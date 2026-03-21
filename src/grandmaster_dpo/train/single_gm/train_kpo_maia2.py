from __future__ import annotations

import argparse
import chess
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math

import torch
from torch.utils.data import DataLoader, Dataset

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move


# ----------------------------
# Dataset
# ----------------------------

class DpoPairs(Dataset):
    def __init__(self, jsonl_path: str):
        self.path = jsonl_path
        self.rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        p = r["prompt"]
        return {
            "fen": p["fen"],
            "elo_self": int(p.get("elo_self", 2800)),
            "elo_oppo": int(p.get("elo_oppo", 2800)),
            "chosen": r["chosen"],       # UCI
            "rejected": r["rejected"],   # UCI
            "label": int(r.get("preference", {}).get("label", 1)),
            "meta": r.get("meta", {}),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {"fen": [], "elo_self": [], "elo_oppo": [], "chosen": [], "rejected": [], "meta": []}
    for b in batch:
        for k in out:
            out[k].append(b[k])
    return out

@dataclass
class TrainConfig:
    gm_name: str
    device: str
    beta: float
    gamma: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    maia_type: str
    train_val_folder: str
    out_dir: str
    run_name: str

# ----------------------------
# Helpers
# ----------------------------

def uci_to_vocab_index(all_moves_dict: Dict[str, int], fen: str, uci: str) -> int:
    side = fen.split(" ")[1]
    uci_eff = mirror_move(uci) if side == "b" else uci
    return int(all_moves_dict.get(uci_eff, -1))

def extract_top_engine_cp(meta: dict) -> float:
    sf_moves = meta["stockfish"]["sf_moves_returned"]
    if not sf_moves:
        return 0.0
    return float(max(cp for _, cp in sf_moves))

def extract_chosen_cp(meta: dict, chosen_uci: str) -> float:
    sf_moves = meta["stockfish"]["sf_moves_returned"]

    # Try to find the chosen move in the Stockfish list
    for uci, cp in sf_moves:
        if uci == chosen_uci:
            return float(cp)

    # Fallback: chosen move not in Stockfish top moves
    cp_values = [cp for _, cp in sf_moves]
    fallback_cp = float(min(cp_values)) if cp_values else 0.0

    print(
        f"[WARN] chosen move {chosen_uci} not found in sf_moves_returned "
        f"(game={meta.get('game_header_hash')}, ply={meta.get('ply_idx')}). "
        f"Using fallback cp={fallback_cp}"
    )

    return fallback_cp

def extract_rest_engine_moves_and_cps(
    meta: dict,
    chosen_uci: str,
    pad_to: int | None = None,
) -> tuple[list[str], list[float]]:
    """
    Returns non-chosen stockfish candidate moves and cps.
    Keeps Stockfish ordering as provided in sf_moves_returned.
    """
    sf_moves = meta["stockfish"]["sf_moves_returned"]

    rest_moves = []
    rest_cps = []

    for uci, cp in sf_moves:
        if uci == chosen_uci:
            continue
        rest_moves.append(uci)
        rest_cps.append(float(cp))

    if pad_to is not None and len(rest_moves) < pad_to:
        pad_n = pad_to - len(rest_moves)
        rest_moves.extend([rest_moves[-1] if rest_moves else chosen_uci] * pad_n)
        rest_cps.extend([rest_cps[-1] if rest_cps else 0.0] * pad_n)

    return rest_moves, rest_cps

def move_logprob_matrix_from_logits(
    logits: torch.Tensor,              # [B, V]
    fens: list[str],
    all_moves_dict,
    moves_2d: list[list[str]],         # [B][K]
    device: torch.device,
) -> torch.Tensor:
    """
    Returns log-prob matrix [B, K] for per-row move lists.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    B = len(fens)
    K = len(moves_2d[0])

    out = torch.empty((B, K), dtype=log_probs.dtype, device=device)

    for i in range(B):
        for j in range(K):
            uci = moves_2d[i][j]
            move_idx = uci_to_vocab_index(all_moves_dict, fens[i], uci)
            out[i, j] = log_probs[i, move_idx]

    return out

def build_rest_candidates_batch(
    meta_list: list[dict],
    chosen_list: list[str],
) -> tuple[list[list[str]], torch.Tensor]:
    """
    Returns:
      rest_moves_batch: list of length B, each a list[str] of equal length K
      rest_cps_batch:   tensor [B, K]
    """
    raw = []
    max_k = 0

    for meta, chosen_uci in zip(meta_list, chosen_list):
        moves, cps = extract_rest_engine_moves_and_cps(meta, chosen_uci)
        raw.append((moves, cps))
        max_k = max(max_k, len(moves))

    rest_moves_batch = []
    rest_cps_batch = []

    for (moves, cps), chosen_uci in zip(raw, chosen_list):
        if len(moves) == 0:
            # degenerate fallback
            moves = [chosen_uci]
            cps = [0.0]

        if len(moves) < max_k:
            pad_n = max_k - len(moves)
            moves = moves + [moves[-1]] * pad_n
            cps = cps + [cps[-1]] * pad_n

        rest_moves_batch.append(moves)
        rest_cps_batch.append(cps)

    rest_cps_tensor = torch.tensor(rest_cps_batch, dtype=torch.float32)
    return rest_moves_batch, rest_cps_tensor

def kl_pi_ref_from_logits(
    logits_pi: torch.Tensor,   # [B, V] already legal-masked (illegal = -inf)
    logits_ref: torch.Tensor,  # [B, V] already legal-masked
) -> torch.Tensor:
    """
    Returns KL(pi || ref) per example: [B]
    """
    logp_pi = torch.log_softmax(logits_pi, dim=-1)     # [B, V]
    logp_ref = torch.log_softmax(logits_ref, dim=-1)   # [B, V]
    p_pi = logp_pi.exp()
    # KL(pi||ref) = sum_a pi(a) (log pi(a) - log ref(a))
    kl = (p_pi * (logp_pi - logp_ref)).sum(dim=-1)     # [B]
    return kl

def ply_from_fen(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    ply = 2 * (fullmove - 1)
    if side == "b":
        ply += 1
    return ply

def device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s in ("mps",):
        return torch.device("mps")
    return torch.device(s)


def max_elo_supported(elo_dict: dict) -> int:
    # find keys like ">=2000" and return 2000
    mx = None
    for k in elo_dict.keys():
        m = re.match(r"^>=\s*(\d+)$", k)
        if m:
            mx = max(mx or 0, int(m.group(1)))
    return mx if mx is not None else 3000


def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    # legal_moves is 0/1 mask same shape as logits
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))


def batch_preprocess(
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    fens: List[str],
    elo_self: List[int],
    elo_oppo: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calls maia2.inference.preprocessing (repo version you pasted):
      preprocessing(fen, elo_self, elo_oppo, elo_dict, all_moves_dict)
        -> board_input, elo_self_cat, elo_oppo_cat, legal_moves_mask
    """
    board_inputs = []
    legal_moves = []
    elo_self_cats = []
    elo_oppo_cats = []

    mx = max_elo_supported(elo_dict)

    for fen, es, eo in zip(fens, elo_self, elo_oppo):
        es = min(int(es), mx)
        eo = min(int(eo), mx)

        bi, es_cat, eo_cat, lm = inference.preprocessing(
            fen, es, eo, elo_dict, all_moves_dict
        )

        board_inputs.append(bi)
        legal_moves.append(lm)
        elo_self_cats.append(int(es_cat))
        elo_oppo_cats.append(int(eo_cat))

    board_input = torch.stack(board_inputs, dim=0).to(device)         # [B, C, 8, 8]
    legal_moves_t = torch.stack(legal_moves, dim=0).to(device)        # [B, V]
    elo_self_t = torch.tensor(elo_self_cats, device=device).long()    # [B]
    elo_oppo_t = torch.tensor(elo_oppo_cats, device=device).long()    # [B]
    return board_input, legal_moves_t, elo_self_t, elo_oppo_t


def forward_logits(
    maia2_model: torch.nn.Module,
    board_input: torch.Tensor,
    elo_self_tensor: torch.Tensor,
    elo_oppo_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Maia2 repo inference uses:
      logits_maia, _, logits_value = model(boards, elos_self, elos_oppo)
    """
    logits_maia, _, _ = maia2_model(board_input, elo_self_tensor, elo_oppo_tensor)
    return logits_maia


def move_logprob_from_logits(
    logits: torch.Tensor,
    fens: List[str],
    all_moves_dict: Dict[str, int],
    moves_uci: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert UCI -> index in Maia vocab (mirroring if fen is black-to-move),
    then gather logprob from logits.
    """
    logp_all = torch.log_softmax(logits, dim=-1)

    idxs: List[int] = []
    for fen, uci in zip(fens, moves_uci):
        side = fen.split(" ")[1]
        uci_eff = mirror_move(uci) if side == "b" else uci
        idx = all_moves_dict.get(uci_eff, None)
        idxs.append(-1 if idx is None else int(idx))

    idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
    safe_idx = idx_t.clamp(min=0)

    gathered = logp_all.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)
    gathered = torch.where(idx_t >= 0, gathered, torch.full_like(gathered, -1e9))
    return gathered

def _pairwise_bucket_loss(
    logp_pi_a: torch.Tensor,   # [B, NA]
    logp_pi_b: torch.Tensor,   # [B, NB]
    logp_ref_a: torch.Tensor,  # [B, NA]
    logp_ref_b: torch.Tensor,  # [B, NB]
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """
    Average DPO-with-margin loss over all pairs a in bucket A, b in bucket B.
    """
    pi_gap = logp_pi_a.unsqueeze(2) - logp_pi_b.unsqueeze(1)     # [B, NA, NB]
    ref_gap = logp_ref_a.unsqueeze(2) - logp_ref_b.unsqueeze(1)  # [B, NA, NB]
    logits = beta * (pi_gap - ref_gap) - gamma
    return (-torch.nn.functional.logsigmoid(logits)).mean()


def _gather_masked_pairs(
    values: torch.Tensor,   # [B, K]
    mask: torch.Tensor,     # [B, K] bool
) -> list[torch.Tensor]:
    """
    Returns a list of length B, each entry shape [Ni] for the masked values in that row.
    """
    out = []
    for b in range(values.shape[0]):
        out.append(values[b][mask[b]])
    return out


def _is_positional(board: chess.Board, move: chess.Move) -> bool:
    """
    Simple heuristic:
    positional/quiet = not capture, not check, not promotion.
    """
    if move.promotion is not None:
        return False
    if board.is_capture(move):
        return False
    if board.gives_check(move):
        return False
    return True


def _same_piece_type(board: chess.Board, move_a: chess.Move, move_b: chess.Move) -> bool:
    piece_a = board.piece_at(move_a.from_square)
    piece_b = board.piece_at(move_b.from_square)
    if piece_a is None or piece_b is None:
        return False
    return piece_a.piece_type == piece_b.piece_type


def _style_similarity_score(
    board: chess.Board,
    chosen_move: chess.Move,
    alt_move: chess.Move,
    chosen_cp: float,
    alt_cp: float,
    cp_scale: float,
    piece_bonus: float,
    positional_bonus: float,
) -> float:
    same_piece = _same_piece_type(board, chosen_move, alt_move)
    same_positional = (_is_positional(board, chosen_move) == _is_positional(board, alt_move))

    piece_mult = piece_bonus if same_piece else 1.0
    positional_mult = positional_bonus if same_positional else 1.0
    cp_mult = math.exp(-abs(float(alt_cp) - float(chosen_cp)) / max(cp_scale, 1e-6))

    return piece_mult * positional_mult * cp_mult


def kpo_style_ranked_loss(
    logp_pi_ch: torch.Tensor,         # [B]
    logp_pi_rest: torch.Tensor,       # [B, K]
    logp_ref_ch: torch.Tensor,        # [B]
    logp_ref_rest: torch.Tensor,      # [B, K]
    fens: List[str],                  # len B
    chosen_uci: List[str],            # len B
    rest_uci: List[List[str]],        # len B, each len K
    chosen_cp: torch.Tensor,          # [B]
    rest_cps: torch.Tensor,           # [B, K]
    beta: float,
    gamma: float,
    cp_scale: float = 40.0,
    piece_bonus: float = 2.0,
    positional_bonus: float = 2.0,
    w_chosen_top: float = 1.0,
    w_chain: float = 0.5,
) -> torch.Tensor:
    """
    Style-ranked KPO loss.

    For each example:
      1) rank all alternatives by style similarity to the chosen move
      2) enforce:
            chosen > alt_1
            alt_1 > alt_2
            alt_2 > alt_3
            ...
    using a DPO/KPO-style pairwise loss.

    This is intended to preserve graded structure among alternatives rather than
    putting them into hard CP buckets.
    """
    device = logp_pi_rest.device
    dtype = logp_pi_rest.dtype
    B, K = logp_pi_rest.shape

    total = torch.zeros((), device=device, dtype=dtype)
    total_w = torch.zeros((), device=device, dtype=dtype)

    for b in range(B):
        fen = fens[b]
        board = chess.Board(fen)

        ch_move = chess.Move.from_uci(chosen_uci[b])
        if ch_move not in board.legal_moves:
            continue

        scored = []
        for k in range(K):
            uci = rest_uci[b][k]
            alt_move = chess.Move.from_uci(uci)
            if alt_move not in board.legal_moves:
                continue

            score = _style_similarity_score(
                board=board,
                chosen_move=ch_move,
                alt_move=alt_move,
                chosen_cp=float(chosen_cp[b].item()),
                alt_cp=float(rest_cps[b, k].item()),
                cp_scale=cp_scale,
                piece_bonus=piece_bonus,
                positional_bonus=positional_bonus,
            )
            scored.append((score, k))

        if len(scored) == 0:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked_idx = [k for _, k in scored]

        # chosen > top-ranked alternative
        top_k = ranked_idx[0]
        loss_top = _pairwise_bucket_loss(
            logp_pi_ch[b:b+1].unsqueeze(1),           # [1, 1]
            logp_pi_rest[b:b+1, top_k:top_k+1],      # [1, 1]
            logp_ref_ch[b:b+1].unsqueeze(1),         # [1, 1]
            logp_ref_rest[b:b+1, top_k:top_k+1],     # [1, 1]
            beta=beta,
            gamma=gamma,
        )
        total = total + w_chosen_top * loss_top
        total_w = total_w + w_chosen_top

        # chain among ranked alternatives
        if len(ranked_idx) >= 2:
            chain_losses = []
            for i in range(len(ranked_idx) - 1):
                ka = ranked_idx[i]
                kb = ranked_idx[i + 1]

                loss_i = _pairwise_bucket_loss(
                    logp_pi_rest[b:b+1, ka:ka+1],     # [1, 1]
                    logp_pi_rest[b:b+1, kb:kb+1],     # [1, 1]
                    logp_ref_rest[b:b+1, ka:ka+1],    # [1, 1]
                    logp_ref_rest[b:b+1, kb:kb+1],    # [1, 1]
                    beta=beta,
                    gamma=gamma,
                )
                chain_losses.append(loss_i)

            if chain_losses:
                chain_mean = torch.stack(chain_losses).mean()
                total = total + w_chain * chain_mean
                total_w = total_w + w_chain

    if total_w.item() == 0.0:
        return (logp_pi_ch.mean() + logp_pi_rest.mean()) * 0.0

    return total / total_w

# ----------------------------
# Eval
# ----------------------------

@torch.no_grad()
def evaluate(
    policy: torch.nn.Module,
    ref: torch.nn.Module,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    loader: DataLoader,
    device: torch.device,
    beta: float,
    gamma: float,
) -> Dict[str, float]:
    policy.eval()
    ref.eval()

    total_loss = 0.0
    n = 0

    for batch in loader:
        fens = batch["fen"]
        chosen = batch["chosen"]
        bs = len(fens)
        board_input, legal_moves, es_t, eo_t = batch_preprocess(
            all_moves_dict, elo_dict, batch["fen"], batch["elo_self"], batch["elo_oppo"], device
        )

        logits_pi = forward_logits(policy, board_input, es_t, eo_t)
        logits_ref = forward_logits(ref, board_input, es_t, eo_t)

        logits_pi = apply_legal_mask(logits_pi, legal_moves)
        logits_ref = apply_legal_mask(logits_ref, legal_moves)

        logp_pi_ch = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["chosen"], device)
        logp_pi_rj = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["rejected"], device)

        logp_ref_ch = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["chosen"], device)
        logp_ref_rj = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["rejected"], device)
        
        # ----------------------------
        # build non-chosen engine candidate set from meta
        # ----------------------------
        meta_list = batch["meta"]

        rest_moves_batch, rest_cps = build_rest_candidates_batch(
            meta_list=meta_list,
            chosen_list=chosen,
        )
        rest_cps = rest_cps.to(device)

        # NEW: chosen CP instead of best_engine_cp
        chosen_cps = torch.tensor(
            [extract_chosen_cp(m, uci) for m, uci in zip(meta_list, chosen)],
            dtype=torch.float32,
            device=device,
        )

        # ----------------------------
        # logprobs for non-chosen engine candidates
        # ----------------------------
        logp_pi_rest = move_logprob_matrix_from_logits(
            logits_pi,
            fens,
            all_moves_dict,
            rest_moves_batch,
            device,
        )

        with torch.no_grad():
            logp_ref_rest = move_logprob_matrix_from_logits(
                logits_ref,
                fens,
                all_moves_dict,
                rest_moves_batch,
                device,
            )

        # ----------------------------
        # NEW style-ranked KPO loss
        # ----------------------------
        loss = kpo_style_ranked_loss(
            logp_pi_ch=logp_pi_ch,
            logp_pi_rest=logp_pi_rest,
            logp_ref_ch=logp_ref_ch,
            logp_ref_rest=logp_ref_rest,
            fens=fens,
            chosen_uci=chosen,
            rest_uci=rest_moves_batch,
            chosen_cp=chosen_cps,
            rest_cps=rest_cps,
            beta=beta,
            gamma=gamma,
            cp_scale=40.0,
            piece_bonus=2.0,
            positional_bonus=2.0,
            w_chosen_top=1.0,
            w_chain=0.5,
        )

        bs = len(batch["fen"])
        total_loss += float(loss) * bs
        n += bs

    return {"loss": total_loss / max(1, n)}


# ----------------------------
# Train
# ----------------------------

def train_one_run(cfg: TrainConfig) -> None:

    train_jsonl = Path(f"{cfg.train_val_folder}/{cfg.gm_name}_train_dpo.jsonl")
    val_jsonl = Path(f"{cfg.train_val_folder}/{cfg.gm_name}_val_dpo.jsonl")
    out_dir = Path(f"{cfg.out_dir}/{cfg.gm_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_from_str(cfg.device)

    # Load Maia-2 base weights twice
    policy = maia_model.from_pretrained(type=cfg.maia_type, device=str(device))
    policy.train()
    ref = maia_model.from_pretrained(type=cfg.maia_type, device=str(device))
    ref.eval()

    policy.to(device)
    ref.to(device)
    for p in ref.parameters():
        p.requires_grad_(False)

    # Repo version: prepare() returns [all_moves_dict, elo_dict, all_moves_dict_reversed]
    prep = inference.prepare()
    all_moves_dict, elo_dict, _ = prep

    train_ds = DpoPairs(train_jsonl)
    val_ds = DpoPairs(val_jsonl)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    optim = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        policy.train()
        running = 0.0
        seen = 0

        for batch in train_loader:
            fens = batch["fen"]
            chosen = batch["chosen"]
            bs = len(fens)
            step += 1

            board_input, legal_moves, es_t, eo_t = batch_preprocess(
                all_moves_dict, elo_dict, batch["fen"], batch["elo_self"], batch["elo_oppo"], device
            )

            logits_pi = forward_logits(policy, board_input, es_t, eo_t)
            with torch.no_grad():
                logits_ref = forward_logits(ref, board_input, es_t, eo_t)

            logits_pi = apply_legal_mask(logits_pi, legal_moves)
            logits_ref = apply_legal_mask(logits_ref, legal_moves)

            logp_pi_ch = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["chosen"], device)
            logp_pi_rj = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["rejected"], device)

            with torch.no_grad():
                logp_ref_ch = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["chosen"], device)
                logp_ref_rj = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["rejected"], device)

            # ----------------------------
            # build non-chosen engine candidate set from meta
            # ----------------------------
            meta_list = batch["meta"]

            rest_moves_batch, rest_cps = build_rest_candidates_batch(
                meta_list=meta_list,
                chosen_list=chosen,
            )
            rest_cps = rest_cps.to(device)

            # NEW: chosen CP instead of best_engine_cp
            chosen_cps = torch.tensor(
                [extract_chosen_cp(m, uci) for m, uci in zip(meta_list, chosen)],
                dtype=torch.float32,
                device=device,
            )

            # ----------------------------
            # logprobs for non-chosen engine candidates
            # ----------------------------
            logp_pi_rest = move_logprob_matrix_from_logits(
                logits_pi,
                fens,
                all_moves_dict,
                rest_moves_batch,
                device,
            )

            with torch.no_grad():
                logp_ref_rest = move_logprob_matrix_from_logits(
                    logits_ref,
                    fens,
                    all_moves_dict,
                    rest_moves_batch,
                    device,
                )

            # ----------------------------
            # NEW style-ranked KPO loss
            # ----------------------------
            loss = kpo_style_ranked_loss(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rest=logp_pi_rest,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rest=logp_ref_rest,
                fens=fens,
                chosen_uci=chosen,
                rest_uci=rest_moves_batch,
                chosen_cp=chosen_cps,
                rest_cps=rest_cps,
                beta=cfg.beta,
                gamma=cfg.gamma,
                cp_scale=40.0,
                piece_bonus=2.0,
                positional_bonus=2.0,
                w_chosen_top=1.0,
                w_chain=0.5,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            optim.step()

            bs = len(batch["fen"])
            running += float(loss.detach()) * bs
            seen += bs

            if step % 50 == 0:
                print(f"[epoch {epoch}] step={step} train_kpo_loss={running/max(1,seen):.4f}")

        metrics = evaluate(policy, ref, all_moves_dict, elo_dict, val_loader, device=device, beta=cfg.beta, gamma=cfg.gamma)
        val_loss = metrics["loss"]
        print(f"[epoch {epoch}] val_kpo_loss={val_loss:.4f}")

        ckpt_path = out_dir / f"policy_epoch{epoch}_kpo_{make_run_name(cfg)}.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / f"policy_best_kpo_{make_run_name(cfg)}.pt"
            torch.save(policy.state_dict(), best_path)
            print(f"Saved best: {best_path} (val_kpo_loss={best_val:.4f})")

def make_run_name(cfg: TrainConfig) -> str:
    return f"beta={cfg.beta:.2f}_gamma={cfg.gamma:.2f}"

def build_runs(args) -> list[TrainConfig]:
    base = dict(
        gm_name=args.gm_name,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        maia_type=args.maia_type,
        train_val_folder=args.train_val_folder,
        out_dir=args.out_dir,
    )

    runs = []

    if args.preset == "single":
        cfg = TrainConfig(
            **base,
            beta=args.beta,
            gamma=args.gamma,
            run_name=f"single_beta={args.beta:.2f}_gamma={args.gamma:.2f}",
        )
        runs.append(cfg)

    elif args.preset == "overnight":
        combos = [
            {"beta": 0.4, "gamma": 0.5},
            {"beta": 0.4, "gamma": 1.0},
            {"beta": 0.6, "gamma": 0.5},
            {"beta": 0.6, "gamma": 1.0},
            {"beta": 0.8, "gamma": 0.5},
            {"beta": 0.8, "gamma": 1.0},
            {"beta": 1.0, "gamma": 0.5},
            {"beta": 1.0, "gamma": 1.0},
        ]
        for d in combos:
            beta = d["beta"]
            gamma = d["gamma"]
            cfg = TrainConfig(
                **base,
                beta=beta,
                gamma=gamma,
                run_name=f"beta={beta:.2f}_gamma={gamma:.2f}",
            )
            runs.append(cfg)

    elif args.preset == "overnight4_part2":
        combos = [
            {"beta": 1.2, "gamma": 0.5},
            {"beta": 1.2, "gamma": 1.0},
            {"beta": 0.6, "gamma": 0.5},
            {"beta": 0.6, "gamma": 1.0},
        ]
        for d in combos:
            beta = d["beta"]
            gamma = d["gamma"]
            cfg = TrainConfig(
                **base,
                beta=beta,
                gamma=gamma,
                run_name=f"beta={beta:.2f}_gamma={gamma:.2f}",
            )
            runs.append(cfg)

    return runs

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/train/single_gm/train_kpo_maia2.py --gm_name caruana --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic --beta 0.8 --gamma 1.2 --preset overnight
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", type=str, required=True)

    ap.add_argument("--device", type=str, default="cpu")  # "mps" works too if your torch build supports it
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--gamma", type=float, default=1.2)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--train_val_folder", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--preset", type=str, default="single", choices=["single", "overnight", "overnight4_part2"])

    args = ap.parse_args()

    runs = build_runs(args)

    print("Generated run configs:")
    for cfg in runs:
        print(make_run_name(cfg))

    for i, cfg in enumerate(runs, start=1):
        print("=" * 80)
        print(f"Starting run {i}/{len(runs)}: {cfg.run_name}")
        print(cfg)
        print("=" * 80)
        train_one_run(cfg)


if __name__ == "__main__":
    main()



