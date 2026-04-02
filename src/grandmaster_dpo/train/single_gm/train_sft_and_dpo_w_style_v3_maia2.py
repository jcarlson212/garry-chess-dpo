from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
import chess
import math

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move

from grandmaster_dpo.utilities.shared_style_emb_model_utils import StyleEncoder, add_batch_dim, move_feature_dict_to_device, move_feature_dict_to_device, pick_device, raw_example_to_cached_arrays, raw_example_to_model_features
from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import make_config

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

        self.game_id_and_ply_to_prev_10_plys = {}
        self.game_id_and_ply_to_fut_10_plys = {}

        def create_window_item(rows, index, target_game):
            if index < 0:
                return None 
            elif index >= len(rows):
                return None
            else:
                if rows[index]["meta"]["game_header_hash"] != target_game:
                    return None
                return rows[index]
        
        for i, r in enumerate(self.rows):
            hash_key = f'{r["meta"]["game_header_hash"]}_{r["meta"]["ply_idx"]}'
            self.game_id_and_ply_to_prev_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i-10, i)]
            self.game_id_and_ply_to_fut_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i+1, i+11)]


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


# ----------------------------
# Helpers
# ----------------------------

def key_game_ply(meta: Dict[str, Any]) -> str:
    return f'{meta["game_header_hash"]}_{meta["ply_idx"]}'


def safe_get_prev_fens(
    prev_map: Dict[str, List[Dict[str, Any]]],
    meta: Dict[str, Any],
    n: int = 5,
) -> List[Optional[str]]:
    """
    Returns the previous n FENs for this training row, padded on the left with None.
    Output length is exactly n.
    """
    key = key_game_ply(meta)
    rows = prev_map.get(key, [])

    # keep only fen strings if present
    fens = []
    for r in rows[-n:]:
        try:
            fens.append(r["prompt"]["fen"])
        except Exception as e:
            if r != None:
                raise(e)
            else:
                fens.append(None)

    # left-pad with None so length == n
    if len(fens) < n:
        fens = [None] * (n - len(fens)) + fens

    return fens

def safe_get_next_fens_chosen(
    fut_map: Dict[str, List[Dict[str, Any]]],
    meta: Dict[str, Any],
    n: int = 5,
) -> List[Optional[str]]:
    """
    Returns the next n recorded FENs along the chosen/game trajectory, padded on the right with None.
    Output length is exactly n.
    """
    key = key_game_ply(meta)
    rows = fut_map.get(key, [])

    fens = []
    for r in rows[:n]:
        try:
            fens.append(r["prompt"]["fen"])
        except Exception as e:
            if r != None:
                raise(e)
            else:
                fens.append(None)

    if len(fens) < n:
        fens = fens + [None] * (n - len(fens))

    return fens

def fen_after_move(fen: str, uci: str) -> Optional[str]:
    """
    Applies a legal UCI move to fen and returns resulting fen.
    Returns None on failure.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise ValueError("Engine proposed move was not legal somehow")
    board.push(move)
    return board.fen()


def safe_get_next_fens_rejected(
    fen: str,
    rejected_uci: str,
    n: int = 5,
) -> List[Optional[str]]:
    """
    We do NOT have true future trajectory for rejected moves in the dataset.
    So we use only the immediate board after rejected move, then pad with None.
    Output length is exactly n.
    """
    out = [fen_after_move(fen, rejected_uci)]
    if len(out) < n:
        out = out + [None] * (n - len(out))
    return out[:n]

def extract_move_cp(meta: dict, uci: str) -> float:
    sf_moves = meta["stockfish"]["sf_moves_returned"]
    for sf_uci, cp in sf_moves:
        if sf_uci == uci:
            return float(cp)
        
    cp_values = [cp for _, cp in sf_moves]
    fallback_cp = float(min(cp_values)) if cp_values else 0.0
    #print(
     #   f"[WARN] move {uci} not found in sf_moves_returned "
    #    f"(game={meta.get('game_header_hash')}, ply={meta.get('ply_idx')}). "
    #    f"Using fallback cp={fallback_cp}"
    #)
    return fallback_cp

# ============================================================
# Main style score
# ============================================================

def compute_style_score_v3(
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    style_embedding_model: nn.Module,
    event: str,
    style_tau: float,
    prev_fens: Optional[Sequence[Optional[str]]] = None,
) -> float:
    game_type = "classical"
    if "blitz" in event.lower():
        game_type = "blitz"
    elif "rapid" in event.lower():
        game_type = "rapid"

    device = next(style_embedding_model.parameters()).device

    style_model_features_chosen = add_batch_dim(move_feature_dict_to_device(
        raw_example_to_model_features(
            {
                "board_t_minus_4": prev_fens[-4] if prev_fens and len(prev_fens) >= 4 else None,
                "board_t_minus_3": prev_fens[-3] if prev_fens and len(prev_fens) >= 3 else None,
                "board_t_minus_2": prev_fens[-2] if prev_fens and len(prev_fens) >= 2 else None,
                "board_t_minus_1": prev_fens[-1] if prev_fens and len(prev_fens) >= 1 else None,
                "board_t": fen,
                "move_played": chosen_uci,
                "game_type": game_type, 
            },
            "phi1"
        ), 
        device))
    
    style_model_features_rejected = add_batch_dim(move_feature_dict_to_device(
        raw_example_to_model_features(
            {
                "board_t_minus_4": prev_fens[-4] if prev_fens and len(prev_fens) >= 4 else None,
                "board_t_minus_3": prev_fens[-3] if prev_fens and len(prev_fens) >= 3 else None,
                "board_t_minus_2": prev_fens[-2] if prev_fens and len(prev_fens) >= 2 else None,
                "board_t_minus_1": prev_fens[-1] if prev_fens and len(prev_fens) >= 1 else None,
                "board_t": fen,
                "move_played": rejected_uci,
                "game_type": game_type, 
            },
            "phi1"
        ), 
        device))

    style_embedding_chosen = style_embedding_model(style_model_features_chosen).squeeze(0)
    style_embedding_rejected = style_embedding_model(style_model_features_rejected).squeeze(0)

    style_score = torch.dot(style_embedding_chosen, style_embedding_rejected) / style_tau

    return style_score

def dpo_loss_style_weighted_v3(
    logp_pi_ch: torch.Tensor,
    logp_pi_rj: torch.Tensor,
    logp_ref_ch: torch.Tensor,
    logp_ref_rj: torch.Tensor,
    style_score: torch.Tensor,
    beta: float,
    tau: float,
) -> torch.Tensor:
    u = torch.exp(-(style_score) / tau)
    dpo = -torch.nn.functional.logsigmoid(beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj)))
    loss = (u.detach() * dpo).mean()
    return loss

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

def chosen_index_tensor(
    fens: List[str],
    all_moves_dict: Dict[str, int],
    moves_uci: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert UCI -> Maia vocab index (mirroring if fen is black-to-move).
    Returns idx_t with -1 for unknown moves (should be rare; those will be ignored).
    """
    idxs: List[int] = []
    for fen, uci in zip(fens, moves_uci):
        side = fen.split(" ")[1]
        uci_eff = mirror_move(uci) if side == "b" else uci
        idx = all_moves_dict.get(uci_eff, None)
        idxs.append(-1 if idx is None else int(idx))
    return torch.tensor(idxs, device=device, dtype=torch.long)

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


def dpo_loss_style_weighted(
    logp_pi_ch: torch.Tensor,
    logp_pi_rj: torch.Tensor,
    logp_ref_ch: torch.Tensor,
    logp_ref_rj: torch.Tensor,
    style_score: torch.Tensor,
    beta: float,
    tau: float,
) -> torch.Tensor:
    pi_gap = logp_pi_ch - logp_pi_rj
    ref_gap = logp_ref_ch - logp_ref_rj
    x = beta * (pi_gap - ref_gap)

    per_example_loss = -torch.nn.functional.logsigmoid(x)
    weights = torch.exp(-style_score / tau)
    return (weights * per_example_loss).sum() / weights.sum().clamp_min(1e-12)

def supervised_nll_loss(
    logits_masked: torch.Tensor,
    idx_t: torch.Tensor,
) -> torch.Tensor:
    """
    Standard supervised fine-tuning objective:
      loss = -mean(log p(chosen_move))
    ignoring examples where chosen move isn't in vocab (idx == -1).
    """
    logp_all = torch.log_softmax(logits_masked, dim=-1)  # [B, V]

    valid = idx_t >= 0
    if valid.sum().item() == 0:
        # return a zero scalar that still has grad
        return logits_masked.sum() * 0.0

    safe_idx = idx_t.clamp(min=0)
    gathered = logp_all.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)  # [B]
    gathered = gathered[valid]
    return (-gathered).mean()
# ----------------------------
# Eval
# ----------------------------

@torch.no_grad()
def evaluate(
    policy: torch.nn.Module,
    ref: torch.nn.Module,
    style_embedding_model: torch.nn.Module,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    loader: DataLoader,
    device: torch.device,
    beta: float,
    dpo_loss_weight: float,
    style_tau: float,
    game_id_and_ply_to_prev_10_plys: Dict[str, list],
    game_id_and_ply_to_fut_10_plys: Dict[str, list]
) -> Dict[str, float]:
    policy.eval()
    ref.eval()
    style_embedding_model.eval()

    total_loss = 0.0
    n = 0

    for batch in loader:
        fens = batch["fen"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        meta_list = batch["meta"]
        ply_idxs = [r["ply_idx"] for r in meta_list]
        board_input, legal_moves, es_t, eo_t = batch_preprocess(
            all_moves_dict, elo_dict, batch["fen"], batch["elo_self"], batch["elo_oppo"], device
        )

        logits_pi = forward_logits(policy, board_input, es_t, eo_t)
        logits_ref = forward_logits(ref, board_input, es_t, eo_t)

        logits_pi = apply_legal_mask(logits_pi, legal_moves)
        logits_ref = apply_legal_mask(logits_ref, legal_moves)
        idx_t = chosen_index_tensor(batch["fen"], all_moves_dict, batch["chosen"], device)
        
        logp_pi_ch = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["chosen"], device)
        logp_pi_rj = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["rejected"], device)

        logp_ref_ch = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["chosen"], device)
        logp_ref_rj = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["rejected"], device)
        
        # NEW: chosen/rejected CPs
        chosen_cps = [extract_move_cp(m, ch) for m, ch in zip(meta_list, chosen)]
        rejected_cps = [extract_move_cp(m, rj) for m, rj in zip(meta_list, rejected)]

        prev_fens_batch = [
            safe_get_prev_fens(game_id_and_ply_to_prev_10_plys, m, n=5)
            for m in meta_list
        ]

        next_fens_chosen_batch = [
            safe_get_next_fens_chosen(game_id_and_ply_to_fut_10_plys, m, n=5)
            for m in meta_list
        ]

        next_fens_rejected_batch = [
            safe_get_next_fens_rejected(fen, rj, n=5)
            for fen, rj in zip(fens, rejected)
        ]
        # NEW: style similarity scores
        style_scores = torch.tensor(
            [
                compute_style_score_v3(
                    fen=fen,
                    chosen_uci=ch,
                    rejected_uci=rj,
                    style_embedding_model=style_embedding_model,
                    event=m["event"],
                    style_tau=style_tau,
                    prev_fens=prev_fens,
                )
                for fen, ch, rj, ch_cp, rj_cp, ply_idx, prev_fens, next_fens_chosen, next_fens_rejected, m
                in zip(
                    fens,
                    chosen,
                    rejected,
                    chosen_cps,
                    rejected_cps,
                    ply_idxs,
                    prev_fens_batch,
                    next_fens_chosen_batch,
                    next_fens_rejected_batch,
                    meta_list
                )
            ],
            dtype=torch.float32,
            device=device,
        )

        loss = (
            dpo_loss_weight
            * dpo_loss_style_weighted(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
                style_score=style_scores,
                beta=beta,
                tau=style_tau,
            )
            + supervised_nll_loss(logits_pi, idx_t)
        )

        bs = len(batch["fen"])
        total_loss += float(loss) * bs
        n += bs

    return {"loss": total_loss / max(1, n)}


# ----------------------------
# Train
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/train/single_gm/train_sft_and_dpo_w_style_v3_maia2.py --gm_name caruana --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic --dpo_loss_weight 0.1
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", type=str, required=True)

    ap.add_argument("--device", type=str, default="cpu")  # "mps" works too if your torch build supports it
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--dpo_loss_weight", type=float, default=0.1)
    ap.add_argument("--style_tau", type=float, default=0.75)
    ap.add_argument("--style_embedding_model_checkpoint", type=str, default=(
            "./final_experiments_for_paper/experiment2_style_model/trained_models/"
            "super_v2_phi1_tau0_25_if_winner__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42/"
            "best.pt"
    ))

    #
    #cp_scale:        20, 40
    #piece_bonus:     1.0, 1.3
    #positional_bonus:2.0
    #tau:             5, 10, 20

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--train_val_folder", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()

    train_jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_train_dpo.jsonl")
    val_jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_val_dpo.jsonl")
    out_dir = Path(f"{args.out_dir}/{args.gm_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device("auto")

    # Load Maia-2 base weights twice
    policy = maia_model.from_pretrained(type=args.maia_type, device=str(device))
    policy.train()
    ref = maia_model.from_pretrained(type=args.maia_type, device=str(device))
    ref.eval()

    # This is the style embedding model
    style_encoder_training_cfg = make_config(
        study_name="super_v3_phi1_tau0_25_warm_from_v2final",
        train_dir="./final_experiments_for_paper/experiment2_style_model/pairs_v3_cached/train",
        eval_dir="./final_experiments_for_paper/experiment2_style_model/pairs_v2_cached/eval",
        pair_variant="v3",
        seed=42,
        embedding_dim=256,
        batch_size=4096,
        lr=3e-4,
        tau=0.25,
        phi_variant="phi1",
        epochs=8,
        max_steps_per_epoch=5000,
        max_eval_batches=150,
        num_workers=0,
        init_from_checkpoint=args.style_embedding_model_checkpoint,
        init_reset_optimizer=True,
        init_strict_load=True,
    )

    style_embedding_model = StyleEncoder(style_encoder_training_cfg)
    style_embedding_model.eval()

    policy.to(device)
    ref.to(device)
    for p in ref.parameters():
        p.requires_grad_(False)

    # Repo version: prepare() returns [all_moves_dict, elo_dict, all_moves_dict_reversed]
    prep = inference.prepare()
    all_moves_dict, elo_dict, _ = prep

    train_ds = DpoPairs(train_jsonl)
    game_id_and_ply_to_prev_10_plys = train_ds.game_id_and_ply_to_prev_10_plys
    game_id_and_ply_to_fut_10_plys = train_ds.game_id_and_ply_to_fut_10_plys
    val_ds = DpoPairs(val_jsonl)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        running = 0.0
        seen = 0

        for batch in train_loader:
            fens = batch["fen"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            meta_list = batch["meta"]
            ply_idxs = [r["ply_idx"] for r in meta_list]
            step += 1

            board_input, legal_moves, es_t, eo_t = batch_preprocess(
                all_moves_dict, elo_dict, batch["fen"], batch["elo_self"], batch["elo_oppo"], device
            )

            logits_pi = forward_logits(policy, board_input, es_t, eo_t)
            with torch.no_grad():
                logits_ref = forward_logits(ref, board_input, es_t, eo_t)

            logits_pi = apply_legal_mask(logits_pi, legal_moves)
            logits_ref = apply_legal_mask(logits_ref, legal_moves)
            idx_t = chosen_index_tensor(batch["fen"], all_moves_dict, batch["chosen"], device)

            logp_pi_ch = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["chosen"], device)
            logp_pi_rj = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["rejected"], device)

            with torch.no_grad():
                logp_ref_ch = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["chosen"], device)
                logp_ref_rj = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["rejected"], device)

            # NEW: chosen/rejected CPs
            chosen_cps = [extract_move_cp(m, ch) for m, ch in zip(meta_list, chosen)]
            rejected_cps = [extract_move_cp(m, rj) for m, rj in zip(meta_list, rejected)]

            game_id_and_ply_to_prev_10_plys = train_ds.game_id_and_ply_to_prev_10_plys
            game_id_and_ply_to_fut_10_plys = train_ds.game_id_and_ply_to_fut_10_plys

            prev_fens_batch = [
                safe_get_prev_fens(game_id_and_ply_to_prev_10_plys, m, n=5)
                for m in meta_list
            ]

            next_fens_chosen_batch = [
                safe_get_next_fens_chosen(game_id_and_ply_to_fut_10_plys, m, n=5)
                for m in meta_list
            ]

            next_fens_rejected_batch = [
                safe_get_next_fens_rejected(fen, rj, n=5)
                for fen, rj in zip(fens, rejected)
            ]

            # NEW: style similarity scores
            style_scores = torch.tensor(
                [

                    compute_style_score_v3(
                        fen=fen,
                        chosen_uci=ch,
                        rejected_uci=rj,
                        style_embedding_model=style_embedding_model,
                        event=m["event"],
                        style_tau=args.style_tau,
                        prev_fens=prev_fens,
                    )
                    for fen, ch, rj, ch_cp, rj_cp, ply_idx, prev_fens, next_fens_chosen, next_fens_rejected, m
                    in zip(
                        fens,
                        chosen,
                        rejected,
                        chosen_cps,
                        rejected_cps,
                        ply_idxs,
                        prev_fens_batch,
                        next_fens_chosen_batch,
                        next_fens_rejected_batch,
                        meta_list
                    )
                ],
                dtype=torch.float32,
                device=device,
            )

            loss = (
                args.dpo_loss_weight
                * dpo_loss_style_weighted(
                    logp_pi_ch=logp_pi_ch,
                    logp_pi_rj=logp_pi_rj,
                    logp_ref_ch=logp_ref_ch,
                    logp_ref_rj=logp_ref_rj,
                    style_score=style_scores,
                    beta=args.beta,
                    tau=args.style_tau,
                )
                + supervised_nll_loss(logits_pi, idx_t)
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optim.step()

            bs = len(batch["fen"])
            running += float(loss.detach()) * bs
            seen += bs

            if step % 50 == 0:
                print(f"[epoch {epoch}] step={step} train_sft_and_dpo_w_style_v3_loss={running/max(1,seen):.4f}")

        metrics = evaluate(policy, ref, style_embedding_model, all_moves_dict, elo_dict, val_loader, device=device, beta=args.beta, dpo_loss_weight=args.dpo_loss_weight,
                           style_tau=args.style_tau, game_id_and_ply_to_prev_10_plys=train_ds.game_id_and_ply_to_prev_10_plys, game_id_and_ply_to_fut_10_plys=train_ds.game_id_and_ply_to_fut_10_plys)
        val_loss = metrics["loss"]
        print(f"[epoch {epoch}] val_sft_and_dpo_w_style_v3_loss={val_loss:.4f}")


        ckpt_path = out_dir / f"policy_epoch{epoch}_sft_and_dpo_w_style_v3_beta={args.beta:.2f}_dpo_loss_weight={args.dpo_loss_weight:.2f}_style_tau={args.style_tau:.2f}_embedding_model={args.style_embedding_model_checkpoint.split('/')[-1]}.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / f"policy_best_sft_and_dpo_w_style_v3_beta={args.beta:.2f}_dpo_loss_weight={args.dpo_loss_weight:.2f}_style_tau={args.style_tau:.2f}_embedding_model={args.style_embedding_model_checkpoint.split('/')[-1]}.pt"
            torch.save(policy.state_dict(), best_path)
            print(f"Saved best: {best_path} (val_sft_and_dpo_w_style_v3_loss={best_val:.4f})")


if __name__ == "__main__":
    main()



