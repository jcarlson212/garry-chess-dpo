from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import statistics
import chess

import torch
from torch.utils.data import DataLoader, Dataset

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move


# ----------------------------
# Dataset
# ----------------------------


class DpoPairs(Dataset):
    def __init__(self, jsonl_path: str, debug: bool = False):
        self.rows: List[Dict[str, Any]] = []
        self.debug = debug
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
        p = r.get("prompt", {}) or {}
        meta = r.get("meta", {}) or {}

        # Correct location: meta['game_header_hash'] (not top-level).
        gh = meta.get("game_header_hash")
        game_id = str(gh)

        if self.debug and idx < 3:
            print(f"r keys: {r.keys()}")
            print(f"meta keys: {meta.keys()}")
            print(f"p keys: {p.keys()}")
            print(f"meta.game_header_hash: {gh!r}")
            print(f"computed game_id: {game_id}")

        # Minimal required keys + metadata keys used by eval
        return {
            "fen": p["fen"],
            "elo_self": int(p.get("elo_self", 2800)),
            "elo_oppo": int(p.get("elo_oppo", 2800)),
            "chosen": r["chosen"],
            "rejected": r["rejected"],

            "game_id": game_id,
            "ply_idx": int(meta.get("ply_idx", -1)),
            "fullmove_number": int(meta.get("fullmove_number", -1)),
            "side_to_move": str(meta.get("side_to_move", "")),
            "opening_prefix_uci_20": meta.get("opening_prefix_uci_20") or [],
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Collate *all* fields your eval loop may read.
    """
    out: Dict[str, List[Any]] = {
        "fen": [],
        "elo_self": [],
        "elo_oppo": [],
        "chosen": [],
        "rejected": [],
        "game_id": [],
        "ply_idx": [],
        "fullmove_number": [],
        "side_to_move": [],
        "opening_prefix_uci_20": [],
    }
    for b in batch:
        for k in out:
            out[k].append(b.get(k))
    return out


# ----------------------------
# Helpers (match training)
# ----------------------------

@torch.no_grad()
def probe_opening_distributions_from_policy(
    policy: torch.nn.Module,
    *,
    maia_type: str,
    device: torch.device,
    all_moves: List[str],
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    elo_self: int = 2800,
    elo_oppo: int = 2800,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "white_first_move_probs": {uci: prob, ...},
        "black_reply_probs_cond_on_white": {
            white_uci: {black_uci: prob, ...},
            ...
        },
        "meta": {...}
      }

    Notes:
      - Uses fine-tuned policy logits (legal-masked) in *canonical* opening states.
      - White distribution is computed at the initial position.
      - Black conditional distributions are computed at positions after each probed white first move.
      - Probabilities are over a curated move set; we also return 'other_mass' remainder.
    """

    def apply_legal_mask_row(logits_row: torch.Tensor, legal_row: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(logits_row.dtype).min
        return torch.where(legal_row > 0, logits_row, torch.full_like(logits_row, neg_inf))

    def uci_to_vocab_index_local(fen: str, uci: str) -> int:
        side = fen.split(" ")[1]  # 'w' or 'b'
        # Maia vocab is stored in "white perspective".
        # For real black moves, map them into that space using mirror_move (involution).
        uci_eff = mirror_move(uci) if side == "b" else uci
        return int(all_moves_dict.get(uci_eff, -1))

    def probs_for_ucis(fen: str, logits_masked_row: torch.Tensor, ucis: List[str]) -> Tuple[Dict[str, float], float]:
        # temperature
        if temperature <= 0:
            probs = torch.softmax(logits_masked_row, dim=-1)
        else:
            probs = torch.softmax(logits_masked_row / temperature, dim=-1)

        out: Dict[str, float] = {}
        used = 0.0
        for u in ucis:
            j = uci_to_vocab_index_local(fen, u)
            p = float(probs[j].item()) if j >= 0 else 0.0
            out[u] = p
            used += p
        other = max(0.0, 1.0 - used)
        return out, other

    # ---- canonical states we probe ----

    # Initial position (white to move)
    start_fen = chess.STARTING_FEN  # "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # White first moves we probe (you can expand this list)
    white_first_moves = [
        "e2e4", "d2d4", "c2c4", "g1f3", "g2g3", "b2b3", "f2f4", "b2b4", "a2a4",
    ]

    # Black reply moves we probe (common replies; expand as you like)
    black_replies = [
        "c7c5", "e7e5", "e7e6", "c7c6", "d7d5", "g8f6", "g7g6", "d7d6",
    ]

    # ---- helper to run model on one fen ----
    def logits_masked_for_fen(fen: str) -> torch.Tensor:
        bi, es_cat, eo_cat, lm = inference.preprocessing(
            fen, int(elo_self), int(elo_oppo), elo_dict, all_moves_dict
        )
        board_input = bi.unsqueeze(0).to(device)            # [1,...]
        legal = lm.unsqueeze(0).to(device)                  # [1,V]
        es_t = torch.tensor([int(es_cat)], device=device).long()
        eo_t = torch.tensor([int(eo_cat)], device=device).long()

        logits, _, _ = policy(board_input, es_t, eo_t)       # [1,V]
        logits = logits.squeeze(0)                           # [V]
        legal = legal.squeeze(0)                             # [V]
        return apply_legal_mask_row(logits, legal)

    # ---- compute white distribution ----
    logits_start = logits_masked_for_fen(start_fen)
    white_probs, white_other = probs_for_ucis(start_fen, logits_start, white_first_moves)

    

    # ---- compute black conditional distributions ----
    black_cond: Dict[str, Any] = {}
    for w in white_first_moves:
        b = chess.Board(start_fen)
        b.push_uci(w)
        fen_b = b.fen()

        logits_b = logits_masked_for_fen(fen_b)
        probs_b, other_b = probs_for_ucis(fen_b, logits_b, black_replies)

        black_cond[w] = {
            "fen_after_white": fen_b,
            "black_reply_probs": probs_b,
            "other_mass": other_b,
        }

    return {
        "white_first_move_probs": white_probs,
        "white_other_mass": white_other,
        "black_reply_probs_cond_on_white": black_cond,
        "meta": {
            "elo_self": elo_self,
            "elo_oppo": elo_oppo,
            "temperature": temperature,
            "maia_type": maia_type,
            "white_moves_probed": white_first_moves,
            "black_replies_probed": black_replies,
        },
    }

def device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s in ("mps",):
        return torch.device("mps")
    return torch.device(s)


def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
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
    board_inputs = []
    legal_moves = []
    elo_self_cats = []
    elo_oppo_cats = []

    for fen, es, eo in zip(fens, elo_self, elo_oppo):
        bi, es_cat, eo_cat, lm = inference.preprocessing(fen, int(es), int(eo), elo_dict, all_moves_dict)
        board_inputs.append(bi)
        legal_moves.append(lm)
        elo_self_cats.append(int(es_cat))
        elo_oppo_cats.append(int(eo_cat))

    board_input = torch.stack(board_inputs, dim=0).to(device)
    legal_moves_t = torch.stack(legal_moves, dim=0).to(device)
    elo_self_t = torch.tensor(elo_self_cats, device=device).long()
    elo_oppo_t = torch.tensor(elo_oppo_cats, device=device).long()
    return board_input, legal_moves_t, elo_self_t, elo_oppo_t


def forward_logits(m: torch.nn.Module, board_input: torch.Tensor, es: torch.Tensor, eo: torch.Tensor) -> torch.Tensor:
    logits_maia, _, _ = m(board_input, es, eo)
    return logits_maia


def uci_to_vocab_index(all_moves_dict: Dict[str, int], fen: str, uci: str) -> int:
    side = fen.split(" ")[1]
    uci_eff = mirror_move(uci) if side == "b" else uci
    return int(all_moves_dict.get(uci_eff, -1))

def gather_logprob(logits_masked: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    # logits_masked already has illegal moves at -inf; safe to log_softmax
    logp_all = torch.log_softmax(logits_masked, dim=-1)
    safe_idx = idxs.clamp(min=0)
    gathered = logp_all.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)
    gathered = torch.where(idxs >= 0, gathered, torch.full_like(gathered, -1e9))
    return gathered


def dpo_loss(logp_pi_ch, logp_pi_rj, logp_ref_ch, logp_ref_rj, beta: float) -> torch.Tensor:
    x = beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj))
    return -torch.nn.functional.logsigmoid(x).mean()


@torch.no_grad()
def kl_policy_base_from_logits(logits_pi_masked: torch.Tensor, logits_ref_masked: torch.Tensor) -> torch.Tensor:
    # KL( pi || ref ) over vocab
    p = torch.softmax(logits_pi_masked, dim=-1)
    logp = torch.log_softmax(logits_pi_masked, dim=-1)
    logq = torch.log_softmax(logits_ref_masked, dim=-1)
    kl = (p * (logp - logq)).sum(dim=-1)  # [B]
    return kl


@torch.no_grad()
def top1_accuracy(logits_masked: torch.Tensor, fens: List[str], all_moves_dict: Dict[str, int], chosen_uci: List[str]) -> torch.Tensor:
    # top1 index in vocab
    top_idx = logits_masked.argmax(dim=-1)  # [B]
    chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, uci) for fen, uci in zip(fens, chosen_uci)],
                              device=logits_masked.device, dtype=torch.long)
    return (top_idx == chosen_idx).float()  # [B]


@torch.no_grad()
def chosen_probability(logits_masked: torch.Tensor, fens: List[str], all_moves_dict: Dict[str, int], chosen_uci: List[str]) -> torch.Tensor:
    probs = torch.softmax(logits_masked, dim=-1)
    chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, uci) for fen, uci in zip(fens, chosen_uci)],
                              device=logits_masked.device, dtype=torch.long)
    safe_idx = chosen_idx.clamp(min=0)
    p = probs.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)
    p = torch.where(chosen_idx >= 0, p, torch.zeros_like(p))
    return p

@torch.no_grad()
def chosen_rank(logits_masked: torch.Tensor, chosen_idx: torch.Tensor) -> torch.Tensor:
    """Rank of chosen move under logits (1 is best). O(B*V) but OK for eval sizes."""
    chosen_idx_safe = chosen_idx.clamp(min=0)
    chosen_logit = logits_masked.gather(1, chosen_idx_safe.view(-1, 1)).squeeze(1)
    greater = (logits_masked > chosen_logit.unsqueeze(1)).sum(dim=1)
    rank = greater + 1
    rank = torch.where(chosen_idx >= 0, rank, torch.full_like(rank, 10**9))
    return rank


@torch.no_grad()
def hit_at_k(logits_masked: torch.Tensor, chosen_idx: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros((logits_masked.size(0),), device=logits_masked.device)
    topk = torch.topk(logits_masked, k=k, dim=-1).indices
    chosen_idx_safe = chosen_idx.clamp(min=0).view(-1, 1)
    hit = (topk == chosen_idx_safe).any(dim=1).float()
    hit = torch.where(chosen_idx >= 0, hit, torch.zeros_like(hit))
    return hit

# ----------------------------
# Stats helpers
# ----------------------------

def fen_to_ply(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return 2 * (fullmove - 1) + (1 if side == "b" else 0)

def ply_to_phase(ply: int) -> str:
    if ply < 20:
        return "opening"
    if ply < 60:
        return "middlegame"
    return "endgame"

def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))

import math
import random
def quantiles(xs: List[float], ps=(0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    if not xs:
        return {f"p{int(p*100)}": float("nan") for p in ps}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    out = {}
    for p in ps:
        k = max(0, min(n - 1, int(math.ceil(p * n) - 1)))
        out[f"p{int(p*100)}"] = float(xs_sorted[k])
    return out

def bootstrap_ci(values: List[float], stat_fn, n_boot=2000, alpha=0.05, seed=0) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rnd = random.Random(seed)
    base = stat_fn(values)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean": float(base), "lo": float(lo), "hi": float(hi)}

def cluster_bootstrap_ci(
    per_rows: List[Dict[str, Any]],
    key: str,
    metric_field: str,
    stat_fn,
    n_boot=2000,
    alpha=0.05,
    seed=0,
) -> Optional[Dict[str, float]]:
    """
    Cluster bootstrap over key (e.g. game_id). Draw games with replacement,
    include all their rows, compute stat over metric_field.
    """
    # group
    groups = defaultdict(list)
    for r in per_rows:
        gid = r.get(key, "")
        if gid:
            groups[gid].append(r)

    if len(groups) < 2:
        return None

    gids = list(groups.keys())
    rnd = random.Random(seed)

    def stat_from_rows(rows: List[Dict[str, Any]]) -> float:
        xs = [float(rr[metric_field]) for rr in rows]
        return stat_fn(xs)

    base = stat_from_rows(per_rows)
    boots = []
    G = len(gids)
    for _ in range(n_boot):
        sampled = [groups[gids[rnd.randrange(G)]] for _ in range(G)]
        flat = [rr for grp in sampled for rr in grp]
        boots.append(stat_from_rows(flat))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean": float(base), "lo": float(lo), "hi": float(hi), "n_clusters": len(groups)}

# ----------------------------
# Opening family (coarse heuristic)
# ----------------------------

def coarse_opening_family_from_prefix(prefix_uci: List[str]) -> str:
    """
    Very coarse: classify based on first black reply or early structure.
    This is a placeholder until you wire ECO book matching.
    """
    if len(prefix_uci) < 2:
        return "Unknown"
    black_reply = prefix_uci[1]
    # replies to 1.e4
    if black_reply == "c7c5":
        return "Sicilian"
    if black_reply == "e7e5":
        return "Open Game (1...e5)"
    if black_reply == "c7c6":
        return "Caro-Kann"
    if black_reply == "e7e6":
        return "French / e6"
    # replies to 1.d4 (common)
    if black_reply == "d7d5":
        return "Queen's Pawn (1...d5)"
    if black_reply == "g8f6":
        return "Indian Defense (1...Nf6)"
    return "Other"

def vocab_index_to_uci(all_moves: List[str], fen: str, idx: int) -> str:
    """
    Convert a vocab index back to a real UCI move for this position.
    Handles Maia's mirroring convention for black-to-move.
    """
    if idx < 0 or idx >= len(all_moves):
        return ""

    uci_eff = all_moves[idx]  # Maia vocab is in "white perspective"
    side = fen.split(" ")[1]  # 'w' or 'b'
    return mirror_move(uci_eff) if side == "b" else uci_eff

# ----------------------------
# Main eval
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/eval/single_gm/eval_sft_maia_single_gm.py --gm_name magnus
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--split_name", required=False, default="val", help="train or val")
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--n_boot", type=int, default=100, help="Number of bootstrap resamples for confidence intervals")
    
    args = ap.parse_args()
    jsonl = Path(f"./processed/single_gm/train_val/{args.gm_name}_{args.split_name}_dpo.jsonl")
    policy_pt = Path(f"./processed/single_gm/train_val/{args.gm_name}/policy_sft_best.pt")
    out_dir = Path(f"./processed/single_gm/train_val/validation_results/{args.gm_name}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_str(args.device)

    # Build vocab + elo dict deterministically (avoid prepare() ordering issues)
    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    # Load base twice; then load policy weights into one
    base = maia_model.from_pretrained(type=args.maia_type, device=str(device)).to(device)
    policy = maia_model.from_pretrained(type=args.maia_type, device=str(device)).to(device)

    sd = torch.load(policy_pt, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (showing 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}")

    base.eval()
    policy.eval()

    opening_probe = probe_opening_distributions_from_policy(
        policy,
        maia_type=args.maia_type,
        device=device,
        all_moves=all_moves,
        all_moves_dict=all_moves_dict,
        elo_dict=elo_dict,
        elo_self=2800,
        elo_oppo=2800,
        temperature=1.0,
    )
    out_dir.joinpath("opening_probe_policy.json").write_text(json.dumps(opening_probe, indent=2))
    print(f"Opening probe saved to {out_dir.joinpath('opening_probe_policy.json')}")

    ds = DpoPairs(jsonl)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    # Aggregate metrics
    n = 0
    sum_dpo = 0.0

    sum_pi_gap = 0.0
    sum_ref_gap = 0.0
    sum_gap_improvement = 0.0

    sum_top1_pi = 0.0
    sum_top1_ref = 0.0

    sum_p_chosen_pi = 0.0
    sum_p_chosen_ref = 0.0

    sum_kl = 0.0

    # NEW: per-row metrics store
    per_rows: List[Dict[str, Any]] = []

    # NEW: phase buckets for tails
    phase_buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    # NEW: opening family distribution (per game)
    opening_by_game: Dict[str, str] = {}



    for batch in loader:
        fens = batch["fen"]
        es = batch["elo_self"]
        eo = batch["elo_oppo"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        bs = len(fens)
        game_ids = batch["game_id"]
        ply_idxs = batch["ply_idx"]
        opening_prefixes = batch["opening_prefix_uci_20"]

        board_input, legal_moves, es_t, eo_t = batch_preprocess(all_moves_dict, elo_dict, fens, es, eo, device)

        logits_pi = forward_logits(policy, board_input, es_t, eo_t)
        logits_ref = forward_logits(base, board_input, es_t, eo_t)

        logits_pi_m = apply_legal_mask(logits_pi, legal_moves)
        logits_ref_m = apply_legal_mask(logits_ref, legal_moves)

        # indices for chosen/rejected
        chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, u) for fen, u in zip(fens, chosen)],
                                  device=device, dtype=torch.long)
        rejected_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, u) for fen, u in zip(fens, rejected)],
                                    device=device, dtype=torch.long)

        chosen_ok = (chosen_idx >= 0) & (legal_moves.gather(1, chosen_idx.clamp(min=0).view(-1,1)).squeeze(1) > 0)
        rejected_ok = (rejected_idx >= 0) & (legal_moves.gather(1, rejected_idx.clamp(min=0).view(-1,1)).squeeze(1) > 0)
        bad = ~(chosen_ok & rejected_ok)
        if bad.any():
            j = int(bad.nonzero()[0])
            raise RuntimeError(f"Illegal chosen/rejected under mask. fen={fens[j]} chosen={chosen[j]} rejected={rejected[j]}")

        logp_pi_ch = gather_logprob(logits_pi_m, chosen_idx)
        logp_pi_rj = gather_logprob(logits_pi_m, rejected_idx)
        logp_ref_ch = gather_logprob(logits_ref_m, chosen_idx)
        logp_ref_rj = gather_logprob(logits_ref_m, rejected_idx)

        loss = dpo_loss(logp_pi_ch, logp_pi_rj, logp_ref_ch, logp_ref_rj, beta=args.beta)

        pi_gap = (logp_pi_ch - logp_pi_rj)          # [B]
        ref_gap = (logp_ref_ch - logp_ref_rj)       # [B]
        gap_improve = (pi_gap - ref_gap)            # [B]

        top1_pi = top1_accuracy(logits_pi_m, fens, all_moves_dict, chosen)
        top1_ref = top1_accuracy(logits_ref_m, fens, all_moves_dict, chosen)

        p_chosen_pi = chosen_probability(logits_pi_m, fens, all_moves_dict, chosen)
        p_chosen_ref = chosen_probability(logits_ref_m, fens, all_moves_dict, chosen)

        kl = kl_policy_base_from_logits(logits_pi_m, logits_ref_m)     # [B]

        # NEW: ranking metrics
        rank_ch = chosen_rank(logits_pi_m, chosen_idx)  # [B]
        hit3 = hit_at_k(logits_pi_m, chosen_idx, k=3)
        hit5 = hit_at_k(logits_pi_m, chosen_idx, k=5)
        hit10 = hit_at_k(logits_pi_m, chosen_idx, k=10)

        # NEW: predicted UCI (top-1)
        pred_idx = logits_pi_m.argmax(dim=-1).tolist()
        pred_uci = [vocab_index_to_uci(all_moves, fen, i) for fen, i in zip(fens, pred_idx)]

        n += bs
        sum_dpo += float(loss) * bs

        sum_pi_gap += float(pi_gap.mean()) * bs
        sum_ref_gap += float(ref_gap.mean()) * bs
        sum_gap_improvement += float(gap_improve.mean()) * bs

        sum_top1_pi += float(top1_pi.mean()) * bs
        sum_top1_ref += float(top1_ref.mean()) * bs

        sum_p_chosen_pi += float(p_chosen_pi.mean()) * bs
        sum_p_chosen_ref += float(p_chosen_ref.mean()) * bs

        sum_kl += float(kl.mean()) * bs

        # NEW: per-row output + phase tails
        for i in range(bs):
            fen = fens[i]
            ply_abs = fen_to_ply(fen)
            phase = ply_to_phase(ply_abs)

            gid = str(game_ids[i] or "")
            if gid and gid not in opening_by_game:
                pref = opening_prefixes[i] or []
                opening_by_game[gid] = coarse_opening_family_from_prefix(pref)

            correct = float(top1_pi[i].item())
            # binary precision/recall/F1 are identical to accuracy in this formulation
            precision = correct
            recall = correct
            f1 = correct

            r = {
                "game_id": gid,
                "ply_idx": int(ply_idxs[i]) if ply_idxs[i] is not None else -1,
                "ply_abs": int(ply_abs),
                "phase": phase,
                "fen": fen,
                "chosen_uci": chosen[i],
                "rejected_uci": rejected[i],
                "pred_uci": pred_uci[i],
                "correct_top1": correct,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hit_top3": float(hit3[i].item()),
                "hit_top5": float(hit5[i].item()),
                "hit_top10": float(hit10[i].item()),
                "rank_chosen": int(rank_ch[i].item()) if rank_ch[i].item() < 1e8 else -1,
                "mrr": float(0.0 if rank_ch[i].item() >= 1e8 else 1.0 / float(rank_ch[i].item())),
                "logp_gap_pi": float(pi_gap[i].item()),
                "logp_gap_ref": float(ref_gap[i].item()),
                "gap_improve": float(gap_improve[i].item()),
                "p_chosen_pi": float(p_chosen_pi[i].item()),
                "p_chosen_ref": float(p_chosen_ref[i].item()),
                "kl_pi_ref": float(kl[i].item()),
                "nll_chosen_pi": float((-logp_pi_ch[i]).item()),
            }
            per_rows.append(r)

            phase_buckets[("kl_pi_ref", phase)].append(r["kl_pi_ref"])
            phase_buckets[("logp_gap_pi", phase)].append(r["logp_gap_pi"])
            phase_buckets[("p_chosen_pi", phase)].append(r["p_chosen_pi"])
            phase_buckets[("correct_top1", phase)].append(r["correct_top1"])



    def avg(x: float) -> float:
        return x / max(1, n)

    # ----------------------------
    # Phase-wise tails (median + p90/p95/p99)
    # ----------------------------
    phase_summary: Dict[str, Dict[str, Any]] = {}
    for (metric, phase), xs in phase_buckets.items():
        phase_summary.setdefault(metric, {})
        phase_summary[metric][phase] = {
            "n": len(xs),
            "mean": mean(xs),
            "median": statistics.median(xs) if xs else float("nan"),
            **quantiles(xs, ps=(0.9, 0.95, 0.99)),
        }

    # ----------------------------
    # Opening fingerprint distribution (per game)
    # ----------------------------
    opening_counts = Counter(opening_by_game.values())
    opening_dist = {k: v for k, v in opening_counts.most_common()}

    # ----------------------------
    # Bootstrap confidence intervals
    # ----------------------------
    acc_vals = [r["correct_top1"] for r in per_rows]
    gap_vals = [r["logp_gap_pi"] for r in per_rows]
    pch_vals = [r["p_chosen_pi"] for r in per_rows]
    mrr_vals = [r["mrr"] for r in per_rows]

    ci_row = {
        "accuracy_top1": bootstrap_ci(acc_vals, mean, n_boot=args.n_boot, seed=0),
        "mean_logp_gap_pi": bootstrap_ci(gap_vals, mean, n_boot=args.n_boot, seed=1),
        "mean_p_chosen_pi": bootstrap_ci(pch_vals, mean, n_boot=args.n_boot, seed=2),
        "mrr": bootstrap_ci(mrr_vals, mean, n_boot=args.n_boot, seed=3),
    }

    # cluster bootstrap if we have game ids
    ci_cluster = {
        "accuracy_top1": cluster_bootstrap_ci(per_rows, "game_id", "correct_top1", mean, n_boot=args.n_boot, seed=10),
        "mean_logp_gap_pi": cluster_bootstrap_ci(per_rows, "game_id", "logp_gap_pi", mean, n_boot=args.n_boot, seed=11),
        "mean_p_chosen_pi": cluster_bootstrap_ci(per_rows, "game_id", "p_chosen_pi", mean, n_boot=args.n_boot, seed=12),
        "mrr": cluster_bootstrap_ci(per_rows, "game_id", "mrr", mean, n_boot=args.n_boot, seed=13),
    }
    # drop Nones
    ci_cluster = {k: v for k, v in ci_cluster.items() if v is not None}


    print("\n=== Eval summary ===")
    print(f"GM: {args.gm_name}")
    print(f"examples: {n}")
    print(f"dpo_loss (policy vs base ref): {avg(sum_dpo):.4f}")
    print("")
    print(f"mean logp_gap policy (chosen - rejected): {avg(sum_pi_gap):.4f}")
    print(f"mean logp_gap base   (chosen - rejected): {avg(sum_ref_gap):.4f}")
    print(f"mean gap improvement (policy - base):     {avg(sum_gap_improvement):.4f}")
    print("")
    print(f"top1 accuracy on chosen (policy): {avg(sum_top1_pi):.4f}")
    print(f"top1 accuracy on chosen (base):   {avg(sum_top1_ref):.4f}")
    print("")
    print(f"mean P(chosen) (policy): {avg(sum_p_chosen_pi):.4f}")
    print(f"mean P(chosen) (base):   {avg(sum_p_chosen_ref):.4f}")
    print("")
    print(f"mean KL(policy || base) over legal moves: {avg(sum_kl):.4f}")
    print("")

    agg = {
        "dpo_loss": avg(sum_dpo),
        "mean_logp_gap_policy_chosen_rejected": avg(sum_pi_gap),
        "mean_logp_gap_base_chosen_rejected": avg(sum_ref_gap),
        "mean_gap_improvement": avg(sum_gap_improvement),
        "top1_accuracy_on_chosen_policy": avg(sum_top1_pi),
        "top1_accuracy_on_chosen_base": avg(sum_top1_ref),
        "mean_p_chosen_policy": avg(sum_p_chosen_pi),
        "mean_p_chosen_base": avg(sum_p_chosen_ref),
        "mean_kl": avg(sum_kl),
    }


    out_dir.joinpath(f"eval_results_sft_{args.split_name}.json").write_text(json.dumps(agg))
    print(f"Eval results saved to {out_dir.joinpath(f'eval_results_sft_{args.split_name}.json')}")
    print(f"Eval results saved to {out_dir.joinpath(f'eval_results_sft_{args.split_name}.json')}")
    # Now we write csv to out_dir.joinpath(f"eval_results_{args.split_name}.csv")

    # 2) Extended JSON: phase tails + CIs + opening dist
    
    ext = {
        **agg,
        "n_rows": len(per_rows),
        "phase_summary": phase_summary,
        "bootstrap_ci_row": ci_row,
        "bootstrap_ci_cluster_by_game": ci_cluster,
        "opening_family_counts_by_game": opening_dist,
        "notes": {
            "opening_family_is_coarse_heuristic": True,
            "precision_recall_f1_equals_accuracy_for_top1_hit": True,
        },
    }
    out_ext = out_dir.joinpath(f"eval_results_sft_extended_{args.split_name}.json")
    out_ext.write_text(json.dumps(ext, indent=2))
    print(f"Extended eval saved to {out_ext}")


    import csv
    with open(out_dir.joinpath(f"eval_results_sft_{args.split_name}.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["dpo_loss", "mean_logp_gap_policy_chosen_rejected", "mean_logp_gap_base_chosen_rejected", "mean_gap_improvement", "top1_accuracy_on_chosen_policy", "top1_accuracy_on_chosen_base", "mean_p_chosen_policy", "mean_p_chosen_base", "mean_kl"])
        writer.writerow([avg(sum_dpo), avg(sum_pi_gap), avg(sum_ref_gap), avg(sum_gap_improvement), avg(sum_top1_pi), avg(sum_top1_ref), avg(sum_p_chosen_pi), avg(sum_p_chosen_ref), avg(sum_kl)])
    print(f"CSV saved to {out_dir.joinpath(f'eval_results_sft_{args.split_name}.csv')}")

    # 4) Per-row metrics CSV
    if per_rows:
        per_csv = out_dir.joinpath(f"eval_per_row_metrics_sft_{args.split_name}.csv")
        fieldnames = list(per_rows[0].keys())
        with open(per_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_rows)
        print(f"Per-row metrics saved to {per_csv}")

if __name__ == "__main__":
    main()
