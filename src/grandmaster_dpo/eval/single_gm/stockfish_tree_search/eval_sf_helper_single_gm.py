#!/usr/bin/env python3
"""
eval_sf_helper_sweep.py

SF-helper evaluation for multiple GMs, multiple finetuned checkpoints, multiple Stockfish Elo settings,
and multiple MultiPV top-K values.

What this script does
---------------------
For each GM in --gms (and corresponding finetuned checkpoint in --policy_pts), it evaluates:

1) Base Maia2 model + Stockfish candidate filter ("SF-helper")
2) Finetuned Maia2 model + Stockfish candidate filter ("SF-helper")

Across sweeps:
- Stockfish Elo levels: --sf_elos (e.g., "none,1600,2000,2400,2800")
- MultiPV top-K values: --topks (e.g., "5,10,15,20,25")

It prints and saves, for each (GM, sf_elo, topk), *two blocks* per model:
- Policy + Stockfish-help (candidate filter)
- SF-helper conditional logp metrics (q = pi restricted to candidates)

Efficiency
----------
- Uses *all CPU cores* (or --workers) by parallelizing Stockfish analysis across processes.
  Each worker owns its own Stockfish engine instance (safe).
- Models run on CPU by default; you can set --device cuda/mps if your Maia2 supports it.
  Stockfish analysis remains multiprocessing.

Determinism vs stochasticity
----------------------------
- By default selection within candidate set is deterministic argmax (no sampling).
- If --sample is set, selection is stochastic via sampling from q (seeded with --seed).
- Stockfish with fixed depth is typically deterministic; still, different builds may vary slightly.

Granular per-row output
-----------------------
For each (GM, sf_elo, topk), saves a JSONL containing per-position records so you can derive custom metrics later.
Records include:
- fen, chosen_uci, rejected_uci
- ply_abs (0=white first move, 1=black first reply, ...)
- SF candidates [(uci, cp)], best_cp
- For each model (base/ft):
    selected_uci, is_best, cp_gap, entropy
    p_chosen_cond, p_rejected_cond, gap_logp_cond
    full_hit@1/@5/@10 on chosen (from full legal-move distribution)
    cand_hit@1/@5/@10 on chosen (within candidate distribution q)
    logp_selected_full (under full masked policy)
    kl_q_vs_base (KL(q || base_full) over candidate support)
- Opening move distributions:
    - selected move distribution at ply_abs==0 (white’s first move)
    - selected move distribution at ply_abs==1 (black’s first reply)

Inputs
------
By default expects the same dataset path pattern you’ve been using:
  ./processed/single_gm/train_val/{gm}_{split}_dpo.jsonl

You can override via --jsonl_template, e.g.:
  --jsonl_template "/path/to/{gm}_{split}.jsonl"

Outputs
-------
For each GM:
  {out_root}/{gm}/sf_helper_eval_{ft_tag}_{split}/
      summary__sfelo={sf_elo}__topk={K}.json
      per_row__sfelo={sf_elo}__topk={K}.jsonl
      opening__sfelo={sf_elo}__topk={K}.json

Example
-------
python eval_sf_helper_sweep.py \
  --gms firouzja,nakamura \
  --policy_pts ./.../firouzja/policy_dpo_best.pt,./.../nakamura/policy_dpo_best.pt \
  --ft_tag dpo \
  --split val \
  --maia_type blitz \
  --device cpu \
  --sf_path /usr/local/bin/stockfish \
  --sf_elos none,1600,2000,2400,2800 \
  --topks 5,10,15,20,25 \
  --restrict_cp_window 60 \
  --sf_depth 10 \
  --workers 0 \
  --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine
from grandmaster_dpo.utilities.shared_style_emb_model_utils import pick_device
import torch
from torch.utils.data import DataLoader, Dataset

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move
from grandmaster_dpo.eval.stockfish_helpers import make_stockfish


# ----------------------------
# Dataset
# ----------------------------

class DpoPairs(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        self.rows = self.rows[:500]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        p = r.get("prompt", {}) or {}
        meta = r.get("meta", {}) or {}
        gh = meta.get("game_header_hash")
        game_id = str(gh)
        return {
            "fen": p["fen"],
            "elo_self": int(p.get("elo_self", 2800)),
            "elo_oppo": int(p.get("elo_oppo", 2800)),
            "chosen": r["chosen"],
            "rejected": r["rejected"],
            "game_id": game_id,
            "ply_idx": int(meta.get("ply_idx", -1)),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {
        "fen": [],
        "elo_self": [],
        "elo_oppo": [],
        "chosen": [],
        "rejected": [],
        "game_id": [],
        "ply_idx": [],
    }
    for b in batch:
        for k in out:
            out[k].append(b.get(k))
    return out


# ----------------------------
# Core helpers (mirroring + indexing)
# ----------------------------

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

def mirror_uci_like_board_mirror(uci: str) -> str:
    mv = chess.Move.from_uci(uci)
    f = chess.square_mirror(mv.from_square)
    t = chess.square_mirror(mv.to_square)
    return chess.Move(f, t, promotion=mv.promotion).uci()

def uci_to_vocab_index(all_moves_dict: Dict[str, int], fen: str, uci: str) -> int:
    side = fen.split(" ")[1]
    uci_eff = mirror_uci_like_board_mirror(uci) if side == "b" else uci
    return int(all_moves_dict.get(uci_eff, -1))

def vocab_index_to_uci(all_moves: List[str], fen: str, idx: int) -> str:
    if idx < 0 or idx >= len(all_moves):
        return ""
    uci_eff = all_moves[idx]
    side = fen.split(" ")[1]
    return mirror_move(uci_eff) if side == "b" else uci_eff

def fen_to_ply_abs(fen: str) -> int:
    parts = fen.split()
    side = parts[1]          # 'w' or 'b'
    fullmove = int(parts[5]) # 1-based
    # ply_abs: 0=white to move at fullmove=1, 1=black to move at fullmove=1, ...
    return 2 * (fullmove - 1) + (1 if side == "b" else 0)


# ----------------------------
# Preprocess / forward
# ----------------------------

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
        bi, es_cat, eo_cat, lm = inference.preprocessing(
            fen, int(es), int(eo), elo_dict, all_moves_dict
        )
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

def gather_logprob_from_masked_logits(logits_masked: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    """log p(idxs) under full masked distribution."""
    logp_all = torch.log_softmax(logits_masked, dim=-1)
    safe = idxs.clamp(min=0)
    out = logp_all.gather(1, safe.view(-1, 1)).squeeze(1)
    out = torch.where(idxs >= 0, out, torch.full_like(out, torch.finfo(out.dtype).min))
    return out

def full_hit_at_k(logits_masked: torch.Tensor, chosen_idx: torch.Tensor, ks: Tuple[int, ...]) -> Dict[int, torch.Tensor]:
    """
    Returns dict k -> hit@k tensor [B], where hit@k is 1 if chosen is in top-k moves of full masked distribution.
    """
    max_k = max(ks)
    topk = torch.topk(logits_masked, k=max_k, dim=-1).indices  # [B,max_k]
    chosen_safe = chosen_idx.clamp(min=0).view(-1, 1)
    hits = {}
    for k in ks:
        hit = (topk[:, :k] == chosen_safe).any(dim=1).float()
        hit = torch.where(chosen_idx >= 0, hit, torch.zeros_like(hit))
        hits[k] = hit
    return hits


# ----------------------------
# Stockfish multiprocessing
# ----------------------------

_SF_ENGINE: Optional[chess.engine.SimpleEngine] = None
_SF_LIMIT: Optional[chess.engine.Limit] = None
_SF_MULTIPV: int = 10

def _score_to_cp(score: chess.engine.PovScore, mate_score: int = 100_000) -> int:
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
        m = rel.mate()
        if m is not None:
            return mate_score if m > 0 else -mate_score
        return 0
    return int(cp)

def _sf_worker_init(
    stockfish_path: str,
    threads: int,
    hash_mb: int,
    uci_elo: Optional[int],
    depth: int,
    multipv: int,
    timeout: float,
) -> None:
    global _SF_ENGINE, _SF_LIMIT, _SF_MULTIPV
    _SF_ENGINE = make_stockfish(
        stockfish_path,
        threads=threads,
        hash_mb=hash_mb,
        uci_elo=uci_elo,      # may be ignored if build doesn't support
        skill_level=None,
        timeout=timeout,
    )
    _SF_LIMIT = chess.engine.Limit(depth=int(depth))
    _SF_MULTIPV = int(multipv)

def _sf_worker_analyse_fen(fen: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "fen": fen,
        "is_game_over": bool,
        "cands": [(uci, cp), ...]  # up to multipv
        "best_cp": int,
      }
    """
    global _SF_ENGINE, _SF_LIMIT, _SF_MULTIPV
    assert _SF_ENGINE is not None and _SF_LIMIT is not None
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=True):
        return {"fen": fen, "is_game_over": True, "cands": [], "best_cp": 0}

    infos = _SF_ENGINE.analyse(board, _SF_LIMIT, multipv=_SF_MULTIPV)

    cands: List[Tuple[str, int]] = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        uci = pv[0].uci()
        cp = _score_to_cp(score)
        cands.append((uci, cp))

    if not cands:
        return {"fen": fen, "is_game_over": False, "cands": [], "best_cp": 0}

    best_cp = max(cp for _, cp in cands)
    return {"fen": fen, "is_game_over": False, "cands": cands, "best_cp": int(best_cp)}

def _sf_worker_shutdown() -> None:
    global _SF_ENGINE
    if _SF_ENGINE is not None:
        try:
            _SF_ENGINE.quit()
        except Exception:
            pass
        _SF_ENGINE = None


# ----------------------------
# SF-helper evaluation logic for one model on one batch
# ----------------------------

@dataclass
class ModelBatchResult:
    # selection stats
    selected_uci: str
    is_best_sf: bool
    cp_selected: int
    cp_best: int
    cp_gap: float
    entropy: float
    logp_selected_full: float

    # conditional probs (q over candidates)
    p_chosen_cond: float
    p_rejected_cond: float
    logp_chosen_cond: float
    logp_rejected_cond: float
    gap_logp_cond: float

    # hits full distribution
    full_hit1: float
    full_hit5: float
    full_hit10: float

    # hits candidate distribution
    cand_hit1: float
    cand_hit5: float
    cand_hit10: float

    # divergence
    kl_q_vs_base: float


def _entropy(probs: List[float], eps: float) -> float:
    if not probs:
        return 0.0
    s = 0.0
    for p in probs:
        pp = max(p, eps)
        s -= pp * math.log(pp)
    return float(s)

@torch.no_grad()
def compute_sf_helper_for_one_position(
    *,
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    cands: List[Tuple[str, int]],
    best_cp: int,
    logits_masked: torch.Tensor,        # [V]
    base_logp_full: torch.Tensor,       # [V] log-softmax of base masked logits
    all_moves_dict: Dict[str, int],
    restrict_cp_window: Optional[int],
    temperature: float,
    sample: bool,
    rng: random.Random,
    eps: float,
    # precomputed full hits for chosen
    full_hit: Dict[int, int],
) -> Optional[Tuple[ModelBatchResult, Dict[str, Any]]]:
    """
    Returns (ModelBatchResult, debug_blob) or None if invalid / no candidates.
    debug_blob contains candidate probs aligned to cands (post-filter), for JSONL.
    """
    if not cands:
        return None

    # filter by cp window
    kept = cands
    if restrict_cp_window is not None:
        w = int(restrict_cp_window)
        filt = [(m, cp) for (m, cp) in kept if cp >= best_cp - w]
        if filt:
            kept = filt

    if not kept:
        return None

    # q distribution over candidates from policy logits
    t = max(float(temperature), 1e-6)
    logp_all = torch.log_softmax(logits_masked / t, dim=-1)  # [V]

    cand_idxs: List[int] = []
    cand_logps: List[torch.Tensor] = []
    for (uci, _cp) in kept:
        idx = uci_to_vocab_index(all_moves_dict, fen, uci)
        cand_idxs.append(idx)
        if idx < 0:
            cand_logps.append(torch.tensor(torch.finfo(logp_all.dtype).min, device=logp_all.device))
        else:
            cand_logps.append(logp_all[idx])

    cand_logps_t = torch.stack(cand_logps, dim=0)  # [K]
    cand_probs_t = torch.softmax(cand_logps_t, dim=0)  # q over candidates
    cand_probs = cand_probs_t.detach().cpu().tolist()

    # selection (deterministic argmax or stochastic sample)
    if sample:
        # python RNG to choose index based on probs (stable across torch versions)
        r = rng.random()
        acc = 0.0
        sel_i = 0
        for j, p in enumerate(cand_probs):
            acc += float(p)
            if r <= acc:
                sel_i = j
                break
    else:
        sel_i = int(torch.argmax(cand_probs_t).item())

    selected_uci, cp_selected = kept[sel_i]
    cp_gap = float(best_cp - cp_selected)
    is_best = (cp_gap <= 1e-9)

    ent = _entropy(cand_probs, eps)

    # logp(selected) under full distribution
    sel_idx = uci_to_vocab_index(all_moves_dict, fen, selected_uci)
    if sel_idx >= 0:
        logp_selected_full = float(logp_all[sel_idx].item())
    else:
        logp_selected_full = float("-inf")

    # conditional probs for chosen / rejected (0 if absent)
    cand_moves = [m for (m, _cp) in kept]
    p_ch = float(cand_probs[cand_moves.index(chosen_uci)]) if chosen_uci in cand_moves else 0.0
    p_rj = float(cand_probs[cand_moves.index(rejected_uci)]) if rejected_uci in cand_moves else 0.0

    # FIX: clamp before log; missing => 0 but log uses eps
    logp_ch = math.log(max(p_ch, eps))
    logp_rj = math.log(max(p_rj, eps))
    gap_logp = float(logp_ch - logp_rj)

    # candidate hits for chosen (rank in candidate q)
    # get top indices in candidate probs
    K = len(cand_probs)
    order = sorted(range(K), key=lambda j: cand_probs[j], reverse=True)
    def cand_hit_at(k: int) -> float:
        if k <= 0:
            return 0.0
        k = min(k, K)
        top_moves = [cand_moves[j] for j in order[:k]]
        return 1.0 if chosen_uci in top_moves else 0.0

    # KL(q || base_full) over candidate support:
    # KL = Σ q(m) [log q(m) - log p_base(m)]
    kl = 0.0
    for (uci, _cp), q in zip(kept, cand_probs):
        idx = uci_to_vocab_index(all_moves_dict, fen, uci)
        if idx < 0:
            continue
        logq = math.log(max(float(q), eps))
        logp_b = float(base_logp_full[idx].item())
        kl += float(q) * (logq - logp_b)

    res = ModelBatchResult(
        selected_uci=selected_uci,
        is_best_sf=is_best,
        cp_selected=int(cp_selected),
        cp_best=int(best_cp),
        cp_gap=float(cp_gap),
        entropy=float(ent),
        logp_selected_full=float(logp_selected_full),

        p_chosen_cond=float(p_ch),
        p_rejected_cond=float(p_rj),
        logp_chosen_cond=float(logp_ch),
        logp_rejected_cond=float(logp_rj),
        gap_logp_cond=float(gap_logp),

        full_hit1=float(full_hit[1]),
        full_hit5=float(full_hit[5]),
        full_hit10=float(full_hit[10]),

        cand_hit1=float(cand_hit_at(1)),
        cand_hit5=float(cand_hit_at(5)),
        cand_hit10=float(cand_hit_at(10)),

        kl_q_vs_base=float(kl),
    )

    debug_blob = {
        "cands_kept": kept,             # [(uci,cp)]
        "q_probs": cand_probs,          # aligned with cands_kept
        "selected_index": int(sel_i),
    }
    return res, debug_blob


# ----------------------------
# Aggregation & reporting
# ----------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")

def print_blocks(tag: str, agg: Dict[str, Any]) -> None:
    print(f"=== {tag}: Policy + Stockfish-help (candidate filter) ===")
    print(f"sf_valid_rows: {agg['sf_valid_rows']}/{agg['sf_total_rows']}")
    print(f"top1 accuracy vs chosen:           {agg['sf_help_top1_acc']:.4f}")
    print(f"top5 hit vs chosen (cand q):       {agg['sf_help_top5_hit_cand']:.4f}")
    print(f"top10 hit vs chosen (cand q):      {agg['sf_help_top10_hit_cand']:.4f}")
    print(f"mean cp_gap vs best SF:            {agg['sf_help_mean_cp_gap']:.2f}")
    print(f"picked best-SF move rate:          {agg['sf_help_best_sf_rate']:.4f}")
    print(f"mean candidate-dist entropy:       {agg['sf_help_mean_entropy']:.4f}")
    print(f"mean logp(policy(selected)):       {agg['sf_help_mean_logp_selected_full']:.4f}")
    print(f"chosen-in-candidates rate:         {agg['sf_help_chosen_in_cands_rate']:.4f}")
    print(f"mean P(chosen | in candidates):    {agg['sf_help_mean_p_chosen_given_in']:.4f}")
    print("")
    print(f"=== {tag}: SF-helper conditional logp metrics (q = pi restricted to cands) ===")
    print(f"mean P(chosen | cands):            {agg['sf_help_mean_p_chosen_cond']:.6f}")
    print(f"mean P(rejected | cands):          {agg['sf_help_mean_p_rejected_cond']:.6f}")
    print(f"mean logp(chosen | cands):         {agg['sf_help_mean_logp_chosen_cond']:.4f}")
    print(f"mean logp(rejected | cands):       {agg['sf_help_mean_logp_rejected_cond']:.4f}")
    print(f"mean gap logp(ch) - logp(rj):      {agg['sf_help_mean_gap_logp_cond']:.4f}")
    print(f"mean KL(q || base_full):           {agg['sf_help_mean_kl_q_vs_base']:.4f}")
    print("")
    print(f"--- Full-policy hit@k on chosen (legal-move distro) ---")
    print(f"full hit@1:                        {agg['full_hit1']:.4f}")
    print(f"full hit@5:                        {agg['full_hit5']:.4f}")
    print(f"full hit@10:                       {agg['full_hit10']:.4f}")
    print("")


def opening_summary(opening_counts: Dict[str, Counter], topn: int = 30) -> Dict[str, Any]:
    out = {}
    for k, ctr in opening_counts.items():
        out[k] = [{"uci": u, "count": c} for (u, c) in ctr.most_common(topn)]
    return out


# ----------------------------
# Main driver
# ----------------------------

def parse_csv_list(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in parse_csv_list(s)]

def parse_sf_elos(s: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for x in parse_csv_list(s):
        xl = x.lower()
        if xl in ("none", "null", "full", "max"):
            out.append(None)
        else:
            out.append(int(x))
    return out

def load_policy_weights_into(model: torch.nn.Module, pt_path: str) -> None:
    sd = torch.load(pt_path, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (showing 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gms", required=True, help="Comma-separated GM names.")
    ap.add_argument("--policy_pts", required=True, help="Comma-separated .pt paths, same order as --gms.")
    ap.add_argument("--ft_tag", required=True, help="Nickname like sft or dpo.")

    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--jsonl_template", default="./processed/single_gm/train_val/{gm}_{split}_dpo.jsonl")
    ap.add_argument("--out_root", default="./processed/single_gm/train_val/validation_results")

    # Stockfish settings
    ap.add_argument("--sf_path", default="/usr/local/bin/stockfish")
    ap.add_argument("--sf_threads_per_worker", type=int, default=1,
                    help="Threads per Stockfish process. Usually 1 when using many processes.")
    ap.add_argument("--sf_hash_mb", type=int, default=128)
    ap.add_argument("--sf_depth", type=int, default=10)
    ap.add_argument("--sf_timeout", type=float, default=30.0)

    ap.add_argument("--sf_elos", default="none",
                    help='Comma list, e.g. "none,1600,2000,2400,2800". Use "none" for full strength.')

    # MultiPV topK sweep
    ap.add_argument("--topks", default="10", help="Comma list, e.g. 5,10,15,20,25")

    # Candidate filtering + policy sampling
    ap.add_argument("--restrict_cp_window", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sample", action="store_true", help="If set, sample from q; default deterministic argmax.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-12)

    # Multiprocessing
    ap.add_argument("--workers", type=int, default=0, help="0 => use os.cpu_count()")

    ap.add_argument("--max_rows", type=int, default=0, help="0 => all rows; else truncate dataset for quick runs.")

    args = ap.parse_args()

    gms = parse_csv_list(args.gms)
    pts = parse_csv_list(args.policy_pts)
    if len(gms) != len(pts):
        raise ValueError(f"--gms has {len(gms)} items but --policy_pts has {len(pts)} items")

    sf_elos = parse_sf_elos(args.sf_elos)
    topks = parse_int_list(args.topks)
    if not topks:
        raise ValueError("No --topks provided")

    device = pick_device("auto")

    # seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # vocab + elo dict (stable)
    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    # workers
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    for gm, pt_path in zip(gms, pts):
        print(f"\n==============================")
        print(f"GM: {gm}")
        print(f"checkpoint: {pt_path}")
        print(f"split: {args.split}")
        print(f"==============================\n")

        jsonl_path = args.jsonl_template.format(gm=gm, split=args.split)
        ds = DpoPairs(jsonl_path)
        if args.max_rows and args.max_rows > 0:
            ds.rows = ds.rows[: args.max_rows]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

        # models
        base_model = maia_model.from_pretrained(type=args.maia_type, device=str(device)).to(device)
        ft_model = maia_model.from_pretrained(type=args.maia_type, device=str(device)).to(device)
        load_policy_weights_into(ft_model, pt_path)
        base_model.eval()
        ft_model.eval()

        gm_out_dir = Path(args.out_root) / gm / f"sf_helper_eval_{args.ft_tag}_{args.split}"
        gm_out_dir.mkdir(parents=True, exist_ok=True)

        # Sweep SF Elo
        for sf_elo in sf_elos:
            sf_elo_tag = "none" if sf_elo is None else str(sf_elo)

            # Sweep MultiPV topK
            for topk in topks:
                print(f"\n--- Running sf_elo={sf_elo_tag} topk={topk} ---\n")

                # multiprocessing pool for SF analysis
                # NOTE: each worker has its own engine
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                pool = ctx.Pool(
                    processes=workers,
                    initializer=_sf_worker_init,
                    initargs=(
                        args.sf_path,
                        int(args.sf_threads_per_worker),
                        int(args.sf_hash_mb),
                        sf_elo,
                        int(args.sf_depth),
                        int(topk),
                        float(args.sf_timeout),
                    ),
                )

                # aggregates per model
                def make_agg() -> Dict[str, Any]:
                    return {
                        "sf_total_rows": 0,
                        "sf_valid_rows": 0,
                        "sf_help_top1_acc": 0.0,
                        "sf_help_top5_hit_cand": 0.0,
                        "sf_help_top10_hit_cand": 0.0,
                        "sf_help_mean_cp_gap": 0.0,
                        "sf_help_best_sf_rate": 0.0,
                        "sf_help_mean_entropy": 0.0,
                        "sf_help_mean_logp_selected_full": 0.0,
                        "sf_help_chosen_in_cands": 0.0,
                        "sf_help_p_chosen_in_cands_sum": 0.0,

                        "sf_help_mean_p_chosen_cond": 0.0,
                        "sf_help_mean_p_rejected_cond": 0.0,
                        "sf_help_mean_logp_chosen_cond": 0.0,
                        "sf_help_mean_logp_rejected_cond": 0.0,
                        "sf_help_mean_gap_logp_cond": 0.0,
                        "sf_help_mean_kl_q_vs_base": 0.0,

                        "full_hit1": 0.0,
                        "full_hit5": 0.0,
                        "full_hit10": 0.0,
                    }

                agg_ft = make_agg()
                agg_base = make_agg()

                # opening distributions at ply_abs==0 and ply_abs==1
                opening_counts_ft: Dict[str, Counter] = {"white_ply0": Counter(), "black_ply1": Counter()}
                opening_counts_base: Dict[str, Counter] = {"white_ply0": Counter(), "black_ply1": Counter()}

                # per-row jsonl
                per_row_path = gm_out_dir / f"per_row__sfelo={sf_elo_tag}__topk={topk}.jsonl"
                f_per = open(per_row_path, "w", encoding="utf-8")

                # deterministic sampling across rows when --sample
                row_rng_ft = random.Random(args.seed + 1000)
                row_rng_base = random.Random(args.seed + 2000)

                try:
                    for batch_idx, batch in enumerate(loader):
                        fens = batch["fen"]
                        es = batch["elo_self"]
                        eo = batch["elo_oppo"]
                        chosen = batch["chosen"]
                        rejected = batch["rejected"]
                        bs = len(fens)

                        # Run Stockfish in parallel for this batch
                        sf_results = pool.map(_sf_worker_analyse_fen, fens, chunksize=max(1, bs // (workers * 2) or 1))

                        # Compute logits for both models (single forward per model)
                        board_input, legal_moves, es_t, eo_t = batch_preprocess(all_moves_dict, elo_dict, fens, es, eo, device)

                        logits_base = forward_logits(base_model, board_input, es_t, eo_t)
                        logits_ft = forward_logits(ft_model, board_input, es_t, eo_t)

                        logits_base_m = apply_legal_mask(logits_base, legal_moves)
                        logits_ft_m = apply_legal_mask(logits_ft, legal_moves)

                        # base logp full distribution used for KL(q||base_full)
                        logp_base_full = torch.log_softmax(logits_base_m, dim=-1)  # [B,V]

                        # Full hit@k for chosen under each model
                        chosen_idx = torch.tensor(
                            [uci_to_vocab_index(all_moves_dict, fen, u) for fen, u in zip(fens, chosen)],
                            device=device,
                            dtype=torch.long,
                        )
                        hits_base = full_hit_at_k(logits_base_m, chosen_idx, ks=(1, 5, 10))
                        hits_ft = full_hit_at_k(logits_ft_m, chosen_idx, ks=(1, 5, 10))

                        for i in range(bs):
                            fen_i = fens[i]
                            chosen_i = chosen[i]
                            rejected_i = rejected[i]
                            ply_abs = fen_to_ply_abs(fen_i)

                            sr = sf_results[i]
                            cands = sr["cands"]
                            best_cp = int(sr["best_cp"])
                            is_over = bool(sr["is_game_over"])

                            # record "total rows" even if invalid so sf_valid_rows makes sense
                            agg_base["sf_total_rows"] += 1
                            agg_ft["sf_total_rows"] += 1

                            if is_over or not cands:
                                # still write a minimal row so you can filter later
                                f_per.write(json.dumps({
                                    "fen": fen_i,
                                    "ply_abs": ply_abs,
                                    "chosen_uci": chosen_i,
                                    "rejected_uci": rejected_i,
                                    "sf": {"is_game_over": is_over, "cands": cands, "best_cp": best_cp},
                                    "base": None,
                                    args.ft_tag: None,
                                }) + "\n")
                                continue

                            # base model eval for this position
                            base_out = compute_sf_helper_for_one_position(
                                fen=fen_i,
                                chosen_uci=chosen_i,
                                rejected_uci=rejected_i,
                                cands=cands,
                                best_cp=best_cp,
                                logits_masked=logits_base_m[i],
                                base_logp_full=logp_base_full[i],
                                all_moves_dict=all_moves_dict,
                                restrict_cp_window=args.restrict_cp_window,
                                temperature=args.temperature,
                                sample=args.sample,
                                rng=row_rng_base,
                                eps=args.eps,
                                full_hit={
                                    1: int(hits_base[1][i].item()),
                                    5: int(hits_base[5][i].item()),
                                    10: int(hits_base[10][i].item()),
                                },
                            )

                            # finetuned model eval for this position (KL still vs base_full)
                            ft_out = compute_sf_helper_for_one_position(
                                fen=fen_i,
                                chosen_uci=chosen_i,
                                rejected_uci=rejected_i,
                                cands=cands,
                                best_cp=best_cp,
                                logits_masked=logits_ft_m[i],
                                base_logp_full=logp_base_full[i],
                                all_moves_dict=all_moves_dict,
                                restrict_cp_window=args.restrict_cp_window,
                                temperature=args.temperature,
                                sample=args.sample,
                                rng=row_rng_ft,
                                eps=args.eps,
                                full_hit={
                                    1: int(hits_ft[1][i].item()),
                                    5: int(hits_ft[5][i].item()),
                                    10: int(hits_ft[10][i].item()),
                                },
                            )

                            if base_out is None and ft_out is None:
                                f_per.write(json.dumps({
                                    "fen": fen_i,
                                    "ply_abs": ply_abs,
                                    "chosen_uci": chosen_i,
                                    "rejected_uci": rejected_i,
                                    "sf": {"is_game_over": is_over, "cands": cands, "best_cp": best_cp},
                                    "base": None,
                                    args.ft_tag: None,
                                }) + "\n")
                                continue

                            # aggregate helper
                            def update_agg(agg: Dict[str, Any], res: ModelBatchResult) -> None:
                                agg["sf_valid_rows"] += 1
                                agg["sf_help_top1_acc"] += 1.0 if res.selected_uci == chosen_i else 0.0
                                agg["sf_help_top5_hit_cand"] += float(res.cand_hit5)
                                agg["sf_help_top10_hit_cand"] += float(res.cand_hit10)
                                agg["sf_help_mean_cp_gap"] += float(res.cp_gap)
                                agg["sf_help_best_sf_rate"] += 1.0 if res.is_best_sf else 0.0
                                agg["sf_help_mean_entropy"] += float(res.entropy)
                                agg["sf_help_mean_logp_selected_full"] += float(res.logp_selected_full)

                                # chosen-in-cands & p(chosen|in)
                                # (chosen in candidates iff p_chosen_cond>0)
                                chosen_in = 1.0 if res.p_chosen_cond > 0.0 else 0.0
                                agg["sf_help_chosen_in_cands"] += chosen_in
                                if chosen_in:
                                    agg["sf_help_p_chosen_in_cands_sum"] += float(res.p_chosen_cond)

                                agg["sf_help_mean_p_chosen_cond"] += float(res.p_chosen_cond)
                                agg["sf_help_mean_p_rejected_cond"] += float(res.p_rejected_cond)
                                agg["sf_help_mean_logp_chosen_cond"] += float(res.logp_chosen_cond)
                                agg["sf_help_mean_logp_rejected_cond"] += float(res.logp_rejected_cond)
                                agg["sf_help_mean_gap_logp_cond"] += float(res.gap_logp_cond)
                                agg["sf_help_mean_kl_q_vs_base"] += float(res.kl_q_vs_base)

                                agg["full_hit1"] += float(res.full_hit1)
                                agg["full_hit5"] += float(res.full_hit5)
                                agg["full_hit10"] += float(res.full_hit10)

                            base_res, base_dbg = base_out if base_out is not None else (None, None)
                            ft_res, ft_dbg = ft_out if ft_out is not None else (None, None)

                            if base_res is not None:
                                update_agg(agg_base, base_res)
                                if ply_abs == 0:
                                    opening_counts_base["white_ply0"][base_res.selected_uci] += 1
                                if ply_abs == 1:
                                    opening_counts_base["black_ply1"][base_res.selected_uci] += 1

                            if ft_res is not None:
                                update_agg(agg_ft, ft_res)
                                if ply_abs == 0:
                                    opening_counts_ft["white_ply0"][ft_res.selected_uci] += 1
                                if ply_abs == 1:
                                    opening_counts_ft["black_ply1"][ft_res.selected_uci] += 1

                            # write per-row for custom analysis
                            f_per.write(json.dumps({
                                "fen": fen_i,
                                "ply_abs": ply_abs,
                                "chosen_uci": chosen_i,
                                "rejected_uci": rejected_i,
                                "sf": {
                                    "is_game_over": is_over,
                                    "best_cp": best_cp,
                                    "cands_raw": cands,
                                },
                                "base": None if base_res is None else {
                                    **base_res.__dict__,
                                    **base_dbg,
                                },
                                args.ft_tag: None if ft_res is None else {
                                    **ft_res.__dict__,
                                    **ft_dbg,
                                },
                            }) + "\n")

                finally:
                    f_per.close()
                    pool.close()
                    pool.terminate()
                    pool.join()

                # finalize averages
                def finalize(agg: Dict[str, Any]) -> Dict[str, Any]:
                    v = int(agg["sf_valid_rows"])
                    total = int(agg["sf_total_rows"])
                    out = dict(agg)
                    out["sf_total_rows"] = total
                    out["sf_valid_rows"] = v

                    out["sf_help_top1_acc"] = _safe_div(out["sf_help_top1_acc"], v)
                    out["sf_help_top5_hit_cand"] = _safe_div(out["sf_help_top5_hit_cand"], v)
                    out["sf_help_top10_hit_cand"] = _safe_div(out["sf_help_top10_hit_cand"], v)
                    out["sf_help_mean_cp_gap"] = _safe_div(out["sf_help_mean_cp_gap"], v)
                    out["sf_help_best_sf_rate"] = _safe_div(out["sf_help_best_sf_rate"], v)
                    out["sf_help_mean_entropy"] = _safe_div(out["sf_help_mean_entropy"], v)
                    out["sf_help_mean_logp_selected_full"] = _safe_div(out["sf_help_mean_logp_selected_full"], v)

                    out["sf_help_chosen_in_cands_rate"] = _safe_div(out["sf_help_chosen_in_cands"], v)
                    out["sf_help_mean_p_chosen_given_in"] = _safe_div(out["sf_help_p_chosen_in_cands_sum"], max(1.0, out["sf_help_chosen_in_cands"]))

                    out["sf_help_mean_p_chosen_cond"] = _safe_div(out["sf_help_mean_p_chosen_cond"], v)
                    out["sf_help_mean_p_rejected_cond"] = _safe_div(out["sf_help_mean_p_rejected_cond"], v)
                    out["sf_help_mean_logp_chosen_cond"] = _safe_div(out["sf_help_mean_logp_chosen_cond"], v)
                    out["sf_help_mean_logp_rejected_cond"] = _safe_div(out["sf_help_mean_logp_rejected_cond"], v)
                    out["sf_help_mean_gap_logp_cond"] = _safe_div(out["sf_help_mean_gap_logp_cond"], v)
                    out["sf_help_mean_kl_q_vs_base"] = _safe_div(out["sf_help_mean_kl_q_vs_base"], v)

                    out["full_hit1"] = _safe_div(out["full_hit1"], v)
                    out["full_hit5"] = _safe_div(out["full_hit5"], v)
                    out["full_hit10"] = _safe_div(out["full_hit10"], v)
                    return out

                agg_base_f = finalize(agg_base)
                agg_ft_f = finalize(agg_ft)

                print_blocks(args.ft_tag, agg_ft_f)
                print_blocks("base", agg_base_f)

                # deltas
                def delta(key: str) -> float:
                    return float(agg_ft_f.get(key, 0.0)) - float(agg_base_f.get(key, 0.0))

                print("=== Delta (finetuned - base) ===")
                print(f"top1 accuracy vs chosen:           {delta('sf_help_top1_acc'):+.4f}")
                print(f"top5 hit vs chosen (cand q):       {delta('sf_help_top5_hit_cand'):+.4f}")
                print(f"top10 hit vs chosen (cand q):      {delta('sf_help_top10_hit_cand'):+.4f}")
                print(f"mean cp_gap vs best SF:            {delta('sf_help_mean_cp_gap'):+.2f}")
                print(f"picked best-SF move rate:          {delta('sf_help_best_sf_rate'):+.4f}")
                print(f"chosen-in-candidates rate:         {delta('sf_help_chosen_in_cands_rate'):+.4f}")
                print(f"mean gap logp(ch)-logp(rj):        {delta('sf_help_mean_gap_logp_cond'):+.4f}")
                print(f"mean KL(q||base_full):             {delta('sf_help_mean_kl_q_vs_base'):+.4f}")
                print("")

                # save summary json
                summary = {
                    "gm": gm,
                    "ft_tag": args.ft_tag,
                    "split": args.split,
                    "maia_type": args.maia_type,
                    "device": str(device),
                    "sf": {
                        "path": args.sf_path,
                        "uci_elo": sf_elo,
                        "threads_per_worker": int(args.sf_threads_per_worker),
                        "hash_mb": int(args.sf_hash_mb),
                        "depth": int(args.sf_depth),
                        "multipv_topk": int(topk),
                        "restrict_cp_window": int(args.restrict_cp_window),
                        "temperature": float(args.temperature),
                        "sample": bool(args.sample),
                        "seed": int(args.seed),
                        "eps": float(args.eps),
                        "workers": int(workers),
                    },
                    "base": agg_base_f,
                    args.ft_tag: agg_ft_f,
                    "delta_ft_minus_base": {k: delta(k) for k in [
                        "sf_help_top1_acc",
                        "sf_help_top5_hit_cand",
                        "sf_help_top10_hit_cand",
                        "sf_help_mean_cp_gap",
                        "sf_help_best_sf_rate",
                        "sf_help_chosen_in_cands_rate",
                        "sf_help_mean_gap_logp_cond",
                        "sf_help_mean_kl_q_vs_base",
                        "full_hit1",
                        "full_hit5",
                        "full_hit10",
                    ]},
                    "artifacts": {
                        "per_row_jsonl": str(per_row_path),
                    }
                }

                summary_path = gm_out_dir / f"summary__sfelo={sf_elo_tag}__topk={topk}.json"
                summary_path.write_text(json.dumps(summary, indent=2))
                print(f"[saved] {summary_path}")

                # save opening distributions
                opening = {
                    "gm": gm,
                    "sf_elo": sf_elo,
                    "topk": topk,
                    "base_opening_selected": opening_summary(opening_counts_base, topn=50),
                    f"{args.ft_tag}_opening_selected": opening_summary(opening_counts_ft, topn=50),
                }
                opening_path = gm_out_dir / f"opening__sfelo={sf_elo_tag}__topk={topk}.json"
                opening_path.write_text(json.dumps(opening, indent=2))
                print(f"[saved] {opening_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
