#!/usr/bin/env python3
"""
eval_sf_helper_ray.py

Ray-native SF-helper evaluation.

Key design goals
----------------
- Parallelize **Stockfish analysis** using Ray *actors*:
    each StockfishActor owns a persistent Stockfish engine process.
- Parallelize **evaluation over dataset rows** using Ray *tasks*:
    dataset is sharded; each shard is processed by a Ray task.
- Avoid repeatedly loading models:
    each shard task loads base+ft models once per task *by default* (simple + robust).
    If you want maximum throughput, use --use_model_actors to keep models warm
    (ModelPairActor per evaluator). That’s included.

What it computes
----------------
For each (GM, sf_elo, topk):
- Base Maia2 + SF-helper metrics
- Finetuned Maia2 + SF-helper metrics
- Deltas (ft - base)
- Opening move distribution of *selected moves* at:
    - ply_abs==0 (white first move)
    - ply_abs==1 (black first reply)

TopK metrics
------------
- Candidate-q hits: cand_hit@1/@5/@10 (chosen is in top-k of q over candidates)
- Full-policy hits: full_hit@1/@5/@10 (chosen is in top-k over all legal moves)

Determinism / stochasticity
---------------------------
- Default is deterministic within candidates: argmax of q (no sampling).
- If --sample is set, sampling uses a seeded Python RNG per shard.
  (Stockfish at fixed depth is typically deterministic.)

Usage example
-------------
python eval_sf_helper_ray.py \
  --ray_address auto \
  --gms firouzja,nakamura \
  --policy_pts ./.../firouzja/policy_dpo_best.pt,./.../nakamura/policy_dpo_best.pt \
  --ft_tag dpo \
  --split val \
  --maia_type blitz \
  --device cpu \
  --stockfish_path /usr/local/bin/stockfish \
  --sf_elos none,1600,2000,2400,2800 \
  --topks 5,10,15,20,25 \
  --sf_depth 10 \
  --restrict_cp_window 60 \
  --num_sf_actors 32 \
  --sf_threads_per_actor 1 \
  --sf_hash_mb 128 \
  --shard_size 256 \
  --max_rows 0

Notes
-----
- For best throughput on CPU clusters:
    set --num_sf_actors ~= total cores, and --sf_threads_per_actor=1.
- If RAM is tight, keep --sf_hash_mb modest (64-256).
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
import ray
import torch

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move
from grandmaster_dpo.eval.stockfish_helpers import make_stockfish


# ----------------------------
# Basic helpers
# ----------------------------

def parse_csv_list(s: str) -> List[str]:
    s = (s or "").strip()
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

def device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s in ("mps",):
        return torch.device("mps")
    return torch.device(s)

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

def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))

def fen_to_ply_abs(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return 2 * (fullmove - 1) + (1 if side == "b" else 0)

def load_policy_weights_into(model: torch.nn.Module, pt_path: str) -> None:
    sd = torch.load(pt_path, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (showing 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}")

def _score_to_cp(score: chess.engine.PovScore, mate_score: int = 100_000) -> int:
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
        m = rel.mate()
        if m is not None:
            return mate_score if m > 0 else -mate_score
        return 0
    return int(cp)

def _entropy(probs: List[float], eps: float) -> float:
    if not probs:
        return 0.0
    s = 0.0
    for p in probs:
        pp = max(float(p), eps)
        s -= pp * math.log(pp)
    return float(s)

def full_hit_at_k(logits_masked: torch.Tensor, chosen_idx: torch.Tensor, ks: Tuple[int, ...]) -> Dict[int, torch.Tensor]:
    max_k = max(ks)
    topk = torch.topk(logits_masked, k=max_k, dim=-1).indices
    chosen_safe = chosen_idx.clamp(min=0).view(-1, 1)
    hits = {}
    for k in ks:
        hit = (topk[:, :k] == chosen_safe).any(dim=1).float()
        hit = torch.where(chosen_idx >= 0, hit, torch.zeros_like(hit))
        hits[k] = hit
    return hits


# ----------------------------
# Stockfish Ray Actor
# ----------------------------

@ray.remote
class StockfishActor:
    """
    Persistent Stockfish engine in an actor for fast repeated analyses.
    """
    def __init__(
        self,
        stockfish_path: str,
        *,
        threads: int,
        hash_mb: int,
        uci_elo: Optional[int],
        depth: int,
        multipv: int,
        timeout: float,
    ):
        self.stockfish_path = stockfish_path
        self.uci_elo = uci_elo
        self.depth = int(depth)
        self.multipv = int(multipv)
        self.engine = make_stockfish(
            stockfish_path,
            threads=int(threads),
            hash_mb=int(hash_mb),
            uci_elo=uci_elo,
            skill_level=None,
            timeout=float(timeout),
        )
        self.limit = chess.engine.Limit(depth=self.depth)

    def analyse_batch(self, fens: List[str]) -> List[Dict[str, Any]]:
        """
        Returns list aligned with fens:
          { "fen": fen, "is_game_over": bool, "cands": [(uci,cp),...], "best_cp": int }
        """
        out: List[Dict[str, Any]] = []
        for fen in fens:
            board = chess.Board(fen)
            if board.is_game_over(claim_draw=True):
                out.append({"fen": fen, "is_game_over": True, "cands": [], "best_cp": 0})
                continue

            infos = self.engine.analyse(board, self.limit, multipv=self.multipv)
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
                out.append({"fen": fen, "is_game_over": False, "cands": [], "best_cp": 0})
                continue

            best_cp = max(cp for _, cp in cands)
            out.append({"fen": fen, "is_game_over": False, "cands": cands, "best_cp": int(best_cp)})

        return out

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass


# ----------------------------
# Optional: Model Pair Ray Actor (keeps models warm)
# ----------------------------

@ray.remote
class ModelPairActor:
    """
    Keeps base + finetuned models loaded once, and runs forward passes on demand.
    """
    def __init__(self, maia_type: str, device_str: str, policy_pt: str):
        self.device = device_from_str(device_str)
        self.base = maia_model.from_pretrained(type=maia_type, device=str(self.device)).to(self.device)
        self.ft = maia_model.from_pretrained(type=maia_type, device=str(self.device)).to(self.device)
        load_policy_weights_into(self.ft, policy_pt)
        self.base.eval()
        self.ft.eval()

        self.all_moves = get_all_possible_moves()
        self.all_moves_dict = {m: i for i, m in enumerate(self.all_moves)}
        self.elo_dict = create_elo_dict()

    def forward_batch(
        self,
        fens: List[str],
        elo_self: List[int],
        elo_oppo: List[int],
    ) -> Dict[str, Any]:
        """
        Returns:
          logits_base_m: [B,V] legal-masked
          logits_ft_m:   [B,V] legal-masked
          logp_base_full:[B,V] log-softmax of base masked
          chosen_idx helper should be done outside with all_moves_dict.
        """
        board_inputs = []
        legal_moves = []
        es_cats = []
        eo_cats = []

        for fen, es, eo in zip(fens, elo_self, elo_oppo):
            bi, es_cat, eo_cat, lm = inference.preprocessing(fen, int(es), int(eo), self.elo_dict, self.all_moves_dict)
            board_inputs.append(bi)
            legal_moves.append(lm)
            es_cats.append(int(es_cat))
            eo_cats.append(int(eo_cat))

        board_input = torch.stack(board_inputs, dim=0).to(self.device)
        legal_moves_t = torch.stack(legal_moves, dim=0).to(self.device)
        es_t = torch.tensor(es_cats, device=self.device).long()
        eo_t = torch.tensor(eo_cats, device=self.device).long()

        with torch.no_grad():
            lb, _, _ = self.base(board_input, es_t, eo_t)
            lf, _, _ = self.ft(board_input, es_t, eo_t)

        lb_m = apply_legal_mask(lb, legal_moves_t)
        lf_m = apply_legal_mask(lf, legal_moves_t)
        logp_base_full = torch.log_softmax(lb_m, dim=-1)

        # move to CPU for cheap transfer (still large but OK; shard_size controls this)
        return {
            "logits_base_m": lb_m.detach().cpu(),
            "logits_ft_m": lf_m.detach().cpu(),
            "logp_base_full": logp_base_full.detach().cpu(),
        }

    def get_vocab(self) -> Dict[str, Any]:
        return {"all_moves": self.all_moves, "all_moves_dict": self.all_moves_dict}


# ----------------------------
# Aggregation structures
# ----------------------------

@dataclass
class PartialAgg:
    # counts
    total_rows: int = 0
    valid_rows: int = 0

    # selection stats
    top1_vs_chosen: float = 0.0
    cand_hit5: float = 0.0
    cand_hit10: float = 0.0
    cp_gap_sum: float = 0.0
    best_sf_rate: float = 0.0
    entropy_sum: float = 0.0
    logp_selected_full_sum: float = 0.0
    chosen_in_cands_sum: float = 0.0
    p_chosen_if_in_sum: float = 0.0

    # conditional q metrics
    p_chosen_cond_sum: float = 0.0
    p_rejected_cond_sum: float = 0.0
    logp_chosen_cond_sum: float = 0.0
    logp_rejected_cond_sum: float = 0.0
    gap_logp_cond_sum: float = 0.0
    kl_q_vs_base_sum: float = 0.0

    # full-policy hits
    full_hit1_sum: float = 0.0
    full_hit5_sum: float = 0.0
    full_hit10_sum: float = 0.0

    # opening distributions
    opening_white_ply0: Counter = None
    opening_black_ply1: Counter = None

    def __post_init__(self):
        if self.opening_white_ply0 is None:
            self.opening_white_ply0 = Counter()
        if self.opening_black_ply1 is None:
            self.opening_black_ply1 = Counter()

    def to_json(self) -> Dict[str, Any]:
        return {
            "sf_total_rows": int(self.total_rows),
            "sf_valid_rows": int(self.valid_rows),
            "sf_help_top1_accuracy_vs_chosen": self._avg(self.top1_vs_chosen),
            "sf_help_top5_hit_vs_chosen_cand_q": self._avg(self.cand_hit5),
            "sf_help_top10_hit_vs_chosen_cand_q": self._avg(self.cand_hit10),
            "sf_help_mean_cp_gap_vs_best": self._avg(self.cp_gap_sum),
            "sf_help_picked_best_sf_rate": self._avg(self.best_sf_rate),
            "sf_help_mean_entropy": self._avg(self.entropy_sum),
            "sf_help_mean_logp_selected_full": self._avg(self.logp_selected_full_sum),
            "sf_help_chosen_in_candidates_rate": self._avg(self.chosen_in_cands_sum),
            "sf_help_mean_p_chosen_given_in_candidates": (
                (self.p_chosen_if_in_sum / max(1.0, self.chosen_in_cands_sum))
                if self.valid_rows > 0 else float("nan")
            ),
            "sf_help_mean_p_chosen_cond": self._avg(self.p_chosen_cond_sum),
            "sf_help_mean_p_rejected_cond": self._avg(self.p_rejected_cond_sum),
            "sf_help_mean_logp_chosen_cond": self._avg(self.logp_chosen_cond_sum),
            "sf_help_mean_logp_rejected_cond": self._avg(self.logp_rejected_cond_sum),
            "sf_help_mean_gap_logp_cond": self._avg(self.gap_logp_cond_sum),
            "sf_help_mean_kl_q_vs_base": self._avg(self.kl_q_vs_base_sum),
            "full_hit1": self._avg(self.full_hit1_sum),
            "full_hit5": self._avg(self.full_hit5_sum),
            "full_hit10": self._avg(self.full_hit10_sum),
        }

    def openings_json(self, topn: int = 50) -> Dict[str, Any]:
        return {
            "white_ply0": [{"uci": u, "count": c} for (u, c) in self.opening_white_ply0.most_common(topn)],
            "black_ply1": [{"uci": u, "count": c} for (u, c) in self.opening_black_ply1.most_common(topn)],
        }

    def merge(self, other: "PartialAgg") -> None:
        self.total_rows += other.total_rows
        self.valid_rows += other.valid_rows

        self.top1_vs_chosen += other.top1_vs_chosen
        self.cand_hit5 += other.cand_hit5
        self.cand_hit10 += other.cand_hit10
        self.cp_gap_sum += other.cp_gap_sum
        self.best_sf_rate += other.best_sf_rate
        self.entropy_sum += other.entropy_sum
        self.logp_selected_full_sum += other.logp_selected_full_sum
        self.chosen_in_cands_sum += other.chosen_in_cands_sum
        self.p_chosen_if_in_sum += other.p_chosen_if_in_sum

        self.p_chosen_cond_sum += other.p_chosen_cond_sum
        self.p_rejected_cond_sum += other.p_rejected_cond_sum
        self.logp_chosen_cond_sum += other.logp_chosen_cond_sum
        self.logp_rejected_cond_sum += other.logp_rejected_cond_sum
        self.gap_logp_cond_sum += other.gap_logp_cond_sum
        self.kl_q_vs_base_sum += other.kl_q_vs_base_sum

        self.full_hit1_sum += other.full_hit1_sum
        self.full_hit5_sum += other.full_hit5_sum
        self.full_hit10_sum += other.full_hit10_sum

        self.opening_white_ply0.update(other.opening_white_ply0)
        self.opening_black_ply1.update(other.opening_black_ply1)

    def _avg(self, s: float) -> float:
        return s / max(1, self.valid_rows)


# ----------------------------
# Shard evaluation (Ray task)
# ----------------------------

@ray.remote
def eval_shard_task(
    shard_rows: List[Dict[str, Any]],
    *,
    maia_type: str,
    device_str: str,
    policy_pt: str,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    sf_actor: ray.actor.ActorHandle,
    topk: int,
    sf_restrict_cp_window: Optional[int],
    temperature: float,
    sample: bool,
    seed: int,
    eps: float,
    use_model_actors: bool,
    model_actor: Optional[ray.actor.ActorHandle],
) -> Dict[str, Any]:
    """
    Returns dict with:
      base_partial: PartialAgg serialized
      ft_partial:   PartialAgg serialized
    """
    # Build batch lists
    fens = [r["fen"] for r in shard_rows]
    elo_self = [int(r.get("elo_self", 2800)) for r in shard_rows]
    elo_oppo = [int(r.get("elo_oppo", 2800)) for r in shard_rows]
    chosen = [r["chosen"] for r in shard_rows]
    rejected = [r["rejected"] for r in shard_rows]

    # SF analysis (batched)
    sf_res = ray.get(sf_actor.analyse_batch.remote(fens))

    # Get logits for base/ft + base logp_full
    if use_model_actors and model_actor is not None:
        fw = ray.get(model_actor.forward_batch.remote(fens, elo_self, elo_oppo))
        logits_base_m = fw["logits_base_m"]  # CPU tensors
        logits_ft_m = fw["logits_ft_m"]
        logp_base_full = fw["logp_base_full"]
        # tensors on CPU already
    else:
        device = device_from_str(device_str)
        base = maia_model.from_pretrained(type=maia_type, device=str(device)).to(device)
        ft = maia_model.from_pretrained(type=maia_type, device=str(device)).to(device)
        load_policy_weights_into(ft, policy_pt)
        base.eval()
        ft.eval()

        board_inputs = []
        legal_moves = []
        es_cats = []
        eo_cats = []

        for fen, es, eo in zip(fens, elo_self, elo_oppo):
            bi, es_cat, eo_cat, lm = inference.preprocessing(fen, int(es), int(eo), elo_dict, all_moves_dict)
            board_inputs.append(bi)
            legal_moves.append(lm)
            es_cats.append(int(es_cat))
            eo_cats.append(int(eo_cat))

        board_input = torch.stack(board_inputs, dim=0).to(device)
        legal_moves_t = torch.stack(legal_moves, dim=0).to(device)
        es_t = torch.tensor(es_cats, device=device).long()
        eo_t = torch.tensor(eo_cats, device=device).long()

        with torch.no_grad():
            lb, _, _ = base(board_input, es_t, eo_t)
            lf, _, _ = ft(board_input, es_t, eo_t)

        logits_base_m = apply_legal_mask(lb, legal_moves_t).detach().cpu()
        logits_ft_m = apply_legal_mask(lf, legal_moves_t).detach().cpu()
        logp_base_full = torch.log_softmax(logits_base_m, dim=-1).detach().cpu()

    # full hits for chosen (per model)
    chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, u) for fen, u in zip(fens, chosen)], dtype=torch.long)
    hits_base = full_hit_at_k(logits_base_m, chosen_idx, ks=(1, 5, 10))
    hits_ft = full_hit_at_k(logits_ft_m, chosen_idx, ks=(1, 5, 10))

    rng_base = random.Random(seed + 2000)
    rng_ft = random.Random(seed + 1000)

    base_agg = PartialAgg()
    ft_agg = PartialAgg()

    def compute_one(
        *,
        agg: PartialAgg,
        fen: str,
        chosen_uci: str,
        rejected_uci: str,
        sf_cands: List[Tuple[str, int]],
        best_cp: int,
        logits_m: torch.Tensor,          # [V], CPU
        base_logp_full_row: torch.Tensor,# [V], CPU
        full_hit_row: Dict[int, int],
        rng: random.Random,
    ) -> Optional[str]:
        # filter cands by cp window
        kept = sf_cands
        if sf_restrict_cp_window is not None:
            w = int(sf_restrict_cp_window)
            filt = [(m, cp) for (m, cp) in kept if cp >= best_cp - w]
            if filt:
                kept = filt
        if not kept:
            return None

        # q over candidates from logits
        t = max(float(temperature), 1e-6)
        logp_all = torch.log_softmax(logits_m / t, dim=-1)  # [V]

        cand_moves = [m for (m, _cp) in kept]
        cand_logps = []
        for uci in cand_moves:
            idx = uci_to_vocab_index(all_moves_dict, fen, uci)
            if idx < 0:
                cand_logps.append(torch.tensor(torch.finfo(logp_all.dtype).min))
            else:
                cand_logps.append(logp_all[idx])
        cand_logps_t = torch.stack(cand_logps, dim=0)
        q = torch.softmax(cand_logps_t, dim=0).detach().cpu().tolist()

        # select move
        if sample:
            r = rng.random()
            acc = 0.0
            sel_i = 0
            for j, p in enumerate(q):
                acc += float(p)
                if r <= acc:
                    sel_i = j
                    break
        else:
            sel_i = max(range(len(q)), key=lambda j: q[j])

        selected_uci, cp_sel = kept[sel_i]
        cp_gap = float(best_cp - int(cp_sel))
        is_best = 1.0 if cp_gap <= 1e-9 else 0.0

        # entropy
        ent = _entropy(q, eps)

        # logp(selected) under full masked distribution
        sel_idx = uci_to_vocab_index(all_moves_dict, fen, selected_uci)
        logp_selected_full = float(logp_all[sel_idx].item()) if sel_idx >= 0 else float("-inf")

        # chosen/rejected conditional probs
        p_ch = float(q[cand_moves.index(chosen_uci)]) if chosen_uci in cand_moves else 0.0
        p_rj = float(q[cand_moves.index(rejected_uci)]) if rejected_uci in cand_moves else 0.0

        logp_ch = math.log(max(p_ch, eps))
        logp_rj = math.log(max(p_rj, eps))
        gap_logp = logp_ch - logp_rj

        # candidate hits for chosen
        order = sorted(range(len(q)), key=lambda j: q[j], reverse=True)
        def cand_hit(k: int) -> float:
            kk = min(k, len(order))
            top_moves = [cand_moves[j] for j in order[:kk]]
            return 1.0 if chosen_uci in top_moves else 0.0

        # KL(q || base_full) over candidate support
        kl = 0.0
        for uci, qq in zip(cand_moves, q):
            idx = uci_to_vocab_index(all_moves_dict, fen, uci)
            if idx < 0:
                continue
            logq = math.log(max(float(qq), eps))
            logp_b = float(base_logp_full_row[idx].item())
            kl += float(qq) * (logq - logp_b)

        # update agg
        agg.total_rows += 1
        agg.valid_rows += 1
        agg.top1_vs_chosen += 1.0 if selected_uci == chosen_uci else 0.0
        agg.cand_hit5 += cand_hit(5)
        agg.cand_hit10 += cand_hit(10)
        agg.cp_gap_sum += cp_gap
        agg.best_sf_rate += is_best
        agg.entropy_sum += ent
        agg.logp_selected_full_sum += logp_selected_full

        chosen_in = 1.0 if p_ch > 0.0 else 0.0
        agg.chosen_in_cands_sum += chosen_in
        if chosen_in:
            agg.p_chosen_if_in_sum += p_ch

        agg.p_chosen_cond_sum += p_ch
        agg.p_rejected_cond_sum += p_rj
        agg.logp_chosen_cond_sum += logp_ch
        agg.logp_rejected_cond_sum += logp_rj
        agg.gap_logp_cond_sum += gap_logp
        agg.kl_q_vs_base_sum += kl

        agg.full_hit1_sum += float(full_hit_row[1])
        agg.full_hit5_sum += float(full_hit_row[5])
        agg.full_hit10_sum += float(full_hit_row[10])

        ply_abs = fen_to_ply_abs(fen)
        if ply_abs == 0:
            agg.opening_white_ply0[selected_uci] += 1
        if ply_abs == 1:
            agg.opening_black_ply1[selected_uci] += 1

        return selected_uci

    # iterate shard rows
    for i, row in enumerate(shard_rows):
        fen = row["fen"]
        ch = chosen[i]
        rj = rejected[i]

        sr = sf_res[i]
        sf_cands = sr["cands"]
        best_cp = int(sr["best_cp"])
        is_over = bool(sr["is_game_over"])

        # count totals even if invalid (for denominator clarity)
        base_agg.total_rows += 1
        ft_agg.total_rows += 1

        if is_over or not sf_cands:
            continue

        # base
        compute_one(
            agg=base_agg,
            fen=fen,
            chosen_uci=ch,
            rejected_uci=rj,
            sf_cands=sf_cands,
            best_cp=best_cp,
            logits_m=logits_base_m[i],
            base_logp_full_row=logp_base_full[i],
            full_hit_row={
                1: int(hits_base[1][i].item()),
                5: int(hits_base[5][i].item()),
                10: int(hits_base[10][i].item()),
            },
            rng=rng_base,
        )

        # finetuned (KL vs base_full)
        compute_one(
            agg=ft_agg,
            fen=fen,
            chosen_uci=ch,
            rejected_uci=rj,
            sf_cands=sf_cands,
            best_cp=best_cp,
            logits_m=logits_ft_m[i],
            base_logp_full_row=logp_base_full[i],
            full_hit_row={
                1: int(hits_ft[1][i].item()),
                5: int(hits_ft[5][i].item()),
                10: int(hits_ft[10][i].item()),
            },
            rng=rng_ft,
        )

    # serialize counters (ray can return Counters but JSON is easier)
    return {
        "base": {
            "agg": base_agg.to_json(),
            "openings": base_agg.openings_json(),
        },
        "ft": {
            "agg": ft_agg.to_json(),
            "openings": ft_agg.openings_json(),
        },
        "base_raw": base_agg,  # kept for reducer merge (Ray can pickle)
        "ft_raw": ft_agg,
    }


# ----------------------------
# Reducer
# ----------------------------

def merge_partials(parts: List[Dict[str, Any]]) -> Tuple[PartialAgg, PartialAgg]:
    base = PartialAgg()
    ft = PartialAgg()
    for p in parts:
        base.merge(p["base_raw"])
        ft.merge(p["ft_raw"])
    return base, ft


# ----------------------------
# Data loading
# ----------------------------

def load_jsonl_rows(path: str, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p = r.get("prompt", {}) or {}
            rows.append({
                "fen": p["fen"],
                "elo_self": int(p.get("elo_self", 2800)),
                "elo_oppo": int(p.get("elo_oppo", 2800)),
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            })
            if max_rows and len(rows) >= max_rows:
                break
    return rows

def shard_rows(rows: List[Dict[str, Any]], shard_size: int) -> List[List[Dict[str, Any]]]:
    out = []
    for i in range(0, len(rows), shard_size):
        out.append(rows[i:i+shard_size])
    return out


# ----------------------------
# Pretty printing
# ----------------------------

def print_blocks(tag: str, agg: Dict[str, Any]) -> None:
    print(f"=== {tag}: Policy + Stockfish-help (candidate filter) ===")
    print(f"sf_valid_rows: {agg['sf_valid_rows']}/{agg['sf_total_rows']}")
    print(f"top1 accuracy vs chosen:           {agg['sf_help_top1_accuracy_vs_chosen']:.4f}")
    print(f"top5 hit vs chosen (cand q):       {agg['sf_help_top5_hit_vs_chosen_cand_q']:.4f}")
    print(f"top10 hit vs chosen (cand q):      {agg['sf_help_top10_hit_vs_chosen_cand_q']:.4f}")
    print(f"mean cp_gap vs best SF:            {agg['sf_help_mean_cp_gap_vs_best']:.2f}")
    print(f"picked best-SF move rate:          {agg['sf_help_picked_best_sf_rate']:.4f}")
    print(f"mean candidate-dist entropy:       {agg['sf_help_mean_entropy']:.4f}")
    print(f"mean logp(policy(selected)):       {agg['sf_help_mean_logp_selected_full']:.4f}")
    print(f"chosen-in-candidates rate:         {agg['sf_help_chosen_in_candidates_rate']:.4f}")
    print(f"mean P(chosen | in candidates):    {agg['sf_help_mean_p_chosen_given_in_candidates']:.4f}")
    print("")
    print(f"=== {tag}: SF-helper conditional logp metrics (q = pi restricted to cands) ===")
    print(f"mean P(chosen | cands):            {agg['sf_help_mean_p_chosen_cond']:.6f}")
    print(f"mean P(rejected | cands):          {agg['sf_help_mean_p_rejected_cond']:.6f}")
    print(f"mean logp(chosen | cands):         {agg['sf_help_mean_logp_chosen_cond']:.4f}")
    print(f"mean logp(rejected | cands):       {agg['sf_help_mean_logp_rejected_cond']:.4f}")
    print(f"mean gap logp(ch) - logp(rj):      {agg['sf_help_mean_gap_logp_cond']:.4f}")
    print(f"mean KL(q || base_full):           {agg['sf_help_mean_kl_q_vs_base']:.4f}")
    print("")
    print("--- Full-policy hit@k on chosen (legal-move distro) ---")
    print(f"full hit@1:                        {agg['full_hit1']:.4f}")
    print(f"full hit@5:                        {agg['full_hit5']:.4f}")
    print(f"full hit@10:                       {agg['full_hit10']:.4f}")
    print("")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ray_address", default="auto", help='Ray address. Use "auto" on cluster; "" for local.')

    ap.add_argument("--gms", required=True, help="Comma-separated GM names.")
    ap.add_argument("--policy_pts", required=True, help="Comma-separated finetuned .pt paths, same order as --gms.")
    ap.add_argument("--ft_tag", required=True)

    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jsonl_template", default="./processed/single_gm/train_val/{gm}_{split}_dpo.jsonl")
    ap.add_argument("--out_root", default="./processed/single_gm/train_val/validation_results")

    # Stockfish
    ap.add_argument("--stockfish_path", required=True, help="Path to stockfish binary inside the Ray runtime/image.")
    ap.add_argument("--sf_threads_per_actor", type=int, default=1)
    ap.add_argument("--sf_hash_mb", type=int, default=128)
    ap.add_argument("--sf_depth", type=int, default=10)
    ap.add_argument("--sf_timeout", type=float, default=30.0)
    ap.add_argument("--sf_elos", default="none", help='Comma list, e.g. "none,1600,2000,2400,2800".')

    # multipv topk sweep
    ap.add_argument("--topks", default="10", help="Comma list, e.g. 5,10,15,20,25")

    # helper
    ap.add_argument("--restrict_cp_window", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-12)

    # Ray parallelism
    ap.add_argument("--num_sf_actors", type=int, default=16)
    ap.add_argument("--shard_size", type=int, default=256)
    ap.add_argument("--max_rows", type=int, default=0)

    # Model actor option
    ap.add_argument("--use_model_actors", action="store_true",
                    help="Keep models warm in Ray actors (faster). If off, each shard task loads models.")
    ap.add_argument("--num_model_actors", type=int, default=4,
                    help="Only used when --use_model_actors; number of model actors to create.")

    args = ap.parse_args()

    # Ray init
    if args.ray_address == "":
        ray.init()
    else:
        ray.init(address=args.ray_address)

    gms = parse_csv_list(args.gms)
    pts = parse_csv_list(args.policy_pts)
    if len(gms) != len(pts):
        raise ValueError(f"--gms has {len(gms)} items but --policy_pts has {len(pts)} items")

    sf_elos = parse_sf_elos(args.sf_elos)
    topks = parse_int_list(args.topks)
    if not topks:
        raise ValueError("No topks provided")

    # vocab + elo dict (stable)
    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    out_root = Path(args.out_root)

    for gm, pt_path in zip(gms, pts):
        print(f"\n==============================")
        print(f"GM: {gm}")
        print(f"checkpoint: {pt_path}")
        print(f"split: {args.split}")
        print(f"==============================\n")

        jsonl_path = args.jsonl_template.format(gm=gm, split=args.split)
        rows = load_jsonl_rows(jsonl_path, max_rows=args.max_rows)
        shards = shard_rows(rows, shard_size=max(1, int(args.shard_size)))

        gm_out_dir = out_root / gm / f"sf_helper_eval_{args.ft_tag}_{args.split}_ray"
        gm_out_dir.mkdir(parents=True, exist_ok=True)

        for sf_elo in sf_elos:
            sf_elo_tag = "none" if sf_elo is None else str(sf_elo)

            for topk in topks:
                print(f"\n--- Running sf_elo={sf_elo_tag} topk={topk} ---\n")

                # Create Stockfish actor pool (each actor has its own engine)
                sf_actors = [
                    StockfishActor.remote(
                        args.stockfish_path,
                        threads=int(args.sf_threads_per_actor),
                        hash_mb=int(args.sf_hash_mb),
                        uci_elo=sf_elo,
                        depth=int(args.sf_depth),
                        multipv=int(topk),
                        timeout=float(args.sf_timeout),
                    )
                    for _ in range(int(args.num_sf_actors))
                ]

                # Optional: model actors pool (keeps models warm)
                model_actors: List[ray.actor.ActorHandle] = []
                if args.use_model_actors:
                    for _ in range(int(args.num_model_actors)):
                        model_actors.append(ModelPairActor.remote(args.maia_type, args.device, pt_path))

                # Launch shard tasks
                futures = []
                for si, shard in enumerate(shards):
                    sf_actor = sf_actors[si % len(sf_actors)]
                    model_actor = model_actors[si % len(model_actors)] if model_actors else None
                    futures.append(
                        eval_shard_task.remote(
                            shard,
                            maia_type=args.maia_type,
                            device_str=args.device,
                            policy_pt=pt_path,
                            all_moves_dict=all_moves_dict,
                            elo_dict=elo_dict,
                            sf_actor=sf_actor,
                            topk=int(topk),
                            sf_restrict_cp_window=int(args.restrict_cp_window) if args.restrict_cp_window is not None else None,
                            temperature=float(args.temperature),
                            sample=bool(args.sample),
                            seed=int(args.seed + si * 9973),
                            eps=float(args.eps),
                            use_model_actors=bool(args.use_model_actors),
                            model_actor=model_actor,
                        )
                    )

                parts = ray.get(futures)
                base_agg, ft_agg = merge_partials(parts)

                base_json = base_agg.to_json()
                ft_json = ft_agg.to_json()

                print_blocks("base", base_json)
                print_blocks(args.ft_tag, ft_json)

                # deltas
                def d(k: str) -> float:
                    return float(ft_json.get(k, 0.0)) - float(base_json.get(k, 0.0))

                print("=== Delta (finetuned - base) ===")
                print(f"top1 accuracy vs chosen:           {d('sf_help_top1_accuracy_vs_chosen'):+.4f}")
                print(f"top5 hit vs chosen (cand q):       {d('sf_help_top5_hit_vs_chosen_cand_q'):+.4f}")
                print(f"top10 hit vs chosen (cand q):      {d('sf_help_top10_hit_vs_chosen_cand_q'):+.4f}")
                print(f"mean cp_gap vs best SF:            {d('sf_help_mean_cp_gap_vs_best'):+.2f}")
                print(f"picked best-SF move rate:          {d('sf_help_picked_best_sf_rate'):+.4f}")
                print(f"chosen-in-candidates rate:         {d('sf_help_chosen_in_candidates_rate'):+.4f}")
                print(f"mean gap logp(ch)-logp(rj):        {d('sf_help_mean_gap_logp_cond'):+.4f}")
                print(f"mean KL(q||base_full):             {d('sf_help_mean_kl_q_vs_base'):+.4f}")
                print("")

                # Save summary
                summary = {
                    "gm": gm,
                    "ft_tag": args.ft_tag,
                    "split": args.split,
                    "maia_type": args.maia_type,
                    "device": args.device,
                    "sf": {
                        "stockfish_path": args.stockfish_path,
                        "uci_elo": sf_elo,
                        "depth": int(args.sf_depth),
                        "multipv_topk": int(topk),
                        "restrict_cp_window": int(args.restrict_cp_window),
                        "temperature": float(args.temperature),
                        "sample": bool(args.sample),
                        "seed": int(args.seed),
                        "actors": int(args.num_sf_actors),
                        "threads_per_actor": int(args.sf_threads_per_actor),
                        "hash_mb": int(args.sf_hash_mb),
                    },
                    "counts": {"rows": len(rows), "shards": len(shards), "shard_size": int(args.shard_size)},
                    "base": base_json,
                    args.ft_tag: ft_json,
                    "delta_ft_minus_base": {k: d(k) for k in [
                        "sf_help_top1_accuracy_vs_chosen",
                        "sf_help_top5_hit_vs_chosen_cand_q",
                        "sf_help_top10_hit_vs_chosen_cand_q",
                        "sf_help_mean_cp_gap_vs_best",
                        "sf_help_picked_best_sf_rate",
                        "sf_help_chosen_in_candidates_rate",
                        "sf_help_mean_gap_logp_cond",
                        "sf_help_mean_kl_q_vs_base",
                        "full_hit1",
                        "full_hit5",
                        "full_hit10",
                    ]},
                }
                summary_path = gm_out_dir / f"summary__sfelo={sf_elo_tag}__topk={topk}.json"
                summary_path.write_text(json.dumps(summary, indent=2))
                print(f"[saved] {summary_path}")

                # Save openings
                openings = {
                    "gm": gm,
                    "sf_elo": sf_elo,
                    "topk": int(topk),
                    "base_opening_selected": base_agg.openings_json(topn=50),
                    f"{args.ft_tag}_opening_selected": ft_agg.openings_json(topn=50),
                }
                opening_path = gm_out_dir / f"openings__sfelo={sf_elo_tag}__topk={topk}.json"
                opening_path.write_text(json.dumps(openings, indent=2))
                print(f"[saved] {opening_path}")

                # Cleanup actors for this sweep (important)
                for a in sf_actors:
                    try:
                        ray.get(a.close.remote())
                    except Exception:
                        pass
                    try:
                        ray.kill(a)
                    except Exception:
                        pass

                for a in model_actors:
                    try:
                        ray.kill(a)
                    except Exception:
                        pass

    print("\nDone.\n")


if __name__ == "__main__":
    main()
