# src/grandmaster_dpo/eval/eval_abstractions.py
from __future__ import annotations

import csv
import json
import math
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
from grandmaster_dpo.eval.chess_utils import batch_preprocess, coarse_opening_family_from_prefix, extract_move_cp, fen_to_ply_abs, forward_logits, ply_to_phase, vocab_index_to_uci
from grandmaster_dpo.eval.configs import OpeningLogitDistConfig, SfConfig
from grandmaster_dpo.eval.dataset import DpoPairs
from grandmaster_dpo.eval.opening_metrics import summarize_opening_distribution, update_opening_distributions_from_logits
from grandmaster_dpo.eval.single_gm.shared_eval_metric_utilities import add_piece_selection_per_row_stats, apply_legal_mask, collate_batch, hit_at_k
from grandmaster_dpo.eval.tensor_metrics import chosen_probability, chosen_rank, gather_logprob_from_masked_logits, kl_policy_base_from_logits
from grandmaster_dpo.eval.trajectory_utils import safe_get_next_fens_chosen, safe_get_next_fens_rejected, safe_get_prev_fens
from grandmaster_dpo.eval.types import _SFCandidate, _SfBatchContext, EvalAggMetrics, EvalPerRowInput, EvalRowMetrics, SfHelperEvalAggregate, SfPerPosResult
import torch
from torch.utils.data import DataLoader
import numpy as np

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves

from grandmaster_dpo.eval.stockfish_helpers import make_stockfish, uci_to_vocab_index


def entropy_from_logits(masked_logits: torch.Tensor) -> torch.Tensor:
    logp = torch.nn.functional.log_softmax(masked_logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def _safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def _binary_mean(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return _safe_mean(vals)


def _precision_at_k(rows: List[Dict[str, Any]], hit_key: str) -> float:
    return _binary_mean(rows, hit_key)


def _recall_at_k(rows: List[Dict[str, Any]], hit_key: str) -> float:
    return _binary_mean(rows, hit_key)


def _f1_from_pr(p: float, r: float) -> float:
    if p + r <= 0:
        return 0.0
    return 2.0 * p * r / (p + r)



def _bootstrap_ci(
    vals: List[float],
    stat_fn: Callable[[List[float]], float] = _safe_mean,
    n_boot: int = 1000,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(stat_fn(sample.tolist())))
    boots = np.asarray(boots, dtype=float)
    return {
        "mean": float(stat_fn(arr.tolist())),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "n": int(len(arr)),
    }


def _cluster_bootstrap_ci(
    rows: List[Dict[str, Any]],
    cluster_key: str,
    value_key: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    grouped: Dict[Any, List[float]] = defaultdict(list)
    for r in rows:
        cluster = r.get(cluster_key)
        val = r.get(value_key)
        if cluster is None or val is None:
            continue
        grouped[cluster].append(float(val))

    clusters = list(grouped.keys())
    if not clusters:
        return None

    rng = np.random.default_rng(seed)

    def stat(sampled_clusters: List[Any]) -> float:
        vals: List[float] = []
        for c in sampled_clusters:
            vals.extend(grouped[c])
        return _safe_mean(vals)

    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True).tolist()
        boots.append(stat(sampled))

    point = stat(clusters)
    return {
        "mean": float(point),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "n_clusters": int(len(clusters)),
    }

def _score_to_cp(
    score,
    *,
    turn: bool | None = None,
    mate_score: int = 100_000,
) -> int:
    """
    Convert python-chess engine score -> integer centipawns (mate mapped to +/- mate_score).

    Supports:
      - PovScore (has .pov(turn))
      - Score: Cp / Mate
    """
    # If it's PovScore, convert to POV for 'turn' if provided; else keep its own POV
    if hasattr(score, "pov"):
        if turn is None:
            # PovScore carries its own POV side; python-chess exposes it as .turn on PovScore
            # If .turn doesn't exist in your version, we fall back to not passing turn.
            t = getattr(score, "turn", None)
            score = score.pov(t) if t is not None else score.pov(True)
        else:
            score = score.pov(turn)

    # Now 'score' should be a Score (Cp/Mate). Convert to cp-like int.
    v = score.score(mate_score=mate_score)
    if v is None:
        # Shouldn't happen often, but keep it safe.
        return 0
    return int(v)

def _entropy(probs: List[float], eps: float) -> float:
    s = 0.0
    for p in probs:
        pp = max(float(p), eps)
        s -= pp * math.log(pp)
    return float(s)

def _get_stockfish_candidates_in_window(sf_candidates: List[_SFCandidate], adaptive_restrict_cp_window: float) -> List[_SFCandidate]:
    # --------- cp-window filter ----------
    best_cp = max(data.cp for data in sf_candidates)
    return sorted([data for data in sf_candidates if data.cp >= best_cp - adaptive_restrict_cp_window], key=lambda x: x.cp, reverse=True)

def _get_engine_plus_style_logits(logp_ref_all: torch.Tensor,
                                  logits_masked_1d: torch.Tensor, 
                                  kept: List[_SFCandidate], 
                                  all_moves_dict: Dict[str, int], 
                                  fen: str, 
                                  cp_cap: int, 
                                  cp_scale: float, 
                                  lam: float):
    # --------- compute pi_ref over kept moves (from policy logits) ----------
    # This is the ref distribution in the KL term.
    cand_moves: List[str] = [data.uci for data in kept]
    cand_idxs: List[int] = [uci_to_vocab_index(all_moves_dict, fen, u) for u in cand_moves]

    neg_inf = torch.finfo(logits_masked_1d.dtype).min
    cand_logp_ref = []
    for idx in cand_idxs:
        if idx < 0:
            cand_logp_ref.append(torch.tensor(neg_inf, device=logp_ref_all.device, dtype=logp_ref_all.dtype))
        else:
            cand_logp_ref.append(logp_ref_all[idx])
    cand_logp_ref_t = torch.stack(cand_logp_ref, dim=0)  # [K]

    # --------- Gibbs / KL-regularized policy improvement ----------
    # pi_new(a) ∝ pi_ref(a) * exp(Q(a)/lambda)
    cps = torch.tensor(
        [int(round(data.cp)) for data in kept],
        device=cand_logp_ref_t.device,
        dtype=cand_logp_ref_t.dtype,
    ).clamp(min=-cp_cap, max=cp_cap)

    Q = cps / cp_scale
    combined = cand_logp_ref_t + (Q / max(lam, 1e-6))          # [K]
    return combined

@torch.inference_mode()
def compute_sf_helper_w_gibbs_for_one_position(
    *,
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    logits_masked_1d: torch.Tensor,   # [V] masked logits for policy
    base_logp_full_1d: torch.Tensor,  # [V] log-softmax of base masked logits
    all_moves_dict: Dict[str, int],   # uci_eff -> idx
    cfg: SfConfig,
    full_hit: Dict[int, int],         # {1:0/1, 5:0/1, 10:0/1}
    rng: random.Random,
    sf_inference_candidates: List[_SFCandidate],
    sf_reference_candidates: Optional[List[_SFCandidate]],
) -> Optional[Tuple[SfPerPosResult, Dict[str, Any]]]:

    # --------- basic guards ----------
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=True):
        return None

    # IMPORTANT: ply index must come from FEN, not move_stack
    ply_abs = int(fen_to_ply_abs(fen))  # you already have this util elsewhere

    V = int(logits_masked_1d.numel())

    # Build idx->uci_eff list so we can invert topk indices into actual UCI for this FEN
    idx_to_uci_eff: List[Optional[str]] = [None] * V
    for u, i in all_moves_dict.items():
        ii = int(i)
        if 0 <= ii < V:
            idx_to_uci_eff[ii] = u

    # Build all_moves list for vocab_index_to_uci if you want to reuse it
    # (it expects a list: index -> uci_eff)
    all_moves_list: List[str] = [m if m is not None else "??" for m in idx_to_uci_eff]

    # --------- hyperparams (with sane defaults) ----------
    k = int(getattr(cfg, "k", 40))
    k = max(1, min(k, V))

    restrict_base = float(getattr(cfg, "restrict_cp_window_base", getattr(cfg, "restrict_cp_window", 20) or 20))
    restrict_max_factor = float(getattr(cfg, "restrict_cp_window_max_factor", 6.0))  # base*(~6) ~= 120 if base=20
    game_len = float(getattr(cfg, "game_len_plies", 80.0))
    frac = min(1.0, max(0.0, ply_abs / game_len))
    adaptive_restrict_cp_window = restrict_base * (1.0 + (restrict_max_factor - 1.0) * frac)

    # lambda schedule (you called it temperature): higher early -> less SF influence
    base_lambda = float(getattr(cfg, "base_temperature", getattr(cfg, "temperature", 0.3)))
    min_lambda = float(getattr(cfg, "min_temperature", 1e-3))
    max_lambda = float(getattr(cfg, "max_temperature", 5.0))
    # Example: start high, end low (more SF late). You can flip if you want.
    # High early -> low late:
    lam = base_lambda * (1.0 + (1.0 - frac) * 4.0)   # ~5x early, ~1x late
    lam = float(min(max(lam, min_lambda), max_lambda))

    # Q scale
    cp_cap = int(getattr(cfg, "cp_cap", 2000))
    cp_scale = float(getattr(cfg, "cp_scale", 150.0))  # 150cp -> +1.0 logit bonus at lambda=1

    # --------- policy top-k inference stockfish candidates ----------
    # logits are already masked; softmax is safe (masked moves -> prob 0)
    policy_probs = torch.softmax(logits_masked_1d, dim=-1)  # [V]
    topv, topi = torch.topk(policy_probs, k=k)

    policy_root_moves: List[chess.Move] = []
    policy_cand_details: List[Tuple[str, float]] = []  # [(uci_actual, prob)]

    for p, idx in zip(topv.tolist(), topi.tolist()):
        idx = int(idx)
        if not (0 <= idx < V):
            continue
        uci_actual = vocab_index_to_uci(all_moves_list, fen, idx)
        if uci_actual is None:
            continue
        try:
            mv = chess.Move.from_uci(uci_actual)
        except Exception:
            continue
        if mv in board.legal_moves:
            policy_root_moves.append(mv)
            policy_cand_details.append((uci_actual, float(p)))

    if not policy_root_moves:
        return None

    # --------- stockfish eval restricted to those root moves ----------
    if not sf_inference_candidates or len(sf_inference_candidates) == 0:
        return None

    kept = _get_stockfish_candidates_in_window(sf_inference_candidates, adaptive_restrict_cp_window)
    kept_reference = _get_stockfish_candidates_in_window(sf_reference_candidates, adaptive_restrict_cp_window) if sf_reference_candidates else None
    cand_moves = [data.uci for data in kept]
    cand_ref_moves = [data.uci for data in kept_reference] if kept_reference else None
    best_cp = max((data.cp for data in kept))

    # Combining style logits and engine "logits" from (cp_score / cp_scaled)*1/lam
    logp_ref_all = torch.log_softmax(logits_masked_1d, dim=-1)  # [V]
    combined = _get_engine_plus_style_logits(logp_ref_all, logits_masked_1d, kept, all_moves_dict, fen, cp_cap, cp_scale, lam)
    cand_probs_t = torch.softmax(combined, dim=0)              # [K]
    cand_probs: List[float] = cand_probs_t.detach().cpu().tolist()
    cand_idxs: List[int] = [uci_to_vocab_index(all_moves_dict, fen, u) for u in cand_moves]

    # Build a full-V "logits" tensor that corresponds to combined scores on candidates
    neg_inf = torch.finfo(logits_masked_1d.dtype).min
    cands_logits_full = torch.full_like(logits_masked_1d, neg_inf)  # [V]
    for idx, comb in zip(cand_idxs, combined):
        if idx >= 0:
            cands_logits_full[idx] = comb

    # --------- select within candidates ----------
    # if bool(getattr(cfg, "sample", True)):
    #    sel_i = int(torch.multinomial(cand_probs_t, 1).item())
    # else:
    #    sel_i = int(torch.argmax(cand_probs_t).item())


    best_cp_ref = max([d.cp for d in sf_reference_candidates]) if sf_reference_candidates else None
    cp_chosen_ref = [d.cp for d in sf_reference_candidates if d.uci == chosen_uci][0] if sf_reference_candidates and chosen_uci in [d.uci for d in sf_reference_candidates] else None
    cp_chosen_inf = [d.cp for d in sf_inference_candidates if d.uci == chosen_uci][0] if chosen_uci in [d.uci for d in sf_inference_candidates] else None 
    cp_rejected_ref = [d.cp for d in sf_reference_candidates if d.uci == rejected_uci][0] if sf_reference_candidates and rejected_uci in [d.uci for d in sf_reference_candidates] else None
    cp_rejected_inf = [d.cp for d in sf_inference_candidates if d.uci == rejected_uci][0] if rejected_uci in [d.uci for d in sf_inference_candidates] else None
    
    def get_stockfish_rank(cands, move_uci):
        sorted_ref = sorted(cands, key=lambda d: d.cp, reverse=True)
        ranked = [(i+1, d) for i, d in enumerate(sorted_ref)]
        return [rnk for rnk, d in ranked if d.uci == move_uci][0] if move_uci in [d.uci for rnk, d in ranked] else None
    
    cp_chosen_ref_rank = get_stockfish_rank(sf_reference_candidates, chosen_uci) if sf_reference_candidates else None
    cp_chosen_inf_rank = get_stockfish_rank(sf_inference_candidates, chosen_uci)

    cp_rejected_ref_rank = get_stockfish_rank(sf_reference_candidates, rejected_uci) if sf_reference_candidates else None
    cp_rejected_inf_rank = get_stockfish_rank(sf_inference_candidates, rejected_uci)

    probs_np = cand_probs_t.detach().cpu().numpy()
    eps = float(getattr(cfg, "eps", 1e-12))
    ent = float(-(probs_np * np.log(probs_np + eps)).sum()) if probs_np.size else 0.0

    # chosen/rejected conditional probs in candidate set
    p_ch = float(cand_probs[cand_moves.index(chosen_uci)]) if chosen_uci in cand_moves else 0.0
    p_rj = float(cand_probs[cand_moves.index(rejected_uci)]) if rejected_uci in cand_moves else 0.0
    logp_ch = math.log(max(p_ch, eps))
    logp_rj = math.log(max(p_rj, eps))
    gap_logp = float(logp_ch - logp_rj)

    # candidate hits on chosen (based on q ordering)
    K = len(cand_probs)
    order = sorted(range(K), key=lambda j: cand_probs[j], reverse=True)

    def cand_hit_at(k_: int) -> float:
        if k_ <= 0:
            return 0.0
        kk = min(int(k_), K)
        top_moves = [cand_moves[j] for j in order[:kk]]
        return 1.0 if chosen_uci in top_moves else 0.0

    # KL(q || base_full) over candidate support
    kl = 0.0
    for cand, q in zip(kept, cand_probs):
        idx = uci_to_vocab_index(all_moves_dict, fen, cand.uci)
        if idx < 0:
            continue
        logq = math.log(max(float(q), eps))
        logp_b = float(base_logp_full_1d[idx].item())
        kl += float(q) * (logq - logp_b)
    
    res = SfPerPosResult(
        inference_cp_best=int(best_cp),
        reference_cp_best=int(best_cp_ref) if best_cp_ref else None,
        entropy_cond_inference=float(ent),

        num_candidates=len(sf_inference_candidates),
        num_candidates_in_inference_window=len(kept),
        num_candidates_in_reference_window=len(kept_reference) if kept_reference else None,

        p_chosen_cond_inference=float(p_ch),
        p_rejected_cond_inference=float(p_rj),
        logp_chosen_cond_inference=float(logp_ch),
        logp_rejected_cond_inference=float(logp_rj),
        gap_logp_cond_inference=float(gap_logp),

        cand_hit1=float(cand_hit_at(1)),
        cand_hit3=float(cand_hit_at(3)),
        cand_hit5=float(cand_hit_at(5)),
        cand_hit10=float(cand_hit_at(10)),

        full_hit1=float(full_hit.get(1, 0)),
        full_hit3=float(full_hit.get(3, 0)),
        full_hit5=float(full_hit.get(5, 0)),
        full_hit10=float(full_hit.get(10, 0)),

        kl_q_vs_base=float(kl),

        expected_cp_cond_inference=sum([data.cp for data in kept]) / (len(kept) + eps),
        cp_std_cond_inference=float(np.std([float(data.cp) for data in kept])),
        top1_cp_cond_inference=best_cp,
        top1_uci_cond_inference=[data for data in kept if data.cp == best_cp][0].uci if len(kept) > 0 else "",
        expected_cp_cond_reference=sum([data.cp for data in kept_reference]) / (len(kept_reference) + eps) if kept_reference else None,
        cp_std_cond_reference=float(np.std([float(data.cp) for data in kept_reference])) if kept_reference else None,
        top1_cp_cond_reference=best_cp_ref if sf_reference_candidates else None,
        top1_uci_cond_reference=[data for data in kept_reference if data.cp == best_cp_ref][0].uci if kept_reference and len(kept_reference) > 0 else "",
        
        chosen_cp_inference=cp_chosen_inf,
        chosen_cp_reference=cp_chosen_ref if sf_reference_candidates else None,
        chosen_rank_reference=cp_chosen_ref_rank if sf_reference_candidates else None,
        chosen_rank_inference=cp_chosen_inf_rank,

        rejected_cp_inference=cp_rejected_inf,
        rejected_cp_reference=cp_rejected_ref if sf_reference_candidates else None,
        rejected_rank_reference=cp_rejected_ref_rank if sf_reference_candidates else None,
        rejected_rank_inference=cp_rejected_inf_rank,
    )

    dbg = {
        "ply_abs": int(ply_abs),
        "lambda": float(lam),
        "adaptive_restrict_cp_window": float(adaptive_restrict_cp_window),
        "cp_scale": float(cp_scale),
        "cp_cap": int(cp_cap),

        "policy_topk": policy_cand_details,   # [(uci, prob)] before SF filtering
        "cands_kept": kept,                   # [(uci, cp)]
        "q_probs": cand_probs,                # q over kept

        # Full-V tensor with combined scores on candidate indices, -inf elsewhere
        "cands_logits_full": cands_logits_full.detach().cpu(),  # [V]
    }
    return res, dbg


@torch.no_grad()
def compute_sf_helper_for_one_position(
    *,
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    logits_masked_1d: torch.Tensor,   # [V] for model under eval
    base_logp_full_1d: torch.Tensor,  # [V] log-softmax of base masked logits
    all_moves_dict: Dict[str, int],
    cfg: SfConfig,
    full_hit: Dict[int, int],         # {1:0/1, 3:0/1, 5:0/1, 10:0/1}
    rng: random.Random,
    sf_inference_candidates: List[_SFCandidate],
    sf_reference_candidates: Optional[List[_SFCandidate]],
) -> Optional[Tuple[SfPerPosResult, Dict[str, Any]]]:
    if not sf_inference_candidates:
        return None

    best_cp_inference = max(d.cp for d in sf_inference_candidates)
    best_cp_reference = max(d.cp for d in sf_reference_candidates) if sf_reference_candidates else None

    kept = sf_inference_candidates
    kept_reference = sf_reference_candidates

    if cfg.restrict_cp_window is not None:
        w = int(cfg.restrict_cp_window)

        filt_inf = [d for d in kept if d.cp >= best_cp_inference - w]
        if filt_inf:
            kept = filt_inf

        filt_ref = [d for d in kept_reference if d.cp >= best_cp_reference - w] if kept_reference and best_cp_reference else None
        if filt_ref:
            kept_reference = filt_ref

    if not kept:
        return None

    eps = float(getattr(cfg, "eps", 1e-12))
    t = max(float(cfg.temperature), 1e-6)

    # q distribution over kept inference candidates from model logits only (non-Gibbs)
    logp_all = torch.log_softmax(logits_masked_1d / t, dim=-1)  # [V]

    cand_moves: List[str] = [d.uci for d in kept]
    cand_idxs: List[int] = [uci_to_vocab_index(all_moves_dict, fen, u) for u in cand_moves]

    neg_inf = torch.finfo(logp_all.dtype).min
    cand_logps = []
    for idx in cand_idxs:
        if idx < 0:
            cand_logps.append(torch.tensor(neg_inf, device=logp_all.device, dtype=logp_all.dtype))
        else:
            cand_logps.append(logp_all[idx])

    cand_logps_t = torch.stack(cand_logps, dim=0)     # [K]
    cand_probs_t = torch.softmax(cand_logps_t, dim=0) # [K]
    cand_probs: List[float] = cand_probs_t.detach().cpu().tolist()

    # Full-V tensor with candidate logits only, -inf elsewhere
    cands_logits_full = torch.full_like(logits_masked_1d, neg_inf)
    for idx in cand_idxs:
        if idx >= 0:
            cands_logits_full[idx] = logits_masked_1d[idx]

    # select within inference candidate set
    if cfg.sample:
        r = rng.random()
        acc = 0.0
        sel_i = len(cand_probs) - 1
        for j, p in enumerate(cand_probs):
            acc += float(p)
            if r <= acc:
                sel_i = j
                break
    else:
        sel_i = int(torch.argmax(cand_probs_t).item())

    selected = kept[sel_i]
    selected_uci = selected.uci
    cp_selected = selected.cp
    cp_gap = float(best_cp_inference - cp_selected)
    is_best = (cp_gap <= 1e-9)

    probs_np = cand_probs_t.detach().cpu().numpy()
    ent = float(-(probs_np * np.log(probs_np + eps)).sum()) if probs_np.size else 0.0

    sel_idx = uci_to_vocab_index(all_moves_dict, fen, selected_uci)
    logp_selected_full = float(logp_all[sel_idx].item()) if sel_idx >= 0 else float("-inf")

    # chosen/rejected conditional probs in inference candidate set
    p_ch = float(cand_probs[cand_moves.index(chosen_uci)]) if chosen_uci in cand_moves else 0.0
    p_rj = float(cand_probs[cand_moves.index(rejected_uci)]) if rejected_uci in cand_moves else 0.0
    logp_ch = math.log(max(p_ch, eps))
    logp_rj = math.log(max(p_rj, eps))
    gap_logp = float(logp_ch - logp_rj)

    # candidate hits on chosen (based on q ordering)
    K = len(cand_probs)
    order = sorted(range(K), key=lambda j: cand_probs[j], reverse=True)

    def cand_hit_at(k_: int) -> float:
        if k_ <= 0:
            return 0.0
        kk = min(int(k_), K)
        top_moves = [cand_moves[j] for j in order[:kk]]
        return 1.0 if chosen_uci in top_moves else 0.0

    # KL(q || base_full) over inference candidate support
    kl = 0.0
    for cand, q in zip(kept, cand_probs):
        idx = uci_to_vocab_index(all_moves_dict, fen, cand.uci)
        if idx < 0:
            continue
        logq = math.log(max(float(q), eps))
        logp_b = float(base_logp_full_1d[idx].item())
        kl += float(q) * (logq - logp_b)

    def get_cp(cands: List[_SFCandidate], move_uci: str) -> Optional[int]:
        for d in cands:
            if d.uci == move_uci:
                return d.cp
        return None

    def get_stockfish_rank(cands: List[_SFCandidate], move_uci: str) -> Optional[int]:
        ranked = sorted(cands, key=lambda d: d.cp, reverse=True)
        for i, d in enumerate(ranked, start=1):
            if d.uci == move_uci:
                return i
        return None

    cp_chosen_inf = get_cp(sf_inference_candidates, chosen_uci)
    cp_chosen_ref = get_cp(sf_reference_candidates, chosen_uci) if sf_reference_candidates else None
    cp_rejected_inf = get_cp(sf_inference_candidates, rejected_uci)
    cp_rejected_ref = get_cp(sf_reference_candidates, rejected_uci) if sf_reference_candidates else None

    cp_chosen_inf_rank = get_stockfish_rank(sf_inference_candidates, chosen_uci)
    cp_chosen_ref_rank = get_stockfish_rank(sf_reference_candidates, chosen_uci) if sf_reference_candidates else None
    cp_rejected_inf_rank = get_stockfish_rank(sf_inference_candidates, rejected_uci)
    cp_rejected_ref_rank = get_stockfish_rank(sf_reference_candidates, rejected_uci) if sf_reference_candidates else None

    cps_kept_inf = [float(d.cp) for d in kept]
    cps_kept_ref = [float(d.cp) for d in kept_reference] if kept_reference else None

    top1_cp_cond_inference = int(max(d.cp for d in kept))
    top1_uci_cond_inference = next(d.uci for d in kept if d.cp == top1_cp_cond_inference)

    top1_cp_cond_reference = int(max(d.cp for d in kept_reference)) if kept_reference else None
    top1_uci_cond_reference = next(d.uci for d in kept_reference if d.cp == top1_cp_cond_reference) if kept_reference else None

    # Keeping naming/behavior aligned with your Gibbs version:
    # this is a plain mean over CPs in the window, not q-weighted expectation.
    expected_cp_cond_inference = float(sum(d.cp for d in kept) / (len(kept) + eps))
    expected_cp_cond_reference = float(sum(d.cp for d in kept_reference) / (len(kept_reference) + eps)) if kept_reference else None

    cp_std_cond_inference = float(np.std(cps_kept_inf))
    cp_std_cond_reference = float(np.std(cps_kept_ref)) if cps_kept_ref else None

    res = SfPerPosResult(
        inference_cp_best=int(best_cp_inference),
        reference_cp_best=int(best_cp_reference) if best_cp_reference else None,
        entropy_cond_inference=float(ent),

        num_candidates=len(sf_inference_candidates),
        num_candidates_in_inference_window=len(kept),
        num_candidates_in_reference_window=len(kept_reference) if kept_reference else None,

        p_chosen_cond_inference=float(p_ch),
        p_rejected_cond_inference=float(p_rj),
        logp_chosen_cond_inference=float(logp_ch),
        logp_rejected_cond_inference=float(logp_rj),
        gap_logp_cond_inference=float(gap_logp),

        cand_hit1=float(cand_hit_at(1)),
        cand_hit3=float(cand_hit_at(3)),
        cand_hit5=float(cand_hit_at(5)),
        cand_hit10=float(cand_hit_at(10)),

        full_hit1=float(full_hit.get(1, 0)),
        full_hit3=float(full_hit.get(3, 0)),
        full_hit5=float(full_hit.get(5, 0)),
        full_hit10=float(full_hit.get(10, 0)),

        kl_q_vs_base=float(kl),

        expected_cp_cond_inference=expected_cp_cond_inference,
        cp_std_cond_inference=cp_std_cond_inference,
        top1_cp_cond_inference=top1_cp_cond_inference,
        top1_uci_cond_inference=top1_uci_cond_inference,

        expected_cp_cond_reference=expected_cp_cond_reference if sf_reference_candidates else None,
        cp_std_cond_reference=cp_std_cond_reference if sf_reference_candidates else None,
        top1_cp_cond_reference=top1_cp_cond_reference if sf_reference_candidates else None,
        top1_uci_cond_reference=top1_uci_cond_reference if sf_reference_candidates else None,

        chosen_cp_inference=cp_chosen_inf,
        chosen_cp_reference=cp_chosen_ref if sf_reference_candidates else None,
        chosen_rank_reference=cp_chosen_ref_rank if sf_reference_candidates else None,
        chosen_rank_inference=cp_chosen_inf_rank,

        rejected_cp_inference=cp_rejected_inf,
        rejected_cp_reference=cp_rejected_ref if sf_reference_candidates else None,
        rejected_rank_reference=cp_rejected_ref_rank if sf_reference_candidates else None,
        rejected_rank_inference=cp_rejected_inf_rank,
    )

    dbg = {
        "cands_kept": kept,
        "cands_reference_kept": kept_reference if sf_reference_candidates else None,
        "q_probs": cand_probs,
        "selected_index": int(sel_i),
        "cands_logits_full": cands_logits_full.detach().cpu(),  # [V]
    }
    return res, dbg


# ============================================================
# Model abstractions
# ============================================================

class EvalModel(ABC):
    """
    Parent abstraction:
    - Loads a Maia2 base ref model
    - Loads a Maia2 policy model (possibly same as base, or with weights)
    - Computes *shared* metrics on (policy, base) on DpoPairs rows
    - Optionally computes SF-helper metrics (policy restricted to SF top-k candidates)

    Child classes override:
      - name/tag
      - policy weight path behavior
      - how to compute "training loss style" metric (dpo vs sft vs pairwise)
    """

    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        sf_cfg: Optional[SfConfig] = None,
        sf_engine: Optional[Any] = None
    ):
        self.maia_type = maia_type
        self.device = device
        self.policy_pt_path = policy_pt_path
        self.beta = float(beta)
        self.sf_cfg = sf_cfg

        # vocab + elo dict stable
        self.all_moves: List[str] = get_all_possible_moves()
        self.all_moves_dict: Dict[str, int] = {m: i for i, m in enumerate(self.all_moves)}
        self.elo_dict = create_elo_dict()

        # models
        self.base = maia_model.from_pretrained(type=self.maia_type, device=str(self.device)).to(self.device)
        self.policy = maia_model.from_pretrained(type=self.maia_type, device=str(self.device)).to(self.device)

        if self.policy_pt_path:
            self._load_policy_weights(self.policy, self.policy_pt_path)

        self.base.eval()
        self.policy.eval()

        # SF engine (single-process) if enabled
        self._sf_engine = sf_engine

    @property
    @abstractmethod
    def tag(self) -> str:
        """Used for output naming (e.g., 'dpo', 'sft', 'sft_pairwise', 'base')."""

    @staticmethod
    def _load_policy_weights(model: torch.nn.Module, pt_path: str) -> None:
        sd = torch.load(pt_path, map_location="cpu")
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] missing keys: {len(missing)} (showing 10): {missing[:10]}")
        if unexpected:
            print(f"[WARN] unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}")

    def close(self) -> None:
        if self._sf_engine is not None:
            try:
                self._sf_engine.quit()
            except Exception:
                pass
            self._sf_engine = None

    # -----------------------
    # Loss heads (override)
    # -----------------------

    @abstractmethod
    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
        logits_pi_m, 
        logits_ref_m, 
        idx_t, 
        batch_meta_data,
    ) -> torch.Tensor:
        """
        Return a scalar tensor loss that corresponds to the training objective style.

        - DPO: -logsigmoid(beta*((pi_gap)-(ref_gap)))
        - SFT: mean NLL chosen (policy only)
        - Pairwise-SFT: -logsigmoid((pi_gap)) (no ref)
        """

    # -----------------------
    # Shared eval
    # -----------------------


    def debug_opening_distribution(self, policy, device, topk: int = 20):
        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)

        def _as_batched_long(x):
            # embedding() requires LongTensor indices; also ensure batch dim [B]
            if isinstance(x, int):
                return torch.tensor([x], device=device, dtype=torch.long)
            x = _to_tensor(x)
            if x.dim() == 0:
                x = x.unsqueeze(0)
            elif x.dim() > 1:
                x = x.reshape(-1)  # defensive: flatten to [B] if something odd comes back
            return x.to(device=device, dtype=torch.long)

        def _as_batched_board(board_input):
            # Maia expects [B, 1152] so it can view -> [B, C, 8, 8] with C=18
            x = _to_tensor(board_input)

            # Common shapes we might get back from preprocessing:
            #   [18, 8, 8]  (C,8,8)
            #   [18, 64]    (C,64)
            #   [1152]      (flat)
            #   [B,1152]    (already batched)
            if x.dim() == 3 and x.shape[-2:] == (8, 8):
                x = x.reshape(1, -1)          # [1, 18*8*8]
            elif x.dim() == 2 and x.shape[-1] == 64:
                x = x.reshape(1, -1)          # [1, 18*64]
            elif x.dim() == 1:
                x = x.unsqueeze(0)            # [1, 1152]
            elif x.dim() == 2 and x.shape[-1] == 1152:
                pass                          # already [B,1152]
            else:
                # Last-resort: make it [1, -1] and hope it is 1152
                x = x.reshape(1, -1)

            # Keep dtype consistent with the model; float is safe here.
            return x.to(device=device, dtype=torch.float32)

        def _as_batched_mask(legal_moves):
            m = _to_tensor(legal_moves)
            # expected either [V] or [B,V] or something like [1,V]
            if m.dim() == 1:
                m = m.unsqueeze(0)
            return m.to(device=device)

        # --------------------------
        # Build opening position inputs
        # --------------------------
        board = chess.Board()
        fen = board.fen()

        # NOTE: inference.preprocessing signature can differ by maia2 version.
        # Your earlier error showed it returns 4 values in your env, so we handle both.
        prep_out = inference.preprocessing(
            fen,
            int(2000),
            int(2000),
            self.elo_dict,
            self.all_moves_dict,
        )

        if len(prep_out) == 4:
            board_input, es_t, eo_t, legal_moves = prep_out
        elif len(prep_out) == 5:
            board_input, es_t, eo_t, legal_moves, _ = prep_out
        else:
            raise ValueError(f"inference.preprocessing returned {len(prep_out)} values, expected 4 or 5")

        board_input = _as_batched_board(board_input)
        es_t = _as_batched_long(es_t)
        eo_t = _as_batched_long(eo_t)
        legal_moves = _as_batched_mask(legal_moves)

        # --------------------------
        # Forward + mask to legal
        # --------------------------
        policy.eval()
        with torch.no_grad():
            logits = forward_logits(policy, board_input, es_t, eo_t)
            # allow either [V] or [B,V]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            logits = logits[0]  # [V]

        mask = legal_moves[0].bool()  # [V]

        print("=== Opening distribution ===")
        print(f"fen: {fen}")
        print(f"board_input shape: {tuple(board_input.shape)} (expected [1,1152])")
        print(f"logits shape: {tuple(logits.shape)}")
        print(f"mask shape: {tuple(mask.shape)}  num_legal(mask.sum)={int(mask.sum())}")

        # If this is 0, your legal mask is in a different space than the policy vocab
        if int(mask.sum()) == 0:
            # extra diagnostics to help you immediately see what's off
            nz = torch.nonzero(mask, as_tuple=False).reshape(-1)
            print(f"mask nonzero count: {nz.numel()}")
            print("WARNING: num legal is 0. This usually means legal_moves mask is not aligned with policy vocab.")
            print("First 64 mask values:", mask[:64].to(torch.int).tolist())
            return

        legal_logits = logits[mask]
        probs = torch.softmax(legal_logits, dim=-1)

        ent = float(-(probs * torch.log(probs + 1e-12)).sum())
        print("entropy:", ent)

        k = min(topk, probs.numel())
        vals, idxs = torch.topk(probs, k=k)

        # Map back from "legal-space index" -> "global vocab index"
        legal_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)

        for rank, (p, j) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
            vocab_idx = int(legal_indices[j].item())
            print(f"[{rank:02d}] vocab_idx={vocab_idx:4d}  p={p:.4f}")

    def _compute_eval_per_row_input(
        self,
        batch: Dict[str, Any],
        opening_counts_adv: Dict[str, Counter],
        dataset: DpoPairs,
        opening_cfg: OpeningLogitDistConfig,
        opening_by_game: Dict[str, str],
    ) -> EvalPerRowInput:
        fens = batch["fen"]
        es = batch["elo_self"]
        eo = batch["elo_oppo"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        batch_size = len(fens)

        game_ids = batch["game_id"]
        ply_idxs = batch["ply_idx"]
        opening_prefixes = batch["opening_prefix_uci_20"]
        meta_list = batch["meta"]

        board_input, legal_moves, es_t, eo_t = batch_preprocess(
            all_moves_dict=self.all_moves_dict,
            elo_dict=self.elo_dict,
            fens=fens,
            elo_self=es,
            elo_oppo=eo,
            device=self.device,
        )

        logits_pi = forward_logits(self.policy, board_input, es_t, eo_t)
        logits_ref = forward_logits(self.base, board_input, es_t, eo_t)

        logits_pi_m = apply_legal_mask(logits_pi, legal_moves)
        logits_ref_m = apply_legal_mask(logits_ref, legal_moves)

        chosen_idx = torch.tensor(
            [uci_to_vocab_index(self.all_moves_dict, fen, u) for fen, u in zip(fens, chosen)],
            device=self.device,
            dtype=torch.long,
        )
        rejected_idx = torch.tensor(
            [uci_to_vocab_index(self.all_moves_dict, fen, u) for fen, u in zip(fens, rejected)],
            device=self.device,
            dtype=torch.long,
        )

        chosen_ok = (chosen_idx >= 0) & (
            legal_moves.gather(1, chosen_idx.clamp(min=0).view(-1, 1)).squeeze(1) > 0
        )
        rejected_ok = (rejected_idx >= 0) & (
            legal_moves.gather(1, rejected_idx.clamp(min=0).view(-1, 1)).squeeze(1) > 0
        )
        bad = ~(chosen_ok & rejected_ok)
        if bad.any():
            j = int(bad.nonzero()[0])
            raise RuntimeError(
                f"Illegal chosen/rejected. fen={fens[j]} chosen={chosen[j]} rejected={rejected[j]}"
            )

        logp_pi_ch = gather_logprob_from_masked_logits(logits_pi_m, chosen_idx)
        logp_pi_rj = gather_logprob_from_masked_logits(logits_pi_m, rejected_idx)
        logp_ref_ch = gather_logprob_from_masked_logits(logits_ref_m, chosen_idx)
        logp_ref_rj = gather_logprob_from_masked_logits(logits_ref_m, rejected_idx)

        chosen_cps = [extract_move_cp(m, ch) for m, ch in zip(meta_list, chosen)]
        rejected_cps = [extract_move_cp(m, rj) for m, rj in zip(meta_list, rejected)]

        prev_fens_batch = [
            safe_get_prev_fens(dataset.game_id_and_ply_to_prev_10_plys, m, n=5)
            for m in meta_list
        ]
        next_fens_chosen_batch = [
            safe_get_next_fens_chosen(dataset.game_id_and_ply_to_fut_10_plys, m, n=5)
            for m in meta_list
        ]
        next_fens_rejected_batch = [
            safe_get_next_fens_rejected(fen, rj, n=5)
            for fen, rj in zip(fens, rejected)
        ]

        batch_meta_data = list(
            zip(
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
        )

        loss = self.compute_training_style_loss(
            logp_pi_ch=logp_pi_ch,
            logp_pi_rj=logp_pi_rj,
            logp_ref_ch=logp_ref_ch,
            logp_ref_rj=logp_ref_rj,
            logits_pi_m=logits_pi_m,
            logits_ref_m=logits_ref_m,
            idx_t=chosen_idx,
            batch_meta_data=batch_meta_data,
        )

        update_opening_distributions_from_logits(
            opening_counts=opening_counts_adv,
            fens=fens,
            logits_masked=logits_pi_m,
            all_moves=self.all_moves,
            cfg=opening_cfg,
        )

        for gid, prefix in zip(game_ids, opening_prefixes):
            if gid not in opening_by_game:
                opening_by_game[gid] = coarse_opening_family_from_prefix(prefix)

        return EvalPerRowInput.from_dict({
            "logp_pi_ch": logp_pi_ch,
            "logp_pi_rj": logp_pi_rj,
            "logp_ref_ch": logp_ref_ch,
            "logp_ref_rj": logp_ref_rj,
            "logits_pi_m": logits_pi_m,
            "logits_ref_m": logits_ref_m,
            "chosen_idx": chosen_idx,
            "batch_size": batch_size,
            "ply_idxs": ply_idxs,
            "opening_counts_adv": opening_counts_adv,
            "fens": fens,
            "loss": loss,
            "game_ids": game_ids,
            "chosen": chosen,
            "rejected": rejected,
            "opening_cfg": opening_cfg,
            "opening_by_game": opening_by_game,
            "opening_prefixes": opening_prefixes,
        })

    def _add_per_row_metrics(
        self,
        eval_input: EvalPerRowInput,
        sums: Dict[str, float],
        rows: List[Dict[str, Any]],
        phase_buckets: Dict[Tuple[str, str], List[float]],
    ) -> None:
        logp_pi_ch = eval_input.logp_pi_ch
        logp_pi_rj = eval_input.logp_pi_rj
        logp_ref_ch = eval_input.logp_ref_ch
        logp_ref_rj = eval_input.logp_ref_rj
        logits_pi_m = eval_input.logits_pi_m
        logits_ref_m = eval_input.logits_ref_m
        chosen_idx = eval_input.chosen_idx
        fens = eval_input.fens
        game_ids = eval_input.game_ids
        chosen = eval_input.chosen
        rejected = eval_input.rejected

        p_chosen_pi = chosen_probability(logits_pi_m, fens, self.all_moves_dict, chosen)
        p_chosen_ref = chosen_probability(logits_ref_m, fens, self.all_moves_dict, chosen)

        hit1_pi = hit_at_k(logits_pi_m, chosen_idx, 1)
        hit5_pi = hit_at_k(logits_pi_m, chosen_idx, 5)
        hit10_pi = hit_at_k(logits_pi_m, chosen_idx, 10)

        hit1_ref = hit_at_k(logits_ref_m, chosen_idx, 1)
        hit5_ref = hit_at_k(logits_ref_m, chosen_idx, 5)
        hit10_ref = hit_at_k(logits_ref_m, chosen_idx, 10)

        rank_pi = chosen_rank(logits_pi_m, chosen_idx)
        rank_ref = chosen_rank(logits_ref_m, chosen_idx)

        kl_pi_ref = kl_policy_base_from_logits(logits_pi_m, logits_ref_m)
        entropy_pi = entropy_from_logits(logits_pi_m)
        entropy_ref = entropy_from_logits(logits_ref_m)

        pi_gap = logp_pi_ch - logp_pi_rj
        ref_gap = logp_ref_ch - logp_ref_rj
        gap_improve = pi_gap - ref_gap
        mrr = 1.0 / rank_pi.float()

        probs_pi = torch.softmax(logits_pi_m, dim=-1)
        probs_ref = torch.softmax(logits_ref_m, dim=-1)

        top3_pi = torch.topk(logits_pi_m, k=min(3, logits_pi_m.size(-1)), dim=-1).indices
        top10_pi = torch.topk(logits_pi_m, k=min(10, logits_pi_m.size(-1)), dim=-1).indices

        for i in range(eval_input.batch_size):
            ply_abs = fen_to_ply_abs(fens[i])
            phase = ply_to_phase(ply_abs)

            pred_idx = int(torch.argmax(logits_pi_m[i]).item())
            pred_uci = vocab_index_to_uci(self.all_moves, fens[i], pred_idx)

            chosen_vocab_idx = int(chosen_idx[i].item())
            chosen_is_in_top3 = bool((top3_pi[i] == chosen_vocab_idx).any().item())
            chosen_is_in_top10 = bool((top10_pi[i] == chosen_vocab_idx).any().item())

            row = {
                "game_id": game_ids[i],
                "ply_idx": int(eval_input.ply_idxs[i]),
                "ply_abs": int(ply_abs),
                "phase": phase,
                "fen": fens[i],
                "chosen_uci": chosen[i],
                "rejected_uci": rejected[i],
                "pred_uci": pred_uci,

                "correct_top1": float(hit1_pi[i].item()),
                "hit_top3": float(chosen_is_in_top3),
                "hit_top5": float(hit5_pi[i].item()),
                "hit_top10": float(hit10_pi[i].item()),
                "rank_chosen": int(rank_pi[i].item()),
                "mrr": float(mrr[i].item()),

                "logp_gap_pi": float(pi_gap[i].item()),
                "logp_gap_ref": float(ref_gap[i].item()),
                "gap_improve": float(gap_improve[i].item()),

                "p_chosen_pi": float(p_chosen_pi[i].item()),
                "p_chosen_ref": float(p_chosen_ref[i].item()),
                "kl_pi_ref": float(kl_pi_ref[i].item()),
                "nll_chosen_pi": float((-logp_pi_ch[i]).item()),

                "entropy_pi": float(entropy_pi[i].item()),
                "entropy_ref": float(entropy_ref[i].item()),

                "correct_top1_base": float(hit1_ref[i].item()),
                "hit_top5_base": float(hit5_ref[i].item()),
                "hit_top10_base": float(hit10_ref[i].item()),
                "rank_chosen_base": int(rank_ref[i].item()),

                "chosen_is_in_top_ten": float(chosen_is_in_top10),
                "chosen_is_in_top_three": float(chosen_is_in_top3),

                "opening_family": eval_input.opening_by_game.get(game_ids[i], "Unknown"),

                # these are useful for old enrichment helpers
                "prob": float(torch.max(probs_pi[i]).item()),
                "prob_ref": float(torch.max(probs_ref[i]).item()),
            }
            rows.append(row)

            sums["pi_gap"] += float(pi_gap[i].item())
            sums["ref_gap"] += float(ref_gap[i].item())
            sums["gap_improve"] += float(gap_improve[i].item())
            sums["top1"] += float(hit1_pi[i].item())
            sums["top1_base"] += float(hit1_ref[i].item())
            sums["hit3"] += float(chosen_is_in_top3)
            sums["hit5"] += float(hit5_pi[i].item())
            sums["hit10"] += float(hit10_pi[i].item())
            sums["mrr"] += float(mrr[i].item())
            sums["p_chosen"] += float(p_chosen_pi[i].item())
            sums["p_chosen_base"] += float(p_chosen_ref[i].item())
            sums["kl"] += float(kl_pi_ref[i].item())
            sums["ent_pi"] += float(entropy_pi[i].item())
            sums["ent_ref"] += float(entropy_ref[i].item())

            if chosen_is_in_top10:
                sums["pi_gap_cond_in_top10"] += float(pi_gap[i].item())
                sums["ref_gap_cond_in_top10"] += float(ref_gap[i].item())
                sums["gap_improve_cond_in_top10"] += float(gap_improve[i].item())
                sums["p_chosen_cond_in_top10"] += float(p_chosen_pi[i].item())
                sums["p_chosen_base_cond_in_top10"] += float(p_chosen_ref[i].item())
                sums["kl_cond_in_top10"] += float(kl_pi_ref[i].item())
                sums["top5_cond_in_top10"] += float(hit5_pi[i].item())
                sums["top10_cond_in_top10"] += float(hit10_pi[i].item())
                sums["n_in_top10"] += 1.0
            else:
                sums["pi_gap_cond_not_in_top10"] += float(pi_gap[i].item())
                sums["ref_gap_cond_not_in_top10"] += float(ref_gap[i].item())
                sums["gap_improve_cond_not_in_top10"] += float(gap_improve[i].item())
                sums["p_chosen_cond_not_in_top10"] += float(p_chosen_pi[i].item())
                sums["p_chosen_base_cond_not_in_top10"] += float(p_chosen_ref[i].item())
                sums["kl_cond_not_in_top10"] += float(kl_pi_ref[i].item())
                sums["n_not_in_top10"] += 1.0

            phase_buckets[("kl_pi_ref", phase)].append(float(kl_pi_ref[i].item()))
            phase_buckets[("logp_gap_pi", phase)].append(float(pi_gap[i].item()))
            phase_buckets[("p_chosen_pi", phase)].append(float(p_chosen_pi[i].item()))
            phase_buckets[("correct_top1", phase)].append(float(hit1_pi[i].item()))
            phase_buckets[("entropy_pi", phase)].append(float(entropy_pi[i].item()))
            phase_buckets[("entropy_ref", phase)].append(float(entropy_ref[i].item()))
            
        sums["loss"] += float(eval_input.loss.item()) * eval_input.batch_size

    def generate_aggregate_eval_metrics(
        self,
        num_rows: int,
        opening_by_game: Dict[str, str],
        sums: Dict[str, float],
        gm_name: str,
        dataset: DpoPairs,
        batch_size: int,
        opening_counts_adv: Dict[str, Counter],
        per_rows: List[Dict[str, Any]],
        phase_buckets: Dict[Tuple[str, str], List[float]],
        n_boot: int = 1000,
    ) -> Dict[str, Any]:
        def avg_total(key: str) -> float:
            return float(sums[key] / max(1, num_rows))

        def avg_cond(key: str, denom_key: str) -> float:
            return float(sums[key] / max(1.0, sums[denom_key]))

        opening_counts = Counter(opening_by_game.values())
        opening_dist = {k: v for k, v in opening_counts.most_common()}
        opening_summary = summarize_opening_distribution(opening_counts_adv, topn=50, normalize=True)

        stockfish_data = None
        if self.sf_cfg is not None:
            stockfish_data = self._run_sf_helper_eval(ds=dataset, batch_size=batch_size)

        top1_p = _precision_at_k(per_rows, "correct_top1")
        top1_r = _recall_at_k(per_rows, "correct_top1")
        top3_p = _precision_at_k(per_rows, "hit_top3")
        top3_r = _recall_at_k(per_rows, "hit_top3")
        top5_p = _precision_at_k(per_rows, "hit_top5")
        top5_r = _recall_at_k(per_rows, "hit_top5")
        top10_p = _precision_at_k(per_rows, "hit_top10")
        top10_r = _recall_at_k(per_rows, "hit_top10")

        chosen_in_top10_rows = [r for r in per_rows if float(r.get("chosen_is_in_top_ten", 0.0)) > 0.0]
        chosen_not_in_top10_rows = [r for r in per_rows if float(r.get("chosen_is_in_top_ten", 0.0)) <= 0.0]

        phase_summary = {
            f"{metric}__{phase}": {
                "mean": _safe_mean(vals),
                "n": len(vals),
            }
            for (metric, phase), vals in phase_buckets.items()
            if vals
        }

        agg = {
            "gm": gm_name,
            "tag": self.tag,
            "maia_type": self.maia_type,
            "device": str(self.device),
            "num_rows": num_rows,

            "loss": avg_total("loss"),
            "mean_logp_gap_policy_chosen_rejected": avg_total("pi_gap"),
            "mean_logp_gap_base_chosen_rejected": avg_total("ref_gap"),
            "mean_gap_improvement": avg_total("gap_improve"),

            "top1_accuracy_on_chosen_policy": avg_total("top1"),
            "top1_accuracy_on_chosen_base": avg_total("top1_base"),
            "hit3_policy": avg_total("hit3"),
            "hit5_policy": avg_total("hit5"),
            "hit10_policy": avg_total("hit10"),
            "mrr": avg_total("mrr"),

            "mean_p_chosen_policy": avg_total("p_chosen"),
            "mean_p_chosen_base": avg_total("p_chosen_base"),
            "mean_kl": avg_total("kl"),
            "mean_ent_pi": avg_total("ent_pi"),
            "mean_ent_ref": avg_total("ent_ref"),

            "top1_precision": top1_p,
            "top1_recall": top1_r,
            "top1_f1": _f1_from_pr(top1_p, top1_r),
            "top3_precision": top3_p,
            "top3_recall": top3_r,
            "top3_f1": _f1_from_pr(top3_p, top3_r),
            "top5_precision": top5_p,
            "top5_recall": top5_r,
            "top5_f1": _f1_from_pr(top5_p, top5_r),
            "top10_precision": top10_p,
            "top10_recall": top10_r,
            "top10_f1": _f1_from_pr(top10_p, top10_r),

            "mean_logp_gap_policy_chosen_rejected_cond_on_in_top_ten": avg_cond("pi_gap_cond_in_top10", "n_in_top10"),
            "mean_logp_gap_base_chosen_rejected_cond_on_in_top_ten": avg_cond("ref_gap_cond_in_top10", "n_in_top10"),
            "mean_gap_improvement_cond_on_in_top_ten": avg_cond("gap_improve_cond_in_top10", "n_in_top10"),
            "mean_p_chosen_policy_cond_on_in_top_ten": avg_cond("p_chosen_cond_in_top10", "n_in_top10"),
            "mean_p_chosen_base_cond_on_in_top_ten": avg_cond("p_chosen_base_cond_in_top10", "n_in_top10"),
            "mean_kl_cond_on_in_top_ten": avg_cond("kl_cond_in_top10", "n_in_top10"),

            "mean_logp_gap_policy_chosen_rejected_cond_on_not_in_top_ten": avg_cond("pi_gap_cond_not_in_top10", "n_not_in_top10"),
            "mean_logp_gap_base_chosen_rejected_cond_on_not_in_top_ten": avg_cond("ref_gap_cond_not_in_top10", "n_not_in_top10"),
            "mean_gap_improvement_cond_on_not_in_top_ten": avg_cond("gap_improve_cond_not_in_top10", "n_not_in_top10"),
            "mean_p_chosen_policy_cond_on_not_in_top_ten": avg_cond("p_chosen_cond_not_in_top10", "n_not_in_top10"),
            "mean_p_chosen_base_cond_on_not_in_top_ten": avg_cond("p_chosen_base_cond_on_not_in_top10", "n_not_in_top10") if "p_chosen_base_cond_on_not_in_top10" in sums else avg_cond("p_chosen_base_cond_not_in_top10", "n_not_in_top10"),
            "mean_kl_cond_on_not_in_top_ten": avg_cond("kl_cond_not_in_top10", "n_not_in_top10"),

            "top5_precision_cond_on_in_top_ten": _precision_at_k(chosen_in_top10_rows, "hit_top5"),
            "top5_recall_cond_on_in_top_ten": _recall_at_k(chosen_in_top10_rows, "hit_top5"),
            "top5_f1_cond_on_in_top_ten": _f1_from_pr(
                _precision_at_k(chosen_in_top10_rows, "hit_top5"),
                _recall_at_k(chosen_in_top10_rows, "hit_top5"),
            ),
            "top10_precision_cond_on_in_top_ten": _precision_at_k(chosen_in_top10_rows, "hit_top10"),
            "top10_recall_cond_on_in_top_ten": _recall_at_k(chosen_in_top10_rows, "hit_top10"),
            "top10_f1_cond_on_in_top_ten": _f1_from_pr(
                _precision_at_k(chosen_in_top10_rows, "hit_top10"),
                _recall_at_k(chosen_in_top10_rows, "hit_top10"),
            ),

            "opening_family_counts_by_game": opening_dist,
            "opening_summary": opening_summary,
            "phase_summary": phase_summary,
            "stockfish": stockfish_data,

            "bootstrap_ci_row": {
                "accuracy_top1": _bootstrap_ci([float(r["correct_top1"]) for r in per_rows], n_boot=n_boot, seed=1),
                "mean_logp_gap_pi": _bootstrap_ci([float(r["logp_gap_pi"]) for r in per_rows], n_boot=n_boot, seed=2),
                "mean_p_chosen_pi": _bootstrap_ci([float(r["p_chosen_pi"]) for r in per_rows], n_boot=n_boot, seed=3),
                "mrr": _bootstrap_ci([float(r["mrr"]) for r in per_rows], n_boot=n_boot, seed=4),
            },
            "bootstrap_ci_cluster_by_game_player_chosen": {
                "accuracy_top1": _cluster_bootstrap_ci(per_rows, "game_id", "correct_top1", n_boot=n_boot, seed=10),
                "mean_logp_gap_pi": _cluster_bootstrap_ci(per_rows, "game_id", "logp_gap_pi", n_boot=n_boot, seed=11),
                "mean_p_chosen_pi": _cluster_bootstrap_ci(per_rows, "game_id", "p_chosen_pi", n_boot=n_boot, seed=12),
                "mrr": _cluster_bootstrap_ci(per_rows, "game_id", "mrr", n_boot=n_boot, seed=13),
            },
            "notes": {
                "opening_family_is_coarse_heuristic_player_chosen": True,
                "precision_recall_f1_equals_accuracy_for_top1_hit": True,
            },
        }

        return agg

    @torch.inference_mode()
    def run_eval(
        self,
        *,
        ds: DpoPairs,
        batch_size: int,
        n_boot: int,
        out_dir: Path,
        gm_name: str,
    ) -> Dict[str, Any]:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )

        num_rows = 0
        sums = defaultdict(float)
        per_rows: List[Dict[str, Any]] = []
        phase_buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        opening_by_game: Dict[str, str] = {}

        opening_counts_adv = {"ply0_white": Counter(), "ply1_black": Counter()}
        opening_cfg = OpeningLogitDistConfig(plies=(0, 1), temperature=1.0, topk=50)

        for batch in loader:
            eval_input = self._compute_eval_per_row_input(
                batch=batch,
                opening_counts_adv=opening_counts_adv,
                dataset=ds,
                opening_cfg=opening_cfg,
                opening_by_game=opening_by_game,
            )
            self._add_per_row_metrics(eval_input, sums, per_rows, phase_buckets)
            num_rows += eval_input.batch_size

        per_rows_sorted = sorted(per_rows, key=lambda r: (r.get("game_id"), r.get("ply_idx")))
        per_rows_sorted = add_piece_selection_per_row_stats(per_rows_sorted)

        eval_agg_metrics = self.generate_aggregate_eval_metrics(
            num_rows=num_rows,
            opening_by_game=opening_by_game,
            sums=sums,
            gm_name=gm_name,
            dataset=ds,
            batch_size=batch_size,
            opening_counts_adv=opening_counts_adv,
            per_rows=per_rows_sorted,
            phase_buckets=phase_buckets,
            n_boot=n_boot,
        )

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

            json_path = out_dir / f"eval_results__{self.tag}.json"
            json_path.write_text(json.dumps(eval_agg_metrics, indent=2))
            print(f"Eval results saved to {json_path}")

            if per_rows_sorted:
                per_row_jsonl = out_dir / f"eval_per_row__{self.tag}.jsonl"
                with open(per_row_jsonl, "w", encoding="utf-8") as f:
                    for row in per_rows_sorted:
                        f.write(json.dumps(row) + "\n")

                per_csv = out_dir / f"eval_per_row__{self.tag}.csv"
                with open(per_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(per_rows_sorted[0].keys()))
                    writer.writeheader()
                    writer.writerows(per_rows_sorted)

                summary_csv = out_dir / f"eval_summary__{self.tag}.csv"
                with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "loss",
                        "mean_logp_gap_policy_chosen_rejected",
                        "mean_logp_gap_base_chosen_rejected",
                        "mean_gap_improvement",
                        "top1_accuracy_on_chosen_policy",
                        "top1_accuracy_on_chosen_base",
                        "mean_p_chosen_policy",
                        "mean_p_chosen_base",
                        "mean_kl",
                        "mean_ent_pi",
                        "mean_ent_ref",
                    ])
                    writer.writerow([
                        eval_agg_metrics["loss"],
                        eval_agg_metrics["mean_logp_gap_policy_chosen_rejected"],
                        eval_agg_metrics["mean_logp_gap_base_chosen_rejected"],
                        eval_agg_metrics["mean_gap_improvement"],
                        eval_agg_metrics["top1_accuracy_on_chosen_policy"],
                        eval_agg_metrics["top1_accuracy_on_chosen_base"],
                        eval_agg_metrics["mean_p_chosen_policy"],
                        eval_agg_metrics["mean_p_chosen_base"],
                        eval_agg_metrics["mean_kl"],
                        eval_agg_metrics["mean_ent_pi"],
                        eval_agg_metrics["mean_ent_ref"],
                    ])

        return eval_agg_metrics

    def _run_sf_helper_eval(self, *, ds: DpoPairs, batch_size: int = 64) -> Dict[str, Any]:
        """
        SF-helper evaluation:
        - SF gives top-k PV candidates at depth=cfg.depth (multipv=cfg.multipv_topk)
        - Policy logits restricted to candidate set to produce q over the inference-window candidates
        - Metrics computed on q, on chosen/rejected diagnostics, and on inference/reference windows
        - KL(q || base_full) computed on candidate support using base masked log-softmax
        """
        assert self.sf_cfg is not None
        assert self._sf_engine is not None

        sf_cfg = self.sf_cfg
        rng = random.Random(sf_cfg.seed)

        dataloader = self._build_sf_helper_eval_dataloader(ds=ds, batch_size=batch_size)
        opening_logits_cfg = OpeningLogitDistConfig(plies=(0, 1), temperature=1.0, topk=50)
        opening_distribution_counts = {"ply0_white": Counter(), "ply1_black": Counter()}
        aggregate = SfHelperEvalAggregate()

        for batch in dataloader:
            batch_ctx = self._prepare_sf_eval_batch_context(batch=batch)
            opening_fens: List[str] = []
            opening_candidate_logits: List[torch.Tensor] = []

            for row_index in range(len(batch_ctx.fens)):
                aggregate.total_rows += 1

                row_output = self._evaluate_single_sf_row(
                    batch_ctx=batch_ctx,
                    row_index=row_index,
                    sf_cfg=sf_cfg,
                    rng=rng,
                )
                if row_output is None:
                    continue

                result, debug_info = row_output
                fen = batch_ctx.fens[row_index]
                ply_abs = fen_to_ply_abs(fen)

                aggregate.add_processed_row(
                    result=result,
                    ply_abs=ply_abs,
                )

                if ply_abs in opening_logits_cfg.plies:
                    opening_fens.append(fen)
                    opening_candidate_logits.append(debug_info["cands_logits_full"])

            self._update_sf_opening_distributions(
                opening_distribution_counts=opening_distribution_counts,
                opening_fens=opening_fens,
                opening_candidate_logits=opening_candidate_logits,
                opening_logits_cfg=opening_logits_cfg,
            )

        opening_summary = summarize_opening_distribution(
            opening_distribution_counts,
            topn=50,
            normalize=True,
        )
        return aggregate.to_dict(
            sf_config=sf_cfg,
            sf_opening_summary=opening_summary,
        )

    def _build_sf_helper_eval_dataloader(self, *, ds: DpoPairs, batch_size: int) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )


    def _prepare_sf_eval_batch_context(self, *, batch: Dict[str, Any]) -> _SfBatchContext:
        fens = batch["fen"]
        elo_self = batch["elo_self"]
        elo_oppo = batch["elo_oppo"]
        chosen_uci_list = batch["chosen"]
        rejected_uci_list = batch["rejected"]
        stockfish_inference = [batch["meta"][i].get("stockfish_inference", None) for i in range(len(batch["meta"]))]
        stockfish_reference = [batch["meta"][i].get("stockfish_reference", None) for i in range(len(batch["meta"]))]

        board_input, legal_moves, elo_self_t, elo_oppo_t = batch_preprocess(
            all_moves_dict=self.all_moves_dict,
            elo_dict=self.elo_dict,
            fens=fens,
            elo_self=elo_self,
            elo_oppo=elo_oppo,
            device=self.device,
        )

        policy_logits = forward_logits(self.policy, board_input, elo_self_t, elo_oppo_t)
        base_logits = forward_logits(self.base, board_input, elo_self_t, elo_oppo_t)

        masked_policy_logits = apply_legal_mask(policy_logits, legal_moves)
        masked_base_logits = apply_legal_mask(base_logits, legal_moves)
        base_full_log_probs = torch.log_softmax(masked_base_logits, dim=-1)

        chosen_move_indices = torch.tensor(
            [uci_to_vocab_index(self.all_moves_dict, fen, uci) for fen, uci in zip(fens, chosen_uci_list)],
            device=self.device,
            dtype=torch.long,
        )

        full_hit1 = hit_at_k(masked_policy_logits, chosen_move_indices, 1)
        full_hit5 = hit_at_k(masked_policy_logits, chosen_move_indices, 5)
        full_hit10 = hit_at_k(masked_policy_logits, chosen_move_indices, 10)

        return _SfBatchContext(
            fens=fens,
            chosen_uci_list=chosen_uci_list,
            rejected_uci_list=rejected_uci_list,
            masked_policy_logits=masked_policy_logits,
            base_full_log_probs=base_full_log_probs,
            full_hit1=full_hit1,
            full_hit5=full_hit5,
            full_hit10=full_hit10,
            stockfish_inference=stockfish_inference,
            stockfish_reference=stockfish_reference
        )


    def _evaluate_single_sf_row(
        self,
        *,
        batch_ctx: _SfBatchContext,
        row_index: int,
        sf_cfg: Any,
        rng: random.Random,
    ) -> Optional[Tuple[SfPerPosResult, Dict[str, Any]]]:
        fen = batch_ctx.fens[row_index]
        chosen_uci = batch_ctx.chosen_uci_list[row_index]
        rejected_uci = batch_ctx.rejected_uci_list[row_index]

        board = chess.Board(fen)
        if board.is_game_over(claim_draw=True):
            return None

        row_full_hit = {
            1: int(batch_ctx.full_hit1[row_index].item()),
            5: int(batch_ctx.full_hit5[row_index].item()),
            10: int(batch_ctx.full_hit10[row_index].item()),
        }

        if sf_cfg.use_gibbs:
            return compute_sf_helper_w_gibbs_for_one_position(
                fen=fen,
                chosen_uci=chosen_uci,
                rejected_uci=rejected_uci,
                logits_masked_1d=batch_ctx.masked_policy_logits[row_index],
                base_logp_full_1d=batch_ctx.base_full_log_probs[row_index],
                all_moves_dict=self.all_moves_dict,
                cfg=sf_cfg,
                full_hit=row_full_hit,
                rng=rng,
                sf_inference_candidates=batch_ctx.stockfish_inference[row_index],
                sf_reference_candidates=batch_ctx.stockfish_reference[row_index],
            )

        return compute_sf_helper_for_one_position(
            fen=fen,
            chosen_uci=chosen_uci,
            rejected_uci=rejected_uci,
            logits_masked_1d=batch_ctx.masked_policy_logits[row_index],
            base_logp_full_1d=batch_ctx.base_full_log_probs[row_index],
            all_moves_dict=self.all_moves_dict,
            cfg=sf_cfg,
            full_hit=row_full_hit,
            rng=rng,
            sf_inference_candidates=batch_ctx.stockfish_inference[row_index],
            sf_reference_candidates=batch_ctx.stockfish_reference[row_index],
        )

    def _update_sf_opening_distributions(
        self,
        *,
        opening_distribution_counts: Dict[str, Counter],
        opening_fens: List[str],
        opening_candidate_logits: List[torch.Tensor],
        opening_logits_cfg: Any,
    ) -> None:
        if not opening_candidate_logits:
            return

        logits_stack = torch.stack(opening_candidate_logits, dim=0).to(self.device)
        update_opening_distributions_from_logits(
            opening_counts=opening_distribution_counts,
            fens=opening_fens,
            logits_masked=logits_stack,
            all_moves=self.all_moves,
            cfg=opening_logits_cfg,
        )

def _serialize_sf_info(info: Dict[str, Any], board: chess.Board) -> Optional[Dict[str, Any]]:
    pv = info.get("pv")
    score = info.get("score")

    if not pv or score is None:
        return None

    try:
        pv_uci = [mv.uci() for mv in pv]
    except Exception:
        pv_uci = []

    out: Dict[str, Any] = {
        "pv": pv_uci,
        "best_move": pv_uci[0] if pv_uci else None,
        "cp": _score_to_cp(score, turn=board.turn),
    }

    if "depth" in info:
        try:
            out["depth"] = int(info["depth"])
        except Exception:
            pass

    if "seldepth" in info:
        try:
            out["seldepth"] = int(info["seldepth"])
        except Exception:
            pass

    if "nodes" in info:
        try:
            out["nodes"] = int(info["nodes"])
        except Exception:
            pass

    if "time" in info:
        try:
            out["time"] = float(info["time"])
        except Exception:
            pass

    if "nps" in info:
        try:
            out["nps"] = int(info["nps"])
        except Exception:
            pass

    return out

import time 

def _safe_get_move_fields(row: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Best-effort extraction of move-ish fields from a row for debugging.
    Adjust keys if your dataset uses different names.
    """
    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}

    return {
        "chosen_move": row.get("chosen_move") or meta.get("chosen_move"),
        "rejected_move": row.get("rejected_move") or meta.get("rejected_move"),
        "move": row.get("move") or meta.get("move"),
    }


def _base_stockfish_meta(
    sf_cfg: "SfConfig",
    reason: str,
    *,
    error: Optional[str] = None,
    legal_moves: Optional[int] = None,
    elapsed_s: Optional[float] = None,
    sf_moves_returned: Optional[List[List[Any]]] = None,
    best_cp: Optional[int] = None,
    infos: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "depth": int(sf_cfg.depth),
        "multipv_requested": int(sf_cfg.multipv_topk),
        "reason": reason,
        "error": error,
        "legal_moves": legal_moves,
        "elapsed_s": elapsed_s,
        "sf_moves_returned": sf_moves_returned or [],
        "best_cp": best_cp,
        "infos": infos or [],
    }


def _make_debug_prefix(i: int, total: Optional[int], fen: str) -> str:
    if total is None:
        return f"[stockfish-cache] row={i} fen={fen}"
    return f"[stockfish-cache] row={i}/{total} fen={fen}"


def generate_stockfish_cache(
    sf_cfg: SfConfig,
    jsonl_path: Path,
    sf_cache_jsonl_path: Path,
    max_rows: Optional[int] = None,
) -> None:
    sf_cache_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if sf_cache_jsonl_path.exists:
        print(f"Skipping generating sf cache for {sf_cfg} as data has already bin generated")
        return

    sf_engine = make_stockfish(
        sf_cfg.stockfish_path,
        threads=int(sf_cfg.threads),
        hash_mb=int(sf_cfg.hash_mb),
        uci_elo=sf_cfg.uci_elo,
        skill_level=None,
        timeout=float(sf_cfg.timeout_s),
    )

    try:
        ds = DpoPairs(jsonl_path=str(jsonl_path))
        if max_rows:
            random.seed(max_rows)
            random.shuffle(ds.rows)
            ds.rows = ds.rows[:500]
            
        limit = chess.engine.Limit(depth=int(sf_cfg.depth))

        try:
            total_rows = len(ds)
        except Exception:
            total_rows = None

        with open(sf_cache_jsonl_path, "w", encoding="utf-8") as sf_cache_file:
            for i, r in enumerate(ds):
                fen = r["fen"]
                move_fields = _safe_get_move_fields(r)

                debug_prefix = _make_debug_prefix(i, total_rows, fen)

                print(
                    f"{debug_prefix} "
                    f"chosen={move_fields['chosen_move']} "
                    f"rejected={move_fields['rejected_move']} "
                    f"move={move_fields['move']}"
                )

                out_row = {**r}

                try:
                    board = chess.Board(fen)
                except Exception as e:
                    err = f"invalid_fen: {type(e).__name__}: {e}"
                    print(f"{debug_prefix} ERROR {err}")
                    out_row.setdefault("meta", {})
                    out_row["meta"]["stockfish"] = _base_stockfish_meta(
                        sf_cfg,
                        "invalid_fen",
                        error=err,
                    )
                    sf_cache_file.write(json.dumps(out_row) + "\n")
                    sf_cache_file.flush()
                    continue

                legal_moves = board.legal_moves.count()
                print(f"{debug_prefix} legal_moves={legal_moves}")

                if not board.is_valid():
                    err = "board.is_valid() == False"
                    print(f"{debug_prefix} ERROR {err}")
                    out_row.setdefault("meta", {})
                    out_row["meta"]["stockfish"] = _base_stockfish_meta(
                        sf_cfg,
                        "invalid_board",
                        error=err,
                        legal_moves=legal_moves,
                    )
                    sf_cache_file.write(json.dumps(out_row) + "\n")
                    sf_cache_file.flush()
                    continue

                if board.is_game_over(claim_draw=True):
                    reason = "game_over"
                    print(f"{debug_prefix} skipping: {reason}")
                    out_row.setdefault("meta", {})
                    out_row["meta"]["stockfish"] = _base_stockfish_meta(
                        sf_cfg,
                        reason,
                        legal_moves=legal_moves,
                    )
                    sf_cache_file.write(json.dumps(out_row) + "\n")
                    sf_cache_file.flush()
                    continue

                started = time.time()
                try:
                    infos = sf_engine.analyse(
                        board,
                        limit,
                        multipv=int(sf_cfg.multipv_topk),
                    )
                except Exception as e:
                    elapsed_s = time.time() - started
                    err = f"{type(e).__name__}: {e}"
                    print(f"{debug_prefix} ANALYSE_ERROR after {elapsed_s:.2f}s: {err}")

                    out_row.setdefault("meta", {})
                    out_row["meta"]["stockfish"] = _base_stockfish_meta(
                        sf_cfg,
                        "analyse_exception",
                        error=err,
                        legal_moves=legal_moves,
                        elapsed_s=elapsed_s,
                    )
                    sf_cache_file.write(json.dumps(out_row) + "\n")
                    sf_cache_file.flush()

                    # Optional: restart engine after an exception in case the process wedged.
                    try:
                        sf_engine.quit()
                    except Exception:
                        pass
                    sf_engine = make_stockfish(
                        sf_cfg.stockfish_path,
                        threads=int(sf_cfg.threads),
                        hash_mb=int(sf_cfg.hash_mb),
                        uci_elo=sf_cfg.uci_elo,
                        skill_level=None,
                        timeout=float(sf_cfg.timeout_s),
                    )
                    continue

                elapsed_s = time.time() - started

                if isinstance(infos, dict):
                    infos = [infos]

                serialized_infos: List[Dict[str, Any]] = []
                sf_moves_returned: List[List[Any]] = []

                serialize_error: Optional[str] = None

                for info_idx, info in enumerate(infos):
                    try:
                        s = _serialize_sf_info(info, board=board)
                    except Exception as e:
                        serialize_error = (
                            f"_serialize_sf_info failed on info_idx={info_idx}: "
                            f"{type(e).__name__}: {e}"
                        )
                        print(f"{debug_prefix} SERIALIZE_ERROR {serialize_error}")
                        continue

                    if s is None:
                        continue

                    serialized_infos.append(s)

                    if s.get("best_move") is not None:
                        sf_moves_returned.append([s["best_move"], s.get("cp")])

                best_cp = max(
                    (cp for _, cp in sf_moves_returned if cp is not None),
                    default=None,
                )

                if len(serialized_infos) < int(sf_cfg.multipv_topk):
                    print(
                        f"{debug_prefix} returned fewer PVs than requested: "
                        f"requested={int(sf_cfg.multipv_topk)} returned={len(serialized_infos)} "
                        f"legal_moves={legal_moves}"
                    )

                print(
                    f"{debug_prefix} analysed in {elapsed_s:.2f}s "
                    f"returned_infos={len(serialized_infos)}"
                )

                out_row.setdefault("meta", {})
                out_row["meta"]["stockfish"] = _base_stockfish_meta(
                    sf_cfg,
                    "ok" if serialize_error is None else "ok_with_serialize_warnings",
                    error=serialize_error,
                    legal_moves=legal_moves,
                    elapsed_s=elapsed_s,
                    sf_moves_returned=sf_moves_returned,
                    best_cp=best_cp,
                    infos=serialized_infos,
                )

                sf_cache_file.write(json.dumps(out_row) + "\n")
                sf_cache_file.flush()

                if (i + 1) % 100 == 0:
                    if total_rows is None:
                        print(f"[stockfish-cache] processed {i + 1} rows")
                    else:
                        print(f"[stockfish-cache] processed {i + 1}/{total_rows} rows")

                # Optional safety valve: periodically restart engine
                if (i + 1) % 500 == 0:
                    print(f"[stockfish-cache] restarting engine after {i + 1} rows")
                    try:
                        sf_engine.quit()
                    except Exception:
                        pass
                    sf_engine = make_stockfish(
                        sf_cfg.stockfish_path,
                        threads=int(sf_cfg.threads),
                        hash_mb=int(sf_cfg.hash_mb),
                        uci_elo=sf_cfg.uci_elo,
                        skill_level=None,
                        timeout=float(sf_cfg.timeout_s),
                    )

    finally:
        try:
            sf_engine.quit()
        except Exception:
            pass

