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
from typing import Any, Dict, List, Optional, Tuple

import chess
from grandmaster_dpo.eval.chess_utils import batch_preprocess, coarse_opening_family_from_prefix, extract_move_cp, fen_to_ply_abs, ply_to_phase, vocab_index_to_uci
from grandmaster_dpo.eval.configs import OpeningLogitDistConfig, SfConfig
from grandmaster_dpo.eval.eval_abstractions import forward_logits
from grandmaster_dpo.eval.opening_metrics import summarize_opening_distribution, update_opening_distributions_from_logits
from grandmaster_dpo.eval.single_gm.shared_eval_metric_utilities import DpoPairs, apply_legal_mask, collate_batch, hit_at_k
from grandmaster_dpo.eval.tensor_metrics import chosen_probability, chosen_rank, gather_logprob_from_masked_logits, kl_policy_base_from_logits
from grandmaster_dpo.eval.trajectory_utils import safe_get_next_fens_chosen, safe_get_next_fens_rejected, safe_get_prev_fens
from grandmaster_dpo.eval.types import _SfBatchContext, EvalAggMetrics, EvalPerRowInput, EvalRowMetrics, SfHelperEvalAggregate, SfPerPosResult
import torch
from torch.utils.data import DataLoader
import numpy as np

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves

from grandmaster_dpo.eval.stockfish_helpers import make_stockfish, uci_to_vocab_index


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

@torch.inference_mode()
def compute_sf_helper_w_gibbs_for_one_position(
    *,
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    sf_engine: chess.engine.SimpleEngine,
    logits_masked_1d: torch.Tensor,   # [V] masked logits for policy
    base_logp_full_1d: torch.Tensor,  # [V] log-softmax of base masked logits
    all_moves_dict: Dict[str, int],   # uci_eff -> idx
    cfg: SfConfig,
    full_hit: Dict[int, int],         # {1:0/1, 5:0/1, 10:0/1}
    rng: random.Random,
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

    # --------- policy top-k candidates ----------
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
    limit = chess.engine.Limit(depth=int(cfg.depth))
    multipv = k

    try:
        infos = sf_engine.analyse(
            board,
            limit,
            multipv=min(multipv, len(policy_root_moves)),
            root_moves=policy_root_moves,
        )
    except Exception:
        return None

    assert len(infos) <= len(topv.tolist())

    sf_cands: List[Tuple[str, int]] = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        uci = pv[0].uci()
        # Convert to POV for side to move so cp comparisons are consistent
        pov_score = score.pov(board.turn) if hasattr(score, "pov") else score
        cp = _score_to_cp(score, turn=board.turn)
        sf_cands.append((uci, cp))

    if not sf_cands:
        return None

    # --------- cp-window filter ----------
    best_cp = max(cp for _, cp in sf_cands)
    kept: List[Tuple[str, int]] = sf_cands
    w = int(adaptive_restrict_cp_window)
    filtered = [(m, cp) for (m, cp) in kept if cp >= best_cp - w]
    if filtered:
        kept = filtered
    if not kept:
        return None

    # Ensure stable ordering for reporting
    kept = sorted(kept, key=lambda x: -x[1])

    # --------- compute pi_ref over kept moves (from policy logits) ----------
    # This is the ref distribution in the KL term.
    logp_ref_all = torch.log_softmax(logits_masked_1d, dim=-1)  # [V]

    cand_moves: List[str] = [m for (m, _cp) in kept]
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
        [int(cp) for (_uci, cp) in kept],
        device=cand_logp_ref_t.device,
        dtype=cand_logp_ref_t.dtype,
    ).clamp(min=-cp_cap, max=cp_cap)

    Q = cps / cp_scale
    combined = cand_logp_ref_t + (Q / max(lam, 1e-6))          # [K]
    cand_probs_t = torch.softmax(combined, dim=0)              # [K]
    cand_probs: List[float] = cand_probs_t.detach().cpu().tolist()

    # Build a full-V "logits" tensor that corresponds to combined scores on candidates
    cands_logits_full = torch.full_like(logits_masked_1d, neg_inf)  # [V]
    for idx, comb in zip(cand_idxs, combined):
        if idx >= 0:
            cands_logits_full[idx] = comb

    # --------- select within candidates ----------
    if bool(getattr(cfg, "sample", True)):
        sel_i = int(torch.multinomial(cand_probs_t, 1).item())
    else:
        sel_i = int(torch.argmax(cand_probs_t).item())

    selected_uci, cp_selected = kept[sel_i]
    cp_gap = float(best_cp - cp_selected)
    is_best = (cp_gap <= 1e-9)

    probs_np = cand_probs_t.detach().cpu().numpy()
    eps = float(getattr(cfg, "eps", 1e-12))
    ent = float(-(probs_np * np.log(probs_np + eps)).sum()) if probs_np.size else 0.0

    # Log-prob of selected move under *full policy* (for comparability with other helper)
    sel_idx = uci_to_vocab_index(all_moves_dict, fen, selected_uci)
    logp_selected_full = float(logp_ref_all[sel_idx].item()) if sel_idx >= 0 else float("-inf")

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
    for (uci, _cp), q in zip(kept, cand_probs):
        idx = uci_to_vocab_index(all_moves_dict, fen, uci)
        if idx < 0:
            continue
        logq = math.log(max(float(q), eps))
        logp_b = float(base_logp_full_1d[idx].item())
        kl += float(q) * (logq - logp_b)

    res = SfPerPosResult(
        selected_uci=str(selected_uci),
        is_best_sf=bool(is_best),
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

        cand_hit1=float(cand_hit_at(1)),
        cand_hit5=float(cand_hit_at(5)),
        cand_hit10=float(cand_hit_at(10)),

        full_hit1=float(full_hit.get(1, 0)),
        full_hit5=float(full_hit.get(5, 0)),
        full_hit10=float(full_hit.get(10, 0)),

        kl_q_vs_base=float(kl),
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
        "selected_index": int(sel_i),

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
    cands: List[Tuple[str, int]],     # [(uci,cp)]
    best_cp: int,
    logits_masked_1d: torch.Tensor,   # [V] for model under eval
    base_logp_full_1d: torch.Tensor,  # [V] log-softmax of base masked logits
    all_moves_dict: Dict[str, int],
    cfg: SfConfig,
    full_hit: Dict[int, int],         # {1:0/1, 5:0/1, 10:0/1}
    rng: random.Random,
) -> Optional[Tuple[SfPerPosResult, Dict[str, Any]]]:
    if not cands:
        return None

    kept = cands
    if cfg.restrict_cp_window is not None:
        w = int(cfg.restrict_cp_window)
        filt = [(m, cp) for (m, cp) in kept if cp >= best_cp - w]
        if filt:
            kept = filt
    if not kept:
        return None

    t = max(float(cfg.temperature), 1e-6)
    logp_all = torch.log_softmax(logits_masked_1d / t, dim=-1)  # [V]

    cand_moves = [m for (m, _cp) in kept]
    cand_idxs: List[int] = [uci_to_vocab_index(all_moves_dict, fen, u) for u in cand_moves]

    cand_logps = []
    for idx in cand_idxs:
        if idx < 0:
            cand_logps.append(torch.tensor(torch.finfo(logp_all.dtype).min, device=logp_all.device))
        else:
            cand_logps.append(logp_all[idx])
    cand_logps_t = torch.stack(cand_logps, dim=0)          # [K]
    cand_probs_t = torch.softmax(cand_logps_t, dim=0)      # [K]
    cand_probs = cand_probs_t.detach().cpu().tolist()

    neg_inf = torch.finfo(logits_masked_1d.dtype).min
    cands_logits_full = torch.full_like(logits_masked_1d, neg_inf)
    cands_logits_full[cand_idxs] = logits_masked_1d[cand_idxs]  # logits, not probs


    # select within candidate set
    if cfg.sample:
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
    ent = _entropy(cand_probs, cfg.eps)

    sel_idx = uci_to_vocab_index(all_moves_dict, fen, selected_uci)
    logp_selected_full = float(logp_all[sel_idx].item()) if sel_idx >= 0 else float("-inf")

    p_ch = float(cand_probs[cand_moves.index(chosen_uci)]) if chosen_uci in cand_moves else 0.0
    p_rj = float(cand_probs[cand_moves.index(rejected_uci)]) if rejected_uci in cand_moves else 0.0
    logp_ch = math.log(max(p_ch, cfg.eps))
    logp_rj = math.log(max(p_rj, cfg.eps))
    gap_logp = float(logp_ch - logp_rj)

    # candidate hits on chosen (based on q ordering)
    K = len(cand_probs)
    order = sorted(range(K), key=lambda j: cand_probs[j], reverse=True)
    def cand_hit_at(k: int) -> float:
        if k <= 0:
            return 0.0
        k = min(k, K)
        top_moves = [cand_moves[j] for j in order[:k]]
        return 1.0 if chosen_uci in top_moves else 0.0

    # KL(q || base_full) over candidate support
    kl = 0.0
    for (uci, _cp), q in zip(kept, cand_probs):
        idx = uci_to_vocab_index(all_moves_dict, fen, uci)
        if idx < 0:
            continue
        logq = math.log(max(float(q), cfg.eps))
        logp_b = float(base_logp_full_1d[idx].item())
        kl += float(q) * (logq - logp_b)

    res = SfPerPosResult(
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

        cand_hit1=float(cand_hit_at(1)),
        cand_hit5=float(cand_hit_at(5)),
        cand_hit10=float(cand_hit_at(10)),

        full_hit1=float(full_hit[1]),
        full_hit5=float(full_hit[5]),
        full_hit10=float(full_hit[10]),

        kl_q_vs_base=float(kl),
    )

    dbg = {
        "cands_kept": kept,
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
        self._sf_engine = None
        if self.sf_cfg is not None:
            self._sf_engine = make_stockfish(
                self.sf_cfg.stockfish_path,
                threads=int(self.sf_cfg.threads),
                hash_mb=int(self.sf_cfg.hash_mb),
                uci_elo=self.sf_cfg.uci_elo,
                skill_level=None,
                timeout=float(self.sf_cfg.timeout_s),
            )

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

    def _compute_eval_per_row_input(self, 
                                    batch: Dict[str, Any], 
                                    opening_counts_adv: Dict[str, Counter], 
                                    dataset: DpoPairs, 
                                    opening_cfg: OpeningLogitDistConfig,
                                    opening_by_game: Dict[str, str]):
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

        print("=== DEBUG: first batch ===")
        print("batch size:", len(fens))
        print("ply_idx sample:", ply_idxs[:10])

        # Always print first few rows no matter what (so you know logging works)
        for i in range(min(5, len(fens))):
            lm_sum = int(legal_moves[i].sum().item())
            print(f"[row {i}] ply_idx={ply_idxs[i]} ply_abs={fen_to_ply_abs(fens[i])} lm_sum={lm_sum}")
            legal_idxs = torch.nonzero(legal_moves[i] > 0).squeeze(1)[:30].tolist()
            print("  first legal idxs:", legal_idxs)
    
        self.debug_opening_distribution(self.policy, self.device)

        logits_pi = forward_logits(self.policy, board_input, es_t, eo_t)
        logits_ref = forward_logits(self.base, board_input, es_t, eo_t)

        logits_pi_m = apply_legal_mask(logits_pi, legal_moves)

        # After logits_pi_m is computed:
        # find first ply_abs==0 row in this batch
        ply0 = [i for i, fen in enumerate(fens) if fen_to_ply_abs(fen) == 0]
        if ply0:
            i = ply0[0]
            row = logits_pi_m[i]                 # already masked
            probs = torch.softmax(row, dim=-1)   # over vocab indices

            # show top legal moves by prob, decoded to uci
            vals, idxs = torch.topk(probs, k=20)
            for k in range(20):
                print("\n=== PLY0 ROW (from batch) distribution ===")
            print("fen:", fens[i])
            print("elo_self raw:", es[i], "elo_oppo raw:", eo[i])
            print("elo_self_cat:", int(es_t[i].item()), "elo_oppo_cat:", int(eo_t[i].item()))
            ent = float(-(probs * torch.log(probs + 1e-12)).sum().item())
            print("entropy:", ent)

            for r, (p, j) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
                uci = vocab_index_to_uci(self.all_moves, fens[i], int(j))
                print(f"[{r:02d}] idx={int(j):4d} uci={uci:6s} p={p:.6f}")


        logits_ref_m = apply_legal_mask(logits_ref, legal_moves)

        chosen_idx = torch.tensor(
            [uci_to_vocab_index(self.all_moves_dict, fen, u) for fen, u in zip(fens, chosen)],
            device=self.device, dtype=torch.long
        )
        rejected_idx = torch.tensor(
            [uci_to_vocab_index(self.all_moves_dict, fen, u) for fen, u in zip(fens, rejected)],
            device=self.device, dtype=torch.long
        )

        # legality checks (match your prior behavior)
        chosen_ok = (chosen_idx >= 0) & (legal_moves.gather(1, chosen_idx.clamp(min=0).view(-1,1)).squeeze(1) > 0)
        rejected_ok = (rejected_idx >= 0) & (legal_moves.gather(1, rejected_idx.clamp(min=0).view(-1,1)).squeeze(1) > 0)
        bad = ~(chosen_ok & rejected_ok)
        if bad.any():
            j = int(bad.nonzero()[0])
            raise RuntimeError(f"Illegal chosen/rejected. fen={fens[j]} chosen={chosen[j]} rejected={rejected[j]}")

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

        loss = self.compute_training_style_loss(
            logp_pi_ch=logp_pi_ch,
            logp_pi_rj=logp_pi_rj,
            logp_ref_ch=logp_ref_ch,
            logp_ref_rj=logp_ref_rj,
            logits_pi_m=logits_pi_m,
            logits_ref_m=logits_ref_m,
            idx_t=chosen_idx,
            batch_meta_data=zip(fens,
                                chosen,
                                rejected,
                                chosen_cps,
                                rejected_cps,
                                ply_idxs,
                                prev_fens_batch,
                                next_fens_chosen_batch,
                                next_fens_rejected_batch,
                                meta_list),
        )

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
                "opening_prefixes": opening_prefixes
            })

    def _add_per_row_metrics(self, eval_per_row_input: EvalPerRowInput, sums: Dict[str, float], rows: List[EvalRowMetrics], phase_buckets: Dict[Tuple[str, str], List]):
        pi_gap = (eval_per_row_input.logp_pi_ch - eval_per_row_input.logp_pi_rj)
        ref_gap = (eval_per_row_input.logp_ref_ch - eval_per_row_input.logp_ref_rj)
        gap_improve = (pi_gap - ref_gap)

        # core metrics
        top1 = (eval_per_row_input.logits_pi_m.argmax(dim=-1) == eval_per_row_input.chosen_idx).float()
        hit5 = hit_at_k(eval_per_row_input.logits_pi_m, eval_per_row_input.chosen_idx, 5)
        hit10 = hit_at_k(eval_per_row_input.logits_pi_m, eval_per_row_input.chosen_idx, 10)
        rank = chosen_rank(eval_per_row_input.logits_pi_m, eval_per_row_input.chosen_idx)
        mrr = torch.where(rank < 1e8, 1.0 / rank.float(), torch.zeros_like(rank.float()))

        p_chosen = chosen_probability(eval_per_row_input.logits_pi_m, eval_per_row_input.fens, self.all_moves_dict, eval_per_row_input.chosen)
        kl = kl_policy_base_from_logits(eval_per_row_input.logits_pi_m, eval_per_row_input.logits_ref_m)

        # accumulate
        sums["loss"] += float(eval_per_row_input.loss) * eval_per_row_input.batch_size
        sums["pi_gap"] += float(pi_gap.mean()) * eval_per_row_input.batch_size
        sums["ref_gap"] += float(ref_gap.mean()) * eval_per_row_input.batch_size
        sums["gap_improve"] += float(gap_improve.mean()) * eval_per_row_input.batch_size
        sums["top1"] += float(top1.mean()) * eval_per_row_input.batch_size
        sums["hit5"] += float(hit5.mean()) * eval_per_row_input.batch_size
        sums["hit10"] += float(hit10.mean()) * eval_per_row_input.batch_size
        sums["p_chosen"] += float(p_chosen.mean()) * eval_per_row_input.batch_size
        sums["kl"] += float(kl.mean()) * eval_per_row_input.batch_size
        sums["mrr"] += float(mrr.mean()) * eval_per_row_input.batch_size

        update_opening_distributions_from_logits(
            opening_counts=eval_per_row_input.opening_counts_adv,
            fens=eval_per_row_input.fens,
            logits_masked=eval_per_row_input.logits_pi_m,
            all_moves=self.all_moves,
            cfg=eval_per_row_input.opening_cfg,
        )

        pred_idx = eval_per_row_input.logits_pi_m.argmax(dim=-1).tolist()
        pred_uci = [vocab_index_to_uci(self.all_moves, fen, i) for fen, i in zip(eval_per_row_input.fens, pred_idx)]

        for i in range(eval_per_row_input.batch_size):
            fen = eval_per_row_input.fens[i]
            ply_abs = fen_to_ply_abs(fen)
            phase = ply_to_phase(ply_abs)

            gid = str(eval_per_row_input.game_ids[i] or "")
            if gid and gid not in eval_per_row_input.opening_by_game:
                pref = eval_per_row_input.opening_prefixes[i] or []
                eval_per_row_input.opening_by_game[gid] = coarse_opening_family_from_prefix(pref)

            r = EvalRowMetrics.from_dict({
                "game_id": gid,
                "ply_idx": int(eval_per_row_input.ply_idxs[i]) if eval_per_row_input.ply_idxs[i] is not None else -1,
                "ply_abs": int(ply_abs),
                "phase": phase,
                "fen": fen,
                "chosen_uci": eval_per_row_input.chosen[i],
                "rejected_uci": eval_per_row_input.rejected[i],
                "pred_uci": pred_uci[i],
                "correct_top1": float(top1[i].item()),
                "hit_top5": float(hit5[i].item()),
                "hit_top10": float(hit10[i].item()),
                "rank_chosen": int(rank[i].item()) if rank[i].item() < 1e8 else -1,
                "mrr": float(mrr[i].item()),
                "logp_gap_pi": float(pi_gap[i].item()),
                "logp_gap_ref": float(ref_gap[i].item()),
                "gap_improve": float(gap_improve[i].item()),
                "p_chosen_pi": float(p_chosen[i].item()),
                "kl_pi_ref": float(kl[i].item()),
                "nll_chosen_pi": float((-eval_per_row_input.logp_pi_ch[i]).item()),
            })
            rows.append(r)

            phase_buckets[("kl_pi_ref", phase)].append(r.kl_pi_ref)
            phase_buckets[("logp_gap_pi", phase)].append(r.logp_gap_pi)
            phase_buckets[("p_chosen_pi", phase)].append(r.p_chosen_pi)
            phase_buckets[("correct_top1", phase)].append(r.correct_top1)

    def generate_aggregate_eval_metrics(self,
                                        num_rows: int,
                                        opening_by_game: Dict[str, str],
                                        sums: Dict[str, float],
                                        gm_name: str,
                                        dataset: DpoPairs,
                                        batch_size: int,
                                        opening_counts_adv: Dict[str, Counter]) -> EvalAggMetrics:
        def avg(x: float) -> float:
            return x / max(1, num_rows)

        opening_counts = Counter(opening_by_game.values())
        opening_dist = {k: v for k, v in opening_counts.most_common()}
        opening_summary = summarize_opening_distribution(opening_counts_adv, topn=50, normalize=True)
        stockfish_data = None

        if self.sf_cfg is not None:
            stockfish_data = self._run_sf_helper_eval(ds=dataset, batch_size=batch_size)

        agg = EvalAggMetrics.from_dict({
            "gm": gm_name,
            "tag": self.tag,
            "maia_type": self.maia_type,
            "device": str(self.device),
            "num_rows": num_rows,
            "mean_loss": avg(sums["loss"]),
            "mean_logp_gap_pi": avg(sums["pi_gap"]),
            "mean_logp_gap_ref": avg(sums["ref_gap"]),
            "mean_gap_improve": avg(sums["gap_improve"]),
            "top1_accuracy": avg(sums["top1"]),
            "hit5": avg(sums["hit5"]),
            "hit10": avg(sums["hit10"]),
            "mrr": avg(sums["mrr"]),
            "mean_p_chosen": avg(sums["p_chosen"]),
            "mean_kl_pi_ref": avg(sums["kl"]),
            "opening_family_counts_by_game": opening_dist,
            "opening_summary": opening_summary,
            "stockfish": stockfish_data,
        })
        
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
    ) -> EvalAggMetrics:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

        # aggregate scalars
        num_rows = 0
        sums = defaultdict(float)

        per_rows: List[EvalRowMetrics] = []
        phase_buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        opening_by_game: Dict[str, str] = {}

        opening_counts_adv = {"ply0_white": Counter(), "ply1_black": Counter()}
        opening_cfg = OpeningLogitDistConfig(plies=(0,1), temperature=1.0, topk=50)

        for batch in loader:
            eval_per_row_input = self._compute_eval_per_row_input(batch, opening_counts_adv, ds, opening_cfg, opening_by_game)
            self._add_per_row_metrics(eval_per_row_input, sums, per_rows, phase_buckets)
            num_rows += eval_per_row_input.batch_size

        eval_agg_metrics = self.generate_aggregate_eval_metrics(num_rows, opening_by_game, sums, gm_name, ds, batch_size, opening_counts_adv)

        # Write outputs
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"eval_results__{self.tag}.json").write_text(json.dumps(eval_agg_metrics, indent=2))

            # per-row
            if per_rows:
                per_csv = out_dir / f"eval_per_row__{self.tag}.csv"
                with open(per_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(asdict(per_rows[0]).keys()))
                    w.writeheader()
                    w.writerows((asdict(r) for r in per_rows))

        return eval_agg_metrics

    def _run_sf_helper_eval(self, *, ds: DpoPairs, batch_size: int = 64) -> Dict[str, Any]:
        """
        SF-helper evaluation:
        - SF gives top-k PV candidates at depth=cfg.depth (multipv=cfg.multipv_topk)
        - Policy logits restricted to candidate set to produce q
        - Metrics computed on q and on selection within candidates
        - KL(q || base_full) computed on candidate support using base masked log-softmax
        """
        assert self.sf_cfg is not None
        assert self._sf_engine is not None

        sf_cfg = self.sf_cfg
        rng = random.Random(sf_cfg.seed)
        analysis_limit = chess.engine.Limit(depth=int(sf_cfg.depth))

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
                    analysis_limit=analysis_limit,
                    rng=rng,
                )
                if row_output is None:
                    continue

                result, debug_info = row_output
                fen = batch_ctx.fens[row_index]
                chosen_uci = batch_ctx.chosen_uci_list[row_index]
                ply_abs = fen_to_ply_abs(fen)

                aggregate.add_processed_row(
                    selected_matches_chosen=(result.selected_uci == chosen_uci),
                    cand_hit5=result.cand_hit5,
                    cand_hit10=result.cand_hit10,
                    cp_gap=result.cp_gap,
                    is_best_sf=result.is_best_sf,
                    entropy=result.entropy,
                    logp_selected_full=result.logp_selected_full,
                    p_chosen_cond=result.p_chosen_cond,
                    p_rejected_cond=result.p_rejected_cond,
                    logp_chosen_cond=result.logp_chosen_cond,
                    logp_rejected_cond=result.logp_rejected_cond,
                    gap_logp_cond=result.gap_logp_cond,
                    kl_q_vs_base=result.kl_q_vs_base,
                    full_hit1=result.full_hit1,
                    full_hit5=result.full_hit5,
                    full_hit10=result.full_hit10,
                    selected_uci=result.selected_uci,
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
        return aggregate.to_dict(sf_config=sf_cfg, sf_opening_summary=opening_summary)


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
        )


    def _evaluate_single_sf_row(
        self,
        *,
        batch_ctx: _SfBatchContext,
        row_index: int,
        sf_cfg: Any,
        analysis_limit: chess.engine.Limit,
        rng: random.Random,
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
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
                sf_engine=self._sf_engine,
                logits_masked_1d=batch_ctx.masked_policy_logits[row_index],
                base_logp_full_1d=batch_ctx.base_full_log_probs[row_index],
                all_moves_dict=self.all_moves_dict,
                cfg=sf_cfg,
                full_hit=row_full_hit,
                rng=rng,
            )

        sf_candidates = self._get_sf_candidates_for_board(
            board=board,
            analysis_limit=analysis_limit,
            multipv_topk=int(sf_cfg.multipv_topk),
        )
        if not sf_candidates:
            return None

        best_cp = max(cp for _, cp in sf_candidates)
        return compute_sf_helper_for_one_position(
            fen=fen,
            chosen_uci=chosen_uci,
            rejected_uci=rejected_uci,
            cands=sf_candidates,
            best_cp=int(best_cp),
            logits_masked_1d=batch_ctx.masked_policy_logits[row_index],
            base_logp_full_1d=batch_ctx.base_full_log_probs[row_index],
            all_moves_dict=self.all_moves_dict,
            cfg=sf_cfg,
            full_hit=row_full_hit,
            rng=rng,
        )


    def _get_sf_candidates_for_board(
        self,
        *,
        board: chess.Board,
        analysis_limit: chess.engine.Limit,
        multipv_topk: int,
    ) -> List[Tuple[str, int]]:
        assert self._sf_engine is not None

        infos = self._sf_engine.analyse(board, analysis_limit, multipv=multipv_topk)
        candidates: List[Tuple[str, int]] = []

        for info in infos:
            pv = info.get("pv")
            score = info.get("score")
            if not pv or score is None:
                continue

            uci = pv[0].uci()
            cp = _score_to_cp(score, turn=board.turn)
            candidates.append((uci, cp))

        return candidates


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
