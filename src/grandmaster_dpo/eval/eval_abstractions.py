# src/grandmaster_dpo/eval/eval_abstractions.py
from __future__ import annotations

import csv
import json
import math
import os
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chess
from grandmaster_dpo.eval.single_gm.eval_sft_and_dpo_w_style_sim_utility_weight_maia2 import compute_style_score, dpo_loss_style_weighted, supervised_nll_loss
from grandmaster_dpo.eval.single_gm.eval_sft_and_dpo_w_style_v2_maia_single_gm import compute_style_score_v2
from grandmaster_dpo.eval.single_gm.eval_sft_and_dpo_w_style_v3_maia_single_gm import compute_style_score_v3, dpo_loss_style_weighted_v3
from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import make_config
from grandmaster_dpo.utilities.shared_style_emb_model_utils import StyleEncoder
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move

# If you already have this helper, keep using it.
# It should return chess.engine.SimpleEngine configured with threads/hash/uci_elo, etc.
from grandmaster_dpo.eval.stockfish_helpers import make_stockfish


# ============================================================
# Dataset (keep your existing one; shown here for completeness)
# ============================================================

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
            print(f"Loaded {len(self.rows)} rows from {jsonl_path}")

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
            "fullmove_number": int(meta.get("fullmove_number", -1)),
            "side_to_move": str(meta.get("side_to_move", "")),
            "opening_prefix_uci_20": meta.get("opening_prefix_uci_20") or [],
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
        "fullmove_number": [],
        "side_to_move": [],
        "opening_prefix_uci_20": [],
    }
    for b in batch:
        for k in out:
            out[k].append(b.get(k))
    return out


@dataclass
class OpeningLogitDistConfig:
    # Which early plies to track (absolute ply count from FEN):
    # 0 = white's first move, 1 = black's first reply, etc.
    plies: Tuple[int, ...] = (0, 1)

    # Use probabilities derived from logits/temperature
    temperature: float = 1.0

    # For efficiency + stability, only allocate probability mass over top-K moves
    # (renormalized within top-K). Set to 0 to use full distribution (expensive).
    topk: int = 50

    # If you set min_prob_mass < 1, you can grow K until cumulative prob reaches this mass.
    # (Requires sorting; only meaningful when topk==0 or you choose to implement dynamic K.)
    min_prob_mass: Optional[float] = None


def update_opening_distributions_from_logits(
    *,
    opening_counts: Dict[str, Counter],
    fens: List[str],
    logits_masked: torch.Tensor,  # [B, V] or [V]
    all_moves: List[str],
    cfg: OpeningLogitDistConfig = OpeningLogitDistConfig(),
) -> None:
    """
    Accumulate *soft* opening distributions using only model logits.

    opening_counts:
      - dict[str, Counter] you maintain across the eval loop, e.g.:
          opening_counts = {"ply0_white": Counter(), "ply1_black": Counter()}
      - We add *float mass* to Counter values (Counter supports float increments).

    Strategy:
      - For each position whose ply_abs is in cfg.plies:
          probs = softmax(logits / T)
          take top-K (cfg.topk) for efficiency
          renormalize within top-K
          add prob mass to the UCI move key

    Notes:
      - logits_masked should already have illegal moves at -inf (your apply_legal_mask),
        so softmax puts ~0 mass on illegal moves automatically.
      - Using soft counts avoids brittle argmax and captures uncertainty.
    """
    if logits_masked.dim() == 1:
        logits_masked = logits_masked.unsqueeze(0)

    assert logits_masked.dim() == 2, f"expected [B,V], got {tuple(logits_masked.shape)}"
    B, V = logits_masked.shape
    assert len(fens) == B, f"len(fens)={len(fens)} must match B={B}"
    assert len(all_moves) == V, f"len(all_moves)={len(all_moves)} must match V={V}"

    T = max(float(cfg.temperature), 1e-6)

    # Build a quick mapping: ply_abs -> bucket name
    # (You can customize naming; these two are usually what you want.)
    ply_to_bucket: Dict[int, str] = {}
    for p in cfg.plies:
        if p == 0:
            ply_to_bucket[p] = "ply0_white"
        elif p == 1:
            ply_to_bucket[p] = "ply1_black"
        else:
            ply_to_bucket[p] = f"ply{p}"

    # Ensure all buckets exist
    for b in ply_to_bucket.values():
        opening_counts.setdefault(b, Counter())

    with torch.no_grad():
        # We operate per-row because each row needs fen-dependent mirroring for idx->uci.
        for i in range(B):
            fen = fens[i]
            ply_abs = fen_to_ply_abs(fen)
            if ply_abs not in ply_to_bucket:
                continue

            bucket = ply_to_bucket[ply_abs]
            row_logits = logits_masked[i] / T

            if cfg.topk and cfg.topk > 0 and cfg.topk < V:
                vals, idxs = torch.topk(row_logits, k=int(cfg.topk), dim=-1)
                probs = torch.softmax(vals, dim=-1)  # renormalized within top-K
                idxs_list = idxs.tolist()
                probs_list = probs.detach().cpu().tolist()
                for j, p in zip(idxs_list, probs_list):
                    uci = vocab_index_to_uci(all_moves, fen, int(j))
                    if uci:
                        opening_counts[bucket][uci] += float(p)
            else:
                # Full distribution (can be slower)
                probs_full = torch.softmax(row_logits, dim=-1).detach().cpu()
                # If you want to be safe about numeric noise, you can skip tiny probs
                # but iterating V can be expensive.
                for j in range(V):
                    p = float(probs_full[j].item())
                    if p <= 0.0:
                        continue
                    uci = vocab_index_to_uci(all_moves, fen, j)
                    if uci:
                        opening_counts[bucket][uci] += p


def summarize_opening_distribution(
    opening_counts: Dict[str, Counter],
    *,
    topn: int = 30,
    normalize: bool = True,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Convert Counters into a JSON-friendly summary.
    If normalize=True, convert masses to probabilities per bucket.
    """
    out: Dict[str, List[Dict[str, float]]] = {}
    for bucket, ctr in opening_counts.items():
        if not ctr:
            out[bucket] = []
            continue
        items = ctr.most_common(topn)
        if normalize:
            total = float(sum(ctr.values()))
            out[bucket] = [{"uci": u, "p": float(c) / max(1e-12, total)} for (u, c) in items]
        else:
            out[bucket] = [{"uci": u, "mass": float(c)} for (u, c) in items]
    return out


# ============================================================
# Shared chess helpers
# ============================================================

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
    uci_eff = all_moves[idx]          # Maia vocab is white-perspective
    side = fen.split(" ")[1]
    return mirror_move(uci_eff) if side == "b" else uci_eff

def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))

def extract_move_cp(meta: dict, uci: str) -> float:
    sf_moves = meta["stockfish"]["sf_moves_returned"]
    for sf_uci, cp in sf_moves:
        if sf_uci == uci:
            return float(cp)
        
    cp_values = [cp for _, cp in sf_moves]
    fallback_cp = float(min(cp_values)) if cp_values else 0.0
    #print(
        #f"[WARN] move {uci} not found in sf_moves_returned "
        #f"(game={meta.get('game_header_hash')}, ply={meta.get('ply_idx')}). "
        #f"Using fallback cp={fallback_cp}"
    #)
    return fallback_cp

def batch_preprocess(
    *,
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
        es, eo = 2000, 2000
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
    logits, _, _ = m(board_input, es, eo)
    return logits

def gather_logprob_from_masked_logits(logits_masked: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    logp_all = torch.log_softmax(logits_masked, dim=-1)
    safe = idxs.clamp(min=0)
    out = logp_all.gather(1, safe.view(-1, 1)).squeeze(1)
    out = torch.where(idxs >= 0, out, torch.full_like(out, -1e9))
    return out

@torch.no_grad()
def kl_policy_base_from_logits(logits_pi_masked: torch.Tensor, logits_ref_masked: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits_pi_masked, dim=-1)
    logp = torch.log_softmax(logits_pi_masked, dim=-1)
    logq = torch.log_softmax(logits_ref_masked, dim=-1)
    return (p * (logp - logq)).sum(dim=-1)

@torch.no_grad()
def chosen_probability(logits_masked: torch.Tensor, fens: List[str], all_moves_dict: Dict[str, int], chosen_uci: List[str]) -> torch.Tensor:
    probs = torch.softmax(logits_masked, dim=-1)
    chosen_idx = torch.tensor(
        [uci_to_vocab_index(all_moves_dict, fen, uci) for fen, uci in zip(fens, chosen_uci)],
        device=logits_masked.device,
        dtype=torch.long,
    )
    safe_idx = chosen_idx.clamp(min=0)
    p = probs.gather(1, safe_idx.view(-1, 1)).squeeze(1)
    return torch.where(chosen_idx >= 0, p, torch.zeros_like(p))

@torch.no_grad()
def hit_at_k(logits_masked: torch.Tensor, chosen_idx: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros((logits_masked.size(0),), device=logits_masked.device)
    topk = torch.topk(logits_masked, k=min(k, logits_masked.size(-1)), dim=-1).indices
    chosen_safe = chosen_idx.clamp(min=0).view(-1, 1)
    hit = (topk == chosen_safe).any(dim=1).float()
    return torch.where(chosen_idx >= 0, hit, torch.zeros_like(hit))

@torch.no_grad()
def chosen_rank(logits_masked: torch.Tensor, chosen_idx: torch.Tensor) -> torch.Tensor:
    chosen_safe = chosen_idx.clamp(min=0)
    chosen_logit = logits_masked.gather(1, chosen_safe.view(-1, 1)).squeeze(1)
    greater = (logits_masked > chosen_logit.unsqueeze(1)).sum(dim=1)
    rank = greater + 1
    return torch.where(chosen_idx >= 0, rank, torch.full_like(rank, 10**9))

def fen_to_ply_abs(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return 2 * (fullmove - 1) + (1 if side == "b" else 0)

def ply_to_phase(ply_abs: int) -> str:
    if ply_abs < 20:
        return "opening"
    if ply_abs < 60:
        return "middlegame"
    return "endgame"

def coarse_opening_family_from_prefix(prefix_uci: List[str]) -> str:
    if len(prefix_uci) < 2:
        return "Unknown"
    black_reply = prefix_uci[1]
    if black_reply == "c7c5":
        return "Sicilian"
    if black_reply == "e7e5":
        return "Open Game (1...e5)"
    if black_reply == "c7c6":
        return "Caro-Kann"
    if black_reply == "e7e6":
        return "French / e6"
    if black_reply == "d7d5":
        return "Queen's Pawn (1...d5)"
    if black_reply == "g8f6":
        return "Indian Defense (1...Nf6)"
    return "Other"


# ============================================================
# Stockfish helper bits (single-process engine; keep it simple)
# ============================================================

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

@dataclass
class SfConfig:
    stockfish_path: str
    depth: int = 10
    multipv_topk: int = 10
    uci_elo: Optional[int] = None           # None => full strength
    threads: int = 1
    hash_mb: int = 128
    timeout_s: float = 30.0

    restrict_cp_window: Optional[int] = 60  # keep moves with cp >= best_cp - window
    temperature: float = 1.0
    sample: bool = False
    seed: int = 0
    eps: float = 1e-12

    use_gibbs: bool = False
    k: int = 40

@dataclass
class SfPerPosResult:
    selected_uci: str
    is_best_sf: bool
    cp_selected: int
    cp_best: int
    cp_gap: float
    entropy: float
    logp_selected_full: float

    # q over candidates (policy restricted to SF top-k set after cp-window filter)
    p_chosen_cond: float
    p_rejected_cond: float
    logp_chosen_cond: float
    logp_rejected_cond: float
    gap_logp_cond: float

    cand_hit1: float
    cand_hit5: float
    cand_hit10: float

    # full-policy hits on chosen
    full_hit1: float
    full_hit5: float
    full_hit10: float

    # divergence vs base full dist on candidate support
    kl_q_vs_base: float

def _entropy(probs: List[float], eps: float) -> float:
    s = 0.0
    for p in probs:
        pp = max(float(p), eps)
        s -= pp * math.log(pp)
    return float(s)

@torch.no_grad()
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
        import chess
        import torch

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


    @torch.no_grad()
    def run_eval(
        self,
        *,
        ds: DpoPairs,
        batch_size: int = 64,
        n_boot: int = 0,  # you can wire your bootstrap back in later
        out_dir: Optional[Path] = None,
        gm_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

        # aggregate scalars
        n = 0
        sums = defaultdict(float)

        per_rows: List[Dict[str, Any]] = []
        phase_buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        opening_by_game: Dict[str, str] = {}

        opening_counts_adv = {"ply0_white": Counter(), "ply1_black": Counter()}
        opening_cfg = OpeningLogitDistConfig(plies=(0,1), temperature=1.0, topk=50)

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
            with torch.no_grad():
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
                safe_get_prev_fens(ds.game_id_and_ply_to_prev_10_plys, m, n=5)
                for m in meta_list
            ]

            next_fens_chosen_batch = [
                safe_get_next_fens_chosen(ds.game_id_and_ply_to_fut_10_plys, m, n=5)
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

            pi_gap = (logp_pi_ch - logp_pi_rj)
            ref_gap = (logp_ref_ch - logp_ref_rj)
            gap_improve = (pi_gap - ref_gap)

            # core metrics
            top1 = (logits_pi_m.argmax(dim=-1) == chosen_idx).float()
            hit5 = hit_at_k(logits_pi_m, chosen_idx, 5)
            hit10 = hit_at_k(logits_pi_m, chosen_idx, 10)
            rank = chosen_rank(logits_pi_m, chosen_idx)
            mrr = torch.where(rank < 1e8, 1.0 / rank.float(), torch.zeros_like(rank.float()))

            p_chosen = chosen_probability(logits_pi_m, fens, self.all_moves_dict, chosen)
            kl = kl_policy_base_from_logits(logits_pi_m, logits_ref_m)

            # accumulate
            n += bs
            sums["loss"] += float(loss) * bs
            sums["pi_gap"] += float(pi_gap.mean()) * bs
            sums["ref_gap"] += float(ref_gap.mean()) * bs
            sums["gap_improve"] += float(gap_improve.mean()) * bs
            sums["top1"] += float(top1.mean()) * bs
            sums["hit5"] += float(hit5.mean()) * bs
            sums["hit10"] += float(hit10.mean()) * bs
            sums["p_chosen"] += float(p_chosen.mean()) * bs
            sums["kl"] += float(kl.mean()) * bs
            sums["mrr"] += float(mrr.mean()) * bs

            update_opening_distributions_from_logits(
                opening_counts=opening_counts_adv,
                fens=fens,
                logits_masked=logits_pi_m,
                all_moves=self.all_moves,
                cfg=opening_cfg,
            )

            pred_idx = logits_pi_m.argmax(dim=-1).tolist()
            pred_uci = [vocab_index_to_uci(self.all_moves, fen, i) for fen, i in zip(fens, pred_idx)]

            for i in range(bs):
                fen = fens[i]
                ply_abs = fen_to_ply_abs(fen)
                phase = ply_to_phase(ply_abs)

                gid = str(game_ids[i] or "")
                if gid and gid not in opening_by_game:
                    pref = opening_prefixes[i] or []
                    opening_by_game[gid] = coarse_opening_family_from_prefix(pref)

                r = {
                    "game_id": gid,
                    "ply_idx": int(ply_idxs[i]) if ply_idxs[i] is not None else -1,
                    "ply_abs": int(ply_abs),
                    "phase": phase,
                    "fen": fen,
                    "chosen_uci": chosen[i],
                    "rejected_uci": rejected[i],
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
                    "nll_chosen_pi": float((-logp_pi_ch[i]).item()),
                }
                per_rows.append(r)

                phase_buckets[("kl_pi_ref", phase)].append(r["kl_pi_ref"])
                phase_buckets[("logp_gap_pi", phase)].append(r["logp_gap_pi"])
                phase_buckets[("p_chosen_pi", phase)].append(r["p_chosen_pi"])
                phase_buckets[("correct_top1", phase)].append(r["correct_top1"])

        def avg(x: float) -> float:
            return x / max(1, n)

        opening_counts = Counter(opening_by_game.values())
        opening_dist = {k: v for k, v in opening_counts.most_common()}

        agg = {
            "gm": gm_name,
            "tag": self.tag,
            "maia_type": self.maia_type,
            "device": str(self.device),
            "n": n,
            f"{self.tag}_loss": avg(sums["loss"]),
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
        }

        opening_summary = summarize_opening_distribution(opening_counts_adv, topn=50, normalize=True)
        agg["opening_summary"] = opening_summary


        # Optional SF-helper add-on
        if self.sf_cfg is not None:
            sf = self.run_sf_helper_eval(ds=ds, batch_size=batch_size)
            agg["sf_helper"] = sf

        # Write outputs
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"eval_results__{self.tag}.json").write_text(json.dumps(agg, indent=2))

            # per-row
            if per_rows:
                per_csv = out_dir / f"eval_per_row__{self.tag}.csv"
                with open(per_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
                    w.writeheader()
                    w.writerows(per_rows)

        return agg

    @torch.no_grad()
    def run_sf_helper_eval(self, *, ds: DpoPairs, batch_size: int = 64) -> Dict[str, Any]:
        """
        SF-helper evaluation:
          - SF gives top-k PV candidates at depth=cfg.depth (multipv=cfg.multipv_topk)
          - Policy logits restricted to candidate set to produce q
          - Metrics computed on q and on selection within candidates
          - KL(q || base_full) computed on candidate support using base masked log-softmax
        """
        assert self.sf_cfg is not None
        assert self._sf_engine is not None

        cfg = self.sf_cfg
        rng = random.Random(cfg.seed)

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

        # aggregates
        agg = {
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

            # opening distributions for “human-likeness fingerprints”
            "opening_selected_white_ply0": Counter(),
            "opening_selected_black_ply1": Counter(),
        }

        # SF config -> analysis limit
        limit = chess.engine.Limit(depth=int(cfg.depth))

        def finalize_div(num_key: str, denom: float) -> None:
            agg[num_key] = (agg[num_key] / denom) if denom > 0 else float("nan")

        opening_counts_sf_adv = {"ply0_white": Counter(), "ply1_black": Counter()}
        opening_cfg_sf = OpeningLogitDistConfig(plies=(0,1), temperature=1.0, topk=50)

        for batch in loader:
            fens = batch["fen"]
            es = batch["elo_self"]
            eo = batch["elo_oppo"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            bs = len(fens)

            board_input, legal_moves, es_t, eo_t = batch_preprocess(
                all_moves_dict=self.all_moves_dict,
                elo_dict=self.elo_dict,
                fens=fens,
                elo_self=es,
                elo_oppo=eo,
                device=self.device,
            )

            logits_pi = forward_logits(self.policy, board_input, es_t, eo_t)
            logits_base = forward_logits(self.base, board_input, es_t, eo_t)

            logits_pi_m = apply_legal_mask(logits_pi, legal_moves)
            logits_base_m = apply_legal_mask(logits_base, legal_moves)
            base_logp_full = torch.log_softmax(logits_base_m, dim=-1)  # [B,V]

            chosen_idx = torch.tensor(
                [uci_to_vocab_index(self.all_moves_dict, fen, u) for fen, u in zip(fens, chosen)],
                device=self.device,
                dtype=torch.long,
            )
            full_hit1 = hit_at_k(logits_pi_m, chosen_idx, 1)
            full_hit5 = hit_at_k(logits_pi_m, chosen_idx, 5)
            full_hit10 = hit_at_k(logits_pi_m, chosen_idx, 10)

            # collect candidate-restricted logits for opening distribution
            batch_open_fens: List[str] = []
            batch_open_logits: List[torch.Tensor] = []

            for i in range(bs):
                agg["sf_total_rows"] += 1

                fen = fens[i]
                board = chess.Board(fen)
                if board.is_game_over(claim_draw=True):
                    continue

                if cfg.use_gibbs:
                    out = compute_sf_helper_w_gibbs_for_one_position(
                        fen=fen,
                        chosen_uci=chosen[i],
                        rejected_uci=rejected[i],
                        sf_engine=self._sf_engine,
                        logits_masked_1d=logits_pi_m[i],
                        base_logp_full_1d=base_logp_full[i],
                        all_moves_dict=self.all_moves_dict,
                        cfg=cfg,
                        full_hit={
                            1: int(full_hit1[i].item()),
                            5: int(full_hit5[i].item()),
                            10: int(full_hit10[i].item()),
                        },
                        rng=rng,
                    )
                else:
                    infos = self._sf_engine.analyse(board, limit, multipv=int(cfg.multipv_topk))
                    cands: List[Tuple[str, int]] = []
                    for info in infos:
                        pv = info.get("pv")
                        score = info.get("score")
                        if not pv or score is None:
                            continue
                        uci = pv[0].uci()
                        cp = _score_to_cp(score, turn=board.turn)
                        cands.append((uci, cp))
                    if not cands:
                        continue

                    best_cp = max(cp for _, cp in cands)
                    out = compute_sf_helper_for_one_position(
                        fen=fen,
                        chosen_uci=chosen[i],
                        rejected_uci=rejected[i],
                        cands=cands,
                        best_cp=int(best_cp),
                        logits_masked_1d=logits_pi_m[i],
                        base_logp_full_1d=base_logp_full[i],
                        all_moves_dict=self.all_moves_dict,
                        cfg=cfg,
                        full_hit={
                            1: int(full_hit1[i].item()),
                            5: int(full_hit5[i].item()),
                            10: int(full_hit10[i].item()),
                        },
                        rng=rng,
                    )
                if out is None:
                    continue
                res, _dbg = out
                ply_abs = fen_to_ply_abs(fen)
                if ply_abs in opening_cfg_sf.plies:
                    batch_open_fens.append(fen)
                    batch_open_logits.append(_dbg["cands_logits_full"])

                agg["sf_valid_rows"] += 1
                agg["sf_help_top1_acc"] += 1.0 if res.selected_uci == chosen[i] else 0.0
                agg["sf_help_top5_hit_cand"] += float(res.cand_hit5)
                agg["sf_help_top10_hit_cand"] += float(res.cand_hit10)
                agg["sf_help_mean_cp_gap"] += float(res.cp_gap)
                agg["sf_help_best_sf_rate"] += 1.0 if res.is_best_sf else 0.0
                agg["sf_help_mean_entropy"] += float(res.entropy)
                agg["sf_help_mean_logp_selected_full"] += float(res.logp_selected_full)

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

                # opening move distributions (first ply for white and black)
                ply_abs = fen_to_ply_abs(fen)
                if ply_abs == 0:
                    agg["opening_selected_white_ply0"][res.selected_uci] += 1
                if ply_abs == 1:
                    agg["opening_selected_black_ply1"][res.selected_uci] += 1

            if batch_open_logits:
                logits_stack = torch.stack(batch_open_logits, dim=0).to(self.device)  # [B_open, V]
                update_opening_distributions_from_logits(
                    opening_counts=opening_counts_sf_adv,
                    fens=batch_open_fens,
                    logits_masked=logits_stack,
                    all_moves=self.all_moves,
                    cfg=opening_cfg_sf,
                )
        opening_summary_sf = summarize_opening_distribution(opening_counts_sf_adv, topn=50, normalize=True)


        v = float(agg["sf_valid_rows"])
        finalize_div("sf_help_top1_acc", v)
        finalize_div("sf_help_top5_hit_cand", v)
        finalize_div("sf_help_top10_hit_cand", v)
        finalize_div("sf_help_mean_cp_gap", v)
        finalize_div("sf_help_best_sf_rate", v)
        finalize_div("sf_help_mean_entropy", v)
        finalize_div("sf_help_mean_logp_selected_full", v)

        finalize_div("sf_help_mean_p_chosen_cond", v)
        finalize_div("sf_help_mean_p_rejected_cond", v)
        finalize_div("sf_help_mean_logp_chosen_cond", v)
        finalize_div("sf_help_mean_logp_rejected_cond", v)
        finalize_div("sf_help_mean_gap_logp_cond", v)
        finalize_div("sf_help_mean_kl_q_vs_base", v)

        finalize_div("full_hit1", v)
        finalize_div("full_hit5", v)
        finalize_div("full_hit10", v)

        agg["sf_help_chosen_in_cands_rate"] = (agg["sf_help_chosen_in_cands"] / v) if v > 0 else float("nan")
        denom_in = max(1.0, agg["sf_help_chosen_in_cands"])
        agg["sf_help_mean_p_chosen_given_in"] = (agg["sf_help_p_chosen_in_cands_sum"] / denom_in)

        # serialize opening counters
        agg["opening_selected_white_ply0"] = [{"uci": u, "count": c} for u, c in agg["opening_selected_white_ply0"].most_common(50)]
        agg["opening_selected_black_ply1"] = [{"uci": u, "count": c} for u, c in agg["opening_selected_black_ply1"].most_common(50)]

        agg["sf_config"] = {
            "stockfish_path": cfg.stockfish_path,
            "depth": int(cfg.depth),
            "multipv_topk": int(getattr(cfg, "multipv_topk", getattr(cfg, "multipv", 10))),

            "use_gibbs": bool(getattr(cfg, "use_gibbs", False)),

            # candidate selection (policy-side)
            "k": int(getattr(cfg, "k", 40)),

            # adaptive cp window
            "restrict_cp_window": cfg.restrict_cp_window,  # keep legacy if you still set it
            "restrict_cp_window_base": float(getattr(cfg, "restrict_cp_window_base", getattr(cfg, "restrict_cp_window", 20) or 20)),
            "restrict_cp_window_max_factor": float(getattr(cfg, "restrict_cp_window_max_factor", 6.0)),
            "game_len_plies": float(getattr(cfg, "game_len_plies", 80.0)),

            # KL-regularized improvement hyperparams
            "base_temperature": float(getattr(cfg, "base_temperature", getattr(cfg, "temperature", 0.3))),
            "min_temperature": float(getattr(cfg, "min_temperature", 1e-3)),
            "max_temperature": float(getattr(cfg, "max_temperature", 5.0)),
            "cp_scale": float(getattr(cfg, "cp_scale", 150.0)),
            "cp_cap": int(getattr(cfg, "cp_cap", 2000)),

            # sampling
            "sample": bool(getattr(cfg, "sample", True)),
            "seed": int(cfg.seed),

            # misc
            "uci_elo": cfg.uci_elo,
            "threads": int(getattr(cfg, "threads", 1)),
            "hash_mb": int(getattr(cfg, "hash_mb", 64)),
            "timeout_s": float(getattr(cfg, "timeout_s", getattr(cfg, "timeout", 0.0))),
        }

        agg["sf_opening_summary"] = opening_summary_sf
        return agg


# ============================================================
# Concrete model types
# ============================================================

class BaseMaia2(EvalModel):
    @property
    def tag(self) -> str:
        return "base_maia2"

    @torch.inference_mode()
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
        batch_meta_data
    ) -> torch.Tensor:
        # Not trained; define something stable for reporting.
        # Here: mean NLL on chosen.
        return (-logp_pi_ch).mean()


class DpoModel(EvalModel):
    @property
    def tag(self) -> str:
        return "dpo"

    @torch.inference_mode()
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
        batch_meta_data
    ) -> torch.Tensor:
        x = self.beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj))
        return -torch.nn.functional.logsigmoid(x).mean()


class SftModel(EvalModel):
    @property
    def tag(self) -> str:
        return "sft"

    @torch.inference_mode()
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
        batch_meta_data
    ) -> torch.Tensor:
        # SFT objective approximated as NLL on chosen.
        return (-logp_pi_ch).mean()


class SftPairwiseModel(EvalModel):
    @property
    def tag(self) -> str:
        return "sft_pairwise"

    @torch.inference_mode()
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
        batch_meta_data
    ) -> torch.Tensor:
        # Pairwise logistic loss without reference:
        # -log(sigmoid(logp(ch) - logp(rj)))
        x = (logp_pi_ch - logp_pi_rj)
        return -torch.nn.functional.logsigmoid(x).mean()
    

class SftAndDpo(EvalModel):

    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        sf_cfg: Optional[SfConfig] = None,
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg)
        self.dpo_loss_weight = dpo_loss_weight

    @property
    def tag(self) -> str:
        return f"sft_and_dpo_beta={self.beta:.2f}"

    @torch.inference_mode()
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
        batch_meta_data
    ) -> torch.Tensor:
        dpo_loss = self.beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj))
        sft_loss = -logp_pi_ch
        return (-self.dpo_loss_weight * torch.nn.functional.logsigmoid(dpo_loss) + sft_loss).mean()

# Todo: add style loss params to constructor and heuristics for v1/v2. V3 should use actual style embedding model
class SftAndDpoWStyleV1(EvalModel):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        sf_cfg: Optional[SfConfig] = None,
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg)
        self.dpo_loss_weight = dpo_loss_weight
        self.style_tau = style_tau
        self.style_cp_scale = 40.0
        self.style_piece_bonus = 1.0
        self.style_positional_bonus = 2.0
    
    @property
    def tag(self) -> str:
        return "sft_and_dpo_w_style_sim_utility_weight"

    @torch.inference_mode()
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
    
        style_scores = torch.tensor(
            [
                compute_style_score(
                    fen=fen,
                    chosen_uci=ch,
                    rejected_uci=rj,
                    chosen_cp=ch_cp,
                    rejected_cp=rj_cp,
                    cp_scale=self.style_cp_scale,
                    piece_bonus=self.style_piece_bonus,
                    positional_bonus=self.style_positional_bonus,
                )
                for fen, ch, rj, ch_cp, rj_cp, _, _, _, _, _ in batch_meta_data
            ],
            dtype=torch.float32,
            device=self.device,
        )

        loss = (
            self.dpo_loss_weight
            * dpo_loss_style_weighted(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
                style_score=style_scores,
                beta=self.beta,
                tau=self.style_tau,
            )
            + supervised_nll_loss(logits_pi_m, idx_t)
        )

        return loss
    

class SftAndDpoWStyleV2(EvalModel):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        sf_cfg: Optional[SfConfig] = None,
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg)
        self.dpo_loss_weight = dpo_loss_weight
        self.style_tau = style_tau
        self.style_cp_scale = 40.0
        self.style_piece_bonus = 1.0
        self.style_positional_bonus = 2.0

    @property
    def tag(self) -> str:
        return "sft_and_dpo_w_style_v2"

    @torch.inference_mode()
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
        batch_meta_data
    ) -> torch.Tensor:
        style_scores = torch.tensor(
            [
                compute_style_score_v2(
                        fen=fen,
                        chosen_uci=ch,
                        rejected_uci=rj,
                        chosen_cp=ch_cp,
                        rejected_cp=rj_cp,
                        prev_fens=prev_fens,
                        next_fens_chosen=next_fens_chosen,
                        next_fens_rejected=next_fens_rejected,
                        ply_idx=ply_idx,
                        phase=None,
                    )
                    for fen, ch, rj, ch_cp, rj_cp, ply_idx, prev_fens, next_fens_chosen, next_fens_rejected, meta_list
                    in batch_meta_data
            ],
            dtype=torch.float32,
            device=self.device,
        )

        loss = (
            self.dpo_loss_weight
            * dpo_loss_style_weighted(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
                style_score=style_scores,
                beta=self.beta,
                tau=self.style_tau,
            )
            + supervised_nll_loss(logits_pi_m, idx_t)
        )
        return loss
    
class SftAndDpoWStyleV3(EvalModel):

    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        style_embedding_chkpt: str = "",
        sf_cfg: Optional[SfConfig] = None,
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg)
        self.dpo_loss_weight = dpo_loss_weight
        self.style_tau = style_tau
        self.style_embedding_chkpt = style_embedding_chkpt

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
            init_from_checkpoint=self.style_embedding_chkpt, # this is all that really matters for inference
            init_reset_optimizer=True,
            init_strict_load=True,
        )
        self.style_embedding_model = StyleEncoder(style_encoder_training_cfg)
        self.style_embedding_model.eval()

    @property
    def tag(self) -> str:
        return "sft_and_dpo_w_style_v3"

    @torch.inference_mode()
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
        batch_meta_data
    ) -> torch.Tensor:
        style_scores = torch.tensor(
            [
                compute_style_score_v3(
                    fen=fen,
                    chosen_uci=ch,
                    rejected_uci=rj,
                    prev_fens=prev_fens,
                    style_embedding_model=self.style_embedding_model,
                    event=meta_list["event"],
                    style_tau=self.style_tau,
                )
                for fen, ch, rj, ch_cp, rj_cp, ply_idx, prev_fens, next_fens_chosen, next_fens_rejected, meta_list
                in batch_meta_data
            ],
            dtype=torch.float32,
            device=self.device,
        )

        loss = (
            self.dpo_loss_weight
            * dpo_loss_style_weighted_v3(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
                style_score=style_scores,
                beta=self.beta,
                tau=self.style_tau,
            )
            + supervised_nll_loss(logits_pi_m, idx_t)
        )
        return loss
    

# ------------------------------------------------------------


# Convenience “with SF helper” variants
# (These are thin wrappers that only change tag; SF is enabled by passing sf_cfg)
class DpoWithSfHelper(DpoModel):

    def __init__(self, *, maia_type: str = "blitz", device: torch.device, policy_pt_path: Optional[str] = None, beta: float = 0.1, sf_cfg: Optional[SfConfig] = None):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg)
        self.depth = sf_cfg.depth
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        
    @property
    def tag(self) -> str:
        return f"dpo_w_sf_helper_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}"

class SftWithSfHelper(SftModel):

    def __init__(self, *, maia_type: str = "blitz", device: torch.device, policy_pt_path: Optional[str] = None, beta: float = 0.1, sf_cfg: Optional[SfConfig] = None):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg)
        self.depth = sf_cfg.depth
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        
    @property
    def tag(self) -> str:
        return f"sft_w_sf_helper_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}"

class SftPairwiseWithSfHelper(SftPairwiseModel):

    def __init__(self, *, maia_type: str = "blitz", device: torch.device, policy_pt_path: Optional[str] = None, beta: float = 0.1, sf_cfg: Optional[SfConfig] = None):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg)
        self.depth = sf_cfg.depth
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        
    @property
    def tag(self) -> str:
        return f"sft_pairwise_w_sf_helper_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}_{self.sf_cfg.use_gibbs}"

class SftAndDpoWithSfHelper(SftAndDpo):

    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        sf_cfg: Optional[SfConfig] = None,
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, dpo_loss_weight=dpo_loss_weight, sf_cfg=sf_cfg)
        self.depth = sf_cfg.depth = sf_cfg.depth 
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window

    @property
    def tag(self) -> str:
        return f"sft_and_dpo_beta={self.beta:.2f}_dpo_w_{self.dpo_loss_weight:.2f}_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}"
    
    
class SftAndDpoWStyleV1WithSfHelper(SftAndDpoWStyleV1):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        sf_cfg: Optional[SfConfig] = None,
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, dpo_loss_weight=dpo_loss_weight, style_tau=style_tau, sf_cfg=sf_cfg)
        self.depth = sf_cfg.depth = sf_cfg.depth 
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window

    @property
    def tag(self) -> str:
        return f"sft_and_dpo_w_style_v1_beta={self.beta:.2f}_dpo_w_{self.dpo_loss_weight:.2f}_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}"
    
class SftAndDpoWStyleV2WithSfHelper(SftAndDpoWStyleV2):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        sf_cfg: Optional[SfConfig] = None,
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, dpo_loss_weight=dpo_loss_weight, style_tau=style_tau, sf_cfg=sf_cfg)
        self.depth = sf_cfg.depth = sf_cfg.depth 
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window

    @property
    def tag(self) -> str:
        return f"sft_and_dpo_w_style_v2_beta={self.beta:.2f}_dpo_w_{self.dpo_loss_weight:.2f}_tau_{self.style_tau:.2f}_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}"
    
class SftAndDpoWStyleV3WithSfHelper(SftAndDpoWStyleV3):
    def __init__(
        self,
        *,
        maia_type: str = "blitz",
        device: torch.device,
        policy_pt_path: Optional[str] = None,
        beta: float = 0.1,
        dpo_loss_weight: float = 1.0,
        style_tau: float = 0.1,
        embedding_model_chkpt_name: str = "",
        sf_cfg: Optional[SfConfig] = None,
    ):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, dpo_loss_weight=dpo_loss_weight, style_tau=style_tau, style_embedding_chkpt=embedding_model_chkpt_name, sf_cfg=sf_cfg)
        self.depth = sf_cfg.depth = sf_cfg.depth 
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        self.style_embedding_chkpt = embedding_model_chkpt_name

    @property
    def tag(self) -> str:
        return f"sft_and_dpo_w_style_v3_beta={self.beta:.2f}_dpo_w_{self.dpo_loss_weight:.2f}_tau_{self.style_tau:.2f}_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}_emb_chkpt_{self.style_embedding_chkpt}"


# ============================================================
# Factory: instantiate the family you want with shared args
# ============================================================

def build_models_for_gm(
    *,
    maia_type: str,
    device: torch.device,
    gm_dir: Path,
    sf_cfgs: Optional[List[SfConfig]],
    beta: float,
    disable_initial_model_types: bool = False,
) -> List[EvalModel]:
    """
    gm_dir expected to contain:
      - policy_dpo_best.pt
      - policy_sft_best.pt
      - policy_pairwise_sft_best.pt
    Adjust filenames as needed.
    """
    dpo_pt = gm_dir / "policy_dpo_best.pt"
    sft_pt = gm_dir / "policy_sft_best.pt"
    pw_pt = gm_dir / "policy_pairwise_sft_best.pt"

    

    models: List[EvalModel] = []

    # base only
    models.append(BaseMaia2(maia_type=maia_type, device=device, policy_pt_path=None, beta=beta, sf_cfg=None))
    if not disable_initial_model_types:
        # non-SF runs
        models.append(DpoModel(maia_type=maia_type, device=device, policy_pt_path=str(dpo_pt), beta=beta, sf_cfg=None))
        models.append(SftModel(maia_type=maia_type, device=device, policy_pt_path=str(sft_pt), beta=beta, sf_cfg=None))
        models.append(SftPairwiseModel(maia_type=maia_type, device=device, policy_pt_path=str(pw_pt), beta=beta, sf_cfg=None))

    # SF-helper runs (depth lives in sf_cfg.depth)
    if sf_cfgs is not None:
        for sf_cfg in sf_cfgs:
            models.append(DpoWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=str(dpo_pt), beta=beta, sf_cfg=sf_cfg))
            models.append(SftWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=str(sft_pt), beta=beta, sf_cfg=sf_cfg))
            models.append(SftPairwiseWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=str(pw_pt), beta=beta, sf_cfg=sf_cfg))
            beta = 0.6
            dpo_loss_weight = 0.1
            style_tau = 0.25
            models.append(SftAndDpoWithSfHelper(maia_type=maia_type, 
                                                device=device, 
                                                policy_pt_path=f"{gm_dir}/policy_best_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}.pt", 
                                                beta=beta, 
                                                dpo_loss_weight=dpo_loss_weight,
                                                sf_cfg=sf_cfg))
            
            models.append(SftAndDpoWStyleV1WithSfHelper(maia_type=maia_type, 
                                                        device=device, 
                                                        policy_pt_path=f"{gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau={style_tau:.2f}.pt", 
                                                        beta=beta, 
                                                        dpo_loss_weight=dpo_loss_weight, 
                                                        style_tau=style_tau, 
                                                        sf_cfg=sf_cfg))
            
            models.append(SftAndDpoWStyleV2WithSfHelper(maia_type=maia_type, 
                                                        device=device, 
                                                        policy_pt_path=f"{gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau={style_tau:.2f}.pt", 
                                                        beta=beta, 
                                                        dpo_loss_weight=dpo_loss_weight,
                                                        style_tau=style_tau,
                                                        sf_cfg=sf_cfg))

    return models


def build_sf_models_for_gm(
    *,
    maia_type: str,
    device: torch.device,
    experiment1_gm_dir: Path = Path("./final_experiments_for_paper/experiments1/trained_models_twic/carlsen/"),
    experiment2_gm_dir: Path = Path("./final_experiments_for_paper/experiments2_style_model/trained_models_single_gm_twic/carlsen/"),
    style_embedding_model_dir: Path = Path('./final_experiments_for_paper/experiments2_style_model/trained_models/'),
    sf_cfgs: List[SfConfig],
) -> List[EvalModel]:
    """
    gm_dir expected to contain:
      - policy_dpo_best.pt
      - policy_sft_best.pt
      - policy_pairwise_sft_best.pt
    Adjust filenames as needed.
    """
    dpo_pt = experiment1_gm_dir / "policy_dpo_best.pt"
    sft_pt = experiment1_gm_dir / "policy_sft_best.pt"
    pw_pt = experiment1_gm_dir / "policy_pairwise_sft_best.pt"
    sft_and_dpo_pt = experiment1_gm_dir / "policy_sft_and_dpo_best.pt"
    sft_and_dpo_w_style_v1_pt = experiment1_gm_dir / "policy_sft_and_dpo_w_style_v1_best.pt"
    sft_and_dpo_w_style_v2_pt = experiment1_gm_dir / "policy_sft_and_dpo_w_style_v2_best.pt"

    sft_and_dpo_w_style_v3_pt = experiment2_gm_dir / "policy_sft_and_dpo_w_style_v3_best.pt"

    final_v2_embedding_model_name = "final_v2_phi1_tau0_25_if_winner__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42"
    final_v3_embedding_model_name = "final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42"

    models: List[EvalModel] = []


    # SF-helper runs (depth lives in sf_cfg.depth)
    for sf_cfg in sf_cfgs:
        models.append(DpoWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_dpo_beta=0.60.pt", beta=0.6, sf_cfg=sf_cfg))
        models.append(SftWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_sft_best.pt", beta=0.6, sf_cfg=sf_cfg))
        models.append(SftPairwiseWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_pairwise_sft_best.pt", beta=0.6, sf_cfg=sf_cfg))

        beta = 0.6
        dpo_loss_weight = 0.1
        style_tau = 0.25
        models.append(SftAndDpoWithSfHelper(maia_type=maia_type, 
                                            device=device, 
                                            policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}.pt", 
                                            beta=beta, 
                                            dpo_loss_weight=dpo_loss_weight,
                                            sf_cfg=sf_cfg))        
        #models.append(SftAndDpoWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_beta=0.60_dpo_loss_weight=0.20.pt", beta=0.6, dpo_loss_weight=0.2, sf_cfg=sf_cfg))
        #models.append(SftAndDpoWithSfHelper(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_beta=0.60_dpo_loss_weight=0.40.pt", beta=0.6, dpo_loss_weight=0.4, sf_cfg=sf_cfg))


        models.append(SftAndDpoWStyleV1WithSfHelper(maia_type=maia_type, 
                                                    device=device, 
                                                    policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau={style_tau:.2f}.pt", 
                                                    beta=beta, 
                                                    dpo_loss_weight=dpo_loss_weight, 
                                                    style_tau=style_tau, 
                                                    sf_cfg=sf_cfg))        #models.append(SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75.pt", beta=0.6, dpo_loss_weight=0.1, style_tau=0.75, sf_cfg=sf_cfg))
        #models.append(SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25.pt", beta=0.6, dpo_loss_weight=0.1, style_tau=1.25, sf_cfg=sf_cfg))
        #models.append(SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=0.25, sf_cfg=sf_cfg))
        #models.append(SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=0.75, sf_cfg=sf_cfg))
        #models.append(SftAndDpoWithSfHelperWStyleV1(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=1.25, sf_cfg=sf_cfg))

        models.append(SftAndDpoWStyleV2WithSfHelper(maia_type=maia_type, 
                                                    device=device, 
                                                    policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau={style_tau:.2f}.pt", 
                                                    beta=beta, 
                                                    dpo_loss_weight=dpo_loss_weight,
                                                    style_tau=style_tau,
                                                    sf_cfg=sf_cfg))        #models.append(SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75.pt", beta=0.6, dpo_loss_weight=0.1, style_tau=0.75, sf_cfg=sf_cfg))
        #models.append(SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25.pt", beta=0.6, dpo_loss_weight=0.1, style_tau=1.25, sf_cfg=sf_cfg))
        #models.append(SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=0.25, sf_cfg=sf_cfg))
        #models.append(SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=0.75, sf_cfg=sf_cfg))
        #models.append(SftAndDpoWithSfHelperWStyleV2(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25.pt", beta=0.6, dpo_loss_weight=0.2, style_tau=1.25, sf_cfg=sf_cfg))

        models.append(SftAndDpoWStyleV3WithSfHelper(maia_type=maia_type, 
                                                    device=device, 
                                                    policy_pt_path=f"{experiment2_gm_dir}/policy_best_sft_and_dpo_w_style_v3_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_tau={style_tau:.2f}_embedding_model=final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42.pt", 
                                                    beta=beta, 
                                                    dpo_loss_weight=dpo_loss_weight,
                                                    style_tau=style_tau,
                                                    embedding_model_chkpt_name=f"{style_embedding_model_dir}/final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42/best.pt",
                                                    sf_cfg=sf_cfg))        
        #models.append(SftAndDpoWithSfHelperWStyleV3(maia_type=maia_type, device=device, policy_pt_path=f"{experiment2_gm_dir}/policy_best_sft_and_dpo_w_style_v3_beta=0.60_dpo_loss_weight=0.10_style_tau=0.25_embedding_model={final_v3_embedding_model_name}.pt", beta=0.6, dpo_loss_weight=0.1, style_tau_inference=0.25, embedding_model_name=final_v3_embedding_model_name, sf_cfg=sf_cfg))

        for dpo_loss_weight in [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:
            for style_tau in [0.25, 0.75, 1.25]:
                #models.append(SftAndDpoWithSfHelperWStyleV3(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v3_beta=0.60_dpo_loss_weight={dpo_loss_weight:.2f}_style_tau={style_tau:.2f}_embedding_model={final_v2_embedding_model_name}.pt", beta=0.6, dpo_loss_weight=dpo_loss_weight, style_tau_inference=style_tau, embedding_model_name=final_v2_embedding_model_name, sf_cfg=sf_cfg))
                #models.append(SftAndDpoWithSfHelperWStyleV3(maia_type=maia_type, device=device, policy_pt_path=f"{experiment1_gm_dir}/policy_best_sft_and_dpo_w_style_v3_beta=0.60_dpo_loss_weight={dpo_loss_weight:.2f}_style_tau={style_tau:.2f}_embedding_model={final_v3_embedding_model_name}.pt", beta=0.6, dpo_loss_weight=dpo_loss_weight, style_tau_inference=style_tau, embedding_model_name=final_v3_embedding_model_name, sf_cfg=sf_cfg))
                pass
        


    return models

