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
import torch
from torch.utils.data import DataLoader, Dataset

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

def _score_to_cp(score: chess.engine.PovScore, mate_score: int = 100_000) -> int:
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
        m = rel.mate()
        if m is not None:
            return mate_score if m > 0 else -mate_score
        return 0
    return int(cp)

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

            loss = self.compute_training_style_loss(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
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

                infos = self._sf_engine.analyse(board, limit, multipv=int(cfg.multipv_topk))
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
            "multipv_topk": int(cfg.multipv_topk),
            "uci_elo": cfg.uci_elo,
            "restrict_cp_window": cfg.restrict_cp_window,
            "temperature": float(cfg.temperature),
            "sample": bool(cfg.sample),
            "seed": int(cfg.seed),
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

    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
    ) -> torch.Tensor:
        # Not trained; define something stable for reporting.
        # Here: mean NLL on chosen.
        return (-logp_pi_ch).mean()


class DpoModel(EvalModel):
    @property
    def tag(self) -> str:
        return "dpo"

    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
    ) -> torch.Tensor:
        x = self.beta * ((logp_pi_ch - logp_pi_rj) - (logp_ref_ch - logp_ref_rj))
        return -torch.nn.functional.logsigmoid(x).mean()


class SftModel(EvalModel):
    @property
    def tag(self) -> str:
        return "sft"

    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
    ) -> torch.Tensor:
        # SFT objective approximated as NLL on chosen.
        return (-logp_pi_ch).mean()


class SftPairwiseModel(EvalModel):
    @property
    def tag(self) -> str:
        return "sft_pairwise"

    def compute_training_style_loss(
        self,
        *,
        logp_pi_ch: torch.Tensor,
        logp_pi_rj: torch.Tensor,
        logp_ref_ch: torch.Tensor,
        logp_ref_rj: torch.Tensor,
    ) -> torch.Tensor:
        # Pairwise logistic loss without reference:
        # -log(sigmoid(logp(ch) - logp(rj)))
        x = (logp_pi_ch - logp_pi_rj)
        return -torch.nn.functional.logsigmoid(x).mean()


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
        return f"dpo_w_sf_helper_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}"

class SftWithSfHelper(SftModel):

    def __init__(self, *, maia_type: str = "blitz", device: torch.device, policy_pt_path: Optional[str] = None, beta: float = 0.1, sf_cfg: Optional[SfConfig] = None):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg)
        self.depth = sf_cfg.depth
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        
    @property
    def tag(self) -> str:
        return f"sft_w_sf_helper_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}"

class SftPairwiseWithSfHelper(SftPairwiseModel):

    def __init__(self, *, maia_type: str = "blitz", device: torch.device, policy_pt_path: Optional[str] = None, beta: float = 0.1, sf_cfg: Optional[SfConfig] = None):
        super().__init__(maia_type=maia_type, device=device, policy_pt_path=policy_pt_path, beta=beta, sf_cfg=sf_cfg)
        self.depth = sf_cfg.depth
        self.multipv_topk = sf_cfg.multipv_topk
        self.restrict_cp_window = sf_cfg.restrict_cp_window
        
    @property
    def tag(self) -> str:
        return f"sft_pairwise_w_sf_helper_depth_{self.depth}_multipv_topk_{self.multipv_topk}_restrict_cp_window_{self.restrict_cp_window}"


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

    return models
