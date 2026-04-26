from __future__ import annotations

import logging
import hashlib
import math
import os
import random
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chess
import torch

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move

from grandmaster_dpo.eval.stockfish_helpers import make_stockfish
from grandmaster_dpo.website.policy_only.schemas import ClockState, EngineConfigRequest, GameStatusResponse
from grandmaster_dpo.website.policy_only.service.opening_book import maybe_get_opening_book

logger = logging.getLogger(__name__)


def mirror_uci_like_board_mirror(uci: str) -> str:
    mv = chess.Move.from_uci(uci)
    f = chess.square_mirror(mv.from_square)
    t = chess.square_mirror(mv.to_square)
    return chess.Move(f, t, promotion=mv.promotion).uci()


def uci_to_vocab_index(all_moves_dict: dict[str, int], fen: str, uci: str) -> int:
    side = fen.split(" ")[1]
    uci_eff = mirror_uci_like_board_mirror(uci) if side == "b" else uci
    return int(all_moves_dict.get(uci_eff, -1))


def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))


def batch_preprocess_single(
    *,
    all_moves_dict: dict[str, int],
    elo_dict: dict[str, int],
    fen: str,
    elo_self: int,
    elo_oppo: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bi, es_cat, eo_cat, lm = inference.preprocessing(
        fen,
        int(elo_self),
        int(elo_oppo),
        elo_dict,
        all_moves_dict,
    )
    board_input = bi.unsqueeze(0).to(device)
    legal_moves = lm.unsqueeze(0).to(device)
    es_t = torch.tensor([int(es_cat)], device=device).long()
    eo_t = torch.tensor([int(eo_cat)], device=device).long()
    return board_input, legal_moves, es_t, eo_t


def forward_logits(
    model: torch.nn.Module,
    board_input: torch.Tensor,
    es: torch.Tensor,
    eo: torch.Tensor,
) -> torch.Tensor:
    logits, _, _ = model(board_input, es, eo)
    return logits


def load_state_dict_fuzzy(pt: Path, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    sd = torch.load(pt, map_location=map_location)
    if isinstance(sd, dict) and "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
        sd = sd["model_state_dict"]
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format in {pt}: {type(sd)}")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def get_module_by_dotted_name(root: torch.nn.Module, dotted: str) -> Optional[torch.nn.Module]:
    cur: Any = root
    for part in dotted.split("."):
        if part.isdigit():
            idx = int(part)
            if isinstance(cur, (torch.nn.ModuleList, list, tuple)):
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
            else:
                return None
        else:
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
    return cur if isinstance(cur, torch.nn.Module) else None


def max_elo_supported(elo_dict: dict[str, int]) -> int:
    mx = None
    for key in elo_dict.keys():
        match = re.match(r"^>=\s*(\d+)$", str(key))
        if match:
            mx = max(mx or 0, int(match.group(1)))
    return mx if mx is not None else 3000


def fen_ply_abs(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return 2 * (fullmove - 1) + (1 if side == "b" else 0)


def game_phase_from_ply_abs(ply_abs: int) -> str:
    if ply_abs < 20:
        return "opening"
    if ply_abs < 60:
        return "middlegame"
    return "endgame"


def position_identity(board: chess.Board) -> str:
    return " ".join(board.fen().split(" ")[:4])


def game_status_from_board(board: chess.Board) -> GameStatusResponse:
    if board.is_checkmate():
        winner = "black" if board.turn == chess.WHITE else "white"
        return GameStatusResponse(state="checkmate", winner=winner, reason="checkmate")
    if board.is_stalemate():
        return GameStatusResponse(state="stalemate", winner=None, reason="stalemate")
    if board.is_insufficient_material():
        return GameStatusResponse(state="draw", winner=None, reason="insufficient_material")
    if board.can_claim_draw():
        return GameStatusResponse(state="draw", winner=None, reason="claimable_draw")
    if board.is_game_over(claim_draw=True):
        return GameStatusResponse(state="draw", winner=None, reason="game_over")
    return GameStatusResponse(state="ongoing", winner=None, reason="")


def _score_to_cp(score: chess.engine.PovScore, mate_score: int = 100_000) -> int:
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
        mate = rel.mate()
        if mate is not None:
            return mate_score if mate > 0 else -mate_score
        return 0
    return int(cp)


def _phase_penalty_for_current_position(engine_config: EngineConfigRequest, *, penalty_name: str, phase: str) -> int:
    draw_penalties = getattr(engine_config, "draw_penalties", None)
    if draw_penalties is None or not draw_penalties.enabled:
        return 0
    penalty_cfg = getattr(draw_penalties, penalty_name, None)
    if penalty_cfg is None:
        return 0
    value = getattr(penalty_cfg, phase, 0)
    return max(0, int(value or 0))


def _is_drawish_non_checkmate(board: chess.Board) -> bool:
    return board.can_claim_draw() or (board.is_game_over(claim_draw=True) and not board.is_checkmate())


def _has_reply_leading_to_draw(board: chess.Board) -> bool:
    if _is_drawish_non_checkmate(board):
        return True
    for reply in board.legal_moves:
        nxt = board.copy(stack=True)
        nxt.push(reply)
        if _is_drawish_non_checkmate(nxt):
            return True
    return False


def _has_reply_leading_to_draw_with_counts(
    board: chess.Board,
    *,
    position_counts: dict[str, int],
) -> bool:
    if _is_drawish_non_checkmate(board):
        return True
    for reply in board.legal_moves:
        nxt = board.copy(stack=False)
        nxt.push(reply)
        if position_counts.get(position_identity(nxt), 0) >= 2 or _is_drawish_non_checkmate(nxt):
            return True
    return False


def build_position_counts(*, start_fen: str, played_moves_uci: list[str]) -> dict[str, int]:
    board = chess.Board(start_fen)
    counts: Counter[str] = Counter({position_identity(board): 1})
    for move_uci in played_moves_uci:
        move = chess.Move.from_uci(str(move_uci))
        if move not in board.legal_moves:
            raise ValueError(f"Illegal historical move {move_uci} for reconstructed board")
        board.push(move)
        counts[position_identity(board)] += 1
    return dict(counts)


def apply_draw_penalties_to_candidate(
    board: chess.Board,
    candidate: dict[str, Any],
    *,
    engine_config: EngineConfigRequest,
    position_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    phase = game_phase_from_ply_abs(fen_ply_abs(board.fen()))
    repetition_penalty = 0
    one_move_penalty = 0
    try:
        move = chess.Move.from_uci(str(candidate["uci"]))
    except Exception:
        adjusted_cp = int(candidate["cp"])
        return {
            **candidate,
            "adjusted_cp": adjusted_cp,
            "draw_penalty_cp": 0,
            "repetition_x2_penalty_cp": None,
            "one_move_from_draw_penalty_cp": None,
        }
    if move in board.legal_moves:
        child = board.copy(stack=True)
        child.push(move)
        if position_counts is not None:
            child_key = position_identity(child)
            child_seen = int(position_counts.get(child_key, 0))
            child_counts = dict(position_counts)
            child_counts[child_key] = child_seen + 1
            if child_seen >= 1:
                repetition_penalty = _phase_penalty_for_current_position(
                    engine_config,
                    penalty_name="repetition_x2_penalty_cp",
                    phase=phase,
                )
            if child_counts[child_key] >= 3 or _has_reply_leading_to_draw_with_counts(
                child,
                position_counts=child_counts,
            ):
                one_move_penalty = _phase_penalty_for_current_position(
                    engine_config,
                    penalty_name="one_move_from_draw_penalty_cp",
                    phase=phase,
                )
        else:
            if child.is_repetition(2):
                repetition_penalty = _phase_penalty_for_current_position(
                    engine_config,
                    penalty_name="repetition_x2_penalty_cp",
                    phase=phase,
                )
            if _has_reply_leading_to_draw(child):
                one_move_penalty = _phase_penalty_for_current_position(
                    engine_config,
                    penalty_name="one_move_from_draw_penalty_cp",
                    phase=phase,
                )
    total_penalty = repetition_penalty + one_move_penalty
    adjusted_cp = int(candidate["cp"]) - total_penalty
    return {
        **candidate,
        "adjusted_cp": adjusted_cp,
        "draw_penalty_cp": total_penalty if total_penalty > 0 else None,
        "repetition_x2_penalty_cp": repetition_penalty if repetition_penalty > 0 else None,
        "one_move_from_draw_penalty_cp": one_move_penalty if one_move_penalty > 0 else None,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _phase_multiplier(phase_cfg: Any, phase: str) -> float:
    value = getattr(phase_cfg, phase, 1.0)
    return max(0.0, float(value if value is not None else 1.0))


def _mood_gate_enabled(*, random_seed: int, move_uci: str, feature_name: str, probability: float) -> bool:
    probability = _clamp01(probability)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    key = f"{int(random_seed)}:{move_uci}:{feature_name}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return value < probability


def _stable_probability(*, random_seed: int, fen: str, feature_name: str) -> float:
    key = f"{int(random_seed)}:{fen}:{feature_name}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _should_apply_position_gate(*, engine_config: EngineConfigRequest, fen: str, feature_name: str, probability: float) -> bool:
    probability = _clamp01(probability)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return _stable_probability(
        random_seed=engine_config.random_seed,
        fen=fen,
        feature_name=feature_name,
    ) < probability


def _candidate_mood_scores(
    board: chess.Board,
    candidate: dict[str, Any],
    *,
    engine_config: EngineConfigRequest,
    phase: str,
    best_adjusted_cp: int,
    cp_gap_window: int | None,
    max_rank: int,
) -> dict[str, float]:
    move_uci_raw = str(candidate.get("uci") or "")
    novelty_enabled = _mood_gate_enabled(
        random_seed=engine_config.random_seed,
        move_uci=move_uci_raw,
        feature_name="novelty",
        probability=engine_config.novelty_weight_prob
        * _phase_multiplier(engine_config.novelty_weight_phase, phase),
    )
    risk_enabled = _mood_gate_enabled(
        random_seed=engine_config.random_seed,
        move_uci=move_uci_raw,
        feature_name="risk",
        probability=engine_config.risk_weight_prob
        * _phase_multiplier(engine_config.risk_weight_phase, phase),
    )
    attack_enabled = _mood_gate_enabled(
        random_seed=engine_config.random_seed,
        move_uci=move_uci_raw,
        feature_name="attack",
        probability=engine_config.attack_weight_prob
        * _phase_multiplier(engine_config.attack_weight_phase, phase),
    )

    rank = int(candidate.get("multipv_rank") or max_rank or 1)
    rank_novelty = 0.0 if max_rank <= 1 else (rank - 1) / max(1, max_rank - 1)
    cp = int(candidate.get("adjusted_cp", candidate.get("cp", best_adjusted_cp)))
    cp_window = max(1, int(cp_gap_window if cp_gap_window is not None else 300))
    cp_drop = max(0, best_adjusted_cp - cp)
    cp_novelty = _clamp01(cp_drop / float(cp_window))
    novelty_score = _clamp01((0.65 * rank_novelty) + (0.35 * cp_novelty)) if novelty_enabled else 0.0

    attack_score = 0.0
    risk_score = 0.0
    try:
        move = chess.Move.from_uci(move_uci_raw)
    except Exception:
        return {"novelty_score": novelty_score, "risk_score": 0.0, "attack_score": 0.0}
    if move not in board.legal_moves:
        return {"novelty_score": novelty_score, "risk_score": 0.0, "attack_score": 0.0}

    moved_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)
    child = board.copy(stack=False)
    child.push(move)
    opponent = child.turn
    mover = not opponent

    if attack_enabled:
        if child.is_check():
            attack_score += 0.45
        if captured_piece is not None:
            attack_score += 0.20
        if moved_piece is not None:
            to_rank = chess.square_rank(move.to_square)
            if moved_piece.color == chess.WHITE:
                attack_score += 0.15 * _clamp01(to_rank / 7.0)
            else:
                attack_score += 0.15 * _clamp01((7 - to_rank) / 7.0)
        opponent_king = child.king(opponent)
        if opponent_king is not None:
            distance = chess.square_distance(move.to_square, opponent_king)
            attack_score += 0.20 * _clamp01((7 - distance) / 7.0)
        attack_score = _clamp01(attack_score)

    if risk_enabled:
        risk_score += 0.50 * cp_novelty
        if moved_piece is not None and child.is_attacked_by(opponent, move.to_square):
            piece_value = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}.get(
                moved_piece.piece_type,
                0,
            )
            if not child.is_attacked_by(mover, move.to_square):
                risk_score += 0.35
            risk_score += min(0.30, piece_value / 30.0)
        if captured_piece is not None:
            risk_score += 0.10
        risk_score = _clamp01(risk_score)

    return {
        "novelty_score": novelty_score,
        "risk_score": risk_score,
        "attack_score": attack_score,
    }


def _apply_weird_move_filters(
    *,
    kept: list[dict[str, Any]],
    engine_config: EngineConfigRequest,
    fen: str,
    phase: str,
    best_adjusted_cp: int,
) -> list[dict[str, Any]]:
    filtered = list(kept)
    if len(filtered) <= 1:
        return filtered

    if _should_apply_position_gate(
        engine_config=engine_config,
        fen=fen,
        feature_name="top_move_suppression",
        probability=engine_config.top_move_suppression_prob
        * _phase_multiplier(engine_config.top_move_suppression_phase, phase),
    ):
        best_cp = max(int(item.get("adjusted_cp", item["cp"])) for item in filtered)
        without_best = [
            item for item in filtered if int(item.get("adjusted_cp", item["cp"])) < best_cp
        ]
        if without_best:
            filtered = without_best

    if _should_apply_position_gate(
        engine_config=engine_config,
        fen=fen,
        feature_name="weird_move",
        probability=engine_config.weird_move_prob
        * _phase_multiplier(engine_config.weird_move_phase, phase),
    ):
        min_loss = max(0, int(engine_config.weird_move_min_cp_loss))
        max_loss = max(min_loss, int(engine_config.weird_move_max_cp_loss))
        weird = [
            item
            for item in filtered
            if min_loss <= best_adjusted_cp - int(item.get("adjusted_cp", item["cp"])) <= max_loss
        ]
        if weird:
            filtered = weird

    return filtered or kept


@dataclass(frozen=True)
class EngineProfile:
    gm_name: str
    maia_type: str = "blitz"
    elo_self: int = 2800
    elo_oppo: int = 2800


@dataclass
class TimerConfig:
    hook_layer: str = "last_ln"
    logits_feature: str = "masked_logits"
    ply_norm: float = 120.0
    min_think_ms: int = 50
    max_think_ms: int = 10_000
    safety_ms: int = 50


class TimerHead(torch.nn.Module):
    def __init__(self, in_dim: int, hidden1: int, hidden2: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden1),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_timer_head_from_ckpt(timer_pt: Path, device: torch.device) -> tuple[TimerHead, int]:
    sd = load_state_dict_fuzzy(timer_pt, map_location="cpu")
    w0 = sd.get("net.0.weight")
    w1 = sd.get("net.3.weight")
    if w0 is None or w1 is None:
        raise ValueError(f"Timer head checkpoint missing expected keys: {timer_pt}")
    hidden1 = int(w0.shape[0])
    in_dim = int(w0.shape[1])
    hidden2 = int(w1.shape[0])
    head = TimerHead(in_dim=in_dim, hidden1=hidden1, hidden2=hidden2, dropout=0.1)
    head.load_state_dict(sd, strict=True)
    head.to(device)
    head.eval()
    feat_dim = in_dim - 7
    if feat_dim <= 0:
        raise ValueError(f"Timer in_dim={in_dim} implies invalid feat_dim={feat_dim}")
    return head, feat_dim


class PolicyTimerRuntime:
    def __init__(
        self,
        maia: torch.nn.Module,
        all_moves_dict: dict[str, int],
        elo_dict: dict[str, int],
        device: torch.device,
        timer_cfg: TimerConfig,
    ) -> None:
        self.maia = maia
        self.all_moves_dict = all_moves_dict
        self.elo_dict = elo_dict
        self.device = device
        self.cfg = timer_cfg
        self._hook_buf: Optional[torch.Tensor] = None
        self._hooked = False
        self._hook_handle = None
        if timer_cfg.hook_layer:
            mod = get_module_by_dotted_name(self.maia, timer_cfg.hook_layer)
            if mod is not None:
                self._hook_handle = mod.register_forward_hook(self._forward_hook)
                self._hooked = True

    def _forward_hook(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        out = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(out):
            self._hook_buf = None
            return
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))
        self._hook_buf = out

    @torch.no_grad()
    def forward_once(self, fen: str, elo_self: int, elo_oppo: int) -> tuple[torch.Tensor, torch.Tensor]:
        mx = max_elo_supported(self.elo_dict)
        elo_self = min(int(elo_self), mx)
        elo_oppo = min(int(elo_oppo), mx)
        board_input, legal_moves, es_t, eo_t = batch_preprocess_single(
            all_moves_dict=self.all_moves_dict,
            elo_dict=self.elo_dict,
            fen=fen,
            elo_self=elo_self,
            elo_oppo=elo_oppo,
            device=self.device,
        )
        self._hook_buf = None
        logits = forward_logits(self.maia, board_input, es_t, eo_t)
        masked_logits = apply_legal_mask(logits, legal_moves)
        if self._hooked and self._hook_buf is not None:
            feats = self._hook_buf
        else:
            if self.cfg.logits_feature == "logprobs":
                feats = torch.log_softmax(masked_logits, dim=-1)
            else:
                feats = masked_logits
        return feats, masked_logits


@dataclass
class ModelBundle:
    profile: EngineProfile
    policy_runtime: PolicyTimerRuntime
    timer_head: Optional[TimerHead]
    timer_feat_dim: Optional[int]
    stockfish: chess.engine.SimpleEngine


@dataclass
class BotMoveResult:
    move_uci: str
    eval_cp: int
    pv_uci: list[str]
    candidate_moves: list[dict[str, Any]]
    stockfish_metrics: dict[str, Any]
    selected_probability: Optional[float]
    requested_think_ms: Optional[int]
    actual_think_ms: int
    engine_limit: dict[str, Any]


def _choose_book_move(
    *,
    gm_name: str,
    board: chess.Board,
    engine_config: EngineConfigRequest,
    played_moves_uci: list[str] | None,
) -> BotMoveResult | None:
    opening_book = maybe_get_opening_book(
        gm_name=gm_name,
        board=board,
        played_moves_uci=played_moves_uci,
    )
    if opening_book is None:
        return None

    _, engine_limit_dict, requested_think_ms = _limit_from_engine_config(engine_config, None)
    ranked_moves = sorted(
        opening_book.probabilities.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    candidate_moves: list[dict[str, Any]] = []
    for rank, (move_uci, prob) in enumerate(ranked_moves, start=1):
        candidate_moves.append(
            {
                "uci": move_uci,
                "cp": 0,
                "adjusted_cp": 0,
                "draw_penalty_cp": None,
                "repetition_x2_penalty_cp": None,
                "one_move_from_draw_penalty_cp": None,
                "mate": None,
                "pv_uci": [move_uci],
                "multipv_rank": rank,
                "in_cp_gap_window": True,
                "prob": float(prob),
                "depth": None,
                "seldepth": None,
                "nodes": None,
                "nps": None,
                "time_ms": 0,
                "tbhits": None,
            }
        )

    if engine_config.sample:
        rng = random.Random(int(engine_config.random_seed))
        cutoff = rng.random()
        total = 0.0
        chosen_idx = 0
        for idx, (_, prob) in enumerate(ranked_moves):
            total += float(prob)
            if cutoff <= total:
                chosen_idx = idx
                break
    else:
        chosen_idx = 0

    chosen_move, chosen_prob = ranked_moves[chosen_idx]
    return BotMoveResult(
        move_uci=chosen_move,
        eval_cp=0,
        pv_uci=[chosen_move],
        candidate_moves=candidate_moves,
        stockfish_metrics={
            "requested_multipv_topk": max(1, int(engine_config.stockfish_multipv_topk)),
            "returned_candidate_count": len(candidate_moves),
            "cp_gap_window": engine_config.cp_gap_window,
            "max_depth": None,
            "max_seldepth": None,
            "total_nodes": None,
            "max_nps": None,
            "max_time_ms": 0,
            "best_cp": 0,
            "best_move_uci": chosen_move,
            "best_move_mate": None,
            "selected_move_rank_by_cp": 1,
            "selected_move_rank_by_prob_within_window": chosen_idx + 1,
            "opening_book_branch": opening_book.branch_name,
            "opening_book_gm_name": gm_name,
        },
        selected_probability=float(chosen_prob),
        requested_think_ms=requested_think_ms,
        actual_think_ms=0,
        engine_limit=engine_limit_dict,
    )


_GLOBAL_LOCK = threading.Lock()
_GLOBALS: dict[str, Any] = {
    "device": None,
    "elo_dict": None,
    "all_moves": None,
    "all_moves_dict": None,
    "stockfish": None,
    "models": OrderedDict(),
}


def get_device() -> torch.device:
    return torch.device(os.environ.get("MAIA_DEVICE", "cpu"))


def get_global_context() -> tuple[torch.device, dict[str, int], list[str], dict[str, int]]:
    with _GLOBAL_LOCK:
        if _GLOBALS["device"] is None:
            _GLOBALS["device"] = get_device()
        if _GLOBALS["elo_dict"] is None:
            _GLOBALS["elo_dict"] = create_elo_dict()
        if _GLOBALS["all_moves"] is None or _GLOBALS["all_moves_dict"] is None:
            all_moves = get_all_possible_moves()
            _GLOBALS["all_moves"] = all_moves
            _GLOBALS["all_moves_dict"] = {m: i for i, m in enumerate(all_moves)}
    return (
        _GLOBALS["device"],
        _GLOBALS["elo_dict"],
        _GLOBALS["all_moves"],
        _GLOBALS["all_moves_dict"],
    )


def get_stockfish() -> chess.engine.SimpleEngine:
    with _GLOBAL_LOCK:
        if _GLOBALS["stockfish"] is None:
            stockfish_path = os.environ.get("STOCKFISH_PATH", "/opt/bin/stockfish")
            sf_threads = int(os.environ.get("STOCKFISH_THREADS", "1"))
            sf_hash_mb = int(os.environ.get("STOCKFISH_HASH_MB", "128"))
            sf_timeout_s = float(os.environ.get("STOCKFISH_TIMEOUT_S", "20.0"))
            _GLOBALS["stockfish"] = make_stockfish(
                stockfish_path,
                threads=sf_threads,
                hash_mb=sf_hash_mb,
                uci_elo=None,
                skill_level=None,
                timeout=sf_timeout_s,
            )
    return _GLOBALS["stockfish"]


def _load_policy_weights(model: torch.nn.Module, pt_path: Path) -> None:
    sd = load_state_dict_fuzzy(pt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing policy keys for %s: %s", pt_path, missing[:10])
    if unexpected:
        logger.warning("Unexpected policy keys for %s: %s", pt_path, unexpected[:10])


def _resolve_policy_path(model_root: Path, gm_name: str) -> Path:
    candidates = [
        model_root / "style_policy" / gm_name / "policy_dpo_best.pt",
        model_root / gm_name / "policy_dpo_best.pt",
        model_root / f"{gm_name}_policy_dpo_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find policy weights for gm_name={gm_name} under {model_root}"
    )


def _resolve_timer_path(model_root: Path, gm_name: str) -> Optional[Path]:
    candidates = [
        model_root / "timer_models" / gm_name / "timer_head_best.pt",
        model_root / gm_name / "timer_head_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    fallback = model_root / "timer_models" / "carlsen" / "timer_head_best.pt"
    if fallback.exists():
        logger.warning("No timer head checkpoint found for %s; using carlsen timer head fallback", gm_name)
        return fallback
    return None


def _canonical_gm_name(gm_name: str | None) -> str:
    return str(gm_name or "").strip().lower().replace("-", "_")


def _split_gm_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    names = [
        _canonical_gm_name(part)
        for part in re.split(r"[\s,]+", raw)
        if _canonical_gm_name(part)
    ]
    return list(dict.fromkeys(names))


def configured_gm_names() -> list[str]:
    names = _split_gm_names(
        os.environ.get("POLICY_ONLY_GM_NAMES")
        or os.environ.get("SERVE_GM_NAMES")
        or os.environ.get("GM_NAMES")
    )
    if names:
        return names
    return _split_gm_names(
        os.environ.get("POLICY_ONLY_GM_NAME")
        or os.environ.get("SERVE_GM_NAME")
        or os.environ.get("GM_NAME")
        or "carlsen"
    )


def default_gm_name() -> str:
    names = configured_gm_names()
    return names[0] if names else "carlsen"


def _infer_gm_name_from_game_type(game_type_id: str) -> str | None:
    if game_type_id.startswith("gm_") and "_" in game_type_id[3:]:
        return game_type_id.split("_", 2)[1]
    return None


def resolve_profile(game_type_id: str, gm_name: str | None = None) -> EngineProfile:
    configured_names = configured_gm_names()
    selected_gm = _canonical_gm_name(gm_name) or _canonical_gm_name(_infer_gm_name_from_game_type(game_type_id))
    if not selected_gm:
        selected_gm = default_gm_name()
    if configured_names and selected_gm not in configured_names:
        allowed = ", ".join(configured_names)
        raise ValueError(f"gm_name={selected_gm} is not available in this container; allowed: {allowed}")
    maia_type = "rapid" if "rapid" in game_type_id else "blitz"
    return EngineProfile(gm_name=selected_gm, maia_type=maia_type)


def get_or_load_bundle(profile: EngineProfile) -> ModelBundle:
    key = f"{profile.gm_name}::{profile.maia_type}"
    with _GLOBAL_LOCK:
        cached = _GLOBALS["models"].get(key)
        if cached is not None:
            _GLOBALS["models"].move_to_end(key)
            return cached

    device, elo_dict, _all_moves, all_moves_dict = get_global_context()
    policy = maia_model.from_pretrained(type=profile.maia_type, device=str(device)).to(device).eval()
    model_root = Path(os.environ.get("MODEL_ROOT", "/opt/models"))
    policy_path = _resolve_policy_path(model_root, profile.gm_name)
    _load_policy_weights(policy, policy_path)

    timer_cfg = TimerConfig()
    policy_runtime = PolicyTimerRuntime(
        maia=policy,
        all_moves_dict=all_moves_dict,
        elo_dict=elo_dict,
        device=device,
        timer_cfg=timer_cfg,
    )

    timer_head = None
    timer_feat_dim = None
    timer_path = _resolve_timer_path(model_root, profile.gm_name)
    if timer_path is not None:
        try:
            timer_head, timer_feat_dim = build_timer_head_from_ckpt(timer_path, device)
            logger.info("Loaded timer head for %s from %s", profile.gm_name, timer_path)
        except Exception as exc:
            logger.warning("Failed to load timer head for %s: %s", profile.gm_name, exc)
    else:
        logger.warning("No timer head checkpoint found for %s under %s", profile.gm_name, model_root)

    bundle = ModelBundle(
        profile=profile,
        policy_runtime=policy_runtime,
        timer_head=timer_head,
        timer_feat_dim=timer_feat_dim,
        stockfish=get_stockfish(),
    )
    with _GLOBAL_LOCK:
        _GLOBALS["models"][key] = bundle
        cache_max_raw = os.environ.get("POLICY_ONLY_MODEL_CACHE_MAX")
        if cache_max_raw is not None:
            try:
                cache_max = max(1, int(cache_max_raw))
            except ValueError:
                logger.warning("Invalid POLICY_ONLY_MODEL_CACHE_MAX=%r; using configured GM count", cache_max_raw)
                cache_max = max(1, len(configured_gm_names()) * 2)
        else:
            cache_max = max(1, len(configured_gm_names()) * 2)
        while len(_GLOBALS["models"]) > cache_max:
            evicted_key, _evicted_bundle = _GLOBALS["models"].popitem(last=False)
            logger.info("Evicted policy model bundle from cache: %s", evicted_key)
    return bundle


def _predict_think_ms(
    *,
    timer_head: TimerHead,
    timer_cfg: TimerConfig,
    feats: torch.Tensor,
    ply_idx: int,
    side_is_white: bool,
    prev5_ms: list[int],
    prev_clock_w_ms: int,
    prev_clock_b_ms: int,
) -> int:
    device = feats.device
    prev5 = [float(x) for x in prev5_ms[-5:]]
    while len(prev5) < 5:
        prev5.insert(0, 0.0)
    prev5_t = torch.tensor([prev5], device=device, dtype=torch.float32)
    prev5_feat = torch.log1p(torch.clamp(prev5_t, min=0.0))
    side = torch.tensor([1 if side_is_white else 0], device=device, dtype=torch.long)
    cw = torch.tensor([float(prev_clock_w_ms)], device=device)
    cb = torch.tensor([float(prev_clock_b_ms)], device=device)
    clock_left_ms = torch.where(side == 1, cw, cb).clamp(min=0.0)
    clock_feat = torch.log1p(clock_left_ms).unsqueeze(-1)
    ply = torch.tensor([float(ply_idx)], device=device).unsqueeze(-1)
    ply_feat = (ply / float(timer_cfg.ply_norm)).clamp(min=0.0, max=10.0)
    x = torch.cat([feats, prev5_feat, clock_feat, ply_feat], dim=-1)
    pred_log = timer_head(x).squeeze(0)
    pred_ms = float(torch.expm1(pred_log).clamp(min=0.0).item())
    pred_ms_i = int(round(pred_ms))
    pred_ms_i = max(timer_cfg.min_think_ms, min(timer_cfg.max_think_ms, pred_ms_i))
    mover_clock = prev_clock_w_ms if side_is_white else prev_clock_b_ms
    pred_ms_i = min(pred_ms_i, max(timer_cfg.min_think_ms, mover_clock - timer_cfg.safety_ms))
    return max(timer_cfg.min_think_ms, pred_ms_i)


def _limit_from_engine_config(
    engine_config: EngineConfigRequest,
    predicted_think_ms: Optional[int],
) -> tuple[chess.engine.Limit, dict[str, Any], Optional[int]]:
    def _time_s(ms: int) -> float:
        return max(0.001, int(ms) / 1000.0)

    if engine_config.limit is not None:
        if engine_config.limit.type == "depth":
            depth_value = int(engine_config.limit.value)
            if predicted_think_ms is not None:
                return chess.engine.Limit(depth=depth_value, time=_time_s(predicted_think_ms)), {
                    "type": "depth+time_ms",
                    "depth": depth_value,
                    "time_ms": predicted_think_ms,
                    "time_source": "timer_head",
                }, predicted_think_ms
            return chess.engine.Limit(depth=depth_value), {
                "type": "depth",
                "value": depth_value,
            }, None
        if engine_config.limit.type == "nodes":
            return chess.engine.Limit(nodes=engine_config.limit.value), {
                "type": "nodes",
                "value": engine_config.limit.value,
            }, None
        value = max(1, int(engine_config.limit.value))
        return chess.engine.Limit(time=_time_s(value)), {
            "type": "time_ms",
            "value": value,
        }, value

    depth_value = engine_config.stockfish_tree_search_depth or engine_config.stockfish_engine_depth
    if depth_value is not None:
        depth_value = int(depth_value)
        if predicted_think_ms is not None:
            return chess.engine.Limit(depth=depth_value, time=_time_s(predicted_think_ms)), {
                "type": "depth+time_ms",
                "depth": depth_value,
                "time_ms": predicted_think_ms,
                "time_source": "timer_head",
            }, predicted_think_ms
        return chess.engine.Limit(depth=depth_value), {"type": "depth", "value": depth_value}, None

    if engine_config.stockfish_engine_nodes is not None:
        return chess.engine.Limit(nodes=engine_config.stockfish_engine_nodes), {
            "type": "nodes",
            "value": engine_config.stockfish_engine_nodes,
        }, None

    if engine_config.stockfish_max_time_ms is not None:
        time_ms = max(1, int(engine_config.stockfish_max_time_ms))
        return chess.engine.Limit(time=_time_s(time_ms)), {
            "type": "time_ms",
            "value": time_ms,
        }, time_ms

    if predicted_think_ms is not None:
        return chess.engine.Limit(time=_time_s(predicted_think_ms)), {
            "type": "time_ms",
            "value": predicted_think_ms,
            "source": "timer_head",
        }, predicted_think_ms

    return chess.engine.Limit(time=0.2), {"type": "time_ms", "value": 200, "source": "default"}, 200


@torch.no_grad()
def choose_bot_move(
    *,
    bundle: ModelBundle,
    fen: str,
    clock: ClockState,
    last_ply_times_ms: list[int],
    engine_config: EngineConfigRequest,
    start_fen: str | None = None,
    played_moves_uci: list[str] | None = None,
) -> BotMoveResult:
    device, elo_dict, _all_moves, all_moves_dict = get_global_context()
    _ = device, elo_dict
    board = chess.Board(fen)
    opening_book_result = _choose_book_move(
        gm_name=bundle.profile.gm_name,
        board=board,
        engine_config=engine_config,
        played_moves_uci=played_moves_uci,
    )
    if opening_book_result is not None:
        logger.info(
            "choose_bot_move_opening_book gm=%s fen_ply=%s branch=%s",
            bundle.profile.gm_name,
            fen_ply_abs(fen),
            opening_book_result.stockfish_metrics.get("opening_book_branch"),
        )
        return opening_book_result

    side_is_white = board.turn == chess.WHITE
    feats, masked_logits = bundle.policy_runtime.forward_once(
        fen,
        bundle.profile.elo_self,
        bundle.profile.elo_oppo,
    )
    predicted_think_ms = None
    if engine_config.use_timer_head and bundle.timer_head is not None:
        predicted_think_ms = _predict_think_ms(
            timer_head=bundle.timer_head,
            timer_cfg=bundle.policy_runtime.cfg,
            feats=feats,
            ply_idx=fen_ply_abs(fen),
            side_is_white=side_is_white,
            prev5_ms=last_ply_times_ms,
            prev_clock_w_ms=int(clock.white_ms or 0),
            prev_clock_b_ms=int(clock.black_ms or 0),
        )
    elif engine_config.use_timer_head:
        logger.warning(
            "Timer head requested but unavailable for gm=%s; falling back to non-timer engine limit",
            bundle.profile.gm_name,
        )

    engine_limit, engine_limit_dict, requested_think_ms = _limit_from_engine_config(
        engine_config,
        predicted_think_ms,
    )
    logger.info(
        "choose_bot_move gm=%s fen_ply=%s timer_requested=%s predicted_think_ms=%s engine_limit=%s",
        bundle.profile.gm_name,
        fen_ply_abs(fen),
        engine_config.use_timer_head,
        predicted_think_ms,
        engine_limit_dict,
    )
    position_counts = None
    if start_fen is not None:
        try:
            position_counts = build_position_counts(
                start_fen=start_fen,
                played_moves_uci=list(played_moves_uci or []),
            )
        except Exception:
            logger.exception(
                "choose_bot_move_failed_to_reconstruct_history gm=%s start_fen=%s move_count=%s",
                bundle.profile.gm_name,
                start_fen,
                len(list(played_moves_uci or [])),
            )

    started = time.time()
    infos = bundle.stockfish.analyse(
        board,
        engine_limit,
        multipv=max(1, int(engine_config.stockfish_multipv_topk)),
    )
    actual_think_ms = int(round((time.time() - started) * 1000.0))

    candidates: list[dict[str, Any]] = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        move = pv[0].uci()
        cp = _score_to_cp(score)
        rel = score.relative
        mate = rel.mate()
        pv_uci = [m.uci() for m in pv[:8]]
        candidates.append(
            {
                "uci": move,
                "cp": int(cp),
                "mate": int(mate) if mate is not None else None,
                "pv_uci": pv_uci,
                "multipv_rank": int(info.get("multipv")) if info.get("multipv") is not None else None,
                "depth": int(info.get("depth")) if info.get("depth") is not None else None,
                "seldepth": int(info.get("seldepth")) if info.get("seldepth") is not None else None,
                "nodes": int(info.get("nodes")) if info.get("nodes") is not None else None,
                "nps": int(info.get("nps")) if info.get("nps") is not None else None,
                "time_ms": int(round(float(info.get("time")) * 1000.0)) if info.get("time") is not None else None,
                "tbhits": int(info.get("tbhits")) if info.get("tbhits") is not None else None,
            }
        )

    if not candidates:
        fallback = next(iter(board.legal_moves))
        return BotMoveResult(
            move_uci=fallback.uci(),
            eval_cp=0,
            pv_uci=[fallback.uci()],
            candidate_moves=[
                {
                    "uci": fallback.uci(),
                    "cp": 0,
                    "mate": None,
                    "pv_uci": [fallback.uci()],
                    "multipv_rank": 1,
                    "in_cp_gap_window": True,
                    "prob": 1.0,
                    "depth": None,
                    "seldepth": None,
                    "nodes": None,
                    "nps": None,
                    "time_ms": actual_think_ms,
                    "tbhits": None,
                }
            ],
            stockfish_metrics={
                "requested_multipv_topk": max(1, int(engine_config.stockfish_multipv_topk)),
                "returned_candidate_count": 1,
                "cp_gap_window": engine_config.cp_gap_window,
                "max_depth": None,
                "max_seldepth": None,
                "total_nodes": None,
                "max_nps": None,
                "max_time_ms": actual_think_ms,
                "best_cp": 0,
                "best_move_uci": fallback.uci(),
                "best_move_mate": None,
                "selected_move_rank_by_cp": 1,
                "selected_move_rank_by_prob_within_window": 1,
            },
            selected_probability=1.0,
            requested_think_ms=requested_think_ms,
            actual_think_ms=actual_think_ms,
            engine_limit=engine_limit_dict,
        )

    best_cp = max(item["cp"] for item in candidates)
    best_candidate = max(candidates, key=lambda item: item["cp"])
    best_pv = best_candidate["pv_uci"]
    adjusted_candidates = [
        apply_draw_penalties_to_candidate(
            board,
            candidate,
            engine_config=engine_config,
            position_counts=position_counts,
        )
        for candidate in candidates
    ]
    best_adjusted_cp = max(int(item.get("adjusted_cp", item["cp"])) for item in adjusted_candidates)
    max_multipv_rank = max((int(item.get("multipv_rank") or 1) for item in adjusted_candidates), default=1)
    phase = game_phase_from_ply_abs(fen_ply_abs(fen))
    adjusted_candidates = [
        {
            **candidate,
            **_candidate_mood_scores(
                board,
                candidate,
                engine_config=engine_config,
                phase=phase,
                best_adjusted_cp=best_adjusted_cp,
                cp_gap_window=engine_config.cp_gap_window,
                max_rank=max_multipv_rank,
            ),
        }
        for candidate in adjusted_candidates
    ]
    kept = adjusted_candidates
    if engine_config.cp_gap_window is not None:
        kept = [
            item
            for item in adjusted_candidates
            if int(item.get("adjusted_cp", item["cp"])) >= best_adjusted_cp - int(engine_config.cp_gap_window)
        ]
        if not kept:
            kept = adjusted_candidates
    kept = _apply_weird_move_filters(
        kept=kept,
        engine_config=engine_config,
        fen=fen,
        phase=phase,
        best_adjusted_cp=best_adjusted_cp,
    )

    style_temp = max(1e-6, float(engine_config.style_temperature))
    logp_all = torch.log_softmax(masked_logits[0] / style_temp, dim=-1)
    beta_engine = (
        float(engine_config.beta_engine)
        if engine_config.beta_engine is not None
        else (1.0 if engine_config.use_gibbs else 0.0)
    )
    alpha_style = float(engine_config.alpha_style)
    engine_temp = max(1e-6, float(engine_config.engine_temp))
    score_terms: list[torch.Tensor] = []
    serializable_candidates: list[dict[str, Any]] = []
    for candidate in kept:
        move_uci = str(candidate["uci"])
        cp = int(candidate.get("adjusted_cp", candidate["cp"]))
        idx = uci_to_vocab_index(all_moves_dict, fen, move_uci)
        if idx < 0:
            base_term = torch.tensor(torch.finfo(logp_all.dtype).min, device=logp_all.device)
        else:
            base_term = logp_all[idx]
        capped_cp = max(-int(engine_config.cp_cap), min(int(engine_config.cp_cap), int(cp)))
        engine_term = torch.tensor(
            float(capped_cp) / max(float(engine_config.cp_scale), 1e-6) / engine_temp,
            device=logp_all.device,
            dtype=logp_all.dtype,
        )
        mood_term = torch.tensor(
            (float(engine_config.novelty_weight) * float(candidate.get("novelty_score") or 0.0))
            + (float(engine_config.risk_weight) * float(candidate.get("risk_score") or 0.0))
            + (float(engine_config.attack_weight) * float(candidate.get("attack_score") or 0.0)),
            device=logp_all.device,
            dtype=logp_all.dtype,
        )
        score_term = (alpha_style * base_term) + (beta_engine * engine_term) + mood_term
        score_terms.append(score_term)
        serializable_candidates.append({**candidate, "in_cp_gap_window": True})

    probs_t = torch.softmax(torch.stack(score_terms, dim=0), dim=0)
    probs = probs_t.detach().cpu().tolist()
    for candidate, prob in zip(serializable_candidates, probs):
        candidate["prob"] = float(prob)

    kept_by_uci = {candidate["uci"]: candidate for candidate in serializable_candidates}
    all_candidate_moves: list[dict[str, Any]] = []
    for candidate in adjusted_candidates:
        move_uci = str(candidate["uci"])
        kept_candidate = kept_by_uci.get(move_uci)
        if kept_candidate is not None:
            all_candidate_moves.append(kept_candidate)
            continue
        all_candidate_moves.append({**candidate, "in_cp_gap_window": False, "prob": None})

    if engine_config.sample:
        rng = random.Random(int(engine_config.random_seed))
        cutoff = rng.random()
        total = 0.0
        chosen_idx = 0
        for idx, prob in enumerate(probs):
            total += float(prob)
            if cutoff <= total:
                chosen_idx = idx
                break
    else:
        chosen_idx = int(torch.argmax(probs_t).item())

    chosen_move = serializable_candidates[chosen_idx]["uci"]
    chosen_prob = float(probs[chosen_idx]) if probs else None
    rank_by_cp = next(
        (rank for rank, candidate in enumerate(sorted(candidates, key=lambda item: item["cp"], reverse=True), start=1) if candidate["uci"] == chosen_move),
        None,
    )
    rank_by_prob = next(
        (rank for rank, candidate in enumerate(sorted(serializable_candidates, key=lambda item: float(item.get("prob") or 0.0), reverse=True), start=1) if candidate["uci"] == chosen_move),
        None,
    )
    stockfish_metrics = {
        "requested_multipv_topk": max(1, int(engine_config.stockfish_multipv_topk)),
        "returned_candidate_count": len(candidates),
        "cp_gap_window": engine_config.cp_gap_window,
        "max_depth": max((candidate["depth"] for candidate in candidates if candidate.get("depth") is not None), default=None),
        "max_seldepth": max((candidate["seldepth"] for candidate in candidates if candidate.get("seldepth") is not None), default=None),
        "total_nodes": sum(candidate["nodes"] for candidate in candidates if candidate.get("nodes") is not None) or None,
        "max_nps": max((candidate["nps"] for candidate in candidates if candidate.get("nps") is not None), default=None),
        "max_time_ms": max((candidate["time_ms"] for candidate in candidates if candidate.get("time_ms") is not None), default=actual_think_ms),
        "best_cp": int(best_cp),
        "best_move_uci": str(best_candidate["uci"]),
        "best_move_mate": best_candidate.get("mate"),
        "selected_move_rank_by_cp": rank_by_cp,
        "selected_move_rank_by_prob_within_window": rank_by_prob,
    }
    return BotMoveResult(
        move_uci=chosen_move,
        eval_cp=int(best_cp),
        pv_uci=best_pv,
        candidate_moves=all_candidate_moves,
        stockfish_metrics=stockfish_metrics,
        selected_probability=chosen_prob,
        requested_think_ms=requested_think_ms,
        actual_think_ms=actual_think_ms,
        engine_limit=engine_limit_dict,
    )
