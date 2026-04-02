from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass
import chess
import itertools
import math

import torch

from grandmaster_dpo.eval.single_gm.shared_eval_metric_utilities import run_eval


# ----------------------------
# Helpers (match training)
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
        #f"[WARN] move {uci} not found in sf_moves_returned "
        #f"(game={meta.get('game_header_hash')}, ply={meta.get('ply_idx')}). "
        #f"Using fallback cp={fallback_cp}"
    #)
    return fallback_cp

def infer_phase_from_ply(ply_idx: int) -> str:
    if ply_idx <= 20:
        return "opening"
    if ply_idx <= 50:
        return "middlegame"
    return "endgame"


def phase_weights(phase: str) -> Dict[str, float]:
    """
    Base weights for local similarity terms before context adaptation.
    """
    if phase == "opening":
        return {
            "cp": 0.90,
            "forcing": 0.70,
            "centralization": 1.10,
            "distance": 0.70,
            "activity": 0.80,
            "king_pressure": 0.50,
            "mobility": 0.60,
            "opp_reply": 0.60,
            "pawn_break": 1.10,
            "development": 1.40,
            "castle": 1.30,
            "region": 0.90,
            "side": 0.60,
            "retreat": 0.50,
            "quiet": 0.60,
            "piece": 0.20,
        }
    if phase == "middlegame":
        return {
            "cp": 1.00,
            "forcing": 1.30,
            "centralization": 0.90,
            "distance": 0.80,
            "activity": 1.10,
            "king_pressure": 1.20,
            "mobility": 0.90,
            "opp_reply": 1.00,
            "pawn_break": 1.20,
            "development": 0.30,
            "castle": 0.20,
            "region": 0.80,
            "side": 0.70,
            "retreat": 0.60,
            "quiet": 0.60,
            "piece": 0.15,
        }
    return {
        "cp": 1.00,
        "forcing": 0.50,
        "centralization": 1.20,
        "distance": 0.90,
        "activity": 1.00,
        "king_pressure": 0.40,
        "mobility": 1.20,
        "opp_reply": 0.70,
        "pawn_break": 0.40,
        "development": 0.00,
        "castle": 0.00,
        "region": 0.60,
        "side": 0.50,
        "retreat": 0.70,
        "quiet": 0.90,
        "piece": 0.10,
    }


# ============================================================
# Geometry / board helpers
# ============================================================

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXT_CENTER_SQUARES = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}


def normalize_similarity(diff: float, scale: float) -> float:
    return math.exp(-abs(diff) / max(scale, 1e-6))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def square_region(square: chess.Square) -> str:
    if square in CENTER_SQUARES:
        return "center"
    if square in EXT_CENTER_SQUARES:
        return "extended_center"
    return "flank"


def file_group(square: chess.Square) -> str:
    f = chess.square_file(square)
    if f <= 2:
        return "queenside"
    if f >= 5:
        return "kingside"
    return "center"


def manhattan_distance(sq1: chess.Square, sq2: chess.Square) -> int:
    f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
    f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
    return abs(f1 - f2) + abs(r1 - r2)


def centrality(square: chess.Square) -> float:
    f = chess.square_file(square)
    r = chess.square_rank(square)
    return -abs(f - 3.5) - abs(r - 3.5)


def move_piece_type(board: chess.Board, move: chess.Move) -> Optional[int]:
    piece = board.piece_at(move.from_square)
    return None if piece is None else piece.piece_type


def is_castle(move: chess.Move) -> bool:
    return move in [
        chess.Move.from_uci("e1g1"),
        chess.Move.from_uci("e1c1"),
        chess.Move.from_uci("e8g8"),
        chess.Move.from_uci("e8c8"),
    ]


def is_positional(board: chess.Board, move: chess.Move) -> bool:
    if move.promotion is not None:
        return False
    if board.is_capture(move):
        return False
    if board.gives_check(move):
        return False
    return True


def is_development_move(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False
    if piece.piece_type not in (chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.ROOK):
        return False

    home_rank = 0 if piece.color == chess.WHITE else 7
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    return from_rank == home_rank and to_rank != home_rank


def is_retreat(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False
    color = piece.color
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    return (to_rank < from_rank) if color == chess.WHITE else (to_rank > from_rank)


def opens_or_advances_center_pawn(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    return from_file in {2, 3, 4, 5} or to_file in {2, 3, 4, 5}


def is_pawn_break_like(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False

    if board.is_capture(move):
        return True

    b = board.copy(stack=False)
    b.push(move)
    sq = move.to_square
    color = piece.color
    enemy = not color

    for df in (-1, 1):
        f = chess.square_file(sq) + df
        r = chess.square_rank(sq)
        if 0 <= f < 8:
            adj_sq = chess.square(f, r)
            p = b.piece_at(adj_sq)
            if p is not None and p.color == enemy and p.piece_type == chess.PAWN:
                return True

    return opens_or_advances_center_pawn(board, move)


def legal_mobility(board: chess.Board) -> int:
    return sum(1 for _ in board.legal_moves)


def legal_mobility_after(board: chess.Board, move: chess.Move) -> int:
    b = board.copy(stack=False)
    b.push(move)
    return legal_mobility(b)


def attacked_squares_by_moved_piece_after(board: chess.Board, move: chess.Move) -> int:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0

    b = board.copy(stack=False)
    b.push(move)
    moved_piece = b.piece_at(move.to_square)
    if moved_piece is None:
        return 0

    return len(b.attacks(move.to_square))


def king_zone_pressure_after(board: chess.Board, move: chess.Move) -> int:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0

    b = board.copy(stack=False)
    b.push(move)
    enemy_king_sq = b.king(not piece.color)
    if enemy_king_sq is None:
        return 0

    king_zone = {enemy_king_sq} | set(b.attacks(enemy_king_sq))
    moved_piece = b.piece_at(move.to_square)
    if moved_piece is None:
        return 0

    attacked = set(b.attacks(move.to_square))
    return len(attacked & king_zone)


def centralization_gain(board: chess.Board, move: chess.Move) -> float:
    return centrality(move.to_square) - centrality(move.from_square)


def opponent_forcing_reply_count(board: chess.Board, move: chess.Move) -> int:
    b = board.copy(stack=False)
    b.push(move)
    cnt = 0
    for mv in b.legal_moves:
        if b.is_capture(mv) or b.gives_check(mv):
            cnt += 1
    return cnt


# ============================================================
# Board-summary helpers for prev/next fen windows
# ============================================================

def count_legal_captures(board: chess.Board) -> int:
    return sum(1 for mv in board.legal_moves if board.is_capture(mv))


def count_legal_checks(board: chess.Board) -> int:
    return sum(1 for mv in board.legal_moves if board.gives_check(mv))


def center_occupancy(board: chess.Board) -> int:
    return sum(1 for sq in CENTER_SQUARES if board.piece_at(sq) is not None)


def ext_center_occupancy(board: chess.Board) -> int:
    return sum(1 for sq in EXT_CENTER_SQUARES if board.piece_at(sq) is not None)


def center_control(board: chess.Board, color: chess.Color) -> int:
    return sum(1 for sq in CENTER_SQUARES if board.is_attacked_by(color, sq))


def count_open_files(board: chess.Board) -> int:
    c = 0
    for f in range(8):
        has_pawn = False
        for r in range(8):
            p = board.piece_at(chess.square(f, r))
            if p is not None and p.piece_type == chess.PAWN:
                has_pawn = True
                break
        if not has_pawn:
            c += 1
    return c


def count_semi_open_files(board: chess.Board, color: chess.Color) -> int:
    c = 0
    enemy = not color
    for f in range(8):
        own_pawn = False
        enemy_pawn = False
        for r in range(8):
            p = board.piece_at(chess.square(f, r))
            if p is None or p.piece_type != chess.PAWN:
                continue
            if p.color == color:
                own_pawn = True
            elif p.color == enemy:
                enemy_pawn = True
        if (not own_pawn) and enemy_pawn:
            c += 1
    return c


def total_non_pawn_material(board: chess.Board) -> int:
    values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    s = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None:
            s += values.get(p.piece_type, 0)
    return s


def has_queen(board: chess.Board, color: chess.Color) -> int:
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None and p.color == color and p.piece_type == chess.QUEEN:
            return 1
    return 0


def is_castled_position(board: chess.Board, color: chess.Color) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    if color == chess.WHITE:
        return 1 if king_sq in (chess.G1, chess.C1) else 0
    return 1 if king_sq in (chess.G8, chess.C8) else 0


def king_zone_pressure_board(board: chess.Board, attacker_color: chess.Color) -> int:
    enemy_king_sq = board.king(not attacker_color)
    if enemy_king_sq is None:
        return 0
    zone = {enemy_king_sq} | set(board.attacks(enemy_king_sq))
    total = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None and p.color == attacker_color:
            total += len(set(board.attacks(sq)) & zone)
    return total


def unfinished_development_proxy(board: chess.Board, color: chess.Color) -> int:
    """
    Larger means more undeveloped.
    """
    count = 0
    home_rank = 0 if color == chess.WHITE else 7
    home_knights = [chess.B1, chess.G1] if color == chess.WHITE else [chess.B8, chess.G8]
    home_bishops = [chess.C1, chess.F1] if color == chess.WHITE else [chess.C8, chess.F8]
    home_rooks = [chess.A1, chess.H1] if color == chess.WHITE else [chess.A8, chess.H8]
    home_queen = chess.D1 if color == chess.WHITE else chess.D8

    for sq in home_knights + home_bishops + home_rooks + [home_queen]:
        p = board.piece_at(sq)
        if p is not None and p.color == color:
            count += 1

    king_sq = board.king(color)
    if king_sq is not None and chess.square_rank(king_sq) == home_rank and not is_castled_position(board, color):
        count += 1

    return count


@dataclass
class BoardSummary:
    mobility: float
    legal_captures: float
    legal_checks: float
    center_occ: float
    ext_center_occ: float
    own_center_control: float
    enemy_center_control: float
    own_king_pressure: float
    enemy_king_pressure: float
    open_files: float
    semi_open_files: float
    non_pawn_material: float
    own_has_queen: float
    enemy_has_queen: float
    own_castled: float
    enemy_castled: float
    own_unfinished_dev: float
    enemy_unfinished_dev: float


def summarize_single_board(board: chess.Board, perspective_color: chess.Color) -> BoardSummary:
    return BoardSummary(
        mobility=float(legal_mobility(board)),
        legal_captures=float(count_legal_captures(board)),
        legal_checks=float(count_legal_checks(board)),
        center_occ=float(center_occupancy(board)),
        ext_center_occ=float(ext_center_occupancy(board)),
        own_center_control=float(center_control(board, perspective_color)),
        enemy_center_control=float(center_control(board, not perspective_color)),
        own_king_pressure=float(king_zone_pressure_board(board, perspective_color)),
        enemy_king_pressure=float(king_zone_pressure_board(board, not perspective_color)),
        open_files=float(count_open_files(board)),
        semi_open_files=float(count_semi_open_files(board, perspective_color)),
        non_pawn_material=float(total_non_pawn_material(board)),
        own_has_queen=float(has_queen(board, perspective_color)),
        enemy_has_queen=float(has_queen(board, not perspective_color)),
        own_castled=float(is_castled_position(board, perspective_color)),
        enemy_castled=float(is_castled_position(board, not perspective_color)),
        own_unfinished_dev=float(unfinished_development_proxy(board, perspective_color)),
        enemy_unfinished_dev=float(unfinished_development_proxy(board, not perspective_color)),
    )


@dataclass
class WindowSummary:
    mean_mobility: float
    mean_legal_captures: float
    mean_legal_checks: float
    mean_center_occ: float
    mean_ext_center_occ: float
    mean_own_center_control: float
    mean_enemy_center_control: float
    mean_own_king_pressure: float
    mean_enemy_king_pressure: float
    mean_open_files: float
    mean_semi_open_files: float
    mean_non_pawn_material: float
    mean_own_has_queen: float
    mean_enemy_has_queen: float
    mean_own_castled: float
    mean_enemy_castled: float
    mean_own_unfinished_dev: float
    mean_enemy_unfinished_dev: float
    trend_mobility: float
    trend_legal_captures: float
    trend_legal_checks: float
    trend_center_occ: float
    trend_own_king_pressure: float
    trend_enemy_king_pressure: float
    trend_non_pawn_material: float


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def summarize_fen_window(
    fens: Optional[Sequence[Optional[str]]],
    perspective_color: chess.Color,
) -> Optional[WindowSummary]:
    if not fens:
        return None

    boards: List[chess.Board] = []
    for fen in fens:
        if fen is None:
            continue
        try:
            boards.append(chess.Board(fen))
        except Exception:
            continue

    if not boards:
        return None

    rows = [summarize_single_board(b, perspective_color) for b in boards]

    def collect(name: str) -> List[float]:
        return [getattr(r, name) for r in rows]

    def trend(name: str) -> float:
        vals = collect(name)
        if len(vals) <= 1:
            return 0.0
        return vals[-1] - vals[0]

    return WindowSummary(
        mean_mobility=_mean(collect("mobility")),
        mean_legal_captures=_mean(collect("legal_captures")),
        mean_legal_checks=_mean(collect("legal_checks")),
        mean_center_occ=_mean(collect("center_occ")),
        mean_ext_center_occ=_mean(collect("ext_center_occ")),
        mean_own_center_control=_mean(collect("own_center_control")),
        mean_enemy_center_control=_mean(collect("enemy_center_control")),
        mean_own_king_pressure=_mean(collect("own_king_pressure")),
        mean_enemy_king_pressure=_mean(collect("enemy_king_pressure")),
        mean_open_files=_mean(collect("open_files")),
        mean_semi_open_files=_mean(collect("semi_open_files")),
        mean_non_pawn_material=_mean(collect("non_pawn_material")),
        mean_own_has_queen=_mean(collect("own_has_queen")),
        mean_enemy_has_queen=_mean(collect("enemy_has_queen")),
        mean_own_castled=_mean(collect("own_castled")),
        mean_enemy_castled=_mean(collect("enemy_castled")),
        mean_own_unfinished_dev=_mean(collect("own_unfinished_dev")),
        mean_enemy_unfinished_dev=_mean(collect("enemy_unfinished_dev")),
        trend_mobility=trend("mobility"),
        trend_legal_captures=trend("legal_captures"),
        trend_legal_checks=trend("legal_checks"),
        trend_center_occ=trend("center_occ"),
        trend_own_king_pressure=trend("own_king_pressure"),
        trend_enemy_king_pressure=trend("enemy_king_pressure"),
        trend_non_pawn_material=trend("non_pawn_material"),
    )


# ============================================================
# Move-level feature extraction
# ============================================================

@dataclass
class MoveStyleFeatures:
    piece_type: Optional[int]
    is_quiet: bool
    is_capture: bool
    gives_check: bool
    is_castle: bool
    is_development: bool
    is_retreat: bool
    is_pawn_break: bool
    region_to: str
    file_group_to: str
    move_distance: int
    moved_piece_activity: int
    king_pressure: int
    mobility_after: int
    centralization_gain: float
    opponent_forcing_replies: int
    forcingness: float


def extract_style_features(board: chess.Board, move: chess.Move) -> MoveStyleFeatures:
    is_capture = board.is_capture(move)
    gives_check = board.gives_check(move)
    is_pawn_break = is_pawn_break_like(board, move)
    king_pressure = king_zone_pressure_after(board, move)

    forcingness = (
        1.5 * float(is_capture) +
        1.5 * float(gives_check) +
        1.0 * float(is_pawn_break) +
        0.5 * float(king_pressure >= 2)
    )

    return MoveStyleFeatures(
        piece_type=move_piece_type(board, move),
        is_quiet=is_positional(board, move),
        is_capture=is_capture,
        gives_check=gives_check,
        is_castle=is_castle(move),
        is_development=is_development_move(board, move),
        is_retreat=is_retreat(board, move),
        is_pawn_break=is_pawn_break,
        region_to=square_region(move.to_square),
        file_group_to=file_group(move.to_square),
        move_distance=manhattan_distance(move.from_square, move.to_square),
        moved_piece_activity=attacked_squares_by_moved_piece_after(board, move),
        king_pressure=king_pressure,
        mobility_after=legal_mobility_after(board, move),
        centralization_gain=centralization_gain(board, move),
        opponent_forcing_replies=opponent_forcing_reply_count(board, move),
        forcingness=forcingness,
    )


# ============================================================
# Adaptive weight modulation from prev_fens
# ============================================================

def adapt_weights_from_history(
    base_weights: Dict[str, float],
    hist: Optional[WindowSummary],
    phase: str,
) -> Dict[str, float]:
    w = dict(base_weights)

    if hist is None:
        return w

    sharpness = hist.mean_legal_captures + 1.5 * hist.mean_legal_checks + 0.25 * hist.mean_own_king_pressure
    quietness = hist.mean_mobility - 0.8 * hist.mean_legal_captures - hist.mean_legal_checks
    dev_need = hist.mean_own_unfinished_dev - 0.7 * hist.mean_own_castled
    simplification = -hist.trend_non_pawn_material
    center_focus = hist.mean_center_occ + 0.5 * hist.mean_ext_center_occ + 0.3 * hist.mean_own_center_control

    sharp_gate = clamp01(sharpness / 8.0)
    quiet_gate = clamp01((quietness - 20.0) / 15.0)
    dev_gate = clamp01(dev_need / 4.0)
    simplify_gate = clamp01(simplification / 6.0)
    center_gate = clamp01(center_focus / 8.0)

    w["forcing"] *= (1.0 + 0.50 * sharp_gate)
    w["king_pressure"] *= (1.0 + 0.55 * sharp_gate)
    w["pawn_break"] *= (1.0 + 0.35 * sharp_gate)
    w["opp_reply"] *= (1.0 + 0.40 * sharp_gate)

    w["centralization"] *= (1.0 + 0.35 * quiet_gate + 0.25 * center_gate)
    w["activity"] *= (1.0 + 0.30 * quiet_gate)
    w["quiet"] *= (1.0 + 0.25 * quiet_gate)

    w["development"] *= (1.0 + 0.80 * dev_gate)
    w["castle"] *= (1.0 + 0.90 * dev_gate)

    w["mobility"] *= (1.0 + 0.30 * simplify_gate)
    w["retreat"] *= (1.0 + 0.20 * simplify_gate)

    if phase == "endgame":
        w["development"] = 0.0
        w["castle"] = 0.0

    return w


# ============================================================
# Future-window similarity
# ============================================================

def future_window_similarity(
    next_fens_chosen: Optional[Sequence[Optional[str]]],
    next_fens_rejected: Optional[Sequence[Optional[str]]],
    perspective_color: chess.Color,
) -> Optional[float]:
    ch = summarize_fen_window(next_fens_chosen, perspective_color)
    rj = summarize_fen_window(next_fens_rejected, perspective_color)
    if ch is None or rj is None:
        return None

    sims = [
        (normalize_similarity(ch.mean_mobility - rj.mean_mobility, 8.0), 1.0),
        (normalize_similarity(ch.mean_legal_captures - rj.mean_legal_captures, 3.0), 1.2),
        (normalize_similarity(ch.mean_legal_checks - rj.mean_legal_checks, 2.0), 1.1),
        (normalize_similarity(ch.mean_center_occ - rj.mean_center_occ, 2.0), 0.9),
        (normalize_similarity(ch.mean_own_center_control - rj.mean_own_center_control, 3.0), 0.8),
        (normalize_similarity(ch.mean_own_king_pressure - rj.mean_own_king_pressure, 4.0), 1.2),
        (normalize_similarity(ch.mean_enemy_king_pressure - rj.mean_enemy_king_pressure, 4.0), 0.7),
        (normalize_similarity(ch.mean_open_files - rj.mean_open_files, 2.0), 0.5),
        (normalize_similarity(ch.mean_non_pawn_material - rj.mean_non_pawn_material, 6.0), 1.0),
        (normalize_similarity(ch.mean_own_has_queen - rj.mean_own_has_queen, 0.7), 0.6),
        (normalize_similarity(ch.mean_enemy_has_queen - rj.mean_enemy_has_queen, 0.7), 0.5),
        (normalize_similarity(ch.trend_mobility - rj.trend_mobility, 6.0), 0.7),
        (normalize_similarity(ch.trend_legal_captures - rj.trend_legal_captures, 2.0), 0.9),
        (normalize_similarity(ch.trend_legal_checks - rj.trend_legal_checks, 1.5), 0.8),
        (normalize_similarity(ch.trend_center_occ - rj.trend_center_occ, 1.5), 0.7),
        (normalize_similarity(ch.trend_own_king_pressure - rj.trend_own_king_pressure, 3.0), 1.0),
        (normalize_similarity(ch.trend_non_pawn_material - rj.trend_non_pawn_material, 4.0), 0.9),
    ]

    num = sum(v * w for v, w in sims)
    den = sum(w for _, w in sims) + 1e-12
    return num / den


# ============================================================
# Main style score
# ============================================================

def compute_style_score_v2(
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    chosen_cp: float,
    rejected_cp: float,
    prev_fens: Optional[Sequence[Optional[str]]] = None,
    next_fens_chosen: Optional[Sequence[Optional[str]]] = None,
    next_fens_rejected: Optional[Sequence[Optional[str]]] = None,
    ply_idx: Optional[int] = None,
    phase: Optional[str] = None,
    cp_scale: float = 35.0,
    activity_scale: float = 3.0,
    mobility_scale: float = 6.0,
    distance_scale: float = 2.5,
    centralization_scale: float = 1.0,
    opp_reply_scale: float = 3.0,
) -> float:
    """
    Higher score => chosen/rejected are MORE stylistically similar,
    so the DPO loss should downweight this pair more.

    prev_fens:
        Recent historical positions BEFORE the current fen.
        These are used to adapt feature weights contextually.

    next_fens_chosen / next_fens_rejected:
        Future trajectories after each candidate move.
        These are used to compare trajectory-style similarity directly.

    Notes:
    - This is offline-only style labeling, which is fine for DPO weighting.
    - At inference, the model still only consumes the current board position.
    """
    board = chess.Board(fen)
    ch = chess.Move.from_uci(chosen_uci)
    rj = chess.Move.from_uci(rejected_uci)

    if phase is None:
        phase = infer_phase_from_ply(ply_idx if ply_idx is not None else board.fullmove_number * 2)

    perspective_color = board.turn

    base_w = phase_weights(phase)
    hist_summary = summarize_fen_window(prev_fens, perspective_color)
    w = adapt_weights_from_history(base_w, hist_summary, phase)

    f_ch = extract_style_features(board, ch)
    f_rj = extract_style_features(board, rj)

    # ----- local smooth similarities -----
    cp_sim = normalize_similarity(float(chosen_cp) - float(rejected_cp), cp_scale)
    forcing_sim = normalize_similarity(f_ch.forcingness - f_rj.forcingness, 1.5)
    centralization_sim = normalize_similarity(
        f_ch.centralization_gain - f_rj.centralization_gain,
        centralization_scale,
    )
    distance_sim = normalize_similarity(
        f_ch.move_distance - f_rj.move_distance,
        distance_scale,
    )
    activity_sim = normalize_similarity(
        f_ch.moved_piece_activity - f_rj.moved_piece_activity,
        activity_scale,
    )
    king_pressure_sim = normalize_similarity(
        f_ch.king_pressure - f_rj.king_pressure,
        activity_scale,
    )
    mobility_sim = normalize_similarity(
        f_ch.mobility_after - f_rj.mobility_after,
        mobility_scale,
    )
    opp_reply_sim = normalize_similarity(
        f_ch.opponent_forcing_replies - f_rj.opponent_forcing_replies,
        opp_reply_scale,
    )

    # ----- binary / low-cardinality similarities -----
    pawn_break_sim = 1.0 if f_ch.is_pawn_break == f_rj.is_pawn_break else 0.0
    development_sim = 1.0 if f_ch.is_development == f_rj.is_development else 0.0
    castle_sim = 1.0 if f_ch.is_castle == f_rj.is_castle else 0.0
    region_sim = 1.0 if f_ch.region_to == f_rj.region_to else 0.0
    side_sim = 1.0 if f_ch.file_group_to == f_rj.file_group_to else 0.0
    retreat_sim = 1.0 if f_ch.is_retreat == f_rj.is_retreat else 0.0
    quiet_sim = 1.0 if f_ch.is_quiet == f_rj.is_quiet else 0.0
    piece_sim = 1.0 if f_ch.piece_type == f_rj.piece_type else 0.0

    local_parts = [
        (cp_sim, w["cp"]),
        (forcing_sim, w["forcing"]),
        (centralization_sim, w["centralization"]),
        (distance_sim, w["distance"]),
        (activity_sim, w["activity"]),
        (king_pressure_sim, w["king_pressure"]),
        (mobility_sim, w["mobility"]),
        (opp_reply_sim, w["opp_reply"]),
        (pawn_break_sim, w["pawn_break"]),
        (development_sim, w["development"]),
        (castle_sim, w["castle"]),
        (region_sim, w["region"]),
        (side_sim, w["side"]),
        (retreat_sim, w["retreat"]),
        (quiet_sim, w["quiet"]),
        (piece_sim, w["piece"]),
    ]
    local_num = sum(v * ww for v, ww in local_parts)
    local_den = sum(ww for _, ww in local_parts) + 1e-12
    local_sim = local_num / local_den

    # ----- future trajectory similarity -----
    fut_sim = future_window_similarity(next_fens_chosen, next_fens_rejected, perspective_color)

    # ----- blend local and future by phase -----
    if fut_sim is None:
        sim = local_sim
    else:
        if phase == "opening":
            alpha_local = 0.65
        elif phase == "middlegame":
            alpha_local = 0.45
        else:
            alpha_local = 0.60
        sim = alpha_local * local_sim + (1.0 - alpha_local) * fut_sim

    return float(max(sim, 1e-6))


# ============================================================
# Optional debug version
# ============================================================

def compute_style_score_v2_debug(
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    chosen_cp: float,
    rejected_cp: float,
    prev_fens: Optional[Sequence[Optional[str]]] = None,
    next_fens_chosen: Optional[Sequence[Optional[str]]] = None,
    next_fens_rejected: Optional[Sequence[Optional[str]]] = None,
    ply_idx: Optional[int] = None,
    phase: Optional[str] = None,
) -> Dict[str, float]:
    board = chess.Board(fen)
    ch = chess.Move.from_uci(chosen_uci)
    rj = chess.Move.from_uci(rejected_uci)

    if phase is None:
        phase = infer_phase_from_ply(ply_idx if ply_idx is not None else board.fullmove_number * 2)

    perspective_color = board.turn
    base_w = phase_weights(phase)
    hist_summary = summarize_fen_window(prev_fens, perspective_color)
    w = adapt_weights_from_history(base_w, hist_summary, phase)

    f_ch = extract_style_features(board, ch)
    f_rj = extract_style_features(board, rj)

    vals = {
        "cp_sim": normalize_similarity(float(chosen_cp) - float(rejected_cp), 35.0),
        "forcing_sim": normalize_similarity(f_ch.forcingness - f_rj.forcingness, 1.5),
        "centralization_sim": normalize_similarity(f_ch.centralization_gain - f_rj.centralization_gain, 1.0),
        "distance_sim": normalize_similarity(f_ch.move_distance - f_rj.move_distance, 2.5),
        "activity_sim": normalize_similarity(f_ch.moved_piece_activity - f_rj.moved_piece_activity, 3.0),
        "king_pressure_sim": normalize_similarity(f_ch.king_pressure - f_rj.king_pressure, 3.0),
        "mobility_sim": normalize_similarity(f_ch.mobility_after - f_rj.mobility_after, 6.0),
        "opp_reply_sim": normalize_similarity(f_ch.opponent_forcing_replies - f_rj.opponent_forcing_replies, 3.0),
        "pawn_break_sim": 1.0 if f_ch.is_pawn_break == f_rj.is_pawn_break else 0.0,
        "development_sim": 1.0 if f_ch.is_development == f_rj.is_development else 0.0,
        "castle_sim": 1.0 if f_ch.is_castle == f_rj.is_castle else 0.0,
        "region_sim": 1.0 if f_ch.region_to == f_rj.region_to else 0.0,
        "side_sim": 1.0 if f_ch.file_group_to == f_rj.file_group_to else 0.0,
        "retreat_sim": 1.0 if f_ch.is_retreat == f_rj.is_retreat else 0.0,
        "quiet_sim": 1.0 if f_ch.is_quiet == f_rj.is_quiet else 0.0,
        "piece_sim": 1.0 if f_ch.piece_type == f_rj.piece_type else 0.0,
    }

    local_num = sum(vals[k] * wname for k, wname in [
        ("cp_sim", w["cp"]),
        ("forcing_sim", w["forcing"]),
        ("centralization_sim", w["centralization"]),
        ("distance_sim", w["distance"]),
        ("activity_sim", w["activity"]),
        ("king_pressure_sim", w["king_pressure"]),
        ("mobility_sim", w["mobility"]),
        ("opp_reply_sim", w["opp_reply"]),
        ("pawn_break_sim", w["pawn_break"]),
        ("development_sim", w["development"]),
        ("castle_sim", w["castle"]),
        ("region_sim", w["region"]),
        ("side_sim", w["side"]),
        ("retreat_sim", w["retreat"]),
        ("quiet_sim", w["quiet"]),
        ("piece_sim", w["piece"]),
    ])
    local_den = sum(w.values()) + 1e-12
    local_sim = local_num / local_den

    future_sim = future_window_similarity(next_fens_chosen, next_fens_rejected, perspective_color)
    if future_sim is None:
        final_sim = local_sim
    else:
        alpha_local = 0.65 if phase == "opening" else (0.45 if phase == "middlegame" else 0.60)
        final_sim = alpha_local * local_sim + (1.0 - alpha_local) * future_sim

    out = {
        "phase": phase,
        "local_sim": float(local_sim),
        "future_sim": float(future_sim) if future_sim is not None else -1.0,
        "final_style_score": float(final_sim),
    }
    out.update({f"weight_{k}": float(v) for k, v in w.items()})
    out.update({f"feat_{k}": float(v) for k, v in vals.items()})
    return out

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

PIECE_TYPE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

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
# Main eval
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/eval/single_gm/eval_sft_and_dpo_w_style_v2_maia_single_gm.py --gm_name caruana --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/eval_results_twic --model_dir ./final_experiments_for_paper/experiment1/trained_models_twic
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--split_name", required=False, default="val", help="train or val")
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.6], help="List of beta values (e.g. --betas 0.1 0.2 0.4)")
    ap.add_argument("--dpo_loss_weights", type=float, nargs="+", default=[0.1, 0.2, 0.4], help="List of beta values (e.g. --dpo_loss_weight 0.1 0.2 0.4)")
    
    ap.add_argument("--style_cp_scale", type=float, default=40)
    ap.add_argument("--style_piece_bonus", type=float, default=1.0)
    ap.add_argument("--style_positional_bonus", type=float, default=2.0)
    ap.add_argument("--style_taus", type=float, nargs="+", default=[0.25, 0.75, 1.25], help="List of style taus for tuning how much style similarity reweights loss")

    ap.add_argument("--n_boot", type=int, default=100, help="Number of bootstrap resamples for confidence intervals")
    ap.add_argument("--train_val_folder", required=True, help="Train/val folder.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--model_dir", required=True, help="Model directory.")

    args = ap.parse_args()

    for beta, dpo_loss_weight, style_tau in itertools.product(args.betas, args.dpo_loss_weights, args.style_taus):
        full_name = f"sft_and_dpo_w_style_v2_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale={args.style_cp_scale:.2f}_style_piece_bonus={args.style_piece_bonus:.2f}_style_positional_bonus={args.style_positional_bonus:.2f}_style_tau={style_tau:.2f}"
        jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_{args.split_name}_dpo.jsonl")
        
        def supplied_loss_function(logp_pi_ch, 
                                    logp_pi_rj, 
                                    logp_ref_ch, 
                                    logp_ref_rj, 
                                    logits_pi_m, 
                                    logits_ref_m, 
                                    idx_t, 
                                    chosen_cps, 
                                    rejected_cps, 
                                    prev_fens_batch,
                                    next_fens_chosen_batch,
                                    next_fens_rejected_batch,
                                    batch_meta_data
        ):
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
                device=args.device,
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
                + supervised_nll_loss(logits_pi_m, idx_t)
            )
            return loss
            
        run_eval(jsonl, 
                 f"{args.model_dir}/{args.gm_name}/policy_best_{full_name}.pt", 
                 args.out_dir, 
                 args.gm_name, 
                 args.device, 
                 args.maia_type, 
                 f"opening_probe_policy_{full_name}.json",
                 args.n_boot,
                 args.batch_size,
                 args.split_name,
                 f"eval_results_{full_name}_{args.split_name}.json",
                 f"eval_results_extended_{full_name}_{args.split_name}.json",
                 f"eval_results_{full_name}_{args.split_name}.csv",
                 f"eval_per_row_metrics_{full_name}_{args.split_name}.jsonl",
                 supplied_loss_function
        )

if __name__ == "__main__":
    main()
