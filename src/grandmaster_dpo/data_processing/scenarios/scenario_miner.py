#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import chess
import chess.engine
import chess.pgn

SCENARIO_TYPES: Tuple[str, ...] = (
    "convert",
    "defend",
    "hold_nerves",
    "counterpunch",
    "simplify_correctly",
)

LENGTH_BINS: Tuple[Tuple[int, int], ...] = (
    (1, 2),
    (3, 8),
    (9, 16),
    (17, 32),
)

MATE_CP = 100_000
ROOT_MULTIPV = 3
TREE_DEPTH_PLIES = 4
GOOD_MOVE_GAP_CP = 35
BLUNDER_GAP_CP = 120
KEEP_SAME_EPS_CP = 15
IMPROVES_EPS_CP = 20
SHALLOW_DEPTH_DEFAULT = 2
TACTICAL_SWING_CP = 80
FORCING_LINE_PLIES = 4
TACTICAL_MATERIAL_GAIN_PAWNS = 1


@dataclass(frozen=True)
class CandidateScenario:
    candidate_id: str
    game_index: int
    game_id: str
    white: str
    black: str
    result: str
    event: str
    date: str
    time_class: str
    phase: str
    scenario_type: str
    start_ply: int
    start_fen: str
    length_bin: str
    length_min: int
    length_max: int
    max_rollout_length: int
    root_eval_cp: int
    root_eval_status: str
    best_move_uci: str
    best_move_cp: int
    cp_gap_top1_top2: int
    top_moves_digest: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class EngineAnalysisConfig:
    scan_depth: int
    tree_depth: int
    shallow_depth: int
    tactical_swing_cp: int
    branch_width: int
    tree_depth_plies: int


@dataclass(frozen=True)
class SelectionConfig:
    target_n: int
    max_per_game: int
    max_per_event: int
    max_positions_per_game: int
    seed: int


@dataclass(frozen=True)
class ScanConfig:
    sample_games: Optional[int]
    stop_when_enough: bool
    log_every_games: int


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    output_jsonl: str
    summary_json: str


@dataclass
class SerializationStats:
    written: int = 0
    scenario_counts: Dict[Tuple[str, str, str], int] | None = None
    event_counts: Dict[str, int] | None = None
    difficulty_counts: Dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.scenario_counts is None:
            self.scenario_counts = defaultdict(int)
        if self.event_counts is None:
            self.event_counts = defaultdict(int)
        if self.difficulty_counts is None:
            self.difficulty_counts = defaultdict(int)


def safe_int(x: Optional[str], default: int = -1) -> int:
    if x is None:
        return default
    try:
        return int(str(x).strip())
    except Exception:
        return default


def format_length_bin(bounds: Tuple[int, int]) -> str:
    return f"{bounds[0]}_{bounds[1]}"


def score_to_cp(score: chess.engine.PovScore, *, mate_score: int = MATE_CP) -> int:
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
        mate = rel.mate()
        if mate is not None:
            return mate_score if mate > 0 else -mate_score
        return 0
    return int(cp)


def make_local_stockfish(
    stockfish_path: str,
    *,
    threads: int = 1,
    hash_mb: int = 256,
    timeout: float = 30.0,
) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout=timeout)
    opts: Dict[str, Any] = {}
    if "Threads" in engine.options:
        opts["Threads"] = int(threads)
    if "Hash" in engine.options:
        opts["Hash"] = int(hash_mb)
    if opts:
        engine.configure(opts)
    return engine


def iter_games(path: str) -> Iterator[Tuple[int, chess.pgn.Game]]:
    with open(path, "r", encoding="utf-8", errors="ignore", buffering=1024 * 1024 * 16) as handle:
        game_index = 0
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            game_index += 1
            yield game_index, game


def game_id_from_headers(headers: chess.pgn.Headers, game_index: int) -> str:
    white = (headers.get("White") or "Unknown").strip()
    black = (headers.get("Black") or "Unknown").strip()
    date = (headers.get("Date") or "????.??.??").strip()
    return f"g{game_index}_{white}_vs_{black}_{date}".replace(" ", "_")


def parse_time_control_seconds(headers: chess.pgn.Headers) -> Optional[int]:
    tc = (headers.get("TimeControl") or "").strip()
    if not tc or tc in {"-", "?"}:
        return None
    if "/" in tc:
        tc = tc.split(":", 1)[-1]
    primary = tc.split(":", 1)[0]
    if primary in {"-", "?"}:
        return None
    if "+" in primary:
        base_s, inc_s = primary.split("+", 1)
    else:
        base_s, inc_s = primary, "0"
    try:
        base = int(base_s)
        inc = int(inc_s)
    except Exception:
        return None
    return base + 40 * inc


def classify_time_control(headers: chess.pgn.Headers) -> str:
    event_type = (headers.get("EventType") or "").strip().lower()
    if "bullet" in event_type:
        return "bullet"
    if "blitz" in event_type:
        return "blitz"
    if "rapid" in event_type:
        return "rapid"
    if "classical" in event_type or "standard" in event_type:
        return "classical"

    approx_seconds = parse_time_control_seconds(headers)
    if approx_seconds is None:
        return "unknown"
    if approx_seconds <= 180:
        return "bullet"
    if approx_seconds <= 600:
        return "blitz"
    if approx_seconds <= 3600:
        return "rapid"
    return "classical"


def is_standard_non_bullet_game(game: chess.pgn.Game) -> bool:
    headers = game.headers
    variant = (headers.get("Variant") or "").strip().lower()
    if variant and variant not in {"standard", "chess"}:
        return False
    return classify_time_control(headers) != "bullet"


def classify_eval_status(cp: int) -> str:
    if cp >= 300:
        return "winning"
    if cp >= 120:
        return "comfortable_edge"
    if cp >= 40:
        return "slight_edge"
    if cp > -40:
        return "roughly_equal"
    if cp > -120:
        return "slightly_worse"
    if cp > -300:
        return "worse"
    return "losing"


def print_progress(message: str) -> None:
    print(f"[scenario_miner] {message}", flush=True)


def build_output_paths(args: argparse.Namespace) -> OutputPaths:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = str(output_dir / "scenario_miner_output.jsonl") if args.output_jsonl is None else args.output_jsonl
    summary_json = str(output_dir / "scenario_miner_summary.json") if args.summary_json is None else args.summary_json
    return OutputPaths(output_dir=output_dir, output_jsonl=output_jsonl, summary_json=summary_json)


def build_engine_analysis_config(args: argparse.Namespace) -> EngineAnalysisConfig:
    return EngineAnalysisConfig(
        scan_depth=args.scan_depth,
        tree_depth=args.tree_depth,
        shallow_depth=args.shallow_depth,
        tactical_swing_cp=args.tactical_swing_cp,
        branch_width=args.branch_width,
        tree_depth_plies=args.tree_depth_plies,
    )


def build_selection_config(args: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        target_n=args.target_n,
        max_per_game=args.max_per_game,
        max_per_event=args.max_per_event,
        max_positions_per_game=args.max_positions_per_game,
        seed=args.seed,
    )


def build_scan_config(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        sample_games=args.sample_games,
        stop_when_enough=args.stop_when_enough_candidates,
        log_every_games=args.log_every_games,
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.stockfish_path:
        raise SystemExit("Could not find Stockfish. Pass --stockfish-path or set STOCKFISH_PATH.")
    if args.branch_width < 1:
        raise SystemExit("--branch-width must be >= 1")
    if args.tree_depth_plies < 1:
        raise SystemExit("--tree-depth-plies must be >= 1")
    if args.shallow_depth < 1:
        raise SystemExit("--shallow-depth must be >= 1")


def phase_from_ply(ply_abs: int) -> str:
    if ply_abs < 20:
        return "opening"
    if ply_abs < 60:
        return "middlegame"
    return "endgame"


def material_snapshot(board: chess.Board) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for piece_type, name in (
        (chess.PAWN, "pawns"),
        (chess.KNIGHT, "knights"),
        (chess.BISHOP, "bishops"),
        (chess.ROOK, "rooks"),
        (chess.QUEEN, "queens"),
    ):
        out[f"white_{name}"] = len(board.pieces(piece_type, chess.WHITE))
        out[f"black_{name}"] = len(board.pieces(piece_type, chess.BLACK))
    out["non_king_piece_count"] = sum(
        len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    return out


def material_balance_pawns(board: chess.Board, color: chess.Color) -> int:
    weights = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    own = 0
    opp = 0
    for piece_type, weight in weights.items():
        own += weight * len(board.pieces(piece_type, color))
        opp += weight * len(board.pieces(piece_type, not color))
    return own - opp


def parse_move(board: chess.Board, uci: str) -> Optional[chess.Move]:
    try:
        move = chess.Move.from_uci(uci)
    except Exception:
        return None
    return move if move in board.legal_moves else None


def is_positional_heuristic(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False
    if board.is_capture(move) or board.gives_check(move) or move.promotion is not None:
        return False
    if board.is_castling(move):
        return True
    if piece.piece_type in (chess.BISHOP, chess.KNIGHT):
        start_rank = chess.square_rank(move.from_square)
        if piece.color == chess.WHITE and start_rank == 0:
            return True
        if piece.color == chess.BLACK and start_rank == 7:
            return True
    if piece.piece_type == chess.PAWN:
        to_file = chess.square_file(move.to_square)
        if to_file in (2, 3, 4, 5):
            return True
        return True
    if piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KNIGHT):
        return True
    return False


def classify_move_style(board: chess.Board, uci: str) -> Dict[str, Any]:
    move = parse_move(board, uci)
    if move is None:
        return {
            "is_legal_on_board": False,
            "is_capture": False,
            "is_check": False,
            "is_promotion": False,
            "is_castle": False,
            "is_tactical": False,
            "is_positional": False,
            "is_quiet": False,
            "is_aggressive": False,
            "piece_type": None,
        }

    piece = board.piece_at(move.from_square)
    is_capture = board.is_capture(move)
    is_check = board.gives_check(move)
    is_promotion = move.promotion is not None
    is_castle = board.is_castling(move)
    is_tactical = is_capture or is_check or is_promotion
    is_positional = is_positional_heuristic(board, move)
    is_quiet = not is_tactical
    to_rank = chess.square_rank(move.to_square)
    aggressive_by_space = (
        piece is not None
        and piece.piece_type in (chess.PAWN, chess.QUEEN, chess.ROOK)
        and ((piece.color == chess.WHITE and to_rank >= 4) or (piece.color == chess.BLACK and to_rank <= 3))
    )
    return {
        "is_legal_on_board": True,
        "is_capture": is_capture,
        "is_check": is_check,
        "is_promotion": is_promotion,
        "is_castle": is_castle,
        "is_tactical": is_tactical,
        "is_positional": is_positional,
        "is_quiet": is_quiet,
        "is_aggressive": bool(is_tactical or aggressive_by_space),
        "piece_type": None if piece is None else chess.piece_name(piece.piece_type),
    }


def board_after_uci(board: chess.Board, uci: str) -> Optional[chess.Board]:
    move = parse_move(board, uci)
    if move is None:
        return None
    nxt = board.copy(stack=False)
    nxt.push(move)
    return nxt


def analyse_single_move_score(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move: chess.Move,
    *,
    depth: int,
) -> Dict[str, Any]:
    next_board = board.copy(stack=False)
    next_board.push(move)
    info = engine.analyse(
        next_board,
        chess.engine.Limit(depth=depth),
        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )
    score = info.get("score")
    cp_from_parent_pov = 0 if score is None else -score_to_cp(score)
    pv = info.get("pv") or []
    return {
        "cp": int(cp_from_parent_pov),
        "pv_uci": [mv.uci() for mv in pv],
    }


def evaluate_forcing_tactical_line(
    board: chess.Board,
    move: chess.Move,
    deep_pv: Sequence[str],
) -> Dict[str, Any]:
    mover = board.turn
    start_material = material_balance_pawns(board, mover)
    line_board = board.copy(stack=False)
    sequence: List[Dict[str, Any]] = []
    forcing_plies = 0

    first_move = move
    line_board.push(first_move)
    first_forcing = board.is_capture(first_move) or board.gives_check(first_move)
    sequence.append(
        {
            "uci": first_move.uci(),
            "is_capture": board.is_capture(first_move),
            "is_check": board.gives_check(first_move),
        }
    )
    if first_forcing:
        forcing_plies += 1

    for uci in list(deep_pv)[: max(0, FORCING_LINE_PLIES - 1)]:
        nxt = parse_move(line_board, uci)
        if nxt is None:
            break
        is_capture = line_board.is_capture(nxt)
        is_check = line_board.gives_check(nxt)
        sequence.append({"uci": uci, "is_capture": is_capture, "is_check": is_check})
        if is_capture or is_check:
            forcing_plies += 1
            line_board.push(nxt)
            continue
        break

    end_material = material_balance_pawns(line_board, mover)
    material_gain = end_material - start_material
    is_mateish = line_board.is_checkmate()
    forcing_capture_check_line = first_forcing and forcing_plies >= 2 and (
        material_gain >= TACTICAL_MATERIAL_GAIN_PAWNS or is_mateish
    )
    return {
        "forcing_sequence_plies": int(forcing_plies),
        "forcing_sequence": sequence,
        "material_gain_pawns": int(material_gain),
        "results_in_checkmate": bool(is_mateish),
        "is_forcing_capture_check_tactic": bool(forcing_capture_check_line),
    }


def compute_tactical_oracle(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move_uci: str,
    deep_cp: int,
    deep_pv: Sequence[str],
    *,
    shallow_depth: int,
    tactical_swing_cp: int,
) -> Dict[str, Any]:
    move = parse_move(board, move_uci)
    if move is None:
        return {
            "is_tactical": False,
            "is_forcing_capture_check_tactic": False,
            "is_shallow_deep_eval_tactic": False,
            "shallow_cp": None,
            "deep_cp": int(deep_cp),
            "shallow_deep_cp_gap": None,
            "forcing_sequence_plies": 0,
            "forcing_sequence": [],
            "material_gain_pawns": 0,
            "results_in_checkmate": False,
        }

    shallow = analyse_single_move_score(engine, board, move, depth=shallow_depth)
    shallow_cp = int(shallow["cp"])
    move_style = classify_move_style(board, move_uci)
    is_attacking_move = bool(
        move_style["is_check"]
        or move_style["is_capture"]
        or move_style["is_aggressive"]
    )

    forcing = evaluate_forcing_tactical_line(board, move, deep_pv)
    shallow_deep_gap = int(deep_cp - shallow_cp)
    is_shallow_deep_eval_tactic = is_attacking_move and shallow_deep_gap >= tactical_swing_cp

    return {
        "is_tactical": bool(forcing["is_forcing_capture_check_tactic"] or is_shallow_deep_eval_tactic),
        "is_forcing_capture_check_tactic": bool(forcing["is_forcing_capture_check_tactic"]),
        "is_shallow_deep_eval_tactic": bool(is_shallow_deep_eval_tactic),
        "shallow_cp": shallow_cp,
        "deep_cp": int(deep_cp),
        "shallow_deep_cp_gap": shallow_deep_gap,
        **forcing,
    }


def analyse_top_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    depth: int,
    multipv: int,
    shallow_depth: int = SHALLOW_DEPTH_DEFAULT,
    tactical_swing_cp: int = TACTICAL_SWING_CP,
) -> Dict[str, Any]:
    infos = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=multipv,
        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )

    top_moves: List[Dict[str, Any]] = []
    best_cp: Optional[int] = None

    for rank, info in enumerate(infos, start=1):
        pv = info.get("pv") or []
        score = info.get("score")
        if not pv or score is None:
            continue
        root_move = pv[0]
        root_uci = root_move.uci()
        cp = score_to_cp(score)
        move_style = classify_move_style(board, root_uci)
        tactical_oracle = compute_tactical_oracle(
            engine,
            board,
            root_uci,
            cp,
            [mv.uci() for mv in pv[1:]],
            shallow_depth=shallow_depth,
            tactical_swing_cp=tactical_swing_cp,
        )

        san = None
        try:
            san = board.san(root_move)
        except Exception:
            san = None

        top_moves.append(
            {
                "rank": rank,
                "uci": root_uci,
                "san": san,
                "cp": cp,
                "pv_uci": [mv.uci() for mv in pv],
                **move_style,
                "tactical_oracle": tactical_oracle,
            }
        )
        if best_cp is None:
            best_cp = cp

    top_moves.sort(key=lambda row: row["rank"])
    if best_cp is None:
        best_cp = 0

    second_cp = top_moves[1]["cp"] if len(top_moves) > 1 else best_cp
    return {
        "root_eval_cp": best_cp,
        "root_eval_status": classify_eval_status(best_cp),
        "top_moves": top_moves,
        "cp_gap_top1_top2": int(best_cp - second_cp),
    }


def classify_scenario(board: chess.Board, phase: str, summary: Dict[str, Any]) -> Optional[str]:
    top_moves = summary["top_moves"]
    if len(top_moves) < 2:
        return None

    root_cp = int(summary["root_eval_cp"])
    best = top_moves[0]
    tactical_near_best = [
        row
        for row in top_moves
        if row["tactical_oracle"]["is_tactical"] and int(best["cp"]) - int(row["cp"]) <= 40
    ]
    quiet_near_best = [row for row in top_moves if row["is_quiet"] and int(best["cp"]) - int(row["cp"]) <= 35]
    top3_tactical_count = sum(1 for row in top_moves[:3] if row["tactical_oracle"]["is_tactical"])
    has_trade_candidate = any(row["is_capture"] for row in top_moves[:2])
    material = material_snapshot(board)
    low_material = material["non_king_piece_count"] <= 12
    top1_top2_gap = int(summary["cp_gap_top1_top2"])
    best_minus_root = int(best["cp"]) - root_cp

    if (
        root_cp >= 50
        and best["is_capture"]
        and (low_material or phase == "endgame")
        and top1_top2_gap >= 55
        and has_trade_candidate
    ):
        return "simplify_correctly"

    if (
        root_cp >= 40
        and root_cp <= 220
        and best["is_quiet"]
        and best["is_positional"]
        and not tactical_near_best
        and best_minus_root >= 10
    ):
        return "convert"

    if (
        root_cp <= -40
        and root_cp >= -220
        and tactical_near_best
        and best["is_aggressive"]
    ):
        return "counterpunch"

    if (
        root_cp <= -30
        and root_cp >= -180
        and best["is_aggressive"]
        and not tactical_near_best
        and best_minus_root >= 10
    ):
        return "defend"

    if (
        abs(root_cp) <= 25
        and len(quiet_near_best) >= 2
        and top1_top2_gap <= 20
        and best["is_quiet"]
        and top3_tactical_count == 0
        and all((not row["is_check"]) for row in top_moves[:2])
    ):
        return "hold_nerves"

    return None


def estimate_scenario_difficulty(
    *,
    scenario_type: str,
    root_eval_cp: int,
    cp_gap_top1_top2: int,
    top_moves: Sequence[Dict[str, Any]],
    phase: str,
    rollout_len: int,
) -> str:
    score = 0
    abs_cp = abs(int(root_eval_cp))

    if abs_cp <= 20:
        score += 3
    elif abs_cp <= 60:
        score += 2
    elif abs_cp <= 120:
        score += 1

    if int(cp_gap_top1_top2) <= 10:
        score += 3
    elif int(cp_gap_top1_top2) <= 25:
        score += 2
    elif int(cp_gap_top1_top2) <= 50:
        score += 1

    tactical_count = sum(1 for row in top_moves[:3] if row.get("labels", {}).get("is_tactical", False))
    score += tactical_count

    if phase == "endgame":
        score += 1
    if rollout_len >= 17:
        score += 2
    elif rollout_len >= 9:
        score += 1

    if scenario_type in {"hold_nerves", "counterpunch"}:
        score += 1
    if scenario_type == "simplify_correctly":
        score += 1

    if score <= 2:
        return "1000"
    if score <= 4:
        return "1500"
    if score <= 6:
        return "2000"
    if score <= 8:
        return "2500"
    return "3000+"


def feasible_length_bins(remaining_plies: int) -> List[Tuple[int, int]]:
    return [bounds for bounds in LENGTH_BINS if remaining_plies >= bounds[0]]


def pick_exact_rollout_length(
    rng: random.Random,
    length_min: int,
    length_max: int,
    remaining_plies: int,
) -> int:
    upper = min(length_max, remaining_plies)
    return upper if upper <= length_min else rng.randint(length_min, upper)


def collect_candidates(
    *,
    input_pgn: str,
    engine: chess.engine.SimpleEngine,
    depth: int,
    shallow_depth: int,
    tactical_swing_cp: int,
    selection_cfg: SelectionConfig,
    scan_cfg: ScanConfig,
) -> List[CandidateScenario]:
    rng = random.Random(selection_cfg.seed)
    candidates: List[CandidateScenario] = []

    for game_index, game in iter_games(input_pgn):
        if scan_cfg.sample_games is not None and game_index > scan_cfg.sample_games:
            break
        if not is_standard_non_bullet_game(game):
            continue

        headers = game.headers
        time_class = classify_time_control(headers)
        game_id = game_id_from_headers(headers, game_index)
        board = game.board()
        moves = list(game.mainline_moves())
        if len(moves) < 20:
            continue

        kept_for_game = 0
        for ply_zero_index, move in enumerate(moves):
            ply_abs = ply_zero_index + 1
            phase = phase_from_ply(ply_abs)
            remaining_plies = len(moves) - ply_zero_index
            if phase not in {"middlegame", "endgame"}:
                board.push(move)
                continue
            if remaining_plies < LENGTH_BINS[0][0]:
                break
            if board.is_game_over(claim_draw=True):
                break

            try:
                summary = analyse_top_moves(
                    engine,
                    board,
                    depth=depth,
                    multipv=ROOT_MULTIPV,
                    shallow_depth=shallow_depth,
                    tactical_swing_cp=tactical_swing_cp,
                )
            except Exception:
                board.push(move)
                continue

            scenario_type = classify_scenario(board, phase, summary)
            if scenario_type is None:
                board.push(move)
                continue

            for length_bounds in feasible_length_bins(remaining_plies):
                candidate = CandidateScenario(
                    candidate_id=f"{game_id}_p{ply_abs}_{scenario_type}_{format_length_bin(length_bounds)}",
                    game_index=game_index,
                    game_id=game_id,
                    white=(headers.get("White") or "").strip(),
                    black=(headers.get("Black") or "").strip(),
                    result=(headers.get("Result") or "").strip(),
                    event=(headers.get("Event") or "").strip(),
                    date=(headers.get("Date") or "").strip(),
                    time_class=time_class,
                    phase=phase,
                    scenario_type=scenario_type,
                    start_ply=ply_abs,
                    start_fen=board.fen(),
                    length_bin=format_length_bin(length_bounds),
                    length_min=length_bounds[0],
                    length_max=length_bounds[1],
                    max_rollout_length=remaining_plies,
                    root_eval_cp=int(summary["root_eval_cp"]),
                    root_eval_status=str(summary["root_eval_status"]),
                    best_move_uci=str(summary["top_moves"][0]["uci"]),
                    best_move_cp=int(summary["top_moves"][0]["cp"]),
                    cp_gap_top1_top2=int(summary["cp_gap_top1_top2"]),
                    top_moves_digest=tuple((str(row["uci"]), int(row["cp"])) for row in summary["top_moves"]),
                )
                candidates.append(candidate)

            kept_for_game += 1
            # Keep candidate density modest in a single game. We only need one final sample per game by default.
            if kept_for_game >= selection_cfg.max_positions_per_game:
                break

            board.push(move)
        else:
            continue

        if scan_cfg.log_every_games > 0 and game_index % scan_cfg.log_every_games == 0:
            tentative_selected = select_balanced_candidates(
                candidates,
                target_n=selection_cfg.target_n,
                max_per_game=selection_cfg.max_per_game,
                max_per_event=selection_cfg.max_per_event,
                seed=selection_cfg.seed,
            )
            print_progress(
                f"scanned_games={game_index} candidates={len(candidates)} "
                f"selectable_now={len(tentative_selected)}"
            )

        if scan_cfg.stop_when_enough and len(candidates) >= selection_cfg.target_n:
            tentative_selected = select_balanced_candidates(
                candidates,
                target_n=selection_cfg.target_n,
                max_per_game=selection_cfg.max_per_game,
                max_per_event=selection_cfg.max_per_event,
                seed=selection_cfg.seed,
            )
            if len(tentative_selected) >= selection_cfg.target_n:
                print_progress(
                    f"early stop after game {game_index}: candidates={len(candidates)} "
                    f"already support target_n={selection_cfg.target_n}"
                )
                break

    rng.shuffle(candidates)
    return candidates


def select_balanced_candidates(
    candidates: Sequence[CandidateScenario],
    *,
    target_n: int,
    max_per_game: int,
    max_per_event: int,
    seed: int,
) -> List[CandidateScenario]:
    rng = random.Random(seed)
    by_stratum: Dict[Tuple[str, str, str], List[CandidateScenario]] = defaultdict(list)
    for cand in candidates:
        by_stratum[(cand.phase, cand.scenario_type, cand.length_bin)].append(cand)

    strata = list(by_stratum.keys())
    for rows in by_stratum.values():
        rng.shuffle(rows)
    strata.sort(key=lambda key: len(by_stratum[key]))

    selected: List[CandidateScenario] = []
    used_by_game: Dict[int, int] = defaultdict(int)
    used_by_event: Dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()

    made_progress = True
    while len(selected) < target_n and made_progress:
        made_progress = False
        for stratum in strata:
            pool = by_stratum[stratum]
            while pool:
                cand = pool.pop()
                if cand.candidate_id in seen_ids:
                    continue
                if used_by_game[cand.game_index] >= max_per_game:
                    continue
                event_key = cand.event.strip() or "__unknown_event__"
                if used_by_event[event_key] >= max_per_event:
                    continue
                selected.append(cand)
                used_by_game[cand.game_index] += 1
                used_by_event[event_key] += 1
                seen_ids.add(cand.candidate_id)
                made_progress = True
                break
            if len(selected) >= target_n:
                break

    return selected


def move_labels_for_candidate(
    *,
    board: chess.Board,
    move_row: Dict[str, Any],
    parent_cp: int,
    best_cp: int,
) -> Dict[str, bool]:
    move_cp = int(move_row["cp"])
    cp_gap = int(best_cp - move_cp)
    base_style = classify_move_style(board, str(move_row["uci"]))

    improves_vs_parent = move_cp >= parent_cp + IMPROVES_EPS_CP
    worsens_vs_parent = move_cp <= parent_cp - IMPROVES_EPS_CP
    keeps_same = abs(move_cp - parent_cp) <= KEEP_SAME_EPS_CP

    return {
        "is_tactical": bool(move_row.get("tactical_oracle", {}).get("is_tactical", base_style["is_tactical"])),
        "is_positional": bool(base_style["is_positional"]),
        "is_engine_best": int(move_row["rank"]) == 1,
        "is_aggressive": bool(base_style["is_aggressive"]),
        "is_improves_odds": improves_vs_parent,
        "is_worsens_odds": worsens_vs_parent,
        "is_keeps_odds_roughly_same": keeps_same,
        "is_a_better_move": move_cp >= parent_cp + KEEP_SAME_EPS_CP,
        "is_blunder": cp_gap >= BLUNDER_GAP_CP,
        "is_better_move_but_good": (0 < cp_gap <= GOOD_MOVE_GAP_CP),
        "is_forcing_capture_check_tactic": bool(move_row.get("tactical_oracle", {}).get("is_forcing_capture_check_tactic", False)),
        "is_shallow_deep_eval_tactic": bool(move_row.get("tactical_oracle", {}).get("is_shallow_deep_eval_tactic", False)),
    }


def build_light_tree(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    depth: int,
    branch_width: int,
    remaining_tree_plies: int,
    shallow_depth: int,
    tactical_swing_cp: int,
) -> Dict[str, Any]:
    summary = analyse_top_moves(
        engine,
        board,
        depth=depth,
        multipv=branch_width,
        shallow_depth=shallow_depth,
        tactical_swing_cp=tactical_swing_cp,
    )
    root_cp = int(summary["root_eval_cp"])
    node: Dict[str, Any] = {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "eval_cp": root_cp,
        "eval_status": classify_eval_status(root_cp),
        "top_moves": [],
    }

    if remaining_tree_plies <= 0:
        return node

    for move_row in summary["top_moves"]:
        labels = move_labels_for_candidate(
            board=board,
            move_row=move_row,
            parent_cp=root_cp,
            best_cp=int(summary["top_moves"][0]["cp"]),
        )
        move_entry: Dict[str, Any] = {
            "rank": int(move_row["rank"]),
            "uci": str(move_row["uci"]),
            "san": move_row["san"],
            "cp": int(move_row["cp"]),
            "cp_gap_from_best": int(summary["top_moves"][0]["cp"]) - int(move_row["cp"]),
            "pv_uci": list(move_row["pv_uci"]),
            "tactical_oracle": move_row.get("tactical_oracle", {}),
            "labels": labels,
        }
        child_board = board_after_uci(board, str(move_row["uci"]))
        if child_board is not None and remaining_tree_plies > 1 and not child_board.is_game_over(claim_draw=True):
            move_entry["child"] = build_light_tree(
                engine,
                child_board,
                depth=depth,
                branch_width=branch_width,
                remaining_tree_plies=remaining_tree_plies - 1,
                shallow_depth=shallow_depth,
                tactical_swing_cp=tactical_swing_cp,
            )
        node["top_moves"].append(move_entry)

    return node


def evaluate_played_move(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    played_move: chess.Move,
    *,
    depth: int,
    multipv: int,
    shallow_depth: int,
    tactical_swing_cp: int,
) -> Dict[str, Any]:
    summary = analyse_top_moves(
        engine,
        board,
        depth=depth,
        multipv=multipv,
        shallow_depth=shallow_depth,
        tactical_swing_cp=tactical_swing_cp,
    )
    best_cp = int(summary["root_eval_cp"])
    played_uci = played_move.uci()

    played_row = None
    for row in summary["top_moves"]:
        if row["uci"] == played_uci:
            played_row = row
            break

    if played_row is None:
        next_board = board.copy(stack=False)
        next_board.push(played_move)
        reply_info = engine.analyse(next_board, chess.engine.Limit(depth=max(8, depth // 2)))
        reply_score = reply_info.get("score")
        played_cp = 0 if reply_score is None else -score_to_cp(reply_score)
        played_row = {
            "rank": multipv + 1,
            "uci": played_uci,
            "san": board.san(played_move),
            "cp": played_cp,
            "pv_uci": [played_uci],
            "tactical_oracle": compute_tactical_oracle(
                engine,
                board,
                played_uci,
                played_cp,
                [],
                shallow_depth=shallow_depth,
                tactical_swing_cp=tactical_swing_cp,
            ),
        }

    labels = move_labels_for_candidate(board=board, move_row=played_row, parent_cp=best_cp, best_cp=best_cp)
    return {
        "fen": board.fen(),
        "played_move_uci": played_uci,
        "played_move_san": board.san(played_move),
        "position_eval_cp": best_cp,
        "position_eval_status": classify_eval_status(best_cp),
        "played_move_cp": int(played_row["cp"]),
        "played_move_rank_if_in_topk": int(played_row["rank"]) if int(played_row["rank"]) <= multipv else None,
        "played_move_labels": labels,
        "top_moves": [
            {
                "rank": int(row["rank"]),
                "uci": str(row["uci"]),
                "san": row["san"],
                "cp": int(row["cp"]),
                "tactical_oracle": row.get("tactical_oracle", {}),
                "labels": move_labels_for_candidate(board=board, move_row=row, parent_cp=best_cp, best_cp=best_cp),
            }
            for row in summary["top_moves"]
        ],
    }


def refine_scenario_from_tree(
    initial_scenario: str,
    tree: Dict[str, Any],
) -> str:
    root_cp = int(tree["eval_cp"])
    top_moves = tree.get("top_moves", [])
    if not top_moves:
        return initial_scenario
    tactical_near_best = [
        row for row in top_moves
        if row["labels"]["is_tactical"] and int(top_moves[0]["cp"]) - int(row["cp"]) <= 40
    ]
    if initial_scenario == "defend" and tactical_near_best:
        return "counterpunch"
    if initial_scenario == "counterpunch" and not tactical_near_best:
        return "defend"
    if initial_scenario == "hold_nerves" and abs(root_cp) > 45:
        return "convert" if root_cp > 0 else "defend"
    return initial_scenario


def build_headers_payload(headers: chess.pgn.Headers) -> Dict[str, Optional[str]]:
    return {
        "White": headers.get("White"),
        "Black": headers.get("Black"),
        "WhiteElo": headers.get("WhiteElo"),
        "BlackElo": headers.get("BlackElo"),
        "Event": headers.get("Event"),
        "Site": headers.get("Site"),
        "Date": headers.get("Date"),
        "Round": headers.get("Round"),
        "Result": headers.get("Result"),
        "ECO": headers.get("ECO"),
        "TimeControl": headers.get("TimeControl"),
        "EventType": headers.get("EventType"),
    }


def update_serialization_stats(
    stats: SerializationStats,
    *,
    candidate: CandidateScenario,
    refined_scenario: str,
    difficulty_estimate: str,
) -> None:
    stats.written += 1
    stats.scenario_counts[(candidate.phase, refined_scenario, candidate.length_bin)] += 1
    stats.event_counts[candidate.event.strip() or "__unknown_event__"] += 1
    stats.difficulty_counts[difficulty_estimate] += 1


def build_summary_payload(stats: SerializationStats, requested_scenarios: int) -> Dict[str, Any]:
    return {
        "written_scenarios": stats.written,
        "requested_scenarios": requested_scenarios,
        "scenario_counts": {
            f"{phase}|{scenario}|{length_bin}": count
            for (phase, scenario, length_bin), count in sorted(stats.scenario_counts.items())
        },
        "event_counts": dict(sorted(stats.event_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "difficulty_counts": dict(sorted(stats.difficulty_counts.items())),
    }


def build_scenario_row(
    *,
    candidate: CandidateScenario,
    headers: chess.pgn.Headers,
    input_pgn: str,
    stockfish_depth: int,
    branch_width: int,
    tree_depth_plies: int,
    shallow_depth: int,
    tactical_swing_cp: int,
    rollout_len: int,
    refined_scenario: str,
    difficulty_estimate: str,
    light_tree: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
    trajectory_final_fen: str,
    start_board: chess.Board,
) -> Dict[str, Any]:
    return {
        "scenario_id": candidate.candidate_id,
        "game_index": candidate.game_index,
        "game_id": candidate.game_id,
        "headers": build_headers_payload(headers),
        "phase": candidate.phase,
        "scenario_type": refined_scenario,
        "time_class": candidate.time_class,
        "length_bin": candidate.length_bin,
        "sampled_rollout_length_plies": rollout_len,
        "difficulty_estimate_elo": difficulty_estimate,
        "start_ply": candidate.start_ply,
        "fen": candidate.start_fen,
        "root_eval_cp": candidate.root_eval_cp,
        "root_eval_status": candidate.root_eval_status,
        "light_tree_depth_plies": tree_depth_plies,
        "light_tree_branch_width": branch_width,
        "light_tree": light_tree,
        "trajectory": trajectory,
        "trajectory_final_fen": trajectory_final_fen,
        "material": material_snapshot(start_board),
        "source": {
            "input_pgn": input_pgn,
            "stockfish_depth": stockfish_depth,
            "tactical_oracle_shallow_depth": shallow_depth,
            "tactical_oracle_swing_cp": tactical_swing_cp,
            "stockfish_multipv": branch_width,
        },
    }


def serialize_selected_scenarios(
    *,
    input_pgn: str,
    selected: Sequence[CandidateScenario],
    engine: chess.engine.SimpleEngine,
    depth: int,
    branch_width: int,
    tree_depth_plies: int,
    shallow_depth: int,
    tactical_swing_cp: int,
    output_jsonl: str,
    summary_json: str,
    seed: int,
) -> None:
    rng = random.Random(seed)
    selected_by_game: Dict[int, List[CandidateScenario]] = defaultdict(list)
    for cand in selected:
        selected_by_game[cand.game_index].append(cand)

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    stats = SerializationStats()
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for game_index, game in iter_games(input_pgn):
            wanted = selected_by_game.get(game_index)
            if not wanted:
                continue

            headers = game.headers
            moves = list(game.mainline_moves())
            for cand in wanted:
                rollout_len = pick_exact_rollout_length(
                    rng,
                    cand.length_min,
                    cand.length_max,
                    cand.max_rollout_length,
                )
                board = game.board()
                for move in moves[: cand.start_ply - 1]:
                    board.push(move)

                start_board = board.copy(stack=False)
                light_tree = build_light_tree(
                    engine,
                    start_board,
                    depth=depth,
                    branch_width=branch_width,
                    remaining_tree_plies=tree_depth_plies,
                    shallow_depth=shallow_depth,
                    tactical_swing_cp=tactical_swing_cp,
                )
                refined_scenario = refine_scenario_from_tree(cand.scenario_type, light_tree)
                difficulty_estimate = estimate_scenario_difficulty(
                    scenario_type=refined_scenario,
                    root_eval_cp=cand.root_eval_cp,
                    cp_gap_top1_top2=cand.cp_gap_top1_top2,
                    top_moves=light_tree.get("top_moves", []),
                    phase=cand.phase,
                    rollout_len=rollout_len,
                )

                trajectory: List[Dict[str, Any]] = []
                cursor = start_board.copy(stack=False)
                for move in moves[cand.start_ply - 1 : cand.start_ply - 1 + rollout_len]:
                    trajectory.append(
                        evaluate_played_move(
                            engine,
                            cursor,
                            move,
                            depth=depth,
                            multipv=branch_width,
                            shallow_depth=shallow_depth,
                            tactical_swing_cp=tactical_swing_cp,
                        )
                    )
                    cursor.push(move)

                row = build_scenario_row(
                    candidate=cand,
                    headers=headers,
                    input_pgn=input_pgn,
                    stockfish_depth=depth,
                    branch_width=branch_width,
                    tree_depth_plies=tree_depth_plies,
                    shallow_depth=shallow_depth,
                    tactical_swing_cp=tactical_swing_cp,
                    rollout_len=rollout_len,
                    refined_scenario=refined_scenario,
                    difficulty_estimate=difficulty_estimate,
                    light_tree=light_tree,
                    trajectory=trajectory,
                    trajectory_final_fen=cursor.fen(),
                    start_board=start_board,
                )
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                update_serialization_stats(
                    stats,
                    candidate=cand,
                    refined_scenario=refined_scenario,
                    difficulty_estimate=difficulty_estimate,
                )

    summary_payload = build_summary_payload(stats, len(selected))
    with open(summary_json, "w", encoding="utf-8") as fout:
        json.dump(summary_payload, fout, indent=2)


def default_stockfish_path() -> Optional[str]:
    env = os.environ.get("STOCKFISH_PATH", "").strip()
    if env:
        return env
    return shutil.which("stockfish")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Mine stratified middlegame/endgame scenario positions from a large PGN."
    )
    ap.add_argument("--input-pgn", required=True, help="Path to the large TWIC PGN file.")
    ap.add_argument(
        "--output-dir",
        default="./website_data_mining_experiments",
        help="Directory for mined scenario outputs.",
    )
    ap.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional explicit JSONL path. Defaults inside --output-dir.",
    )
    ap.add_argument(
        "--summary-json",
        default=None,
        help="Optional explicit summary path. Defaults inside --output-dir.",
    )
    ap.add_argument("--stockfish-path", default=default_stockfish_path(), help="Path to Stockfish binary.")
    ap.add_argument("--target-n", type=int, default=500, help="Number of scenarios to write.")
    ap.add_argument("--scan-depth", type=int, default=12, help="Stockfish depth for candidate scanning.")
    ap.add_argument("--tree-depth", type=int, default=16, help="Stockfish depth for tree/trajectory evaluation.")
    ap.add_argument("--shallow-depth", type=int, default=SHALLOW_DEPTH_DEFAULT, help="Shallow depth used by the tactical oracle.")
    ap.add_argument("--tactical-swing-cp", type=int, default=TACTICAL_SWING_CP, help="Min deep-minus-shallow cp gain for the tactical oracle.")
    ap.add_argument("--branch-width", type=int, default=3, help="Top-k moves to store per state.")
    ap.add_argument("--tree-depth-plies", type=int, default=TREE_DEPTH_PLIES, help="Max plies in the light tree.")
    ap.add_argument("--max-per-game", type=int, default=1, help="Maximum selected scenarios from any single game.")
    ap.add_argument("--max-per-event", type=int, default=10, help="Maximum selected scenarios from any single event.")
    ap.add_argument("--max-positions-per-game", type=int, default=2, help="Cap candidate roots examined per game.")
    ap.add_argument("--threads", type=int, default=1, help="Stockfish Threads option.")
    ap.add_argument("--hash-mb", type=int, default=2048, help="Stockfish Hash option in MB.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed.")
    ap.add_argument("--sample-games", type=int, default=None, help="Optional cap for smoke tests.")
    ap.add_argument("--log-every-games", type=int, default=25, help="Emit progress every N scanned games.")
    ap.add_argument("--stop-when-enough-candidates", dest="stop_when_enough_candidates", action="store_true", default=True, help="Stop scanning once the current candidate pool can already satisfy target_n.")
    ap.add_argument("--no-stop-when-enough-candidates", dest="stop_when_enough_candidates", action="store_false", help="Keep scanning all games even after enough candidates are available.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    output_paths = build_output_paths(args)
    engine_cfg = build_engine_analysis_config(args)
    selection_cfg = build_selection_config(args)
    scan_cfg = build_scan_config(args)
    print_progress(
        f"starting scan_depth={engine_cfg.scan_depth} tree_depth={engine_cfg.tree_depth} "
        f"target_n={selection_cfg.target_n} output_dir={output_paths.output_dir}"
    )

    engine = make_local_stockfish(
        args.stockfish_path,
        threads=args.threads,
        hash_mb=args.hash_mb,
        timeout=30.0,
    )

    try:
        candidates = collect_candidates(
            input_pgn=args.input_pgn,
            engine=engine,
            depth=engine_cfg.scan_depth,
            shallow_depth=engine_cfg.shallow_depth,
            tactical_swing_cp=engine_cfg.tactical_swing_cp,
            selection_cfg=selection_cfg,
            scan_cfg=scan_cfg,
        )
        print_progress(f"candidate_scan_complete candidates={len(candidates)}")
        selected = select_balanced_candidates(
            candidates,
            target_n=selection_cfg.target_n,
            max_per_game=selection_cfg.max_per_game,
            max_per_event=selection_cfg.max_per_event,
            seed=selection_cfg.seed,
        )
        print_progress(f"selection_complete selected={len(selected)}")
        serialize_selected_scenarios(
            input_pgn=args.input_pgn,
            selected=selected,
            engine=engine,
            depth=engine_cfg.tree_depth,
            branch_width=engine_cfg.branch_width,
            tree_depth_plies=engine_cfg.tree_depth_plies,
            shallow_depth=engine_cfg.shallow_depth,
            tactical_swing_cp=engine_cfg.tactical_swing_cp,
            output_jsonl=output_paths.output_jsonl,
            summary_json=output_paths.summary_json,
            seed=selection_cfg.seed,
        )
        print_progress(
            f"done wrote_jsonl={output_paths.output_jsonl} wrote_summary={output_paths.summary_json}"
        )
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
