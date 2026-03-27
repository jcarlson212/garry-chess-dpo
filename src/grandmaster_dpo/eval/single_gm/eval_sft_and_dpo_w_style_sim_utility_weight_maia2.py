from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import chess
import itertools

import torch

from maia2.utils import mirror_move
from grandmaster_dpo.eval.single_gm.shared_eval_metric_utilities import run_eval


# ----------------------------
# Helpers (match training)
# ----------------------------

def is_positional(board: chess.Board, move: chess.Move) -> bool:
    if move.promotion is not None:
        return False
    if board.is_capture(move):
        return False
    if board.gives_check(move):
        return False
    return True


def same_piece_type(board: chess.Board, move_a: chess.Move, move_b: chess.Move) -> bool:
    piece_a = board.piece_at(move_a.from_square)
    piece_b = board.piece_at(move_b.from_square)
    if piece_a is None or piece_b is None:
        return False
    return piece_a.piece_type == piece_b.piece_type


def compute_style_score(
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    chosen_cp: float,
    rejected_cp: float,
    cp_scale: float = 40.0,
    piece_bonus: float = 2.0,
    positional_bonus: float = 2.0,
) -> float:
    board = chess.Board(fen)
    ch = chess.Move.from_uci(chosen_uci)
    rj = chess.Move.from_uci(rejected_uci)

    same_piece = same_piece_type(board, ch, rj)
    same_pos = (is_positional(board, ch) == is_positional(board, rj))
    cp_sim = math.exp(-abs(float(chosen_cp) - float(rejected_cp)) / max(cp_scale, 1e-6))

    score = cp_sim
    if same_piece:
        score *= piece_bonus
    if same_pos:
        score *= positional_bonus
    return float(score)

def chosen_index_tensor(
    fens: List[str],
    all_moves_dict: Dict[str, int],
    moves_uci: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert UCI -> Maia vocab index (mirroring if fen is black-to-move).
    Returns idx_t with -1 for unknown moves (should be rare; those will be ignored).
    """
    idxs: List[int] = []
    for fen, uci in zip(fens, moves_uci):
        side = fen.split(" ")[1]
        uci_eff = mirror_move(uci) if side == "b" else uci
        idx = all_moves_dict.get(uci_eff, None)
        idxs.append(-1 if idx is None else int(idx))
    return torch.tensor(idxs, device=device, dtype=torch.long)

PIECE_TYPE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

PIECE_NAME_LIST = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def add_piece_selection_per_row_stats(
    per_rows_sorted: List[Dict[str, Any]],
    cp_gap_thresholds: tuple[int, ...] = (0, 10, 20, 40, 80, 120),
    seq_windows: tuple[int, ...] = (3, 5),
) -> List[Dict[str, Any]]:
    """
    Augment each row with:
      1) piece-type selection stats
      2) engine-like metrics from stored stockfish top moves
      3) tactical / positional heuristics
      4) sequence probability metrics using neighboring rows in same game

    Notes / limitations:
    - Piece-type probability mass is exact over the stored top-k if `prob` is present.
      If `prob` is missing, it falls back to softmax(logit) over the stored top-k only.
    - Exact cp gap for pred_uci or chosen_uci is only available if that move is present
      in stockfish['sf_moves_returned'].
    - Tactical / positional labels are heuristic, not engine-ground-truth.
    """

    def safe_float(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def safe_log_prob_from_prob(p: Any, eps: float = 1e-12) -> float:
        p = safe_float(p, 0.0)
        p = max(p, eps)
        return math.log(p)

    def board_from_fen(fen: str) -> Optional[chess.Board]:
        try:
            return chess.Board(fen)
        except Exception:
            return None

    def parse_move(board: Optional[chess.Board], uci: Optional[str]) -> Optional[chess.Move]:
        if board is None or not uci:
            return None
        try:
            mv = chess.Move.from_uci(uci)
            return mv
        except Exception:
            return None

    def move_piece_type_name(board: Optional[chess.Board], uci: Optional[str]) -> Optional[str]:
        mv = parse_move(board, uci)
        if board is None or mv is None:
            return None
        piece = board.piece_at(mv.from_square)
        if piece is None:
            return None
        return PIECE_TYPE_NAMES.get(piece.piece_type)

    def is_capture(board: Optional[chess.Board], mv: Optional[chess.Move]) -> bool:
        if board is None or mv is None:
            return False
        try:
            return board.is_capture(mv)
        except Exception:
            return False

    def is_check(board: Optional[chess.Board], mv: Optional[chess.Move]) -> bool:
        if board is None or mv is None:
            return False
        try:
            return board.gives_check(mv)
        except Exception:
            return False

    def is_promotion(mv: Optional[chess.Move]) -> bool:
        return mv is not None and mv.promotion is not None

    def is_castle(board: Optional[chess.Board], mv: Optional[chess.Move]) -> bool:
        if board is None or mv is None:
            return False
        try:
            return board.is_castling(mv)
        except Exception:
            return False

    def is_minor_piece_development(board: Optional[chess.Board], mv: Optional[chess.Move]) -> bool:
        if board is None or mv is None:
            return False
        piece = board.piece_at(mv.from_square)
        if piece is None:
            return False
        if piece.piece_type not in (chess.KNIGHT, chess.BISHOP):
            return False

        from_rank = chess.square_rank(mv.from_square)
        if piece.color == chess.WHITE:
            return from_rank == 0
        return from_rank == 7

    def is_central_pawn_push(board: Optional[chess.Board], mv: Optional[chess.Move]) -> bool:
        if board is None or mv is None:
            return False
        piece = board.piece_at(mv.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
        to_file = chess.square_file(mv.to_square)
        return to_file in (2, 3, 4, 5)  # c,d,e,f files
    
    def is_positional_heuristic(board: Optional[chess.Board], mv: Optional[chess.Move]) -> bool:
        if board is None or mv is None:
            return False

        piece = board.piece_at(mv.from_square)
        if piece is None:
            return False

        # Tactical overrides
        if board.is_capture(mv) or board.gives_check(mv) or mv.promotion is not None:
            return False

        # Quiet moves are candidates
        if board.is_castling(mv):
            return True

        # Minor piece development
        if is_minor_piece_development(board, mv):
            return True

        # Central pawn push
        if is_central_pawn_push(board, mv):
            return True

        # Quiet bishop / queen / rook repositioning often positional
        if piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            return True

        # Non-capturing pawn moves in opening/middlegame are often positional/prophylactic
        if piece.piece_type == chess.PAWN:
            return True

        # Quiet knight moves that are not checks/captures are often positional too
        if piece.piece_type == chess.KNIGHT:
            return True

        return False

    def classify_move_style(board: Optional[chess.Board], uci: Optional[str]) -> Dict[str, Any]:
        mv = parse_move(board, uci)
        if board is None or mv is None:
            return {
                "piece_type": None,
                "is_legal_on_board": False,
                "is_capture": False,
                "is_check": False,
                "is_promotion": False,
                "is_castle": False,
                "is_tactical": False,
                "is_positional": False,
                "is_quiet": False,
            }

        legal = mv in board.legal_moves
        piece_type = move_piece_type_name(board, uci)
        cap = is_capture(board, mv)
        chk = is_check(board, mv)
        promo = is_promotion(mv)
        castle = is_castle(board, mv)

        tactical = cap or chk or promo
        quiet = (not cap) and (not chk) and (not promo)

        positional = is_positional_heuristic(board, mv)

        return {
            "piece_type": piece_type,
            "is_legal_on_board": legal,
            "is_capture": cap,
            "is_check": chk,
            "is_promotion": promo,
            "is_castle": castle,
            "is_tactical": tactical,
            "is_positional": positional,
            "is_quiet": quiet,
        }

    def topk_piece_mass(top_moves: Any, board: Optional[chess.Board]) -> Dict[str, Optional[float]]:
        """
        Computes piece-type probability mass using stored top-k moves.

        Priority:
            1. Use exact `prob` if available
            2. Otherwise reconstruct from logits

        Returns normalized mass over the provided top-k moves.
        """
        out = {f"piece_mass_topk_{p}": None for p in PIECE_NAME_LIST}
        out["topk_mass_total"] = None

        if board is None or not isinstance(top_moves, list) or len(top_moves) == 0:
            return out

        probs: List[Optional[float]] = []
        logits: List[float] = []
        piece_types: List[Optional[str]] = []

        has_any_prob = False
        for x in top_moves:
            if not isinstance(x, dict):
                continue
            uci = x.get("uci")
            piece_name = move_piece_type_name(board, uci)
            piece_types.append(piece_name)

            p = x.get("prob")
            if p is not None:
                probs.append(float(p))
                logits.append(0.0)
                has_any_prob = True
            else:
                probs.append(None)
                logits.append(float(x.get("logit", 0.0)))

        if not piece_types:
            return out

        if has_any_prob:
            weights = [p if p is not None else 0.0 for p in probs]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                return out
        else:
            if not logits:
                return out
            m = max(logits)
            exps = [math.exp(z - m) for z in logits]
            denom = sum(exps)
            if denom <= 0.0:
                return out
            weights = [e / denom for e in exps]

        mass = {p: 0.0 for p in PIECE_NAME_LIST}
        for w, piece_name in zip(weights, piece_types):
            if piece_name in mass:
                mass[piece_name] += w

        for p in PIECE_NAME_LIST:
            out[f"piece_mass_topk_{p}"] = mass[p]
        out["topk_mass_total"] = 1.0
        return out

    def stockfish_move_map(stockfish_blob: Any) -> Dict[str, Any]:
        best_cp_all = None
        sf_map: Dict[str, float] = {}
        sf_ordered_uci: List[str] = []

        if isinstance(stockfish_blob, dict):
            best_cp_all = stockfish_blob.get("best_cp_all")
            moves = stockfish_blob.get("sf_moves_returned", [])
            if isinstance(moves, list):
                for item in moves:
                    if (
                        isinstance(item, (list, tuple))
                        and len(item) >= 2
                        and isinstance(item[0], str)
                    ):
                        uci = item[0]
                        cp = safe_float(item[1], default=float("nan"))
                        sf_map[uci] = cp
                        sf_ordered_uci.append(uci)

        return {
            "best_cp_all": best_cp_all,
            "sf_map": sf_map,
            "sf_ordered_uci": sf_ordered_uci,
            "engine_best_uci": sf_ordered_uci[0] if sf_ordered_uci else None,
        }

    def add_engine_metrics(row: Dict[str, Any], board: Optional[chess.Board]) -> None:
        sf_info = stockfish_move_map(row.get("stockfish"))
        sf_map = sf_info["sf_map"]
        sf_ordered_uci = sf_info["sf_ordered_uci"]
        best_cp_all = sf_info["best_cp_all"]
        engine_best_uci = sf_info["engine_best_uci"]

        pred_uci = row.get("pred_uci")
        chosen_uci = row.get("chosen_uci")

        row["engine_best_uci"] = engine_best_uci
        row["engine_top1_uci"] = engine_best_uci

        for k in (1, 2, 3, 4, 5, 10):
            row[f"pred_in_engine_top{k}"] = float(pred_uci in sf_ordered_uci[:k]) if pred_uci else 0.0
            row[f"chosen_in_engine_top{k}"] = float(chosen_uci in sf_ordered_uci[:k]) if chosen_uci else 0.0

        row["pred_matches_engine_best"] = float(pred_uci == engine_best_uci) if pred_uci and engine_best_uci else 0.0
        row["chosen_matches_engine_best"] = float(chosen_uci == engine_best_uci) if chosen_uci and engine_best_uci else 0.0

        pred_cp = sf_map.get(pred_uci)
        chosen_cp = sf_map.get(chosen_uci)

        row["pred_engine_cp"] = pred_cp if pred_cp is not None else None
        row["chosen_engine_cp"] = chosen_cp if chosen_cp is not None else None

        if pred_cp is not None and best_cp_all is not None:
            pred_gap = abs(float(best_cp_all) - float(pred_cp))
            row["pred_cp_gap_to_engine_best"] = pred_gap
            for x in cp_gap_thresholds:
                row[f"pred_cp_gap_le_{x}"] = float(pred_gap <= x)
        else:
            row["pred_cp_gap_to_engine_best"] = None
            for x in cp_gap_thresholds:
                row[f"pred_cp_gap_le_{x}"] = None

        if chosen_cp is not None and best_cp_all is not None:
            chosen_gap = abs(float(best_cp_all) - float(chosen_cp))
            row["chosen_cp_gap_to_engine_best"] = chosen_gap
            for x in cp_gap_thresholds:
                row[f"chosen_cp_gap_le_{x}"] = float(chosen_gap <= x)
        else:
            row["chosen_cp_gap_to_engine_best"] = None
            for x in cp_gap_thresholds:
                row[f"chosen_cp_gap_le_{x}"] = None

        row["pred_more_engine_like_than_chosen_top1"] = float(
            row["pred_matches_engine_best"] > row["chosen_matches_engine_best"]
        )
        row["pred_more_engine_like_than_chosen_top3"] = float(
            row["pred_in_engine_top3"] > row["chosen_in_engine_top3"]
        )
        row["pred_more_engine_like_than_chosen_top5"] = float(
            row["pred_in_engine_top5"] > row["chosen_in_engine_top5"]
        )

        engine_style = classify_move_style(board, engine_best_uci)
        row["engine_best_piece_type"] = engine_style["piece_type"]
        row["engine_best_is_tactical"] = float(engine_style["is_tactical"])
        row["engine_best_is_positional"] = float(engine_style["is_positional"])
        row["engine_best_is_capture"] = float(engine_style["is_capture"])
        row["engine_best_is_check"] = float(engine_style["is_check"])

    def add_piece_type_metrics(row: Dict[str, Any], board: Optional[chess.Board]) -> None:
        chosen_style = classify_move_style(board, row.get("chosen_uci"))
        pred_style = classify_move_style(board, row.get("pred_uci"))

        ref_top1_uci = None
        ref_top = row.get("top_max10_ref_w_logits")
        if isinstance(ref_top, list) and ref_top:
            ref_top1_uci = ref_top[0].get("uci")

        ref_style = classify_move_style(board, ref_top1_uci)

        row["chosen_piece_type"] = chosen_style["piece_type"]
        row["pred_piece_type_pi"] = pred_style["piece_type"]
        row["top1_piece_type_ref"] = ref_style["piece_type"]

        pi_mass = topk_piece_mass(row.get("top_max10_pi_w_logits"), board)
        ref_mass = topk_piece_mass(row.get("top_max10_ref_w_logits"), board)

        for p in PIECE_NAME_LIST:
            row[f"chosen_is_{p}"] = float(chosen_style["piece_type"] == p)
            row[f"pred_pi_is_{p}"] = float(pred_style["piece_type"] == p)
            row[f"pred_ref_is_{p}"] = float(ref_style["piece_type"] == p)

            row[f"pi_top1_matches_player_piece_type_{p}"] = float(
                (chosen_style["piece_type"] == p) and (pred_style["piece_type"] == p)
            )
            row[f"ref_top1_matches_player_piece_type_{p}"] = float(
                (chosen_style["piece_type"] == p) and (ref_style["piece_type"] == p)
            )

            row[f"pi_top1_selects_{p}_when_player_not_{p}"] = float(
                (chosen_style["piece_type"] != p) and (pred_style["piece_type"] == p)
            )
            row[f"ref_top1_selects_{p}_when_player_not_{p}"] = float(
                (chosen_style["piece_type"] != p) and (ref_style["piece_type"] == p)
            )

            row[f"pi_topk_piece_mass_{p}"] = pi_mass[f"piece_mass_topk_{p}"]
            row[f"ref_topk_piece_mass_{p}"] = ref_mass[f"piece_mass_topk_{p}"]

            chosen_is_p = (chosen_style["piece_type"] == p)
            row[f"pi_topk_piece_mass_{p}_when_player_{p}"] = (
                pi_mass[f"piece_mass_topk_{p}"] if chosen_is_p else None
            )
            row[f"pi_topk_piece_mass_{p}_when_player_not_{p}"] = (
                pi_mass[f"piece_mass_topk_{p}"] if not chosen_is_p else None
            )
            row[f"ref_topk_piece_mass_{p}_when_player_{p}"] = (
                ref_mass[f"piece_mass_topk_{p}"] if chosen_is_p else None
            )
            row[f"ref_topk_piece_mass_{p}_when_player_not_{p}"] = (
                ref_mass[f"piece_mass_topk_{p}"] if not chosen_is_p else None
            )

    def add_style_heuristics(row: Dict[str, Any], board: Optional[chess.Board]) -> None:
        chosen_style = classify_move_style(board, row.get("chosen_uci"))
        pred_style = classify_move_style(board, row.get("pred_uci"))
        engine_best_uci = row.get("engine_best_uci")
        engine_style = classify_move_style(board, engine_best_uci)

        prefixes = {
            "chosen": chosen_style,
            "pred_pi": pred_style,
            "engine_best": engine_style,
        }

        for prefix, st in prefixes.items():
            row[f"{prefix}_is_capture"] = float(st["is_capture"])
            row[f"{prefix}_is_check"] = float(st["is_check"])
            row[f"{prefix}_is_promotion"] = float(st["is_promotion"])
            row[f"{prefix}_is_castle"] = float(st["is_castle"])
            row[f"{prefix}_is_tactical"] = float(st["is_tactical"])
            row[f"{prefix}_is_positional"] = float(st["is_positional"])
            row[f"{prefix}_is_quiet"] = float(st["is_quiet"])

        row["pi_matches_player_tactical"] = float(
            chosen_style["is_tactical"] and pred_style["is_tactical"]
        )
        row["pi_matches_player_positional"] = float(
            chosen_style["is_positional"] and pred_style["is_positional"]
        )
        row["ref_tactical_proxy_from_engine_best"] = float(engine_style["is_tactical"])
        row["ref_positional_proxy_from_engine_best"] = float(engine_style["is_positional"])

        row["missed_engine_tactic_by_player"] = float(
            engine_style["is_tactical"] and not chosen_style["is_tactical"]
        )
        row["missed_engine_tactic_by_pi_top1"] = float(
            engine_style["is_tactical"] and not pred_style["is_tactical"]
        )

        row["missed_engine_positional_by_player"] = float(
            engine_style["is_positional"] and not chosen_style["is_positional"]
        )
        row["missed_engine_positional_by_pi_top1"] = float(
            engine_style["is_positional"] and not pred_style["is_positional"]
        )

        row["player_vs_engine_style_agree_tactical"] = float(
            chosen_style["is_tactical"] == engine_style["is_tactical"]
        )
        row["player_vs_pi_style_agree_tactical"] = float(
            chosen_style["is_tactical"] == pred_style["is_tactical"]
        )
        row["player_vs_engine_style_agree_positional"] = float(
            chosen_style["is_positional"] == engine_style["is_positional"]
        )
        row["player_vs_pi_style_agree_positional"] = float(
            chosen_style["is_positional"] == pred_style["is_positional"]
        )

    def finalize_sequence_metrics_for_game(game_rows: List[Dict[str, Any]]) -> None:
        if not game_rows:
            return

        for i, row in enumerate(game_rows):
            p_pi = safe_float(row.get("p_chosen_pi"), 0.0)
            p_ref = safe_float(row.get("p_chosen_ref"), 0.0)

            logp_pi = safe_log_prob_from_prob(p_pi)
            logp_ref = safe_log_prob_from_prob(p_ref)

            row["_tmp_logp_pi"] = logp_pi
            row["_tmp_logp_ref"] = logp_ref

            for w in seq_windows:
                start = max(0, i - w + 1)
                window = game_rows[start : i + 1]

                sum_log_pi = sum(r["_tmp_logp_pi"] for r in window)
                sum_log_ref = sum(r["_tmp_logp_ref"] for r in window)

                row[f"chosen_seq_logprob_pi_last{w}"] = sum_log_pi
                row[f"chosen_seq_logprob_ref_last{w}"] = sum_log_ref
                row[f"chosen_seq_prob_pi_last{w}"] = math.exp(sum_log_pi)
                row[f"chosen_seq_prob_ref_last{w}"] = math.exp(sum_log_ref)
                row[f"chosen_seq_logprob_improve_pi_minus_ref_last{w}"] = sum_log_pi - sum_log_ref

        n = len(game_rows)
        for i, row in enumerate(game_rows):
            cur_tac = bool(row.get("chosen_is_tactical", 0.0))
            prev_tac = bool(game_rows[i - 1].get("chosen_is_tactical", 0.0)) if i > 0 else False
            next_tac = bool(game_rows[i + 1].get("chosen_is_tactical", 0.0)) if i + 1 < n else False

            cur_pos = bool(row.get("chosen_is_positional", 0.0))
            prev_pos = bool(game_rows[i - 1].get("chosen_is_positional", 0.0)) if i > 0 else False
            next_pos = bool(game_rows[i + 1].get("chosen_is_positional", 0.0)) if i + 1 < n else False

            row["chosen_tactic_starts"] = float(cur_tac and not prev_tac)
            row["chosen_tactic_completes"] = float(cur_tac and not next_tac and prev_tac)
            row["chosen_positional_starts"] = float(cur_pos and not prev_pos)
            row["chosen_positional_completes"] = float(cur_pos and not next_pos and prev_pos)

            for w in seq_windows:
                if cur_tac:
                    row[f"tactical_seq_logprob_pi_last{w}"] = row[f"chosen_seq_logprob_pi_last{w}"]
                    row[f"tactical_seq_logprob_ref_last{w}"] = row[f"chosen_seq_logprob_ref_last{w}"]
                    row[f"tactical_seq_prob_pi_last{w}"] = row[f"chosen_seq_prob_pi_last{w}"]
                    row[f"tactical_seq_prob_ref_last{w}"] = row[f"chosen_seq_prob_ref_last{w}"]
                else:
                    row[f"tactical_seq_logprob_pi_last{w}"] = None
                    row[f"tactical_seq_logprob_ref_last{w}"] = None
                    row[f"tactical_seq_prob_pi_last{w}"] = None
                    row[f"tactical_seq_prob_ref_last{w}"] = None

                if cur_pos:
                    row[f"positional_seq_logprob_pi_last{w}"] = row[f"chosen_seq_logprob_pi_last{w}"]
                    row[f"positional_seq_logprob_ref_last{w}"] = row[f"chosen_seq_logprob_ref_last{w}"]
                    row[f"positional_seq_prob_pi_last{w}"] = row[f"chosen_seq_prob_pi_last{w}"]
                    row[f"positional_seq_prob_ref_last{w}"] = row[f"chosen_seq_prob_ref_last{w}"]
                else:
                    row[f"positional_seq_logprob_pi_last{w}"] = None
                    row[f"positional_seq_logprob_ref_last{w}"] = None
                    row[f"positional_seq_prob_pi_last{w}"] = None
                    row[f"positional_seq_prob_ref_last{w}"] = None

        for row in game_rows:
            row.pop("_tmp_logp_pi", None)
            row.pop("_tmp_logp_ref", None)

    current_gid = None
    game_rows: List[Dict[str, Any]] = []

    for row in per_rows_sorted:
        gid = row.get("game_id")
        if current_gid is None:
            current_gid = gid

        if gid != current_gid:
            finalize_sequence_metrics_for_game(game_rows)
            game_rows = []
            current_gid = gid

        board = board_from_fen(row.get("fen", ""))

        add_piece_type_metrics(row, board)
        add_engine_metrics(row, board)
        add_style_heuristics(row, board)

        row["pi_beats_ref_on_gap_improve"] = float(safe_float(row.get("gap_improve"), 0.0) > 0.0)
        row["pi_lower_entropy_than_ref"] = float(
            safe_float(row.get("entropy_pi"), 0.0) < safe_float(row.get("entropy_ref"), 0.0)
        )
        row["pi_higher_entropy_than_ref"] = float(
            safe_float(row.get("entropy_pi"), 0.0) > safe_float(row.get("entropy_ref"), 0.0)
        )

        game_rows.append(row)

    if game_rows:
        finalize_sequence_metrics_for_game(game_rows)

    return per_rows_sorted

def uci_to_vocab_index(all_moves_dict: Dict[str, int], fen: str, uci: str) -> int:
    side = fen.split(" ")[1]
    uci_eff = mirror_move(uci) if side == "b" else uci
    return int(all_moves_dict.get(uci_eff, -1))

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
    # Example usage: python ./src/grandmaster_dpo/eval/single_gm/eval_sft_and_dpo_w_style_sim_utility_weight_maia2.py --gm_name caruana --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/eval_results_twic --model_dir ./final_experiments_for_paper/experiment1/trained_models_twic
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
        full_name = f"sft_and_dpo_w_style_sim_utility_weight_beta={beta:.2f}_dpo_loss_weight={dpo_loss_weight:.2f}_style_cp_scale={args.style_cp_scale:.2f}_style_piece_bonus={args.style_piece_bonus:.2f}_style_positional_bonus={args.style_positional_bonus:.2f}_style_tau={style_tau:.2f}"
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
                    compute_style_score(
                        fen=fen,
                        chosen_uci=ch,
                        rejected_uci=rj,
                        chosen_cp=ch_cp,
                        rejected_cp=rj_cp,
                        cp_scale=args.style_cp_scale,
                        piece_bonus=args.style_piece_bonus,
                        positional_bonus=args.style_positional_bonus,
                    )
                    for fen, ch, rj, ch_cp, rj_cp, _, _, _, _ in batch_meta_data
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
