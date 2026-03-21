from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import statistics
import chess

import torch
from torch.utils.data import DataLoader, Dataset

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move


# ----------------------------
# Dataset
# ----------------------------


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

        self.game_id_and_ply_to_prev_10_plys = {}
        self.game_id_and_ply_to_fut_10_plys = {}

        def create_window_item(rows, index, target_game):
            if index < 0:
                return None 
            elif index >= len(rows):
                return None
            else:
                if rows[index]["meta"]["game_header_hash"] != target_game:
                    return None
                return rows[index]

        for i, r in enumerate(self.rows):
            hash_key = f'{r["meta"]["game_header_hash"]}_{r["meta"]["ply_idx"]}'
            self.game_id_and_ply_to_prev_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i-10, i)]
            self.game_id_and_ply_to_fut_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i+1, i+11)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        p = r.get("prompt", {}) or {}
        meta = r.get("meta", {}) or {}

        # Correct location: meta['game_header_hash'] (not top-level).
        gh = meta.get("game_header_hash")
        game_id = str(gh)

        if self.debug and idx < 3:
            print(f"r keys: {r.keys()}")
            print(f"meta keys: {meta.keys()}")
            print(f"p keys: {p.keys()}")
            print(f"meta.game_header_hash: {gh!r}")
            print(f"computed game_id: {game_id}")

        # Minimal required keys + metadata keys used by eval
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
            "meta": meta,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Collate *all* fields your eval loop may read.
    """
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
        "meta": [],
    }
    for b in batch:
        for k in out:
            out[k].append(b.get(k))
    return out


# ----------------------------
# Helpers (match training)
# ----------------------------

def key_game_ply(meta: Dict[str, Any]) -> str:
    return f'{meta["game_header_hash"]}_{meta["ply_idx"]}'

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

def is_chosen_in_top_10_engine(meta: dict, uci: str) -> float:
    sf_moves = meta["stockfish"]["sf_moves_returned"]
    for sf_uci, cp in sf_moves:
        if sf_uci == uci:
            return True
        
    return False




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

def entropy_from_logits(masked_logits: torch.Tensor) -> torch.Tensor:
    """
    masked_logits: [B, V] with illegal moves already masked to -inf or very negative
    returns: [B] entropy for each example in the batch
    """
    logp = torch.nn.functional.log_softmax(masked_logits, dim=-1)   # [B, V]
    p = logp.exp()                                # [B, V]
    entropy = -(p * logp).sum(dim=-1)            # [B]
    return entropy


@torch.no_grad()
def probe_opening_distributions_from_policy(
    policy: torch.nn.Module,
    *,
    maia_type: str,
    device: torch.device,
    all_moves: List[str],
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    elo_self: int = 2800,
    elo_oppo: int = 2800,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "white_first_move_probs": {uci: prob, ...},
        "black_reply_probs_cond_on_white": {
            white_uci: {black_uci: prob, ...},
            ...
        },
        "meta": {...}
      }

    Notes:
      - Uses fine-tuned policy logits (legal-masked) in *canonical* opening states.
      - White distribution is computed at the initial position.
      - Black conditional distributions are computed at positions after each probed white first move.
      - Probabilities are over a curated move set; we also return 'other_mass' remainder.
    """

    def apply_legal_mask_row(logits_row: torch.Tensor, legal_row: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(logits_row.dtype).min
        return torch.where(legal_row > 0, logits_row, torch.full_like(logits_row, neg_inf))

    def uci_to_vocab_index_local(fen: str, uci: str) -> int:
        side = fen.split(" ")[1]  # 'w' or 'b'
        # Maia vocab is stored in "white perspective".
        # For real black moves, map them into that space using mirror_move (involution).
        uci_eff = mirror_move(uci) if side == "b" else uci
        return int(all_moves_dict.get(uci_eff, -1))

    def probs_for_ucis(fen: str, logits_masked_row: torch.Tensor, ucis: List[str]) -> Tuple[Dict[str, float], float]:
        # temperature
        if temperature <= 0:
            probs = torch.softmax(logits_masked_row, dim=-1)
        else:
            probs = torch.softmax(logits_masked_row / temperature, dim=-1)

        out: Dict[str, float] = {}
        used = 0.0
        for u in ucis:
            j = uci_to_vocab_index_local(fen, u)
            p = float(probs[j].item()) if j >= 0 else 0.0
            out[u] = p
            used += p
        other = max(0.0, 1.0 - used)
        return out, other

    # ---- canonical states we probe ----

    # Initial position (white to move)
    start_fen = chess.STARTING_FEN  # "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # White first moves we probe (you can expand this list)
    white_first_moves = [
        "e2e4", "d2d4", "c2c4", "g1f3", "g2g3", "b2b3", "f2f4", "b2b4", "a2a4",
    ]

    # Black reply moves we probe (common replies; expand as you like)
    black_replies = [
        "c7c5", "e7e5", "e7e6", "c7c6", "d7d5", "g8f6", "g7g6", "d7d6",
    ]

    # ---- helper to run model on one fen ----
    def logits_masked_for_fen(fen: str) -> torch.Tensor:
        bi, es_cat, eo_cat, lm = inference.preprocessing(
            fen, int(elo_self), int(elo_oppo), elo_dict, all_moves_dict
        )
        board_input = bi.unsqueeze(0).to(device)            # [1,...]
        legal = lm.unsqueeze(0).to(device)                  # [1,V]
        es_t = torch.tensor([int(es_cat)], device=device).long()
        eo_t = torch.tensor([int(eo_cat)], device=device).long()

        logits, _, _ = policy(board_input, es_t, eo_t)       # [1,V]
        logits = logits.squeeze(0)                           # [V]
        legal = legal.squeeze(0)                             # [V]
        return apply_legal_mask_row(logits, legal)

    # ---- compute white distribution ----
    logits_start = logits_masked_for_fen(start_fen)
    white_probs, white_other = probs_for_ucis(start_fen, logits_start, white_first_moves)

    

    # ---- compute black conditional distributions ----
    black_cond: Dict[str, Any] = {}
    for w in white_first_moves:
        b = chess.Board(start_fen)
        b.push_uci(w)
        fen_b = b.fen()

        logits_b = logits_masked_for_fen(fen_b)
        probs_b, other_b = probs_for_ucis(fen_b, logits_b, black_replies)

        black_cond[w] = {
            "fen_after_white": fen_b,
            "black_reply_probs": probs_b,
            "other_mass": other_b,
        }

    return {
        "white_first_move_probs": white_probs,
        "white_other_mass": white_other,
        "black_reply_probs_cond_on_white": black_cond,
        "meta": {
            "elo_self": elo_self,
            "elo_oppo": elo_oppo,
            "temperature": temperature,
            "maia_type": maia_type,
            "white_moves_probed": white_first_moves,
            "black_replies_probed": black_replies,
        },
    }

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
        bi, es_cat, eo_cat, lm = inference.preprocessing(fen, int(es), int(eo), elo_dict, all_moves_dict)
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


def uci_to_vocab_index(all_moves_dict: Dict[str, int], fen: str, uci: str) -> int:
    side = fen.split(" ")[1]
    uci_eff = mirror_move(uci) if side == "b" else uci
    return int(all_moves_dict.get(uci_eff, -1))

def gather_logprob(logits_masked: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    # logits_masked already has illegal moves at -inf; safe to log_softmax
    logp_all = torch.log_softmax(logits_masked, dim=-1)
    safe_idx = idxs.clamp(min=0)
    gathered = logp_all.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)
    gathered = torch.where(idxs >= 0, gathered, torch.full_like(gathered, -1e9))
    return gathered

@torch.no_grad()
def kl_policy_base_from_logits(logits_pi_masked: torch.Tensor, logits_ref_masked: torch.Tensor) -> torch.Tensor:
    # KL( pi || ref ) over vocab
    p = torch.softmax(logits_pi_masked, dim=-1)
    logp = torch.log_softmax(logits_pi_masked, dim=-1)
    logq = torch.log_softmax(logits_ref_masked, dim=-1)
    kl = (p * (logp - logq)).sum(dim=-1)  # [B]
    return kl


@torch.no_grad()
def top1_accuracy(logits_masked: torch.Tensor, fens: List[str], all_moves_dict: Dict[str, int], chosen_uci: List[str]) -> torch.Tensor:
    # top1 index in vocab
    top_idx = logits_masked.argmax(dim=-1)  # [B]
    chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, uci) for fen, uci in zip(fens, chosen_uci)],
                              device=logits_masked.device, dtype=torch.long)
    return (top_idx == chosen_idx).float()  # [B]


@torch.no_grad()
def chosen_probability(logits_masked: torch.Tensor, fens: List[str], all_moves_dict: Dict[str, int], chosen_uci: List[str]) -> torch.Tensor:
    probs = torch.softmax(logits_masked, dim=-1)
    chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, uci) for fen, uci in zip(fens, chosen_uci)],
                              device=logits_masked.device, dtype=torch.long)
    safe_idx = chosen_idx.clamp(min=0)
    p = probs.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)
    p = torch.where(chosen_idx >= 0, p, torch.zeros_like(p))
    return p

@torch.no_grad()
def chosen_rank(logits_masked: torch.Tensor, chosen_idx: torch.Tensor) -> torch.Tensor:
    """Rank of chosen move under logits (1 is best). O(B*V) but OK for eval sizes."""
    chosen_idx_safe = chosen_idx.clamp(min=0)
    chosen_logit = logits_masked.gather(1, chosen_idx_safe.view(-1, 1)).squeeze(1)
    greater = (logits_masked > chosen_logit.unsqueeze(1)).sum(dim=1)
    rank = greater + 1
    rank = torch.where(chosen_idx >= 0, rank, torch.full_like(rank, 10**9))
    return rank


@torch.no_grad()
def hit_at_k(logits_masked: torch.Tensor, chosen_idx: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros((logits_masked.size(0),), device=logits_masked.device)
    topk = torch.topk(logits_masked, k=k, dim=-1).indices
    chosen_idx_safe = chosen_idx.clamp(min=0).view(-1, 1)
    hit = (topk == chosen_idx_safe).any(dim=1).float()
    hit = torch.where(chosen_idx >= 0, hit, torch.zeros_like(hit))
    return hit

# ----------------------------
# Stats helpers
# ----------------------------

def fen_to_ply(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return 2 * (fullmove - 1) + (1 if side == "b" else 0)

def ply_to_phase(ply: int) -> str:
    if ply < 20:
        return "opening"
    if ply < 60:
        return "middlegame"
    return "endgame"

def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))

import math
import random
def quantiles(xs: List[float], ps=(0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    if not xs:
        return {f"p{int(p*100)}": float("nan") for p in ps}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    out = {}
    for p in ps:
        k = max(0, min(n - 1, int(math.ceil(p * n) - 1)))
        out[f"p{int(p*100)}"] = float(xs_sorted[k])
    return out

def bootstrap_ci(values: List[float], stat_fn, n_boot=2000, alpha=0.05, seed=0) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rnd = random.Random(seed)
    base = stat_fn(values)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean": float(base), "lo": float(lo), "hi": float(hi)}

def cluster_bootstrap_ci(
    per_rows: List[Dict[str, Any]],
    key: str,
    metric_field: str,
    stat_fn,
    n_boot=2000,
    alpha=0.05,
    seed=0,
) -> Optional[Dict[str, float]]:
    """
    Cluster bootstrap over key (e.g. game_id). Draw games with replacement,
    include all their rows, compute stat over metric_field.
    """
    # group
    groups = defaultdict(list)
    for r in per_rows:
        gid = r.get(key, "")
        if gid:
            groups[gid].append(r)

    if len(groups) < 2:
        return None

    gids = list(groups.keys())
    rnd = random.Random(seed)

    def stat_from_rows(rows: List[Dict[str, Any]]) -> float:
        xs = [float(rr[metric_field]) for rr in rows]
        return stat_fn(xs)

    base = stat_from_rows(per_rows)
    boots = []
    G = len(gids)
    for _ in range(n_boot):
        sampled = [groups[gids[rnd.randrange(G)]] for _ in range(G)]
        flat = [rr for grp in sampled for rr in grp]
        boots.append(stat_from_rows(flat))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean": float(base), "lo": float(lo), "hi": float(hi), "n_clusters": len(groups)}

# ----------------------------
# Opening family (coarse heuristic)
# ----------------------------

def coarse_opening_family_from_prefix(prefix_uci: List[str]) -> str:
    """
    Very coarse: classify based on first black reply or early structure.
    This is a placeholder until you wire ECO book matching.
    """
    if len(prefix_uci) < 2:
        return "Unknown"
    black_reply = prefix_uci[1]
    # replies to 1.e4
    if black_reply == "c7c5":
        return "Sicilian"
    if black_reply == "e7e5":
        return "Open Game (1...e5)"
    if black_reply == "c7c6":
        return "Caro-Kann"
    if black_reply == "e7e6":
        return "French / e6"
    # replies to 1.d4 (common)
    if black_reply == "d7d5":
        return "Queen's Pawn (1...d5)"
    if black_reply == "g8f6":
        return "Indian Defense (1...Nf6)"
    return "Other"

def vocab_index_to_uci(all_moves: List[str], fen: str, idx: int) -> str:
    """
    Convert a vocab index back to a real UCI move for this position.
    Handles Maia's mirroring convention for black-to-move.
    """
    if idx < 0 or idx >= len(all_moves):
        return ""

    uci_eff = all_moves[idx]  # Maia vocab is in "white perspective"
    side = fen.split(" ")[1]  # 'w' or 'b'
    return mirror_move(uci_eff) if side == "b" else uci_eff

# ----------------------------
# Main eval
# ----------------------------

def run_eval(jsonl_path: str, 
             pt_path: str, 
             out_dir_base: str, 
             gm_name: str, 
             device: str, 
             maia_type: str, 
             opening_probe_policy_filename: str,
             n_boot: int,
             batch_size: int,
             split_name: str,
             agg_results_filename: str,
             eval_results_extended_filename: str,
             eval_results_summary_csv_filename: str,
             per_row_filename: str,
             supplied_loss_function
    ) -> None:
    jsonl = Path(jsonl_path)

    policy_pt = Path(pt_path)
    out_dir = Path(f"{out_dir_base}/{gm_name}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_str(device)

    # Build vocab + elo dict deterministically (avoid prepare() ordering issues)
    prep = inference.prepare()
    all_moves_dict, elo_dict, all_moves_dict_reversed = prep
    all_moves = [None] * len(all_moves_dict)
    for mv, idx in all_moves_dict.items():
        all_moves[idx] = mv

    # if you need all_moves as a list:
    all_moves = [None] * len(all_moves_dict_reversed)
    for idx, uci in all_moves_dict_reversed.items():
        all_moves[idx] = uci

    # Load base twice; then load policy weights into one
    base = maia_model.from_pretrained(type=maia_type, device=str(device)).to(device)
    policy = maia_model.from_pretrained(type=maia_type, device=str(device)).to(device)

    sd = torch.load(policy_pt, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print("missing", len(missing))
    print("unexpected", len(unexpected))
    print("sample missing:", missing[:20])
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (showing 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}")

    base.eval()
    policy.eval()

    opening_probe = probe_opening_distributions_from_policy(
        policy,
        maia_type=maia_type,
        device=device,
        all_moves=all_moves,
        all_moves_dict=all_moves_dict,
        elo_dict=elo_dict,
        elo_self=2800,
        elo_oppo=2800,
        temperature=1.0,
    )
    out_dir.joinpath(opening_probe_policy_filename).write_text(json.dumps(opening_probe, indent=2))
    print(f"Opening probe saved to {out_dir.joinpath(opening_probe_policy_filename)}")

    opening_probe = probe_opening_distributions_from_policy(
        base,
        maia_type=maia_type,
        device=device,
        all_moves=all_moves,
        all_moves_dict=all_moves_dict,
        elo_dict=elo_dict,
        elo_self=2800,
        elo_oppo=2800,
        temperature=1.0,
    )
    out_dir.joinpath(f"opening_probe_base.json").write_text(json.dumps(opening_probe, indent=2))
    print(f"Opening probe saved to {out_dir.joinpath(f'opening_probe_base.json')}")

    ds = DpoPairs(jsonl)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    # Aggregate metrics
    n = 0
    sum_loss = 0.0

    sum_pi_gap = 0.0
    sum_ref_gap = 0.0
    sum_gap_improvement = 0.0

    sum_top1_pi = 0.0
    sum_top1_ref = 0.0

    sum_p_chosen_pi = 0.0
    sum_p_chosen_ref = 0.0

    sum_kl = 0.0

    sum_ent_pi = 0.0
    sum_ent_ref = 0.0

    # NEW: per-row metrics store
    per_rows: List[Dict[str, Any]] = []

    # NEW: phase buckets for tails
    phase_buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    # NEW: opening family distribution (per game)
    opening_by_game: Dict[str, str] = {}

    # NEW: empirical player opening distribution from actual played games
    opening_prefix_by_game: Dict[str, List[str]] = {}


    for batch in loader:
        fens = batch["fen"]
        es = batch["elo_self"]
        eo = batch["elo_oppo"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        bs = len(fens)
        game_ids = batch["game_id"]
        ply_idxs = batch["ply_idx"]
        meta_list = batch["meta"]
        opening_prefixes = batch["opening_prefix_uci_20"]

        board_input, legal_moves, es_t, eo_t = batch_preprocess(all_moves_dict, elo_dict, fens, es, eo, device)

        logits_pi = forward_logits(policy, board_input, es_t, eo_t)
        logits_ref = forward_logits(base, board_input, es_t, eo_t)

        logits_pi_m = apply_legal_mask(logits_pi, legal_moves)
        logits_ref_m = apply_legal_mask(logits_ref, legal_moves)

        probs_pi = torch.softmax(logits_pi_m, dim=-1)
        probs_ref = torch.softmax(logits_ref_m, dim=-1)

        entropy_pi = entropy_from_logits(logits_pi_m)     # [B]
        entropy_ref = entropy_from_logits(logits_ref_m)   # [B]

        entropy_diff = entropy_pi - entropy_ref           # [B]

        # indices for chosen/rejected
        chosen_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, u) for fen, u in zip(fens, chosen)],
                                device=device, dtype=torch.long)
        rejected_idx = torch.tensor([uci_to_vocab_index(all_moves_dict, fen, u) for fen, u in zip(fens, rejected)],
                                    device=device, dtype=torch.long)

        chosen_ok = (chosen_idx >= 0) & (legal_moves.gather(1, chosen_idx.clamp(min=0).view(-1,1)).squeeze(1) > 0)
        rejected_ok = (rejected_idx >= 0) & (legal_moves.gather(1, rejected_idx.clamp(min=0).view(-1,1)).squeeze(1) > 0)
        bad = ~(chosen_ok & rejected_ok)
        if bad.any():
            j = int(bad.nonzero()[0])
            raise RuntimeError(f"Illegal chosen/rejected under mask. fen={fens[j]} chosen={chosen[j]} rejected={rejected[j]}")

        logp_pi_ch = gather_logprob(logits_pi_m, chosen_idx)
        logp_pi_rj = gather_logprob(logits_pi_m, rejected_idx)
        logp_ref_ch = gather_logprob(logits_ref_m, chosen_idx)
        logp_ref_rj = gather_logprob(logits_ref_m, rejected_idx)

        idx_t = chosen_index_tensor(batch["fen"], all_moves_dict, batch["chosen"], device)

        chosen_cps = [extract_move_cp(m, ch) for m, ch in zip(meta_list, chosen)]
        rejected_cps = [extract_move_cp(m, rj) for m, rj in zip(meta_list, rejected)]

        chosen_is_in_top_tens = [is_chosen_in_top_10_engine(m, ch) for m, ch in zip(meta_list, chosen)]

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

        loss = supplied_loss_function(logp_pi_ch, 
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
                                      zip(
                                        fens,
                                        chosen,
                                        rejected,
                                        chosen_cps,
                                        rejected_cps,
                                        ply_idxs,
                                        prev_fens_batch,
                                        next_fens_chosen_batch,
                                        next_fens_rejected_batch
                                    )
        )

        pi_gap = (logp_pi_ch - logp_pi_rj)          # [B]
        ref_gap = (logp_ref_ch - logp_ref_rj)       # [B]
        gap_improve = (pi_gap - ref_gap)            # [B]

        top1_pi = top1_accuracy(logits_pi_m, fens, all_moves_dict, chosen)
        top1_ref = top1_accuracy(logits_ref_m, fens, all_moves_dict, chosen)

        p_chosen_pi = chosen_probability(logits_pi_m, fens, all_moves_dict, chosen)
        p_chosen_ref = chosen_probability(logits_ref_m, fens, all_moves_dict, chosen)

        kl = kl_policy_base_from_logits(logits_pi_m, logits_ref_m)     # [B]

        # NEW: ranking metrics
        rank_ch = chosen_rank(logits_pi_m, chosen_idx)  # [B]
        hit3 = hit_at_k(logits_pi_m, chosen_idx, k=3)
        hit5 = hit_at_k(logits_pi_m, chosen_idx, k=5)
        hit10 = hit_at_k(logits_pi_m, chosen_idx, k=10)

        # NEW: predicted UCI (top-1)
        pred_idx = logits_pi_m.argmax(dim=-1).tolist()
        pred_uci = [vocab_index_to_uci(all_moves, fen, i) for fen, i in zip(fens, pred_idx)]

        n += bs
        sum_loss += float(loss) * bs

        sum_pi_gap += float(pi_gap.mean()) * bs
        sum_ref_gap += float(ref_gap.mean()) * bs
        sum_gap_improvement += float(gap_improve.mean()) * bs

        sum_top1_pi += float(top1_pi.mean()) * bs
        sum_top1_ref += float(top1_ref.mean()) * bs

        sum_p_chosen_pi += float(p_chosen_pi.mean()) * bs
        sum_p_chosen_ref += float(p_chosen_ref.mean()) * bs

        sum_kl += float(kl.mean()) * bs

        sum_ent_pi += float(entropy_pi.mean()) * bs
        sum_ent_ref += float(entropy_ref.mean()) * bs

        # NEW: predicted UCI (top-1)
        pred_idx = logits_pi_m.argmax(dim=-1).tolist()
        pred_uci = [vocab_index_to_uci(all_moves, fen, i) for fen, i in zip(fens, pred_idx)]

        # NEW: top-10 moves for policy and base
        k = min(10, logits_pi_m.shape[-1])

        topk_pi_idx = torch.topk(logits_pi_m, k=k, dim=-1).indices      # [B, k]
        topk_ref_idx = torch.topk(logits_ref_m, k=k, dim=-1).indices    # [B, k]

        topk_pi_uci = [
            [
                {
                    "uci": vocab_index_to_uci(all_moves, fens[i], int(idx)),
                    "logit": float(logits_pi_m[i, idx].item()),
                    "prob": float(probs_pi[i, idx].item())
                }
                for idx in topk_pi_idx[i].tolist()
            ]
            for i in range(bs)
        ]

        topk_ref_uci = [
            [
                {
                    "uci": vocab_index_to_uci(all_moves, fens[i], int(idx)),
                    "logit": float(logits_ref_m[i, idx].item()),
                    "prob": float(probs_ref[i, idx].item())
                }
                for idx in topk_ref_idx[i].tolist()
            ]
            for i in range(bs)
        ]

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

        # NEW: per-row output + phase tails
        for i in range(bs):
            fen = fens[i]
            ply_abs = fen_to_ply(fen)
            phase = ply_to_phase(ply_abs)

            gid = str(game_ids[i] or "")
            if gid:
                pref = opening_prefixes[i] or []

                if gid not in opening_by_game:
                    opening_by_game[gid] = coarse_opening_family_from_prefix(pref)

                # Keep one empirical opening prefix per game so we can reconstruct
                # actual player opening frequencies from played games.
                if gid not in opening_prefix_by_game:
                    opening_prefix_by_game[gid] = list(pref)

            correct = float(top1_pi[i].item())
            correct_ref = float(top1_ref[i].item())

            r = {
                "game_id": gid,
                "ply_idx": int(ply_idxs[i]) if ply_idxs[i] is not None else -1,
                "ply_abs": int(ply_abs),
                "phase": phase,
                "fen": fen,
                "chosen_uci": chosen[i],
                "rejected_uci": rejected[i],
                "pred_uci": pred_uci[i],
                "correct_top1": correct,
                "correct_top1_ref": correct_ref,
                "hit_top3": float(hit3[i].item()),
                "hit_top5": float(hit5[i].item()),
                "hit_top10": float(hit10[i].item()),
                "rank_chosen": int(rank_ch[i].item()),
                "mrr": float(1.0 / float(rank_ch[i].item())),
                "logp_gap_pi": float(pi_gap[i].item()),
                "logp_gap_ref": float(ref_gap[i].item()),
                "gap_improve": float(gap_improve[i].item()),
                "p_chosen_pi": float(p_chosen_pi[i].item()),
                "p_chosen_ref": float(p_chosen_ref[i].item()),
                "kl_pi_ref": float(kl[i].item()),
                "nll_chosen_pi": float((-logp_pi_ch[i]).item()),
                "top_max10_pi_w_logits": topk_pi_uci[i],
                "top_max10_ref_w_logits": topk_ref_uci[i],
                "stockfish": batch["meta"][i]["stockfish"],
                "entropy_pi": float(entropy_pi[i].item()),
                "entropy_ref": float(entropy_ref[i].item()),
                "entropy_diff_pi_vs_ref": float(entropy_diff[i].item()),
                "prev_fens": prev_fens_batch[i],
                "next_fens_chosen": next_fens_chosen_batch[i],
                "next_fens_rejected": next_fens_rejected_batch[i],
                "chosen_is_in_top_ten": chosen_is_in_top_tens[i],
            }

            per_rows.append(r)

            phase_buckets[("hit_top3", phase)].append(r["hit_top3"])
            phase_buckets[("hit_top5", phase)].append(r["hit_top5"])
            phase_buckets[("hit_top10", phase)].append(r["hit_top10"])
            phase_buckets[("rank_chosen", phase)].append(r["rank_chosen"])
            phase_buckets[("mrr", phase)].append(r["mrr"])
            phase_buckets[("entropy_pi", phase)].append(r["entropy_pi"])
            phase_buckets[("entropy_ref", phase)].append(r["entropy_ref"])
            phase_buckets[("kl_pi_ref", phase)].append(r["kl_pi_ref"])
            phase_buckets[("logp_gap_pi", phase)].append(r["logp_gap_pi"])
            phase_buckets[("p_chosen_pi", phase)].append(r["p_chosen_pi"])
            phase_buckets[("correct_top1", phase)].append(r["correct_top1"])


    def precision_top1(eval_rows):
        correct_count = 0
        incorrect_count = 0 
        for row in eval_rows:
            if row["correct_top1"] == 1:
                correct_count += 1
            else:
                incorrect_count += 1
        
        return correct_count / (correct_count + incorrect_count)

    def recall_top1(eval_rows):
        return sum([r["correct_top1"] for r in eval_rows]) / len(eval_rows)

    def f1_top1(eval_rows):
        return 2 / ((1.0 / precision_top1(eval_rows)) + (1.0 / recall_top1(eval_rows)))

    def precision_top3(eval_rows):
        correct_count = 0
        incorrect_count = 0 
        for row in eval_rows:
            if row["hit_top3"] == 1:
                correct_count += 1
            else:
                incorrect_count += 1
        
        return correct_count / (correct_count + incorrect_count)

    def recall_top3(eval_rows):
        return sum([r["hit_top3"] for r in eval_rows]) / len(eval_rows)

    def f1_top3(eval_rows):
        return 2 / ((1.0 / precision_top3(eval_rows)) + (1.0 / recall_top3(eval_rows)))

    def precision_top5(eval_rows):
        correct_count = 0
        incorrect_count = 0 
        for row in eval_rows:
            if row["hit_top5"] == 1:
                correct_count += 1
            else:
                incorrect_count += 1
        
        return correct_count / (correct_count + incorrect_count)
    
    def percent_in_top_ten_engine(eval_rows):
        in_top_ten = 0
        not_in_top_ten = 0

        for row in eval_rows:
            if row["chosen_is_in_top_ten"]:
                in_top_ten += 1
            else:
                not_in_top_ten += 1
        return in_top_ten / (in_top_ten + not_in_top_ten + 0.001)
            # probability we pick the move when it is in the top 10 engine moves
        # probability we pick the move when it is not in the top 10 engine moves
        # percent of time the chosen move is in the top 10 engine moves

    def recall_top5(eval_rows):
        return sum([r["hit_top5"] for r in eval_rows]) / len(eval_rows)

    def f1_top5(eval_rows):
        return 2 / ((1.0 / precision_top5(eval_rows)) + (1.0 / recall_top5(eval_rows)))
    
    def precision_top10(eval_rows):
        correct_count = 0
        incorrect_count = 0 
        for row in eval_rows:
            if row["hit_top10"] == 1:
                correct_count += 1
            else:
                incorrect_count += 1
        
        return correct_count / (correct_count + incorrect_count)

    def recall_top10(eval_rows):
        return sum([r["hit_top10"] for r in eval_rows]) / len(eval_rows)

    def f1_top10(eval_rows):
        return 2 / ((1.0 / precision_top10(eval_rows)) + (1.0 / recall_top10(eval_rows)))

    def avg(x: float) -> float:
        return x / max(1, n)

    # ----------------------------
    # Phase-wise tails (median + p90/p95/p99)
    # ----------------------------
    phase_summary: Dict[str, Dict[str, Any]] = {}
    phase_summary.setdefault("top1_precision", {})
    phase_summary.setdefault("top1_recall", {})
    phase_summary.setdefault("top1_f1", {})
    phase_summary.setdefault("top3_precision", {})
    phase_summary.setdefault("top3_recall", {})
    phase_summary.setdefault("top3_f1", {})
    phase_summary.setdefault("top5_precision", {})
    phase_summary.setdefault("top5_recall", {})
    phase_summary.setdefault("top5_f1", {})
    phase_summary.setdefault("top10_precision", {})
    phase_summary.setdefault("top10_recall", {})
    phase_summary.setdefault("top10_f1", {})
    phase_summary.setdefault("percent_chosen_in_top_ten", {})
    for (metric, phase), xs in phase_buckets.items():
        phase_summary.setdefault(metric, {})
        phase_summary[metric][phase] = {
            "n": len(xs),
            "mean": mean(xs),
            "median": statistics.median(xs) if xs else float("nan"),
            **quantiles(xs, ps=(0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)),
        }
        if phase not in phase_summary["top1_precision"]:
            phase_summary["top1_precision"][phase] = precision_top1([r for r in per_rows if r["phase"] == phase])
            phase_summary["top1_recall"][phase] = recall_top1([r for r in per_rows if r["phase"] == phase])
            phase_summary["top1_f1"][phase] = f1_top1([r for r in per_rows if r["phase"] == phase])

            phase_summary["top3_precision"][phase] = precision_top3([r for r in per_rows if r["phase"] == phase])
            phase_summary["top3_recall"][phase] = recall_top3([r for r in per_rows if r["phase"] == phase])
            phase_summary["top3_f1"][phase] = f1_top3([r for r in per_rows if r["phase"] == phase])

            phase_summary["top5_precision"][phase] = precision_top5([r for r in per_rows if r["phase"] == phase])
            phase_summary["top5_recall"][phase] = recall_top5([r for r in per_rows if r["phase"] == phase])
            phase_summary["top5_f1"][phase] = f1_top5([r for r in per_rows if r["phase"] == phase])

            phase_summary["top10_precision"][phase] = precision_top10([r for r in per_rows if r["phase"] == phase])
            phase_summary["top10_recall"][phase] = recall_top10([r for r in per_rows if r["phase"] == phase])
            phase_summary["top10_f1"][phase] = f1_top10([r for r in per_rows if r["phase"] == phase])

            phase_summary["percent_chosen_in_top_ten"][phase] = percent_in_top_ten_engine([r for r in per_rows if r["phase"] == phase])

    # ----------------------------
    # Opening fingerprint distribution (per game)
    # ----------------------------
    opening_counts = Counter(opening_by_game.values())
    opening_dist = {k: v for k, v in opening_counts.most_common()}

    # ----------------------------
    # Empirical player opening distribution from played games
    # ----------------------------
    def normalize_counter(counter: Counter) -> Dict[str, float]:
        total = sum(counter.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in counter.items()}

    white_first_moves_probed = [
        "e2e4", "d2d4", "c2c4", "g1f3", "g2g3", "b2b3", "f2f4", "b2b4", "a2a4",
    ]
    black_replies_probed = [
        "c7c5", "e7e5", "e7e6", "c7c6", "d7d5", "g8f6", "g7g6", "d7d6",
    ]

    player_white_first_counts = Counter()
    player_black_reply_cond_counts: Dict[str, Counter] = defaultdict(Counter)

    n_games_with_prefix = 0
    n_games_with_white_first = 0
    n_games_with_black_reply = 0

    for gid, pref in opening_prefix_by_game.items():
        if not pref:
            continue

        n_games_with_prefix += 1

        white_first = pref[0] if len(pref) >= 1 else None
        black_reply = pref[1] if len(pref) >= 2 else None

        if white_first:
            n_games_with_white_first += 1
            player_white_first_counts[white_first] += 1

        if white_first and black_reply:
            n_games_with_black_reply += 1
            player_black_reply_cond_counts[white_first][black_reply] += 1

    # Match probe_opening_distributions_from_policy JSON shape
    player_white_first_move_probs = {}
    white_total = sum(player_white_first_counts.values())
    for u in white_first_moves_probed:
        player_white_first_move_probs[u] = (
            player_white_first_counts[u] / white_total if white_total > 0 else 0.0
        )

    player_white_other_mass = 0.0
    if white_total > 0:
        used = sum(player_white_first_counts[u] for u in white_first_moves_probed)
        player_white_other_mass = max(0.0, 1.0 - (used / white_total))

    player_black_reply_probs_cond_on_white = {}
    for white_uci in white_first_moves_probed:
        ctr = player_black_reply_cond_counts.get(white_uci, Counter())
        total_for_white = sum(ctr.values())

        probs = {}
        for black_uci in black_replies_probed:
            probs[black_uci] = ctr[black_uci] / total_for_white if total_for_white > 0 else 0.0

        other_mass = 0.0
        if total_for_white > 0:
            used = sum(ctr[u] for u in black_replies_probed)
            other_mass = max(0.0, 1.0 - (used / total_for_white))

        # Reconstruct canonical fen_after_white for consistency with probe output
        try:
            b = chess.Board(chess.STARTING_FEN)
            b.push_uci(white_uci)
            fen_after_white = b.fen()
        except Exception:
            fen_after_white = None

        player_black_reply_probs_cond_on_white[white_uci] = {
            "fen_after_white": fen_after_white,
            "black_reply_probs": probs,
            "other_mass": other_mass,
            "n_games_with_this_white_move": int(total_for_white),
        }

    player_opening_probe_empirical = {
        "white_first_move_probs": player_white_first_move_probs,
        "white_other_mass": player_white_other_mass,
        "black_reply_probs_cond_on_white": player_black_reply_probs_cond_on_white,
        "meta": {
            "source": "empirical_player_games",
            "n_games_total": len(opening_prefix_by_game),
            "n_games_with_prefix": n_games_with_prefix,
            "n_games_with_white_first": n_games_with_white_first,
            "n_games_with_black_reply": n_games_with_black_reply,
            "white_moves_probed": white_first_moves_probed,
            "black_replies_probed": black_replies_probed,
        },
    }

    # ----------------------------
    # Bootstrap confidence intervals
    # ----------------------------
    rows_in_top_ten = [r for r in per_rows if r["chosen_is_in_top_ten"]]
    rows_not_in_top_ten = [r for r in per_rows if not r["chosen_is_in_top_ten"]]
    num_not_in_top_ten = len(rows_not_in_top_ten)
    num_in_top_ten = len(rows_in_top_ten)

    acc_vals = [r["correct_top1"] for r in per_rows]
    gap_vals = [r["logp_gap_pi"] for r in per_rows]
    pch_vals = [r["p_chosen_pi"] for r in per_rows]
    mrr_vals = [r["mrr"] for r in per_rows]
    ent_pi_vals = [r["entropy_pi"] for r in per_rows]
    ent_ref_vals = [r["entropy_ref"] for r in per_rows]
    hit_top3_vals = [r["hit_top3"] for r in per_rows]
    hit_top5_vals = [r["hit_top5"] for r in per_rows]
    hit_top10_vals = [r["hit_top10"] for r in per_rows]
    kl_pi_ref_vals = [r["kl_pi_ref"] for r in per_rows]
    p_chosen_pi_vals = [r["p_chosen_pi"] for r in per_rows]
    p_chosen_ref_vals = [r["p_chosen_ref"] for r in per_rows]

    acc_vals_cond_on_not_in_top_ten = [r["correct_top1"] for r in rows_not_in_top_ten]
    gap_vals_cond_on_not_in_top_ten = [r["logp_gap_pi"] for r in rows_not_in_top_ten]
    pch_vals_cond_on_not_in_top_ten = [r["p_chosen_pi"] for r in rows_not_in_top_ten]
    mrr_vals_cond_on_not_in_top_ten = [r["mrr"] for r in rows_not_in_top_ten]
    ent_pi_vals_cond_on_not_in_top_ten = [r["entropy_pi"] for r in rows_not_in_top_ten]
    ent_ref_vals_cond_on_not_in_top_ten = [r["entropy_ref"] for r in rows_not_in_top_ten]
    hit_top3_vals_cond_on_not_in_top_ten = [r["hit_top3"] for r in rows_not_in_top_ten]
    hit_top5_vals_cond_on_not_in_top_ten = [r["hit_top5"] for r in rows_not_in_top_ten]
    hit_top10_vals_cond_on_not_in_top_ten = [r["hit_top10"] for r in rows_not_in_top_ten]
    kl_pi_ref_vals_cond_on_not_in_top_ten = [r["kl_pi_ref"] for r in rows_not_in_top_ten]
    p_chosen_pi_vals_cond_on_not_in_top_ten = [r["p_chosen_pi"] for r in rows_not_in_top_ten]
    p_chosen_ref_vals_cond_on_not_in_top_ten = [r["p_chosen_ref"] for r in rows_not_in_top_ten]

    acc_vals_cond_on_in_top_ten = [r["correct_top1"] for r in rows_in_top_ten]
    gap_vals_cond_on_in_top_ten = [r["logp_gap_pi"] for r in rows_in_top_ten]
    pch_vals_cond_on_in_top_ten = [r["p_chosen_pi"] for r in rows_in_top_ten]
    mrr_vals_cond_on_in_top_ten = [r["mrr"] for r in rows_in_top_ten]
    ent_pi_vals_cond_on_in_top_ten = [r["entropy_pi"] for r in rows_in_top_ten]
    ent_ref_vals_cond_on_in_top_ten = [r["entropy_ref"] for r in rows_in_top_ten]
    hit_top3_vals_cond_on_in_top_ten = [r["hit_top3"] for r in rows_in_top_ten]
    hit_top5_vals_cond_on_in_top_ten = [r["hit_top5"] for r in rows_in_top_ten]
    hit_top10_vals_cond_on_in_top_ten = [r["hit_top10"] for r in rows_in_top_ten]
    kl_pi_ref_vals_cond_on_in_top_ten = [r["kl_pi_ref"] for r in rows_in_top_ten]
    p_chosen_pi_vals_cond_on_in_top_ten = [r["p_chosen_pi"] for r in rows_in_top_ten]
    p_chosen_ref_vals_cond_on_in_top_ten = [r["p_chosen_ref"] for r in rows_in_top_ten]

    ci_row = {
        "accuracy_top1": bootstrap_ci(acc_vals, mean, n_boot=n_boot, seed=0),
        "mean_logp_gap_pi": bootstrap_ci(gap_vals, mean, n_boot=n_boot, seed=1),
        "mean_p_chosen_pi": bootstrap_ci(pch_vals, mean, n_boot=n_boot, seed=2),
        "mrr": bootstrap_ci(mrr_vals, mean, n_boot=n_boot, seed=3),
        "entropy_pi": bootstrap_ci(ent_pi_vals, mean, n_boot=n_boot, seed=3),
        "entropy_ref": bootstrap_ci(ent_ref_vals, mean, n_boot=n_boot, seed=3),
        "hit_top3": bootstrap_ci(hit_top3_vals, mean, n_boot=n_boot, seed=3),
        "hit_top5": bootstrap_ci(hit_top5_vals, mean, n_boot=n_boot, seed=3),
        "hit_top10": bootstrap_ci(hit_top10_vals, mean, n_boot=n_boot, seed=3),
        "kl_pi_ref": bootstrap_ci(kl_pi_ref_vals, mean, n_boot=n_boot, seed=3),
        "p_chosen_pi": bootstrap_ci(p_chosen_pi_vals, mean, n_boot=n_boot, seed=3),
        "p_chosen_ref": bootstrap_ci(p_chosen_ref_vals, mean, n_boot=n_boot, seed=3),

        "accuracy_top1_cond_on_not_in_top_ten": bootstrap_ci(acc_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=0),
        "mean_logp_gap_pi_cond_on_not_in_top_ten": bootstrap_ci(gap_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=1),
        "mean_p_chosen_pi_cond_on_not_in_top_ten": bootstrap_ci(pch_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=2),
        "mrr_cond_on_not_in_top_ten": bootstrap_ci(mrr_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=3),
        "entropy_pi_cond_on_not_in_top_ten": bootstrap_ci(ent_pi_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=3),
        "entropy_ref_cond_on_not_in_top_ten": bootstrap_ci(ent_ref_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=3),
        "hit_top3_cond_on_not_in_top_ten": bootstrap_ci(hit_top3_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=3),
        "hit_top5_cond_on_not_in_top_ten": bootstrap_ci(hit_top5_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=3),
        "hit_top10_cond_on_not_in_top_ten": bootstrap_ci(hit_top10_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=3),
        "kl_pi_ref_cond_on_not_in_top_ten": bootstrap_ci(kl_pi_ref_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=3),
        "p_chosen_pi_cond_on_not_in_top_ten": bootstrap_ci(p_chosen_pi_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=3),
        "p_chosen_ref_cond_on_not_in_top_ten": bootstrap_ci(p_chosen_ref_vals_cond_on_not_in_top_ten, mean, n_boot=n_boot, seed=3),

        "accuracy_top1_cond_on_in_top_ten": bootstrap_ci(acc_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=0),
        "mean_logp_gap_pi_cond_on_in_top_ten": bootstrap_ci(gap_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=1),
        "mean_p_chosen_pi_cond_on_in_top_ten": bootstrap_ci(pch_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=2),
        "mrr_cond_on_in_top_ten": bootstrap_ci(mrr_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=3),
        "entropy_pi_cond_on_in_top_ten": bootstrap_ci(ent_pi_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=3),
        "entropy_ref_cond_on_in_top_ten": bootstrap_ci(ent_ref_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=3),
        "hit_top3_cond_on_in_top_ten": bootstrap_ci(hit_top3_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=3),
        "hit_top5_cond_on_in_top_ten": bootstrap_ci(hit_top5_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=3),
        "hit_top10_cond_on_in_top_ten": bootstrap_ci(hit_top10_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=3),
        "kl_pi_ref_cond_on_in_top_ten": bootstrap_ci(kl_pi_ref_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=3),
        "p_chosen_pi_cond_on_in_top_ten": bootstrap_ci(p_chosen_pi_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=3),
        "p_chosen_ref_cond_on_in_top_ten": bootstrap_ci(p_chosen_ref_vals_cond_on_in_top_ten, mean, n_boot=n_boot, seed=3),
    }

    # cluster bootstrap if we have game ids
    ci_cluster = {
        "accuracy_top1": cluster_bootstrap_ci(per_rows, "game_id", "correct_top1", mean, n_boot=n_boot, seed=10),
        "mean_logp_gap_pi": cluster_bootstrap_ci(per_rows, "game_id", "logp_gap_pi", mean, n_boot=n_boot, seed=11),
        "mean_p_chosen_pi": cluster_bootstrap_ci(per_rows, "game_id", "p_chosen_pi", mean, n_boot=n_boot, seed=12),
        "mrr": cluster_bootstrap_ci(per_rows, "game_id", "mrr", mean, n_boot=n_boot, seed=13),
    }
    # drop Nones
    ci_cluster = {k: v for k, v in ci_cluster.items() if v is not None}


    print("\n=== Eval summary ===")
    print(f"GM: {gm_name}")
    print(f"examples: {n}")
    print(f"loss: {avg(sum_loss):.4f}")
    print("")
    print(f"mean logp_gap policy (chosen - rejected): {avg(sum_pi_gap):.4f}")
    print(f"mean logp_gap base   (chosen - rejected): {avg(sum_ref_gap):.4f}")
    print(f"mean gap improvement (policy - base):     {avg(sum_gap_improvement):.4f}")
    print("")
    print(f"top1 accuracy on chosen (policy): {avg(sum_top1_pi):.4f}")
    print(f"top1 accuracy on chosen (base):   {avg(sum_top1_ref):.4f}")
    print("")
    print(f"mean P(chosen) (policy): {avg(sum_p_chosen_pi):.4f}")
    print(f"mean P(chosen) (base):   {avg(sum_p_chosen_ref):.4f}")
    print("")
    print(f"mean KL(policy || base) over legal moves: {avg(sum_kl):.4f}")
    print("")


    agg = {
        "loss": avg(sum_loss),
        "mean_logp_gap_policy_chosen_rejected": avg(sum_pi_gap),
        "mean_logp_gap_base_chosen_rejected": avg(sum_ref_gap),
        "mean_gap_improvement": avg(sum_gap_improvement),
        "top1_accuracy_on_chosen_policy": avg(sum_top1_pi),
        "top1_accuracy_on_chosen_base": avg(sum_top1_ref),
        "mean_p_chosen_policy": avg(sum_p_chosen_pi),
        "mean_p_chosen_base": avg(sum_p_chosen_ref),
        "mean_kl": avg(sum_kl),
        "mean_ent_pi": avg(sum_ent_pi),
        "mean_ent_ref": avg(sum_ent_ref),
        "top1_precision": precision_top1(per_rows),
        "top1_recall": recall_top1(per_rows),
        "top1_f1": f1_top1(per_rows),
        "top3_precision": precision_top3(per_rows),
        "top3_recall": recall_top3(per_rows),
        "top3_f1": f1_top3(per_rows),
        "top5_precision": precision_top5(per_rows),
        "top5_recall": recall_top5(per_rows),
        "top5_f1": f1_top5(per_rows),
        "top10_precision": precision_top10(per_rows),
        "top10_recall": recall_top10(per_rows),
        "top10_f1": f1_top10(per_rows),
    }

    def avg_over_chosen_not_in_top_ten(metric: str) -> float:
        return sum([r[metric] for r in per_rows if not r["chosen_is_in_top_ten"]]) / max(num_not_in_top_ten, 1.0)

    def avg_over_chosen_in_top_ten(metric: str) -> float:
        return sum([r[metric] for r in per_rows if r["chosen_is_in_top_ten"]]) / max(num_in_top_ten, 1.0)
    
    # add conditional metrics on chosen not in top 10 stockfish multipv
    agg = {
        **agg,
        "mean_logp_gap_policy_chosen_rejected_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("logp_gap_pi"),
        "mean_logp_gap_base_chosen_rejected_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("logp_gap_ref"),
        "mean_gap_improvement_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("gap_improve"),
        "top1_accuracy_on_chosen_policy_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("correct_top1"),
        "top1_accuracy_on_chosen_base_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("correct_top1_ref"),
        "mean_p_chosen_policy_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("p_chosen_pi"),
        "mean_p_chosen_base_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("p_chosen_ref"),
        "mean_kl_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("kl_pi_ref"),
        "mean_ent_pi_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("entropy_pi"),
        "mean_ent_ref_cond_on_not_in_top_ten": avg_over_chosen_not_in_top_ten("entropy_ref"),
        "top1_precision_cond_on_not_in_top_ten": precision_top1([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top1_recall_cond_on_not_in_top_ten": recall_top1([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top1_f1_cond_on_not_in_top_ten": f1_top1([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top3_precision_cond_on_not_in_top_ten": precision_top3([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top3_recall_cond_on_not_in_top_ten": recall_top3([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top3_f1_cond_on_not_in_top_ten": f1_top3([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top5_precision_cond_on_not_in_top_ten": precision_top5([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top5_recall_cond_on_not_in_top_ten": recall_top5([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top5_f1_cond_on_not_in_top_ten": f1_top5([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top10_precision_cond_on_not_in_top_ten": precision_top10([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top10_recall_cond_on_not_in_top_ten": recall_top10([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
        "top10_f1_cond_on_not_in_top_ten": f1_top10([r for r in per_rows if not r["chosen_is_in_top_ten"]]),
    }

    agg = {
        **agg,
        "mean_logp_gap_policy_chosen_rejected_cond_on_in_top_ten": avg_over_chosen_in_top_ten("logp_gap_pi"),
        "mean_logp_gap_base_chosen_rejected_cond_on_in_top_ten": avg_over_chosen_in_top_ten("logp_gap_ref"),
        "mean_gap_improvement_cond_on_in_top_ten": avg_over_chosen_in_top_ten("gap_improve"),
        "top1_accuracy_on_chosen_policy_cond_on_in_top_ten": avg_over_chosen_in_top_ten("correct_top1"),
        "top1_accuracy_on_chosen_base_cond_on_in_top_ten": avg_over_chosen_in_top_ten("correct_top1_ref"),
        "mean_p_chosen_policy_cond_on_in_top_ten": avg_over_chosen_in_top_ten("p_chosen_pi"),
        "mean_p_chosen_base_cond_on_in_top_ten": avg_over_chosen_in_top_ten("p_chosen_ref"),
        "mean_kl_cond_on_in_top_ten": avg_over_chosen_in_top_ten("kl_pi_ref"),
        "mean_ent_pi_cond_on_in_top_ten": avg_over_chosen_in_top_ten("entropy_pi"),
        "mean_ent_ref_cond_on_in_top_ten": avg_over_chosen_in_top_ten("entropy_ref"),
        "top1_precision_cond_on_in_top_ten": precision_top1([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top1_recall_cond_on_in_top_ten": recall_top1([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top1_f1_cond_on_in_top_ten": f1_top1([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top3_precision_cond_on_in_top_ten": precision_top3([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top3_recall_cond_on_in_top_ten": recall_top3([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top3_f1_cond_on_in_top_ten": f1_top3([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top5_precision_cond_on_in_top_ten": precision_top5([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top5_recall_cond_on_in_top_ten": recall_top5([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top5_f1_cond_on_in_top_ten": f1_top5([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top10_precision_cond_on_in_top_ten": precision_top10([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top10_recall_cond_on_in_top_ten": recall_top10([r for r in per_rows if r["chosen_is_in_top_ten"]]),
        "top10_f1_cond_on_in_top_ten": f1_top10([r for r in per_rows if r["chosen_is_in_top_ten"]]),
    }

    out_dir.joinpath(agg_results_filename).write_text(json.dumps(agg))
    print(f"Eval results saved to {out_dir.joinpath(agg_results_filename)}")
    print(f"Eval results saved to {out_dir.joinpath(agg_results_filename)}")
    # Now we write csv to out_dir.joinpath(f"eval_results_dpo_{args.split_name}.csv")

    # 2) Extended JSON: phase tails + CIs + opening dist
    
    ext = {
        **agg,
        "n_rows": len(per_rows),
        "phase_summary": phase_summary,
        "bootstrap_ci_row": ci_row,
        "bootstrap_ci_cluster_by_game_player_chosen": ci_cluster,
        "opening_family_counts_by_game_player_chosen": opening_dist,
        "player_opening_probe_empirical": player_opening_probe_empirical,
        "notes": {
            "opening_family_is_coarse_heuristic_player_chosen": True,
            "precision_recall_f1_equals_accuracy_for_top1_hit": True,
        },
    }
    out_ext = out_dir.joinpath(eval_results_extended_filename)
    out_ext.write_text(json.dumps(ext, indent=2))
    print(f"Extended eval saved to {out_ext}")


    import csv
    with open(out_dir.joinpath(eval_results_summary_csv_filename), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["loss", "mean_logp_gap_policy_chosen_rejected", "mean_logp_gap_base_chosen_rejected", 
                        "mean_gap_improvement", "top1_accuracy_on_chosen_policy", "top1_accuracy_on_chosen_base", 
                        "mean_p_chosen_policy", "mean_p_chosen_base", "mean_kl", "mean_ent_pi", "mean_ent_ref"])
        writer.writerow([avg(sum_loss), avg(sum_pi_gap), avg(sum_ref_gap), avg(sum_gap_improvement), avg(sum_top1_pi), avg(sum_top1_ref), avg(sum_p_chosen_pi), avg(sum_p_chosen_ref), avg(sum_kl), avg(sum_ent_pi), avg(sum_ent_ref)])
    print(f"CSV saved to {out_dir.joinpath(eval_results_summary_csv_filename)}")

    # 4) Per-row metrics CSV
    if per_rows:
        per_row_json_path = out_dir.joinpath(per_row_filename)

        per_rows_sorted = sorted(
            per_rows,
            key=lambda r: (r.get("game_id"), r.get("ply_idx"))
        )

        per_rows_sorted = add_piece_selection_per_row_stats(per_rows_sorted)

        with open(per_row_json_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row) + "\n" for row in per_rows_sorted)

