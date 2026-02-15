#!/usr/bin/env python3
"""
Parse Lichess PGNs that include per-move clock annotations like:
  1. c4 { [%clk 0:03:00] } 1... g6 { [%clk 0:03:00] } ...

and produce JSONL datasets for timing-model training:

Output row schema (per *GM move* only):
{
  "prompt": {"fen": <FEN_BEFORE_MOVE>, "elo_self": <int>, "elo_oppo": <int>},
  "time_to_make_chosen_move_ms": <int>,
  "previous_five_ply_move_times_ms": [<int> x5],
  "chosen": "<uci>",
  "meta": {... game headers + ply_idx + gm_side + usernames ...}
}

Splits 80/20 into:
  processed/single_gm/time_per_move/train_val/{gm_name}/{gm_name}_train.jsonl
  processed/single_gm/time_per_move/train_val/{gm_name}/{gm_name}_val.jsonl

Example:
  python ./src/grandmaster_dpo/data_processing/single_gm/timer_data_per_gm/generate_timing_training_eval_data.py \
    --gm_name carlsen \
    --nickname DrNykterstein \
    --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn


CLK_RE = re.compile(r"\[%clk\s+([0-9]+):([0-9]{2}):([0-9]{2})\]")

def parse_hms_to_seconds(h: int, m: int, s: int) -> int:
    return int(h) * 3600 + int(m) * 60 + int(s)

def parse_clk_comment(comment: str) -> Optional[int]:
    """
    Extract clock time (seconds remaining) from a PGN node comment.
    Lichess typically stores it in a comment like: "{ [%clk 0:02:59] }"
    """
    if not comment:
        return None
    m = CLK_RE.search(comment)
    if not m:
        return None
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return parse_hms_to_seconds(hh, mm, ss)

def parse_timecontrol(tc: str) -> Tuple[Optional[int], int]:
    """
    Parse TimeControl header.
    Common Lichess: "180+0" meaning base seconds + increment seconds.
    Sometimes "0+1" etc.
    Returns (base_seconds_or_None, increment_seconds).
    """
    if not tc:
        return (None, 0)
    if "+" in tc:
        base_str, inc_str = tc.split("+", 1)
        inc = int(base_str and inc_str) if inc_str.isdigit() else int(float(inc_str)) if inc_str else 0
    else:
        base_str, inc = tc, 0

    base_str = base_str.strip()
    if base_str.isdigit():
        return (int(base_str), inc)

    # Occasionally a clock string might appear; handle "0:03:00" just in case
    if ":" in base_str:
        parts = base_str.split(":")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return (parse_hms_to_seconds(int(parts[0]), int(parts[1]), int(parts[2])), inc)

    return (None, inc)

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def choose_input_pgn(input_dir: Path, gm_name: str) -> Path:
    """
    Find input PGN file(s) that contain gm_name in filename.
    Prefers exact common pattern, otherwise first lexicographic match.
    """
    preferred = input_dir / f"{gm_name}_lichess_blitz_1k_w_times.pgn"
    if preferred.exists():
        return preferred

    matches = sorted(input_dir.glob(f"{gm_name}*.pgn"))
    if not matches:
        raise FileNotFoundError(f"No PGN found in {input_dir} matching {gm_name}*.pgn")
    return matches[0]

def iter_games(pgn_path: Path):
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game

def build_rows_from_game(game: chess.pgn.Game, nickname: str) -> List[Dict[str, Any]]:
    """
    Returns rows for GM moves only in this game.
    If any ply is missing a %clk comment, we stop parsing that game to avoid corrupt timing deltas.
    """
    headers = game.headers
    white = headers.get("White", "")
    black = headers.get("Black", "")

    if white != nickname and black != nickname:
        return []

    gm_side = "white" if white == nickname else "black"

    white_elo = safe_int(headers.get("WhiteElo", 0), 0)
    black_elo = safe_int(headers.get("BlackElo", 0), 0)

    elo_self = white_elo if gm_side == "white" else black_elo
    elo_oppo = black_elo if gm_side == "white" else white_elo

    base_sec, inc_sec = parse_timecontrol(headers.get("TimeControl", ""))
    # If base not present/parseable, infer from first clock tag (common fallback)
    # We'll set prev_clock_* lazily on first move for each side if needed.
    prev_clock_w: Optional[int] = base_sec
    prev_clock_b: Optional[int] = base_sec

    board = game.board()
    node = game

    times_global_ms: List[int] = []  # all previous ply times in ms (both sides), in chronological ply order
    out: List[Dict[str, Any]] = []

    ply_idx = 0  # 0-based ply index (half-moves)
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move

        mover_is_white = board.turn == chess.WHITE
        mover_side = "white" if mover_is_white else "black"

        # clock AFTER this move is recorded in the node comment
        clk_after = parse_clk_comment(next_node.comment)
        if clk_after is None:
            # Timing deltas become unreliable without this; stop this game.
            break

        # initialize prev clocks from first seen clock tag if base time was unknown
        if mover_is_white and prev_clock_w is None:
            prev_clock_w = clk_after
        if (not mover_is_white) and prev_clock_b is None:
            prev_clock_b = clk_after

        prev_clock = prev_clock_w if mover_is_white else prev_clock_b
        if prev_clock is None:
            # still unknown => bail out
            break

        # time spent for this ply (seconds)
        spent_sec = prev_clock - clk_after + inc_sec
        if spent_sec < 0:
            # can happen due to rounding / lag; clamp
            spent_sec = 0

        spent_ms = int(round(spent_sec * 1000.0))

        fen_before = board.fen()
        chosen_uci = move.uci()

        # previous 5 ply move times (prior to current move)
        prev5 = times_global_ms[-5:]
        if len(prev5) < 5:
            prev5 = [0] * (5 - len(prev5)) + prev5

        # record example if GM is the mover on this ply
        if mover_side == gm_side:
            meta = {
                "event": headers.get("Event", ""),
                "site": headers.get("Site", ""),
                "date": headers.get("Date", ""),
                "round": headers.get("Round", ""),
                "result": headers.get("Result", ""),
                "game_id": headers.get("GameId", ""),
                "utc_date": headers.get("UTCDate", ""),
                "utc_time": headers.get("UTCTime", ""),
                "time_control": headers.get("TimeControl", ""),
                "eco": headers.get("ECO", ""),
                "opening": headers.get("Opening", ""),
                "termination": headers.get("Termination", ""),
                "white": white,
                "black": black,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "gm_side": gm_side,
                "ply_idx": ply_idx,
            }

            out.append({
                "prompt": {
                    "fen": fen_before,
                    "elo_self": int(elo_self) if elo_self > 0 else 2800,
                    "elo_oppo": int(elo_oppo) if elo_oppo > 0 else 2800,
                },
                "time_to_make_chosen_move_ms": spent_ms,
                "previous_five_ply_move_times_ms": prev5,
                "prev_clock_w": prev_clock_w,
                "prev_clock_b": prev_clock_b,
                "player_side": mover_side,
                "ply_idx": ply_idx,
                "chosen": chosen_uci,
                "meta": meta,
            })

        # advance board + timing state
        board.push(move)
        times_global_ms.append(spent_ms)

        if mover_is_white:
            prev_clock_w = clk_after
        else:
            prev_clock_b = clk_after

        node = next_node
        ply_idx += 1

    return out

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", type=str, required=True)
    ap.add_argument("--nickname", type=str, required=True, help="Exact Lichess username in the PGNs (e.g., DrNykterstein).")
    ap.add_argument("--input_dir", type=str, default="./data/raw/lichess/blitz/timers_included/")
    ap.add_argument("--output_base", type=str, default="./processed/single_gm/time_per_move/train_val/")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val_frac", type=float, default=0.20)
    args = ap.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    pgn_path = choose_input_pgn(input_dir, args.gm_name)
    print(f"[timers] input_pgn={pgn_path}")

    all_rows: List[Dict[str, Any]] = []
    n_games = 0
    n_used_games = 0

    for game in iter_games(pgn_path):
        n_games += 1
        rows = build_rows_from_game(game, nickname=args.nickname)
        if rows:
            n_used_games += 1
            all_rows.extend(rows)

    print(f"[timers] games_total={n_games} games_matched_nickname={n_used_games} rows={len(all_rows)}")

    if not all_rows:
        raise SystemExit("No rows produced. Check --nickname matches PGN White/Black exactly and that [%clk ...] tags exist.")

    random.shuffle(all_rows)

    n_val = max(1, int(len(all_rows) * args.val_frac))
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    out_dir = Path(args.output_base) / args.gm_name
    train_path = out_dir / f"{args.gm_name}_train.jsonl"
    val_path = out_dir / f"{args.gm_name}_val.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    print(f"[timers] wrote train={train_path} ({len(train_rows)})")
    print(f"[timers] wrote   val={val_path} ({len(val_rows)})")

if __name__ == "__main__":
    main()
