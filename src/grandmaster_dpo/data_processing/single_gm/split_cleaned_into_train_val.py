from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterator, List, Tuple

import chess.pgn


def iter_pgn_games(pgn_path: str) -> Iterator[chess.pgn.Game]:
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            yield g


def game_hash(g: chess.pgn.Game) -> str:
    """
    Stable-ish hash to dedupe / split deterministically:
    hash(initial FEN + mainline UCI moves).
    """
    board = g.board()
    fen0 = board.fen()

    moves = []
    for mv in g.mainline_moves():
        moves.append(mv.uci())

    s = fen0 + "|" + " ".join(moves)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def write_pgn(games: List[chess.pgn.Game], out_path: str) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        exporter = chess.pgn.FileExporter(out)
        for g in games:
            g.accept(exporter)
            out.write("\n\n")
            n += 1
    return n


def split_games(
    games: List[chess.pgn.Game],
    *,
    val_frac: float,
    seed: int,
    dedupe: bool = True,
) -> Tuple[List[chess.pgn.Game], List[chess.pgn.Game]]:
    """
    Deterministic split:
    - optionally dedup by game_hash
    - sort by hash
    - assign every k-th game to val based on val_frac
    """
    if dedupe:
        seen = set()
        uniq = []
        for g in games:
            h = game_hash(g)
            if h in seen:
                continue
            seen.add(h)
            uniq.append(g)
        games = uniq

    # Deterministic ordering independent of file ordering
    games = sorted(games, key=game_hash)

    n = len(games)
    n_val = max(1, int(round(n * val_frac)))

    # Deterministic pseudo-shuffle via hashing with seed
    def seeded_key(g: chess.pgn.Game) -> str:
        h = game_hash(g)
        return hashlib.sha1(f"{seed}:{h}".encode("utf-8")).hexdigest()

    games = sorted(games, key=seeded_key)

    val = games[:n_val]
    train = games[n_val:]
    return train, val


def main():
    # Example usage: python ./src/grandmaster_dpo/data_processing/single_gm/split_cleaned_into_train_val.py --gm_name carlsen --in_dir ./final_experiments_for_paper/experiment1/cleaned_and_filtered_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/train_val_pgns_twic
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction (default 0.2).")
    ap.add_argument("--seed", type=int, default=7, help="Deterministic split seed.")
    ap.add_argument("--no_dedupe", action="store_true", help="Disable deduplication.")
    ap.add_argument("--in_dir", required=True, help="Input directory.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    args = ap.parse_args()
    in_pgn = Path(f"{args.in_dir}/{args.gm_name}.pgn")
    out_dir = Path(f"{args.out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    games = list(iter_pgn_games(in_pgn))
    print(f"Loaded games: {len(games)}")

    train, val = split_games(
        games,
        val_frac=args.val_frac,
        seed=args.seed,
        dedupe=not args.no_dedupe,
    )
    print(f"Train: {len(train)} | Val: {len(val)}")

    train_path = out_dir / f"{args.gm_name}_train.pgn"
    val_path = out_dir / f"{args.gm_name}_val.pgn"

    write_pgn(train, str(train_path))
    write_pgn(val, str(val_path))

    print(f"Wrote: {train_path}")
    print(f"Wrote: {val_path}")


if __name__ == "__main__":
    main()
