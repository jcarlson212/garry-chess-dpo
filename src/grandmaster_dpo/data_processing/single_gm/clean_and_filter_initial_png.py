"""
Preprocess PGN -> (filtered blitz games) -> batches -> Torch-ready tensors.

Assumptions:
- Input is a single PGN file (e.g., 1.7MB, ~2.3k games).
- Tokenization is a simple UCI-move vocabulary built from your corpus (good enough to get
  a DataLoader working now; you can swap in Maia's native tokenization later).


Event types are expected to be roughly of the following:
swiss (rapid)
match (rapid)
k.o. (rapid)
k.o. (blitz)
tourn (rapid)
swiss (blitz)
swiss
tourn (blitz)
tourn

To generate the dataset, we assume-after lowercasing- anything with blitz in it is blitz, anything with rapid in it is rapid, and otherwise it's classical.
We target 60% blitz games, 30% rapid games, and 10% classical games to give a balance between immediate
stylistic preferences (blitz), interesting middlegame positions (rapid), and purely tactical positions (classical). This is expected 
to help generalize the model to scale for different time controls and levels of play.

Or import and use PGNBlitzDataset for a DataLoader.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import chess
import chess.pgn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


import chess

import chess

def is_variant_game(g: chess.pgn.Game) -> bool:
    # Many PGNs explicitly mark it:
    variant = (g.headers.get("Variant") or "").lower()
    if variant != "":
        print(f"Skipping variant: {variant}")
        return True
    # Some lichess PGNs use:
    if g.headers.get("FEN") and "960" in (g.headers.get("SetUp") or ""):
        return True
    return False

def canonical_uci(board: chess.Board, move_str: str) -> str:
    """
    Canonicalize ANY move (UCI or SAN) into a legal, board-consistent UCI.

    - Works for standard chess AND Chess960
    - Validates legality
    - Handles castling, promotions, en passant automatically
    - Raises if the move cannot be interpreted
    """
    move_str = move_str.strip()

    # Try UCI first
    try:
        mv = chess.Move.from_uci(move_str)
        if mv in board.legal_moves:
            return mv.uci()
    except Exception:
        pass

    # Try SAN (PGNs often store SAN internally)
    try:
        mv = board.parse_san(move_str)
        if mv in board.legal_moves:
            return mv.uci()
    except Exception:
        pass

    raise ValueError(f"Illegal or unparseable move {move_str!r} for FEN {board.fen()}")

from typing import Optional, List

def game_to_canonical_uci_moves(
    game: chess.pgn.Game,
    *,
    drop_illegal: bool = True,
) -> Optional[List[str]]:
    board = game.board()
    out: List[str] = []

    for node in game.mainline():
        move = node.move
        try:
            uci = canonical_uci(board, move.uci())
            out.append(uci)
        except Exception:
            if drop_illegal:
                return None   # drop entire game
            else:
                pass
        board.push(move)

    return out


# ----------------------------
# Blitz detection
# ----------------------------

_TC_RE = re.compile(r"^\s*(\d+)\s*([+|/])\s*(\d+)\s*$")  # supports "300+0" or "300/0"
_INT_RE = re.compile(r"^\s*(\d+)\s*$")


def _parse_timecontrol_seconds(tc: str) -> Optional[int]:
    """
    Parse PGN TimeControl into an *estimate* of total seconds per player.
    Handles common formats like:
      - "300+0" (5+0)
      - "180+2" (3+2)
      - "600" (10 minutes base only)
    Returns None if unknown / unsupported (e.g. "?" or "-").
    """
    if not tc:
        return None
    tc = tc.strip()
    if tc in {"?", "-", "0"}:
        return None

    m = _TC_RE.match(tc)
    if m:
        base = int(m.group(1))
        inc = int(m.group(3))
        # crude estimate: base + 40*increment (typical middlegame length)
        return base + 40 * inc

    m2 = _INT_RE.match(tc)
    if m2:
        return int(m2.group(1))

    # Some PGNs have multiple controls like "40/7200:3600" (classical). Treat as non-blitz.
    if ":" in tc or "/" in tc:
        return None

    return None


def classify_time_control(headers: Dict[str, str]) -> str:
    et = headers.get("EventType", "").lower()

    if "bullet" in et:
        return "bullet"

    if "rapid" in et:
        return "rapid"

    if "blitz" in et:
        return "blitz"

    # everything else is slow / classical
    return "classical"


# ----------------------------
# Streaming PGN read/write
# ----------------------------

def iter_pgn_games(pgn_path: str) -> Iterator[chess.pgn.Game]:
    """Stream games from a PGN file without loading it all into memory."""
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def write_pgn(games: Iterator[chess.pgn.Game], out_path: str) -> int:
    """Write games to a PGN file. Returns number of games written."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        exporter = chess.pgn.FileExporter(out)
        for g in games:
            g.accept(exporter)
            out.write("\n\n")
            n += 1
    return n


def filter_for_game_type(
    pgn_path: str,
    *,
    require_min_plies: int = 20,  # e.g. >=10 full moves
    game_type: str = "blitz",
) -> Iterator[chess.pgn.Game]:
    """
    Yield blitz games from PGN.
    Also drops very short games (often junk / forfeits / malformed).
    """
    for g in iter_pgn_games(pgn_path):
        if is_variant_game(g):
            continue # this is a variant of chess where the back rank is not the same as the front rank
        if classify_time_control(g.headers) != game_type:
            continue
        plies = sum(1 for _ in g.mainline_moves())
        if plies < require_min_plies:
            continue

        uci_moves = game_to_canonical_uci_moves(g)
        if uci_moves is None:
            print(f"Dropping game {g.headers['Event']} {g.headers['Site']} {g.headers['Date']} {g.headers['Round']} because it has illegal moves")
            continue  # drop malformed / illegal games

        # rebuild game with canonical moves
        new_game = chess.pgn.Game()
        new_game.headers = g.headers.copy()

        board = new_game.board()
        node = new_game
        for u in uci_moves:
            mv = chess.Move.from_uci(u)
            node = node.add_variation(mv)
            board.push(mv)

        yield new_game



# ----------------------------
# Tokenization: UCI move vocab
# (swap this out later for Maia's native representation)
# ----------------------------

@dataclass(frozen=True)
class MoveVocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int

    @staticmethod
    def build_from_games(
        games: Iterator[chess.pgn.Game],
        *,
        min_freq: int = 1,
        add_specials: bool = True,
        max_moves: Optional[int] = None,  # cap distinct moves if you want
    ) -> "MoveVocab":
        from collections import Counter

        ctr = Counter()
        for g in games:
            uci_moves = game_to_canonical_uci_moves(g)
            if not uci_moves:
                continue
            for u in uci_moves:
                ctr[u] += 1

        moves = [m for m, c in ctr.items() if c >= min_freq]
        moves.sort()
        if max_moves is not None:
            moves = moves[:max_moves]

        itos: List[str] = []
        stoi: Dict[str, int] = {}

        def _add(tok: str) -> int:
            idx = len(itos)
            itos.append(tok)
            stoi[tok] = idx
            return idx

        if add_specials:
            pad_id = _add("<pad>")
            bos_id = _add("<bos>")
            eos_id = _add("<eos>")
            unk_id = _add("<unk>")
        else:
            pad_id = bos_id = eos_id = unk_id = -1

        for m in moves:
            _add(m)

        return MoveVocab(stoi=stoi, itos=itos, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, unk_id=unk_id)

    def encode_moves(self, uci_moves: Sequence[str], *, add_bos_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos_eos:
            ids.append(self.bos_id)
        for m in uci_moves:
            ids.append(self.stoi.get(m, self.unk_id))
        if add_bos_eos:
            ids.append(self.eos_id)
        return ids


def collate_autoreg(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Pad variable-length sequences into a batch.
    Returns:
      input_ids: (B, T)
      labels:    (B, T)
      attention_mask: (B, T) 1 for real tokens, 0 for pad
    """
    lens = [item["input_ids"].numel() for item in batch]
    max_len = max(lens)

    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)  # ignore index for loss
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        x = item["input_ids"]
        y = item["labels"]
        n = x.numel()
        input_ids[i, :n] = x
        labels[i, :n] = y
        attention_mask[i, :n] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ----------------------------
# Batching utilities (non-torch)
# ----------------------------

def batch_games(games: Sequence[chess.pgn.Game], batch_size: int) -> Iterator[List[chess.pgn.Game]]:
    for i in range(0, len(games), batch_size):
        yield list(games[i : i + batch_size])


# ----------------------------
# CLI: filter blitz + build vocab + quick DataLoader smoke test
# ----------------------------

def main():
    # Example usage: python ./src/grandmaster_dpo/data_processing/single_gm/clean_and_filter_initial_png.py  --pgn ./final_experiments_for_paper/experiment1/raw_pgns_twic/carlsen.pgn --gm_name carlsen --out_dir ./final_experiments_for_paper/experiment1/cleaned_and_filtered_pgns_twic
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="Path to input PGN.")
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--max_games", type=int, default=500, help="Maximum number of games to filter.")
    ap.add_argument("--min_plies", type=int, default=20)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    args = ap.parse_args()

    out_path = Path(f"{args.out_dir}/{args.gm_name}.pgn")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Filter blitz games (stream)
    blitz_games = list(
        filter_for_game_type(
            args.pgn,
            game_type="blitz",
        )
    )

    rapid_games = list(
        filter_for_game_type(
            args.pgn,
            game_type="rapid",
        )
    )

    classical_games = list(
        filter_for_game_type(
            args.pgn,
            game_type="classical",
        )
    )
    games = []
    games.extend(blitz_games[:int(args.max_games * 0.6)])
    games.extend(rapid_games[:int(args.max_games * 0.3)])
    games.extend(classical_games[:int(args.max_games * 0.1)])
    np.random.shuffle(games)
    print(f"Filtered blitz games: {len(blitz_games)}")
    print(f"Filtered rapid games: {len(rapid_games)}")
    print(f"Filtered classical games: {len(classical_games)}")
    print(f"Total games: {len(blitz_games) + len(rapid_games) + len(classical_games)}")
    print(f"Total games after filtering to target 60% blitz, 30% rapid, 10% classical (should be {args.max_games}): {len(games)}")

    # 2) Optionally write them out as a new PGN
    if out_path:
        n = write_pgn(iter(games), out_path)
        print(f"Wrote {n} games -> {out_path}")


if __name__ == "__main__":
    main()

