"""
Preprocess PGN -> (filtered blitz games) -> batches -> Torch-ready tensors.

Assumptions:
- Input is a single PGN file (e.g., 1.7MB, ~2.3k games).
- "Blitz" is determined primarily by [TimeControl], falling back to event/site strings.
- Tokenization is a simple UCI-move vocabulary built from your corpus (good enough to get
  a DataLoader working now; you can swap in Maia's native tokenization later).

Install deps (you already have these):
  pip install chess torch

Usage:
    python ./src/grandmaster_dpo/data_processing/single_gm/clean_and_filter_initial_png.py  --pgn ./data/raw/pgndownload_magnus.pgn --gm_name magnus
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
from torch.utils.data import Dataset, DataLoader


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


def is_blitz_game(
    headers: Dict[str, str],
    *,
    blitz_max_seconds: int = 8 * 60,  # generous: up to ~8 minutes estimated total
) -> bool:
    """
    Decide whether a game is blitz.
    - First: use TimeControl seconds estimate if present.
    - Fallback: keyword search in Event/Site.
    """
    tc = headers.get("TimeControl", "")
    secs = _parse_timecontrol_seconds(tc)
    if secs is not None:
        return secs <= blitz_max_seconds

    # fallback heuristic if TimeControl missing
    hay = " ".join(
        [
            headers.get("Event", ""),
            headers.get("Site", ""),
            headers.get("Round", ""),
        ]
    ).lower()
    return any(k in hay for k in ["blitz", "bullet", "3+0", "5+0", "rapid blitz"])


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


def filter_blitz_games(
    pgn_path: str,
    *,
    blitz_max_seconds: int = 8 * 60,
    require_min_plies: int = 20,  # e.g. >=10 full moves
) -> Iterator[chess.pgn.Game]:
    """
    Yield blitz games from PGN.
    Also drops very short games (often junk / forfeits / malformed).
    """
    for g in iter_pgn_games(pgn_path):
        if not is_blitz_game(g.headers, blitz_max_seconds=blitz_max_seconds):
            continue
        plies = sum(1 for _ in g.mainline_moves())
        if plies < require_min_plies:
            continue
        yield g


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
            for mv in g.mainline_moves():
                u = mv.uci()
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


def game_to_uci_moves(game: chess.pgn.Game) -> List[str]:
    return [mv.uci() for mv in game.mainline_moves()]


# ----------------------------
# Dataset + collate -> tensors
# Autoregressive next-move: x = tokens[:-1], y = tokens[1:]
# ----------------------------

class PGNBlitzDataset(Dataset):
    """
    Loads filtered blitz games into token sequences.
    For now this keeps sequences in memory (1.7MB PGN is tiny).
    """

    def __init__(
        self,
        pgn_path: str,
        vocab: MoveVocab,
        *,
        blitz_max_seconds: int = 8 * 60,
        require_min_plies: int = 20,
        max_seq_len: int = 256,  # truncate long games
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        games = list(
            filter_blitz_games(
                pgn_path,
                blitz_max_seconds=blitz_max_seconds,
                require_min_plies=require_min_plies,
            )
        )

        self.games_headers: List[Dict[str, str]] = [dict(g.headers) for g in games]
        self.seqs: List[List[int]] = []
        for g in games:
            uci = game_to_uci_moves(g)
            ids = vocab.encode_moves(uci, add_bos_eos=True)
            if len(ids) > max_seq_len:
                ids = ids[: max_seq_len - 1] + [vocab.eos_id]
            self.seqs.append(ids)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.seqs[idx]
        # autoregressive next-token prediction
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


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
    # Example usage: python ./src/grandmaster_dpo/data_processing/single_gm/clean_and_filter_initial_png.py  --pgn ./data/raw/pgndownload_magnus.pgn --gm_name magnus
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="Path to input PGN.")
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--blitz_max_seconds", type=int, default=8 * 60)
    ap.add_argument("--min_plies", type=int, default=20)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    out_path = Path(f"./processed/single_gm/{args.gm_name}.pgn")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Filter blitz games (stream)
    blitz_games = list(
        filter_blitz_games(
            args.pgn,
            blitz_max_seconds=args.blitz_max_seconds,
            require_min_plies=args.min_plies,
        )
    )
    print(f"Filtered blitz games: {len(blitz_games)}")

    # 2) Optionally write them out as a new PGN
    if out_path:
        n = write_pgn(iter(blitz_games), out_path)
        print(f"Wrote {n} games -> {out_path}")

    # 3) Build move vocab from filtered games
    vocab = MoveVocab.build_from_games(iter(blitz_games))
    print(f"Vocab size (incl specials): {len(vocab.itos)}")

    # 4) Torch Dataset + DataLoader smoke test
    #    (Re-use the original PGN path for simplicity; dataset will re-filter)
    ds = PGNBlitzDataset(
        args.pgn,
        vocab,
        blitz_max_seconds=args.blitz_max_seconds,
        require_min_plies=args.min_plies,
        max_seq_len=args.max_seq_len,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_autoreg(b, pad_id=vocab.pad_id),
    )

    batch = next(iter(dl))
    print("Batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")

    # Example: how you’d compute loss later
    # logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])  # (B,T,V)
    # loss = torch.nn.functional.cross_entropy(
    #     logits.view(-1, logits.size(-1)),
    #     batch["labels"].view(-1),
    #     ignore_index=-100,
    # )


if __name__ == "__main__":
    main()

