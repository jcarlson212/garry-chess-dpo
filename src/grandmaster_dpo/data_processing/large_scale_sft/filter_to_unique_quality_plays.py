"""
Preprocess a directory of PGNs -> unique (position, move) plays -> JSONL for Maia2 finetuning.

Each output row is one "ply" (half-move): board state (FEN), UCI move played, and ELOs.
Uniqueness is by hash of (FEN, UCI move) so duplicate positions across games are dropped.
Output format matches what Maia2 expects for inference/training:
  - board: FEN from active player's view (white = normal FEN, black = board.mirror().fen())
  - move: UCI move (for black, mirrored so it's in "white's view" board coordinates)
  - active_win: 1 if active player won, -1 if lost, 0 draw (for value head training)

No ELO fields (GM-only corpus). See: https://github.com/CSSLab/maia2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import chess
import chess.pgn

try:
    import orjson
    def _json_line(row: Dict[str, Any]) -> bytes:
        return orjson.dumps(row) + b"\n"
except ImportError:
    orjson = None
    def _json_line(row: Dict[str, Any]) -> bytes:
        return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")

# Larger write buffer for partition files (fewer syscalls over millions of lines)
_FILE_WRITE_BUFFER = 4 * 2**20  # 4 MiB


def mirror_square(square: str) -> str:
    """Mirror a square for black's view (rank 1<->8)."""
    file_c, rank = square[0], square[1]
    return file_c + str(9 - int(rank))


def mirror_move_uci(move_uci: str) -> str:
    """Mirror UCI move for black's view (maia2 convention)."""
    if len(move_uci) > 4:
        start, end, promo = move_uci[:2], move_uci[2:4], move_uci[4:]
        return mirror_square(start) + mirror_square(end) + promo
    return mirror_square(move_uci[:2]) + mirror_square(move_uci[2:4])

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
    with open(pgn_path, "r", encoding="utf-8", errors="ignore", buffering=_FILE_WRITE_BUFFER) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def iter_pgn_files(pgn_dir: str, pattern: str = "*.pgn") -> Iterator[Path]:
    """Yield paths to PGN files under pgn_dir (optional .pgn.zst not included by default)."""
    root = Path(pgn_dir)
    if not root.is_dir():
        return
    for p in sorted(root.rglob(pattern)):
        if p.is_file():
            yield p


def play_row_hash(board_fen: str, move_uci: str) -> bytes:
    """Stable 32-byte hash for (position, move) to deduplicate plays. Saves memory vs hex string."""
    blob = f"{board_fen}|{move_uci}".encode("utf-8")
    return hashlib.sha256(blob).digest()


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
# Per-ply extraction for Maia2 (FEN, UCI, ELO, active_win)
# ----------------------------

def _parse_result(result: str) -> Optional[int]:
    """Map PGN Result to white_win: 1=white, -1=black, 0=draw."""
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    if result == "1/2-1/2":
        return 0
    return None


def iter_plays_from_game(
    game: chess.pgn.Game,
    *,
    require_min_plies: int = 20,
    first_n_moves: int = 0,
    max_ply: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield one dict per ply (half-move) in game mainline. Single pass: validates moves and checks min plies.
    Each dict has board (FEN), move (UCI), active_win. No ELO fields (GM-only corpus).
    Board/move are from the active player's view (black positions use mirror FEN and mirrored UCI).
    """
    white_win = _parse_result(game.headers.get("Result", "?"))
    if white_win is None:
        return

    board = game.board()
    rows: List[Dict[str, Any]] = []
    ply_count = 0
    for node in game.mainline():
        move = node.move
        try:
            canonical_uci(board, move.uci())
        except Exception:
            return  # invalid game
        ply_count += 1
        if ply_count <= first_n_moves:
            board.push(move)
            continue
        if max_ply is not None and ply_count > max_ply:
            break
        if board.turn == chess.WHITE:
            board_fen = board.fen()
            move_uci = move.uci()
            active_win = white_win
        else:
            board_fen = board.mirror().fen()
            move_uci = mirror_move_uci(move.uci())
            active_win = -white_win
        rows.append({"board": board_fen, "move": move_uci, "active_win": active_win})
        board.push(move)

    if ply_count < require_min_plies:
        return
    for row in rows:
        yield row


def iter_all_plays(
    pgn_paths: Iterator[Path],
    *,
    is_variant_game_fn: Optional[Any] = None,
    game_type_filter: Optional[str] = None,
    require_min_plies: int = 20,
    first_n_moves: int = 0,
    max_ply: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield play dicts from all games in the given PGN files.
    If game_type_filter is set (e.g. 'blitz'), only games of that type are used.
    """
    is_variant = is_variant_game_fn or is_variant_game
    for pgn_path in pgn_paths:
        for game in iter_pgn_games(str(pgn_path)):
            if is_variant(game):
                continue
            if game_type_filter is not None:
                if classify_time_control(game.headers) != game_type_filter:
                    continue
            for row in iter_plays_from_game(
                game,
                require_min_plies=require_min_plies,
                first_n_moves=first_n_moves,
                max_ply=max_ply,
            ):
                yield row


def filter_to_unique_plays(
    play_iter: Iterator[Dict[str, Any]],
    *,
    seen_hashes: Optional[set] = None,
    buffer_size: int = 50_000,
) -> Iterator[Dict[str, Any]]:
    """
    Yield only plays whose (board, move) hash has not been seen.
    Optionally pass in a set to persist seen hashes across runs.
    """
    seen = set() if seen_hashes is None else seen_hashes
    buffer: List[Dict[str, Any]] = []
    for row in play_iter:
        h = play_row_hash(row["board"], row["move"])
        if h in seen:
            continue
        seen.add(h)
        buffer.append(row)
        if len(buffer) >= buffer_size:
            for r in buffer:
                yield r
            buffer = []
    for r in buffer:
        yield r


def write_plays_jsonl(play_iter: Iterator[Dict[str, Any]], out_path: str) -> int:
    """Write one JSON object per line. Returns number of lines written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "wb", buffering=_FILE_WRITE_BUFFER) as f:
        for row in play_iter:
            f.write(_json_line(row))
            n += 1
    return n


def write_plays_jsonl_partitioned(
    play_iter: Iterator[Dict[str, Any]],
    out_dir: str,
    *,
    lines_per_partition: int = 10_000_000,
    partition_prefix: str = "actions",
) -> Tuple[int, List[Path]]:
    """
    Stream plays into partition files under out_dir: actions_1.jsonl, actions_2.jsonl, ...
    Each partition has at most lines_per_partition lines. Writes as it iterates (no large in-memory buffer).
    Uses a 4 MiB write buffer and orjson if available for speed.
    Returns (total lines written, list of partition paths).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    partition_index = 1
    partition_paths: List[Path] = []
    current_path = out / f"{partition_prefix}_{partition_index}.jsonl"
    partition_paths.append(current_path)
    f = open(current_path, "wb", buffering=_FILE_WRITE_BUFFER)
    lines_in_current = 0
    try:
        for row in play_iter:
            f.write(_json_line(row))
            lines_in_current += 1
            total += 1
            if lines_in_current >= lines_per_partition:
                f.close()
                partition_index += 1
                current_path = out / f"{partition_prefix}_{partition_index}.jsonl"
                partition_paths.append(current_path)
                f = open(current_path, "wb", buffering=_FILE_WRITE_BUFFER)
                lines_in_current = 0
    finally:
        f.close()
    if lines_in_current == 0 and partition_paths:
        # Last partition is empty; remove it
        last_path = partition_paths.pop()
        if last_path.exists():
            last_path.unlink()
    return total, partition_paths


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


def collate_autoreg(batch: List[Dict], pad_id: int) -> Dict:
    """
    Pad variable-length sequences into a batch.
    Returns:
      input_ids: (B, T)
      labels:    (B, T)
      attention_mask: (B, T) 1 for real tokens, 0 for pad
    """
    import torch
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
# CLI: PGN dir -> unique plays -> JSONL for Maia2
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/data_processing/large_scale_sft/filter_to_unique_quality_plays.py

    # or this: python ./src/grandmaster_dpo/data_processing/large_scale_sft/filter_to_unique_quality_plays.py \
    # --pgn_dir=./data/raw/twic/large_scale_finetuning_4k_plus \
    #--out=./data/processed/large_scale_fine_tuning_4k_plus
    ap = argparse.ArgumentParser(
        description="Read a directory of PGNs, extract unique (position,move) plays, write JSONL for Maia2 finetuning."
    )
    ap.add_argument(
        "--pgn_dir",
        default="./data/raw/twic/large_scale_finetuning_4k_plus",
        help="Directory containing PGN files (recursively searched for *.pgn).",
    )
    ap.add_argument(
        "--out",
        default="./data/processed/large_scale_fine_tuning_4k_plus",
        help="Output directory for partition files (e.g. actions_1.jsonl, actions_2.jsonl, ...).",
    )
    ap.add_argument(
        "--lines_per_partition",
        type=int,
        default=10_000_000,
        help="Max lines per partition file (default 10M).",
    )
    ap.add_argument(
        "--game_type",
        type=str,
        default=None,
        choices=["blitz", "rapid", "classical", "bullet"],
        help="If set, only include games of this time control (from EventType). Default: all.",
    )
    ap.add_argument(
        "--min_plies",
        type=int,
        default=20,
        help="Skip games with fewer than this many half-moves.",
    )
    ap.add_argument(
        "--first_n_moves",
        type=int,
        default=0,
        help="Skip the first N half-moves of each game (opening book).",
    )
    ap.add_argument(
        "--max_ply",
        type=int,
        default=None,
        help="Cap plies per game (optional).",
    )
    ap.add_argument(
        "--buffer_size",
        type=int,
        default=50_000,
        help="Buffer size before flushing unique plays (for progress).",
    )
    ap.add_argument(
        "--pgn_glob",
        type=str,
        default="*.pgn",
        help="Glob for PGN files under pgn_dir (default: *.pgn).",
    )
    args = ap.parse_args()

    pgn_dir = Path(args.pgn_dir)
    if not pgn_dir.is_dir():
        raise SystemExit(f"Not a directory: {pgn_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pgn_paths = list(iter_pgn_files(str(pgn_dir), pattern=args.pgn_glob))
    print(f"Found {len(pgn_paths)} PGN file(s) under {pgn_dir}")

    play_iter = iter_all_plays(
        iter(pgn_paths),
        game_type_filter=args.game_type,
        require_min_plies=args.min_plies,
        first_n_moves=args.first_n_moves,
        max_ply=args.max_ply,
    )
    unique_plays = filter_to_unique_plays(play_iter, buffer_size=args.buffer_size)
    n, partition_paths = write_plays_jsonl_partitioned(
        unique_plays,
        str(out_dir),
        lines_per_partition=args.lines_per_partition,
    )
    print(f"Wrote {n} unique plays to {len(partition_paths)} partition(s) under {out_dir}")
    for p in partition_paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()





