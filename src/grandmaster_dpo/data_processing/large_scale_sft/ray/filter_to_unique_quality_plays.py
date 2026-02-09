#!/usr/bin/env python3
"""
Ray/Anyscale-friendly version of filter_to_unique_quality_plays.py

What it does:
- Reads PGN files from input_s3_path (s3://bucket/path/to/pgns/)
- If there's a single large PGN, splits it into chunks for parallel processing
- Extracts unique (position, move) plays per chunk using Ray tasks
- Writes partitioned JSONL output to output_s3_path

Output format (one JSON per line):
  - board: FEN from active player's view (black positions mirrored)
  - move: UCI move (black moves mirrored to white's view)
  - active_win: 1 if active player won, -1 if lost, 0 draw

Typical Anyscale usage:
    python filter_to_unique_quality_plays.py \
      --input_s3_path s3://bucket/raw/pgns/ \
      --output_s3_path s3://bucket/processed/actions/ \
      --num_output_partitions 1000 \
      --games_per_chunk 5000

Dependencies:
- python-chess
- orjson (optional, for faster JSON)
- fsspec, s3fs (for S3 access)
- ray
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import chess
import chess.pgn
import fsspec
import ray
from fsspec.core import url_to_fs

# Optional faster JSON
try:
    import orjson

    _HAS_ORJSON = True

    def _json_dumps_line(row: Dict[str, Any]) -> bytes:
        return orjson.dumps(row) + b"\n"

except ImportError:
    _HAS_ORJSON = False

    def _json_dumps_line(row: Dict[str, Any]) -> bytes:
        return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FILE_WRITE_BUFFER = 4 * 2**20  # 4 MiB
_DEFAULT_MIN_PLIES = 20
_DEFAULT_FIRST_N_MOVES = 0
_DEFAULT_NUM_CHUNKS_FOR_LARGE_FILES = 1000  # Split large PGNs into this many chunks
_DEFAULT_LOG_INTERVAL = 10_000
_LARGE_FILE_THRESHOLD_MB = 50  # Files larger than this get chunked


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# S3/fsspec utilities
# ---------------------------------------------------------------------------

def _join_uri(dir_uri: str, filename: str) -> str:
    if dir_uri.endswith("/"):
        return dir_uri + filename
    return dir_uri + "/" + filename


def _get_file_size(uri: str) -> int:
    """Get file size in bytes."""
    fs, path = url_to_fs(uri)
    try:
        info = fs.info(path)
        return info.get("size", 0)
    except Exception:
        return 0


def _discover_pgn_files(input_dir: str, pattern: str = "*.pgn") -> List[Tuple[str, int]]:
    """
    Discover PGN files in the input directory (local or S3).
    Returns list of (uri, size_bytes) tuples.
    """
    glob_pattern = _join_uri(input_dir, pattern)
    fs, fs_pattern = url_to_fs(glob_pattern)
    paths = fs.glob(fs_pattern)
    
    # Convert to full URIs and get sizes
    protocol = input_dir.split("://")[0] if "://" in input_dir else "file"
    results = []
    
    for p in sorted(paths):
        if protocol == "file":
            uri = p
        else:
            uri = f"{protocol}://{p}"
        
        size = _get_file_size(uri)
        results.append((uri, size))
    
    return results


def _read_file_bytes(uri: str) -> bytes:
    """Read entire file into memory as bytes."""
    with fsspec.open(uri, "rb") as f:
        return f.read()


def _write_file_bytes(uri: str, data: bytes) -> None:
    """Write bytes to a file (local or S3)."""
    with fsspec.open(uri, "wb") as f:
        f.write(data)


def _stream_file_text(uri: str) -> Iterator[str]:
    """Stream file line by line as text."""
    with fsspec.open(uri, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


# ---------------------------------------------------------------------------
# Chess utilities (from original filter_to_unique_quality_plays.py)
# ---------------------------------------------------------------------------

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
    """Check if game is a chess variant (960, etc.)."""
    variant = (g.headers.get("Variant") or "").lower()
    if variant != "":
        return True
    if g.headers.get("FEN") and "960" in (g.headers.get("SetUp") or ""):
        return True
    return False


def canonical_uci(board: chess.Board, move_str: str) -> str:
    """
    Canonicalize ANY move (UCI or SAN) into a legal, board-consistent UCI.
    """
    move_str = move_str.strip()

    # Try UCI first
    try:
        mv = chess.Move.from_uci(move_str)
        if mv in board.legal_moves:
            return mv.uci()
    except Exception:
        pass

    # Try SAN
    try:
        mv = board.parse_san(move_str)
        if mv in board.legal_moves:
            return mv.uci()
    except Exception:
        pass

    raise ValueError(f"Illegal or unparseable move {move_str!r} for FEN {board.fen()}")


def _parse_result(result: str) -> Optional[int]:
    """Map PGN Result to white_win: 1=white, -1=black, 0=draw."""
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    if result == "1/2-1/2":
        return 0
    return None


# ---------------------------------------------------------------------------
# Time control classification
# ---------------------------------------------------------------------------

_TC_RE = re.compile(r"^\s*(\d+)\s*([+|/])\s*(\d+)\s*$")
_INT_RE = re.compile(r"^\s*(\d+)\s*$")


def classify_time_control(headers: Dict[str, str]) -> str:
    et = headers.get("EventType", "").lower()

    if "bullet" in et:
        return "bullet"
    if "rapid" in et:
        return "rapid"
    if "blitz" in et:
        return "blitz"

    return "classical"


# ---------------------------------------------------------------------------
# PGN parsing utilities
# ---------------------------------------------------------------------------

def iter_pgn_games_from_string(pgn_text: str) -> Iterator[chess.pgn.Game]:
    """Stream games from a PGN string without loading all into memory."""
    pgn_io = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        yield game


def split_large_pgn_fast(
    pgn_uri: str,
    num_chunks: int,
) -> List[ray.ObjectRef]:
    """
    Fast split of a large PGN file into chunks by finding game boundaries.
    
    Instead of parsing each game (slow), we:
    1. Read the raw text
    2. Find all positions where games start (lines beginning with '[Event ')
    3. Return chunk boundaries (start_game_idx, end_game_idx) for lazy loading
    
    Returns a list of (start_game_idx, end_game_idx) tuples.
    The actual content is extracted later to avoid holding everything in memory.
    """
    logging.info("Fast-splitting large PGN file: %s into ~%d chunks", pgn_uri, num_chunks)
    start = time.perf_counter()
    
    # Read raw file content
    with fsspec.open(pgn_uri, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    read_time = time.perf_counter() - start
    logging.info("  Read %.1f MB in %.1fs", len(content) / (1024 * 1024), read_time)
    
    # Find all game start positions (lines starting with [Event)
    # We look for newline followed by [Event or start of file with [Event
    game_starts: List[int] = []
    
    # Check if file starts with [Event
    if content.startswith("[Event "):
        game_starts.append(0)
    
    # Find all \n[Event positions
    pos = 0
    search_pattern = "\n[Event "
    while True:
        pos = content.find(search_pattern, pos)
        if pos == -1:
            break
        game_starts.append(pos + 1)  # +1 to skip the newline
        pos += 1
    
    total_games = len(game_starts)
    if total_games == 0:
        logging.warning("No games found in %s", pgn_uri)
        return []
    
    find_time = time.perf_counter() - start - read_time
    logging.info("  Found %d games in %.1fs", total_games, find_time)
    
    # Calculate how many games per chunk
    games_per_chunk = max(1, total_games // num_chunks)
    
    # Build chunk boundaries and extract content immediately, putting into Ray object store
    # This way we don't hold all chunks in Python memory at once
    chunk_refs: List[ray.ObjectRef] = []
    chunk_start_idx = 0
    chunks_created = 0
    
    while chunk_start_idx < total_games:
        chunk_end_idx = min(chunk_start_idx + games_per_chunk, total_games)
        
        # Get byte positions
        start_pos = game_starts[chunk_start_idx]
        if chunk_end_idx < total_games:
            end_pos = game_starts[chunk_end_idx]
        else:
            end_pos = len(content)
        
        chunk_text = content[start_pos:end_pos].strip()
        if chunk_text:
            # Immediately put into Ray object store and release Python reference
            chunk_refs.append(ray.put(chunk_text))
            chunks_created += 1
        
        chunk_start_idx = chunk_end_idx
    
    # Release the large content string
    del content
    del game_starts
    
    elapsed = time.perf_counter() - start
    logging.info(
        "  Split complete: %d games -> %d chunks in %.1fs (%.0f games/s)",
        total_games, chunks_created, elapsed, total_games / elapsed if elapsed > 0 else 0
    )
    
    return chunk_refs


def iter_plays_from_game(
    game: chess.pgn.Game,
    *,
    require_min_plies: int = 20,
    first_n_moves: int = 0,
    max_ply: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield one dict per ply (half-move) in game mainline.
    Each dict has board (FEN), move (UCI), active_win.
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


def play_row_hash(board_fen: str, move_uci: str) -> bytes:
    """Stable 32-byte hash for (position, move) to deduplicate plays."""
    blob = f"{board_fen}|{move_uci}".encode("utf-8")
    return hashlib.sha256(blob).digest()


# ---------------------------------------------------------------------------
# Processing options dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessingOpts:
    min_plies: int = _DEFAULT_MIN_PLIES
    first_n_moves: int = _DEFAULT_FIRST_N_MOVES
    max_ply: Optional[int] = None
    game_type: Optional[str] = None
    log_interval: int = _DEFAULT_LOG_INTERVAL


# ---------------------------------------------------------------------------
# Ray task: Process a PGN chunk and write directly to S3
# ---------------------------------------------------------------------------

@ray.remote
def process_and_write_chunk(
    chunk_id: int,
    pgn_text: str,
    output_uri: str,
    opts_dict: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    """
    Process a PGN text chunk and write directly to S3.
    Deduplication happens within this chunk only (no global dedup).
    
    Returns (chunk_id, games_processed, games_skipped, lines_written).
    """
    opts = ProcessingOpts(**opts_dict)
    pid = os.getpid()
    
    start = time.perf_counter()
    logging.info("[pid=%s] Starting chunk %d -> %s", pid, chunk_id, output_uri.split("/")[-1])
    
    seen_hashes: set = set()
    games_processed = 0
    games_skipped = 0
    lines_written = 0
    
    # Build output in memory buffer, then write once
    output_buffer = io.BytesIO()
    
    for game in iter_pgn_games_from_string(pgn_text):
        # Skip variants
        if is_variant_game(game):
            games_skipped += 1
            continue
        
        # Filter by game type if specified
        if opts.game_type is not None:
            if classify_time_control(game.headers) != opts.game_type:
                games_skipped += 1
                continue
        
        # Extract plays and write immediately
        for row in iter_plays_from_game(
            game,
            require_min_plies=opts.min_plies,
            first_n_moves=opts.first_n_moves,
            max_ply=opts.max_ply,
        ):
            h = play_row_hash(row["board"], row["move"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                output_buffer.write(_json_dumps_line(row))
                lines_written += 1
        
        games_processed += 1
    
    # Write to S3
    try:
        _write_file_bytes(output_uri, output_buffer.getvalue())
    except Exception as e:
        logging.error("[pid=%s] Failed to write chunk %d to %s: %s", pid, chunk_id, output_uri, e)
        return chunk_id, games_processed, games_skipped, 0
    
    elapsed = time.perf_counter() - start
    logging.info(
        "[pid=%s] Finished chunk %d: %d games (%d skipped), %d lines written to %s in %.1fs",
        pid, chunk_id, games_processed, games_skipped, lines_written, output_uri.split("/")[-1], elapsed
    )
    
    return chunk_id, games_processed, games_skipped, lines_written


@ray.remote
def process_and_write_file(
    file_id: int,
    pgn_uri: str,
    output_uri: str,
    opts_dict: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    """
    Process a single PGN file and write directly to S3.
    Deduplication happens within this file only (no global dedup).
    
    Returns (file_id, games_processed, games_skipped, lines_written).
    """
    opts = ProcessingOpts(**opts_dict)
    pid = os.getpid()
    
    start = time.perf_counter()
    logging.info("[pid=%s] Starting file %s -> %s", pid, pgn_uri.split("/")[-1], output_uri.split("/")[-1])
    
    # Read PGN file
    try:
        pgn_bytes = _read_file_bytes(pgn_uri)
        pgn_text = pgn_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logging.error("[pid=%s] Failed to read %s: %s", pid, pgn_uri, e)
        return file_id, 0, 0, 0
    
    seen_hashes: set = set()
    games_processed = 0
    games_skipped = 0
    lines_written = 0
    
    # Build output in memory buffer
    output_buffer = io.BytesIO()
    
    for game in iter_pgn_games_from_string(pgn_text):
        # Skip variants
        if is_variant_game(game):
            games_skipped += 1
            continue
        
        # Filter by game type if specified
        if opts.game_type is not None:
            if classify_time_control(game.headers) != opts.game_type:
                games_skipped += 1
                continue
        
        # Extract plays and write immediately
        for row in iter_plays_from_game(
            game,
            require_min_plies=opts.min_plies,
            first_n_moves=opts.first_n_moves,
            max_ply=opts.max_ply,
        ):
            h = play_row_hash(row["board"], row["move"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                output_buffer.write(_json_dumps_line(row))
                lines_written += 1
        
        games_processed += 1
    
    # Write to S3
    try:
        _write_file_bytes(output_uri, output_buffer.getvalue())
    except Exception as e:
        logging.error("[pid=%s] Failed to write file %d to %s: %s", pid, file_id, output_uri, e)
        return file_id, games_processed, games_skipped, 0
    
    elapsed = time.perf_counter() - start
    logging.info(
        "[pid=%s] Finished file %s: %d games (%d skipped), %d lines written in %.1fs",
        pid, pgn_uri.split("/")[-1], games_processed, games_skipped, lines_written, elapsed
    )
    
    return file_id, games_processed, games_skipped, lines_written


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    # python filter_to_unique_quality_plays.py --num_chunks_for_largo-ml-artifacts/chess-engine/raw_pgns/twic_1k_plus_player_games/ --output_s3_path s3://crljaso-ml-ard_state_actions/twic_1k_plus_player_games/
    ap = argparse.ArgumentParser(
        description="Ray-parallelized PGN -> unique plays JSONL processor for Maia2 finetuning."
    )
    ap.add_argument(
        "--input_s3_path",
        type=str,
        required=True,
        help="S3 URI to directory containing PGN files (e.g., s3://bucket/raw/pgns/).",
    )
    ap.add_argument(
        "--output_s3_path",
        type=str,
        required=True,
        help="S3 URI for output directory (e.g., s3://bucket/processed/actions/).",
    )
    ap.add_argument(
        "--num_chunks_for_large_files",
        type=int,
        default=_DEFAULT_NUM_CHUNKS_FOR_LARGE_FILES,
        help=f"Number of chunks to split large PGN files into (default: {_DEFAULT_NUM_CHUNKS_FOR_LARGE_FILES}).",
    )
    ap.add_argument(
        "--large_file_threshold_mb",
        type=int,
        default=_LARGE_FILE_THRESHOLD_MB,
        help=f"Files larger than this (MB) will be split into chunks (default: {_LARGE_FILE_THRESHOLD_MB}).",
    )
    ap.add_argument(
        "--game_type",
        type=str,
        default=None,
        choices=["blitz", "rapid", "classical", "bullet"],
        help="If set, only include games of this time control. Default: all.",
    )
    ap.add_argument(
        "--min_plies",
        type=int,
        default=_DEFAULT_MIN_PLIES,
        help=f"Skip games with fewer than this many half-moves (default: {_DEFAULT_MIN_PLIES}).",
    )
    ap.add_argument(
        "--first_n_moves",
        type=int,
        default=_DEFAULT_FIRST_N_MOVES,
        help="Skip the first N half-moves of each game (opening book).",
    )
    ap.add_argument(
        "--max_ply",
        type=int,
        default=None,
        help="Cap plies per game (optional).",
    )
    ap.add_argument(
        "--pgn_glob",
        type=str,
        default="*.pgn",
        help="Glob pattern for PGN files (default: *.pgn).",
    )
    ap.add_argument(
        "--partition_prefix",
        type=str,
        default="actions",
        help="Prefix for output partition files (default: actions).",
    )
    ap.add_argument(
        "--log_interval",
        type=int,
        default=_DEFAULT_LOG_INTERVAL,
        help="Log progress every N items.",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = ap.parse_args()

    _setup_logging(args.verbose)

    # Initialize Ray
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)
    
    logging.info(
        "Starting filter_to_unique_quality_plays (Ray): input=%s output=%s",
        args.input_s3_path,
        args.output_s3_path,
    )

    # ---------------------------------------------------------------------------
    # Step 1: Discover PGN files and their sizes
    # ---------------------------------------------------------------------------
    pgn_files_with_sizes = _discover_pgn_files(args.input_s3_path, pattern=args.pgn_glob)
    if not pgn_files_with_sizes:
        logging.error("No PGN files found in %s with pattern %s", args.input_s3_path, args.pgn_glob)
        return
    
    total_size_mb = sum(size for _, size in pgn_files_with_sizes) / (1024 * 1024)
    logging.info("Found %d PGN file(s) (%.1f MB total)", len(pgn_files_with_sizes), total_size_mb)
    for uri, size in pgn_files_with_sizes[:5]:
        logging.info("  - %s (%.1f MB)", uri.split("/")[-1], size / (1024 * 1024))
    if len(pgn_files_with_sizes) > 5:
        logging.info("  ... and %d more", len(pgn_files_with_sizes) - 5)

    # ---------------------------------------------------------------------------
    # Step 2: Decide processing strategy based on file sizes
    # ---------------------------------------------------------------------------
    large_file_threshold_bytes = args.large_file_threshold_mb * 1024 * 1024
    
    # Separate large files (need chunking) from small files (process directly)
    large_files = [(uri, size) for uri, size in pgn_files_with_sizes if size > large_file_threshold_bytes]
    small_files = [(uri, size) for uri, size in pgn_files_with_sizes if size <= large_file_threshold_bytes]
    
    logging.info(
        "Processing strategy: %d large file(s) to chunk, %d small file(s) to process directly",
        len(large_files), len(small_files)
    )

    opts_dict = {
        "min_plies": args.min_plies,
        "first_n_moves": args.first_n_moves,
        "max_ply": args.max_ply,
        "game_type": args.game_type,
        "log_interval": args.log_interval,
    }

    start_process = time.perf_counter()
    futures = []
    partition_counter = 0
    # Estimate total partitions for zero-padding (large files chunks + small files)
    estimated_partitions = len(large_files) * args.num_chunks_for_large_files + len(small_files)
    pad_width = max(4, len(str(estimated_partitions)))  # At least 4 digits

    # ---------------------------------------------------------------------------
    # Step 2a: Split large files into chunks and submit as process+write tasks
    # ---------------------------------------------------------------------------
    for pgn_uri, size in large_files:
        logging.info(
            "Fast-splitting large file: %s (%.1f MB) into ~%d chunks",
            pgn_uri.split("/")[-1], size / (1024 * 1024), args.num_chunks_for_large_files
        )
        
        # Fast split by finding game boundaries (no PGN parsing)
        # Returns Ray object refs already in the object store
        chunk_refs = split_large_pgn_fast(pgn_uri, args.num_chunks_for_large_files)
        
        logging.info("Submitting %d process+write tasks for %s", len(chunk_refs), pgn_uri.split("/")[-1])
        
        # Submit each chunk as a separate Ray task that processes AND writes
        for i, chunk_ref in enumerate(chunk_refs):
            partition_counter += 1
            filename = f"{args.partition_prefix}_{str(partition_counter).zfill(pad_width)}.jsonl"
            output_uri = _join_uri(args.output_s3_path, filename)
            futures.append(process_and_write_chunk.remote(partition_counter, chunk_ref, output_uri, opts_dict))

    # ---------------------------------------------------------------------------
    # Step 2b: Submit small files directly as process+write tasks
    # ---------------------------------------------------------------------------
    if small_files:
        logging.info("Submitting %d small file process+write tasks...", len(small_files))
        for pgn_uri, _ in small_files:
            partition_counter += 1
            filename = f"{args.partition_prefix}_{str(partition_counter).zfill(pad_width)}.jsonl"
            output_uri = _join_uri(args.output_s3_path, filename)
            futures.append(process_and_write_file.remote(partition_counter, pgn_uri, output_uri, opts_dict))

    # ---------------------------------------------------------------------------
    # Step 3: Wait for all tasks to complete (they write as they finish)
    # ---------------------------------------------------------------------------
    logging.info("Waiting for %d process+write tasks to complete...", len(futures))
    logging.info("(Each task writes directly to S3 as it finishes - no blocking)")
    
    results = ray.get(futures)
    
    elapsed_total = time.perf_counter() - start_process

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total_games = sum(r[1] for r in results)
    total_skipped = sum(r[2] for r in results)
    total_written = sum(r[3] for r in results)
    
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info("  Input PGN files: %d", len(pgn_files_with_sizes))
    logging.info("  Large files chunked: %d", len(large_files))
    logging.info("  Total processing tasks: %d", len(futures))
    logging.info("  Total games processed: %d", total_games)
    logging.info("  Total games skipped: %d", total_skipped)
    logging.info("  Output partitions written: %d", partition_counter)
    logging.info("  Total lines written: %d", total_written)
    logging.info("  Total time: %.1fs", elapsed_total)
    logging.info("  Throughput: %.0f lines/s", total_written / elapsed_total if elapsed_total > 0 else 0)
    logging.info("=" * 60)
    logging.info("")
    logging.info("NOTE: Deduplication was done per-chunk only (not globally).")
    logging.info("      Some duplicate (board, move) pairs may exist across partitions.")


if __name__ == "__main__":
    main()
