"""
Label filter_to_unique_quality_plays JSONL output with Stockfish top-10 moves and scores.

Reads partitioned actions_*.jsonl from an input directory, runs Stockfish multipv analysis
per position, and writes each line with:
  - "top_moves": [{"move": "e2e4", "score_cp": 20, "score_wdl": 0.10}, ...] (up to multipv entries)
      - score_cp: raw centipawns from side-to-move POV
      - score_wdl: sigmoid(cp/400) mapped to [-1, 1] for value head training
  - "value_cp": Stockfish eval in centipawns from side-to-move POV (best move's score).
  - "value_wdl": float in [-1, 1], sigmoid mapping of value_cp for Maia2 value head training.
    Maia2's value head is trained on +1/0/-1 outcome; value_wdl is the expected-outcome proxy
    from Stockfish or from game result if game over.

Designed for scale: hundreds of millions of positions across many partition files, with
multiprocessing (one engine per worker) and streaming I/O.

Usage (use the project venv so chess, orjson, etc. are available):
  source .venv/bin/activate   # or: source venv/bin/activate
  python -m grandmaster_dpo.data_processing.large_scale_sft.label_w_stockfish --stockfish_path /path/to/stockfish [options]
"""

from __future__ import annotations

import argparse
import json
import math
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess
import chess.engine

try:
    import orjson
    _HAS_ORJSON = True
    def _json_line(row: Dict[str, Any]) -> bytes:
        return orjson.dumps(row) + b"\n"
except ImportError:
    _HAS_ORJSON = False
    def _json_line(row: Dict[str, Any]) -> bytes:
        return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")

# Larger write buffer for partition files
_FILE_WRITE_BUFFER = 4 * 2**20  # 4 MiB

# Default analysis limit (depth is faster per position; nodes is more consistent across positions)
_DEFAULT_DEPTH = 14
_DEFAULT_NODES = None
_MATE_CP_CONVENTION = 100_000  # mate scores mapped to ±(MATE_CP_CONVENTION - ply)
# Scale for cp -> win-prob: sigmoid(cp / CP_SCALE) maps typical evals to (0, 1)
_CP_SCALE = 400.0


def _cp_to_value_target(cp: int) -> float:
    """Map centipawns (side-to-move POV) to outcome-style target in [-1, 1] for Maia2 value head."""
    # win_prob = 1 / (1 + exp(-cp/scale)); value_target = 2*win_prob - 1
    x = cp / _CP_SCALE
    if x >= 0:
        win_prob = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        win_prob = exp_x / (1.0 + exp_x)
    return 2.0 * win_prob - 1.0


def _score_to_cp(score: chess.engine.PovScore, *, mate_score: int = _MATE_CP_CONVENTION) -> int:
    """Convert engine PovScore to centipawns (from side-to-move POV). Mate mapped to ±MATE_CP."""
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
        # Fallback for mate: use mate_score with sign from mate()
        m = rel.mate()
        if m is not None:
            return mate_score if m > 0 else -mate_score
        return 0
    return cp


def _make_stockfish(
    stockfish_path: str,
    *,
    threads: int = 1,
    hash_mb: int = 128,
    timeout: float = 30.0,
) -> chess.engine.SimpleEngine:
    """Create and configure a Stockfish engine (same pattern as gauntlet_fixed_compute)."""
    eng = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout=timeout)
    try:
        opts: Dict[str, Any] = {}
        if "Threads" in eng.options:
            opts["Threads"] = int(threads)
        if "Hash" in eng.options:
            opts["Hash"] = int(hash_mb)
        # Note: Do NOT set MultiPV here - it's automatically managed by eng.analyse(multipv=...)
        if opts:
            eng.configure(opts)
    except Exception as e:
        logging.warning("[PID=%s] Stockfish configure error: %s", os.getpid(), e)
    return eng


def _analyse_top_moves(
    eng: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    limit: chess.engine.Limit,
    multipv: int = 10,
) -> List[Dict[str, Any]]:
    """Return list of {move, score_cp, score_wdl} for top multipv moves (from side-to-move POV)."""
    infos = eng.analyse(board, limit, multipv=multipv)
    out: List[Dict[str, Any]] = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        root_move = pv[0]
        uci = root_move.uci()
        cp = _score_to_cp(score)
        wdl = _cp_to_value_target(cp)
        out.append({"move": uci, "score_cp": cp, "score_wdl": round(wdl, 4)})
    return out


def _process_one_partition(
    partition_id: int,
    input_path: Path,
    output_path: Path,
    stockfish_path: str,
    *,
    depth: Optional[int] = _DEFAULT_DEPTH,
    nodes: Optional[int] = _DEFAULT_NODES,
    multipv: int = 10,
    threads: int = 1,
    hash_mb: int = 128,
    log_interval: int = 50_000,
) -> int:
    """
    Process a single partition file: read line-by-line, label with Stockfish top moves, write.
    Returns number of lines written.
    """
    if nodes is not None and nodes > 0:
        limit = chess.engine.Limit(nodes=nodes)
    else:
        limit = chess.engine.Limit(depth=depth or _DEFAULT_DEPTH)

    eng = _make_stockfish(stockfish_path, threads=threads, hash_mb=hash_mb)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        errors = 0
        start = time.perf_counter()
        last_log = start

        with open(input_path, "rb") as fin, open(output_path, "wb", buffering=_FILE_WRITE_BUFFER) as fout:
            for raw in fin:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    row = orjson.loads(raw) if _HAS_ORJSON else json.loads(line)
                except Exception as e:
                    logging.warning("[partition %s] JSON parse error: %s", partition_id, e)
                    errors += 1
                    continue

                board_fen = row.get("board")
                if not board_fen:
                    errors += 1
                    continue
                try:
                    board = chess.Board(board_fen)
                except Exception as e:
                    logging.warning("[partition %s] Invalid FEN: %s", partition_id, e)
                    errors += 1
                    continue

                if board.is_game_over(claim_draw=True):
                    # Keep row but with empty top_moves; value from game result
                    row["top_moves"] = []
                    active_win = row.get("active_win", 0)
                    row["value_cp"] = active_win * _MATE_CP_CONVENTION  # +100k win, -100k loss, 0 draw
                    row["value_wdl"] = float(active_win)  # +1 win, -1 loss, 0 draw
                else:
                    try:
                        top_moves = _analyse_top_moves(eng, board, limit=limit, multipv=multipv)
                        row["top_moves"] = top_moves
                        # Position value: best move's score (side-to-move POV)
                        if top_moves:
                            row["value_cp"] = top_moves[0]["score_cp"]
                            row["value_wdl"] = top_moves[0]["score_wdl"]
                        else:
                            row["value_cp"] = 0
                            row["value_wdl"] = 0.0
                    except Exception as e:
                        logging.warning("[partition %s] Engine error: %s", partition_id, e)
                        errors += 1
                        row["top_moves"] = []
                        row["value_cp"] = 0
                        row["value_wdl"] = 0.0

                fout.write(_json_line(row))
                total += 1

                if total % log_interval == 0 and total > 0:
                    now = time.perf_counter()
                    elapsed = now - start
                    rate = total / elapsed if elapsed > 0 else 0
                    logging.info(
                        "[partition %s] %s lines | %.1f lines/s | errors=%s | elapsed=%.1fs",
                        partition_id, total, rate, errors, elapsed,
                    )

        elapsed = time.perf_counter() - start
        rate = total / elapsed if elapsed > 0 else 0
        logging.info(
            "[partition %s] DONE: %s lines in %.1fs (%.1f lines/s) | errors=%s",
            partition_id, total, elapsed, rate, errors,
        )
        return total
    finally:
        eng.quit()


def _run_worker(
    partition_ids: List[int],
    input_dir: Path,
    output_dir: Path,
    stockfish_path: str,
    opts: Dict[str, Any],
) -> int:
    """Process a list of partition IDs (for one worker). Returns total lines processed."""
    total = 0
    for pid in partition_ids:
        in_path = input_dir / f"actions_{pid}.jsonl"
        out_path = output_dir / f"actions_{pid}.jsonl"
        if not in_path.exists():
            logging.warning("Skip partition %s: %s not found", pid, in_path)
            continue
        total += _process_one_partition(
            pid, in_path, out_path, stockfish_path, **opts
        )
    return total


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/data_processing/large_scale_sft/label_w_stockfish.py --stockfish_path=/usr/local/bin/stockfish --workers=10
    ap = argparse.ArgumentParser(
        description="Label actions_*.jsonl with Stockfish top-10 moves and scores (multipv)."
    )
    ap.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./data/processed/large_scale_fine_tuning_4k_plus"),
        help="Directory containing actions_1.jsonl, actions_2.jsonl, ...",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./data/processed/large_scale_fine_tuning_4k_plus_labeled"),
        help="Output directory for labeled partition files.",
    )
    ap.add_argument(
        "--stockfish_path",
        type=str,
        required=True,
        help="Path to Stockfish UCI binary.",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=_DEFAULT_DEPTH,
        help="Analysis depth (used if --nodes not set).",
    )
    ap.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Analysis node limit (overrides depth if set).",
    )
    ap.add_argument(
        "--multipv",
        type=int,
        default=10,
        help="Number of top moves to record per position.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (each has one Stockfish engine).",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Stockfish Threads per engine (1 recommended when using multiple workers).",
    )
    ap.add_argument(
        "--hash_mb",
        type=int,
        default=128,
        help="Stockfish Hash size in MB per engine.",
    )
    ap.add_argument(
        "--partitions",
        type=str,
        default=None,
        help="Comma-separated partition indices to process (e.g. 1,2,3). Default: all found.",
    )
    ap.add_argument(
        "--log_interval",
        type=int,
        default=50_000,
        help="Log progress every N lines per partition.",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = ap.parse_args()

    _setup_logging(args.verbose)
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Discover or parse partition IDs
    if args.partitions is not None:
        partition_ids = [int(x.strip()) for x in args.partitions.split(",") if x.strip()]
    else:
        partition_ids = []
        for p in sorted(input_dir.glob("actions_*.jsonl")):
            try:
                # actions_3.jsonl -> 3
                num = int(p.stem.split("_")[-1])
                partition_ids.append(num)
            except ValueError:
                continue
        partition_ids.sort()

    if not partition_ids:
        raise SystemExit("No partition files found (actions_1.jsonl, actions_2.jsonl, ...).")

    logging.info(
        "Starting label_w_stockfish: input_dir=%s output_dir=%s partitions=%s workers=%s depth=%s nodes=%s multipv=%s",
        input_dir, output_dir, partition_ids, args.workers, args.depth, args.nodes, args.multipv,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    opts: Dict[str, Any] = {
        "depth": args.depth,
        "nodes": args.nodes,
        "multipv": args.multipv,
        "threads": args.threads,
        "hash_mb": args.hash_mb,
        "log_interval": args.log_interval,
    }

    if args.workers <= 1:
        total = _run_worker(partition_ids, input_dir, output_dir, args.stockfish_path, opts)
        logging.info("Total lines labeled: %s", total)
        return

    # Distribute partitions across workers (each worker gets its own Stockfish engine)
    import multiprocessing as mp
    chunks: List[List[int]] = [[] for _ in range(args.workers)]
    for i, pid in enumerate(partition_ids):
        chunks[i % args.workers].append(pid)

    with mp.Pool(args.workers) as pool:
        totals = pool.starmap(
            _run_worker,
            [
                (chunk, input_dir, output_dir, args.stockfish_path, opts)
                for chunk in chunks
            ],
        )

    logging.info("Total lines labeled: %s", sum(totals))


if __name__ == "__main__":
    main()
