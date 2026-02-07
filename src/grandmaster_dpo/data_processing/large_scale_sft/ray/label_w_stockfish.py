#!/usr/bin/env python3
"""
Ray/Anyscale-friendly version of label_w_stockfish.py

What it does
- Reads partitioned actions_*.jsonl from input_dir (local or s3://...)
- Runs Stockfish MultiPV analysis per position
- Writes labeled JSONL to output_dir with:
    top_moves: [{"move","score_cp","score_wdl"}, ...] up to --multipv
    value_cp / value_wdl: best move score (side-to-move POV)
- Scales horizontally on a Ray cluster using *actors* (1 Stockfish engine per actor).

Typical Anyscale usage
- Put Stockfish binary in your image (recommended) and pass --stockfish_path
- Run on the cluster:
    python label_w_stockfish_ray.py \
      --input_dir s3://.../large_scale_fine_tuning_4k_plus \
      --output_dir s3://.../large_scale_fine_tuning_4k_plus_labeled \
      --stockfish_path /usr/local/bin/stockfish \
      --actors 64 --threads 1 --hash_mb 128 --depth 14 --multipv 10

Dependencies
- python-chess
- orjson (optional)
- fsspec (and s3fs if using s3://)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from fsspec.core import url_to_fs

import chess
import chess.engine
import ray

# Optional faster JSON
try:
    import orjson

    _HAS_ORJSON = True

    def _json_loads(b: bytes) -> Dict[str, Any]:
        return orjson.loads(b)

    def _json_dumps_line(row: Dict[str, Any]) -> bytes:
        return orjson.dumps(row) + b"\n"

except ImportError:
    _HAS_ORJSON = False

    def _json_loads(b: bytes) -> Dict[str, Any]:
        return json.loads(b.decode("utf-8"))

    def _json_dumps_line(row: Dict[str, Any]) -> bytes:
        return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")


# If you use s3://, install s3fs. We use fsspec so the same code works for local + S3.
import fsspec


_DEFAULT_DEPTH = 14
_DEFAULT_NODES = None
_MATE_CP_CONVENTION = 100_000
_CP_SCALE = 400.0

# Larger write buffer (helps local FS; fsspec backends vary)
_FILE_WRITE_BUFFER = 4 * 2**20  # 4 MiB


def _cp_to_value_target(cp: int) -> float:
    """Map centipawns (side-to-move POV) to outcome-style target in [-1, 1]."""
    x = cp / _CP_SCALE
    if x >= 0:
        win_prob = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        win_prob = exp_x / (1.0 + exp_x)
    return 2.0 * win_prob - 1.0


def _score_to_cp(score: chess.engine.PovScore, *, mate_score: int = _MATE_CP_CONVENTION) -> int:
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
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
    eng = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout=timeout)
    try:
        opts: Dict[str, Any] = {}
        if "Threads" in eng.options:
            opts["Threads"] = int(threads)
        if "Hash" in eng.options:
            opts["Hash"] = int(hash_mb)
        if opts:
            eng.configure(opts)
    except Exception as e:
        logging.warning("[pid=%s] Stockfish configure error: %s", os.getpid(), e)
    return eng


def _analyse_top_moves(
    eng: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    limit: chess.engine.Limit,
    multipv: int = 10,
) -> List[Dict[str, Any]]:
    infos = eng.analyse(board, limit, multipv=multipv)
    out: List[Dict[str, Any]] = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        root_move = pv[0]
        cp = _score_to_cp(score)
        wdl = _cp_to_value_target(cp)
        out.append({"move": root_move.uci(), "score_cp": cp, "score_wdl": round(wdl, 4)})
    return out


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _join_uri(dir_uri: str, filename: str) -> str:
    if dir_uri.endswith("/"):
        return dir_uri + filename
    return dir_uri + "/" + filename


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _fs_for(uri: str):
    # fsspec can infer filesystem from URL protocol
    # - local paths: protocol="file"
    # - s3://...: protocol="s3" (requires s3fs)
    return fsspec.open(uri, "rb").fs


def _discover_partitions(input_dir: str, *, partitions_arg: Optional[str]) -> Dict[int, str]:
    """
    Returns {partition_id: basename}, preserving any zero-padding in the original filename.
    """
    if partitions_arg:
        ids = [int(x.strip()) for x in partitions_arg.split(",") if x.strip()]
        # if user forces ids, we can't know padding; we'll just use non-padded names later
        return {pid: f"actions_{pid}.jsonl" for pid in sorted(set(ids))}

    pattern = _join_uri(input_dir, "actions_*.jsonl")
    fs, fs_pattern = url_to_fs(pattern)
    paths = fs.glob(fs_pattern)

    out: Dict[int, str] = {}
    for p in paths:
        base = p.rsplit("/", 1)[-1]
        if not base.startswith("actions_") or not base.endswith(".jsonl"):
            continue
        s = base[len("actions_"):-len(".jsonl")]
        try:
            pid = int(s)
        except ValueError:
            continue
        out[pid] = base  # <-- preserve exact basename (including zero padding)
    return dict(sorted(out.items()))


@dataclass(frozen=True)
class LabelOpts:
    depth: Optional[int] = _DEFAULT_DEPTH
    nodes: Optional[int] = _DEFAULT_NODES
    multipv: int = 10
    threads: int = 1
    hash_mb: int = 128
    log_interval: int = 50_000


@ray.remote
class StockfishLabeler:
    """
    One actor == one persistent Stockfish engine.
    Process one or more partitions sequentially inside the actor.
    """

    def __init__(self, stockfish_path: str, opts: Dict[str, Any]):
        self.stockfish_path = stockfish_path
        self.opts = LabelOpts(**opts)

        if self.opts.nodes is not None and self.opts.nodes > 0:
            self.limit = chess.engine.Limit(nodes=int(self.opts.nodes))
        else:
            self.limit = chess.engine.Limit(depth=int(self.opts.depth or _DEFAULT_DEPTH))

        self.eng = _make_stockfish(
            stockfish_path,
            threads=int(self.opts.threads),
            hash_mb=int(self.opts.hash_mb),
        )
        self.pid = os.getpid()
        logging.info("[actor pid=%s] Stockfish ready (threads=%s hash_mb=%s)", self.pid, self.opts.threads, self.opts.hash_mb)

    def close(self) -> None:
        try:
            self.eng.quit()
        except Exception:
            pass

    def process_partition(self, partition_id: int, input_uri: str, output_uri: str) -> Tuple[int, int]:
        """
        Returns (lines_written, errors)
        """
        total = 0
        errors = 0
        start = time.perf_counter()

        # Stream read + stream write (works for local and s3://)
        with fsspec.open(input_uri, "rb") as fin:
            # fsspec write buffering varies by backend; still keep a Python-side buffer
            with fsspec.open(output_uri, "wb") as fout:
                last_log = start
                for raw in fin:
                    if not raw:
                        continue
                    # Fast-path: avoid decode unless needed for json fallback
                    try:
                        row = _json_loads(raw)
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            logging.warning("[partition %s] JSON parse error: %s", partition_id, e)
                        continue

                    board_fen = row.get("board")
                    if not board_fen:
                        errors += 1
                        continue

                    try:
                        board = chess.Board(board_fen)
                    except Exception:
                        errors += 1
                        continue

                    if board.is_game_over(claim_draw=True):
                        row["top_moves"] = []
                        active_win = row.get("active_win", 0)
                        row["value_cp"] = int(active_win) * _MATE_CP_CONVENTION
                        row["value_wdl"] = float(active_win)
                    else:
                        try:
                            top_moves = _analyse_top_moves(
                                self.eng,
                                board,
                                limit=self.limit,
                                multipv=int(self.opts.multipv),
                            )
                            row["top_moves"] = top_moves
                            if top_moves:
                                row["value_cp"] = top_moves[0]["score_cp"]
                                row["value_wdl"] = top_moves[0]["score_wdl"]
                            else:
                                row["value_cp"] = 0
                                row["value_wdl"] = 0.0
                        except Exception as e:
                            errors += 1
                            if errors <= 5:
                                logging.warning("[partition %s] Engine error: %s", partition_id, e)
                            row["top_moves"] = []
                            row["value_cp"] = 0
                            row["value_wdl"] = 0.0

                    fout.write(_json_dumps_line(row))
                    total += 1

                    if total % int(self.opts.log_interval) == 0:
                        now = time.perf_counter()
                        elapsed = now - start
                        rate = total / elapsed if elapsed > 0 else 0.0
                        logging.info(
                            "[partition %s | actor pid=%s] %s lines | %.1f lines/s | errors=%s | elapsed=%.1fs",
                            partition_id,
                            self.pid,
                            total,
                            rate,
                            errors,
                            elapsed,
                        )

        elapsed = time.perf_counter() - start
        rate = total / elapsed if elapsed > 0 else 0.0
        logging.info(
            "[partition %s | actor pid=%s] DONE: %s lines in %.1fs (%.1f lines/s) | errors=%s",
            partition_id,
            self.pid,
            total,
            elapsed,
            rate,
            errors,
        )
        return total, errors


def _round_robin_assign(partition_ids: Sequence[int], n: int) -> List[List[int]]:
    chunks: List[List[int]] = [[] for _ in range(max(1, n))]
    for i, pid in enumerate(partition_ids):
        chunks[i % len(chunks)].append(pid)
    return chunks


def main() -> None:
    global _CP_SCALE
    ap = argparse.ArgumentParser(description="Label actions_*.jsonl with Stockfish top-N moves using Ray actors.")
    ap.add_argument("--input_dir", type=str, required=True, help="Directory URI containing actions_*.jsonl (local or s3://).")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory URI (local or s3://).")
    ap.add_argument("--stockfish_path", type=str, required=True, help="Path to Stockfish binary on each node.")
    ap.add_argument("--depth", type=int, default=_DEFAULT_DEPTH, help="Analysis depth (used if --nodes not set).")
    ap.add_argument("--nodes", type=int, default=None, help="Analysis node limit (overrides depth if set).")
    ap.add_argument("--multipv", type=int, default=10, help="Number of top moves to record per position.")
    ap.add_argument("--actors", type=int, default=1, help="Number of Ray actors (1 engine per actor).")
    ap.add_argument("--threads", type=int, default=1, help="Stockfish Threads per engine (1 recommended with many actors).")
    ap.add_argument("--hash_mb", type=int, default=128, help="Stockfish Hash size (MB) per engine.")
    ap.add_argument("--partitions", type=str, default=None, help="Comma-separated partition indices (e.g. 1,2,3). Default: discover actions_*.jsonl.")
    ap.add_argument("--log_interval", type=int, default=50_000, help="Log progress every N lines per partition.")
    ap.add_argument("--cp_scale", type=float, default=_CP_SCALE, help="(Advanced) CP->sigmoid scale. Default 400.")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = ap.parse_args()

    # Allow overriding CP scale from CLI (kept simple; applies globally)
    _CP_SCALE = float(args.cp_scale)

    _setup_logging(args.verbose)

    # Ray cluster init (Anyscale: address auto)
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)

    # Discover partitions
    parts = _discover_partitions(args.input_dir, partitions_arg=args.partitions)
    partition_ids = list(parts.keys())
    logging.info(
        "Starting label_w_stockfish_ray: input_dir=%s output_dir=%s partitions=%s actors=%s depth=%s nodes=%s multipv=%s threads=%s hash_mb=%s",
        args.input_dir,
        args.output_dir,
        partition_ids[:20] + (["..."] if len(partition_ids) > 20 else []),
        args.actors,
        args.depth,
        args.nodes,
        args.multipv,
        args.threads,
        args.hash_mb,
    )

    opts: Dict[str, Any] = {
        "depth": args.depth,
        "nodes": args.nodes,
        "multipv": args.multipv,
        "threads": args.threads,
        "hash_mb": args.hash_mb,
        "log_interval": args.log_interval,
    }

    # Create actors
    n_actors = max(1, int(args.actors))
    actors = [StockfishLabeler.remote(args.stockfish_path, opts) for _ in range(n_actors)]

    # Assign partitions round-robin, but execute per-partition so stragglers don’t block whole chunks.
    # (This tends to balance better because some partitions are harder.)
    pending = []
    for i, pid in enumerate(partition_ids):
        actor = actors[i % n_actors]
        fname = parts[pid]
        in_uri = _join_uri(args.input_dir, fname)
        out_uri = _join_uri(args.output_dir, fname)
        pending.append(actor.process_partition.remote(pid, in_uri, out_uri))

    total_lines = 0
    total_errors = 0
    # Consume as they finish
    for (lines, errs) in ray.get(pending):
        total_lines += int(lines)
        total_errors += int(errs)

    logging.info("ALL DONE: total_lines=%s total_errors=%s partitions=%s", total_lines, total_errors, len(partition_ids))

    # Cleanly shut down engines
    ray.get([a.close.remote() for a in actors])


if __name__ == "__main__":
    main()
