#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chess
import chess.engine


JSONDict = Dict[str, Any]


def iter_partition_files(input_dir: str) -> List[str]:
    files: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".jsonl"):
            files.append(path)
    return files



def read_jsonl(path: str) -> Iterable[JSONDict]:
    with open(path, "r", encoding="utf-8", errors="ignore", buffering=1024 * 1024 * 16) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON on line {line_num} of {path}: {e}") from e



def safe_float(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)



def score_to_cp_and_mate(score: chess.engine.PovScore, turn: chess.Color) -> Tuple[Optional[int], Optional[int]]:
    rel = score.pov(turn)
    cp = rel.score(mate_score=100000)
    mate = rel.mate()
    return cp, mate



def score_to_wdl(score: chess.engine.PovScore, turn: chess.Color, ply: int) -> Dict[str, Optional[float]]:
    rel = score.pov(turn)
    try:
        wdl = rel.wdl(model="sf16", ply=max(1, ply))
        total = wdl.wins + wdl.draws + wdl.losses
        if total <= 0:
            return {
                "win_prob": None,
                "draw_prob": None,
                "loss_prob": None,
                "expected_score": None,
            }
        win_prob = wdl.wins / total
        draw_prob = wdl.draws / total
        loss_prob = wdl.losses / total
        expected_score = (wdl.wins + 0.5 * wdl.draws) / total
        return {
            "win_prob": safe_float(win_prob),
            "draw_prob": safe_float(draw_prob),
            "loss_prob": safe_float(loss_prob),
            "expected_score": safe_float(expected_score),
        }
    except Exception:
        return {
            "win_prob": None,
            "draw_prob": None,
            "loss_prob": None,
            "expected_score": None,
        }



def board_after_uci(board: chess.Board, move_uci: str) -> Optional[chess.Board]:
    try:
        move = chess.Move.from_uci(move_uci)
    except Exception:
        return None
    if move not in board.legal_moves:
        return None
    b = board.copy(stack=False)
    b.push(move)
    return b



def analyse_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move_played_uci: str,
    depth: int,
    multipv: int,
) -> Dict[str, Any]:
    info_list = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=multipv,
        info=(
            chess.engine.INFO_SCORE
            | chess.engine.INFO_PV
            | chess.engine.INFO_REFUTATION
            | chess.engine.INFO_CURRLINE
        ),
    )

    turn = board.turn
    ply = board.fullmove_number * 2 - (0 if turn == chess.WHITE else 1)

    top_moves: List[Dict[str, Any]] = []
    best_cp: Optional[int] = None
    best_expected_score: Optional[float] = None
    best_move_uci: Optional[str] = None

    for idx, info in enumerate(info_list, start=1):
        pv = info.get("pv") or []
        if not pv:
            continue

        first_move = pv[0]
        score = info.get("score")
        if score is None:
            continue

        cp_score, mate_score = score_to_cp_and_mate(score, turn)
        wdl = score_to_wdl(score, turn, ply)

        move_board = board.copy(stack=False)
        san: Optional[str] = None
        try:
            san = move_board.san(first_move)
        except Exception:
            san = None

        pv_uci = [m.uci() for m in pv]
        entry = {
            "rank": idx,
            "uci": first_move.uci(),
            "san": san,
            "cp_score": cp_score,
            "mate_score": mate_score,
            "win_prob": wdl["win_prob"],
            "draw_prob": wdl["draw_prob"],
            "loss_prob": wdl["loss_prob"],
            "expected_score": wdl["expected_score"],
            "pv_uci": pv_uci,
        }
        top_moves.append(entry)

        if idx == 1:
            best_cp = cp_score
            best_expected_score = wdl["expected_score"]
            best_move_uci = first_move.uci()

    played_board = board_after_uci(board, move_played_uci)
    played_meta: Dict[str, Any] = {
        "uci": move_played_uci,
        "legal": played_board is not None,
        "cp_score": None,
        "mate_score": None,
        "win_prob": None,
        "draw_prob": None,
        "loss_prob": None,
        "expected_score": None,
        "cp_gap_from_best": None,
        "expected_score_drop_from_best": None,
        "top_ten_rank_if_present": None,
        "is_in_stockfish_top_ten": False,
        "is_best_move": False,
    }

    for entry in top_moves:
        if entry["uci"] == move_played_uci:
            played_meta.update(
                {
                    "cp_score": entry["cp_score"],
                    "mate_score": entry["mate_score"],
                    "win_prob": entry["win_prob"],
                    "draw_prob": entry["draw_prob"],
                    "loss_prob": entry["loss_prob"],
                    "expected_score": entry["expected_score"],
                    "top_ten_rank_if_present": entry["rank"],
                    "is_in_stockfish_top_ten": True,
                    "is_best_move": entry["rank"] == 1,
                }
            )
            break

    if played_board is not None and not played_meta["is_in_stockfish_top_ten"]:
        played_info = engine.analyse(
            played_board,
            chess.engine.Limit(depth=max(8, min(depth, 18))),
            info=chess.engine.INFO_SCORE,
        )
        response_score = played_info.get("score")
        if response_score is not None:
            # Score after the played move is from the opponent's perspective to move,
            # so negate it to get evaluation from the original mover's perspective.
            cp_opp, mate_opp = score_to_cp_and_mate(response_score, played_board.turn)
            cp_score = -cp_opp if cp_opp is not None else None
            mate_score = -mate_opp if mate_opp is not None else None

            played_meta["cp_score"] = cp_score
            played_meta["mate_score"] = mate_score

            try:
                opp_rel = response_score.pov(played_board.turn)
                wdl = opp_rel.wdl(model="sf16", ply=max(1, ply + 1))
                total = wdl.wins + wdl.draws + wdl.losses
                if total > 0:
                    played_meta["win_prob"] = safe_float(wdl.losses / total)
                    played_meta["draw_prob"] = safe_float(wdl.draws / total)
                    played_meta["loss_prob"] = safe_float(wdl.wins / total)
                    played_meta["expected_score"] = safe_float((wdl.losses + 0.5 * wdl.draws) / total)
            except Exception:
                pass

    if best_cp is not None and played_meta["cp_score"] is not None:
        played_meta["cp_gap_from_best"] = int(best_cp - played_meta["cp_score"])

    if best_expected_score is not None and played_meta["expected_score"] is not None:
        played_meta["expected_score_drop_from_best"] = safe_float(best_expected_score - played_meta["expected_score"])

    return {
        "stockfish_depth": depth,
        "stockfish_multipv": multipv,
        "stockfish_best_move_uci": best_move_uci,
        "stockfish_best_cp_score": best_cp,
        "stockfish_best_expected_score": best_expected_score,
        "stockfish_top_ten_moves": top_moves,
        "stockfish_data_for_player_move": played_meta,
    }


_ENGINE: Optional[chess.engine.SimpleEngine] = None
_ENGINE_PATH: Optional[str] = None
_THREADS: int = 1
_HASH_MB: int = 256



def worker_init(engine_path: str, threads: int, hash_mb: int) -> None:
    global _ENGINE, _ENGINE_PATH, _THREADS, _HASH_MB
    _ENGINE_PATH = engine_path
    _THREADS = threads
    _HASH_MB = hash_mb
    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _ENGINE.configure({"Threads": threads, "Hash": hash_mb})



def worker_shutdown() -> None:
    global _ENGINE
    if _ENGINE is not None:
        try:
            _ENGINE.quit()
        except Exception:
            pass
        _ENGINE = None



def process_partition(
    input_path: str,
    output_path: str,
    depth: int,
    multipv: int,
    flush_every: int,
    overwrite: bool,
) -> Dict[str, Any]:
    global _ENGINE
    if _ENGINE is None:
        raise RuntimeError("Engine was not initialized in worker")

    if os.path.exists(output_path) and not overwrite:
        return {
            "partition": os.path.basename(input_path),
            "status": "skipped_exists",
            "input_path": input_path,
            "output_path": output_path,
            "rows": 0,
            "ok_rows": 0,
            "bad_rows": 0,
            "illegal_rows": 0,
            "elapsed_sec": 0.0,
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = 0
    ok_rows = 0
    bad_rows = 0
    illegal_rows = 0
    start = time.time()

    with open(output_path, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as out_f:
        for row in read_jsonl(input_path):
            rows += 1
            try:
                board_fen = row["board_t"]
                move_played = row["move_played"]
                board = chess.Board(board_fen)

                sf_meta = analyse_position(
                    engine=_ENGINE,
                    board=board,
                    move_played_uci=move_played,
                    depth=depth,
                    multipv=multipv,
                )

                out_row = dict(row)
                out_row.update(sf_meta)
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

                if not sf_meta["stockfish_data_for_player_move"]["legal"]:
                    illegal_rows += 1
                else:
                    ok_rows += 1

            except Exception as e:
                bad_rows += 1
                err_row = dict(row)
                err_row["stockfish_error"] = str(e)
                out_f.write(json.dumps(err_row, ensure_ascii=False) + "\n")

            if rows % flush_every == 0:
                out_f.flush()
                elapsed = time.time() - start
                rate = rows / max(elapsed, 1e-9)
                print(
                    f"[{os.path.basename(input_path)}] rows={rows:,} ok={ok_rows:,} bad={bad_rows:,} "
                    f"illegal={illegal_rows:,} rate={rate:.1f} rows/sec",
                    file=sys.stderr,
                )

    elapsed = time.time() - start
    return {
        "partition": os.path.basename(input_path),
        "status": "done",
        "input_path": input_path,
        "output_path": output_path,
        "rows": rows,
        "ok_rows": ok_rows,
        "bad_rows": bad_rows,
        "illegal_rows": illegal_rows,
        "elapsed_sec": elapsed,
    }



def build_default_output_root(input_root: str) -> str:
    input_root = os.path.abspath(input_root)
    parent = os.path.dirname(input_root)
    base = os.path.basename(input_root.rstrip(os.sep))
    if base == "splits":
        return os.path.join(parent, "splits_w_post_processing_metadata")
    return os.path.join(parent, f"{base}_w_post_processing_metadata")



def process_split_dir(
    split_name: str,
    input_dir: str,
    output_dir: str,
    engine_path: str,
    workers: int,
    threads_per_worker: int,
    hash_mb_per_worker: int,
    depth: int,
    multipv: int,
    flush_every: int,
    overwrite: bool,
) -> Dict[str, Any]:
    input_files = iter_partition_files(input_dir)
    if not input_files:
        print(f"[warn] no files found in {input_dir}", file=sys.stderr)
        return {
            "split": split_name,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "partitions": [],
            "totals": {
                "rows": 0,
                "ok_rows": 0,
                "bad_rows": 0,
                "illegal_rows": 0,
                "elapsed_sec": 0.0,
            },
        }

    os.makedirs(output_dir, exist_ok=True)
    print(
        f"[main] split={split_name} files={len(input_files)} workers={workers} "
        f"threads/worker={threads_per_worker} hash_mb/worker={hash_mb_per_worker}",
        file=sys.stderr,
    )

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=worker_init,
        initargs=(engine_path, threads_per_worker, hash_mb_per_worker),
    ) as ex:
        futures = []
        for input_path in input_files:
            output_path = os.path.join(output_dir, os.path.basename(input_path))
            futures.append(
                ex.submit(
                    process_partition,
                    input_path,
                    output_path,
                    depth,
                    multipv,
                    flush_every,
                    overwrite,
                )
            )

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(
                f"[{split_name}] {res['partition']}: status={res['status']} rows={res['rows']:,} "
                f"ok={res['ok_rows']:,} bad={res['bad_rows']:,} illegal={res['illegal_rows']:,} "
                f"elapsed={res['elapsed_sec']:.1f}s",
                file=sys.stderr,
            )

    results_sorted = sorted(results, key=lambda x: x["partition"])
    return {
        "split": split_name,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "partitions": results_sorted,
        "totals": {
            "rows": sum(x["rows"] for x in results_sorted),
            "ok_rows": sum(x["ok_rows"] for x in results_sorted),
            "bad_rows": sum(x["bad_rows"] for x in results_sorted),
            "illegal_rows": sum(x["illegal_rows"] for x in results_sorted),
            "elapsed_sec": sum(x["elapsed_sec"] for x in results_sorted),
        },
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, help="Root dir containing train/eval/test split dirs")
    ap.add_argument(
        "--output-root",
        default=None,
        help="Root dir for enriched output. Defaults to sibling dir named splits_w_post_processing_metadata",
    )
    ap.add_argument("--engine-path", required=True, help="Path to Stockfish binary")
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--multipv", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 8) // 2)))
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--hash-mb-per-worker", type=int, default=1024)
    ap.add_argument("--flush-every", type=int, default=5000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest-path", default=None)
    args = ap.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root or build_default_output_root(input_root))

    train_in = os.path.join(input_root, "train")
    eval_in = os.path.join(input_root, "eval")
    test_in = os.path.join(input_root, "test")

    train_out = os.path.join(output_root, "train")
    eval_out = os.path.join(output_root, "eval")
    test_out = os.path.join(output_root, "test")

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not os.path.exists(args.engine_path):
        raise FileNotFoundError(f"Stockfish binary not found: {args.engine_path}")

    t0 = time.time()

    train_summary = process_split_dir(
        split_name="train",
        input_dir=train_in,
        output_dir=train_out,
        engine_path=args.engine_path,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        hash_mb_per_worker=args.hash_mb_per_worker,
        depth=args.depth,
        multipv=args.multipv,
        flush_every=args.flush_every,
        overwrite=args.overwrite,
    )
    eval_summary = process_split_dir(
        split_name="eval",
        input_dir=eval_in,
        output_dir=eval_out,
        engine_path=args.engine_path,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        hash_mb_per_worker=args.hash_mb_per_worker,
        depth=args.depth,
        multipv=args.multipv,
        flush_every=args.flush_every,
        overwrite=args.overwrite,
    )
    test_summary = process_split_dir(
        split_name="test",
        input_dir=test_in,
        output_dir=test_out,
        engine_path=args.engine_path,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        hash_mb_per_worker=args.hash_mb_per_worker,
        depth=args.depth,
        multipv=args.multipv,
        flush_every=args.flush_every,
        overwrite=args.overwrite,
    )

    manifest = {
        "input_root": input_root,
        "output_root": output_root,
        "engine_path": os.path.abspath(args.engine_path),
        "depth": args.depth,
        "multipv": args.multipv,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "hash_mb_per_worker": args.hash_mb_per_worker,
        "flush_every": args.flush_every,
        "overwrite": args.overwrite,
        "splits": {
            "train": train_summary,
            "eval": eval_summary,
            "test": test_summary,
        },
        "elapsed_wall_sec": time.time() - t0,
    }

    manifest_path = args.manifest_path or os.path.join(output_root, "manifest_stockfish_post_processing.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[main] output_root={output_root}", file=sys.stderr)
    print(f"[main] manifest={manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
Example:

python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/add_stockfish_post_processing_metadata.py \
    --input-root ./final_experiments_for_paper/experiment2_style_model/splits \
    --engine-path /opt/homebrew/bin/stockfish \
    --depth 18 \
    --multipv 10 \
    --workers 6 \
    --threads-per-worker 2 \
    --hash-mb-per-worker 1024 \
    --overwrite

Notes:
- Start with workers=4 to 6 on Apple Silicon and then scale up.
- Stockfish uses CPU, not GPU, so the main knobs are workers, threads-per-worker, and hash.
- The output rows preserve the original schema and add Stockfish-derived metadata.
"""
