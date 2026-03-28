#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Set, Tuple


def iter_partition_files(input_dir: str) -> List[str]:
    files: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".jsonl"):
            files.append(path)
    return files


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore", buffering=1024 * 1024 * 16) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON on line {line_num} of {path}: {e}") from e


def collect_players_from_partition(path: str) -> Dict[str, object]:
    players: Set[str] = set()
    rows = 0
    bad_rows = 0

    for row in read_jsonl(path):
        rows += 1
        try:
            p = str(row["player_id"]).strip()
            o = str(row["opponent_id"]).strip()
            if p:
                players.add(p)
            if o:
                players.add(o)
        except Exception:
            bad_rows += 1

    return {
        "partition": os.path.splitext(os.path.basename(path))[0],
        "path": path,
        "rows": rows,
        "bad_rows": bad_rows,
        "players": sorted(players),
    }


def split_players(
    all_players: List[str],
    seed: int,
    train_frac: float,
    eval_frac: float,
    test_frac: float,
) -> Tuple[Set[str], Set[str], Set[str]]:
    if not math.isclose(train_frac + eval_frac + test_frac, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Split fractions must sum to 1.0, got "
            f"{train_frac} + {eval_frac} + {test_frac} = {train_frac + eval_frac + test_frac}"
        )

    players = sorted(set(all_players))
    rng = random.Random(seed)
    rng.shuffle(players)

    n = len(players)
    n_train = int(n * train_frac)
    n_eval = int(n * eval_frac)
    n_test = n - n_train - n_eval

    train_players = set(players[:n_train])
    eval_players = set(players[n_train:n_train + n_eval])
    test_players = set(players[n_train + n_eval:])

    assert len(train_players) == n_train
    assert len(eval_players) == n_eval
    assert len(test_players) == n_test
    assert len(train_players | eval_players | test_players) == n

    return train_players, eval_players, test_players


def classify_row(
    row: dict,
    train_players: Set[str],
    eval_players: Set[str],
    test_players: Set[str],
) -> str | None:
    """
    Rules:
    - train: mover in train AND opponent in train
    - eval: mover in eval AND opponent in train or eval (not test)
    - test: mover in test AND opponent can be anywhere
    - else: discard
    """
    mover = str(row.get("player_id", "")).strip()
    opp = str(row.get("opponent_id", "")).strip()

    if mover in train_players and opp in train_players:
        return "train"

    if mover in eval_players and (opp in train_players or opp in eval_players):
        return "eval"

    if mover in test_players:
        return "test"

    return None


def split_partition_rows(
    input_path: str,
    train_dir: str,
    eval_dir: str,
    test_dir: str,
    train_players: Set[str],
    eval_players: Set[str],
    test_players: Set[str],
    flush_every: int = 100000,
) -> Dict[str, object]:
    partition_filename = os.path.basename(input_path)
    partition_stem, _ = os.path.splitext(partition_filename)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_path = os.path.join(train_dir, f"{partition_stem}.jsonl")
    eval_path = os.path.join(eval_dir, f"{partition_stem}.jsonl")
    test_path = os.path.join(test_dir, f"{partition_stem}.jsonl")

    total_rows = 0
    bad_rows = 0
    dropped_rows = 0
    train_rows = 0
    eval_rows = 0
    test_rows = 0

    with (
        open(train_path, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as train_f,
        open(eval_path, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as eval_f,
        open(test_path, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as test_f,
    ):
        for row in read_jsonl(input_path):
            total_rows += 1

            try:
                split_name = classify_row(row, train_players, eval_players, test_players)

                if split_name == "train":
                    train_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    train_rows += 1
                elif split_name == "eval":
                    eval_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    eval_rows += 1
                elif split_name == "test":
                    test_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    test_rows += 1
                else:
                    dropped_rows += 1

                if total_rows % flush_every == 0:
                    train_f.flush()
                    eval_f.flush()
                    test_f.flush()
                    print(
                        f"[{partition_stem}] rows={total_rows:,} "
                        f"train={train_rows:,} eval={eval_rows:,} test={test_rows:,} "
                        f"dropped={dropped_rows:,} bad_rows={bad_rows:,}",
                        file=sys.stderr,
                    )

            except Exception:
                bad_rows += 1

    return {
        "partition": partition_stem,
        "input_path": input_path,
        "train_path": train_path,
        "eval_path": eval_path,
        "test_path": test_path,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "test_rows": test_rows,
        "dropped_rows": dropped_rows,
        "bad_rows": bad_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing partitioned input .jsonl files")
    ap.add_argument("--train-dir", required=True, help="Output directory for train partition .jsonl files")
    ap.add_argument("--eval-dir", required=True, help="Output directory for eval partition .jsonl files")
    ap.add_argument("--test-dir", required=True, help="Output directory for test partition .jsonl files")
    ap.add_argument("--manifest-path", required=True, help="Path to output manifest json")
    ap.add_argument("--seed", type=int, required=True, help="Random seed for player split assignment")
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--eval-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))))
    args = ap.parse_args()

    input_files = iter_partition_files(args.input_dir)
    if not input_files:
        raise FileNotFoundError(f"No .jsonl files found in: {args.input_dir}")

    print(f"[main] found {len(input_files)} input partitions", file=sys.stderr)
    print(f"[main] using {args.workers} worker processes", file=sys.stderr)

    # -------------------------
    # Pass 1: collect all players
    # -------------------------
    collect_results: List[Dict[str, object]] = []
    all_players: Set[str] = set()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
    ) as ex:
        futures = [ex.submit(collect_players_from_partition, path) for path in input_files]

        for fut in as_completed(futures):
            res = fut.result()
            collect_results.append(res)
            players = set(res["players"])
            all_players.update(players)
            print(
                f"[collect] {res['partition']}: rows={res['rows']:,} "
                f"players={len(players):,} bad_rows={res['bad_rows']:,}",
                file=sys.stderr,
            )

    all_players_sorted = sorted(all_players)
    print(f"[main] unique players={len(all_players_sorted):,}", file=sys.stderr)

    # -------------------------
    # Split players
    # -------------------------
    train_players, eval_players, test_players = split_players(
        all_players=all_players_sorted,
        seed=args.seed,
        train_frac=args.train_frac,
        eval_frac=args.eval_frac,
        test_frac=args.test_frac,
    )

    print(
        f"[main] player split sizes -> "
        f"train={len(train_players):,} eval={len(eval_players):,} test={len(test_players):,}",
        file=sys.stderr,
    )

    # -------------------------
    # Pass 2: write train/eval/test rows
    # -------------------------
    split_results: List[Dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
    ) as ex:
        futures = [
            ex.submit(
                split_partition_rows,
                input_path,
                args.train_dir,
                args.eval_dir,
                args.test_dir,
                train_players,
                eval_players,
                test_players,
            )
            for input_path in input_files
        ]

        for fut in as_completed(futures):
            res = fut.result()
            split_results.append(res)
            print(
                f"[split] {res['partition']}: total={res['total_rows']:,} "
                f"train={res['train_rows']:,} eval={res['eval_rows']:,} test={res['test_rows']:,} "
                f"dropped={res['dropped_rows']:,} bad_rows={res['bad_rows']:,}",
                file=sys.stderr,
            )

    # -------------------------
    # Manifest
    # -------------------------
    split_results_sorted = sorted(split_results, key=lambda x: x["partition"])
    manifest = {
        "input_dir": args.input_dir,
        "train_dir": args.train_dir,
        "eval_dir": args.eval_dir,
        "test_dir": args.test_dir,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "eval_frac": args.eval_frac,
        "test_frac": args.test_frac,
        "num_unique_players": len(all_players_sorted),
        "num_train_players": len(train_players),
        "num_eval_players": len(eval_players),
        "num_test_players": len(test_players),
        "train_players": sorted(train_players),
        "eval_players": sorted(eval_players),
        "test_players": sorted(test_players),
        "partitions": split_results_sorted,
        "totals": {
            "total_rows": sum(x["total_rows"] for x in split_results_sorted),
            "train_rows": sum(x["train_rows"] for x in split_results_sorted),
            "eval_rows": sum(x["eval_rows"] for x in split_results_sorted),
            "test_rows": sum(x["test_rows"] for x in split_results_sorted),
            "dropped_rows": sum(x["dropped_rows"] for x in split_results_sorted),
            "bad_rows": sum(x["bad_rows"] for x in split_results_sorted),
        },
    }

    manifest_dir = os.path.dirname(args.manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[main] manifest -> {args.manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    # python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/split_player_conditioned_jsonl.py --input-dir ./final_experiments_for_paper/experiment2_style_model/flattened_style_jsonl --train-dir ./final_experiments_for_paper/experiment2_style_model/splits/train --eval-dir ./final_experiments_for_paper/experiment2_style_model/splits/eval --test-dir ./final_experiments_for_paper/experiment2_style_model/splits/test --manifest-path ./final_experiments_for_paper/experiment2_style_model/splits/manifest.json --seed 42 --workers 16
    main()
