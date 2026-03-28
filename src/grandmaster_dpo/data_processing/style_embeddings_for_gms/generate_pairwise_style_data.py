from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from array import array
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Tuple


PHASE_TO_CODE = {"opening": 0, "middlegame": 1, "endgame": 2}
CODE_TO_PHASE = {v: k for k, v in PHASE_TO_CODE.items()}


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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def slim_example(row: dict) -> dict:
    return {
        "example_id": row["example_id"],
        "player_id": row["player_id"],
        "opponent_id": row.get("opponent_id"),
        "game_id": row.get("game_id"),
        "ply_idx": row.get("ply_idx"),
        "move_color": row.get("move_color"),
        "game_type": row.get("game_type"),
        "opening_bucket": row.get("opening_bucket"),
        "phase": row["phase"],
        "board_t_minus_5": row.get("board_t_minus_5"),
        "board_t_minus_4": row.get("board_t_minus_4"),
        "board_t_minus_3": row.get("board_t_minus_3"),
        "board_t_minus_2": row.get("board_t_minus_2"),
        "board_t_minus_1": row.get("board_t_minus_1"),
        "board_t": row.get("board_t"),
        "move_played": row.get("move_played"),
    }


def get_or_add_code(value: str, mapping: Dict[str, int]) -> int:
    code = mapping.get(value)
    if code is None:
        code = len(mapping)
        mapping[value] = code
    return code


def row_id_to_int(row_id: int) -> int:
    return int(row_id)


class InMemoryRowStore:
    """
    Stores only slim rows in RAM (local machine 128 GB ram)
    """
    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: List[dict] = []

    def append(self, row: dict) -> int:
        row_id = len(self.rows)
        self.rows.append(slim_example(row))
        return row_id

    def get(self, row_id: int) -> dict:
        return self.rows[row_id]


def build_in_memory_index(split_dir: str) -> Dict[str, object]:
    """
    Build all metadata and slim rows fully in memory.

    We intentionally use:
      - arrays for dense numeric metadata
      - defaultdict(array('I')) for match buckets
      - list[dict] only for slim payload rows
    """
    player_code_map: Dict[str, int] = {}
    game_type_code_map: Dict[str, int] = {}

    row_store = InMemoryRowStore()
    row_id_by_example_id: Dict[str, int] = {}

    player_codes = array("I")
    phase_codes = array("B")
    game_type_codes = array("B")

    by_player_phase_game_type: DefaultDict[Tuple[int, int, int], array] = defaultdict(lambda: array("I"))
    by_game_type: DefaultDict[int, array] = defaultdict(lambda: array("I"))

    game_type_counts: DefaultDict[int, int] = defaultdict(int)
    game_type_player_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)

    stats = {
        "rows": 0,
        "bad_rows": 0,
        "partitions": 0,
        "duplicate_example_ids": 0,
    }

    for path in iter_partition_files(split_dir):
        stats["partitions"] += 1
        for row in read_jsonl(path):
            try:
                example_id = str(row["example_id"])
                if example_id in row_id_by_example_id:
                    stats["duplicate_example_ids"] += 1
                    continue

                player_id = str(row["player_id"])
                phase = str(row["phase"])
                game_type = str(row["game_type"])

                phase_code = PHASE_TO_CODE.get(phase)
                if phase_code is None:
                    stats["bad_rows"] += 1
                    continue

                player_code = get_or_add_code(player_id, player_code_map)
                gt_code = get_or_add_code(game_type, game_type_code_map)

                row_id = row_store.append(row)
                row_id_by_example_id[example_id] = row_id

                player_codes.append(player_code)
                phase_codes.append(phase_code)
                game_type_codes.append(gt_code)

                by_player_phase_game_type[(player_code, phase_code, gt_code)].append(row_id)
                by_game_type[gt_code].append(row_id)

                game_type_counts[gt_code] += 1
                game_type_player_counts[(gt_code, player_code)] += 1
                stats["rows"] += 1
            except Exception:
                stats["bad_rows"] += 1

    return {
        "row_store": row_store,
        "row_id_by_example_id": row_id_by_example_id,
        "player_codes": player_codes,
        "phase_codes": phase_codes,
        "game_type_codes": game_type_codes,
        "by_player_phase_game_type": by_player_phase_game_type,
        "by_game_type": by_game_type,
        "game_type_counts": dict(game_type_counts),
        "game_type_player_counts": dict(game_type_player_counts),
        "player_code_map": player_code_map,
        "game_type_code_map": game_type_code_map,
        "stats": stats,
    }


def sample_without_self(bucket: array, self_row_id: int, max_count: int, rng: random.Random) -> List[int]:
    n = len(bucket)
    if n <= 1 or max_count == 0:
        return []

    eligible = n - 1
    if eligible <= 0:
        return []

    if max_count < 0 or eligible <= max_count:
        out = [row_id_to_int(rid) for rid in bucket if rid != self_row_id]
        rng.shuffle(out)
        return out

    target = min(max_count, eligible)
    seen: set[int] = set()
    out: List[int] = []
    while len(out) < target:
        rid = row_id_to_int(bucket[rng.randrange(n)])
        if rid == self_row_id or rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
    return out


def is_diff_gm(
    candidate_row_id: int,
    anchor_row_id: int,
    anchor_player_code: int,
    anchor_game_type_code: int,
    player_codes: array,
    game_type_codes: array,
) -> bool:
    return (
        candidate_row_id != anchor_row_id
        and game_type_codes[candidate_row_id] == anchor_game_type_code
        and player_codes[candidate_row_id] != anchor_player_code
    )


def sample_diff_gm_exact(
    anchor_row_id: int,
    anchor_player_code: int,
    anchor_game_type_code: int,
    by_game_type: Dict[int, array],
    game_type_counts: Dict[int, int],
    game_type_player_counts: Dict[Tuple[int, int], int],
    player_codes: array,
    game_type_codes: array,
    max_count: int,
    rng: random.Random,
) -> Tuple[List[int], int]:
    total = max(
        0,
        game_type_counts.get(anchor_game_type_code, 0)
        - game_type_player_counts.get((anchor_game_type_code, anchor_player_code), 0)
    )
    if total == 0 or max_count == 0:
        return [], total

    pool = by_game_type.get(anchor_game_type_code)
    if not pool:
        return [], total

    target = total if max_count < 0 else min(max_count, total)
    out: List[int] = []
    seen: set[int] = set()

    # Fast path: rejection sample out of same-game-type pool.
    # Usually different-GM dominates, so this is cheap.
    max_tries = max(256, target * 50)
    tries = 0
    pool_len = len(pool)

    while len(out) < target and tries < max_tries:
        tries += 1
        rid = row_id_to_int(pool[rng.randrange(pool_len)])
        if rid in seen:
            continue
        if is_diff_gm(
            candidate_row_id=rid,
            anchor_row_id=anchor_row_id,
            anchor_player_code=anchor_player_code,
            anchor_game_type_code=anchor_game_type_code,
            player_codes=player_codes,
            game_type_codes=game_type_codes,
        ):
            seen.add(rid)
            out.append(rid)

    if len(out) < target:
        for rid_raw in pool:
            rid = row_id_to_int(rid_raw)
            if rid in seen:
                continue
            if is_diff_gm(
                candidate_row_id=rid,
                anchor_row_id=anchor_row_id,
                anchor_player_code=anchor_player_code,
                anchor_game_type_code=anchor_game_type_code,
                player_codes=player_codes,
                game_type_codes=game_type_codes,
            ):
                seen.add(rid)
                out.append(rid)
                if len(out) >= target:
                    break

    return out, total


def sample_same_player_diff_phase(
    anchor_row_id: int,
    anchor_player_code: int,
    anchor_phase_code: int,
    anchor_game_type_code: int,
    by_player_phase_game_type: Dict[Tuple[int, int, int], array],
    max_count: int,
    rng: random.Random,
) -> Tuple[List[int], int]:
    merged: List[int] = []
    total = 0

    for phase_code in (0, 1, 2):
        if phase_code == anchor_phase_code:
            continue
        bucket = by_player_phase_game_type.get((anchor_player_code, phase_code, anchor_game_type_code))
        if not bucket:
            continue
        total += len(bucket)
        merged.extend(row_id_to_int(rid) for rid in bucket if rid != anchor_row_id)

    if not merged or max_count == 0:
        return [], total

    if max_count < 0 or len(merged) <= max_count:
        rng.shuffle(merged)
        return merged, total

    return rng.sample(merged, max_count), total


def generate_negatives_v1(
    anchor_row_id: int,
    anchor_player_code: int,
    anchor_phase_code: int,
    anchor_game_type_code: int,
    by_player_phase_game_type: Dict[Tuple[int, int, int], array],
    by_game_type: Dict[int, array],
    game_type_counts: Dict[int, int],
    game_type_player_counts: Dict[Tuple[int, int], int],
    player_codes: array,
    game_type_codes: array,
    max_negatives_per_anchor: int,
    rng: random.Random,
) -> Tuple[List[int], int]:
    """
    Variant 1:
      negatives = same game_type AND (
          different GM, any phase
          OR same GM, different phase
      )
    """
    same_phase_shift_ids, same_phase_shift_total = sample_same_player_diff_phase(
        anchor_row_id=anchor_row_id,
        anchor_player_code=anchor_player_code,
        anchor_phase_code=anchor_phase_code,
        anchor_game_type_code=anchor_game_type_code,
        by_player_phase_game_type=by_player_phase_game_type,
        max_count=max_negatives_per_anchor if max_negatives_per_anchor >= 0 else -1,
        rng=rng,
    )

    diff_ids, diff_total = sample_diff_gm_exact(
        anchor_row_id=anchor_row_id,
        anchor_player_code=anchor_player_code,
        anchor_game_type_code=anchor_game_type_code,
        by_game_type=by_game_type,
        game_type_counts=game_type_counts,
        game_type_player_counts=game_type_player_counts,
        player_codes=player_codes,
        game_type_codes=game_type_codes,
        max_count=max_negatives_per_anchor if max_negatives_per_anchor >= 0 else -1,
        rng=rng,
    )

    # True eligible union count:
    neg_total = same_phase_shift_total + diff_total

    # Sample from union.
    union_ids = same_phase_shift_ids + diff_ids
    if not union_ids:
        return [], neg_total

    if max_negatives_per_anchor < 0 or len(union_ids) <= max_negatives_per_anchor:
        rng.shuffle(union_ids)
        return union_ids, neg_total

    return rng.sample(union_ids, max_negatives_per_anchor), neg_total


def generate_negatives_v2(
    anchor_row_id: int,
    anchor_player_code: int,
    anchor_game_type_code: int,
    by_game_type: Dict[int, array],
    game_type_counts: Dict[int, int],
    game_type_player_counts: Dict[Tuple[int, int], int],
    player_codes: array,
    game_type_codes: array,
    max_negatives_per_anchor: int,
    rng: random.Random,
) -> Tuple[List[int], int]:
    """
    Variant 2:
      negatives = same game_type AND different GM
      phase may be same or different
    """
    return sample_diff_gm_exact(
        anchor_row_id=anchor_row_id,
        anchor_player_code=anchor_player_code,
        anchor_game_type_code=anchor_game_type_code,
        by_game_type=by_game_type,
        game_type_counts=game_type_counts,
        game_type_player_counts=game_type_player_counts,
        player_codes=player_codes,
        game_type_codes=game_type_codes,
        max_count=max_negatives_per_anchor,
        rng=rng,
    )


def emit_pairs_for_partition(
    input_path: str,
    output_dir: str,
    row_store: InMemoryRowStore,
    row_id_by_example_id: Dict[str, int],
    player_codes: array,
    phase_codes: array,
    game_type_codes: array,
    by_player_phase_game_type: Dict[Tuple[int, int, int], array],
    by_game_type: Dict[int, array],
    game_type_counts: Dict[int, int],
    game_type_player_counts: Dict[Tuple[int, int], int],
    variant: str,
    seed: int,
    max_positives_per_anchor: int,
    max_negatives_per_anchor: int,
    require_positive: bool,
    require_negative: bool,
    flush_every: int = 20000,
) -> Dict[str, object]:
    partition_filename = os.path.basename(input_path)
    partition_stem, _ = os.path.splitext(partition_filename)
    output_path = os.path.join(output_dir, f"{partition_stem}.jsonl")
    ensure_dir(output_dir)

    total_rows = 0
    emitted_rows = 0
    skipped_no_positive = 0
    skipped_no_negative = 0
    bad_rows = 0

    rng = random.Random(seed + stable_int_hash(partition_stem))

    with open(output_path, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as out_f:
        write_buffer: List[str] = []

        for row_num, row in enumerate(read_jsonl(input_path), start=1):
            total_rows += 1
            try:
                example_id = str(row["example_id"])
                anchor_row_id = row_id_by_example_id.get(example_id)
                if anchor_row_id is None:
                    bad_rows += 1
                    continue

                anchor_player_code = player_codes[anchor_row_id]
                anchor_phase_code = phase_codes[anchor_row_id]
                anchor_game_type_code = game_type_codes[anchor_row_id]

                pos_bucket = by_player_phase_game_type.get(
                    (anchor_player_code, anchor_phase_code, anchor_game_type_code)
                )
                if pos_bucket is None:
                    pos_ids = []
                    pos_total = 0
                else:
                    pos_ids = sample_without_self(
                        bucket=pos_bucket,
                        self_row_id=anchor_row_id,
                        max_count=max_positives_per_anchor,
                        rng=rng,
                    )
                    pos_total = max(0, len(pos_bucket) - 1)

                if variant == "v1":
                    neg_ids, neg_total = generate_negatives_v1(
                        anchor_row_id=anchor_row_id,
                        anchor_player_code=anchor_player_code,
                        anchor_phase_code=anchor_phase_code,
                        anchor_game_type_code=anchor_game_type_code,
                        by_player_phase_game_type=by_player_phase_game_type,
                        by_game_type=by_game_type,
                        game_type_counts=game_type_counts,
                        game_type_player_counts=game_type_player_counts,
                        player_codes=player_codes,
                        game_type_codes=game_type_codes,
                        max_negatives_per_anchor=max_negatives_per_anchor,
                        rng=rng,
                    )
                elif variant == "v2":
                    neg_ids, neg_total = generate_negatives_v2(
                        anchor_row_id=anchor_row_id,
                        anchor_player_code=anchor_player_code,
                        anchor_game_type_code=anchor_game_type_code,
                        by_game_type=by_game_type,
                        game_type_counts=game_type_counts,
                        game_type_player_counts=game_type_player_counts,
                        player_codes=player_codes,
                        game_type_codes=game_type_codes,
                        max_negatives_per_anchor=max_negatives_per_anchor,
                        rng=rng,
                    )
                else:
                    raise ValueError(f"Unsupported variant: {variant}")

                if require_positive and not pos_ids:
                    skipped_no_positive += 1
                    continue

                if require_negative and not neg_ids:
                    skipped_no_negative += 1
                    continue

                positives = [row_store.get(rid) for rid in pos_ids]
                negatives = [row_store.get(rid) for rid in neg_ids]

                pair_row = {
                    "anchor": row_store.get(anchor_row_id),
                    "positives": positives,
                    "negatives": negatives,
                    "meta": {
                        "variant": variant,
                        "matching_rules": {
                            "positive": "same player_id, same phase, same game_type",
                            "negative_v1": "same game_type and (different player_id any phase OR same player_id different phase)",
                            "negative_v2": "same game_type and different player_id any phase",
                        },
                        "num_positive_candidates_total": pos_total,
                        "num_negative_candidates_total": neg_total,
                        "num_positives_sampled": len(positives),
                        "num_negatives_sampled": len(negatives),
                    },
                }

                write_buffer.append(json.dumps(pair_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                emitted_rows += 1

                if len(write_buffer) >= flush_every:
                    out_f.writelines(write_buffer)
                    out_f.flush()
                    write_buffer.clear()
                    print(
                        f"[{partition_stem}] rows={total_rows:,} emitted={emitted_rows:,} "
                        f"skipped_no_positive={skipped_no_positive:,} "
                        f"skipped_no_negative={skipped_no_negative:,} "
                        f"bad_rows={bad_rows:,}",
                        file=sys.stderr,
                    )

            except Exception as e:
                bad_rows += 1
                print(f"[{partition_stem}] ERROR row {row_num}: {e}", file=sys.stderr)

        if write_buffer:
            out_f.writelines(write_buffer)
            out_f.flush()

    return {
        "partition": partition_stem,
        "input_path": input_path,
        "output_path": output_path,
        "rows": total_rows,
        "emitted_rows": emitted_rows,
        "skipped_no_positive": skipped_no_positive,
        "skipped_no_negative": skipped_no_negative,
        "bad_rows": bad_rows,
    }


def process_split(
    split_name: str,
    input_dir: str,
    output_dir: str,
    variant: str,
    seed: int,
    max_positives_per_anchor: int,
    max_negatives_per_anchor: int,
    require_positive: bool,
    require_negative: bool,
) -> Dict[str, object]:
    input_files = iter_partition_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No .jsonl files found in: {input_dir}")

    ensure_dir(output_dir)

    print(f"[{split_name}] building full in-memory index from {input_dir}", file=sys.stderr)
    bundle = build_in_memory_index(input_dir)

    row_store: InMemoryRowStore = bundle["row_store"]
    row_id_by_example_id = bundle["row_id_by_example_id"]
    player_codes = bundle["player_codes"]
    phase_codes = bundle["phase_codes"]
    game_type_codes = bundle["game_type_codes"]
    by_player_phase_game_type = bundle["by_player_phase_game_type"]
    by_game_type = bundle["by_game_type"]
    game_type_counts = bundle["game_type_counts"]
    game_type_player_counts = bundle["game_type_player_counts"]
    load_stats = bundle["stats"]

    print(
        f"[{split_name}] loaded rows={load_stats['rows']:,} "
        f"bad_rows={load_stats['bad_rows']:,} "
        f"duplicate_example_ids={load_stats['duplicate_example_ids']:,} "
        f"partitions={load_stats['partitions']:,}",
        file=sys.stderr,
    )
    print(
        f"[{split_name}] unique_examples={len(row_id_by_example_id):,} "
        f"player_phase_game_type_buckets={len(by_player_phase_game_type):,} "
        f"game_type_buckets={len(by_game_type):,}",
        file=sys.stderr,
    )

    results: List[Dict[str, object]] = []
    for input_path in input_files:
        print(f"[{split_name}] processing partition {os.path.basename(input_path)}", file=sys.stderr)
        res = emit_pairs_for_partition(
            input_path=input_path,
            output_dir=output_dir,
            row_store=row_store,
            row_id_by_example_id=row_id_by_example_id,
            player_codes=player_codes,
            phase_codes=phase_codes,
            game_type_codes=game_type_codes,
            by_player_phase_game_type=by_player_phase_game_type,
            by_game_type=by_game_type,
            game_type_counts=game_type_counts,
            game_type_player_counts=game_type_player_counts,
            variant=variant,
            seed=seed,
            max_positives_per_anchor=max_positives_per_anchor,
            max_negatives_per_anchor=max_negatives_per_anchor,
            require_positive=require_positive,
            require_negative=require_negative,
        )
        results.append(res)
        print(
            f"[{split_name}][done] {res['partition']}: "
            f"rows={res['rows']:,} emitted={res['emitted_rows']:,} "
            f"skipped_no_positive={res['skipped_no_positive']:,} "
            f"skipped_no_negative={res['skipped_no_negative']:,} "
            f"bad_rows={res['bad_rows']:,} -> {res['output_path']}",
            file=sys.stderr,
        )

    manifest = {
        "split_name": split_name,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "variant": variant,
        "seed": seed,
        "max_positives_per_anchor": max_positives_per_anchor,
        "max_negatives_per_anchor": max_negatives_per_anchor,
        "require_positive": require_positive,
        "require_negative": require_negative,
        "num_rows_loaded": load_stats["rows"],
        "num_unique_examples": len(row_id_by_example_id),
        "num_player_phase_game_type_buckets": len(by_player_phase_game_type),
        "num_game_type_buckets": len(by_game_type),
        "partitions": sorted(results, key=lambda x: x["partition"]),
        "totals": {
            "rows": sum(int(x["rows"]) for x in results),
            "emitted_rows": sum(int(x["emitted_rows"]) for x in results),
            "skipped_no_positive": sum(int(x["skipped_no_positive"]) for x in results),
            "skipped_no_negative": sum(int(x["skipped_no_negative"]) for x in results),
            "bad_rows": sum(int(x["bad_rows"]) for x in results),
        },
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[{split_name}] manifest -> {manifest_path}", file=sys.stderr)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--test-dir", required=True)

    ap.add_argument("--train-out-dir", required=True)
    ap.add_argument("--eval-out-dir", required=True)
    ap.add_argument("--test-out-dir", required=True)

    ap.add_argument("--variant", choices=["v1", "v2"], required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-positives-per-anchor", type=int, default=8)
    ap.add_argument("--max-negatives-per-anchor", type=int, default=32)

    ap.add_argument("--require-positive", action="store_true")
    ap.add_argument("--require-negative", action="store_true")
    args = ap.parse_args()

    process_split(
        split_name="train",
        input_dir=args.train_dir,
        output_dir=args.train_out_dir,
        variant=args.variant,
        seed=args.seed,
        max_positives_per_anchor=args.max_positives_per_anchor,
        max_negatives_per_anchor=args.max_negatives_per_anchor,
        require_positive=args.require_positive,
        require_negative=args.require_negative,
    )

    process_split(
        split_name="eval",
        input_dir=args.eval_dir,
        output_dir=args.eval_out_dir,
        variant=args.variant,
        seed=args.seed + 1,
        max_positives_per_anchor=args.max_positives_per_anchor,
        max_negatives_per_anchor=args.max_negatives_per_anchor,
        require_positive=args.require_positive,
        require_negative=args.require_negative,
    )

    process_split(
        split_name="test",
        input_dir=args.test_dir,
        output_dir=args.test_out_dir,
        variant=args.variant,
        seed=args.seed + 2,
        max_positives_per_anchor=args.max_positives_per_anchor,
        max_negatives_per_anchor=args.max_negatives_per_anchor,
        require_positive=args.require_positive,
        require_negative=args.require_negative,
    )


if __name__ == "__main__":
    main()