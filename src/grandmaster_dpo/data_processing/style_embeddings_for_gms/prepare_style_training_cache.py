#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import orjson

from grandmaster_dpo.utilities.shared_style_emb_model_utils import (
    raw_example_to_cached_arrays,
)


def stable_example_hash64(ex: Dict[str, Any]) -> int:
    """
    Stable 64-bit key for shard-local dedupe.
    We prefer example_id if present, but hash it into uint64 to avoid large Python strings as dict keys.
    """
    ex_id = ex.get("example_id")
    if ex_id:
        raw = str(ex_id).encode("utf-8")
    else:
        payload = b"|".join(
            [
                str(ex.get("player_id", "")).encode("utf-8"),
                str(ex.get("opponent_id", "")).encode("utf-8"),
                str(ex.get("game_id", "")).encode("utf-8"),
                str(ex.get("ply_idx", -1)).encode("utf-8"),
                str(ex.get("move_played", "")).encode("utf-8"),
                str(ex.get("board_t", "")).encode("utf-8"),
            ]
        )
        raw = payload

    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def iter_jsonl_rows(input_dir: Path, max_rows: Optional[int]) -> Iterable[Dict[str, Any]]:
    seen = 0
    for path in sorted(input_dir.glob("*.jsonl")):
        print(f"[read] file={path}")
        with path.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                yield orjson.loads(line)
                seen += 1
                if seen % 10_000 == 0:
                    print(f"[read] rows={seen:,}")
                if max_rows is not None and seen >= max_rows:
                    return


@dataclass
class ShardBuilder:
    shard_idx: int

    example_hash_to_local_idx: Dict[int, int] = field(default_factory=dict)

    example_boards: List[np.ndarray] = field(default_factory=list)
    example_moves: List[np.ndarray] = field(default_factory=list)
    example_game_types: List[np.uint8] = field(default_factory=list)

    pair_anchor_idx: List[int] = field(default_factory=list)

    pos_flat: List[int] = field(default_factory=list)
    pos_offsets: List[int] = field(default_factory=lambda: [0])

    neg_flat: List[int] = field(default_factory=list)
    neg_offsets: List[int] = field(default_factory=lambda: [0])

    rows_read_in_shard: int = 0
    rows_kept_in_shard: int = 0
    total_pos_candidates: int = 0
    total_neg_candidates: int = 0
    skipped_no_pos: int = 0
    skipped_no_neg: int = 0

    def get_or_add_example(self, ex: Dict[str, Any]) -> int:
        h = stable_example_hash64(ex)
        maybe_idx = self.example_hash_to_local_idx.get(h)
        if maybe_idx is not None:
            return maybe_idx

        boards, move, game_type = raw_example_to_cached_arrays(ex)
        idx = len(self.example_boards)

        self.example_hash_to_local_idx[h] = idx
        self.example_boards.append(boards)
        self.example_moves.append(move)
        self.example_game_types.append(game_type)
        return idx

    def add_pair_row(self, row: Dict[str, Any]) -> None:
        self.rows_read_in_shard += 1

        anchor_idx = self.get_or_add_example(row["anchor"])
        pos_indices = [self.get_or_add_example(x) for x in row.get("positives", [])]
        neg_indices = [self.get_or_add_example(x) for x in row.get("negatives", [])]

        if not pos_indices:
            self.skipped_no_pos += 1
            return
        if not neg_indices:
            self.skipped_no_neg += 1
            return

        self.pair_anchor_idx.append(anchor_idx)

        self.pos_flat.extend(pos_indices)
        self.pos_offsets.append(len(self.pos_flat))

        self.neg_flat.extend(neg_indices)
        self.neg_offsets.append(len(self.neg_flat))

        self.rows_kept_in_shard += 1
        self.total_pos_candidates += len(pos_indices)
        self.total_neg_candidates += len(neg_indices)

    def num_kept_pairs(self) -> int:
        return self.rows_kept_in_shard

    def num_unique_examples(self) -> int:
        return len(self.example_boards)

    def is_empty(self) -> bool:
        return self.rows_kept_in_shard == 0

    def clear(self) -> None:
        self.example_hash_to_local_idx.clear()
        self.example_boards.clear()
        self.example_moves.clear()
        self.example_game_types.clear()
        self.pair_anchor_idx.clear()
        self.pos_flat.clear()
        self.pos_offsets = [0]
        self.neg_flat.clear()
        self.neg_offsets = [0]

        self.rows_read_in_shard = 0
        self.rows_kept_in_shard = 0
        self.total_pos_candidates = 0
        self.total_neg_candidates = 0
        self.skipped_no_pos = 0
        self.skipped_no_neg = 0


def save_shard(builder: ShardBuilder, split_out_dir: Path) -> Dict[str, Any]:
    shard_dir = split_out_dir / f"shard_{builder.shard_idx:06d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if builder.is_empty():
        meta = {
            "shard_idx": builder.shard_idx,
            "rows_read_in_shard": builder.rows_read_in_shard,
            "rows_kept_in_shard": 0,
            "num_unique_examples": 0,
            "skipped_no_pos": builder.skipped_no_pos,
            "skipped_no_neg": builder.skipped_no_neg,
            "empty_shard": True,
        }
        with (shard_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta

    boards = np.stack(builder.example_boards, axis=0).astype(np.uint8, copy=False)       # [N, 5, 64]
    moves = np.stack(builder.example_moves, axis=0).astype(np.uint8, copy=False)          # [N, 3]
    game_types = np.asarray(builder.example_game_types, dtype=np.uint8)                    # [N]

    pair_anchor_idx = np.asarray(builder.pair_anchor_idx, dtype=np.int32)                  # [M]
    pos_flat = np.asarray(builder.pos_flat, dtype=np.int32)
    pos_offsets = np.asarray(builder.pos_offsets, dtype=np.int64)
    neg_flat = np.asarray(builder.neg_flat, dtype=np.int32)
    neg_offsets = np.asarray(builder.neg_offsets, dtype=np.int64)

    np.save(shard_dir / "examples_board_tokens.uint8.npy", boards, allow_pickle=False)
    np.save(shard_dir / "examples_moves.uint8.npy", moves, allow_pickle=False)
    np.save(shard_dir / "examples_game_type.uint8.npy", game_types, allow_pickle=False)

    np.save(shard_dir / "pair_anchor_idx.int32.npy", pair_anchor_idx, allow_pickle=False)
    np.save(shard_dir / "pair_pos_flat.int32.npy", pos_flat, allow_pickle=False)
    np.save(shard_dir / "pair_pos_offsets.int64.npy", pos_offsets, allow_pickle=False)
    np.save(shard_dir / "pair_neg_flat.int32.npy", neg_flat, allow_pickle=False)
    np.save(shard_dir / "pair_neg_offsets.int64.npy", neg_offsets, allow_pickle=False)

    meta = {
        "shard_idx": builder.shard_idx,
        "rows_read_in_shard": builder.rows_read_in_shard,
        "rows_kept_in_shard": int(builder.rows_kept_in_shard),
        "num_unique_examples": int(boards.shape[0]),
        "boards_shape": list(boards.shape),
        "moves_shape": list(moves.shape),
        "game_types_shape": list(game_types.shape),
        "pair_anchor_idx_shape": list(pair_anchor_idx.shape),
        "pair_pos_flat_shape": list(pos_flat.shape),
        "pair_pos_offsets_shape": list(pos_offsets.shape),
        "pair_neg_flat_shape": list(neg_flat.shape),
        "pair_neg_offsets_shape": list(neg_offsets.shape),
        "avg_pos_candidates_per_pair": float(builder.total_pos_candidates / max(1, builder.rows_kept_in_shard)),
        "avg_neg_candidates_per_pair": float(builder.total_neg_candidates / max(1, builder.rows_kept_in_shard)),
        "skipped_no_pos": builder.skipped_no_pos,
        "skipped_no_neg": builder.skipped_no_neg,
        "dtype_notes": {
            "boards": "uint8 piece ids, [N, 5, 64]",
            "moves": "uint8 move encoding [from, to, promo]",
            "game_types": "uint8 ids",
            "pair_anchor_idx": "int32 local example index",
            "pair_pos_flat": "int32 local example indices",
            "pair_pos_offsets": "int64 offsets into pair_pos_flat",
            "pair_neg_flat": "int32 local example indices",
            "pair_neg_offsets": "int64 offsets into pair_neg_flat",
        },
    }

    with (shard_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def build_split_sharded(
    input_dir: Path,
    output_dir: Path,
    rows_per_shard: int,
    max_rows: Optional[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    builder = ShardBuilder(shard_idx=shard_idx)

    split_rows_read = 0
    split_rows_kept = 0
    split_examples = 0
    split_pos = 0
    split_neg = 0
    split_skipped_no_pos = 0
    split_skipped_no_neg = 0
    shard_count = 0

    for row in iter_jsonl_rows(input_dir, max_rows=max_rows):
        split_rows_read += 1
        builder.add_pair_row(row)

        if split_rows_read % 10_000 == 0:
            print(
                f"[build] split={input_dir.name} rows={split_rows_read:,} "
                f"current_shard={builder.shard_idx} "
                f"kept_in_shard={builder.num_kept_pairs():,} "
                f"unique_examples_in_shard={builder.num_unique_examples():,}"
            )

        if builder.rows_read_in_shard >= rows_per_shard:
            meta = save_shard(builder, output_dir)
            shard_count += 1

            split_rows_kept += meta.get("rows_kept_in_shard", 0)
            split_examples += meta.get("num_unique_examples", 0)
            split_pos += int(meta.get("pair_pos_flat_shape", [0])[0]) if "pair_pos_flat_shape" in meta else 0
            split_neg += int(meta.get("pair_neg_flat_shape", [0])[0]) if "pair_neg_flat_shape" in meta else 0
            split_skipped_no_pos += meta.get("skipped_no_pos", 0)
            split_skipped_no_neg += meta.get("skipped_no_neg", 0)

            print(
                f"[flush] split={input_dir.name} shard={builder.shard_idx:06d} "
                f"kept={meta.get('rows_kept_in_shard', 0):,} "
                f"unique_examples={meta.get('num_unique_examples', 0):,}"
            )

            builder.clear()
            gc.collect()

            shard_idx += 1
            builder = ShardBuilder(shard_idx=shard_idx)

    if builder.rows_read_in_shard > 0:
        meta = save_shard(builder, output_dir)
        shard_count += 1

        split_rows_kept += meta.get("rows_kept_in_shard", 0)
        split_examples += meta.get("num_unique_examples", 0)
        split_pos += int(meta.get("pair_pos_flat_shape", [0])[0]) if "pair_pos_flat_shape" in meta else 0
        split_neg += int(meta.get("pair_neg_flat_shape", [0])[0]) if "pair_neg_flat_shape" in meta else 0
        split_skipped_no_pos += meta.get("skipped_no_pos", 0)
        split_skipped_no_neg += meta.get("skipped_no_neg", 0)

        print(
            f"[flush] split={input_dir.name} shard={builder.shard_idx:06d} "
            f"kept={meta.get('rows_kept_in_shard', 0):,} "
            f"unique_examples={meta.get('num_unique_examples', 0):,}"
        )

    split_meta = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rows_per_shard": rows_per_shard,
        "max_rows": max_rows,
        "num_shards": shard_count,
        "num_pair_rows_read": split_rows_read,
        "num_pair_rows_kept": split_rows_kept,
        "sum_unique_examples_across_shards": split_examples,
        "sum_pos_candidates_across_shards": split_pos,
        "sum_neg_candidates_across_shards": split_neg,
        "skipped_no_pos": split_skipped_no_pos,
        "skipped_no_neg": split_skipped_no_neg,
        "note": "Examples are deduped within shard only, not globally across the whole split.",
    }

    with (output_dir / "_split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print(json.dumps(split_meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, required=True)
    ap.add_argument("--eval-dir", type=str, required=True)
    ap.add_argument("--test-dir", type=str, default=None)
    ap.add_argument("--out-root", type=str, required=True)

    ap.add_argument("--rows-per-shard", type=int, default=100_000)

    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-eval-rows", type=int, default=None)
    ap.add_argument("--max-test-rows", type=int, default=None)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=== BUILD TRAIN SPLIT ===")
    build_split_sharded(
        input_dir=Path(args.train_dir),
        output_dir=out_root / "train",
        rows_per_shard=args.rows_per_shard,
        max_rows=args.max_train_rows,
    )

    print("\n=== BUILD EVAL SPLIT ===")
    build_split_sharded(
        input_dir=Path(args.eval_dir),
        output_dir=out_root / "eval",
        rows_per_shard=args.rows_per_shard,
        max_rows=args.max_eval_rows,
    )

    if args.test_dir:
        print("\n=== BUILD TEST SPLIT ===")
        build_split_sharded(
            input_dir=Path(args.test_dir),
            output_dir=out_root / "test",
            rows_per_shard=args.rows_per_shard,
            max_rows=args.max_test_rows,
        )


if __name__ == "__main__":
    main()
