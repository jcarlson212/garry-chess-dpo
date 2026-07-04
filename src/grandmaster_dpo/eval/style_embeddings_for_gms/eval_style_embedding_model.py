from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import orjson
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Adjust these imports to match your repo layout.
from grandmaster_dpo.train.style_embeddings_for_gms.dataset_schema import TrainConfig
from grandmaster_dpo.utilities.jsonl_io import open_jsonl_binary, sorted_jsonl_paths
from grandmaster_dpo.train.style_embeddings_for_gms.train_style_encoder import StyleEncoder
from grandmaster_dpo.utilities.shared_style_emb_model_utils import (
    PHASE_TO_ID,
    assert_model_dir_matches_variant,
    model_variant_uses_game_type,
    model_variant_uses_opponent_context,
    move_feature_dict_to_device,
    pick_device,
    raw_example_to_cached_arrays,
    resolve_checkpoint,
    set_seed,
    stack_feature_dicts,
)


DEFAULT_EVAL_TAUS = [0.05, 0.1, 0.25, 0.75, 1.25, 1.75, 2.25]
DEFAULT_PERCENTILES = [10, 25, 50, 75, 90]



@dataclass
class ExampleMeta:
    player_id: Optional[str]
    opponent_id: Optional[str]
    game_id: Optional[str]
    ply_idx: Optional[int]
    move_color: Optional[str]
    game_type: Optional[str]
    opening_bucket: Optional[str]
    phase: Optional[str]
    example_id: Optional[str]
    engine_rank: Optional[int] = None
    engine_cp_gap: Optional[float] = None


@dataclass
class EvalCacheShardBuilder:
    shard_idx: int
    example_key_to_local_idx: Dict[Tuple[str, str], int] = field(default_factory=dict)
    example_boards: List[np.ndarray] = field(default_factory=list)
    example_moves: List[np.ndarray] = field(default_factory=list)
    example_game_types: List[np.uint8] = field(default_factory=list)
    example_player_ids: List[str] = field(default_factory=list)
    example_phase_ids: List[np.uint8] = field(default_factory=list)
    example_engine_ranks: List[np.int16] = field(default_factory=list)
    example_engine_cp_gaps: List[np.float32] = field(default_factory=list)
    example_ids: List[str] = field(default_factory=list)

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

    def is_empty(self) -> bool:
        return self.rows_kept_in_shard == 0

    def num_unique_examples(self) -> int:
        return len(self.example_boards)

    def clear(self) -> None:
        self.example_key_to_local_idx.clear()
        self.example_boards.clear()
        self.example_moves.clear()
        self.example_game_types.clear()
        self.example_player_ids.clear()
        self.example_phase_ids.clear()
        self.example_engine_ranks.clear()
        self.example_engine_cp_gaps.clear()
        self.example_ids.clear()
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

def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[Dict[str, Any], TrainConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = TrainConfig.from_dict(ckpt["config"])
    return ckpt, cfg

def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[StyleEncoder, TrainConfig, Dict[str, Any]]:
    ckpt, cfg = load_checkpoint(checkpoint_path, device)
    model = StyleEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt

def example_meta_from_raw(ex: Dict[str, Any]) -> ExampleMeta:
    engine_rank = None
    engine_cp_gap = None
    for key in ["engine_rank", "stockfish_rank", "move_engine_rank"]:
        if ex.get(key) is not None:
            engine_rank = int(ex[key])
            break
    for key in ["engine_cp_gap", "stockfish_cp_gap", "move_cp_gap"]:
        if ex.get(key) is not None:
            engine_cp_gap = float(ex[key])
            break
    return ExampleMeta(
        player_id=ex.get("player_id"),
        opponent_id=ex.get("opponent_id"),
        game_id=ex.get("game_id"),
        ply_idx=int(ex["ply_idx"]) if ex.get("ply_idx") is not None else None,
        move_color=ex.get("move_color"),
        game_type=ex.get("game_type"),
        opening_bucket=ex.get("opening_bucket"),
        phase=ex.get("phase"),
        example_id=ex.get("example_id"),
        engine_rank=engine_rank,
        engine_cp_gap=engine_cp_gap,
    )


def iter_jsonl_rows(input_dir: Path, max_rows: Optional[int], skip_rows: int = 0) -> Iterable[Dict[str, Any]]:
    seen = 0
    for path in sorted_jsonl_paths(input_dir):
        print(f"[read] file={path}")
        with open_jsonl_binary(path) as f:
            for line in f:
                if not line.strip():
                    continue
                seen += 1
                if skip_rows > 0 and seen <= skip_rows:
                    continue
                yield orjson.loads(line)
                if seen % 10_000 == 0:
                    print(f"[read] rows={seen:,}")
                if max_rows is not None and seen >= max_rows:
                    return

def normalize_string_array(values: Sequence[Optional[str]]) -> Tuple[np.ndarray, List[str]]:
    vocab: Dict[str, int] = {}
    rev: List[str] = [""]
    out = np.zeros(len(values), dtype=np.int32)
    for i, value in enumerate(values):
        key = "" if value is None else str(value)
        maybe = vocab.get(key)
        if maybe is None:
            maybe = len(rev)
            vocab[key] = maybe
            rev.append(key)
        out[i] = maybe
    return out, rev


def get_or_add_example(builder: EvalCacheShardBuilder, ex: Dict[str, Any], dataset_tag: str) -> int:
    meta = example_meta_from_raw(ex)
    example_id = meta.example_id or ""
    key = (dataset_tag, example_id)
    if example_id:
        maybe_idx = builder.example_key_to_local_idx.get(key)
        if maybe_idx is not None:
            return maybe_idx

    boards, move, game_type = raw_example_to_cached_arrays(ex)
    idx = len(builder.example_boards)

    if example_id:
        builder.example_key_to_local_idx[key] = idx

    builder.example_boards.append(boards)
    builder.example_moves.append(move)
    builder.example_game_types.append(game_type)
    builder.example_player_ids.append(meta.player_id or "")
    builder.example_phase_ids.append(np.uint8(PHASE_TO_ID.get(meta.phase or "", 0)))
    builder.example_engine_ranks.append(np.int16(meta.engine_rank if meta.engine_rank is not None else -1))
    builder.example_engine_cp_gaps.append(np.float32(meta.engine_cp_gap if meta.engine_cp_gap is not None else math.nan))
    builder.example_ids.append(example_id)
    return idx


def add_pair_row(builder: EvalCacheShardBuilder, row: Dict[str, Any], dataset_tag: str) -> None:
    builder.rows_read_in_shard += 1

    anchor_idx = get_or_add_example(builder, row["anchor"], dataset_tag)
    pos_indices = [get_or_add_example(builder, x, dataset_tag) for x in row.get("positives", [])]
    neg_indices = [get_or_add_example(builder, x, dataset_tag) for x in row.get("negatives", [])]

    if not pos_indices:
        builder.skipped_no_pos += 1
        return
    if not neg_indices:
        builder.skipped_no_neg += 1
        return

    builder.pair_anchor_idx.append(anchor_idx)
    builder.pos_flat.extend(pos_indices)
    builder.pos_offsets.append(len(builder.pos_flat))
    builder.neg_flat.extend(neg_indices)
    builder.neg_offsets.append(len(builder.neg_flat))
    builder.rows_kept_in_shard += 1
    builder.total_pos_candidates += len(pos_indices)
    builder.total_neg_candidates += len(neg_indices)


def save_eval_cache_shard(builder: EvalCacheShardBuilder, split_out_dir: Path) -> Dict[str, Any]:
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

    boards = np.stack(builder.example_boards, axis=0).astype(np.uint8, copy=False)
    moves = np.stack(builder.example_moves, axis=0).astype(np.uint8, copy=False)
    game_types = np.asarray(builder.example_game_types, dtype=np.uint8)
    player_ids, player_vocab = normalize_string_array(builder.example_player_ids)
    example_ids, example_vocab = normalize_string_array(builder.example_ids)
    phase_ids = np.asarray(builder.example_phase_ids, dtype=np.uint8)
    engine_ranks = np.asarray(builder.example_engine_ranks, dtype=np.int16)
    engine_cp_gaps = np.asarray(builder.example_engine_cp_gaps, dtype=np.float32)

    pair_anchor_idx = np.asarray(builder.pair_anchor_idx, dtype=np.int32)
    pos_flat = np.asarray(builder.pos_flat, dtype=np.int32)
    pos_offsets = np.asarray(builder.pos_offsets, dtype=np.int64)
    neg_flat = np.asarray(builder.neg_flat, dtype=np.int32)
    neg_offsets = np.asarray(builder.neg_offsets, dtype=np.int64)

    np.save(shard_dir / "examples_board_tokens.uint8.npy", boards, allow_pickle=False)
    np.save(shard_dir / "examples_moves.uint8.npy", moves, allow_pickle=False)
    np.save(shard_dir / "examples_game_type.uint8.npy", game_types, allow_pickle=False)
    np.save(shard_dir / "examples_player_id.int32.npy", player_ids, allow_pickle=False)
    np.save(shard_dir / "examples_phase_id.uint8.npy", phase_ids, allow_pickle=False)
    np.save(shard_dir / "examples_engine_rank.int16.npy", engine_ranks, allow_pickle=False)
    np.save(shard_dir / "examples_engine_cp_gap.float32.npy", engine_cp_gaps, allow_pickle=False)
    np.save(shard_dir / "examples_id.int32.npy", example_ids, allow_pickle=False)

    np.save(shard_dir / "pair_anchor_idx.int32.npy", pair_anchor_idx, allow_pickle=False)
    np.save(shard_dir / "pair_pos_flat.int32.npy", pos_flat, allow_pickle=False)
    np.save(shard_dir / "pair_pos_offsets.int64.npy", pos_offsets, allow_pickle=False)
    np.save(shard_dir / "pair_neg_flat.int32.npy", neg_flat, allow_pickle=False)
    np.save(shard_dir / "pair_neg_offsets.int64.npy", neg_offsets, allow_pickle=False)

    with (shard_dir / "player_id_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(player_vocab, f, indent=2)
    with (shard_dir / "example_id_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(example_vocab, f, indent=2)

    meta = {
        "shard_idx": builder.shard_idx,
        "rows_read_in_shard": builder.rows_read_in_shard,
        "rows_kept_in_shard": int(builder.rows_kept_in_shard),
        "num_unique_examples": int(boards.shape[0]),
        "boards_shape": list(boards.shape),
        "moves_shape": list(moves.shape),
        "game_types_shape": list(game_types.shape),
        "player_ids_shape": list(player_ids.shape),
        "phase_ids_shape": list(phase_ids.shape),
        "engine_ranks_shape": list(engine_ranks.shape),
        "engine_cp_gaps_shape": list(engine_cp_gaps.shape),
        "example_ids_shape": list(example_ids.shape),
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
            "player_ids": "int32 ids with player_id_vocab.json",
            "phase_ids": "uint8 ids",
            "engine_ranks": "int16, -1 means missing",
            "engine_cp_gaps": "float32, NaN means missing",
            "example_ids": "int32 ids with example_id_vocab.json",
        },
    }
    with (shard_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _parse_shard_index(shard_dir: Path) -> Optional[int]:
    name = shard_dir.name
    if not name.startswith("shard_"):
        return None
    suffix = name.split("shard_", 1)[1]
    try:
        return int(suffix)
    except Exception:
        return None


def _required_shard_file_names() -> Tuple[str, ...]:
    return (
        "meta.json",
        "examples_board_tokens.uint8.npy",
        "examples_moves.uint8.npy",
        "examples_game_type.uint8.npy",
        "pair_anchor_idx.int32.npy",
        "pair_pos_flat.int32.npy",
        "pair_pos_offsets.int64.npy",
        "pair_neg_flat.int32.npy",
        "pair_neg_offsets.int64.npy",
    )


def collect_resumable_shards(output_dir: Path) -> Tuple[List[Path], List[Dict[str, Any]], List[Path]]:
    """
    Keep only the longest contiguous prefix of complete shard dirs [0..k-1].
    Any trailing/incomplete/non-contiguous shard dirs are returned as stale.
    """
    shard_dirs = [p for p in output_dir.glob("shard_*") if p.is_dir()]
    indexed: List[Tuple[int, Path]] = []
    stale: List[Path] = []
    for sd in shard_dirs:
        idx = _parse_shard_index(sd)
        if idx is None:
            stale.append(sd)
            continue
        indexed.append((idx, sd))
    indexed.sort(key=lambda x: x[0])

    keep_dirs: List[Path] = []
    keep_meta: List[Dict[str, Any]] = []
    expected_idx = 0
    req = _required_shard_file_names()

    for idx, sd in indexed:
        if idx != expected_idx:
            stale.append(sd)
            continue
        if any(not (sd / rel).exists() for rel in req):
            stale.append(sd)
            break
        meta = read_json_if_exists(sd / "meta.json")
        if meta is None:
            stale.append(sd)
            break
        keep_dirs.append(sd)
        keep_meta.append(meta)
        expected_idx += 1

    kept_set = {p.resolve() for p in keep_dirs}
    for _, sd in indexed:
        if sd.resolve() not in kept_set and sd not in stale:
            stale.append(sd)

    return keep_dirs, keep_meta, stale


def build_eval_cache_from_pairs(
    input_dir: Path,
    output_dir: Path,
    rows_per_shard: int,
    max_rows: Optional[int],
    dataset_tag: str,
    *,
    resume_from_existing: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    split_rows_read = 0
    split_rows_kept = 0
    split_examples = 0
    split_pos = 0
    split_neg = 0
    split_skipped_no_pos = 0
    split_skipped_no_neg = 0
    shard_count = 0

    if resume_from_existing and output_dir.exists():
        keep_dirs, keep_meta, stale_dirs = collect_resumable_shards(output_dir)
        if stale_dirs:
            for sd in stale_dirs:
                shutil.rmtree(sd, ignore_errors=True)
        if keep_meta:
            for meta in keep_meta:
                split_rows_read += int(meta.get("rows_read_in_shard", 0))
                split_rows_kept += int(meta.get("rows_kept_in_shard", 0))
                split_examples += int(meta.get("num_unique_examples", 0))
                split_pos += int(meta.get("pair_pos_flat_shape", [0])[0]) if "pair_pos_flat_shape" in meta else 0
                split_neg += int(meta.get("pair_neg_flat_shape", [0])[0]) if "pair_neg_flat_shape" in meta else 0
                split_skipped_no_pos += int(meta.get("skipped_no_pos", 0))
                split_skipped_no_neg += int(meta.get("skipped_no_neg", 0))
            shard_count = len(keep_meta)
            shard_idx = shard_count
            print(
                f"[cache-resume] split={input_dir.name} keeping {shard_count} shard(s), "
                f"rows_read_already={split_rows_read:,}"
            )

    builder = EvalCacheShardBuilder(shard_idx=shard_idx)
    skip_rows = split_rows_read

    if max_rows is not None and split_rows_read >= max_rows:
        print(f"[cache-resume] split={input_dir.name} already reached max_rows={max_rows:,}; no new cache rows needed")
    for row in iter_jsonl_rows(input_dir, max_rows=max_rows, skip_rows=skip_rows):
        split_rows_read += 1
        add_pair_row(builder, row, dataset_tag=dataset_tag)

        if split_rows_read % 10_000 == 0:
            print(
                f"[cache-build] split={input_dir.name} rows={split_rows_read:,} "
                f"current_shard={builder.shard_idx} kept_in_shard={builder.rows_kept_in_shard:,} "
                f"unique_examples_in_shard={builder.num_unique_examples():,}"
            )

        if builder.rows_read_in_shard >= rows_per_shard:
            meta = save_eval_cache_shard(builder, output_dir)
            shard_count += 1
            split_rows_kept += meta.get("rows_kept_in_shard", 0)
            split_examples += meta.get("num_unique_examples", 0)
            split_pos += int(meta.get("pair_pos_flat_shape", [0])[0]) if "pair_pos_flat_shape" in meta else 0
            split_neg += int(meta.get("pair_neg_flat_shape", [0])[0]) if "pair_neg_flat_shape" in meta else 0
            split_skipped_no_pos += meta.get("skipped_no_pos", 0)
            split_skipped_no_neg += meta.get("skipped_no_neg", 0)
            print(
                f"[cache-flush] split={input_dir.name} shard={builder.shard_idx:06d} "
                f"kept={meta.get('rows_kept_in_shard', 0):,} unique_examples={meta.get('num_unique_examples', 0):,}"
            )
            builder.clear()
            gc.collect()
            shard_idx += 1
            builder = EvalCacheShardBuilder(shard_idx=shard_idx)

    if builder.rows_read_in_shard > 0:
        meta = save_eval_cache_shard(builder, output_dir)
        shard_count += 1
        split_rows_kept += meta.get("rows_kept_in_shard", 0)
        split_examples += meta.get("num_unique_examples", 0)
        split_pos += int(meta.get("pair_pos_flat_shape", [0])[0]) if "pair_pos_flat_shape" in meta else 0
        split_neg += int(meta.get("pair_neg_flat_shape", [0])[0]) if "pair_neg_flat_shape" in meta else 0
        split_skipped_no_pos += meta.get("skipped_no_pos", 0)
        split_skipped_no_neg += meta.get("skipped_no_neg", 0)
        print(
            f"[cache-flush] split={input_dir.name} shard={builder.shard_idx:06d} "
            f"kept={meta.get('rows_kept_in_shard', 0):,} unique_examples={meta.get('num_unique_examples', 0):,}"
        )

    split_meta = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rows_per_shard": rows_per_shard,
        "max_rows": max_rows,
        "dataset_tag": dataset_tag,
        "num_shards": shard_count,
        "num_pair_rows_read": split_rows_read,
        "num_pair_rows_kept": split_rows_kept,
        "sum_unique_examples_across_shards": split_examples,
        "sum_pos_candidates_across_shards": split_pos,
        "sum_neg_candidates_across_shards": split_neg,
        "skipped_no_pos": split_skipped_no_pos,
        "skipped_no_neg": split_skipped_no_neg,
        "resume_from_existing": bool(resume_from_existing),
        "rows_skipped_from_existing_cache": int(skip_rows),
        "note": "Examples are deduped within shard only, not globally across the whole split.",
    }
    with (output_dir / "_split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    return split_meta


@dataclass
class OptionalFeatureFiles:
    player_ids: Optional[Path] = None
    phase_ids: Optional[Path] = None
    engine_rank: Optional[Path] = None
    engine_cp_gap: Optional[Path] = None
    example_ids: Optional[Path] = None
    player_vocab: Optional[Path] = None
    example_vocab: Optional[Path] = None


def detect_optional_feature_files(shard_dir: Path) -> OptionalFeatureFiles:
    def pick(*names: str) -> Optional[Path]:
        for name in names:
            path = shard_dir / name
            if path.exists():
                return path
        return None

    return OptionalFeatureFiles(
        player_ids=pick(
            "examples_player_id.int32.npy",
            "examples_player_ids.int32.npy",
            "examples_player.uint16.npy",
            "examples_player.uint32.npy",
        ),
        phase_ids=pick(
            "examples_phase.uint8.npy",
            "examples_phase_id.uint8.npy",
            "examples_phase_ids.uint8.npy",
        ),
        engine_rank=pick(
            "examples_engine_rank.int16.npy",
            "examples_engine_rank.uint8.npy",
            "examples_stockfish_rank.uint8.npy",
            "examples_move_engine_rank.uint8.npy",
        ),
        engine_cp_gap=pick(
            "examples_engine_cp_gap.float32.npy",
            "examples_stockfish_cp_gap.float32.npy",
            "examples_move_cp_gap.float32.npy",
        ),
        example_ids=pick(
            "examples_id.int32.npy",
            "examples_id.npy",
            "examples_ids.npy",
            "examples_example_id.npy",
        ),
        player_vocab=pick("player_id_vocab.json"),
        example_vocab=pick("example_id_vocab.json"),
    )


def load_vocab(path: Optional[Path]) -> Optional[List[str]]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data)

class PairEvalDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        model_variant_name: str,
        max_rows: Optional[int] = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.model_variant_name = model_variant_name

        shard_dirs = sorted(self.input_dir.glob("shard_*"))
        if not shard_dirs:
            raise ValueError(f"No shard_* dirs found in {self.input_dir}")

        self.shards: List[Dict[str, Any]] = []
        self.lengths: List[int] = []
        self.optional_files: List[OptionalFeatureFiles] = []
        self.player_vocabs: List[Optional[List[str]]] = []
        self.example_vocabs: List[Optional[List[str]]] = []

        total_rows = 0
        for sd in shard_dirs:
            shard = {
                "path": sd,
                "boards": np.load(sd / "examples_board_tokens.uint8.npy", mmap_mode="r"),
                "moves": np.load(sd / "examples_moves.uint8.npy", mmap_mode="r"),
                "game_types": np.load(sd / "examples_game_type.uint8.npy", mmap_mode="r"),
                "anchor_idx": np.load(sd / "pair_anchor_idx.int32.npy", mmap_mode="r"),
                "pos_flat": np.load(sd / "pair_pos_flat.int32.npy", mmap_mode="r"),
                "pos_offsets": np.load(sd / "pair_pos_offsets.int64.npy", mmap_mode="r"),
                "neg_flat": np.load(sd / "pair_neg_flat.int32.npy", mmap_mode="r"),
                "neg_offsets": np.load(sd / "pair_neg_offsets.int64.npy", mmap_mode="r"),
            }
            self.shards.append(shard)
            self.lengths.append(len(shard["anchor_idx"]))

            files = detect_optional_feature_files(sd)
            self.optional_files.append(files)
            self.player_vocabs.append(load_vocab(files.player_vocab))
            self.example_vocabs.append(load_vocab(files.example_vocab))

            total_rows += len(shard["anchor_idx"])
            if max_rows is not None and total_rows >= max_rows:
                break

        self.cum_lengths = np.cumsum(self.lengths)
        print(f"[pair-dataset] loaded {len(self.shards)} shards from {self.input_dir}")
        print(f"[pair-dataset] total rows={len(self):,}")

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:
        shard_id = int(np.searchsorted(self.cum_lengths, idx, side="right"))
        prev = 0 if shard_id == 0 else int(self.cum_lengths[shard_id - 1])
        local_idx = idx - prev
        return shard_id, local_idx

    def _candidate_indices(self, flat: np.ndarray, offsets: np.ndarray, row_idx: int) -> np.ndarray:
        start = int(offsets[row_idx])
        end = int(offsets[row_idx + 1])
        if end <= start:
            raise ValueError(f"Row {row_idx} has no candidates in {self.input_dir}")
        return np.asarray(flat[start:end], dtype=np.int64)

    def _load_optional(self, shard_id: int, key: str) -> Optional[np.ndarray]:
        files = self.optional_files[shard_id]
        path = getattr(files, key)
        if path is None:
            return None
        return np.load(path, mmap_mode="r")

    def _decode_vocab(self, vocab: Optional[List[str]], encoded_value: Any) -> Optional[str]:
        if vocab is None or encoded_value is None:
            return None
        idx = int(encoded_value)
        if idx < 0 or idx >= len(vocab):
            return None
        value = vocab[idx]
        return value if value != "" else None

    def _example_features(self, shard: Dict[str, Any], ex_idx: int) -> Dict[str, torch.Tensor]:
        boards = torch.from_numpy(shard["boards"][ex_idx]).long()
        boards = F.one_hot(boards, num_classes=13)[..., 1:]
        boards = boards.permute(0, 2, 1).reshape(5, 12, 8, 8).float()

        feat: Dict[str, torch.Tensor] = {
            "boards": boards,
            "move": torch.from_numpy(shard["moves"][ex_idx]).long(),
        }
        if model_variant_uses_game_type(self.model_variant_name):
            feat["game_type"] = torch.tensor(int(shard["game_types"][ex_idx]), dtype=torch.long)
        if model_variant_uses_opponent_context(self.model_variant_name):
            feat["opponent_context"] = torch.zeros(32, dtype=torch.float32)
        return feat

    def _many_example_features(self, shard: Dict[str, Any], ex_indices: np.ndarray) -> Dict[str, torch.Tensor]:
        boards = torch.from_numpy(np.asarray(shard["boards"][ex_indices])).long()   # [K, 5, 64]
        boards = F.one_hot(boards, num_classes=13)[..., 1:]                         # [K, 5, 64, 12]
        boards = boards.permute(0, 1, 3, 2).reshape(len(ex_indices), 5, 12, 8, 8).float()

        feat: Dict[str, torch.Tensor] = {
            "boards": boards,
            "move": torch.from_numpy(np.asarray(shard["moves"][ex_indices])).long(),   # [K, 3]
        }
        if model_variant_uses_game_type(self.model_variant_name):
            feat["game_type"] = torch.from_numpy(
                np.asarray(shard["game_types"][ex_indices])
            ).long()  # [K]
        if model_variant_uses_opponent_context(self.model_variant_name):
            feat["opponent_context"] = torch.zeros((len(ex_indices), 32), dtype=torch.float32)
        return feat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_id, local_idx = self._locate(idx)
        shard = self.shards[shard_id]

        anchor_idx = int(shard["anchor_idx"][local_idx])
        pos_indices = self._candidate_indices(shard["pos_flat"], shard["pos_offsets"], local_idx)
        neg_indices = self._candidate_indices(shard["neg_flat"], shard["neg_offsets"], local_idx)

        player_ids = self._load_optional(shard_id, "player_ids")
        phase_ids = self._load_optional(shard_id, "phase_ids")
        engine_rank = self._load_optional(shard_id, "engine_rank")
        engine_cp_gap = self._load_optional(shard_id, "engine_cp_gap")
        example_ids = self._load_optional(shard_id, "example_ids")
        player_vocab = self.player_vocabs[shard_id]
        example_vocab = self.example_vocabs[shard_id]

        return {
            "anchor": self._example_features(shard, anchor_idx),
            "positive_all": self._many_example_features(shard, pos_indices),
            "negative_all": self._many_example_features(shard, neg_indices),
            "pos_count": int(len(pos_indices)),
            "neg_count": int(len(neg_indices)),
            "meta": {
                "shard_id": shard_id,
                "row_idx": local_idx,
                "anchor_idx": anchor_idx,
                "anchor_player_id": self._decode_vocab(player_vocab, player_ids[anchor_idx]) if player_ids is not None else None,
                "anchor_phase_id": int(phase_ids[anchor_idx]) if phase_ids is not None else None,
                "anchor_engine_rank": int(engine_rank[anchor_idx]) if engine_rank is not None and int(engine_rank[anchor_idx]) >= 0 else None,
                "anchor_engine_cp_gap": float(engine_cp_gap[anchor_idx]) if engine_cp_gap is not None and not math.isnan(float(engine_cp_gap[anchor_idx])) else None,
                "anchor_example_id": self._decode_vocab(example_vocab, example_ids[anchor_idx]) if example_ids is not None else None,
                "anchor_game_type_id": int(shard["game_types"][anchor_idx]) if "game_types" in shard else None,
                "n_pos": int(len(pos_indices)),
                "n_neg": int(len(neg_indices)),
            },
        }

class ExampleEmbeddingDataset(Dataset):
    def __init__(self, input_dir: str | Path, model_variant_name: str, max_examples: Optional[int] = None) -> None:
        self.input_dir = Path(input_dir)
        self.model_variant_name = model_variant_name
        self.shards: List[Dict[str, Any]] = []
        self.lengths: List[int] = []
        self.optional_files: List[OptionalFeatureFiles] = []
        self.player_vocabs: List[Optional[List[str]]] = []
        self.example_vocabs: List[Optional[List[str]]] = []

        shard_dirs = sorted(self.input_dir.glob("shard_*"))
        if not shard_dirs:
            raise ValueError(f"No shard_* dirs found in {self.input_dir}")

        total_examples = 0
        for sd in shard_dirs:
            shard = {
                "boards": np.load(sd / "examples_board_tokens.uint8.npy", mmap_mode="r"),
                "moves": np.load(sd / "examples_moves.uint8.npy", mmap_mode="r"),
                "game_types": np.load(sd / "examples_game_type.uint8.npy", mmap_mode="r"),
            }
            self.shards.append(shard)
            self.lengths.append(len(shard["boards"]))

            files = detect_optional_feature_files(sd)
            self.optional_files.append(files)
            self.player_vocabs.append(load_vocab(files.player_vocab))
            self.example_vocabs.append(load_vocab(files.example_vocab))

            total_examples += len(shard["boards"])
            if max_examples is not None and total_examples >= max_examples:
                break

        self.cum_lengths = np.cumsum(self.lengths)
        print(f"[example-dataset] loaded {len(self.shards)} shards from {self.input_dir}")
        print(f"[example-dataset] total examples={len(self):,}")

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:
        shard_id = int(np.searchsorted(self.cum_lengths, idx, side="right"))
        prev = 0 if shard_id == 0 else int(self.cum_lengths[shard_id - 1])
        local_idx = idx - prev
        return shard_id, local_idx

    def _load_optional(self, shard_id: int, key: str) -> Optional[np.ndarray]:
        files = self.optional_files[shard_id]
        path = getattr(files, key)
        if path is None:
            return None
        return np.load(path, mmap_mode="r")

    def _decode_vocab(self, vocab: Optional[List[str]], encoded_value: Any) -> Optional[str]:
        if vocab is None or encoded_value is None:
            return None
        idx = int(encoded_value)
        if idx < 0 or idx >= len(vocab):
            return None
        value = vocab[idx]
        return value if value != "" else None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_id, local_idx = self._locate(idx)
        shard = self.shards[shard_id]
        boards = torch.from_numpy(shard["boards"][local_idx]).long()
        boards = F.one_hot(boards, num_classes=13)[..., 1:]
        boards = boards.permute(0, 2, 1).reshape(5, 12, 8, 8).float()

        feat: Dict[str, torch.Tensor] = {
            "boards": boards,
            "move": torch.from_numpy(shard["moves"][local_idx]).long(),
        }
        if model_variant_uses_game_type(self.model_variant_name):
            feat["game_type"] = torch.tensor(int(shard["game_types"][local_idx]), dtype=torch.long)
        if model_variant_uses_opponent_context(self.model_variant_name):
            feat["opponent_context"] = torch.zeros(32, dtype=torch.float32)

        player_ids = self._load_optional(shard_id, "player_ids")
        phase_ids = self._load_optional(shard_id, "phase_ids")
        engine_rank = self._load_optional(shard_id, "engine_rank")
        engine_cp_gap = self._load_optional(shard_id, "engine_cp_gap")
        example_ids = self._load_optional(shard_id, "example_ids")
        player_vocab = self.player_vocabs[shard_id]
        example_vocab = self.example_vocabs[shard_id]

        return {
            "features": feat,
            "meta": {
                "shard_id": shard_id,
                "example_idx": local_idx,
                "player_id": self._decode_vocab(player_vocab, player_ids[local_idx]) if player_ids is not None else None,
                "phase_id": int(phase_ids[local_idx]) if phase_ids is not None else None,
                "engine_rank": int(engine_rank[local_idx]) if engine_rank is not None and int(engine_rank[local_idx]) >= 0 else None,
                "engine_cp_gap": float(engine_cp_gap[local_idx]) if engine_cp_gap is not None and not math.isnan(float(engine_cp_gap[local_idx])) else None,
                "example_id": self._decode_vocab(example_vocab, example_ids[local_idx]) if example_ids is not None else None,
            },
        }

def collate_pair_eval(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    def stack_dicts(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for item in items:
            for k, v in item.items():
                out[k].append(v)
        return {k: torch.stack(v, dim=0) for k, v in out.items()}

    def concat_dicts(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for item in items:
            for k, v in item.items():
                out[k].append(v)
        return {k: torch.cat(v, dim=0) for k, v in out.items()}

    return {
        "anchor": stack_dicts([x["anchor"] for x in batch]),                 # [B, ...]
        "positive_all": concat_dicts([x["positive_all"] for x in batch]),    # [sum_pos, ...]
        "negative_all": concat_dicts([x["negative_all"] for x in batch]),    # [sum_neg, ...]
        "pos_counts": torch.tensor([x["pos_count"] for x in batch], dtype=torch.long),
        "neg_counts": torch.tensor([x["neg_count"] for x in batch], dtype=torch.long),
        "meta": [x["meta"] for x in batch],
    }


def collate_example_embeddings(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for row in batch:
        for k, v in row["features"].items():
            out[k].append(v)
    return {
        "features": {k: torch.stack(v, dim=0) for k, v in out.items()},
        "meta": [x["meta"] for x in batch],
    }


def move_to_device(batch_part: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch_part.items()}


def percentiles_dict(values: np.ndarray, prefix: str, percentiles: Sequence[int] = DEFAULT_PERCENTILES) -> Dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_mean": math.nan, **{f"{prefix}_p{p}": math.nan for p in percentiles}}
    out = {f"{prefix}_mean": float(np.mean(values))}
    for p in percentiles:
        out[f"{prefix}_p{p}"] = float(np.percentile(values, p))
    return out


def safe_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = float(ranks[y_true == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def precision_recall_curve_points(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    s_sorted = scores[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((y_true == 1).sum()), 1)
    return precision, recall, s_sorted


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    precision, recall, _ = precision_recall_curve_points(y_true, scores)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def _reservoir_update_player_samples(
    sample_store: Dict[str, Dict[str, Any]],
    player_id: Optional[str],
    embedding: np.ndarray,
    meta: Dict[str, Any],
    per_player_cap: int,
    rng: np.random.Generator,
) -> None:
    if player_id is None or per_player_cap <= 0:
        return

    bucket = sample_store.get(player_id)
    if bucket is None:
        bucket = {"seen": 0, "embeddings": [], "meta": []}
        sample_store[player_id] = bucket

    bucket["seen"] += 1
    emb_copy = embedding.astype(np.float32, copy=True)
    meta_copy = dict(meta)

    if len(bucket["embeddings"]) < per_player_cap:
        bucket["embeddings"].append(emb_copy)
        bucket["meta"].append(meta_copy)
        return

    j = int(rng.integers(0, bucket["seen"]))
    if j < per_player_cap:
        bucket["embeddings"][j] = emb_copy
        bucket["meta"][j] = meta_copy


def finalize_sampled_anchor_embeddings(
    sample_store: Dict[str, Dict[str, Any]],
    *,
    max_players: Optional[int],
    min_examples_per_player: int,
    player_selection: str,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    eligible: List[Tuple[str, Dict[str, Any]]] = []
    for pid, bucket in sample_store.items():
        n_kept = len(bucket["embeddings"])
        if n_kept >= min_examples_per_player:
            eligible.append((pid, bucket))

    if not eligible:
        return (
            np.zeros((0, 1), dtype=np.float32),
            [],
            {
                "sampling_method": "anchor_reservoir",
                "n_players_seen": int(len(sample_store)),
                "n_players_eligible": 0,
                "n_players_selected": 0,
                "n_embeddings": 0,
                "player_selection": player_selection,
                "min_examples_per_player": int(min_examples_per_player),
                "max_players": None if max_players is None else int(max_players),
            },
        )

    if player_selection == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(eligible)
    else:
        eligible.sort(
            key=lambda item: (int(item[1]["seen"]), len(item[1]["embeddings"]), str(item[0])),
            reverse=True,
        )

    if max_players is not None:
        eligible = eligible[:max_players]

    embeddings: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []
    players_summary: Dict[str, Any] = {}
    for pid, bucket in eligible:
        player_embs = [np.asarray(x, dtype=np.float32) for x in bucket["embeddings"]]
        player_meta = [dict(m) for m in bucket["meta"]]
        embeddings.extend(player_embs)
        meta_rows.extend(player_meta)
        players_summary[str(pid)] = {
            "n_seen_anchor_rows": int(bucket["seen"]),
            "n_sampled_embeddings": int(len(player_embs)),
        }

    if embeddings:
        emb = normalize_rows(np.stack(embeddings, axis=0).astype(np.float32, copy=False))
    else:
        emb = np.zeros((0, 1), dtype=np.float32)

    summary = {
        "sampling_method": "anchor_reservoir",
        "n_players_seen": int(len(sample_store)),
        "n_players_eligible": int(len(players_summary)),
        "n_players_selected": int(len(players_summary)),
        "n_embeddings": int(len(meta_rows)),
        "player_selection": player_selection,
        "min_examples_per_player": int(min_examples_per_player),
        "max_players": None if max_players is None else int(max_players),
        "players": players_summary,
    }
    return emb, meta_rows, summary


def _decode_vocab_value(vocab: Optional[List[str]], encoded_value: Any) -> Optional[str]:
    if vocab is None or encoded_value is None:
        return None
    try:
        idx = int(encoded_value)
    except Exception:
        return None
    if idx < 0 or idx >= len(vocab):
        return None
    value = vocab[idx]
    return value if value != "" else None


def _checkpoint_signature(checkpoint_path: Optional[Path]) -> str:
    if checkpoint_path is None:
        return "unknown_checkpoint"
    try:
        stat = checkpoint_path.stat()
        return f"{checkpoint_path.resolve()}|{int(stat.st_mtime)}|{int(stat.st_size)}"
    except Exception:
        return str(checkpoint_path)


def _embedding_cache_paths(cache_dir: Path, shard_name: str) -> Tuple[Path, Path, Path]:
    emb_path = cache_dir / f"{shard_name}.embeddings.npz"
    meta_path = cache_dir / f"{shard_name}.meta.json"
    tmp_emb_path = cache_dir / f".{shard_name}.embeddings.tmp.npz"
    return emb_path, meta_path, tmp_emb_path


def _load_cached_shard_embeddings(
    *,
    emb_path: Path,
    meta_path: Path,
    expected_examples: int,
    model_signature: str,
) -> Optional[np.ndarray]:
    meta = read_json_if_exists(meta_path)
    if meta is None:
        return None
    if str(meta.get("model_signature", "")) != model_signature:
        return None
    if int(meta.get("n_examples", -1)) != int(expected_examples):
        return None
    if not emb_path.exists():
        return None
    try:
        with np.load(emb_path) as data:
            emb = np.asarray(data["embeddings"], dtype=np.float32)
    except Exception:
        return None
    if emb.ndim != 2 or emb.shape[0] != int(expected_examples):
        return None
    return emb


def _save_cached_shard_embeddings(
    *,
    emb_path: Path,
    meta_path: Path,
    tmp_emb_path: Path,
    embeddings: np.ndarray,
    model_signature: str,
) -> None:
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp_emb_path, embeddings=np.asarray(embeddings, dtype=np.float32))
    os.replace(tmp_emb_path, emb_path)
    write_json(
        meta_path,
        {
            "model_signature": model_signature,
            "n_examples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        },
    )


def _encode_shard_embeddings(
    *,
    model: torch.nn.Module,
    device: torch.device,
    model_variant_name: str,
    boards_u8: np.ndarray,
    moves_u8: np.ndarray,
    game_types_u8: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    n = int(boards_u8.shape[0])
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            boards = torch.from_numpy(np.asarray(boards_u8[start:end])).long()
            boards = F.one_hot(boards, num_classes=13)[..., 1:]
            boards = boards.permute(0, 1, 3, 2).reshape(end - start, 5, 12, 8, 8).float()

            feats: Dict[str, torch.Tensor] = {
                "boards": boards.to(device),
                "move": torch.from_numpy(np.asarray(moves_u8[start:end])).long().to(device),
            }
            if model_variant_uses_game_type(model_variant_name):
                feats["game_type"] = torch.from_numpy(np.asarray(game_types_u8[start:end])).long().to(device)
            if model_variant_uses_opponent_context(model_variant_name):
                feats["opponent_context"] = torch.zeros((end - start, 32), dtype=torch.float32, device=device)
            z = model(feats).detach().cpu().numpy().astype(np.float32, copy=False)
            chunks.append(z)
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def compute_pair_metrics_with_embedding_cache(
    *,
    model: torch.nn.Module,
    device: torch.device,
    model_variant_name: str,
    cached_split_dir: Path,
    eval_taus: Sequence[float],
    batch_size: int,
    max_batches: Optional[int],
    progress_every_batches: int,
    sample_anchor_embeddings_per_player: int,
    sampled_embedding_max_players: Optional[int],
    sampled_embedding_min_examples_per_player: int,
    sampled_embedding_player_selection: str,
    sampling_seed: int,
    embedding_cache_dir: Path,
    model_signature: str,
) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    def concat_or_empty(xs: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs) if xs else np.array([], dtype=np.float32)

    shard_dirs = sorted([p for p in cached_split_dir.glob("shard_*") if p.is_dir()])
    if not shard_dirs:
        raise ValueError(f"No shard_* dirs found in {cached_split_dir}")

    cand_pos_l2_all: List[np.ndarray] = []
    cand_neg_l2_all: List[np.ndarray] = []
    cand_pos_cos_all: List[np.ndarray] = []
    cand_neg_cos_all: List[np.ndarray] = []

    row_mean_pos_l2_all: List[np.ndarray] = []
    row_mean_neg_l2_all: List[np.ndarray] = []
    row_best_pos_l2_all: List[np.ndarray] = []
    row_hardest_neg_l2_all: List[np.ndarray] = []
    row_mean_pos_cos_all: List[np.ndarray] = []
    row_mean_neg_cos_all: List[np.ndarray] = []
    row_best_pos_cos_all: List[np.ndarray] = []
    row_hardest_neg_cos_all: List[np.ndarray] = []

    engine_buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    sample_store: Dict[str, Dict[str, Any]] = {}
    rng = np.random.default_rng(sampling_seed)

    total_rows = 0
    for sd in shard_dirs:
        anchor_idx = np.load(sd / "pair_anchor_idx.int32.npy", mmap_mode="r")
        total_rows += int(len(anchor_idx))
    max_rows = int(total_rows if max_batches is None else min(total_rows, max_batches * batch_size))
    print(f"[eval-progress] pair-metrics (embedding-cache) start: total_rows={total_rows:,} max_rows={max_rows:,}")

    processed_rows = 0
    start_t = time.time()

    for sd in shard_dirs:
        boards = np.load(sd / "examples_board_tokens.uint8.npy", mmap_mode="r")
        moves = np.load(sd / "examples_moves.uint8.npy", mmap_mode="r")
        game_types = np.load(sd / "examples_game_type.uint8.npy", mmap_mode="r")
        anchor_idx = np.load(sd / "pair_anchor_idx.int32.npy", mmap_mode="r")
        pos_flat = np.load(sd / "pair_pos_flat.int32.npy", mmap_mode="r")
        pos_offsets = np.load(sd / "pair_pos_offsets.int64.npy", mmap_mode="r")
        neg_flat = np.load(sd / "pair_neg_flat.int32.npy", mmap_mode="r")
        neg_offsets = np.load(sd / "pair_neg_offsets.int64.npy", mmap_mode="r")

        files = detect_optional_feature_files(sd)
        player_ids = np.load(files.player_ids, mmap_mode="r") if files.player_ids is not None else None
        phase_ids = np.load(files.phase_ids, mmap_mode="r") if files.phase_ids is not None else None
        engine_rank = np.load(files.engine_rank, mmap_mode="r") if files.engine_rank is not None else None
        engine_cp_gap = np.load(files.engine_cp_gap, mmap_mode="r") if files.engine_cp_gap is not None else None
        example_ids = np.load(files.example_ids, mmap_mode="r") if files.example_ids is not None else None
        player_vocab = load_vocab(files.player_vocab)
        example_vocab = load_vocab(files.example_vocab)

        emb_path, meta_path, tmp_emb_path = _embedding_cache_paths(embedding_cache_dir, sd.name)
        emb = _load_cached_shard_embeddings(
            emb_path=emb_path,
            meta_path=meta_path,
            expected_examples=int(boards.shape[0]),
            model_signature=model_signature,
        )
        if emb is None:
            print(f"[embed-cache] building {sd.name} embeddings cache")
            emb = _encode_shard_embeddings(
                model=model,
                device=device,
                model_variant_name=model_variant_name,
                boards_u8=boards,
                moves_u8=moves,
                game_types_u8=game_types,
                batch_size=max(1, int(batch_size)),
            )
            _save_cached_shard_embeddings(
                emb_path=emb_path,
                meta_path=meta_path,
                tmp_emb_path=tmp_emb_path,
                embeddings=emb,
                model_signature=model_signature,
            )

        n_rows = int(len(anchor_idx))
        for row_i in range(n_rows):
            if processed_rows >= max_rows:
                break
            aidx = int(anchor_idx[row_i])
            p0, p1 = int(pos_offsets[row_i]), int(pos_offsets[row_i + 1])
            n0, n1 = int(neg_offsets[row_i]), int(neg_offsets[row_i + 1])
            pos_idx = np.asarray(pos_flat[p0:p1], dtype=np.int64)
            neg_idx = np.asarray(neg_flat[n0:n1], dtype=np.int64)
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                processed_rows += 1
                continue

            z_anchor = emb[aidx]
            z_pos = emb[pos_idx]
            z_neg = emb[neg_idx]

            pos_diff = z_pos - z_anchor[None, :]
            neg_diff = z_neg - z_anchor[None, :]
            pos_l2_row = np.linalg.norm(pos_diff, axis=1).astype(np.float32)
            neg_l2_row = np.linalg.norm(neg_diff, axis=1).astype(np.float32)
            pos_cos_row = (z_pos @ z_anchor).astype(np.float32)
            neg_cos_row = (z_neg @ z_anchor).astype(np.float32)

            cand_pos_l2_all.append(pos_l2_row)
            cand_neg_l2_all.append(neg_l2_row)
            cand_pos_cos_all.append(pos_cos_row)
            cand_neg_cos_all.append(neg_cos_row)

            row_mean_pos_l2 = float(np.mean(pos_l2_row))
            row_mean_neg_l2 = float(np.mean(neg_l2_row))
            row_best_pos_l2 = float(np.min(pos_l2_row))
            row_hardest_neg_l2 = float(np.min(neg_l2_row))
            row_mean_pos_cos = float(np.mean(pos_cos_row))
            row_mean_neg_cos = float(np.mean(neg_cos_row))
            row_best_pos_cos = float(np.max(pos_cos_row))
            row_hardest_neg_cos = float(np.max(neg_cos_row))

            row_mean_pos_l2_all.append(np.asarray([row_mean_pos_l2], dtype=np.float32))
            row_mean_neg_l2_all.append(np.asarray([row_mean_neg_l2], dtype=np.float32))
            row_best_pos_l2_all.append(np.asarray([row_best_pos_l2], dtype=np.float32))
            row_hardest_neg_l2_all.append(np.asarray([row_hardest_neg_l2], dtype=np.float32))
            row_mean_pos_cos_all.append(np.asarray([row_mean_pos_cos], dtype=np.float32))
            row_mean_neg_cos_all.append(np.asarray([row_mean_neg_cos], dtype=np.float32))
            row_best_pos_cos_all.append(np.asarray([row_best_pos_cos], dtype=np.float32))
            row_hardest_neg_cos_all.append(np.asarray([row_hardest_neg_cos], dtype=np.float32))

            anchor_player_id = _decode_vocab_value(player_vocab, player_ids[aidx]) if player_ids is not None else None
            anchor_phase_id = int(phase_ids[aidx]) if phase_ids is not None else None
            anchor_engine_rank = int(engine_rank[aidx]) if engine_rank is not None and int(engine_rank[aidx]) >= 0 else None
            anchor_engine_cp_gap = (
                float(engine_cp_gap[aidx]) if engine_cp_gap is not None and not math.isnan(float(engine_cp_gap[aidx])) else None
            )
            anchor_example_id = _decode_vocab_value(example_vocab, example_ids[aidx]) if example_ids is not None else None
            anchor_game_type_id = int(game_types[aidx]) if game_types is not None else None

            sampled_meta = {
                "player_id": anchor_player_id,
                "phase_id": anchor_phase_id,
                "game_type_id": anchor_game_type_id,
                "engine_rank": anchor_engine_rank,
                "engine_cp_gap": anchor_engine_cp_gap,
                "example_id": anchor_example_id,
                "source": "anchor_row_sample",
                "row_idx": int(row_i),
                "shard_id": str(sd.name),
            }
            _reservoir_update_player_samples(
                sample_store=sample_store,
                player_id=sampled_meta["player_id"],
                embedding=np.asarray(z_anchor, dtype=np.float32),
                meta=sampled_meta,
                per_player_cap=sample_anchor_embeddings_per_player,
                rng=rng,
            )

            rank = anchor_engine_rank
            if rank is not None:
                if rank <= 3:
                    bucket = "engine_like"
                elif rank > 10:
                    bucket = "engine_unlike"
                else:
                    bucket = "engine_middle"
                engine_buckets[bucket]["row_mean_pos_l2"].append(row_mean_pos_l2)
                engine_buckets[bucket]["row_mean_neg_l2"].append(row_mean_neg_l2)
                engine_buckets[bucket]["row_best_pos_l2"].append(row_best_pos_l2)
                engine_buckets[bucket]["row_hardest_neg_l2"].append(row_hardest_neg_l2)
                engine_buckets[bucket]["row_mean_pos_cos"].append(row_mean_pos_cos)
                engine_buckets[bucket]["row_mean_neg_cos"].append(row_mean_neg_cos)
                engine_buckets[bucket]["row_best_pos_cos"].append(row_best_pos_cos)
                engine_buckets[bucket]["row_hardest_neg_cos"].append(row_hardest_neg_cos)

            processed_rows += 1
            if progress_every_batches > 0 and (processed_rows % (progress_every_batches * max(1, int(batch_size))) == 0):
                elapsed = max(1e-9, time.time() - start_t)
                rps = processed_rows / elapsed
                rem = max(max_rows - processed_rows, 0)
                eta_min = rem / max(rps, 1e-9) / 60.0
                print(
                    f"[eval-progress] pair-metrics rows={processed_rows:,}/{max_rows:,} "
                    f"elapsed_min={elapsed/60.0:.1f} rows_per_sec={rps:.1f} eta_min={eta_min:.1f}"
                )

        if processed_rows >= max_rows:
            break

    print(
        f"[eval-progress] pair-metrics (embedding-cache) done: rows={processed_rows:,} "
        f"elapsed_min={(time.time()-start_t)/60.0:.1f}"
    )

    cand_pos_l2 = concat_or_empty(cand_pos_l2_all)
    cand_neg_l2 = concat_or_empty(cand_neg_l2_all)
    cand_pos_cos = concat_or_empty(cand_pos_cos_all)
    cand_neg_cos = concat_or_empty(cand_neg_cos_all)
    row_mean_pos_l2 = concat_or_empty(row_mean_pos_l2_all)
    row_mean_neg_l2 = concat_or_empty(row_mean_neg_l2_all)
    row_best_pos_l2 = concat_or_empty(row_best_pos_l2_all)
    row_hardest_neg_l2 = concat_or_empty(row_hardest_neg_l2_all)
    row_mean_pos_cos = concat_or_empty(row_mean_pos_cos_all)
    row_mean_neg_cos = concat_or_empty(row_mean_neg_cos_all)
    row_best_pos_cos = concat_or_empty(row_best_pos_cos_all)
    row_hardest_neg_cos = concat_or_empty(row_hardest_neg_cos_all)

    out: Dict[str, Any] = {
        "n_anchor_rows": int(len(row_mean_pos_l2)),
        "n_pos_candidates": int(len(cand_pos_l2)),
        "n_neg_candidates": int(len(cand_neg_l2)),
        "candidate_level": {
            "l2": {
                "positive": percentiles_dict(cand_pos_l2, "pos_l2"),
                "negative": percentiles_dict(cand_neg_l2, "neg_l2"),
                "gap_mean": float(np.mean(cand_neg_l2) - np.mean(cand_pos_l2)) if len(cand_pos_l2) else math.nan,
            },
            "cosine": {
                "positive": percentiles_dict(cand_pos_cos, "pos_cos"),
                "negative": percentiles_dict(cand_neg_cos, "neg_cos"),
                "gap_mean": float(np.mean(cand_pos_cos) - np.mean(cand_neg_cos)) if len(cand_pos_cos) else math.nan,
            },
        },
        "row_aggregated": {
            "l2": {
                "mean_positive": percentiles_dict(row_mean_pos_l2, "row_mean_pos_l2"),
                "mean_negative": percentiles_dict(row_mean_neg_l2, "row_mean_neg_l2"),
                "best_positive": percentiles_dict(row_best_pos_l2, "row_best_pos_l2"),
                "hardest_negative": percentiles_dict(row_hardest_neg_l2, "row_hardest_neg_l2"),
                "mean_gap": float(np.mean(row_mean_neg_l2) - np.mean(row_mean_pos_l2)) if len(row_mean_pos_l2) else math.nan,
                "hard_gap": float(np.mean(row_hardest_neg_l2) - np.mean(row_mean_pos_l2)) if len(row_mean_pos_l2) else math.nan,
            },
            "cosine": {
                "mean_positive": percentiles_dict(row_mean_pos_cos, "row_mean_pos_cos"),
                "mean_negative": percentiles_dict(row_mean_neg_cos, "row_mean_neg_cos"),
                "best_positive": percentiles_dict(row_best_pos_cos, "row_best_pos_cos"),
                "hardest_negative": percentiles_dict(row_hardest_neg_cos, "row_hardest_neg_cos"),
                "mean_gap": float(np.mean(row_mean_pos_cos) - np.mean(row_mean_neg_cos)) if len(row_mean_pos_cos) else math.nan,
                "hard_gap": float(np.mean(row_mean_pos_cos) - np.mean(row_hardest_neg_cos)) if len(row_mean_pos_cos) else math.nan,
                "pair_acc_mean_vs_hardest": float(np.mean(row_mean_pos_cos > row_hardest_neg_cos)) if len(row_mean_pos_cos) else math.nan,
            },
            "by_eval_tau": {},
        },
        "engine_conditioned": {},
    }

    for tau in eval_taus:
        row_mean_pos_scaled = row_mean_pos_cos / tau
        row_mean_neg_scaled = row_mean_neg_cos / tau
        row_hard_neg_scaled = row_hardest_neg_cos / tau
        row_mean_pos_exp = np.exp(np.clip(row_mean_pos_scaled, -60.0, 60.0))
        row_mean_neg_exp = np.exp(np.clip(row_mean_neg_scaled, -60.0, 60.0))
        row_hard_neg_exp = np.exp(np.clip(row_hard_neg_scaled, -60.0, 60.0))
        logits = np.stack([row_mean_pos_scaled, row_hard_neg_scaled], axis=1)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        loss = -np.log(np.clip(probs[:, 0], 1e-12, None))
        out["row_aggregated"]["by_eval_tau"][str(tau)] = {
            "dot_over_tau": {
                "mean_positive": percentiles_dict(row_mean_pos_scaled, "row_mean_pos_dot_over_tau"),
                "mean_negative": percentiles_dict(row_mean_neg_scaled, "row_mean_neg_dot_over_tau"),
                "hardest_negative": percentiles_dict(row_hard_neg_scaled, "row_hard_neg_dot_over_tau"),
                "mean_gap": float(np.mean(row_mean_pos_scaled) - np.mean(row_mean_neg_scaled)) if len(row_mean_pos_scaled) else math.nan,
                "hard_gap": float(np.mean(row_mean_pos_scaled) - np.mean(row_hard_neg_scaled)) if len(row_mean_pos_scaled) else math.nan,
            },
            "exp_dot_over_tau": {
                "mean_positive": percentiles_dict(row_mean_pos_exp, "row_mean_pos_exp_dot_over_tau"),
                "mean_negative": percentiles_dict(row_mean_neg_exp, "row_mean_neg_exp_dot_over_tau"),
                "hardest_negative": percentiles_dict(row_hard_neg_exp, "row_hard_neg_exp_dot_over_tau"),
                "mean_gap": float(np.mean(row_mean_pos_exp) - np.mean(row_mean_neg_exp)) if len(row_mean_pos_exp) else math.nan,
                "hard_gap": float(np.mean(row_mean_pos_exp) - np.mean(row_hard_neg_exp)) if len(row_mean_pos_exp) else math.nan,
            },
            "infonce_like_loss_mean_vs_hardest_neg": float(np.mean(loss)) if len(loss) else math.nan,
        }

    for bucket, vals in engine_buckets.items():
        pos_cos_b = np.asarray(vals.get("row_mean_pos_cos", []), dtype=np.float32)
        neg_cos_b = np.asarray(vals.get("row_mean_neg_cos", []), dtype=np.float32)
        hard_neg_cos_b = np.asarray(vals.get("row_hardest_neg_cos", []), dtype=np.float32)
        pos_l2_b = np.asarray(vals.get("row_mean_pos_l2", []), dtype=np.float32)
        neg_l2_b = np.asarray(vals.get("row_mean_neg_l2", []), dtype=np.float32)
        hard_neg_l2_b = np.asarray(vals.get("row_hardest_neg_l2", []), dtype=np.float32)
        out["engine_conditioned"][bucket] = {
            "n_anchor_rows": int(len(pos_cos_b)),
            "l2_mean_gap": float(np.mean(neg_l2_b) - np.mean(pos_l2_b)) if len(pos_l2_b) else math.nan,
            "l2_hard_gap": float(np.mean(hard_neg_l2_b) - np.mean(pos_l2_b)) if len(pos_l2_b) else math.nan,
            "cos_mean_gap": float(np.mean(pos_cos_b) - np.mean(neg_cos_b)) if len(pos_cos_b) else math.nan,
            "cos_hard_gap": float(np.mean(pos_cos_b) - np.mean(hard_neg_cos_b)) if len(pos_cos_b) else math.nan,
        }
        for tau in eval_taus:
            out["engine_conditioned"][bucket][f"dot_over_tau_mean_gap@{tau}"] = (
                float(np.mean(pos_cos_b / tau) - np.mean(neg_cos_b / tau)) if len(pos_cos_b) else math.nan
            )
            out["engine_conditioned"][bucket][f"dot_over_tau_hard_gap@{tau}"] = (
                float(np.mean(pos_cos_b / tau) - np.mean(hard_neg_cos_b / tau)) if len(pos_cos_b) else math.nan
            )

    sampled_embeddings, sampled_meta_rows, sample_summary = finalize_sampled_anchor_embeddings(
        sample_store,
        max_players=sampled_embedding_max_players,
        min_examples_per_player=sampled_embedding_min_examples_per_player,
        player_selection=sampled_embedding_player_selection,
        seed=sampling_seed,
    )

    rng_diag = np.random.default_rng(42)
    max_diag = 300_000

    def _subsample(arr: np.ndarray) -> np.ndarray:
        if len(arr) <= max_diag:
            return arr
        idx = rng_diag.choice(len(arr), size=max_diag, replace=False)
        return arr[idx]

    pos_c = _subsample(cand_pos_cos)
    neg_c = _subsample(cand_neg_cos)
    hard_n = _subsample(row_hardest_neg_cos)
    soft_n = _subsample(row_mean_neg_cos)
    pos_r = _subsample(row_mean_pos_cos)
    score_distributions = {
        "positive_candidate_cosine": _distribution_summary(pos_c),
        "negative_candidate_cosine": _distribution_summary(neg_c),
        "positive_row_mean_cosine": _distribution_summary(pos_r),
        "hard_negative_row_cosine": _distribution_summary(hard_n),
        "soft_negative_row_cosine": _distribution_summary(soft_n),
    }
    pairwise_auc = {
        "positive_vs_all_negative": _pair_auc_ap(
            np.concatenate([np.ones(len(pos_c), dtype=np.int64), np.zeros(len(neg_c), dtype=np.int64)]),
            np.concatenate([pos_c, neg_c]),
        )
        if len(pos_c) and len(neg_c)
        else {"n_pairs": 0, "roc_auc": math.nan, "average_precision": math.nan},
        "positive_vs_hard_negative": _pair_auc_ap(
            np.concatenate([np.ones(len(pos_r), dtype=np.int64), np.zeros(len(hard_n), dtype=np.int64)]),
            np.concatenate([pos_r, hard_n]),
        )
        if len(pos_r) and len(hard_n)
        else {"n_pairs": 0, "roc_auc": math.nan, "average_precision": math.nan},
        "positive_vs_soft_negative": _pair_auc_ap(
            np.concatenate([np.ones(len(pos_r), dtype=np.int64), np.zeros(len(soft_n), dtype=np.int64)]),
            np.concatenate([pos_r, soft_n]),
        )
        if len(pos_r) and len(soft_n)
        else {"n_pairs": 0, "roc_auc": math.nan, "average_precision": math.nan},
    }
    extra_pair_diagnostics = {"score_distributions": score_distributions, "pairwise_auc": pairwise_auc}
    return out, sampled_embeddings, sampled_meta_rows, sample_summary, extra_pair_diagnostics


def compute_pair_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_taus: Sequence[float],
    max_batches: Optional[int] = None,
    *,
    sample_anchor_embeddings_per_player: int = 16,
    sampled_embedding_max_players: Optional[int] = 500,
    sampled_embedding_min_examples_per_player: int = 2,
    sampled_embedding_player_selection: str = "most_seen",
    sampling_seed: int = 42,
    progress_every_batches: int = 100,
) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    def concat_or_empty(xs: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs) if xs else np.array([], dtype=np.float32)

    cand_pos_l2_all: List[np.ndarray] = []
    cand_neg_l2_all: List[np.ndarray] = []
    cand_pos_cos_all: List[np.ndarray] = []
    cand_neg_cos_all: List[np.ndarray] = []

    row_mean_pos_l2_all: List[np.ndarray] = []
    row_mean_neg_l2_all: List[np.ndarray] = []
    row_best_pos_l2_all: List[np.ndarray] = []
    row_hardest_neg_l2_all: List[np.ndarray] = []

    row_mean_pos_cos_all: List[np.ndarray] = []
    row_mean_neg_cos_all: List[np.ndarray] = []
    row_best_pos_cos_all: List[np.ndarray] = []
    row_hardest_neg_cos_all: List[np.ndarray] = []

    engine_buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    sample_store: Dict[str, Dict[str, Any]] = {}
    rng = np.random.default_rng(sampling_seed)

    start_t = time.time()
    total_batches: Optional[int] = None
    try:
        total_batches = len(loader)
    except Exception:
        total_batches = None
    print(
        f"[eval-progress] pair-metrics start: total_batches="
        f"{total_batches if total_batches is not None else 'unknown'} "
        f"max_batches={max_batches if max_batches is not None else 'none'}"
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            anchor = move_feature_dict_to_device(batch["anchor"], device)
            positive_all = move_feature_dict_to_device(batch["positive_all"], device)
            negative_all = move_feature_dict_to_device(batch["negative_all"], device)

            pos_counts = batch["pos_counts"].to(device)
            neg_counts = batch["neg_counts"].to(device)
            meta = batch["meta"]

            z_anchor = model(anchor)                 # [B, D]
            z_pos_all = model(positive_all)         # [sum_pos, D]
            z_neg_all = model(negative_all)         # [sum_neg, D]

            z_anchor_np = z_anchor.detach().cpu().numpy().astype(np.float32, copy=False)
            for i, m in enumerate(meta):
                sampled_meta = {
                    "player_id": m.get("anchor_player_id"),
                    "phase_id": m.get("anchor_phase_id"),
                    "game_type_id": m.get("anchor_game_type_id"),
                    "engine_rank": m.get("anchor_engine_rank"),
                    "engine_cp_gap": m.get("anchor_engine_cp_gap"),
                    "example_id": m.get("anchor_example_id"),
                    "source": "anchor_row_sample",
                    "row_idx": m.get("row_idx"),
                    "shard_id": m.get("shard_id"),
                }
                _reservoir_update_player_samples(
                    sample_store=sample_store,
                    player_id=sampled_meta["player_id"],
                    embedding=z_anchor_np[i],
                    meta=sampled_meta,
                    per_player_cap=sample_anchor_embeddings_per_player,
                    rng=rng,
                )

            pos_owner = torch.repeat_interleave(
                torch.arange(z_anchor.shape[0], device=device),
                pos_counts,
            )
            neg_owner = torch.repeat_interleave(
                torch.arange(z_anchor.shape[0], device=device),
                neg_counts,
            )

            pos_anchor = z_anchor[pos_owner]
            neg_anchor = z_anchor[neg_owner]

            pos_l2_flat = torch.linalg.norm(pos_anchor - z_pos_all, dim=-1).cpu().numpy()
            neg_l2_flat = torch.linalg.norm(neg_anchor - z_neg_all, dim=-1).cpu().numpy()
            pos_cos_flat = torch.sum(pos_anchor * z_pos_all, dim=-1).cpu().numpy()
            neg_cos_flat = torch.sum(neg_anchor * z_neg_all, dim=-1).cpu().numpy()

            cand_pos_l2_all.append(pos_l2_flat)
            cand_neg_l2_all.append(neg_l2_flat)
            cand_pos_cos_all.append(pos_cos_flat)
            cand_neg_cos_all.append(neg_cos_flat)

            pos_counts_np = batch["pos_counts"].cpu().numpy().astype(np.int64)
            neg_counts_np = batch["neg_counts"].cpu().numpy().astype(np.int64)
            pos_offsets = np.concatenate([[0], np.cumsum(pos_counts_np)])
            neg_offsets = np.concatenate([[0], np.cumsum(neg_counts_np)])

            row_mean_pos_l2 = np.zeros(len(meta), dtype=np.float32)
            row_mean_neg_l2 = np.zeros(len(meta), dtype=np.float32)
            row_best_pos_l2 = np.zeros(len(meta), dtype=np.float32)
            row_hardest_neg_l2 = np.zeros(len(meta), dtype=np.float32)

            row_mean_pos_cos = np.zeros(len(meta), dtype=np.float32)
            row_mean_neg_cos = np.zeros(len(meta), dtype=np.float32)
            row_best_pos_cos = np.zeros(len(meta), dtype=np.float32)
            row_hardest_neg_cos = np.zeros(len(meta), dtype=np.float32)

            for i, m in enumerate(meta):
                p0, p1 = int(pos_offsets[i]), int(pos_offsets[i + 1])
                n0, n1 = int(neg_offsets[i]), int(neg_offsets[i + 1])

                pos_l2_row = pos_l2_flat[p0:p1]
                neg_l2_row = neg_l2_flat[n0:n1]
                pos_cos_row = pos_cos_flat[p0:p1]
                neg_cos_row = neg_cos_flat[n0:n1]

                row_mean_pos_l2[i] = float(np.mean(pos_l2_row))
                row_mean_neg_l2[i] = float(np.mean(neg_l2_row))
                row_best_pos_l2[i] = float(np.min(pos_l2_row))
                row_hardest_neg_l2[i] = float(np.min(neg_l2_row))

                row_mean_pos_cos[i] = float(np.mean(pos_cos_row))
                row_mean_neg_cos[i] = float(np.mean(neg_cos_row))
                row_best_pos_cos[i] = float(np.max(pos_cos_row))
                row_hardest_neg_cos[i] = float(np.max(neg_cos_row))

                rank = m.get("anchor_engine_rank")
                if rank is None:
                    continue
                if rank <= 3:
                    bucket = "engine_like"
                elif rank > 10:
                    bucket = "engine_unlike"
                else:
                    bucket = "engine_middle"

                engine_buckets[bucket]["row_mean_pos_l2"].append(float(row_mean_pos_l2[i]))
                engine_buckets[bucket]["row_mean_neg_l2"].append(float(row_mean_neg_l2[i]))
                engine_buckets[bucket]["row_best_pos_l2"].append(float(row_best_pos_l2[i]))
                engine_buckets[bucket]["row_hardest_neg_l2"].append(float(row_hardest_neg_l2[i]))

                engine_buckets[bucket]["row_mean_pos_cos"].append(float(row_mean_pos_cos[i]))
                engine_buckets[bucket]["row_mean_neg_cos"].append(float(row_mean_neg_cos[i]))
                engine_buckets[bucket]["row_best_pos_cos"].append(float(row_best_pos_cos[i]))
                engine_buckets[bucket]["row_hardest_neg_cos"].append(float(row_hardest_neg_cos[i]))

            row_mean_pos_l2_all.append(row_mean_pos_l2)
            row_mean_neg_l2_all.append(row_mean_neg_l2)
            row_best_pos_l2_all.append(row_best_pos_l2)
            row_hardest_neg_l2_all.append(row_hardest_neg_l2)

            row_mean_pos_cos_all.append(row_mean_pos_cos)
            row_mean_neg_cos_all.append(row_mean_neg_cos)
            row_best_pos_cos_all.append(row_best_pos_cos)
            row_hardest_neg_cos_all.append(row_hardest_neg_cos)

            if progress_every_batches > 0 and ((batch_idx + 1) % progress_every_batches == 0):
                elapsed = max(1e-9, time.time() - start_t)
                done = batch_idx + 1
                bps = done / elapsed
                msg = (
                    f"[eval-progress] pair-metrics batches={done}"
                    f" elapsed_min={elapsed/60.0:.1f}"
                    f" batches_per_sec={bps:.2f}"
                )
                if total_batches is not None:
                    rem = max(total_batches - done, 0)
                    eta_sec = rem / max(bps, 1e-9)
                    msg += f" eta_min={eta_sec/60.0:.1f}"
                print(msg)

    elapsed_total = max(1e-9, time.time() - start_t)
    print(
        f"[eval-progress] pair-metrics done: batches={batch_idx + 1 if 'batch_idx' in locals() else 0} "
        f"elapsed_min={elapsed_total/60.0:.1f}"
    )

    cand_pos_l2 = concat_or_empty(cand_pos_l2_all)
    cand_neg_l2 = concat_or_empty(cand_neg_l2_all)
    cand_pos_cos = concat_or_empty(cand_pos_cos_all)
    cand_neg_cos = concat_or_empty(cand_neg_cos_all)

    row_mean_pos_l2 = concat_or_empty(row_mean_pos_l2_all)
    row_mean_neg_l2 = concat_or_empty(row_mean_neg_l2_all)
    row_best_pos_l2 = concat_or_empty(row_best_pos_l2_all)
    row_hardest_neg_l2 = concat_or_empty(row_hardest_neg_l2_all)

    row_mean_pos_cos = concat_or_empty(row_mean_pos_cos_all)
    row_mean_neg_cos = concat_or_empty(row_mean_neg_cos_all)
    row_best_pos_cos = concat_or_empty(row_best_pos_cos_all)
    row_hardest_neg_cos = concat_or_empty(row_hardest_neg_cos_all)

    out: Dict[str, Any] = {
        "n_anchor_rows": int(len(row_mean_pos_l2)),
        "n_pos_candidates": int(len(cand_pos_l2)),
        "n_neg_candidates": int(len(cand_neg_l2)),
        "candidate_level": {
            "l2": {
                "positive": percentiles_dict(cand_pos_l2, "pos_l2"),
                "negative": percentiles_dict(cand_neg_l2, "neg_l2"),
                "gap_mean": float(np.mean(cand_neg_l2) - np.mean(cand_pos_l2)) if len(cand_pos_l2) else math.nan,
            },
            "cosine": {
                "positive": percentiles_dict(cand_pos_cos, "pos_cos"),
                "negative": percentiles_dict(cand_neg_cos, "neg_cos"),
                "gap_mean": float(np.mean(cand_pos_cos) - np.mean(cand_neg_cos)) if len(cand_pos_cos) else math.nan,
            },
        },
        "row_aggregated": {
            "l2": {
                "mean_positive": percentiles_dict(row_mean_pos_l2, "row_mean_pos_l2"),
                "mean_negative": percentiles_dict(row_mean_neg_l2, "row_mean_neg_l2"),
                "best_positive": percentiles_dict(row_best_pos_l2, "row_best_pos_l2"),
                "hardest_negative": percentiles_dict(row_hardest_neg_l2, "row_hardest_neg_l2"),
                "mean_gap": float(np.mean(row_mean_neg_l2) - np.mean(row_mean_pos_l2)) if len(row_mean_pos_l2) else math.nan,
                "hard_gap": float(np.mean(row_hardest_neg_l2) - np.mean(row_mean_pos_l2)) if len(row_mean_pos_l2) else math.nan,
            },
            "cosine": {
                "mean_positive": percentiles_dict(row_mean_pos_cos, "row_mean_pos_cos"),
                "mean_negative": percentiles_dict(row_mean_neg_cos, "row_mean_neg_cos"),
                "best_positive": percentiles_dict(row_best_pos_cos, "row_best_pos_cos"),
                "hardest_negative": percentiles_dict(row_hardest_neg_cos, "row_hardest_neg_cos"),
                "mean_gap": float(np.mean(row_mean_pos_cos) - np.mean(row_mean_neg_cos)) if len(row_mean_pos_cos) else math.nan,
                "hard_gap": float(np.mean(row_mean_pos_cos) - np.mean(row_hardest_neg_cos)) if len(row_mean_pos_cos) else math.nan,
                "pair_acc_mean_vs_hardest": float(np.mean(row_mean_pos_cos > row_hardest_neg_cos)) if len(row_mean_pos_cos) else math.nan,
            },
            "by_eval_tau": {},
        },
        "engine_conditioned": {},
    }

    for tau in eval_taus:
        row_mean_pos_scaled = row_mean_pos_cos / tau
        row_mean_neg_scaled = row_mean_neg_cos / tau
        row_hard_neg_scaled = row_hardest_neg_cos / tau

        row_mean_pos_exp = np.exp(np.clip(row_mean_pos_scaled, -60.0, 60.0))
        row_mean_neg_exp = np.exp(np.clip(row_mean_neg_scaled, -60.0, 60.0))
        row_hard_neg_exp = np.exp(np.clip(row_hard_neg_scaled, -60.0, 60.0))

        logits = np.stack([row_mean_pos_scaled, row_hard_neg_scaled], axis=1)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        loss = -np.log(np.clip(probs[:, 0], 1e-12, None))

        out["row_aggregated"]["by_eval_tau"][str(tau)] = {
            "dot_over_tau": {
                "mean_positive": percentiles_dict(row_mean_pos_scaled, "row_mean_pos_dot_over_tau"),
                "mean_negative": percentiles_dict(row_mean_neg_scaled, "row_mean_neg_dot_over_tau"),
                "hardest_negative": percentiles_dict(row_hard_neg_scaled, "row_hard_neg_dot_over_tau"),
                "mean_gap": float(np.mean(row_mean_pos_scaled) - np.mean(row_mean_neg_scaled)) if len(row_mean_pos_scaled) else math.nan,
                "hard_gap": float(np.mean(row_mean_pos_scaled) - np.mean(row_hard_neg_scaled)) if len(row_mean_pos_scaled) else math.nan,
            },
            "exp_dot_over_tau": {
                "mean_positive": percentiles_dict(row_mean_pos_exp, "row_mean_pos_exp_dot_over_tau"),
                "mean_negative": percentiles_dict(row_mean_neg_exp, "row_mean_neg_exp_dot_over_tau"),
                "hardest_negative": percentiles_dict(row_hard_neg_exp, "row_hard_neg_exp_dot_over_tau"),
                "mean_gap": float(np.mean(row_mean_pos_exp) - np.mean(row_mean_neg_exp)) if len(row_mean_pos_exp) else math.nan,
                "hard_gap": float(np.mean(row_mean_pos_exp) - np.mean(row_hard_neg_exp)) if len(row_mean_pos_exp) else math.nan,
            },
            "infonce_like_loss_mean_vs_hardest_neg": float(np.mean(loss)) if len(loss) else math.nan,
        }

    for bucket, vals in engine_buckets.items():
        pos_cos_b = np.asarray(vals.get("row_mean_pos_cos", []), dtype=np.float32)
        neg_cos_b = np.asarray(vals.get("row_mean_neg_cos", []), dtype=np.float32)
        hard_neg_cos_b = np.asarray(vals.get("row_hardest_neg_cos", []), dtype=np.float32)

        pos_l2_b = np.asarray(vals.get("row_mean_pos_l2", []), dtype=np.float32)
        neg_l2_b = np.asarray(vals.get("row_mean_neg_l2", []), dtype=np.float32)
        hard_neg_l2_b = np.asarray(vals.get("row_hardest_neg_l2", []), dtype=np.float32)

        out["engine_conditioned"][bucket] = {
            "n_anchor_rows": int(len(pos_cos_b)),
            "l2_mean_gap": float(np.mean(neg_l2_b) - np.mean(pos_l2_b)) if len(pos_l2_b) else math.nan,
            "l2_hard_gap": float(np.mean(hard_neg_l2_b) - np.mean(pos_l2_b)) if len(pos_l2_b) else math.nan,
            "cos_mean_gap": float(np.mean(pos_cos_b) - np.mean(neg_cos_b)) if len(pos_cos_b) else math.nan,
            "cos_hard_gap": float(np.mean(pos_cos_b) - np.mean(hard_neg_cos_b)) if len(pos_cos_b) else math.nan,
        }
        for tau in eval_taus:
            out["engine_conditioned"][bucket][f"dot_over_tau_mean_gap@{tau}"] = (
                float(np.mean(pos_cos_b / tau) - np.mean(neg_cos_b / tau)) if len(pos_cos_b) else math.nan
            )
            out["engine_conditioned"][bucket][f"dot_over_tau_hard_gap@{tau}"] = (
                float(np.mean(pos_cos_b / tau) - np.mean(hard_neg_cos_b / tau)) if len(pos_cos_b) else math.nan
            )

    sampled_embeddings, sampled_meta_rows, sample_summary = finalize_sampled_anchor_embeddings(
        sample_store,
        max_players=sampled_embedding_max_players,
        min_examples_per_player=sampled_embedding_min_examples_per_player,
        player_selection=sampled_embedding_player_selection,
        seed=sampling_seed,
    )
    rng_diag = np.random.default_rng(42)
    max_diag = 300_000
    def _subsample(arr: np.ndarray) -> np.ndarray:
        if len(arr) <= max_diag:
            return arr
        idx = rng_diag.choice(len(arr), size=max_diag, replace=False)
        return arr[idx]

    pos_c = _subsample(cand_pos_cos)
    neg_c = _subsample(cand_neg_cos)
    hard_n = _subsample(row_hardest_neg_cos)
    soft_n = _subsample(row_mean_neg_cos)
    pos_r = _subsample(row_mean_pos_cos)

    score_distributions = {
        "positive_candidate_cosine": _distribution_summary(pos_c),
        "negative_candidate_cosine": _distribution_summary(neg_c),
        "positive_row_mean_cosine": _distribution_summary(pos_r),
        "hard_negative_row_cosine": _distribution_summary(hard_n),
        "soft_negative_row_cosine": _distribution_summary(soft_n),
    }
    pairwise_auc = {
        "positive_vs_all_negative": _pair_auc_ap(
            np.concatenate([np.ones(len(pos_c), dtype=np.int64), np.zeros(len(neg_c), dtype=np.int64)]),
            np.concatenate([pos_c, neg_c]),
        )
        if len(pos_c) and len(neg_c)
        else {"n_pairs": 0, "roc_auc": math.nan, "average_precision": math.nan},
        "positive_vs_hard_negative": _pair_auc_ap(
            np.concatenate([np.ones(len(pos_r), dtype=np.int64), np.zeros(len(hard_n), dtype=np.int64)]),
            np.concatenate([pos_r, hard_n]),
        )
        if len(pos_r) and len(hard_n)
        else {"n_pairs": 0, "roc_auc": math.nan, "average_precision": math.nan},
        "positive_vs_soft_negative": _pair_auc_ap(
            np.concatenate([np.ones(len(pos_r), dtype=np.int64), np.zeros(len(soft_n), dtype=np.int64)]),
            np.concatenate([pos_r, soft_n]),
        )
        if len(pos_r) and len(soft_n)
        else {"n_pairs": 0, "roc_auc": math.nan, "average_precision": math.nan},
    }
    extra_pair_diagnostics = {
        "score_distributions": score_distributions,
        "pairwise_auc": pairwise_auc,
    }
    return out, sampled_embeddings, sampled_meta_rows, sample_summary, extra_pair_diagnostics


def encode_example_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    embeddings: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            feats = move_to_device(batch["features"], device)
            z = model(feats).cpu().numpy().astype(np.float32)
            embeddings.append(z)
            meta_rows.extend(batch["meta"])

    if not embeddings:
        return np.zeros((0, 1), dtype=np.float32), []
    emb = np.concatenate(embeddings, axis=0)
    return normalize_rows(emb), meta_rows


def compute_retrieval_metrics(embeddings: np.ndarray, meta_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    player_ids = np.asarray([m.get("player_id") for m in meta_rows], dtype=object)
    valid = np.asarray([pid is not None for pid in player_ids])
    if not valid.any():
        return {"skipped": True, "reason": "player_ids_not_available"}

    emb = embeddings[valid]
    pids = player_ids[valid]
    sims = pairwise_cosine(emb, emb)
    np.fill_diagonal(sims, -np.inf)

    recall1 = 0.0
    recall5 = 0.0
    recall10 = 0.0
    recall20 = 0.0
    mrr = 0.0
    ranks: List[int] = []
    positive_recall_sum = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
    positive_counts: List[int] = []
    n = len(pids)
    for i in range(n):
        order = np.argsort(-sims[i])
        ranked_pids = pids[order]
        matches = np.where(ranked_pids == pids[i])[0]
        if len(matches) == 0:
            continue
        pos_total = int(len(matches))
        positive_counts.append(pos_total)
        best_rank = int(matches[0]) + 1
        ranks.append(best_rank)
        recall1 += float(best_rank <= 1)
        recall5 += float(best_rank <= 5)
        recall10 += float(best_rank <= 10)
        recall20 += float(best_rank <= 20)
        mrr += 1.0 / best_rank
        for k in positive_recall_sum:
            in_top_k = int(np.sum(matches < k))
            positive_recall_sum[k] += float(in_top_k / max(pos_total, 1))

    rank_arr = np.asarray(ranks, dtype=np.float32) if ranks else np.asarray([], dtype=np.float32)
    ks = [1, 3, 5, 10, 20, 50]
    recall_at_k = {f"recall_at_{k}": (float(np.mean(rank_arr <= k)) if len(rank_arr) else math.nan) for k in ks}
    positive_recall_at_k = {
        f"positive_recall_at_{k}": (float(positive_recall_sum[k] / len(ranks)) if len(ranks) else math.nan)
        for k in ks
    }
    positive_count_arr = np.asarray(positive_counts, dtype=np.float32) if positive_counts else np.asarray([], dtype=np.float32)

    return {
        "skipped": False,
        "n_examples": int(n),
        "recall_at_1": float(recall1 / n) if n else math.nan,
        "recall_at_5": float(recall5 / n) if n else math.nan,
        "recall_at_10": float(recall10 / n) if n else math.nan,
        "recall_at_20": float(recall20 / n) if n else math.nan,
        "mrr": float(mrr / n) if n else math.nan,
        "rank_mean": float(np.mean(rank_arr)) if len(rank_arr) else math.nan,
        "rank_median": float(np.median(rank_arr)) if len(rank_arr) else math.nan,
        "rank_p90": float(np.percentile(rank_arr, 90)) if len(rank_arr) else math.nan,
        "rank_p95": float(np.percentile(rank_arr, 95)) if len(rank_arr) else math.nan,
        "mean_positive_count_per_query": float(np.mean(positive_count_arr)) if len(positive_count_arr) else math.nan,
        "median_positive_count_per_query": float(np.median(positive_count_arr)) if len(positive_count_arr) else math.nan,
        "rank_values": [int(x) for x in ranks],
        "recall_curve": recall_at_k,
        "positive_recall_curve": positive_recall_at_k,
    }


def compute_binary_classification_metrics(embeddings: np.ndarray, meta_rows: List[Dict[str, Any]], sample_limit: int = 200_000) -> Dict[str, Any]:
    player_ids = np.asarray([m.get("player_id") for m in meta_rows], dtype=object)
    valid_idx = np.where(np.asarray([pid is not None for pid in player_ids]))[0]
    if len(valid_idx) < 2:
        return {"skipped": True, "reason": "player_ids_not_available"}

    rng = np.random.default_rng(42)
    emb = embeddings[valid_idx]
    pids = player_ids[valid_idx]
    n = len(pids)
    pair_count = min(sample_limit, n * (n - 1) // 2)
    if pair_count <= 0:
        return {"skipped": True, "reason": "not_enough_examples"}

    i_idx = rng.integers(0, n, size=pair_count)
    j_idx = rng.integers(0, n, size=pair_count)
    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]

    y_true = (pids[i_idx] == pids[j_idx]).astype(np.int64)
    scores = np.sum(emb[i_idx] * emb[j_idx], axis=1)

    roc_auc = safe_auc_from_scores(y_true, scores)
    ap = average_precision(y_true, scores)

    thresholds = np.linspace(-1.0, 1.0, 401)
    best_f1 = -1.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    for thr in thresholds:
        pred = (scores >= thr).astype(np.int64)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
            best_precision = float(precision)
            best_recall = float(recall)

    return {
        "skipped": False,
        "n_sampled_pairs": int(len(scores)),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "best_f1": float(best_f1),
        "best_threshold": best_threshold,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "positive_pair_rate": float(np.mean(y_true)),
    }


def _distribution_summary(values: np.ndarray, *, bins: int = 80, lo: float = -1.0, hi: float = 1.0) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "p10": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p90": math.nan,
            "hist": {"edges": [], "counts": []},
        }
    hist_counts, hist_edges = np.histogram(arr, bins=bins, range=(lo, hi))
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "hist": {
            "edges": [float(x) for x in hist_edges.tolist()],
            "counts": [int(x) for x in hist_counts.tolist()],
        },
    }


def _pair_auc_ap(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]
    if len(y) == 0:
        return {"n_pairs": 0, "roc_auc": math.nan, "average_precision": math.nan}
    return {
        "n_pairs": int(len(y)),
        "roc_auc": safe_auc_from_scores(y, s),
        "average_precision": average_precision(y, s),
    }


def _threshold_table(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> List[Dict[str, float]]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    out: List[Dict[str, float]] = []
    for thr in thresholds:
        pred = (s >= thr).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        tpr = recall
        fpr = fp / max(fp + tn, 1)
        specificity = tn / max(tn + fp, 1)
        out.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "specificity": float(specificity),
                "tp": float(tp),
                "fp": float(fp),
                "tn": float(tn),
                "fn": float(fn),
            }
        )
    return out


def _calibration_from_scores(y_true: np.ndarray, scores: np.ndarray, num_bins: int = 12) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]
    if len(y) == 0:
        return {"n_pairs": 0, "ece": math.nan, "bins": []}
    # Dot/cosine in [-1, 1] -> pseudo probability.
    p = np.clip((s + 1.0) * 0.5, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: List[Dict[str, Any]] = []
    ece = 0.0
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == num_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n == 0:
            bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0, "mean_conf": math.nan, "empirical_acc": math.nan})
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        w = n / max(len(y), 1)
        ece += w * abs(acc - conf)
        bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": n, "mean_conf": conf, "empirical_acc": acc})
    return {"n_pairs": int(len(y)), "ece": float(ece), "bins": bins}


def _bootstrap_ci(values: np.ndarray, fn, *, n_boot: int = 300, seed: int = 42) -> Dict[str, float]:
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    point = float(fn(arr))
    if len(arr) == 1:
        return {"mean": point, "ci95_lo": point, "ci95_hi": point, "n": 1}
    boots = np.zeros(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(fn(arr[idx]))
    return {
        "mean": point,
        "ci95_lo": float(np.percentile(boots, 2.5)),
        "ci95_hi": float(np.percentile(boots, 97.5)),
        "n": int(n),
    }


def _bootstrap_auc(y_true: np.ndarray, scores: np.ndarray, *, n_boot: int = 300, seed: int = 42) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]
    if len(y) == 0:
        return {"mean": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan, "n": 0}
    point = safe_auc_from_scores(y, s)
    if point is None:
        return {"mean": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan, "n": int(len(y))}
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        auc = safe_auc_from_scores(y[idx], s[idx])
        if auc is not None and np.isfinite(auc):
            boots.append(float(auc))
    if not boots:
        return {"mean": float(point), "ci95_lo": math.nan, "ci95_hi": math.nan, "n": int(n)}
    b = np.asarray(boots, dtype=np.float64)
    return {
        "mean": float(point),
        "ci95_lo": float(np.percentile(b, 2.5)),
        "ci95_hi": float(np.percentile(b, 97.5)),
        "n": int(n),
    }


def build_extended_diagnostics(
    *,
    embeddings: np.ndarray,
    meta_rows: List[Dict[str, Any]],
    retrieval: Dict[str, Any],
    classification: Dict[str, Any],
    pair_metrics: Dict[str, Any],
    score_distributions: Dict[str, Any],
    pairwise_auc: Dict[str, Any],
    sample_limit: int = 250_000,
) -> Dict[str, Any]:
    player_ids = np.asarray([m.get("player_id") for m in meta_rows], dtype=object)
    phase_ids = np.asarray([m.get("phase_id") for m in meta_rows], dtype=object)
    engine_ranks = np.asarray([m.get("engine_rank") for m in meta_rows], dtype=object)
    game_type_ids = np.asarray([m.get("game_type_id") for m in meta_rows], dtype=object)

    valid_idx = np.where(np.asarray([pid is not None for pid in player_ids]))[0]
    emb = embeddings[valid_idx]
    pids = player_ids[valid_idx]
    phs = phase_ids[valid_idx]
    eng = engine_ranks[valid_idx]
    gts = game_type_ids[valid_idx]

    rng = np.random.default_rng(42)
    n = len(pids)
    pair_count = min(sample_limit, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=pair_count)
    j_idx = rng.integers(0, n, size=pair_count)
    keep = i_idx != j_idx
    i_idx = i_idx[keep]
    j_idx = j_idx[keep]
    y = (pids[i_idx] == pids[j_idx]).astype(np.int64)
    s = np.sum(emb[i_idx] * emb[j_idx], axis=1)
    same_phase = (phs[i_idx] == phs[j_idx]) & np.asarray([x is not None for x in phs[i_idx]]) & np.asarray([x is not None for x in phs[j_idx]])

    thresholds = np.linspace(-1.0, 1.0, 201, dtype=np.float64)
    masks = {
        "overall": np.ones(len(y), dtype=bool),
        "same_player_same_phase_vs_diff_player_diff_phase": (y == 1) & same_phase | ((y == 0) & (~same_phase)),
        "hard_negative_vs_positive": (y == 1) | ((y == 0) & same_phase),
        "soft_negative_vs_positive": (y == 1) | ((y == 0) & (~same_phase)),
    }

    threshold_sweep: Dict[str, Any] = {}
    calibration: Dict[str, Any] = {}
    for name, m in masks.items():
        yy = y[m]
        ss = s[m]
        threshold_sweep[name] = {
            "n_pairs": int(len(ss)),
            "summary": _pair_auc_ap(yy, ss),
            "points": _threshold_table(yy, ss, thresholds),
        }
        calibration[name] = _calibration_from_scores(yy, ss)

    # Retrieval@K curve and CI from rank values.
    ranks = np.asarray(retrieval.get("rank_values", []), dtype=np.float64)
    ks = [1, 3, 5, 10, 20, 50, 100]
    retrieval_at_k = {
        "n_ranks": int(len(ranks)),
        "recall_at_k": {f"k{k}": (float(np.mean(ranks <= k)) if len(ranks) else math.nan) for k in ks},
        "positive_recall_at_k": retrieval.get("positive_recall_curve", {}),
        "rank_mean": float(np.mean(ranks)) if len(ranks) else math.nan,
        "rank_median": float(np.median(ranks)) if len(ranks) else math.nan,
        "rank_p90": float(np.percentile(ranks, 90)) if len(ranks) else math.nan,
        "rank_p95": float(np.percentile(ranks, 95)) if len(ranks) else math.nan,
    }

    # Conditioned metrics: phase, game_type, engine bucket.
    def _condition_group(name: str, mask: np.ndarray) -> Dict[str, Any]:
        yy = y[mask]
        ss = s[mask]
        return {
            "name": name,
            "n_pairs": int(len(ss)),
            **_pair_auc_ap(yy, ss),
            "score_distribution": _distribution_summary(ss, bins=60),
        }

    conditioned: Dict[str, Any] = {"phase": [], "game_type": [], "engine_rank_bucket": []}
    phase_vals = sorted({int(x) for x in phs if x is not None})
    for ph in phase_vals:
        mask = np.asarray([(phs[i] == ph and phs[j] == ph) for i, j in zip(i_idx, j_idx)], dtype=bool)
        conditioned["phase"].append(_condition_group(f"phase_{ph}", mask))
    gt_vals = sorted({int(x) for x in gts if x is not None})
    for gt in gt_vals:
        mask = np.asarray([(gts[i] == gt and gts[j] == gt) for i, j in zip(i_idx, j_idx)], dtype=bool)
        conditioned["game_type"].append(_condition_group(f"game_type_{gt}", mask))

    def _eng_bucket(v: Any) -> str:
        if v is None:
            return "unknown"
        vv = int(v)
        if vv <= 3:
            return "engine_like"
        if vv > 10:
            return "engine_unlike"
        return "engine_middle"

    buckets = ["engine_like", "engine_middle", "engine_unlike", "unknown"]
    for b in buckets:
        mask = np.asarray([(_eng_bucket(eng[i]) == b and _eng_bucket(eng[j]) == b) for i, j in zip(i_idx, j_idx)], dtype=bool)
        conditioned["engine_rank_bucket"].append(_condition_group(b, mask))

    # Bootstrap CIs.
    bootstrap = {
        "mrr": _bootstrap_ci(ranks, lambda x: np.mean(1.0 / np.clip(x, 1, None))) if len(ranks) else {"mean": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan, "n": 0},
        "recall_at_1": _bootstrap_ci(ranks, lambda x: np.mean(x <= 1)) if len(ranks) else {"mean": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan, "n": 0},
        "recall_at_5": _bootstrap_ci(ranks, lambda x: np.mean(x <= 5)) if len(ranks) else {"mean": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan, "n": 0},
        "classification_roc_auc": _bootstrap_auc(y, s) if len(s) else {"mean": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan, "n": 0},
        "row_cos_hard_gap": {
            "mean": _safe_float((((pair_metrics.get("row_aggregated") or {}).get("cosine") or {}).get("hard_gap"))),
            "ci95_lo": math.nan,
            "ci95_hi": math.nan,
            "n": int((pair_metrics.get("n_anchor_rows") or 0)),
        },
    }

    # Top confusions via player centroids.
    by_player: Dict[str, List[np.ndarray]] = defaultdict(list)
    for z, pid in zip(emb, pids):
        if pid is not None:
            by_player[str(pid)].append(z)
    confusions: List[Dict[str, Any]] = []
    if len(by_player) >= 2:
        pnames = sorted(by_player.keys())
        cents = []
        counts = []
        for p in pnames:
            mat = np.stack(by_player[p], axis=0)
            c = normalize_rows(mat.mean(axis=0, keepdims=True))[0]
            cents.append(c)
            counts.append(len(mat))
        cm = np.stack(cents, axis=0)
        sim = cm @ cm.T
        for i in range(len(pnames)):
            for j in range(i + 1, len(pnames)):
                confusions.append(
                    {
                        "player_a": pnames[i],
                        "player_b": pnames[j],
                        "cosine": float(sim[i, j]),
                        "n_a": int(counts[i]),
                        "n_b": int(counts[j]),
                    }
                )
        confusions.sort(key=lambda r: r["cosine"], reverse=True)
        confusions = confusions[:100]

    return {
        "threshold_sweep": threshold_sweep,
        "retrieval_at_k": retrieval_at_k,
        "score_distributions": score_distributions,
        "pairwise_auc": pairwise_auc,
        "conditioned_metrics": conditioned,
        "bootstrap_ci": bootstrap,
        "calibration": calibration,
        "top_confusions": {"pairs": confusions, "n_players": int(len(by_player))},
    }


def compute_spread_metrics(embeddings: np.ndarray, meta_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    player_ids = [m.get("player_id") for m in meta_rows]
    by_player: Dict[Any, List[np.ndarray]] = defaultdict(list)
    for z, pid in zip(embeddings, player_ids):
        if pid is not None:
            by_player[pid].append(z)

    if not by_player:
        return {"skipped": True, "reason": "player_ids_not_available"}

    centroids: Dict[Any, np.ndarray] = {}
    intra_spreads: Dict[Any, float] = {}
    for pid, rows in by_player.items():
        mat = np.stack(rows, axis=0)
        centroid = normalize_rows(mat.mean(axis=0, keepdims=True))[0]
        centroids[pid] = centroid
        dists = np.linalg.norm(mat - centroid[None, :], axis=1)
        intra_spreads[pid] = float(np.mean(dists))

    centroid_keys = list(centroids.keys())
    centroid_mat = np.stack([centroids[k] for k in centroid_keys], axis=0)
    centroid_sims = pairwise_cosine(centroid_mat, centroid_mat)
    centroid_dists = 1.0 - centroid_sims
    iu = np.triu_indices_from(centroid_dists, k=1)
    inter = centroid_dists[iu]
    intra = np.asarray(list(intra_spreads.values()), dtype=np.float32)

    return {
        "skipped": False,
        "n_players": int(len(centroid_keys)),
        "intra_player_spread": percentiles_dict(intra, "intra_player_spread"),
        "inter_player_centroid_distance": percentiles_dict(inter.astype(np.float32), "inter_player_centroid_distance"),
        "spread_ratio_mean": float(np.mean(inter) / max(np.mean(intra), 1e-12)) if len(inter) and len(intra) else math.nan,
        "players": {
            str(pid): {
                "n_examples": len(by_player[pid]),
                "intra_spread": intra_spreads[pid],
            }
            for pid in centroid_keys
        },
    }


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _required_shard_files() -> Tuple[str, ...]:
    return (
        "meta.json",
        "examples_board_tokens.uint8.npy",
        "examples_moves.uint8.npy",
        "examples_game_type.uint8.npy",
        "pair_anchor_idx.int32.npy",
        "pair_pos_flat.int32.npy",
        "pair_pos_offsets.int64.npy",
        "pair_neg_flat.int32.npy",
        "pair_neg_offsets.int64.npy",
    )


def validate_cached_split_dir(cached_split_dir: Path) -> Tuple[bool, str]:
    """
    A cache is reusable only if build summary exists and shard layout is complete.
    This guards against partial cache writes after interrupted runs.
    """
    if not cached_split_dir.exists() or not cached_split_dir.is_dir():
        return False, "missing_cache_dir"

    summary_path = cached_split_dir / "_build_summary.json"
    summary = read_json_if_exists(summary_path)
    if summary is None:
        return False, "missing_or_invalid_build_summary"

    num_shards = summary.get("num_shards")
    try:
        num_shards_int = int(num_shards)
    except Exception:
        return False, "invalid_num_shards_in_summary"

    shard_dirs = sorted(p for p in cached_split_dir.glob("shard_*") if p.is_dir())
    if len(shard_dirs) != num_shards_int:
        return False, f"shard_count_mismatch(expected={num_shards_int},found={len(shard_dirs)})"

    expected_names = {f"shard_{i:06d}" for i in range(num_shards_int)}
    actual_names = {p.name for p in shard_dirs}
    if actual_names != expected_names:
        return False, "non_contiguous_or_stale_shard_dirs"

    req = _required_shard_files()
    for shard_dir in shard_dirs:
        for rel in req:
            path = shard_dir / rel
            if not path.exists() or not path.is_file():
                return False, f"missing_required_file:{path.name}"

    return True, "ok"


def is_prebuilt_cached_split_dir(path: Path) -> Tuple[bool, str]:
    """
    Detect whether `path` already contains shard_* cache layout.
    Useful when input pairs are pre-cached (e.g. pairs_v3_cached/*) rather than JSONL.
    """
    if not path.exists() or not path.is_dir():
        return False, "missing_dir"
    shard_dirs = sorted(p for p in path.glob("shard_*") if p.is_dir())
    if not shard_dirs:
        return False, "no_shard_dirs"
    req = _required_shard_files()
    for shard_dir in shard_dirs:
        for rel in req:
            f = shard_dir / rel
            if not f.exists() or not f.is_file():
                return False, f"missing_required_file:{shard_dir.name}/{rel}"
    return True, "ok"


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def build_pair_score_components(pair_metrics: Dict[str, Any]) -> Dict[str, Any]:
    row_cos = (((pair_metrics.get("row_aggregated") or {}).get("cosine")) or {})
    out = {
        "pos_mean_cos": _safe_float((((row_cos.get("mean_positive") or {}).get("row_mean_pos_cos_mean")))),
        "hardneg_mean_cos": _safe_float((((row_cos.get("hardest_negative") or {}).get("row_hardest_neg_cos_mean")))),
        "meanneg_mean_cos": _safe_float((((row_cos.get("mean_negative") or {}).get("row_mean_neg_cos_mean")))),
        "bestpos_mean_cos": _safe_float((((row_cos.get("best_positive") or {}).get("row_best_pos_cos_mean")))),
        "hard_gap_cos": _safe_float(row_cos.get("hard_gap")),
        "mean_gap_cos": _safe_float(row_cos.get("mean_gap")),
        "pair_acc_hardest": _safe_float(row_cos.get("pair_acc_mean_vs_hardest")),
    }
    return out


def flatten_metrics_for_split(
    *,
    retrieval: Dict[str, Any],
    classification: Dict[str, Any],
    spread: Dict[str, Any],
    pair_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    row_cos = (((pair_metrics.get("row_aggregated") or {}).get("cosine")) or {})
    flat: Dict[str, Any] = {
        "mrr": _safe_float(retrieval.get("mrr")),
        "recall_at_1": _safe_float(retrieval.get("recall_at_1")),
        "recall_at_5": _safe_float(retrieval.get("recall_at_5")),
        "recall_at_10": _safe_float(retrieval.get("recall_at_10")),
        "recall_at_20": _safe_float(retrieval.get("recall_at_20")),
        "positive_recall_at_1": _safe_float(retrieval.get("positive_recall_at_1")),
        "positive_recall_at_5": _safe_float(retrieval.get("positive_recall_at_5")),
        "positive_recall_at_10": _safe_float(retrieval.get("positive_recall_at_10")),
        "positive_recall_at_20": _safe_float(retrieval.get("positive_recall_at_20")),
        "mean_positive_count_per_query": _safe_float(retrieval.get("mean_positive_count_per_query")),
        "rank_mean": _safe_float(retrieval.get("rank_mean")),
        "rank_median": _safe_float(retrieval.get("rank_median")),
        "rank_p90": _safe_float(retrieval.get("rank_p90")),
        "rank_p95": _safe_float(retrieval.get("rank_p95")),
        "classification_roc_auc": _safe_float(classification.get("roc_auc")),
        "classification_ap": _safe_float(classification.get("average_precision")),
        "classification_best_f1": _safe_float(classification.get("best_f1")),
        "classification_best_threshold": _safe_float(classification.get("best_threshold")),
        "classification_positive_pair_rate": _safe_float(classification.get("positive_pair_rate")),
        "spread_ratio": _safe_float(spread.get("spread_ratio_mean")),
        "row_cos_hard_gap": _safe_float(row_cos.get("hard_gap")),
        "row_cos_mean_gap": _safe_float(row_cos.get("mean_gap")),
        "pair_acc_hardest": _safe_float(row_cos.get("pair_acc_mean_vs_hardest")),
        "pos_mean_cos": _safe_float((((row_cos.get("mean_positive") or {}).get("row_mean_pos_cos_mean")))),
        "hardneg_mean_cos": _safe_float((((row_cos.get("hardest_negative") or {}).get("row_hardest_neg_cos_mean")))),
        "meanneg_mean_cos": _safe_float((((row_cos.get("mean_negative") or {}).get("row_mean_neg_cos_mean")))),
    }
    recall_curve = retrieval.get("recall_curve")
    if isinstance(recall_curve, dict):
        for k, v in recall_curve.items():
            flat[str(k)] = _safe_float(v)
    pos_curve = retrieval.get("positive_recall_curve")
    if isinstance(pos_curve, dict):
        for k, v in pos_curve.items():
            flat[str(k)] = _safe_float(v)
    return flat



def run_eval_for_split(
    split: str,
    cached_split_dir: Path,
    output_root: Path,
    model: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    eval_taus: Sequence[float],
    batch_size_override: Optional[int],
    num_workers: int,
    max_pair_batches: Optional[int],
    max_embed_batches: Optional[int],
    max_examples: Optional[int],
    save_embeddings: bool,
    *,
    sampled_embedding_max_players: Optional[int],
    sampled_embedding_max_examples_per_player: int,
    sampled_embedding_min_examples_per_player: int,
    sampled_embedding_player_selection: str,
    sampled_embedding_seed: int,
    progress_every_batches: int,
    use_embedding_cache: bool,
    embedding_cache_batch_size: int,
    model_signature: str,
) -> Dict[str, Any]:
    print(f"[eval] split={split} dir={cached_split_dir}")
    split_out = output_root / split
    split_out.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size_override or cfg.batch_size

    if use_embedding_cache:
        pair_metrics, embeddings, meta_rows, sample_summary, extra_pair_diag = compute_pair_metrics_with_embedding_cache(
            model=model,
            device=device,
            model_variant_name=cfg.model.variant_name,
            cached_split_dir=cached_split_dir,
            eval_taus=eval_taus,
            batch_size=max(1, int(embedding_cache_batch_size)),
            max_batches=max_pair_batches,
            progress_every_batches=progress_every_batches,
            sample_anchor_embeddings_per_player=sampled_embedding_max_examples_per_player,
            sampled_embedding_max_players=sampled_embedding_max_players,
            sampled_embedding_min_examples_per_player=sampled_embedding_min_examples_per_player,
            sampled_embedding_player_selection=sampled_embedding_player_selection,
            sampling_seed=sampled_embedding_seed,
            embedding_cache_dir=split_out / "embedding_cache",
            model_signature=model_signature,
        )
    else:
        pair_ds = PairEvalDataset(
            input_dir=cached_split_dir,
            model_variant_name=cfg.model.variant_name,
        )

        pair_loader = DataLoader(
            pair_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_pair_eval,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
            pin_memory=False,
        )
        pair_metrics, embeddings, meta_rows, sample_summary, extra_pair_diag = compute_pair_metrics(
            model,
            pair_loader,
            device,
            eval_taus,
            max_batches=max_pair_batches,
            sample_anchor_embeddings_per_player=sampled_embedding_max_examples_per_player,
            sampled_embedding_max_players=sampled_embedding_max_players,
            sampled_embedding_min_examples_per_player=sampled_embedding_min_examples_per_player,
            sampled_embedding_player_selection=sampled_embedding_player_selection,
            sampling_seed=sampled_embedding_seed,
            progress_every_batches=progress_every_batches,
        )
    write_json(split_out / "pair_metrics.json", pair_metrics)
    write_json(split_out / "sampled_embedding_summary.json", sample_summary)

    retrieval = compute_retrieval_metrics(embeddings, meta_rows)
    classification = compute_binary_classification_metrics(embeddings, meta_rows)
    spread = compute_spread_metrics(embeddings, meta_rows)
    extended = build_extended_diagnostics(
        embeddings=embeddings,
        meta_rows=meta_rows,
        retrieval=retrieval,
        classification=classification,
        pair_metrics=pair_metrics,
        score_distributions=extra_pair_diag.get("score_distributions", {}),
        pairwise_auc=extra_pair_diag.get("pairwise_auc", {}),
    )
    pair_score_components = build_pair_score_components(pair_metrics)
    metrics_flat = flatten_metrics_for_split(
        retrieval=retrieval,
        classification=classification,
        spread=spread,
        pair_metrics=pair_metrics,
    )

    write_json(split_out / "retrieval_metrics.json", retrieval)
    write_json(split_out / "classification_metrics.json", classification)
    write_json(split_out / "spread_metrics.json", spread)
    write_json(split_out / "pair_score_components.json", pair_score_components)
    write_json(split_out / "metrics_flat.json", metrics_flat)
    write_json(split_out / "retrieval_curve.json", {"recall_curve": retrieval.get("recall_curve", {})})
    write_json(split_out / "threshold_sweep.json", extended.get("threshold_sweep", {}))
    write_json(split_out / "retrieval_at_k.json", extended.get("retrieval_at_k", {}))
    write_json(split_out / "score_distributions.json", extended.get("score_distributions", {}))
    write_json(split_out / "pairwise_auc.json", extended.get("pairwise_auc", {}))
    write_json(split_out / "conditioned_metrics.json", extended.get("conditioned_metrics", {}))
    write_json(split_out / "bootstrap_ci.json", extended.get("bootstrap_ci", {}))
    write_json(split_out / "calibration.json", extended.get("calibration", {}))
    write_json(split_out / "top_confusions.json", extended.get("top_confusions", {}))

    if save_embeddings:
        np.savez_compressed(
            split_out / "embeddings_and_meta.npz",
            embeddings=embeddings,
            player_id=np.asarray([m.get("player_id") for m in meta_rows], dtype=object),
            phase_id=np.asarray([m.get("phase_id") for m in meta_rows], dtype=object),
            engine_rank=np.asarray([m.get("engine_rank") for m in meta_rows], dtype=object),
            engine_cp_gap=np.asarray([m.get("engine_cp_gap") for m in meta_rows], dtype=object),
            example_id=np.asarray([m.get("example_id") for m in meta_rows], dtype=object),
        )

    summary = {
        "split": split,
        "cached_split_dir": str(cached_split_dir),
        "pair_metrics_path": str(split_out / "pair_metrics.json"),
        "retrieval_metrics_path": str(split_out / "retrieval_metrics.json"),
        "classification_metrics_path": str(split_out / "classification_metrics.json"),
        "spread_metrics_path": str(split_out / "spread_metrics.json"),
        "pair_score_components_path": str(split_out / "pair_score_components.json"),
        "metrics_flat_path": str(split_out / "metrics_flat.json"),
        "retrieval_curve_path": str(split_out / "retrieval_curve.json"),
        "threshold_sweep_path": str(split_out / "threshold_sweep.json"),
        "retrieval_at_k_path": str(split_out / "retrieval_at_k.json"),
        "score_distributions_path": str(split_out / "score_distributions.json"),
        "pairwise_auc_path": str(split_out / "pairwise_auc.json"),
        "conditioned_metrics_path": str(split_out / "conditioned_metrics.json"),
        "bootstrap_ci_path": str(split_out / "bootstrap_ci.json"),
        "calibration_path": str(split_out / "calibration.json"),
        "top_confusions_path": str(split_out / "top_confusions.json"),
        "sampled_embedding_summary_path": str(split_out / "sampled_embedding_summary.json"),
        "embedding_source": "sampled_anchor_rows_from_pair_eval",
        "n_embeddings": int(len(embeddings)),
        "saved_embeddings": bool(save_embeddings),
        "max_embed_batches_used": max_embed_batches,
        "max_examples_used": max_examples,
    }
    write_json(split_out / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Directory containing best.pt / epoch_*.pt")
    ap.add_argument("--pairs-dir", required=True, help="Raw pairs parent dir containing eval/*.jsonl and test/*.jsonl")
    ap.add_argument("--output-dir", required=True, help="Root output dir")
    ap.add_argument("--checkpoint-name", default="best", help="best | last | exact file name like epoch_003.pt")
    ap.add_argument("--splits", nargs="+", default=["eval", "test"])
    ap.add_argument("--eval-taus", nargs="+", type=float, default=DEFAULT_EVAL_TAUS)
    ap.add_argument("--rows-per-shard", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-eval-rows", type=int, default=None)
    ap.add_argument("--max-test-rows", type=int, default=None)
    ap.add_argument("--max-pair-batches", type=int, default=None)
    ap.add_argument("--max-embed-batches", type=int, default=None, help="Unused in sampled-anchor mode; kept for compatibility")
    ap.add_argument("--max-examples", type=int, default=None, help="Unused in sampled-anchor mode; kept for compatibility")
    ap.add_argument("--sampled-embedding-max-players", type=int, default=500)
    ap.add_argument("--sampled-embedding-max-examples-per-player", type=int, default=16)
    ap.add_argument("--sampled-embedding-min-examples-per-player", type=int, default=2)
    ap.add_argument("--sampled-embedding-player-selection", choices=["most_seen", "random"], default="most_seen")
    ap.add_argument("--sampled-embedding-seed", type=int, default=42)
    ap.add_argument("--progress-every-batches", type=int, default=100, help="Print progress every N pair batches (<=0 disables).")
    ap.add_argument(
        "--use-embedding-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache per-shard example embeddings and reuse across reruns for faster eval.",
    )
    ap.add_argument(
        "--embedding-cache-batch-size",
        type=int,
        default=8192,
        help="Batch size for shard embedding-cache encoding path.",
    )
    ap.add_argument("--save-embeddings", action="store_true")
    ap.add_argument("--keep-cached", action="store_true", help="Keep generated cached eval dirs instead of deleting later")
    ap.add_argument(
        "--reuse-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing generated_eval_cache/<split> when valid; disable with --no-reuse-cache.",
    )
    ap.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Always rebuild generated_eval_cache/<split> even when reusable cache exists.",
    )
    ap.add_argument(
        "--resume-cache-build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When cache is not fully reusable, resume cache construction from completed shard_* dirs.",
    )
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    set_seed(args.seed)

    model_dir = Path(args.model_dir)
    pairs_dir = Path(args.pairs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    checkpoint_path = resolve_checkpoint(model_dir, args.checkpoint_name)
    model, cfg, ckpt = build_model_from_checkpoint(checkpoint_path, device)
    assert_model_dir_matches_variant(model_dir, cfg)

    pair_variant = getattr(cfg, "pair_variant", None)
    if pair_variant is not None and pair_variant not in str(pairs_dir):
        print(f"[warn] checkpoint pair_variant={pair_variant} but pairs_dir does not contain that substring: {pairs_dir}")

    cache_root = output_dir / "generated_eval_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "model_dir": str(model_dir),
        "pairs_dir": str(pairs_dir),
        "cache_root": str(cache_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "eval_taus": list(args.eval_taus),
        "checkpoint_epoch": ckpt.get("epoch"),
        "model_variant": cfg.model.variant_name,
        "pair_variant": pair_variant,
        "train_tau": getattr(cfg, "tau", None),
        "rows_per_shard": args.rows_per_shard,
        "splits": args.splits,
        "embedding_source": "sampled_anchor_rows_from_pair_eval",
        "sampled_embedding_max_players": args.sampled_embedding_max_players,
        "sampled_embedding_max_examples_per_player": args.sampled_embedding_max_examples_per_player,
        "sampled_embedding_min_examples_per_player": args.sampled_embedding_min_examples_per_player,
        "sampled_embedding_player_selection": args.sampled_embedding_player_selection,
        "sampled_embedding_seed": args.sampled_embedding_seed,
        "progress_every_batches": args.progress_every_batches,
        "use_embedding_cache": bool(args.use_embedding_cache),
        "embedding_cache_batch_size": int(args.embedding_cache_batch_size),
        "reuse_cache": bool(args.reuse_cache),
        "force_rebuild_cache": bool(args.force_rebuild_cache),
        "resume_cache_build": bool(args.resume_cache_build),
    }
    write_json(output_dir / "manifest.json", manifest)

    split_summaries: List[Dict[str, Any]] = []
    for split in args.splits:
        raw_split_dir = pairs_dir / split
        if not raw_split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {raw_split_dir}")

        max_rows = args.max_eval_rows if split == "eval" else args.max_test_rows if split == "test" else None
        input_is_cached, input_cached_reason = is_prebuilt_cached_split_dir(raw_split_dir)
        if input_is_cached:
            cached_split_dir = raw_split_dir
            cache_summary_path = cached_split_dir / "_build_summary.json"
            print(f"[stage] using pre-cached pair split for split={split} from {cached_split_dir}")
            if max_rows is not None:
                print(
                    f"[stage] note: max_rows={max_rows:,} is ignored when using pre-cached split input "
                    f"(evaluate-time row limiting remains available via max_pair_batches/max_examples)"
                )
            cache_meta = read_json_if_exists(cache_summary_path) or {
                "input_dir": str(cached_split_dir),
                "output_dir": str(cached_split_dir),
                "source": "prebuilt_cached_pairs_input",
            }
            cache_reused = True
        else:
            if raw_split_dir.exists():
                print(f"[stage] split={split} input cache detection: {input_cached_reason}; expecting JSONL input")
            cached_split_dir = cache_root / split
            cache_summary_path = cached_split_dir / "_build_summary.json"

            cache_reused = False
            if bool(args.reuse_cache) and not bool(args.force_rebuild_cache):
                reusable, reason = validate_cached_split_dir(cached_split_dir)
                if reusable:
                    print(f"[stage] reusing eval cache for split={split} from {cached_split_dir}")
                    cache_meta = read_json_if_exists(cache_summary_path) or {}
                    cache_reused = True
                else:
                    print(f"[stage] cache not reusable for split={split}: {reason}; rebuilding")

            if not cache_reused:
                resume_build = bool(args.resume_cache_build) and cached_split_dir.exists() and not bool(args.force_rebuild_cache)
                if cached_split_dir.exists() and not resume_build:
                    # Explicit clean rebuild path.
                    shutil.rmtree(cached_split_dir)
                if resume_build:
                    print(f"[stage] resuming eval cache build for split={split}")
                else:
                    print(f"[stage] building eval cache for split={split}")
                cache_meta = build_eval_cache_from_pairs(
                    input_dir=raw_split_dir,
                    output_dir=cached_split_dir,
                    rows_per_shard=args.rows_per_shard,
                    max_rows=max_rows,
                    dataset_tag=split,
                    resume_from_existing=resume_build,
                )
                write_json(cache_summary_path, cache_meta)

        cache_meta = dict(cache_meta)
        cache_meta["cache_reused"] = bool(cache_reused)
        cache_meta["cache_dir"] = str(cached_split_dir)
        cache_meta["cache_build_summary_path"] = str(cache_summary_path)

        split_summary = run_eval_for_split(
            split=split,
            cached_split_dir=cached_split_dir,
            output_root=output_dir,
            model=model,
            cfg=cfg,
            device=device,
            eval_taus=args.eval_taus,
            batch_size_override=args.batch_size,
            num_workers=args.num_workers,
            max_pair_batches=args.max_pair_batches,
            max_embed_batches=args.max_embed_batches,
            max_examples=args.max_examples,
            save_embeddings=args.save_embeddings,
            sampled_embedding_max_players=args.sampled_embedding_max_players,
            sampled_embedding_max_examples_per_player=args.sampled_embedding_max_examples_per_player,
            sampled_embedding_min_examples_per_player=args.sampled_embedding_min_examples_per_player,
            sampled_embedding_player_selection=args.sampled_embedding_player_selection,
            sampled_embedding_seed=args.sampled_embedding_seed,
            progress_every_batches=args.progress_every_batches,
            use_embedding_cache=bool(args.use_embedding_cache),
            embedding_cache_batch_size=int(args.embedding_cache_batch_size),
            model_signature=_checkpoint_signature(checkpoint_path),
        )
        split_summary["cache_build_summary_path"] = str(cache_summary_path)
        split_summary["cache_reused"] = bool(cache_reused)
        split_summaries.append(split_summary)

    final_summary = {
        "manifest_path": str(output_dir / "manifest.json"),
        "splits": split_summaries,
        "cache_root": str(cache_root),
        "keep_cached": bool(args.keep_cached),
    }
    write_json(output_dir / "final_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
