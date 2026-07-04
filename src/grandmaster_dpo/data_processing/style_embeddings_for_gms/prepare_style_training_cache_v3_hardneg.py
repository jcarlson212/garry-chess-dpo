from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from grandmaster_dpo.utilities.npy_io import load_npy
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_MODEL_PATH = (
    "/Users/jasoncarlson/Documents/GitHub/garry-chess-dpo/"
    "final_experiments_for_paper/experiment2_style_model/trained_models/"
    "screen_v1_phi0_tau0_75__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.75__seed-42/"
    "best.pt"
)


def pick_device(user_device: Optional[str]) -> torch.device:
    if user_device:
        return torch.device(user_device)
    if torch.backends.mps.is_available():
        print("[device] using MPS (Apple GPU)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[device] using CUDA")
        return torch.device("cuda")
    print("[device] using CPU")
    return torch.device("cpu")


class BoardCNN(nn.Module):
    def __init__(self, board_embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, board_embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleEncoder(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int,
        token_embed_dim: int,
        dropout: float,
        variant_name: str,
        board_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.variant_name = variant_name
        self.num_boards = 5
        self.board_embed_dim = board_embed_dim

        self.move_embed = nn.Embedding(70, token_embed_dim)
        self.game_type_embed = nn.Embedding(8, 16)

        self.board_cnn = BoardCNN(
            board_embed_dim=self.board_embed_dim,
            dropout=dropout,
        )

        board_in_dim = self.num_boards * self.board_embed_dim
        move_in_dim = 3 * token_embed_dim

        aux_dim = 0
        if variant_name in {"phi1", "phi3"}:
            aux_dim += 16
        if variant_name == "phi3":
            aux_dim += 32

        self.mlp = nn.Sequential(
            nn.Linear(board_in_dim + move_in_dim + aux_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        boards = feats["boards"]  # [B, 5, 12, 8, 8]
        move = feats["move"]      # [B, 3]

        bsz, num_boards, channels, h, w = boards.shape
        board_inputs = boards.reshape(bsz * num_boards, channels, h, w)
        board_vecs = self.board_cnn(board_inputs)
        board_vecs = board_vecs.reshape(bsz, num_boards * self.board_embed_dim)

        move_vec = self.move_embed(move).reshape(bsz, -1)

        parts = [board_vecs, move_vec]

        if "game_type" in feats:
            parts.append(self.game_type_embed(feats["game_type"]))

        if "opponent_context" in feats:
            parts.append(feats["opponent_context"])

        x = torch.cat(parts, dim=-1)
        z = self.mlp(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


def infer_model_hparams(state_dict: Dict[str, torch.Tensor], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    variant_name = "phi0"
    dropout = 0.0
    if config:
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            variant_name = model_cfg.get("variant_name", variant_name)
            dropout = float(model_cfg.get("dropout", dropout))

    move_embed_weight = state_dict["move_embed.weight"]
    token_embed_dim = int(move_embed_weight.shape[1])

    mlp0_weight = state_dict["mlp.0.weight"]
    hidden_dim = int(mlp0_weight.shape[0])

    mlp6_weight = state_dict["mlp.6.weight"]
    embedding_dim = int(mlp6_weight.shape[0])

    return {
        "variant_name": variant_name,
        "dropout": dropout,
        "token_embed_dim": token_embed_dim,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
    }


def load_model(model_path: str, device: torch.device) -> Tuple[StyleEncoder, Dict[str, Any]]:
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    config = ckpt.get("config", {})
    hps = infer_model_hparams(state_dict, config)

    model = StyleEncoder(
        embedding_dim=hps["embedding_dim"],
        hidden_dim=hps["hidden_dim"],
        token_embed_dim=hps["token_embed_dim"],
        dropout=hps["dropout"],
        variant_name=hps["variant_name"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, hps


def shard_dirs(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.glob("shard_*") if p.is_dir()])


def load_shard(shard_dir: Path) -> Dict[str, np.ndarray]:
    return {
        "boards": load_npy(shard_dir / "examples_board_tokens.uint8.npy", mmap_mode="r"),
        "moves": load_npy(shard_dir / "examples_moves.uint8.npy", mmap_mode="r"),
        "game_types": load_npy(shard_dir / "examples_game_type.uint8.npy", mmap_mode="r"),
        "anchor_idx": load_npy(shard_dir / "pair_anchor_idx.int32.npy", mmap_mode="r"),
        "pos_flat": load_npy(shard_dir / "pair_pos_flat.int32.npy", mmap_mode="r"),
        "pos_offsets": load_npy(shard_dir / "pair_pos_offsets.int64.npy", mmap_mode="r"),
        "neg_flat": load_npy(shard_dir / "pair_neg_flat.int32.npy", mmap_mode="r"),
        "neg_offsets": load_npy(shard_dir / "pair_neg_offsets.int64.npy", mmap_mode="r"),
    }


def compact_to_batch(
    boards_u8: np.ndarray,
    moves_u8: np.ndarray,
    game_types_u8: np.ndarray,
    *,
    variant_name: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    boards = torch.from_numpy(boards_u8.astype(np.int64, copy=False))
    boards = F.one_hot(boards, num_classes=13)[..., 1:]
    boards = boards.permute(0, 1, 3, 2).reshape(boards.shape[0], 5, 12, 8, 8).float()

    out: Dict[str, torch.Tensor] = {
        "boards": boards.to(device),
        "move": torch.from_numpy(moves_u8.astype(np.int64, copy=False)).to(device),
    }

    if variant_name in {"phi1", "phi3"}:
        out["game_type"] = torch.from_numpy(game_types_u8.astype(np.int64, copy=False)).to(device)

    if variant_name == "phi3":
        out["opponent_context"] = torch.zeros((boards.shape[0], 32), dtype=torch.float32, device=device)

    return out


@torch.no_grad()
def compute_embeddings_for_shard(
    model: StyleEncoder,
    shard: Dict[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    boards = shard["boards"]
    moves = shard["moves"]
    game_types = shard["game_types"]

    out_chunks: List[np.ndarray] = []
    n = int(boards.shape[0])

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = compact_to_batch(
            boards[start:end],
            moves[start:end],
            game_types[start:end],
            variant_name=model.variant_name,
            device=device,
        )
        z = model(batch)
        out_chunks.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
        if start // batch_size % 10 == 0:
            print(f"[embed] rows={end:,}/{n:,}")

    return np.concatenate(out_chunks, axis=0)


def choose_hard_negatives(
    embeddings: np.ndarray,
    anchor_idx: np.ndarray,
    neg_flat: np.ndarray,
    neg_offsets: np.ndarray,
    *,
    hard_negatives_per_pair: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    new_neg_flat: List[np.ndarray] = []
    new_neg_offsets: List[int] = [0]

    pair_count = int(anchor_idx.shape[0])
    total_original = 0
    total_kept = 0
    all_pool_mean_l2: List[float] = []
    all_kept_mean_l2: List[float] = []

    for i in range(pair_count):
        a_idx = int(anchor_idx[i])
        ns = int(neg_offsets[i])
        ne = int(neg_offsets[i + 1])
        cand = neg_flat[ns:ne].astype(np.int32, copy=False)

        if cand.size == 0:
            selected = cand
        elif hard_negatives_per_pair <= 0 or cand.size <= hard_negatives_per_pair:
            selected = cand
            dists = np.linalg.norm(embeddings[cand] - embeddings[a_idx], axis=1)
            if dists.size > 0:
                all_pool_mean_l2.append(float(dists.mean()))
                all_kept_mean_l2.append(float(dists.mean()))
        else:
            dists = np.linalg.norm(embeddings[cand] - embeddings[a_idx], axis=1)
            order = np.argsort(dists, kind="stable")
            keep = order[:hard_negatives_per_pair]
            selected = cand[keep]
            all_pool_mean_l2.append(float(dists.mean()))
            all_kept_mean_l2.append(float(dists[keep].mean()))

        new_neg_flat.append(selected.astype(np.int32, copy=False))
        new_neg_offsets.append(new_neg_offsets[-1] + int(selected.size))
        total_original += int(cand.size)
        total_kept += int(selected.size)

        if (i + 1) % 10000 == 0:
            print(f"[mine] pairs={i + 1:,}/{pair_count:,}")

    meta = {
        "num_pairs": float(pair_count),
        "total_original_negatives": float(total_original),
        "total_kept_negatives": float(total_kept),
        "avg_original_negatives_per_pair": float(total_original / max(1, pair_count)),
        "avg_kept_negatives_per_pair": float(total_kept / max(1, pair_count)),
        "mean_pool_l2": float(np.mean(all_pool_mean_l2)) if all_pool_mean_l2 else 0.0,
        "mean_kept_l2": float(np.mean(all_kept_mean_l2)) if all_kept_mean_l2 else 0.0,
    }

    neg_flat_out = np.concatenate(new_neg_flat, axis=0) if new_neg_flat else np.zeros((0,), dtype=np.int32)
    neg_offsets_out = np.asarray(new_neg_offsets, dtype=np.int64)
    return neg_flat_out, neg_offsets_out, meta


def save_shard_copy(
    src_shard_dir: Path,
    out_shard_dir: Path,
    shard: Dict[str, np.ndarray],
    new_neg_flat: np.ndarray,
    new_neg_offsets: np.ndarray,
    extra_meta: Dict[str, Any],
) -> Dict[str, Any]:
    out_shard_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_shard_dir / "examples_board_tokens.uint8.npy", np.asarray(shard["boards"]), allow_pickle=False)
    np.save(out_shard_dir / "examples_moves.uint8.npy", np.asarray(shard["moves"]), allow_pickle=False)
    np.save(out_shard_dir / "examples_game_type.uint8.npy", np.asarray(shard["game_types"]), allow_pickle=False)
    np.save(out_shard_dir / "pair_anchor_idx.int32.npy", np.asarray(shard["anchor_idx"]), allow_pickle=False)
    np.save(out_shard_dir / "pair_pos_flat.int32.npy", np.asarray(shard["pos_flat"]), allow_pickle=False)
    np.save(out_shard_dir / "pair_pos_offsets.int64.npy", np.asarray(shard["pos_offsets"]), allow_pickle=False)
    np.save(out_shard_dir / "pair_neg_flat.int32.npy", new_neg_flat, allow_pickle=False)
    np.save(out_shard_dir / "pair_neg_offsets.int64.npy", new_neg_offsets, allow_pickle=False)

    src_meta_path = src_shard_dir / "meta.json"
    src_meta = {}
    if src_meta_path.exists():
        with src_meta_path.open("r", encoding="utf-8") as f:
            src_meta = json.load(f)

    meta = dict(src_meta)
    meta.update({
        "source_shard_dir": str(src_shard_dir),
        "hard_negative_mining": {
            "method": "fast_easy_v3_from_cached_v2",
            **extra_meta,
        },
        "pair_neg_flat_shape": list(new_neg_flat.shape),
        "pair_neg_offsets_shape": list(new_neg_offsets.shape),
    })

    with (out_shard_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def process_split(
    split_name: str,
    in_split_dir: Path,
    out_split_dir: Path,
    *,
    model: StyleEncoder,
    device: torch.device,
    hard_negatives_per_pair: int,
    embedding_batch_size: int,
    max_shards: Optional[int],
) -> Dict[str, Any]:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    shards = shard_dirs(in_split_dir)
    if max_shards is not None:
        shards = shards[:max_shards]

    split_stats: List[Dict[str, Any]] = []
    total_pairs = 0
    total_original_negs = 0
    total_kept_negs = 0

    for idx, shard_dir in enumerate(shards):
        print(f"\n[{split_name}] shard {idx + 1}/{len(shards)} -> {shard_dir.name}")
        shard = load_shard(shard_dir)

        embeddings = compute_embeddings_for_shard(
            model=model,
            shard=shard,
            batch_size=embedding_batch_size,
            device=device,
        )

        new_neg_flat, new_neg_offsets, mining_meta = choose_hard_negatives(
            embeddings=embeddings,
            anchor_idx=shard["anchor_idx"],
            neg_flat=shard["neg_flat"],
            neg_offsets=shard["neg_offsets"],
            hard_negatives_per_pair=hard_negatives_per_pair,
        )

        out_shard_dir = out_split_dir / shard_dir.name
        shard_meta = save_shard_copy(
            src_shard_dir=shard_dir,
            out_shard_dir=out_shard_dir,
            shard=shard,
            new_neg_flat=new_neg_flat,
            new_neg_offsets=new_neg_offsets,
            extra_meta={
                "hard_negatives_per_pair": hard_negatives_per_pair,
                "embedding_batch_size": embedding_batch_size,
                "embedding_dim": int(embeddings.shape[1]),
                **mining_meta,
            },
        )
        split_stats.append(shard_meta)

        total_pairs += int(shard["anchor_idx"].shape[0])
        total_original_negs += int(mining_meta["total_original_negatives"])
        total_kept_negs += int(mining_meta["total_kept_negatives"])

    split_meta = {
        "split_name": split_name,
        "input_dir": str(in_split_dir),
        "output_dir": str(out_split_dir),
        "num_shards": len(shards),
        "num_pairs": total_pairs,
        "total_original_negatives": total_original_negs,
        "total_kept_negatives": total_kept_negs,
        "avg_original_negatives_per_pair": float(total_original_negs / max(1, total_pairs)),
        "avg_kept_negatives_per_pair": float(total_kept_negs / max(1, total_pairs)),
        "method": "fast_easy_v3_from_cached_v2",
        "note": "Hard negatives are chosen only from each pair row's existing cached v2 negative pool.",
    }
    with (out_split_dir / "_split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    print(json.dumps(split_meta, indent=2))
    return split_meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build fast/easy v3 hard-negative cached shards from an existing cached v2 dataset."
    )
    ap.add_argument("--in-root", type=str, required=True, help="Root cached v2 dir containing train/eval/(test)")
    ap.add_argument("--out-root", type=str, required=True, help="Output root for new cached v3-hardneg dir")
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--embedding-batch-size", type=int, default=4096)
    ap.add_argument("--hard-negatives-per-pair", type=int, default=8)
    ap.add_argument("--max-train-shards", type=int, default=None)
    ap.add_argument("--max-eval-shards", type=int, default=None)
    ap.add_argument("--max-test-shards", type=int, default=None)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model, model_meta = load_model(args.model_path, device=device)

    manifest = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "model_path": args.model_path,
        "model_meta": model_meta,
        "embedding_batch_size": args.embedding_batch_size,
        "hard_negatives_per_pair": args.hard_negatives_per_pair,
        "method": "fast_easy_v3_from_cached_v2",
        "note": "Chooses closest-in-L2 negatives from each pair row's existing cached negative pool.",
        "splits": {},
    }

    for split_name, max_shards in [
        ("train", args.max_train_shards),
        ("eval", args.max_eval_shards),
        ("test", args.max_test_shards),
    ]:
        in_split = in_root / split_name
        if not in_split.exists():
            continue
        manifest["splits"][split_name] = process_split(
            split_name=split_name,
            in_split_dir=in_split,
            out_split_dir=out_root / split_name,
            model=model,
            device=device,
            hard_negatives_per_pair=args.hard_negatives_per_pair,
            embedding_batch_size=args.embedding_batch_size,
            max_shards=max_shards,
        )

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] manifest -> {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
