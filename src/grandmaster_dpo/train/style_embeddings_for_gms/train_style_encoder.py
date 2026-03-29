from __future__ import annotations

import argparse
import orjson as json
import math
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # must be set to use new apple GPU
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from .dataset_schema import ExampleRow, PairRow, TrainConfig
from .pair_variants import validate_pair_row
from .train_configs import STUDIES


PIECE_TO_ID = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}
GAME_TYPE_TO_ID = {"blitz": 1, "rapid": 2, "classical": 3}
PHASE_TO_ID = {"opening": 1, "middlegame": 2, "endgame": 3}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("[device] using MPS (Apple GPU)")
        return torch.device("mps")
    return torch.device("cpu")


def encode_square(s: str) -> int:
    file_idx = ord(s[0]) - ord("a")
    rank_idx = int(s[1]) - 1
    return 1 + rank_idx * 8 + file_idx


def encode_move_uci(move: str) -> List[int]:
    # simple fixed-length move tokenization: from, to, promo
    if len(move) < 4:
        return [0, 0, 0]
    frm = encode_square(move[:2])
    to = encode_square(move[2:4])
    promo = 0
    if len(move) >= 5:
        promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
        promo = promo_map.get(move[4].lower(), 0)
    return [frm, to, promo]


def encode_fen_board_planes(fen: str) -> List[List[List[int]]]:
    planes = [[[0 for _ in range(8)] for _ in range(8)] for _ in range(12)]

    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }

    board = fen.split()[0]
    row = 0
    col = 0

    for ch in board:
        if ch == "/":
            row += 1
            col = 0
        elif ch.isdigit():
            col += int(ch)
        else:
            plane_idx = piece_map[ch]
            planes[plane_idx][row][col] = 1
            col += 1

    return planes  # [12, 8, 8]


def example_to_features(ex: Dict[str, Any], variant_name: str) -> Dict[str, torch.Tensor]:
    boards = torch.tensor([
        encode_fen_board_planes(ex["board_t_minus_4"]),
        encode_fen_board_planes(ex["board_t_minus_3"]),
        encode_fen_board_planes(ex["board_t_minus_2"]),
        encode_fen_board_planes(ex["board_t_minus_1"]),
        encode_fen_board_planes(ex["board_t"]),
    ], dtype=torch.float32)  # [5, 12, 8, 8]
    move = encode_move_uci(ex["move_played"])

    feat: Dict[str, torch.Tensor] = {
        "boards": boards,  # [5, 12, 8, 8]
        "move": torch.tensor(move, dtype=torch.long),      # [3]
    }

    if variant_name in {"phi1", "phi3"}:
        feat["game_type"] = torch.tensor(
            GAME_TYPE_TO_ID.get(ex["game_type"], 0), dtype=torch.long
        )

    if variant_name == "phi3":
        feat["opponent_context"] = torch.zeros(32, dtype=torch.float32)

    return feat

class PairShardedDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        model_variant_name: str,
        max_rows: Optional[int] = None,
    ) -> None:
        self.model_variant_name = model_variant_name
    
        shard_dirs = sorted(Path(input_dir).glob("shard_*"))
        if not shard_dirs:
            raise ValueError(f"No shard_* dirs found in {input_dir}")

        self.shards = []
        self.lengths = []

        total_rows = 0
        for sd in shard_dirs:
            shard = {
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

            total_rows += len(shard["anchor_idx"])
            if max_rows and total_rows > max_rows:
                break

        # prefix sum for global indexing
        self.cum_lengths = np.cumsum(self.lengths)

        print(f"[dataset] loaded {len(self.shards)} shards")
        print(f"[dataset] total rows={self.__len__():,}")

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def _locate(self, idx: int):
        shard_id = np.searchsorted(self.cum_lengths, idx, side="right")
        prev = 0 if shard_id == 0 else self.cum_lengths[shard_id - 1]
        local_idx = idx - prev
        return shard_id, local_idx

    def _example_features(self, shard, ex_idx: int) -> Dict[str, torch.Tensor]:
        boards = shard["boards"][ex_idx]   # [5,64]
        move = shard["moves"][ex_idx]      # [3]

        # convert to one-hot planes [5,12,8,8]
        boards = torch.from_numpy(boards).long()  # [5,64]
        boards = F.one_hot(boards, num_classes=13)[..., 1:]  # drop empty
        boards = boards.permute(0, 2, 1).reshape(5, 12, 8, 8).float()

        feat = {
            "boards": boards,
            "move": torch.from_numpy(move).long(),
        }

        if self.model_variant_name in {"phi1", "phi3"}:
            feat["game_type"] = torch.tensor(
                int(shard["game_types"][ex_idx]),
                dtype=torch.long,
            )

        if self.model_variant_name == "phi3":
            feat["opponent_context"] = torch.zeros(32, dtype=torch.float32)

        return feat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_id, local_idx = self._locate(idx)
        shard = self.shards[shard_id]

        anchor_idx = int(shard["anchor_idx"][local_idx])

        ps = int(shard["pos_offsets"][local_idx])
        pe = int(shard["pos_offsets"][local_idx + 1])
        ns = int(shard["neg_offsets"][local_idx])
        ne = int(shard["neg_offsets"][local_idx + 1])

        pos_idx = int(shard["pos_flat"][np.random.randint(ps, pe)])
        neg_idx = int(shard["neg_flat"][np.random.randint(ns, ne)])

        return {
            "anchor": self._example_features(shard, anchor_idx),
            "positive": self._example_features(shard, pos_idx),
            "negative": self._example_features(shard, neg_idx),
            "anchor_player_id": 0,  # optional now
        }

def _stack_feature_dict(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for item in items:
        for k, v in item.items():
            out[k].append(v)
    return {k: torch.stack(v, dim=0) for k, v in out.items()}


def collate_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "anchor": _stack_feature_dict([x["anchor"] for x in batch]),
        "positive": _stack_feature_dict([x["positive"] for x in batch]),
        "negative": _stack_feature_dict([x["negative"] for x in batch]),
        "anchor_player_id": [x["anchor_player_id"] for x in batch],
    }

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
        # x: [B, 12, 8, 8]
        return self.net(x)

class StyleEncoder(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        m = cfg.model

        self.num_boards = 5
        self.board_embed_dim = 128

        self.move_embed = nn.Embedding(70, m.token_embed_dim)
        self.game_type_embed = nn.Embedding(8, 16)

        self.board_cnn = BoardCNN(
            board_embed_dim=self.board_embed_dim,
            dropout=m.dropout,
        )

        board_in_dim = self.num_boards * self.board_embed_dim
        move_in_dim = 3 * m.token_embed_dim

        aux_dim = 0
        if m.variant_name in {"phi1", "phi3"}:
            aux_dim += 16
        if m.variant_name == "phi3":
            aux_dim += 32

        hidden_dim = m.hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(board_in_dim + move_in_dim + aux_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(m.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(m.dropout),
            nn.Linear(hidden_dim, m.embedding_dim),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        boards = feats["boards"]   # [B, 5, 12, 8, 8]
        move = feats["move"]       # [B, 3]

        bsz, num_boards, channels, h, w = boards.shape

        # flatten batch and board-time dimension so each board goes through the same CNN
        board_inputs = boards.reshape(bsz * num_boards, channels, h, w)   # [B*5, 12, 8, 8]

        # encode each board independently
        board_vecs = self.board_cnn(board_inputs)                         # [B*5, board_embed_dim]

        # concatenate the 5 board embeddings
        board_vecs = board_vecs.reshape(bsz, num_boards * self.board_embed_dim)

        # encode move
        m_emb = self.move_embed(move).reshape(bsz, -1)

        parts = [board_vecs, m_emb]

        if "game_type" in feats:
            gt = self.game_type_embed(feats["game_type"])
            parts.append(gt)

        if "opponent_context" in feats:
            parts.append(feats["opponent_context"])

        x = torch.cat(parts, dim=-1)
        z = self.mlp(x)
        z = F.normalize(z, p=2, dim=-1)
        return z

def move_to_device(batch_part: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch_part.items():
        out[k] = v.to(device)
    return out


def info_nce_triplet_loss(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    tau: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pos_logits = (z_anchor * z_pos).sum(dim=-1) / tau
    neg_logits = (z_anchor * z_neg).sum(dim=-1) / tau
    logits = torch.stack([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(z_anchor.shape[0], dtype=torch.long, device=z_anchor.device)
    loss = F.cross_entropy(logits, labels)

    pos_l2 = (z_anchor - z_pos).pow(2).sum(dim=-1).sqrt().mean()
    neg_l2 = (z_anchor - z_neg).pow(2).sum(dim=-1).sqrt().mean()
    pos_cos = (z_anchor * z_pos).sum(dim=-1).mean()
    neg_cos = (z_anchor * z_neg).sum(dim=-1).mean()
    acc = (pos_logits > neg_logits).float().mean()

    stats = {
        "pair_acc": float(acc.detach().cpu().item()),
        "pos_l2": float(pos_l2.detach().cpu().item()),
        "neg_l2": float(neg_l2.detach().cpu().item()),
        "pos_cos": float(pos_cos.detach().cpu().item()),
        "neg_cos": float(neg_cos.detach().cpu().item()),
        "margin_cos": float((pos_cos - neg_cos).detach().cpu().item()),
    }
    return loss, stats


def summarize_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {k: sum(m[k] for m in metrics) / len(metrics) for k in keys}


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row).decode() + "\n")


def save_checkpoint(
    checkpoint_dir: Path,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
    extra: Dict[str, Any],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.to_dict(),
            "extra": extra,
        },
        path,
    )
    return path


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int, keep_names: Sequence[str]) -> None:
    pts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    keep_set = set(keep_names)
    removable = [p for p in pts if p.name not in keep_set]
    if len(removable) <= keep_last_n:
        return
    for p in removable[:-keep_last_n]:
        p.unlink(missing_ok=True)


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tau: float,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    all_metrics: List[Dict[str, float]] = []
    losses: List[float] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        anchor = move_to_device(batch["anchor"], device)
        positive = move_to_device(batch["positive"], device)
        negative = move_to_device(batch["negative"], device)

        z_anchor = model(anchor)
        z_pos = model(positive)
        z_neg = model(negative)

        loss, stats = info_nce_triplet_loss(z_anchor, z_pos, z_neg, tau=tau)
        losses.append(float(loss.detach().cpu().item()))
        all_metrics.append(stats)

    out = summarize_metrics(all_metrics)
    out["loss"] = sum(losses) / max(1, len(losses))
    return out


def train_one_run(cfg: TrainConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = pick_device()
    print(f"[train] device={device}")
    print(f"[train] run_name={cfg.run_name()}")
    print(f"[train] model_variant={cfg.model.variant_name}")
    print(f"[train] pair_variant={cfg.pair_variant}")
    print(f"[train] max_train_rows={cfg.max_train_rows}")
    print(f"[train] max_eval_rows={cfg.max_eval_rows}")
    print(f"[train] train dir={cfg.train_dir}")
    print(f"[train] eval dir={cfg.eval_dir}")

    train_ds = PairShardedDataset(
        input_dir=cfg.train_dir,
        model_variant_name=cfg.model.variant_name,
        max_rows=cfg.max_train_rows,
    )

    eval_ds = PairShardedDataset(
        input_dir=cfg.eval_dir,
        model_variant_name=cfg.model.variant_name,
        max_rows=cfg.max_eval_rows,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_pairs,
        persistent_workers=(cfg.num_workers > 0 and cfg.persistent_workers),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        pin_memory=cfg.pin_memory,
    )

    model = StyleEncoder(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    summary_path = cfg.summary_path()
    checkpoint_dir = cfg.checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_eval_loss = math.inf
    best_ckpt_name = None
    start_time = time.time()

    append_jsonl(summary_path, {
        "event": "run_start",
        "time": time.time(),
        "config": cfg.to_dict(),
        "device": str(device),
        "num_train_rows": len(train_ds),
        "num_eval_rows": len(eval_ds),
    })

    total_samples_per_hour = 0.0
    number_of_samples_measured = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_metrics: List[Dict[str, float]] = []
        train_losses: List[float] = []

        start_time = time.time()
        for step_idx, batch in enumerate(train_loader, start=1):
            if cfg.max_steps_per_epoch is not None and step_idx > cfg.max_steps_per_epoch:
                break
            anchor = move_to_device(batch["anchor"], device)
            positive = move_to_device(batch["positive"], device)
            negative = move_to_device(batch["negative"], device)

            optimizer.zero_grad(set_to_none=True)

            z_anchor = model(anchor)
            z_pos = model(positive)
            z_neg = model(negative)

            loss, stats = info_nce_triplet_loss(z_anchor, z_pos, z_neg, tau=cfg.tau)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))
            train_metrics.append(stats)
            if step_idx % 50 == 0:
                print(f"[train] epoch {epoch} step {step_idx} loss={loss.item():.4f} acc={stats['pair_acc']:.4f}", end="\r")
            if step_idx % 100 == 0 and step_idx > 0:
                new_start_time = time.time()
                elapsed = new_start_time - start_time
                total_samples_per_hour += step_idx * cfg.batch_size / elapsed* 3600
                number_of_samples_measured += 1
                print(f"[train] samples / hour: {step_idx * cfg.batch_size / elapsed * 3600:.2f}")
                
                avg_step_time = elapsed / step_idx

                steps_left_this_epoch = len(train_loader) - step_idx
                steps_left_future_epochs = (cfg.epochs - epoch) * len(train_loader)

                total_steps_left = steps_left_this_epoch + steps_left_future_epochs

                print(
                    f"[train] estimated total time remaining: "
                    f"{total_steps_left * avg_step_time / 3600:.2f} hours"
                )
                print(f"[train] estimated time remaining for this epoch: "
                    f"{steps_left_this_epoch * avg_step_time / 3600:.2f} hours"
                )
                start_time = new_start_time

        train_summary = summarize_metrics(train_metrics)
        train_summary["loss"] = sum(train_losses) / max(1, len(train_losses))

        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=max(0, min(4, cfg.num_workers)),
            collate_fn=collate_pairs,
            persistent_workers=False,
            prefetch_factor=(2 if cfg.num_workers > 0 else None),
            pin_memory=cfg.pin_memory,
        )

        eval_summary = run_eval(
            model=model,
            loader=eval_loader,
            device=device,
            tau=cfg.tau,
            max_batches=cfg.max_eval_batches,
        )

        row = {
            "event": "epoch_end",
            "time": time.time(),
            "epoch": epoch,
            "train": train_summary,
            "eval": eval_summary,
            "samples_per_hour": total_samples_per_hour / max(1, number_of_samples_measured),
        }
        append_jsonl(summary_path, row)

        ckpt_name = f"epoch_{epoch:03d}"
        if cfg.save_every_epoch:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                name=ckpt_name,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                extra={"train": train_summary, "eval": eval_summary},
            )

        if eval_summary["loss"] < best_eval_loss:
            best_eval_loss = eval_summary["loss"]
            best_ckpt_name = "best"
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                name="best",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                extra={"train": train_summary, "eval": eval_summary},
            )

        cleanup_old_checkpoints(
            checkpoint_dir=checkpoint_dir,
            keep_last_n=cfg.keep_last_n_checkpoints,
            keep_names=["best.pt"],
        )

        if cfg.timeout_minutes is not None:
            elapsed_minutes = (time.time() - start_time) / 60.0
            if elapsed_minutes > cfg.timeout_minutes:
                append_jsonl(summary_path, {
                    "event": "timeout_stop",
                    "time": time.time(),
                    "epoch": epoch,
                    "elapsed_minutes": elapsed_minutes,
                })
                break

    append_jsonl(summary_path, {
        "event": "run_end",
        "time": time.time(),
        "best_eval_loss": best_eval_loss,
        "best_checkpoint": best_ckpt_name,
    })

    return {
        "run_name": cfg.run_name(),
        "best_eval_loss": best_eval_loss,
        "summary_path": str(summary_path),
        "checkpoint_dir": str(checkpoint_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=sorted(STUDIES.keys()))
    args = ap.parse_args()

    cfg = STUDIES[args.study]
    out = train_one_run(cfg)
    print(json.dumps(out, option=json.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()