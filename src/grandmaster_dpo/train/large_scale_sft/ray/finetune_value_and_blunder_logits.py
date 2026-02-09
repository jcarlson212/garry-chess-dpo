"""
Fine-tune Maia2 value head and action logits on Stockfish-labeled data using Ray Train.

This script:
1. Trains the value head against Stockfish's value_wdl (sigmoid-mapped centipawn eval)
2. Trains action logits to match Stockfish's top-k move distribution while preserving
   human-like decision making (only adjusts logits for moves in Stockfish's top-k)

Input: S3 path to JSONL files with Stockfish labels (from label_w_stockfish.py)
Output: S3 path for model checkpoints (model_best.pt, model_latest.pt, model_epoch_N.pt)

Example row format:
{
  "board": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "move": "e2e4",
  "active_win": 0,
  "top_moves": [{"move": "e2e4", "score_cp": 38, "score_wdl": 0.0906}, ...],
  "value_cp": 38,
  "value_wdl": 0.0906
}

Usage:
  python finetune_value_and_blunder_logits.py \
    --input_s3_path s3://bucket/labeled_data/ \
    --model_output_dir s3://bucket/models/maia2_finetuned/ \
    --num_workers 4 \
    --epochs 3

Output files in model_output_dir:
  - model_best.pt: Best checkpoint (lowest loss)
  - model_latest.pt: Most recent checkpoint
  - model_epoch_N.pt: Checkpoint for each epoch
  - training_config.json: Training configuration and final metrics
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import ray
from ray import train
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer

# Maia2 imports
from maia2 import model as maia_model
from maia2 import inference
from maia2.utils import get_all_possible_moves, create_elo_dict, mirror_move


# ----------------------------
# Constants
# ----------------------------

# Maia2's ELO range is capped at 2000. We use the max to get the strongest baseline
# behavior before fine-tuning. After fine-tuning, the model loses its ability to
# play at arbitrary skill levels but gains better value predictions and fewer blunders.
DEFAULT_ELO_SELF = 2000
DEFAULT_ELO_OPPO = 2000
TOP_K_MOVES = 10  # Number of Stockfish top moves to use for action loss


# ----------------------------
# Data Loading
# ----------------------------

def parse_jsonl_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single JSONL line, returning None on error."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


class StockfishLabeledDataset(IterableDataset):
    """
    Iterable dataset that streams JSONL files from S3 or local paths.
    Each worker gets a shard of the files.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        all_moves_dict: Dict[str, int],
        elo_dict: Dict[str, int],
        elo_self: int = DEFAULT_ELO_SELF,
        elo_oppo: int = DEFAULT_ELO_OPPO,
        top_k: int = TOP_K_MOVES,
        ray_worker_rank: int = 0,
        ray_world_size: int = 1,
    ):
        self.file_paths = file_paths
        self.all_moves_dict = all_moves_dict
        self.elo_dict = elo_dict
        self.elo_self = elo_self
        self.elo_oppo = elo_oppo
        self.top_k = top_k
        self.num_moves = len(all_moves_dict)
        self.ray_worker_rank = ray_worker_rank
        self.ray_world_size = ray_world_size
    
    def __iter__(self):
        # First, shard files across Ray workers (DDP ranks)
        # Each Ray worker gets a disjoint subset of files
        ray_sharded_files = self.file_paths[self.ray_worker_rank::self.ray_world_size]
        
        # Then, within each Ray worker, shard across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process loading
            file_iter = iter(ray_sharded_files)
        else:
            # Multi-process loading: shard files across DataLoader workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            file_iter = iter(ray_sharded_files[worker_id::num_workers])
        
        for file_path in file_iter:
            yield from self._process_file(file_path)
    
    def _process_file(self, file_path: str):
        """Process a single JSONL file, yielding preprocessed samples."""
        import smart_open  # For S3 support
        
        with smart_open.open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                row = parse_jsonl_line(line)
                if row is None:
                    continue
                
                sample = self._preprocess_row(row)
                if sample is not None:
                    yield sample
    
    def _preprocess_row(self, row: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Preprocess a single row into tensors for training.
        
        Note: All boards in the labeled data are already from white's perspective
        (black positions were mirrored during filter_to_unique_quality_plays).
        """
        board_fen = row.get("board")
        top_moves = row.get("top_moves", [])
        value_wdl = row.get("value_wdl", 0.0)
        
        if not board_fen or not top_moves:
            return None
        
        try:
            # Preprocess board using Maia2's preprocessing
            # Since boards are already from white's perspective, we treat them as white-to-move
            board_input, elo_self_cat, elo_oppo_cat, legal_moves = inference.preprocessing(
                board_fen, self.elo_self, self.elo_oppo, self.elo_dict, self.all_moves_dict
            )
        except Exception:
            return None
        
        # Build target distribution for top-k moves
        # We'll create a sparse target: only the top-k moves have non-zero targets
        top_k_indices = []
        top_k_wdls = []
        
        for move_info in top_moves[:self.top_k]:
            move_uci = move_info.get("move")
            score_wdl = move_info.get("score_wdl", 0.0)
            
            if move_uci is None:
                continue
            
            # Get move index in Maia2's vocabulary
            # Since board is already in white's perspective, no mirroring needed
            move_idx = self.all_moves_dict.get(move_uci, -1)
            if move_idx >= 0:
                top_k_indices.append(move_idx)
                top_k_wdls.append(score_wdl)
        
        if not top_k_indices:
            return None
        
        # Convert to tensors
        top_k_indices_t = torch.tensor(top_k_indices, dtype=torch.long)
        top_k_wdls_t = torch.tensor(top_k_wdls, dtype=torch.float32)
        
        # Create mask for top-k moves (sparse representation)
        top_k_mask = torch.zeros(self.num_moves, dtype=torch.float32)
        top_k_mask[top_k_indices_t] = 1.0
        
        # Create target WDL distribution (sparse)
        top_k_targets = torch.zeros(self.num_moves, dtype=torch.float32)
        top_k_targets[top_k_indices_t] = top_k_wdls_t
        
        return {
            "board_input": board_input,
            "elo_self": torch.tensor(elo_self_cat, dtype=torch.long),
            "elo_oppo": torch.tensor(elo_oppo_cat, dtype=torch.long),
            "legal_moves": legal_moves,
            "top_k_mask": top_k_mask,
            "top_k_targets": top_k_targets,
            "value_target": torch.tensor(value_wdl, dtype=torch.float32),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples into batched tensors."""
    return {
        "board_input": torch.stack([s["board_input"] for s in batch]),
        "elo_self": torch.stack([s["elo_self"] for s in batch]),
        "elo_oppo": torch.stack([s["elo_oppo"] for s in batch]),
        "legal_moves": torch.stack([s["legal_moves"] for s in batch]),
        "top_k_mask": torch.stack([s["top_k_mask"] for s in batch]),
        "top_k_targets": torch.stack([s["top_k_targets"] for s in batch]),
        "value_target": torch.stack([s["value_target"] for s in batch]),
    }


# ----------------------------
# Loss Functions
# ----------------------------

def value_loss(pred_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    """
    MSE loss for value head.
    
    Maia2's value head is trained as regression in [-1, 1] (losing -> winning).
    The raw logits are already in this range by design, so we just clamp to be safe.
    
    pred_value: Model's value prediction (raw logits in ~[-1, 1])
    target_value: Stockfish's value_wdl in [-1, 1]
    """
    # Maia2's value head outputs values in [-1, 1] range by design
    # Just clamp to ensure we stay in bounds, no additional nonlinearity needed
    pred_clamped = pred_value.clamp(-1.0, 1.0)
    return F.mse_loss(pred_clamped, target_value)


import torch
import torch.nn.functional as F

def action_loss_topk_blunder_prevent(
    logits: torch.Tensor,         # [B, V]
    legal_moves: torch.Tensor,    # [B, V] {0,1}
    top_k_mask: torch.Tensor,     # [B, V] {0,1}
    top_k_scores: torch.Tensor,   # [B, V] score for top-k, 0 elsewhere
    tau_model: float = 1.0,
    tau_target: float = 1.0,
    lambda_mass: float = 1.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    # mask illegal moves out for stability
    very_neg = -1e9 if logits.dtype in (torch.float16, torch.bfloat16) else -1e30
    legal_logits = logits.masked_fill(legal_moves <= 0, very_neg)

    # Only allow gradients through top-k logits; others affect normalization but are frozen.
    mixed_logits = torch.where(top_k_mask > 0, legal_logits, legal_logits.detach())

    # Full softmax over legal moves => can shift mass from outside top-k into top-k
    model_probs = F.softmax(mixed_logits / tau_model, dim=-1)

    # Build a target distribution supported on top-k.
    # Prefer exp/softmax on scores over "shift then normalize" to avoid degenerate all-zero cases.
    target_logits = (top_k_scores / tau_target).masked_fill(top_k_mask <= 0, very_neg)
    target_probs = F.softmax(target_logits, dim=-1)  # sums to 1 over top-k, ~0 elsewhere

    # Cross-entropy (equivalently KL up to constant)
    ce = -(target_probs * torch.log(model_probs.clamp_min(eps))).sum(dim=-1)

    # Explicitly penalize probability mass outside top-k (blunders)
    mass_topk = (model_probs * top_k_mask).sum(dim=-1).clamp_min(eps)
    mass_loss = -torch.log(mass_topk)

    return (ce + lambda_mass * mass_loss).mean()


# ----------------------------
# Training Loop
# ----------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_weight: float = 1.0,
    action_weight: float = 1.0,
    grad_clip: float = 1.0,
    log_interval: int = 100,
) -> Dict[str, float]:
    """Train for one epoch, returning metrics."""
    model.train()
    
    total_loss = 0.0
    total_value_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        board_input = batch["board_input"].to(device)
        elo_self = batch["elo_self"].to(device)
        elo_oppo = batch["elo_oppo"].to(device)
        legal_moves = batch["legal_moves"].to(device)
        top_k_mask = batch["top_k_mask"].to(device)
        top_k_targets = batch["top_k_targets"].to(device)
        value_target = batch["value_target"].to(device)
        
        # Forward pass
        logits_maia, _, logits_value = model(board_input, elo_self, elo_oppo)
        
        # Compute losses
        v_loss = value_loss(logits_value, value_target)
        a_loss = action_loss_topk_blunder_prevent(logits_maia, legal_moves, top_k_mask, top_k_targets)
        
        loss = value_weight * v_loss + action_weight * a_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_value_loss += v_loss.item()
        total_action_loss += a_loss.item()
        num_batches += 1
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_v = total_value_loss / num_batches
            avg_a = total_action_loss / num_batches
            print(f"  Batch {batch_idx + 1}: loss={avg_loss:.4f} value={avg_v:.4f} action={avg_a:.4f}")
    
    return {
        "loss": total_loss / max(1, num_batches),
        "value_loss": total_value_loss / max(1, num_batches),
        "action_loss": total_action_loss / max(1, num_batches),
        "num_batches": num_batches,
    }


# ----------------------------
# Ray Train Function
# ----------------------------

def train_func(config: Dict[str, Any]):
    """
    Ray Train worker function.
    
    This function runs on each worker and handles:
    - Loading the model
    - Creating the dataset and dataloader
    - Training loop with checkpointing
    """
    import smart_open  # Ensure available in worker
    
    # Extract config
    input_s3_path = config["input_s3_path"]
    maia_type = config.get("maia_type", "blitz")
    epochs = config.get("epochs", 3)
    batch_size = config.get("batch_size", 256)
    lr = config.get("lr", 1e-5)
    weight_decay = config.get("weight_decay", 0.0)
    grad_clip = config.get("grad_clip", 1.0)
    value_weight = config.get("value_weight", 1.0)
    action_weight = config.get("action_weight", 1.0)
    num_dataloader_workers = config.get("num_dataloader_workers", 4)
    log_interval = config.get("log_interval", 100)
    elo_self = config.get("elo_self", DEFAULT_ELO_SELF)
    elo_oppo = config.get("elo_oppo", DEFAULT_ELO_OPPO)
    top_k = config.get("top_k", TOP_K_MOVES)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Maia2 model
    print(f"Loading Maia2 model (type={maia_type})...")
    model = maia_model.from_pretrained(type=maia_type, device="cpu")
    model = model.to(device)
    
    # Prepare model for distributed training
    # Note: Maia2's forward returns (logits_maia, logits_side_info, logits_value)
    # but we only use logits_maia and logits_value in our loss, so we need
    # find_unused_parameters=True to handle the unused logits_side_info parameters
    model = train.torch.prepare_model(
        model,
        parallel_strategy_kwargs={"find_unused_parameters": True},
    )
    
    # Get vocab and ELO dict
    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    
    # Discover input files
    print(f"Discovering files from {input_s3_path}...")
    file_paths = _discover_jsonl_files(input_s3_path)
    print(f"Found {len(file_paths)} JSONL files")
    
    if not file_paths:
        raise ValueError(f"No JSONL files found at {input_s3_path}")
    
    # Get Ray worker rank and world size for proper file sharding across DDP workers
    ray_context = train.get_context()
    ray_worker_rank = ray_context.get_world_rank()
    ray_world_size = ray_context.get_world_size()
    
    # Calculate how many files this worker will process
    files_for_this_worker = len(file_paths[ray_worker_rank::ray_world_size])
    print(f"Ray worker {ray_worker_rank}/{ray_world_size} will process {files_for_this_worker} files")
    
    # Create dataset and dataloader
    # Note: We pass ray_worker_rank and ray_world_size so the dataset can shard files
    # across Ray workers, ensuring each worker processes a disjoint subset of files
    dataset = StockfishLabeledDataset(
        file_paths=file_paths,
        all_moves_dict=all_moves_dict,
        elo_dict=elo_dict,
        elo_self=elo_self,
        elo_oppo=elo_oppo,
        top_k=top_k,
        ray_worker_rank=ray_worker_rank,
        ray_world_size=ray_world_size,
    )
    
    # Note: We don't use train.torch.prepare_data_loader() because we're manually
    # sharding files across Ray workers. Using prepare_data_loader would add another
    # layer of sharding via DistributedSampler which doesn't work well with IterableDataset.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_dataloader_workers,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # Get model output dir from config
    model_output_dir = config.get("model_output_dir")
    
    # Training loop
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        
        metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            value_weight=value_weight,
            action_weight=action_weight,
            grad_clip=grad_clip,
            log_interval=log_interval,
        )
        
        print(f"Epoch {epoch} complete: {metrics}")
        
        # Get the underlying model if wrapped by DDP
        model_to_save = model.module if hasattr(model, "module") else model
        
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "maia_type": maia_type,
        }
        
        # Save checkpoint to Ray (for fault tolerance and tracking)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            torch.save(checkpoint_data, checkpoint_path)
            
            checkpoint = Checkpoint.from_directory(tmpdir)
            train.report(metrics, checkpoint=checkpoint)
        
        # Also save directly to model_output_dir (only on rank 0)
        if train.get_context().get_world_rank() == 0 and model_output_dir:
            _save_checkpoint_to_s3(
                checkpoint_data, 
                model_output_dir, 
                epoch, 
                is_best=(metrics["loss"] < best_loss)
            )
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
    
    print("\nTraining complete!")


def _save_checkpoint_to_s3(
    checkpoint_data: Dict[str, Any],
    model_output_dir: str,
    epoch: int,
    is_best: bool = False,
):
    """
    Save checkpoint directly to S3 model_output_dir.
    
    Saves:
    - model_epoch_{epoch}.pt: Epoch checkpoint
    - model_best.pt: Best checkpoint (if is_best=True)
    - model_latest.pt: Always updated to latest
    """
    import smart_open
    import io
    
    # Prepare clean checkpoint (just model state for inference)
    clean_checkpoint = {
        "model_state_dict": checkpoint_data["model_state_dict"],
        "maia_type": checkpoint_data.get("maia_type"),
        "epoch": checkpoint_data.get("epoch"),
        "metrics": checkpoint_data.get("metrics"),
    }
    
    def save_to_path(path: str):
        buffer = io.BytesIO()
        torch.save(clean_checkpoint, buffer)
        buffer.seek(0)
        with smart_open.open(path, "wb") as f:
            f.write(buffer.read())
    
    # Save epoch checkpoint
    epoch_path = f"{model_output_dir}/model_epoch_{epoch}.pt"
    print(f"Saving epoch {epoch} checkpoint to {epoch_path}")
    save_to_path(epoch_path)
    
    # Save as latest
    latest_path = f"{model_output_dir}/model_latest.pt"
    save_to_path(latest_path)
    
    # Save as best if applicable
    if is_best:
        best_path = f"{model_output_dir}/model_best.pt"
        print(f"New best model! Saving to {best_path}")
        save_to_path(best_path)


def _discover_jsonl_files(s3_path: str) -> List[str]:
    """
    Discover all JSONL files at the given S3 path.
    
    Supports both:
    - Direct file: s3://bucket/path/file.jsonl
    - Directory: s3://bucket/path/ (lists all .jsonl files)
    """
    import boto3
    from urllib.parse import urlparse
    
    parsed = urlparse(s3_path)
    
    if not parsed.scheme == "s3":
        # Local path
        path = Path(s3_path)
        if path.is_file():
            return [str(path)]
        elif path.is_dir():
            return sorted([str(p) for p in path.glob("*.jsonl")])
        else:
            return []
    
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    
    # Check if it's a direct file reference
    if prefix.endswith(".jsonl"):
        return [s3_path]
    
    # List files in the prefix
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".jsonl"):
                files.append(f"s3://{bucket}/{key}")
    
    return sorted(files)


def _save_final_model(result, model_output_dir: str, maia_type: str):
    """
    Save the best checkpoint to the model_output_dir as a clean PyTorch model file.
    
    This saves:
    - model_best.pt: The best model state dict
    - model_final.pt: The final epoch model state dict  
    - training_config.json: Training configuration and final metrics
    """
    import smart_open
    import io
    
    print(f"\nSaving final model to {model_output_dir}...")
    
    # Get best checkpoint
    best_checkpoint = result.best_checkpoints
    if best_checkpoint:
        best_ckpt, best_metrics = best_checkpoint[0]
        
        # Load the checkpoint
        with best_ckpt.as_directory() as ckpt_dir:
            ckpt_path = Path(ckpt_dir) / "model.pt"
            if ckpt_path.exists():
                checkpoint_data = torch.load(ckpt_path, map_location="cpu")
                
                # Save best model state dict
                best_model_path = f"{model_output_dir}/model_best.pt"
                print(f"Saving best model to {best_model_path}")
                
                # Create a clean checkpoint with just the model state dict
                clean_checkpoint = {
                    "model_state_dict": checkpoint_data["model_state_dict"],
                    "maia_type": maia_type,
                    "epoch": checkpoint_data.get("epoch"),
                    "metrics": checkpoint_data.get("metrics"),
                }
                
                # Save to S3 or local
                buffer = io.BytesIO()
                torch.save(clean_checkpoint, buffer)
                buffer.seek(0)
                
                with smart_open.open(best_model_path, "wb") as f:
                    f.write(buffer.read())
                
                print(f"Best model saved with metrics: {best_metrics}")
    
    # Save training config and results
    config_path = f"{model_output_dir}/training_config.json"
    config_data = {
        "maia_type": maia_type,
        "final_metrics": result.metrics,
        "best_checkpoint_metrics": best_metrics if best_checkpoint else None,
    }
    
    with smart_open.open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Training config saved to {config_path}")
    print(f"\nModel artifacts saved to {model_output_dir}/")


# ----------------------------
# Main Entry Point
# ----------------------------

def main():
    # Usage: python finetune_value_and_blunder_logits.py --input_s3_path s3://crljaso-ml-artifacts/chess-engine/all_twic_4k_plus_games_plus_labeled/ --model_output_dir s3://crljaso-ml-artifacts/chess-engine/maia2_finetuned/4k_test/ --ray_storage_path s3://crljaso-ml-artifacts/chess-engine/maia2_finetuned/4k_test/ray_logs/ --num_workers 4conda activate base
    parser = argparse.ArgumentParser(
        description="Fine-tune Maia2 on Stockfish-labeled data using Ray Train"
    )
    
    # Data paths
    parser.add_argument(
        "--input_s3_path",
        type=str,
        required=True,
        help="S3 path to input JSONL files (directory or single file)",
    )
    parser.add_argument(
        "--model_output_dir",
        type=str,
        required=True,
        help="S3 path for final model checkpoint output (e.g., s3://bucket/models/maia2_finetuned/)",
    )
    parser.add_argument(
        "--ray_storage_path",
        type=str,
        default=None,
        help="S3 path for Ray Train logs/intermediate checkpoints (defaults to model_output_dir/ray_logs/)",
    )
    
    # Model config
    parser.add_argument(
        "--maia_type",
        type=str,
        default="blitz",
        choices=["blitz", "rapid"],
        help="Maia2 model type",
    )
    
    # Training config
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per worker")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--value_weight", type=float, default=1.0, help="Weight for value loss")
    parser.add_argument("--action_weight", type=float, default=1.0, help="Weight for action loss")
    
    # Ray config
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Ray workers")
    parser.add_argument("--num_gpus_per_worker", type=float, default=1.0, help="GPUs per worker")
    parser.add_argument("--num_dataloader_workers", type=int, default=4, help="DataLoader workers")
    
    # Other
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N batches")
    parser.add_argument("--elo_self", type=int, default=DEFAULT_ELO_SELF, help="Self ELO for inference (max 2000 for Maia2)")
    parser.add_argument("--elo_oppo", type=int, default=DEFAULT_ELO_OPPO, help="Opponent ELO for inference (max 2000 for Maia2)")
    parser.add_argument("--top_k", type=int, default=TOP_K_MOVES, help="Top-k moves for action loss")
    
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)
    
    # Set up paths
    model_output_dir = args.model_output_dir.rstrip("/")
    ray_storage_path = args.ray_storage_path or f"{model_output_dir}/ray_logs"
    
    # Build config dict
    train_config = {
        "input_s3_path": args.input_s3_path,
        "model_output_dir": model_output_dir,
        "maia_type": args.maia_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "value_weight": args.value_weight,
        "action_weight": args.action_weight,
        "num_dataloader_workers": args.num_dataloader_workers,
        "log_interval": args.log_interval,
        "elo_self": args.elo_self,
        "elo_oppo": args.elo_oppo,
        "top_k": args.top_k,
    }
    
    # Configure Ray Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.num_gpus_per_worker > 0,
            resources_per_worker={
                "GPU": args.num_gpus_per_worker,
            },
        ),
        run_config=train.RunConfig(
            name="maia2_finetune_value_blunder",
            storage_path=ray_storage_path,
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="loss",
                checkpoint_score_order="min",
            ),
        ),
    )
    
    # Run training
    print(f"Starting training with {args.num_workers} workers...")
    print(f"Input: {args.input_s3_path}")
    print(f"Model output: {model_output_dir}")
    print(f"Ray storage: {ray_storage_path}")
    
    result = trainer.fit()
    
    print("\n=== Training Complete ===")
    print(f"Best checkpoint: {result.best_checkpoints}")
    print(f"Final metrics: {result.metrics}")
    
    # Save final/best model to model_output_dir
    _save_final_model(result, model_output_dir, args.maia_type)


if __name__ == "__main__":
    main()
