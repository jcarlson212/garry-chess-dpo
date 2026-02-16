#!/usr/bin/env python3
"""
Train a "timer model" on top of a *frozen* Maia2 model that you've already fine-tuned
(e.g., your DPO policy_best.pt).

Input JSONL rows look like:

{
  "prompt": {"fen": "...", "elo_self": 2800, "elo_oppo": 2700},
  "time_to_make_chosen_move_ms": 512,
  "previous_five_ply_move_times_ms": [412, 510, 100, 150, 400],
  "chosen": "f2f3",
  ...
}

We:
  - load Maia2 base model (blitz/rapid)
  - load your fine-tuned policy weights into Maia2
  - freeze Maia2
  - extract a representation from Maia2 (either a named internal layer via hook,
    OR fallback to using the (masked) policy logits as a representation)
  - concatenate that representation with the previous five move times
  - predict y = log1p(time_to_make_chosen_move_ms)
  - train only the timer head

Notes:
  - If you know a good internal layer name to hook (recommended), pass --hook_layer.
    Example: --hook_layer "trunk.blocks.7"  (depends on maia2 implementation)
  - If hook layer is not found, we automatically fall back to using logits as features.

Example usage:
  python ./src/grandmaster_dpo/train/single_gm/train_think_time_model.py --gm_name magnus 
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move


# ----------------------------
# Utils
# ----------------------------

def device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s in ("mps",):
        return torch.device("mps")
    return torch.device(s)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def max_elo_supported(elo_dict: dict) -> int:
    mx = None
    for k in elo_dict.keys():
        m = re.match(r"^>=\s*(\d+)$", k)
        if m:
            mx = max(mx or 0, int(m.group(1)))
    return mx if mx is not None else 3000


def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))


def batch_preprocess(
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    fens: List[str],
    elo_self: List[int],
    elo_oppo: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    board_inputs = []
    legal_moves = []
    elo_self_cats = []
    elo_oppo_cats = []

    mx = max_elo_supported(elo_dict)

    for fen, es, eo in zip(fens, elo_self, elo_oppo):
        es = min(int(es), mx)
        eo = min(int(eo), mx)

        bi, es_cat, eo_cat, lm = inference.preprocessing(
            fen, es, eo, elo_dict, all_moves_dict
        )
        board_inputs.append(bi)
        legal_moves.append(lm)
        elo_self_cats.append(int(es_cat))
        elo_oppo_cats.append(int(eo_cat))

    board_input = torch.stack(board_inputs, dim=0).to(device)         # [B, C, 8, 8]
    legal_moves_t = torch.stack(legal_moves, dim=0).to(device)        # [B, V]
    elo_self_t = torch.tensor(elo_self_cats, device=device).long()    # [B]
    elo_oppo_t = torch.tensor(elo_oppo_cats, device=device).long()    # [B]
    return board_input, legal_moves_t, elo_self_t, elo_oppo_t


def get_module_by_dotted_name(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    """
    Resolve "a.b.3.c" style names into a submodule if possible.
    """
    cur: Any = root
    for part in dotted.split("."):
        if part.isdigit():
            idx = int(part)
            if isinstance(cur, (nn.ModuleList, list, tuple)):
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
            else:
                return None
        else:
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
    return cur if isinstance(cur, nn.Module) else None


# ----------------------------
# Dataset
# ----------------------------

class TimingJsonl(Dataset):
    def __init__(self, jsonl_path: Path):
        self.rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        p = r["prompt"]
        prev = r.get("previous_five_ply_move_times_ms", r.get("previous_five_ply_move_times_ms".replace("_", ""), None))
        if prev is None:
            prev = r.get("previous_five_ply_move_times_ms", [])
        if not isinstance(prev, list):
            prev = []

        # Ensure exactly 5 values (pad with 0, truncate if longer)
        prev5 = [float(x) for x in prev[:5]]
        while len(prev5) < 5:
            prev5.append(0.0)

        return {
            "fen": p["fen"],
            "elo_self": int(p.get("elo_self", 2800)),
            "elo_oppo": int(p.get("elo_oppo", 2800)),
            "target_ms": float(r["time_to_make_chosen_move_ms"]),
            "prev5_ms": prev5,
            "ply_idx": int(r.get("ply_idx")),
            "player_side": int(1 if r.get("player_side") == "white" else 0),
            "prev_clock_w": int(r.get("prev_clock_w")),
            "prev_clock_b": int(r.get("prev_clock_b")),
        }


def collate_timing(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "fen": [b["fen"] for b in batch],
        "elo_self": [b["elo_self"] for b in batch],
        "elo_oppo": [b["elo_oppo"] for b in batch],
        "target_ms": torch.tensor([b["target_ms"] for b in batch], dtype=torch.float32),
        "prev5_ms": torch.tensor([b["prev5_ms"] for b in batch], dtype=torch.float32),
        "ply_idx": torch.tensor([b["ply_idx"] for b in batch], dtype=torch.int32),
        "player_side": torch.tensor([b["player_side"] for b in batch], dtype=torch.int32),
        "prev_clock_w": torch.tensor([b["prev_clock_w"] for b in batch], dtype=torch.int32),
        "prev_clock_b": torch.tensor([b["prev_clock_b"] for b in batch], dtype=torch.int32),
    }


# ----------------------------
# Model: Frozen Maia2 feature extractor + small timer head
# ----------------------------

@dataclass
class FeatureConfig:
    hook_layer: Optional[str] = None          # dotted module name inside Maia2 to hook
    use_logits_fallback: bool = True          # if hook fails, use masked logits as feature
    logits_feature: str = "masked_logits"     # "masked_logits" or "logprobs"


class FrozenMaia2Featurizer(nn.Module):
    """
    Extracts a representation from a Maia2 model:
      - preferred: forward hook on a named internal layer (hook_layer)
      - fallback: use (masked) policy logits as representation

    Output: [B, D]
    """
    def __init__(
        self,
        maia2_model: nn.Module,
        all_moves_dict: Dict[str, int],
        elo_dict: Dict[str, int],
        device: torch.device,
        cfg: FeatureConfig,
    ):
        super().__init__()
        self.m = maia2_model
        self.all_moves_dict = all_moves_dict
        self.elo_dict = elo_dict
        self.device = device
        self.cfg = cfg

        self._hook_handle = None
        self._hook_buf: Optional[torch.Tensor] = None
        self._hooked = False

        if cfg.hook_layer:
            mod = get_module_by_dotted_name(self.m, cfg.hook_layer)
            if mod is not None:
                self._hook_handle = mod.register_forward_hook(self._forward_hook)
                self._hooked = True
                print(f"[featurizer] Hooking Maia2 layer: {cfg.hook_layer}")
            else:
                print(f"[featurizer] WARNING: hook_layer not found: {cfg.hook_layer}. Will fallback to logits.")

        # Freeze Maia2 params
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.m.eval()

    def _forward_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        # Normalize output to tensor [B, D]
        if isinstance(output, (tuple, list)):
            out = output[0]
        else:
            out = output
        if not torch.is_tensor(out):
            self._hook_buf = None
            return
        # If output is [B, C, H, W], pool it; if [B, D], keep.
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))
        self._hook_buf = out

    @torch.no_grad()
    def forward(
        self,
        fens: List[str],
        elo_self: List[int],
        elo_oppo: List[int],
    ) -> torch.Tensor:
        board_input, legal_moves, es_t, eo_t = batch_preprocess(
            self.all_moves_dict, self.elo_dict, fens, elo_self, elo_oppo, self.device
        )

        self._hook_buf = None
        logits_maia, _, _ = self.m(board_input, es_t, eo_t)  # policy logits

        # Prefer hook feature if available and captured
        if self._hooked and self._hook_buf is not None:
            feats = self._hook_buf
            return feats

        if not self.cfg.use_logits_fallback:
            raise RuntimeError(
                "Hook feature not available and logits fallback disabled. "
                "Pass a valid --hook_layer or enable fallback."
            )

        # Fallback: use masked logits (or logprobs) as representation
        logits = apply_legal_mask(logits_maia, legal_moves)
        if self.cfg.logits_feature == "logprobs":
            feats = torch.log_softmax(logits, dim=-1)
        else:
            feats = logits
        return feats


class TimerHead(nn.Module):
    def __init__(self, in_dim: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TimerModel(nn.Module):
    def __init__(self, featurizer: FrozenMaia2Featurizer, head: TimerHead):
        super().__init__()
        self.featurizer = featurizer
        self.head = head

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        feats = self.featurizer(batch["fen"], batch["elo_self"], batch["elo_oppo"])  # [B, D]

        prev5 = batch["prev5_ms"].to(feats.device)  # [B, 5]
        prev5_feat = torch.log1p(torch.clamp(prev5, min=0.0))

        # clock-left before move for mover (ms)
        # player_side: 1=white, 0=black
        side = batch["player_side"].to(feats.device).long()
        cw = batch["prev_clock_w"].to(feats.device).float() * 1000.0 # convert to ms
        cb = batch["prev_clock_b"].to(feats.device).float() * 1000.0 # convert to ms
        clock_left_ms = torch.where(side == 1, cw, cb).clamp(min=0.0)  # [B]
        clock_feat = torch.log1p(clock_left_ms).unsqueeze(-1)          # [B, 1]

        # ply index feature (scaled)
        ply = batch["ply_idx"].to(feats.device).float().unsqueeze(-1)  # [B, 1]
        ply_feat = ply / 120.0  # rough normalization; tweak if you want

        x = torch.cat([feats, prev5_feat, clock_feat, ply_feat], dim=-1)
        return self.head(x)

# ----------------------------
# Train / Eval
# ----------------------------

@torch.no_grad()
def eval_epoch(model: TimerModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0.0
    total_mae_ms = 0.0
    n = 0

    for batch in loader:
        # Move tensor fields to device
        batch = dict(batch)
        batch["target_ms"] = batch["target_ms"].to(device)
        batch["prev5_ms"] = batch["prev5_ms"].to(device)

        pred_log = model(batch)  # predicts log1p(ms)
        tgt_log = torch.log1p(torch.clamp(batch["target_ms"], min=0.0))

        # Huber in log-space
        loss = torch.nn.functional.smooth_l1_loss(pred_log, tgt_log, beta=0.2, reduction="mean")

        # also report MAE in ms-space (roughly)
        pred_ms = torch.expm1(pred_log).clamp(min=0.0)
        mae_ms = torch.mean(torch.abs(pred_ms - batch["target_ms"]))

        bs = batch["target_ms"].shape[0]
        total += float(loss) * bs
        total_mae_ms += float(mae_ms) * bs
        n += bs

    return {
        "loss_log_huber": total / max(1, n),
        "mae_ms": total_mae_ms / max(1, n),
    }

@torch.no_grad()
def eval_epoch(model: TimerModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    losses = []
    abs_err_ms = []
    abs_err_sec_round = []
    acc_sec_round = []

    total = 0.0
    total_mae_ms = 0.0
    n = 0

    for batch in loader:
        batch = dict(batch)
        batch["target_ms"] = batch["target_ms"].to(device)
        batch["prev5_ms"] = batch["prev5_ms"].to(device)

        # keep these if your TimerModel uses them
        if "ply_idx" in batch:
            batch["ply_idx"] = batch["ply_idx"].to(device)
        if "player_side" in batch:
            batch["player_side"] = batch["player_side"].to(device)
        if "prev_clock_w" in batch:
            batch["prev_clock_w"] = batch["prev_clock_w"].to(device)
        if "prev_clock_b" in batch:
            batch["prev_clock_b"] = batch["prev_clock_b"].to(device)

        pred_log = model(batch)  # predicts log1p(ms)
        tgt_log = torch.log1p(torch.clamp(batch["target_ms"], min=0.0))

        loss = torch.nn.functional.smooth_l1_loss(pred_log, tgt_log, beta=0.2, reduction="mean")

        pred_ms = torch.expm1(pred_log).clamp(min=0.0)
        ae_ms = torch.abs(pred_ms - batch["target_ms"])

        # Metrics that respect clock quantization: round to nearest second
        pred_sec_r = torch.round(pred_ms / 1000.0)
        tgt_sec_r = torch.round(batch["target_ms"] / 1000.0)
        ae_sec_r = torch.abs(pred_sec_r - tgt_sec_r)
        acc_sec = (pred_sec_r == tgt_sec_r).float()

        bs = batch["target_ms"].shape[0]
        total += float(loss) * bs
        total_mae_ms += float(ae_ms.mean()) * bs
        n += bs

        losses.append(loss.detach().cpu())
        abs_err_ms.append(ae_ms.detach().cpu())
        abs_err_sec_round.append(ae_sec_r.detach().cpu())
        acc_sec_round.append(acc_sec.detach().cpu())

    # Aggregate
    if n == 0:
        return {"loss_log_huber": float("nan"), "mae_ms": float("nan")}

    ae_ms_all = torch.cat(abs_err_ms, dim=0)
    ae_sec_all = torch.cat(abs_err_sec_round, dim=0)
    acc_sec_all = torch.cat(acc_sec_round, dim=0)

    def pct(x: torch.Tensor, q: float) -> float:
        return float(torch.quantile(x, q).item())

    return {
        "loss_log_huber": total / max(1, n),
        "mae_ms": total_mae_ms / max(1, n),
        "med_ae_ms": float(torch.median(ae_ms_all).item()),
        "p90_ae_ms": pct(ae_ms_all, 0.90),
        "p95_ae_ms": pct(ae_ms_all, 0.95),
        "p99_ae_ms": pct(ae_ms_all, 0.99),
        "mae_sec_round": float(ae_sec_all.mean().item()),
        "acc_sec_round": float(acc_sec_all.mean().item()),
    }


def train(
    model: TimerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    metrics_out_dir: Optional[Path] = None,   # NEW: write per-epoch metrics JSONL here
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_out_dir is None:
        metrics_out_dir = out_dir / "metrics"
    metrics_out_dir.mkdir(parents=True, exist_ok=True)

    metrics_jsonl = metrics_out_dir / "train_val_metrics.jsonl"

    # Only train head (featurizer is frozen by construction)
    optim = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for batch in train_loader:
            step += 1

            batch = dict(batch)
            batch["target_ms"] = batch["target_ms"].to(device)
            batch["prev5_ms"] = batch["prev5_ms"].to(device)
            # keep these if your TimerModel uses them
            if "ply_idx" in batch:
                batch["ply_idx"] = batch["ply_idx"].to(device)
            if "player_side" in batch:
                batch["player_side"] = batch["player_side"].to(device)
            if "prev_clock_w" in batch:
                batch["prev_clock_w"] = batch["prev_clock_w"].to(device)
            if "prev_clock_b" in batch:
                batch["prev_clock_b"] = batch["prev_clock_b"].to(device)

            pred_log = model(batch)
            tgt_log = torch.log1p(torch.clamp(batch["target_ms"], min=0.0))

            loss = torch.nn.functional.smooth_l1_loss(pred_log, tgt_log, beta=0.2, reduction="mean")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
            optim.step()

            bs = batch["target_ms"].shape[0]
            running += float(loss.detach()) * bs
            seen += bs

            if step % 100 == 0:
                print(f"[epoch {epoch}] step={step} train_loss_log_huber={running/max(1,seen):.4f}")

        # ---- NEW: richer eval metrics ----
        val_metrics = eval_epoch(model, val_loader, device=device)

        # Pretty print core metrics
        msg = (
            f"[epoch {epoch}] "
            f"val_loss_log_huber={val_metrics['loss_log_huber']:.4f}  "
            f"val_mae_ms={val_metrics['mae_ms']:.2f}  "
            f"val_med_ae_ms={val_metrics.get('med_ae_ms', float('nan')):.2f}  "
            f"val_mae_sec_round={val_metrics.get('mae_sec_round', float('nan')):.3f}  "
            f"val_acc_sec_round={val_metrics.get('acc_sec_round', float('nan')):.3f}  "
            f"val_p90_ae_ms={val_metrics.get('p90_ae_ms', float('nan')):.2f}  "
            f"val_p95_ae_ms={val_metrics.get('p95_ae_ms', float('nan')):.2f}"
        )
        print(msg)

        # Write metrics row (JSONL) so you can plot later
        with open(metrics_jsonl, "a", encoding="utf-8") as f:
            row = {"epoch": epoch, "step": step, "train_loss_log_huber": running / max(1, seen)}
            row.update(val_metrics)
            f.write(json.dumps(row) + "\n")

        # Save last
        torch.save(model.head.state_dict(), out_dir / f"timer_head_epoch{epoch}.pt")

        # Save best
        if val_metrics["loss_log_huber"] < best:
            best = val_metrics["loss_log_huber"]
            torch.save(model.head.state_dict(), out_dir / "timer_head_best.pt")
            print(f"Saved best: {out_dir/'timer_head_best.pt'} (val_loss_log_huber={best:.4f})")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    # --hook_layer last_ln
    ap.add_argument("--gm_name", type=str, required=True)
    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"])

    # Data: either provide train/val explicitly OR a single file + val split
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    

    # Feature extraction
    ap.add_argument("--hook_layer", type=str, default="last_ln", help="Optional dotted module name inside Maia2 to hook for features.")
    ap.add_argument("--logits_feature", type=str, default="masked_logits", choices=["masked_logits", "logprobs"])

    # Head sizing (small by default)
    ap.add_argument("--head_hidden1", type=int, default=128)
    ap.add_argument("--head_hidden2", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()
    set_seed(args.seed)

    policy_ckpt = Path(f"./processed/single_gm/train_val/{args.gm_name}/policy_dpo_best.pt")
    out_dir = Path(f"./processed/single_gm/time_per_move/train_val/{args.gm_name}")
    train_jsonl = Path(f"./processed/single_gm/time_per_move/train_val/{args.gm_name}/{args.gm_name}_train.jsonl")
    val_jsonl = Path(f"./processed/single_gm/time_per_move/train_val/{args.gm_name}/{args.gm_name}_val.jsonl")

    metrics_out_dir = Path(f"./processed/single_gm/time_per_move/train_val/{args.gm_name}/metrics")
    metrics_out_dir.mkdir(parents=True, exist_ok=True)

    device = device_from_str(args.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load Maia2 + your fine-tuned weights
    # ----------------------------
    maia = maia_model.from_pretrained(type=args.maia_type, device=str(device))
    maia.to(device)

    ckpt = torch.load(policy_ckpt, map_location=device)
    maia.load_state_dict(ckpt, strict=False)
    print(f"Loaded policy weights: {policy_ckpt}")

    # prepare() provides vocab + elo mapping
    all_moves_dict, elo_dict, _ = inference.prepare()

    feat_cfg = FeatureConfig(
        hook_layer=args.hook_layer.strip() or None,
        use_logits_fallback=True,
        logits_feature=args.logits_feature,
    )
    featurizer = FrozenMaia2Featurizer(
        maia2_model=maia,
        all_moves_dict=all_moves_dict,
        elo_dict=elo_dict,
        device=device,
        cfg=feat_cfg,
    )

    # Infer feature dim with a tiny dry-run
    with torch.no_grad():
        dummy = {
            "fen": ["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"],
            "elo_self": [2800],
            "elo_oppo": [2800],
        }
        f = featurizer(dummy["fen"], dummy["elo_self"], dummy["elo_oppo"])
        feat_dim = int(f.shape[-1])
    in_dim = feat_dim + 5 + 2 # + prev5 (log1p) features + clock-left (log1p) features + ply index (scaled)
    print(f"[timer] feature_dim={feat_dim}  in_dim={in_dim}")

    head = TimerHead(
        in_dim=in_dim,
        hidden1=args.head_hidden1,
        hidden2=args.head_hidden2,
        dropout=args.dropout,
    ).to(device)

    model = TimerModel(featurizer=featurizer, head=head).to(device)

    # ----------------------------
    # Load data
    # ----------------------------
    if train_jsonl and val_jsonl:
        train_path = Path(train_jsonl)
        val_path = Path(val_jsonl)
        train_ds = TimingJsonl(train_path)
        val_ds = TimingJsonl(val_path)
    else:
        raise SystemExit("Provide either --train_jsonl and --val_jsonl, or --data_jsonl to split.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_timing,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_timing,
        pin_memory=(device.type == "cuda"),
    )

    # ----------------------------
    # Train
    # ----------------------------
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=out_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        metrics_out_dir=metrics_out_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
