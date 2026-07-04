from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import orjson
import torch
import torch.nn.functional as F

from grandmaster_dpo.train.style_embeddings_for_gms.dataset_schema import TrainConfig
from grandmaster_dpo.train.style_embeddings_for_gms.train_style_encoder import StyleEncoder
from grandmaster_dpo.utilities.jsonl_io import open_jsonl_binary, sorted_jsonl_paths
from grandmaster_dpo.utilities.shared_style_emb_model_utils import (
    model_variant_uses_game_type,
    model_variant_uses_opponent_context,
    pick_device,
    raw_example_to_cached_arrays,
    resolve_checkpoint,
)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ============================================================
# IEEE CoG paper plotting defaults
# ============================================================

plt.rcParams.update(
    {
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.size": 7,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "lines.linewidth": 1.4,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    }
)

PAIR_VARIANT_COLORS = {
    "v1": "#2F6BFF",
    "v2": "#00A6A6",
    "v3": "#E67E22",
}

MODEL_VARIANT_COLORS = {
    "phi0": "#2F6BFF",
    "phi1": "#7E57C2",
    "phi2": "#00A6A6",
    "phi3": "#7D6608",
}

TAU_COLORS = {
    0.05: "#FAD7D7",
    0.10: "#F5B7B1",
    0.25: "#F1948A",
    0.75: "#EC7063",
    1.25: "#E74C3C",
    1.75: "#CA6F1E",
    2.25: "#7D6608",
}

DEFAULT_COLOR = "#999999"


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def finish_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_color(meta: Dict[str, Any]) -> str:
    pair_variant = meta.get("pair_variant")
    if pair_variant in PAIR_VARIANT_COLORS:
        return PAIR_VARIANT_COLORS[pair_variant]

    model_variant = meta.get("model_variant")
    if model_variant in MODEL_VARIANT_COLORS:
        return MODEL_VARIANT_COLORS[model_variant]

    tau = meta.get("train_tau")
    if tau is not None:
        try:
            tau = round(float(tau), 2)
            if tau in TAU_COLORS:
                return TAU_COLORS[tau]
        except Exception:
            pass

    return DEFAULT_COLOR


DEFAULT_LABEL_PLAYERS = ["Kasparov, G.", "Anand, V.", "Karpov, Ana", 
    "Karpov, A.", "Kramnik, V.", "Kramnik, V", "Nepomniachtchi, I.",
    "Nepomniachtchi, I", "Topalov, V.", "Topalov, V", "Carlsen, M.",
    "Niemann, H.", "Niemann, H", "Caruana, F.", "Caruana, F.",
    "Carlsen, M", "Caruana, F", "Ding, L.", "Ding, L", "Giri, A.", "Giri, A",
    "Firouzja, A.", "Firouzja, A", "So, W.", "So, W",
]

DEFAULT_METRICS = [
    ("retrieval.recall_at_1", "Recall@1"),
    ("retrieval.recall_at_5", "Recall@5"),
    ("retrieval.mrr", "MRR"),
    ("classification.roc_auc", "ROC AUC"),
    ("classification.average_precision", "Average Precision"),
    ("classification.best_f1", "Best F1"),
    ("spread.spread_ratio_mean", "Spread Ratio"),
    ("pair.row_cos_mean_gap", "Row Cosine Mean Gap"),
    ("pair.row_cos_hard_gap", "Row Cosine Hard Gap"),
    ("pair.pair_acc_mean_vs_hardest", "Pair Acc (mean pos > hardest neg)"),
]


@dataclass
class EvalRun:
    run_dir: Path
    split: str
    run_name: str
    pair_metrics: Dict[str, Any]
    retrieval_metrics: Dict[str, Any]
    classification_metrics: Dict[str, Any]
    spread_metrics: Dict[str, Any]
    manifest: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class TrainingCurve:
    path: Path
    name: str
    meta: Dict[str, Any]
    x: np.ndarray
    train_loss: np.ndarray
    eval_loss: Optional[np.ndarray]


@dataclass
class PlayerExample:
    player_id: str
    game_id: str
    example: Dict[str, Any]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def parse_float_from_name(patterns: Sequence[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1).replace("_", "."))
    return None


def parse_int_from_name(patterns: Sequence[str], text: str) -> Optional[int]:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None


def parse_model_metadata(name: str, manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "pair_variant": None,
        "model_variant": None,
        "train_tau": None,
        "embedding_dim": None,
        "batch_size": None,
        "lr": None,
        "seed": None,
    }

    manifest = manifest or {}
    if manifest:
        meta["pair_variant"] = manifest.get("pair_variant")
        meta["model_variant"] = manifest.get("model_variant")
        train_tau = manifest.get("train_tau")
        meta["train_tau"] = float(train_tau) if train_tau is not None else None

    m = re.search(r"pair[-_](v\d+)", name)
    if m and meta["pair_variant"] is None:
        meta["pair_variant"] = m.group(1)

    m = re.search(r"phi[-_](phi\d+)", name)
    if m and meta["model_variant"] is None:
        meta["model_variant"] = m.group(1)
    if meta["model_variant"] is None:
        m = re.search(r"_(phi\d+)(?:_|$)", name)
        if m:
            meta["model_variant"] = m.group(1)

    if meta["train_tau"] is None:
        meta["train_tau"] = parse_float_from_name(
            [r"tau[-_](\d+(?:[._]\d+)?)", r"tau(\d+(?:_\d+)?)"],
            name,
        )

    meta["embedding_dim"] = parse_int_from_name([r"edim[-_](\d+)"], name)
    meta["batch_size"] = parse_int_from_name([r"bs[-_](\d+)"], name)
    meta["lr"] = parse_float_from_name([r"lr[-_](\d+(?:[._]\d+)?)"], name)
    meta["seed"] = parse_int_from_name([r"seed[-_](\d+)"], name)

    label_parts: List[str] = []
    if meta["pair_variant"]:
        label_parts.append(str(meta["pair_variant"]))
    if meta["model_variant"]:
        label_parts.append(str(meta["model_variant"]))
    if meta["train_tau"] is not None:
        label_parts.append(f"τ={meta['train_tau']}")
    meta["short_label"] = " | ".join(label_parts) if label_parts else name
    return meta


def discover_eval_runs(eval_root: Path, split: str) -> List[EvalRun]:
    runs: List[EvalRun] = []
    for run_dir in sorted([p for p in eval_root.iterdir() if p.is_dir()]):
        split_dir = run_dir / split
        pair_path = split_dir / "pair_metrics.json"
        retrieval_path = split_dir / "retrieval_metrics.json"
        class_path = split_dir / "classification_metrics.json"
        spread_path = split_dir / "spread_metrics.json"
        if not (pair_path.exists() and retrieval_path.exists() and class_path.exists() and spread_path.exists()):
            continue
        manifest_path = run_dir / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        meta = parse_model_metadata(run_dir.name, manifest)
        runs.append(
            EvalRun(
                run_dir=run_dir,
                split=split,
                run_name=run_dir.name,
                pair_metrics=read_json(pair_path),
                retrieval_metrics=read_json(retrieval_path),
                classification_metrics=read_json(class_path),
                spread_metrics=read_json(spread_path),
                manifest=manifest,
                meta=meta,
            )
        )
    if not runs:
        raise FileNotFoundError(
            f"No eval runs found under {eval_root} with split='{split}' and the expected metrics JSON files."
        )
    return runs


def nested_get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def flatten_eval_run(run: EvalRun) -> Dict[str, Any]:
    pair_row = run.pair_metrics.get("row_aggregated", {}).get("cosine", {})
    spread = run.spread_metrics
    train_tau = run.meta.get("train_tau")
    tau_metrics = None
    if train_tau is not None:
        tau_metrics = run.pair_metrics.get("row_aggregated", {}).get("by_eval_tau", {}).get(str(train_tau))
    if tau_metrics is None:
        by_tau = run.pair_metrics.get("row_aggregated", {}).get("by_eval_tau", {})
        if by_tau:
            first_key = sorted(by_tau.keys(), key=lambda x: float(x))[0]
            tau_metrics = by_tau[first_key]

    row: Dict[str, Any] = {
        "run_name": run.run_name,
        "run_dir": str(run.run_dir),
        "short_label": run.meta.get("short_label", run.run_name),
        **run.meta,
        "retrieval.recall_at_1": nested_get(run.retrieval_metrics, "recall_at_1"),
        "retrieval.recall_at_5": nested_get(run.retrieval_metrics, "recall_at_5"),
        "retrieval.mrr": nested_get(run.retrieval_metrics, "mrr"),
        "classification.roc_auc": nested_get(run.classification_metrics, "roc_auc"),
        "classification.average_precision": nested_get(run.classification_metrics, "average_precision"),
        "classification.best_f1": nested_get(run.classification_metrics, "best_f1"),
        "spread.spread_ratio_mean": nested_get(spread, "spread_ratio_mean"),
        "spread.n_players": nested_get(spread, "n_players"),
        "spread.intra_mean": nested_get(spread, "intra_player_spread.intra_player_spread_mean"),
        "spread.inter_mean": nested_get(spread, "inter_player_centroid_distance.inter_player_centroid_distance_mean"),
        "pair.row_cos_mean_gap": pair_row.get("mean_gap"),
        "pair.row_cos_hard_gap": pair_row.get("hard_gap"),
        "pair.pair_acc_mean_vs_hardest": pair_row.get("pair_acc_mean_vs_hardest"),
        "pair.candidate_cos_gap": nested_get(run.pair_metrics, "candidate_level.cosine.gap_mean"),
    }
    if tau_metrics is not None:
        row["pair.dot_over_tau_mean_gap"] = nested_get(tau_metrics, "dot_over_tau.mean_gap")
        row["pair.dot_over_tau_hard_gap"] = nested_get(tau_metrics, "dot_over_tau.hard_gap")
        row["pair.infonce_like_loss"] = nested_get(tau_metrics, "infonce_like_loss_mean_vs_hardest_neg")
    return row


def maybe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        out = float(x)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def load_training_curve(path: Path) -> Optional[TrainingCurve]:
    rows: List[Dict[str, Any]] = []
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            obj = orjson.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    if not rows:
        return None

    x_candidates = ["step", "global_step", "train_step", "update", "iteration", "iter", "epoch"]
    train_candidates = [
        "train_loss",
        "loss",
        "avg_loss",
        "mean_loss",
        "objective",
        "train.total_loss",
    ]
    eval_candidates = ["eval_loss", "val_loss", "validation_loss", "dev_loss"]

    def pick_value(obj: Dict[str, Any], candidates: Sequence[str], default: Optional[float]) -> Optional[float]:
        for key in candidates:
            val = nested_get(obj, key)
            out = maybe_float(val)
            if out is not None:
                return out
        return default

    x_vals: List[float] = []
    train_vals: List[float] = []
    eval_vals: List[float] = []
    have_eval = False
    for idx, row in enumerate(rows):
        x = pick_value(row, x_candidates, float(idx))
        train = pick_value(row, train_candidates, None)
        eval_v = pick_value(row, eval_candidates, None)
        if x is None or train is None:
            continue
        x_vals.append(x)
        train_vals.append(train)
        if eval_v is not None:
            eval_vals.append(eval_v)
            have_eval = True
        else:
            eval_vals.append(np.nan)

    if not x_vals:
        return None

    meta = parse_model_metadata(path.stem)
    return TrainingCurve(
        path=path,
        name=path.stem,
        meta=meta,
        x=np.asarray(x_vals, dtype=np.float32),
        train_loss=np.asarray(train_vals, dtype=np.float32),
        eval_loss=np.asarray(eval_vals, dtype=np.float32) if have_eval else None,
    )


def load_training_curves(training_summary_dir: Path) -> List[TrainingCurve]:
    curves: List[TrainingCurve] = []
    for path in sorted(training_summary_dir.glob("*.jsonl")):
        curve = load_training_curve(path)
        if curve is not None:
            curves.append(curve)
    return curves


def sort_rows_for_plot(rows: List[Dict[str, Any]], metric_key: str) -> List[Dict[str, Any]]:
    def key_fn(r: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            str(r.get("pair_variant") or "zzz"),
            str(r.get("model_variant") or "zzz"),
            float(r.get("train_tau")) if r.get("train_tau") is not None else 1e9,
            -(float(r.get(metric_key)) if r.get(metric_key) is not None else -1e9),
            str(r.get("run_name")),
        )
    return sorted(rows, key=key_fn)


def plot_metric_bars(rows: List[Dict[str, Any]], metric_key: str, metric_label: str, output_path: Path) -> None:
    rows = [r for r in rows if maybe_float(r.get(metric_key)) is not None]
    if not rows:
        return
    rows = sort_rows_for_plot(rows, metric_key)
    labels = [r.get("short_label", r["run_name"]) for r in rows]
    vals = [float(r[metric_key]) for r in rows]

    fig_h = max(4.5, 0.35 * len(rows))
    plt.figure(figsize=(11, fig_h))
    y = np.arange(len(rows))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.xlabel(metric_label)
    plt.title(f"{metric_label} ({rows[0].get('split', 'metrics') if rows else 'metrics'})")
    for yi, val in zip(y, vals):
        plt.text(val, yi, f" {val:.4f}", va="center")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_loss_curves(curves: List[TrainingCurve], output_dir: Path, overlay_limit: int) -> List[str]:
    created: List[str] = []
    if not curves:
        return created

    indiv_dir = output_dir / "loss_curves"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    for curve in curves:
        plt.figure(figsize=(9, 5))
        plt.plot(curve.x, curve.train_loss, label="train_loss")
        if curve.eval_loss is not None:
            plt.plot(curve.x, curve.eval_loss, label="eval_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(curve.meta.get("short_label", curve.name))
        plt.legend()
        plt.tight_layout()
        out_path = indiv_dir / f"{slugify(curve.name)}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        created.append(str(out_path))

    chosen = curves[:overlay_limit]
    plt.figure(figsize=(10, 6))
    for curve in chosen:
        plt.plot(curve.x, curve.train_loss, label=curve.meta.get("short_label", curve.name))
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.title("Training loss curves (overlay)")
    if len(chosen) <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    overlay_path = output_dir / "loss_curves_overlay.png"
    plt.savefig(overlay_path, dpi=180, bbox_inches="tight")
    plt.close()
    created.append(str(overlay_path))
    return created


def plot_margin_distribution_bands(runs: List[EvalRun], output_path: Path, use_train_tau: bool) -> None:
    labels: List[str] = []
    pos_p10: List[float] = []
    pos_p50: List[float] = []
    pos_p90: List[float] = []
    neg_p10: List[float] = []
    neg_p50: List[float] = []
    neg_p90: List[float] = []

    for run in sort_rows_for_plot([flatten_eval_run(r) for r in runs], "pair.row_cos_mean_gap"):
        src_run = next(r for r in runs if r.run_name == run["run_name"])
        if use_train_tau:
            tau = src_run.meta.get("train_tau")
            by_tau = src_run.pair_metrics.get("row_aggregated", {}).get("by_eval_tau", {})
            tau_block = by_tau.get(str(tau)) if tau is not None else None
            if tau_block is None and by_tau:
                first_key = sorted(by_tau.keys(), key=lambda x: float(x))[0]
                tau_block = by_tau[first_key]
            if tau_block is None:
                continue
            pos = nested_get(tau_block, "dot_over_tau.mean_positive")
            neg = nested_get(tau_block, "dot_over_tau.hardest_negative")
            title = "Margin distribution bands (dot/τ, mean positive vs hardest negative)"
            xlabel = "dot/τ"
        else:
            pos = nested_get(src_run.pair_metrics, "row_aggregated.cosine.mean_positive")
            neg = nested_get(src_run.pair_metrics, "row_aggregated.cosine.hardest_negative")
            title = "Margin distribution bands (row cosine: mean positive vs hardest negative)"
            xlabel = "cosine"

        if not isinstance(pos, dict) or not isinstance(neg, dict):
            continue
        labels.append(src_run.meta.get("short_label", src_run.run_name))
        pos_p10.append(float(pos[next(k for k in pos if k.endswith("_p10"))]))
        pos_p50.append(float(pos[next(k for k in pos if k.endswith("_p50"))]))
        pos_p90.append(float(pos[next(k for k in pos if k.endswith("_p90"))]))
        neg_p10.append(float(neg[next(k for k in neg if k.endswith("_p10"))]))
        neg_p50.append(float(neg[next(k for k in neg if k.endswith("_p50"))]))
        neg_p90.append(float(neg[next(k for k in neg if k.endswith("_p90"))]))

    if not labels:
        return

    y = np.arange(len(labels))
    plt.figure(figsize=(12, max(4.5, 0.4 * len(labels))))
    plt.hlines(y, neg_p10, neg_p90, linewidth=3, label="hardest negative p10-p90")
    plt.scatter(neg_p50, y, marker="x", s=40, label="hardest negative p50")
    plt.hlines(y, pos_p10, pos_p90, linewidth=3, label="mean positive p10-p90")
    plt.scatter(pos_p50, y, marker="o", s=35, label="mean positive p50")
    plt.yticks(y, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_spread_histograms(runs: List[EvalRun], output_path: Path, limit: int) -> None:
    chosen = runs[:limit]
    if not chosen:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    any_data = False
    for run in chosen:
        players = run.spread_metrics.get("players", {})
        vals = [maybe_float(v.get("intra_spread")) for v in players.values() if isinstance(v, dict)]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            continue
        any_data = True
        ax.hist(
            vals,
            bins=24,
            histtype="step",
            linewidth=1.4,
            label=run.meta.get("short_label", run.run_name),
            color=run_color(run.meta),
        )
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("Per-player intra spread")
    ax.set_ylabel("Count")
    ax.set_title("Spread histograms")
    style_axes(ax)
    if len(chosen) <= 6:
        ax.legend()
    finish_figure(fig, output_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[Dict[str, Any], TrainConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = TrainConfig.from_dict(ckpt["config"])
    return ckpt, cfg


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[StyleEncoder, TrainConfig, Dict[str, Any]]:
    ckpt, cfg = load_checkpoint(checkpoint_path, device)
    model = StyleEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def iter_jsonl_rows(input_dir: Path, max_rows: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    seen = 0
    for path in sorted_jsonl_paths(input_dir):
        with open_jsonl_binary(path) as f:
            for line in f:
                if not line.strip():
                    continue
                yield orjson.loads(line)
                seen += 1
                if max_rows is not None and seen >= max_rows:
                    return


def compact_arrays_to_batch(
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
    if model_variant_uses_game_type(variant_name):
        out["game_type"] = torch.from_numpy(game_types_u8.astype(np.int64, copy=False)).to(device)
    if model_variant_uses_opponent_context(variant_name):
        out["opponent_context"] = torch.zeros((boards.shape[0], 32), dtype=torch.float32, device=device)
    return out


@torch.no_grad()
def encode_raw_examples(
    model: torch.nn.Module,
    variant_name: str,
    examples: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not examples:
        return np.zeros((0, 1), dtype=np.float32)

    board_list: List[np.ndarray] = []
    move_list: List[np.ndarray] = []
    game_type_list: List[np.uint8] = []
    for ex in examples:
        boards, move, game_type = raw_example_to_cached_arrays(ex)
        board_list.append(boards)
        move_list.append(move)
        game_type_list.append(game_type)

    boards_arr = np.stack(board_list, axis=0).astype(np.uint8, copy=False)
    moves_arr = np.stack(move_list, axis=0).astype(np.uint8, copy=False)
    game_types_arr = np.asarray(game_type_list, dtype=np.uint8)

    chunks: List[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        end = min(start + batch_size, len(examples))
        feats = compact_arrays_to_batch(
            boards_arr[start:end],
            moves_arr[start:end],
            game_types_arr[start:end],
            variant_name=variant_name,
            device=device,
        )
        z = model(feats).detach().cpu().numpy().astype(np.float32)
        chunks.append(z)
    emb = np.concatenate(chunks, axis=0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return emb / norms


def reservoir_replace(example: Dict[str, Any], current: Optional[Dict[str, Any]], seen_count: int, rng: np.random.Generator) -> Dict[str, Any]:
    if current is None:
        return example
    if rng.integers(0, seen_count) == 0:
        return example
    return current


def collect_pca_examples(
    pairs_split_dir: Path,
    n_players: int,
    samples_per_player: int,
    label_players: Sequence[str],
    seed: int,
    max_rows: Optional[int],
    player_selection: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    per_player_per_game: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    per_player_game_seen: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in iter_jsonl_rows(pairs_split_dir, max_rows=max_rows):
        ex = row.get("anchor")
        if not isinstance(ex, dict):
            continue
        pid = ex.get("player_id")
        gid = ex.get("game_id")
        if not pid or not gid:
            continue
        seen = per_player_game_seen[pid][gid] + 1
        per_player_game_seen[pid][gid] = seen
        cur = per_player_per_game[pid].get(gid)
        per_player_per_game[pid][gid] = reservoir_replace(ex, cur, seen, rng)

    eligible = [pid for pid, by_game in per_player_per_game.items() if len(by_game) >= samples_per_player]
    chosen: List[str] = []

    for pid in label_players:
        if pid in eligible and pid not in chosen:
            chosen.append(pid)

    remaining = [pid for pid in eligible if pid not in chosen]
    if player_selection == "most_games":
        remaining.sort(key=lambda pid: len(per_player_per_game[pid]), reverse=True)
    else:
        rng.shuffle(remaining)

    for pid in remaining:
        if len(chosen) >= n_players:
            break
        chosen.append(pid)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for pid in chosen:
        games = list(per_player_per_game[pid].keys())
        rng.shuffle(games)
        selected_games = games[:samples_per_player]
        out[pid] = [per_player_per_game[pid][gid] for gid in selected_games]

    stats = {
        "n_players_requested": n_players,
        "samples_per_player_requested": samples_per_player,
        "n_players_eligible": len(eligible),
        "n_players_selected": len(out),
        "selected_players": list(out.keys()),
        "missing_label_players": [pid for pid in label_players if pid not in out],
        "player_game_counts": {pid: len(per_player_per_game[pid]) for pid in out},
        "sampling_note": "Uses at most one anchor example per (player_id, game_id), then samples games randomly per player.",
    }
    return out, stats


def pca_2d(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(x_centered, full_matrices=False)
    if x.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    comps = u[:, :2] * s[:2]
    if comps.shape[1] == 1:
        comps = np.concatenate([comps, np.zeros((comps.shape[0], 1), dtype=comps.dtype)], axis=1)
    return comps.astype(np.float32, copy=False)


def plot_player_pca(
    model_dir: Path,
    checkpoint_name: str,
    pairs_split_dir: Path,
    output_dir: Path,
    split_name: str,
    n_players: int,
    samples_per_player: int,
    label_players: Sequence[str],
    seed: int,
    device: torch.device,
    batch_size: int,
    max_rows: Optional[int],
    player_selection: str,
) -> Optional[Dict[str, Any]]:
    checkpoint_path = resolve_checkpoint(model_dir, checkpoint_name)
    model, cfg, ckpt = build_model_from_checkpoint(checkpoint_path, device)
    sampled, sample_stats = collect_pca_examples(
        pairs_split_dir=pairs_split_dir,
        n_players=n_players,
        samples_per_player=samples_per_player,
        label_players=label_players,
        seed=seed,
        max_rows=max_rows,
        player_selection=player_selection,
    )
    if not sampled:
        return None

    players = list(sampled.keys())
    centroids: List[np.ndarray] = []
    per_player_mean_norm: Dict[str, float] = {}
    for pid in players:
        emb = encode_raw_examples(
            model=model,
            variant_name=cfg.model.variant_name,
            examples=sampled[pid],
            device=device,
            batch_size=batch_size,
        )
        centroid = emb.mean(axis=0, keepdims=True)
        centroid = centroid / np.clip(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-12, None)
        centroids.append(centroid[0])
        per_player_mean_norm[pid] = float(np.linalg.norm(emb.mean(axis=0)))

    centroid_mat = np.stack(centroids, axis=0)
    points = pca_2d(centroid_mat)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"pca_player_plot_{split_name}.png"
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    marker_sizes = []
    for pid in players:
        player_vals = sampled[pid]
        n = len(player_vals)
        marker_sizes.append(16 + 0.6 * n)
    ax.scatter(points[:, 0], points[:, 1], s=np.asarray(marker_sizes), color="#B0B0B0", alpha=0.8)

    highlighted = set(label_players)
    for i, pid in enumerate(players):
        if pid in highlighted:
            ax.scatter(points[i, 0], points[i, 1], s=marker_sizes[i] * 1.2, color="#C0392B", alpha=0.95)
            ax.annotate(pid, (points[i, 0], points[i, 1]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of player centroids ({split_name})")
    style_axes(ax)
    finish_figure(fig, fig_path)

    stats = {
        "figure_path": str(fig_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "model_variant": cfg.model.variant_name,
        "pair_variant": getattr(cfg, "pair_variant", None),
        "player_centroids": {
            pid: {
                "pc1": float(points[i, 0]),
                "pc2": float(points[i, 1]),
                "n_samples": len(sampled[pid]),
                "mean_pre_norm_centroid_norm": per_player_mean_norm[pid],
            }
            for i, pid in enumerate(players)
        },
        **sample_stats,
    }
    write_json(output_dir / f"pca_player_plot_{split_name}.json", stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate experiment 2 paper plots from eval outputs and training summaries.")
    ap.add_argument("--eval-runs-root", type=str, required=True, help="Root dir containing per-model eval output dirs.")
    ap.add_argument("--training-summary-dir", type=str, required=True, help="Directory containing training summary JSONL files.")
    ap.add_argument("--output-dir", type=str, required=True, help="Directory for generated plots and summary JSON.")
    ap.add_argument("--split", type=str, default="test", help="Metric split to plot (default: test).")
    ap.add_argument("--loss-overlay-limit", type=int, default=12)
    ap.add_argument("--spread-hist-limit", type=int, default=8)
    ap.add_argument(
        "--ablation-metrics",
        nargs="*",
        default=[m[0] for m in DEFAULT_METRICS],
        help="Metric keys to plot. Defaults cover retrieval/classification/spread/pair metrics.",
    )

    ap.add_argument("--pairs-dir", type=str, default=None, help="Raw pairs parent dir containing eval/test/*.jsonl, used for PCA player plot.")
    ap.add_argument("--pca-model-dir", type=str, default=None, help="Model dir containing best.pt / epoch_*.pt, used for PCA player plot.")
    ap.add_argument("--checkpoint-name", type=str, default="best")
    ap.add_argument("--pca-split", type=str, default="test")
    ap.add_argument("--pca-num-players", type=int, default=100)
    ap.add_argument("--pca-samples-per-player", type=int, default=100)
    ap.add_argument("--pca-batch-size", type=int, default=2048)
    ap.add_argument("--pca-max-rows", type=int, default=None)
    ap.add_argument("--pca-player-selection", choices=["random", "most_games"], default="random")
    ap.add_argument("--label-players", nargs="*", default=DEFAULT_LABEL_PLAYERS)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eval_runs_root = Path(args.eval_runs_root)
    training_summary_dir = Path(args.training_summary_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)

    runs = discover_eval_runs(eval_runs_root, split=args.split)
    flat_rows = [flatten_eval_run(r) for r in runs]
    for row in flat_rows:
        row["split"] = args.split

    curves = load_training_curves(training_summary_dir)
    created_plots: List[str] = []
    created_plots.extend(plot_loss_curves(curves, output_dir, overlay_limit=args.loss_overlay_limit))

    metric_to_label = {k: label for k, label in DEFAULT_METRICS}
    for metric_key in args.ablation_metrics:
        metric_label = metric_to_label.get(metric_key, metric_key)
        out_path = output_dir / "ablations" / f"{slugify(metric_key)}.png"
        plot_metric_bars(flat_rows, metric_key, metric_label, out_path)
        if out_path.exists():
            created_plots.append(str(out_path))

    margin_cos_path = output_dir / "margin_distribution_cosine.png"
    plot_margin_distribution_bands(runs, margin_cos_path, use_train_tau=False)
    if margin_cos_path.exists():
        created_plots.append(str(margin_cos_path))

    margin_tau_path = output_dir / "margin_distribution_dot_over_tau.png"
    plot_margin_distribution_bands(runs, margin_tau_path, use_train_tau=True)
    if margin_tau_path.exists():
        created_plots.append(str(margin_tau_path))

    spread_hist_path = output_dir / "spread_histograms.png"
    plot_spread_histograms(runs, spread_hist_path, limit=args.spread_hist_limit)
    if spread_hist_path.exists():
        created_plots.append(str(spread_hist_path))

    pca_stats: Optional[Dict[str, Any]] = None
    if args.pairs_dir and args.pca_model_dir:
        pairs_split_dir = Path(args.pairs_dir) / args.pca_split
        if not pairs_split_dir.exists():
            raise FileNotFoundError(f"Missing PCA split dir: {pairs_split_dir}")
        pca_stats = plot_player_pca(
            model_dir=Path(args.pca_model_dir),
            checkpoint_name=args.checkpoint_name,
            pairs_split_dir=pairs_split_dir,
            output_dir=output_dir,
            split_name=args.pca_split,
            n_players=args.pca_num_players,
            samples_per_player=args.pca_samples_per_player,
            label_players=args.label_players,
            seed=args.seed,
            device=device,
            batch_size=args.pca_batch_size,
            max_rows=args.pca_max_rows,
            player_selection=args.pca_player_selection,
        )
        if pca_stats is not None:
            created_plots.append(str(output_dir / f"pca_player_plot_{args.pca_split}.png"))

    summary = {
        "eval_runs_root": str(eval_runs_root),
        "training_summary_dir": str(training_summary_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "n_eval_runs": len(runs),
        "eval_runs": [
            {
                "run_name": r.run_name,
                "run_dir": str(r.run_dir),
                **r.meta,
            }
            for r in runs
        ],
        "n_training_curves": len(curves),
        "generated_plots": created_plots,
        "pca": pca_stats,
        "notes": [
            "Loss curves are parsed heuristically from training summary JSONL rows.",
            "Margin distribution plots use the stored p10/p50/p90 summary statistics from pair_metrics.json, not raw re-encoded candidate margins.",
            "PCA player plot samples at most one anchor example per (player_id, game_id), then samples games randomly per player.",
            "The engine scatter plot is intentionally omitted per request.",
        ],
    }
    write_json(output_dir / "plot_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
