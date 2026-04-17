from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paper plotting defaults (IEEE CoG friendly)
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
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
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

GM_ORDER = [
    "carlsen",
    "caruana",
    "wei",
    "vincent",
    "nakamura",
    "gukesh",
    "giri",
    "firouzja",
    "praggnanandhaa",
]

DEPTHS = [2, 4, 6, 8, 10, 12]
TEMPS = [0.25, 0.50, 0.75, 1.00, 1.50]
REFERENCE_RANK_BUCKETS = [1, 3, 5, 10]
REFERENCE_CP_GAP_BUCKETS = [10, 20, 40, 80, 160, 320, 640]

# Edit these after you inspect the first few plots.
BEST_TEMPS: Dict[str, float] = {
    "dpo": 1.00,
    "sft_and_dpo": 1.00,
    "sft_and_dpo_w_style_v3": 1.00,
    "sft": 1.00,
    "sft_pairwise": 1.00,
}

FIXED_DEPTH_FOR_TEMP_SWEEP = 10

FAMILY_ORDER = [
    "sft",
    "sft_pairwise",
    "dpo",
    "sft_and_dpo",
    "sft_and_dpo_w_style_v3",
]

FAMILY_LABELS = {
    "sft": "NLL",
    "sft_pairwise": "Pairwise",
    "dpo": "DPO",
    "sft_and_dpo": "NLL + DPO",
    "sft_and_dpo_w_style_v3": "NLL + DPO + Style-v3",
}

FAMILY_COLORS = {
    "sft": "#00A6A6",
    "sft_pairwise": "#7E57C2",
    "dpo": "#C0392B",
    "sft_and_dpo": "#F39C12",
    "sft_and_dpo_w_style_v3": "#6E2C00",
}

DEPTH_LINESTYLES = {
    2: "-",
    4: (0, (5, 1)),
    6: (0, (3, 1, 1, 1)),
    8: (0, (2, 1)),
    10: (0, (5, 2, 1, 2)),
    12: (0, (1, 1)),
}

T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


# ============================================================
# Parsing / discovery
# ============================================================

TAG_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "dpo",
        re.compile(
            r"^dpo_w_sf_depth_(?P<depth>\d+)_pv_(?P<pv>\d+)_cp_w_(?P<cpw>\d+)_(?P<gibbs>True|False)(?:_t(?P<temp>[0-9.]+))?$"
        ),
    ),
    (
        "sft_and_dpo",
        re.compile(
            r"^sft_and_dpo_beta=(?P<beta>[0-9.]+)_d_(?P<depth>\d+)_pv_(?P<pv>\d+)_cp_w_(?P<cpw>\d+)_(?P<gibbs>True|False)(?:_t(?P<temp>[0-9.]+))?$"
        ),
    ),
    (
        "sft_and_dpo_w_style_v3",
        re.compile(
            r"^sft_and_dpo_w_style_v3_d_(?P<depth>\d+)_pv_(?P<pv>\d+)_cp_w_(?P<cpw>\d+)_(?P<gibbs>True|False)(?:_t(?P<temp>[0-9.]+))?(?:_.*)?$"
        ),
    ),
    (
        "sft_pairwise",
        re.compile(
            r"^sft_pairwise_d_(?P<depth>\d+)_pv_(?P<pv>\d+)_cp_w_(?P<cpw>\d+)_(?P<gibbs>True|False)(?:_t(?P<temp>[0-9.]+))?$"
        ),
    ),
    (
        "sft",
        re.compile(
            r"^sft_d_(?P<depth>\d+)_pv_(?P<pv>\d+)_cp_w_(?P<cpw>\d+)_(?P<gibbs>True|False)(?:_t(?P<temp>[0-9.]+))?$"
        ),
    ),
]


@dataclass(frozen=True)
class RunKey:
    gm: str
    family: str
    depth: int
    gibbs: bool
    temperature: float
    tag: str


@dataclass
class RunRecord:
    key: RunKey
    run_dir: Path
    summary_path: Path
    summary: Dict[str, Any]


def parse_tag(tag: str) -> Optional[RunKey]:
    for family, pattern in TAG_PATTERNS:
        m = pattern.match(tag)
        if not m:
            continue
        gd = m.groupdict()
        depth = int(gd["depth"])
        gibbs = gd["gibbs"] == "True"
        temp = float(gd["temp"]) if gd.get("temp") else 1.00
        return RunKey(
            gm="",
            family=family,
            depth=depth,
            gibbs=gibbs,
            temperature=temp,
            tag=tag,
        )
    return None


def _find_summary_json(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("eval_results__*.json"))
    if candidates:
        return candidates[0]
    return None


def discover_runs(eval_root: Path, gm_order: Sequence[str]) -> List[RunRecord]:
    records: List[RunRecord] = []
    for gm in gm_order:
        gm_root = eval_root / gm / "family_eval_val"
        if not gm_root.exists():
            print(f"[WARN] Missing GM folder: {gm_root}")
            continue
        for run_dir in sorted(gm_root.iterdir()):
            if not run_dir.is_dir():
                continue
            parsed = parse_tag(run_dir.name)
            if parsed is None:
                continue
            summary_path = _find_summary_json(run_dir)
            if summary_path is None:
                print(f"[WARN] Missing summary JSON inside {run_dir}")
                continue
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            key = RunKey(
                gm=gm,
                family=parsed.family,
                depth=parsed.depth,
                gibbs=parsed.gibbs,
                temperature=parsed.temperature,
                tag=parsed.tag,
            )
            records.append(
                RunRecord(
                    key=key,
                    run_dir=run_dir,
                    summary_path=summary_path,
                    summary=summary,
                )
            )
    return records


# ============================================================
# Metric extraction
# ============================================================


def _safe_get(d: Dict[str, Any], path: Sequence[str], default: float = float("nan")) -> float:
    cur: Any = d
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def get_main_metric(summary: Dict[str, Any], metric_name: str) -> float:
    # Use inference-conditioned Stockfish summary when available.
    if metric_name == "p_chosen_cond_inference":
        return _safe_get(summary, ["stockfish", "metrics", "p_chosen_cond_inference", "mean"])
    if metric_name == "mrr_cond_inference":
        return _safe_get(summary, ["stockfish", "metrics", "mrr", "mean"])
    if metric_name == "top1_cond_inference":
        return _safe_get(summary, ["stockfish", "metrics", "cand_hit1", "mean"])
    if metric_name == "top3_cond_inference":
        return _safe_get(summary, ["stockfish", "metrics", "cand_hit3", "mean"])
    if metric_name == "top5_cond_inference":
        return _safe_get(summary, ["stockfish", "metrics", "cand_hit5", "mean"])
    if metric_name == "top10_cond_inference":
        return _safe_get(summary, ["stockfish", "metrics", "cand_hit10", "mean"])
    raise KeyError(f"Unsupported metric_name: {metric_name}")


def get_rank_bucket_metric(
    summary: Dict[str, Any],
    rank_bucket: int,
    metric_name: str = "p_chosen_cond_inference",
    bucket_kind: str = "exact",
    engine_source: str = "reference",
) -> float:
    if bucket_kind == "exact":
        path = [
            "stockfish",
            "chosen_quality_by_rank_exact",
            engine_source,
            str(rank_bucket),
            metric_name,
            "mean",
        ]
    elif bucket_kind == "leq":
        path = [
            "stockfish",
            "chosen_quality_by_rank_leq",
            engine_source,
            str(rank_bucket),
            metric_name,
            "mean",
        ]
    else:
        raise ValueError(f"Unsupported bucket_kind: {bucket_kind}")
    return _safe_get(summary, path)


def get_cp_gap_bucket_metric(
    summary: Dict[str, Any],
    cp_gap_bucket: int,
    metric_name: str = "p_chosen_cond_inference",
    engine_source: str = "reference",
) -> float:
    return _safe_get(
        summary,
        [
            "stockfish",
            "chosen_quality_by_cp_gap_to_best",
            engine_source,
            str(cp_gap_bucket),
            metric_name,
            "mean",
        ],
    )


# ============================================================
# Aggregation
# ============================================================


def t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    df = n - 1
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    return 1.96


def mean_ci(values: Sequence[float]) -> Tuple[float, float, float, int]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean, 1
    sem = float(arr.std(ddof=1) / math.sqrt(n))
    half = t_critical_95(n) * sem
    return mean, mean - half, mean + half, n


def build_records_table(records: Sequence[RunRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rows.append(
            {
                "gm": rec.key.gm,
                "family": rec.key.family,
                "depth": rec.key.depth,
                "gibbs": rec.key.gibbs,
                "temperature": rec.key.temperature,
                "tag": rec.key.tag,
                "summary_path": str(rec.summary_path),
                "p_chosen_cond_inference": get_main_metric(rec.summary, "p_chosen_cond_inference"),
                "mrr_cond_inference": get_main_metric(rec.summary, "mrr_cond_inference"),
                "top1_cond_inference": get_main_metric(rec.summary, "top1_cond_inference"),
                "top3_cond_inference": get_main_metric(rec.summary, "top3_cond_inference"),
                "top5_cond_inference": get_main_metric(rec.summary, "top5_cond_inference"),
                "top10_cond_inference": get_main_metric(rec.summary, "top10_cond_inference"),
                **{
                    f"rank_exact_ref_{r}_pchosen": get_rank_bucket_metric(
                        rec.summary,
                        rank_bucket=r,
                        metric_name="p_chosen_cond_inference",
                        bucket_kind="exact",
                        engine_source="reference",
                    )
                    for r in REFERENCE_RANK_BUCKETS
                },
                **{
                    f"rank_leq_ref_{r}_pchosen": get_rank_bucket_metric(
                        rec.summary,
                        rank_bucket=r,
                        metric_name="p_chosen_cond_inference",
                        bucket_kind="leq",
                        engine_source="reference",
                    )
                    for r in REFERENCE_RANK_BUCKETS
                },
                **{
                    f"cp_gap_ref_{gap}_pchosen": get_cp_gap_bucket_metric(
                        rec.summary,
                        cp_gap_bucket=gap,
                        metric_name="p_chosen_cond_inference",
                        engine_source="reference",
                    )
                    for gap in REFERENCE_CP_GAP_BUCKETS
                },
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Plot helpers
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def nice_metric_label(metric: str) -> str:
    mapping = {
        "p_chosen_cond_inference": "P(chosen | inference candidates)",
        "mrr_cond_inference": "MRR (inference candidates)",
        "top1_cond_inference": "Recall@1 (inference candidates)",
        "top3_cond_inference": "Recall@3 (inference candidates)",
        "top5_cond_inference": "Recall@5 (inference candidates)",
        "top10_cond_inference": "Recall@10 (inference candidates)",
    }
    return mapping.get(metric, metric)


def plot_line_with_ci(
    ax: plt.Axes,
    x: Sequence[float],
    means: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    *,
    color: str,
    label: str,
    linestyle: Any = "-",
    marker: str = "o",
    alpha_fill: float = 0.14,
) -> None:
    x_arr = np.asarray(x, dtype=float)
    m_arr = np.asarray(means, dtype=float)
    lo_arr = np.asarray(lows, dtype=float)
    hi_arr = np.asarray(highs, dtype=float)
    mask = np.isfinite(m_arr)
    if not mask.any():
        return
    ax.plot(x_arr[mask], m_arr[mask], color=color, label=label, linestyle=linestyle, marker=marker)
    if np.isfinite(lo_arr[mask]).any() and np.isfinite(hi_arr[mask]).any():
        ax.fill_between(x_arr[mask], lo_arr[mask], hi_arr[mask], color=color, alpha=alpha_fill)


def aggregate_over_gms(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(list(group_cols), dropna=False)
    for group_key, subdf in grouped:
        values = [float(v) for v in subdf[value_col].tolist() if pd.notna(v)]
        mean, lo, hi, n = mean_ci(values)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: val for col, val in zip(group_cols, group_key)}
        row.update({"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_gms": n})
        rows.append(row)
    return pd.DataFrame(rows)


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family)


# ============================================================
# Main figures
# ============================================================


def plot_family_depth_sweep(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    metric_col: str = "p_chosen_cond_inference",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.2), sharey=True, constrained_layout=True)
    for ax, gibbs in zip(axes, [False, True]):
        for family in FAMILY_ORDER:
            temp = BEST_TEMPS[family] if gibbs else 1.00
            sub = df[
                (df["family"] == family)
                & (df["gibbs"] == gibbs)
                & np.isclose(df["temperature"], temp)
                & (df["depth"].isin(DEPTHS))
            ]
            agg = aggregate_over_gms(sub, group_cols=["depth"], value_col=metric_col).sort_values("depth")
            if agg.empty:
                continue
            plot_line_with_ci(
                ax,
                agg["depth"],
                agg["mean"],
                agg["ci_lo"],
                agg["ci_hi"],
                color=FAMILY_COLORS[family],
                label=family_label(family),
                linestyle="-",
                marker="o",
            )
        ax.set_title(f"{'Gibbs' if gibbs else 'No Gibbs'}")
        ax.set_xlabel("Inference Stockfish depth")
        ax.set_xticks(DEPTHS)
        ax.set_xlim(min(DEPTHS) - 0.4, max(DEPTHS) + 0.4)
    axes[0].set_ylabel(nice_metric_label(metric_col))
    axes[1].legend(ncol=1, loc="best")
    fig.suptitle("Depth sweep by training family", y=1.03)
    save_figure(fig, out_dir, f"depth_sweep_by_family__{metric_col}")


def plot_temp_sweep_fixed_depth(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    fixed_depth: int,
    metric_col: str = "p_chosen_cond_inference",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.2), sharey=True, constrained_layout=True)

    # Gibbs panel: actual temperature sweep.
    ax = axes[0]
    for family in FAMILY_ORDER:
        sub = df[
            (df["family"] == family)
            & (df["gibbs"] == True)
            & (df["depth"] == fixed_depth)
            & (df["temperature"].isin(TEMPS))
        ]
        agg = aggregate_over_gms(sub, group_cols=["temperature"], value_col=metric_col).sort_values("temperature")
        if agg.empty:
            continue
        plot_line_with_ci(
            ax,
            agg["temperature"],
            agg["mean"],
            agg["ci_lo"],
            agg["ci_hi"],
            color=FAMILY_COLORS[family],
            label=family_label(family),
            linestyle="-",
            marker="o",
        )
    ax.set_title(f"Gibbs, depth={fixed_depth}")
    ax.set_xlabel("Gibbs temperature")
    ax.set_xticks(TEMPS)

    # No-Gibbs panel: repeat baseline horizontally so the pair stays visually comparable.
    ax = axes[1]
    for family in FAMILY_ORDER:
        sub = df[
            (df["family"] == family)
            & (df["gibbs"] == False)
            & (df["depth"] == fixed_depth)
            & np.isclose(df["temperature"], 1.00)
        ]
        agg = aggregate_over_gms(sub, group_cols=["depth"], value_col=metric_col)
        if agg.empty:
            continue
        mean = float(agg.iloc[0]["mean"])
        lo = float(agg.iloc[0]["ci_lo"])
        hi = float(agg.iloc[0]["ci_hi"])
        xs = np.asarray(TEMPS, dtype=float)
        ax.plot(xs, np.full_like(xs, mean), color=FAMILY_COLORS[family], label=family_label(family), marker="o")
        if np.isfinite(lo) and np.isfinite(hi):
            ax.fill_between(xs, np.full_like(xs, lo), np.full_like(xs, hi), color=FAMILY_COLORS[family], alpha=0.14)
    ax.set_title(f"No Gibbs baseline, depth={fixed_depth}")
    ax.set_xlabel("Temperature axis (baseline repeated)")
    ax.set_xticks(TEMPS)

    axes[0].set_ylabel(nice_metric_label(metric_col))
    axes[1].legend(ncol=1, loc="best")
    fig.suptitle("Temperature sweep at fixed depth", y=1.03)
    save_figure(fig, out_dir, f"temp_sweep_depth_{fixed_depth}__{metric_col}")


def plot_cross_family_best_temp_depth_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    metric_col: str = "p_chosen_cond_inference",
) -> None:
    # This is the "big graph" requested: x=depth, y=p(chosen inference), lines=families.
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.2), sharey=True, constrained_layout=True)
    for ax, gibbs in zip(axes, [False, True]):
        for family in FAMILY_ORDER:
            chosen_temp = BEST_TEMPS[family] if gibbs else 1.00
            sub = df[
                (df["family"] == family)
                & (df["gibbs"] == gibbs)
                & np.isclose(df["temperature"], chosen_temp)
            ]
            agg = aggregate_over_gms(sub, group_cols=["depth"], value_col=metric_col).sort_values("depth")
            if agg.empty:
                continue
            plot_line_with_ci(
                ax,
                agg["depth"],
                agg["mean"],
                agg["ci_lo"],
                agg["ci_hi"],
                color=FAMILY_COLORS[family],
                label=family_label(family),
                linestyle="-",
                marker="o",
            )
        ax.set_title(f"{'Gibbs' if gibbs else 'No Gibbs'}")
        ax.set_xlabel("Inference Stockfish depth")
        ax.set_xticks(DEPTHS)
    axes[0].set_ylabel(nice_metric_label(metric_col))
    axes[1].legend(ncol=1, loc="best")
    fig.suptitle("Cross-family depth comparison using chosen best temperatures", y=1.03)
    save_figure(fig, out_dir, f"cross_family_best_temp_depth__{metric_col}")


def plot_reference_rank_conditioning(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    family: str,
    bucket_kind: str = "exact",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.2), sharey=True, constrained_layout=True)
    ranks = REFERENCE_RANK_BUCKETS
    col_prefix = "rank_exact_ref" if bucket_kind == "exact" else "rank_leq_ref"

    for ax, gibbs in zip(axes, [False, True]):
        chosen_temp = BEST_TEMPS[family] if gibbs else 1.00
        fam = df[
            (df["family"] == family)
            & (df["gibbs"] == gibbs)
            & np.isclose(df["temperature"], chosen_temp)
            & (df["depth"].isin(DEPTHS))
        ]
        for depth in DEPTHS:
            sub = fam[fam["depth"] == depth]
            means, los, his, xs = [], [], [], []
            for rank in ranks:
                col = f"{col_prefix}_{rank}_pchosen"
                agg = aggregate_over_gms(sub, group_cols=["depth"], value_col=col)
                if agg.empty:
                    means.append(float("nan"))
                    los.append(float("nan"))
                    his.append(float("nan"))
                else:
                    means.append(float(agg.iloc[0]["mean"]))
                    los.append(float(agg.iloc[0]["ci_lo"]))
                    his.append(float(agg.iloc[0]["ci_hi"]))
                xs.append(rank)
            plot_line_with_ci(
                ax,
                xs,
                means,
                los,
                his,
                color=FAMILY_COLORS[family],
                label=f"d={depth}",
                linestyle=DEPTH_LINESTYLES.get(depth, "-"),
                marker="o",
            )
        title_bucket = "exact rank" if bucket_kind == "exact" else "rank ≤ k"
        ax.set_title(f"{family_label(family)} — {'Gibbs' if gibbs else 'No Gibbs'}")
        ax.set_xlabel(f"Reference-engine {title_bucket}")
        ax.set_xticks(ranks)
    axes[0].set_ylabel("P(chosen | inference candidates)")
    axes[1].legend(ncol=2, loc="best")
    fig.suptitle(f"Reference-engine rank conditioning ({family_label(family)})", y=1.03)
    save_figure(fig, out_dir, f"rank_{bucket_kind}_conditioning__{family}")


def plot_reference_cp_gap_conditioning(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    family: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.2), sharey=True, constrained_layout=True)
    gaps = REFERENCE_CP_GAP_BUCKETS
    for ax, gibbs in zip(axes, [False, True]):
        chosen_temp = BEST_TEMPS[family] if gibbs else 1.00
        fam = df[
            (df["family"] == family)
            & (df["gibbs"] == gibbs)
            & np.isclose(df["temperature"], chosen_temp)
            & (df["depth"].isin(DEPTHS))
        ]
        for depth in DEPTHS:
            sub = fam[fam["depth"] == depth]
            means, los, his, xs = [], [], [], []
            for gap in gaps:
                col = f"cp_gap_ref_{gap}_pchosen"
                agg = aggregate_over_gms(sub, group_cols=["depth"], value_col=col)
                if agg.empty:
                    means.append(float("nan"))
                    los.append(float("nan"))
                    his.append(float("nan"))
                else:
                    means.append(float(agg.iloc[0]["mean"]))
                    los.append(float(agg.iloc[0]["ci_lo"]))
                    his.append(float(agg.iloc[0]["ci_hi"]))
                xs.append(gap)
            plot_line_with_ci(
                ax,
                xs,
                means,
                los,
                his,
                color=FAMILY_COLORS[family],
                label=f"d={depth}",
                linestyle=DEPTH_LINESTYLES.get(depth, "-"),
                marker="o",
            )
        ax.set_title(f"{family_label(family)} — {'Gibbs' if gibbs else 'No Gibbs'}")
        ax.set_xlabel("Reference-engine CP gap bucket")
        ax.set_xticks(gaps)
        ax.set_xscale("log", base=2)
    axes[0].set_ylabel("P(chosen | inference candidates)")
    axes[1].legend(ncol=2, loc="best")
    fig.suptitle(f"Reference-engine CP-gap conditioning ({family_label(family)})", y=1.03)
    save_figure(fig, out_dir, f"cp_gap_conditioning__{family}")


# ============================================================
# Reporting / exports
# ============================================================


def export_run_table(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    df.sort_values(["family", "gibbs", "temperature", "depth", "gm"]).to_csv(
        out_dir / "depth_study_run_table.csv", index=False
    )


def export_best_temp_stub(out_dir: Path) -> None:
    rows = []
    for family in FAMILY_ORDER:
        rows.append({"family": family, "label": family_label(family), "best_temp": BEST_TEMPS[family]})
    pd.DataFrame(rows).to_csv(out_dir / "best_temps_used.csv", index=False)


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate IEEE-CoG-quality plots for experiment3 depth-study validation results. "
            "Uses inference-conditioned p(chosen) for primary plots and reference-engine conditioning "
            "for rank / CP-gap plots."
        )
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Path to final_experiments_for_paper/experiment3/depth_study_validation_results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where figures / csv exports will be written",
    )
    parser.add_argument(
        "--fixed-depth-for-temp-sweep",
        type=int,
        default=FIXED_DEPTH_FOR_TEMP_SWEEP,
        help="Depth to use for temperature sweeps (default: 10)",
    )
    parser.add_argument(
        "--gms",
        nargs="*",
        default=GM_ORDER,
        help="Subset of grandmasters to include (default: all 9)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    records = discover_runs(args.eval_root, args.gms)
    if not records:
        raise SystemExit("No matching experiment3 runs were discovered.")

    df = build_records_table(records)
    export_run_table(df, args.out_dir)
    export_best_temp_stub(args.out_dir)

    # Primary paper figures.
    plot_family_depth_sweep(df, args.out_dir, metric_col="p_chosen_cond_inference")
    plot_family_depth_sweep(df, args.out_dir, metric_col="mrr_cond_inference")
    plot_temp_sweep_fixed_depth(
        df,
        args.out_dir,
        fixed_depth=args.fixed_depth_for_temp_sweep,
        metric_col="p_chosen_cond_inference",
    )
    plot_temp_sweep_fixed_depth(
        df,
        args.out_dir,
        fixed_depth=args.fixed_depth_for_temp_sweep,
        metric_col="mrr_cond_inference",
    )
    plot_cross_family_best_temp_depth_comparison(df, args.out_dir, metric_col="p_chosen_cond_inference")
    plot_cross_family_best_temp_depth_comparison(df, args.out_dir, metric_col="mrr_cond_inference")

    # Reference-engine conditioning figures.
    for family in FAMILY_ORDER:
        plot_reference_rank_conditioning(df, args.out_dir, family=family, bucket_kind="exact")
        plot_reference_rank_conditioning(df, args.out_dir, family=family, bucket_kind="leq")
        plot_reference_cp_gap_conditioning(df, args.out_dir, family=family)

    print(f"[OK] Wrote figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
