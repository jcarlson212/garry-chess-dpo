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
# IEEE CoG / paper-ready defaults
# ============================================================

plt.rcParams.update(
    {
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "legend.fontsize": 6.3,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.2,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "axes.axisbelow": True,
    }
)

# ------------------------------------------------------------
# Semantic colors:
#   family hue = v1 / v2 / v3
#   shade      = phi0 / phi1 / phi3
# ------------------------------------------------------------

FAMILY_COLORS = {
    "v1": {"phi0": "#4C78A8", "phi1": "#2A5A88", "phi3": "#173A5A"},
    "v2": {"phi0": "#F58518", "phi1": "#C86A00", "phi3": "#8F4B00"},
    "v3": {"phi0": "#54A24B", "phi1": "#2F7E2A", "phi3": "#165016"},
    "other": {"phi0": "#777777", "phi1": "#666666", "phi3": "#555555"},
}

RUN_STAGE_ORDER = {
    "screen": 0,
    "ablation": 1,
    "final": 2,
    "super": 3,
    "stress_test": 4,
    "other": 5,
}

PAIR_ORDER = {"v1": 0, "v2": 1, "v3": 2}
PHI_ORDER = {"phi0": 0, "phi1": 1, "phi3": 2}

# ============================================================
# Metric definitions
# ============================================================

# Main-paper metrics. These are what you should actually tell the story with.
PRIMARY_METRICS = {
    "mrr": {
        "title": "MRR",
        "keys": ["mrr"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "recall_at_1": {
        "title": "Recall@1",
        "keys": ["recall_at_1", "recall@1", "top1_acc", "top1_accuracy", "accuracy_top1"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "recall_at_5": {
        "title": "Recall@5",
        "keys": ["recall_at_5", "recall@5", "top5_recall", "hit_top5"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "mean_logp_gap": {
        "title": "Mean log-prob gap",
        "keys": [
            "mean_logp_gap",
            "mean_logp_gap_pi",
            "mean_logp_gap_policy_chosen_rejected",
        ],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "pair_acc_hardest": {
        "title": "Pair acc vs hardest neg",
        "keys": [
            "pair_acc_mean_vs_hardest",
            "pair_acc_mean_pos_gt_hardest_neg",
            "pair_acc_mean_pos_hardest_neg",
        ],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "row_cos_hard_gap": {
        "title": "Row cosine hard gap",
        "keys": ["row_cos_hard_gap", "row_cosine_hard_gap"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "row_cos_mean_gap": {
        "title": "Row cosine mean gap",
        "keys": ["row_cos_mean_gap", "row_cosine_mean_gap"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "spread_ratio": {
        "title": "Spread ratio",
        "keys": ["spread_ratio", "spread_ratio_mean", "spread.spread_ratio_mean"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "mean_kl": {
        "title": "KL vs reference",
        "keys": ["mean_kl", "kl_pi_ref"],
        "is_percent": False,
        "bigger_is_better": False,
    },
    "mean_ent_pi": {
        "title": "Entropy",
        "keys": ["mean_ent_pi", "entropy_pi"],
        "is_percent": False,
        "bigger_is_better": False,
    },
}

# Conditional metrics that are especially useful for appendix / style-sensitive story.
CONDITIONAL_METRICS = {
    "mean_logp_gap_cond_not_top10": {
        "title": "Gap | chosen not in SF top-10",
        "keys": [
            "mean_logp_gap_cond_not_top10",
            "mean_logp_gap_cond_on_not_in_top_ten",
            "mean_logp_gap_pi_cond_on_not_in_top_ten",
            "mean_logp_gap_policy_chosen_rejected_cond_on_not_in_top_ten",
        ],
        "is_percent": False,
    },
    "mrr_cond_not_top10": {
        "title": "MRR | chosen not in SF top-10",
        "keys": ["mrr_cond_not_top10", "mrr_cond_on_not_in_top_ten"],
        "is_percent": False,
    },
    "recall_at_1_cond_not_top10": {
        "title": "Recall@1 | chosen not in SF top-10",
        "keys": [
            "recall_at_1_cond_not_top10",
            "top1_acc_cond_not_top10",
            "top1_accuracy_cond_on_not_in_top_ten",
            "accuracy_top1_cond_on_not_in_top_ten",
        ],
        "is_percent": False,
    },
    "mean_logp_gap_cond_in_top10": {
        "title": "Gap | chosen in SF top-10",
        "keys": [
            "mean_logp_gap_cond_in_top10",
            "mean_logp_gap_cond_on_in_top_ten",
            "mean_logp_gap_pi_cond_on_in_top_ten",
            "mean_logp_gap_policy_chosen_rejected_cond_on_in_top_ten",
        ],
        "is_percent": False,
    },
    "mrr_cond_in_top10": {
        "title": "MRR | chosen in SF top-10",
        "keys": ["mrr_cond_in_top10", "mrr_cond_on_in_top_ten"],
        "is_percent": False,
    },
    "recall_at_1_cond_in_top10": {
        "title": "Recall@1 | chosen in SF top-10",
        "keys": [
            "recall_at_1_cond_in_top10",
            "top1_acc_cond_in_top10",
            "top1_accuracy_cond_on_in_top_ten",
            "accuracy_top1_cond_on_in_top_ten",
        ],
        "is_percent": False,
    },
}


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def try_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def canonicalize_tau_str(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    raw = raw.replace("_", ".")
    try:
        val = float(raw)
    except Exception:
        return raw
    # Canonical short numeric string without unnecessary zeros
    return f"{val:g}"


def parse_tau_value(raw: Optional[str]) -> float:
    c = canonicalize_tau_str(raw)
    if c is None:
        return float("nan")
    try:
        return float(c)
    except Exception:
        return float("nan")


def ci95(vals: Sequence[float]) -> Tuple[float, float, float, int]:
    arr = np.asarray([try_float(v) for v in vals], dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean, 1
    se = float(arr.std(ddof=1) / math.sqrt(n))
    half = 1.96 * se
    return mean, mean - half, mean + half, n


def recursive_find_first_number(obj: Any, candidate_keys: Sequence[str]) -> float:
    """
    Flexible metric extraction from nested dict/json structures.
    """
    candidate_keys_lower = {k.lower() for k in candidate_keys}

    def _walk(x: Any) -> Optional[float]:
        if isinstance(x, dict):
            # direct hit
            for k, v in x.items():
                if str(k).lower() in candidate_keys_lower:
                    if isinstance(v, dict):
                        for subk in ["mean", "value", "avg"]:
                            if subk in v and np.isfinite(try_float(v[subk])):
                                return float(v[subk])
                    val = try_float(v)
                    if np.isfinite(val):
                        return val
            # bootstrap row style
            for k, v in x.items():
                if str(k).lower() == "bootstrap_ci_row" and isinstance(v, dict):
                    for inner_k, inner_v in v.items():
                        if str(inner_k).lower() in candidate_keys_lower and isinstance(inner_v, dict):
                            for subk in ["mean", "value", "avg"]:
                                if subk in inner_v and np.isfinite(try_float(inner_v[subk])):
                                    return float(inner_v[subk])
            # recurse
            for _, v in x.items():
                found = _walk(v)
                if found is not None:
                    return found
        elif isinstance(x, list):
            for v in x:
                found = _walk(v)
                if found is not None:
                    return found
        return None

    found = _walk(obj)
    return float("nan") if found is None else found


def pretty_method_label(pair_version: str, phi: str, tau: Optional[str]) -> str:
    tau_part = f"τ={canonicalize_tau_str(tau)}" if tau is not None else "τ=?"
    return f"{pair_version}-{phi}-{tau_part}"


def run_sort_key(run: "RunRecord") -> Tuple[int, int, int, float, str]:
    return (
        RUN_STAGE_ORDER.get(run.stage, 99),
        PAIR_ORDER.get(run.pair_version, 99),
        PHI_ORDER.get(run.phi, 99),
        run.tau_value if np.isfinite(run.tau_value) else 999.0,
        run.name,
    )


def bar_colors(runs: Sequence["RunRecord"]) -> List[str]:
    out = []
    for r in runs:
        fam = FAMILY_COLORS.get(r.pair_version, FAMILY_COLORS["other"])
        out.append(fam.get(r.phi, "#777777"))
    return out


# ============================================================
# Run metadata
# ============================================================

RUN_NAME_RE = re.compile(
    r"""
    ^
    (?P<stage>screen|final|super|stress_test|ablation)?
    _?
    (?P<pair>v[123])
    _
    (?P<phi>phi[013])
    (?:
        _warm(?:_from_[^_]+)?
    )?
    (?:
        _tau(?P<tau>[0-9_]+)
    )?
    .*?
    __pair-(?P<pair_confirm>v[123])
    __phi-(?P<phi_confirm>phi[013])
    .*?
    __tau-(?P<tau_confirm>[0-9.]+)
    __seed-(?P<seed>\d+)
    $
    """,
    re.VERBOSE,
)


@dataclass
class RunRecord:
    name: str
    path: Path
    source_kind: str  # "eval" or "train"
    stage: str
    pair_version: str
    phi: str
    tau: Optional[str]
    tau_value: float
    seed: Optional[int]
    training_summary_path: Optional[Path] = None
    eval_summary_path: Optional[Path] = None
    per_row_metrics_path: Optional[Path] = None
    raw_eval_json: Optional[Dict[str, Any]] = None
    training_df: Optional[pd.DataFrame] = None

    @property
    def method_label(self) -> str:
        return pretty_method_label(self.pair_version, self.phi, self.tau)

    @property
    def family_key(self) -> str:
        return f"{self.pair_version}|{self.phi}"

    @property
    def finalist_key(self) -> str:
        return f"{self.pair_version}|{self.phi}|tau={canonicalize_tau_str(self.tau)}"

    def metric(self, metric_key: str) -> float:
        if self.raw_eval_json is None:
            return float("nan")
        spec = PRIMARY_METRICS.get(metric_key) or CONDITIONAL_METRICS.get(metric_key)
        if spec is None:
            return float("nan")
        return recursive_find_first_number(self.raw_eval_json, spec["keys"])


def parse_run_name(name: str) -> Tuple[str, str, str, Optional[str], float, Optional[int]]:
    m = RUN_NAME_RE.match(name)
    if not m:
        # fallback parse
        stage = "other"
        pair = re.search(r"__(?:pair-)?(v[123])__", f"__{name}__")
        phi = re.search(r"__(?:phi-)?(phi[013])__", f"__{name}__")
        tau = re.search(r"__(?:tau-)?([0-9.]+)__", f"__{name}__")
        return (
            stage,
            pair.group(1) if pair else "other",
            phi.group(1) if phi else "phi0",
            canonicalize_tau_str(tau.group(1)) if tau else None,
            parse_tau_value(tau.group(1)) if tau else float("nan"),
            None,
        )

    stage = m.group("stage") or "other"
    pair = m.group("pair_confirm") or m.group("pair") or "other"
    phi = m.group("phi_confirm") or m.group("phi") or "phi0"
    tau_raw = m.group("tau_confirm") or m.group("tau")
    tau = canonicalize_tau_str(tau_raw)
    tau_val = parse_tau_value(tau_raw)
    seed = int(m.group("seed")) if m.group("seed") else None
    return stage, pair, phi, tau, tau_val, seed


# ============================================================
# Discovery
# ============================================================

def find_best_matching_file(run_dir: Path, split: str, patterns: Sequence[str]) -> Optional[Path]:
    """
    Robust file discovery:
      - prefer explicit split match (test/eval/val)
      - otherwise accept generic file
    """
    files = [p for p in run_dir.iterdir() if p.is_file()]
    split = split.lower()

    split_hits: List[Path] = []
    generic_hits: List[Path] = []

    for p in files:
        name = p.name.lower()
        if any(re.search(pattern, name) for pattern in patterns):
            if split in name:
                split_hits.append(p)
            else:
                generic_hits.append(p)

    if split_hits:
        split_hits.sort()
        return split_hits[0]
    if generic_hits:
        generic_hits.sort()
        return generic_hits[0]
    return None


def discover_eval_runs(eval_runs_root: Path, split: str) -> List[RunRecord]:
    runs: List[RunRecord] = []
    if not eval_runs_root.exists():
        return runs

    for run_dir in sorted([p for p in eval_runs_root.iterdir() if p.is_dir()]):
        stage, pair, phi, tau, tau_val, seed = parse_run_name(run_dir.name)

        eval_summary = find_best_matching_file(
            run_dir,
            split=split,
            patterns=[
                r"eval_results",
                r"summary",
                r"metrics",
            ],
        )
        per_row = find_best_matching_file(
            run_dir,
            split=split,
            patterns=[
                r"per[_\-]?row",
                r"row[_\-]?metrics",
            ],
        )

        rec = RunRecord(
            name=run_dir.name,
            path=run_dir,
            source_kind="eval",
            stage=stage,
            pair_version=pair,
            phi=phi,
            tau=tau,
            tau_value=tau_val,
            seed=seed,
            eval_summary_path=eval_summary,
            per_row_metrics_path=per_row,
        )

        if eval_summary and eval_summary.suffix.lower() == ".json":
            try:
                rec.raw_eval_json = load_json(eval_summary)
            except Exception:
                rec.raw_eval_json = None
        elif eval_summary and eval_summary.suffix.lower() == ".jsonl":
            try:
                rec.raw_eval_json = {"rows": load_jsonl(eval_summary)}
            except Exception:
                rec.raw_eval_json = None

        runs.append(rec)

    return sorted(runs, key=run_sort_key)


def discover_training_runs(training_summary_dir: Path) -> List[RunRecord]:
    runs: List[RunRecord] = []
    if not training_summary_dir.exists():
        return runs

    for p in sorted(training_summary_dir.glob("*.jsonl")):
        stem = p.stem
        stage, pair, phi, tau, tau_val, seed = parse_run_name(stem)
        rec = RunRecord(
            name=stem,
            path=p,
            source_kind="train",
            stage=stage,
            pair_version=pair,
            phi=phi,
            tau=tau,
            tau_value=tau_val,
            seed=seed,
            training_summary_path=p,
        )
        try:
            rows = load_jsonl(p)
            rec.training_df = pd.DataFrame(rows)
        except Exception:
            rec.training_df = pd.DataFrame()

        runs.append(rec)

    return sorted(runs, key=run_sort_key)


def attach_training_to_eval(eval_runs: List[RunRecord], train_runs: List[RunRecord]) -> None:
    by_name = {r.name: r for r in train_runs}
    for ev in eval_runs:
        tr = by_name.get(ev.name)
        if tr is not None:
            ev.training_summary_path = tr.training_summary_path
            ev.training_df = tr.training_df


# ============================================================
# Filtering / selecting
# ============================================================

def drop_duplicate_canonical_runs(runs: List[RunRecord]) -> List[RunRecord]:
    """
    Collapse duplicates like tau0_10 vs tau0_1 by canonical metadata.
    Prefer runs with eval summary, then per-row, then lexicographically first name.
    """
    grouped: Dict[Tuple[str, str, str, Optional[str], Optional[int]], List[RunRecord]] = {}
    for r in runs:
        key = (r.stage, r.pair_version, r.phi, canonicalize_tau_str(r.tau), r.seed)
        grouped.setdefault(key, []).append(r)

    out: List[RunRecord] = []
    for _, items in grouped.items():
        items = sorted(
            items,
            key=lambda r: (
                0 if r.eval_summary_path else 1,
                0 if r.per_row_metrics_path else 1,
                r.name,
            ),
        )
        out.append(items[0])
    return sorted(out, key=run_sort_key)


def choose_best_finalists(eval_runs: List[RunRecord]) -> List[RunRecord]:
    """
    Keep one best run per pair/phi among final/super/stress_test if available,
    otherwise fall back to screen.
    Primary ranking:
      1) stage priority: super > final > stress_test > screen
      2) MRR
      3) hard-gap
      4) Recall@1
    """
    stage_priority = {"super": 0, "final": 1, "stress_test": 2, "screen": 3, "ablation": 4, "other": 5}
    grouped: Dict[Tuple[str, str], List[RunRecord]] = {}
    for r in eval_runs:
        grouped.setdefault((r.pair_version, r.phi), []).append(r)

    selected: List[RunRecord] = []
    for _, items in grouped.items():
        items = sorted(
            items,
            key=lambda r: (
                stage_priority.get(r.stage, 99),
                -(r.metric("mrr") if np.isfinite(r.metric("mrr")) else -1e9),
                -(r.metric("row_cos_hard_gap") if np.isfinite(r.metric("row_cos_hard_gap")) else -1e9),
                -(r.metric("recall_at_1") if np.isfinite(r.metric("recall_at_1")) else -1e9),
                r.tau_value if np.isfinite(r.tau_value) else 999.0,
                r.name,
            ),
        )
        selected.append(items[0])

    return sorted(selected, key=lambda r: (PAIR_ORDER.get(r.pair_version, 99), PHI_ORDER.get(r.phi, 99)))


def runs_for_tau_sweep(eval_runs: List[RunRecord], pair_version: str, phi: str) -> List[RunRecord]:
    candidates = [r for r in eval_runs if r.pair_version == pair_version and r.phi == phi]
    # Prefer screen/final/super in that order to show the actual sweep history.
    candidates = sorted(
        candidates,
        key=lambda r: (
            0 if r.stage == "screen" else (1 if r.stage == "final" else (2 if r.stage == "super" else 3)),
            r.tau_value if np.isfinite(r.tau_value) else 999.0,
            r.name,
        ),
    )
    # Deduplicate by canonical tau, keeping the best stage candidate
    best_by_tau: Dict[str, RunRecord] = {}
    for r in candidates:
        tau = canonicalize_tau_str(r.tau)
        if tau is None:
            continue
        if tau not in best_by_tau:
            best_by_tau[tau] = r
    return sorted(best_by_tau.values(), key=lambda r: r.tau_value)


def training_runs_for_ablation(train_runs: List[RunRecord], pattern: str) -> List[RunRecord]:
    return sorted([r for r in train_runs if pattern in r.name], key=run_sort_key)


# ============================================================
# Data export
# ============================================================

def export_run_table(runs: Sequence[RunRecord], out_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in runs:
        row = {
            "name": r.name,
            "stage": r.stage,
            "pair_version": r.pair_version,
            "phi": r.phi,
            "tau": canonicalize_tau_str(r.tau),
            "seed": r.seed,
            "method_label": r.method_label,
        }
        for metric_key in PRIMARY_METRICS:
            row[metric_key] = r.metric(metric_key)
        for metric_key in CONDITIONAL_METRICS:
            row[metric_key] = r.metric(metric_key)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    df.to_json(out_path.with_suffix(".json"), orient="records", indent=2)
    return df


# ============================================================
# Plot helpers
# ============================================================

def annotate_top(ax: plt.Axes, xs: Sequence[float], ys: Sequence[float], fmt: str = "{:.3f}", k: int = 1) -> None:
    arr = np.asarray(ys, dtype=float)
    valid = np.where(np.isfinite(arr))[0]
    if len(valid) == 0:
        return
    order = valid[np.argsort(arr[valid])[::-1]]
    for idx in order[:k]:
        ax.text(xs[idx], arr[idx], fmt.format(arr[idx]), fontsize=6.2, ha="center", va="bottom")


def grouped_bar(
    ax: plt.Axes,
    runs: Sequence[RunRecord],
    metric_key: str,
    title: str,
) -> None:
    xs = np.arange(len(runs))
    ys = [r.metric(metric_key) for r in runs]
    colors = bar_colors(runs)
    ax.bar(xs, ys, color=colors, width=0.72)
    ax.set_xticks(xs)
    ax.set_xticklabels([r.method_label for r in runs], rotation=25, ha="right")
    ax.set_title(title, pad=4)
    annotate_top(ax, xs, ys, fmt="{:.4f}", k=2)


def line_tau(
    ax: plt.Axes,
    runs: Sequence[RunRecord],
    metric_key: str,
    title: str,
    ylabel: Optional[str] = None,
) -> None:
    xs = np.asarray([r.tau_value for r in runs], dtype=float)
    ys = np.asarray([r.metric(metric_key) for r in runs], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    if len(xs) == 0:
        ax.set_visible(False)
        return

    color = FAMILY_COLORS.get(runs[0].pair_version, FAMILY_COLORS["other"]).get(runs[0].phi, "#666666")
    ax.plot(xs, ys, marker="o", color=color)
    ax.set_title(title, pad=4)
    ax.set_xlabel("Training τ")
    if ylabel:
        ax.set_ylabel(ylabel)
    annotate_top(ax, xs, ys, fmt="{:.4f}", k=1)


def plot_training_curve(ax: plt.Axes, run: RunRecord, title: str, smooth_window: int = 5) -> None:
    if run.training_df is None or run.training_df.empty:
        ax.set_visible(False)
        return

    df = run.training_df.copy()
    step_col = None
    loss_col = None

    for c in df.columns:
        cl = c.lower()
        if step_col is None and cl in {"step", "steps", "global_step"}:
            step_col = c
        if loss_col is None and cl in {"train_loss", "loss", "train/loss"}:
            loss_col = c

    if step_col is None or loss_col is None:
        ax.set_visible(False)
        return

    x = pd.to_numeric(df[step_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[loss_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        ax.set_visible(False)
        return

    order = np.argsort(x)
    x, y = x[order], y[order]
    color = FAMILY_COLORS.get(run.pair_version, FAMILY_COLORS["other"]).get(run.phi, "#666666")

    ax.plot(x, y, alpha=0.30, color=color)
    if len(y) >= smooth_window:
        sm = pd.Series(y).rolling(smooth_window, min_periods=1).mean().to_numpy()
        ax.plot(x, sm, color=color, linewidth=1.6)

    ax.set_title(title, pad=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")


def ranked_dotplot(ax: plt.Axes, runs: Sequence[RunRecord], metric_key: str, title: str) -> None:
    runs = sorted(
        runs,
        key=lambda r: r.metric(metric_key) if np.isfinite(r.metric(metric_key)) else -1e9,
        reverse=True,
    )
    xs = np.arange(len(runs))
    ys = [r.metric(metric_key) for r in runs]
    colors = bar_colors(runs)
    ax.scatter(xs, ys, c=colors, s=24)
    for i, r in enumerate(runs):
        ax.text(i, ys[i], r.method_label, fontsize=5.8, rotation=55, ha="left", va="bottom")
    ax.set_xticks([])
    ax.set_title(title, pad=4)


# ============================================================
# Figure families
# ============================================================

def plot_finalist_main_results(out_dir: Path, finalists: Sequence[RunRecord]) -> None:
    metrics = [
        ("mrr", PRIMARY_METRICS["mrr"]["title"]),
        ("recall_at_1", PRIMARY_METRICS["recall_at_1"]["title"]),
        ("mean_logp_gap", PRIMARY_METRICS["mean_logp_gap"]["title"]),
        ("mean_kl", PRIMARY_METRICS["mean_kl"]["title"]),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(10.6, 2.7), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        grouped_bar(ax, finalists, metric_key, title)
    savefig(fig, out_dir, "fig_finalist_main_results")


def plot_hard_negative_results(out_dir: Path, finalists: Sequence[RunRecord]) -> None:
    metrics = [
        ("pair_acc_hardest", PRIMARY_METRICS["pair_acc_hardest"]["title"]),
        ("row_cos_hard_gap", PRIMARY_METRICS["row_cos_hard_gap"]["title"]),
        ("row_cos_mean_gap", PRIMARY_METRICS["row_cos_mean_gap"]["title"]),
        ("spread_ratio", PRIMARY_METRICS["spread_ratio"]["title"]),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(10.8, 2.8), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        grouped_bar(ax, finalists, metric_key, title)
    savefig(fig, out_dir, "fig_hard_negative_results")


def plot_phi_comparisons(out_dir: Path, finalists: Sequence[RunRecord]) -> None:
    """
    Compare phi0 vs phi1 within v1 and v2.
    """
    comparisons = []
    for pair in ["v1", "v2"]:
        sub = [r for r in finalists if r.pair_version == pair and r.phi in {"phi0", "phi1"}]
        if len(sub) >= 2:
            comparisons.extend(sorted(sub, key=lambda r: PHI_ORDER.get(r.phi, 99)))

    if not comparisons:
        return

    metrics = [
        ("mrr", "MRR"),
        ("recall_at_1", "Recall@1"),
        ("mean_logp_gap", "Mean log-prob gap"),
        ("row_cos_hard_gap", "Hard gap"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 2.7), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        grouped_bar(ax, comparisons, metric_key, title)
    savefig(fig, out_dir, "fig_phi_comparisons")


def plot_tau_sweeps(out_dir: Path, eval_runs: Sequence[RunRecord]) -> None:
    panels = [
        ("v1", "phi0"),
        ("v1", "phi1"),
        ("v2", "phi0"),
        ("v2", "phi1"),
        ("v3", "phi0"),
        ("v3", "phi1"),
    ]

    metric_pairs = [
        ("mrr", "MRR"),
        ("mean_kl", "KL vs ref"),
    ]

    fig, axes = plt.subplots(len(panels), len(metric_pairs), figsize=(6.6, 9.2), constrained_layout=True)
    for i, (pair, phi) in enumerate(panels):
        sweep = runs_for_tau_sweep(list(eval_runs), pair, phi)
        for j, (metric_key, ylabel) in enumerate(metric_pairs):
            ax = axes[i, j]
            title = f"{pair}-{phi}"
            line_tau(ax, sweep, metric_key, title=title, ylabel=ylabel if j == 0 else None)
            if j == 1:
                ax.set_ylabel("")

    savefig(fig, out_dir, "fig_tau_sweeps")


def plot_training_diagnostics(out_dir: Path, finalists: Sequence[RunRecord]) -> None:
    if not finalists:
        return

    finalists = sorted(finalists, key=lambda r: (PAIR_ORDER.get(r.pair_version, 99), PHI_ORDER.get(r.phi, 99)))
    n = min(4, len(finalists))
    chosen = finalists[:n]

    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 2.5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, run in zip(axes, chosen):
        plot_training_curve(ax, run, run.method_label)
    savefig(fig, out_dir, "fig_training_diagnostics_finalists")


def plot_ablation_batchsize(out_dir: Path, train_runs: Sequence[RunRecord], eval_runs: Sequence[RunRecord]) -> None:
    """
    Focus on the around-best-recipe batch-size ablation for v1 phi0.
    """
    batch_runs = [r for r in train_runs if "ablation_v1_phi0" in r.name and "_bs" in r.name]
    if not batch_runs:
        return

    rows: List[Dict[str, Any]] = []
    eval_lookup = {r.name: r for r in eval_runs}
    for r in batch_runs:
        m = re.search(r"__bs-(\d+)__", f"__{r.name}__")
        bs = int(m.group(1)) if m else None
        if bs is None:
            continue
        train_final_loss = float("nan")
        if r.training_df is not None and not r.training_df.empty:
            loss_col = None
            for c in r.training_df.columns:
                if c.lower() in {"train_loss", "loss", "train/loss"}:
                    loss_col = c
                    break
            if loss_col is not None:
                series = pd.to_numeric(r.training_df[loss_col], errors="coerce").dropna()
                if not series.empty:
                    train_final_loss = float(series.iloc[-1])

        ev = eval_lookup.get(r.name)
        rows.append(
            {
                "name": r.name,
                "batch_size": bs,
                "train_final_loss": train_final_loss,
                "mrr": ev.metric("mrr") if ev else float("nan"),
                "row_cos_hard_gap": ev.metric("row_cos_hard_gap") if ev else float("nan"),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("batch_size")
    df.to_csv(out_dir / "ablation_batchsize.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.5), constrained_layout=True)
    for ax, col, title in zip(
        axes,
        ["train_final_loss", "mrr", "row_cos_hard_gap"],
        ["Final train loss", "MRR", "Hard gap"],
    ):
        ax.plot(df["batch_size"], df[col], marker="o")
        ax.set_title(title, pad=4)
        ax.set_xlabel("Batch size")
    savefig(fig, out_dir, "fig_ablation_batchsize")


def plot_ablation_lr(out_dir: Path, train_runs: Sequence[RunRecord], eval_runs: Sequence[RunRecord]) -> None:
    lr_runs = [r for r in train_runs if "ablation_v1_phi0" in r.name and "_lr" in r.name]
    if not lr_runs:
        return

    rows: List[Dict[str, Any]] = []
    eval_lookup = {r.name: r for r in eval_runs}
    for r in lr_runs:
        m = re.search(r"__lr-([0-9.]+)__", f"__{r.name}__")
        lr = float(m.group(1)) if m else None
        if lr is None:
            continue
        train_final_loss = float("nan")
        if r.training_df is not None and not r.training_df.empty:
            loss_col = None
            for c in r.training_df.columns:
                if c.lower() in {"train_loss", "loss", "train/loss"}:
                    loss_col = c
                    break
            if loss_col is not None:
                series = pd.to_numeric(r.training_df[loss_col], errors="coerce").dropna()
                if not series.empty:
                    train_final_loss = float(series.iloc[-1])

        ev = eval_lookup.get(r.name)
        rows.append(
            {
                "name": r.name,
                "lr": lr,
                "train_final_loss": train_final_loss,
                "mrr": ev.metric("mrr") if ev else float("nan"),
                "row_cos_hard_gap": ev.metric("row_cos_hard_gap") if ev else float("nan"),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("lr")
    df.to_csv(out_dir / "ablation_lr.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.5), constrained_layout=True)
    for ax, col, title in zip(
        axes,
        ["train_final_loss", "mrr", "row_cos_hard_gap"],
        ["Final train loss", "MRR", "Hard gap"],
    ):
        ax.plot(df["lr"], df[col], marker="o")
        ax.set_xscale("log")
        ax.set_title(title, pad=4)
        ax.set_xlabel("Learning rate")
    savefig(fig, out_dir, "fig_ablation_lr")


def plot_promotion_scatter(out_dir: Path, eval_runs: Sequence[RunRecord]) -> None:
    """
    screen -> final/super sanity plot
    """
    rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str, str], Dict[str, RunRecord]] = {}

    for r in eval_runs:
        tau = canonicalize_tau_str(r.tau)
        if tau is None:
            continue
        key = (r.pair_version, r.phi, tau)
        by_key.setdefault(key, {})
        by_key[key][r.stage] = r

    for key, stages in by_key.items():
        scr = stages.get("screen")
        fin = stages.get("final") or stages.get("super")
        if scr is None or fin is None:
            continue
        rows.append(
            {
                "pair_version": key[0],
                "phi": key[1],
                "tau": key[2],
                "screen_mrr": scr.metric("mrr"),
                "final_mrr": fin.metric("mrr"),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "promotion_scatter.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    for _, row in df.iterrows():
        color = FAMILY_COLORS.get(row["pair_version"], FAMILY_COLORS["other"]).get(row["phi"], "#666666")
        ax.scatter(row["screen_mrr"], row["final_mrr"], color=color, s=28)
        ax.text(row["screen_mrr"], row["final_mrr"], f'{row["pair_version"]}-{row["phi"]}-τ={row["tau"]}', fontsize=5.6)
    ax.set_xlabel("Screen MRR")
    ax.set_ylabel("Final/Super MRR")
    ax.set_title("Promotion sanity check", pad=4)
    savefig(fig, out_dir, "fig_promotion_scatter")


def plot_appendix_conditionals(out_dir: Path, finalists: Sequence[RunRecord]) -> None:
    metrics = [
        ("mean_logp_gap_cond_not_top10", CONDITIONAL_METRICS["mean_logp_gap_cond_not_top10"]["title"]),
        ("mrr_cond_not_top10", CONDITIONAL_METRICS["mrr_cond_not_top10"]["title"]),
        ("mean_logp_gap_cond_in_top10", CONDITIONAL_METRICS["mean_logp_gap_cond_in_top10"]["title"]),
        ("mrr_cond_in_top10", CONDITIONAL_METRICS["mrr_cond_in_top10"]["title"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes.flatten(), metrics):
        grouped_bar(ax, finalists, metric_key, title)
    savefig(fig, out_dir, "appendix_conditionals")


def plot_appendix_ranked_metrics(out_dir: Path, eval_runs: Sequence[RunRecord]) -> None:
    metrics = [
        ("mrr", "All runs ranked by MRR"),
        ("row_cos_hard_gap", "All runs ranked by hard gap"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.0), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        ranked_dotplot(ax, eval_runs, metric_key, title)
    savefig(fig, out_dir, "appendix_ranked_metrics")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coherent IEEE CoG paper plots for Experiment 2 style embeddings.")
    parser.add_argument("--eval-runs-root", type=Path, required=True)
    parser.add_argument("--training-summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--include-appendix",
        action="store_true",
        help="Also emit appendix-style conditional and ranked diagnostic plots.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    main_dir = args.output_dir / "main_paper"
    appendix_dir = args.output_dir / "appendix"
    tables_dir = args.output_dir / "tables"
    ensure_dir(main_dir)
    ensure_dir(appendix_dir)
    ensure_dir(tables_dir)

    eval_runs = discover_eval_runs(args.eval_runs_root, split=args.split)
    eval_runs = drop_duplicate_canonical_runs(eval_runs)

    train_runs = discover_training_runs(args.training_summary_dir)
    train_runs = drop_duplicate_canonical_runs(train_runs)

    attach_training_to_eval(eval_runs, train_runs)

    # Export raw discovered tables
    export_run_table(eval_runs, tables_dir / "eval_runs")
    export_run_table(train_runs, tables_dir / "training_runs")

    finalists = choose_best_finalists(eval_runs)
    export_run_table(finalists, tables_dir / "finalists")

    # -----------------------
    # Main paper figure set
    # -----------------------
    plot_tau_sweeps(main_dir, eval_runs)
    plot_phi_comparisons(main_dir, finalists)
    plot_finalist_main_results(main_dir, finalists)
    plot_hard_negative_results(main_dir, finalists)
    plot_training_diagnostics(main_dir, finalists)
    plot_promotion_scatter(main_dir, eval_runs)

    # -----------------------
    # Ablations
    # -----------------------
    plot_ablation_batchsize(main_dir, train_runs, eval_runs)
    plot_ablation_lr(main_dir, train_runs, eval_runs)

    # -----------------------
    # Appendix
    # -----------------------
    if args.include_appendix:
        plot_appendix_conditionals(appendix_dir, finalists)
        plot_appendix_ranked_metrics(appendix_dir, eval_runs)

    # -----------------------
    # Human-readable notes
    # -----------------------
    notes_path = args.output_dir / "README_plot_story.txt"
    with notes_path.open("w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "Experiment 2 paper plot story",
                    "=============================",
                    "",
                    "Main-paper figure families:",
                    "1. fig_tau_sweeps.pdf/png",
                    "   - Screening story: choose tau per family.",
                    "   - Read as: moderate tau should improve retrieval/hard-gap vs weak tau; too-aggressive settings may hurt KL/entropy.",
                    "",
                    "2. fig_phi_comparisons.pdf/png",
                    "   - Metadata story: phi1 vs phi0 at matched family budget.",
                    "   - Read as: phi1 should help most on retrieval and hard-negative-aware metrics.",
                    "",
                    "3. fig_finalist_main_results.pdf/png",
                    "   - Finalist overall story: MRR, Recall@1, mean log-prob gap, KL.",
                    "   - Read as: best overall finalist should balance retrieval/separation with controlled drift.",
                    "",
                    "4. fig_hard_negative_results.pdf/png",
                    "   - Application story: v3 > v2 > v1 should show up most clearly here.",
                    "   - Read as: easy-task metrics can flatter v1, but hard-negative-aware metrics are the main decision metrics.",
                    "",
                    "5. fig_training_diagnostics_finalists.pdf/png",
                    "   - Stability story: finalists should train smoothly enough without obvious collapse.",
                    "",
                    "6. fig_promotion_scatter.pdf/png",
                    "   - Process story: screening metrics should roughly predict final metrics, justifying promotion logic.",
                    "",
                    "Ablations:",
                    "- fig_ablation_batchsize.pdf/png",
                    "- fig_ablation_lr.pdf/png",
                    "",
                    "Appendix-only figures:",
                    "- appendix_conditionals.pdf/png",
                    "- appendix_ranked_metrics.pdf/png",
                    "",
                    "Deliberately omitted from main paper:",
                    "- AP / F1 / ROC AUC",
                    "- giant spaghetti loss overlays",
                    "- raw histogram-heavy spread plots",
                    "- single highlighted PCA panel",
                    "",
                    "Naming fixes:",
                    "- tau0_10 and tau0_1 are canonicalized to tau=0.1",
                    "- run discovery is split-aware and does not assume only *_val.json",
                ]
            )
        )

    print(f"[done] wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
