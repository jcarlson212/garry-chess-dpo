from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


# ============================================================
# Paper plotting defaults
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

OPENING_MOVE_ORDER = ["e2e4", "d2d4", "c2c4", "g1f3", "g2g3", "b2b3", "f2f4", "b2b4", "a2a4"]
PIECE_ORDER = ["pawn", "knight", "bishop", "rook", "queen", "king"]

METHOD_PATTERNS = {
    "summary_ext_json": re.compile(
        r"^eval_results_(?:extended_(?P<method1>.+?)|(?P<method2>.+?)_extended)_val\.json$"
    ),
    "summary_json": re.compile(r"^eval_results_(?P<method>.+?)_val\.json$"),
    "opening_probe_json": re.compile(r"^opening_probe_policy_(?P<method>.+?)\.json$"),
    # handles eval_per_Row_metrics_<method>_val.json
    "per_row_json": re.compile(r"^eval_per_[Rr]ow_metrics_(?P<method>.+?)_val\.jsonl?$"),
}

METHOD_COLORS = {
    "maia2": "#2F6BFF",
    "sft": "#00A6A6",  # NLL
    "sft_pairwise": "#7E57C2",
    "dpo_beta=0.02": "#FAD7D7",
    "dpo_beta=0.05": "#F5B7B1",
    "dpo_beta=0.10": "#F1948A",
    "dpo_beta=0.20": "#EC7063",
    "dpo_beta=0.40": "#E74C3C",
    "dpo_beta=0.60": "#B03A2E",
    "dpo": "#C0392B",
    "sft_and_dpo": "#F39C12",
    "unknown": "#999999",
}

STYLE_V1_COLOR = "#7D6608"
STYLE_V2_COLOR = "#6E2C00"


# ============================================================
# Naming helpers
# ============================================================

def paper_method_label(method_key: str) -> str:
    if method_key == "maia2":
        return "Maia-2"
    if method_key == "sft":
        return "NLL"
    if method_key == "sft_pairwise":
        return "Pairwise"
    if method_key == "dpo":
        return "DPO"

    m = re.match(r"dpo_beta=(?P<beta>[0-9.]+)", method_key)
    if m:
        beta = float(m.group("beta"))
        return f"DPO (β={beta:g})"

    m = re.match(
        r"sft_and_dpo_beta=(?P<beta>[0-9.]+)_dpo_loss_weight=(?P<weight>[0-9.]+)",
        method_key,
    )
    if m:
        beta = float(m.group("beta"))
        weight = float(m.group("weight"))
        return f"NLL + DPO (λ={weight:g})"

    if method_key == "sft_and_dpo":
        return "NLL + DPO"

    return method_key


def opening_panel_title(gm_name: str) -> str:
    mapping = {
        "carlsen": "Carlsen",
        "caruana": "Caruana",
        "wei": "Wei",
        "vincent": "Vincent",
        "nakamura": "Nakamura",
        "gukesh": "Gukesh",
        "giri": "Giri",
        "firouzja": "Firouzja",
        "praggnanandhaa": "Praggnanandhaa",
    }
    return mapping.get(gm_name, gm_name.capitalize())


def method_color(method_key: str) -> str:
    # exact matches first
    if method_key in METHOD_COLORS:
        return METHOD_COLORS[method_key]

    # DPO beta sweep
    if method_key.startswith("dpo_beta="):
        return METHOD_COLORS.get(method_key, "#E74C3C")

    # NLL + DPO mixes (Experiment 2)
    m = re.match(
        r"sft_and_dpo_beta=(?P<beta>[0-9.]+)_dpo_loss_weight=(?P<w>[0-9.]+)",
        method_key,
    )
    if m:
        w = float(m.group("w"))

        # progressively darker orange as weight increases
        if w <= 0.10:
            return "#F8C471"
        if w <= 0.20:
            return "#F5B041"
        if w <= 0.40:
            return "#EB984E"
        return "#CA6F1E"

    if "style_sim_utility_weight" in method_key:
        return STYLE_V1_COLOR

    if "style_v2" in method_key:
        return STYLE_V2_COLOR

    return METHOD_COLORS["unknown"]


# ============================================================
# Bundle models
# ============================================================

@dataclass
class MethodBundle:
    gm_name: str
    method_key: str
    gm_dir: Path
    summary_ext_json: Optional[Path] = None
    summary_json: Optional[Path] = None
    opening_probe_json: Optional[Path] = None
    per_row_json: Optional[Path] = None

    summary_ext: Optional[Dict[str, Any]] = None
    summary_json_obj: Optional[Dict[str, Any]] = None
    opening_probe: Optional[Dict[str, Any]] = None
    per_row_df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        if self.summary_ext is None and self.summary_ext_json and self.summary_ext_json.exists():
            with self.summary_ext_json.open("r", encoding="utf-8") as f:
                self.summary_ext = json.load(f)

        if self.summary_json_obj is None and self.summary_json and self.summary_json.exists():
            with self.summary_json.open("r", encoding="utf-8") as f:
                self.summary_json_obj = json.load(f)

        if self.opening_probe is None and self.opening_probe_json and self.opening_probe_json.exists():
            with self.opening_probe_json.open("r", encoding="utf-8") as f:
                self.opening_probe = json.load(f)

        if self.per_row_df is None and self.per_row_json and self.per_row_json.exists():
            self.per_row_df = load_jsonl_records(self.per_row_json)

    @property
    def label(self) -> str:
        return paper_method_label(self.method_key)

    @property
    def color(self) -> str:
        return method_color(self.method_key)


# ============================================================
# Discovery
# ============================================================

def discover_method_bundles(gm_dir: Path) -> Dict[str, MethodBundle]:
    methods: Dict[str, MethodBundle] = {}
    if not gm_dir.exists():
        return methods

    for path in gm_dir.iterdir():
        if not path.is_file():
            continue
        for attr_name, pattern in METHOD_PATTERNS.items():
            m = pattern.match(path.name)
            if not m:
                continue
            if attr_name == "summary_ext_json":
                method = m.group("method1") or m.group("method2")
            else:
                method = m.group("method")
            bundle = methods.setdefault(
                method,
                MethodBundle(gm_name=gm_dir.name, method_key=method, gm_dir=gm_dir),
            )
            setattr(bundle, attr_name, path)
            break
    return methods


# ============================================================
# IO helpers
# ============================================================

def load_jsonl_records(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        first_nonempty = None
        all_lines = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            if first_nonempty is None:
                first_nonempty = s
            all_lines.append(s)

    if first_nonempty is None:
        return pd.DataFrame()

    # line-delimited JSON objects
    if first_nonempty.startswith("{"):
        for s in all_lines:
            rows.append(json.loads(s))
        return pd.DataFrame(rows)

    # fallback for array JSON
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    return pd.DataFrame([obj])


# ============================================================
# Metric helpers
# ============================================================

# ============================================================
# Metric registry
# ============================================================

METRIC_SPECS: Dict[str, Dict[str, Sequence[str]]] = {
    # -------------------------
    # existing global metrics
    # -------------------------
    "top1_acc": {
        "value_keys": [
            "top1_accuracy_on_chosen_policy",
            "accuracy_top1",
            "top1_accuracy",
        ],
        "bootstrap_keys": ["accuracy_top1"],
    },
    "top3_recall": {
        "value_keys": ["hit_top3", "top3_recall", "recall_top3"],
        "bootstrap_keys": ["hit_top3"],
    },
    "top5_recall": {
        "value_keys": ["hit_top5", "top5_recall", "recall_top5"],
        "bootstrap_keys": ["hit_top5"],
    },
    "top10_recall": {
        "value_keys": ["hit_top10", "top10_recall", "recall_top10"],
        "bootstrap_keys": ["hit_top10"],
    },
    "mrr": {
        "value_keys": ["mrr"],
        "bootstrap_keys": ["mrr"],
    },
    "mean_p_chosen": {
        "value_keys": ["mean_p_chosen_policy", "mean_p_chosen_pi", "p_chosen_pi"],
        "bootstrap_keys": ["mean_p_chosen_pi", "p_chosen_pi"],
    },
    "mean_logp_gap": {
        "value_keys": ["mean_logp_gap_policy_chosen_rejected", "mean_logp_gap_pi"],
        "bootstrap_keys": ["mean_logp_gap_pi"],
    },
    "mean_gap_improvement": {
        "value_keys": ["mean_gap_improvement", "gap_improve"],
        "bootstrap_keys": [],  # no upstream bootstrap key shown for this currently
    },
    "mean_kl": {
        "value_keys": ["mean_kl", "kl_pi_ref"],
        "bootstrap_keys": ["kl_pi_ref"],
    },
    "mean_ent_pi": {
        "value_keys": ["mean_ent_pi", "entropy_pi"],
        "bootstrap_keys": ["entropy_pi"],
    },
    "mean_ent_ref": {
        "value_keys": ["mean_ent_ref", "entropy_ref"],
        "bootstrap_keys": ["entropy_ref"],
    },

    # -------------------------
    # chosen NOT in Stockfish top-10
    # -------------------------
    "top1_acc_cond_not_top10": {
        "value_keys": [
            "top1_accuracy_on_chosen_policy_cond_on_not_in_top_ten",
            "accuracy_top1_cond_on_not_in_top_ten",
        ],
        "bootstrap_keys": ["accuracy_top1_cond_on_not_in_top_ten"],
    },
    "top3_recall_cond_not_top10": {
        "value_keys": [
            "top3_recall_cond_on_not_in_top_ten",
            "hit_top3_cond_on_not_in_top_ten",
        ],
        "bootstrap_keys": ["hit_top3_cond_on_not_in_top_ten"],
    },
    "top5_recall_cond_not_top10": {
        "value_keys": [
            "top5_recall_cond_on_not_in_top_ten",
            "hit_top5_cond_on_not_in_top_ten",
        ],
        "bootstrap_keys": ["hit_top5_cond_on_not_in_top_ten"],
    },
    "top10_recall_cond_not_top10": {
        "value_keys": [
            "top10_recall_cond_on_not_in_top_ten",
            "hit_top10_cond_on_not_in_top_ten",
        ],
        "bootstrap_keys": ["hit_top10_cond_on_not_in_top_ten"],
    },
    "mean_logp_gap_cond_not_top10": {
        "value_keys": [
            "mean_logp_gap_policy_chosen_rejected_cond_on_not_in_top_ten",
            "mean_logp_gap_pi_cond_on_not_in_top_ten",
        ],
        "bootstrap_keys": ["mean_logp_gap_pi_cond_on_not_in_top_ten"],
    },
    "mean_gap_improvement_cond_not_top10": {
        "value_keys": ["mean_gap_improvement_cond_on_not_in_top_ten"],
        "bootstrap_keys": [],
    },
    "mean_p_chosen_cond_not_top10": {
        "value_keys": [
            "mean_p_chosen_policy_cond_on_not_in_top_ten",
            "mean_p_chosen_pi_cond_on_not_in_top_ten",
            "p_chosen_pi_cond_on_not_in_top_ten",
        ],
        "bootstrap_keys": [
            "mean_p_chosen_pi_cond_on_not_in_top_ten",
            "p_chosen_pi_cond_on_not_in_top_ten",
        ],
    },
    "mean_kl_cond_not_top10": {
        "value_keys": [
            "mean_kl_cond_on_not_in_top_ten",
            "kl_pi_ref_cond_on_not_in_top_ten",
        ],
        "bootstrap_keys": ["kl_pi_ref_cond_on_not_in_top_ten"],
    },
    "mean_ent_pi_cond_not_top10": {
        "value_keys": [
            "mean_ent_pi_cond_on_not_in_top_ten",
            "entropy_pi_cond_on_not_in_top_ten",
        ],
        "bootstrap_keys": ["entropy_pi_cond_on_not_in_top_ten"],
    },
    "mean_ent_ref_cond_not_top10": {
        "value_keys": [
            "mean_ent_ref_cond_on_not_in_top_ten",
            "entropy_ref_cond_on_not_in_top_ten",
        ],
        "bootstrap_keys": ["entropy_ref_cond_on_not_in_top_ten"],
    },

    # -------------------------
    # chosen IN Stockfish top-10
    # -------------------------
    "top1_acc_cond_in_top10": {
        "value_keys": [
            "top1_accuracy_on_chosen_policy_cond_on_in_top_ten",
            "accuracy_top1_cond_on_in_top_ten",
        ],
        "bootstrap_keys": ["accuracy_top1_cond_on_in_top_ten"],
    },
    "top3_recall_cond_in_top10": {
        "value_keys": [
            "top3_recall_cond_on_in_top_ten",
            "hit_top3_cond_on_in_top_ten",
        ],
        "bootstrap_keys": ["hit_top3_cond_on_in_top_ten"],
    },
    "top5_recall_cond_in_top10": {
        "value_keys": [
            "top5_recall_cond_on_in_top_ten",
            "hit_top5_cond_on_in_top_ten",
        ],
        "bootstrap_keys": ["hit_top5_cond_on_in_top_ten"],
    },
    "top10_recall_cond_in_top10": {
        "value_keys": [
            "top10_recall_cond_on_in_top_ten",
            "hit_top10_cond_on_in_top_ten",
        ],
        "bootstrap_keys": ["hit_top10_cond_on_in_top_ten"],
    },
    "mean_logp_gap_cond_in_top10": {
        "value_keys": [
            "mean_logp_gap_policy_chosen_rejected_cond_on_in_top_ten",
            "mean_logp_gap_pi_cond_on_in_top_ten",
        ],
        "bootstrap_keys": ["mean_logp_gap_pi_cond_on_in_top_ten"],
    },
    "mean_gap_improvement_cond_in_top10": {
        "value_keys": ["mean_gap_improvement_cond_on_in_top_ten"],
        "bootstrap_keys": [],
    },
    "mean_p_chosen_cond_in_top10": {
        "value_keys": [
            "mean_p_chosen_policy_cond_on_in_top_ten",
            "mean_p_chosen_pi_cond_on_in_top_ten",
            "p_chosen_pi_cond_on_in_top_ten",
        ],
        "bootstrap_keys": [
            "mean_p_chosen_pi_cond_on_in_top_ten",
            "p_chosen_pi_cond_on_in_top_ten",
        ],
    },
    "mean_kl_cond_in_top10": {
        "value_keys": [
            "mean_kl_cond_on_in_top_ten",
            "kl_pi_ref_cond_on_in_top_ten",
        ],
        "bootstrap_keys": ["kl_pi_ref_cond_on_in_top_ten"],
    },
    "mean_ent_pi_cond_in_top10": {
        "value_keys": [
            "mean_ent_pi_cond_on_in_top_ten",
            "entropy_pi_cond_on_in_top_ten",
        ],
        "bootstrap_keys": ["entropy_pi_cond_on_in_top_ten"],
    },
    "mean_ent_ref_cond_in_top10": {
        "value_keys": [
            "mean_ent_ref_cond_on_in_top_ten",
            "entropy_ref_cond_on_in_top_ten",
        ],
        "bootstrap_keys": ["entropy_ref_cond_on_in_top_ten"],
    },
}

ALL_TABLE_METRICS = [
    # global
    "top1_acc",
    "top3_recall",
    "top5_recall",
    "top10_recall",
    "mrr",
    "mean_p_chosen",
    "mean_logp_gap",
    "mean_gap_improvement",
    "mean_kl",
    "mean_ent_pi",
    "mean_ent_ref",

    # not in top-10
    "top1_acc_cond_not_top10",
    "top3_recall_cond_not_top10",
    "top5_recall_cond_not_top10",
    "top10_recall_cond_not_top10",
    "mean_logp_gap_cond_not_top10",
    "mean_gap_improvement_cond_not_top10",
    "mean_p_chosen_cond_not_top10",
    "mean_kl_cond_not_top10",
    "mean_ent_pi_cond_not_top10",
    "mean_ent_ref_cond_not_top10",

    # in top-10
    "top1_acc_cond_in_top10",
    "top3_recall_cond_in_top10",
    "top5_recall_cond_in_top10",
    "top10_recall_cond_in_top10",
    "mean_logp_gap_cond_in_top10",
    "mean_gap_improvement_cond_in_top10",
    "mean_p_chosen_cond_in_top10",
    "mean_kl_cond_in_top10",
    "mean_ent_pi_cond_in_top10",
    "mean_ent_ref_cond_in_top10",
]

def try_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def get_summary_metric(bundle: MethodBundle, keys: Sequence[str]) -> float:
    for src in [bundle.summary_ext, bundle.summary_json_obj]:
        if src is None:
            continue

        for k in keys:
            if k in src and src[k] is not None:
                v = try_float(src[k])
                if not math.isnan(v):
                    return v

            bootstrap = src.get("bootstrap_ci_row", {})
            if k in bootstrap and bootstrap[k].get("mean") is not None:
                v = try_float(bootstrap[k]["mean"])
                if not math.isnan(v):
                    return v

    return float("nan")


def mean_ci_from_values(values: Sequence[float]) -> Tuple[float, float, float, int]:
    arr = np.asarray([float(v) for v in values if not math.isnan(float(v))], dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean, 1
    se = float(arr.std(ddof=1) / math.sqrt(n))
    half = 1.96 * se
    return mean, mean - half, mean + half, n


def get_opening_distribution(bundle: MethodBundle) -> Optional[Dict[str, float]]:
    if not bundle.opening_probe:
        return None
    white = bundle.opening_probe.get("white_first_move_probs", {}) or {}
    return {m: float(white.get(m, 0.0)) for m in OPENING_MOVE_ORDER}


def get_empirical_player_opening_distribution(bundle: MethodBundle) -> Optional[Dict[str, float]]:
    src = bundle.summary_ext or {}
    player_probe = src.get("player_opening_probe_empirical", {}) or {}
    if not player_probe:
        return None
    white_emp = player_probe.get("white_first_move_probs", {}) or {}
    return {m: float(white_emp.get(m, 0.0)) for m in OPENING_MOVE_ORDER}


def get_metric_value(bundle: MethodBundle, metric_name: str) -> float:
    spec = METRIC_SPECS[metric_name]
    return get_summary_metric(bundle, spec["value_keys"])


def get_metric_bootstrap_ci(bundle: MethodBundle, metric_name: str) -> Tuple[float, float, float]:
    spec = METRIC_SPECS[metric_name]
    for src in [bundle.summary_ext, bundle.summary_json_obj]:
        if src is None:
            continue
        bootstrap = src.get("bootstrap_ci_row", {}) or {}
        for k in spec["bootstrap_keys"]:
            obj = bootstrap.get(k)
            if not isinstance(obj, dict):
                continue
            mean = try_float(obj.get("mean"))
            lo = try_float(obj.get("lo"))
            hi = try_float(obj.get("hi"))
            if not (math.isnan(mean) and math.isnan(lo) and math.isnan(hi)):
                return mean, lo, hi
    return float("nan"), float("nan"), float("nan")


def derive_metrics(bundle: MethodBundle, metric_names: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric_name in metric_names:
        out[metric_name] = get_metric_value(bundle, metric_name)
        boot_mean, boot_lo, boot_hi = get_metric_bootstrap_ci(bundle, metric_name)
        out[f"{metric_name}__boot_mean"] = boot_mean
        out[f"{metric_name}__boot_lo"] = boot_lo
        out[f"{metric_name}__boot_hi"] = boot_hi
    return out


def safe_series_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def subset_mean(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    sub = df.loc[mask, col]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub, errors="coerce").dropna().mean())


def bool_equality_mean(df: pd.DataFrame, col_a: str, col_b: str, condition: Optional[pd.Series] = None) -> float:
    if col_a not in df.columns or col_b not in df.columns:
        return float("nan")
    sub = df
    if condition is not None:
        sub = df.loc[condition]
    if sub.empty:
        return float("nan")
    a = pd.to_numeric(sub[col_a], errors="coerce")
    b = pd.to_numeric(sub[col_b], errors="coerce")
    valid = ~(a.isna() | b.isna())
    if valid.sum() == 0:
        return float("nan")
    return float((a[valid] == b[valid]).mean())


# ============================================================
# Requirement checks
# ============================================================

def check_required_methods(
    eval_root: Path,
    gm_names: Sequence[str],
    required_methods: Sequence[str],
    require_opening_probe_for_methods: Optional[Sequence[str]] = None,
    require_row_metrics_for_methods: Optional[Sequence[str]] = None,
) -> Tuple[bool, List[str], Dict[str, Dict[str, MethodBundle]]]:
    require_opening_probe_for_methods = set(require_opening_probe_for_methods or [])
    require_row_metrics_for_methods = set(require_row_metrics_for_methods or [])
    missing: List[str] = []
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]] = {}

    for gm in gm_names:
        gm_dir = eval_root / gm
        if not gm_dir.exists():
            missing.append(f"{gm}: missing GM directory {gm_dir}")
            continue

        bundles = discover_method_bundles(gm_dir)
        bundles_by_gm[gm] = bundles

        for method in required_methods:
            if method not in bundles:
                missing.append(f"{gm}: missing method bundle '{method}'")
                continue

            bundle = bundles[method]

            if bundle.summary_ext_json is None and bundle.summary_json is None:
                missing.append(
                    f"{gm}: method '{method}' missing summary file "
                    f"(expected extended or standard eval_results json)"
                )

            if method in require_opening_probe_for_methods and bundle.opening_probe_json is None:
                missing.append(f"{gm}: method '{method}' missing opening probe json")

            if method in require_row_metrics_for_methods and bundle.per_row_json is None:
                missing.append(f"{gm}: method '{method}' missing per-row metrics json")

    return len(missing) == 0, missing, bundles_by_gm


# ============================================================
# Aggregate table builders
# ============================================================

def annotate_top2_bars(
    ax: plt.Axes,
    means: Sequence[float],
    extra_offset_frac: float = 0.0,
    fmt: str = "{:.2f}",
) -> None:
    vals = np.asarray(means, dtype=float)
    valid = np.isfinite(vals)
    if valid.sum() == 0:
        return

    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(vals[valid_idx])[::-1]]
    topk = order[:2]

    y0, y1 = ax.get_ylim()
    y_span = abs(y1 - y0)
    if y_span <= 0:
        y_span = 1.0

    base_offset = 0.02 * y_span + extra_offset_frac * y_span

    placed = []
    for rank, idx in enumerate(topk):
        y = vals[idx] + base_offset

        # if another annotation is too close in x and y, bump it upward
        for prev_x, prev_y in placed:
            if abs(idx - prev_x) <= 1 and abs(y - prev_y) < 0.035 * y_span:
                y += 0.04 * y_span

        ax.text(
            idx,
            y,
            fmt.format(vals[idx]),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold" if rank == 0 else "normal",
        )
        placed.append((idx, y))

def build_player_level_table(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
    metric_names: Sequence[str] = ALL_TABLE_METRICS,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for gm in GM_ORDER:
        if gm not in bundles_by_gm:
            continue
        gm_bundles = bundles_by_gm[gm]
        for method in methods:
            if method not in gm_bundles:
                continue
            bundle = gm_bundles[method]
            bundle.load()
            d = derive_metrics(bundle, metric_names)
            rows.append(
                {
                    "gm": gm,
                    "method_key": method,
                    "method_label": bundle.label,
                    **d,
                }
            )
    return pd.DataFrame(rows)

def build_aggregate_table(
    player_df: pd.DataFrame,
    metric_names: Sequence[str] = ALL_TABLE_METRICS,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for method_key, sub in player_df.groupby("method_key", sort=False):
        row: Dict[str, Any] = {
            "method_key": method_key,
            "method_label": str(sub["method_label"].iloc[0]),
            "n_gms": int(sub["gm"].nunique()),
        }

        for metric in metric_names:
            mean, lo, hi, n = mean_ci_from_values(sub[metric].tolist())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
            row[f"{metric}_n"] = n

            # also export average upstream row-bootstrap summaries across GMs
            # useful for diagnostics / table generation, though not the same thing
            # as across-GM uncertainty
            row[f"{metric}_boot_mean_avg"] = float(
                pd.to_numeric(sub.get(f"{metric}__boot_mean"), errors="coerce").dropna().mean()
            ) if f"{metric}__boot_mean" in sub.columns else float("nan")
            row[f"{metric}_boot_lo_avg"] = float(
                pd.to_numeric(sub.get(f"{metric}__boot_lo"), errors="coerce").dropna().mean()
            ) if f"{metric}__boot_lo" in sub.columns else float("nan")
            row[f"{metric}_boot_hi_avg"] = float(
                pd.to_numeric(sub.get(f"{metric}__boot_hi"), errors="coerce").dropna().mean()
            ) if f"{metric}__boot_hi" in sub.columns else float("nan")

        rows.append(row)

    order_map = {m: i for i, m in enumerate(player_df["method_key"].drop_duplicates().tolist())}
    out = pd.DataFrame(rows)
    if not out.empty:
        out["__order"] = out["method_key"].map(order_map)
        out = out.sort_values("__order").drop(columns="__order")
    return out

# ============================================================
# Figure helpers
# ============================================================

def write_df_exports(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    df.to_json(out_dir / f"{stem}.json", orient="records", indent=2)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finish_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_ci_line(
    ax: plt.Axes,
    x: Sequence[float],
    means: Sequence[float],
    los: Sequence[float],
    his: Sequence[float],
    label: str,
    color: str,
) -> None:
    x_arr = np.asarray(x, dtype=float)
    mean_arr = np.asarray(means, dtype=float)
    lo_arr = np.asarray(los, dtype=float)
    hi_arr = np.asarray(his, dtype=float)

    ax.plot(x_arr, mean_arr, marker="o", color=color, label=label)
    valid = ~(np.isnan(mean_arr) | np.isnan(lo_arr) | np.isnan(hi_arr))
    if valid.any():
        ax.fill_between(x_arr[valid], lo_arr[valid], hi_arr[valid], color=color, alpha=0.18)


def _plot_bar_with_ci(
    ax: plt.Axes,
    labels: Sequence[str],
    means: Sequence[float],
    los: Sequence[float],
    his: Sequence[float],
    colors: Sequence[str],
    title: str,
    percent: bool = False,
) -> None:
    xs = np.arange(len(labels))
    means_arr = np.asarray(means, dtype=float)
    yerr_lo = np.maximum(0.0, means_arr - np.asarray(los, dtype=float))
    yerr_hi = np.maximum(0.0, np.asarray(his, dtype=float) - means_arr)

    ax.bar(xs, means_arr, color=colors, width=0.72)
    ax.errorbar(
        xs,
        means_arr,
        yerr=np.vstack([yerr_lo, yerr_hi]),
        fmt="none",
        capsize=2,
        linewidth=1.0,
    )
    ax.set_title(title, pad=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))


def _plot_grouped_bars(
    ax: plt.Axes,
    category_labels: Sequence[str],
    series_labels: Sequence[str],
    series_means: Sequence[Sequence[float]],
    series_los: Sequence[Sequence[float]],
    series_his: Sequence[Sequence[float]],
    series_colors: Sequence[str],
    percent: bool,
    title: str,
    annotate_top2: bool = False,
) -> None:
    n_cat = len(category_labels)
    n_series = len(series_labels)
    xs = np.arange(n_cat)
    width = 0.82 / max(n_series, 1)

    all_xpos: List[np.ndarray] = []
    all_means: List[np.ndarray] = []

    for i, (label, means, los, his, color) in enumerate(
        zip(series_labels, series_means, series_los, series_his, series_colors)
    ):
        means_arr = np.asarray(means, dtype=float)
        los_arr = np.asarray(los, dtype=float)
        his_arr = np.asarray(his, dtype=float)
        xpos = xs - 0.41 + width / 2 + i * width

        ax.bar(xpos, means_arr, width=width, color=color, label=label)
        yerr_lo = np.maximum(0.0, means_arr - los_arr)
        yerr_hi = np.maximum(0.0, his_arr - means_arr)
        ax.errorbar(
            xpos,
            means_arr,
            yerr=np.vstack([yerr_lo, yerr_hi]),
            fmt="none",
            capsize=2,
            linewidth=0.9,
        )

        all_xpos.append(xpos)
        all_means.append(means_arr)

    ax.set_xticks(xs)
    ax.set_xticklabels(category_labels)
    ax.set_title(title, pad=4)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    if annotate_top2:
        # For each category (piece type), label the top 2 method means
        for j in range(n_cat):
            vals = np.array([all_means[i][j] for i in range(n_series)], dtype=float)
            valid = np.isfinite(vals)
            if valid.sum() == 0:
                continue

            valid_idx = np.where(valid)[0]
            order = valid_idx[np.argsort(vals[valid_idx])[::-1]]
            topk = order[:2]

            for rank, i in enumerate(topk):
                x = all_xpos[i][j]
                y = vals[i]
                ax.text(
                    x,
                    y + (0.012 if percent else 0.02),
                    f"{y:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontweight="bold" if rank == 0 else "normal",
                    rotation=90,
                )


# ============================================================
# Row-level figure aggregation for Experiment 3
# ============================================================

def compute_prob_over_entropy_vs_engine_likeness_player_table(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for gm in GM_ORDER:
        gm_bundles = bundles_by_gm.get(gm, {})
        for method in methods:
            bundle = gm_bundles.get(method)
            if bundle is None:
                continue

            bundle.load()
            df = bundle.per_row_df
            if df is None or df.empty:
                continue

            work = df.copy()

            # ---------- y-axis pieces ----------
            # avg probability the policy assigns to the chosen move
            work["p_chosen_pi_num"] = pd.to_numeric(work.get("p_chosen_pi"), errors="coerce")

            # avg entropy of the policy
            entropy_candidates = [
                "entropy_pi",
                "ent_pi",
                "pi_entropy",
                "policy_entropy",
            ]
            entropy_col = None
            for c in entropy_candidates:
                if c in work.columns:
                    entropy_col = c
                    break

            if entropy_col is None:
                continue

            work["entropy_pi_num"] = pd.to_numeric(work[entropy_col], errors="coerce")

            # ---------- x-axis piece ----------
            # "engine likeness" = average CP gap from best engine move
            # Use per-row predicted gap if present; otherwise fall back to:
            #   best_cp_all - min(stockfish top-10 cp)
            work["pred_cp_gap_num"] = pd.to_numeric(
                work.get("pred_cp_gap_to_engine_best"),
                errors="coerce",
            )

            def fallback_pred_gap_from_stockfish(stockfish_obj: Any) -> float:
                if not isinstance(stockfish_obj, dict):
                    return float("nan")

                best_cp = stockfish_obj.get("best_cp_all")
                moves = stockfish_obj.get("sf_moves_returned")

                try:
                    best_cp = float(best_cp)
                except Exception:
                    return float("nan")

                if not isinstance(moves, list) or len(moves) == 0:
                    return float("nan")

                cps: List[float] = []
                for item in moves:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    try:
                        cps.append(float(item[1]))
                    except Exception:
                        continue

                if not cps:
                    return float("nan")

                # min CP among returned MultiPV=10 candidates
                min_cp = min(cps)
                return best_cp - min_cp

            if "stockfish" in work.columns:
                fallback_gap = work["stockfish"].apply(fallback_pred_gap_from_stockfish)
                work["pred_cp_gap_num"] = work["pred_cp_gap_num"].fillna(fallback_gap)

            valid = ~(
                work["p_chosen_pi_num"].isna()
                | work["entropy_pi_num"].isna()
                | work["pred_cp_gap_num"].isna()
            )

            valid &= work["entropy_pi_num"] > 0.0

            if valid.sum() == 0:
                continue

            mean_p = float(work.loc[valid, "p_chosen_pi_num"].mean())
            mean_entropy = float(work.loc[valid, "entropy_pi_num"].mean())

            if not np.isfinite(mean_entropy) or mean_entropy <= 0.0:
                continue

            rows.append(
                {
                    "gm": gm,
                    "method_key": method,
                    "mean_p_chosen_pi": mean_p,
                    "mean_entropy_pi": mean_entropy,
                    "prob_over_entropy": mean_p / mean_entropy,
                    "mean_pred_cp_gap_to_engine_best": float(
                        work.loc[valid, "pred_cp_gap_num"].mean()
                    ),
                    "n_rows": int(valid.sum()),
                }
            )

    return pd.DataFrame(rows)

def _compute_engine_likeness_from_per_row_df(df: pd.DataFrame) -> pd.Series:
    pred_gap = pd.to_numeric(df.get("pred_cp_gap_to_engine_best"), errors="coerce")

    def fallback_pred_gap_from_stockfish(stockfish_obj: Any) -> float:
        if not isinstance(stockfish_obj, dict):
            return float("nan")

        best_cp = stockfish_obj.get("best_cp_all")
        moves = stockfish_obj.get("sf_moves_returned")

        try:
            best_cp = float(best_cp)
        except Exception:
            return float("nan")

        if not isinstance(moves, list) or len(moves) == 0:
            return float("nan")

        cps: List[float] = []
        for item in moves:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                cps.append(float(item[1]))
            except Exception:
                continue

        if not cps:
            return float("nan")

        # Use min cp among returned MultiPV=10 candidates
        min_cp = min(cps)
        return best_cp - min_cp

    if "stockfish" in df.columns:
        fallback_gap = df["stockfish"].apply(fallback_pred_gap_from_stockfish)
        pred_gap = pred_gap.fillna(fallback_gap)

    return pred_gap

def compute_entropy_ratio_vs_engine_likeness_player_table(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    entropy_pi_candidates = ["entropy_pi", "ent_pi", "pi_entropy", "policy_entropy"]
    entropy_ref_candidates = ["entropy_ref", "ent_ref", "ref_entropy", "reference_entropy"]

    for gm in GM_ORDER:
        gm_bundles = bundles_by_gm.get(gm, {})
        for method in methods:
            bundle = gm_bundles.get(method)
            if bundle is None:
                continue

            bundle.load()
            df = bundle.per_row_df
            if df is None or df.empty:
                continue

            work = df.copy()

            pi_col = next((c for c in entropy_pi_candidates if c in work.columns), None)
            ref_col = next((c for c in entropy_ref_candidates if c in work.columns), None)

            if pi_col is None or ref_col is None:
                continue

            work["entropy_pi_num"] = pd.to_numeric(work[pi_col], errors="coerce")
            work["entropy_ref_num"] = pd.to_numeric(work[ref_col], errors="coerce")
            work["pred_cp_gap_num"] = _compute_engine_likeness_from_per_row_df(work)

            valid = ~(
                work["entropy_pi_num"].isna()
                | work["entropy_ref_num"].isna()
                | work["pred_cp_gap_num"].isna()
            )
            valid &= work["entropy_ref_num"] > 0.0

            if valid.sum() == 0:
                continue

            mean_entropy_pi = float(work.loc[valid, "entropy_pi_num"].mean())
            mean_entropy_ref = float(work.loc[valid, "entropy_ref_num"].mean())

            if not np.isfinite(mean_entropy_ref) or mean_entropy_ref <= 0.0:
                continue

            rows.append(
                {
                    "gm": gm,
                    "method_key": method,
                    "mean_entropy_pi": mean_entropy_pi,
                    "mean_entropy_ref": mean_entropy_ref,
                    "entropy_ratio_pi_over_ref": mean_entropy_pi / mean_entropy_ref,
                    "mean_pred_cp_gap_to_engine_best": float(
                        work.loc[valid, "pred_cp_gap_num"].mean()
                    ),
                    "n_rows": int(valid.sum()),
                }
            )

    return pd.DataFrame(rows)

def compute_prob_ratio_vs_engine_likeness_player_table(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    p_ref_candidates = [
        "p_chosen_ref",
        "mean_p_chosen_ref",
        "p_chosen_maia2",
        "p_chosen_base",
    ]

    for gm in GM_ORDER:
        gm_bundles = bundles_by_gm.get(gm, {})
        for method in methods:
            bundle = gm_bundles.get(method)
            if bundle is None:
                continue

            bundle.load()
            df = bundle.per_row_df
            if df is None or df.empty:
                continue

            work = df.copy()

            if "p_chosen_pi" not in work.columns:
                continue

            ref_col = next((c for c in p_ref_candidates if c in work.columns), None)
            if ref_col is None:
                continue

            work["p_chosen_pi_num"] = pd.to_numeric(work["p_chosen_pi"], errors="coerce")
            work["p_chosen_ref_num"] = pd.to_numeric(work[ref_col], errors="coerce")
            work["pred_cp_gap_num"] = _compute_engine_likeness_from_per_row_df(work)

            valid = ~(
                work["p_chosen_pi_num"].isna()
                | work["p_chosen_ref_num"].isna()
                | work["pred_cp_gap_num"].isna()
            )
            valid &= work["p_chosen_ref_num"] > 0.0

            if valid.sum() == 0:
                continue

            mean_p_pi = float(work.loc[valid, "p_chosen_pi_num"].mean())
            mean_p_ref = float(work.loc[valid, "p_chosen_ref_num"].mean())

            if not np.isfinite(mean_p_ref) or mean_p_ref <= 0.0:
                continue

            rows.append(
                {
                    "gm": gm,
                    "method_key": method,
                    "mean_p_chosen_pi": mean_p_pi,
                    "mean_p_chosen_ref": mean_p_ref,
                    "p_chosen_ratio_pi_over_ref": mean_p_pi / mean_p_ref,
                    "mean_pred_cp_gap_to_engine_best": float(
                        work.loc[valid, "pred_cp_gap_num"].mean()
                    ),
                    "n_rows": int(valid.sum()),
                }
            )

    return pd.DataFrame(rows)

def compute_prob_vs_cp_gap_player_table(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for gm in GM_ORDER:
        gm_bundles = bundles_by_gm.get(gm, {})
        for method in methods:
            bundle = gm_bundles.get(method)
            if bundle is None:
                continue

            bundle.load()
            df = bundle.per_row_df
            if df is None or df.empty:
                continue

            if "p_chosen_pi" not in df.columns:
                continue

            work = df.copy()
            work["p_chosen_pi_num"] = pd.to_numeric(work["p_chosen_pi"], errors="coerce")
            work["pred_cp_gap_num"] = pd.to_numeric(work.get("pred_cp_gap_to_engine_best"), errors="coerce")

            def fallback_pred_gap_from_stockfish(stockfish_obj: Any) -> float:
                if not isinstance(stockfish_obj, dict):
                    return float("nan")

                best_cp = stockfish_obj.get("best_cp_all")
                moves = stockfish_obj.get("sf_moves_returned")

                try:
                    best_cp = float(best_cp)
                except Exception:
                    return float("nan")

                if not isinstance(moves, list) or len(moves) == 0:
                    return float("nan")

                cps: List[float] = []
                for item in moves:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    try:
                        cps.append(float(item[1]))
                    except Exception:
                        continue

                if not cps:
                    return float("nan")

                worst_cp = min(cps)   # lowest cp = worst move among returned stockfish set
                return best_cp - worst_cp

            if "stockfish" in work.columns:
                fallback_gap = work["stockfish"].apply(fallback_pred_gap_from_stockfish)
                work["pred_cp_gap_num"] = work["pred_cp_gap_num"].fillna(fallback_gap)

            valid = ~(work["p_chosen_pi_num"].isna() | work["pred_cp_gap_num"].isna())
            if valid.sum() == 0:
                continue

            rows.append(
                {
                    "gm": gm,
                    "method_key": method,
                    "mean_p_chosen_pi": float(work.loc[valid, "p_chosen_pi_num"].mean()),
                    "mean_pred_cp_gap_to_engine_best": float(work.loc[valid, "pred_cp_gap_num"].mean()),
                    "n_rows": int(valid.sum()),
                }
            )

    return pd.DataFrame(rows)

def aggregate_ratio_vs_engine_likeness_player_table(
    player_df: pd.DataFrame,
    y_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if player_df.empty:
        return pd.DataFrame()

    for method_key, sub in player_df.groupby("method_key", sort=False):
        x_mean, x_lo, x_hi, x_n = mean_ci_from_values(
            sub["mean_pred_cp_gap_to_engine_best"].tolist()
        )
        y_mean, y_lo, y_hi, y_n = mean_ci_from_values(
            sub[y_col].tolist()
        )

        rows.append(
            {
                "method_key": method_key,
                "x_mean": x_mean,
                "x_ci_lo": x_lo,
                "x_ci_hi": x_hi,
                "x_n_gms": x_n,
                "y_mean": y_mean,
                "y_ci_lo": y_lo,
                "y_ci_hi": y_hi,
                "y_n_gms": y_n,
            }
        )

    return pd.DataFrame(rows)

def aggregate_prob_over_entropy_vs_engine_likeness_player_table(
    player_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if player_df.empty:
        return pd.DataFrame()

    for method_key, sub in player_df.groupby("method_key", sort=False):
        x_mean, x_lo, x_hi, x_n = mean_ci_from_values(
            sub["mean_pred_cp_gap_to_engine_best"].tolist()
        )
        y_mean, y_lo, y_hi, y_n = mean_ci_from_values(
            sub["prob_over_entropy"].tolist()
        )

        rows.append(
            {
                "method_key": method_key,
                "x_mean": x_mean,
                "x_ci_lo": x_lo,
                "x_ci_hi": x_hi,
                "x_n_gms": x_n,
                "y_mean": y_mean,
                "y_ci_lo": y_lo,
                "y_ci_hi": y_hi,
                "y_n_gms": y_n,
            }
        )

    return pd.DataFrame(rows)

def aggregate_prob_vs_cp_gap_player_table(player_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if player_df.empty:
        return pd.DataFrame()

    for method_key, sub in player_df.groupby("method_key", sort=False):
        x_mean, x_lo, x_hi, x_n = mean_ci_from_values(sub["mean_pred_cp_gap_to_engine_best"].tolist())
        y_mean, y_lo, y_hi, y_n = mean_ci_from_values(sub["mean_p_chosen_pi"].tolist())

        rows.append(
            {
                "method_key": method_key,
                "x_mean": x_mean,
                "x_ci_lo": x_lo,
                "x_ci_hi": x_hi,
                "x_n_gms": x_n,
                "y_mean": y_mean,
                "y_ci_lo": y_lo,
                "y_ci_hi": y_hi,
                "y_n_gms": y_n,
            }
        )

    return pd.DataFrame(rows)

def compute_piece_type_player_table(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for gm in GM_ORDER:
        gm_bundles = bundles_by_gm.get(gm, {})
        for method in methods:
            bundle = gm_bundles.get(method)
            if bundle is None:
                continue
            bundle.load()
            df = bundle.per_row_df
            if df is None or df.empty or "chosen_piece_type" not in df.columns:
                continue

            for piece in PIECE_ORDER:
                mask = df["chosen_piece_type"].astype(str) == piece
                col = f"pi_top1_matches_player_piece_type_{piece}"
                val = subset_mean(df, mask, col)
                rows.append(
                    {
                        "gm": gm,
                        "method_key": method,
                        "piece_type": piece,
                        "value": val,
                    }
                )
    return pd.DataFrame(rows)


def compute_surface_player_table(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = [
        ("Quiet", "chosen_is_quiet", "pred_pi_is_quiet"),
        ("Capture", "chosen_is_capture", "pred_pi_is_capture"),
        ("Check", "chosen_is_check", "pred_pi_is_check"),
    ]

    for gm in GM_ORDER:
        gm_bundles = bundles_by_gm.get(gm, {})
        for method in methods:
            bundle = gm_bundles.get(method)
            if bundle is None:
                continue
            bundle.load()
            df = bundle.per_row_df
            if df is None or df.empty:
                continue

            for display_name, chosen_col, pred_col in metrics:
                # condition on player selecting that action type
                if chosen_col not in df.columns or pred_col not in df.columns:
                    val = float("nan")
                else:
                    mask = pd.to_numeric(df[chosen_col], errors="coerce") == 1.0
                    val = subset_mean(df, mask, pred_col)

                rows.append(
                    {
                        "gm": gm,
                        "method_key": method,
                        "metric": display_name,
                        "value": val,
                    }
                )
    return pd.DataFrame(rows)

def compute_deep_player_table(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for gm in GM_ORDER:
        gm_bundles = bundles_by_gm.get(gm, {})
        for method in methods:
            bundle = gm_bundles.get(method)
            if bundle is None:
                continue
            bundle.load()
            df = bundle.per_row_df
            if df is None or df.empty:
                continue

            # Agreement metrics
            tactical_agree = subset_mean(
                df,
                pd.Series([True] * len(df), index=df.index),
                "player_vs_pi_style_agree_tactical",
            )
            positional_agree = subset_mean(
                df,
                pd.Series([True] * len(df), index=df.index),
                "player_vs_pi_style_agree_positional",
            )

            # Conditional MRR metrics
            if "chosen_is_tactical" in df.columns and "mrr" in df.columns:
                tactical_mask = pd.to_numeric(df["chosen_is_tactical"], errors="coerce") == 1.0
                tactical_mrr = subset_mean(df, tactical_mask, "mrr")
            else:
                tactical_mrr = float("nan")

            if "chosen_is_positional" in df.columns and "mrr" in df.columns:
                positional_mask = pd.to_numeric(df["chosen_is_positional"], errors="coerce") == 1.0
                positional_mrr = subset_mean(df, positional_mask, "mrr")
            else:
                positional_mrr = float("nan")

            rows.extend(
                [
                    {"gm": gm, "method_key": method, "metric": "Tactical Agreement", "value": tactical_agree},
                    {"gm": gm, "method_key": method, "metric": "Positional Agreement", "value": positional_agree},
                    {"gm": gm, "method_key": method, "metric": "Tactical MRR", "value": tactical_mrr},
                    {"gm": gm, "method_key": method, "metric": "Positional MRR", "value": positional_mrr},
                ]
            )
    return pd.DataFrame(rows)

def aggregate_player_metric_table(player_metric_df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if player_metric_df.empty:
        return pd.DataFrame()

    for (method_key, category), sub in player_metric_df.groupby(["method_key", category_col], sort=False):
        mean, lo, hi, n = mean_ci_from_values(sub["value"].tolist())
        rows.append(
            {
                "method_key": method_key,
                category_col: category,
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_gms": n,
            }
        )
    return pd.DataFrame(rows)

def _plot_ratio_vs_engine_likeness_scatter(
    exp_dir: Path,
    player_df: pd.DataFrame,
    y_col: str,
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
    stem: str,
    title: str,
    y_label: str,
) -> None:
    agg_df = aggregate_ratio_vs_engine_likeness_player_table(player_df, y_col=y_col)

    player_df.to_csv(exp_dir / f"{stem}_player_level.csv", index=False)
    agg_df.to_csv(exp_dir / f"{stem}_aggregate.csv", index=False)

    agg_lookup = {row["method_key"]: row for _, row in agg_df.iterrows()}

    fig, ax = plt.subplots(figsize=(4.7, 3.35), constrained_layout=True)

    xs_all: List[float] = []
    ys_all: List[float] = []

    for method, label, color in zip(methods, labels, colors):
        row = agg_lookup.get(method)
        if row is None:
            continue

        x = try_float(row["x_mean"])
        x_lo = try_float(row["x_ci_lo"])
        x_hi = try_float(row["x_ci_hi"])

        y = try_float(row["y_mean"])
        y_lo = try_float(row["y_ci_lo"])
        y_hi = try_float(row["y_ci_hi"])

        if math.isnan(x) or math.isnan(y):
            continue

        xs_all.append(x)
        ys_all.append(y)

        xerr = np.array([[max(0.0, x - x_lo)], [max(0.0, x_hi - x)]], dtype=float)
        yerr = np.array([[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]], dtype=float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=color,
            markersize=5.5,
            capsize=2.5,
            linewidth=1.0,
            alpha=0.95,
        )

        ax.annotate(
            label,
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.4,
        )

    ax.set_title(title, pad=4)
    ax.set_xlabel("Engine distance (mean CP gap from engine best)")
    ax.set_ylabel(y_label)

    if xs_all:
        x_min, x_max = min(xs_all), max(xs_all)
        x_span = x_max - x_min
        x_pad = max(3.0, 0.10 * x_span if x_span > 0 else 5.0)
        ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)

    if ys_all:
        y_min, y_max = min(ys_all), max(ys_all)
        y_span = y_max - y_min
        y_pad = max(0.02, 0.10 * y_span if y_span > 0 else 0.05)
        ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    # helpful visual baseline for "same as ref"
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="gray", alpha=0.9)

    finish_figure(fig, exp_dir, stem)

def plot_entropy_ratio_vs_engine_likeness_scatter_from_per_row(
    exp_dir: Path,
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
    stem: str,
    title: str,
) -> None:
    player_df = compute_entropy_ratio_vs_engine_likeness_player_table(
        bundles_by_gm,
        methods,
    )

    _plot_ratio_vs_engine_likeness_scatter(
        exp_dir=exp_dir,
        player_df=player_df,
        y_col="entropy_ratio_pi_over_ref",
        methods=methods,
        labels=labels,
        colors=colors,
        stem=stem,
        title=title,
        y_label=r"Mean entropy ratio $\bar{H}_{\pi} / \bar{H}_{\mathrm{ref}}$",
    )

def plot_prob_ratio_vs_engine_likeness_scatter_from_per_row(
    exp_dir: Path,
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
    stem: str,
    title: str,
) -> None:
    player_df = compute_prob_ratio_vs_engine_likeness_player_table(
        bundles_by_gm,
        methods,
    )

    _plot_ratio_vs_engine_likeness_scatter(
        exp_dir=exp_dir,
        player_df=player_df,
        y_col="p_chosen_ratio_pi_over_ref",
        methods=methods,
        labels=labels,
        colors=colors,
        stem=stem,
        title=title,
        y_label=r"Chosen-probability ratio $\bar{p}_{\mathrm{chosen},\pi} / \bar{p}_{\mathrm{chosen},\mathrm{ref}}$",
    )

def plot_prob_over_entropy_vs_engine_likeness_scatter_from_per_row(
    exp_dir: Path,
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
    stem: str,
    title: str,
) -> None:
    player_df = compute_prob_over_entropy_vs_engine_likeness_player_table(
        bundles_by_gm,
        methods,
    )
    agg_df = aggregate_prob_over_entropy_vs_engine_likeness_player_table(player_df)

    player_df.to_csv(exp_dir / f"{stem}_player_level.csv", index=False)
    agg_df.to_csv(exp_dir / f"{stem}_aggregate.csv", index=False)

    agg_lookup = {row["method_key"]: row for _, row in agg_df.iterrows()}

    fig, ax = plt.subplots(figsize=(4.7, 3.35), constrained_layout=True)

    xs_all: List[float] = []
    ys_all: List[float] = []

    for method, label, color in zip(methods, labels, colors):
        row = agg_lookup.get(method)
        if row is None:
            continue

        x = try_float(row["x_mean"])
        x_lo = try_float(row["x_ci_lo"])
        x_hi = try_float(row["x_ci_hi"])

        y = try_float(row["y_mean"])
        y_lo = try_float(row["y_ci_lo"])
        y_hi = try_float(row["y_ci_hi"])

        if math.isnan(x) or math.isnan(y):
            continue

        xs_all.append(x)
        ys_all.append(y)

        xerr = np.array([[max(0.0, x - x_lo)], [max(0.0, x_hi - x)]], dtype=float)
        yerr = np.array([[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]], dtype=float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=color,
            markersize=5.5,
            capsize=2.5,
            linewidth=1.0,
            alpha=0.95,
        )

        ax.annotate(
            label,
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.4,
        )

    ax.set_title(title, pad=4)
    ax.set_xlabel("Engine likeness (mean CP gap from engine best)")
    ax.set_ylabel("Mean π(chosen) / mean entropy")

    if xs_all:
        x_min, x_max = min(xs_all), max(xs_all)
        x_span = x_max - x_min
        x_pad = max(3.0, 0.10 * x_span if x_span > 0 else 5.0)
        ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)

    if ys_all:
        y_min, y_max = min(ys_all), max(ys_all)
        y_span = y_max - y_min
        y_pad = max(0.01, 0.10 * y_span if y_span > 0 else 0.03)
        ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    finish_figure(fig, exp_dir, stem)

def plot_prob_vs_cp_gap_scatter_from_per_row(
    exp_dir: Path,
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
    stem: str,
    title: str,
) -> None:
    player_df = compute_prob_vs_cp_gap_player_table(bundles_by_gm, methods)
    agg_df = aggregate_prob_vs_cp_gap_player_table(player_df)

    player_df.to_csv(exp_dir / f"{stem}_player_level.csv", index=False)
    agg_df.to_csv(exp_dir / f"{stem}_aggregate.csv", index=False)

    agg_lookup = {row["method_key"]: row for _, row in agg_df.iterrows()}

    fig, ax = plt.subplots(figsize=(4.7, 3.35), constrained_layout=True)

    xs_all: List[float] = []
    ys_all: List[float] = []

    for method, label, color in zip(methods, labels, colors):
        row = agg_lookup.get(method)
        if row is None:
            continue

        x = try_float(row["x_mean"])
        x_lo = try_float(row["x_ci_lo"])
        x_hi = try_float(row["x_ci_hi"])

        y = try_float(row["y_mean"])
        y_lo = try_float(row["y_ci_lo"])
        y_hi = try_float(row["y_ci_hi"])

        if math.isnan(x) or math.isnan(y):
            continue

        xs_all.append(x)
        ys_all.append(y)

        xerr = np.array([[max(0.0, x - x_lo)], [max(0.0, x_hi - x)]], dtype=float)
        yerr = np.array([[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]], dtype=float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=color,
            markersize=5.5,
            capsize=2.5,
            linewidth=1.0,
            alpha=0.95,
        )

        ax.annotate(
            label,
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.4,
        )

    ax.set_title(title, pad=4)
    ax.set_xlabel("Mean CP gap to engine best")
    ax.set_ylabel("Mean π(chosen)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    if xs_all:
        x_min, x_max = min(xs_all), max(xs_all)
        x_span = x_max - x_min
        x_pad = max(3.0, 0.10 * x_span if x_span > 0 else 5.0)
        ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)

    if ys_all:
        y_min, y_max = min(ys_all), max(ys_all)
        y_span = y_max - y_min
        y_pad = max(0.01, 0.10 * y_span if y_span > 0 else 0.03)
        ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))

    finish_figure(fig, exp_dir, stem)

def plot_conditional_metrics_figure(
    exp_dir: Path,
    agg_df: pd.DataFrame,
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
    cond_suffix: str,
    stem: str,
    suptitle: str,
) -> None:
    agg_lookup = {row["method_key"]: row for _, row in agg_df.iterrows()}

    metrics = [
        (f"mean_p_chosen_{cond_suffix}", "Mean π(chosen)", True),
        (f"mean_logp_gap_{cond_suffix}", "Log π gap", False),
        (f"mean_gap_improvement_{cond_suffix}", "Gap improvement", False),
        (f"top1_acc_{cond_suffix}", "Top-1 recall", True),
        (f"mean_kl_{cond_suffix}", "KL vs Maia-2", False),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(13.0, 2.75), constrained_layout=True)

    for ax, (metric, title, percent) in zip(axes, metrics):
        means = [agg_lookup[m].get(f"{metric}_mean", float("nan")) for m in methods]
        los = [agg_lookup[m].get(f"{metric}_ci_lo", float("nan")) for m in methods]
        his = [agg_lookup[m].get(f"{metric}_ci_hi", float("nan")) for m in methods]

        _plot_bar_with_ci(
            ax=ax,
            labels=labels,
            means=means,
            los=los,
            his=his,
            colors=colors,
            title=title,
            percent=percent,
        )
        annotate_top2_bars(ax, means)

    fig.suptitle(suptitle, y=1.03, fontsize=8.5)
    finish_figure(fig, exp_dir, stem)

def plot_conditional_recall_figure(
    exp_dir: Path,
    agg_df: pd.DataFrame,
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
    cond_suffix: str,
    stem: str,
    suptitle: str,
) -> None:
    agg_lookup = {row["method_key"]: row for _, row in agg_df.iterrows()}

    metrics = [
        (f"top1_acc_{cond_suffix}", "Top-1 recall"),
        (f"top3_recall_{cond_suffix}", "Top-3 recall"),
        (f"top5_recall_{cond_suffix}", "Top-5 recall"),
        (f"top10_recall_{cond_suffix}", "Top-10 recall"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.75), constrained_layout=True)

    for ax, (metric, title) in zip(axes, metrics):
        means = [agg_lookup[m].get(f"{metric}_mean", float("nan")) for m in methods]
        los = [agg_lookup[m].get(f"{metric}_ci_lo", float("nan")) for m in methods]
        his = [agg_lookup[m].get(f"{metric}_ci_hi", float("nan")) for m in methods]

        _plot_bar_with_ci(
            ax=ax,
            labels=labels,
            means=means,
            los=los,
            his=his,
            colors=colors,
            title=title,
            percent=True,
        )
        ax.set_ylim(0.0, 1.0)
        annotate_top2_bars(ax, means)

    fig.suptitle(suptitle, y=1.03, fontsize=8.5)
    finish_figure(fig, exp_dir, stem)

def plot_piece_type_fidelity(
    exp_dir: Path,
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
) -> None:
    player_df = compute_piece_type_player_table(bundles_by_gm, methods)
    agg_df = aggregate_player_metric_table(player_df, "piece_type")

    write_df_exports(player_df, exp_dir, "piece_type_player_level")
    write_df_exports(agg_df, exp_dir, "piece_type_aggregate")

    fig, ax = plt.subplots(figsize=(7.0, 2.6), constrained_layout=True)

    series_means, series_los, series_his = [], [], []
    for method in methods:
        sub = agg_df[agg_df["method_key"] == method].set_index("piece_type")
        series_means.append([try_float(sub["mean"].get(piece, float("nan"))) for piece in PIECE_ORDER])
        series_los.append([try_float(sub["ci_lo"].get(piece, float("nan"))) for piece in PIECE_ORDER])
        series_his.append([try_float(sub["ci_hi"].get(piece, float("nan"))) for piece in PIECE_ORDER])

    display_pieces = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]
    _plot_grouped_bars(
        ax=ax,
        category_labels=display_pieces,
        series_labels=labels,
        series_means=series_means,
        series_los=series_los,
        series_his=series_his,
        series_colors=colors,
        percent=True,
        title="Piece-type selection fidelity",
        annotate_top2=True,
    )
    ax.set_ylabel("Match rate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=len(labels), frameon=False)

    finish_figure(fig, exp_dir, "piece_type_fidelity_all_gms")


def plot_surface_style_agreement(
    exp_dir: Path,
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
) -> None:
    player_df = compute_surface_player_table(bundles_by_gm, methods)
    agg_df = aggregate_player_metric_table(player_df, "metric")

    write_df_exports(player_df, exp_dir, "surface_style_player_level")
    write_df_exports(agg_df, exp_dir, "surface_style_aggregate")

    fig, ax = plt.subplots(figsize=(5.8, 2.5), constrained_layout=True)

    metric_order = ["Quiet", "Capture", "Check"]
    series_means, series_los, series_his = [], [], []
    for method in methods:
        sub = agg_df[agg_df["method_key"] == method].set_index("metric")
        series_means.append([try_float(sub["mean"].get(m, float("nan"))) for m in metric_order])
        series_los.append([try_float(sub["ci_lo"].get(m, float("nan"))) for m in metric_order])
        series_his.append([try_float(sub["ci_hi"].get(m, float("nan"))) for m in metric_order])

    _plot_grouped_bars(
        ax=ax,
        category_labels=metric_order,
        series_labels=labels,
        series_means=series_means,
        series_los=series_los,
        series_his=series_his,
        series_colors=colors,
        percent=True,
        title="Surface-level style agreement",
    )
    ax.set_ylabel("Conditional agreement")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=len(labels), frameon=False)

    finish_figure(fig, exp_dir, "surface_style_agreement_all_gms")

def plot_experiment3_recall_figure(
    exp_dir: Path,
    agg_df: pd.DataFrame,
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
) -> None:
    agg_lookup = {row["method_key"]: row for _, row in agg_df.iterrows()}

    recall_metrics = [
        ("top1_acc", "Top-1 recall"),
        ("top3_recall", "Top-3 recall"),
        ("top5_recall", "Top-5 recall"),
        ("top10_recall", "Top-10 recall"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.5), constrained_layout=True)

    for ax, (metric, title) in zip(axes, recall_metrics):
        means = [agg_lookup[m][f"{metric}_mean"] for m in methods]
        los = [agg_lookup[m][f"{metric}_ci_lo"] for m in methods]
        his = [agg_lookup[m][f"{metric}_ci_hi"] for m in methods]

        _plot_bar_with_ci(
            ax=ax,
            labels=labels,
            means=means,
            los=los,
            his=his,
            colors=colors,
            title=title,
            percent=True,
        )
        ax.set_ylim(0.0, 1.0)

        # label top 2 means
        vals = np.array(means, dtype=float)
        valid = np.isfinite(vals)
        if valid.sum() > 0:
            valid_idx = np.where(valid)[0]
            order = valid_idx[np.argsort(vals[valid_idx])[::-1]]
            topk = order[:2]

            for rank, idx in enumerate(topk):
                ax.text(
                    idx,
                    vals[idx] + 0.02,
                    f"{vals[idx]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold" if rank == 0 else "normal",
                )

    finish_figure(fig, exp_dir, "exp3_recall_all_gms")

def plot_deep_style_agreement(
    exp_dir: Path,
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
) -> None:
    player_df = compute_deep_player_table(bundles_by_gm, methods)
    agg_df = aggregate_player_metric_table(player_df, "metric")

    write_df_exports(player_df, exp_dir, "deep_style_player_level")
    write_df_exports(agg_df, exp_dir, "deep_style_aggregate")

    metric_panels = [
        ("Tactical Agreement", True),
        ("Positional Agreement", True),
        ("Tactical MRR", True),
        ("Positional MRR", True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5), constrained_layout=True)

    for ax, (metric_name, percent) in zip(axes, metric_panels):
        sub_metric = agg_df[agg_df["metric"] == metric_name].set_index("method_key")
        means = [try_float(sub_metric["mean"].get(m, float("nan"))) for m in methods]
        los = [try_float(sub_metric["ci_lo"].get(m, float("nan"))) for m in methods]
        his = [try_float(sub_metric["ci_hi"].get(m, float("nan"))) for m in methods]
        _plot_bar_with_ci(ax, labels, means, los, his, colors, title=metric_name, percent=percent)
        
        
        # annotate top-2 means
        vals = np.array(means, dtype=float)
        valid = np.isfinite(vals)

        if valid.sum() > 0:
            order = np.argsort(vals[valid])[::-1]
            valid_idx = np.where(valid)[0]

            topk = valid_idx[order[:2]]

            for rank, idx in enumerate(topk):
                weight = "bold" if rank == 0 else "normal"

                ax.text(
                    idx,
                    vals[idx] + 0.03,
                    f"{vals[idx]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight=weight,
                )
        ax.set_ylim(0.0, 1.0)

    finish_figure(fig, exp_dir, "deep_style_agreement_all_gms")

# ============================================================
# Experiment 1: beta sweep across 9 GMs
# ============================================================

def run_experiment1(eval_root: Path, out_dir: Path) -> None:
    exp_dir = out_dir / "experiment1"
    ensure_dir(exp_dir)

    methods = [
        "maia2",
        "sft",
        "sft_pairwise",
        "dpo_beta=0.02",
        "dpo_beta=0.05",
        "dpo_beta=0.10",
        "dpo_beta=0.20",
        "dpo_beta=0.40",
        "dpo_beta=0.60",
    ]
    ok, missing, bundles_by_gm = check_required_methods(
        eval_root,
        GM_ORDER,
        methods,
        require_row_metrics_for_methods=methods,
    )
    if not ok:
        print("\n[SKIP] Experiment 1 skipped. Missing files / bundles:")
        for line in missing:
            print(" -", line)
        return

    player_df = build_player_level_table(bundles_by_gm, methods, metric_names=ALL_TABLE_METRICS)
    agg_df = build_aggregate_table(player_df, metric_names=ALL_TABLE_METRICS)

    write_df_exports(player_df, exp_dir, "experiment1_player_level")
    write_df_exports(agg_df, exp_dir, "experiment1_aggregate")

    beta_methods = [m for m in methods if m.startswith("dpo_beta=")]
    beta_values = [float(m.split("=")[1]) for m in beta_methods]

    agg_lookup = {row["method_key"]: row for _, row in agg_df.iterrows()}

    metrics = [
        ("mean_p_chosen", "P(chosen)", True),
        ("mrr", "MRR", True),
        ("mean_logp_gap", "Log-probability gap", False),
        ("mean_kl", "KL vs Maia-2", False),
        ("top1_acc", "Top-1 recall", True),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(11.2, 2.55), constrained_layout=True)
    for ax, (metric, title, percent) in zip(axes, metrics):
        means = [agg_lookup[m][f"{metric}_mean"] for m in beta_methods]
        los = [agg_lookup[m][f"{metric}_ci_lo"] for m in beta_methods]
        his = [agg_lookup[m][f"{metric}_ci_hi"] for m in beta_methods]
        _plot_ci_line(
            ax,
            beta_values,
            means,
            los,
            his,
            label="DPO β sweep",
            color=method_color("dpo_beta=0.60"),
        )
        ax.set_xlabel("β")
        ax.set_title(title, pad=4)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        for baseline in ["maia2", "sft", "sft_pairwise"]:
            row = agg_lookup[baseline]
            y = row[f"{metric}_mean"]
            if not math.isnan(float(y)):
                ax.axhline(
                    float(y),
                    linestyle=":",
                    linewidth=1.0,
                    color=method_color(baseline),
                    label=paper_method_label(baseline),
                )

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    fig.legend(uniq_h, uniq_l, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False)

    finish_figure(fig, exp_dir, "exp1_beta_sweep_all_gms")

    labels = [paper_method_label(m) for m in methods]
    colors = [method_color(m) for m in methods]

    plot_conditional_metrics_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_not_top10",
        stem="exp1_cond_not_in_top10_metrics_all_gms",
        suptitle="Chosen move not in Stockfish top-10",
    )
    plot_conditional_metrics_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_in_top10",
        stem="exp1_cond_in_top10_metrics_all_gms",
        suptitle="Chosen move in Stockfish top-10",
    )
    plot_conditional_recall_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_not_top10",
        stem="exp1_cond_not_in_top10_recall_all_gms",
        suptitle="Recall when chosen move is not in Stockfish top-10",
    )
    plot_conditional_recall_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_in_top10",
        stem="exp1_cond_in_top10_recall_all_gms",
        suptitle="Recall when chosen move is in Stockfish top-10",
    )

    plot_prob_vs_cp_gap_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp1_prob_chosen_vs_pred_cp_gap_scatter_all_gms",
        title="Mean π(chosen) vs predicted CP gap",
    )

    plot_prob_over_entropy_vs_engine_likeness_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp1_prob_over_entropy_vs_engine_likeness_scatter_all_gms",
        title="Mean π(chosen) / entropy vs Distance from engine-best move (CP)",
    )

    plot_entropy_ratio_vs_engine_likeness_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp1_entropy_ratio_vs_engine_likeness_scatter_all_gms",
        title="Entropy retention vs Distance from engine-best move (CP)",
    )

    plot_prob_ratio_vs_engine_likeness_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp1_prob_ratio_vs_engine_likeness_scatter_all_gms",
        title="Chosen-move probability gain vs Distance from engine-best move (CP)",
    )
    
    print("[OK] Experiment 1 complete:", exp_dir)


# ============================================================
# Experiment 2: NLL + vanilla DPO across 9 GMs
# ============================================================

def run_experiment2(eval_root: Path, out_dir: Path) -> None:
    exp_dir = out_dir / "experiment2"
    ensure_dir(exp_dir)

    methods = [
        "maia2",
        "sft",
        "sft_pairwise",
        "dpo_beta=0.60",
        "sft_and_dpo_beta=0.60_dpo_loss_weight=0.10",
        "sft_and_dpo_beta=0.60_dpo_loss_weight=0.20",
        "sft_and_dpo_beta=0.60_dpo_loss_weight=0.40",
    ]
    ok, missing, bundles_by_gm = check_required_methods(
        eval_root,
        GM_ORDER,
        methods,
        require_row_metrics_for_methods=methods,
    )
    if not ok:
        print("\n[SKIP] Experiment 2 skipped. Missing files / bundles:")
        for line in missing:
            print(" -", line)
        return

    player_df = build_player_level_table(bundles_by_gm, methods, metric_names=ALL_TABLE_METRICS)
    agg_df = build_aggregate_table(player_df, metric_names=ALL_TABLE_METRICS)

    write_df_exports(player_df, exp_dir, "experiment2_player_level")
    write_df_exports(agg_df, exp_dir, "experiment2_aggregate")

    labels = [paper_method_label(m) for m in methods]
    colors = [method_color(m) for m in methods]
    agg_lookup = {row["method_key"]: row for _, row in agg_df.iterrows()}

    metrics = [
        ("mean_p_chosen", "P(chosen)", True),
        ("mean_logp_gap", "Log-probability gap", False),
        ("mrr", "MRR", True),
        ("top1_acc", "Top-1 recall", True),
        ("mean_kl", "KL vs Maia-2", False),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(11.0, 2.55), constrained_layout=True)
    for ax, (metric, title, percent) in zip(axes, metrics):
        means = [agg_lookup[m][f"{metric}_mean"] for m in methods]
        los = [agg_lookup[m][f"{metric}_ci_lo"] for m in methods]
        his = [agg_lookup[m][f"{metric}_ci_hi"] for m in methods]
        _plot_bar_with_ci(ax, labels, means, los, his, colors, title=title, percent=percent)

    finish_figure(fig, exp_dir, "exp2_nll_plus_dpo_all_gms")

    plot_conditional_metrics_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_not_top10",
        stem="exp2_cond_not_in_top10_metrics_all_gms",
        suptitle="Chosen move not in Stockfish top-10",
    )
    plot_conditional_metrics_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_in_top10",
        stem="exp2_cond_in_top10_metrics_all_gms",
        suptitle="Chosen move in Stockfish top-10",
    )
    plot_conditional_recall_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_not_top10",
        stem="exp2_cond_not_in_top10_recall_all_gms",
        suptitle="Recall when chosen move is not in Stockfish top-10",
    )
    plot_conditional_recall_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_in_top10",
        stem="exp2_cond_in_top10_recall_all_gms",
        suptitle="Recall when chosen move is in Stockfish top-10",
    )
    plot_prob_vs_cp_gap_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp2_prob_chosen_vs_pred_cp_gap_scatter_all_gms",
        title="Mean π(chosen) vs predicted CP gap",
    )
    plot_prob_over_entropy_vs_engine_likeness_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp2_prob_over_entropy_vs_engine_likeness_scatter_all_gms",
        title="Mean π(chosen) / entropy vs Distance from engine-best move (CP)",
    )

    plot_entropy_ratio_vs_engine_likeness_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp2_entropy_ratio_vs_engine_likeness_scatter_all_gms",
        title="Entropy retention vs Distance from engine-best move (CP)",
    )

    plot_prob_ratio_vs_engine_likeness_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp2_prob_ratio_vs_engine_likeness_scatter_all_gms",
        title="Chosen-move probability gain vs Distance from engine-best move (CP)",
    )
    print("[OK] Experiment 2 complete:", exp_dir)


# ============================================================
# Experiment 3: style-reweighted variants across 9 GMs
# ============================================================

def choose_best_style_method_keys(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    variant: str,
) -> List[str]:
    gm_candidates: Dict[str, List[str]] = {}

    for gm, gm_bundles in bundles_by_gm.items():
        keys = list(gm_bundles.keys())
        if variant == "v1":
            cand = [k for k in keys if "style_sim_utility_weight" in k]
        elif variant == "v2":
            cand = [k for k in keys if "style_v2" in k]
        else:
            cand = []
        gm_candidates[gm] = sorted(cand)

    common = None
    for gm in GM_ORDER:
        s = set(gm_candidates.get(gm, []))
        common = s if common is None else common.intersection(s)

    if not common:
        return []

    return sorted(common)


def best_method_by_metric(
    bundles_by_gm: Dict[str, Dict[str, MethodBundle]],
    candidate_methods: Sequence[str],
    metric: str = "mean_logp_gap",
) -> Optional[str]:
    rows = []
    for gm in GM_ORDER:
        gm_bundles = bundles_by_gm.get(gm, {})
        for method in candidate_methods:
            if method not in gm_bundles:
                continue
            bundle = gm_bundles[method]
            bundle.load()
            d = derive_metrics(bundle, [metric])
            rows.append({"gm": gm, "method": method, metric: d.get(metric, float("nan"))})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    means = df.groupby("method")[metric].mean()
    if means.empty:
        return None
    return str(means.idxmax())


def run_experiment3(eval_root: Path, out_dir: Path) -> Optional[Tuple[str, str]]:
    exp_dir = out_dir / "experiment3"
    ensure_dir(exp_dir)

    all_bundles_by_gm: Dict[str, Dict[str, MethodBundle]] = {}
    for gm in GM_ORDER:
        all_bundles_by_gm[gm] = discover_method_bundles(eval_root / gm)

    v1_candidates = choose_best_style_method_keys(all_bundles_by_gm, "v1")
    v2_candidates = choose_best_style_method_keys(all_bundles_by_gm, "v2")

    if not v1_candidates or not v2_candidates:
        print("\n[SKIP] Experiment 3 skipped. Could not find common style-v1 / style-v2 method keys across all 9 GMs.")
        if not v1_candidates:
            print(" - no common v1 keys containing 'style_sim_utility_weight'")
        if not v2_candidates:
            print(" - no common v2 keys containing 'style_v2'")
        return None

    best_v1 = best_method_by_metric(all_bundles_by_gm, v1_candidates, metric="mean_logp_gap")
    best_v2 = best_method_by_metric(all_bundles_by_gm, v2_candidates, metric="mean_logp_gap")

    if best_v1 is None or best_v2 is None:
        print("\n[SKIP] Experiment 3 skipped. Failed to select best v1 / v2 methods.")
        return None

    methods = [
        "maia2",
        "sft",
        "dpo_beta=0.60",
        best_v1,
        best_v2,
    ]

    ok, missing, bundles_by_gm = check_required_methods(
        eval_root,
        GM_ORDER,
        methods,
        require_opening_probe_for_methods=[best_v1, best_v2, "maia2", "sft", "dpo_beta=0.60"],
        require_row_metrics_for_methods=methods,
    )
    if not ok:
        print("\n[SKIP] Experiment 3 skipped. Missing files / bundles:")
        for line in missing:
            print(" -", line)
        return None

    player_df = build_player_level_table(bundles_by_gm, methods, metric_names=ALL_TABLE_METRICS)
    agg_df = build_aggregate_table(player_df, metric_names=ALL_TABLE_METRICS)

    write_df_exports(player_df, exp_dir, "experiment3_player_level")
    write_df_exports(agg_df, exp_dir, "experiment3_aggregate")

    labels = [
        "Maia-2",
        "NLL",
        "DPO (β=0.6)",
        "NLL + reweighted v1",
        "NLL + reweighted v2",
    ]
    colors = [
        method_color("maia2"),
        method_color("sft"),
        method_color("dpo_beta=0.60"),
        STYLE_V1_COLOR,
        STYLE_V2_COLOR,
    ]
    agg_lookup = {row["method_key"]: row for _, row in agg_df.iterrows()}

    metrics = [
        ("mean_p_chosen", "P(chosen)", True),
        ("mrr", "MRR", True),
        ("mean_logp_gap", "Log-probability gap", False),
        #("mean_gap_improvement", "Gap improvement", False),
        ("top1_acc", "Top-1 recall", True),
        ("mean_kl", "KL vs Maia-2", False),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(11.0, 2.55), constrained_layout=True)
    for ax, (metric, title, percent) in zip(axes, metrics):
        means = [agg_lookup[m][f"{metric}_mean"] for m in methods]
        los = [agg_lookup[m][f"{metric}_ci_lo"] for m in methods]
        his = [agg_lookup[m][f"{metric}_ci_hi"] for m in methods]
        _plot_bar_with_ci(ax, labels, means, los, his, colors, title=title, percent=percent)

    finish_figure(fig, exp_dir, "exp3_style_reweighted_all_gms")

    # NEW: extra Exp3 figures from per-row metrics
    plot_experiment3_recall_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
    )
    plot_piece_type_fidelity(exp_dir, bundles_by_gm, methods, labels, colors)
    plot_surface_style_agreement(exp_dir, bundles_by_gm, methods, labels, colors)
    plot_deep_style_agreement(exp_dir, bundles_by_gm, methods, labels, colors)

    plot_conditional_metrics_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_not_top10",
        stem="exp3_cond_not_in_top10_metrics_all_gms",
        suptitle="Chosen move not in Stockfish top-10",
    )
    plot_conditional_metrics_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_in_top10",
        stem="exp3_cond_in_top10_metrics_all_gms",
        suptitle="Chosen move in Stockfish top-10",
    )
    plot_conditional_recall_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_not_top10",
        stem="exp3_cond_not_in_top10_recall_all_gms",
        suptitle="Recall when chosen move is not in Stockfish top-10",
    )
    plot_conditional_recall_figure(
        exp_dir=exp_dir,
        agg_df=agg_df,
        methods=methods,
        labels=labels,
        colors=colors,
        cond_suffix="cond_in_top10",
        stem="exp3_cond_in_top10_recall_all_gms",
        suptitle="Recall when chosen move is in Stockfish top-10",
    )
    plot_prob_vs_cp_gap_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp3_prob_chosen_vs_pred_cp_gap_scatter_all_gms",
        title="Mean π(chosen) vs predicted CP gap",
    )
    plot_prob_over_entropy_vs_engine_likeness_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp3_prob_over_entropy_vs_engine_likeness_scatter_all_gms",
        title="Mean π(chosen) / entropy vs Distance from engine-best move (CP)",
    )
    plot_entropy_ratio_vs_engine_likeness_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp3_entropy_ratio_vs_engine_likeness_scatter_all_gms",
        title="Entropy retention vs Distance from engine-best move (CP)",
    )

    plot_prob_ratio_vs_engine_likeness_scatter_from_per_row(
        exp_dir=exp_dir,
        bundles_by_gm=bundles_by_gm,
        methods=methods,
        labels=labels,
        colors=colors,
        stem="exp3_prob_ratio_vs_engine_likeness_scatter_all_gms",
        title="Chosen-move probability gain vs Distance from engine-best move (CP)",
    )
    print("[OK] Experiment 3 complete:", exp_dir)
    print(f"      selected best_v1 = {best_v1}")
    print(f"      selected best_v2 = {best_v2}")
    return best_v1, best_v2


# ============================================================
# Opening 3x3 grid
# ============================================================

def run_opening_grid(
    eval_root: Path,
    out_dir: Path,
    best_v1: Optional[str],
    best_v2: Optional[str],
) -> None:
    grid_dir = out_dir / "opening_grid"
    ensure_dir(grid_dir)

    if best_v1 is None or best_v2 is None:
        print("\n[SKIP] Opening grid skipped because Experiment 3 did not produce best_v1 / best_v2.")
        return

    methods = ["maia2", "sft", "dpo_beta=0.60", best_v1, best_v2]
    ok, missing, bundles_by_gm = check_required_methods(
        eval_root,
        GM_ORDER,
        methods,
        require_opening_probe_for_methods=methods,
    )
    if not ok:
        print("\n[SKIP] Opening grid skipped. Missing files / bundles:")
        for line in missing:
            print(" -", line)
        return

    fig, axes = plt.subplots(3, 3, figsize=(7.1, 5.8), constrained_layout=True)
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for idx, gm in enumerate(GM_ORDER):
        ax = axes[idx]
        gm_bundles = bundles_by_gm[gm]
        for m in methods:
            gm_bundles[m].load()

        empirical = get_empirical_player_opening_distribution(gm_bundles["maia2"])
        if empirical is None:
            ax.set_visible(False)
            continue

        x = np.arange(len(OPENING_MOVE_ORDER))

        ax.plot(
            x,
            [empirical.get(k, 0.0) for k in OPENING_MOVE_ORDER],
            linestyle="--",
            marker="o",
            linewidth=1.1,
            markersize=2.5,
            color="black",
            label="Player",
        )

        plotted_labels = {
            "maia2": "Maia-2",
            "sft": "NLL",
            "dpo_beta=0.60": "DPO",
            best_v1: "RW-v1",
            best_v2: "RW-v2",
        }

        for m in methods:
            dist = get_opening_distribution(gm_bundles[m])
            if dist is None:
                continue
            ax.plot(
                x,
                [dist.get(k, 0.0) for k in OPENING_MOVE_ORDER],
                linewidth=1.0,
                markersize=0,
                color=gm_bundles[m].color if m not in {best_v1, best_v2} else (STYLE_V1_COLOR if m == best_v1 else STYLE_V2_COLOR),
                label=plotted_labels[m],
            )

        ax.set_title(opening_panel_title(gm), pad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["e4", "d4", "c4", "Nf3", "g3", "b3", "f4", "b4", "a4"], rotation=45, ha="right")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.tick_params(axis="both", which="major", pad=1)

        if idx % 3 == 0:
            ax.set_ylabel("Prob.")
        if idx >= 6:
            ax.set_xlabel("White first move")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles and legend_labels:
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(legend_handles, legend_labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
        fig.legend(
            uniq_h,
            uniq_l,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=6,
            frameon=False,
            columnspacing=0.9,
            handletextpad=0.4,
        )

    finish_figure(fig, grid_dir, "opening_first_move_grid_all_gms")
    print("[OK] Opening grid complete:", grid_dir)


# ============================================================
# Manifest
# ============================================================

def write_manifest(out_dir: Path, best_v1: Optional[str], best_v2: Optional[str]) -> None:
    manifest = {
        "gm_order": GM_ORDER,
        "best_style_v1_method": best_v1,
        "best_style_v2_method": best_v2,
    }
    with (out_dir / "paper_figure_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate paper-ready aggregate figures across 9 GMs. "
            "Experiments are run in order: Exp1, Exp2, Exp3, then opening grid."
        )
    )
    p.add_argument(
        "--eval_root",
        required=True,
        help="Root eval directory containing GM subfolders.",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for paper figures and CSV tables.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("Running paper figure generation across 9 GMs:")
    print("  " + ", ".join(GM_ORDER))
    print()

    run_experiment1(eval_root, out_dir)
    run_experiment2(eval_root, out_dir)
    style_keys = run_experiment3(eval_root, out_dir)

    best_v1 = None
    best_v2 = None
    if style_keys is not None:
        best_v1, best_v2 = style_keys

    run_opening_grid(eval_root, out_dir, best_v1, best_v2)
    write_manifest(out_dir, best_v1, best_v2)

    print("\nDone. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()

