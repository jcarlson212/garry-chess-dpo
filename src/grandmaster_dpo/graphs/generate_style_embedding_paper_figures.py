from __future__ import annotations

import argparse
import hashlib
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

# Names (or surname roots) that we annotate when present in test split.
FAMOUS_PLAYER_HINTS = [
    "kasparov",
    "karpov",
    "kramnik",
    "anand",
    "carlsen",
    "caruana",
    "nepomniachtchi",
    "ding",
    "topalov",
    "giri",
    "nakamura",
    "fischer",
    "botvinnik",
    "capablanca",
    "alekhine",
    "petrosian",
    "spassky",
    "tal",
    "lasker",
    "morphy",
]

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
    "positive_recall_at_5": {
        "title": "Positive Recall@5",
        "keys": ["positive_recall_at_5", "positive_recall@5"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "recall_at_10": {
        "title": "Recall@10",
        "keys": ["recall_at_10", "recall@10"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "recall_at_20": {
        "title": "Recall@20",
        "keys": ["recall_at_20", "recall@20"],
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
            "row_aggregated__cosine__pair_acc_mean_vs_hardest",
        ],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "row_cos_hard_gap": {
        "title": "Row cosine hard gap",
        "keys": [
            "row_cos_hard_gap",
            "row_cosine_hard_gap",
            "row_aggregated__cosine__hard_gap",
            "cosine__hard_gap",
        ],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "row_cos_mean_gap": {
        "title": "Row cosine mean gap",
        "keys": [
            "row_cos_mean_gap",
            "row_cosine_mean_gap",
            "row_aggregated__cosine__mean_gap",
            "cosine__mean_gap",
        ],
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
    "classification_roc_auc": {
        "title": "ROC AUC",
        "keys": ["roc_auc", "classification_roc_auc"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "classification_ap": {
        "title": "Average precision",
        "keys": ["average_precision", "ap", "classification_ap"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "classification_best_f1": {
        "title": "Best F1",
        "keys": ["best_f1", "classification_best_f1"],
        "is_percent": False,
        "bigger_is_better": True,
    },
    "classification_best_threshold": {
        "title": "Best threshold",
        "keys": ["best_threshold", "classification_best_threshold"],
        "is_percent": False,
        "bigger_is_better": False,
    },
    "classification_positive_pair_rate": {
        "title": "Positive-pair rate",
        "keys": ["positive_pair_rate", "classification_positive_pair_rate"],
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


def try_int(x: Any) -> Optional[int]:
    try:
        v = int(x)
        return v
    except Exception:
        return None


def _as_str_array(x: np.ndarray) -> np.ndarray:
    out: List[str] = []
    for v in x:
        if v is None:
            out.append("")
            continue
        s = str(v)
        out.append(s)
    return np.asarray(out, dtype=object)


def _canonical_phase_label(x: Any) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip()
    if not s:
        return "unknown"
    si = try_int(s)
    if si is not None:
        # Phase ids in training/eval caches use PHASE_TO_ID; keep unknown fallback.
        id_to_name = {0: "unknown", 1: "opening", 2: "middlegame", 3: "endgame"}
        return id_to_name.get(si, f"phase_{si}")
    return s


def _game_bucket_from_example_id(example_id: Any) -> str:
    s = str(example_id).strip() if example_id is not None else ""
    if not s:
        return "unknown_game"
    for sep in ("::", "|"):
        if sep in s:
            return s.split(sep, 1)[0] or "unknown_game"
    # Best-effort strip of trailing move/ply token.
    s2 = re.sub(r"([:_#/\-](?:ply|move)?\d+)$", "", s, flags=re.IGNORECASE)
    return s2 if s2 else s


def _stable_run_seed(base_seed: int, run_name: str) -> int:
    digest = hashlib.md5(run_name.encode("utf-8")).hexdigest()[:8]
    bump = int(digest, 16)
    return int((int(base_seed) + bump) % (2**31 - 1))

def normalize_metric_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def metric_aliases(metric_key: str) -> List[str]:
    spec = PRIMARY_METRICS.get(metric_key) or CONDITIONAL_METRICS.get(metric_key)
    aliases = [metric_key]
    if spec is not None:
        aliases.extend(spec.get("keys", []))

    out = set()
    for alias in aliases:
        norm = normalize_metric_name(alias)
        if not norm:
            continue
        out.add(norm)
        out.add(norm.replace("recall_at_", "recall_"))
        out.add(norm.replace("top1_accuracy", "top1_acc"))
        out.add(norm.replace("top5_recall", "recall_at_5"))
        out.add(norm.replace("mean_logp_", "logp_"))
    return sorted(out)


def collect_numeric_paths(obj: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}

    def _join(a: str, b: str) -> str:
        return b if not a else f"{a}__{b}"

    if isinstance(obj, dict):
        for k, v in obj.items():
            nk = normalize_metric_name(k)
            new_prefix = _join(prefix, nk) if nk else prefix

            if isinstance(v, dict):
                for subk in ("mean", "value", "avg", "estimate"):
                    if subk in v and np.isfinite(try_float(v[subk])):
                        out[new_prefix] = float(v[subk])
                        out[_join(new_prefix, subk)] = float(v[subk])
                out.update(collect_numeric_paths(v, new_prefix))
            elif isinstance(v, list):
                out.update(collect_numeric_paths(v, new_prefix))
            else:
                fv = try_float(v)
                if np.isfinite(fv) and new_prefix:
                    out[new_prefix] = fv
    elif isinstance(obj, list):
        for item in obj:
            out.update(collect_numeric_paths(item, prefix))

    return out


def extract_metric_map_from_obj(obj: Any) -> Dict[str, float]:
    raw = collect_numeric_paths(obj)
    out: Dict[str, float] = {}
    for path_key, value in raw.items():
        if not np.isfinite(value):
            continue
        parts = [p for p in path_key.split("__") if p]
        if not parts:
            continue
        out.setdefault(path_key, value)
        out.setdefault(path_key.replace("__", "_"), value)
        out.setdefault(parts[-1], value)
        if parts[-1] in {"mean", "value", "avg", "estimate"} and len(parts) >= 2:
            out.setdefault(parts[-2], value)
            out.setdefault("__".join(parts[:-1]), value)
            out.setdefault("_".join(parts[:-1]), value)
    return out


def extract_metric_map_from_dataframe(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df is None or df.empty:
        return out

    norm_cols = {normalize_metric_name(c): c for c in df.columns}

    # Case 1: one-row wide summary table where metric names are columns.
    for c in df.columns:
        series = pd.to_numeric(df[c], errors="coerce").dropna()
        if not series.empty:
            out.setdefault(normalize_metric_name(c), float(series.iloc[0]))

    # Case 2: long-form table with metric/key/name and mean/value/avg columns.
    name_col = next((norm_cols[k] for k in ["metric", "key", "name", "metric_name", "stat"] if k in norm_cols), None)
    value_col = next((norm_cols[k] for k in ["mean", "value", "avg", "estimate", "metric_value"] if k in norm_cols), None)
    if name_col is not None and value_col is not None:
        for _, row in df.iterrows():
            name = normalize_metric_name(row[name_col])
            value = try_float(row[value_col])
            if name and np.isfinite(value):
                out[name] = float(value)

    # Case 3: embedded json-ish bootstrap_ci_row column.
    if "bootstrap_ci_row" in norm_cols:
        col = norm_cols["bootstrap_ci_row"]
        for raw in df[col].dropna():
            if isinstance(raw, str):
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                out.update(extract_metric_map_from_obj({"bootstrap_ci_row": obj}))
            elif isinstance(raw, dict):
                out.update(extract_metric_map_from_obj({"bootstrap_ci_row": raw}))

    return out


def load_eval_artifact(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            obj = load_json(path)
            return obj, extract_metric_map_from_obj(obj)
        if suffix == ".jsonl":
            rows = load_jsonl(path)
            obj = {"rows": rows}
            metrics = extract_metric_map_from_obj(obj)
            metrics.update(extract_metric_map_from_dataframe(pd.DataFrame(rows)))
            return obj, metrics
        if suffix in {".csv", ".tsv"}:
            df = pd.read_csv(path, sep="	" if suffix == ".tsv" else ",")
            return None, extract_metric_map_from_dataframe(df)
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
            return None, extract_metric_map_from_dataframe(df)
    except Exception:
        return None, {}
    return None, {}


def available_metric_count(run: "RunRecord") -> int:
    keys = ["mrr", "recall_at_1", "mean_logp_gap", "row_cos_hard_gap", "mean_kl"]
    return sum(int(np.isfinite(run.metric(k))) for k in keys)


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


def run_identity_key(
    stage: str,
    pair_version: str,
    phi: str,
    tau: Optional[str],
    seed: Optional[int],
) -> Tuple[str, str, str, Optional[str], Optional[int]]:
    return (stage, pair_version, phi, canonicalize_tau_str(tau), seed)


def run_identity_key_from_name(name: str) -> Tuple[str, str, str, Optional[str], Optional[int]]:
    stage, pair, phi, tau, _tau_val, seed = parse_run_name(name)
    return run_identity_key(stage, pair, phi, tau, seed)


def run_identity_key_from_record(run: "RunRecord") -> Tuple[str, str, str, Optional[str], Optional[int]]:
    return run_identity_key(run.stage, run.pair_version, run.phi, run.tau, run.seed)


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
    eval_metrics: Optional[Dict[str, float]] = None
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
        aliases = metric_aliases(metric_key)

        if self.eval_metrics:
            for alias in aliases:
                if alias in self.eval_metrics and np.isfinite(self.eval_metrics[alias]):
                    return float(self.eval_metrics[alias])
            for key, value in self.eval_metrics.items():
                if not np.isfinite(value):
                    continue
                parts = [p for p in str(key).split("__") if p]
                for alias in aliases:
                    if key == alias or key.endswith(f"__{alias}") or alias in parts:
                        return float(value)

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

def find_best_matching_file(
    run_dir: Path,
    split: str,
    patterns: Sequence[str],
    exclude_patterns: Optional[Sequence[str]] = None,
    preferred_suffixes: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """
    Search recursively and score candidates rather than taking the first lexicographic hit.
    This matters because eval folders often contain nested outputs and multiple files whose
    names all contain "metrics".
    """
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    split = split.lower()
    exclude_patterns = list(exclude_patterns or [])
    preferred_suffixes = [s.lower() for s in (preferred_suffixes or [".json", ".jsonl", ".parquet", ".pq", ".csv", ".tsv"])]

    scored: List[Tuple[int, Path]] = []
    for p in files:
        name = p.name.lower()
        full = str(p).lower()
        if not any(re.search(pattern, name) or re.search(pattern, full) for pattern in patterns):
            continue
        if any(re.search(pattern, name) or re.search(pattern, full) for pattern in exclude_patterns):
            continue

        score = 0
        if split and (split in name or f"/{split}/" in full or f"_{split}" in name):
            score += 100
        suffix = p.suffix.lower()
        if suffix in preferred_suffixes:
            score += 40 + max(0, 10 - preferred_suffixes.index(suffix))
        if "bootstrap" in name:
            score += 40
        if "summary" in name:
            score += 35
        if "sampled_embedding_summary" in name:
            score -= 120
        if "retrieval_metrics" in name:
            score += 110
        if "pair_metrics" in name:
            score += 105
        if "spread_metrics" in name:
            score += 95
        if "classification_metrics" in name:
            score += 90
        if "aggregate" in name or "overall" in name:
            score += 25
        if "eval_results" in name:
            score += 25
        if "metrics" in name:
            score += 15
        if "per_row" in name or "row_metrics" in name or "cache" in full:
            score -= 120
        score -= len(p.relative_to(run_dir).parts)
        scored.append((score, p))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return scored[0][1]


def merge_metric_maps(target: Dict[str, float], new_metrics: Optional[Dict[str, float]]) -> None:
    if not new_metrics:
        return
    for k, v in new_metrics.items():
        if np.isfinite(try_float(v)):
            target[k] = float(v)


def load_summary_referenced_artifacts(summary_obj: Optional[Dict[str, Any]], summary_path: Path) -> Dict[str, float]:
    """
    Current eval layout often stores primary metrics in sibling files
    referenced by `summary.json` (e.g. retrieval_metrics_path).
    """
    out: Dict[str, float] = {}
    if not isinstance(summary_obj, dict):
        return out

    for k, v in summary_obj.items():
        if not isinstance(v, str):
            continue
        key_norm = normalize_metric_name(k)
        if not key_norm.endswith("_path"):
            continue
        p = Path(v)
        if not p.exists():
            p = summary_path.parent / Path(v).name
        if not p.exists() or not p.is_file():
            continue
        _obj, metrics = load_eval_artifact(p)
        merge_metric_maps(out, metrics)
    return out


def load_split_local_metric_files(run_dir: Path, split: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    split_dir = run_dir / split
    if not split_dir.exists() or not split_dir.is_dir():
        return out

    preferred_names = [
        "metrics_flat.json",
        "pair_score_components.json",
        "retrieval_curve.json",
        "retrieval_at_k.json",
        "threshold_sweep.json",
        "score_distributions.json",
        "pairwise_auc.json",
        "conditioned_metrics.json",
        "bootstrap_ci.json",
        "calibration.json",
        "top_confusions.json",
        "retrieval_metrics.json",
        "pair_metrics.json",
        "spread_metrics.json",
        "classification_metrics.json",
        "summary.json",
    ]
    for name in preferred_names:
        p = split_dir / name
        if not p.exists():
            continue
        obj, metrics = load_eval_artifact(p)
        merge_metric_maps(out, metrics)
        if name == "summary.json":
            merge_metric_maps(out, load_summary_referenced_artifacts(obj, p))

    for p in sorted(split_dir.glob("*.json")):
        lname = p.name.lower()
        if lname in set(preferred_names):
            continue
        if "sampled_embedding_summary" in lname:
            continue
        if "metrics" not in lname and lname != "summary.json":
            continue
        _obj, metrics = load_eval_artifact(p)
        merge_metric_maps(out, metrics)
    return out


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
                r"bootstrap",
                r"eval_results",
                r"summary",
                r"aggregate",
                r"overall",
                r"metrics",
            ],
            exclude_patterns=[
                r"per[_\-]?row",
                r"row[_\-]?metrics",
                r"/generated_eval_cache/",
                r"manifest",
            ],
            preferred_suffixes=[".json", ".jsonl", ".parquet", ".pq", ".csv", ".tsv"],
        )
        per_row = find_best_matching_file(
            run_dir,
            split=split,
            patterns=[
                r"per[_\-]?row",
                r"row[_\-]?metrics",
            ],
            preferred_suffixes=[".parquet", ".pq", ".json", ".jsonl", ".csv", ".tsv"],
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

        if eval_summary is not None:
            rec.raw_eval_json, rec.eval_metrics = load_eval_artifact(eval_summary)
        else:
            rec.eval_metrics = {}

        if eval_summary is not None and eval_summary.name.lower() == "summary.json":
            merge_metric_maps(rec.eval_metrics, load_summary_referenced_artifacts(rec.raw_eval_json, eval_summary))

        merge_metric_maps(rec.eval_metrics, load_split_local_metric_files(run_dir, split))

        if (not rec.eval_metrics) and per_row is not None:
            _obj, per_row_metrics = load_eval_artifact(per_row)
            if per_row_metrics:
                rec.eval_metrics = per_row_metrics

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
    by_identity = {run_identity_key_from_record(r): r for r in train_runs}
    for ev in eval_runs:
        tr = by_name.get(ev.name)
        if tr is None:
            tr = by_identity.get(run_identity_key_from_record(ev))
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
                0 if available_metric_count(r) > 0 else 1,
                -available_metric_count(r),
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
                0 if available_metric_count(r) > 0 else 1,
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
            0 if available_metric_count(r) > 0 else 1,
            0 if r.stage == "screen" else (1 if r.stage == "final" else (2 if r.stage == "super" else 3)),
            -available_metric_count(r),
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

def export_discovery_debug(runs: Sequence[RunRecord], out_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in runs:
        rows.append(
            {
                "name": r.name,
                "stage": r.stage,
                "pair_version": r.pair_version,
                "phi": r.phi,
                "tau": canonicalize_tau_str(r.tau),
                "eval_summary_path": str(r.eval_summary_path) if r.eval_summary_path else "",
                "per_row_metrics_path": str(r.per_row_metrics_path) if r.per_row_metrics_path else "",
                "available_metric_count": available_metric_count(r),
                "mrr": r.metric("mrr"),
                "recall_at_1": r.metric("recall_at_1"),
                "mean_logp_gap": r.metric("mean_logp_gap"),
                "row_cos_hard_gap": r.metric("row_cos_hard_gap"),
                "mean_kl": r.metric("mean_kl"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    df.to_json(out_path.with_suffix(".json"), orient="records", indent=2)
    return df


def export_metric_coverage(runs: Sequence[RunRecord], out_path: Path) -> pd.DataFrame:
    metrics = [
        "mrr",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mean_logp_gap",
        "mean_kl",
        "pair_acc_hardest",
        "row_cos_hard_gap",
        "row_cos_mean_gap",
        "spread_ratio",
        "classification_roc_auc",
        "classification_ap",
        "classification_best_f1",
        "classification_best_threshold",
    ]
    rows: List[Dict[str, Any]] = []
    n = max(1, len(runs))
    for m in metrics:
        vals = np.asarray([r.metric(m) for r in runs], dtype=float)
        finite = np.isfinite(vals)
        rows.append(
            {
                "metric_key": m,
                "metric_title": PRIMARY_METRICS.get(m, {}).get("title", m),
                "n_finite": int(finite.sum()),
                "coverage_frac": float(finite.mean()),
                "mean": float(np.nanmean(vals)) if finite.any() else float("nan"),
                "min": float(np.nanmin(vals)) if finite.any() else float("nan"),
                "max": float(np.nanmax(vals)) if finite.any() else float("nan"),
                "n_runs": n,
            }
        )
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
    ys = np.asarray([r.metric(metric_key) for r in runs], dtype=float)
    colors = bar_colors(runs)
    valid = np.isfinite(ys)
    if valid.any():
        ax.bar(xs[valid], ys[valid], color=[colors[i] for i in np.where(valid)[0]], width=0.72)
        annotate_top(ax, xs[valid], ys[valid], fmt="{:.4f}", k=min(2, int(valid.sum())))
    else:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes, fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r.method_label for r in runs], rotation=25, ha="right")
    ax.set_title(title, pad=4)


def line_tau(
    ax: plt.Axes,
    runs: Sequence[RunRecord],
    metric_key: str,
    title: str,
    ylabel: Optional[str] = None,
) -> None:
    xs = np.asarray([r.tau_value for r in runs], dtype=float)
    ys = np.asarray([r.metric(metric_key) for r in runs], dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    if len(xs) == 0:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes, fontsize=6.5)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Training τ")
        if ylabel:
            ax.set_ylabel(ylabel)
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


def ranked_dotplot(
    ax: plt.Axes,
    runs: Sequence[RunRecord],
    metric_key: str,
    title: str,
    ylabel: Optional[str] = None,
    annotate_top_k: int = 3,
) -> None:
    runs = [r for r in runs if np.isfinite(r.metric(metric_key))]
    runs = sorted(runs, key=lambda r: r.metric(metric_key), reverse=True)
    if not runs:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes, fontsize=6.5)
        ax.set_xticks([])
        ax.set_xlabel("Rank")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_title(title, pad=4)
        return
    xs = np.arange(1, len(runs) + 1)
    ys = [r.metric(metric_key) for r in runs]
    colors = bar_colors(runs)
    ax.scatter(xs, ys, c=colors, s=24, zorder=3)
    ax.plot(xs, ys, color="#777777", alpha=0.25, linewidth=0.9, zorder=2)
    ax.set_xlim(0.5, len(runs) + 0.5)
    tick_count = min(8, len(runs))
    xticks = np.unique(np.round(np.linspace(1, len(runs), tick_count)).astype(int))
    ax.set_xticks(xticks)
    ax.set_xlabel("Rank (1 = best)")
    if ylabel:
        ax.set_ylabel(ylabel)
    for i in range(min(annotate_top_k, len(runs))):
        ax.text(
            xs[i],
            ys[i],
            runs[i].method_label,
            fontsize=5.8,
            ha="left",
            va="bottom",
            clip_on=True,
        )
    ax.set_title(title, pad=4)


def metric_has_coverage(runs: Sequence[RunRecord], metric_key: str, min_count: int = 1) -> bool:
    vals = np.asarray([r.metric(metric_key) for r in runs], dtype=float)
    return int(np.isfinite(vals).sum()) >= int(min_count)


def metric_from_aliases(run: RunRecord, aliases: Sequence[str]) -> float:
    metrics = run.eval_metrics or {}
    norm_aliases = [normalize_metric_name(a) for a in aliases if a]
    if not norm_aliases:
        return float("nan")
    for alias in norm_aliases:
        if alias in metrics and np.isfinite(metrics[alias]):
            return float(metrics[alias])
    for key, value in metrics.items():
        if not np.isfinite(value):
            continue
        parts = [p for p in str(key).split("__") if p]
        norm_key = normalize_metric_name(key)
        for alias in norm_aliases:
            if norm_key == alias or norm_key.endswith(f"__{alias}") or alias in parts:
                return float(value)
    return float("nan")


def pick_main_result_metrics(finalists: Sequence[RunRecord]) -> List[Tuple[str, str]]:
    # Always show retrieval first.
    chosen: List[Tuple[str, str]] = [
        ("mrr", PRIMARY_METRICS["mrr"]["title"]),
        ("recall_at_1", PRIMARY_METRICS["recall_at_1"]["title"]),
    ]

    # Prefer original paper metrics when present, otherwise switch to available metrics.
    tail_candidates = [
        ("mean_logp_gap", PRIMARY_METRICS["mean_logp_gap"]["title"]),
        ("mean_kl", PRIMARY_METRICS["mean_kl"]["title"]),
        ("pair_acc_hardest", PRIMARY_METRICS["pair_acc_hardest"]["title"]),
        ("row_cos_hard_gap", PRIMARY_METRICS["row_cos_hard_gap"]["title"]),
        ("classification_roc_auc", PRIMARY_METRICS["classification_roc_auc"]["title"]),
        ("classification_ap", PRIMARY_METRICS["classification_ap"]["title"]),
    ]
    for metric_key, title in tail_candidates:
        if metric_key in {m for m, _ in chosen}:
            continue
        if metric_has_coverage(finalists, metric_key, min_count=1):
            chosen.append((metric_key, title))
        if len(chosen) >= 4:
            break
    return chosen[:4]


# ============================================================
# Figure families
# ============================================================

def plot_finalist_main_results(out_dir: Path, finalists: Sequence[RunRecord]) -> None:
    metrics = pick_main_result_metrics(finalists)

    fig, axes = plt.subplots(1, len(metrics), figsize=(10.6, 2.7), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
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

    metrics = [("mrr", "MRR"), ("recall_at_1", "Recall@1"), ("row_cos_hard_gap", "Hard gap"), ("pair_acc_hardest", "Pair acc hard")]

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
        ("recall_at_1", "Recall@1"),
        ("recall_at_5", "Recall@5"),
        ("row_cos_hard_gap", "Hard gap"),
    ]

    fig, axes = plt.subplots(len(panels), len(metric_pairs), figsize=(12.6, 9.2), constrained_layout=True)
    for i, (pair, phi) in enumerate(panels):
        sweep = runs_for_tau_sweep(list(eval_runs), pair, phi)
        for j, (metric_key, ylabel) in enumerate(metric_pairs):
            ax = axes[i, j]
            title = f"{pair}-{phi}"
            line_tau(ax, sweep, metric_key, title=title, ylabel=ylabel if j == 0 else None)
            if j > 0:
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
    eval_lookup_by_identity = {run_identity_key_from_record(r): r for r in eval_runs}
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
        if ev is None:
            ev = eval_lookup_by_identity.get(run_identity_key_from_record(r))
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
    eval_lookup_by_identity = {run_identity_key_from_record(r): r for r in eval_runs}
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
        if ev is None:
            ev = eval_lookup_by_identity.get(run_identity_key_from_record(r))
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


def plot_classification_summary(out_dir: Path, runs: Sequence[RunRecord]) -> None:
    metrics = [
        ("classification_roc_auc", PRIMARY_METRICS["classification_roc_auc"]["title"]),
        ("classification_ap", PRIMARY_METRICS["classification_ap"]["title"]),
        ("classification_best_f1", PRIMARY_METRICS["classification_best_f1"]["title"]),
        ("classification_best_threshold", PRIMARY_METRICS["classification_best_threshold"]["title"]),
    ]
    metrics = [(k, t) for (k, t) in metrics if metric_has_coverage(runs, k, min_count=1)]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.7 * len(metrics), 2.7), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric_key, title) in zip(axes, metrics):
        grouped_bar(ax, runs, metric_key, title)
        if metric_key in {"classification_roc_auc", "classification_ap"}:
            # Useful visual baseline: random classifier.
            baseline = 0.5 if metric_key == "classification_roc_auc" else 0.005
            ax.axhline(baseline, color="#999999", linestyle="--", linewidth=0.9)
            ax.text(0.02, 0.98, "random baseline", transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color="#666666")
    savefig(fig, out_dir, "fig_classification_summary")


def plot_pair_score_components(out_dir: Path, runs: Sequence[RunRecord]) -> None:
    """
    Explicitly compare positive, hardest-negative, and mean-negative cosine levels.
    This gives a more interpretable view than only reporting a single gap metric.
    """
    if not runs:
        return
    labels = [r.method_label for r in runs]
    xs = np.arange(len(runs))
    pos_mean = np.asarray(
        [
            metric_from_aliases(
                r,
                [
                    "row_aggregated__cosine__mean_positive__row_mean_pos_cos_mean",
                    "pos_mean_cos",
                ],
            )
            for r in runs
        ],
        dtype=float,
    )
    hardneg_mean = np.asarray(
        [
            metric_from_aliases(
                r,
                [
                    "row_aggregated__cosine__hardest_negative__row_hardest_neg_cos_mean",
                    "hardneg_mean_cos",
                ],
            )
            for r in runs
        ],
        dtype=float,
    )
    meanneg_mean = np.asarray(
        [
            metric_from_aliases(
                r,
                [
                    "row_aggregated__cosine__mean_negative__row_mean_neg_cos_mean",
                    "meanneg_mean_cos",
                ],
            )
            for r in runs
        ],
        dtype=float,
    )
    hard_gap = np.asarray([r.metric("row_cos_hard_gap") for r in runs], dtype=float)
    mean_gap = np.asarray([r.metric("row_cos_mean_gap") for r in runs], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.2), constrained_layout=True)
    ax0, ax1 = axes

    for series, name, color in [
        (pos_mean, "Positive mean cosine", "#1f77b4"),
        (hardneg_mean, "Hardest-neg mean cosine", "#d62728"),
        (meanneg_mean, "Mean-neg mean cosine", "#ff7f0e"),
    ]:
        valid = np.isfinite(series)
        if valid.any():
            ax0.plot(xs[valid], series[valid], marker="o", linewidth=1.2, label=name, color=color)
    ax0.set_xticks(xs)
    ax0.set_xticklabels(labels, rotation=20, ha="right")
    ax0.set_ylabel("Cosine similarity")
    ax0.set_title("Score Components (Pos vs Hard/Soft Neg)", pad=4)
    ax0.legend(loc="best", fontsize=5.8)

    width = 0.36
    valid_h = np.isfinite(hard_gap)
    valid_m = np.isfinite(mean_gap)
    if valid_h.any():
        ax1.bar(xs[valid_h] - width / 2, hard_gap[valid_h], width=width, label="Hard gap", color="#4C78A8")
    if valid_m.any():
        ax1.bar(xs[valid_m] + width / 2, mean_gap[valid_m], width=width, label="Mean gap", color="#F58518")
    ax1.axhline(0.0, color="#888888", linewidth=0.8)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Gap (pos - neg)")
    ax1.set_title("Gap Components", pad=4)
    ax1.legend(loc="best", fontsize=5.8)

    savefig(fig, out_dir, "fig_pair_score_components")


def extract_eval_tau_curve(run: RunRecord, value_key: str) -> Tuple[np.ndarray, np.ndarray]:
    metrics = run.eval_metrics or {}
    pts: List[Tuple[float, float]] = []
    pattern = re.compile(rf"row_aggregated__by_eval_tau__([0-9_]+)__{re.escape(value_key)}$")
    for k, v in metrics.items():
        m = pattern.match(k)
        if not m:
            continue
        tau = parse_tau_value(m.group(1))
        val = try_float(v)
        if np.isfinite(tau) and np.isfinite(val):
            pts.append((tau, float(val)))
    if not pts:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    pts = sorted(pts, key=lambda x: x[0])
    return np.asarray([p[0] for p in pts], dtype=float), np.asarray([p[1] for p in pts], dtype=float)


def plot_eval_tau_sensitivity(out_dir: Path, eval_runs: Sequence[RunRecord]) -> None:
    """
    x-axis: eval tau used in score scaling
    color: training tau of the model checkpoint
    """
    panels = [("v1", "phi0"), ("v1", "phi1"), ("v2", "phi0"), ("v2", "phi1"), ("v3", "phi0"), ("v3", "phi1")]
    keys_and_titles = [
        ("dot_over_tau__hard_gap", "Hard gap vs eval tau"),
        ("infonce_like_loss_mean_vs_hardest_neg", "InfoNCE-like loss vs eval tau"),
    ]

    fig, axes = plt.subplots(len(panels), 2, figsize=(9.0, 10.5), constrained_layout=True)
    for i, (pair, phi) in enumerate(panels):
        sweep = runs_for_tau_sweep(list(eval_runs), pair, phi)
        if not sweep:
            for j in range(2):
                ax = axes[i, j]
                ax.text(0.5, 0.5, "no runs", ha="center", va="center", transform=ax.transAxes, fontsize=6.2)
                ax.set_title(f"{pair}-{phi}")
                ax.set_xlabel("Eval tau")
            continue

        train_taus = sorted({r.tau_value for r in sweep if np.isfinite(r.tau_value)})
        color_map = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(train_taus))))
        color_by_tau = {t: color_map[k] for k, t in enumerate(train_taus)}

        for j, (value_key, title) in enumerate(keys_and_titles):
            ax = axes[i, j]
            plotted = False
            for r in sweep:
                xs, ys = extract_eval_tau_curve(r, value_key)
                if len(xs) == 0:
                    continue
                train_tau = r.tau_value
                color = color_by_tau.get(train_tau, "#666666")
                label = f"train τ={train_tau:g}" if np.isfinite(train_tau) else r.method_label
                ax.plot(xs, ys, marker="o", color=color, alpha=0.9, linewidth=1.1, label=label)
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "no eval-tau curve data", ha="center", va="center", transform=ax.transAxes, fontsize=6.2)
            ax.set_title(f"{pair}-{phi}: {title}", pad=4)
            ax.set_xlabel("Eval tau")
            if j == 0:
                ax.set_ylabel("Gap (higher better)")
            else:
                ax.set_ylabel("Loss (lower better)")
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # dedupe legend entries
                    dedup = dict(zip(labels, handles))
                    ax.legend(dedup.values(), dedup.keys(), title="Training tau", loc="best", fontsize=5.6, title_fontsize=5.8)

    savefig(fig, out_dir, "fig_eval_tau_sensitivity")


def export_super_run_comparison(out_dir: Path, eval_runs: Sequence[RunRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], Dict[str, List[RunRecord]]] = {}
    for r in eval_runs:
        key = (r.pair_version, r.phi)
        grouped.setdefault(key, {})
        grouped[key].setdefault(r.stage, []).append(r)

    for (pair, phi), stages in sorted(grouped.items(), key=lambda x: (PAIR_ORDER.get(x[0][0], 99), PHI_ORDER.get(x[0][1], 99))):
        final_candidates = stages.get("final", [])
        super_candidates = stages.get("super", [])
        final_best = sorted(final_candidates, key=lambda r: r.metric("mrr"), reverse=True)[0] if final_candidates else None
        super_best = sorted(super_candidates, key=lambda r: r.metric("mrr"), reverse=True)[0] if super_candidates else None
        rows.append(
            {
                "pair_version": pair,
                "phi": phi,
                "has_final": bool(final_best is not None),
                "has_super": bool(super_best is not None),
                "final_name": final_best.name if final_best else "",
                "super_name": super_best.name if super_best else "",
                "final_mrr": final_best.metric("mrr") if final_best else float("nan"),
                "super_mrr": super_best.metric("mrr") if super_best else float("nan"),
                "final_hard_gap": final_best.metric("row_cos_hard_gap") if final_best else float("nan"),
                "super_hard_gap": super_best.metric("row_cos_hard_gap") if super_best else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "super_vs_final.csv", index=False)
    return df


def plot_super_vs_final(out_dir: Path, eval_runs: Sequence[RunRecord]) -> None:
    df = export_super_run_comparison(out_dir, eval_runs)
    if df.empty:
        return
    if not (df["has_final"].any() or df["has_super"].any()):
        return
    labels = [f"{row.pair_version}-{row.phi}" for row in df.itertuples()]
    xs = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["mrr", "hard_gap"],
        ["Final vs Super: MRR", "Final vs Super: Hard gap"],
    ):
        fcol = "final_mrr" if metric == "mrr" else "final_hard_gap"
        scol = "super_mrr" if metric == "mrr" else "super_hard_gap"
        y_final = pd.to_numeric(df[fcol], errors="coerce").to_numpy(dtype=float)
        y_super = pd.to_numeric(df[scol], errors="coerce").to_numpy(dtype=float)
        ax.scatter(xs, y_final, marker="o", s=20, label="final")
        ax.scatter(xs, y_super, marker="^", s=24, label="super")
        for i in range(len(xs)):
            if np.isfinite(y_final[i]) and np.isfinite(y_super[i]):
                ax.plot([xs[i], xs[i]], [y_final[i], y_super[i]], color="#999999", linewidth=0.9, alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title, pad=4)
    axes[0].legend(loc="best", fontsize=5.9)
    savefig(fig, out_dir, "fig_super_vs_final")


def normalize_player_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def is_famous_player_name(name: str, hints: Sequence[str]) -> bool:
    norm = normalize_player_name(name)
    if not norm:
        return False
    return any(h in norm for h in hints)


def pca_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x_centered = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(x_centered, full_matrices=False)
    comp = x_centered @ vt[:2].T
    if comp.shape[1] == 1:
        comp = np.concatenate([comp, np.zeros((comp.shape[0], 1), dtype=comp.dtype)], axis=1)
    return comp.astype(np.float32, copy=False)


def _pairwise_l2(x: np.ndarray) -> np.ndarray:
    # x: [N, D]
    g = x @ x.T
    sq = np.clip(np.diag(g)[:, None] + np.diag(g)[None, :] - 2.0 * g, 0.0, None)
    return np.sqrt(sq).astype(np.float64, copy=False)


def _run_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def plot_famous_player_heatmaps_for_run(
    out_dir: Path,
    *,
    run: RunRecord,
    player_order: Sequence[str],
    centroid_mat: np.ndarray,
    tau: float,
) -> Optional[Path]:
    if len(player_order) < 2 or centroid_mat.shape[0] < 2:
        return None
    c = np.clip(centroid_mat @ centroid_mat.T, -1.0, 1.0)
    l2 = _pairwise_l2(centroid_mat)
    tau_eff = float(tau) if np.isfinite(tau) and float(tau) > 1e-9 else 1.0
    exp_term = np.exp(-c / tau_eff)

    mats = [
        (c, "Cosine(centroid_i, centroid_j)"),
        (l2, "L2(centroid_i, centroid_j)"),
        (exp_term, f"exp(-dot/tau), tau={tau_eff:g}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), constrained_layout=True)
    for ax, (mat, title) in zip(axes, mats):
        im = ax.imshow(mat, cmap="viridis", interpolation="nearest")
        ax.set_title(title, pad=4)
        ax.set_xticks(np.arange(len(player_order)))
        ax.set_yticks(np.arange(len(player_order)))
        ax.set_xticklabels(player_order, rotation=45, ha="right")
        ax.set_yticklabels(player_order)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    stem = f"fig_famous_player_heatmaps_{_run_slug(run.name)}"
    savefig(fig, out_dir, stem)
    return out_dir / f"{stem}.pdf"


def _weighted_sample_without_replacement(
    items: Sequence[str],
    n: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> List[str]:
    if n <= 0 or not items:
        return []
    n = min(n, len(items))
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != len(items) or not np.isfinite(w).any() or float(w.sum()) <= 0.0:
        idx = rng.choice(len(items), size=n, replace=False)
        return [items[int(i)] for i in idx]
    p = w / w.sum()
    idx = rng.choice(len(items), size=n, replace=False, p=p)
    return [items[int(i)] for i in idx]


def _stratified_pick_indices(
    indices: np.ndarray,
    phase_labels: np.ndarray,
    game_buckets: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for ix in indices:
        key = (_canonical_phase_label(phase_labels[ix]), str(game_buckets[ix]))
        buckets.setdefault(key, []).append(int(ix))
    for k in buckets:
        rng.shuffle(buckets[k])
    keys = sorted(buckets.keys())
    selected: List[int] = []
    live = True
    while live and len(selected) < cap:
        live = False
        for k in keys:
            arr = buckets[k]
            if not arr:
                continue
            selected.append(arr.pop())
            live = True
            if len(selected) >= cap:
                break
    return np.asarray(selected, dtype=np.int64)


def _load_test_embeddings_with_meta(run: RunRecord) -> Optional[Dict[str, np.ndarray]]:
    p = run.path / "test" / "embeddings_and_meta.npz"
    if not p.exists() or not p.is_file():
        return None
    try:
        with np.load(p, allow_pickle=True) as data:
            emb = np.asarray(data["embeddings"], dtype=np.float32)
            pid = _as_str_array(np.asarray(data["player_id"], dtype=object))
            phase = _as_str_array(np.asarray(data["phase_id"], dtype=object))
            example = _as_str_array(np.asarray(data["example_id"], dtype=object))
        if emb.ndim != 2 or emb.shape[0] == 0:
            return None
        if not (len(pid) == len(phase) == len(example) == emb.shape[0]):
            return None
        return {"embeddings": emb, "player_id": pid, "phase_id": phase, "example_id": example}
    except Exception:
        return None


def plot_test_player_pca(
    out_dir: Path,
    tables_dir: Path,
    runs: Sequence[RunRecord],
    *,
    num_players: int = 100,
    per_player_cap: int = 16,
    base_seed: int = 42,
    famous_hints: Sequence[str] = FAMOUS_PLAYER_HINTS,
) -> None:
    if not runs:
        return

    runs = sorted(list(runs), key=run_sort_key)
    ncols = min(3, max(1, len(runs)))
    nrows = int(math.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)

    sampled_rows: List[Dict[str, Any]] = []
    famous_rows: List[Dict[str, Any]] = []
    heatmap_rows: List[Dict[str, Any]] = []

    for i, run in enumerate(runs):
        ax = axes_arr[i]
        loaded = _load_test_embeddings_with_meta(run)
        if loaded is None:
            ax.text(0.5, 0.5, "missing test/embeddings_and_meta.npz\n(rerun eval with --save-embeddings)", ha="center", va="center", fontsize=6.3, transform=ax.transAxes)
            ax.set_title(run.method_label, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        emb = loaded["embeddings"]
        pid = loaded["player_id"]
        phase = loaded["phase_id"]
        example = loaded["example_id"]
        game_bucket = np.asarray([_game_bucket_from_example_id(x) for x in example], dtype=object)

        valid = np.asarray([bool(str(x).strip()) for x in pid], dtype=bool)
        if not valid.any():
            ax.text(0.5, 0.5, "no labeled players in test embeddings", ha="center", va="center", fontsize=6.3, transform=ax.transAxes)
            ax.set_title(run.method_label, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        emb = emb[valid]
        pid = pid[valid]
        phase = phase[valid]
        game_bucket = game_bucket[valid]

        uniq, inv, counts = np.unique(pid, return_inverse=True, return_counts=True)
        player_counts = {str(uniq[k]): int(counts[k]) for k in range(len(uniq))}
        famous_present = sorted(
            [p for p in player_counts if is_famous_player_name(p, famous_hints)],
            key=lambda p: player_counts[p],
            reverse=True,
        )
        famous_rows.extend(
            [{"run": run.name, "player_id": p, "n_examples": player_counts[p]} for p in famous_present]
        )

        rng = np.random.default_rng(_stable_run_seed(base_seed, run.name))
        force_n = min(len(famous_present), max(8, int(num_players * 0.2)))
        selected_players: List[str] = list(famous_present[:force_n])
        remainder = [p for p in player_counts.keys() if p not in set(selected_players)]
        weights = np.asarray([math.sqrt(max(1, player_counts[p])) for p in remainder], dtype=float)
        need = max(0, num_players - len(selected_players))
        selected_players.extend(_weighted_sample_without_replacement(remainder, need, weights, rng))

        centroids: List[np.ndarray] = []
        labels: List[str] = []
        n_used: List[int] = []
        for p in selected_players:
            idx = np.where(pid == p)[0]
            if len(idx) == 0:
                continue
            picked = _stratified_pick_indices(idx, phase, game_bucket, cap=per_player_cap, rng=rng)
            z = emb[picked]
            centroid = z.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(centroid, axis=1, keepdims=True)
            centroid = centroid / np.clip(norm, 1e-12, None)
            centroids.append(centroid[0])
            labels.append(p)
            n_used.append(int(len(picked)))
            sampled_rows.append(
                {
                    "run": run.name,
                    "pair_version": run.pair_version,
                    "phi": run.phi,
                    "tau": run.tau,
                    "player_id": p,
                    "n_examples_available": int(player_counts.get(p, 0)),
                    "n_examples_used": int(len(picked)),
                    "is_famous": bool(is_famous_player_name(p, famous_hints)),
                }
            )

        if not centroids:
            ax.text(0.5, 0.5, "no selectable players", ha="center", va="center", fontsize=6.3, transform=ax.transAxes)
            ax.set_title(run.method_label, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        points = pca_2d(np.stack(centroids, axis=0))
        marker_sizes = 14.0 + 0.9 * np.asarray(n_used, dtype=float)
        ax.scatter(points[:, 0], points[:, 1], s=marker_sizes, color="#B0B0B0", alpha=0.82, edgecolors="none")

        for j, p in enumerate(labels):
            if not is_famous_player_name(p, famous_hints):
                continue
            ax.scatter(points[j, 0], points[j, 1], s=marker_sizes[j] * 1.25, color="#C0392B", alpha=0.95, edgecolors="none")
            ax.annotate(p, (points[j, 0], points[j, 1]), fontsize=5.8, xytext=(2, 2), textcoords="offset points")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{run.method_label} (n={len(labels)})", pad=4)

        # Per-run famous-player heatmaps from stratified centroids.
        famous_limit = 12
        famous_for_heatmap = famous_present[:famous_limit]
        famous_centroids: List[np.ndarray] = []
        famous_labels: List[str] = []
        for p in famous_for_heatmap:
            idx = np.where(pid == p)[0]
            if len(idx) == 0:
                continue
            picked = _stratified_pick_indices(idx, phase, game_bucket, cap=per_player_cap, rng=rng)
            z = emb[picked]
            centroid = z.mean(axis=0, keepdims=True)
            centroid = centroid / np.clip(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-12, None)
            famous_centroids.append(centroid[0])
            famous_labels.append(p)
            heatmap_rows.append(
                {
                    "run": run.name,
                    "player_id": p,
                    "n_examples_available": int(player_counts.get(p, 0)),
                    "n_examples_used": int(len(picked)),
                }
            )
        if len(famous_centroids) >= 2:
            plot_famous_player_heatmaps_for_run(
                out_dir,
                run=run,
                player_order=famous_labels,
                centroid_mat=np.stack(famous_centroids, axis=0),
                tau=run.tau_value,
            )

    for k in range(len(runs), len(axes_arr)):
        axes_arr[k].axis("off")

    savefig(fig, out_dir, "fig_test_player_pca")

    if sampled_rows:
        pd.DataFrame(sampled_rows).to_csv(tables_dir / "test_player_pca_samples.csv", index=False)
    if famous_rows:
        pd.DataFrame(famous_rows).drop_duplicates(subset=["run", "player_id"]).to_csv(
            tables_dir / "famous_players_in_test_split.csv",
            index=False,
        )
    if heatmap_rows:
        pd.DataFrame(heatmap_rows).to_csv(tables_dir / "famous_player_heatmap_players.csv", index=False)


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
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        ylabel = "MRR" if metric_key == "mrr" else "Row cosine hard gap"
        ranked_dotplot(ax, eval_runs, metric_key, title, ylabel=ylabel, annotate_top_k=4)
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
    parser.add_argument("--player-pca-num-players", type=int, default=100)
    parser.add_argument("--player-pca-max-examples-per-player", type=int, default=16)
    parser.add_argument("--player-pca-seed", type=int, default=42)
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

    loaded_eval = sum(int(available_metric_count(r) > 0) for r in eval_runs)
    print(f"[plotter] discovered eval runs={len(eval_runs)} with usable metrics={loaded_eval}")
    for r in eval_runs:
        print(
            f"[plotter] {r.name}: summary={r.eval_summary_path} per_row={r.per_row_metrics_path} metric_count={available_metric_count(r)} mrr={r.metric('mrr')}"
        )

    # Export raw discovered tables
    export_run_table(eval_runs, tables_dir / "eval_runs")
    export_run_table(train_runs, tables_dir / "training_runs")
    export_discovery_debug(eval_runs, tables_dir / "eval_discovery_debug")
    export_metric_coverage(eval_runs, tables_dir / "metric_coverage")

    finalists = choose_best_finalists(eval_runs)
    export_run_table(finalists, tables_dir / "finalists")

    # -----------------------
    # Main paper figure set
    # -----------------------
    plot_tau_sweeps(main_dir, eval_runs)
    plot_phi_comparisons(main_dir, finalists)
    plot_finalist_main_results(main_dir, finalists)
    plot_hard_negative_results(main_dir, finalists)
    plot_pair_score_components(main_dir, finalists)
    plot_classification_summary(main_dir, finalists)
    plot_eval_tau_sensitivity(main_dir, eval_runs)
    plot_super_vs_final(main_dir, eval_runs)
    plot_test_player_pca(
        main_dir,
        tables_dir,
        finalists,
        num_players=args.player_pca_num_players,
        per_player_cap=args.player_pca_max_examples_per_player,
        base_seed=args.player_pca_seed,
    )
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
                    "   - Training-tau sweep for MRR / Recall@1 / hard-gap.",
                    "   - Clarifies retrieval quality as training tau changes.",
                    "",
                    "2. fig_phi_comparisons.pdf/png",
                    "   - Metadata story: phi1 vs phi0 at matched family budget.",
                    "   - Read as: phi1 should help most on retrieval and hard-negative-aware metrics.",
                    "",
                    "3. fig_finalist_main_results.pdf/png",
                    "   - Finalist overall story with automatic fallback when metrics are missing.",
                    "   - In this dataset, mean_logp_gap/KL are largely unavailable, so available metrics are shown instead.",
                    "",
                    "4. fig_hard_negative_results.pdf/png",
                    "   - Application story: v3 > v2 > v1 should show up most clearly here.",
                    "   - Read as: hard-negative-aware metrics are main decision metrics.",
                    "",
                    "5. fig_pair_score_components.pdf/png",
                    "   - Score-level diagnostics: positive vs hardest-negative vs soft-negative cosine means and resulting gaps.",
                    "   - Useful for diagnosing cases where hard gap goes negative but other metrics seem stable.",
                    "",
                    "6. fig_classification_summary.pdf/png",
                    "   - Classification summary: ROC AUC, AP, best F1, threshold.",
                    "",
                    "7. fig_eval_tau_sensitivity.pdf/png",
                    "   - Eval-time tau sensitivity (hard-gap and InfoNCE-like loss from pair metrics).",
                    "   - This is not retrieval recall-vs-threshold; raw threshold curves are not emitted by current eval artifacts.",
                    "",
                    "8. fig_super_vs_final.pdf/png",
                    "   - Stage handling overview for final vs super runs (including missing-stage cases).",
                    "",
                    "9. fig_training_diagnostics_finalists.pdf/png",
                    "   - Stability story: finalists should train smoothly enough without obvious collapse.",
                    "",
                    "10. fig_test_player_pca.pdf/png",
                    "   - PCA(2) of per-player test centroids for each finalist model.",
                    "   - Samples up to 100 players with bias toward high-coverage/famous labeled players; centroid average is stratified across phase and game buckets.",
                    "",
                    "11. fig_promotion_scatter.pdf/png",
                    "   - Process story: screening metrics should roughly predict final metrics, justifying promotion logic.",
                    "",
                    "Metric definitions:",
                    "- Recall@1: fraction of anchors where the true positive is ranked #1 among candidates.",
                    "- MRR: mean reciprocal rank of the true positive; 1.0 is perfect, higher is better.",
                    "- Pair acc vs hardest neg: probability positive score > hardest negative score.",
                    "- Hard gap: mean(pos - hardest_neg) in cosine space.",
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
                    "",
                    "Coverage diagnostics:",
                    "- See tables/metric_coverage.csv for which metrics are actually present.",
                    "- See main_paper/super_vs_final.csv for stage availability and comparisons.",
                    "- See tables/famous_players_in_test_split.csv for famous players found per run.",
                    "- See tables/test_player_pca_samples.csv for sampled players used in PCA plots.",
                ]
            )
        )

    print(f"[done] wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
