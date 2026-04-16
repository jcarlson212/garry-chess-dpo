from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
import math
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np
from grandmaster_dpo.eval.configs import OpeningLogitDistConfig
import torch


@dataclass
class CandidatePolicy:
    """
    Distribution q over a candidate set, plus aligned metadata.
    """
    cand_moves: List[str]               # [K]
    cand_cps: List[int]                 # [K]
    cand_idxs: List[int]                # [K] vocab idx for each move, -1 if missing
    cand_logits: torch.Tensor           # [K] unnormalized logits used to form q
    cand_probs: List[float]             # [K] normalized q
    best_cp: int

    def index_of(self, uci: str) -> int:
        try:
            return self.cand_moves.index(uci)
        except ValueError:
            return -1

    @property
    def size(self) -> int:
        return len(self.cand_moves)

@dataclass
class SfPerPosResult:
    """
    Stores enough information to support:
      - main paper metrics
      - downstream slicing/aggregation
      - debugging / ablations later
    """
    inference_cp_best: int

    reference_cp_best: Optional[int]

    num_candidates: int
    num_candidates_in_inference_window: int
    num_candidates_in_reference_window: Optional[int]

    entropy_cond_inference: float

    p_chosen_cond_inference: float
    p_rejected_cond_inference: float
    logp_chosen_cond_inference: float
    logp_rejected_cond_inference: float
    gap_logp_cond_inference: float

    cand_hit1: float
    cand_hit3: float
    cand_hit5: float
    cand_hit10: float

    full_hit1: float
    full_hit3: float
    full_hit5: float
    full_hit10: float

    kl_q_vs_base: float

    # Useful downstream summaries
    expected_cp_cond_inference: float
    cp_std_cond_inference: float
    top1_cp_cond_inference: int
    top1_uci_cond_inference: str

    expected_cp_cond_reference: Optional[float]
    cp_std_cond_reference: Optional[float]
    top1_cp_cond_reference: Optional[int]
    top1_uci_cond_reference: Optional[str]
    
    chosen_cp_inference: Optional[float] 
    chosen_cp_reference: Optional[float]
    chosen_rank_reference: Optional[int] 
    chosen_rank_inference: Optional[int]

    rejected_cp_inference: Optional[float] 
    rejected_cp_reference: Optional[float]
    rejected_rank_reference: Optional[int] 
    rejected_rank_inference: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        # Get names of all fields expected by the __init__ method
        field_names = {f.name for f in fields(cls) if f.init}
        # Filter data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

@dataclass
class EvalPerRowInput:
    logp_pi_ch: torch.Tensor
    logp_pi_rj: torch.Tensor
    logp_ref_ch: torch.Tensor
    logp_ref_rj: torch.Tensor
    logits_pi_m: torch.Tensor
    logits_ref_m: torch.Tensor
    chosen_idx: torch.Tensor
    batch_size: int
    ply_idxs: List[int]
    opening_counts_adv: Dict[str, Counter]
    fens: List[str]
    loss: torch.Tensor
    game_ids: List[str]
    chosen: List[str]
    rejected: List[str]
    opening_cfg: OpeningLogitDistConfig
    opening_by_game: Dict[str, str]
    opening_prefixes: List[List[str]]

    @classmethod
    def from_dict(cls, data):
        # Get names of all fields expected by the __init__ method
        field_names = {f.name for f in fields(cls) if f.init}
        # Filter data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

@dataclass
class EvalRowMetrics:
    game_id: str
    ply_idx: int
    ply_abs: int
    phase: str
    fen: str
    chosen_uci: str
    rejected_uci: str
    pred_uci: str
    correct_top1: float
    hit_top5: float
    hit_top10: float
    rank_chosen: int
    mrr: float
    logp_gap_pi: float
    logp_gap_ref: float
    gap_improve: float
    p_chosen_pi: float
    kl_pi_ref: float
    nll_chosen_pi: float

    @classmethod
    def from_dict(cls, data):
        # Get names of all fields expected by the __init__ method
        field_names = {f.name for f in fields(cls) if f.init}
        # Filter data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

@dataclass
class _SfBatchContext:
    fens: List[str]
    chosen_uci_list: List[str]
    rejected_uci_list: List[str]
    masked_policy_logits: torch.Tensor
    base_full_log_probs: torch.Tensor
    full_hit1: torch.Tensor
    full_hit5: torch.Tensor
    full_hit10: torch.Tensor
    stockfish_inference: Dict[Any, Any]
    stockfish_reference: Dict[Any, Any]

@dataclass
class _SFCandidate:
    uci: str
    cp: int

@dataclass
class EvalAggMetrics:
    gm: str
    tag: str
    maia_type: str
    device: str
    num_rows: int
    mean_loss: float
    mean_logp_gap_pi: float
    mean_logp_gap_ref: float
    mean_gap_improve: float
    top1_accuracy: float
    hit5: float
    hit10: float
    mrr: float
    mean_p_chosen: float
    mean_kl_pi_ref: float
    opening_family_counts_by_game: Dict[str, Counter]
    opening_summary: Dict[str, List[Dict[str, float]]]
    stockfish: Dict[Any, Any]

    @classmethod
    def from_dict(cls, data):
        # Get names of all fields expected by the __init__ method
        field_names = {f.name for f in fields(cls) if f.init}
        # Filter data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


@dataclass
class SfHelperEvalAggregate:
    # Row counts
    total_rows: int = 0
    valid_rows: int = 0

    # Per-row metric storage for global summaries
    rows: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Opening fingerprints using top1 move in inference window
    opening_top1_white_ply0: Counter[str] = field(default_factory=Counter)
    opening_top1_black_ply1: Counter[str] = field(default_factory=Counter)

    # Conditional slices:
    # CP gap buckets are cumulative: gap <= threshold
    cp_gap_thresholds: List[int] = field(default_factory=lambda: [10, 20, 40, 80, 160, 320, 640])

    # Rank buckets:
    # exact rank in {1,3,5,10}
    rank_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    # cumulative rank <= k in {1,3,5,10}
    rank_thresholds: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    cp_gap_slices: DefaultDict[str, DefaultDict[int, List[Dict[str, float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    rank_exact_slices: DefaultDict[str, DefaultDict[int, List[Dict[str, float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    rank_leq_slices: DefaultDict[str, DefaultDict[int, List[Dict[str, float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    def _append_metric(self, name: str, value: Optional[float]) -> None:
        if value is None:
            return
        try:
            x = float(value)
        except Exception:
            return
        if math.isnan(x) or math.isinf(x):
            return
        self.rows[name].append(x)

    def _safe_reciprocal_rank(self, rank: Optional[int]) -> Optional[float]:
        if rank is None:
            return None
        if rank <= 0:
            return None
        return 1.0 / float(rank)

    def add_processed_row(
        self,
        *,
        result: "SfPerPosResult",
        ply_abs: int,
    ) -> None:
        self.valid_rows += 1

        # Core per-row metrics
        self._append_metric("p_chosen_cond_inference", result.p_chosen_cond_inference)
        self._append_metric("p_rejected_cond_inference", result.p_rejected_cond_inference)
        self._append_metric("logp_chosen_cond_inference", result.logp_chosen_cond_inference)
        self._append_metric("logp_rejected_cond_inference", result.logp_rejected_cond_inference)
        self._append_metric("gap_logp_cond_inference", result.gap_logp_cond_inference)

        self._append_metric("cand_hit1", result.cand_hit1)
        self._append_metric("cand_hit3", result.cand_hit3)
        self._append_metric("cand_hit5", result.cand_hit5)
        self._append_metric("cand_hit10", result.cand_hit10)

        self._append_metric("full_hit1", result.full_hit1)
        self._append_metric("full_hit3", result.full_hit3)
        self._append_metric("full_hit5", result.full_hit5)
        self._append_metric("full_hit10", result.full_hit10)

        self._append_metric("entropy_cond_inference", result.entropy_cond_inference)
        self._append_metric("kl_q_vs_base", result.kl_q_vs_base)

        self._append_metric("num_candidates", result.num_candidates)
        self._append_metric("num_candidates_in_inference_window", result.num_candidates_in_inference_window)
        self._append_metric("num_candidates_in_reference_window", result.num_candidates_in_reference_window)

        self._append_metric("expected_cp_cond_inference", result.expected_cp_cond_inference)
        self._append_metric("cp_std_cond_inference", result.cp_std_cond_inference)
        self._append_metric("top1_cp_cond_inference", result.top1_cp_cond_inference)

        self._append_metric("expected_cp_cond_reference", result.expected_cp_cond_reference)
        self._append_metric("cp_std_cond_reference", result.cp_std_cond_reference)
        self._append_metric("top1_cp_cond_reference", result.top1_cp_cond_reference)

        if result.chosen_cp_inference is not None:
            self._append_metric("chosen_cp_inference", result.chosen_cp_inference)
            self._append_metric(
                "chosen_cp_gap_inference",
                float(result.inference_cp_best - result.chosen_cp_inference),
            )
        if result.chosen_cp_reference is not None and result.reference_cp_best is not None:
            self._append_metric("chosen_cp_reference", result.chosen_cp_reference)
            self._append_metric(
                "chosen_cp_gap_reference",
                float(result.reference_cp_best - result.chosen_cp_reference),
            )

        if result.rejected_cp_inference is not None:
            self._append_metric("rejected_cp_inference", result.rejected_cp_inference)
            self._append_metric(
                "rejected_cp_gap_inference",
                float(result.inference_cp_best - result.rejected_cp_inference),
            )
        if result.rejected_cp_reference is not None and result.reference_cp_best is not None:
            self._append_metric("rejected_cp_reference", result.rejected_cp_reference)
            self._append_metric(
                "rejected_cp_gap_reference",
                float(result.reference_cp_best - result.rejected_cp_reference),
            )

        if result.chosen_rank_inference is not None:
            self._append_metric("chosen_rank_inference", result.chosen_rank_inference)
            self._append_metric("chosen_mrr_inference", self._safe_reciprocal_rank(result.chosen_rank_inference))
        if result.chosen_rank_reference is not None:
            self._append_metric("chosen_rank_reference", result.chosen_rank_reference)
            self._append_metric("chosen_mrr_reference", self._safe_reciprocal_rank(result.chosen_rank_reference))

        if result.rejected_rank_inference is not None:
            self._append_metric("rejected_rank_inference", result.rejected_rank_inference)
            self._append_metric("rejected_mrr_inference", self._safe_reciprocal_rank(result.rejected_rank_inference))
        if result.rejected_rank_reference is not None:
            self._append_metric("rejected_rank_reference", result.rejected_rank_reference)
            self._append_metric("rejected_mrr_reference", self._safe_reciprocal_rank(result.rejected_rank_reference))

        self._append_metric("chosen_present_in_inference", 1.0 if result.chosen_cp_inference is not None else 0.0)
        self._append_metric("chosen_present_in_reference", 1.0 if result.chosen_cp_reference is not None else 0.0)
        self._append_metric("rejected_present_in_inference", 1.0 if result.rejected_cp_inference is not None else 0.0)
        self._append_metric("rejected_present_in_reference", 1.0 if result.rejected_cp_reference is not None else 0.0)

        self._append_metric("chosen_best_inference", 1.0 if result.chosen_rank_inference == 1 else 0.0)
        self._append_metric("chosen_best_reference", 1.0 if result.chosen_rank_reference == 1 else 0.0)
        self._append_metric("rejected_best_inference", 1.0 if result.rejected_rank_inference == 1 else 0.0)
        self._append_metric("rejected_best_reference", 1.0 if result.rejected_rank_reference == 1 else 0.0)

        self._append_metric("has_reference", 1.0 if result.reference_cp_best is not None else 0.0) # should be 500

        # Opening counters
        if ply_abs == 0:
            self.opening_top1_white_ply0[result.top1_uci_cond_inference] += 1
        if ply_abs == 1:
            self.opening_top1_black_ply1[result.top1_uci_cond_inference] += 1

        # Conditional slices for chosen move quality
        self._add_conditional_slices(result=result)

    def _add_conditional_slices(self, *, result: "SfPerPosResult") -> None:
        row_payload = {
            "p_chosen_cond_inference": float(result.p_chosen_cond_inference),
            "mrr_inference": self._safe_reciprocal_rank(result.chosen_rank_inference),
            "mrr_reference": self._safe_reciprocal_rank(result.chosen_rank_reference),
        }

        # CP gap slices: chosen move gap to best
        if result.chosen_cp_reference is not None and result.reference_cp_best is not None:
            gap_ref = float(result.reference_cp_best - result.chosen_cp_reference)
            for thr in self.cp_gap_thresholds:
                if gap_ref <= thr:
                    self.cp_gap_slices["reference"][thr].append(dict(row_payload))

        if result.chosen_cp_inference is not None:
            gap_inf = float(result.inference_cp_best - result.chosen_cp_inference)
            for thr in self.cp_gap_thresholds:
                if gap_inf <= thr:
                    self.cp_gap_slices["inference"][thr].append(dict(row_payload))

        # Exact rank slices
        if result.chosen_rank_reference is not None:
            rk_ref = int(result.chosen_rank_reference)
            if rk_ref in self.rank_values:
                self.rank_exact_slices["reference"][rk_ref].append(dict(row_payload))
            for thr in self.rank_thresholds:
                if rk_ref <= thr:
                    self.rank_leq_slices["reference"][thr].append(dict(row_payload))

        if result.chosen_rank_inference is not None:
            rk_inf = int(result.chosen_rank_inference)
            if rk_inf in self.rank_values:
                self.rank_exact_slices["inference"][rk_inf].append(dict(row_payload))
            for thr in self.rank_thresholds:
                if rk_inf <= thr:
                    self.rank_leq_slices["inference"][thr].append(dict(row_payload))

    def _serialize_counter(self, counter: Counter[str], topn: int = 50) -> List[Dict[str, Any]]:
        return [{"uci": uci, "count": count} for uci, count in counter.most_common(topn)]

    def _series_summary(self, values: List[float]) -> Dict[str, Any]:
        vals = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
        n = len(vals)
        if n == 0:
            return {
                "n": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "p5": float("nan"),
                "p25": float("nan"),
                "p50": float("nan"),
                "p75": float("nan"),
                "p95": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "ci95_half_width": float("nan"),
            }

        arr = np.asarray(vals, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        if n <= 1:
            ci_half = 0.0
        else:
            sem = std / math.sqrt(n)
            if scipy_stats is not None:
                tcrit = float(scipy_stats.t.ppf(0.975, df=n - 1))
            else:
                tcrit = 1.96
            ci_half = float(tcrit * sem)

        return {
            "n": n,
            "mean": mean,
            "std": std,
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "ci95_low": mean - ci_half,
            "ci95_high": mean + ci_half,
            "ci95_half_width": ci_half,
        }

    def _summarize_named_metrics(self, metric_names: List[str]) -> Dict[str, Any]:
        return {name: self._series_summary(self.rows.get(name, [])) for name in metric_names}

    def _summarize_slice_rows(
        self,
        rows: List[Dict[str, float]],
        *,
        include_reference_mrr: bool,
    ) -> Dict[str, Any]:
        p_chosen_vals = [r["p_chosen_cond_inference"] for r in rows if r.get("p_chosen_cond_inference") is not None]
        out = {
            "p_chosen_cond_inference": self._series_summary(p_chosen_vals),
        }

        if include_reference_mrr:
            mrr_vals = [r["mrr_reference"] for r in rows if r.get("mrr_reference") is not None]
            out["mrr"] = self._series_summary(mrr_vals)
        else:
            mrr_vals = [r["mrr_inference"] for r in rows if r.get("mrr_inference") is not None]
            out["mrr"] = self._series_summary(mrr_vals)

        return out

    def _summarize_cp_gap_slices(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"reference": {}, "inference": {}}

        for thr in self.cp_gap_thresholds:
            out["reference"][str(thr)] = self._summarize_slice_rows(
                self.cp_gap_slices["reference"][thr],
                include_reference_mrr=True,
            )
            out["inference"][str(thr)] = self._summarize_slice_rows(
                self.cp_gap_slices["inference"][thr],
                include_reference_mrr=False,
            )
        return out

    def _summarize_rank_exact_slices(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"reference": {}, "inference": {}}

        for rk in self.rank_values:
            out["reference"][str(rk)] = self._summarize_slice_rows(
                self.rank_exact_slices["reference"][rk],
                include_reference_mrr=True,
            )
            out["inference"][str(rk)] = self._summarize_slice_rows(
                self.rank_exact_slices["inference"][rk],
                include_reference_mrr=False,
            )
        return out

    def _summarize_rank_leq_slices(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"reference": {}, "inference": {}}

        for rk in self.rank_thresholds:
            out["reference"][str(rk)] = self._summarize_slice_rows(
                self.rank_leq_slices["reference"][rk],
                include_reference_mrr=True,
            )
            out["inference"][str(rk)] = self._summarize_slice_rows(
                self.rank_leq_slices["inference"][rk],
                include_reference_mrr=False,
            )
        return out

    def to_dict(self, *, sf_config: Any, sf_opening_summary: Any) -> Dict[str, Any]:
        key_metrics = [
            "p_chosen_cond_inference",
            "p_rejected_cond_inference",
            "logp_chosen_cond_inference",
            "logp_rejected_cond_inference",
            "gap_logp_cond_inference",
            "cand_hit1",
            "cand_hit3",
            "cand_hit5",
            "cand_hit10",
            "full_hit1",
            "full_hit3",
            "full_hit5",
            "full_hit10",
            "entropy_cond_inference",
            "kl_q_vs_base",
            "num_candidates",
            "num_candidates_in_inference_window",
            "num_candidates_in_reference_window",
            "expected_cp_cond_inference",
            "cp_std_cond_inference",
            "top1_cp_cond_inference",
            "expected_cp_cond_reference",
            "cp_std_cond_reference",
            "top1_cp_cond_reference",
            "chosen_cp_inference",
            "chosen_cp_reference",
            "chosen_cp_gap_inference",
            "chosen_cp_gap_reference",
            "chosen_rank_inference",
            "chosen_rank_reference",
            "chosen_mrr_inference",
            "chosen_mrr_reference",
            "rejected_cp_inference",
            "rejected_cp_reference",
            "rejected_cp_gap_inference",
            "rejected_cp_gap_reference",
            "rejected_rank_inference",
            "rejected_rank_reference",
            "rejected_mrr_inference",
            "rejected_mrr_reference",
            "chosen_present_in_inference",
            "chosen_present_in_reference",
            "rejected_present_in_inference",
            "rejected_present_in_reference",
            "chosen_best_inference",
            "chosen_best_reference",
            "rejected_best_inference",
            "rejected_best_reference",
        ]

        return {
            "sf_total_rows": self.total_rows,
            "sf_valid_rows": self.valid_rows,
            "metrics": self._summarize_named_metrics(key_metrics),

            # Conditional chosen-quality analyses
            "chosen_quality_by_cp_gap_to_best": self._summarize_cp_gap_slices(),
            "chosen_quality_by_rank_exact": self._summarize_rank_exact_slices(),
            "chosen_quality_by_rank_leq": self._summarize_rank_leq_slices(),

            # Opening fingerprints
            "opening_top1_white_ply0": self._serialize_counter(self.opening_top1_white_ply0),
            "opening_top1_black_ply1": self._serialize_counter(self.opening_top1_black_ply1),

            "sf_config": asdict(sf_config),
            "sf_opening_summary": sf_opening_summary,
        }
    
