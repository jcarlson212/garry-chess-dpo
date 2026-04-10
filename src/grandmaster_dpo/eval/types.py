from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List
from grandmaster_dpo.eval.configs import OpeningLogitDistConfig
import torch


@dataclass
class SfPerPosResult:
    selected_uci: str
    is_best_sf: bool
    cp_selected: int
    cp_best: int
    cp_gap: float
    entropy: float
    logp_selected_full: float

    # q over candidates (policy restricted to SF top-k set after cp-window filter)
    p_chosen_cond: float
    p_rejected_cond: float
    logp_chosen_cond: float
    logp_rejected_cond: float
    gap_logp_cond: float

    cand_hit1: float
    cand_hit5: float
    cand_hit10: float

    # full-policy hits on chosen
    full_hit1: float
    full_hit5: float
    full_hit10: float

    # divergence vs base full dist on candidate support
    kl_q_vs_base: float

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
    
@dataclass
class SfHelperEvalAggregate:
    # Row counts
    total_rows: int = 0
    valid_rows: int = 0

    # Candidate-restricted / SF-helper metrics (stored as running sums until finalized)
    top1_selected_matches_chosen_sum: float = 0.0
    top5_candidate_hit_sum: float = 0.0
    top10_candidate_hit_sum: float = 0.0
    cp_gap_sum: float = 0.0
    best_sf_selected_sum: float = 0.0
    entropy_sum: float = 0.0
    selected_full_logp_sum: float = 0.0

    chosen_present_in_candidates_sum: float = 0.0
    chosen_present_probability_sum: float = 0.0

    chosen_conditional_probability_sum: float = 0.0
    rejected_conditional_probability_sum: float = 0.0
    chosen_conditional_logp_sum: float = 0.0
    rejected_conditional_logp_sum: float = 0.0
    conditional_logp_gap_sum: float = 0.0
    kl_q_vs_base_sum: float = 0.0

    # Full-policy hit metrics
    full_hit1_sum: float = 0.0
    full_hit5_sum: float = 0.0
    full_hit10_sum: float = 0.0

    # Human-likeness opening fingerprints
    opening_selected_white_ply0: Counter[str] = field(default_factory=Counter)
    opening_selected_black_ply1: Counter[str] = field(default_factory=Counter)

    def add_processed_row(
        self,
        *,
        selected_matches_chosen: bool,
        cand_hit5: float,
        cand_hit10: float,
        cp_gap: float,
        is_best_sf: bool,
        entropy: float,
        logp_selected_full: float,
        p_chosen_cond: float,
        p_rejected_cond: float,
        logp_chosen_cond: float,
        logp_rejected_cond: float,
        gap_logp_cond: float,
        kl_q_vs_base: float,
        full_hit1: float,
        full_hit5: float,
        full_hit10: float,
        selected_uci: str,
        ply_abs: int,
    ) -> None:
        self.valid_rows += 1

        self.top1_selected_matches_chosen_sum += 1.0 if selected_matches_chosen else 0.0
        self.top5_candidate_hit_sum += float(cand_hit5)
        self.top10_candidate_hit_sum += float(cand_hit10)
        self.cp_gap_sum += float(cp_gap)
        self.best_sf_selected_sum += 1.0 if is_best_sf else 0.0
        self.entropy_sum += float(entropy)
        self.selected_full_logp_sum += float(logp_selected_full)

        chosen_is_in_candidates = 1.0 if p_chosen_cond > 0.0 else 0.0
        self.chosen_present_in_candidates_sum += chosen_is_in_candidates
        if chosen_is_in_candidates:
            self.chosen_present_probability_sum += float(p_chosen_cond)

        self.chosen_conditional_probability_sum += float(p_chosen_cond)
        self.rejected_conditional_probability_sum += float(p_rejected_cond)
        self.chosen_conditional_logp_sum += float(logp_chosen_cond)
        self.rejected_conditional_logp_sum += float(logp_rejected_cond)
        self.conditional_logp_gap_sum += float(gap_logp_cond)
        self.kl_q_vs_base_sum += float(kl_q_vs_base)

        self.full_hit1_sum += float(full_hit1)
        self.full_hit5_sum += float(full_hit5)
        self.full_hit10_sum += float(full_hit10)

        if ply_abs == 0:
            self.opening_selected_white_ply0[selected_uci] += 1
        if ply_abs == 1:
            self.opening_selected_black_ply1[selected_uci] += 1

    def _safe_mean(self, total: float, denom: float) -> float:
        return total / denom if denom > 0 else float("nan")

    def _serialize_counter(self, counter: Counter[str], topn: int = 50) -> List[Dict[str, Any]]:
        return [{"uci": uci, "count": count} for uci, count in counter.most_common(topn)]

    def to_dict(self, *, sf_config: Any, sf_opening_summary: Any) -> Dict[str, Any]:
        valid_row_count = float(self.valid_rows)

        result = {
            "sf_total_rows": self.total_rows,
            "sf_valid_rows": self.valid_rows,

            "sf_help_top1_acc": self._safe_mean(self.top1_selected_matches_chosen_sum, valid_row_count),
            "sf_help_top5_hit_cand": self._safe_mean(self.top5_candidate_hit_sum, valid_row_count),
            "sf_help_top10_hit_cand": self._safe_mean(self.top10_candidate_hit_sum, valid_row_count),
            "sf_help_mean_cp_gap": self._safe_mean(self.cp_gap_sum, valid_row_count),
            "sf_help_best_sf_rate": self._safe_mean(self.best_sf_selected_sum, valid_row_count),
            "sf_help_mean_entropy": self._safe_mean(self.entropy_sum, valid_row_count),
            "sf_help_mean_logp_selected_full": self._safe_mean(self.selected_full_logp_sum, valid_row_count),

            "sf_help_chosen_in_cands_rate": self._safe_mean(self.chosen_present_in_candidates_sum, valid_row_count),
            "sf_help_mean_p_chosen_cond": self._safe_mean(self.chosen_conditional_probability_sum, valid_row_count),
            "sf_help_mean_p_rejected_cond": self._safe_mean(self.rejected_conditional_probability_sum, valid_row_count),
            "sf_help_mean_logp_chosen_cond": self._safe_mean(self.chosen_conditional_logp_sum, valid_row_count),
            "sf_help_mean_logp_rejected_cond": self._safe_mean(self.rejected_conditional_logp_sum, valid_row_count),
            "sf_help_mean_gap_logp_cond": self._safe_mean(self.conditional_logp_gap_sum, valid_row_count),
            "sf_help_mean_kl_q_vs_base": self._safe_mean(self.kl_q_vs_base_sum, valid_row_count),

            "full_hit1": self._safe_mean(self.full_hit1_sum, valid_row_count),
            "full_hit5": self._safe_mean(self.full_hit5_sum, valid_row_count),
            "full_hit10": self._safe_mean(self.full_hit10_sum, valid_row_count),

            "sf_help_mean_p_chosen_given_in": self._safe_mean(
                self.chosen_present_probability_sum,
                self.chosen_present_in_candidates_sum,
            ),

            "opening_selected_white_ply0": self._serialize_counter(self.opening_selected_white_ply0),
            "opening_selected_black_ply1": self._serialize_counter(self.opening_selected_black_ply1),

            "sf_config": asdict(sf_config),
            "sf_opening_summary": sf_opening_summary,
        }
        return result