
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import chess
from matplotlib.lines import Line2D


# ============================================================
# IEEE CoG-ish plotting defaults
# ============================================================

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 7,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.2,
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

METHOD_COLORS = {
    "maia2": "#03FF6C",
    "sft": "#017BFD",
    "sft_pairwise": "#8303FA",
    "dpo": "#581E1E",
    "dpo_beta=0.02": "#220101",
    "dpo_beta=0.05": "#4B0202",
    "dpo_beta=0.10": "#830202",
    "dpo_beta=0.20": "#990202",
    "dpo_beta=0.40": "#CA0303",
    "dpo_beta=0.60": "#FD0303",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25": "#434B01",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75": "#738001",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25": "#DAF102",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25": "#494D27",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75": "#7B8143",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25": "#E3EC8B",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25": "#4B2801",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75": "#805401",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25": "#F1B102",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25": "#4D4127",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75": "#817043",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25": "#ECC58B",
    "orpo": "#F58518",
    "ipo": "#54A24B",
    "unknown": "#999999",
}

METHOD_TO_METHOD_NICKNAME = {
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25": "sft_and_dpo_w_style_v1_w=0.1_tau=0.25",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75": "sft_and_dpo_w_style_v1_w=0.1_tau=0.75",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25": "sft_and_dpo_w_style_v1_w=0.1_tau=1.25",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25": "sft_and_dpo_w_style_v1_w=0.2_tau=0.25",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75": "sft_and_dpo_w_style_v1_w=0.2_tau=0.75",
    "sft_and_dpo_w_style_sim_utility_weight_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25": "sft_and_dpo_w_style_v1_w=0.2_tau=1.25",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25": "sft_and_dpo_w_style_v2_w=0.1_tau=0.25",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75": "sft_and_dpo_w_style_v2_w=0.1_tau=0.75",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.10_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25": "sft_and_dpo_w_style_v2_w=0.1_tau=1.25",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.25": "sft_and_dpo_w_style_v2_w=0.2_tau=0.25",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=0.75": "sft_and_dpo_w_style_v2_w=0.2_tau=0.75",
    "sft_and_dpo_w_style_v2_beta=0.60_dpo_loss_weight=0.20_style_cp_scale=40.00_style_piece_bonus=1.00_style_positional_bonus=2.00_style_tau=1.25": "sft_and_dpo_w_style_v2_w=0.2_tau=1.25",
}

PHASE_ORDER = ["opening", "middlegame", "endgame"]
PIECE_TYPES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
OPENING_MOVE_ORDER = ["e2e4", "d2d4", "c2c4", "g1f3", "g2g3", "b2b3", "f2f4", "b2b4", "a2a4"]
BLACK_REPLY_ORDER = ["e7e5", "c7c5", "g8f6", "d7d5", "e7e6", "c7c6", "g7g6", "d7d6"]


# ============================================================
# Discovery / loading
# ============================================================

def compact_method_label(method_key: str) -> str:
    # Explicit compact names for the methods you care about most
    manual = {
        "sft": "SFT",
        "maia2": "Maia2",
        "sft_pairwise": "SFT-pair",
        "dpo": "DPO",
        "orpo": "ORPO",
        "ipo": "IPO",
    }
    if method_key in manual:
        return manual[method_key]

    # Handle nickname-based methods first
    nickname = METHOD_TO_METHOD_NICKNAME.get(method_key, method_key)

    # Very compact formatting for the style-v1 / style-v2 models
    m = re.match(
        r"sft_and_dpo_w_style_(v\d)_w=(?P<w>[0-9.]+)_tau=(?P<tau>[0-9.]+)",
        nickname
    )
    if m:
        v = m.group(1).upper()      # V1 / V2
        w = m.group("w")
        tau = m.group("tau")
        return f"SFT+DPO-{v} (w={w}, τ={tau})"

    # DPO beta sweeps
    m = re.match(r"dpo_beta=(?P<beta>[0-9.]+)", method_key)
    if m:
        return f"DPO (β={m.group('beta')})"

    # fallback
    return pretty_method_name(nickname)

@dataclass
class MethodBundle:
    method_key: str
    gm_name: str
    gm_dir: Path
    row_jsonl: Optional[Path] = None
    summary_ext_json: Optional[Path] = None
    summary_json: Optional[Path] = None
    summary_csv: Optional[Path] = None
    opening_probe_json: Optional[Path] = None
    row_df: Optional[pd.DataFrame] = None
    summary_ext: Optional[Dict[str, Any]] = None
    summary_json_obj: Optional[Dict[str, Any]] = None
    summary_csv_df: Optional[pd.DataFrame] = None
    opening_probe: Optional[Dict[str, Any]] = None
    derived: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return compact_method_label(self.method_key)

    @property
    def color(self) -> str:
        return METHOD_COLORS.get(self.method_key, METHOD_COLORS["unknown"])


@dataclass
class PlotContext:
    gm_name: str
    gm_dir: Path
    out_dir: Path
    methods: Dict[str, MethodBundle]
    chosen_methods: List[str]

    def bundles(self) -> List[MethodBundle]:
        return [self.methods[m] for m in self.chosen_methods if m in self.methods]


METHOD_PATTERNS = {
    "row_jsonl": re.compile(r"^eval_per_row_metrics_(?P<method>.+?)_val\.jsonl$"),
    "summary_ext_json": re.compile(
        r"^eval_results_(?:extended_(?P<method1>.+?)|(?P<method2>.+?)_extended)_val\.json$"
    ),
    "summary_json": re.compile(r"^eval_results_(?P<method>.+?)_val\.json$"),
    "summary_csv": re.compile(r"^eval_results_(?P<method>.+?)_val\.csv$"),
    "opening_probe_json": re.compile(r"^opening_probe_policy_(?P<method>.+?)\.json$"),
}


def pretty_method_name(method_key: str) -> str:
    parts = method_key.split("_")
    pretty = []
    for p in parts:
        if p.upper() in {"DPO", "SFT", "IPO", "ORPO"}:
            pretty.append(p.upper())
        else:
            pretty.append(p.capitalize())
    return "-".join(pretty)


def discover_method_bundles(gm_dir: Path) -> Dict[str, MethodBundle]:
    methods: Dict[str, MethodBundle] = {}
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
                MethodBundle(method_key=method, gm_name=gm_dir.name, gm_dir=gm_dir),
            )
            setattr(bundle, attr_name, path)
            break
    return methods


def load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_json(path, lines=True)


def load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


# ============================================================
# Metrics helpers
# ============================================================


def safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.mean())


def safe_median(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.median())


def get_summary_metric(bundle: MethodBundle, keys: Sequence[str]) -> float:
    for src in [bundle.summary_ext, bundle.summary_json_obj]:
        if src is None:
            continue
        for k in keys:
            if k in src and src[k] is not None:
                try:
                    return float(src[k])
                except Exception:
                    pass
    if bundle.summary_csv_df is not None and not bundle.summary_csv_df.empty:
        row = bundle.summary_csv_df.iloc[0]
        for k in keys:
            if k in row and pd.notna(row[k]):
                return float(row[k])
    if bundle.row_df is not None:
        for k in keys:
            if k in bundle.row_df.columns:
                return safe_mean(bundle.row_df[k])
    return float("nan")


def phase_means_from_rows(bundle: MethodBundle, metric_col: str) -> Dict[str, float]:
    out = {p: float("nan") for p in PHASE_ORDER}
    if bundle.row_df is None or "phase" not in bundle.row_df.columns or metric_col not in bundle.row_df.columns:
        return out
    for phase in PHASE_ORDER:
        sub = bundle.row_df.loc[bundle.row_df["phase"] == phase, metric_col]
        out[phase] = safe_mean(sub)
    return out


def phase_means_from_summary(bundle: MethodBundle, metric_col: str) -> Dict[str, float]:
    out = {p: float("nan") for p in PHASE_ORDER}
    src = bundle.summary_ext or {}
    phase_summary = src.get("phase_summary", {})
    if metric_col not in phase_summary:
        return out
    for phase in PHASE_ORDER:
        maybe = phase_summary.get(metric_col, {}).get(phase, {})
        if "mean" in maybe and maybe["mean"] is not None:
            out[phase] = float(maybe["mean"])
    return out


def phase_metric(bundle: MethodBundle, metric_col: str) -> Dict[str, float]:
    out = phase_means_from_summary(bundle, metric_col)
    if any(not math.isnan(v) for v in out.values()):
        return out
    return phase_means_from_rows(bundle, metric_col)


def detect_engine_likeness_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    candidates = {
        "more_engine_like": [
            "pred_more_engine_like_than_chosen_top1",
            "pred_more_engine_like_than_chosen_top3",
            "pred_matches_engine_best",
            "pred_in_engine_top1",
        ],
        "cp_gap": ["pred_cp_gap_to_engine_best", "chosen_cp_gap_to_engine_best"],
    }
    out: Dict[str, Optional[str]] = {}
    for key, names in candidates.items():
        found = None
        for n in names:
            if n in df.columns:
                found = n
                break
        out[key] = found
    return out

def safe_board(fen: Optional[str]) -> Optional[chess.Board]:
    if fen is None:
        return None
    try:
        return chess.Board(fen)
    except Exception:
        return None


def safe_fen_list(x: Any) -> List[Optional[str]]:
    if isinstance(x, list):
        return x
    return []


def count_legal_captures(board: chess.Board) -> int:
    return sum(1 for mv in board.legal_moves if board.is_capture(mv))


def count_legal_checks(board: chess.Board) -> int:
    return sum(1 for mv in board.legal_moves if board.gives_check(mv))


def count_forcing_moves(board: chess.Board) -> int:
    cnt = 0
    for mv in board.legal_moves:
        if board.is_capture(mv) or board.gives_check(mv) or mv.promotion is not None:
            cnt += 1
    return cnt


def center_control_occupancy(board: chess.Board) -> int:
    center = [chess.D4, chess.E4, chess.D5, chess.E5]
    return sum(1 for sq in center if board.piece_at(sq) is not None)


def extended_center_occupancy(board: chess.Board) -> int:
    ext = [
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.D4, chess.E4, chess.F4,
        chess.C5, chess.D5, chess.E5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6,
    ]
    return sum(1 for sq in ext if board.piece_at(sq) is not None)


def minor_piece_development_count(board: chess.Board, color: bool) -> int:
    developed = 0
    piece_map = board.piece_map()

    if color == chess.WHITE:
        home = {
            chess.B1: chess.KNIGHT,
            chess.G1: chess.KNIGHT,
            chess.C1: chess.BISHOP,
            chess.F1: chess.BISHOP,
        }
    else:
        home = {
            chess.B8: chess.KNIGHT,
            chess.G8: chess.KNIGHT,
            chess.C8: chess.BISHOP,
            chess.F8: chess.BISHOP,
        }

    for sq, ptype in home.items():
        p = piece_map.get(sq)
        if p is None or p.piece_type != ptype or p.color != color:
            developed += 1
    return developed


def king_is_castled(board: chess.Board, color: bool) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    if color == chess.WHITE:
        return int(king_sq in {chess.G1, chess.C1})
    return int(king_sq in {chess.G8, chess.C8})


def board_style_features(board: chess.Board, color_to_measure: bool) -> Dict[str, float]:
    return {
        "forcing_moves": float(count_forcing_moves(board)),
        "legal_captures": float(count_legal_captures(board)),
        "legal_checks": float(count_legal_checks(board)),
        "center_occ": float(center_control_occupancy(board)),
        "ext_center_occ": float(extended_center_occupancy(board)),
        "minor_dev": float(minor_piece_development_count(board, color_to_measure)),
        "castled": float(king_is_castled(board, color_to_measure)),
    }


def derive_sequence_style_metrics_for_row(row: pd.Series) -> Dict[str, float]:
    """
    Deeper style metrics using prev_fens / next_fens_chosen.
    All metrics are from the moving side's perspective at the current row.
    """
    out: Dict[str, float] = {}

    cur_board = safe_board(row.get("fen"))
    if cur_board is None:
        return out

    mover = cur_board.turn
    prev_fens = safe_fen_list(row.get("prev_fens"))
    next_fens = safe_fen_list(row.get("next_fens_chosen"))

    prev_board = None
    for x in reversed(prev_fens):
        prev_board = safe_board(x)
        if prev_board is not None:
            break

    next_boards = [safe_board(x) for x in next_fens if x is not None]
    next_boards = [b for b in next_boards if b is not None]

    cur_feats = board_style_features(cur_board, mover)
    out["style::forcing_moves_now"] = cur_feats["forcing_moves"]
    out["style::legal_captures_now"] = cur_feats["legal_captures"]
    out["style::legal_checks_now"] = cur_feats["legal_checks"]
    out["style::center_occ_now"] = cur_feats["center_occ"]
    out["style::ext_center_occ_now"] = cur_feats["ext_center_occ"]
    out["style::minor_dev_now"] = cur_feats["minor_dev"]
    out["style::castled_now"] = cur_feats["castled"]

    if prev_board is not None:
        prev_feats = board_style_features(prev_board, mover)
        out["style::forcing_moves_delta_vs_prev"] = cur_feats["forcing_moves"] - prev_feats["forcing_moves"]
        out["style::center_occ_delta_vs_prev"] = cur_feats["center_occ"] - prev_feats["center_occ"]
        out["style::ext_center_occ_delta_vs_prev"] = cur_feats["ext_center_occ"] - prev_feats["ext_center_occ"]
        out["style::minor_dev_delta_vs_prev"] = cur_feats["minor_dev"] - prev_feats["minor_dev"]
        out["style::castled_delta_vs_prev"] = cur_feats["castled"] - prev_feats["castled"]

    if next_boards:
        fut_forcing = []
        fut_center = []
        fut_ext_center = []
        fut_dev = []
        fut_castled = []

        for nb in next_boards:
            f = board_style_features(nb, mover)
            fut_forcing.append(f["forcing_moves"])
            fut_center.append(f["center_occ"])
            fut_ext_center.append(f["ext_center_occ"])
            fut_dev.append(f["minor_dev"])
            fut_castled.append(f["castled"])

        out["style::future_forcing_mean_5"] = float(np.mean(fut_forcing))
        out["style::future_center_mean_5"] = float(np.mean(fut_center))
        out["style::future_ext_center_mean_5"] = float(np.mean(fut_ext_center))
        out["style::future_minor_dev_mean_5"] = float(np.mean(fut_dev))
        out["style::future_castled_mean_5"] = float(np.mean(fut_castled))

        out["style::future_forcing_max_5"] = float(np.max(fut_forcing))
        out["style::future_center_max_5"] = float(np.max(fut_center))
        out["style::future_minor_dev_max_5"] = float(np.max(fut_dev))
        out["style::future_castled_max_5"] = float(np.max(fut_castled))

        out["style::future_forcing_delta_mean_5"] = float(np.mean(fut_forcing) - cur_feats["forcing_moves"])
        out["style::future_center_delta_mean_5"] = float(np.mean(fut_center) - cur_feats["center_occ"])
        out["style::future_minor_dev_delta_mean_5"] = float(np.mean(fut_dev) - cur_feats["minor_dev"])
        out["style::future_castled_delta_mean_5"] = float(np.mean(fut_castled) - cur_feats["castled"])

        # Crude continuation signals:
        out["style::tactical_followthrough_5"] = float(
            any(
                (count_legal_captures(nb) > 0) or (count_legal_checks(nb) > 0)
                for nb in next_boards[:3]
            )
        )
        out["style::positional_followthrough_5"] = float(
            (np.mean(fut_center) >= cur_feats["center_occ"])
            or (np.mean(fut_dev) > cur_feats["minor_dev"])
            or (np.mean(fut_castled) > cur_feats["castled"])
        )

        # Sequence volatility proxy:
        if len(fut_forcing) >= 2:
            out["style::future_forcing_std_5"] = float(np.std(fut_forcing))
        if len(fut_center) >= 2:
            out["style::future_center_std_5"] = float(np.std(fut_center))

    chosen_is_tactical = float(row.get("chosen_is_tactical", np.nan))
    chosen_is_positional = float(row.get("chosen_is_positional", np.nan))
    pred_is_tactical = float(row.get("pred_pi_is_tactical", np.nan))
    pred_is_positional = float(row.get("pred_pi_is_positional", np.nan))

    if not math.isnan(chosen_is_tactical):
        out["style::chosen_tactical_x_followthrough"] = (
            chosen_is_tactical * out.get("style::tactical_followthrough_5", float("nan"))
        )
    if not math.isnan(chosen_is_positional):
        out["style::chosen_positional_x_followthrough"] = (
            chosen_is_positional * out.get("style::positional_followthrough_5", float("nan"))
        )
    if not math.isnan(pred_is_tactical):
        out["style::pred_tactical_flag"] = pred_is_tactical
    if not math.isnan(pred_is_positional):
        out["style::pred_positional_flag"] = pred_is_positional

    return out


def add_deep_style_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    rows = []
    for _, row in df.iterrows():
        rows.append(derive_sequence_style_metrics_for_row(row))
    deep_df = pd.DataFrame(rows)
    if deep_df.empty:
        return df
    return pd.concat([df.reset_index(drop=True), deep_df.reset_index(drop=True)], axis=1)


def wilson_interval(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0 or math.isnan(p):
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return center - half, center + half


def get_bootstrap_ci(bundle: MethodBundle, key: str) -> Optional[Tuple[float, float, float]]:
    src = bundle.summary_ext or {}
    ci = src.get("bootstrap_ci_row", {})

    aliases = {
        "top1_acc": ["accuracy_top1"],
        "mrr": ["mrr"],
        "mean_p_chosen": ["mean_p_chosen_pi", "p_chosen_pi"],
        "mean_kl": ["kl_pi_ref"],
        "mean_ent_pi": ["entropy_pi"],
        "hit_top3": ["hit_top3"],
        "hit_top5": ["hit_top5"],
        "hit_top10": ["hit_top10"],
        "mean_logp_gap": ["mean_logp_gap_pi"],
    }
    for cand in aliases.get(key, [key]):
        if cand in ci:
            item = ci[cand]
            return float(item["mean"]), float(item["lo"]), float(item["hi"])
    return None


def get_phase_binary_ci(bundle: MethodBundle, metric_col: str, phase: str) -> Tuple[float, float, float]:
    src = bundle.summary_ext or {}
    phase_summary = src.get("phase_summary", {})
    metric_block = phase_summary.get(metric_col, {})
    item = metric_block.get(phase, {})
    if isinstance(item, dict) and "mean" in item and "n" in item:
        mean = float(item["mean"])
        n = int(item["n"])
        lo, hi = wilson_interval(mean, n)
        return mean, lo, hi
    return float("nan"), float("nan"), float("nan")

def compute_derived(bundle: MethodBundle) -> None:
    d: Dict[str, Any] = {}

    d["top1_acc"] = get_summary_metric(bundle, ["top1_accuracy_on_chosen_policy", "accuracy_top1", "top1_accuracy"])
    d["mrr"] = get_summary_metric(bundle, ["mrr"])
    d["mean_logp_gap"] = get_summary_metric(bundle, ["mean_logp_gap_policy_chosen_rejected", "mean_logp_gap_pi"])
    d["mean_p_chosen"] = get_summary_metric(bundle, ["mean_p_chosen_policy", "mean_p_chosen_pi"])
    d["mean_kl"] = get_summary_metric(bundle, ["mean_kl", "kl_pi_ref"])
    d["mean_ent_pi"] = get_summary_metric(bundle, ["mean_ent_pi", "entropy_pi"])
    d["mean_ent_ref"] = get_summary_metric(bundle, ["mean_ent_ref", "entropy_ref"])
    d["mean_gap_improvement"] = get_summary_metric(bundle, ["mean_gap_improvement", "gap_improve"])
    d["top1_acc_by_phase"] = phase_metric(bundle, "correct_top1")
    d["entropy_by_phase"] = phase_metric(bundle, "entropy_pi")
    d["kl_by_phase"] = phase_metric(bundle, "kl_pi_ref")
    d["logp_gap_by_phase"] = phase_metric(bundle, "logp_gap_pi")

    df = bundle.row_df
    if df is not None and not df.empty:
        df = add_deep_style_columns(df)
        bundle.row_df = df

        cols = detect_engine_likeness_cols(df)
        more_engine_col = cols["more_engine_like"]
        cp_gap_col = "pred_cp_gap_to_engine_best" if "pred_cp_gap_to_engine_best" in df.columns else None
        chosen_cp_gap_col = "chosen_cp_gap_to_engine_best" if "chosen_cp_gap_to_engine_best" in df.columns else None
        d["n_rows"] = int(len(df))
        d["engine_like_rate"] = safe_mean(df[more_engine_col]) if more_engine_col else float("nan")
        d["pred_cp_gap_mean"] = safe_mean(df[cp_gap_col]) if cp_gap_col else float("nan")
        d["chosen_cp_gap_mean"] = safe_mean(df[chosen_cp_gap_col]) if chosen_cp_gap_col else float("nan")

        for k in [
            "pi_matches_player_tactical",
            "player_vs_pi_style_agree_tactical",
            "pi_matches_player_positional",
            "player_vs_pi_style_agree_positional",
            "chosen_is_tactical",
            "chosen_is_positional",
            "pred_pi_is_tactical",
            "pred_pi_is_positional",
            "chosen_is_quiet",
            "pred_pi_is_quiet",
            "chosen_is_capture",
            "pred_pi_is_capture",
            "chosen_is_check",
            "pred_pi_is_check",
            "gap_improve",
            "entropy_pi",
            "entropy_ref",
            "entropy_diff_pi_vs_ref",
            "nll_chosen_pi",
            "mrr",
            "correct_top1",
            "p_chosen_pi",
            "logp_gap_pi",
            "kl_pi_ref",
            "chosen_seq_logprob_improve_pi_minus_ref_last3",
            "chosen_seq_logprob_improve_pi_minus_ref_last5",
            "tactical_seq_logprob_pi_last3",
            "tactical_seq_logprob_ref_last3",
            "positional_seq_logprob_pi_last3",
            "positional_seq_logprob_ref_last3",
        ]:
            if k in df.columns:
                d[f"mean::{k}"] = safe_mean(df[k])

        for k in [
            "style::forcing_moves_now",
            "style::legal_captures_now",
            "style::legal_checks_now",
            "style::center_occ_now",
            "style::ext_center_occ_now",
            "style::minor_dev_now",
            "style::castled_now",
            "style::forcing_moves_delta_vs_prev",
            "style::center_occ_delta_vs_prev",
            "style::ext_center_occ_delta_vs_prev",
            "style::minor_dev_delta_vs_prev",
            "style::castled_delta_vs_prev",
            "style::future_forcing_mean_5",
            "style::future_center_mean_5",
            "style::future_ext_center_mean_5",
            "style::future_minor_dev_mean_5",
            "style::future_castled_mean_5",
            "style::future_forcing_max_5",
            "style::future_center_max_5",
            "style::future_minor_dev_max_5",
            "style::future_castled_max_5",
            "style::future_forcing_delta_mean_5",
            "style::future_center_delta_mean_5",
            "style::future_minor_dev_delta_mean_5",
            "style::future_castled_delta_mean_5",
            "style::tactical_followthrough_5",
            "style::positional_followthrough_5",
            "style::chosen_tactical_x_followthrough",
            "style::chosen_positional_x_followthrough",
            "style::future_forcing_std_5",
            "style::future_center_std_5",
        ]:
            if k in df.columns:
                d[f"mean::{k}"] = safe_mean(df[k])

        # threshold-style engine diagnostics
        for thr in [0, 10, 20, 40, 80, 120]:
            col_pred = f"pred_cp_gap_le_{thr}"
            col_chosen = f"chosen_cp_gap_le_{thr}"
            if col_pred in df.columns:
                d[f"mean::{col_pred}"] = safe_mean(df[col_pred])
            if col_chosen in df.columns:
                d[f"mean::{col_chosen}"] = safe_mean(df[col_chosen])

        piece_match: Dict[str, float] = {}
        piece_mass: Dict[str, float] = {}
        piece_overselect: Dict[str, float] = {}
        for piece in PIECE_TYPES:
            mc = f"pi_top1_matches_player_piece_type_{piece}"
            mass_col = f"pi_topk_piece_mass_{piece}"
            oversel_col = f"pi_top1_selects_{piece}_when_player_not_{piece}"
            if mc in df.columns:
                piece_match[piece] = safe_mean(df[mc])
            if mass_col in df.columns:
                piece_mass[piece] = safe_mean(df[mass_col])
            if oversel_col in df.columns:
                piece_overselect[piece] = safe_mean(df[oversel_col])
        d["piece_match"] = piece_match
        d["piece_mass"] = piece_mass
        d["piece_overselect"] = piece_overselect

    if bundle.opening_probe:
        probe = bundle.opening_probe
        white = probe.get("white_first_move_probs", {}) or {}
        d["opening_white_first_move_probs"] = {m: float(white.get(m, 0.0)) for m in OPENING_MOVE_ORDER}

        black = probe.get("black_reply_probs_cond_on_white", {}) or {}
        black_matrix = {}
        for white_move in OPENING_MOVE_ORDER:
            black_dist = ((black.get(white_move) or {}).get("black_reply_probs") or {})
            black_matrix[white_move] = {bm: float(black_dist.get(bm, 0.0)) for bm in BLACK_REPLY_ORDER}
        d["opening_black_reply_matrix"] = black_matrix

        d["opening_white_entropy"] = shannon_entropy(list(d["opening_white_first_move_probs"].values()))

    src = bundle.summary_ext or {}
    player_probe = src.get("player_opening_probe_empirical", {}) or {}
    if player_probe:
        white_emp = player_probe.get("white_first_move_probs", {}) or {}
        d["player_opening_white_first_move_probs"] = {
            m: float(white_emp.get(m, 0.0)) for m in OPENING_MOVE_ORDER
        }
        d["player_opening_white_entropy"] = shannon_entropy(list(d["player_opening_white_first_move_probs"].values()))

        black_emp = player_probe.get("black_reply_probs_cond_on_white", {}) or {}
        black_emp_matrix = {}
        for white_move in OPENING_MOVE_ORDER:
            black_dist = ((black_emp.get(white_move) or {}).get("black_reply_probs") or {})
            black_emp_matrix[white_move] = {bm: float(black_dist.get(bm, 0.0)) for bm in BLACK_REPLY_ORDER}
        d["player_opening_black_reply_matrix"] = black_emp_matrix

    if d.get("opening_white_first_move_probs") and d.get("player_opening_white_first_move_probs"):
        p = [d["player_opening_white_first_move_probs"].get(m, 0.0) for m in OPENING_MOVE_ORDER]
        q = [d["opening_white_first_move_probs"].get(m, 0.0) for m in OPENING_MOVE_ORDER]
        d["opening_white_kl_player_to_model"] = kl_discrete(p, q)
        d["opening_white_l1_player_to_model"] = float(np.sum(np.abs(np.asarray(p) - np.asarray(q))))

    bundle.derived = d


# ============================================================
# Plot helpers
# ============================================================

def dynamic_fig_width(n_methods: int, base: float = 5.8, per_method: float = 0.42, max_width: float = 9.5) -> float:
    return min(max_width, base + per_method * max(0, n_methods - 3))


def legend_ncols(n_methods: int) -> int:
    if n_methods <= 3:
        return n_methods
    if n_methods <= 6:
        return 3
    return 4


def add_top_legend(fig: plt.Figure, ax: plt.Axes, n_methods: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=legend_ncols(n_methods),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

def shannon_entropy(probs: Sequence[float]) -> float:
    arr = np.asarray([p for p in probs if p is not None], dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return float("nan")
    return float(-(arr * np.log(arr)).sum())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finish_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def bundle_series(ctx: PlotContext, key: str) -> Tuple[List[str], np.ndarray, List[str]]:
    labels: List[str] = []
    vals: List[float] = []
    colors: List[str] = []
    for b in ctx.bundles():
        labels.append(b.label)
        vals.append(float(b.derived.get(key, float("nan"))))
        colors.append(b.color)
    return labels, np.asarray(vals, dtype=float), colors

def grouped_phase_plot(
    ctx: PlotContext,
    phase_key: str,
    title: str,
    ylabel: str,
    stem: str,
    percent: bool = False,
) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return

    fig_w = dynamic_fig_width(len(bundles), base=5.8, per_method=0.45, max_width=8.8)
    fig, ax = plt.subplots(figsize=(fig_w, 2.7), constrained_layout=True)

    x = np.arange(len(PHASE_ORDER))
    width = 0.78 / max(1, len(bundles))
    plotted = False

    for i, b in enumerate(bundles):
        data = b.derived.get(phase_key, {})
        vals = [data.get(p, float("nan")) for p in PHASE_ORDER]
        if np.all(np.isnan(vals)):
            continue
        plotted = True
        ax.bar(
            x + (i - (len(bundles) - 1) / 2.0) * width,
            vals,
            width=width,
            label=b.label,
            color=b.color,
        )

    if not plotted:
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PHASE_ORDER])
    ax.set_title(title, pad=6)
    ax.set_ylabel(ylabel)

    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    add_top_legend(fig, ax, len(bundles))
    finish_figure(fig, ctx.out_dir, stem)

def single_bar_plot(
    ctx: PlotContext,
    keys: Sequence[str],
    xlabels: Sequence[str],
    title: str,
    ylabel: str,
    stem: str,
    percent: bool = False,
) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return

    fig_w = dynamic_fig_width(len(bundles), base=6.0, per_method=0.48, max_width=9.2)
    fig, ax = plt.subplots(figsize=(fig_w, 2.9), constrained_layout=True)

    x = np.arange(len(keys))
    width = 0.78 / max(1, len(bundles))
    plotted = False

    for i, b in enumerate(bundles):
        vals = [float(b.derived.get(k, float("nan"))) for k in keys]
        if np.all(np.isnan(vals)):
            continue
        plotted = True
        ax.bar(
            x + (i - (len(bundles) - 1) / 2.0) * width,
            vals,
            width=width,
            label=b.label,
            color=b.color,
        )

    if not plotted:
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_title(title, pad=6)
    ax.set_ylabel(ylabel)

    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    add_top_legend(fig, ax, len(bundles))
    finish_figure(fig, ctx.out_dir, stem)

# ============================================================
# Plot functions
# ============================================================


def plot_core_overview_with_ci(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return

    metrics = [
        ("top1_acc", "Top-1 acc"),
        ("mrr", "MRR"),
        ("mean_p_chosen", "P(chosen)"),
        ("mean_kl", "KL"),
        ("mean_ent_pi", "Entropy"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(10.0, 2.8), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, metrics):
        xs = np.arange(len(bundles))
        vals = []
        lows = []
        highs = []
        colors = []
        labels = []

        for b in bundles:
            v = float(b.derived.get(key, float("nan")))
            ci = get_bootstrap_ci(b, key)
            if math.isnan(v):
                vals.append(np.nan)
                lows.append(0.0)
                highs.append(0.0)
            else:
                vals.append(v)
                if ci is not None:
                    _, lo, hi = ci
                    lows.append(max(0.0, v - lo))
                    highs.append(max(0.0, hi - v))
                else:
                    lows.append(0.0)
                    highs.append(0.0)
            colors.append(b.color)
            labels.append(b.label)

        ax.bar(xs, vals, color=colors, width=0.72)
        ax.errorbar(xs, vals, yerr=np.vstack([lows, highs]), fmt="none", capsize=2, linewidth=1.0)
        ax.set_title(title, pad=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=6.2)

        if key in {"top1_acc", "mrr", "mean_p_chosen"}:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    fig.suptitle(f"{ctx.gm_name}: core metrics with bootstrap CIs", y=1.03)
    finish_figure(fig, ctx.out_dir, "01b_core_overview_with_ci")


def plot_accuracy_by_phase(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return

    fig_w = dynamic_fig_width(len(bundles), base=6.2, per_method=0.50, max_width=9.2)
    fig, ax = plt.subplots(figsize=(fig_w, 3.0), constrained_layout=True)

    x = np.arange(len(PHASE_ORDER))
    width = 0.78 / max(1, len(bundles))
    plotted = False

    for i, b in enumerate(bundles):
        vals, lo_err, hi_err = [], [], []
        for phase in PHASE_ORDER:
            mean, lo, hi = get_phase_binary_ci(b, "correct_top1", phase)
            vals.append(mean)
            lo_err.append(max(0.0, mean - lo) if not math.isnan(mean) else 0.0)
            hi_err.append(max(0.0, hi - mean) if not math.isnan(mean) else 0.0)

        if np.all(np.isnan(vals)):
            continue

        plotted = True
        xpos = x + (i - (len(bundles) - 1) / 2.0) * width
        ax.bar(xpos, vals, width=width, label=b.label, color=b.color)
        ax.errorbar(xpos, vals, yerr=np.vstack([lo_err, hi_err]), fmt="none", capsize=2, linewidth=0.9)

    if not plotted:
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PHASE_ORDER])
    ax.set_title(f"{ctx.gm_name}: top-1 fidelity by phase", pad=6)
    ax.set_ylabel("Top-1 accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    add_top_legend(fig, ax, len(bundles))
    finish_figure(fig, ctx.out_dir, "02_accuracy_by_phase")


def plot_entropy_by_phase(ctx: PlotContext) -> None:
    grouped_phase_plot(
        ctx,
        phase_key="entropy_by_phase",
        title=f"{ctx.gm_name}: policy entropy by phase",
        ylabel="Entropy",
        stem="03_entropy_by_phase",
        percent=False,
    )


def plot_kl_by_phase(ctx: PlotContext) -> None:
    grouped_phase_plot(
        ctx,
        phase_key="kl_by_phase",
        title=f"{ctx.gm_name}: KL(policy || base) by phase",
        ylabel="KL",
        stem="04_kl_by_phase",
        percent=False,
    )

def plot_opening_alignment_to_player(ctx: PlotContext) -> None:
    single_bar_plot(
        ctx,
        keys=["opening_white_kl_player_to_model", "opening_white_l1_player_to_model", "opening_white_entropy", "player_opening_white_entropy"],
        xlabels=["KL(player||model)", "L1 dist", "Model entropy", "Player entropy"],
        title=f"{ctx.gm_name}: opening alignment to empirical player distribution",
        ylabel="Value",
        stem="22_opening_alignment_to_player",
        percent=False,
    )

def plot_engine_likeness_vs_player_fidelity(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    maia_bundle = next((b for b in bundles if b.label == "Maia2"), None)
    if maia_bundle is None:
        return

    x0 = float(maia_bundle.derived.get("engine_like_rate", float("nan")))
    y0 = float(maia_bundle.derived.get("top1_acc", float("nan")))
    if math.isnan(x0) or math.isnan(y0):
        return

    pts = []
    for b in bundles:
        x = float(b.derived.get("engine_like_rate", float("nan")))
        y = float(b.derived.get("top1_acc", float("nan")))
        if math.isnan(x) or math.isnan(y):
            continue
        pts.append((b, x - x0, y - y0))
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)

    xs = []
    ys = []
    for b, x, y in pts:
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=44, color=b.color, label=b.label)

    # zero baselines = same as Maia
    ax.axvline(0.0, linestyle=":", color="gray", linewidth=1.0)
    ax.axhline(0.0, linestyle=":", color="gray", linewidth=1.0)

    # diagonal: equal increase in engine-likeness and player fidelity
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    lo = min(xmin, ymin)
    hi = max(xmax, ymax)

    pad = 0.05 * (hi - lo) if hi > lo else 0.001
    lo -= pad
    hi += pad

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle=":",
        color="red",
        linewidth=1.2,
        label="equal change"
    )

    # nice limits around the data
    xpad = 0.12 * (xmax - xmin) if xmax > xmin else 0.001
    ypad = 0.12 * (ymax - ymin) if ymax > ymin else 0.001
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.set_xlabel("Δ engine-like move rate vs Maia2")
    ax.set_ylabel("Δ top-1 player fidelity vs Maia2")
    ax.set_title(f"{ctx.gm_name}: fidelity gain vs engine-likeness gain", pad=6)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    add_top_legend(fig, ax, len(pts) + 1)
    finish_figure(fig, ctx.out_dir, "05_engine_likeness_tradeoff")


def plot_engine_gap_bars(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    vals = []
    chosen_gap = None

    for b in bundles:
        pred_gap = float(b.derived.get("pred_cp_gap_mean", float("nan")))
        if math.isnan(pred_gap):
            continue
        vals.append((b.label, pred_gap, b.color))

        if chosen_gap is None:
            cg = float(b.derived.get("chosen_cp_gap_mean", float("nan")))
            if not math.isnan(cg):
                chosen_gap = cg

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)

    labels = [v[0] for v in vals]
    heights = [v[1] for v in vals]
    colors = [v[2] for v in vals]
    xs = np.arange(len(vals))

    ax.bar(xs, heights, color=colors, width=0.72)

    if chosen_gap is not None:
        ax.axhline(
            chosen_gap,
            color="red",
            linestyle=":",
            linewidth=1.4,
            label="Human chosen move baseline"
        )
        ax.text(
            len(vals) - 0.4,
            chosen_gap,
            " human chosen baseline",
            color="red",
            va="bottom",
            ha="right",
            fontsize=7,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean CP gap to Stockfish best move")
    ax.set_title(f"{ctx.gm_name}: predicted move engine proximity", pad=6)

    if chosen_gap is not None:
        ax.legend(loc="upper right", frameon=False)

    finish_figure(fig, ctx.out_dir, "06_engine_gap_diagnostics")


def plot_style_agreement_bars(ctx: PlotContext) -> None:
    metrics = [
        "mean::pi_matches_player_tactical",
        "mean::pi_matches_player_positional",
        "mean::style::tactical_followthrough_5",
        "mean::style::positional_followthrough_5",
        "mean::style::chosen_tactical_x_followthrough",
        "mean::style::chosen_positional_x_followthrough",
    ]
    single_bar_plot(
        ctx,
        keys=metrics,
        xlabels=[
            "Tac match",
            "Pos match",
            "Tac seq",
            "Pos seq",
            "Chosen tac→seq",
            "Chosen pos→seq",
        ],
        title=f"{ctx.gm_name}: deeper style agreement",
        ylabel="Rate / mean",
        stem="07_style_agreement",
        percent=False,
    )


def plot_action_type_balance(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6), sharey=True, constrained_layout=True)

    triplets = [
        ("Quiet", "mean::pred_pi_is_quiet", "mean::chosen_is_quiet"),
        ("Capture", "mean::pred_pi_is_capture", "mean::chosen_is_capture"),
        ("Check", "mean::pred_pi_is_check", "mean::chosen_is_check"),
    ]

    short_ids = [f"M{i+1}" for i in range(len(bundles))]
    anything = False

    for ax, (name, pred_key, chosen_key) in zip(axes, triplets):
        x = np.arange(len(bundles))
        pred = np.array([float(b.derived.get(pred_key, float("nan"))) for b in bundles])
        chosen = np.array([float(b.derived.get(chosen_key, float("nan"))) for b in bundles])

        if np.all(np.isnan(pred)) and np.all(np.isnan(chosen)):
            continue

        anything = True
        ax.bar(x - 0.18, chosen, width=0.36, label="Player", color="#BDBDBD")
        ax.bar(x + 0.18, pred, width=0.36, label="Model", color="#5DA5DA")
        ax.set_title(name, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(short_ids, rotation=0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    if not anything:
        plt.close(fig)
        return

    axes[0].set_ylabel("Rate")
    fig.suptitle(f"{ctx.gm_name}: action-type balance vs player", y=1.03)

    # Player / Model legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    # Method mapping underneath
    method_text = "   ".join([f"{sid}={b.label}" for sid, b in zip(short_ids, bundles)])
    fig.text(0.5, -0.02, method_text, ha="center", va="top", fontsize=6.2)

    finish_figure(fig, ctx.out_dir, "08_action_type_balance")


def plot_piece_type_heatmap(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    rows = []
    labels = []
    for b in bundles:
        pm = b.derived.get("piece_match", {})
        vals = [pm.get(piece, float("nan")) for piece in PIECE_TYPES]
        if np.all(np.isnan(vals)):
            continue
        rows.append(vals)
        labels.append(b.label)
    if not rows:
        return
    arr = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(6.1, 1.6 + 0.28 * len(labels)), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(np.arange(len(PIECE_TYPES)))
    ax.set_xticklabels([p.capitalize() for p in PIECE_TYPES], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(f"{ctx.gm_name}: piece-type top-1 fidelity")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Rate")
    finish_figure(fig, ctx.out_dir, "09_piece_type_heatmap")

def plot_deep_style_summary(ctx: PlotContext) -> None:
    metrics = [
        "mean::style::tactical_followthrough_5",
        "mean::style::positional_followthrough_5",
        "mean::style::future_forcing_delta_mean_5",
        "mean::style::future_center_delta_mean_5",
        "mean::style::future_minor_dev_delta_mean_5",
        "mean::style::future_castled_delta_mean_5",
    ]
    single_bar_plot(
        ctx,
        keys=metrics,
        xlabels=[
            "Tac follow",
            "Pos follow",
            "Δ forcing",
            "Δ center",
            "Δ dev",
            "Δ castled",
        ],
        title=f"{ctx.gm_name}: deeper sequence-style metrics",
        ylabel="Mean value",
        stem="17_deep_style_summary",
        percent=False,
    )

def plot_core_overview(ctx: PlotContext) -> None:
    single_bar_plot(
        ctx,
        keys=["top1_acc", "mrr", "mean_p_chosen", "mean_logp_gap", "mean_gap_improvement", "mean_kl"],
        xlabels=["Top-1 acc", "MRR", "P(chosen)", "LogP gap", "Gap improve", "KL"],
        title=f"{ctx.gm_name}: core evaluation metrics",
        ylabel="Metric value",
        stem="01_core_overview",
        percent=False,
    )


def plot_tactical_positional_followthrough(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return

    fig, ax = plt.subplots(figsize=(6.6, 3.0), constrained_layout=True)

    x = np.arange(len(bundles))
    chosen_tac = np.array([float(b.derived.get("mean::style::chosen_tactical_x_followthrough", float("nan"))) for b in bundles])
    chosen_pos = np.array([float(b.derived.get("mean::style::chosen_positional_x_followthrough", float("nan"))) for b in bundles])

    ax.bar(x - 0.18, chosen_tac, width=0.36, label="Chosen tactical → follow-through")
    ax.bar(x + 0.18, chosen_pos, width=0.36, label="Chosen positional → follow-through")

    ax.set_xticks(x)
    ax.set_xticklabels([b.label for b in bundles], rotation=22, ha="right")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(f"{ctx.gm_name}: tactical vs positional follow-through", pad=6)
    ax.legend(frameon=False)

    finish_figure(fig, ctx.out_dir, "18_tactical_positional_followthrough")

def plot_style_volatility(ctx: PlotContext) -> None:
    metrics = [
        "mean::style::future_forcing_std_5",
        "mean::style::future_center_std_5",
        "mean_ent_pi",
        "mean_kl",
    ]
    single_bar_plot(
        ctx,
        keys=metrics,
        xlabels=["Force std", "Center std", "Entropy", "KL"],
        title=f"{ctx.gm_name}: continuation volatility and distribution sharpness",
        ylabel="Mean value",
        stem="19_style_volatility",
        percent=False,
    )

def plot_style_by_phase(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return

    metrics = [
        ("pi_matches_player_tactical", "Tactical match"),
        ("pi_matches_player_positional", "Positional match"),
    ]

    for metric, label in metrics:
        fig_w = dynamic_fig_width(len(bundles), base=5.8, per_method=0.45, max_width=8.8)
        fig, ax = plt.subplots(figsize=(fig_w, 2.8), constrained_layout=True)

        x = np.arange(len(PHASE_ORDER))
        width = 0.78 / max(1, len(bundles))
        plotted = False

        for i, b in enumerate(bundles):
            vals = phase_metric(b, metric)
            y = [vals.get(p, float("nan")) for p in PHASE_ORDER]
            if np.all(np.isnan(y)):
                continue
            plotted = True
            ax.bar(
                x + (i - (len(bundles) - 1) / 2.0) * width,
                y,
                width=width,
                label=b.label,
                color=b.color,
            )

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in PHASE_ORDER])
        ax.set_ylabel("Rate")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_title(f"{ctx.gm_name}: {label.lower()} by phase", pad=6)

        add_top_legend(fig, ax, len(bundles))
        finish_figure(fig, ctx.out_dir, f"20_{metric}_by_phase")

def plot_opening_first_move_probs(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return

    fig_w = dynamic_fig_width(len(bundles), base=6.8, per_method=0.55, max_width=10.0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.2), constrained_layout=True)

    x = np.arange(len(OPENING_MOVE_ORDER))
    width = 0.72 / max(1, len(bundles))
    plotted = False

    for i, b in enumerate(bundles):
        dist = b.derived.get("opening_white_first_move_probs")
        if not dist:
            continue
        plotted = True
        vals = [dist.get(m, 0.0) for m in OPENING_MOVE_ORDER]
        ax.bar(
            x + (i - (len(bundles) - 1) / 2.0) * width,
            vals,
            width=width,
            label=b.label,
            color=b.color,
            alpha=0.86,
        )

    # Overlay empirical player opening rates once
    emp = None
    for b in bundles:
        emp = b.derived.get("player_opening_white_first_move_probs")
        if emp:
            break

    if emp:
        emp_vals = [emp.get(m, 0.0) for m in OPENING_MOVE_ORDER]
        ax.plot(
            x,
            emp_vals,
            marker="o",
            linestyle="--",
            linewidth=1.4,
            label="Player empirical",
        )

    if not plotted:
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels(OPENING_MOVE_ORDER, rotation=28, ha="right")
    ax.set_ylabel("Probability / empirical rate")
    ax.set_title(f"{ctx.gm_name}: opening first-move distribution vs player empirical", pad=6)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    add_top_legend(fig, ax, len(bundles) + (1 if emp else 0))
    finish_figure(fig, ctx.out_dir, "10_opening_first_move_probs")


def plot_opening_entropy_vs_accuracy(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    pts = []
    maia_pt = None

    for b in bundles:
        x = float(b.derived.get("opening_white_entropy", float("nan")))
        y = float((b.derived.get("top1_acc_by_phase") or {}).get("opening", float("nan")))
        if math.isnan(x) or math.isnan(y):
            continue
        pts.append((b, x, y))
        if b.label.upper() == "MAIA2":
            maia_pt = (x, y)

    if not pts:
        return

    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)

    xs, ys = [], []
    for b, x, y in pts:
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=44, color=b.color, label=b.label)

    if maia_pt is not None:
        x0, y0 = maia_pt
        ax.axvline(x0, linestyle=":", color="gray", linewidth=1.0)
        ax.axhline(y0, linestyle=":", color="gray", linewidth=1.0)

        ax.annotate(
            "preferred",
            xy=(x0 - 0.01, y0 + 0.002),
            xytext=(x0 + 0.01, y0 + 0.003),
            arrowprops=dict(arrowstyle="->", linestyle=":"),
            fontsize=7,
        )

    xpad = 0.08 * (max(xs) - min(xs)) if max(xs) > min(xs) else 0.01
    ypad = 0.08 * (max(ys) - min(ys)) if max(ys) > min(ys) else 0.002
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    ax.set_xlabel("Opening prediction entropy")
    ax.set_ylabel("Opening top-1 fidelity")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(f"{ctx.gm_name}: entropy vs opening fidelity", pad=6)

    add_top_legend(fig, ax, len(pts))
    finish_figure(fig, ctx.out_dir, "11_opening_entropy_vs_accuracy")

def plot_engine_gap_thresholds(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return

    thresholds = [0, 10, 20, 40, 80, 120]
    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)

    for b in bundles:
        y = [float(b.derived.get(f"mean::pred_cp_gap_le_{thr}", float("nan"))) for thr in thresholds]
        if np.all(np.isnan(y)):
            continue
        ax.plot(thresholds, y, marker="o", linewidth=1.4, label=b.label, color=b.color)

    # chosen-human baseline from first available bundle
    for b in bundles:
        human = [float(b.derived.get(f"mean::chosen_cp_gap_le_{thr}", float("nan"))) for thr in thresholds]
        if not np.all(np.isnan(human)):
            ax.plot(thresholds, human, marker="s", linestyle="--", linewidth=1.2, label="Player chosen baseline")
            break

    ax.set_xlabel("CP gap threshold")
    ax.set_ylabel("Rate within threshold")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(f"{ctx.gm_name}: engine proximity threshold curves", pad=6)
    ax.legend(frameon=False, ncol=2)

    finish_figure(fig, ctx.out_dir, "21_engine_gap_thresholds")

def kl_discrete(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

def plot_black_reply_heatmaps(ctx: PlotContext) -> None:
    for b in ctx.bundles():
        matrix = b.derived.get("opening_black_reply_matrix")
        if not matrix:
            continue
        arr = np.array([[matrix[w].get(br, 0.0) for br in BLACK_REPLY_ORDER] for w in OPENING_MOVE_ORDER], dtype=float)
        fig, ax = plt.subplots(figsize=(6.3, 3.2), constrained_layout=True)
        im = ax.imshow(arr, aspect="auto", cmap="magma")
        ax.set_xticks(np.arange(len(BLACK_REPLY_ORDER)))
        ax.set_xticklabels(BLACK_REPLY_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(OPENING_MOVE_ORDER)))
        ax.set_yticklabels(OPENING_MOVE_ORDER)
        ax.set_title(f"{ctx.gm_name}: {b.label} opening reply matrix")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Probability")
        finish_figure(fig, ctx.out_dir, f"12_opening_reply_heatmap_{b.method_key}")


def plot_distribution_boxplots(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    metrics = [
        ("entropy_pi", "Entropy", "13_box_entropy"),
        ("gap_improve", "Gap improvement", "14_box_gap_improve"),
        ("pred_cp_gap_to_engine_best", "Pred CP gap to engine best", "15_box_pred_cp_gap"),
    ]

    for col, title_y, stem in metrics:
        data = []
        labels = []
        colors = []

        for b in bundles:
            if b.row_df is None or col not in b.row_df.columns:
                continue
            vals = pd.to_numeric(b.row_df[col], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(b.label)
            colors.append(b.color)

        if not data:
            continue

        fig_w = dynamic_fig_width(len(labels), base=5.6, per_method=0.45, max_width=9.0)
        fig, ax = plt.subplots(figsize=(fig_w, 2.8), constrained_layout=True)

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)

        ax.set_ylabel(title_y)
        ax.set_title(f"{ctx.gm_name}: distribution of {title_y.lower()}", pad=6)
        ax.tick_params(axis="x", rotation=18, labelsize=6.5)

        finish_figure(fig, ctx.out_dir, stem)


def plot_phase_gap_improvement(ctx: PlotContext) -> None:
    bundles = ctx.bundles()
    if not bundles:
        return
    fig, ax = plt.subplots(figsize=(6.9, 2.6), constrained_layout=True)
    x = np.arange(len(PHASE_ORDER))
    width = 0.8 / max(1, len(bundles))
    plotted = False
    for i, b in enumerate(bundles):
        vals = phase_metric(b, "gap_improve")
        y = [vals.get(p, float("nan")) for p in PHASE_ORDER]
        if np.all(np.isnan(y)):
            continue
        plotted = True
        ax.bar(x + (i - (len(bundles) - 1) / 2.0) * width, y, width=width, label=b.label, color=b.color)
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PHASE_ORDER])
    ax.set_ylabel("Gap improvement")
    ax.set_title(f"{ctx.gm_name}: gap improvement by phase")
    ax.legend(ncol=min(4, len(bundles)), frameon=False)
    finish_figure(fig, ctx.out_dir, "16_gap_improve_by_phase")


PLOT_REGISTRY: List[Tuple[str, Callable[[PlotContext], None]]] = [
    ("core_overview", plot_core_overview),
    ("core_overview_with_ci", plot_core_overview_with_ci),
    ("accuracy_by_phase", plot_accuracy_by_phase),
    ("entropy_by_phase", plot_entropy_by_phase),
    ("kl_by_phase", plot_kl_by_phase),
    ("engine_likeness_tradeoff", plot_engine_likeness_vs_player_fidelity),
    ("engine_gap_diagnostics", plot_engine_gap_bars),
    ("engine_gap_thresholds", plot_engine_gap_thresholds),
    ("style_agreement", plot_style_agreement_bars),
    ("deep_style_summary", plot_deep_style_summary),
    ("tactical_positional_followthrough", plot_tactical_positional_followthrough),
    ("style_volatility", plot_style_volatility),
    ("style_by_phase", plot_style_by_phase),
    ("action_type_balance", plot_action_type_balance),
    ("piece_type_heatmap", plot_piece_type_heatmap),
    ("opening_first_move_probs", plot_opening_first_move_probs),
    ("opening_alignment_to_player", plot_opening_alignment_to_player),
    ("opening_entropy_vs_accuracy", plot_opening_entropy_vs_accuracy),
    ("opening_reply_heatmaps", plot_black_reply_heatmaps),
    ("distribution_boxplots", plot_distribution_boxplots),
    ("gap_improve_by_phase", plot_phase_gap_improvement),
]


# ============================================================
# Tables / report helpers
# ============================================================


def write_method_summary_csv(ctx: PlotContext) -> None:
    rows = []
    for b in ctx.bundles():
        rows.append(
            {
                "method": b.method_key,
                "label": b.label,
                "top1_acc": b.derived.get("top1_acc"),
                "mrr": b.derived.get("mrr"),
                "mean_p_chosen": b.derived.get("mean_p_chosen"),
                "mean_logp_gap": b.derived.get("mean_logp_gap"),
                "mean_gap_improvement": b.derived.get("mean_gap_improvement"),
                "mean_kl": b.derived.get("mean_kl"),
                "mean_ent_pi": b.derived.get("mean_ent_pi"),
                "mean_ent_ref": b.derived.get("mean_ent_ref"),
                "engine_like_rate": b.derived.get("engine_like_rate"),
                "pred_cp_gap_mean": b.derived.get("pred_cp_gap_mean"),
                "opening_probe_entropy": b.derived.get("opening_white_entropy"),
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(ctx.out_dir / "method_summary_table.csv", index=False)


def write_manifest(ctx: PlotContext) -> None:
    manifest = {
        "gm_name": ctx.gm_name,
        "gm_dir": str(ctx.gm_dir),
        "out_dir": str(ctx.out_dir),
        "methods": {},
        "plots": [name for name, _ in PLOT_REGISTRY],
    }
    for b in ctx.bundles():
        manifest["methods"][b.method_key] = {
            "row_jsonl": str(b.row_jsonl) if b.row_jsonl else None,
            "summary_ext_json": str(b.summary_ext_json) if b.summary_ext_json else None,
            "summary_json": str(b.summary_json) if b.summary_json else None,
            "summary_csv": str(b.summary_csv) if b.summary_csv else None,
            "opening_probe_json": str(b.opening_probe_json) if b.opening_probe_json else None,
        }
    with (ctx.out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    # python .\src\grandmaster_dpo\graphs\single_gm\generate_graphs.py --gm_name caruana --eval_root .\final_experiments_for_paper\experiment1\eval_results_twic --out_dir .\final_experiments_for_paper\experiment1\eval_graphs_twic
    p = argparse.ArgumentParser(
        description=(
            "Generate IEEE CoG-style figures for grandmaster eval outputs. "
            "The script discovers method bundles inside eval_root/gm_name."
        )
    )
    p.add_argument("--gm_name", required=True, help="Grandmaster folder name, e.g. caruana")
    p.add_argument(
        "--eval_root",
        required=True,
        help=r"Path to root eval directory, e.g. final_experiments_for_paper\experiment1\eval_results_twic",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        required=True,
        help="Explicit output directory. Should be at same level as eval_root",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional subset of discovered methods, e.g. dpo sft sft_pairwise",
    )
    p.add_argument(
        "--plots",
        nargs="*",
        default=None,
        help="Optional subset of plot registry names to run",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gm_dir = Path(args.eval_root) / args.gm_name
    if not gm_dir.exists():
        raise FileNotFoundError(f"GM directory does not exist: {gm_dir}")

    methods = discover_method_bundles(gm_dir)
    if not methods:
        raise RuntimeError(f"No method bundles discovered in {gm_dir}")

    chosen_methods = list(methods.keys()) if args.methods is None else [m for m in args.methods if m in methods]
    if not chosen_methods:
        raise RuntimeError(f"No requested methods found. Discovered methods: {sorted(methods.keys())}")

    for method in chosen_methods:
        bundle = methods[method]
        bundle.row_df = load_jsonl(bundle.row_jsonl)
        bundle.summary_ext = load_json(bundle.summary_ext_json)
        bundle.summary_json_obj = load_json(bundle.summary_json)
        bundle.summary_csv_df = load_csv(bundle.summary_csv)
        bundle.opening_probe = load_json(bundle.opening_probe_json)
        compute_derived(bundle)

    out_dir = Path(f"{args.out_dir}/{args.gm_name}/figures_cog")
    ensure_dir(out_dir)

    ctx = PlotContext(
        gm_name=args.gm_name,
        gm_dir=gm_dir,
        out_dir=out_dir,
        methods=methods,
        chosen_methods=chosen_methods,
    )

    selected_plot_names = {name for name, _ in PLOT_REGISTRY} if args.plots is None else set(args.plots)
    for name, fn in PLOT_REGISTRY:
        if name in selected_plot_names:
            try:
                fn(ctx)
            except Exception as exc:
                print(f"[WARN] plot '{name}' failed: {exc}")

    write_method_summary_csv(ctx)
    write_manifest(ctx)
    print(f"Done. Figures written to: {out_dir}")
    print(f"Methods: {', '.join(chosen_methods)}")


if __name__ == "__main__":
    main()
