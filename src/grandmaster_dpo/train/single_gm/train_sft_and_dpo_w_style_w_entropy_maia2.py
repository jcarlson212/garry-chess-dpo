from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import chess
import math

import torch
from torch.utils.data import DataLoader, Dataset

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move


# ----------------------------
# Dataset
# ----------------------------

class DpoPairs(Dataset):
    def __init__(self, jsonl_path: str):
        self.path = jsonl_path
        self.rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        self.game_id_and_ply_to_prev_10_plys = {}
        self.game_id_and_ply_to_fut_10_plys = {}

        def create_window_item(rows, index, target_game):
            if index < 0:
                return None 
            elif index >= len(rows):
                return None
            else:
                if rows[index]["meta"]["game_header_hash"] != target_game:
                    return None
                return rows[index]

        for i, r in enumerate(self.rows):
            hash_key = f'{r["meta"]["game_header_hash"]}_{r["meta"]["ply_idx"]}'
            self.game_id_and_ply_to_prev_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i-10, i)]
            self.game_id_and_ply_to_fut_10_plys[hash_key] = [create_window_item(self.rows, i, r["meta"]["game_header_hash"]) for i in range(i+1, i+11)]
            
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        p = r["prompt"]
        return {
            "fen": p["fen"],
            "elo_self": int(p.get("elo_self", 2800)),
            "elo_oppo": int(p.get("elo_oppo", 2800)),
            "chosen": r["chosen"],       # UCI
            "rejected": r["rejected"],   # UCI
            "label": int(r.get("preference", {}).get("label", 1)),
            "meta": r.get("meta", {}),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {"fen": [], "elo_self": [], "elo_oppo": [], "chosen": [], "rejected": [], "meta": []}
    for b in batch:
        for k in out:
            out[k].append(b[k])
    return out


# ----------------------------
# Helpers
# ----------------------------

def extract_move_cp(meta: dict, uci: str) -> float:
    sf_moves = meta["stockfish"]["sf_moves_returned"]
    for sf_uci, cp in sf_moves:
        if sf_uci == uci:
            return float(cp)
        
    cp_values = [cp for _, cp in sf_moves]
    fallback_cp = float(min(cp_values)) if cp_values else 0.0
    #print(
     #   f"[WARN] move {uci} not found in sf_moves_returned "
    #    f"(game={meta.get('game_header_hash')}, ply={meta.get('ply_idx')}). "
    #    f"Using fallback cp={fallback_cp}"
    #)
    return fallback_cp

# ----------------------------
# Phase helpers
# ----------------------------

def infer_phase_from_ply(ply_idx: int) -> str:
    if ply_idx <= 20:
        return "opening"
    if ply_idx <= 50:
        return "middlegame"
    return "endgame"


def phase_weights(phase: str) -> Dict[str, float]:
    """
    Relative importance of style dimensions by phase.
    """
    if phase == "opening":
        return {
            "cp": 1.0,
            "piece": 1.2,
            "positional": 0.7,
            "tactical": 0.7,
            "development": 1.5,
            "castle": 1.5,
            "center": 1.2,
            "wing": 0.7,
            "pawn_break": 1.1,
            "king_pressure": 0.6,
            "mobility_delta": 0.6,
            "distance": 0.8,
        }
    if phase == "middlegame":
        return {
            "cp": 1.0,
            "piece": 1.0,
            "positional": 1.0,
            "tactical": 1.3,
            "development": 0.5,
            "castle": 0.4,
            "center": 0.9,
            "wing": 0.9,
            "pawn_break": 1.2,
            "king_pressure": 1.3,
            "mobility_delta": 1.0,
            "distance": 0.9,
        }
    return {
        "cp": 1.0,
        "piece": 0.8,
        "positional": 1.1,
        "tactical": 0.7,
        "development": 0.1,
        "castle": 0.0,
        "center": 0.5,
        "wing": 0.5,
        "pawn_break": 0.6,
        "king_pressure": 0.6,
        "mobility_delta": 1.4,
        "distance": 1.0,
    }


# ----------------------------
# Geometry / board helpers
# ----------------------------

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXT_CENTER_SQUARES = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}


def manhattan_distance(sq1: chess.Square, sq2: chess.Square) -> int:
    f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
    f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
    return abs(f1 - f2) + abs(r1 - r2)


def file_group(square: chess.Square) -> str:
    f = chess.square_file(square)
    if f <= 2:
        return "queenside"
    if f >= 5:
        return "kingside"
    return "center"


def square_region(square: chess.Square) -> str:
    if square in CENTER_SQUARES:
        return "center"
    if square in EXT_CENTER_SQUARES:
        return "extended_center"
    return "flank"


def is_development_move(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False

    # Focus on opening-style development patterns
    if piece.piece_type not in (chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.ROOK):
        return False

    if piece.color == chess.WHITE:
        home_rank = 0
    else:
        home_rank = 7

    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)

    # Leaving back rank, usually toward activity
    if from_rank == home_rank and to_rank != home_rank:
        return True

    return False


def is_retreat(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False

    color = piece.color
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)

    if color == chess.WHITE:
        return to_rank < from_rank
    return to_rank > from_rank


def is_positional(board: chess.Board, move: chess.Move) -> bool:
    if move.promotion is not None:
        return False
    if board.is_capture(move):
        return False
    if board.gives_check(move):
        return False
    return True


def same_piece_type(board: chess.Board, move_a: chess.Move, move_b: chess.Move) -> bool:
    piece_a = board.piece_at(move_a.from_square)
    piece_b = board.piece_at(move_b.from_square)
    if piece_a is None or piece_b is None:
        return False
    return piece_a.piece_type == piece_b.piece_type


def move_piece_type(board: chess.Board, move: chess.Move) -> int | None:
    piece = board.piece_at(move.from_square)
    return None if piece is None else piece.piece_type


def is_castle(move: chess.Move) -> bool:
    return move in [
        chess.Move.from_uci("e1g1"),
        chess.Move.from_uci("e1c1"),
        chess.Move.from_uci("e8g8"),
        chess.Move.from_uci("e8c8"),
    ]


def gives_check_safe(board: chess.Board, move: chess.Move) -> bool:
    return board.gives_check(move)


def is_capture_safe(board: chess.Board, move: chess.Move) -> bool:
    return board.is_capture(move)


def legal_mobility_after(board: chess.Board, move: chess.Move) -> int:
    b = board.copy(stack=False)
    b.push(move)
    return sum(1 for _ in b.legal_moves)


def attacked_squares_by_moved_piece_after(board: chess.Board, move: chess.Move) -> int:
    """
    Approximate activity of the moved piece after the move.
    """
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0

    b = board.copy(stack=False)
    b.push(move)
    moved_piece = b.piece_at(move.to_square)
    if moved_piece is None:
        return 0

    return len(b.attacks(move.to_square))


def king_zone_pressure_after(board: chess.Board, move: chess.Move) -> int:
    """
    Rough proxy: number of attacked squares around enemy king after move.
    """
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0

    b = board.copy(stack=False)
    b.push(move)

    enemy_king_sq = b.king(not piece.color)
    if enemy_king_sq is None:
        return 0

    king_zone = {enemy_king_sq} | set(b.attacks(enemy_king_sq))
    moved_piece = b.piece_at(move.to_square)
    if moved_piece is None:
        return 0

    attacked = set(b.attacks(move.to_square))
    return len(attacked & king_zone)


def opens_or_advances_center_pawn(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False

    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)

    # c/d/e/f pawns often drive structure/style
    return from_file in {2, 3, 4, 5} or to_file in {2, 3, 4, 5}


def is_pawn_break_like(board: chess.Board, move: chess.Move) -> bool:
    """
    Lightweight pawn-break proxy.
    """
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False

    if board.is_capture(move):
        return True

    # pawn advancing into contact / central thrust-ish
    b = board.copy(stack=False)
    b.push(move)
    sq = move.to_square
    color = piece.color

    # Is pawn now adjacent to enemy pawns / creating direct tension?
    enemy = not color
    for df in (-1, 1):
        f = chess.square_file(sq) + df
        r = chess.square_rank(sq)
        if 0 <= f < 8:
            adj_sq = chess.square(f, r)
            p = b.piece_at(adj_sq)
            if p is not None and p.color == enemy and p.piece_type == chess.PAWN:
                return True

    return opens_or_advances_center_pawn(board, move)


def normalize_similarity(diff: float, scale: float) -> float:
    return math.exp(-abs(diff) / max(scale, 1e-6))


# ----------------------------
# Feature extraction
# ----------------------------

@dataclass
class MoveStyleFeatures:
    piece_type: int | None
    is_quiet: bool
    is_capture: bool
    gives_check: bool
    is_castle: bool
    is_development: bool
    is_retreat: bool
    is_pawn_break: bool
    region_to: str
    file_group_to: str
    move_distance: int
    moved_piece_activity: int
    king_pressure: int
    mobility_after: int


def extract_style_features(board: chess.Board, move: chess.Move) -> MoveStyleFeatures:
    return MoveStyleFeatures(
        piece_type=move_piece_type(board, move),
        is_quiet=is_positional(board, move),
        is_capture=is_capture_safe(board, move),
        gives_check=gives_check_safe(board, move),
        is_castle=is_castle(move),
        is_development=is_development_move(board, move),
        is_retreat=is_retreat(board, move),
        is_pawn_break=is_pawn_break_like(board, move),
        region_to=square_region(move.to_square),
        file_group_to=file_group(move.to_square),
        move_distance=manhattan_distance(move.from_square, move.to_square),
        moved_piece_activity=attacked_squares_by_moved_piece_after(board, move),
        king_pressure=king_zone_pressure_after(board, move),
        mobility_after=legal_mobility_after(board, move),
    )


# ----------------------------
# Main style score
# ----------------------------

def compute_style_score(
    fen: str,
    chosen_uci: str,
    rejected_uci: str,
    chosen_cp: float,
    rejected_cp: float,
    ply_idx: int | None = None,
    phase: str | None = None,
    cp_scale: float = 35.0,
    activity_scale: float = 3.0,
    mobility_scale: float = 6.0,
    distance_scale: float = 2.5,
    bonus_same_piece: float = 1.25,
    bonus_same_region: float = 1.10,
    bonus_same_side: float = 1.08,
) -> float:
    """
    Higher score => chosen/rejected are stylistically MORE similar.

    This is designed for use in DPO weighting where similar moves
    should get LOWER training emphasis.
    """
    board = chess.Board(fen)
    ch = chess.Move.from_uci(chosen_uci)
    rj = chess.Move.from_uci(rejected_uci)

    if phase is None:
        phase = infer_phase_from_ply(ply_idx if ply_idx is not None else board.fullmove_number * 2)

    pw = phase_weights(phase)

    f_ch = extract_style_features(board, ch)
    f_rj = extract_style_features(board, rj)

    # Continuous similarities
    cp_sim = normalize_similarity(float(chosen_cp) - float(rejected_cp), cp_scale)
    activity_sim = normalize_similarity(
        f_ch.moved_piece_activity - f_rj.moved_piece_activity,
        activity_scale,
    )
    king_pressure_sim = normalize_similarity(
        f_ch.king_pressure - f_rj.king_pressure,
        activity_scale,
    )
    mobility_sim = normalize_similarity(
        f_ch.mobility_after - f_rj.mobility_after,
        mobility_scale,
    )
    distance_sim = normalize_similarity(
        f_ch.move_distance - f_rj.move_distance,
        distance_scale,
    )

    # Binary / categorical similarities
    piece_sim = 1.0 if f_ch.piece_type == f_rj.piece_type else 0.0
    positional_sim = 1.0 if f_ch.is_quiet == f_rj.is_quiet else 0.0
    tactical_sim = 1.0 if (
        f_ch.is_capture == f_rj.is_capture and
        f_ch.gives_check == f_rj.gives_check
    ) else 0.0
    development_sim = 1.0 if f_ch.is_development == f_rj.is_development else 0.0
    castle_sim = 1.0 if f_ch.is_castle == f_rj.is_castle else 0.0
    center_sim = 1.0 if f_ch.region_to == f_rj.region_to else 0.0
    wing_sim = 1.0 if f_ch.file_group_to == f_rj.file_group_to else 0.0
    pawn_break_sim = 1.0 if f_ch.is_pawn_break == f_rj.is_pawn_break else 0.0

    # Weighted average of style-dimension similarities
    parts = [
        (cp_sim, pw["cp"]),
        (piece_sim, pw["piece"]),
        (positional_sim, pw["positional"]),
        (tactical_sim, pw["tactical"]),
        (development_sim, pw["development"]),
        (castle_sim, pw["castle"]),
        (center_sim, pw["center"]),
        (wing_sim, pw["wing"]),
        (pawn_break_sim, pw["pawn_break"]),
        (king_pressure_sim, pw["king_pressure"]),
        (mobility_sim, pw["mobility_delta"]),
        (distance_sim, pw["distance"]),
    ]

    num = sum(v * w for v, w in parts)
    den = sum(w for _, w in parts) + 1e-12
    sim = num / den

    # Mild multiplicative bonuses for especially close stylistic matches
    if piece_sim > 0.5:
        sim *= bonus_same_piece
    if center_sim > 0.5:
        sim *= bonus_same_region
    if wing_sim > 0.5:
        sim *= bonus_same_side

    # Keep score positive and bounded-ish
    # Typical range ends up around [0.0, ~1.7]
    return float(max(sim, 1e-6))

def kl_pi_ref_from_logits(
    logits_pi: torch.Tensor,   # [B, V] already legal-masked (illegal = -inf)
    logits_ref: torch.Tensor,  # [B, V] already legal-masked
) -> torch.Tensor:
    """
    Returns KL(pi || ref) per example: [B]
    """
    logp_pi = torch.log_softmax(logits_pi, dim=-1)     # [B, V]
    logp_ref = torch.log_softmax(logits_ref, dim=-1)   # [B, V]
    p_pi = logp_pi.exp()
    # KL(pi||ref) = sum_a pi(a) (log pi(a) - log ref(a))
    kl = (p_pi * (logp_pi - logp_ref)).sum(dim=-1)     # [B]
    return kl

def ply_from_fen(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    ply = 2 * (fullmove - 1)
    if side == "b":
        ply += 1
    return ply

def device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s in ("mps",):
        return torch.device("mps")
    return torch.device(s)


def max_elo_supported(elo_dict: dict) -> int:
    # find keys like ">=2000" and return 2000
    mx = None
    for k in elo_dict.keys():
        m = re.match(r"^>=\s*(\d+)$", k)
        if m:
            mx = max(mx or 0, int(m.group(1)))
    return mx if mx is not None else 3000


def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    # legal_moves is 0/1 mask same shape as logits
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))

def chosen_index_tensor(
    fens: List[str],
    all_moves_dict: Dict[str, int],
    moves_uci: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert UCI -> Maia vocab index (mirroring if fen is black-to-move).
    Returns idx_t with -1 for unknown moves (should be rare; those will be ignored).
    """
    idxs: List[int] = []
    for fen, uci in zip(fens, moves_uci):
        side = fen.split(" ")[1]
        uci_eff = mirror_move(uci) if side == "b" else uci
        idx = all_moves_dict.get(uci_eff, None)
        idxs.append(-1 if idx is None else int(idx))
    return torch.tensor(idxs, device=device, dtype=torch.long)

def batch_preprocess(
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    fens: List[str],
    elo_self: List[int],
    elo_oppo: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calls maia2.inference.preprocessing (repo version you pasted):
      preprocessing(fen, elo_self, elo_oppo, elo_dict, all_moves_dict)
        -> board_input, elo_self_cat, elo_oppo_cat, legal_moves_mask
    """
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


def forward_logits(
    maia2_model: torch.nn.Module,
    board_input: torch.Tensor,
    elo_self_tensor: torch.Tensor,
    elo_oppo_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Maia2 repo inference uses:
      logits_maia, _, logits_value = model(boards, elos_self, elos_oppo)
    """
    logits_maia, _, _ = maia2_model(board_input, elo_self_tensor, elo_oppo_tensor)
    return logits_maia


def move_logprob_from_logits(
    logits: torch.Tensor,
    fens: List[str],
    all_moves_dict: Dict[str, int],
    moves_uci: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert UCI -> index in Maia vocab (mirroring if fen is black-to-move),
    then gather logprob from logits.
    """
    logp_all = torch.log_softmax(logits, dim=-1)

    idxs: List[int] = []
    for fen, uci in zip(fens, moves_uci):
        side = fen.split(" ")[1]
        uci_eff = mirror_move(uci) if side == "b" else uci
        idx = all_moves_dict.get(uci_eff, None)
        idxs.append(-1 if idx is None else int(idx))

    idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
    safe_idx = idx_t.clamp(min=0)

    gathered = logp_all.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)
    gathered = torch.where(idx_t >= 0, gathered, torch.full_like(gathered, -1e9))
    return gathered


def dpo_loss_style_weighted(
    logp_pi_ch: torch.Tensor,
    logp_pi_rj: torch.Tensor,
    logp_ref_ch: torch.Tensor,
    logp_ref_rj: torch.Tensor,
    style_score: torch.Tensor,
    beta: float,
    tau: float,
) -> torch.Tensor:
    pi_gap = logp_pi_ch - logp_pi_rj
    ref_gap = logp_ref_ch - logp_ref_rj
    x = beta * (pi_gap - ref_gap)

    per_example_loss = -torch.nn.functional.logsigmoid(x)
    weights = torch.exp(-style_score / tau)
    return (weights * per_example_loss).sum() / weights.sum().clamp_min(1e-12)

def supervised_nll_loss(
    logits_masked: torch.Tensor,
    idx_t: torch.Tensor,
) -> torch.Tensor:
    """
    Standard supervised fine-tuning objective:
      loss = -mean(log p(chosen_move))
    ignoring examples where chosen move isn't in vocab (idx == -1).
    """
    logp_all = torch.log_softmax(logits_masked, dim=-1)  # [B, V]

    valid = idx_t >= 0
    if valid.sum().item() == 0:
        # return a zero scalar that still has grad
        return logits_masked.sum() * 0.0

    safe_idx = idx_t.clamp(min=0)
    gathered = logp_all.gather(dim=1, index=safe_idx.view(-1, 1)).squeeze(1)  # [B]
    gathered = gathered[valid]
    return (-gathered).mean()
# ----------------------------
# Eval
# ----------------------------

@torch.no_grad()
def evaluate(
    policy: torch.nn.Module,
    ref: torch.nn.Module,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    loader: DataLoader,
    device: torch.device,
    beta: float,
    dpo_loss_weight: float,
    style_tau: float,
    style_cp_scale: float,
    style_piece_bonus: float,
    style_positional_bonus: float
) -> Dict[str, float]:
    policy.eval()
    ref.eval()

    total_loss = 0.0
    n = 0

    for batch in loader:
        fens = batch["fen"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        meta_list = batch["meta"]
        ply_idxs = [r["ply_idx"] for r in meta_list]
        board_input, legal_moves, es_t, eo_t = batch_preprocess(
            all_moves_dict, elo_dict, batch["fen"], batch["elo_self"], batch["elo_oppo"], device
        )

        logits_pi = forward_logits(policy, board_input, es_t, eo_t)
        logits_ref = forward_logits(ref, board_input, es_t, eo_t)

        logits_pi = apply_legal_mask(logits_pi, legal_moves)
        logits_ref = apply_legal_mask(logits_ref, legal_moves)
        idx_t = chosen_index_tensor(batch["fen"], all_moves_dict, batch["chosen"], device)
        
        logp_pi_ch = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["chosen"], device)
        logp_pi_rj = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["rejected"], device)

        logp_ref_ch = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["chosen"], device)
        logp_ref_rj = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["rejected"], device)
        
        # NEW: chosen/rejected CPs
        chosen_cps = [extract_move_cp(m, ch) for m, ch in zip(meta_list, chosen)]
        rejected_cps = [extract_move_cp(m, rj) for m, rj in zip(meta_list, rejected)]

        # NEW: style similarity scores
        style_scores = torch.tensor(
            [
                compute_style_score(
                    fen=fen,
                    chosen_uci=ch,
                    rejected_uci=rj,
                    chosen_cp=ch_cp,
                    rejected_cp=rj_cp,
                    ply_idx=ply_idx,
                    phase=None,
                )
                for fen, ch, rj, ch_cp, rj_cp, ply_idx in zip(fens, chosen, rejected, chosen_cps, rejected_cps, ply_idxs)
            ],
            dtype=torch.float32,
            device=device,
        )

        loss = (
            dpo_loss_weight
            * dpo_loss_style_weighted(
                logp_pi_ch=logp_pi_ch,
                logp_pi_rj=logp_pi_rj,
                logp_ref_ch=logp_ref_ch,
                logp_ref_rj=logp_ref_rj,
                style_score=style_scores,
                beta=beta,
                tau=style_tau,
            )
            + supervised_nll_loss(logits_pi, idx_t)
        )

        bs = len(batch["fen"])
        total_loss += float(loss) * bs
        n += bs

    return {"loss": total_loss / max(1, n)}


# ----------------------------
# Train
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/train/single_gm/train_sft_and_dpo_w_style_sim_utility_weight_maia2.py --gm_name caruana --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic --dpo_loss_weight 0.1
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", type=str, required=True)

    ap.add_argument("--device", type=str, default="cpu")  # "mps" works too if your torch build supports it
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--dpo_loss_weight", type=float, default=0.1)
    ap.add_argument("--style_cp_scale", type=float, default=40)
    ap.add_argument("--style_piece_bonus", type=float, default=1.0)
    ap.add_argument("--style_positional_bonus", type=float, default=2.0)
    ap.add_argument("--style_tau", type=float, default=0.75)

    #
    #cp_scale:        20, 40
    #piece_bonus:     1.0, 1.3
    #positional_bonus:2.0
    #tau:             5, 10, 20

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--train_val_folder", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()

    train_jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_train_dpo.jsonl")
    val_jsonl = Path(f"{args.train_val_folder}/{args.gm_name}_val_dpo.jsonl")
    out_dir = Path(f"{args.out_dir}/{args.gm_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_from_str(args.device)

    # Load Maia-2 base weights twice
    policy = maia_model.from_pretrained(type=args.maia_type, device=str(device))
    policy.train()
    ref = maia_model.from_pretrained(type=args.maia_type, device=str(device))
    ref.eval()

    policy.to(device)
    ref.to(device)
    for p in ref.parameters():
        p.requires_grad_(False)

    # Repo version: prepare() returns [all_moves_dict, elo_dict, all_moves_dict_reversed]
    prep = inference.prepare()
    all_moves_dict, elo_dict, _ = prep

    train_ds = DpoPairs(train_jsonl)
    val_ds = DpoPairs(val_jsonl)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        running = 0.0
        seen = 0

        for batch in train_loader:
            fens = batch["fen"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            meta_list = batch["meta"]
            ply_idxs = [r["ply_idx"] for r in meta_list]
            step += 1

            board_input, legal_moves, es_t, eo_t = batch_preprocess(
                all_moves_dict, elo_dict, batch["fen"], batch["elo_self"], batch["elo_oppo"], device
            )

            logits_pi = forward_logits(policy, board_input, es_t, eo_t)
            with torch.no_grad():
                logits_ref = forward_logits(ref, board_input, es_t, eo_t)

            logits_pi = apply_legal_mask(logits_pi, legal_moves)
            logits_ref = apply_legal_mask(logits_ref, legal_moves)
            idx_t = chosen_index_tensor(batch["fen"], all_moves_dict, batch["chosen"], device)

            logp_pi_ch = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["chosen"], device)
            logp_pi_rj = move_logprob_from_logits(logits_pi, batch["fen"], all_moves_dict, batch["rejected"], device)

            with torch.no_grad():
                logp_ref_ch = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["chosen"], device)
                logp_ref_rj = move_logprob_from_logits(logits_ref, batch["fen"], all_moves_dict, batch["rejected"], device)

            # NEW: chosen/rejected CPs
            chosen_cps = [extract_move_cp(m, ch) for m, ch in zip(meta_list, chosen)]
            rejected_cps = [extract_move_cp(m, rj) for m, rj in zip(meta_list, rejected)]

            # NEW: style similarity scores
            style_scores = torch.tensor(
                [
                    compute_style_score(
                        fen=fen,
                        chosen_uci=ch,
                        rejected_uci=rj,
                        chosen_cp=ch_cp,
                        rejected_cp=rj_cp,
                        ply_idx=ply_idx,
                        phase=None,
                    )
                    for fen, ch, rj, ch_cp, rj_cp, ply_idx in zip(fens, chosen, rejected, chosen_cps, rejected_cps, ply_idxs)
                ],
                dtype=torch.float32,
                device=device,
            )

            loss = (
                args.dpo_loss_weight
                * dpo_loss_style_weighted(
                    logp_pi_ch=logp_pi_ch,
                    logp_pi_rj=logp_pi_rj,
                    logp_ref_ch=logp_ref_ch,
                    logp_ref_rj=logp_ref_rj,
                    style_score=style_scores,
                    beta=args.beta,
                    tau=args.style_tau,
                )
                + supervised_nll_loss(logits_pi, idx_t)
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optim.step()

            bs = len(batch["fen"])
            running += float(loss.detach()) * bs
            seen += bs

            if step % 50 == 0:
                print(f"[epoch {epoch}] step={step} train_sft_and_dpo_w_style_sim_utility_weight_loss={running/max(1,seen):.4f}")

        metrics = evaluate(policy, ref, all_moves_dict, elo_dict, val_loader, device=device, beta=args.beta, dpo_loss_weight=args.dpo_loss_weight,
                           style_cp_scale=args.style_cp_scale, style_piece_bonus=args.style_piece_bonus, style_positional_bonus=args.style_positional_bonus,
                           style_tau=args.style_tau)
        val_loss = metrics["loss"]
        print(f"[epoch {epoch}] val_sft_and_dpo_w_style_sim_utility_weight_loss={val_loss:.4f}")


        ckpt_path = out_dir / f"policy_epoch{epoch}_sft_and_dpo_w_style_sim_utility_weight_beta={args.beta:.2f}_dpo_loss_weight={args.dpo_loss_weight:.2f}_style_cp_scale={args.style_cp_scale:.2f}_style_piece_bonus={args.style_piece_bonus:.2f}_style_positional_bonus={args.style_positional_bonus:.2f}_style_tau={args.style_tau:.2f}.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / f"policy_best_sft_and_dpo_w_style_sim_utility_weight_beta={args.beta:.2f}_dpo_loss_weight={args.dpo_loss_weight:.2f}_style_cp_scale={args.style_cp_scale:.2f}_style_piece_bonus={args.style_piece_bonus:.2f}_style_positional_bonus={args.style_positional_bonus:.2f}_style_tau={args.style_tau:.2f}.pt"
            torch.save(policy.state_dict(), best_path)
            print(f"Saved best: {best_path} (val_sft_and_dpo_w_style_sim_utility_weight_loss={best_val:.4f})")


if __name__ == "__main__":
    main()

