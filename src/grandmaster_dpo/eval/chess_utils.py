import chess
from typing import Dict, List, Optional, Tuple

from maia2.utils import mirror_move
import torch
from maia2 import inference

def fen_after_move(fen: str, uci: str) -> Optional[str]:
    """
    Applies a legal UCI move to fen and returns resulting fen.
    Returns None on failure.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise ValueError("Engine proposed move was not legal somehow")
    board.push(move)
    return board.fen()

def fen_to_ply_abs(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return 2 * (fullmove - 1) + (1 if side == "b" else 0)

def ply_to_phase(ply_abs: int) -> str:
    if ply_abs < 20:
        return "opening"
    if ply_abs < 60:
        return "middlegame"
    return "endgame"

def coarse_opening_family_from_prefix(prefix_uci: List[str]) -> str:
    if len(prefix_uci) < 2:
        return "Unknown"
    black_reply = prefix_uci[1]
    if black_reply == "c7c5":
        return "Sicilian"
    if black_reply == "e7e5":
        return "Open Game (1...e5)"
    if black_reply == "c7c6":
        return "Caro-Kann"
    if black_reply == "e7e6":
        return "French / e6"
    if black_reply == "d7d5":
        return "Queen's Pawn (1...d5)"
    if black_reply == "g8f6":
        return "Indian Defense (1...Nf6)"
    return "Other"

def mirror_uci_like_board_mirror(uci: str) -> str:
    mv = chess.Move.from_uci(uci)
    f = chess.square_mirror(mv.from_square)
    t = chess.square_mirror(mv.to_square)
    return chess.Move(f, t, promotion=mv.promotion).uci()

def uci_to_vocab_index(all_moves_dict: Dict[str, int], fen: str, uci: str) -> int:
    side = fen.split(" ")[1]
    uci_eff = mirror_uci_like_board_mirror(uci) if side == "b" else uci
    return int(all_moves_dict.get(uci_eff, -1))

def vocab_index_to_uci(all_moves: List[str], fen: str, idx: int) -> str:
    if idx < 0 or idx >= len(all_moves):
        return ""
    uci_eff = all_moves[idx]          # Maia vocab is white-perspective
    side = fen.split(" ")[1]
    return mirror_move(uci_eff) if side == "b" else uci_eff

def extract_move_cp(meta: dict, uci: str) -> float:
    sf_moves = meta["stockfish"]["sf_moves_returned"]
    for sf_uci, cp in sf_moves:
        if sf_uci == uci:
            return float(cp)
        
    cp_values = [cp for _, cp in sf_moves]
    fallback_cp = float(min(cp_values)) if cp_values else 0.0
    #print(
        #f"[WARN] move {uci} not found in sf_moves_returned "
        #f"(game={meta.get('game_header_hash')}, ply={meta.get('ply_idx')}). "
        #f"Using fallback cp={fallback_cp}"
    #)
    return fallback_cp

def batch_preprocess(
    *,
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

    for fen, es, eo in zip(fens, elo_self, elo_oppo):
        es, eo = 2000, 2000
        bi, es_cat, eo_cat, lm = inference.preprocessing(
            fen, int(es), int(eo), elo_dict, all_moves_dict
        )
        board_inputs.append(bi)
        legal_moves.append(lm)
        elo_self_cats.append(int(es_cat))
        elo_oppo_cats.append(int(eo_cat))

    board_input = torch.stack(board_inputs, dim=0).to(device)
    legal_moves_t = torch.stack(legal_moves, dim=0).to(device)
    elo_self_t = torch.tensor(elo_self_cats, device=device).long()
    elo_oppo_t = torch.tensor(elo_oppo_cats, device=device).long()
    return board_input, legal_moves_t, elo_self_t, elo_oppo_t

def forward_logits(m: torch.nn.Module, board_input: torch.Tensor, es: torch.Tensor, eo: torch.Tensor) -> torch.Tensor:
    logits, _, _ = m(board_input, es, eo)
    return logits