import math
import chess
import chess.engine
import torch
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from maia2.utils import mirror_move


def mirror_uci_like_board_mirror(uci: str) -> str:
    """Mirror a UCI move the same way chess.Board(...).mirror() changes coordinates."""
    mv = chess.Move.from_uci(uci)
    f = chess.square_mirror(mv.from_square)
    t = chess.square_mirror(mv.to_square)
    return chess.Move(f, t, promotion=mv.promotion).uci()

def uci_to_vocab_index(all_moves_dict: Dict[str, int], fen: str, uci: str) -> int:
    side = fen.split(" ")[1]
    uci_eff = mirror_uci_like_board_mirror(uci) if side == "b" else uci
    return int(all_moves_dict.get(uci_eff, -1))

def vocab_index_to_uci(all_moves: List[str], fen: str, idx: int) -> str:
    """
    Convert a vocab index back to a real UCI move for this position.
    Handles Maia's mirroring convention for black-to-move.
    """
    if idx < 0 or idx >= len(all_moves):
        return ""

    uci_eff = all_moves[idx]  # Maia vocab is in "white perspective"
    side = fen.split(" ")[1]  # 'w' or 'b'
    return mirror_move(uci_eff) if side == "b" else uci_eff

# ----------------------------
# Stockfish rerank helpers
# ----------------------------

def _stable_sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def make_stockfish(
    stockfish_path: str,
    *,
    threads: int = 1,
    hash_mb: int = 256,
    uci_elo: Optional[int] = None,   # if supported by your SF build
    skill_level: Optional[int] = None, # if supported (0..20 on many builds)
    timeout: float = 30.0,
) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout=timeout)

    opts: Dict[str, Any] = {}
    # common options
    if "Threads" in eng.options:
        opts["Threads"] = int(threads)
    if "Hash" in eng.options:
        opts["Hash"] = int(hash_mb)

    # Elo limiting (only if your Stockfish supports it)
    if uci_elo is not None and "UCI_LimitStrength" in eng.options and "UCI_Elo" in eng.options:
        opts["UCI_LimitStrength"] = True
        opts["UCI_Elo"] = int(uci_elo)

    # Skill Level (alternative weakening mechanism)
    if skill_level is not None and "Skill Level" in eng.options:
        opts["Skill Level"] = int(skill_level)

    if opts:
        eng.configure(opts)
    return eng


def _score_to_cp(score: chess.engine.PovScore, mate_score: int = 100_000) -> int:
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
        m = rel.mate()
        if m is not None:
            return mate_score if m > 0 else -mate_score
        return 0
    return int(cp)

@torch.no_grad()
def choose_move_sf_topk_biased_by_policy(
    *,
    fen: str,
    logits_pi_masked: torch.Tensor,     # [V] legal-masked (illegal == -inf)
    all_moves_dict: Dict[str, int],
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    multipv: int = 10,
    restrict_cp_window: Optional[int] = 60,  # keep moves with cp >= best_cp - window
    temperature: float = 1.0,                # policy sampling temperature within candidate set
    sample: bool = True,                      # True for sampling, False for argmax (deterministic eval)
) -> Tuple[str, str, np.ndarray, List[Tuple[str, int]], int, int, float]:
    """
    Returns:
      (fen, uci_selected, cand_probs, cands[(uci,cp)], cp_selected, best_cp, entropy)

    Process:
      1) Stockfish MultiPV -> candidate moves + cp
      2) filter by cp-gap
      3) sample using Maia policy probs restricted to remaining candidates
    """
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=True):
        return fen, "", np.zeros((0,), dtype=np.float64), [], 0, 0, 0.0

    infos = engine.analyse(board, limit, multipv=int(multipv))
    cands: List[Tuple[str, int]] = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        uci = pv[0].uci()
        cp = _score_to_cp(score)
        cands.append((uci, cp))

    if not cands:
        idx = int(torch.argmax(logits_pi_masked).item())
        uci_fb = vocab_index_to_uci(all_moves_dict, fen, idx)
        return fen, uci_fb, np.array([1.0]), [(uci_fb, 0)]

    best_cp = max(cp for _, cp in cands)

    # filter to near-best to avoid blunders
    if restrict_cp_window is not None:
        w = int(restrict_cp_window)
        filt = [(m, cp) for (m, cp) in cands if cp >= best_cp - w]
        if filt:
            cands = filt

    # policy probs restricted to candidate set
    t = max(float(temperature), 1e-6)
    logp_all = torch.log_softmax(logits_pi_masked / t, dim=-1)  # [V]

    cand_logps = []
    for (uci, _cp) in cands:
        idx = uci_to_vocab_index(all_moves_dict, fen, uci)  # your helper (handles black mirror)
        if idx < 0:
            cand_logps.append(torch.tensor(-1e9, device=logp_all.device, dtype=logp_all.dtype))
        else:
            cand_logps.append(logp_all[idx])

    cand_logps_t = torch.stack(cand_logps, dim=0)  # [K]
    cand_probs_t = torch.softmax(cand_logps_t, dim=0)  # sums to 1 over candidates

    if sample:
        sel_i = int(torch.multinomial(cand_probs_t, 1).item())
    else:
        sel_i = int(torch.argmax(cand_probs_t).item())

    uci_selected, cp_selected = cands[sel_i]

    probs_np = cand_probs_t.detach().cpu().numpy()

    # entropy (0 when argmax selection is used)
    eps = 1e-12
    ent = float(-(probs_np * np.log(probs_np + eps)).sum()) if probs_np.size else 0.0

    return fen, uci_selected, probs_np, cands, int(cp_selected), int(best_cp), ent

@torch.no_grad()
def batch_choose_moves_sf_topk_biased_by_policy(
    *,
    fens: List[str],
    logits_pi_masked: torch.Tensor,     # [B, V]
    all_moves_dict: Dict[str, int],
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    multipv: int = 10,
    restrict_cp_window: Optional[int] = 60,
    temperature: float = 1.0,
    sample: bool = True,
) -> List[Tuple[str, str, np.ndarray, List[Tuple[str, int]], int, int, float]]:
    print("batch_choose_moves_sf_topk_biased_by_policy")
    out = []
    for i, fen in enumerate(fens):
        out.append(
            choose_move_sf_topk_biased_by_policy(
                fen=fen,
                logits_pi_masked=logits_pi_masked[i],
                all_moves_dict=all_moves_dict,
                engine=engine,
                limit=limit,
                multipv=multipv,
                restrict_cp_window=restrict_cp_window,
                temperature=temperature,
                sample=sample,
            )
        )
    return out
