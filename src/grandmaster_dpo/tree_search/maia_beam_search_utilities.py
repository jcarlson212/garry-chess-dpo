import math
import chess
import heapq
from functools import lru_cache

def choose_move_depth_limited(policy, prepared, board: chess.Board, elo_self: int, elo_oppo: int,
                              depth: int = 4, beam: int = 12, inference=None) -> tuple[str, str]:
    """
    Returns (best_uci, candidates_text) using depth-limited lookahead.
    Uses Maia's move_probs as transition weights and win_prob as leaf eval.
    win_prob is assumed to be "white POV" in [0,1].
    """

    def leaf_eval(b: chess.Board) -> float:
        # Terminal handling
        if b.is_game_over():
            res = b.result()
            if res == "1-0": return 1.0
            if res == "0-1": return 0.0
            return 0.5

        _, wp = inference.inference_each(policy, prepared, b.fen(), int(elo_self), int(elo_oppo))
        if wp is None:
            return 0.5
        return float(wp)

    def recurse(b: chess.Board, d: int) -> float:
        if d == 0 or b.is_game_over():
            return leaf_eval(b)

        move_probs, wp = inference.inference_each(policy, prepared, b.fen(), int(elo_self), int(elo_oppo))
        # Take top beam moves for tractability
        items = sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:beam]

        # Minimax over white POV value
        if b.turn == chess.WHITE:
            best = -1.0
            for uci, _p in items:
                mv = chess.Move.from_uci(uci)
                if mv not in b.legal_moves:
                    continue
                b.push(mv)
                v = recurse(b, d - 1)
                b.pop()
                best = max(best, v)
            return best if best >= 0 else leaf_eval(b)
        else:
            best = 2.0
            for uci, _p in items:
                mv = chess.Move.from_uci(uci)
                if mv not in b.legal_moves:
                    continue
                b.push(mv)
                v = recurse(b, d - 1)
                b.pop()
                best = min(best, v)
            return best if best <= 1 else leaf_eval(b)

    # root: score candidate moves
    root_probs, root_wp = inference.inference_each(policy, prepared, board.fen(), int(elo_self), int(elo_oppo))
    root_items = sorted(root_probs.items(), key=lambda kv: kv[1], reverse=True)[:beam]

    scored = []
    for uci, p in root_items:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            continue
        board.push(mv)
        v = recurse(board, depth - 1)
        board.pop()
        scored.append((uci, float(p), float(v)))

    if not scored:
        # fallback: argmax prob
        print(f"fallback: no scored moves... might be a bug in the tree search")
        best = max(root_probs.items(), key=lambda kv: kv[1])[0]
        return best, "fallback: no scored moves"

    # pick best by minimax-backed value (white POV)
    if board.turn == chess.WHITE:
        best_uci, _, _ = max(scored, key=lambda t: t[2])
    else:
        best_uci, _, _ = min(scored, key=lambda t: t[2])

    # side panel
    lines = []
    lines.append(f"depth={depth} beam={beam}")
    lines.append(f"root win_prob (white POV): {float(root_wp):.4f}" if root_wp is not None else "root win_prob: None")
    lines.append("")
    lines.append("uci     p(policy)   v(lookahead)")
    for uci, p, v in sorted(scored, key=lambda t: t[2], reverse=(board.turn==chess.WHITE))[:8]:
        lines.append(f"{uci:6s}  {p:8.4f}    {v:8.4f}")

    return best_uci, "\n".join(lines)

def choose_move_depth_limited_fast(
    policy,
    prepared,
    board: chess.Board,
    elo_self: int,
    elo_oppo: int,
    *,
    depth: int = 4,
    beam: int = 12,
    inference=None,
    tt_max: int = 200_000,   # cap caches
) -> tuple[str, str]:
    """
    Faster depth-limited minimax using:
      - caching (transposition) keyed by (fen, depth)
      - caching of inference_each keyed by fen
      - heapq.nlargest for top-beam extraction
    Assumes win_prob is white POV in [0,1].
    """

    assert inference is not None, "pass inference=maia2.inference module (or object with inference_each)"

    es = int(elo_self)
    eo = int(elo_oppo)

    # --- Inference cache: fen -> (top_items, win_prob, legal_set)
    # Store ONLY top 'beam' items to save memory/time.
    @lru_cache(maxsize=tt_max)
    def infer_top(fen: str):
        move_probs, win_prob = inference.inference_each(policy, prepared, fen, es, eo)

        # keep top beam without sorting whole dict
        items = heapq.nlargest(beam, move_probs.items(), key=lambda kv: kv[1])

        # also keep a set of UCIs we consider (for sanity / quick checks)
        return items, (0.5 if win_prob is None else float(win_prob))

    # --- Value cache: (fen, depth_remaining, turn) -> backed-up value
    @lru_cache(maxsize=tt_max)
    def recurse_fen(fen: str, d: int) -> float:
        b = chess.Board(fen)

        # terminal or leaf
        if b.is_game_over():
            res = b.result()
            if res == "1-0": return 1.0
            if res == "0-1": return 0.0
            return 0.5
        if d == 0:
            # leaf eval uses model value head (win_prob)
            _items, wp = infer_top(fen)
            return wp

        items, _wp = infer_top(fen)

        # minimax in white POV
        if b.turn == chess.WHITE:
            best = -1.0
            for uci, _p in items:
                mv = chess.Move.from_uci(uci)
                # mv should be legal by construction; avoid expensive membership checks
                if mv not in b.legal_moves:
                    continue
                b.push(mv)
                v = recurse_fen(b.fen(), d - 1)
                b.pop()
                if v > best:
                    best = v
                    # optional pruning: can't beat 1.0
                    if best >= 1.0:
                        break
            return best if best >= 0 else _wp
        else:
            best = 2.0
            for uci, _p in items:
                mv = chess.Move.from_uci(uci)
                if mv not in b.legal_moves:
                    continue
                b.push(mv)
                v = recurse_fen(b.fen(), d - 1)
                b.pop()
                if v < best:
                    best = v
                    if best <= 0.0:
                        break
            return best if best <= 1 else _wp

    # --- Root scoring
    root_fen = board.fen()
    root_items, root_wp = infer_top(root_fen)

    scored = []
    for uci, p in root_items:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            continue
        board.push(mv)
        v = recurse_fen(board.fen(), depth - 1)
        board.pop()
        scored.append((uci, float(p), float(v)))

    if not scored:
        best = max(dict(root_items).items(), key=lambda kv: kv[1])[0] if root_items else None
        if best is None:
            # absolute fallback: first legal move
            best = next(iter(board.legal_moves)).uci()
        return best, "fallback: no scored moves"

    # choose by minimax-backed value
    if board.turn == chess.WHITE:
        best_uci, _, _ = max(scored, key=lambda t: t[2])
        scored_sorted = sorted(scored, key=lambda t: t[2], reverse=True)
    else:
        best_uci, _, _ = min(scored, key=lambda t: t[2])
        scored_sorted = sorted(scored, key=lambda t: t[2], reverse=False)

    # panel
    lines = [
        f"depth={depth} beam={beam} (cached)",
        f"root win_prob (white POV): {root_wp:.4f}",
        "",
        "uci     p(policy)   v(lookahead)",
    ]
    for uci, p, v in scored_sorted[:8]:
        lines.append(f"{uci:6s}  {p:8.4f}    {v:8.4f}")

    return best_uci, "\n".join(lines)

def legal_uci_set(b: chess.Board) -> set[str]:
    return {m.uci() for m in b.legal_moves}

import math
import chess

def choose_move_depth_limited_ab(
    policy,
    prepared,
    board: chess.Board,
    elo_self: int,
    elo_oppo: int,
    depth: int = 4,
    beam_root: int = 12,
    beam_sched: tuple[int, ...] = (12, 8, 6, 4),
    inference=None,
    style_lambda: float = 0.05,
) -> tuple[str, str]:
    """
    Returns (best_uci, candidates_text) using depth-limited minimax with alpha-beta pruning.

    - Uses Maia's move_probs to ORDER moves (best-first) and to TRUNCATE branching (beam).
    - Uses Maia's win_prob as evaluation at leaves.
    - win_prob is assumed to be "white POV" in [0,1].

    beam_root controls number of root candidates shown/considered.
    beam_sched controls branching factor by ply under the root, e.g. (12,8,6,4).
      If depth exceeds len(beam_sched), the last value is reused.
    """

    if inference is None:
        raise ValueError("inference must be provided and expose inference_each(...)")

    # --- small caches to avoid repeated model calls within one search ---
    _infer_cache: dict[str, tuple[dict[str, float], float | None]] = {}
    _eval_cache: dict[str, float] = {}

    def style_bonus(p: float, lam: float, eps: float = 1e-12) -> float:
        return lam * math.log(max(p, eps))

    def _cache_key(b: chess.Board) -> str:
        # include side-to-move etc via full FEN; elos are fixed per call
        return b.fen()

    def infer_once(b: chess.Board) -> tuple[dict[str, float], float | None]:
        k = _cache_key(b)
        if k in _infer_cache:
            return _infer_cache[k]
        move_probs, wp = inference.inference_each(policy, prepared, k, int(elo_self), int(elo_oppo))
        if move_probs is None:
            move_probs = {}
        _infer_cache[k] = (move_probs, wp)
        return _infer_cache[k]

    def leaf_eval(b: chess.Board) -> float:
        k = _cache_key(b)
        if k in _eval_cache:
            return _eval_cache[k]

        # Terminal handling
        if b.is_game_over():
            res = b.result()
            v = 1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5
            _eval_cache[k] = v
            return v

        _move_probs, wp = infer_once(b)
        v = 0.5 if wp is None else float(wp)
        _eval_cache[k] = v
        return v

    def beam_for_ply(ply_from_root: int) -> int:
        # ply_from_root = 0 at root children, 1 for grandchildren, ...
        if not beam_sched:
            return beam_root
        idx = min(ply_from_root, len(beam_sched) - 1)
        return int(beam_sched[idx])

    def ordered_moves(b: chess.Board, k: int) -> list[tuple[chess.Move, float]]:
        move_probs, _wp = infer_once(b)
        if not move_probs:
            return []

        # Order by policy probability desc, then validate legality.
        # Keep top-k *after* sorting; legality filtering happens as we iterate.
        items = sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)

        out: list[tuple[chess.Move, float]] = []
        for uci, p in items:
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                continue
            if mv in b.legal_moves:
                out.append((mv, float(p)))
                if len(out) >= k:
                    break
        return out

    def alphabeta(b: chess.Board, d: int, alpha: float, beta: float, ply_from_root: int) -> float:
        if d == 0 or b.is_game_over():
            return leaf_eval(b)

        k = beam_for_ply(ply_from_root)
        moves = ordered_moves(b, k)

        # If the policy produced no legal moves (or empty dict), fall back to leaf eval.
        if not moves:
            return leaf_eval(b)

        if b.turn == chess.WHITE:
            v = -math.inf
            for mv, _p in moves:
                b.push(mv)
                child = alphabeta(b, d - 1, alpha, beta, ply_from_root + 1)
                b.pop()
                child = child + style_bonus(p, style_lambda)

                if child > v:
                    v = child
                if v > alpha:
                    alpha = v
                if alpha >= beta:
                    break  # beta cut
            return v if v != -math.inf else leaf_eval(b)
        else:
            v = math.inf
            for mv, _p in moves:
                b.push(mv)
                child = alphabeta(b, d - 1, alpha, beta, ply_from_root + 1)
                b.pop()
                child = child + style_bonus(p, style_lambda)

                if child < v:
                    v = child
                if v < beta:
                    beta = v
                if alpha >= beta:
                    break  # alpha cut
            return v if v != math.inf else leaf_eval(b)

    # --- root scoring ---
    root_probs, root_wp = infer_once(board)
    # root candidates ordered by policy prob, truncated to beam_root
    root_items = []
    for uci, p in sorted(root_probs.items(), key=lambda kv: kv[1], reverse=True):
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            continue
        if mv in board.legal_moves:
            root_items.append((uci, mv, float(p)))
            if len(root_items) >= int(beam_root):
                break

    scored: list[tuple[str, float, float]] = []
    for uci, mv, p in root_items:
        board.push(mv)
        v = alphabeta(board, depth - 1, alpha=-math.inf, beta=math.inf, ply_from_root=0)
        board.pop()
        scored.append((uci, p, float(v)))

    if not scored:
        # fallback: argmax prob among whatever we got (even if illegal filter removed all)
        if root_probs:
            best = max(root_probs.items(), key=lambda kv: kv[1])[0]
            return best, "fallback: no scored legal moves"
        # absolute last resort
        return "0000", "fallback: no policy moves available"

    # pick best by minimax value (white POV)
    if board.turn == chess.WHITE:
        best_uci, _, _ = max(scored, key=lambda t: t[2])
    else:
        best_uci, _, _ = min(scored, key=lambda t: t[2])

    # side panel
    lines = []
    lines.append(f"depth={depth} beam_root={beam_root} beam_sched={beam_sched}")
    lines.append(f"root win_prob (white POV): {float(root_wp):.4f}" if root_wp is not None else "root win_prob: None")
    lines.append("")
    lines.append("uci     p(policy)   v(ab)")
    # show top 8 sorted by side-to-move preference
    for uci, p, v in sorted(scored, key=lambda t: t[2], reverse=(board.turn == chess.WHITE))[:8]:
        lines.append(f"{uci:6s}  {p:8.4f}    {v:8.4f}")

    return best_uci, "\n".join(lines)
