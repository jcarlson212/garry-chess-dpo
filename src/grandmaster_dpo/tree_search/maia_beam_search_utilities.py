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


def choose_move_depth_limited_ab(
    policy,
    prepared,
    board: chess.Board,
    elo_self: int,
    elo_oppo: int,
    *,
    depth: int = 4,
    beam_root: int = 12,
    beam_sched=(12, 8, 6, 4),   # per ply beams (depth=4 uses 4 plies)
    inference=None,
    tt_max: int = 200_000,
):
    assert inference is not None
    es, eo = int(elo_self), int(elo_oppo)

    # Cache inference per fen: returns (top_items, win_prob)
    @lru_cache(maxsize=tt_max)
    def infer_cached(fen: str, beam: int):
        move_probs, wp = inference.inference_each(policy, prepared, fen, es, eo)
        items = heapq.nlargest(beam, move_probs.items(), key=lambda kv: kv[1])
        wpv = 0.5 if wp is None else float(wp)
        return items, wpv

    # Cache backed-up value: (fen, d, alpha_bucket, beta_bucket, beam) is too big, so cache just (fen,d,turn,beam)
    @lru_cache(maxsize=tt_max)
    def minimax(fen: str, d: int, ply: int) -> float:
        b = chess.Board(fen)

        if b.is_game_over():
            res = b.result()
            if res == "1-0": return 1.0
            if res == "0-1": return 0.0
            return 0.5
        if d == 0:
            # leaf eval = model value
            _items, wp = infer_cached(fen, 1)  # beam irrelevant for leaf
            return wp

        beam = beam_sched[min(ply, len(beam_sched) - 1)]
        items, wp_here = infer_cached(fen, beam)

        # We'll do alpha-beta *outside* caching to keep cache keys small.
        # This function is used by ab() below.
        # Return a tuple would help, but keep float.
        # wp_here is a good fallback if no moves can be pushed.
        if b.turn == chess.WHITE:
            best = -1.0
            for uci, _p in items:
                mv = chess.Move.from_uci(uci)
                try:
                    b.push(mv)
                except Exception:
                    continue
                v = ab(b, d - 1, ply + 1, -1.0, 2.0)
                b.pop()
                if v > best:
                    best = v
                    if best >= 1.0:
                        break
            return best if best >= 0 else wp_here
        else:
            best = 2.0
            for uci, _p in items:
                mv = chess.Move.from_uci(uci)
                try:
                    b.push(mv)
                except Exception:
                    continue
                v = ab(b, d - 1, ply + 1, -1.0, 2.0)
                b.pop()
                if v < best:
                    best = v
                    if best <= 0.0:
                        break
            return best if best <= 1 else wp_here

    def ab(b: chess.Board, d: int, ply: int, alpha: float, beta: float) -> float:
        fen = b.fen()
        if d == 0 or b.is_game_over():
            return minimax(fen, 0, ply)

        beam = beam_sched[min(ply, len(beam_sched) - 1)]
        items, wp_here = infer_cached(fen, beam)

        if b.turn == chess.WHITE:
            v = -1.0
            for uci, _p in items:
                mv = chess.Move.from_uci(uci)
                try:
                    b.push(mv)
                except Exception:
                    continue
                v = max(v, ab(b, d - 1, ply + 1, alpha, beta))
                b.pop()
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v if v >= 0 else wp_here
        else:
            v = 2.0
            for uci, _p in items:
                mv = chess.Move.from_uci(uci)
                try:
                    b.push(mv)
                except Exception:
                    continue
                v = min(v, ab(b, d - 1, ply + 1, alpha, beta))
                b.pop()
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v if v <= 1 else wp_here

    # root
    root_fen = board.fen()
    
    root_items, root_wp = infer_cached(root_fen, beam_root)

    scored = []
    for uci, p in root_items:
        mv = chess.Move.from_uci(uci)
        try:
            board.push(mv)
        except Exception:
            continue
        v = ab(board, depth - 1, 1, -1.0, 2.0)
        board.pop()
        scored.append((uci, float(p), float(v)))

    if not scored:
        # fallback: best by policy
        best_uci = max(root_items, key=lambda kv: kv[1])[0]
        return best_uci, "fallback: no scored moves"

    if board.turn == chess.WHITE:
        best_uci, _, _ = max(scored, key=lambda t: t[2])
        scored_sorted = sorted(scored, key=lambda t: t[2], reverse=True)
    else:
        best_uci, _, _ = min(scored, key=lambda t: t[2])
        scored_sorted = sorted(scored, key=lambda t: t[2])

    lines = [
        f"depth={depth} beam_root={beam_root} beam_sched={beam_sched} (alpha-beta + cached)",
        f"root win_prob (white POV): {root_wp:.4f}",
        "",
        "uci     p(policy)   v(lookahead)",
    ]
    for uci, p, v in scored_sorted[:8]:
        lines.append(f"{uci:6s}  {p:8.4f}    {v:8.4f}")

    return best_uci, "\n".join(lines)
