#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import chess
import chess.engine
import torch

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move
from grandmaster_dpo.tree_search.maia_beam_search_utilities import choose_move_depth_limited


# ----------------------------
# Helpers
# ----------------------------

def device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s in ("mps",):
        return torch.device("mps")
    return torch.device(s)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def max_elo_supported(elo_dict: dict) -> int:
    import re
    mx = None
    for k in elo_dict.keys():
        m = re.match(r"^>=\s*(\d+)$", k)
        if m:
            mx = max(mx or 0, int(m.group(1)))
    return mx if mx is not None else 3000


def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))


# ----------------------------
# Maia2 move selection
# ----------------------------

@torch.no_grad()
def maia2_pick_move_uci(
    policy: torch.nn.Module,
    fen: str,
    elo_self: int,
    elo_oppo: int,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    all_moves_rev: Dict[int, str],
    device: torch.device,
    temperature: float = 0.0,
    top_k: int = 0,
    tuned_model_depth: int = 0,
) -> str:
    """
    Returns real-board UCI string. Maia2 vocabulary is white-canonical.
    preprocessing mirrors black internally; we mirror the chosen move back.
    """
    mx = max_elo_supported(elo_dict)
    es = min(int(elo_self), mx)
    eo = min(int(elo_oppo), mx)

    if tuned_model_depth and tuned_model_depth > 0:
        board, _, _, _ = inference.preprocessing(fen, es, eo, elo_dict, all_moves_dict)
        return choose_move_depth_limited(
            policy,
            (all_moves_dict, elo_dict, all_moves_rev),
            board,
            es,
            eo,
            depth=tuned_model_depth,
            beam=4,
        )[0]

    board_input, es_cat, eo_cat, legal_mask = inference.preprocessing(
        fen, es, eo, elo_dict, all_moves_dict
    )
    board_input = board_input.unsqueeze(0).to(device)
    legal_mask = legal_mask.unsqueeze(0).to(device)
    es_t = torch.tensor([int(es_cat)], device=device).long()
    eo_t = torch.tensor([int(eo_cat)], device=device).long()

    logits, _, _ = policy(board_input, es_t, eo_t)               # [1, V]
    logits = apply_legal_mask(logits, legal_mask).squeeze(0)     # [V]

    if temperature is None or temperature <= 0.0:
        idx = int(torch.argmax(logits).item())
    else:
        if top_k and top_k > 0:
            vals, inds = torch.topk(logits, k=min(top_k, logits.numel()))
            probs = torch.softmax(vals / temperature, dim=-1)
            pick = int(torch.multinomial(probs, num_samples=1).item())
            idx = int(inds[pick].item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx = int(torch.multinomial(probs, num_samples=1).item())

    uci_eff = all_moves_rev.get(idx)
    if uci_eff is None:
        legal_idxs = torch.nonzero(legal_mask.squeeze(0) > 0, as_tuple=False).view(-1).tolist()
        if not legal_idxs:
            raise RuntimeError("No legal moves according to Maia2 legal mask.")
        idx = int(random.choice(legal_idxs))
        uci_eff = all_moves_rev[int(idx)]

    side = fen.split(" ")[1]
    return mirror_move(uci_eff) if side == "b" else uci_eff


# ----------------------------
# Stockfish (UCI_Elo strength limit)
# ----------------------------

def make_stockfish_limited(
    stockfish_path: str,
    *,
    threads: int,
    hash_mb: int,
    limit_strength: bool,
    uci_elo: Optional[int],
) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    cfg: Dict[str, object] = {
        "Threads": int(threads),
        "Hash": int(hash_mb),
    }
    # Strength limit knobs (supported by Stockfish)
    cfg["UCI_LimitStrength"] = bool(limit_strength)
    if limit_strength and uci_elo is not None:
        cfg["UCI_Elo"] = int(uci_elo)
    try:
        eng.configure(cfg)
    except Exception:
        # Some builds/options may differ; we still try to run with clocks.
        pass
    return eng


def stockfish_pick_move_3p2(
    eng: chess.engine.SimpleEngine,
    board: chess.Board,
    w_clock: float,
    b_clock: float,
    inc: float,
) -> chess.Move:
    limit = chess.engine.Limit(
        white_clock=max(0.001, float(w_clock)),
        black_clock=max(0.001, float(b_clock)),
        white_inc=float(inc),
        black_inc=float(inc),
    )
    res = eng.play(board, limit)
    return res.move


# ----------------------------
# Player definitions
# ----------------------------

@dataclass(frozen=True)
class PlayerSpec:
    key: str                  # unique identifier in tables/csv
    kind: str                 # "sf_uci" or "maia"
    param: str                # e.g. "uci=2600" or "dpo"
    sf_uci_elo: Optional[int] = None
    ckpt: Optional[str] = None
    tuned_model_depth: int = 0


# ----------------------------
# Model loading
# ----------------------------

def load_policy(
    maia_type: str,
    device: torch.device,
    ckpt_path: Optional[Path],
) -> torch.nn.Module:
    policy = maia_model.from_pretrained(type=maia_type, device=str(device))

    if ckpt_path is not None:
        sd = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
            sd = sd["model_state_dict"]
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        missing, unexpected = policy.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARN] unexpected keys: {len(unexpected)}")

    policy.to(device)
    policy.eval()
    return policy


# ----------------------------
# 3+2 game runner
# ----------------------------

def play_one_game_3p2(
    *,
    white: PlayerSpec,
    black: PlayerSpec,
    policy_cache: Dict[str, torch.nn.Module],
    device: torch.device,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    all_moves_rev: Dict[int, str],
    stockfish_path: str,
    threads: int,
    hash_mb: int,
    fixed_maia_elo: int,
    temperature: float,
    top_k: int,
    max_plies: int,
    seed: int,
) -> Tuple[float, str]:
    """
    Returns (score_for_white, result_string).
    Enforces 3+2 by tracking clocks, subtracting wall time for Maia inference.
    """
    random.seed(seed)
    board = chess.Board()

    start_time = 180.0
    inc = 2.0
    w_clock = start_time
    b_clock = start_time

    eng_w = None
    eng_b = None

    try:
        if white.kind == "sf_uci":
            eng_w = make_stockfish_limited(
                stockfish_path,
                threads=threads,
                hash_mb=hash_mb,
                limit_strength=True,
                uci_elo=white.sf_uci_elo,
            )
        if black.kind == "sf_uci":
            eng_b = make_stockfish_limited(
                stockfish_path,
                threads=threads,
                hash_mb=hash_mb,
                limit_strength=True,
                uci_elo=black.sf_uci_elo,
            )

        for _ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break

            if board.turn == chess.WHITE:
                # White to move
                if white.kind == "sf_uci":
                    t0 = time.perf_counter()
                    mv = stockfish_pick_move_3p2(eng_w, board, w_clock, b_clock, inc)
                    dt = time.perf_counter() - t0
                    w_clock -= dt
                    if w_clock <= 0:
                        return 0.0, "0-1(time)"
                    w_clock += inc
                else:
                    policy = policy_cache[white.key]
                    t0 = time.perf_counter()
                    uci = maia2_pick_move_uci(
                        policy=policy,
                        fen=board.fen(),
                        elo_self=fixed_maia_elo,
                        elo_oppo=fixed_maia_elo,
                        all_moves_dict=all_moves_dict,
                        elo_dict=elo_dict,
                        all_moves_rev=all_moves_rev,
                        device=device,
                        temperature=temperature,
                        top_k=top_k,
                        tuned_model_depth=white.tuned_model_depth,
                    )
                    dt = time.perf_counter() - t0
                    w_clock -= dt
                    if w_clock <= 0:
                        return 0.0, "0-1(time)"
                    w_clock += inc
                    mv = chess.Move.from_uci(uci)

                if mv not in board.legal_moves:
                    mv = random.choice(list(board.legal_moves))
                board.push(mv)

            else:
                # Black to move
                if black.kind == "sf_uci":
                    t0 = time.perf_counter()
                    mv = stockfish_pick_move_3p2(eng_b, board, w_clock, b_clock, inc)
                    dt = time.perf_counter() - t0
                    b_clock -= dt
                    if b_clock <= 0:
                        return 1.0, "1-0(time)"
                    b_clock += inc
                else:
                    policy = policy_cache[black.key]
                    t0 = time.perf_counter()
                    uci = maia2_pick_move_uci(
                        policy=policy,
                        fen=board.fen(),
                        elo_self=fixed_maia_elo,
                        elo_oppo=fixed_maia_elo,
                        all_moves_dict=all_moves_dict,
                        elo_dict=elo_dict,
                        all_moves_rev=all_moves_rev,
                        device=device,
                        temperature=temperature,
                        top_k=top_k,
                        tuned_model_depth=black.tuned_model_depth,
                    )
                    dt = time.perf_counter() - t0
                    b_clock -= dt
                    if b_clock <= 0:
                        return 1.0, "1-0(time)"
                    b_clock += inc
                    mv = chess.Move.from_uci(uci)

                if mv not in board.legal_moves:
                    mv = random.choice(list(board.legal_moves))
                board.push(mv)

        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return 0.5, "1/2-1/2"

        res = outcome.result()
        if res == "1-0":
            return 1.0, "1-0"
        if res == "0-1":
            return 0.0, "0-1"
        return 0.5, "1/2-1/2"

    finally:
        for eng in (eng_w, eng_b):
            if eng is not None:
                try:
                    eng.quit()
                except Exception:
                    pass


# ----------------------------
# Elo fit (Bradley–Terry / Elo logistic MLE)
# ----------------------------

def fit_elos_mle(
    players: List[str],
    games: List[Tuple[str, str, float]],
    *,
    anchor_player: str,
    anchor_elo: float,
    iters: int = 3500,
    lr: float = 2.0,
) -> Dict[str, float]:
    """
    Fit Elo ratings via logistic MLE.
    games: (white_key, black_key, score_white) where score_white in {1, 0.5, 0}.
    Ratings are identifiable only up to additive constant; we fix anchor_player to anchor_elo.
    """
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    r = [0.0] * n  # relative ratings
    k = math.log(10.0) / 400.0
    anchor_i = idx[anchor_player]

    step = lr
    for _ in range(iters):
        grad = [0.0] * n

        for w, b, s in games:
            iw = idx[w]
            ib = idx[b]
            x = (r[ib] - r[iw]) / 400.0
            p = 1.0 / (1.0 + (10.0 ** x))
            g = (s - p) * k
            grad[iw] += g
            grad[ib] -= g

        grad[anchor_i] = 0.0

        for i in range(n):
            if i == anchor_i:
                continue
            r[i] += step * grad[i]

        step *= 0.9995

    shift = anchor_elo - r[anchor_i]
    return {p: (r[idx[p]] + shift) for p in players}


# ----------------------------
# Pool runner (single-process recommended)
# ----------------------------

def run_pool(
    *,
    player_specs: List[PlayerSpec],
    maia_type: str,
    device_str: str,
    stockfish_path: str,
    threads: int,
    hash_mb: int,
    fixed_maia_elo: int,
    temperature: float,
    top_k: int,
    max_plies: int,
    games_per_pair: int,
    seed: int,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, int, int, int]]]:
    """
    Returns:
      - games list: (white_key, black_key, score_white)
      - pair summary rows: (a, b, games, a_wins, draws, a_losses) where a<b lexicographically
    """
    device = device_from_str(device_str)
    all_moves_dict, elo_dict, all_moves_rev = inference.prepare()

    specs_by_key = {p.key: p for p in player_specs}
    keys = [p.key for p in player_specs]

    # Load Maia policies once (single-process caching)
    policy_cache: Dict[str, torch.nn.Module] = {}
    for ps in player_specs:
        if ps.kind == "maia":
            ckpt_path = Path(ps.ckpt) if ps.ckpt is not None else None
            policy_cache[ps.key] = load_policy(maia_type, device, ckpt_path)

    def add_pair_result(pair_stats: Dict[Tuple[str, str], List[int]], w: str, b: str, s_white: float) -> None:
        a, c = (w, b) if w < b else (b, w)
        if (a, c) not in pair_stats:
            # [a_wins, draws, a_losses]
            pair_stats[(a, c)] = [0, 0, 0]
        st = pair_stats[(a, c)]
        # convert to a-perspective
        if a == w:
            s_a = s_white
        else:
            s_a = 1.0 - s_white
        if s_a == 1.0:
            st[0] += 1
        elif s_a == 0.5:
            st[1] += 1
        else:
            st[2] += 1

    all_games: List[Tuple[str, str, float]] = []
    pair_stats: Dict[Tuple[str, str], List[int]] = {}

    pair_list: List[Tuple[str, str]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pair_list.append((keys[i], keys[j]))

    for pi, (a_key, b_key) in enumerate(pair_list):
        pair_seed = seed + 100000 * pi
        for g in range(games_per_pair):
            # Alternate colors
            if g % 2 == 0:
                white_key, black_key = a_key, b_key
            else:
                white_key, black_key = b_key, a_key

            score_w, _res = play_one_game_3p2(
                white=specs_by_key[white_key],
                black=specs_by_key[black_key],
                policy_cache=policy_cache,
                device=device,
                all_moves_dict=all_moves_dict,
                elo_dict=elo_dict,
                all_moves_rev=all_moves_rev,
                stockfish_path=stockfish_path,
                threads=threads,
                hash_mb=hash_mb,
                fixed_maia_elo=fixed_maia_elo,
                temperature=temperature,
                top_k=top_k,
                max_plies=max_plies,
                seed=pair_seed + g * 7919,
            )
            all_games.append((white_key, black_key, float(score_w)))
            add_pair_result(pair_stats, white_key, black_key, float(score_w))

        print(f"[PAIR {pi+1}/{len(pair_list)}] {a_key} vs {b_key} finished ({games_per_pair} games)")

    pair_rows: List[Tuple[str, str, int, int, int, int]] = []
    for (a, b), (aw, dr, al) in sorted(pair_stats.items()):
        pair_rows.append((a, b, aw + dr + al, aw, dr, al))

    return all_games, pair_rows


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pool Elo estimator: Stockfish(UCI_Elo) + Maia models under 3+2, fitted jointly."
    )
    ap.add_argument("--gm_name", type=str, required=True)

    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", type=str, default="cpu", help="cpu|mps|cuda")

    ap.add_argument("--stockfish_path", type=str, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash_mb", type=int, default=256)

    ap.add_argument("--fixed_maia_elo", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)

    ap.add_argument("--max_plies", type=int, default=400)

    ap.add_argument("--sf_elos", type=str, default="1600,1800,2000,2200,2400,2600,2800,3000",
                    help="Comma-separated Stockfish UCI_Elo levels to include in pool.")
    ap.add_argument("--games_per_pair", type=int, default=40, help="Games per unordered pairing; colors alternate.")
    ap.add_argument("--seed", type=int, default=0)

    # Maia tree-search depth (optional)
    ap.add_argument("--depth_level", type=int, default=0, help="Tree-search depth for tuned models (0 disables).")

    # Elo anchoring: pick which player to pin to a value (avoid 'Maia is 2k' optics by anchoring a Stockfish level)
    ap.add_argument("--anchor_key", type=str, default="sf_uci_2000",
                    help="Player key to anchor (e.g., sf_uci_2000 or maia_base).")
    ap.add_argument("--anchor_elo_value", type=float, default=2000.0, help="Elo to assign to anchor_key on output scale.")

    args = ap.parse_args()
    set_seed(args.seed)

    sf_elos = [int(x.strip()) for x in args.sf_elos.split(",") if x.strip()]
    if not sf_elos:
        raise ValueError("No --sf_elos provided.")

    # Check checkpoints for GM
    gm_dir = Path(f"./processed/single_gm/train_val/{args.gm_name}")
    dpo_model_path = gm_dir / "policy_best.pt"
    sft_model_path = gm_dir / "policy_sft_best.pt"
    pairwise_model_path = gm_dir / "policy_pairwise_sft_best.pt"

    for p in (dpo_model_path, sft_model_path, pairwise_model_path):
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")

    # Build player pool
    player_specs: List[PlayerSpec] = []
    for e in sf_elos:
        player_specs.append(PlayerSpec(
            key=f"sf_uci_{e}",
            kind="sf_uci",
            param=f"uci={e}",
            sf_uci_elo=e,
        ))

    player_specs.extend([
        PlayerSpec(key="maia_base", kind="maia", param="base", ckpt=None, tuned_model_depth=0),
        PlayerSpec(key="maia_dpo", kind="maia", param="dpo", ckpt=str(dpo_model_path), tuned_model_depth=args.depth_level),
        PlayerSpec(key="maia_sft", kind="maia", param="sft", ckpt=str(sft_model_path), tuned_model_depth=args.depth_level),
        PlayerSpec(key="maia_pairwise_sft", kind="maia", param="pairwise_sft", ckpt=str(pairwise_model_path), tuned_model_depth=args.depth_level),
    ])

    keys = [p.key for p in player_specs]
    if args.anchor_key not in keys:
        raise ValueError(f"--anchor_key={args.anchor_key} not found in pool keys: {keys}")

    # Run all pairings (single-process recommended for caching)
    games, pair_rows = run_pool(
        player_specs=player_specs,
        maia_type=args.maia_type,
        device_str=args.device,
        stockfish_path=args.stockfish_path,
        threads=args.threads,
        hash_mb=args.hash_mb,
        fixed_maia_elo=args.fixed_maia_elo,
        temperature=args.temperature,
        top_k=args.top_k,
        max_plies=args.max_plies,
        games_per_pair=args.games_per_pair,
        seed=args.seed,
    )

    # Fit Elo jointly and anchor
    players = [p.key for p in player_specs]
    elos = fit_elos_mle(
        players=players,
        games=games,
        anchor_player=args.anchor_key,
        anchor_elo=args.anchor_elo_value,
        iters=3500,
        lr=2.0,
    )

    # Write CSV outputs
    out_dir = Path(f"./processed/single_gm/train_val/validation_results/{args.gm_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"pool_elos_uci_3p2_depth{args.depth_level}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gm_name", "player_key", "kind", "param", "elo", "anchor_key", "anchor_elo"])
        for ps in sorted(player_specs, key=lambda p: elos[p.key], reverse=True):
            w.writerow([args.gm_name, ps.key, ps.kind, ps.param, f"{elos[ps.key]:.1f}", args.anchor_key, f"{args.anchor_elo_value:.1f}"])

    pair_csv = out_dir / f"pool_pairs_uci_3p2_depth{args.depth_level}.csv"
    with open(pair_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "games", "a_wins", "draws", "a_losses", "a_score"])
        for a, b, g, aw, dr, al in pair_rows:
            a_score = (aw + 0.5 * dr) / max(1, g)
            w.writerow([a, b, g, aw, dr, al, f"{a_score:.3f}"])

    print("\n=== Elo results (joint-fit, anchored) ===")
    print(f"Anchor: {args.anchor_key} = {args.anchor_elo_value:.1f}")
    for ps in sorted(player_specs, key=lambda p: elos[p.key], reverse=True):
        print(f"{ps.key:20s} {ps.kind:7s} {ps.param:12s} elo={elos[ps.key]:7.1f}")

    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {pair_csv}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
