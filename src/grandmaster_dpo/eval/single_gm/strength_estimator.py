#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import chess
import chess.engine
import torch

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move
from concurrent.futures import ProcessPoolExecutor, as_completed
from grandmaster_dpo.tree_search.maia_beam_search_utilities import choose_move_depth_limited


# ----------------------------
# Device / misc helpers
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
    # find keys like ">=2000" and return 2000
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


def clamp_score(s: float, eps: float = 1e-6) -> float:
    return max(eps, min(1.0 - eps, s))


def elo_diff_from_score_simple(score: float) -> float:
    """
    Convert score S (expected points per game) into Elo difference Δ
    using: S = 1 / (1 + 10^(-Δ/400)).
    """
    s = clamp_score(score)
    return 400.0 * math.log10(s / (1.0 - s))


# ----------------------------
# Maia2 move selection (uses maia2.inference.preprocessing which already mirrors black)
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
    Returns a *real-board* UCI move string for the given FEN.

    Important: Maia2 move vocabulary is treated as "white-to-move canonical".
    If it's black to move in `fen`, maia2.preprocessing mirrors internally and
    we mirror the selected canonical move back with mirror_move().
    """
    if tuned_model_depth == 0:
        mx = max_elo_supported(elo_dict)
        es = min(int(elo_self), mx)
        eo = min(int(elo_oppo), mx)

        board_input, es_cat, eo_cat, legal_mask = inference.preprocessing(
            fen, es, eo, elo_dict, all_moves_dict
        )
        board_input = board_input.unsqueeze(0).to(device)            # [1, C, 8, 8]
        legal_mask = legal_mask.unsqueeze(0).to(device)              # [1, V]
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
        uci_real = mirror_move(uci_eff) if side == "b" else uci_eff
        return uci_real
    else:        
        mx = max_elo_supported(elo_dict)
        es = min(int(elo_self), mx)
        eo = min(int(elo_oppo), mx)

        board, _, _, _ = inference.preprocessing(
            fen, es, eo, elo_dict, all_moves_dict
        )
        return choose_move_depth_limited(policy, (all_moves_dict, elo_dict, all_moves_rev), board, 2800, 2800, depth=tuned_model_depth, beam=4)[0]


# ----------------------------
# Stockfish wrapper
# ----------------------------


def make_stockfish(stockfish_path: str, threads: int, hash_mb: int) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        eng.configure({"Threads": int(threads), "Hash": int(hash_mb)})
    except Exception:
        pass
    return eng



def stockfish_pick_move(
    eng: chess.engine.SimpleEngine,
    board: chess.Board,
    movetime_ms: Optional[int],
    depth: Optional[int],
) -> chess.Move:
    if movetime_ms is not None and movetime_ms > 0:
        limit = chess.engine.Limit(time=movetime_ms / 1000.0)
    elif depth is not None and depth > 0:
        limit = chess.engine.Limit(depth=depth)
    else:
        limit = chess.engine.Limit(time=0.05)

    res = eng.play(board, limit)
    return res.move


# ----------------------------
# Match runner
# ----------------------------

@dataclass
class AggWDL:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(1, self.games)


def play_one_game(
    policy: torch.nn.Module,
    eng: chess.engine.SimpleEngine,
    model_plays_white: bool,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    all_moves_rev: Dict[int, str],
    device: torch.device,
    elo_self: int,
    elo_oppo: int,
    movetime_ms: Optional[int],
    depth: Optional[int],
    max_plies: int,
    temperature: float,
    top_k: int,
    tuned_model_depth: int = 0,
) -> float:
    """
    Returns game score from model perspective: 1.0 win, 0.5 draw, 0.0 loss
    """
    board = chess.Board()
    policy.eval()

    for _ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        model_to_move = (board.turn == chess.WHITE and model_plays_white) or (board.turn == chess.BLACK and not model_plays_white)

        if model_to_move:
            fen = board.fen()
            uci = maia2_pick_move_uci(
                policy=policy,
                fen=fen,
                elo_self=elo_self,
                elo_oppo=elo_oppo,
                all_moves_dict=all_moves_dict,
                elo_dict=elo_dict,
                all_moves_rev=all_moves_rev,
                device=device,
                temperature=temperature,
                top_k=top_k,
                tuned_model_depth=tuned_model_depth,
            )
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                print(f"Illegal move: {move} on board: {board.fen()}")
                move = random.choice(list(board.legal_moves))
            board.push(move)
        else:
            move = stockfish_pick_move(eng, board, movetime_ms=movetime_ms, depth=depth)
            if move is None or move not in board.legal_moves:
                print(f"Illegal move: {move} on board: {board.fen()}")
                move = random.choice(list(board.legal_moves))
            board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.5

    res = outcome.result()
    if res == "1/2-1/2":
        return 0.5

    model_won = (res == "1-0" and model_plays_white) or (res == "0-1" and not model_plays_white)
    return 1.0 if model_won else 0.0


# ----------------------------
# Model loading
# ----------------------------

def load_policy(
    maia_type: str,
    device: torch.device,
    ckpt_path: Optional[Path],
) -> torch.nn.Module:
    """
    If ckpt_path is None: returns base Maia2.
    Else: loads Maia2 then overlays ckpt weights.
    """
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
# Parallel worker: play chunk of games at fixed Stockfish depth
# ----------------------------

def _depth_chunk_worker(
    *,
    depth: int,
    n_games: int,
    ckpt: Optional[str],         # None => base
    maia_type: str,
    device_str: str,
    stockfish_path: str,
    threads: int,
    hash_mb: int,
    max_plies: int,
    fixed_elo: int,
    temperature: float,
    top_k: int,
    seed: int,
    chunk_id: int,
    tuned_model_depth: int = 0,
) -> Tuple[int, int, int, int]:
    """
    Plays n_games vs Stockfish at fixed depth.
    Returns (depth, wins, draws, losses) from model perspective.
    Loads model + starts Stockfish once per chunk.
    """
    set_seed(seed + chunk_id * 10007 + depth * 97)
    device = device_from_str(device_str)

    all_moves_dict, elo_dict, all_moves_rev = inference.prepare()
    ckpt_path = Path(ckpt) if ckpt is not None else None
    policy = load_policy(maia_type, device, ckpt_path)

    eng = make_stockfish(stockfish_path, threads=threads, hash_mb=hash_mb)

    wins = draws = losses = 0
    try:
        for g in range(n_games):
            model_white = ((chunk_id + g) % 2 == 0)  # alternate colors
            s = play_one_game(
                policy=policy,
                eng=eng,
                model_plays_white=model_white,
                all_moves_dict=all_moves_dict,
                elo_dict=elo_dict,
                all_moves_rev=all_moves_rev,
                device=device,
                elo_self=fixed_elo,
                elo_oppo=fixed_elo,
                movetime_ms=None,
                depth=depth,
                max_plies=max_plies,
                temperature=temperature,
                top_k=top_k,
                tuned_model_depth=tuned_model_depth,
            )
            if s == 1.0:
                wins += 1
            elif s == 0.5:
                draws += 1
            else:
                losses += 1
        return depth, wins, draws, losses
    finally:
        try:
            eng.quit()
        except Exception:
            pass


def run_vs_stockfish_depths(
    *,
    ckpt: Optional[str],  # None => base
    maia_type: str,
    device_str: str,
    stockfish_path: str,
    threads: int,
    hash_mb: int,
    max_plies: int,
    fixed_elo: int,
    temperature: float,
    top_k: int,
    seed: int,
    depths: List[int],
    games_per_depth: int,
    chunk_size: int,
    max_workers: int,
    label: str,
    tuned_model_depth: int = 0,
) -> Dict[int, AggWDL]:
    chunks_per_depth = math.ceil(games_per_depth / chunk_size)
    agg: Dict[int, AggWDL] = {d: AggWDL() for d in depths}

    total_tasks = len(depths) * chunks_per_depth
    print(f"\n=== Running {label} vs Stockfish depths ===")
    print(f"depths={depths}, games_per_depth={games_per_depth}, chunk_size={chunk_size}, total_tasks={total_tasks}, max_workers={max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        chunk_id = 0
        for d in depths:
            for k in range(chunks_per_depth):
                n = min(chunk_size, games_per_depth - (k * chunk_size))
                futs.append(ex.submit(
                    _depth_chunk_worker,
                    depth=d,
                    n_games=n,
                    ckpt=ckpt,
                    maia_type=maia_type,
                    device_str=device_str,
                    stockfish_path=stockfish_path,
                    threads=threads,
                    hash_mb=hash_mb,
                    max_plies=max_plies,
                    fixed_elo=fixed_elo,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                    chunk_id=chunk_id,
                    tuned_model_depth=tuned_model_depth,
                ))
                chunk_id += 1

        for f in as_completed(futs):
            d, w, dr, l = f.result()
            a = agg[d]
            a.wins += w
            a.draws += dr
            a.losses += l
            print(f"[DONE depth={d}] games={a.games:3d} W/D/L={a.wins:3d}/{a.draws:3d}/{a.losses:3d} score={a.score:.3f}")

    return agg


# ----------------------------
# Calibration + Elo estimation
# ----------------------------

def fit_stockfish_depth_to_elo_diff_log(
    base_results: Dict[int, AggWDL],
) -> Tuple[float, float]:
    xs: List[float] = [] # This will be log(depth)
    ys: List[float] = [] # This remains z
    ws: List[float] = []

    for d in sorted(base_results.keys()):
        a = base_results[d]
        if a.games <= 0 or a.score <= 0.01: # Ignore 0% scores, they break the slope
            continue
        z = elo_diff_from_score_simple(a.score)
        xs.append(math.log(float(d))) # <--- Key change
        ys.append(float(z))
        ws.append(float(a.games))

    if len(xs) < 2:
        raise ValueError("Need at least 2 depths with games to calibrate.")

    # Weighted least squares for y = alpha x + beta
    # alpha = cov_w(x,y)/var_w(x)
    wsum = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / wsum
    ybar = sum(w * y for w, y in zip(ws, ys)) / wsum

    num = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    den = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if den <= 1e-12:
        raise ValueError("Degenerate calibration: depths have no variance.")

    alpha = num / den
    beta = ybar - alpha * xbar
    return alpha, beta


def estimate_model_elo_from_results(
    model_results: Dict[int, AggWDL],
    *,
    alpha: float,
    beta: float,
    base_elo: float,
    min_score: float = 0.02,   # drop depths where model is basically shut out
    max_score: float = 0.98,   # symmetric: also drop near-certain wins
) -> Tuple[float, Dict[int, float]]:
    """
    Uses calibration z_base(log depth) = alpha*log(depth) + beta = (base_elo - sf_elo(depth)),
    so sf_elo(depth) = base_elo - z_base(depth).

    For each depth, model score gives z_model(depth) = (model_elo - sf_elo(depth)),
    so model_elo(depth) = sf_elo(depth) + z_model(depth).

    Returns (weighted average Elo, per-depth Elo estimates).
    """
    per_depth: Dict[int, float] = {}
    weights: List[float] = []
    elos: List[float] = []
    print(f"Using depths with score in [0.02, 0.98] for Elo aggregation.")

    for d in sorted(model_results.keys()):
        a = model_results[d]
        if a.games <= 0:
            continue

        # 1) Skip saturated points where Elo-from-score is numerically/ statistically meaningless
        if a.score <= min_score or a.score >= max_score:
            # Optional debug:
            # print(f"[SKIP depth={d}] score={a.score:.3f} games={a.games}")
            
            continue

        # Calibration is in log(depth)
        z_base = alpha * math.log(float(d)) + beta   # base - sf(depth)
        sf_elo_d = float(base_elo) - z_base          # sf(depth)

        z_model = elo_diff_from_score_simple(a.score)  # model - sf(depth)
        model_elo_d = sf_elo_d + z_model

        per_depth[d] = model_elo_d

        # 2) Weight by information ~ n * p * (1-p)
        w = float(a.games) * float(a.score) * float(1.0 - a.score)
        weights.append(w)
        elos.append(model_elo_d)

    if not elos:
        raise ValueError(
            "No usable depths after filtering. "
            "Lower min_score/max_score or include weaker Stockfish depths."
        )

    wsum = sum(weights)
    elo_hat = sum(w * e for w, e in zip(weights, elos)) / wsum
    return elo_hat, per_depth



# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    # Example usage: python ./src/grandmaster_dpo/eval/single_gm/strength_estimator.py --maia_type blitz --device cpu --stockfish_path /usr/local/bin/stockfish --sf_elo 2000 --threads 1 --hash_mb 256 --max_plies 400 --temperature 0.1 --top_k 5 --seed 0 --gm_name firouzja
    ap = argparse.ArgumentParser(
        description=(
            "Calibrate Stockfish depth->Elo using base Maia2 (fixed Elo conditioning), "
            "then estimate Elo of fine-tuned checkpoints by playing vs Stockfish at those depths."
        )
    )
    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", type=str, default="cpu", help="cpu|mps|cuda")

    ap.add_argument("--gm_name", type=str, required=True, help="Grandmaster name (used to find ckpts under processed/single_gm/train_val/{gm_name}/).")
    ap.add_argument("--depth_level", type=int, default=0, help="The depth of tree search to use on the fine-tuned models.")

    ap.add_argument("--stockfish_path", type=str, required=True, help="Path to Stockfish binary (UCI).")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash_mb", type=int, default=256)

    ap.add_argument("--max_plies", type=int, default=400, help="Adjudicate as draw if exceeded.")

    ap.add_argument("--fixed_maia_elo", type=int, default=2000, help="Force Maia2 conditioning Elo (both self and opponent) to this value for ALL games.")
    ap.add_argument("--base_elo_value", type=float, default=2000.0, help="The Elo value you want to assign to *base Maia2* at fixed_maia_elo (anchor for calibration).")

    ap.add_argument("--temperature", type=float, default=0.0, help="0=greedy, >0 sampling temperature for Maia2.")
    ap.add_argument("--top_k", type=int, default=0, help="If sampling, restrict Maia2 to top-k moves before sampling.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--depths", type=str, default="1,3,5,7,9,11,13", help="Comma-separated Stockfish depths to test.")
    ap.add_argument("--games_per_depth", type=int, default=100)
    ap.add_argument("--chunk_size", type=int, default=10)
    ap.add_argument("--max_workers", type=int, default=0, help="0=auto (min(cpu_count,8)).")

    args = ap.parse_args()

    set_seed(args.seed)

    depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
    if not depths:
        raise ValueError("No depths provided.")

    max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else min(os.cpu_count() or 2, 8)

    # Check checkpoints
    dpo_model_path = Path(f"./processed/single_gm/train_val/{args.gm_name}/policy_best.pt")
    sft_model_path = Path(f"./processed/single_gm/train_val/{args.gm_name}/policy_sft_best.pt")
    pairwise_model_path = Path(f"./processed/single_gm/train_val/{args.gm_name}/policy_pairwise_sft_best.pt")

    ckpts: List[Tuple[Optional[str], str]] = [
        (None, "base"),  # for completeness (we'll still run calibrated vs SF in the same framework if you want)
        (str(dpo_model_path), "dpo"),
        (str(sft_model_path), "sft"),
        (str(pairwise_model_path), "pairwise_sft"),
    ]

    for p, name in ckpts:
        if p is not None and not Path(p).exists():
            raise FileNotFoundError(f"{name} model not found: {p}")

    # 1) Calibration: run BASE Maia2 vs Stockfish(depths)
    base_results = run_vs_stockfish_depths(
        ckpt=None,
        maia_type=args.maia_type,
        device_str=args.device,
        stockfish_path=args.stockfish_path,
        threads=args.threads,
        hash_mb=args.hash_mb,
        max_plies=args.max_plies,
        fixed_elo=args.fixed_maia_elo,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        depths=depths,
        games_per_depth=args.games_per_depth,
        chunk_size=args.chunk_size,
        max_workers=max_workers,
        label=f"BASE(maia_type={args.maia_type},fixed_elo={args.fixed_maia_elo})",
    )

    alpha, beta = fit_stockfish_depth_to_elo_diff_log(base_results)
    print("\n=== Calibration fit (base - stockfish(depth)) ===")
    print(f"z_base(depth) = alpha*log(depth) + beta")
    print(f"alpha={alpha:.4f}, beta={beta:.4f}")
    print(f"Stockfish Elo(depth) = base_elo_value - z_base(depth)  (base_elo_value={args.base_elo_value})")

    # Helpful printout: implied SF Elo per depth
    print("\nImplied Stockfish Elo by depth (anchored on base_elo_value):")
    for d in depths:
        z = alpha * math.log(d) + beta
        sf_elo_d = args.base_elo_value - z
        print(f"  depth={d:2d}: sf_elo≈{sf_elo_d:8.1f}   (z_base={z:7.1f})")

    # 2) Evaluate each model vs Stockfish(depths), then convert to Elo using the calibration
    print("\n=== Estimating Elo for models (anchored to base calibration) ===")
    for ckpt_path, model_type in ckpts:
        if model_type == "base":
            model_results = base_results
        else:
            model_results = run_vs_stockfish_depths(
                ckpt=ckpt_path,
                maia_type=args.maia_type,
                device_str=args.device,
                stockfish_path=args.stockfish_path,
                threads=args.threads,
                hash_mb=args.hash_mb,
                max_plies=args.max_plies,
                fixed_elo=args.fixed_maia_elo,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
                depths=depths,
                games_per_depth=args.games_per_depth,
                chunk_size=args.chunk_size,
                max_workers=max_workers,
                label=f"{model_type.upper()}(fixed_elo={args.fixed_maia_elo})",
                tuned_model_depth=args.depth_level,
            )

        elo_hat, per_depth = estimate_model_elo_from_results(
            model_results,
            alpha=alpha,
            beta=beta,
            base_elo=args.base_elo_value,
        )

        print(f"\n--- {model_type} ---")
        if ckpt_path is not None:
            print(f"ckpt: {ckpt_path}")
        print(f"Estimated Elo: {elo_hat:.1f}  (anchored: base_elo_value={args.base_elo_value})")
        with open(f"./processed/single_gm/train_val/validation_results/{args.gm_name}/strength_estimator_{model_type}_elo_hat.txt", "w") as f:
            f.write(f"Estimated Elo: {elo_hat:.1f}  (anchored: base_elo_value={args.base_elo_value})")

        
        print("Per-depth estimates:")
        import csv
        with open(f"./processed/single_gm/train_val/validation_results/{args.gm_name}/strength_estimator_{model_type}_depth_{args.depth_level}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["depth", "elo_estimate", "score", "games", "used_in_fit", "wins", "draws", "losses"])

            for d in depths:
                a = model_results[d]
                if d in per_depth:
                    elo_val = per_depth[d]
                    used = 1
                    print(f"  depth={d:2d}: elo≈{elo_val:8.1f}   score={a.score:.3f}  games={a.games}")
                else:
                    elo_val = ""   # or None
                    used = 0
                    print(f"  depth={d:2d}: SKIPPED (score={a.score:.3f}, games={a.games})")

                writer.writerow([d, elo_val, a.score, a.games, used, a.wins, a.draws, a.losses])


        

    print("\nDone.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
