#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import chess
import chess.engine
import chess.pgn
import torch

from maia2 import inference, model as maia_model
from maia2.utils import mirror_move
from grandmaster_dpo.tree_search.maia_beam_search_utilities import choose_move_depth_limited


# ============================================================
# Player spec
# ============================================================

@dataclass(frozen=True)
class PlayerSpec:
    key: str
    kind: str                 # "sf" or "maia"
    param: str
    ckpt: Optional[str] = None
    tuned_model_depth: int = 0

    # Stockfish compute knobs (used in play() limits)
    sf_depth: Optional[int] = None
    sf_nodes: Optional[int] = None

    # Optional Stockfish strength knobs (used via engine.configure)
    sf_skill: Optional[int] = None             # often 0..20, if supported
    sf_limit_strength: Optional[bool] = None   # enables UCI_Elo, if supported
    sf_uci_elo: Optional[int] = None           # if supported


# ============================================================
# Global worker state
# ============================================================

G: Dict[str, Any] = {}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s in ("cpu",):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda")
    if s == "mps":
        return torch.device("mps")
    return torch.device(s)


def _max_elo_supported(elo_dict: dict) -> int:
    # Maia2 elo_dict keys sometimes include strings like ">=2000"
    import re
    mx = None
    for k in elo_dict.keys():
        m = re.match(r"^>=\s*(\d+)$", k)
        if m:
            mx = max(mx or 0, int(m.group(1)))
    return mx if mx is not None else 3000


def _apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))


@torch.no_grad()
def _maia2_pick_move_uci(
    policy: torch.nn.Module,
    fen: str,
    elo_self: int,
    elo_oppo: int,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    all_moves_rev: Dict[int, str],
    device: torch.device,
    temperature: float,
    top_k: int,
    tuned_model_depth: int,
) -> str:
    """
    If tuned_model_depth > 0: use depth-limited Maia beam search (deterministic-ish).
    Else: sample/greedy from the Maia policy logits (temperature/top_k controlled).
    """
    mx = _max_elo_supported(elo_dict)
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

    logits, _, _ = policy(board_input, es_t, eo_t)
    logits = _apply_legal_mask(logits, legal_mask).squeeze(0)

    if temperature <= 0.0:
        idx = int(torch.argmax(logits).item())
    else:
        if top_k and top_k > 0:
            vals, inds = torch.topk(logits, k=min(top_k, logits.numel()))
            probs = torch.softmax(vals / temperature, dim=-1)
            pick = int(torch.multinomial(probs, 1).item())
            idx = int(inds[pick].item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx = int(torch.multinomial(probs, 1).item())

    uci_eff = all_moves_rev.get(idx)
    if uci_eff is None:
        legal_idxs = torch.nonzero(legal_mask.squeeze(0) > 0, as_tuple=False).view(-1).tolist()
        idx = int(random.choice(legal_idxs))
        uci_eff = all_moves_rev[int(idx)]

    side = fen.split(" ")[1]
    return mirror_move(uci_eff) if side == "b" else uci_eff


def _make_stockfish(stockfish_path: str, cfg: Dict[str, Any], *, timeout: float) -> chess.engine.SimpleEngine:
    # Increase timeout to avoid UCI handshake timeouts under spawn/load.
    eng = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout=timeout)
    try:
        filtered = {k: v for k, v in cfg.items() if k in eng.options}
        if filtered:
            eng.configure(filtered)
    except Exception as e:
        print(f"[PID={os.getpid()}] Stockfish configure error: {e}")
    return eng


def _stockfish_pick_move_fixed(
    eng: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    depth: Optional[int],
    nodes: Optional[int],
) -> chess.Move:
    if nodes is not None and nodes > 0:
        limit = chess.engine.Limit(nodes=int(nodes))
    else:
        limit = chess.engine.Limit(depth=int(depth))
    return eng.play(board, limit).move


def _load_policy(maia_type: str, device: torch.device, ckpt_path: Optional[Path]) -> torch.nn.Module:
    policy = maia_model.from_pretrained(type=maia_type, device=str(device))
    if ckpt_path is not None:
        sd = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
            sd = sd["model_state_dict"]
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        policy.load_state_dict(sd, strict=False)
    policy.to(device)
    policy.eval()
    return policy


def _sf_target_cfg_for(ps: PlayerSpec) -> Dict[str, Any]:
    # Only include strength knobs here; threads/hash are fixed per worker for stability.
    cfg: Dict[str, Any] = {}
    if ps.sf_skill is not None:
        cfg["Skill Level"] = int(ps.sf_skill)
    if ps.sf_limit_strength is not None:
        cfg["UCI_LimitStrength"] = bool(ps.sf_limit_strength)
    if ps.sf_uci_elo is not None:
        cfg["UCI_Elo"] = int(ps.sf_uci_elo)
    return cfg


def _ensure_sf_cfg(ps: PlayerSpec) -> None:
    """
    Reconfigure the single worker Stockfish engine only if this PlayerSpec
    requests different strength knobs than current.
    """
    eng: chess.engine.SimpleEngine = G["sf_engine"]
    desired = _sf_target_cfg_for(ps)

    cur: Dict[str, Any] = G.get("sf_current_cfg", {})
    if desired == cur:
        return

    try:
        filtered = {k: v for k, v in desired.items() if k in eng.options}
        # If engine doesn't support these options, this becomes empty and we treat as "configured".
        if filtered:
            eng.configure(filtered)
    except Exception as e:
        print(f"[PID={os.getpid()}] Stockfish configure error (dynamic): {e}")

    # Record what we *attempted* to set (not necessarily all supported).
    G["sf_current_cfg"] = desired


def _worker_init(state: Dict[str, Any]) -> None:
    G.clear()
    G.update(state)

    _set_seed(G["seed"] + 99991 * (os.getpid() % 9973))
    device = _device_from_str(G["device_str"])
    G["device"] = device

    all_moves_dict, elo_dict, all_moves_rev = inference.prepare()
    G["all_moves_dict"] = all_moves_dict
    G["elo_dict"] = elo_dict
    G["all_moves_rev"] = all_moves_rev

    # Maia policies (once per worker)
    policies: Dict[str, torch.nn.Module] = {}
    for ps in G["player_specs"]:
        if ps.kind == "maia":
            ckpt_path = Path(ps.ckpt) if ps.ckpt is not None else None
            policies[ps.key] = _load_policy(G["maia_type"], device, ckpt_path)
    G["policies"] = policies

    # ONE Stockfish engine per worker
    base_cfg: Dict[str, Any] = {}
    # Keep behavior stable/reproducible:
    base_cfg["Threads"] = int(G["threads"])
    base_cfg["Hash"] = int(G["hash_mb"])

    G["sf_engine"] = _make_stockfish(
        G["stockfish_path"],
        base_cfg,
        timeout=float(G["sf_init_timeout"]),
    )
    G["sf_current_cfg"] = {}  # no strength knobs set initially


# ============================================================
# Game logic (NO TIME LIMITS)
# ============================================================

def _play_one_game_fixed_compute(white: PlayerSpec, black: PlayerSpec, seed: int) -> Tuple[float, str, str]:
    """
    Returns: (score_white in {0,0.5,1}, result_str like "1-0"/"0-1"/"1/2-1/2", pgn_text)
    """
    rng = random.Random(seed)
    board = chess.Board()

    max_plies: int = G["max_plies"]

    game = chess.pgn.Game()
    game.headers["Event"] = "local-gauntlet"
    game.headers["White"] = white.key
    game.headers["Black"] = black.key
    node = game

    for _ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        if board.turn == chess.WHITE:
            mv = _pick_move_for(board, white, opp=black, rng=rng)
        else:
            mv = _pick_move_for(board, black, opp=white, rng=rng)

        if mv not in board.legal_moves:
            mv = rng.choice(list(board.legal_moves))

        board.push(mv)
        node = node.add_variation(mv)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        score_w = 0.5
        res = "1/2-1/2"
    else:
        res = outcome.result()
        if res == "1-0":
            score_w = 1.0
        elif res == "0-1":
            score_w = 0.0
        else:
            score_w = 0.5

    game.headers["Result"] = res
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    pgn_text = game.accept(exporter)

    return float(score_w), res, pgn_text


def _pick_move_for(board: chess.Board, me: PlayerSpec, opp: PlayerSpec, rng: random.Random) -> chess.Move:
    if me.kind == "sf":
        # Reconfigure strength knobs if this spec needs it (still only one engine process).
        _ensure_sf_cfg(me)
        eng = G["sf_engine"]
        mv = _stockfish_pick_move_fixed(
            eng,
            board,
            depth=me.sf_depth,
            nodes=me.sf_nodes,
        )
        return mv

    # Maia
    policy = G["policies"][me.key]
    uci = _maia2_pick_move_uci(
        policy=policy,
        fen=board.fen(),
        elo_self=G["fixed_maia_elo"],
        elo_oppo=G["fixed_maia_elo"],
        all_moves_dict=G["all_moves_dict"],
        elo_dict=G["elo_dict"],
        all_moves_rev=G["all_moves_rev"],
        device=G["device"],
        temperature=G["temperature"],
        top_k=G["top_k"],
        tuned_model_depth=me.tuned_model_depth,
    )
    mv = chess.Move.from_uci(uci)
    return mv


# ============================================================
# Gauntlet scheduling
# ============================================================

def _build_gauntlet_tasks(
    *,
    models: List[str],
    anchors: List[str],
    games_per_match: int,
    seed: int,
) -> List[Tuple[int, str, str, int]]:
    if games_per_match <= 0:
        raise ValueError("games_per_match must be > 0")

    tasks: List[Tuple[int, str, str, int]] = []
    tid = 0
    for m in models:
        for a in anchors:
            tasks.append((tid, m, a, games_per_match))
            tid += 1

    rng = random.Random(seed + 424242)
    rng.shuffle(tasks)
    return tasks


def _run_gauntlet_task(task: Tuple[int, str, str, int]) -> List[Tuple[str, str, float, str, str]]:
    task_id, model_key, anchor_key, n_games = task
    specs: Dict[str, PlayerSpec] = G["specs_by_key"]

    out: List[Tuple[str, str, float, str, str]] = []
    base_seed = G["seed"] + 1000003 * task_id

    for g in range(n_games):
        if g % 2 == 0:
            wk, bk = model_key, anchor_key
        else:
            wk, bk = anchor_key, model_key

        print(f"Game {g} of {n_games} starting. White: {wk}, Black: {bk}")

        score_w, res, pgn_text = _play_one_game_fixed_compute(
            specs[wk], specs[bk], seed=base_seed + 7919 * g
        )
        out.append((wk, bk, score_w, res, pgn_text))
        print(f"Game {g} of {n_games} finished. White: {wk}, Black: {bk}, Score: {score_w}")
    return out


# ============================================================
# Optional quick Elo fit (anchored logistic MLE)
# ============================================================

def fit_elos_torch(
    players: List[str],
    games: List[Tuple[str, str, float]],
    *,
    anchor_key: str,
    anchor_elo: float,
    steps: int,
    lr: float,
    l2: float,
    device: str,
    init_elos: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    aidx = idx[anchor_key]

    wi = torch.tensor([idx[w] for (w, b, s) in games], dtype=torch.long, device=device)
    bi = torch.tensor([idx[b] for (w, b, s) in games], dtype=torch.long, device=device)
    y = torch.tensor([s for (w, b, s) in games], dtype=torch.float32, device=device)

    delta = torch.nn.Parameter(torch.zeros(n, dtype=torch.float32, device=device))
    anchor_mask = torch.ones(n, device=device, dtype=torch.float32)
    anchor_mask[aidx] = 0.0

    if init_elos is not None:
        with torch.no_grad():
            for p, e in init_elos.items():
                if p not in idx or p == anchor_key:
                    continue
                delta[idx[p]] = float(e) - float(anchor_elo)
            delta[aidx] = 0.0

    k = math.log(10.0) / 400.0
    opt = torch.optim.Adam([delta], lr=lr)

    eps = 1e-8
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        delta_eff = delta * anchor_mask
        r = delta_eff + float(anchor_elo)

        logits = (r[wi] - r[bi]) * k
        p = torch.sigmoid(logits)

        loss = -(y * torch.log(p + eps) + (1.0 - y) * torch.log(1.0 - p + eps)).mean()
        loss = loss + l2 * (delta_eff * delta_eff).mean()
        loss.backward()
        if delta.grad is not None:
            delta.grad[aidx] = 0.0
        opt.step()
        with torch.no_grad():
            delta[aidx] = 0.0

    out = {p: float(anchor_elo + float((delta.detach() * anchor_mask)[idx[p]].cpu())) for p in players}
    out[anchor_key] = float(anchor_elo)
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gauntlet eval (no clocks): Maia variants vs Stockfish anchors using fixed depth/nodes. Writes PGN + results CSV."
    )
    ap.add_argument("--gm_name", type=str, required=True)

    ap.add_argument("--maia_type", type=str, default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", type=str, default="cpu", help="device for workers: cpu|mps|cuda")
    ap.add_argument("--fit_device", type=str, default="cpu", help="device for optional Elo fitting: cpu|cuda")

    ap.add_argument("--stockfish_path", type=str, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash_mb", type=int, default=256)
    ap.add_argument("--sf_init_timeout", type=float, default=15.0,
                    help="Timeout (seconds) for Stockfish UCI initialization per worker.")

    # Maia conditioning + move selection
    ap.add_argument("--fixed_maia_elo", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--depth_level", type=int, default=0, help="Depth for Maia tuned_model_depth beam search (0 disables).")

    # Game controls
    ap.add_argument("--max_plies", type=int, default=400)

    # Anchors: choose ONE stable knob (depth recommended here, nodes optional)
    ap.add_argument("--sf_depths", type=str, default="4,6,8,10,12,14",
                    help="Comma-separated Stockfish anchor depths.")
    ap.add_argument("--sf_nodes", type=str, default="",
                    help="Optional comma-separated Stockfish node anchors. If set, overrides sf_depths for anchors.")

    # Gauntlet size
    ap.add_argument("--games_per_match", type=int, default=40,
                    help="Games per (model,anchor) match. Colors are alternated within the match.")

    # Parallelism
    ap.add_argument("--workers", type=int, default=0, help="0 => cpu_count()-1")
    ap.add_argument("--seed", type=int, default=0)

    # Output
    ap.add_argument("--out_tag", type=str, default="gauntlet_fixed_compute")

    # Weak anchors (below depth=1 strength) - achieved primarily via tiny node caps
    ap.add_argument("--add_weak_anchors", action="store_true",
                    help="Add extra weak Stockfish anchors using tiny node caps + (if supported) UCI_Elo/Skill.")
    ap.add_argument("--weak_elos", type=str, default="1500, 1700, 1900",
                    help="Comma-separated UCI_Elo targets for weak anchors (if supported).")
    ap.add_argument("--weak_nodes", type=str, default="10,25,60,150",
                    help="Comma-separated node caps for weak anchors (paired with weak_elos by index).")
    ap.add_argument("--weak_skills", type=str, default="0,2,4,6",
                    help="Comma-separated Skill Level values for weak anchors (paired with weak_elos by index).")

    # Optional quick Elo fit (anchored)
    ap.add_argument("--fit_elos", action="store_true", help="Also fit quick anchored Elo inside this script.")
    ap.add_argument("--anchor_key", type=str, default="", help="Anchor key for in-script Elo (e.g. sf_depth_10).")
    ap.add_argument("--anchor_elo_value", type=float, default=2000.0)
    ap.add_argument("--fit_steps", type=int, default=1500)
    ap.add_argument("--fit_lr", type=float, default=0.05)
    ap.add_argument("--fit_l2", type=float, default=1e-4)

    args = ap.parse_args()
    _set_seed(args.seed)

    # Check checkpoints
    gm_dir = Path(f"./processed/single_gm/train_val/{args.gm_name}")
    dpo_path = gm_dir / "policy_best.pt"
    sft_path = gm_dir / "policy_sft_best.pt"
    pair_path = gm_dir / "policy_pairwise_sft_best.pt"
    for p in (dpo_path, sft_path, pair_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")

    # Decide anchor mode
    use_nodes = bool(args.sf_nodes.strip())
    sf_nodes_list: List[int] = []
    sf_depth_list: List[int] = []

    if use_nodes:
        sf_nodes_list = [int(x.strip()) for x in args.sf_nodes.split(",") if x.strip()]
        if not sf_nodes_list:
            raise ValueError("Provided --sf_nodes but parsed empty list.")
    else:
        sf_depth_list = [int(x.strip()) for x in args.sf_depths.split(",") if x.strip()]
        if not sf_depth_list:
            raise ValueError("No anchor depths parsed from --sf_depths.")

    player_specs: List[PlayerSpec] = []
    anchor_keys: List[str] = []

    # Base anchors (depth or nodes)
    if use_nodes:
        for n in sf_nodes_list:
            k = f"sf_nodes_{n}"
            anchor_keys.append(k)
            player_specs.append(PlayerSpec(
                key=k, kind="sf", param=f"nodes={n}",
                sf_nodes=n, sf_depth=None
            ))
    else:
        for d in sf_depth_list:
            k = f"sf_depth_{d}"
            anchor_keys.append(k)
            player_specs.append(PlayerSpec(
                key=k, kind="sf", param=f"depth={d}",
                sf_depth=d, sf_nodes=None
            ))

    # Weak anchors: depth=1 + tiny node caps (+ optional UCI_Elo/Skill)
    if args.add_weak_anchors:
        weak_elos = [int(x.strip()) for x in args.weak_elos.split(",") if x.strip()]
        weak_nodes = [int(x.strip()) for x in args.weak_nodes.split(",") if x.strip()]
        weak_skills = [int(x.strip()) for x in args.weak_skills.split(",") if x.strip()]
        m = min(len(weak_elos), len(weak_nodes), len(weak_skills))
        if m == 0:
            raise ValueError("Weak anchor lists parsed empty; check --weak_elos/--weak_nodes/--weak_skills.")

        weak_elos = weak_elos[:m]
        weak_nodes = weak_nodes[:m]
        weak_skills = weak_skills[:m]

        for uci_elo, nodes_cap, skill in zip(weak_elos, weak_nodes, weak_skills):
            k = f"sf_weak_{uci_elo}"
            anchor_keys.append(k)
            player_specs.append(
                PlayerSpec(
                    key=k,
                    kind="sf",
                    param=f"depth=1,nodes={nodes_cap},uci_elo={uci_elo},skill={skill}",
                    sf_depth=1,
                    sf_nodes=nodes_cap,
                    sf_skill=skill,
                    sf_limit_strength=True,
                    sf_uci_elo=uci_elo,
                )
            )

    # Maia variants under test
    player_specs.extend([
        PlayerSpec(key="maia_base", kind="maia", param="base", ckpt=None, tuned_model_depth=0),
        PlayerSpec(key="maia_dpo", kind="maia", param="dpo", ckpt=str(dpo_path), tuned_model_depth=args.depth_level),
        PlayerSpec(key="maia_sft", kind="maia", param="sft", ckpt=str(sft_path), tuned_model_depth=args.depth_level),
        PlayerSpec(key="maia_pairwise_sft", kind="maia", param="pairwise_sft", ckpt=str(pair_path), tuned_model_depth=args.depth_level),
    ])
    model_keys = ["maia_base", "maia_dpo", "maia_sft", "maia_pairwise_sft"]

    specs_by_key = {p.key: p for p in player_specs}

    workers = args.workers if args.workers and args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    init_state = dict(
        seed=args.seed,
        device_str=args.device,
        maia_type=args.maia_type,
        stockfish_path=args.stockfish_path,
        threads=args.threads,
        hash_mb=args.hash_mb,
        sf_init_timeout=float(args.sf_init_timeout),
        fixed_maia_elo=args.fixed_maia_elo,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        max_plies=int(args.max_plies),
        player_specs=player_specs,
        specs_by_key=specs_by_key,
    )

    out_dir = Path(f"./processed/single_gm/train_val/validation_results/{args.gm_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "nodes" if use_nodes else "depth"
    depth_part = f"maiaDepth{args.depth_level}"
    anchor_part = f"sf{mode}"
    base_name = f"{args.out_tag}_{anchor_part}_{depth_part}_gpm{args.games_per_match}_seed{args.seed}"

    pgn_path = out_dir / f"{base_name}.pgn"
    results_csv = out_dir / f"{base_name}_results.csv"

    tasks = _build_gauntlet_tasks(
        models=model_keys,
        anchors=anchor_keys,
        games_per_match=args.games_per_match,
        seed=args.seed,
    )
    total_games = len(tasks) * args.games_per_match
    print(f"Models={len(model_keys)} Anchors={len(anchor_keys)} Matches={len(tasks)} TotalGames={total_games}")
    print(f"Writing PGN: {pgn_path}")
    print(f"Writing CSV: {results_csv}")

    ctx = mp.get_context("spawn")
    all_game_rows_for_optional_fit: List[Tuple[str, str, float]] = []

    with open(pgn_path, "w", encoding="utf-8") as pgn_f, open(results_csv, "w", newline="", encoding="utf-8") as csv_f:
        w = csv.writer(csv_f)
        w.writerow(["gm_name", "white", "black", "score_white", "result", "model_key", "anchor_key", "anchor_param", "maia_param"])

        with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(init_state,)) as pool:
            for out in pool.imap_unordered(_run_gauntlet_task, tasks):
                for wk, bk, s_w, res, pgn_text in out:
                    pgn_f.write(pgn_text)
                    pgn_f.write("\n\n")

                    if wk in model_keys:
                        model_key, anchor_key = wk, bk
                    elif bk in model_keys:
                        model_key, anchor_key = bk, wk
                    else:
                        model_key, anchor_key = "", ""

                    anchor_param = specs_by_key[anchor_key].param if anchor_key in specs_by_key else ""
                    maia_param = specs_by_key[model_key].param if model_key in specs_by_key else ""

                    w.writerow([args.gm_name, wk, bk, f"{float(s_w):.1f}", res, model_key, anchor_key, anchor_param, maia_param])

                    if args.fit_elos:
                        all_game_rows_for_optional_fit.append((wk, bk, float(s_w)))

            pool.close()
            pool.join()

    print(f"Done. Wrote:\n  {pgn_path}\n  {results_csv}")

    if args.fit_elos:
        players = [p.key for p in player_specs]
        if not args.anchor_key:
            raise ValueError("--fit_elos requires --anchor_key (e.g. sf_depth_10).")
        if args.anchor_key not in players:
            raise ValueError(f"--anchor_key={args.anchor_key} not found in players.")

        elos = fit_elos_torch(
            players=players,
            games=all_game_rows_for_optional_fit,
            anchor_key=args.anchor_key,
            anchor_elo=args.anchor_elo_value,
            steps=args.fit_steps,
            lr=args.fit_lr,
            l2=args.fit_l2,
            device=args.fit_device,
            init_elos=None,
        )
        out_elo_csv = out_dir / f"{base_name}_quick_anchored_elos.csv"
        with open(out_elo_csv, "w", newline="", encoding="utf-8") as f:
            ww = csv.writer(f)
            ww.writerow(["player_key", "kind", "param", "elo", "anchor_key", "anchor_elo"])
            for ps in sorted(player_specs, key=lambda p: elos[p.key], reverse=True):
                ww.writerow([ps.key, ps.kind, ps.param, f"{elos[ps.key]:.1f}", args.anchor_key, f"{args.anchor_elo_value:.1f}"])

        print("\n=== QUICK ANCHORED ELO (sanity check) ===")
        print(f"Anchor: {args.anchor_key} = {args.anchor_elo_value:.1f}")
        for ps in sorted(player_specs, key=lambda p: elos[p.key], reverse=True):
            print(f"{ps.key:22s} {ps.kind:6s} {ps.param:28s} elo={elos[ps.key]:7.1f}")
        print(f"Wrote: {out_elo_csv}")

    print("\nNext step (recommended): feed the PGN to ordo/BayesElo for Elo + confidence intervals.")


if __name__ == "__main__":
    main()
