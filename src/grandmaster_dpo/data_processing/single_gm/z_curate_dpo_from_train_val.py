from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import chess
import chess.pgn
import torch
import random
import math
import hashlib

from maia2 import inference, model
from maia2.utils import mirror_move

from grandmaster_dpo.eval.stockfish_helpers import make_stockfish


def iter_pgn_games(pgn_path: Path) -> Iterator[chess.pgn.Game]:
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            yield g


def is_gm_white(headers: Dict[str, str], gm_name: str) -> bool:
    return gm_name.lower() in headers.get("White", "").lower()


def is_gm_black(headers: Dict[str, str], gm_name: str) -> bool:
    return gm_name.lower() in headers.get("Black", "").lower()


def load_maia2(
    *,
    maia_type: str,
    device: str,
    pt_path: Optional[Path] = None,
) -> torch.nn.Module:
    """
    Loads a Maia-2 model for inference.

    - If pt_path is None: uses maia2.model.from_pretrained(type=..., device=...)
    - If pt_path is provided: loads pretrained base model, then tries to load state dict from pt.

    This pattern works for:
      - pt that is a raw state_dict
      - pt that is a checkpoint dict containing 'state_dict' or 'model_state_dict'
    """
    m = model.from_pretrained(type=maia_type, device=device)

    if pt_path is None:
        return m

    ckpt = torch.load(pt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading pt: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading pt: {len(unexpected)} (showing up to 10): {unexpected[:10]}")

    return m


@torch.no_grad()
def pick_rejected_from_stockfish(
    maia_model: torch.nn.Module,
    prepared,
    *,
    fen: str,
    chosen_uci: str,
    elo_self: int,
    elo_oppo: int,
    device: torch.device,
    sf_engine: chess.engine.SimpleEngine,
    # Stockfish candidate generation
    multipv: int = 10,
    depth: int = 10,
    # Keep only "near-equal" SF moves (prevents garbage negatives)
    cp_window: int = 40,                 # keep moves with cp >= best_cp - cp_window
    # How to sample within kept SF moves
    sample_mode: str = "uniform",        # "uniform" | "softmax_cp"
    cp_temp: float = 120.0,              # used if sample_mode="softmax_cp"
    rng: random.Random = random.Random(0),
):
    """
    Pick a rejected move using Stockfish instead of Maia.

    Returns: (rejected_uci, chosen_prob, rejected_prob, legal_count)

    Notes:
    - chosen_prob/rejected_prob are reported under Maia's (temperature=1) distribution
      over legal moves in this position, for logging only.
    - Stockfish is used ONLY to define a plausible candidate set for rejected moves.
    - We exclude the chosen move from SF candidates (if present).
    """
    all_moves_dict, elo_dict, all_moves_dict_reversed = prepared

    # --- build board ---
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=True):
        return None

    legal_uci_set = {m.uci() for m in board.legal_moves}
    legal_count = len(legal_uci_set)
    if legal_count <= 1:
        return None

    # chosen must be legal
    if chosen_uci not in legal_uci_set:
        return None

    # --- Maia forward for logging probabilities (not for selecting rejected) ---
    board_input, es_cat, eo_cat, legal_moves = inference.preprocessing(
        fen, elo_self, elo_oppo, elo_dict, all_moves_dict
    )
    board_input = board_input.unsqueeze(0).to(device)
    es_t = torch.tensor([es_cat], device=device).long()
    eo_t = torch.tensor([eo_cat], device=device).long()
    legal = legal_moves.to(device)  # [V]

    logits, _, _ = maia_model(board_input, es_t, eo_t)
    logits = logits.squeeze(0)  # [V]

    # mask illegal with -inf
    neg_inf = torch.finfo(logits.dtype).min
    logits = torch.where(legal > 0, logits, torch.full_like(logits, neg_inf))

    # Maia uses white-perspective vocab; mirror chosen if black to move for indexing
    side = fen.split(" ")[1]
    chosen_eff = mirror_move(chosen_uci) if side == "b" else chosen_uci
    chosen_idx = all_moves_dict.get(chosen_eff, None)

    probs_maia = torch.softmax(logits, dim=-1)  # temperature=1 for logging
    chosen_prob = float(probs_maia[chosen_idx].item()) if chosen_idx is not None else 0.0

    # --- Stockfish candidates (MultiPV) ---
    try:
        infos = sf_engine.analyse(
            board,
            chess.engine.Limit(depth=int(depth)),
            multipv=int(max(1, multipv)),
        )
    except Exception:
        return None

    # `analyse(..., multipv=n)` can return dict or list depending on engine/version
    if isinstance(infos, dict):
        infos = [infos]

    # Extract (uci, cp) from PV[0]
    sf_moves: list[tuple[str, int]] = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        uci = pv[0].uci()
        # ensure legal
        if uci not in legal_uci_set:
            continue

        # Convert score to centipawns from side-to-move POV (consistent comparisons)
        try:
            pov_score = score.pov(board.turn)
        except Exception:
            pov_score = score

        # python-chess scores may be Mate() or Cp(); handle both
        cp: int
        try:
            cp_val = pov_score.score(mate_score=100000)  # mate -> large magnitude
            cp = int(cp_val) if cp_val is not None else 0
        except Exception:
            # very defensive fallback
            cp = 0

        sf_moves.append((uci, cp))

    if not sf_moves:
        return None

    # Exclude the chosen move so it's a proper negative
    best_cp_all = max(cp for _, cp in sf_moves)

    kept = [(u, cp) for (u, cp) in sf_moves
            if cp >= best_cp_all - int(cp_window) and u != chosen_uci]

    if not kept:
        return None

    # --- sample rejected among kept SF candidates ---
    if sample_mode == "uniform":
        rejected_uci, rejected_cp = rng.choice(kept)
    elif sample_mode == "softmax_cp":
        # Higher cp => higher sampling weight; cp_temp controls softness
        cps = [cp for _, cp in kept]
        # stable softmax in python
        m = max(cps)
        exps = [math.exp((cp - m) / max(cp_temp, 1e-6)) for cp in cps]
        s = sum(exps)
        weights = [e / s for e in exps]
        idx = rng.choices(range(len(kept)), weights=weights, k=1)[0]
        rejected_uci, rejected_cp = kept[idx]
    else:
        raise ValueError(f"Unknown sample_mode={sample_mode!r} (use 'uniform' or 'softmax_cp').")

    # --- Maia prob for rejected (for logging only) ---
    rejected_eff = mirror_move(rejected_uci) if side == "b" else rejected_uci
    rejected_idx = all_moves_dict.get(rejected_eff, None)
    rejected_prob = float(probs_maia[rejected_idx].item()) if rejected_idx is not None else 0.0

    return rejected_uci, chosen_prob, rejected_prob, legal_count, rejected_cp, best_cp_all, sf_moves

def stable_game_id(headers: dict, gm_name: str) -> str:
    key = "|".join([
        gm_name,
        headers.get("Event",""),
        headers.get("Site",""),
        headers.get("Date",""),
        headers.get("Round",""),
        headers.get("White",""),
        headers.get("Black",""),
        headers.get("Result",""),
        headers.get("Round","")
    ])
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{gm_name}:{h}"

def opening_prefix_uci_from_game(g: chess.pgn.Game, max_plies: int = 20) -> list[str]:
    out = []
    b = g.board()
    for i, mv in enumerate(g.mainline_moves()):
        if i >= max_plies:
            break
        out.append(mv.uci())
        b.push(mv)
    return out

def make_pairs_for_game(
    g: chess.pgn.Game,
    *,
    gm_name: str,
    maia_model: torch.nn.Module,
    prepared: object,
    elo_self: int,
    elo_oppo_default: int,
    max_pairs_per_game: Optional[int] = None,
    skip_first_plies: int,
    sf_engine: chess.engine.SimpleEngine,
    rng: random.Random
) -> Iterator[dict]:
    """
    Yield DPO pairs for each position where Kasparov is to move.
    Each yielded record is JSON-serializable.
    """
    headers = dict(g.headers)
    gm_w = is_gm_white(headers, gm_name)
    gm_b = is_gm_black(headers, gm_name)
    if not (gm_w or gm_b):
        return

    # If opponent Elo present, use it; else default
    oppo_elo = elo_oppo_default
    # Some PGNs store Elo in WhiteElo/BlackElo. Use the *opponent*.
    try:
        if gm_w and headers.get("BlackElo"):
            oppo_elo = int(headers["BlackElo"])
        elif gm_b and headers.get("WhiteElo"):
            oppo_elo = int(headers["WhiteElo"])
    except Exception:
        oppo_elo = elo_oppo_default

    board = g.board()
    pairs_made = 0

    opening_prefix_uci = opening_prefix_uci_from_game(g, max_plies=20)

    # Iterate moves with board state before each move
    for ply_idx, mv in enumerate(g.mainline_moves()):
        if ply_idx <= skip_first_plies - 1:
            # skip first two moves
            board.push(mv)
            continue
        gm_to_move = (gm_w and board.turn == chess.WHITE) or (gm_b and board.turn == chess.BLACK)
        chosen_uci = mv.uci()

        if gm_to_move:
            fen = board.fen()
            legal_uci = {m.uci() for m in board.legal_moves}

            # Maia expects fen + (elo_self, elo_oppo)
            # Returns move_probs dict and win_prob
            move_probs, win_prob = inference.inference_each(maia_model, prepared, fen, elo_self, oppo_elo)

            # We only keep pairs where chosen is legal (should be) and Maia has an alternative
            if chosen_uci in legal_uci:
                rejected = pick_rejected_from_stockfish(
                    maia_model,
                    prepared,
                    fen=fen,
                    chosen_uci=chosen_uci,
                    elo_self=elo_self,
                    elo_oppo=oppo_elo,
                    device="cpu",
                    rng=rng,
                    sf_engine=sf_engine
                )
                if rejected is not None:
                    side_to_move = "w" if board.turn == chess.WHITE else "b"
                    fullmove_number = board.fullmove_number  # python-chess property
                    chosen_san = board.san(mv)  # BEFORE push
                    is_capture = board.is_capture(mv)
                    gives_check = board.gives_check(mv)
                    is_promo = mv.promotion is not None

                    rejected_uci, chosen_prob, rejected_prob, legal_count, rejected_cp, best_cp_all, sf_moves = rejected
                    print(f"chosen_prob: {chosen_prob}, rejected_prob: {rejected_prob}, win_prob: {win_prob}, rejected_cp: {rejected_cp}, best_cp_all: {best_cp_all}")
                    yield {
                        # DPO core fields
                        "prompt": {
                            "fen": fen,
                            "elo_self": int(elo_self),
                            "elo_oppo": int(oppo_elo),
                        },
                        "chosen": chosen_uci,     # gm move
                        "rejected": rejected_uci, # Maia suggestion (different)
                        # Preference labeling (explicit)
                        "preference": {
                            "winner": gm_name,
                            "loser": "stockfish_candidate",
                            "label": 1,  # 1 means chosen > rejected
                        },
                        # Useful metadata for auditing / filtering
                        "meta": {
                            "event": headers.get("Event", ""),
                            "site": headers.get("Site", ""),
                            "date": headers.get("Date", ""),
                            "round": headers.get("Round", ""),
                            "white": headers.get("White", ""),
                            "black": headers.get("Black", ""),
                            "result": headers.get("Result", ""),
                            "ply_idx": ply_idx,
                            "gm_side": "white" if gm_w else "black",
                            "maia": {
                                "chosen_prob": chosen_prob,
                                "rejected_prob": rejected_prob,
                                "win_prob": float(win_prob) if win_prob is not None else None,
                                "top1": max(move_probs, key=move_probs.get) if move_probs else None,
                            },
                            "stockfish": {
                                "rejected_cp": rejected_cp,
                                "best_cp_all": best_cp_all,
                                "sf_moves_returned": sf_moves
                            },
                            "game_header_hash": stable_game_id(headers, gm_name),
                                    # opening reconstruction helper
                            "opening_prefix_uci_20": opening_prefix_uci,

                            # SAN + tactical flags (for sacrifice/volatility proxies later)
                            "chosen_san": chosen_san,
                            "is_capture": bool(is_capture),
                            "gives_check": bool(gives_check),
                            "is_promotion": bool(is_promo),
                            "side_to_move": side_to_move,
                            "fullmove_number": fullmove_number,
                        },
                    }

                    pairs_made += 1
                    if max_pairs_per_game is not None and pairs_made >= max_pairs_per_game:
                        break

        board.push(mv)


def write_jsonl(records: Iterator[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def process_split(
    gm_name: str,
    split_dir: Path,
    split_name: str,
    *,
    maia_model: torch.nn.Module,
    prepared: object,
    elo_self: int,
    elo_oppo_default: int,
    max_pairs_per_game: Optional[int],
    skip_first_plies: int,
    sf_engine: chess.engine.SimpleEngine
) -> None:
    pgn_path = split_dir / f"{gm_name}_{split_name}.pgn"
    if not pgn_path.exists():
        raise FileNotFoundError(f"Missing PGN: {pgn_path}")

    #out_path = split_dir / f"{gm_name}_{split_name}_dpo_topk{topk}_temperature{temperature}_skipplies{skip_first_plies}.jsonl"
    out_path = split_dir / f"{gm_name}_{split_name}_dpo.jsonl"
    print(f"Writing to: {out_path}")
    def recs():
        rng = random.Random(0 if split_name == "train" else 1)
        for g in iter_pgn_games(pgn_path):
            yield from make_pairs_for_game(
                g,
                gm_name=gm_name,
                maia_model=maia_model,
                prepared=prepared,
                elo_self=elo_self,
                elo_oppo_default=elo_oppo_default,
                max_pairs_per_game=max_pairs_per_game,
                skip_first_plies=skip_first_plies,
                sf_engine=sf_engine,
                rng=rng
            )

    n = write_jsonl(recs(), out_path)
    print(f"[{split_name}] wrote {n} preference pairs -> {out_path}")


def main():
    # Example usage: python ./src/grandmaster_dpo/data_processing/single_gm/z_curate_dpo_from_train_val.py --gm_name carlsen --split_dir ./final_experiments_for_paper/experiment1/train_val_pgns_twic
    # Note: GM name has to be in the PGN file's player names (White or Black)... capitalization doesn't matter
    # If the name is common it will match both, so you need to be careful to use the correct one.
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm_name", required=True, help="Name of the grandmaster.")
    ap.add_argument("--split_dir", required=True, help="Folder containing train.pgn and val.pgn.")
    ap.add_argument("--maia_type", default="blitz", choices=["blitz", "rapid"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--maia_pt", default="", help="Optional .pt weights to load on top of from_pretrained().")
    ap.add_argument("--elo_self", type=int, default=2800, help="GM Elo to condition Maia on.")
    ap.add_argument("--elo_oppo_default", type=int, default=2700, help="Fallback opponent Elo if missing.")
    ap.add_argument("--max_pairs_per_game", type=int, default=0, help="0 means no cap.")
    ap.add_argument("--skip_first_plies", type=int, default=0, help="Number of plies at beg. to skip.")
    ap.add_argument("--sf_path", type=str, default="/usr/local/bin/stockfish")
    ap.add_argument("--sf_threads", type=int, default=8)
    args = ap.parse_args()
    split_dir = Path(f"{args.split_dir}")

    pt = Path(args.maia_pt) if args.maia_pt else None
    max_pairs = None if args.max_pairs_per_game == 0 else int(args.max_pairs_per_game)

    maia = load_maia2(maia_type=args.maia_type, device=args.device, pt_path=pt)
    prepared = inference.prepare()  # Maia-2 inference helper :contentReference[oaicite:1]{index=1}

    sf_engine = make_stockfish(args.sf_path, threads=args.sf_threads, hash_mb=2048)

    process_split(
        args.gm_name,
        split_dir,
        "train",
        maia_model=maia,
        prepared=prepared,
        elo_self=args.elo_self,
        elo_oppo_default=args.elo_oppo_default,
        max_pairs_per_game=max_pairs,
        skip_first_plies=args.skip_first_plies,
        sf_engine=sf_engine
    )
    process_split(
        args.gm_name,
        split_dir,
        "val",
        maia_model=maia,
        prepared=prepared,
        elo_self=args.elo_self,
        elo_oppo_default=args.elo_oppo_default,
        max_pairs_per_game=max_pairs,
        skip_first_plies=args.skip_first_plies,
        sf_engine=sf_engine
    )


if __name__ == "__main__":
    main()
