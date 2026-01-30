from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import chess
import chess.pgn
import torch

from maia2 import inference, model


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


def pick_maia_rejected_move(
    move_probs: Dict[str, float],
    *,
    chosen_uci: str,
    legal_moves_uci: set[str],
) -> Optional[Tuple[str, float]]:
    """
    Choose the Maia move to treat as "rejected" for DPO:
    - pick highest-prob legal move != chosen
    - return (uci, prob) or None if not found
    """
    for uci, prob in sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True):
        if uci == chosen_uci:
            continue
        if uci in legal_moves_uci:
            return uci, float(prob)
    return None


def make_pairs_for_game(
    g: chess.pgn.Game,
    *,
    gm_name: str,
    maia_model: torch.nn.Module,
    prepared: object,
    elo_self: int,
    elo_oppo_default: int,
    max_pairs_per_game: Optional[int] = None,
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

    # Iterate moves with board state before each move
    for ply_idx, mv in enumerate(g.mainline_moves()):
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
                rejected = pick_maia_rejected_move(move_probs, chosen_uci=chosen_uci, legal_moves_uci=legal_uci)
                if rejected is not None:
                    rejected_uci, rejected_prob = rejected
                    chosen_prob = float(move_probs.get(chosen_uci, 0.0))
                    print(f"chosen_prob: {chosen_prob}, rejected_prob: {rejected_prob}, win_prob: {win_prob}")
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
                            "loser": "maia2",
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
) -> None:
    pgn_path = split_dir / f"{gm_name}_{split_name}.pgn"
    if not pgn_path.exists():
        raise FileNotFoundError(f"Missing PGN: {pgn_path}")

    out_path = split_dir / f"{gm_name}_{split_name}_dpo.jsonl"

    def recs():
        for g in iter_pgn_games(pgn_path):
            yield from make_pairs_for_game(
                g,
                gm_name=gm_name,
                maia_model=maia_model,
                prepared=prepared,
                elo_self=elo_self,
                elo_oppo_default=elo_oppo_default,
                max_pairs_per_game=max_pairs_per_game,
            )

    n = write_jsonl(recs(), out_path)
    print(f"[{split_name}] wrote {n} preference pairs -> {out_path}")


def main():
    # Example usage: python ./src/grandmaster_dpo/data_processing/single_gm/z_curate_dpo_from_train_val.py --gm_name magnua --split_dir ./processed/single_gm/train_val/
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
    args = ap.parse_args()
    split_dir = Path(f"./processed/single_gm/train_val/")

    pt = Path(args.maia_pt) if args.maia_pt else None
    max_pairs = None if args.max_pairs_per_game == 0 else int(args.max_pairs_per_game)

    maia = load_maia2(maia_type=args.maia_type, device=args.device, pt_path=pt)
    prepared = inference.prepare()  # Maia-2 inference helper :contentReference[oaicite:1]{index=1}

    process_split(
        args.gm_name,
        split_dir,
        "train",
        maia_model=maia,
        prepared=prepared,
        elo_self=args.elo_self,
        elo_oppo_default=args.elo_oppo_default,
        max_pairs_per_game=max_pairs,
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
    )


if __name__ == "__main__":
    main()
