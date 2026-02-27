# lambda_function.py (Python 3.11)
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import torch

from maia2 import inference, model as maia_model
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move

# your helper (as in your abstractions)
from grandmaster_dpo.eval.stockfish_helpers import make_stockfish


# ----------------------------
# Shared helpers (subset of your eval_abstractions.py)
# ----------------------------

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
    uci_eff = all_moves[idx]  # Maia vocab is white-perspective
    side = fen.split(" ")[1]
    return mirror_move(uci_eff) if side == "b" else uci_eff

def apply_legal_mask(logits: torch.Tensor, legal_moves: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(legal_moves > 0, logits, torch.full_like(logits, neg_inf))

def batch_preprocess_single(
    *,
    all_moves_dict: Dict[str, int],
    elo_dict: Dict[str, int],
    fen: str,
    elo_self: int,
    elo_oppo: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bi, es_cat, eo_cat, lm = inference.preprocessing(
        fen, int(elo_self), int(elo_oppo), elo_dict, all_moves_dict
    )
    board_input = bi.unsqueeze(0).to(device)          # [1,...]
    legal_moves = lm.unsqueeze(0).to(device)          # [1,V]
    es_t = torch.tensor([int(es_cat)], device=device).long()
    eo_t = torch.tensor([int(eo_cat)], device=device).long()
    return board_input, legal_moves, es_t, eo_t

def forward_logits(m: torch.nn.Module, board_input: torch.Tensor, es: torch.Tensor, eo: torch.Tensor) -> torch.Tensor:
    logits, _, _ = m(board_input, es, eo)
    return logits

def _entropy(probs: List[float], eps: float = 1e-12) -> float:
    s = 0.0
    for p in probs:
        pp = max(float(p), eps)
        s -= pp * math.log(pp)
    return float(s)

def _score_to_cp(score: chess.engine.PovScore, mate_score: int = 100_000) -> int:
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
        m = rel.mate()
        if m is not None:
            return mate_score if m > 0 else -mate_score
        return 0
    return int(cp)

def fen_ply_abs(fen: str) -> int:
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return 2 * (fullmove - 1) + (1 if side == "b" else 0)

def game_status_from_board(board: chess.Board) -> Dict[str, Any]:
    # winner: "white"|"black"|None
    # state: ongoing|checkmate|stalemate|draw
    if board.is_checkmate():
        # side to move is checkmated, so the other side won
        winner = "black" if board.turn == chess.WHITE else "white"
        return {"state": "checkmate", "winner": winner, "reason": "checkmate"}
    if board.is_stalemate():
        return {"state": "stalemate", "winner": None, "reason": "stalemate"}
    if board.is_insufficient_material():
        return {"state": "draw", "winner": None, "reason": "insufficient_material"}
    if board.can_claim_draw():
        # could be 50-move / repetition; we don’t know which claim, but it’s drawable
        return {"state": "draw", "winner": None, "reason": "claimable_draw"}
    if board.is_game_over(claim_draw=True):
        # fallback
        return {"state": "draw", "winner": None, "reason": "game_over"}
    return {"state": "ongoing", "winner": None, "reason": ""}

def json_response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(payload),
    }

def error_payload(
    *,
    game_id: str,
    code: str,
    message: str,
    server_ply: int,
    server_fen: str,
    clock: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ok": False,
        "game_id": game_id,
        "error": {"code": code, "message": message},
        "server_ply": server_ply,
        "server_fen": server_fen,
        "clock": clock,
    }


# ----------------------------
# Engine config & mapping
# ----------------------------

@dataclass(frozen=True)
class EngineLimit:
    # request format: { "type": "time_ms|nodes|depth", "value": 200 }
    type: str
    value: int

@dataclass(frozen=True)
class EngineProfile:
    gm_name: str
    maia_type: str = "blitz"
    # Maia elo-conditioning (you can map from game_type_id too)
    elo_self: int = 2800
    elo_oppo: int = 2800

    # Stockfish multipv + filtering (you said “top10”)
    sf_multipv_topk: int = 10
    restrict_cp_window: int = 60

    # Softmax temp & sampling policy
    temperature: float = 1.0
    sample: bool = False

# Example mapping: game_type_id -> style profile
# You can expand this to include variants (e.g., bullet/blitz/classical, different ELO-conditioning)
GAME_TYPE_TO_PROFILE: Dict[str, EngineProfile] = {
    # e.g. "gm_kasparov_blitz": EngineProfile(gm_name="kasparov", maia_type="blitz", temperature=0.9, sample=False),
    "gm_carlsen_blitz": EngineProfile(gm_name="carlsen", maia_type="blitz", temperature=1.0, sample=True),
    "gm_nakamura_blitz": EngineProfile(gm_name="nakamura", maia_type="blitz", temperature=1.0, sample=True),
    "gm_caruana_blitz": EngineProfile(gm_name="caruana", maia_type="blitz", temperature=1.0, sample=True),
    "gm_firouzja_blitz": EngineProfile(gm_name="firouzja", maia_type="blitz", temperature=1.0, sample=True),
    "gm_giri_blitz": EngineProfile(gm_name="giri", maia_type="blitz", temperature=1.0, sample=True),
    "gm_gukesh_blitz": EngineProfile(gm_name="gukesh", maia_type="blitz", temperature=1.0, sample=True),
    "gm_praggnanandhaa_blitz": EngineProfile(gm_name="praggnanandhaa", maia_type="blitz", temperature=1.0, sample=True),
    "gm_vincent_blitz": EngineProfile(gm_name="vincent", maia_type="blitz", temperature=1.0, sample=True),
    "gm_wei_blitz": EngineProfile(gm_name="wei", maia_type="blitz", temperature=1.0, sample=True),

    "gm_alekhine_blitz": EngineProfile(gm_name="alekhine", maia_type="blitz", temperature=1.0, sample=True),
    "gm_anand_blitz": EngineProfile(gm_name="anand", maia_type="blitz", temperature=1.0, sample=True),
    "gm_botvinnik_blitz": EngineProfile(gm_name="botvinnik", maia_type="blitz", temperature=1.0, sample=True),
    "gm_capablanca_blitz": EngineProfile(gm_name="capablanca", maia_type="blitz", temperature=1.0, sample=True),
    "gm_fischer_blitz": EngineProfile(gm_name="fischer", maia_type="blitz", temperature=1.0, sample=True),
    "gm_kasparov_blitz": EngineProfile(gm_name="kasparov", maia_type="blitz", temperature=1.0, sample=True),
    "gm_lasker_blitz": EngineProfile(gm_name="lasker", maia_type="blitz", temperature=1.0, sample=True),
    "gm_polgar_blitz": EngineProfile(gm_name="polgar", maia_type="blitz", temperature=1.0, sample=True),
    "gm_tal_blitz": EngineProfile(gm_name="tal", maia_type="blitz", temperature=1.0, sample=True)
}

def load_game_type_profile(game_type_id: str) -> EngineProfile:
    prof = GAME_TYPE_TO_PROFILE.get(game_type_id)
    if prof is None:
        # sane default if you haven’t wired mapping yet
        return EngineProfile(gm_name="default", maia_type="blitz")
    return prof


# ----------------------------
# Model / Stockfish singletons (warm Lambda reuse)
# ----------------------------

_GLOBALS: Dict[str, Any] = {
    "device": None,
    "elo_dict": None,
    "all_moves": None,
    "all_moves_dict": None,
    "models": {},     # gm_name -> {"policy":..., "sf":..., "maia_type":...}
}

_GLOBALS["sf"] = None

def get_stockfish() -> chess.engine.SimpleEngine:
    if _GLOBALS["sf"] is None:
        stockfish_path = os.environ.get("STOCKFISH_PATH", "/opt/bin/stockfish")
        sf_threads = int(os.environ.get("STOCKFISH_THREADS", "1"))
        sf_hash_mb = int(os.environ.get("STOCKFISH_HASH_MB", "128"))
        sf_timeout_s = float(os.environ.get("STOCKFISH_TIMEOUT_S", "5.0"))
        _GLOBALS["sf"] = make_stockfish(
            stockfish_path,
            threads=sf_threads,
            hash_mb=sf_hash_mb,
            uci_elo=None,          # full strength by default; you can map this from game_type_id too
            skill_level=None,
            timeout=sf_timeout_s,
        )
    return _GLOBALS["sf"]


def get_device() -> torch.device:
    # Lambda is typically CPU unless you’re doing something custom
    return torch.device(os.environ.get("MAIA_DEVICE", "cpu"))

def _load_policy_weights(model: torch.nn.Module, pt_path: str) -> None:
    sd = torch.load(pt_path, map_location="cpu")
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (showing 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}")

def get_globals() -> None:
    if _GLOBALS["device"] is None:
        _GLOBALS["device"] = get_device()
    if _GLOBALS["elo_dict"] is None:
        _GLOBALS["elo_dict"] = create_elo_dict()
    if _GLOBALS["all_moves"] is None or _GLOBALS["all_moves_dict"] is None:
        all_moves = get_all_possible_moves()
        _GLOBALS["all_moves"] = all_moves
        _GLOBALS["all_moves_dict"] = {m: i for i, m in enumerate(all_moves)}

def get_or_load_gm_bundle(profile: EngineProfile) -> Dict[str, Any]:
    """
    Loads:
      - base Maia2 model (reference)
      - policy Maia2 model (fine-tuned DPO weights for gm_name)
      - stockfish engine
    Caches by (gm_name, maia_type) key.
    """
    get_globals()

    gm_name = profile.gm_name
    maia_type = profile.maia_type
    key = f"{gm_name}::{maia_type}"

    if key in _GLOBALS["models"]:
        return _GLOBALS["models"][key]

    device: torch.device = _GLOBALS["device"]

    policy = maia_model.from_pretrained(type=maia_type, device=str(device)).to(device).eval()

    model_root = os.environ.get("MODEL_ROOT", "/opt/models")  # put weights in a Lambda layer or container image
    # you said: only DPO + SF helper for serving
    pt_path = Path(model_root) / gm_name / "policy_dpo_best.pt"
    if not pt_path.exists():
        # allow a single default file too
        alt = Path(model_root) / f"{gm_name}_policy_dpo_best.pt"
        if alt.exists():
            pt_path = alt
        else:
            raise FileNotFoundError(f"Could not find DPO policy weights for gm_name={gm_name} at {pt_path} or {alt}")

    _load_policy_weights(policy, str(pt_path))

    sf = get_stockfish()
    bundle = {
        "policy": policy,
        "sf": sf,
        "maia_type": maia_type,
        "gm_name": gm_name,
    }
    _GLOBALS["models"][key] = bundle
    return bundle


# ----------------------------
# Game state (ElastiCache pseudo-code)
# ----------------------------

def redis_get_game_state_pseudocode(game_id: str) -> Optional[Dict[str, Any]]:
    """
    PSEUDO-CODE ONLY (no-op):
      r = redis.Redis(host=..., port=..., decode_responses=True)
      s = r.get(f"game:{game_id}")
      return json.loads(s) if s else None
    """
    return None

def redis_set_game_state_pseudocode(game_id: str, state: Dict[str, Any], ttl_s: int = 86400) -> None:
    """
    PSEUDO-CODE ONLY (no-op):
      r.set(f"game:{game_id}", json.dumps(state), ex=ttl_s)
    """
    return None


# ----------------------------
# Stockfish limit from request
# ----------------------------

def limit_from_request(limit_obj: Dict[str, Any]) -> chess.engine.Limit:
    t = str(limit_obj.get("type", "time_ms"))
    v = int(limit_obj.get("value", 200))
    if t == "depth":
        return chess.engine.Limit(depth=v)
    if t == "nodes":
        return chess.engine.Limit(nodes=v)
    # default time_ms
    return chess.engine.Limit(time=max(0.001, v / 1000.0))


# ----------------------------
# Core move selection: SF top-10 -> Maia policy conditional dist -> select
# ----------------------------

@torch.no_grad()
def pick_bot_move_with_sf_topk(
    *,
    bundle: Dict[str, Any],
    profile: EngineProfile,
    fen: str,
    engine_limit: chess.engine.Limit,
    random_seed: int,
) -> Tuple[str, int, List[str]]:
    """
    Returns:
      bot_move_uci, bot_eval_cp(best sf cp), bot_pv_uci(best pv line)
    """
    get_globals()
    device: torch.device = _GLOBALS["device"]
    all_moves_dict: Dict[str, int] = _GLOBALS["all_moves_dict"]
    elo_dict: Dict[str, int] = _GLOBALS["elo_dict"]

    board = chess.Board(fen)
    sf = bundle["sf"]

    infos = sf.analyse(board, engine_limit, multipv=int(profile.sf_multipv_topk))
    cands: List[Tuple[str, int, List[str]]] = []  # (uci0, cp, pvline_uci)
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        u0 = pv[0].uci()
        cp = _score_to_cp(score)
        pv_uci = [m.uci() for m in pv[:8]]  # cap PV for response payload
        cands.append((u0, cp, pv_uci))

    if not cands:
        # fallback: choose any legal move
        mv = next(iter(board.legal_moves))
        return mv.uci(), 0, [mv.uci()]

    # best by cp
    best_cp = max(cp for _, cp, _ in cands)
    best_pv = max(cands, key=lambda x: x[1])[2]

    # restrict cp window
    kept = cands
    if profile.restrict_cp_window is not None:
        w = int(profile.restrict_cp_window)
        filt = [x for x in kept if x[1] >= best_cp - w]
        if filt:
            kept = filt

    # compute Maia logits for this position
    board_input, legal_moves, es_t, eo_t = batch_preprocess_single(
        all_moves_dict=all_moves_dict,
        elo_dict=elo_dict,
        fen=fen,
        elo_self=profile.elo_self,
        elo_oppo=profile.elo_oppo,
        device=device,
    )
    logits = forward_logits(bundle["policy"], board_input, es_t, eo_t)  # [1,V]
    logits_m = apply_legal_mask(logits, legal_moves)[0]                # [V]

    t = max(1e-6, float(profile.temperature))
    logp_all = torch.log_softmax(logits_m / t, dim=-1)                 # [V]

    # candidate conditional distribution q over kept
    cand_moves = [u for (u, _cp, _pv) in kept]
    cand_idxs = [uci_to_vocab_index(all_moves_dict, fen, u) for u in cand_moves]

    cand_logps: List[torch.Tensor] = []
    for idx in cand_idxs:
        if idx < 0:
            cand_logps.append(torch.tensor(torch.finfo(logp_all.dtype).min, device=logp_all.device))
        else:
            cand_logps.append(logp_all[idx])
    cand_logps_t = torch.stack(cand_logps, dim=0)      # [K]
    cand_probs_t = torch.softmax(cand_logps_t, dim=0)  # [K]
    cand_probs = cand_probs_t.detach().cpu().tolist()

    rng = random.Random(int(random_seed))
    if bool(profile.sample):
        r = rng.random()
        acc = 0.0
        sel_i = 0
        for j, p in enumerate(cand_probs):
            acc += float(p)
            if r <= acc:
                sel_i = j
                break
    else:
        sel_i = int(torch.argmax(cand_probs_t).item())

    bot_uci = cand_moves[sel_i]
    return bot_uci, int(best_cp), best_pv


# ----------------------------
# Lambda handler
# ----------------------------

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    # Route: POST /games
    try:
        method = (event.get("requestContext", {}).get("http", {}).get("method")
                  or event.get("httpMethod")
                  or "").upper()
        path = event.get("rawPath") or event.get("path") or ""
        if method != "POST" or not path.endswith("/games"):
            return json_response(404, {"ok": False, "error": {"code": "not_found", "message": "Unknown route"}})

        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            # not handling base64 in this minimal example
            return json_response(400, {"ok": False, "error": {"code": "bad_request", "message": "base64 body not supported"}})

        req = json.loads(body)

        # Extract request fields
        game_id = str(req.get("game_id") or "")
        client_ply = int(req.get("client_ply", -1))
        pre_move_fen = str(req.get("pre_move_fen") or "")
        client_uci = str(req.get("client_uci") or "")
        bot_id = str(req.get("bot_id") or "")
        game_type_id = str(req.get("game_type_id") or "")

        clock = req.get("clock") or {"white_ms": None, "black_ms": None}
        timing = req.get("timing") or {}
        elapsed_ms = int(timing.get("player_move_elapsed_ms", 0))

        engine_config = req.get("engine_config") or {}
        limit_obj = (engine_config.get("limit") or {"type": "time_ms", "value": 200})
        random_seed = int((engine_config.get("random_seed") or 0))

        player_color_req = str(req.get("player_color") or "")  # "white"|"black"|"" (maybe empty on subsequent calls)

        if not game_id or not pre_move_fen or not game_type_id:
            return json_response(
                400,
                {"ok": False, "error": {"code": "bad_request", "message": "Missing required fields"}},
            )

        # PSEUDO: lookup state from ElastiCache (no-op)
        state = redis_get_game_state_pseudocode(game_id)
        if state is None:
            if player_color_req not in ("white", "black"):
                return json_response(
                    400,
                    {"ok": False, "error": {"code": "bad_request", "message": "Missing/invalid player_color (white|black)"}},
                )
            player_color = player_color_req
            server_fen = pre_move_fen
            server_ply = client_ply if client_ply >= 0 else fen_ply_abs(pre_move_fen)
        else:
            player_color = str(state.get("player_color", player_color_req or "white"))
            server_fen = str(state.get("fen", pre_move_fen))
            server_ply = int(state.get("ply", fen_ply_abs(server_fen)))

        # desync check
        if client_ply >= 0 and server_ply != client_ply:
            return json_response(
                409,
                error_payload(
                    game_id=game_id,
                    code="desync",
                    message=f"client_ply={client_ply} does not match server_ply={server_ply}",
                    server_ply=server_ply,
                    server_fen=server_fen,
                    clock=clock,
                ),
            )

        # Validate FEN
        try:
            board = chess.Board(server_fen)
        except Exception:
            return json_response(
                400,
                error_payload(
                    game_id=game_id,
                    code="bad_fen",
                    message="Invalid FEN",
                    server_ply=server_ply,
                    server_fen=server_fen,
                    clock=clock,
                ),
            )

        player_is_white = (player_color == "white")
        is_player_turn = (board.turn == chess.WHITE) if player_is_white else (board.turn == chess.BLACK)

        # game over?
        if board.is_game_over(claim_draw=True):
            return json_response(
                409,
                error_payload(
                    game_id=game_id,
                    code="game_over",
                    message="Game already over",
                    server_ply=server_ply,
                    server_fen=board.fen(),
                    clock=clock,
                ),
            )

        server_ply_before = server_ply
        
        if is_player_turn:
            # Apply player move
            try:
                mv = chess.Move.from_uci(client_uci)
            except Exception:
                return json_response(
                    400,
                    error_payload(
                        game_id=game_id,
                        code="illegal_move",
                        message="Invalid UCI format or was player turn and no UCI provided",
                        server_ply=server_ply,
                        server_fen=board.fen(),
                        clock=clock,
                    ),
                )

            if mv not in board.legal_moves:
                return json_response(
                    400,
                    error_payload(
                        game_id=game_id,
                        code="illegal_move",
                        message="Move is not legal in current position",
                        server_ply=server_ply,
                        server_fen=board.fen(),
                        clock=clock,
                    ),
                )

            # Update clock (simple: subtract elapsed from side who moved)
            # NOTE: This assumes the player is the side-to-move in server_fen and client_uci is their move.
            # If your API distinguishes player color, incorporate it here.
            if isinstance(clock, dict) and "white_ms" in clock and "black_ms" in clock:
                try:
                    if board.turn == chess.WHITE:
                        clock["white_ms"] = max(0, int(clock["white_ms"]) - elapsed_ms)
                    else:
                        clock["black_ms"] = max(0, int(clock["black_ms"]) - elapsed_ms)
                except Exception:
                    pass

            board.push(mv)
            server_ply += 1

            # After player move, game might be over
            if board.is_game_over(claim_draw=True):
                status = game_status_from_board(board)
                # PSEUDO: persist
                redis_set_game_state_pseudocode(game_id, {"fen": board.fen(), "ply": server_ply, "clock": clock, "player_color": player_color})
                return json_response(
                    200,
                    {
                        "ok": True,
                        "game_id": game_id,
                        "server_ply_before": server_ply_before,
                        "server_ply_after": server_ply,
                        "new_fen": board.fen(),
                        "player_move_uci": client_uci,
                        "bot_move_uci": "",
                        "bot_id": bot_id,
                        "game_type_id": game_type_id,
                        "clock": clock,
                        "game_status": status,
                        "analysis": {"bot_eval_cp": 0, "bot_pv_uci": []},
                    },
                )

        # Load profile & engine bundle (DPO + SF-top10)
        profile = load_game_type_profile(game_type_id)
        try:
            bundle = get_or_load_gm_bundle(profile)
        except FileNotFoundError as e:
            return json_response(
                500,
                error_payload(
                    game_id=game_id,
                    code="server_error",
                    message=str(e),
                    server_ply=server_ply,
                    server_fen=board.fen(),
                    clock=clock,
                ),
            )

        # Determine SF limit from request
        engine_limit = limit_from_request(limit_obj)

        # Pick bot move
        bot_uci, bot_eval_cp, bot_pv_uci = pick_bot_move_with_sf_topk(
            bundle=bundle,
            profile=profile,
            fen=board.fen(),
            engine_limit=engine_limit,
            random_seed=random_seed,
        )

        try:
            bot_mv = chess.Move.from_uci(bot_uci)
        except Exception:
            # fallback random legal
            bot_mv = next(iter(board.legal_moves))
            bot_uci = bot_mv.uci()

        if bot_mv not in board.legal_moves:
            # fallback random legal
            bot_mv = next(iter(board.legal_moves))
            bot_uci = bot_mv.uci()

        # Bot move timing: if you later track server time, you can subtract here too.
        board.push(bot_mv)
        server_ply += 1

        status = game_status_from_board(board)

        # PSEUDO: update state in ElastiCache
        redis_set_game_state_pseudocode(game_id, {"fen": board.fen(), "ply": server_ply, "clock": clock, "player_color": player_color})

        resp_ok = {
            "ok": True,
            "game_id": game_id,
            "server_ply_before": server_ply_before,  # before player+bot applied
            "server_ply_after": server_ply,
            "new_fen": board.fen(),
            "player_move_uci": client_uci if is_player_turn else "",
            "bot_move_uci": bot_uci,
            "bot_id": bot_id,
            "game_type_id": game_type_id,
            "clock": clock,
            "game_status": status,
            "analysis": {
                "bot_eval_cp": int(bot_eval_cp),
                "bot_pv_uci": bot_pv_uci,
            },
        }
        return json_response(200, resp_ok)

    except Exception as e:
        # last-resort crash guard
        return json_response(
            500,
            {"ok": False, "error": {"code": "server_error", "message": f"Unhandled: {type(e).__name__}: {e}"}},
        )
