from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import chess

from grandmaster_dpo.website.policy_only.schemas import (
    AnalysisResponse,
    ClockState,
    ClockStateResponse,
    ClockSyncRequest,
    ClockUpdateRequest,
    ErrorInfo,
    ErrorResponse,
    GameRequest,
    GameResponse,
    StockfishMetricsResponse,
)
from grandmaster_dpo.website.policy_only.service.completion_events import (
    GameFinishedPublisher,
    NullGameFinishedPublisher,
)
from grandmaster_dpo.website.policy_only.service.runtime import (
    choose_bot_move,
    fen_ply_abs,
    game_status_from_board,
    get_or_load_bundle,
    resolve_profile,
)
from grandmaster_dpo.website.policy_only.service.opening_starts import OpeningStartManager
from grandmaster_dpo.website.policy_only.service.state import GameStateStore, StoredGameState

logger = logging.getLogger(__name__)

FINISHED_EVENT_DEFAULT_CANDIDATE_LIMIT = 3
FINISHED_EVENT_CANDIDATE_KEYS = {
    "uci",
    "prob",
    "probability",
    "in_cp_gap_window",
    "cp",
    "adjusted_cp",
    "draw_penalty_cp",
    "repetition_x2_penalty_cp",
    "one_move_from_draw_penalty_cp",
    "mate",
    "multipv_rank",
    "novelty_score",
    "risk_score",
    "attack_score",
    "sacrifice_score",
}


class GameServiceError(Exception):
    def __init__(self, status_code: int, error: ErrorResponse) -> None:
        super().__init__(error.error.message)
        self.status_code = status_code
        self.error = error


class PolicyOnlyGameService:
    def __init__(
        self,
        store: GameStateStore,
        finished_game_publisher: GameFinishedPublisher | None = None,
        opening_start_manager: OpeningStartManager | None = None,
    ) -> None:
        self.store = store
        self.finished_game_publisher = finished_game_publisher or NullGameFinishedPublisher()
        self.opening_start_manager = opening_start_manager or OpeningStartManager()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _status_from_board_and_clock(board: chess.Board, clock: ClockState) -> Any:
        if clock.white_ms is not None and clock.white_ms <= 0 and clock.black_ms is not None and clock.black_ms <= 0:
            return {"state": "timeout", "winner": None, "reason": "both_flags"}
        if clock.white_ms is not None and clock.white_ms <= 0:
            return {"state": "timeout", "winner": "black", "reason": "white_flag"}
        if clock.black_ms is not None and clock.black_ms <= 0:
            return {"state": "timeout", "winner": "white", "reason": "black_flag"}
        return game_status_from_board(board).model_dump(mode="python")

    @staticmethod
    def _format_bot_label(bot_id: str, gm_name: str) -> str:
        raw = bot_id or gm_name or "bot"
        if "_" in raw:
            raw = raw.replace("_", " ")
        return raw.title()

    @staticmethod
    def _human_actor_id(player_color: str) -> str:
        return f"human_{player_color}"

    @classmethod
    def _resolved_human_actor_id(cls, player_color: str, authenticated_user_id: str | None) -> str:
        if authenticated_user_id:
            return authenticated_user_id
        return cls._human_actor_id(player_color)

    @staticmethod
    def _build_compact_move(
        *,
        ply: int,
        color: str,
        actor_id: str,
        actor_type: str,
        move: chess.Move,
        san: str,
        move_time_ms: int,
        clock: ClockState,
    ) -> dict[str, Any]:
        return {
            "ply": ply,
            "color": color,
            "actor_id": actor_id,
            "actor_type": actor_type,
            "uci": move.uci(),
            "san": san,
            "from_sq": chess.square_name(move.from_square),
            "to_sq": chess.square_name(move.to_square),
            "promotion": chess.piece_symbol(move.promotion) if move.promotion is not None else None,
            "move_time_ms": move_time_ms,
            "clock_after_white_ms": clock.white_ms,
            "clock_after_black_ms": clock.black_ms,
        }

    def _build_inference_trace(
        self,
        *,
        game_id: str,
        ply: int,
        fen_before: str,
        actor_id: str,
        move_result: Any,
        req: GameRequest,
        side_to_move: str,
    ) -> dict[str, Any]:
        stockfish_metrics = dict(move_result.stockfish_metrics or {})
        board_state_analysis = {
            "engine_name": "stockfish",
            "engine_depth": stockfish_metrics.get("max_depth"),
            "engine_nodes": stockfish_metrics.get("total_nodes"),
            "engine_time_ms": stockfish_metrics.get("max_time_ms"),
            "position_eval_cp": move_result.eval_cp,
            "best_move_uci": stockfish_metrics.get("best_move_uci"),
            "candidate_count": stockfish_metrics.get("returned_candidate_count"),
            "pv_uci": list(move_result.pv_uci),
        }
        return {
            "game_id": game_id,
            "ply": ply,
            "fen_before": fen_before,
            "side_to_move": side_to_move,
            "actor_id": actor_id,
            "actor_type": "bot",
            "selected_move_uci": move_result.move_uci,
            "selected_move_san": None,
            "selected_move_probability": move_result.selected_probability,
            "inference_config": {
                "policy_model_name": req.bot_id or None,
                "use_gibbs": req.engine_config.use_gibbs,
                "lam": req.engine_config.lam,
                "temperature": req.engine_config.temperature,
                "alpha_style": req.engine_config.alpha_style,
                "beta_engine": req.engine_config.beta_engine,
                "engine_temp": req.engine_config.engine_temp,
                "style_temperature": req.engine_config.style_temperature,
                "novelty_weight": req.engine_config.novelty_weight,
                "novelty_weight_prob": req.engine_config.novelty_weight_prob,
                "novelty_weight_phase": req.engine_config.novelty_weight_phase.model_dump(mode="python"),
                "risk_weight": req.engine_config.risk_weight,
                "risk_weight_prob": req.engine_config.risk_weight_prob,
                "risk_weight_phase": req.engine_config.risk_weight_phase.model_dump(mode="python"),
                "attack_weight": req.engine_config.attack_weight,
                "attack_weight_prob": req.engine_config.attack_weight_prob,
                "attack_weight_phase": req.engine_config.attack_weight_phase.model_dump(mode="python"),
                "weird_move_prob": req.engine_config.weird_move_prob,
                "weird_move_phase": req.engine_config.weird_move_phase.model_dump(mode="python"),
                "weird_move_min_cp_loss": req.engine_config.weird_move_min_cp_loss,
                "weird_move_max_cp_loss": req.engine_config.weird_move_max_cp_loss,
                "top_move_suppression_prob": req.engine_config.top_move_suppression_prob,
                "top_move_suppression_phase": req.engine_config.top_move_suppression_phase.model_dump(mode="python"),
                "sacrifice_weight": req.engine_config.sacrifice_weight,
                "sacrifice_propensity_phase": req.engine_config.sacrifice_propensity_phase.model_dump(mode="python"),
                "sample": req.engine_config.sample,
                "cp_gap_window": req.engine_config.cp_gap_window,
                "stockfish_multipv_topk": req.engine_config.stockfish_multipv_topk,
                "timer_head_enabled": req.engine_config.use_timer_head,
                "time_control_style_scale": req.engine_config.time_control_style_scale,
                "requested_depth": (
                    req.engine_config.limit.value
                    if req.engine_config.limit is not None and req.engine_config.limit.type == "depth"
                    else req.engine_config.stockfish_tree_search_depth
                    or req.engine_config.stockfish_engine_depth
                ),
                "requested_time_ms": move_result.requested_think_ms,
                "requested_nodes": (
                    req.engine_config.limit.value
                    if req.engine_config.limit is not None and req.engine_config.limit.type == "nodes"
                    else req.engine_config.stockfish_engine_nodes
                ),
                "draw_penalties": req.engine_config.draw_penalties.model_dump(mode="python"),
                "forced_blunder": {
                    **req.engine_config.forced_blunder.model_dump(mode="python"),
                    "attempted": stockfish_metrics.get("forced_blunder_attempted"),
                    "triggered": stockfish_metrics.get("forced_blunder_triggered"),
                    "candidate_count": stockfish_metrics.get("forced_blunder_candidate_count"),
                    "bot_move_number": stockfish_metrics.get("forced_blunder_bot_move_number"),
                    "piece_type": stockfish_metrics.get("forced_blunder_piece_type"),
                    "cp_loss": stockfish_metrics.get("forced_blunder_cp_loss"),
                },
            },
            "board_state_analysis": board_state_analysis,
            "candidate_moves": list(move_result.candidate_moves),
            "created_at_iso": self._now_iso(),
            "engine_limit": dict(move_result.engine_limit or {}),
        }

    @staticmethod
    def _forced_blunder_already_triggered(inference_positions: list[dict[str, Any]]) -> bool:
        for raw in inference_positions:
            if not isinstance(raw, dict):
                continue
            inference_config = raw.get("inference_config")
            if not isinstance(inference_config, dict):
                continue
            forced_blunder = inference_config.get("forced_blunder")
            if isinstance(forced_blunder, dict) and forced_blunder.get("triggered") is True:
                return True
        return False

    @staticmethod
    def _finished_event_candidate_limit() -> int:
        raw = os.environ.get("POLICY_ONLY_FINISHED_EVENT_CANDIDATE_LIMIT")
        if raw is None:
            return FINISHED_EVENT_DEFAULT_CANDIDATE_LIMIT
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning(
                "invalid_finished_event_candidate_limit value=%r using_default=%s",
                raw,
                FINISHED_EVENT_DEFAULT_CANDIDATE_LIMIT,
            )
            return FINISHED_EVENT_DEFAULT_CANDIDATE_LIMIT

    @staticmethod
    def _compact_finished_event_candidate(candidate: Any) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            return {}
        return {
            key: value
            for key, value in candidate.items()
            if key in FINISHED_EVENT_CANDIDATE_KEYS and value is not None
        }

    def _compact_finished_event_candidates(
        self,
        candidates: list[Any],
        *,
        selected_move_uci: str,
    ) -> tuple[list[dict[str, Any]], int]:
        candidate_limit = self._finished_event_candidate_limit()
        if candidate_limit <= 0:
            return [], len(candidates)

        def _candidate_score(candidate: Any) -> tuple[int, float, int]:
            if not isinstance(candidate, dict):
                return (0, 0.0, 0)
            is_selected = 1 if str(candidate.get("uci") or "") == selected_move_uci else 0
            prob = candidate.get("probability", candidate.get("prob"))
            try:
                probability = float(prob) if prob is not None else 0.0
            except (TypeError, ValueError):
                probability = 0.0
            try:
                rank_score = -int(candidate.get("multipv_rank") or 9999)
            except (TypeError, ValueError):
                rank_score = -9999
            return (is_selected, probability, rank_score)

        compacted: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=_candidate_score, reverse=True):
            compacted_candidate = self._compact_finished_event_candidate(candidate)
            if compacted_candidate:
                compacted.append(compacted_candidate)
            if len(compacted) >= candidate_limit:
                break
        return compacted, max(0, len(candidates) - len(compacted))

    def _compact_finished_event_inference_positions(
        self,
        inference_positions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        compacted_positions: list[dict[str, Any]] = []
        truncated_candidate_count = 0
        for raw in inference_positions:
            if not isinstance(raw, dict):
                continue
            compacted = dict(raw)
            candidate_moves = list(raw.get("candidate_moves") or [])
            compacted_candidates, truncated = self._compact_finished_event_candidates(
                candidate_moves,
                selected_move_uci=str(raw.get("selected_move_uci") or ""),
            )
            compacted["candidate_moves"] = compacted_candidates
            if truncated > 0:
                compacted["candidate_moves_truncated_count"] = truncated
                truncated_candidate_count += truncated
            compacted_positions.append(compacted)

        if truncated_candidate_count > 0:
            logger.info(
                "finished_event_inference_candidates_compacted positions=%s truncated_candidates=%s kept_per_position=%s",
                len(compacted_positions),
                truncated_candidate_count,
                self._finished_event_candidate_limit(),
            )
        return compacted_positions

    @staticmethod
    def _json_size_bytes(payload: dict[str, Any]) -> int:
        return len(json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8"))

    def _maybe_publish_finished_event(self, state: StoredGameState) -> None:
        if state.finished_event_published_at_ms is not None or not state.finished_event_payload:
            return
        payload = dict(state.finished_event_payload)
        payload["published_at_iso"] = self._now_iso()
        try:
            message_id = self.finished_game_publisher.publish_finished_game(payload)
        except Exception:
            logger.exception(
                "games_finished_publish_failed game_id=%s event_key=%s",
                state.game_id,
                payload.get("event_key"),
            )
            return
        if message_id is None:
            return
        state.finished_event_message_id = message_id
        state.finished_event_published_at_ms = int(time.time() * 1000)
        state.finished_event_payload = None
        self.store.set(state.game_id, state)
        logger.info(
            "games_finished_published game_id=%s message_id=%s status=%s reason=%s",
            state.game_id,
            (message_id or ""),
            (state.terminal_status or {}).get("state"),
            (state.terminal_status or {}).get("reason"),
        )

    def _build_finished_event_payload(
        self,
        *,
        req: GameRequest,
        response: GameResponse,
        state: StoredGameState,
        completion_origin: str,
    ) -> dict[str, Any]:
        gm_name = response.analysis.gm_name
        bot_actor_id = req.bot_id or gm_name or "bot"
        bot_label = self._format_bot_label(req.bot_id, gm_name)
        human_actor_id = self._resolved_human_actor_id(state.player_color, state.authenticated_user_id)
        if state.player_color == "white":
            white_actor_id = human_actor_id
            white_actor_type = "user"
            white_label = "Human"
            white_family = "human"
            white_actor_username = None
            white_elo_estimate = None
            black_actor_id = bot_actor_id
            black_actor_type = "bot"
            black_label = bot_label
            black_family = "grandmaster"
            black_actor_username = None
            black_elo_estimate = None
        else:
            white_actor_id = bot_actor_id
            white_actor_type = "bot"
            white_label = bot_label
            white_family = "grandmaster"
            white_actor_username = None
            white_elo_estimate = None
            black_actor_id = human_actor_id
            black_actor_type = "user"
            black_label = "Human"
            black_family = "human"
            black_actor_username = None
            black_elo_estimate = None

        game_status = response.game_status.model_dump(mode="python")
        result = "draw"
        white_result = "draw"
        black_result = "draw"
        if game_status.get("winner") == "white":
            result = "white_win"
            white_result = "win"
            black_result = "loss"
        elif game_status.get("winner") == "black":
            result = "black_win"
            white_result = "loss"
            black_result = "win"
        initial_time_ms = (
            state.initial_clock.white_ms
            if state.initial_clock.white_ms is not None
            else state.initial_clock.black_ms
            if state.initial_clock.black_ms is not None
            else 0
        )
        event_key = (
            f"{req.game_id}:{response.server_ply_after}:{response.game_status.state}:"
            f"{response.game_status.reason}:{response.new_fen}"
        )
        request_payload = req.model_dump(mode="python", exclude={"gm_name", "opening_family"})
        payload = {
            "event_type": "game_finished",
            "event_version": 2,
            "delivery_semantics": "at_least_once",
            "source": "policy_only_api",
            "deployment_target": os.environ.get("DEPLOYMENT_TARGET", "ecs-fargate"),
            "event_created_at_iso": self._now_iso(),
            "event_key": event_key,
            "completion_origin": completion_origin,
            "game_id": response.game_id,
            "game_type_id": response.game_type_id,
            "bot_id": response.bot_id,
            "player_color": state.player_color,
            "server_ply_before": response.server_ply_before,
            "server_ply_after": response.server_ply_after,
            "final_fen": response.new_fen,
            "clock": response.clock.model_dump(mode="python"),
            "last_ply_times_ms": list(state.last_ply_times_ms),
            "game_status": game_status,
            "player_move_uci": response.player_move_uci,
            "bot_move_uci": response.bot_move_uci,
            "game": {
                "game_id": response.game_id,
                "start_fen": state.start_fen,
                "final_fen": response.new_fen,
                "move_count": len(state.moves_compact),
                "moves_compact": list(state.moves_compact),
                "inference_positions": self._compact_finished_event_inference_positions(
                    list(state.inference_positions)
                ),
                "white_actor_id": white_actor_id,
                "white_actor_type": white_actor_type,
                "white_label": white_label,
                "white_family": white_family,
                "white_actor_username": white_actor_username,
                "white_elo_estimate": white_elo_estimate,
                "white_user_id": state.authenticated_user_id if state.player_color == "white" else None,
                "black_actor_id": black_actor_id,
                "black_actor_type": black_actor_type,
                "black_label": black_label,
                "black_family": black_family,
                "black_actor_username": black_actor_username,
                "black_elo_estimate": black_elo_estimate,
                "black_user_id": state.authenticated_user_id if state.player_color == "black" else None,
                "player_color": state.player_color,
                "mode_kind": "bot" if bot_actor_id else "unknown",
                "mode_variant": req.bot_id or gm_name,
                "time_control_id": req.game_type_id,
                "initial_time_ms": int(initial_time_ms or 0),
                "increment_ms": 0,
                "result": result,
                "white_result": white_result,
                "black_result": black_result,
                "winner_actor_id": white_actor_id if game_status.get("winner") == "white" else black_actor_id if game_status.get("winner") == "black" else None,
                "winner_actor_color": game_status.get("winner"),
                "termination_reason": game_status.get("reason") or "",
                "created_at_ms": state.created_at_ms,
                "created_at_iso": datetime.fromtimestamp(state.created_at_ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "ended_at_ms": int(time.time() * 1000),
                "ended_at_iso": self._now_iso(),
                "clock_initial": state.initial_clock.model_dump(mode="python"),
                "clock_final": response.clock.model_dump(mode="python"),
            },
            "request": request_payload,
            "response": response.model_dump(mode="python"),
        }
        logger.info(
            "finished_event_payload_built game_id=%s bytes=%s moves=%s inference_positions=%s",
            response.game_id,
            self._json_size_bytes(payload),
            len(state.moves_compact),
            len(state.inference_positions),
        )
        return payload

    def _persist_terminal_state_and_publish(
        self,
        *,
        req: GameRequest,
        response: GameResponse,
        player_color: str,
        clock: ClockState,
        start_fen: str,
        initial_clock: ClockState,
        moves_compact: list[dict[str, Any]],
        inference_positions: list[dict[str, Any]],
        authenticated_user_id: str | None,
        created_at_ms: int,
        last_ply_times_ms: list[int],
        completion_origin: str,
    ) -> None:
        terminal_status = response.game_status.model_dump(mode="python")
        new_state = StoredGameState(
            game_id=req.game_id,
            fen=response.new_fen,
            start_fen=start_fen,
            ply=response.server_ply_after,
            player_color=player_color,
            authenticated_user_id=authenticated_user_id,
            clock=clock,
            initial_clock=initial_clock,
            bot_id=req.bot_id,
            game_type_id=req.game_type_id,
            moves_compact=moves_compact,
            inference_positions=inference_positions,
            last_ply_times_ms=last_ply_times_ms,
            terminal_status=terminal_status,
            created_at_ms=created_at_ms,
        )
        new_state.finished_event_payload = self._build_finished_event_payload(
            req=req,
            response=response,
            state=new_state,
            completion_origin=completion_origin,
        )
        self.store.set(req.game_id, new_state)
        logger.info(
            "game_finished_detected game_id=%s status=%s reason=%s origin=%s",
            req.game_id,
            response.game_status.state,
            response.game_status.reason,
            completion_origin,
        )
        self._maybe_publish_finished_event(new_state)

    def _error(
        self,
        *,
        status_code: int,
        game_id: str,
        code: str,
        message: str,
        server_ply: int,
        server_fen: str,
        clock: ClockState,
    ) -> GameServiceError:
        return GameServiceError(
            status_code=status_code,
            error=ErrorResponse(
                game_id=game_id,
                error=ErrorInfo(code=code, message=message),
                server_ply=server_ply,
                server_fen=server_fen,
                clock=clock,
            ),
        )

    def _load_state(self, req: GameRequest) -> tuple[StoredGameState | None, str, str, int]:
        state = self.store.get(req.game_id)
        if state is None:
            if req.player_color not in ("white", "black"):
                raise self._error(
                    status_code=400,
                    game_id=req.game_id,
                    code="bad_request",
                    message="Missing/invalid player_color (white|black)",
                    server_ply=req.client_ply if req.client_ply >= 0 else 0,
                    server_fen=req.pre_move_fen,
                    clock=req.clock,
                )
            player_color = req.player_color
            if req.opening_family:
                try:
                    opening_start = self.opening_start_manager.get(req.opening_family)
                except KeyError as exc:
                    raise self._error(
                        status_code=400,
                        game_id=req.game_id,
                        code="unknown_opening_family",
                        message=str(exc),
                        server_ply=req.client_ply if req.client_ply >= 0 else 0,
                        server_fen=req.pre_move_fen,
                        clock=req.clock,
                    )
                server_fen = opening_start.fen
                server_ply = req.client_ply if req.client_ply >= 0 else 0
            else:
                server_fen = req.pre_move_fen
                server_ply = req.client_ply if req.client_ply >= 0 else fen_ply_abs(req.pre_move_fen)
            return None, player_color, server_fen, server_ply
        return state, state.player_color, state.fen, state.ply

    def play_turn(self, req: GameRequest) -> GameResponse:
        if not req.game_id or (not req.pre_move_fen and not req.opening_family) or not req.game_type_id:
            raise self._error(
                status_code=400,
                game_id=req.game_id or "",
                code="bad_request",
                message="Missing required fields",
                server_ply=req.client_ply if req.client_ply >= 0 else 0,
                server_fen=req.pre_move_fen or "",
                clock=req.clock,
            )

        state, player_color, server_fen, server_ply = self._load_state(req)
        if req.client_ply >= 0 and req.client_ply != server_ply:
            raise self._error(
                status_code=409,
                game_id=req.game_id,
                code="desync",
                message=f"client_ply={req.client_ply} does not match server_ply={server_ply}",
                server_ply=server_ply,
                server_fen=server_fen,
                clock=state.clock if state is not None else req.clock,
            )

        try:
            board = chess.Board(server_fen)
        except Exception:
            raise self._error(
                status_code=400,
                game_id=req.game_id,
                code="bad_fen",
                message="Invalid FEN",
                server_ply=server_ply,
                server_fen=server_fen,
                clock=state.clock if state is not None else req.clock,
            )

        try:
            profile = resolve_profile(req.game_type_id, req.gm_name)
        except ValueError as exc:
            raise self._error(
                status_code=400,
                game_id=req.game_id,
                code="bad_gm_name",
                message=str(exc),
                server_ply=server_ply,
                server_fen=board.fen(),
                clock=state.clock if state is not None else req.clock,
            )

        if board.is_game_over(claim_draw=True):
            if state is not None:
                self._maybe_publish_finished_event(state)
            raise self._error(
                status_code=409,
                game_id=req.game_id,
                code="game_over",
                message="Game already over",
                server_ply=server_ply,
                server_fen=board.fen(),
                clock=state.clock if state is not None else req.clock,
            )

        clock = state.clock if state is not None else req.clock
        initial_clock = state.initial_clock if state is not None else req.clock.model_copy(deep=True)
        start_fen = state.start_fen if state is not None else server_fen
        moves_compact = list(state.moves_compact) if state is not None else []
        inference_positions = list(state.inference_positions) if state is not None else []
        created_at_ms = state.created_at_ms if state is not None else int(time.time() * 1000)
        last_ply_times_ms = list(state.last_ply_times_ms) if state is not None else []
        authenticated_user_id = (
            state.authenticated_user_id if state is not None else req.authenticated_user_id
        ) or req.authenticated_user_id
        server_ply_before = server_ply
        player_is_white = player_color == "white"
        is_player_turn = (board.turn == chess.WHITE) if player_is_white else (board.turn == chess.BLACK)

        if is_player_turn:
            try:
                player_move = chess.Move.from_uci(req.client_uci)
            except Exception:
                raise self._error(
                    status_code=400,
                    game_id=req.game_id,
                    code="illegal_move",
                    message="Invalid UCI format or missing player move",
                    server_ply=server_ply,
                    server_fen=board.fen(),
                    clock=clock,
                )

            if player_move not in board.legal_moves:
                raise self._error(
                    status_code=400,
                    game_id=req.game_id,
                    code="illegal_move",
                    message="Move is not legal in current position",
                    server_ply=server_ply,
                    server_fen=board.fen(),
                    clock=clock,
                )

            elapsed_ms = max(0, int(req.timing.player_move_elapsed_ms))
            player_move_ply = server_ply + 1
            player_move_color = "w" if board.turn == chess.WHITE else "b"
            player_move_san = board.san(player_move)
            if board.turn == chess.WHITE and clock.white_ms is not None:
                clock = ClockState(white_ms=max(0, int(clock.white_ms) - elapsed_ms), black_ms=clock.black_ms)
            elif board.turn == chess.BLACK and clock.black_ms is not None:
                clock = ClockState(white_ms=clock.white_ms, black_ms=max(0, int(clock.black_ms) - elapsed_ms))
            last_ply_times_ms.append(elapsed_ms)
            last_ply_times_ms = last_ply_times_ms[-5:]

            board.push(player_move)
            server_ply += 1
            moves_compact.append(
                self._build_compact_move(
                    ply=player_move_ply,
                    color=player_move_color,
                    actor_id=self._resolved_human_actor_id(player_color, authenticated_user_id),
                    actor_type="user",
                    move=player_move,
                    san=player_move_san,
                    move_time_ms=elapsed_ms,
                    clock=clock,
                )
            )

            status = self._status_from_board_and_clock(board, clock)
            if status["state"] != "ongoing":
                response = GameResponse(
                    game_id=req.game_id,
                    server_ply_before=server_ply_before,
                    server_ply_after=server_ply,
                    new_fen=board.fen(),
                    player_move_uci=req.client_uci,
                    bot_move_uci="",
                    bot_id=req.bot_id,
                    game_type_id=req.game_type_id,
                    clock=clock,
                    game_status=status,
                    analysis=AnalysisResponse(
                        bot_eval_cp=0,
                        bot_pv_uci=[],
                        candidate_moves=[],
                        stockfish_metrics=StockfishMetricsResponse(
                            requested_multipv_topk=max(1, int(req.engine_config.stockfish_multipv_topk)),
                            returned_candidate_count=0,
                            cp_gap_window=req.engine_config.cp_gap_window,
                        ),
                        selected_move_probability=None,
                        use_gibbs=req.engine_config.use_gibbs,
                        requested_think_ms=None,
                        actual_think_ms=None,
                        engine_limit={},
                        gm_name=profile.gm_name,
                    ),
                )
                self._persist_terminal_state_and_publish(
                    req=req,
                    response=response,
                    player_color=player_color,
                    clock=clock,
                    start_fen=start_fen,
                    initial_clock=initial_clock,
                    moves_compact=moves_compact,
                    inference_positions=inference_positions,
                    authenticated_user_id=authenticated_user_id,
                    created_at_ms=created_at_ms,
                    last_ply_times_ms=last_ply_times_ms,
                    completion_origin="player_move",
                )
                return response

        bundle = get_or_load_bundle(profile)
        logger.info(
            "play_turn game_id=%s gm=%s ply=%s use_gibbs=%s",
            req.game_id,
            profile.gm_name,
            server_ply,
            req.engine_config.use_gibbs,
        )
        move_result = choose_bot_move(
            bundle=bundle,
            fen=board.fen(),
            clock=clock,
            last_ply_times_ms=last_ply_times_ms,
            engine_config=req.engine_config,
            start_fen=start_fen,
            played_moves_uci=[str(item.get("uci") or "") for item in moves_compact],
            forced_blunder_already_triggered=self._forced_blunder_already_triggered(inference_positions),
        )
        try:
            bot_move = chess.Move.from_uci(move_result.move_uci)
        except Exception:
            bot_move = next(iter(board.legal_moves))
        if bot_move not in board.legal_moves:
            bot_move = next(iter(board.legal_moves))

        bot_move_ply = server_ply + 1
        bot_move_color = "w" if board.turn == chess.WHITE else "b"
        bot_move_san = board.san(bot_move)
        bot_fen_before = board.fen()
        if board.turn == chess.WHITE and clock.white_ms is not None:
            clock = ClockState(
                white_ms=max(0, int(clock.white_ms) - move_result.actual_think_ms),
                black_ms=clock.black_ms,
            )
        elif board.turn == chess.BLACK and clock.black_ms is not None:
            clock = ClockState(
                white_ms=clock.white_ms,
                black_ms=max(0, int(clock.black_ms) - move_result.actual_think_ms),
            )
        last_ply_times_ms.append(move_result.actual_think_ms)
        last_ply_times_ms = last_ply_times_ms[-5:]

        board.push(bot_move)
        server_ply += 1
        moves_compact.append(
            self._build_compact_move(
                ply=bot_move_ply,
                color=bot_move_color,
                actor_id=req.bot_id or profile.gm_name,
                actor_type="bot",
                move=bot_move,
                san=bot_move_san,
                move_time_ms=move_result.actual_think_ms,
                clock=clock,
            )
        )
        inference_trace = self._build_inference_trace(
            game_id=req.game_id,
            ply=bot_move_ply,
            fen_before=bot_fen_before,
            actor_id=req.bot_id or profile.gm_name,
            move_result=move_result,
            req=req,
            side_to_move="white" if bot_move_color == "w" else "black",
        )
        inference_trace["selected_move_san"] = bot_move_san
        inference_positions.append(inference_trace)
        status = self._status_from_board_and_clock(board, clock)

        response = GameResponse(
            game_id=req.game_id,
            server_ply_before=server_ply_before,
            server_ply_after=server_ply,
            new_fen=board.fen(),
            player_move_uci=req.client_uci if is_player_turn else "",
            bot_move_uci=bot_move.uci(),
            bot_id=req.bot_id,
            game_type_id=req.game_type_id,
            clock=clock,
            game_status=status,
            analysis=AnalysisResponse(
                bot_eval_cp=move_result.eval_cp,
                bot_pv_uci=move_result.pv_uci,
                candidate_moves=move_result.candidate_moves,
                stockfish_metrics=move_result.stockfish_metrics,
                selected_move_probability=move_result.selected_probability,
                use_gibbs=req.engine_config.use_gibbs,
                requested_think_ms=move_result.requested_think_ms,
                actual_think_ms=move_result.actual_think_ms,
                engine_limit=move_result.engine_limit,
                gm_name=profile.gm_name,
            ),
        )
        if status["state"] != "ongoing":
            self._persist_terminal_state_and_publish(
                req=req,
                response=response,
                player_color=player_color,
                clock=clock,
                start_fen=start_fen,
                initial_clock=initial_clock,
                moves_compact=moves_compact,
                inference_positions=inference_positions,
                authenticated_user_id=authenticated_user_id,
                created_at_ms=created_at_ms,
                last_ply_times_ms=last_ply_times_ms,
                completion_origin="bot_move",
            )
            return response

        new_state = StoredGameState(
            game_id=req.game_id,
            fen=board.fen(),
            start_fen=start_fen,
            ply=server_ply,
            player_color=player_color,
            authenticated_user_id=authenticated_user_id,
            clock=clock,
            initial_clock=initial_clock,
            bot_id=req.bot_id,
            game_type_id=req.game_type_id,
            moves_compact=moves_compact,
            inference_positions=inference_positions,
            last_ply_times_ms=last_ply_times_ms,
            created_at_ms=created_at_ms,
        )
        self.store.set(req.game_id, new_state)
        return response

    def get_clock_state(self, game_id: str) -> ClockStateResponse:
        state = self.store.get(game_id)
        if state is None:
            raise self._error(
                status_code=404,
                game_id=game_id,
                code="not_found",
                message="Unknown game_id",
                server_ply=0,
                server_fen="",
                clock=ClockState(),
            )
        self._maybe_publish_finished_event(state)
        return ClockStateResponse(
            game_id=game_id,
            server_ply=state.ply,
            server_fen=state.fen,
            clock=state.clock,
            player_color=state.player_color,
            last_ply_times_ms=state.last_ply_times_ms,
            desynced=False,
        )

    def update_clock(self, game_id: str, req: ClockUpdateRequest) -> ClockStateResponse:
        state = self.store.get(game_id)
        if state is None:
            raise self._error(
                status_code=404,
                game_id=game_id,
                code="not_found",
                message="Unknown game_id",
                server_ply=0,
                server_fen="",
                clock=ClockState(),
            )
        if req.client_ply is not None and req.client_ply != state.ply:
            raise self._error(
                status_code=409,
                game_id=game_id,
                code="desync",
                message=f"client_ply={req.client_ply} does not match server_ply={state.ply}",
                server_ply=state.ply,
                server_fen=state.fen,
                clock=state.clock,
            )
        self._maybe_publish_finished_event(state)
        state.clock = req.clock
        self.store.set(game_id, state)
        return ClockStateResponse(
            game_id=game_id,
            server_ply=state.ply,
            server_fen=state.fen,
            clock=state.clock,
            player_color=state.player_color,
            last_ply_times_ms=state.last_ply_times_ms,
            desynced=False,
        )

    def sync_clock(self, game_id: str, req: ClockSyncRequest) -> ClockStateResponse:
        state = self.store.get(game_id)
        if state is None:
            raise self._error(
                status_code=404,
                game_id=game_id,
                code="not_found",
                message="Unknown game_id",
                server_ply=0,
                server_fen="",
                clock=ClockState(),
            )
        self._maybe_publish_finished_event(state)
        desynced = False
        if req.client_ply is not None and req.client_ply != state.ply:
            desynced = True
        if req.client_fen is not None and req.client_fen != state.fen:
            desynced = True
        return ClockStateResponse(
            game_id=game_id,
            server_ply=state.ply,
            server_fen=state.fen,
            clock=state.clock,
            player_color=state.player_color,
            last_ply_times_ms=state.last_ply_times_ms,
            desynced=desynced,
        )
