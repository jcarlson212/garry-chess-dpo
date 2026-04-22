from __future__ import annotations

import logging

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
from grandmaster_dpo.website.policy_only.service.runtime import (
    choose_bot_move,
    fen_ply_abs,
    game_status_from_board,
    get_or_load_bundle,
    resolve_profile,
)
from grandmaster_dpo.website.policy_only.service.state import GameStateStore, StoredGameState

logger = logging.getLogger(__name__)


class GameServiceError(Exception):
    def __init__(self, status_code: int, error: ErrorResponse) -> None:
        super().__init__(error.error.message)
        self.status_code = status_code
        self.error = error


class PolicyOnlyGameService:
    def __init__(self, store: GameStateStore) -> None:
        self.store = store

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
            server_fen = req.pre_move_fen
            server_ply = req.client_ply if req.client_ply >= 0 else fen_ply_abs(req.pre_move_fen)
            return None, player_color, server_fen, server_ply
        return state, state.player_color, state.fen, state.ply

    def play_turn(self, req: GameRequest) -> GameResponse:
        if not req.game_id or not req.pre_move_fen or not req.game_type_id:
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

        if board.is_game_over(claim_draw=True):
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
        last_ply_times_ms = list(state.last_ply_times_ms) if state is not None else []
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
            if board.turn == chess.WHITE and clock.white_ms is not None:
                clock = ClockState(white_ms=max(0, int(clock.white_ms) - elapsed_ms), black_ms=clock.black_ms)
            elif board.turn == chess.BLACK and clock.black_ms is not None:
                clock = ClockState(white_ms=clock.white_ms, black_ms=max(0, int(clock.black_ms) - elapsed_ms))
            last_ply_times_ms.append(elapsed_ms)
            last_ply_times_ms = last_ply_times_ms[-5:]

            board.push(player_move)
            server_ply += 1

            if board.is_game_over(claim_draw=True):
                status = game_status_from_board(board)
                new_state = StoredGameState(
                    game_id=req.game_id,
                    fen=board.fen(),
                    ply=server_ply,
                    player_color=player_color,
                    clock=clock,
                    bot_id=req.bot_id,
                    game_type_id=req.game_type_id,
                    last_ply_times_ms=last_ply_times_ms,
                )
                self.store.set(req.game_id, new_state)
                return GameResponse(
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
                        gm_name=resolve_profile(req.game_type_id).gm_name,
                    ),
                )

        profile = resolve_profile(req.game_type_id)
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
        )
        try:
            bot_move = chess.Move.from_uci(move_result.move_uci)
        except Exception:
            bot_move = next(iter(board.legal_moves))
        if bot_move not in board.legal_moves:
            bot_move = next(iter(board.legal_moves))

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
        status = game_status_from_board(board)

        new_state = StoredGameState(
            game_id=req.game_id,
            fen=board.fen(),
            ply=server_ply,
            player_color=player_color,
            clock=clock,
            bot_id=req.bot_id,
            game_type_id=req.game_type_id,
            last_ply_times_ms=last_ply_times_ms,
        )
        self.store.set(req.game_id, new_state)

        return GameResponse(
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
