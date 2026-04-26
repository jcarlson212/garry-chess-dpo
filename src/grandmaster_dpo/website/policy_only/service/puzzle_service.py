from __future__ import annotations

import logging
import os
import random
import secrets
from typing import Any

import chess
import chess.engine

from grandmaster_dpo.website.policy_only.schemas import (
    EngineConfigRequest,
    EngineLimitRequest,
    ErrorInfo,
    PuzzleAdvantage,
    PuzzleCatalogResponse,
    PuzzleCategorySummary,
    PuzzleErrorResponse,
    PuzzleMoveRequest,
    PuzzlePrecomputedHints,
    PuzzleReplayStep,
    PuzzleScenarioSummary,
    PuzzleSessionResponse,
    PuzzleStartRequest,
    PuzzleStateSummary,
)
from grandmaster_dpo.website.policy_only.service.runtime import (
    ClockState,
    _score_to_cp,
    choose_bot_move,
    fen_ply_abs,
    get_or_load_bundle,
    resolve_profile,
)
from grandmaster_dpo.website.policy_only.service.puzzle_registry import (
    PuzzleScenario,
    PuzzleScenarioRegistry,
)
from grandmaster_dpo.website.policy_only.service.puzzle_state import (
    PuzzleStateStore,
    StoredPuzzleState,
)

logger = logging.getLogger(__name__)


class PuzzleServiceError(Exception):
    def __init__(self, status_code: int, error: PuzzleErrorResponse) -> None:
        super().__init__(error.error.message)
        self.status_code = status_code
        self.error = error


class PuzzleService:
    def __init__(
        self,
        *,
        store: PuzzleStateStore,
        scenario_registry: PuzzleScenarioRegistry,
    ) -> None:
        self.store = store
        self.scenario_registry = scenario_registry
        self.branch_width = int(os.environ.get("PUZZLE_BRANCH_WIDTH", "3"))
        self.stockfish_depth = int(os.environ.get("PUZZLE_STOCKFISH_DEPTH", "12"))
        self.performance_well_cp = int(os.environ.get("PUZZLE_PERF_WELL_CP_LOSS", "25"))
        self.performance_okay_cp = int(os.environ.get("PUZZLE_PERF_OKAY_CP_LOSS", "90"))
        self.lose_cp = int(os.environ.get("PUZZLE_LOSE_CP_LOSS", "140"))

    @staticmethod
    def _eval_status(cp: int) -> str:
        if cp >= 300:
            return "winning"
        if cp >= 120:
            return "comfortable_edge"
        if cp >= 40:
            return "slight_edge"
        if cp > -40:
            return "roughly_equal"
        if cp > -120:
            return "slightly_worse"
        if cp > -300:
            return "worse"
        return "losing"

    def _performance_label(self, cp_loss: int) -> PuzzlePerformance:
        if cp_loss <= self.performance_well_cp:
            return "well"
        if cp_loss <= self.performance_okay_cp:
            return "okay"
        return "bad"

    @staticmethod
    def _terminal_summary(state: StoredPuzzleState) -> PuzzleStateSummary:
        return PuzzleStateSummary(
            status=state.status,
            performance=state.performance,
            won=state.status == "won",
            lost=state.status == "lost",
            terminated_early=state.status == "lost",
            termination_reason=state.termination_reason,
        )

    @staticmethod
    def _scenario_summary(state: StoredPuzzleState) -> PuzzleScenarioSummary:
        return PuzzleScenarioSummary(
            phase=state.phase,
            difficulty_estimate_elo=state.difficulty_estimate_elo,
            length_bin=state.length_bin,
            sampled_rollout_length_plies=state.sampled_rollout_length_plies,
        )

    @staticmethod
    def _session_id() -> str:
        return f"pz_{secrets.token_urlsafe(12)}"

    @staticmethod
    def _default_game_type_id(bot_id: str) -> str:
        return f"gm_{bot_id}_rapid"

    def _default_puzzle_engine_config(self) -> EngineConfigRequest:
        return EngineConfigRequest(
            limit=EngineLimitRequest(type="depth", value=self.stockfish_depth),
            random_seed=0,
            stockfish_multipv_topk=max(self.branch_width, 3),
            cp_gap_window=40,
            use_gibbs=False,
            engine_temp=1.0,
            cp_scale=150.0,
            cp_cap=2000,
            style_temperature=0.25,
            sample=False,
            use_timer_head=False,
        )

    def _resolved_engine_config(self, engine_config: EngineConfigRequest) -> EngineConfigRequest:
        config = engine_config.model_copy(deep=True)
        if config.limit is None and config.stockfish_engine_depth is None:
            config.limit = EngineLimitRequest(type="depth", value=self.stockfish_depth)
        if config.stockfish_multipv_topk < 1:
            config.stockfish_multipv_topk = max(self.branch_width, 3)
        return config

    @staticmethod
    def _analysis_depth(engine_config: EngineConfigRequest, fallback_depth: int) -> int:
        if engine_config.limit is not None and engine_config.limit.type == "depth":
            return max(1, int(engine_config.limit.value))
        if engine_config.stockfish_engine_depth is not None:
            return max(1, int(engine_config.stockfish_engine_depth))
        return max(1, int(fallback_depth))

    def _error(
        self,
        *,
        status_code: int,
        session_id: str,
        code: str,
        message: str,
        server_ply: int,
        server_fen: str,
    ) -> PuzzleServiceError:
        return PuzzleServiceError(
            status_code=status_code,
            error=PuzzleErrorResponse(
                session_id=session_id,
                error=ErrorInfo(code=code, message=message),
                server_ply=server_ply,
                server_fen=server_fen,
            ),
        )

    def _analyse_position(
        self,
        *,
        game_type_id: str,
        gm_name: str | None = None,
        engine_config: EngineConfigRequest,
        fen: str,
        played_move_uci: str | None = None,
    ) -> dict[str, Any]:
        profile = resolve_profile(game_type_id, gm_name)
        bundle = get_or_load_bundle(profile)
        board = chess.Board(fen)
        analysis_depth = self._analysis_depth(engine_config, self.stockfish_depth)
        multipv = max(1, int(engine_config.stockfish_multipv_topk))
        infos = bundle.stockfish.analyse(
            board,
            chess.engine.Limit(depth=analysis_depth),
            multipv=multipv,
        )

        top_moves: list[dict[str, Any]] = []
        for rank, info in enumerate(infos, start=1):
            pv = info.get("pv") or []
            score = info.get("score")
            if not pv or score is None:
                continue
            move = pv[0]
            cp = int(_score_to_cp(score))
            top_moves.append(
                {
                    "rank": rank,
                    "uci": move.uci(),
                    "san": board.san(move),
                    "cp": cp,
                }
            )
        if not top_moves:
            fallback = next(iter(board.legal_moves))
            top_moves = [{"rank": 1, "uci": fallback.uci(), "san": board.san(fallback), "cp": 0}]

        best = top_moves[0]
        played_cp = None
        played_rank = None
        if played_move_uci:
            for row in top_moves:
                if row["uci"] == played_move_uci:
                    played_cp = int(row["cp"])
                    played_rank = int(row["rank"])
                    break
            if played_cp is None:
                played_move = chess.Move.from_uci(played_move_uci)
                next_board = board.copy(stack=False)
                next_board.push(played_move)
                reply_info = bundle.stockfish.analyse(
                    next_board,
                    chess.engine.Limit(depth=max(8, analysis_depth // 2)),
                )
                score = reply_info.get("score")
                played_cp = 0 if score is None else -int(_score_to_cp(score))

        return {
            "eval_cp": int(best["cp"]),
            "eval_status": self._eval_status(int(best["cp"])),
            "best_move_uci": str(best["uci"]),
            "best_move_san": best["san"],
            "top_moves": top_moves,
            "played_move_cp": played_cp,
            "played_move_rank_if_in_topk": played_rank,
        }

    def list_categories(self) -> PuzzleCatalogResponse:
        return PuzzleCatalogResponse(
            categories=[
                PuzzleCategorySummary(category=category, count=count)
                for category, count in self.scenario_registry.list_categories()
            ]
        )

    def _response_from_state(
        self,
        *,
        state: StoredPuzzleState,
        advantage: dict[str, Any],
        player_move_uci: str = "",
        bot_move_uci: str = "",
    ) -> PuzzleSessionResponse:
        scenario = self.scenario_registry.get_by_id(state.scenario_id)
        return PuzzleSessionResponse(
            session_id=state.session_id,
            scenario_id=state.scenario_id,
            category=state.category,
            bot_id=state.bot_id,
            game_type_id=state.game_type_id,
            engine_config=state.engine_config,
            player_color=state.player_color,
            fen=state.current_fen,
            server_ply=state.current_ply,
            advantage=PuzzleAdvantage(
                eval_cp=int(advantage["eval_cp"]),
                eval_status=str(advantage["eval_status"]),
                best_move_uci=advantage.get("best_move_uci"),
                best_move_san=advantage.get("best_move_san"),
            ),
            puzzle_state=self._terminal_summary(state),
            scenario=self._scenario_summary(state),
            precomputed_hints=PuzzlePrecomputedHints(
                light_tree=scenario.light_tree,
                trajectory=scenario.trajectory,
            ),
            player_move_uci=player_move_uci,
            bot_move_uci=bot_move_uci,
            solution_replay=[
                PuzzleReplayStep.model_validate(item)
                for item in state.solution_replay
            ] if state.status != "ongoing" else [],
        )

    def start_puzzle(self, req: PuzzleStartRequest) -> PuzzleSessionResponse:
        if req.scenario_id:
            try:
                scenario = self.scenario_registry.get_by_id(req.scenario_id)
            except KeyError:
                raise self._error(
                    status_code=404,
                    session_id="",
                    code="unknown_scenario",
                    message=f"Unknown puzzle scenario: {req.scenario_id}",
                    server_ply=0,
                    server_fen="",
                )
        else:
            if not req.category:
                raise self._error(
                    status_code=400,
                    session_id="",
                    code="missing_category",
                    message="category is required when scenario_id is not provided",
                    server_ply=0,
                    server_fen="",
                )
            try:
                if req.target_elo is not None:
                    missing = [
                        name
                        for name, value in (
                            ("normal_mean", req.normal_mean),
                            ("normal_std", req.normal_std),
                            ("min_elo", req.min_elo),
                            ("max_elo", req.max_elo),
                        )
                        if value is None
                    ]
                    if missing:
                        raise self._error(
                            status_code=400,
                            session_id="",
                            code="missing_elo_distribution",
                            message=f"Missing ELO distribution fields: {', '.join(missing)}",
                            server_ply=0,
                            server_fen="",
                        )
                    if int(req.min_elo) > int(req.max_elo):
                        raise self._error(
                            status_code=400,
                            session_id="",
                            code="invalid_elo_distribution",
                            message="min_elo must be <= max_elo",
                            server_ply=0,
                            server_fen="",
                        )
                    scenario = self.scenario_registry.sample_for_category_and_target_elo(
                        req.category,
                        target_elo=int(req.target_elo),
                        normal_mean=float(req.normal_mean),
                        normal_std=float(req.normal_std),
                        min_elo=int(req.min_elo),
                        max_elo=int(req.max_elo),
                        rng=random.Random(),
                    )
                else:
                    scenario = self.scenario_registry.random_for_category(req.category)
            except KeyError:
                raise self._error(
                    status_code=404,
                    session_id="",
                    code="unknown_category",
                    message=f"Unknown puzzle category: {req.category}",
                    server_ply=0,
                    server_fen="",
                )
        requested_bot_id = req.gm_name or req.bot_id or "carlsen"
        game_type_id = self._default_game_type_id(requested_bot_id)
        try:
            gm_name = resolve_profile(game_type_id, req.gm_name or req.bot_id).gm_name
        except ValueError as exc:
            raise self._error(
                status_code=400,
                session_id="",
                code="bad_gm_name",
                message=str(exc),
                server_ply=0,
                server_fen="",
            )
        game_type_id = self._default_game_type_id(gm_name)
        engine_config = self._resolved_engine_config(req.engine_config or self._default_puzzle_engine_config())
        session_id = self._session_id()
        state = StoredPuzzleState(
            session_id=session_id,
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            bot_id=gm_name,
            game_type_id=game_type_id,
            engine_config=engine_config,
            authenticated_user_id=req.authenticated_user_id,
            player_color=scenario.player_color,
            start_fen=scenario.fen,
            current_fen=scenario.fen,
            current_ply=fen_ply_abs(scenario.fen),
            phase=scenario.phase,
            difficulty_estimate_elo=scenario.difficulty_estimate_elo,
            length_bin=scenario.length_bin,
            sampled_rollout_length_plies=scenario.sampled_rollout_length_plies,
            target_full_plies=max(1, scenario.sampled_rollout_length_plies),
        )
        self.store.set(session_id, state)
        advantage = {
            "eval_cp": scenario.root_eval_cp,
            "eval_status": scenario.root_eval_status,
            "best_move_uci": ((scenario.light_tree.get("top_moves") or [{}])[0]).get("uci"),
            "best_move_san": ((scenario.light_tree.get("top_moves") or [{}])[0]).get("san"),
        }
        logger.info(
            "puzzle_started session_id=%s scenario_id=%s category=%s bot_id=%s",
            session_id,
            scenario.scenario_id,
            scenario.category,
            gm_name,
        )
        return self._response_from_state(state=state, advantage=advantage)

    def get_puzzle(self, session_id: str) -> PuzzleSessionResponse:
        state = self.store.get(session_id)
        if state is None:
            raise self._error(
                status_code=404,
                session_id=session_id,
                code="not_found",
                message="Unknown puzzle session",
                server_ply=0,
                server_fen="",
            )
        advantage = self._analyse_position(
            game_type_id=state.game_type_id,
            engine_config=state.engine_config,
            fen=state.current_fen,
        )
        return self._response_from_state(state=state, advantage=advantage)

    def play_move(self, session_id: str, req: PuzzleMoveRequest) -> PuzzleSessionResponse:
        state = self.store.get(session_id)
        if state is None:
            raise self._error(
                status_code=404,
                session_id=session_id,
                code="not_found",
                message="Unknown puzzle session",
                server_ply=0,
                server_fen="",
            )
        if state.status != "ongoing":
            terminal_advantage = self._analyse_position(
                game_type_id=state.game_type_id,
                gm_name=state.bot_id,
                engine_config=state.engine_config,
                fen=state.current_fen,
            )
            return self._response_from_state(state=state, advantage=terminal_advantage)
        if req.client_ply != state.current_ply:
            raise self._error(
                status_code=409,
                session_id=session_id,
                code="desync",
                message=f"client_ply={req.client_ply} does not match server_ply={state.current_ply}",
                server_ply=state.current_ply,
                server_fen=state.current_fen,
            )
        if req.pre_move_fen != state.current_fen:
            raise self._error(
                status_code=409,
                session_id=session_id,
                code="desync",
                message="client pre_move_fen does not match server_fen",
                server_ply=state.current_ply,
                server_fen=state.current_fen,
            )

        board = chess.Board(state.current_fen)
        expected_turn = chess.WHITE if state.player_color == "white" else chess.BLACK
        if board.turn != expected_turn:
            raise self._error(
                status_code=409,
                session_id=session_id,
                code="not_players_turn",
                message="Puzzle session is not at a player-turn position",
                server_ply=state.current_ply,
                server_fen=state.current_fen,
            )
        try:
            player_move = chess.Move.from_uci(req.client_uci)
        except Exception:
            raise self._error(
                status_code=400,
                session_id=session_id,
                code="illegal_move",
                message="Invalid UCI format",
                server_ply=state.current_ply,
                server_fen=state.current_fen,
            )
        if player_move not in board.legal_moves:
            raise self._error(
                status_code=400,
                session_id=session_id,
                code="illegal_move",
                message="Move is not legal in current position",
                server_ply=state.current_ply,
                server_fen=state.current_fen,
            )

        user_position_analysis = self._analyse_position(
                game_type_id=state.game_type_id,
                gm_name=state.bot_id,
                engine_config=state.engine_config,
                fen=state.current_fen,
                played_move_uci=req.client_uci,
        )
        player_move_san = board.san(player_move)
        played_move_cp = int(user_position_analysis["played_move_cp"] or user_position_analysis["eval_cp"])
        cp_loss = int(user_position_analysis["eval_cp"]) - played_move_cp
        state.max_user_cp_loss = max(state.max_user_cp_loss, cp_loss)
        state.performance = self._performance_label(state.max_user_cp_loss)
        state.solution_replay.append(
            PuzzleReplayStep(
                fen=state.current_fen,
                best_move_uci=user_position_analysis.get("best_move_uci"),
                best_move_san=user_position_analysis.get("best_move_san"),
                position_eval_cp=int(user_position_analysis["eval_cp"]),
                position_eval_status=str(user_position_analysis["eval_status"]),
                player_move_uci=req.client_uci,
                player_move_san=player_move_san,
                cp_loss=cp_loss,
                performance=state.performance,
            ).model_dump(mode="python")
        )

        board.push(player_move)
        state.move_history_uci.append(req.client_uci)
        state.current_ply += 1
        state.plies_played += 1

        player_won = board.is_game_over(claim_draw=True) and (
            (board.is_checkmate() and board.turn != expected_turn)
        )
        player_lost = board.is_game_over(claim_draw=True) and not player_won
        if cp_loss >= self.lose_cp:
            state.status = "lost"
            state.termination_reason = "missed_critical_move"
            state.current_fen = board.fen()
            self.store.set(session_id, state)
            terminal_advantage = self._analyse_position(
                game_type_id=state.game_type_id,
                gm_name=state.bot_id,
                engine_config=state.engine_config,
                fen=state.current_fen,
            )
            return self._response_from_state(
                state=state,
                advantage=terminal_advantage,
                player_move_uci=req.client_uci,
            )
        if player_won:
            state.status = "won"
            state.termination_reason = "checkmate"
            state.current_fen = board.fen()
            self.store.set(session_id, state)
            terminal_advantage = {
                "eval_cp": 100000 if state.player_color == "white" else -100000,
                "eval_status": "winning",
                "best_move_uci": None,
                "best_move_san": None,
            }
            return self._response_from_state(
                state=state,
                advantage=terminal_advantage,
                player_move_uci=req.client_uci,
            )
        if player_lost:
            state.status = "lost"
            state.termination_reason = "game_over"
            state.current_fen = board.fen()
            self.store.set(session_id, state)
            terminal_advantage = {
                "eval_cp": -100000 if state.player_color == "white" else 100000,
                "eval_status": "losing",
                "best_move_uci": None,
                "best_move_san": None,
            }
            return self._response_from_state(
                state=state,
                advantage=terminal_advantage,
                player_move_uci=req.client_uci,
            )

        if state.plies_played >= state.target_full_plies:
            state.status = "won"
            state.termination_reason = "target_completed"
            state.current_fen = board.fen()
            self.store.set(session_id, state)
            terminal_advantage = self._analyse_position(
                game_type_id=state.game_type_id,
                engine_config=state.engine_config,
                fen=state.current_fen,
            )
            return self._response_from_state(
                state=state,
                advantage=terminal_advantage,
                player_move_uci=req.client_uci,
            )

        bot_result = choose_bot_move(
            bundle=get_or_load_bundle(resolve_profile(state.game_type_id, state.bot_id)),
            fen=board.fen(),
            clock=ClockState(white_ms=0, black_ms=0),
            last_ply_times_ms=[],
            engine_config=state.engine_config,
            start_fen=state.start_fen,
            played_moves_uci=state.move_history_uci,
        )
        bot_move = chess.Move.from_uci(bot_result.move_uci)
        board.push(bot_move)
        state.move_history_uci.append(bot_result.move_uci)
        state.current_ply += 1
        state.plies_played += 1
        state.current_fen = board.fen()

        if board.is_game_over(claim_draw=True):
            winner_is_player = board.is_checkmate() and board.turn != expected_turn
            state.status = "won" if winner_is_player else "lost"
            state.termination_reason = "game_over"
            self.store.set(session_id, state)
            terminal_advantage = {
                "eval_cp": (100000 if state.player_color == "white" else -100000) if winner_is_player else (-100000 if state.player_color == "white" else 100000),
                "eval_status": "winning" if winner_is_player else "losing",
                "best_move_uci": None,
                "best_move_san": None,
            }
            return self._response_from_state(
                state=state,
                advantage=terminal_advantage,
                player_move_uci=req.client_uci,
                bot_move_uci=bot_result.move_uci,
            )

        if state.plies_played >= state.target_full_plies:
            state.status = "won"
            state.termination_reason = "target_completed"

        self.store.set(session_id, state)
        next_advantage = self._analyse_position(
                game_type_id=state.game_type_id,
                gm_name=state.bot_id,
                engine_config=state.engine_config,
                fen=state.current_fen,
            )
        return self._response_from_state(
            state=state,
            advantage=next_advantage,
            player_move_uci=req.client_uci,
            bot_move_uci=bot_result.move_uci,
        )
