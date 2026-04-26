from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import boto3
import chess
import chess.engine

from grandmaster_dpo.eval.stockfish_helpers import make_stockfish
from grandmaster_dpo.website.policy_only.db.mapper import PolicyOnlyGamesTable
from grandmaster_dpo.website.policy_only.db.models import (
    BoardStateAnalysisRecord,
    CandidateMoveRecord,
    GameInferencePositionRecord,
    GamePostgameBoardStateRecord,
    GamePostgameMoveRecord,
    GamePostgameSummaryRecord,
    GameRecord,
    InferenceConfigRecord,
    InferenceStatusRecord,
    MoveAnalysisRecord,
    PostgameStatusRecord,
    PostgameSummaryMetricsRecord,
)

logger = logging.getLogger(__name__)

POSTGAME_ANALYSIS_DEPTH = 22
POSTGAME_ANALYSIS_PURPOSE = "postgame_engine_analysis"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _score_to_cp(score: chess.engine.PovScore, mate_score: int = 100_000) -> int:
    rel = score.relative
    cp = rel.score(mate_score=mate_score)
    if cp is None:
        mate = rel.mate()
        if mate is not None:
            return mate_score if mate > 0 else -mate_score
        return 0
    return int(cp)


def unwrap_sqs_sns_message(message_body: str) -> dict[str, Any]:
    body = json.loads(message_body)
    if isinstance(body, dict) and isinstance(body.get("Message"), str):
        return json.loads(body["Message"])
    if isinstance(body, dict):
        return body
    raise ValueError("Unsupported SQS message body format")


class PostgameAnalyzer(Protocol):
    def analyze(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[GamePostgameBoardStateRecord], list[GamePostgameMoveRecord], GamePostgameSummaryRecord]:
        ...


class StockfishPostgameAnalyzer:
    def __init__(
        self,
        *,
        depth: int = POSTGAME_ANALYSIS_DEPTH,
        multipv: int | None = None,
        stockfish_path: str | None = None,
        stockfish_threads: int | None = None,
        stockfish_hash_mb: int | None = None,
        stockfish_timeout_s: float | None = None,
    ) -> None:
        self.depth = int(depth)
        self.multipv = max(2, int(multipv or os.environ.get("POSTGAME_STOCKFISH_MULTIPV", "5")))
        self.engine = make_stockfish(
            stockfish_path or os.environ.get("STOCKFISH_PATH", "/opt/bin/stockfish"),
            threads=int(stockfish_threads or os.environ.get("POSTGAME_STOCKFISH_THREADS", "4")),
            hash_mb=int(stockfish_hash_mb or os.environ.get("POSTGAME_STOCKFISH_HASH_MB", "512")),
            timeout=float(stockfish_timeout_s or os.environ.get("POSTGAME_STOCKFISH_TIMEOUT_S", "60.0")),
        )

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            logger.exception("postgame_stockfish_shutdown_failed")

    def _top_moves(self, board: chess.Board) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        infos = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=self.multipv,
        )
        candidates: list[dict[str, Any]] = []
        for info in infos:
            pv = info.get("pv")
            score = info.get("score")
            if not pv or score is None:
                continue
            move = pv[0]
            candidates.append(
                {
                    "uci": move.uci(),
                    "cp": _score_to_cp(score),
                    "mate": score.relative.mate(),
                    "multipv_rank": int(info.get("multipv")) if info.get("multipv") is not None else None,
                    "pv_uci": [pv_move.uci() for pv_move in pv[:8]],
                    "depth": int(info.get("depth")) if info.get("depth") is not None else None,
                    "seldepth": int(info.get("seldepth")) if info.get("seldepth") is not None else None,
                    "nodes": int(info.get("nodes")) if info.get("nodes") is not None else None,
                    "nps": int(info.get("nps")) if info.get("nps") is not None else None,
                    "time_ms": (
                        int(round(float(info.get("time")) * 1000.0))
                        if info.get("time") is not None
                        else None
                    ),
                    "tbhits": int(info.get("tbhits")) if info.get("tbhits") is not None else None,
                }
            )
        if not candidates:
            raise RuntimeError("Stockfish returned no candidates during postgame analysis")
        best = max(candidates, key=lambda item: int(item["cp"]))
        return candidates, best

    @staticmethod
    def _classify_cp_loss(cp_loss: int) -> str:
        if cp_loss <= 10:
            return "best"
        if cp_loss <= 80:
            return "inaccuracy"
        if cp_loss <= 200:
            return "mistake"
        return "blunder"

    @staticmethod
    def _accuracy_score(cp_loss: int) -> float:
        return round(max(0.0, 1.0 - (float(cp_loss) / 45.0)), 2)

    def analyze(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[GamePostgameBoardStateRecord], list[GamePostgameMoveRecord], GamePostgameSummaryRecord]:
        game = payload["game"]
        game_id = str(game["game_id"])
        board = chess.Board(str(game["start_fen"]))
        moves_compact = list(game.get("moves_compact") or [])
        board_state_records: list[GamePostgameBoardStateRecord] = []
        move_records: list[GamePostgameMoveRecord] = []
        accuracy_scores: list[tuple[str, float]] = []
        best_count = 0
        inaccuracy_count = 0
        mistake_count = 0
        blunder_count = 0

        for move_item in moves_compact:
            ply = int(move_item["ply"])
            fen_before = board.fen()
            side_to_move = "white" if board.turn == chess.WHITE else "black"
            move = chess.Move.from_uci(str(move_item["uci"]))
            if move not in board.legal_moves:
                raise ValueError(f"Illegal stored move {move.uci()} at ply={ply} for game_id={game_id}")

            candidates, best = self._top_moves(board)
            top_moves = [CandidateMoveRecord(**candidate) for candidate in candidates]
            board_state_records.append(
                GamePostgameBoardStateRecord(
                    PK=f"GAME#{game_id}",
                    SK=f"POSTGAME#BOARD_STATE#DEPTH#{self.depth}#PLY#{ply:04d}",
                    entity_type="GAME_POSTGAME_BOARD_STATE",
                    game_id=game_id,
                    ply=ply,
                    analysis_depth=self.depth,
                    analysis_purpose=POSTGAME_ANALYSIS_PURPOSE,
                    fen_before=fen_before,
                    side_to_move=side_to_move,
                    board_state_analysis=BoardStateAnalysisRecord(
                        engine_name="stockfish",
                        engine_depth=self.depth,
                        engine_nodes=sum(candidate.nodes or 0 for candidate in top_moves) or None,
                        engine_time_ms=max((candidate.time_ms or 0 for candidate in top_moves), default=0) or None,
                        position_eval_cp=int(best["cp"]),
                        best_move_uci=str(best["uci"]),
                        candidate_count=len(top_moves),
                        pv_uci=list(best.get("pv_uci") or []),
                    ),
                    top_moves=top_moves,
                    requested_at_iso=payload.get("event_created_at_iso"),
                    completed_at_iso=_now_iso(),
                    version=1,
                )
            )

            board_after = board.copy(stack=False)
            played_san = board.san(move)
            board_after.push(move)
            after_info = self.engine.analyse(board_after, chess.engine.Limit(depth=self.depth))
            after_score = after_info.get("score")
            if after_score is None:
                raise RuntimeError(f"Stockfish returned no played-move score for game_id={game_id} ply={ply}")
            eval_after_cp = -_score_to_cp(after_score)
            best_cp = int(best["cp"])
            cp_loss = max(0, best_cp - eval_after_cp)
            classification = self._classify_cp_loss(cp_loss)
            accuracy_score = self._accuracy_score(cp_loss)
            if classification == "best":
                best_count += 1
            elif classification == "inaccuracy":
                inaccuracy_count += 1
            elif classification == "mistake":
                mistake_count += 1
            else:
                blunder_count += 1
            accuracy_scores.append((str(move_item["color"]), accuracy_score))

            played_rank = next(
                (
                    idx
                    for idx, candidate in enumerate(
                        sorted(candidates, key=lambda item: int(item["cp"]), reverse=True),
                        start=1,
                    )
                    if str(candidate["uci"]) == move.uci()
                ),
                None,
            )
            move_records.append(
                GamePostgameMoveRecord(
                    PK=f"GAME#{game_id}",
                    SK=f"POSTGAME#MOVE#DEPTH#{self.depth}#PLY#{ply:04d}",
                    entity_type="GAME_POSTGAME_MOVE",
                    game_id=game_id,
                    ply=ply,
                    analysis_depth=self.depth,
                    analysis_purpose=POSTGAME_ANALYSIS_PURPOSE,
                    fen_before=fen_before,
                    played_move_uci=move.uci(),
                    played_move_san=played_san,
                    actor_id=str(move_item["actor_id"]),
                    actor_type=str(move_item["actor_type"]),
                    move_analysis=MoveAnalysisRecord(
                        engine_name="stockfish",
                        engine_depth=self.depth,
                        engine_nodes=int(after_info.get("nodes")) if after_info.get("nodes") is not None else None,
                        engine_time_ms=(
                            int(round(float(after_info.get("time")) * 1000.0))
                            if after_info.get("time") is not None
                            else None
                        ),
                        eval_before_cp=best_cp,
                        eval_after_cp=eval_after_cp,
                        best_move_uci=str(best["uci"]),
                        played_move_rank=played_rank,
                        cp_loss=cp_loss,
                        classification=classification,
                        is_best=(move.uci() == str(best["uci"])),
                        is_blunder=(classification == "blunder"),
                        is_mistake=(classification == "mistake"),
                        is_inaccuracy=(classification == "inaccuracy"),
                        accuracy_score=accuracy_score,
                    ),
                    requested_at_iso=payload.get("event_created_at_iso"),
                    completed_at_iso=_now_iso(),
                    version=1,
                )
            )
            board.push(move)

        all_scores = [score for _, score in accuracy_scores]
        white_scores = [score for color, score in accuracy_scores if color == "w"]
        black_scores = [score for color, score in accuracy_scores if color == "b"]
        summary = GamePostgameSummaryRecord(
            PK=f"GAME#{game_id}",
            SK=f"POSTGAME#SUMMARY#DEPTH#{self.depth}#PURPOSE#{POSTGAME_ANALYSIS_PURPOSE}",
            entity_type="GAME_POSTGAME_SUMMARY",
            game_id=game_id,
            analysis_depth=self.depth,
            analysis_purpose=POSTGAME_ANALYSIS_PURPOSE,
            analysis_state="ready",
            summary=PostgameSummaryMetricsRecord(
                sample_count=len(accuracy_scores),
                accuracy_percent=(round(sum(all_scores) / len(all_scores) * 100.0, 1) if all_scores else None),
                white_accuracy_percent=(
                    round(sum(white_scores) / len(white_scores) * 100.0, 1) if white_scores else None
                ),
                black_accuracy_percent=(
                    round(sum(black_scores) / len(black_scores) * 100.0, 1) if black_scores else None
                ),
                best_count=best_count,
                inaccuracy_count=inaccuracy_count,
                mistake_count=mistake_count,
                blunder_count=blunder_count,
            ),
            outputs_s3_key=None,
            requested_at_iso=payload.get("event_created_at_iso"),
            completed_at_iso=_now_iso(),
            version=1,
        )
        return board_state_records, move_records, summary


@dataclass
class ProcessResult:
    game_id: str
    inserted_game: bool
    inserted_inference_positions: int
    inserted_board_states: int
    inserted_move_analyses: int
    inserted_summary: bool
    skipped_existing_summary: bool = False


class GameFinishedPostprocessor:
    def __init__(
        self,
        *,
        mapper: PolicyOnlyGamesTable,
        analyzer: PostgameAnalyzer,
        analysis_depth: int = POSTGAME_ANALYSIS_DEPTH,
    ) -> None:
        self.mapper = mapper
        self.analyzer = analyzer
        self.analysis_depth = analysis_depth

    @staticmethod
    def _game_pk(game_id: str) -> str:
        return f"GAME#{game_id}"

    def _build_game_record(self, payload: dict[str, Any], *, analysis_state: str, version: int) -> GameRecord:
        game = payload["game"]
        game_id = str(game["game_id"])
        created_at_ms = int(game.get("created_at_ms") or 0)
        ended_at_ms = int(game.get("ended_at_ms") or created_at_ms or 0)
        inference_positions = list(game.get("inference_positions") or [])
        latest_inference = inference_positions[-1] if inference_positions else None
        return GameRecord(
            PK=self._game_pk(game_id),
            SK="GAME",
            entity_type="GAME",
            game_id=game_id,
            white_actor_id=str(game["white_actor_id"]),
            white_actor_type=str(game["white_actor_type"]),
            black_actor_id=str(game["black_actor_id"]),
            black_actor_type=str(game["black_actor_type"]),
            white_label=game.get("white_label"),
            black_label=game.get("black_label"),
            white_family=game.get("white_family"),
            black_family=game.get("black_family"),
            white_elo_estimate=game.get("white_elo_estimate"),
            black_elo_estimate=game.get("black_elo_estimate"),
            white_actor_username=game.get("white_actor_username"),
            black_actor_username=game.get("black_actor_username"),
            white_user_id=(
                str(game["white_user_id"])
                if game.get("white_user_id") is not None
                else str(game["white_actor_id"]) if str(game.get("white_actor_type")) == "user" else None
            ),
            black_user_id=(
                str(game["black_user_id"])
                if game.get("black_user_id") is not None
                else str(game["black_actor_id"]) if str(game.get("black_actor_type")) == "user" else None
            ),
            creation_time_since_epoch_ms=created_at_ms,
            ended_time_since_epoch_ms=ended_at_ms,
            created_at_iso=str(game["created_at_iso"]),
            updated_at_iso=str(game.get("ended_at_iso") or payload.get("event_created_at_iso") or _now_iso()),
            mode_kind=str(game.get("mode_kind") or "unknown"),
            mode_variant=str(game.get("mode_variant") or ""),
            time_control_id=str(game.get("time_control_id") or ""),
            initial_time_ms=int(game.get("initial_time_ms") or 0),
            increment_ms=int(game.get("increment_ms") or 0),
            result=str(game.get("result") or "unknown"),
            white_result=str(game.get("white_result") or "unknown"),
            black_result=str(game.get("black_result") or "unknown"),
            winner_actor_id=game.get("winner_actor_id"),
            winner_actor_color=game.get("winner_actor_color"),
            termination_reason=game.get("termination_reason"),
            start_fen=str(game["start_fen"]),
            final_fen=str(game["final_fen"]),
            move_count=int(game.get("move_count") or len(game.get("moves_compact") or [])),
            moves_compact=list(game.get("moves_compact") or []),
            inference_status=InferenceStatusRecord(
                has_inference_traces=bool(inference_positions),
                inference_trace_count=len(inference_positions),
                latest_inference_config=(
                    InferenceConfigRecord.model_validate(latest_inference["inference_config"])
                    if latest_inference and latest_inference.get("inference_config")
                    else None
                ),
            ),
            postgame_analysis_status=PostgameStatusRecord(
                analysis_state=analysis_state,
                latest_requested_depth=self.analysis_depth,
                latest_completed_depth=self.analysis_depth if analysis_state == "ready" else None,
                finalized_depths=[self.analysis_depth] if analysis_state == "ready" else [],
                analysis_updated_at=(
                    str(game.get("ended_at_iso") or payload.get("event_created_at_iso") or _now_iso())
                    if analysis_state == "ready"
                    else None
                ),
            ),
            authoritative_status="final",
            is_ranked=bool(game.get("is_ranked") or False),
            visibility=str(game.get("visibility") or "private"),
            version=version,
            GSI1PK=self.mapper.player_gsi_pk(str(game["white_actor_id"])),
            GSI1SK=f"TS#{created_at_ms}#COLOR#white#GAME#{game_id}",
            GSI2PK=self.mapper.pair_gsi_pk(
                str(game["white_actor_type"]),
                str(game["white_actor_id"]),
                str(game["black_actor_type"]),
                str(game["black_actor_id"]),
            ),
            GSI2SK=f"TS#{created_at_ms}#GAME#{game_id}",
        )

    def _build_inference_records(self, payload: dict[str, Any]) -> list[GameInferencePositionRecord]:
        game = payload["game"]
        game_id = str(game["game_id"])
        records: list[GameInferencePositionRecord] = []
        for raw in game.get("inference_positions") or []:
            records.append(
                GameInferencePositionRecord(
                    PK=self._game_pk(game_id),
                    SK=f"INFERENCE#PLY#{int(raw['ply']):04d}",
                    entity_type="GAME_INFERENCE_POSITION",
                    game_id=game_id,
                    ply=int(raw["ply"]),
                    fen_before=str(raw["fen_before"]),
                    side_to_move=str(raw["side_to_move"]),
                    actor_id=str(raw["actor_id"]),
                    actor_type=str(raw["actor_type"]),
                    selected_move_uci=str(raw["selected_move_uci"]),
                    selected_move_san=raw.get("selected_move_san"),
                    selected_move_probability=raw.get("selected_move_probability"),
                    inference_config=InferenceConfigRecord.model_validate(raw.get("inference_config") or {}),
                    board_state_analysis=BoardStateAnalysisRecord.model_validate(
                        raw.get("board_state_analysis") or {}
                    ),
                    candidate_moves=[
                        CandidateMoveRecord.model_validate(
                            self._normalize_candidate_move(candidate)
                        )
                        for candidate in (raw.get("candidate_moves") or [])
                    ],
                    created_at_iso=str(raw.get("created_at_iso") or payload.get("event_created_at_iso") or _now_iso()),
                    version=1,
                )
            )
        return records

    @staticmethod
    def _normalize_candidate_move(candidate: Any) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            raise TypeError(f"candidate move must be a dict, got {type(candidate)!r}")
        normalized = dict(candidate)
        if "probability" not in normalized and "prob" in normalized:
            normalized["probability"] = normalized.pop("prob")
        return normalized

    def process_payload(self, payload: dict[str, Any]) -> ProcessResult:
        if payload.get("event_type") != "game_finished":
            raise ValueError(f"Unsupported event_type: {payload.get('event_type')}")
        game = payload.get("game")
        if not isinstance(game, dict):
            raise ValueError("Finished-game payload missing `game` object")
        game_id = str(game.get("game_id") or payload.get("game_id") or "")
        if not game_id:
            raise ValueError("Finished-game payload missing game_id")

        logger.info(
            "postprocess_game_started game_id=%s event_key=%s source=%s",
            game_id,
            payload.get("event_key"),
            payload.get("source"),
        )

        existing_summary = self.mapper.get_postgame_summary(
            game_id,
            analysis_depth=self.analysis_depth,
            analysis_purpose=POSTGAME_ANALYSIS_PURPOSE,
        )
        if existing_summary is not None and existing_summary.analysis_state == "ready":
            logger.info(
                "postprocess_game_skipped_existing_summary game_id=%s depth=%s",
                game_id,
                self.analysis_depth,
            )
            return ProcessResult(
                game_id=game_id,
                inserted_game=False,
                inserted_inference_positions=0,
                inserted_board_states=0,
                inserted_move_analyses=0,
                inserted_summary=False,
                skipped_existing_summary=True,
            )

        queued_game = self._build_game_record(payload, analysis_state="queued", version=1)
        inserted_game = self.mapper.put_game_record(queued_game)
        inference_records = self._build_inference_records(payload)
        inserted_inference = sum(1 for record in inference_records if self.mapper.put_inference_position(record))

        board_state_records, move_records, summary_record = self.analyzer.analyze(payload)
        inserted_board_states = sum(
            1 for record in board_state_records if self.mapper.put_postgame_board_state(record)
        )
        inserted_move_analyses = sum(1 for record in move_records if self.mapper.put_postgame_move(record))
        inserted_summary = self.mapper.put_postgame_summary(summary_record)

        ready_game = self._build_game_record(payload, analysis_state="ready", version=2)
        ready_game.updated_at_iso = summary_record.completed_at_iso or _now_iso()
        ready_game.postgame_analysis_status = PostgameStatusRecord(
            analysis_state="ready",
            latest_requested_depth=self.analysis_depth,
            latest_completed_depth=self.analysis_depth,
            finalized_depths=[self.analysis_depth],
            analysis_updated_at=summary_record.completed_at_iso or _now_iso(),
        )
        self.mapper.put_game_record(ready_game)

        logger.info(
            "postprocess_game_completed game_id=%s inserted_game=%s inference=%s board_states=%s moves=%s summary=%s",
            game_id,
            inserted_game,
            inserted_inference,
            inserted_board_states,
            inserted_move_analyses,
            inserted_summary,
        )
        return ProcessResult(
            game_id=game_id,
            inserted_game=inserted_game,
            inserted_inference_positions=inserted_inference,
            inserted_board_states=inserted_board_states,
            inserted_move_analyses=inserted_move_analyses,
            inserted_summary=inserted_summary,
        )


def make_default_postprocessor() -> GameFinishedPostprocessor:
    mapper = PolicyOnlyGamesTable()
    analyzer = StockfishPostgameAnalyzer(depth=int(os.environ.get("POSTGAME_ANALYSIS_DEPTH", "22")))
    return GameFinishedPostprocessor(mapper=mapper, analyzer=analyzer)
