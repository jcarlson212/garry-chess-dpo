from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class DbModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ActorType = Literal["user", "bot", "guest", "system", "imported"]
VisibilityType = Literal["private", "public", "unlisted"]
OutcomeType = Literal["white_win", "black_win", "draw", "aborted", "unknown"]
PlayerResultType = Literal["win", "loss", "draw", "aborted", "unknown"]
AnalysisStateType = Literal["not_started", "queued", "running", "ready", "failed"]
EntityType = Literal[
    "GAME",
    "GAME_INFERENCE_POSITION",
    "GAME_POSTGAME_BOARD_STATE",
    "GAME_POSTGAME_MOVE",
    "GAME_POSTGAME_SUMMARY",
]


class CompactMoveRecord(DbModel):
    ply: int
    color: Literal["w", "b"]
    actor_id: str
    actor_type: ActorType
    uci: str
    san: str
    move_time_ms: Optional[int] = None
    clock_after_white_ms: Optional[int] = None
    clock_after_black_ms: Optional[int] = None
    from_sq: Optional[str] = None
    to_sq: Optional[str] = None
    promotion: Optional[str] = None


class InferenceConfigRecord(DbModel):
    policy_model_name: Optional[str] = None
    use_gibbs: bool = False
    lam: Optional[float] = None
    temperature: Optional[float] = None
    sample: Optional[bool] = None
    cp_gap_window: Optional[int] = None
    stockfish_multipv_topk: Optional[int] = None
    timer_head_enabled: Optional[bool] = None
    requested_depth: Optional[int] = None
    requested_time_ms: Optional[int] = None
    requested_nodes: Optional[int] = None
    draw_penalties: dict[str, Any] = Field(default_factory=dict)


class InferenceStatusRecord(DbModel):
    has_inference_traces: bool = False
    inference_trace_count: int = 0
    latest_inference_config: Optional[InferenceConfigRecord] = None


class CandidateMoveRecord(DbModel):
    uci: str
    probability: Optional[float] = None
    in_cp_gap_window: Optional[bool] = None
    cp: Optional[int] = None
    adjusted_cp: Optional[int] = None
    draw_penalty_cp: Optional[int] = None
    repetition_x2_penalty_cp: Optional[int] = None
    one_move_from_draw_penalty_cp: Optional[int] = None
    mate: Optional[int] = None
    multipv_rank: Optional[int] = None
    pv_uci: list[str] = Field(default_factory=list)
    depth: Optional[int] = None
    seldepth: Optional[int] = None
    nodes: Optional[int] = None
    nps: Optional[int] = None
    time_ms: Optional[int] = None
    tbhits: Optional[int] = None


class BoardStateAnalysisRecord(DbModel):
    engine_name: str
    engine_depth: Optional[int] = None
    engine_nodes: Optional[int] = None
    engine_time_ms: Optional[int] = None
    position_eval_cp: Optional[int] = None
    best_move_uci: Optional[str] = None
    candidate_count: Optional[int] = None
    pv_uci: list[str] = Field(default_factory=list)


class MoveAnalysisRecord(DbModel):
    engine_name: str
    engine_depth: Optional[int] = None
    engine_nodes: Optional[int] = None
    engine_time_ms: Optional[int] = None
    eval_before_cp: Optional[int] = None
    eval_after_cp: Optional[int] = None
    best_move_uci: Optional[str] = None
    played_move_rank: Optional[int] = None
    cp_loss: Optional[int] = None
    classification: Optional[str] = None
    is_best: Optional[bool] = None
    is_blunder: Optional[bool] = None
    is_mistake: Optional[bool] = None
    is_inaccuracy: Optional[bool] = None
    accuracy_score: Optional[float] = None


class PostgameStatusRecord(DbModel):
    analysis_state: AnalysisStateType
    latest_requested_depth: Optional[int] = None
    latest_completed_depth: Optional[int] = None
    finalized_depths: list[int] = Field(default_factory=list)
    analysis_updated_at: Optional[str] = None


class PostgameSummaryMetricsRecord(DbModel):
    sample_count: Optional[int] = None
    accuracy_percent: Optional[float] = None
    white_accuracy_percent: Optional[float] = None
    black_accuracy_percent: Optional[float] = None
    best_count: Optional[int] = None
    inaccuracy_count: Optional[int] = None
    mistake_count: Optional[int] = None
    blunder_count: Optional[int] = None


class GameMetadataRecord(DbModel):
    source: Optional[str] = None
    lifecycle: Optional[str] = None
    rated: Optional[bool] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class GamesTableItem(DbModel):
    PK: str
    SK: str
    entity_type: EntityType
    game_id: str
    version: int = 1


class GameRecord(GamesTableItem):
    entity_type: Literal["GAME"] = "GAME"
    white_actor_id: str
    white_actor_type: ActorType
    black_actor_id: str
    black_actor_type: ActorType
    white_label: Optional[str] = None
    black_label: Optional[str] = None
    white_family: Optional[str] = None
    black_family: Optional[str] = None
    white_elo_estimate: Optional[int] = None
    black_elo_estimate: Optional[int] = None
    white_actor_username: Optional[str] = None
    black_actor_username: Optional[str] = None
    white_user_id: Optional[str] = None
    black_user_id: Optional[str] = None
    creation_time_since_epoch_ms: int
    ended_time_since_epoch_ms: Optional[int] = None
    created_at_iso: str
    updated_at_iso: str
    mode_kind: str
    mode_variant: str
    time_control_id: str
    initial_time_ms: int
    increment_ms: int = 0
    result: OutcomeType = "unknown"
    white_result: PlayerResultType = "unknown"
    black_result: PlayerResultType = "unknown"
    winner_actor_id: Optional[str] = None
    winner_actor_color: Optional[Literal["white", "black"]] = None
    termination_reason: Optional[str] = None
    start_fen: str
    final_fen: str
    move_count: int
    moves_compact: list[CompactMoveRecord] = Field(default_factory=list)
    inference_status: InferenceStatusRecord = Field(default_factory=InferenceStatusRecord)
    postgame_analysis_status: PostgameStatusRecord
    game_metadata: GameMetadataRecord = Field(default_factory=GameMetadataRecord)
    authoritative_status: str = "final"
    is_ranked: bool = False
    visibility: VisibilityType = "private"
    GSI1PK: str
    GSI1SK: str
    GSI2PK: Optional[str] = None
    GSI2SK: Optional[str] = None
    GSI3PK: Optional[str] = None
    GSI3SK: Optional[str] = None


class GameInferencePositionRecord(GamesTableItem):
    entity_type: Literal["GAME_INFERENCE_POSITION"] = "GAME_INFERENCE_POSITION"
    ply: int
    fen_before: str
    side_to_move: Literal["white", "black"]
    actor_id: str
    actor_type: ActorType
    selected_move_uci: str
    selected_move_san: Optional[str] = None
    selected_move_probability: Optional[float] = None
    inference_config: InferenceConfigRecord
    board_state_analysis: BoardStateAnalysisRecord
    candidate_moves: list[CandidateMoveRecord] = Field(default_factory=list)
    created_at_iso: str


class GamePostgameBoardStateRecord(GamesTableItem):
    entity_type: Literal["GAME_POSTGAME_BOARD_STATE"] = "GAME_POSTGAME_BOARD_STATE"
    ply: int
    analysis_depth: int
    analysis_purpose: str
    fen_before: str
    side_to_move: Literal["white", "black"]
    board_state_analysis: BoardStateAnalysisRecord
    top_moves: list[CandidateMoveRecord] = Field(default_factory=list)
    requested_at_iso: Optional[str] = None
    completed_at_iso: Optional[str] = None


class GamePostgameMoveRecord(GamesTableItem):
    entity_type: Literal["GAME_POSTGAME_MOVE"] = "GAME_POSTGAME_MOVE"
    ply: int
    analysis_depth: int
    analysis_purpose: str
    fen_before: str
    played_move_uci: str
    played_move_san: Optional[str] = None
    actor_id: str
    actor_type: ActorType
    move_analysis: MoveAnalysisRecord
    requested_at_iso: Optional[str] = None
    completed_at_iso: Optional[str] = None


class GamePostgameSummaryRecord(GamesTableItem):
    entity_type: Literal["GAME_POSTGAME_SUMMARY"] = "GAME_POSTGAME_SUMMARY"
    analysis_depth: int
    analysis_purpose: str
    analysis_state: AnalysisStateType
    summary: PostgameSummaryMetricsRecord
    outputs_s3_key: Optional[str] = None
    requested_at_iso: Optional[str] = None
    completed_at_iso: Optional[str] = None


GameRecordUnion = Union[
    GameRecord,
    GameInferencePositionRecord,
    GamePostgameBoardStateRecord,
    GamePostgameMoveRecord,
    GamePostgameSummaryRecord,
]
