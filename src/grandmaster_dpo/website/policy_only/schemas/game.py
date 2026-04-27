from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClockState(StrictModel):
    white_ms: Optional[int] = None
    black_ms: Optional[int] = None


class TimingInfo(StrictModel):
    player_move_elapsed_ms: int = 0


class EngineLimitRequest(StrictModel):
    type: Literal["time_ms", "nodes", "depth"] = "time_ms"
    value: int = 200


class PhasePenaltyConfig(StrictModel):
    opening: int = 0
    middlegame: int = 0
    endgame: int = 0


class PhaseFloatConfig(StrictModel):
    opening: float = 1.0
    middlegame: float = 1.0
    endgame: float = 1.0


class PhaseProbabilityConfig(StrictModel):
    opening: float = 0.0
    middlegame: float = 0.0
    endgame: float = 0.0


class DrawPenaltyConfig(StrictModel):
    enabled: bool = False
    repetition_x2_penalty_cp: PhasePenaltyConfig = Field(default_factory=PhasePenaltyConfig)
    one_move_from_draw_penalty_cp: PhasePenaltyConfig = Field(default_factory=PhasePenaltyConfig)


class ForcedBlunderConfig(StrictModel):
    enabled: bool = False
    probability: float = 0.0
    max_bot_move_number: Optional[int] = 7
    piece_types: list[Literal["knight", "bishop", "rook", "queen"]] = Field(
        default_factory=lambda: ["knight", "bishop", "rook", "queen"]
    )
    min_cp_loss: int = 300
    max_cp_loss: int = 1200
    once_per_game: bool = True
    disable_opening_book_until_triggered: bool = True


class EngineConfigRequest(StrictModel):
    limit: Optional[EngineLimitRequest] = None
    random_seed: int = 0
    stockfish_multipv_topk: int = 10
    cp_gap_window: Optional[int] = 60
    use_gibbs: bool = False
    lam: Optional[float] = None
    temperature: Optional[float] = None
    alpha_style: float = 1.0
    beta_engine: Optional[float] = None
    engine_temp: float = 1.0
    style_temperature: float = 1.0
    novelty_weight: float = 0.0
    novelty_weight_prob: float = 1.0
    novelty_weight_phase: PhaseFloatConfig = Field(default_factory=PhaseFloatConfig)
    risk_weight: float = 0.0
    risk_weight_prob: float = 1.0
    risk_weight_phase: PhaseFloatConfig = Field(default_factory=PhaseFloatConfig)
    attack_weight: float = 0.0
    attack_weight_prob: float = 1.0
    attack_weight_phase: PhaseFloatConfig = Field(default_factory=PhaseFloatConfig)
    weird_move_prob: float = 0.0
    weird_move_phase: PhaseFloatConfig = Field(default_factory=PhaseFloatConfig)
    weird_move_min_cp_loss: int = 20
    weird_move_max_cp_loss: int = 120
    top_move_suppression_prob: float = 0.0
    top_move_suppression_phase: PhaseFloatConfig = Field(default_factory=PhaseFloatConfig)
    sacrifice_weight: float = 1.0
    sacrifice_propensity_phase: PhaseProbabilityConfig = Field(default_factory=PhaseProbabilityConfig)
    cp_scale: float = 150.0
    cp_cap: int = 2000
    sample: bool = True
    use_timer_head: bool = True
    time_control_style_scale: float = 1.0
    stockfish_tree_search_depth: Optional[int] = None
    stockfish_engine_depth: Optional[int] = None
    stockfish_engine_nodes: Optional[int] = None
    stockfish_max_time_ms: Optional[int] = None
    draw_penalties: DrawPenaltyConfig = Field(default_factory=DrawPenaltyConfig)
    forced_blunder: ForcedBlunderConfig = Field(default_factory=ForcedBlunderConfig)

    @model_validator(mode="after")
    def _apply_legacy_mood_aliases(self) -> "EngineConfigRequest":
        if self.lam is not None:
            self.engine_temp = float(self.lam)
        if self.temperature is not None:
            self.style_temperature = float(self.temperature)
        return self


class GameRequest(StrictModel):
    game_id: str
    client_ply: int = -1
    pre_move_fen: str = ""
    client_uci: str = ""
    gm_name: Optional[str] = None
    opening_family: Optional[str] = None
    bot_id: str = ""
    game_type_id: str
    clock: ClockState = Field(default_factory=ClockState)
    timing: TimingInfo = Field(default_factory=TimingInfo)
    engine_config: EngineConfigRequest = Field(default_factory=EngineConfigRequest)
    player_color: Optional[Literal["white", "black"]] = None
    authenticated_user_id: Optional[str] = None


class GameStatusResponse(StrictModel):
    state: Literal["ongoing", "checkmate", "stalemate", "draw", "timeout"]
    winner: Optional[Literal["white", "black"]] = None
    reason: str = ""


class StockfishCandidateMoveResponse(StrictModel):
    uci: str
    cp: int
    adjusted_cp: Optional[int] = None
    draw_penalty_cp: Optional[int] = None
    repetition_x2_penalty_cp: Optional[int] = None
    one_move_from_draw_penalty_cp: Optional[int] = None
    mate: Optional[int] = None
    pv_uci: list[str] = Field(default_factory=list)
    multipv_rank: Optional[int] = None
    in_cp_gap_window: bool = False
    prob: Optional[float] = None
    depth: Optional[int] = None
    seldepth: Optional[int] = None
    nodes: Optional[int] = None
    nps: Optional[int] = None
    time_ms: Optional[int] = None
    tbhits: Optional[int] = None
    novelty_score: Optional[float] = None
    risk_score: Optional[float] = None
    attack_score: Optional[float] = None
    sacrifice_score: Optional[float] = None
    forced_blunder_candidate: Optional[bool] = None
    forced_blunder_piece_type: Optional[str] = None
    forced_blunder_cp_loss: Optional[int] = None


class StockfishMetricsResponse(StrictModel):
    requested_multipv_topk: int
    returned_candidate_count: int
    cp_gap_window: Optional[int] = None
    opening_book_branch: Optional[str] = None
    opening_book_gm_name: Optional[str] = None
    max_depth: Optional[int] = None
    max_seldepth: Optional[int] = None
    total_nodes: Optional[int] = None
    max_nps: Optional[int] = None
    max_time_ms: Optional[int] = None
    stockfish_actual_think_ms: Optional[int] = None
    backend_wall_time_ms: Optional[int] = None
    best_cp: Optional[int] = None
    best_move_uci: Optional[str] = None
    best_move_mate: Optional[int] = None
    selected_move_rank_by_cp: Optional[int] = None
    selected_move_rank_by_prob_within_window: Optional[int] = None
    forced_blunder_attempted: Optional[bool] = None
    forced_blunder_triggered: Optional[bool] = None
    forced_blunder_candidate_count: Optional[int] = None
    forced_blunder_bot_move_number: Optional[int] = None
    forced_blunder_piece_type: Optional[str] = None
    forced_blunder_cp_loss: Optional[int] = None


class AnalysisResponse(StrictModel):
    bot_eval_cp: int
    bot_pv_uci: list[str]
    candidate_moves: list[StockfishCandidateMoveResponse] = Field(default_factory=list)
    stockfish_metrics: StockfishMetricsResponse
    selected_move_probability: Optional[float] = None
    use_gibbs: bool = False
    requested_think_ms: Optional[int] = None
    actual_think_ms: Optional[int] = None
    engine_limit: dict[str, Any] = Field(default_factory=dict)
    gm_name: str


class GameResponse(StrictModel):
    ok: Literal[True] = True
    game_id: str
    server_ply_before: int
    server_ply_after: int
    new_fen: str
    player_move_uci: str
    bot_move_uci: str
    bot_id: str
    game_type_id: str
    clock: ClockState
    game_status: GameStatusResponse
    analysis: AnalysisResponse


class ErrorInfo(StrictModel):
    code: str
    message: str


class ErrorResponse(StrictModel):
    ok: Literal[False] = False
    game_id: str
    error: ErrorInfo
    server_ply: int
    server_fen: str
    clock: ClockState


class ClockStateResponse(StrictModel):
    ok: Literal[True] = True
    game_id: str
    server_ply: int
    server_fen: str
    clock: ClockState
    player_color: Literal["white", "black"]
    last_ply_times_ms: list[int] = Field(default_factory=list)
    desynced: bool = False


class ClockUpdateRequest(StrictModel):
    client_ply: Optional[int] = None
    clock: ClockState


class ClockSyncRequest(StrictModel):
    client_ply: Optional[int] = None
    client_fen: Optional[str] = None
