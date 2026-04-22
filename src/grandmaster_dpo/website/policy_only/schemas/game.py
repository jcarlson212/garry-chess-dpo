from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class EngineConfigRequest(StrictModel):
    limit: Optional[EngineLimitRequest] = None
    random_seed: int = 0
    stockfish_multipv_topk: int = 10
    cp_gap_window: Optional[int] = 60
    use_gibbs: bool = False
    lam: float = 1.0
    cp_scale: float = 150.0
    cp_cap: int = 2000
    temperature: float = 1.0
    sample: bool = True
    use_timer_head: bool = True
    stockfish_tree_search_depth: Optional[int] = None
    stockfish_engine_depth: Optional[int] = None
    stockfish_engine_nodes: Optional[int] = None
    stockfish_max_time_ms: Optional[int] = None


class GameRequest(StrictModel):
    game_id: str
    client_ply: int = -1
    pre_move_fen: str
    client_uci: str = ""
    bot_id: str = ""
    game_type_id: str
    clock: ClockState = Field(default_factory=ClockState)
    timing: TimingInfo = Field(default_factory=TimingInfo)
    engine_config: EngineConfigRequest = Field(default_factory=EngineConfigRequest)
    player_color: Optional[Literal["white", "black"]] = None


class GameStatusResponse(StrictModel):
    state: Literal["ongoing", "checkmate", "stalemate", "draw"]
    winner: Optional[Literal["white", "black"]] = None
    reason: str = ""


class StockfishCandidateMoveResponse(StrictModel):
    uci: str
    cp: int
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


class StockfishMetricsResponse(StrictModel):
    requested_multipv_topk: int
    returned_candidate_count: int
    cp_gap_window: Optional[int] = None
    max_depth: Optional[int] = None
    max_seldepth: Optional[int] = None
    total_nodes: Optional[int] = None
    max_nps: Optional[int] = None
    max_time_ms: Optional[int] = None
    best_cp: Optional[int] = None
    best_move_uci: Optional[str] = None
    best_move_mate: Optional[int] = None
    selected_move_rank_by_cp: Optional[int] = None
    selected_move_rank_by_prob_within_window: Optional[int] = None


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
