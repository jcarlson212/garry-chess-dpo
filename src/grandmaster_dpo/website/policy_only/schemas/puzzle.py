from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from .game import EngineConfigRequest, ErrorInfo, StrictModel


PuzzleStatus = Literal["ongoing", "won", "lost"]
PuzzlePerformance = Literal["well", "okay", "bad"]


class PuzzleCategorySummary(StrictModel):
    category: str
    count: int


class PuzzleCatalogResponse(StrictModel):
    ok: Literal[True] = True
    categories: list[PuzzleCategorySummary] = Field(default_factory=list)


class PuzzleStartRequest(StrictModel):
    category: str = ""
    scenario_id: Optional[str] = None
    target_elo: Optional[int] = None
    normal_mean: Optional[float] = None
    normal_std: Optional[float] = None
    min_elo: Optional[int] = None
    max_elo: Optional[int] = None
    gm_name: Optional[str] = None
    bot_id: Optional[str] = None
    authenticated_user_id: Optional[str] = None
    engine_config: EngineConfigRequest = Field(default_factory=EngineConfigRequest)


class PuzzleMoveRequest(StrictModel):
    client_ply: int
    pre_move_fen: str
    client_uci: str


class PuzzleAdvantage(StrictModel):
    eval_cp: int
    eval_status: str
    best_move_uci: Optional[str] = None
    best_move_san: Optional[str] = None
    bar_pov: Literal["white"] = "white"


class PuzzleStateSummary(StrictModel):
    status: PuzzleStatus
    performance: PuzzlePerformance
    won: bool
    lost: bool
    terminated_early: bool
    termination_reason: str = ""


class PuzzleReplayStep(StrictModel):
    fen: str
    best_move_uci: Optional[str] = None
    best_move_san: Optional[str] = None
    position_eval_cp: int
    position_eval_status: str
    player_move_uci: Optional[str] = None
    player_move_san: Optional[str] = None
    cp_loss: Optional[int] = None
    performance: PuzzlePerformance


class PuzzleScenarioSummary(StrictModel):
    phase: str
    difficulty_estimate_elo: str
    length_bin: str
    sampled_rollout_length_plies: int


class PuzzlePrecomputedHints(StrictModel):
    light_tree: dict[str, Any] = Field(default_factory=dict)
    trajectory: list[dict[str, Any]] = Field(default_factory=list)


class PuzzleSessionResponse(StrictModel):
    ok: Literal[True] = True
    session_id: str
    scenario_id: str
    category: str
    bot_id: str
    game_type_id: str
    engine_config: EngineConfigRequest
    player_color: Literal["white", "black"]
    fen: str
    server_ply: int
    advantage: PuzzleAdvantage
    puzzle_state: PuzzleStateSummary
    scenario: PuzzleScenarioSummary
    precomputed_hints: PuzzlePrecomputedHints = Field(default_factory=PuzzlePrecomputedHints)
    player_move_uci: str = ""
    bot_move_uci: str = ""
    solution_replay: list[PuzzleReplayStep] = Field(default_factory=list)


class PuzzleErrorResponse(StrictModel):
    ok: Literal[False] = False
    session_id: str = ""
    error: ErrorInfo
    server_ply: int = 0
    server_fen: str = ""
