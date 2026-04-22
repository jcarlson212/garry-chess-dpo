from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from grandmaster_dpo.website.policy_only.api.dependencies import get_game_service
from grandmaster_dpo.website.policy_only.schemas import (
    ClockStateResponse,
    ClockSyncRequest,
    ClockUpdateRequest,
    ErrorResponse,
)
from grandmaster_dpo.website.policy_only.service.game_service import GameServiceError, PolicyOnlyGameService

router = APIRouter()


@router.get(
    "/games/{game_id}/clock",
    response_model=ClockStateResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_clock(
    game_id: str,
    service: PolicyOnlyGameService = Depends(get_game_service),
) -> ClockStateResponse:
    try:
        return service.get_clock_state(game_id)
    except GameServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump())


@router.post(
    "/games/{game_id}/clock",
    response_model=ClockStateResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def update_clock(
    game_id: str,
    req: ClockUpdateRequest,
    service: PolicyOnlyGameService = Depends(get_game_service),
) -> ClockStateResponse:
    try:
        return service.update_clock(game_id, req)
    except GameServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump())


@router.post(
    "/games/{game_id}/clock/sync",
    response_model=ClockStateResponse,
    responses={404: {"model": ErrorResponse}},
)
def sync_clock(
    game_id: str,
    req: ClockSyncRequest,
    service: PolicyOnlyGameService = Depends(get_game_service),
) -> ClockStateResponse:
    try:
        return service.sync_clock(game_id, req)
    except GameServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump())
