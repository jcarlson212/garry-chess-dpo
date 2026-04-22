from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from grandmaster_dpo.website.policy_only.api.dependencies import get_game_service
from grandmaster_dpo.website.policy_only.schemas import ErrorResponse, GameRequest, GameResponse
from grandmaster_dpo.website.policy_only.service.game_service import GameServiceError, PolicyOnlyGameService

router = APIRouter()


@router.post("/games", response_model=GameResponse, responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def post_games(
    req: GameRequest,
    service: PolicyOnlyGameService = Depends(get_game_service),
) -> GameResponse:
    try:
        return service.play_turn(req)
    except GameServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump())
