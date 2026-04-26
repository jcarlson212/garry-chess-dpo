from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from grandmaster_dpo.website.policy_only.api.dependencies import get_puzzle_service
from grandmaster_dpo.website.policy_only.schemas import (
    PuzzleCatalogResponse,
    PuzzleErrorResponse,
    PuzzleMoveRequest,
    PuzzleSessionResponse,
    PuzzleStartRequest,
)
from grandmaster_dpo.website.policy_only.service.puzzle_service import PuzzleService, PuzzleServiceError

router = APIRouter()


@router.get("/puzzles", response_model=PuzzleCatalogResponse, responses={500: {"model": PuzzleErrorResponse}})
def get_puzzles(
    service: PuzzleService = Depends(get_puzzle_service),
) -> PuzzleCatalogResponse:
    try:
        return service.list_categories()
    except PuzzleServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump())


@router.post("/puzzles/start", response_model=PuzzleSessionResponse, responses={400: {"model": PuzzleErrorResponse}, 404: {"model": PuzzleErrorResponse}, 500: {"model": PuzzleErrorResponse}})
def post_puzzle_start(
    req: PuzzleStartRequest,
    service: PuzzleService = Depends(get_puzzle_service),
) -> PuzzleSessionResponse:
    try:
        return service.start_puzzle(req)
    except PuzzleServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump())


@router.get("/puzzles/{session_id}", response_model=PuzzleSessionResponse, responses={404: {"model": PuzzleErrorResponse}, 500: {"model": PuzzleErrorResponse}})
def get_puzzle_session(
    session_id: str,
    service: PuzzleService = Depends(get_puzzle_service),
) -> PuzzleSessionResponse:
    try:
        return service.get_puzzle(session_id)
    except PuzzleServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump())


@router.post("/puzzles/{session_id}/move", response_model=PuzzleSessionResponse, responses={400: {"model": PuzzleErrorResponse}, 404: {"model": PuzzleErrorResponse}, 409: {"model": PuzzleErrorResponse}, 500: {"model": PuzzleErrorResponse}})
def post_puzzle_move(
    session_id: str,
    req: PuzzleMoveRequest,
    service: PuzzleService = Depends(get_puzzle_service),
) -> PuzzleSessionResponse:
    try:
        return service.play_move(session_id, req)
    except PuzzleServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump())
