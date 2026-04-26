from fastapi import APIRouter

from grandmaster_dpo.website.policy_only.api.routes import clocks, games, health, puzzles

router = APIRouter()
router.include_router(health.router, tags=["health"])
router.include_router(games.router, tags=["games"])
router.include_router(clocks.router, tags=["clock"])
router.include_router(puzzles.router, tags=["puzzles"])
