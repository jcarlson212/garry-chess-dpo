from __future__ import annotations

from grandmaster_dpo.website.policy_only.service.completion_events import resolve_game_finished_publisher
from grandmaster_dpo.website.policy_only.service.game_service import PolicyOnlyGameService
from grandmaster_dpo.website.policy_only.service.puzzle_registry import resolve_puzzle_scenario_registry
from grandmaster_dpo.website.policy_only.service.puzzle_service import PuzzleService
from grandmaster_dpo.website.policy_only.service.puzzle_state import resolve_puzzle_state_store
from grandmaster_dpo.website.policy_only.service.state import resolve_game_state_store

_STORE = resolve_game_state_store()
_PUBLISHER = resolve_game_finished_publisher()
_SERVICE = PolicyOnlyGameService(store=_STORE, finished_game_publisher=_PUBLISHER)
_PUZZLE_STORE = resolve_puzzle_state_store()
_PUZZLE_REGISTRY = resolve_puzzle_scenario_registry()
_PUZZLE_SERVICE = PuzzleService(store=_PUZZLE_STORE, scenario_registry=_PUZZLE_REGISTRY)


def get_game_service() -> PolicyOnlyGameService:
    return _SERVICE


def get_puzzle_service() -> PuzzleService:
    return _PUZZLE_SERVICE


def get_state_store_name() -> str:
    return _STORE.__class__.__name__
