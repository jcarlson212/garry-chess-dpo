from __future__ import annotations

from grandmaster_dpo.website.policy_only.service.game_service import PolicyOnlyGameService
from grandmaster_dpo.website.policy_only.service.state import resolve_game_state_store

_STORE = resolve_game_state_store()
_SERVICE = PolicyOnlyGameService(store=_STORE)


def get_game_service() -> PolicyOnlyGameService:
    return _SERVICE


def get_state_store_name() -> str:
    return _STORE.__class__.__name__
