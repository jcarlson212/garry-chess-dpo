from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from grandmaster_dpo.website.policy_only.schemas import EngineConfigRequest

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


@dataclass
class StoredPuzzleState:
    session_id: str
    scenario_id: str
    category: str
    bot_id: str
    game_type_id: str
    engine_config: EngineConfigRequest
    authenticated_user_id: str | None
    player_color: str
    start_fen: str
    current_fen: str
    current_ply: int
    phase: str
    difficulty_estimate_elo: str
    length_bin: str
    sampled_rollout_length_plies: int
    target_full_plies: int
    plies_played: int = 0
    performance: str = "well"
    status: str = "ongoing"
    termination_reason: str = ""
    max_user_cp_loss: int = 0
    move_history_uci: list[str] = field(default_factory=list)
    solution_replay: list[dict[str, Any]] = field(default_factory=list)
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "scenario_id": self.scenario_id,
            "category": self.category,
            "bot_id": self.bot_id,
            "game_type_id": self.game_type_id,
            "engine_config": self.engine_config.model_dump(mode="python"),
            "authenticated_user_id": self.authenticated_user_id,
            "player_color": self.player_color,
            "start_fen": self.start_fen,
            "current_fen": self.current_fen,
            "current_ply": self.current_ply,
            "phase": self.phase,
            "difficulty_estimate_elo": self.difficulty_estimate_elo,
            "length_bin": self.length_bin,
            "sampled_rollout_length_plies": self.sampled_rollout_length_plies,
            "target_full_plies": self.target_full_plies,
            "plies_played": self.plies_played,
            "performance": self.performance,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "max_user_cp_loss": self.max_user_cp_loss,
            "move_history_uci": self.move_history_uci,
            "solution_replay": self.solution_replay,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "StoredPuzzleState":
        return cls(
            session_id=str(data["session_id"]),
            scenario_id=str(data["scenario_id"]),
            category=str(data["category"]),
            bot_id=str(data["bot_id"]),
            game_type_id=str(data["game_type_id"]),
            engine_config=EngineConfigRequest.model_validate(data.get("engine_config") or {}),
            authenticated_user_id=(
                str(data["authenticated_user_id"])
                if data.get("authenticated_user_id") is not None
                else None
            ),
            player_color=str(data["player_color"]),
            start_fen=str(data["start_fen"]),
            current_fen=str(data["current_fen"]),
            current_ply=int(data["current_ply"]),
            phase=str(data.get("phase") or "unknown"),
            difficulty_estimate_elo=str(data.get("difficulty_estimate_elo") or "unknown"),
            length_bin=str(data.get("length_bin") or ""),
            sampled_rollout_length_plies=int(data.get("sampled_rollout_length_plies") or 0),
            target_full_plies=int(data.get("target_full_plies") or 0),
            plies_played=int(data.get("plies_played") or 0),
            performance=str(data.get("performance") or "well"),
            status=str(data.get("status") or "ongoing"),
            termination_reason=str(data.get("termination_reason") or ""),
            max_user_cp_loss=int(data.get("max_user_cp_loss") or 0),
            move_history_uci=[str(x) for x in (data.get("move_history_uci") or [])],
            solution_replay=[
                item for item in (data.get("solution_replay") or []) if isinstance(item, dict)
            ],
            created_at_ms=int(data.get("created_at_ms") or int(time.time() * 1000)),
            updated_at_ms=int(data.get("updated_at_ms") or int(time.time() * 1000)),
        )


class PuzzleStateStore(Protocol):
    def get(self, session_id: str) -> StoredPuzzleState | None:
        ...

    def set(self, session_id: str, state: StoredPuzzleState) -> None:
        ...


class InMemoryPuzzleStateStore:
    def __init__(self, ttl_seconds: int = 24 * 60 * 60, max_sessions: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, StoredPuzzleState] = {}
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions

    def _prune_expired_locked(self) -> None:
        if self._ttl_seconds <= 0:
            return
        cutoff_ms = int(time.time() * 1000) - (self._ttl_seconds * 1000)
        expired = [session_id for session_id, state in self._data.items() if state.updated_at_ms < cutoff_ms]
        for session_id in expired:
            self._data.pop(session_id, None)

    def _prune_oversized_locked(self) -> None:
        if self._max_sessions <= 0 or len(self._data) <= self._max_sessions:
            return
        ordered = sorted(self._data.items(), key=lambda item: item[1].updated_at_ms)
        to_remove = len(self._data) - self._max_sessions
        for session_id, _state in ordered[:to_remove]:
            self._data.pop(session_id, None)

    def get(self, session_id: str) -> StoredPuzzleState | None:
        with self._lock:
            self._prune_expired_locked()
            return self._data.get(session_id)

    def set(self, session_id: str, state: StoredPuzzleState) -> None:
        state.updated_at_ms = int(time.time() * 1000)
        with self._lock:
            self._prune_expired_locked()
            self._data[session_id] = state
            self._prune_oversized_locked()


class RedisPuzzleStateStore:
    def __init__(self, redis_url: str, ttl_seconds: int = 24 * 60 * 60, key_prefix: str = "policy_only") -> None:
        if redis is None:
            raise RuntimeError("redis package is not installed")
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}:puzzle:{session_id}"

    def get(self, session_id: str) -> StoredPuzzleState | None:
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return None
        return StoredPuzzleState.from_dict(json.loads(raw))

    def set(self, session_id: str, state: StoredPuzzleState) -> None:
        state.updated_at_ms = int(time.time() * 1000)
        self._client.set(self._key(session_id), json.dumps(state.to_dict()), ex=self._ttl_seconds)


def resolve_puzzle_state_store() -> PuzzleStateStore:
    redis_url = (
        os.environ.get("POLICY_ONLY_REDIS_URL")
        or os.environ.get("REDIS_URL")
        or os.environ.get("ELASTICACHE_REDIS_URL")
    )
    ttl_seconds = int(os.environ.get("PUZZLE_STATE_TTL_SECONDS", str(24 * 60 * 60)))
    key_prefix = os.environ.get("GAME_STATE_KEY_PREFIX", "policy_only")
    if redis_url:
        return RedisPuzzleStateStore(redis_url=redis_url, ttl_seconds=ttl_seconds, key_prefix=key_prefix)
    max_sessions = int(os.environ.get("IN_MEMORY_MAX_PUZZLE_SESSIONS", "10000"))
    return InMemoryPuzzleStateStore(ttl_seconds=ttl_seconds, max_sessions=max_sessions)
