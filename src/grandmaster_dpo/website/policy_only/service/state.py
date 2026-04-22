from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from grandmaster_dpo.website.policy_only.schemas import ClockState

try:
    import redis
except Exception:  # pragma: no cover - optional until dependency is installed
    redis = None


@dataclass
class StoredGameState:
    game_id: str
    fen: str
    ply: int
    player_color: str
    clock: ClockState
    bot_id: str
    game_type_id: str
    last_ply_times_ms: list[int] = field(default_factory=list)
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "fen": self.fen,
            "ply": self.ply,
            "player_color": self.player_color,
            "clock": self.clock.model_dump(),
            "bot_id": self.bot_id,
            "game_type_id": self.game_type_id,
            "last_ply_times_ms": self.last_ply_times_ms,
            "updated_at_ms": self.updated_at_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "StoredGameState":
        return cls(
            game_id=str(data["game_id"]),
            fen=str(data["fen"]),
            ply=int(data["ply"]),
            player_color=str(data["player_color"]),
            clock=ClockState.model_validate(data.get("clock") or {}),
            bot_id=str(data.get("bot_id") or ""),
            game_type_id=str(data.get("game_type_id") or ""),
            last_ply_times_ms=[int(x) for x in (data.get("last_ply_times_ms") or [])],
            updated_at_ms=int(data.get("updated_at_ms") or int(time.time() * 1000)),
        )


class GameStateStore(Protocol):
    def get(self, game_id: str) -> StoredGameState | None:
        ...

    def set(self, game_id: str, state: StoredGameState) -> None:
        ...


class InMemoryGameStateStore:
    def __init__(self, ttl_seconds: int = 24 * 60 * 60, max_games: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, StoredGameState] = {}
        self._ttl_seconds = ttl_seconds
        self._max_games = max_games

    def _prune_expired_locked(self) -> None:
        if self._ttl_seconds <= 0:
            return
        cutoff_ms = int(time.time() * 1000) - (self._ttl_seconds * 1000)
        expired = [game_id for game_id, state in self._data.items() if state.updated_at_ms < cutoff_ms]
        for game_id in expired:
            self._data.pop(game_id, None)

    def _prune_oversized_locked(self) -> None:
        if self._max_games <= 0 or len(self._data) <= self._max_games:
            return
        ordered = sorted(self._data.items(), key=lambda item: item[1].updated_at_ms)
        to_remove = len(self._data) - self._max_games
        for game_id, _state in ordered[:to_remove]:
            self._data.pop(game_id, None)

    def get(self, game_id: str) -> StoredGameState | None:
        with self._lock:
            self._prune_expired_locked()
            return self._data.get(game_id)

    def set(self, game_id: str, state: StoredGameState) -> None:
        state.updated_at_ms = int(time.time() * 1000)
        with self._lock:
            self._prune_expired_locked()
            self._data[game_id] = state
            self._prune_oversized_locked()


class RedisGameStateStore:
    def __init__(self, redis_url: str, ttl_seconds: int = 24 * 60 * 60, key_prefix: str = "policy_only") -> None:
        if redis is None:
            raise RuntimeError("redis package is not installed")
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix

    def _key(self, game_id: str) -> str:
        return f"{self._key_prefix}:game:{game_id}"

    def get(self, game_id: str) -> StoredGameState | None:
        raw = self._client.get(self._key(game_id))
        if raw is None:
            return None
        return StoredGameState.from_dict(json.loads(raw))

    def set(self, game_id: str, state: StoredGameState) -> None:
        state.updated_at_ms = int(time.time() * 1000)
        self._client.set(self._key(game_id), json.dumps(state.to_dict()), ex=self._ttl_seconds)


def resolve_game_state_store() -> GameStateStore:
    redis_url = (
        os.environ.get("POLICY_ONLY_REDIS_URL")
        or os.environ.get("REDIS_URL")
        or os.environ.get("ELASTICACHE_REDIS_URL")
    )
    ttl_seconds = int(os.environ.get("GAME_STATE_TTL_SECONDS", str(24 * 60 * 60)))
    key_prefix = os.environ.get("GAME_STATE_KEY_PREFIX", "policy_only")
    if redis_url:
        return RedisGameStateStore(redis_url=redis_url, ttl_seconds=ttl_seconds, key_prefix=key_prefix)
    max_games = int(os.environ.get("IN_MEMORY_MAX_GAMES", "10000"))
    return InMemoryGameStateStore(ttl_seconds=ttl_seconds, max_games=max_games)
