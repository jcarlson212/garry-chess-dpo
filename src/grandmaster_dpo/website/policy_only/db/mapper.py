from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Optional

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError
from pydantic import TypeAdapter

from grandmaster_dpo.website.policy_only.db.models import (
    GameInferencePositionRecord,
    GamePostgameBoardStateRecord,
    GamePostgameMoveRecord,
    GamePostgameSummaryRecord,
    GameRecord,
    GameRecordUnion,
)


@dataclass(frozen=True)
class GameQueryPage:
    items: list[GameRecord]
    next_cursor: Optional[str] = None


@dataclass(frozen=True)
class GameBundleWriteResult:
    game_written: bool
    inference_positions_written: int
    postgame_board_states_written: int
    postgame_moves_written: int
    postgame_summaries_written: int


class PolicyOnlyGamesTable:
    """
    Typed access layer for the `garry-chess-games` single-table design.

    Main client read patterns supported here:
    - fetch a known set of recently completed games by `game_id`
    - fetch postgame analysis and inference traces for one game partition
    - fetch all completed games for a player, paginated
    - fetch games updated since a client watermark (`updated_at_iso`) to support
      app boot / resume flows

    Best-practice note:
    - The `GAME` item is the compact sync source for history and incremental sync.
    - Heavy inference and postgame analysis live under separate `SK`s.
    - Postgame jobs should update both child analysis items and the parent `GAME`
      item's `updated_at_iso` / `postgame_analysis_status`, so incremental client
      syncs can stay centered on `GAME` items.
    """

    DEFAULT_TABLE_NAME = "garry-chess-games"
    GSI_PLAYER_GAMES = "gsi_player_games"
    GSI_PAIR_HISTORY = "gsi_pair_history"
    GSI_PLAYER_UPDATES = "gsi_player_updates"

    def __init__(
        self,
        *,
        table_name: Optional[str] = None,
        region_name: Optional[str] = None,
        dynamodb_resource: Any = None,
        player_updates_index_name: Optional[str] = None,
    ) -> None:
        self.table_name = table_name or os.environ.get(
            "POLICY_ONLY_GAMES_TABLE_NAME",
            self.DEFAULT_TABLE_NAME,
        )
        self.region_name = region_name or os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION",
            "us-east-1",
        )
        self.dynamodb = dynamodb_resource or boto3.resource("dynamodb", region_name=self.region_name)
        self.table = self.dynamodb.Table(self.table_name)
        self.player_updates_index_name = player_updates_index_name or os.environ.get(
            "POLICY_ONLY_GAMES_UPDATES_INDEX_NAME"
        )
        self._item_adapter = TypeAdapter(
            GameRecord
            | GameInferencePositionRecord
            | GamePostgameBoardStateRecord
            | GamePostgameMoveRecord
            | GamePostgameSummaryRecord
        )

    @staticmethod
    def game_pk(game_id: str) -> str:
        return f"GAME#{game_id}"

    @staticmethod
    def game_sk() -> str:
        return "GAME"

    @staticmethod
    def player_gsi_pk(actor_id: str) -> str:
        return f"PLAYER#{actor_id}"

    @staticmethod
    def pair_gsi_pk(actor_a_type: str, actor_a_id: str, actor_b_type: str, actor_b_id: str) -> str:
        pair = sorted([(actor_a_type, actor_a_id), (actor_b_type, actor_b_id)])
        return f"PAIR#{pair[0][0]}#{pair[0][1]}#{pair[1][0]}#{pair[1][1]}"

    @staticmethod
    def encode_cursor(last_evaluated_key: Optional[dict[str, Any]]) -> Optional[str]:
        if not last_evaluated_key:
            return None
        return json.dumps(last_evaluated_key, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def decode_cursor(cursor: Optional[str]) -> Optional[dict[str, Any]]:
        if not cursor:
            return None
        return json.loads(cursor)

    def _dispatch_item(self, item: dict[str, Any]) -> GameRecordUnion:
        return self._item_adapter.validate_python(item)

    def _to_dynamo_compatible(self, value: Any) -> Any:
        if isinstance(value, float):
            return Decimal(str(value))
        if isinstance(value, list):
            return [self._to_dynamo_compatible(item) for item in value]
        if isinstance(value, dict):
            return {key: self._to_dynamo_compatible(item) for key, item in value.items()}
        return value

    def _write_if_newer(self, record: GameRecordUnion) -> bool:
        item = self._to_dynamo_compatible(record.model_dump(mode="python", exclude_none=True))
        try:
            self.table.put_item(
                Item=item,
                ConditionExpression=Attr("version").not_exists() | Attr("version").lte(record.version),
            )
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise

    def put_game_record(self, record: GameRecord) -> bool:
        return self._write_if_newer(record)

    def put_inference_position(self, record: GameInferencePositionRecord) -> bool:
        return self._write_if_newer(record)

    def put_postgame_board_state(self, record: GamePostgameBoardStateRecord) -> bool:
        return self._write_if_newer(record)

    def put_postgame_move(self, record: GamePostgameMoveRecord) -> bool:
        return self._write_if_newer(record)

    def put_postgame_summary(self, record: GamePostgameSummaryRecord) -> bool:
        return self._write_if_newer(record)

    def put_completed_game_bundle(
        self,
        *,
        game: GameRecord,
        inference_positions: Iterable[GameInferencePositionRecord] = (),
        postgame_board_states: Iterable[GamePostgameBoardStateRecord] = (),
        postgame_moves: Iterable[GamePostgameMoveRecord] = (),
        postgame_summaries: Iterable[GamePostgameSummaryRecord] = (),
    ) -> GameBundleWriteResult:
        game_written = self.put_game_record(game)
        inference_written = sum(1 for record in inference_positions if self.put_inference_position(record))
        board_written = sum(
            1 for record in postgame_board_states if self.put_postgame_board_state(record)
        )
        move_written = sum(1 for record in postgame_moves if self.put_postgame_move(record))
        summary_written = sum(1 for record in postgame_summaries if self.put_postgame_summary(record))
        return GameBundleWriteResult(
            game_written=game_written,
            inference_positions_written=inference_written,
            postgame_board_states_written=board_written,
            postgame_moves_written=move_written,
            postgame_summaries_written=summary_written,
        )

    def get_game(self, game_id: str, *, consistent_read: bool = False) -> Optional[GameRecord]:
        resp = self.table.get_item(
            Key={"PK": self.game_pk(game_id), "SK": self.game_sk()},
            ConsistentRead=consistent_read,
        )
        item = resp.get("Item")
        if not item:
            return None
        parsed = self._dispatch_item(item)
        if not isinstance(parsed, GameRecord):
            raise TypeError(f"Expected GAME item for {game_id}, got {type(parsed).__name__}")
        return parsed

    def get_game_partition(
        self,
        game_id: str,
        *,
        consistent_read: bool = False,
    ) -> list[GameRecordUnion]:
        resp = self.table.query(
            KeyConditionExpression=Key("PK").eq(self.game_pk(game_id)),
            ConsistentRead=consistent_read,
        )
        return [self._dispatch_item(item) for item in resp.get("Items", [])]

    def get_postgame_summary(
        self,
        game_id: str,
        *,
        analysis_depth: int,
        analysis_purpose: str,
        consistent_read: bool = False,
    ) -> Optional[GamePostgameSummaryRecord]:
        sk = f"POSTGAME#SUMMARY#DEPTH#{analysis_depth}#PURPOSE#{analysis_purpose}"
        resp = self.table.get_item(
            Key={"PK": self.game_pk(game_id), "SK": sk},
            ConsistentRead=consistent_read,
        )
        item = resp.get("Item")
        if not item:
            return None
        parsed = self._dispatch_item(item)
        if not isinstance(parsed, GamePostgameSummaryRecord):
            raise TypeError(f"Expected postgame summary item, got {type(parsed).__name__}")
        return parsed

    def get_recent_games_by_ids(
        self,
        game_ids: Iterable[str],
        *,
        consistent_read: bool = False,
    ) -> dict[str, GameRecord]:
        keys = [{"PK": self.game_pk(game_id), "SK": self.game_sk()} for game_id in game_ids]
        if not keys:
            return {}
        results: dict[str, GameRecord] = {}
        request_items = {self.table_name: {"Keys": keys, "ConsistentRead": consistent_read}}
        while request_items:
            resp = self.dynamodb.batch_get_item(RequestItems=request_items)
            for item in resp.get("Responses", {}).get(self.table_name, []):
                parsed = self._dispatch_item(item)
                if isinstance(parsed, GameRecord):
                    results[parsed.game_id] = parsed
            request_items = resp.get("UnprocessedKeys") or {}
        return results

    def list_games_for_player(
        self,
        actor_id: str,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
        newest_first: bool = True,
        consistent_read: bool = False,
    ) -> GameQueryPage:
        query_kwargs: dict[str, Any] = {
            "IndexName": self.GSI_PLAYER_GAMES,
            "KeyConditionExpression": Key("GSI1PK").eq(self.player_gsi_pk(actor_id)),
            "Limit": limit,
            "ScanIndexForward": not newest_first,
            "ConsistentRead": consistent_read,
        }
        exclusive_start_key = self.decode_cursor(cursor)
        if exclusive_start_key:
            query_kwargs["ExclusiveStartKey"] = exclusive_start_key
        resp = self.table.query(
            **query_kwargs,
        )
        items = [self._dispatch_item(item) for item in resp.get("Items", [])]
        games = [item for item in items if isinstance(item, GameRecord)]
        return GameQueryPage(items=games, next_cursor=self.encode_cursor(resp.get("LastEvaluatedKey")))

    def list_games_updated_since(
        self,
        actor_id: str,
        *,
        updated_since_iso: str,
        limit: int = 50,
        cursor: Optional[str] = None,
        newest_first: bool = True,
    ) -> GameQueryPage:
        """
        Incremental sync path for app startup / resume.

        Preferred production design:
        - maintain a player-updates GSI keyed by player and `updated_at_iso`
        - set `self.player_updates_index_name` to that GSI name

        Current-table fallback:
        - query `gsi_player_games`
        - filter on the parent `GAME.updated_at_iso`
        - this works but is less efficient when accounts have many historical games
        """
        if self.player_updates_index_name:
            query_kwargs = {
                "IndexName": self.player_updates_index_name,
                "KeyConditionExpression": Key("GSI3PK").eq(self.player_gsi_pk(actor_id))
                & Key("GSI3SK").gte(f"UPDATED#{updated_since_iso}"),
                "Limit": limit,
                "ScanIndexForward": not newest_first,
            }
            exclusive_start_key = self.decode_cursor(cursor)
            if exclusive_start_key:
                query_kwargs["ExclusiveStartKey"] = exclusive_start_key
            resp = self.table.query(**query_kwargs)
            items = [self._dispatch_item(item) for item in resp.get("Items", [])]
            games = [item for item in items if isinstance(item, GameRecord)]
            return GameQueryPage(
                items=games,
                next_cursor=self.encode_cursor(resp.get("LastEvaluatedKey")),
            )

        query_kwargs = {
            "IndexName": self.GSI_PLAYER_GAMES,
            "KeyConditionExpression": Key("GSI1PK").eq(self.player_gsi_pk(actor_id)),
            "FilterExpression": Attr("updated_at_iso").gte(updated_since_iso),
            "Limit": limit,
            "ScanIndexForward": not newest_first,
        }
        exclusive_start_key = self.decode_cursor(cursor)
        if exclusive_start_key:
            query_kwargs["ExclusiveStartKey"] = exclusive_start_key
        resp = self.table.query(**query_kwargs)
        items = [self._dispatch_item(item) for item in resp.get("Items", [])]
        games = [item for item in items if isinstance(item, GameRecord)]
        return GameQueryPage(items=games, next_cursor=self.encode_cursor(resp.get("LastEvaluatedKey")))
