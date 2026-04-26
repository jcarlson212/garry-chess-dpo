from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Protocol

import boto3

logger = logging.getLogger(__name__)

DEFAULT_GAMES_FINISHED_SNS_TOPIC_ARN = (
    "arn:aws:sns:us-east-1:437720536299:garry-chess-games-finished"
)


class GameFinishedPublisher(Protocol):
    def publish_finished_game(self, payload: dict[str, Any]) -> str | None:
        ...


class NullGameFinishedPublisher:
    def publish_finished_game(self, payload: dict[str, Any]) -> None:
        logger.info(
            "games_finished_publish_disabled game_id=%s event_key=%s",
            payload.get("game_id"),
            payload.get("event_key"),
        )
        return None


class SnsGameFinishedPublisher:
    def __init__(
        self,
        *,
        topic_arn: str,
        region_name: str | None = None,
        sns_client: Any = None,
        publish_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
    ) -> None:
        self.topic_arn = topic_arn
        self.region_name = region_name or os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION",
            "us-east-1",
        )
        self.sns_client = sns_client or boto3.client("sns", region_name=self.region_name)
        self.publish_attempts = max(1, int(publish_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    def publish_finished_game(self, payload: dict[str, Any]) -> str:
        message = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
        last_error: Exception | None = None
        for attempt in range(1, self.publish_attempts + 1):
            try:
                resp = self.sns_client.publish(
                    TopicArn=self.topic_arn,
                    Message=message,
                    MessageAttributes={
                        "event_type": {"DataType": "String", "StringValue": "game_finished"},
                        "game_id": {"DataType": "String", "StringValue": str(payload.get("game_id") or "")},
                        "game_status": {
                            "DataType": "String",
                            "StringValue": str((payload.get("game_status") or {}).get("state") or ""),
                        },
                    },
                )
                return str(resp["MessageId"])
            except Exception as exc:  # pragma: no cover - retry path is validated via service tests
                last_error = exc
                logger.warning(
                    "games_finished_publish_attempt_failed game_id=%s event_key=%s attempt=%s/%s error=%s",
                    payload.get("game_id"),
                    payload.get("event_key"),
                    attempt,
                    self.publish_attempts,
                    exc,
                )
                if attempt < self.publish_attempts and self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds)
        assert last_error is not None
        raise last_error


def resolve_game_finished_publisher() -> GameFinishedPublisher:
    raw_topic_arn = os.environ.get(
        "POLICY_ONLY_GAMES_FINISHED_SNS_TOPIC_ARN",
        DEFAULT_GAMES_FINISHED_SNS_TOPIC_ARN,
    )
    topic_arn = raw_topic_arn.strip()
    if topic_arn.lower() in {"", "none", "disabled", "off"}:
        return NullGameFinishedPublisher()
    return SnsGameFinishedPublisher(
        topic_arn=topic_arn,
        publish_attempts=int(os.environ.get("POLICY_ONLY_GAMES_FINISHED_SNS_PUBLISH_ATTEMPTS", "3")),
        retry_backoff_seconds=float(
            os.environ.get("POLICY_ONLY_GAMES_FINISHED_SNS_RETRY_BACKOFF_SECONDS", "0.25")
        ),
    )
