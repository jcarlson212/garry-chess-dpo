from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import boto3

from grandmaster_dpo.website.game_finished_postprocessing.processor import (
    GameFinishedPostprocessor,
    make_default_postprocessor,
    unwrap_sqs_sns_message,
)


def configure_logging() -> None:
    level_name = os.environ.get("GAME_FINISHED_POSTPROCESSING_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _receive_loop(postprocessor: GameFinishedPostprocessor, sqs_client: Any, queue_url: str) -> None:
    logger = logging.getLogger(__name__)
    wait_time_seconds = int(os.environ.get("SQS_WAIT_TIME_SECONDS", "20"))
    visibility_timeout_seconds = int(os.environ.get("SQS_VISIBILITY_TIMEOUT_SECONDS", "900"))
    max_messages = int(os.environ.get("SQS_MAX_NUMBER_OF_MESSAGES", "1"))
    exit_on_idle = os.environ.get("EXIT_ON_IDLE", "false").lower() == "true"

    while True:
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time_seconds,
            VisibilityTimeout=visibility_timeout_seconds,
            AttributeNames=["All"],
            MessageAttributeNames=["All"],
        )
        messages = response.get("Messages") or []
        if not messages:
            logger.info("postprocess_queue_idle queue_url=%s", queue_url)
            if exit_on_idle:
                return
            continue

        for message in messages:
            receipt_handle = message["ReceiptHandle"]
            message_id = message.get("MessageId")
            payload = unwrap_sqs_sns_message(message["Body"])
            game_id = payload.get("game_id") or (payload.get("game") or {}).get("game_id")
            logger.info(
                "postprocess_queue_message_received message_id=%s game_id=%s event_key=%s",
                message_id,
                game_id,
                payload.get("event_key"),
            )
            try:
                result = postprocessor.process_payload(payload)
            except Exception:
                logger.exception(
                    "postprocess_queue_message_failed message_id=%s game_id=%s body=%s",
                    message_id,
                    game_id,
                    json.dumps(payload, default=str),
                )
                raise

            sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            logger.info(
                "postprocess_queue_message_deleted message_id=%s game_id=%s skipped_existing_summary=%s",
                message_id,
                result.game_id,
                result.skipped_existing_summary,
            )


def main() -> int:
    configure_logging()
    logger = logging.getLogger(__name__)
    queue_url = os.environ.get("GAME_FINISHED_SQS_QUEUE_URL") or os.environ.get("SQS_QUEUE_URL")
    if not queue_url:
        raise RuntimeError("Missing GAME_FINISHED_SQS_QUEUE_URL / SQS_QUEUE_URL")

    region_name = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    sqs_client = boto3.client("sqs", region_name=region_name)
    postprocessor = make_default_postprocessor()
    logger.info("postprocess_worker_started queue_url=%s region=%s", queue_url, region_name)
    _receive_loop(postprocessor, sqs_client, queue_url)
    logger.info("postprocess_worker_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
