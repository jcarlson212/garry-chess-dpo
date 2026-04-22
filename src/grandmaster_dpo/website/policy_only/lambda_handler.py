from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from grandmaster_dpo.website.policy_only.api.dependencies import get_game_service
from grandmaster_dpo.website.policy_only.schemas import ClockSyncRequest, ClockUpdateRequest, GameRequest
from grandmaster_dpo.website.policy_only.service.game_service import GameServiceError


def json_response(status_code: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def _http_method(event: dict[str, Any]) -> str:
    return (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or ""
    ).upper()


def _http_path(event: dict[str, Any]) -> str:
    return event.get("rawPath") or event.get("path") or ""


def _parse_body(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("isBase64Encoded"):
        raise ValueError("base64 body not supported")
    body = event.get("body") or "{}"
    return json.loads(body)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    service = get_game_service()
    method = _http_method(event)
    path = _http_path(event)

    try:
        if method == "POST" and path.endswith("/games"):
            req = GameRequest.model_validate(_parse_body(event))
            resp = service.play_turn(req)
            return json_response(200, resp.model_dump())

        if method == "GET" and path.endswith("/healthz"):
            return json_response(
                200,
                {
                    "ok": True,
                    "service": "policy-only-api",
                    "deployment_target": "lambda-compat",
                    "state_store": "in-memory",
                },
            )

        if method == "GET" and "/games/" in path and path.endswith("/clock"):
            game_id = path.rstrip("/").split("/")[-2]
            resp = service.get_clock_state(game_id)
            return json_response(200, resp.model_dump())

        if method == "POST" and "/games/" in path and path.endswith("/clock/sync"):
            game_id = path.rstrip("/").split("/")[-3]
            req = ClockSyncRequest.model_validate(_parse_body(event))
            resp = service.sync_clock(game_id, req)
            return json_response(200, resp.model_dump())

        if method == "POST" and "/games/" in path and path.endswith("/clock"):
            game_id = path.rstrip("/").split("/")[-2]
            req = ClockUpdateRequest.model_validate(_parse_body(event))
            resp = service.update_clock(game_id, req)
            return json_response(200, resp.model_dump())

        return json_response(404, {"ok": False, "error": {"code": "not_found", "message": "Unknown route"}})
    except ValidationError as exc:
        return json_response(
            400,
            {
                "ok": False,
                "error": {"code": "bad_request", "message": exc.errors()},
            },
        )
    except ValueError as exc:
        return json_response(
            400,
            {
                "ok": False,
                "error": {"code": "bad_request", "message": str(exc)},
            },
        )
    except GameServiceError as exc:
        return json_response(exc.status_code, exc.error.model_dump())
    except Exception as exc:
        return json_response(
            500,
            {
                "ok": False,
                "error": {
                    "code": "server_error",
                    "message": f"Unhandled: {type(exc).__name__}: {exc}",
                },
            },
        )
