from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from grandmaster_dpo.website.policy_only.api.router import router

logging.basicConfig(level=logging.INFO)


def _cors_allow_origins() -> list[str]:
    raw = os.environ.get("POLICY_ONLY_CORS_ALLOW_ORIGINS")
    if raw:
        origins = [item.strip() for item in raw.split(",") if item.strip()]
        if origins:
            return origins
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://localhost:5173",
        "https://127.0.0.1:5173",
        "https://garrychess.ai",
        "https://www.garrychess.ai",
    ]


def _apply_cors_headers(request: Request, response: JSONResponse) -> JSONResponse:
    origin = request.headers.get("origin")
    allowed_origins = _cors_allow_origins()
    if origin and (origin in allowed_origins or "*" in allowed_origins):
        response.headers["Access-Control-Allow-Origin"] = origin if origin != "*" else "*"
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app = FastAPI(
    title="Grandmaster DPO Policy-Only API",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ensure_cors_on_error_responses(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as exc:
        logging.exception("Unhandled request error", exc_info=exc)
        response = JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "code": "server_error",
                    "message": f"Unhandled: {type(exc).__name__}: {exc}",
                },
            },
        )
    return _apply_cors_headers(request, response)

app.include_router(router)
