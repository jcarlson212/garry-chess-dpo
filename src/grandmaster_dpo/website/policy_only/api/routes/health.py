from __future__ import annotations

import os

from fastapi import APIRouter

from grandmaster_dpo.website.policy_only.api.dependencies import get_state_store_name
from grandmaster_dpo.website.policy_only.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    gm_name = (
        os.environ.get("POLICY_ONLY_GM_NAME")
        or os.environ.get("SERVE_GM_NAME")
        or os.environ.get("GM_NAME")
    )
    return HealthResponse(
        ok=True,
        service="policy-only-api",
        deployment_target="ecs-fargate",
        gm_name=gm_name,
        state_store=get_state_store_name(),
    )
