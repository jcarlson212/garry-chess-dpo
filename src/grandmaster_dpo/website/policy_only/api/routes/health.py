from __future__ import annotations

from fastapi import APIRouter

from grandmaster_dpo.website.policy_only.api.dependencies import get_state_store_name
from grandmaster_dpo.website.policy_only.schemas import HealthResponse
from grandmaster_dpo.website.policy_only.service.runtime import configured_gm_names, default_gm_name

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    gm_names = configured_gm_names()
    return HealthResponse(
        ok=True,
        service="policy-only-api",
        deployment_target="ecs-fargate",
        gm_name=default_gm_name(),
        gm_names=gm_names,
        state_store=get_state_store_name(),
    )
