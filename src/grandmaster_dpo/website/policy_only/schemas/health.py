from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    service: str
    deployment_target: str
    gm_name: Optional[str] = None
    state_store: str
