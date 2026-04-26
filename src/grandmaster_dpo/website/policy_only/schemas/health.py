from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    service: str
    deployment_target: str
    gm_name: Optional[str] = None
    gm_names: list[str] = Field(default_factory=list)
    state_store: str
