from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    config: dict[str, Any] = Field(..., description="BackendConfig YAML-equivalent mapping")


class ForecastResponse(BaseModel):
    rows: int
    forecasts: list[dict[str, Any]]


class RunResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    artifact_urls: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
