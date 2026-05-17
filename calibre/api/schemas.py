from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from calibre.cli.config import BackendConfig, load_config_from_mapping


class ForecastRequest(BaseModel):
    config: dict[str, Any] = Field(..., description="BackendConfig YAML-equivalent mapping")

    def as_backend_config(self) -> BackendConfig:
        """Parse and validate this request's config into a BackendConfig."""
        return load_config_from_mapping(self.config)


class ForecastResponse(BaseModel):
    rows: int
    forecasts: list[dict[str, Any]]


class RunResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    artifact_urls: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
