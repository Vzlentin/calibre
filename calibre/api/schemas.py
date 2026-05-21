from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from calibre.cli.config import BackendConfig, load_config_from_mapping
from calibre.core.run_status import RunStatus


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
    status: RunStatus
    artifact_urls: dict[str, str] = Field(default_factory=dict)
    row_count: int | None = None
    error: str | None = None


class FitRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    tenant: str
    sku_set: list[str]
    horizon: int
    freq: str = "W"
    history: list[dict[str, Any]]
    forecaster_config: dict[str, Any] = Field(
        ..., description="model_config dict resolved by the forecasting adapter registry"
    )
    future_x: list[dict[str, Any]] | None = None
    conformal_config: dict[str, Any] | None = None


class FitHandle(BaseModel):
    fit_id: str
    session_id: str
    status: RunStatus
    artifact_urls: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class PredictRequest(BaseModel):
    fit_id: str
    origin: str


class PredictResponse(BaseModel):
    rows: int
    forecast: list[dict[str, Any]]


class CalibrateRequest(BaseModel):
    session_id: str
    forecast: list[dict[str, Any]]


class CalibrateResponse(BaseModel):
    rows: int
    calibrated: list[dict[str, Any]]


class OrderRequest(BaseModel):
    calibrated: list[dict[str, Any]]
    ordering: dict[str, Any] = Field(
        ..., description="Ordering policy spec with policy/params/coverage/quantile"
    )
    inventory: list[dict[str, Any]] | None = None
    session_id: str | None = None


class OrderResponse(BaseModel):
    rows: int
    orders: list[dict[str, Any]]


class ObserveRequest(BaseModel):
    session_id: str
    actuals: list[dict[str, Any]]


class ObserveResponse(BaseModel):
    session_id: str
    status: RunStatus


class SessionStateResponse(BaseModel):
    session_id: str
    tenant: str
    unique_id: str
    state: dict[str, dict[str, Any]]
    last_forecast: list[dict[str, Any]] | None = None
    open_orders: list[dict[str, Any]] | None = None
