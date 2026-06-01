from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from calibre.core.run_status import RunStatus


class BacktestRequest(BaseModel):
    config: dict[str, Any] = Field(..., description="BackendConfig YAML-equivalent mapping")


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
    sales_uri: str = Field(
        ..., description="parquet/SQL URI resolved by the SalesAdapter into the history frame"
    )
    as_of: str | None = Field(
        None, description="point-in-time cutoff for sales revisions (as_of <= origin)"
    )
    forecaster_config: dict[str, Any] = Field(
        ..., description="model_config dict resolved by the forecasting adapter registry"
    )
    future_x_uri: str | None = Field(None, description="parquet URI for the future regressor frame")
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
    future_x_override: dict[str, list[dict[str, Any]]] | None = None


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


class TuneRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    tenant: str
    sku_set: list[str]
    horizon: int
    freq: str = "W"
    sales_uri: str = Field(
        ..., description="parquet/SQL URI resolved by the SalesAdapter into the history frame"
    )
    actuals_uri: str = Field(..., description="parquet URI for the realized-actuals frame")
    as_of: str | None = Field(
        None, description="point-in-time cutoff for sales revisions (as_of <= origin)"
    )
    origins: list[str]
    base_model_config: dict[str, Any]
    search_space_id: str
    objective_id: str
    n_trials: int = 20
    hpo_scope: Literal["local", "global"] = "local"
    conformal_config: dict[str, Any] | None = None


class TuneHandle(BaseModel):
    study_id: str
    session_id: str
    status: RunStatus
    error: str | None = None


class TuneCandidatePayload(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_config_values: dict[str, Any] = Field(default_factory=dict)
    conformal_config: dict[str, Any] = Field(default_factory=dict)
    ordering_config: dict[str, Any] = Field(default_factory=dict)


class TuneStudyResponse(BaseModel):
    study_id: str
    session_id: str
    tenant: str
    sku_set: list[str]
    status: RunStatus
    best_candidates: dict[str, TuneCandidatePayload] = Field(default_factory=dict)
    error: str | None = None


class SessionStateResponse(BaseModel):
    session_id: str
    tenant: str
    unique_id: str
    state: dict[str, dict[str, Any]]
    last_forecast: list[dict[str, Any]] | None = None
    open_orders: list[dict[str, Any]] | None = None
