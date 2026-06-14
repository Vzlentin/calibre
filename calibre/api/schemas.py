"""Pydantic request/response schemas for the FastAPI surface.

Each model's docstring is published as its OpenAPI schema ``description``;
per-field ``Field(description=...)`` text becomes the per-property description.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from calibre.core.run_status import RunStatus


class BacktestRequest(BaseModel):
    """Request to run a full backtest from a backend config."""

    config: dict[str, Any] = Field(..., description="BackendConfig YAML-equivalent mapping")


class RunResponse(BaseModel):
    """Status and artifact locations for a submitted backtest run."""

    id: str
    status: RunStatus
    artifact_urls: dict[str, str] = Field(default_factory=dict)
    row_count: int | None = None
    error: str | None = None


class FitRequest(BaseModel):
    """Request to fit a forecaster over a SKU set and history window."""

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
    """Handle to a fit job, carrying its session and artifact locations."""

    fit_id: str
    session_id: str
    status: RunStatus
    artifact_urls: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class PredictRequest(BaseModel):
    """Request to predict from a completed fit at a given origin."""

    fit_id: str
    origin: str
    future_x_override: dict[str, list[dict[str, Any]]] | None = None


class PredictResponse(BaseModel):
    """Forecast rows produced by a predict call."""

    rows: int
    forecast: list[dict[str, Any]]


class CalibrateRequest(BaseModel):
    """Request to conformally calibrate a forecast for a session."""

    session_id: str
    forecast: list[dict[str, Any]]


class CalibrateResponse(BaseModel):
    """Calibrated forecast rows with conformal intervals."""

    rows: int
    calibrated: list[dict[str, Any]]


class OrderRequest(BaseModel):
    """Request to turn a calibrated forecast into order quantities."""

    calibrated: list[dict[str, Any]]
    ordering: dict[str, Any] = Field(
        ...,
        description=(
            "Ordering spec: policy (rs|rss|newsvendor), params, coverage; "
            "quantile applies only to rs; period applies only to newsvendor."
        ),
    )
    inventory: list[dict[str, Any]] | None = None
    session_id: str | None = None


class OrderResponse(BaseModel):
    """Order rows produced by an ordering policy."""

    rows: int
    orders: list[dict[str, Any]]


class ObserveRequest(BaseModel):
    """Request to record realized actuals against an open session."""

    session_id: str
    actuals: list[dict[str, Any]]


class ObserveResponse(BaseModel):
    """Session status after recording observed actuals."""

    session_id: str
    status: RunStatus


class TuneRequest(BaseModel):
    """Request to run hyper-parameter tuning over a search space."""

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
    """Handle to a tuning study, carrying its session and status."""

    study_id: str
    session_id: str
    status: RunStatus
    error: str | None = None


class TuneCandidatePayload(BaseModel):
    """A tuned candidate's model, conformal, and ordering config."""

    model_config = ConfigDict(protected_namespaces=())

    model_config_values: dict[str, Any] = Field(default_factory=dict)
    conformal_config: dict[str, Any] = Field(default_factory=dict)
    ordering_config: dict[str, Any] = Field(default_factory=dict)


class TuneStudyResponse(BaseModel):
    """Tuning study result with the best candidate per objective."""

    study_id: str
    session_id: str
    tenant: str
    sku_set: list[str]
    status: RunStatus
    best_candidates: dict[str, TuneCandidatePayload] = Field(default_factory=dict)
    error: str | None = None


class SessionStateResponse(BaseModel):
    """Persisted per-series session state and latest forecast/orders."""

    session_id: str
    tenant: str
    unique_id: str
    state: dict[str, dict[str, Any]]
    last_forecast: list[dict[str, Any]] | None = None
    open_orders: list[dict[str, Any]] | None = None
