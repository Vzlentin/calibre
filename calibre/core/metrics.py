"""Prometheus metric definitions and the helpers that record into them."""

from __future__ import annotations

from prometheus_client import Gauge, Histogram, start_http_server

forecast_duration_seconds = Histogram(
    "calibre_forecast_duration_seconds",
    "Forecast execution duration.",
    ["model", "phase"],
)
phase_duration_seconds = Histogram(
    "calibre_phase_duration_seconds",
    "Per-origin engine phase duration.",
    ["phase"],
)
conformal_coverage_ratio = Gauge(
    "calibre_conformal_coverage_ratio",
    "Observed conformal coverage ratio.",
    ["model", "mode"],
)
conformal_coverage_drift = Gauge(
    "calibre_conformal_coverage_drift",
    "Adaptive-controller miscoverage drift: mean(error_history) - target_alpha.",
    ["model", "partition"],
)
order_cost = Gauge(
    "calibre_order_cost",
    "Order policy cost.",
    ["currency", "dataset"],
)


def serve(port: int) -> None:
    """Start the Prometheus HTTP exposition server on ``port``."""
    start_http_server(int(port))


def observe_forecast_duration(model: str, phase: str, seconds: float) -> None:
    """Record a forecast-execution duration sample for a model and phase."""
    forecast_duration_seconds.labels(model=model, phase=phase).observe(float(seconds))


def observe_phase_duration(phase: str, seconds: float) -> None:
    """Record a per-origin engine-phase duration sample."""
    phase_duration_seconds.labels(phase=phase).observe(float(seconds))


def set_conformal_coverage(model: str, mode: str, ratio: float) -> None:
    """Set the observed conformal coverage ratio for a model and mode."""
    conformal_coverage_ratio.labels(model=model, mode=mode).set(float(ratio))


def set_conformal_coverage_drift(model: str, partition: str, drift: float) -> None:
    """Set the adaptive-controller miscoverage drift for a model partition."""
    conformal_coverage_drift.labels(model=model, partition=partition).set(float(drift))


def set_order_cost(currency: str, dataset: str, cost: float) -> None:
    """Set the order-policy cost gauge for a currency and dataset."""
    order_cost.labels(currency=currency, dataset=dataset).set(float(cost))
