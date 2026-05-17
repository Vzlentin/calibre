from __future__ import annotations

from prometheus_client import Gauge, Histogram, start_http_server

forecast_duration_seconds = Histogram(
    "calibre_forecast_duration_seconds",
    "Forecast execution duration.",
    ["model", "phase"],
)
conformal_coverage_ratio = Gauge(
    "calibre_conformal_coverage_ratio",
    "Observed conformal coverage ratio.",
    ["model", "mode"],
)
order_cost = Gauge(
    "calibre_order_cost",
    "Order policy cost.",
    ["currency", "dataset"],
)


def serve(port: int) -> None:
    start_http_server(int(port))


def observe_forecast_duration(model: str, phase: str, seconds: float) -> None:
    forecast_duration_seconds.labels(model=model, phase=phase).observe(float(seconds))


def set_conformal_coverage(model: str, mode: str, ratio: float) -> None:
    conformal_coverage_ratio.labels(model=model, mode=mode).set(float(ratio))


def set_order_cost(currency: str, dataset: str, cost: float) -> None:
    order_cost.labels(currency=currency, dataset=dataset).set(float(cost))
