from __future__ import annotations

import io
import json
import logging

from prometheus_client import generate_latest

from calibre.core.logging import setup_logging
from calibre.core.metrics import (
    conformal_coverage_ratio,
    forecast_duration_seconds,
    observe_forecast_duration,
    order_cost,
    set_conformal_coverage,
    set_order_cost,
)
from calibre.core.tracing import span


def test_json_logging_includes_standard_extra_fields() -> None:
    stream = io.StringIO()
    setup_logging(stream=stream)
    logger = logging.getLogger("calibre.test")

    logger.info(
        "done",
        extra={"origin": "2024-01-01", "phase": "fit", "duration_ms": 12.5},
    )

    payload = json.loads(stream.getvalue())
    assert payload["message"] == "done"
    assert payload["origin"] == "2024-01-01"
    assert payload["phase"] == "fit"
    assert payload["duration_ms"] == 12.5


def test_metrics_helpers_populate_prometheus_series() -> None:
    observe_forecast_duration("stub", "predict", 0.01)
    set_conformal_coverage("stub", "cumulative", 0.9)
    set_order_cost("EUR", "unit", 3.5)

    assert forecast_duration_seconds.labels(model="stub", phase="predict")._sum.get() > 0.0
    assert conformal_coverage_ratio.labels(model="stub", mode="cumulative")._value.get() == 0.9
    assert order_cost.labels(currency="EUR", dataset="unit")._value.get() == 3.5


def test_prometheus_export_includes_required_series() -> None:
    observe_forecast_duration("export", "origin", 0.02)
    set_conformal_coverage("export", "cumulative", 0.75)
    set_order_cost("EUR", "export", 12.0)

    payload = generate_latest().decode()

    assert "calibre_forecast_duration_seconds" in payload
    assert "calibre_conformal_coverage_ratio" in payload
    assert "calibre_order_cost" in payload


def test_tracing_span_is_noop_context_manager() -> None:
    with span("unit", origin="2024-01-01"):
        value = 1
    assert value == 1
