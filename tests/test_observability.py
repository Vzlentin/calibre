from __future__ import annotations

import io
import json
import logging

from calibre.core.logging import setup_logging
from calibre.core.metrics import (
    forecast_duration_seconds,
    observe_forecast_duration,
    order_cost,
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
    set_order_cost("EUR", "unit", 3.5)

    assert forecast_duration_seconds.labels(model="stub", phase="predict")._sum.get() > 0.0
    assert order_cost.labels(currency="EUR", dataset="unit")._value.get() == 3.5


def test_tracing_span_is_noop_context_manager() -> None:
    with span("unit", origin="2024-01-01"):
        value = 1
    assert value == 1
