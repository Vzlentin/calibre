from __future__ import annotations

import io
import json
import logging

import pandas as pd
from prometheus_client import generate_latest

from calibre.conformal.runtime import SymmetricIntervalConfig, build_symmetric_interval_runtime
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
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
from calibre.execution.backend import BackendEngine, ExecutionOptions


class _ObservabilityAdapter:
    def fit(self, task: ForecastTask) -> None:
        self.task = task

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id],
                DS: [task.forecast_origin + pd.Timedelta(weeks=1)],
                Y_HAT: [3.0],
                H: [1],
            }
        )


def _json_lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


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


def test_backend_logs_adapter_fit_predict_phases(monkeypatch) -> None:
    stream = io.StringIO()
    setup_logging(stream=stream)
    monkeypatch.setattr(
        "calibre.execution.backend.resolve_adapter",
        lambda _: _ObservabilityAdapter(),
    )
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "A"],
            DS: pd.date_range("2024-01-07", periods=3, freq="W-SUN"),
            Y: [1.0, 2.0, 3.0],
        }
    )
    task = ForecastTask(
        history=history,
        horizon=1,
        model_config={"backend": "stub", "model": "stub_model", "name": "stub_model"},
    )

    BackendEngine(execution=ExecutionOptions(freq="W-SUN")).execute(
        [task], history, [pd.Timestamp("2024-01-28")]
    )

    records = _json_lines(stream)
    by_phase = {record.get("phase"): record for record in records}
    assert by_phase["fit"]["model_name"] == "stub_model"
    assert by_phase["fit"]["unique_id"] == "A"
    assert by_phase["fit"]["duration_ms"] >= 0.0
    assert by_phase["predict"]["model_name"] == "stub_model"
    assert by_phase["predict"]["unique_id"] == "A"
    assert by_phase["predict"]["duration_ms"] >= 0.0


def test_conformal_runtime_logs_apply_and_observe_operations() -> None:
    stream = io.StringIO()
    setup_logging(stream=stream)
    runtime = build_symmetric_interval_runtime(
        SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4)
    )
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            DS: [pd.Timestamp("2024-01-14")],
            Y: [2.0],
            Y_HAT: [2.5],
            H: [1],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-07")],
            MODEL_NAME: ["stub_model"],
        }
    )

    applied = runtime.apply(frame)
    runtime.observe(applied)

    conformal_records = [
        record for record in _json_lines(stream) if record.get("phase") == "conformal"
    ]
    assert [record["operation"] for record in conformal_records] == ["apply", "observe"]
    assert all(record["duration_ms"] >= 0.0 for record in conformal_records)
    assert all(record["rows"] == 1 for record in conformal_records)


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
