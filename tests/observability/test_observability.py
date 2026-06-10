from __future__ import annotations

import io
import json
import logging

import pandas as pd
import pytest
from prometheus_client import REGISTRY, generate_latest

from calibre.conformal.runtime import SymmetricIntervalConfig, build_symmetric_interval_runtime
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.core.logging import setup_logging
from calibre.core.metrics import (
    observe_forecast_duration,
    set_conformal_coverage,
    set_order_cost,
)
from calibre.core.tracing import span
from calibre.execution.backend import BackendEngine, ExecutionOptions
from calibre.execution.task_builder import partition_tasks
from calibre.forecasting.adapter_base import ModelAdapter


class _ObservabilityAdapter(ModelAdapter):
    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del collect_fitted_values
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
    _run_standard_path_engine(monkeypatch, stream)

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

    duration_sum = REGISTRY.get_sample_value(
        "calibre_forecast_duration_seconds_sum", {"model": "stub", "phase": "predict"}
    )
    assert duration_sum == pytest.approx(0.01)
    assert (
        REGISTRY.get_sample_value(
            "calibre_conformal_coverage_ratio", {"model": "stub", "mode": "cumulative"}
        )
        == 0.9
    )
    assert (
        REGISTRY.get_sample_value("calibre_order_cost", {"currency": "EUR", "dataset": "unit"})
        == 3.5
    )


def test_prometheus_export_includes_required_series() -> None:
    observe_forecast_duration("export", "origin", 0.02)
    set_conformal_coverage("export", "cumulative", 0.75)
    set_order_cost("EUR", "export", 12.0)

    payload = generate_latest().decode()

    assert "calibre_forecast_duration_seconds" in payload
    assert "calibre_conformal_coverage_ratio" in payload
    assert "calibre_order_cost" in payload


def _run_standard_path_engine(monkeypatch, stream: io.StringIO) -> None:
    """Run a single-origin standard-path backtest, logging to ``stream``.

    Mirrors ``test_backend_logs_adapter_fit_predict_phases`` so the phase-seam
    and span instrumentation can be asserted on a real engine run.
    """
    setup_logging(stream=stream)
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
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
        partition_tasks([task]), history, [pd.Timestamp("2024-01-28")]
    )


def test_phase_seam_logs_completed_phase_per_phase(monkeypatch) -> None:
    stream = io.StringIO()
    _run_standard_path_engine(monkeypatch, stream)

    phase_records = [
        record for record in _json_lines(stream) if record.get("message") == "completed phase"
    ]
    by_phase = {record["phase"]: record for record in phase_records}
    for name in ("ResolveOpen", "Predict", "Reconcile", "Calibrate", "Order", "Commit"):
        assert name in by_phase, f"missing completed-phase record for {name}"
        assert by_phase[name]["duration_ms"] >= 0.0
        assert by_phase[name]["origin"] == "2024-01-28T00:00:00"


def test_phase_seam_logs_completion_when_phase_raises() -> None:
    stream = io.StringIO()
    setup_logging(stream=stream)
    engine = BackendEngine()
    origin = pd.Timestamp("2024-01-28")

    with (
        pytest.raises(RuntimeError, match=rf"Predict phase failed at origin {origin}"),
        engine._phase("Predict", origin),
    ):
        raise RuntimeError("boom")

    phase_records = [
        record
        for record in _json_lines(stream)
        if record.get("message") == "completed phase" and record.get("phase") == "Predict"
    ]
    assert len(phase_records) == 1
    assert phase_records[0]["origin"] == origin.isoformat()
    assert phase_records[0]["duration_ms"] >= 0.0


def test_phase_seam_failure_preserves_exception_type() -> None:
    engine = BackendEngine()
    origin = pd.Timestamp("2024-01-28")

    with (
        pytest.raises(ValueError, match=rf"Calibrate phase failed at origin {origin}: bad"),
        engine._phase("Calibrate", origin),
    ):
        raise ValueError("bad")


def test_phase_duration_metric_records_after_run(monkeypatch) -> None:
    stream = io.StringIO()
    _run_standard_path_engine(monkeypatch, stream)

    count = REGISTRY.get_sample_value("calibre_phase_duration_seconds_count", {"phase": "Commit"})
    assert count is not None and count >= 1
    assert "calibre_phase_duration_seconds" in generate_latest().decode()


def test_span_emits_completed_span_record() -> None:
    stream = io.StringIO()
    setup_logging(stream=stream)

    with span("x", origin="o"):
        pass

    records = [
        record for record in _json_lines(stream) if record.get("message") == "completed span"
    ]
    assert len(records) == 1
    assert records[0]["span"] == "x"
    assert records[0]["origin"] == "o"
    assert records[0]["duration_ms"] >= 0.0


def test_span_logs_on_exception_and_propagates() -> None:
    stream = io.StringIO()
    setup_logging(stream=stream)

    with pytest.raises(RuntimeError, match="kaboom"), span("y", origin="o"):
        raise RuntimeError("kaboom")

    records = [
        record
        for record in _json_lines(stream)
        if record.get("message") == "completed span" and record.get("span") == "y"
    ]
    assert len(records) == 1
    assert records[0]["duration_ms"] >= 0.0


def test_engine_run_emits_engine_level_spans(monkeypatch) -> None:
    stream = io.StringIO()
    _run_standard_path_engine(monkeypatch, stream)

    span_names = [
        record["span"]
        for record in _json_lines(stream)
        if record.get("message") == "completed span"
    ]
    for name in ("staging_materialize", "conformal_state_restore", "ledger_close"):
        assert span_names.count(name) == 1, f"expected exactly one {name} span"
    # Substep spans fire on origins where resolution work occurs (Commit's resolve).
    assert "ledger_append" in span_names
    assert "ledger_resolution_frame" in span_names


def test_engine_run_does_not_emit_backtest_span(monkeypatch) -> None:
    stream = io.StringIO()
    _run_standard_path_engine(monkeypatch, stream)

    span_names = {
        record.get("span")
        for record in _json_lines(stream)
        if record.get("message") == "completed span"
    }
    assert "backtest" not in span_names
