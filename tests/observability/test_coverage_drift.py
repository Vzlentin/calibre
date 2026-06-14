"""Tests for coverage-drift observability metrics."""

from __future__ import annotations

import pandas as pd
from prometheus_client import REGISTRY

from calibre.conformal.controllers import AdaptiveAlphaController
from calibre.conformal.runtime import (
    SymmetricIntervalConfig,
    build_symmetric_interval_runtime,
)
from calibre.core.forecast_frame import (
    CONFORMAL_PARTITION,
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.core.metrics import conformal_coverage_drift
from calibre.execution.backend import (
    BackendEngine,
    ConformalOptions,
    ExecutionOptions,
)


def _make_runtime() -> object:
    return build_symmetric_interval_runtime(
        SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    )


def _adaptive_controller_with_history(
    *,
    target_alpha: float,
    errors: list[int],
) -> AdaptiveAlphaController:
    controller = AdaptiveAlphaController(alpha=target_alpha, gamma=0.0)
    for error in errors:
        y_pred = {"lower": 0.0, "upper": 1.0} if error else {"lower": 0.0, "upper": 10.0}
        controller.observe(5.0, y_pred, 1)
    return controller


def _drift_gauge(model: str, partition: str) -> float | None:
    return REGISTRY.get_sample_value(
        "calibre_conformal_coverage_drift", {"model": model, "partition": partition}
    )


def _engine(runtime: object) -> BackendEngine:
    return BackendEngine(
        execution=ExecutionOptions(freq="W-SUN"),
        conformal=ConformalOptions(runtime=runtime),
    )


def _resolved_frame(*, partitions: list[str], model: str = "stub") -> pd.DataFrame:
    lower, upper = SymmetricIntervalConfig(
        method="aci", coverage=0.9, calibration_window=4
    ).interval_columns
    rows = []
    for partition in partitions:
        rows.append(
            {
                UNIQUE_ID: "A",
                DS: pd.Timestamp("2024-01-14"),
                Y: 2.0,
                Y_HAT: 2.5,
                H: 1,
                FORECAST_ORIGIN: pd.Timestamp("2024-01-07"),
                MODEL_NAME: model,
                CONFORMAL_PARTITION: partition,
                lower: 1.0,
                upper: 4.0,
            }
        )
    return pd.DataFrame(rows)


def test_drift_gauge_is_empirical_miscoverage_minus_target_per_partition() -> None:
    runtime = _make_runtime()
    # 2 of 4 observations miss -> 0.5 empirical miscoverage; target alpha 0.1
    # leaves a drift of 0.4, asserted as a literal independent of the formula.
    runtime.controller = _adaptive_controller_with_history(target_alpha=0.1, errors=[1, 0, 0, 1])
    frame = _resolved_frame(partitions=["p1", "p2"], model="model_x")

    _engine(runtime)._record_coverage_drift(frame, runtime)

    assert _drift_gauge("model_x", "p1") == 0.4
    assert _drift_gauge("model_x", "p2") == 0.4


def test_drift_falls_back_to_global_partition() -> None:
    runtime = _make_runtime()
    runtime.controller = _adaptive_controller_with_history(target_alpha=0.1, errors=[1, 1, 0, 0])
    frame = _resolved_frame(partitions=["only"], model="model_y").drop(
        columns=[CONFORMAL_PARTITION]
    )

    _engine(runtime)._record_coverage_drift(frame, runtime)

    assert _drift_gauge("model_y", "__global__") == 0.4


def test_fixed_controller_records_no_drift() -> None:
    runtime = build_symmetric_interval_runtime(
        SymmetricIntervalConfig(method="mscp", coverage=0.9, calibration_window=4)
    )
    frame = _resolved_frame(partitions=["p1"], model="model_fixed")
    sentinel = -12345.0
    conformal_coverage_drift.labels(model="model_fixed", partition="p1").set(sentinel)

    _engine(runtime)._record_coverage_drift(frame, runtime)

    assert _drift_gauge("model_fixed", "p1") == sentinel


def test_adaptive_controller_with_empty_history_records_no_drift() -> None:
    runtime = _make_runtime()
    assert runtime.controller.error_history == []
    frame = _resolved_frame(partitions=["p1"], model="model_empty")
    sentinel = -999.0
    conformal_coverage_drift.labels(model="model_empty", partition="p1").set(sentinel)

    _engine(runtime)._record_coverage_drift(frame, runtime)

    assert _drift_gauge("model_empty", "p1") == sentinel
