from __future__ import annotations

import pandas as pd

from calibre.conformal.controllers import AdaptiveAlphaController, FixedAlphaController
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
    _adaptive_controller_drift,
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


def test_adaptive_controller_drift_matches_mean_minus_target() -> None:
    runtime = _make_runtime()
    runtime.controller = _adaptive_controller_with_history(target_alpha=0.1, errors=[1, 0, 0, 1])

    drift = _adaptive_controller_drift(runtime)

    assert drift is not None
    assert drift == 0.5 - 0.1


def test_adaptive_controller_drift_is_none_for_fixed_controller() -> None:
    runtime = build_symmetric_interval_runtime(
        SymmetricIntervalConfig(method="mscp", coverage=0.9, calibration_window=4)
    )
    assert isinstance(runtime.controller, FixedAlphaController)

    assert _adaptive_controller_drift(runtime) is None


def test_adaptive_controller_drift_is_none_when_history_empty() -> None:
    runtime = _make_runtime()
    assert isinstance(runtime.controller, AdaptiveAlphaController)
    assert runtime.controller.error_history == []

    assert _adaptive_controller_drift(runtime) is None


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


def test_record_coverage_drift_emits_one_gauge_per_partition() -> None:
    runtime = _make_runtime()
    runtime.controller = _adaptive_controller_with_history(target_alpha=0.1, errors=[1, 0, 1, 0])
    engine = BackendEngine(
        execution=ExecutionOptions(freq="W-SUN"),
        conformal=ConformalOptions(runtime=runtime),
    )
    frame = _resolved_frame(partitions=["p1", "p2"], model="model_x")
    expected_drift = 0.5 - 0.1

    conformal_coverage_drift.labels(model="model_x", partition="p1").set(-999.0)
    conformal_coverage_drift.labels(model="model_x", partition="p2").set(-999.0)

    engine._record_coverage_drift(frame, runtime)

    assert (
        conformal_coverage_drift.labels(model="model_x", partition="p1")._value.get()
        == expected_drift
    )
    assert (
        conformal_coverage_drift.labels(model="model_x", partition="p2")._value.get()
        == expected_drift
    )


def test_record_coverage_drift_falls_back_to_global_partition() -> None:
    runtime = _make_runtime()
    runtime.controller = _adaptive_controller_with_history(target_alpha=0.1, errors=[1, 1, 0, 0])
    engine = BackendEngine(
        execution=ExecutionOptions(freq="W-SUN"),
        conformal=ConformalOptions(runtime=runtime),
    )
    frame = _resolved_frame(partitions=["only"], model="model_y").drop(
        columns=[CONFORMAL_PARTITION]
    )

    conformal_coverage_drift.labels(model="model_y", partition="__global__").set(-999.0)

    engine._record_coverage_drift(frame, runtime)

    assert (
        conformal_coverage_drift.labels(model="model_y", partition="__global__")._value.get()
        == 0.5 - 0.1
    )


def test_record_coverage_drift_is_noop_for_fixed_controller() -> None:
    runtime = build_symmetric_interval_runtime(
        SymmetricIntervalConfig(method="mscp", coverage=0.9, calibration_window=4)
    )
    engine = BackendEngine(
        execution=ExecutionOptions(freq="W-SUN"),
        conformal=ConformalOptions(runtime=runtime),
    )
    frame = _resolved_frame(partitions=["p1"], model="model_fixed")

    sentinel = -12345.0
    conformal_coverage_drift.labels(model="model_fixed", partition="p1").set(sentinel)

    engine._record_coverage_drift(frame, runtime)

    assert (
        conformal_coverage_drift.labels(model="model_fixed", partition="p1")._value.get()
        == sentinel
    )
