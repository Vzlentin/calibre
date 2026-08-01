"""Run one shared protocol suite over every built-in conformal registration."""

from __future__ import annotations

import inspect
import math

import pandas as pd
import pytest
from tests.conformal_fixtures import delivery_batch

from newcalibre.conformal import (
    CalibrationResult,
    CalibrationSeedBatch,
    ConformalRuntime,
    ForecastKey,
    ResolvedObservation,
    available_methods,
    derive_partition_label,
    method_config_schema,
    resolve_method,
)
from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    CensoringAssertion,
    DecisionScopeKind,
    GuaranteeClaim,
    GuaranteeCurrency,
    interval_columns,
)

pytestmark = pytest.mark.tier1
_METHODS = available_methods()
_ORIGIN = pd.Timestamp("2026-02-02")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku"], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime([_ORIGIN]),
            ACTUAL_VALUE: pd.Series([float("nan")], dtype="float64"),
            POINT_FORECAST: pd.Series([4.0], dtype="float64"),
            HORIZON_STEP: pd.Series([1], dtype="int64"),
            ORIGIN: pd.to_datetime([_ORIGIN]),
            MODEL_NAME: pd.Series(["protocol-model"], dtype="string"),
        }
    )


def _observation(result: CalibrationResult) -> ResolvedObservation:
    row = result.forecasts.iloc[0]
    key = ForecastKey(
        series_key=row[SERIES_KEY],
        origin=pd.Timestamp(row[ORIGIN]),
        horizon_step=int(row[HORIZON_STEP]),
        model_name=row[MODEL_NAME],
    )
    return ResolvedObservation(
        forecast_key=key,
        target_timestamp=pd.Timestamp(row[TARGET_TIMESTAMP]),
        actual=7.0,
        point_forecast=float(row[POINT_FORECAST]),
        censoring_assertion=CensoringAssertion.UNCENSORED,
        availability_bound=None,
        issued=result.issuances[key],
    )


def test_builtin_method_enumeration_is_nonvacuous_and_exact() -> None:
    assert _METHODS == (
        "sequential-adaptive-per-step",
        "split-per-step",
        "split-window-sum",
        "weighted-per-step",
    )


@pytest.mark.parametrize("method", _METHODS)
def test_builtin_registration_has_schema_parity_and_all_three_verbs(method: str) -> None:
    schema = method_config_schema(method)
    runtime = resolve_method({"method": method})
    if method == "weighted-per-step":
        method_fields = {"weight_decay"}
    elif method == "sequential-adaptive-per-step":
        method_fields = {"learning_rate"}
    else:
        method_fields = {
            "upper_floor",
            "upper_cap",
            *({"protection_period"} if method == "split-window-sum" else set()),
        }

    assert isinstance(runtime, ConformalRuntime)
    assert type(runtime.config) is schema
    assert set(schema.model_fields) == {
        "coverage",
        "calibration_window",
        "partition_by",
        *method_fields,
    }
    for verb in ("calibrate", "apply", "observe"):
        assert callable(getattr(runtime, verb))
    assert not hasattr(runtime, "load_state")
    assert "load_state" not in inspect.getsource(ConformalRuntime)


@pytest.mark.parametrize("method", _METHODS)
def test_builtin_protocol_covers_below_and_at_readiness_with_complete_descriptor(
    method: str,
) -> None:
    runtime = resolve_method({"method": method})
    label = derive_partition_label(
        "protocol-model",
        "global",
        runtime.manifest.emission_scope,
    )
    below_states = runtime.calibrate(CalibrationSeedBatch({label: list(range(1, 10))}))
    below = runtime.apply(_frame(), below_states)
    ready_states = runtime.calibrate(CalibrationSeedBatch({label: list(range(1, 11))}))
    ready = runtime.apply(_frame(), ready_states)
    lower, upper = interval_columns(0.9)
    below_facts = next(iter(below.issuances.values()))
    ready_facts = next(iter(ready.issuances.values()))

    assert runtime.manifest.minimum_calibration_scores(runtime.config) == 10
    assert math.isnan(below.forecasts.loc[0, lower])
    assert math.isnan(below.forecasts.loc[0, upper])
    assert below_facts.bounds_null_reason == "warm-up"
    assert ready.forecasts.loc[0, lower] == 0.0
    assert ready.forecasts.loc[0, upper] == 14.0
    assert ready_facts.method_name == method
    assert ready_facts.emission_form is runtime.manifest.emission_form
    assert ready_facts.emission_scope is runtime.manifest.emission_scope
    assert ready_facts.effective_descriptor.type.claim is GuaranteeClaim.ONE_SIDED_COVERAGE
    expected_currency = (
        GuaranteeCurrency.LONG_RUN_PATHWISE
        if method == "sequential-adaptive-per-step"
        else GuaranteeCurrency.FINITE_SAMPLE_MARGINAL
    )
    assert ready_facts.effective_descriptor.type.currency is expected_currency
    assert ready_facts.effective_descriptor.scope.kind is DecisionScopeKind.PER_DECISION_NODE
    assert ready_facts.effective_descriptor.level == 0.9
    assert ready_facts.working_level == pytest.approx(0.1)


@pytest.mark.parametrize("method", _METHODS)
def test_builtin_observe_replay_and_factory_restoration_are_exact(method: str) -> None:
    original = resolve_method({"method": method})
    label = derive_partition_label(
        "protocol-model",
        "global",
        original.manifest.emission_scope,
    )
    states = original.calibrate(CalibrationSeedBatch({label: list(range(1, 11))}))
    issued = original.apply(_frame(), states)
    delivery = delivery_batch(label, (_observation(issued),))

    first = original.observe(delivery, states)
    restored = resolve_method({"method": method}, states=states)
    second = restored.observe(delivery, states)

    assert restored is not original
    assert first.annotations == second.annotations
    assert first.dirty_state == second.dirty_state
    assert resolve_method({"method": method}, states=first.state).manifest == original.manifest
