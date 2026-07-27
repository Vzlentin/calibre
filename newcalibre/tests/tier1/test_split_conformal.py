"""Exercise split-conformal mathematics, state, censoring, and attribution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import pandas as pd
import pytest

import newcalibre.conformal.methods.split as split_method
from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    SPLIT_PER_STEP_MANIFEST,
    CalibrationResult,
    ConformalRegistryError,
    Delivery,
    EmissionForm,
    ForecastKey,
    ResolvedObservation,
    RuntimeContractError,
    derive_partition_label,
    resolve_method,
)
from newcalibre.conformal.state import JsonStateCodec
from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    CensoringAssertion,
    EmissionScope,
    GuaranteeClaim,
    ScoredSeries,
    interval_columns,
)

pytestmark = pytest.mark.tier1
_ORIGIN = pd.Timestamp("2026-01-05")
_MODEL = "fixture-model"


def _frame(
    points: tuple[float, ...] = (4.0,),
    *,
    series_key: str = "sku",
    actuals: tuple[object, ...] | None = None,
) -> pd.DataFrame:
    horizon = tuple(range(1, len(points) + 1))
    values = (float("nan"),) * len(points) if actuals is None else actuals
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series([series_key] * len(points), dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime(
                [_ORIGIN + pd.Timedelta(days=step - 1) for step in horizon]
            ),
            ACTUAL_VALUE: list(values),
            POINT_FORECAST: pd.Series(points, dtype="float64"),
            HORIZON_STEP: pd.Series(horizon, dtype="int64"),
            ORIGIN: pd.to_datetime([_ORIGIN] * len(points)),
            MODEL_NAME: pd.Series([_MODEL] * len(points), dtype="string"),
        }
    )


def _partition(
    scope: EmissionScope,
    *,
    series_key: str = "sku",
    partition_by: str = "global",
    horizon_step: int = 1,
) -> str:
    value = "global" if partition_by == "global" else series_key
    return derive_partition_label(
        _MODEL,
        value,
        scope,
        horizon_step=horizon_step if partition_by == "series-horizon" else None,
    )


def _states(
    method: str,
    scores: list[float],
    *,
    configuration: Mapping[str, object] | None = None,
    series_key: str = "sku",
) -> tuple[object, str, Mapping[str, bytes]]:
    payload = {"method": method, **({} if configuration is None else configuration)}
    runtime = resolve_method(payload)
    label = _partition(
        runtime.manifest.emission_scope,
        series_key=series_key,
        partition_by=str(payload.get("partition_by", "global")),
    )
    states = runtime.calibrate({label: scores})
    return runtime, label, states


def _observations(
    result: CalibrationResult,
    actuals: tuple[float, ...],
    assertions: tuple[CensoringAssertion | None, ...],
) -> tuple[ResolvedObservation, ...]:
    frame = result.forecasts
    observations: list[ResolvedObservation] = []
    for position, row in enumerate(frame.to_dict("records")):
        key = ForecastKey(
            series_key=row[SERIES_KEY],
            origin=pd.Timestamp(row[ORIGIN]),
            horizon_step=row[HORIZON_STEP],
            model_name=row[MODEL_NAME],
        )
        observations.append(
            ResolvedObservation(
                forecast_key=key,
                target_timestamp=pd.Timestamp(row[TARGET_TIMESTAMP]),
                actual=actuals[position],
                point_forecast=row[POINT_FORECAST],
                censoring_assertion=assertions[position],
                availability_bound=None,
                issued=result.issuances[key],
            )
        )
    return tuple(observations)


def _payload(method: str, state: bytes, *, label: str) -> dict[str, object]:
    decoded = JsonStateCodec(method, 1).decode(state, expected_label=label)
    assert isinstance(decoded, dict)
    return decoded


def test_rank_readiness_uses_the_strict_boundary_and_conservative_order_statistic() -> None:
    runtime, label, nine = _states("split-per-step", list(range(1, 10)))
    below = runtime.apply(_frame(), nine)
    lower, upper = interval_columns(0.9)
    facts = next(iter(below.issuances.values()))

    assert SPLIT_PER_STEP_MANIFEST.minimum_calibration_scores(runtime.config) == 10
    assert math.isnan(below.forecasts.loc[0, lower])
    assert math.isnan(below.forecasts.loc[0, upper])
    assert not facts.calibration_ready
    assert facts.bounds_null_reason == "warm-up"

    ten = runtime.calibrate({label: list(range(1, 11))})
    ready = runtime.apply(_frame(), ten)
    assert ready.forecasts.loc[0, lower] == 0.0
    assert ready.forecasts.loc[0, upper] == 14.0
    ready_facts = next(iter(ready.issuances.values()))
    assert ready_facts.calibration_ready
    assert ready_facts.working_level == pytest.approx(0.1)
    assert ready_facts.effective_descriptor.level == 0.9

    nonmax_runtime, nonmax_label, nonmax = _states(
        "split-per-step",
        list(range(1, 11)),
        configuration={"coverage": 0.8},
    )
    nonmax_result = nonmax_runtime.apply(_frame(), nonmax)
    assert nonmax_result.forecasts.loc[0, interval_columns(0.8)[1]] == 13.0
    assert nonmax_label in nonmax


def test_calibration_is_deterministic_bounded_and_rejects_nonfinite_scores() -> None:
    runtime, label, _ = _states(
        "split-per-step",
        [],
        configuration={"coverage": 0.5, "calibration_window": 2},
    )
    first = runtime.calibrate({label: [1.0, 2.0, 3.0]})
    second = runtime.calibrate({label: [1.0, 2.0, 3.0]})
    payload = _payload("split-per-step", first[label], label=label)

    assert first == second
    assert payload["scores"] == [2.0, 3.0]
    assert payload["delivered_score_count"] == 3
    assert payload["scored_series"] == "demand-honest"
    assert _payload(
        "split-per-step",
        first[METHOD_SCOPE_LABEL],
        label=METHOD_SCOPE_LABEL,
    ) == {"issue_counter": 0}

    for score in (math.nan, math.inf, -1.0):
        with pytest.raises(RuntimeContractError, match="scores"):
            runtime.calibrate({label: [score]})


def test_series_partitioning_uses_independent_score_states() -> None:
    runtime = resolve_method(
        {
            "method": "split-per-step",
            "coverage": 0.5,
            "partition_by": "series",
        }
    )
    a_label = _partition(EmissionScope.PER_STEP, series_key="a", partition_by="series")
    b_label = _partition(EmissionScope.PER_STEP, series_key="b", partition_by="series")
    states = runtime.calibrate({a_label: [1.0, 2.0], b_label: [10.0, 20.0]})
    forecasts = pd.concat(
        [_frame(series_key="a"), _frame(series_key="b")],
        ignore_index=True,
    )

    result = runtime.apply(forecasts, states)

    assert result.forecasts[interval_columns(0.5)[1]].tolist() == [6.0, 24.0]
    assert [facts.partition_label for facts in result.issuances.values()] == [
        a_label,
        b_label,
    ]


def test_series_horizon_partitioning_isolates_state_and_readiness_per_step() -> None:
    runtime = resolve_method(
        {
            "method": "split-per-step",
            "coverage": 0.5,
            "partition_by": "series-horizon",
        }
    )
    first_label = _partition(
        EmissionScope.PER_STEP,
        partition_by="series-horizon",
        horizon_step=1,
    )
    second_label = _partition(
        EmissionScope.PER_STEP,
        partition_by="series-horizon",
        horizon_step=2,
    )
    assert first_label != second_label
    states = runtime.calibrate({first_label: [1.0, 2.0], second_label: [10.0]})

    result = runtime.apply(_frame((4.0, 4.0)), states)
    lower, upper = interval_columns(0.5)

    assert result.forecasts[upper].tolist()[0] == 6.0
    assert math.isnan(result.forecasts[lower].tolist()[1])
    assert math.isnan(result.forecasts[upper].tolist()[1])
    assert [facts.partition_label for facts in result.issuances.values()] == [
        first_label,
        second_label,
    ]
    assert [facts.calibration_ready for facts in result.issuances.values()] == [True, False]


def test_series_horizon_delivery_updates_only_its_declared_step() -> None:
    runtime = resolve_method(
        {
            "method": "split-per-step",
            "coverage": 0.5,
            "partition_by": "series-horizon",
        }
    )
    first_label = _partition(
        EmissionScope.PER_STEP,
        partition_by="series-horizon",
        horizon_step=1,
    )
    second_label = _partition(
        EmissionScope.PER_STEP,
        partition_by="series-horizon",
        horizon_step=2,
    )
    states = runtime.calibrate({first_label: [1.0, 2.0], second_label: [10.0, 20.0]})
    issued = runtime.apply(_frame((4.0, 4.0)), states)
    observations = _observations(
        issued,
        (7.0, 8.0),
        (CensoringAssertion.UNCENSORED, CensoringAssertion.UNCENSORED),
    )

    effect = runtime.observe(Delivery(first_label, (observations[0],)), states)

    assert set(effect.state_updates) == {first_label}
    assert second_label not in effect.state_updates
    with pytest.raises(RuntimeContractError, match="issued partition"):
        Delivery(first_label, (observations[1],))


def test_series_horizon_labels_cannot_collide_with_series_values() -> None:
    series_label = derive_partition_label(
        _MODEL,
        "sku@1",
        EmissionScope.PER_STEP,
    )
    horizon_label = derive_partition_label(
        _MODEL,
        "sku",
        EmissionScope.PER_STEP,
        horizon_step=1,
    )

    assert series_label != horizon_label


def test_apply_ignores_poisoned_actuals_and_advances_only_method_issue_state() -> None:
    class Poison:
        def __float__(self) -> float:
            raise AssertionError("apply read actual_value")

    runtime, label, states = _states("split-per-step", list(range(1, 11)))
    result = runtime.apply(_frame(actuals=(Poison(),)), states)

    assert result.forecasts.loc[0, interval_columns(0.9)[1]] == 14.0
    assert set(result.state_updates) == {METHOD_SCOPE_LABEL}
    assert label not in result.state_updates
    assert _payload(
        "split-per-step",
        result.state_updates[METHOD_SCOPE_LABEL],
        label=METHOD_SCOPE_LABEL,
    ) == {"issue_counter": 1}


def test_window_sum_emits_only_on_the_terminal_complete_leading_window() -> None:
    runtime, label, states = _states(
        "split-window-sum",
        list(range(1, 11)),
        configuration={"protection_period": 3},
    )
    result = runtime.apply(_frame((2.0, 3.0, 4.0, 100.0)), states)
    lower, upper = interval_columns(0.9)
    frame = result.forecasts

    assert frame[lower].iloc[:2].isna().all()
    assert frame[upper].iloc[:2].isna().all()
    assert frame.loc[2, upper] == 19.0
    assert math.isnan(frame.loc[3, upper])
    reasons = [facts.bounds_null_reason for facts in result.issuances.values()]
    assert reasons == ["emission-scope", "emission-scope", None, "emission-scope"]
    assert all(facts.partition_label == label for facts in result.issuances.values())

    with pytest.raises(RuntimeContractError, match="leading protection-window"):
        runtime.apply(_frame((2.0, 3.0)).iloc[[1]].assign(horizon_step=3), states)


def test_window_apply_rejects_a_large_incomplete_period_before_range_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protection_period = 10**12
    runtime = resolve_method(
        {
            "method": "split-window-sum",
            "protection_period": protection_period,
        }
    )
    terminal_only = _frame().assign(horizon_step=protection_period)

    def fail_on_range(*_args: object) -> None:
        raise AssertionError("incomplete window allocated over the protection period")

    monkeypatch.setattr(split_method, "range", fail_on_range, raising=False)

    with pytest.raises(RuntimeContractError, match="leading protection-window"):
        runtime.apply(terminal_only, {})


def test_per_step_observe_handles_declared_censored_and_sticky_undeclared_series() -> None:
    runtime, label, states = _states(
        "split-per-step",
        [1.0, 2.0],
        configuration={"coverage": 0.5, "calibration_window": 3},
    )
    issued = runtime.apply(_frame((4.0, 5.0, 6.0)), states)
    delivery = Delivery(
        label,
        _observations(
            issued,
            (7.0, 20.0, 8.0),
            (CensoringAssertion.UNCENSORED, CensoringAssertion.CENSORED, None),
        ),
    )
    effect = runtime.observe(delivery, states)
    payload = _payload("split-per-step", effect.state_updates[label], label=label)

    assert [annotation.score for annotation in effect.annotations] == [3.0, None, 2.0]
    assert effect.annotations[1].exclusion_cause == "declared-censored"
    assert [annotation.advanced_delivered_score for annotation in effect.annotations] == [
        True,
        False,
        True,
    ]
    assert payload["scores"] == [2.0, 3.0, 2.0]
    assert payload["delivered_score_count"] == 4
    assert payload["scored_series"] == "recorded-sales"

    later = runtime.apply(_frame(), {**states, label: effect.state_updates[label]})
    later_facts = next(iter(later.issuances.values()))
    assert later_facts.effective_descriptor.scored_series is ScoredSeries.RECORDED_SALES
    before_reference = next(iter(issued.issuances.values())).state_reference
    assert later_facts.state_reference != before_reference


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("method", "wrong split method"),
        ("form", "wrong emission form"),
        ("scope", "wrong emission scope"),
        ("working-level", "wrong working alpha"),
    ],
)
def test_observe_rejects_tampered_issuance_identity_before_state_advancement(
    mismatch: str,
    message: str,
) -> None:
    runtime, label, states = _states(
        "split-per-step",
        [1.0, 2.0],
        configuration={"coverage": 0.5},
    )
    result = runtime.apply(_frame(), states)
    observation = _observations(
        result,
        (7.0,),
        (CensoringAssertion.UNCENSORED,),
    )[0]
    issued = observation.issued
    if mismatch == "method":
        tampered = replace(issued, method_name="stale-method")
    elif mismatch == "form":
        tampered = replace(issued, emission_form=EmissionForm.ONE_SIDED_LOWER)
    elif mismatch == "scope":
        tampered = replace(
            issued,
            emission_scope=EmissionScope.WINDOW_SUM,
            effective_descriptor=replace(
                issued.effective_descriptor,
                window=EmissionScope.WINDOW_SUM,
            ),
        )
    else:
        tampered = replace(issued, working_level=0.6)
    before = dict(states)

    with pytest.raises(RuntimeContractError, match=message):
        runtime.observe(
            Delivery(label, (replace(observation, issued=tampered),)),
            states,
        )

    assert states == before


def test_declared_censoring_without_an_undeclared_score_stays_demand_honest() -> None:
    runtime, label, states = _states(
        "split-per-step",
        [1.0, 2.0],
        configuration={"coverage": 0.5},
    )
    issued = runtime.apply(_frame((4.0, 5.0)), states)
    effect = runtime.observe(
        Delivery(
            label,
            _observations(
                issued,
                (6.0, 99.0),
                (CensoringAssertion.UNCENSORED, CensoringAssertion.CENSORED),
            ),
        ),
        states,
    )
    payload = _payload("split-per-step", effect.state_updates[label], label=label)

    assert payload["delivered_score_count"] == 3
    assert payload["scored_series"] == "demand-honest"


def test_window_observe_scores_once_on_terminal_and_preserves_canonical_annotations() -> None:
    runtime, label, states = _states(
        "split-window-sum",
        [1.0, 2.0],
        configuration={"coverage": 0.5, "protection_period": 3},
    )
    issued = runtime.apply(_frame((2.0, 3.0, 4.0)), states)
    observations = _observations(
        issued,
        (4.0, 4.0, 7.0),
        (CensoringAssertion.UNCENSORED,) * 3,
    )
    effect = runtime.observe(Delivery(label, observations), states)
    payload = _payload("split-window-sum", effect.state_updates[label], label=label)

    assert [annotation.forecast_key.horizon_step for annotation in effect.annotations] == [
        1,
        2,
        3,
    ]
    assert [annotation.score for annotation in effect.annotations] == [None, None, 6.0]
    assert [annotation.advanced_delivered_score for annotation in effect.annotations] == [
        False,
        False,
        True,
    ]
    assert payload["scores"] == [1.0, 2.0, 6.0]
    assert payload["delivered_score_count"] == 3


def test_window_observe_batches_canonical_windows_like_consecutive_calls() -> None:
    runtime, label, states = _states(
        "split-window-sum",
        [1.0, 2.0],
        configuration={"coverage": 0.5, "protection_period": 2},
    )
    issued = runtime.apply(_frame((2.0, 3.0)), states)
    first = _observations(
        issued,
        (4.0, 5.0),
        (CensoringAssertion.UNCENSORED,) * 2,
    )
    next_origin = _ORIGIN + pd.Timedelta(days=1)
    second = tuple(
        replace(
            observation,
            forecast_key=replace(observation.forecast_key, origin=next_origin),
            target_timestamp=observation.target_timestamp + pd.Timedelta(days=1),
        )
        for observation in first
    )

    batched = runtime.observe(Delivery(label, (*first, *second)), states)
    first_effect = runtime.observe(Delivery(label, first), states)
    consecutive = runtime.observe(
        Delivery(label, second),
        {**states, **first_effect.state_updates},
    )

    assert batched.state_updates == consecutive.state_updates
    assert batched.annotations == (*first_effect.annotations, *consecutive.annotations)
    payload = _payload("split-window-sum", batched.state_updates[label], label=label)
    assert payload["delivered_score_count"] == 4


def test_window_censoring_excludes_the_composite_without_state_advancement() -> None:
    runtime, label, states = _states(
        "split-window-sum",
        [1.0, 2.0],
        configuration={"coverage": 0.5, "protection_period": 3},
    )
    issued = runtime.apply(_frame((2.0, 3.0, 4.0)), states)
    effect = runtime.observe(
        Delivery(
            label,
            _observations(
                issued,
                (4.0, 4.0, 7.0),
                (
                    CensoringAssertion.UNCENSORED,
                    CensoringAssertion.CENSORED,
                    None,
                ),
            ),
        ),
        states,
    )

    assert effect.state_updates[label] == states[label]
    assert all(annotation.score is None for annotation in effect.annotations)
    assert all(
        annotation.exclusion_cause == "declared-censored-window"
        for annotation in effect.annotations
    )


def test_window_observe_refuses_partial_foreign_out_of_range_and_noncanonical_members() -> None:
    runtime, label, states = _states(
        "split-window-sum",
        [1.0, 2.0],
        configuration={"coverage": 0.5, "protection_period": 3},
    )
    issued = runtime.apply(_frame((2.0, 3.0, 4.0)), states)
    observations = _observations(
        issued,
        (4.0, 4.0, 7.0),
        (CensoringAssertion.UNCENSORED,) * 3,
    )

    with pytest.raises(RuntimeContractError, match="complete protection window"):
        runtime.observe(Delivery(label, observations[:2]), states)
    with pytest.raises(RuntimeContractError, match="duplicate forecast key"):
        Delivery(label, (observations[0], observations[0], observations[2]))
    with pytest.raises(RuntimeContractError, match="canonical horizon steps"):
        runtime.observe(
            Delivery(label, (observations[1], observations[0], observations[2])), states
        )

    foreign = replace(
        observations[1],
        forecast_key=replace(observations[1].forecast_key, series_key="foreign"),
    )
    with pytest.raises(
        RuntimeContractError,
        match="declared partition|share series|complete protection window",
    ):
        runtime.observe(Delivery(label, (observations[0], foreign, observations[2])), states)

    out_of_range = replace(
        observations[2],
        forecast_key=replace(observations[2].forecast_key, horizon_step=4),
    )
    with pytest.raises(RuntimeContractError, match="canonical horizon steps"):
        runtime.observe(Delivery(label, (observations[0], observations[1], out_of_range)), states)


def test_clamps_record_binding_per_finite_row_and_void_only_changed_claims() -> None:
    runtime, _label, states = _states(
        "split-per-step",
        [1.0, 2.0],
        configuration={
            "coverage": 0.5,
            "upper_floor": 10.0,
            "upper_cap": 20.0,
        },
    )
    result = runtime.apply(_frame((4.0, 30.0)), states)
    facts = tuple(result.issuances.values())

    assert result.forecasts[interval_columns(0.5)[1]].tolist() == [10.0, 20.0]
    assert [(binding.name, binding.bound) for binding in facts[0].bindings] == [
        ("upper_floor", True),
        ("upper_cap", False),
    ]
    assert [(binding.name, binding.bound) for binding in facts[1].bindings] == [
        ("upper_floor", False),
        ("upper_cap", True),
    ]
    assert all(issued.effective_descriptor.type.claim is GuaranteeClaim.NONE for issued in facts)

    identity_runtime, _identity_label, identity_states = _states(
        "split-per-step",
        [1.0, 2.0],
        configuration={"coverage": 0.5},
    )
    identity = identity_runtime.apply(_frame((4.0, 30.0)), identity_states)
    nonbinding_runtime, _nonbinding_label, nonbinding_states = _states(
        "split-per-step",
        [1.0, 2.0],
        configuration={"coverage": 0.5, "upper_floor": 0.0, "upper_cap": 100.0},
    )
    nonbinding = nonbinding_runtime.apply(_frame((4.0, 30.0)), nonbinding_states)
    assert identity.forecasts.equals(nonbinding.forecasts)
    assert all(
        issued.effective_descriptor.type.claim is GuaranteeClaim.ONE_SIDED_COVERAGE
        for issued in nonbinding.issuances.values()
    )


def test_warmup_never_falls_through_a_configured_clamp() -> None:
    runtime, _label, states = _states(
        "split-per-step",
        list(range(1, 10)),
        configuration={"upper_floor": 100.0, "upper_cap": 200.0},
    )
    result = runtime.apply(_frame(), states)
    facts = next(iter(result.issuances.values()))

    assert math.isnan(result.forecasts.loc[0, interval_columns(0.9)[1]])
    assert facts.bindings == ()
    assert facts.bounds_null_reason == "warm-up"


@pytest.mark.parametrize(
    "configuration",
    [
        {"coverage": 0.0},
        {"coverage": 1.0},
        {"coverage": math.nan},
        {"calibration_window": 0},
        {"calibration_window": 5001},
        {"coverage": 0.9999, "calibration_window": 10},
        {"upper_floor": math.inf},
        {"upper_cap": math.nan},
        {"upper_cap": -1.0},
        {"upper_floor": 2.0, "upper_cap": 1.0},
    ],
)
def test_split_config_rejects_invalid_rank_window_and_clamps(
    configuration: dict[str, object],
) -> None:
    with pytest.raises(ConformalRegistryError, match="invalid configuration"):
        resolve_method({"method": "split-per-step", **configuration})


@pytest.mark.parametrize(
    "configuration",
    [
        {"protection_period": 0},
        {"partition_by": "series-horizon"},
    ],
)
def test_window_config_rejects_invalid_scope_configuration(
    configuration: dict[str, object],
) -> None:
    with pytest.raises(ConformalRegistryError, match="invalid configuration"):
        resolve_method({"method": "split-window-sum", **configuration})


def test_split_private_codec_rejects_wrong_scope_and_malformed_restoration_payloads() -> None:
    wrong_scope = _partition(EmissionScope.WINDOW_SUM)
    codec = JsonStateCodec("split-per-step", 1)
    valid_payload = {
        "delivered_score_count": 1,
        "scored_series": "demand-honest",
        "scores": [1.0],
    }
    wrong_scope_state = codec.encode(wrong_scope, valid_payload)
    malformed_label = _partition(EmissionScope.PER_STEP)
    malformed_state = codec.encode(malformed_label, {**valid_payload, "unknown": True})

    with pytest.raises(ConformalRegistryError, match="wrong method emission scope"):
        resolve_method(
            {"method": "split-per-step"},
            states={wrong_scope: wrong_scope_state},
        )
    with pytest.raises(ConformalRegistryError, match="exact fields"):
        resolve_method(
            {"method": "split-per-step"},
            states={malformed_label: malformed_state},
        )
