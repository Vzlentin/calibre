"""Exercise sequential-adaptive conformal state, issuance, and feedback."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import pandas as pd
import pytest

from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST,
    AssumptionClass,
    CalibrationResult,
    CensoringPolicy,
    ConformalRegistryError,
    Delivery,
    EmissionForm,
    ForecastKey,
    PostWarmupNonFinite,
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
    GuaranteeCurrency,
    ScoredSeries,
    interval_columns,
)

pytestmark = pytest.mark.tier1
_ORIGIN = pd.Timestamp("2026-05-04")
_MODEL = "adaptive-fixture"
_METHOD = "sequential-adaptive-per-step"


def _frame(
    points: tuple[float, ...] = (4.0,),
    *,
    series_key: str = "sku",
    actuals: tuple[object, ...] | None = None,
) -> pd.DataFrame:
    steps = tuple(range(1, len(points) + 1))
    values = (float("nan"),) * len(points) if actuals is None else actuals
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series([series_key] * len(points), dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime(
                [_ORIGIN + pd.Timedelta(days=step - 1) for step in steps]
            ),
            ACTUAL_VALUE: list(values),
            POINT_FORECAST: pd.Series(points, dtype="float64"),
            HORIZON_STEP: pd.Series(steps, dtype="int64"),
            ORIGIN: pd.to_datetime([_ORIGIN] * len(points)),
            MODEL_NAME: pd.Series([_MODEL] * len(points), dtype="string"),
        }
    )


def _partition(*, series_key: str = "sku", partition_by: str = "global") -> str:
    value = "global" if partition_by == "global" else series_key
    return derive_partition_label(_MODEL, value, EmissionScope.PER_STEP)


def _states(
    scores: list[float],
    *,
    configuration: Mapping[str, object] | None = None,
    series_key: str = "sku",
):
    payload = {"method": _METHOD, **({} if configuration is None else configuration)}
    runtime = resolve_method(payload)
    label = _partition(
        series_key=series_key,
        partition_by=str(payload.get("partition_by", "global")),
    )
    return runtime, label, runtime.calibrate({label: scores})


def _observations(
    result: CalibrationResult,
    actuals: tuple[float, ...],
    assertions: tuple[CensoringAssertion | None, ...],
) -> tuple[ResolvedObservation, ...]:
    observations: list[ResolvedObservation] = []
    for position, row in enumerate(result.forecasts.to_dict("records")):
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


def _payload(state: bytes, *, label: str) -> dict[str, object]:
    decoded = JsonStateCodec(_METHOD, 1).decode(state, expected_label=label)
    assert isinstance(decoded, dict)
    return decoded


def _partition_state(
    label: str,
    *,
    scores: list[float],
    raw_alpha: object,
    delivered_score_count: int | None = None,
    feedback_count: int = 0,
    scored_series: str = "demand-honest",
) -> bytes:
    return JsonStateCodec(_METHOD, 1).encode(
        label,
        {
            "delivered_score_count": (
                len(scores) if delivered_score_count is None else delivered_score_count
            ),
            "feedback_count": feedback_count,
            "raw_alpha": raw_alpha,
            "scored_series": scored_series,
            "scores": scores,
        },
    )


def test_manifest_defaults_and_configuration_surface_are_exact() -> None:
    runtime = resolve_method({"method": _METHOD})

    assert runtime.config.model_dump() == {
        "coverage": 0.9,
        "calibration_window": 5000,
        "partition_by": "global",
        "learning_rate": 0.05,
    }
    assert set(type(runtime.config).model_fields) == {
        "coverage",
        "calibration_window",
        "partition_by",
        "learning_rate",
    }
    assert SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.assumption_class is (
        AssumptionClass.SEQUENTIAL_ADAPTIVE
    )
    assert SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.emission_form is EmissionForm.ONE_SIDED_UPPER
    assert SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.emission_scope is EmissionScope.PER_STEP
    assert SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.censoring_policy is (
        CensoringPolicy.CONSUMES_CENSORING_FACTS
    )
    assert SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.state_bound == 5000
    assert SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.order_sensitive
    assert SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.clamps == ()
    assert SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.post_warmup_non_finite is (
        PostWarmupNonFinite.ALLOWED_WITH_ATTRIBUTION
    )
    declaration = SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.guarantees[0]
    assert declaration.claim is GuaranteeClaim.ONE_SIDED_COVERAGE
    assert declaration.currency is GuaranteeCurrency.LONG_RUN_PATHWISE


@pytest.mark.parametrize(
    "configuration",
    [
        {"coverage": 0.0},
        {"coverage": 1.0},
        {"coverage": math.nan},
        {"calibration_window": 0},
        {"calibration_window": 5001},
        {"coverage": 0.9999, "calibration_window": 10},
        {"partition_by": "category"},
        {"learning_rate": -0.1},
        {"learning_rate": math.nan},
        {"learning_rate": math.inf},
        {"burn_in": 10},
        {"prefix_count": 10},
        {"alpha_clamp": True},
    ],
)
def test_configuration_rejects_invalid_values_and_reference_policy_knobs(
    configuration: dict[str, object],
) -> None:
    with pytest.raises(ConformalRegistryError, match="invalid configuration"):
        resolve_method({"method": _METHOD, **configuration})


def test_calibration_is_deterministic_bounded_and_holds_raw_alpha_at_target() -> None:
    runtime, label, _states_value = _states(
        [],
        configuration={"coverage": 0.5, "calibration_window": 3},
    )

    first = runtime.calibrate({label: [8.0, 9.0, 1.0, 2.0]})
    second = runtime.calibrate({label: [8.0, 9.0, 1.0, 2.0]})

    assert first == second
    assert _payload(first[label], label=label) == {
        "delivered_score_count": 4,
        "feedback_count": 0,
        "raw_alpha": 0.5,
        "scored_series": "demand-honest",
        "scores": [9.0, 1.0, 2.0],
    }
    assert _payload(first[METHOD_SCOPE_LABEL], label=METHOD_SCOPE_LABEL) == {"issue_counter": 0}
    for score in (math.nan, math.inf, -1.0):
        with pytest.raises(RuntimeContractError, match="scores"):
            runtime.calibrate({label: [score]})


def test_first_ready_issuance_uses_target_alpha_and_numpy_higher_quantile() -> None:
    runtime, _label, states = _states(
        [9.0, 1.0, 5.0],
        configuration={"coverage": 0.5, "calibration_window": 3},
    )

    result = runtime.apply(_frame(), states)
    facts = next(iter(result.issuances.values()))
    lower, upper = interval_columns(0.5)

    assert result.forecasts.loc[0, lower] == 0.0
    assert result.forecasts.loc[0, upper] == 9.0
    assert facts.working_level == 0.5
    assert facts.effective_descriptor.level == 0.5
    assert facts.effective_descriptor.type.currency is GuaranteeCurrency.LONG_RUN_PATHWISE
    assert facts.bounds_null_reason is None


def test_series_partitions_are_independent_and_apply_ignores_poisoned_actuals() -> None:
    class Poison:
        def __float__(self) -> float:
            raise AssertionError("apply read actual_value")

    runtime = resolve_method(
        {
            "method": _METHOD,
            "coverage": 0.5,
            "partition_by": "series",
        }
    )
    a_label = _partition(series_key="a", partition_by="series")
    b_label = _partition(series_key="b", partition_by="series")
    states = runtime.calibrate({a_label: [1.0, 2.0], b_label: [10.0, 20.0]})
    forecasts = pd.concat(
        [
            _frame(series_key="a", actuals=(Poison(),)),
            _frame(series_key="b", actuals=(Poison(),)),
        ],
        ignore_index=True,
    )

    result = runtime.apply(forecasts, states)

    assert result.forecasts[interval_columns(0.5)[1]].tolist() == [6.0, 24.0]
    assert [facts.partition_label for facts in result.issuances.values()] == [
        a_label,
        b_label,
    ]
    assert set(result.state_updates) == {METHOD_SCOPE_LABEL}


def test_warmup_scores_advance_without_feedback_then_first_ready_issue_stays_at_target() -> None:
    runtime, label, states = _states(
        [1.0],
        configuration={"coverage": 0.5, "learning_rate": 0.25},
    )
    warm = runtime.apply(_frame(), states)
    warm_facts = next(iter(warm.issuances.values()))
    observed = runtime.observe(
        Delivery(
            label,
            _observations(
                warm,
                (7.0,),
                (CensoringAssertion.UNCENSORED,),
            ),
        ),
        states,
    )
    payload = _payload(observed.state_updates[label], label=label)
    ready = runtime.apply(
        _frame(),
        {**states, **warm.state_updates, **observed.state_updates},
    )
    ready_facts = next(iter(ready.issuances.values()))

    assert not warm_facts.calibration_ready
    assert warm_facts.bounds_null_reason == "warm-up"
    assert payload["scores"] == [1.0, 3.0]
    assert payload["feedback_count"] == 0
    assert payload["raw_alpha"] == 0.5
    assert ready_facts.calibration_ready
    assert ready_facts.working_level == 0.5


def test_closed_boundary_hits_and_misses_follow_the_hand_derived_recurrence() -> None:
    runtime, label, states = _states(
        [1.0, 3.0],
        configuration={"coverage": 0.5, "learning_rate": 0.25},
    )
    first = runtime.apply(_frame(), states)
    first_observe = runtime.observe(
        Delivery(
            label,
            _observations(
                first,
                (7.0,),
                (CensoringAssertion.UNCENSORED,),
            ),
        ),
        states,
    )
    after_hit = _payload(first_observe.state_updates[label], label=label)
    second_states = {**states, **first.state_updates, **first_observe.state_updates}
    second = runtime.apply(_frame(), second_states)
    second_observe = runtime.observe(
        Delivery(
            label,
            _observations(
                second,
                (8.0,),
                (CensoringAssertion.UNCENSORED,),
            ),
        ),
        second_states,
    )
    after_miss = _payload(second_observe.state_updates[label], label=label)

    assert first.forecasts.loc[0, interval_columns(0.5)[1]] == 7.0
    assert first_observe.annotations[0].score == 3.0
    assert after_hit["raw_alpha"] == 0.625
    assert after_hit["feedback_count"] == 1
    assert after_miss["raw_alpha"] == 0.5
    assert after_miss["feedback_count"] == 2


def test_raw_alpha_excursions_are_unclipped_but_only_quantile_input_is_clipped() -> None:
    runtime, label, states = _states(
        [1.0, 3.0],
        configuration={"coverage": 0.5, "learning_rate": 2.0},
    )
    issued = runtime.apply(_frame(), states)
    miss = runtime.observe(
        Delivery(
            label,
            _observations(
                issued,
                (8.0,),
                (CensoringAssertion.UNCENSORED,),
            ),
        ),
        states,
    )
    below_payload = _payload(miss.state_updates[label], label=label)
    below_states = {**states, **issued.state_updates, **miss.state_updates}
    unresolvable = runtime.apply(_frame(), below_states)
    returned = runtime.observe(
        Delivery(
            label,
            _observations(
                unresolvable,
                (100.0,),
                (CensoringAssertion.UNCENSORED,),
            ),
        ),
        below_states,
    )

    cover_runtime, cover_label, cover_states = _states(
        [1.0, 3.0],
        configuration={"coverage": 0.5, "learning_rate": 2.0},
    )
    cover_issue = cover_runtime.apply(_frame(), cover_states)
    cover = cover_runtime.observe(
        Delivery(
            cover_label,
            _observations(
                cover_issue,
                (7.0,),
                (CensoringAssertion.UNCENSORED,),
            ),
        ),
        cover_states,
    )
    above_states = {**cover_states, **cover_issue.state_updates, **cover.state_updates}
    clipped = cover_runtime.apply(_frame(), above_states)
    above_payload = _payload(cover.state_updates[cover_label], label=cover_label)
    clipped_facts = next(iter(clipped.issuances.values()))

    assert below_payload["raw_alpha"] == -0.5
    assert next(iter(unresolvable.issuances.values())).bounds_null_reason == (
        "unresolvable-working-level"
    )
    assert _payload(returned.state_updates[label], label=label)["raw_alpha"] == 0.5
    assert above_payload["raw_alpha"] == 1.5
    assert clipped_facts.working_level == 1.5
    assert clipped.forecasts.loc[0, interval_columns(0.5)[1]] == 5.0


def test_exact_active_window_trigger_attributes_nonfinite_and_trivial_cover_returns() -> None:
    runtime = resolve_method(
        {
            "method": _METHOD,
            "coverage": 0.5,
            "learning_rate": 0.2,
        }
    )
    label = _partition()
    state = _partition_state(
        label,
        scores=[1.0, 3.0],
        raw_alpha=1.0 / 3.0,
        feedback_count=1,
    )
    states = {label: state}

    issued = runtime.apply(_frame(), states)
    facts = next(iter(issued.issuances.values()))
    observed = runtime.observe(
        Delivery(
            label,
            _observations(
                issued,
                (100.0,),
                (CensoringAssertion.UNCENSORED,),
            ),
        ),
        states,
    )
    payload = _payload(observed.state_updates[label], label=label)
    later = runtime.apply(
        _frame(),
        {**states, **issued.state_updates, **observed.state_updates},
    )

    assert facts.calibration_ready
    assert facts.bounds_null_reason == "unresolvable-working-level"
    assert math.isnan(facts.lower_bound)
    assert math.isnan(facts.upper_bound)
    assert observed.annotations[0].score == 96.0
    assert payload["feedback_count"] == 2
    assert payload["raw_alpha"] == pytest.approx(1.0 / 3.0 + 0.1)
    assert next(iter(later.issuances.values())).bounds_null_reason is None


def test_declared_censoring_is_excluded_and_recorded_sales_label_is_sticky() -> None:
    runtime, label, states = _states(
        [1.0, 3.0],
        configuration={"coverage": 0.5, "learning_rate": 0.25},
    )
    issued = runtime.apply(_frame((4.0, 4.0)), states)
    effect = runtime.observe(
        Delivery(
            label,
            _observations(
                issued,
                (100.0, 7.0),
                (CensoringAssertion.CENSORED, None),
            ),
        ),
        states,
    )
    payload = _payload(effect.state_updates[label], label=label)
    later = runtime.apply(_frame(), {**states, **effect.state_updates})
    later_facts = next(iter(later.issuances.values()))

    assert effect.annotations[0].exclusion_cause == "declared-censored"
    assert not effect.annotations[0].advanced_delivered_score
    assert effect.annotations[1].score == 3.0
    assert payload["scores"] == [1.0, 3.0, 3.0]
    assert payload["delivered_score_count"] == 3
    assert payload["feedback_count"] == 1
    assert payload["raw_alpha"] == 0.625
    assert later_facts.effective_descriptor.scored_series is ScoredSeries.RECORDED_SALES


def test_issue_counter_reference_progresses_without_mutating_partition_state() -> None:
    runtime, label, states = _states(
        [1.0, 3.0],
        configuration={"coverage": 0.5},
    )

    first = runtime.apply(_frame(), states)
    second = runtime.apply(_frame(), {**states, **first.state_updates})
    first_reference = next(iter(first.issuances.values())).state_reference
    second_reference = next(iter(second.issuances.values())).state_reference

    assert first_reference.startswith(f"{_METHOD}:0:sha256:")
    assert second_reference.startswith(f"{_METHOD}:1:sha256:")
    assert first_reference != second_reference
    assert label not in first.state_updates
    assert label not in second.state_updates


def test_observe_rejects_tampered_identity_before_state_advancement() -> None:
    runtime, label, states = _states(
        [1.0, 3.0],
        configuration={"coverage": 0.5},
    )
    result = runtime.apply(_frame(), states)
    observation = _observations(
        result,
        (7.0,),
        (CensoringAssertion.UNCENSORED,),
    )[0]
    wrong_method = replace(observation.issued, method_name="stale-method")
    wrong_descriptor = replace(
        observation.issued,
        effective_descriptor=replace(observation.issued.effective_descriptor, level=0.6),
    )

    for tampered, message in (
        (wrong_method, "wrong sequential-adaptive method"),
        (wrong_descriptor, "wrong sequential-adaptive descriptor"),
    ):
        before = dict(states)
        with pytest.raises(RuntimeContractError, match=message):
            runtime.observe(
                Delivery(label, (replace(observation, issued=tampered),)),
                states,
            )
        assert states == before


def test_factory_restoration_replays_apply_and_observe_exactly() -> None:
    configuration = {
        "method": _METHOD,
        "coverage": 0.5,
        "calibration_window": 3,
        "learning_rate": 0.25,
    }
    original, label, states = _states(
        [9.0, 1.0, 5.0],
        configuration={key: value for key, value in configuration.items() if key != "method"},
    )
    restored = resolve_method(configuration, states=states)
    original_apply = original.apply(_frame(), states)
    restored_apply = restored.apply(_frame(), states)

    pd.testing.assert_frame_equal(
        original_apply.forecasts,
        restored_apply.forecasts,
        check_exact=True,
    )
    assert original_apply.issuances == restored_apply.issuances
    assert original_apply.state_updates == restored_apply.state_updates

    delivery = Delivery(
        label,
        _observations(
            original_apply,
            (7.0,),
            (CensoringAssertion.UNCENSORED,),
        ),
    )
    assert original.observe(delivery, states) == restored.observe(delivery, states)


def test_private_codec_rejects_wrong_scope_malformed_alpha_and_nonmonotone_counts() -> None:
    codec = JsonStateCodec(_METHOD, 1)
    wrong_scope = derive_partition_label(_MODEL, "global", EmissionScope.WINDOW_SUM)
    wrong_scope_state = _partition_state(
        wrong_scope,
        scores=[1.0],
        raw_alpha=0.1,
    )
    label = _partition()
    malformed = codec.encode(
        label,
        {
            "delivered_score_count": 1,
            "feedback_count": 0,
            "raw_alpha": "0.1",
            "scored_series": "demand-honest",
            "scores": [1.0],
        },
    )
    nonmonotone = _partition_state(
        label,
        scores=[1.0, 2.0],
        raw_alpha=0.1,
        delivered_score_count=1,
    )

    with pytest.raises(ConformalRegistryError, match="wrong method emission scope"):
        resolve_method({"method": _METHOD}, states={wrong_scope: wrong_scope_state})
    with pytest.raises(ConformalRegistryError, match="raw alpha"):
        resolve_method({"method": _METHOD}, states={label: malformed})
    with pytest.raises(ConformalRegistryError, match="no smaller than retained scores"):
        resolve_method({"method": _METHOD}, states={label: nonmonotone})
