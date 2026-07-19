"""Exercise weighted-conformal mathematics, state, censoring, and attribution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import pandas as pd
import pytest

from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    WEIGHTED_PER_STEP_MANIFEST,
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
_ORIGIN = pd.Timestamp("2026-01-05")
_MODEL = "weighted-fixture"


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


def _partition(*, series_key: str = "sku", partition_by: str = "global") -> str:
    value = "global" if partition_by == "global" else series_key
    return derive_partition_label(_MODEL, value, EmissionScope.PER_STEP)


def _states(
    scores: list[float],
    *,
    configuration: Mapping[str, object] | None = None,
    series_key: str = "sku",
) -> tuple[object, str, Mapping[str, bytes]]:
    payload = {
        "method": "weighted-per-step",
        **({} if configuration is None else configuration),
    }
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
    decoded = JsonStateCodec("weighted-per-step", 1).decode(state, expected_label=label)
    assert isinstance(decoded, dict)
    return decoded


def test_manifest_and_default_configuration_declare_weighted_contract() -> None:
    runtime = resolve_method({"method": "weighted-per-step"})

    assert runtime.config.model_dump() == {
        "coverage": 0.9,
        "calibration_window": 5000,
        "partition_by": "global",
        "weight_decay": 0.99,
    }
    assert WEIGHTED_PER_STEP_MANIFEST.assumption_class is AssumptionClass.WEIGHTED
    assert WEIGHTED_PER_STEP_MANIFEST.emission_form is EmissionForm.ONE_SIDED_UPPER
    assert WEIGHTED_PER_STEP_MANIFEST.emission_scope is EmissionScope.PER_STEP
    assert WEIGHTED_PER_STEP_MANIFEST.censoring_policy is CensoringPolicy.CONSUMES_CENSORING_FACTS
    assert WEIGHTED_PER_STEP_MANIFEST.state_bound == 5000
    assert WEIGHTED_PER_STEP_MANIFEST.order_sensitive
    assert WEIGHTED_PER_STEP_MANIFEST.clamps == ()
    assert (
        WEIGHTED_PER_STEP_MANIFEST.post_warmup_non_finite
        is PostWarmupNonFinite.ALLOWED_WITH_ATTRIBUTION
    )
    assert WEIGHTED_PER_STEP_MANIFEST.guarantees[0].claim is GuaranteeClaim.ONE_SIDED_COVERAGE
    assert (
        WEIGHTED_PER_STEP_MANIFEST.guarantees[0].currency
        is GuaranteeCurrency.FINITE_SAMPLE_MARGINAL
    )


def test_calibration_is_deterministic_chronological_bounded_and_strict() -> None:
    runtime, label, _states_value = _states(
        [],
        configuration={"coverage": 0.5, "calibration_window": 3},
    )
    first = runtime.calibrate({label: [8.0, 9.0, 1.0, 2.0]})
    second = runtime.calibrate({label: [8.0, 9.0, 1.0, 2.0]})

    assert first == second
    assert _payload(first[label], label=label) == {
        "delivered_score_count": 4,
        "scored_series": "demand-honest",
        "scores": [9.0, 1.0, 2.0],
    }
    assert _payload(first[METHOD_SCOPE_LABEL], label=METHOD_SCOPE_LABEL) == {"issue_counter": 0}
    for score in (math.nan, math.inf, -1.0):
        with pytest.raises(RuntimeContractError, match="scores"):
            runtime.calibrate({label: [score]})


def test_weighted_quantile_uses_chronology_then_score_order_with_heldout_mass() -> None:
    configuration = {
        "coverage": 0.5,
        "calibration_window": 3,
        "weight_decay": 0.5,
    }
    first_runtime, _first_label, first_states = _states(
        [9.0, 1.0, 5.0],
        configuration=configuration,
    )
    second_runtime, _second_label, second_states = _states(
        [5.0, 1.0, 9.0],
        configuration=configuration,
    )
    upper = interval_columns(0.5)[1]

    first = first_runtime.apply(_frame(), first_states)
    second = second_runtime.apply(_frame(), second_states)

    # Weights are (0.25, 0.5, 1.0), and the held-out unit mass makes the
    # threshold 0.5 * 2.75 = 1.375. Sorting by score selects 5 then 9.
    assert first.forecasts.loc[0, upper] == 9.0
    assert second.forecasts.loc[0, upper] == 13.0


def test_tied_weighted_scores_replay_deterministically() -> None:
    runtime, _label, states = _states(
        [2.0, 1.0, 2.0],
        configuration={
            "coverage": 0.5,
            "calibration_window": 3,
            "weight_decay": 0.5,
        },
    )

    first = runtime.apply(_frame(), states)
    second = runtime.apply(_frame(), states)

    assert first.forecasts.loc[0, interval_columns(0.5)[1]] == 6.0
    pd.testing.assert_frame_equal(first.forecasts, second.forecasts, check_exact=True)
    assert first.issuances == second.issuances
    assert first.state_updates == second.state_updates


def test_uniform_weight_boundary_matches_corrected_unweighted_rank() -> None:
    runtime, _label, states = _states(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        configuration={
            "coverage": 0.8,
            "calibration_window": 5,
            "weight_decay": 1.0,
        },
    )
    result = runtime.apply(_frame(), states)

    assert runtime.manifest.minimum_calibration_scores(runtime.config) == 5
    assert result.forecasts.loc[0, interval_columns(0.8)[1]] == 9.0


def test_default_weighting_is_warm_then_finite_at_exact_count_readiness() -> None:
    runtime, label, nine = _states(list(range(1, 10)))
    below = runtime.apply(_frame(), nine)
    ten = runtime.calibrate({label: list(range(1, 11))})
    ready = runtime.apply(_frame(), ten)
    lower, upper = interval_columns(0.9)
    below_facts = next(iter(below.issuances.values()))
    ready_facts = next(iter(ready.issuances.values()))

    assert runtime.manifest.minimum_calibration_scores(runtime.config) == 10
    assert math.isnan(below.forecasts.loc[0, lower])
    assert math.isnan(below.forecasts.loc[0, upper])
    assert not below_facts.calibration_ready
    assert below_facts.bounds_null_reason == "warm-up"
    assert ready.forecasts.loc[0, lower] == 0.0
    assert ready.forecasts.loc[0, upper] == 14.0
    assert ready_facts.calibration_ready
    assert ready_facts.bounds_null_reason is None
    assert ready_facts.working_level == pytest.approx(0.1)
    assert ready_facts.effective_descriptor.level == 0.9
    assert ready_facts.bindings == ()


def test_aggressive_decay_attributes_persistent_post_warmup_heldout_mass() -> None:
    configuration = {
        "coverage": 0.9,
        "calibration_window": 20,
        "weight_decay": 0.1,
    }
    runtime, label, states = _states(list(range(1, 11)), configuration=configuration)
    first = runtime.apply(_frame(), states)
    lower, upper = interval_columns(0.9)
    first_facts = next(iter(first.issuances.values()))

    assert first_facts.calibration_ready
    assert first_facts.bounds_null_reason == "held-out-weight-mass"
    assert math.isnan(first.forecasts.loc[0, lower])
    assert math.isnan(first.forecasts.loc[0, upper])
    assert first_facts.bindings == ()

    observed = runtime.observe(
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
    evolved = {**states, **first.state_updates, **observed.state_updates}
    second = runtime.apply(_frame(), evolved)
    second_facts = next(iter(second.issuances.values()))

    assert second_facts.calibration_ready
    assert second_facts.bounds_null_reason == "held-out-weight-mass"
    assert math.isnan(second.forecasts.loc[0, upper])


def test_series_partitioning_uses_independent_weighted_score_states() -> None:
    runtime = resolve_method(
        {
            "method": "weighted-per-step",
            "coverage": 0.5,
            "partition_by": "series",
            "weight_decay": 1.0,
        }
    )
    a_label = _partition(series_key="a", partition_by="series")
    b_label = _partition(series_key="b", partition_by="series")
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


def test_apply_ignores_poisoned_actuals_and_advances_only_method_issue_state() -> None:
    class Poison:
        def __float__(self) -> float:
            raise AssertionError("apply read actual_value")

    runtime, label, states = _states(list(range(1, 11)))
    result = runtime.apply(_frame(actuals=(Poison(),)), states)

    assert result.forecasts.loc[0, interval_columns(0.9)[1]] == 14.0
    assert set(result.state_updates) == {METHOD_SCOPE_LABEL}
    assert label not in result.state_updates
    assert _payload(
        result.state_updates[METHOD_SCOPE_LABEL],
        label=METHOD_SCOPE_LABEL,
    ) == {"issue_counter": 1}


def test_repeated_apply_advances_issuance_reference_without_observation() -> None:
    runtime, _label, states = _states(list(range(1, 11)))

    first = runtime.apply(_frame(), states)
    second = runtime.apply(_frame(), {**states, **first.state_updates})
    first_reference = next(iter(first.issuances.values())).state_reference
    second_reference = next(iter(second.issuances.values())).state_reference

    assert first_reference != second_reference
    assert first_reference.startswith("weighted-per-step:0:sha256:")
    assert second_reference.startswith("weighted-per-step:1:sha256:")


def test_observe_preserves_canonical_append_order_censoring_and_sticky_series_label() -> None:
    runtime, label, states = _states(
        [8.0, 9.0],
        configuration={"coverage": 0.5, "calibration_window": 3},
    )
    issued = runtime.apply(_frame((4.0, 5.0, 6.0)), states)
    effect = runtime.observe(
        Delivery(
            label,
            _observations(
                issued,
                (7.0, 20.0, 8.0),
                (CensoringAssertion.UNCENSORED, CensoringAssertion.CENSORED, None),
            ),
        ),
        states,
    )
    payload = _payload(effect.state_updates[label], label=label)

    assert [annotation.score for annotation in effect.annotations] == [3.0, None, 2.0]
    assert effect.annotations[1].exclusion_cause == "declared-censored"
    assert [annotation.advanced_delivered_score for annotation in effect.annotations] == [
        True,
        False,
        True,
    ]
    assert payload == {
        "delivered_score_count": 4,
        "scored_series": "recorded-sales",
        "scores": [9.0, 3.0, 2.0],
    }

    later = runtime.apply(_frame(), {**states, **effect.state_updates})
    later_facts = next(iter(later.issuances.values()))
    issued_facts = next(iter(issued.issuances.values()))
    assert later_facts.effective_descriptor.scored_series is ScoredSeries.RECORDED_SALES
    assert later_facts.state_reference != issued_facts.state_reference


def test_observe_rejects_overflowed_finite_residual_before_state_encoding() -> None:
    runtime, label, states = _states(
        [1.0, 2.0],
        configuration={"coverage": 0.5},
    )
    issued = runtime.apply(_frame((1e308,)), states)
    delivery = Delivery(
        label,
        _observations(
            issued,
            (-1e308,),
            (CensoringAssertion.UNCENSORED,),
        ),
    )
    before = dict(states)

    with pytest.raises(RuntimeContractError, match="weighted residual scores must be finite"):
        runtime.observe(delivery, states)

    assert states == before


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("method", "wrong weighted method"),
        ("form", "wrong emission form"),
        ("scope", "wrong emission scope"),
        ("working-level", "wrong working alpha"),
    ],
)
def test_observe_rejects_tampered_issuance_before_state_advancement(
    mismatch: str,
    message: str,
) -> None:
    runtime, label, states = _states(
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


def test_factory_restoration_replays_apply_and_observe_exactly() -> None:
    configuration = {
        "method": "weighted-per-step",
        "coverage": 0.5,
        "calibration_window": 3,
        "weight_decay": 0.5,
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


@pytest.mark.parametrize(
    "configuration",
    [
        {"coverage": 0.0},
        {"coverage": 1.0},
        {"coverage": math.nan},
        {"calibration_window": 0},
        {"calibration_window": 5001},
        {"coverage": 0.9999, "calibration_window": 10},
        {"weight_decay": 0.0},
        {"weight_decay": -0.1},
        {"weight_decay": 1.1},
        {"weight_decay": math.nan},
        {"upper_cap": 100.0},
    ],
)
def test_weighted_config_rejects_invalid_rank_window_decay_and_clamps(
    configuration: dict[str, object],
) -> None:
    with pytest.raises(ConformalRegistryError, match="invalid configuration"):
        resolve_method({"method": "weighted-per-step", **configuration})


def test_private_codec_rejects_wrong_scope_and_malformed_restoration_payloads() -> None:
    wrong_scope = derive_partition_label(_MODEL, "global", EmissionScope.WINDOW_SUM)
    codec = JsonStateCodec("weighted-per-step", 1)
    valid_payload = {
        "delivered_score_count": 1,
        "scored_series": "demand-honest",
        "scores": [1.0],
    }
    wrong_scope_state = codec.encode(wrong_scope, valid_payload)
    label = _partition()
    malformed_state = codec.encode(label, {**valid_payload, "unknown": True})

    with pytest.raises(ConformalRegistryError, match="wrong method emission scope"):
        resolve_method(
            {"method": "weighted-per-step"},
            states={wrong_scope: wrong_scope_state},
        )
    with pytest.raises(ConformalRegistryError, match="exact fields"):
        resolve_method(
            {"method": "weighted-per-step"},
            states={label: malformed_state},
        )


def test_private_codec_rejects_nonmonotone_counts_and_invalid_method_state() -> None:
    codec = JsonStateCodec("weighted-per-step", 1)
    label = _partition()
    nonmonotone = codec.encode(
        label,
        {
            "delivered_score_count": 1,
            "scored_series": "demand-honest",
            "scores": [1.0, 2.0],
        },
    )
    invalid_method = codec.encode(METHOD_SCOPE_LABEL, {"issue_counter": -1})

    with pytest.raises(ConformalRegistryError, match="no smaller than retained scores"):
        resolve_method(
            {"method": "weighted-per-step"},
            states={label: nonmonotone},
        )
    with pytest.raises(ConformalRegistryError, match="nonnegative integer"):
        resolve_method(
            {"method": "weighted-per-step"},
            states={METHOD_SCOPE_LABEL: invalid_method},
        )
