"""Prove split-conformal continuation is identical after factory restoration."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd
import pytest

from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    CalibrationResult,
    Delivery,
    ForecastKey,
    ResolvedObservation,
    derive_partition_label,
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
    ScoredSeries,
    interval_columns,
)

pytestmark = pytest.mark.tier2
_MODEL = "roundtrip-model"


def _frame(
    points: tuple[float, ...],
    *,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    steps = tuple(range(1, len(points) + 1))
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku"] * len(points), dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime(
                [origin + pd.Timedelta(days=step - 1) for step in steps]
            ),
            ACTUAL_VALUE: pd.Series([float("nan")] * len(points), dtype="float64"),
            POINT_FORECAST: pd.Series(points, dtype="float64"),
            HORIZON_STEP: pd.Series(steps, dtype="int64"),
            ORIGIN: pd.to_datetime([origin] * len(points)),
            MODEL_NAME: pd.Series([_MODEL] * len(points), dtype="string"),
        }
    )


def _delivery(
    result: CalibrationResult,
    actuals: tuple[float, ...],
    assertions: tuple[CensoringAssertion | None, ...],
) -> Delivery:
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
    return Delivery(observations[0].issued.partition_label, tuple(observations))


def _assert_apply_equal(left: CalibrationResult, right: CalibrationResult) -> None:
    pd.testing.assert_frame_equal(left.forecasts, right.forecasts, check_exact=True)
    assert left.issuances == right.issuances
    assert left.state_updates == right.state_updates


def _combined_states(
    seed: Mapping[str, bytes],
    issued: CalibrationResult,
    partition_update: Mapping[str, bytes],
) -> dict[str, bytes]:
    return {**seed, **issued.state_updates, **partition_update}


def test_per_step_restart_preserves_censoring_label_clamp_and_later_replay() -> None:
    configuration = {
        "method": "split-per-step",
        "coverage": 0.5,
        "calibration_window": 4,
        "upper_floor": 10.0,
    }
    runtime = resolve_method(configuration)
    label = derive_partition_label(_MODEL, "global", runtime.manifest.emission_scope)
    seed = runtime.calibrate({label: [1.0, 2.0]})
    issued = runtime.apply(
        _frame((2.0, 3.0), origin=pd.Timestamp("2026-03-02")),
        seed,
    )
    facts = tuple(issued.issuances.values())
    assert all(fact.bindings[0].bound for fact in facts)

    observed = runtime.observe(
        _delivery(
            issued,
            (99.0, 7.0),
            (CensoringAssertion.CENSORED, None),
        ),
        seed,
    )
    states = _combined_states(seed, issued, observed.state_updates)
    restored = resolve_method(configuration, states=states)

    later_frame = _frame((6.0,), origin=pd.Timestamp("2026-03-09"))
    continued_apply = runtime.apply(later_frame, states)
    restored_apply = restored.apply(later_frame, states)
    _assert_apply_equal(continued_apply, restored_apply)
    assert (
        next(iter(continued_apply.issuances.values())).effective_descriptor.scored_series.value
        == "recorded-sales"
    )

    later_delivery = _delivery(
        continued_apply,
        (9.0,),
        (CensoringAssertion.UNCENSORED,),
    )
    continued_observe = runtime.observe(later_delivery, states)
    restored_observe = restored.observe(later_delivery, states)
    assert continued_observe == restored_observe


@pytest.mark.parametrize(
    ("weight_decay", "expected_null_reason"),
    [(0.99, None), (0.1, "held-out-weight-mass")],
)
def test_weighted_restart_matches_finite_and_heldout_mass_continuations(
    weight_decay: float,
    expected_null_reason: str | None,
) -> None:
    configuration = {
        "method": "weighted-per-step",
        "coverage": 0.9,
        "calibration_window": 12,
        "weight_decay": weight_decay,
    }
    runtime = resolve_method(configuration)
    label = derive_partition_label(_MODEL, "global", runtime.manifest.emission_scope)
    seed = runtime.calibrate({label: list(range(1, 11))})
    issued = runtime.apply(
        _frame((2.0, 3.0), origin=pd.Timestamp("2026-04-06")),
        seed,
    )
    observed = runtime.observe(
        _delivery(
            issued,
            (99.0, 7.0),
            (CensoringAssertion.CENSORED, None),
        ),
        seed,
    )
    states = _combined_states(seed, issued, observed.state_updates)
    restored = resolve_method(configuration, states=states)

    assert all(isinstance(state, bytes) for state in states.values())
    later_frame = _frame((6.0,), origin=pd.Timestamp("2026-04-13"))
    continued_apply = runtime.apply(later_frame, states)
    restored_apply = restored.apply(later_frame, states)
    _assert_apply_equal(continued_apply, restored_apply)
    facts = next(iter(continued_apply.issuances.values()))
    upper = interval_columns(0.9)[1]
    assert facts.effective_descriptor.scored_series is ScoredSeries.RECORDED_SALES
    assert facts.calibration_ready
    assert facts.bounds_null_reason == expected_null_reason
    assert math.isnan(continued_apply.forecasts.loc[0, upper]) == (expected_null_reason is not None)

    later_delivery = _delivery(
        continued_apply,
        (9.0,),
        (CensoringAssertion.UNCENSORED,),
    )
    continued_observe = runtime.observe(later_delivery, states)
    restored_observe = restored.observe(later_delivery, states)
    assert continued_observe == restored_observe


def test_sequential_restart_matches_finite_unresolvable_and_trivial_cover_continuation() -> None:
    configuration = {
        "method": "sequential-adaptive-per-step",
        "coverage": 0.5,
        "calibration_window": 6,
        "learning_rate": 2.0,
    }
    runtime = resolve_method(configuration)
    label = derive_partition_label(_MODEL, "global", runtime.manifest.emission_scope)
    seed = runtime.calibrate({label: [1.0, 3.0]})
    finite = runtime.apply(
        _frame((4.0,), origin=pd.Timestamp("2026-05-04")),
        seed,
    )
    missed = runtime.observe(
        _delivery(
            finite,
            (8.0,),
            (CensoringAssertion.UNCENSORED,),
        ),
        seed,
    )
    after_finite = _combined_states(seed, finite, missed.state_updates)
    restored = resolve_method(configuration, states=after_finite)

    next_frame = _frame((4.0,), origin=pd.Timestamp("2026-05-11"))
    continued_unresolvable = runtime.apply(next_frame, after_finite)
    restored_unresolvable = restored.apply(next_frame, after_finite)
    _assert_apply_equal(continued_unresolvable, restored_unresolvable)
    unresolvable_facts = next(iter(continued_unresolvable.issuances.values()))
    assert unresolvable_facts.bounds_null_reason == "unresolvable-working-level"
    assert math.isnan(unresolvable_facts.upper_bound)

    trivial_delivery = _delivery(
        continued_unresolvable,
        (100.0,),
        (CensoringAssertion.UNCENSORED,),
    )
    continued_trivial = runtime.observe(trivial_delivery, after_finite)
    restored_trivial = restored.observe(trivial_delivery, after_finite)
    assert continued_trivial == restored_trivial
    after_trivial = _combined_states(
        after_finite,
        continued_unresolvable,
        continued_trivial.state_updates,
    )
    freshly_restored = resolve_method(configuration, states=after_trivial)

    later_frame = _frame((4.0,), origin=pd.Timestamp("2026-05-18"))
    continued_finite = runtime.apply(later_frame, after_trivial)
    restored_finite = freshly_restored.apply(later_frame, after_trivial)
    _assert_apply_equal(continued_finite, restored_finite)
    assert next(iter(continued_finite.issuances.values())).bounds_null_reason is None

    finite_delivery = _delivery(
        continued_finite,
        (5.0,),
        (CensoringAssertion.UNCENSORED,),
    )
    assert runtime.observe(finite_delivery, after_trivial) == freshly_restored.observe(
        finite_delivery,
        after_trivial,
    )


def test_window_restart_matches_after_incomplete_complete_censored_and_undeclared_script() -> None:
    configuration = {
        "method": "split-window-sum",
        "coverage": 0.5,
        "calibration_window": 4,
        "protection_period": 3,
        "upper_cap": 20.0,
    }
    runtime = resolve_method(configuration)
    label = derive_partition_label(_MODEL, "global", runtime.manifest.emission_scope)
    seed = runtime.calibrate({label: [1.0, 2.0]})

    incomplete = runtime.apply(
        _frame((2.0, 3.0), origin=pd.Timestamp("2026-04-06")),
        seed,
    )
    assert all(
        facts.bounds_null_reason == "emission-scope" for facts in incomplete.issuances.values()
    )

    complete = runtime.apply(
        _frame((8.0, 8.0, 8.0), origin=pd.Timestamp("2026-04-06")),
        {**seed, **incomplete.state_updates},
    )
    terminal = tuple(complete.issuances.values())[-1]
    assert terminal.upper_bound == 20.0
    assert terminal.bindings[0].bound

    censored = runtime.observe(
        _delivery(
            complete,
            (9.0, 9.0, 9.0),
            (
                CensoringAssertion.UNCENSORED,
                CensoringAssertion.CENSORED,
                CensoringAssertion.UNCENSORED,
            ),
        ),
        seed,
    )
    assert censored.state_updates[label] == seed[label]

    undeclared = runtime.observe(
        _delivery(
            complete,
            (9.0, 9.0, 9.0),
            (CensoringAssertion.UNCENSORED, None, CensoringAssertion.UNCENSORED),
        ),
        censored.state_updates,
    )
    states = _combined_states(seed, complete, undeclared.state_updates)
    assert METHOD_SCOPE_LABEL in states
    restored = resolve_method(configuration, states=states)

    later_frame = _frame((4.0, 5.0, 6.0), origin=pd.Timestamp("2026-04-13"))
    continued_apply = runtime.apply(later_frame, states)
    restored_apply = restored.apply(later_frame, states)
    _assert_apply_equal(continued_apply, restored_apply)

    later_delivery = _delivery(
        continued_apply,
        (5.0, 7.0, 9.0),
        (CensoringAssertion.UNCENSORED,) * 3,
    )
    continued_observe = runtime.observe(later_delivery, states)
    restored_observe = restored.observe(later_delivery, states)
    assert continued_observe == restored_observe
