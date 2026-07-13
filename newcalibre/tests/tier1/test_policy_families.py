"""Exercise pure policy families, provenance, modifiers, and refusals."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, cast

import pandas as pd
import pytest

from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    CostStructure,
    DecisionScope,
    DecisionScopeKind,
    DecisionTiming,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    GuaranteeType,
    InventoryPosition,
    ScoredSeries,
    interval_columns,
    quantile_column,
)
from newcalibre.ledger import BoundKey, ForecastIssuance, ForecastKey, GuaranteedSide
from newcalibre.ordering import (
    OrderingConfigError,
    OrderingInputError,
    OrderingSetup,
    PolicyRequest,
    compile_ordering,
    dispatch_policy,
)

pytestmark = pytest.mark.tier1

ORIGIN_VALUE = pd.Timestamp("2026-01-05")
TIMING = DecisionTiming(lead_time=1, review_period=1)
COST = CostStructure(underage=3.0, overage=1.0, holding=0.5, shortage=2.0)


def _configuration(policy: str, **changes: object):
    values: dict[str, object] = {
        "policy": policy,
        "series_keys": ("sku-a",),
        "cost_structure": COST,
        "decision_timing": TIMING,
        "task_horizon": 3,
        "calibration_coverage": 0.8,
    }
    if policy == "rss":
        values["reorder_point"] = 9.0
    values.update(changes)
    return compile_ordering(OrderingSetup(**values))  # type: ignore[arg-type]


def _descriptor(
    *,
    level: float = 0.8,
    window: EmissionScope = EmissionScope.PER_STEP,
    claim: GuaranteeClaim = GuaranteeClaim.ONE_SIDED_COVERAGE,
) -> GuaranteeDescriptor:
    currency = None if claim is GuaranteeClaim.NONE else GuaranteeCurrency.FINITE_SAMPLE_MARGINAL
    return GuaranteeDescriptor(
        type=GuaranteeType(
            claim=claim,
            currency=currency,
            declared_slack=None,
        ),
        level=level,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=window,
        scope=DecisionScope(
            kind=DecisionScopeKind.PER_DECISION_NODE,
            class_system_name=None,
        ),
    )


def _issuance(
    descriptor: GuaranteeDescriptor,
    *,
    finite: bool = True,
    side: GuaranteedSide | None = GuaranteedSide.UPPER,
) -> ForecastIssuance:
    if descriptor.type.claim is not GuaranteeClaim.ONE_SIDED_COVERAGE:
        side = None
    return ForecastIssuance(
        descriptor=descriptor,
        guaranteed_side=side,
        calibration_ready=finite,
        bounds_finite=finite,
        bounds_null_reason=None if finite else "emission-scope",
    )


def _frame(
    *,
    series: tuple[str, ...] = ("sku-a",),
    models: tuple[str, ...] = ("model-a",),
    horizon: int = 3,
    columns: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_key in series:
        for model_name in models:
            for step in range(1, horizon + 1):
                row_index = len(rows)
                row: dict[str, object] = {
                    SERIES_KEY: series_key,
                    TARGET_TIMESTAMP: ORIGIN_VALUE + pd.Timedelta(days=step - 1),
                    ACTUAL_VALUE: math.nan,
                    POINT_FORECAST: 1.0,
                    HORIZON_STEP: step,
                    ORIGIN: ORIGIN_VALUE,
                    MODEL_NAME: model_name,
                }
                for name, values in (columns or {}).items():
                    row[name] = values[row_index]
                rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    frame[SERIES_KEY] = frame[SERIES_KEY].astype("string")
    frame[MODEL_NAME] = frame[MODEL_NAME].astype("string")
    frame[ACTUAL_VALUE] = frame[ACTUAL_VALUE].astype("float64")
    frame[POINT_FORECAST] = frame[POINT_FORECAST].astype("float64")
    frame[HORIZON_STEP] = frame[HORIZON_STEP].astype("int64")
    for name in columns or {}:
        frame[name] = frame[name].astype("float64")
    return frame


def _key(series: str, step: int, model: str = "model-a") -> ForecastKey:
    return (series, ORIGIN_VALUE, step, model)


def _upper_issuances(
    descriptor: GuaranteeDescriptor,
    *,
    horizon: int = 3,
    finite: tuple[bool, ...] | None = None,
    series: tuple[str, ...] = ("sku-a",),
    models: tuple[str, ...] = ("model-a",),
) -> dict[ForecastKey, dict[BoundKey, ForecastIssuance]]:
    upper = interval_columns(descriptor.level)[1]
    flags = finite or (True,) * horizon
    return {
        _key(series_key, step, model_name): {
            (upper,): _issuance(descriptor, finite=flags[step - 1])
        }
        for series_key in series
        for model_name in models
        for step in range(1, horizon + 1)
    }


def _request(
    configuration,
    frame: pd.DataFrame,
    issuances: dict[ForecastKey, dict[BoundKey, ForecastIssuance]],
    *,
    positions: dict[str, InventoryPosition] | None = None,
) -> PolicyRequest:
    return PolicyRequest(
        frame=frame,
        issuances=issuances,
        inventory_positions=positions or {"sku-a": InventoryPosition(0.0, 0.0, 0.0)},
        configuration=configuration,
    )


def test_newsvendor_interpolates_h1_and_ignores_later_horizons() -> None:
    lower, upper = interval_columns(0.8)
    frame = _frame(
        columns={
            lower: [10.0, math.nan, math.inf],
            upper: [18.0, math.nan, -math.inf],
        }
    )
    descriptor = _descriptor()
    issuances = {
        _key("sku-a", step): {
            (lower, upper): _issuance(
                replace(
                    descriptor,
                    type=GuaranteeType(
                        claim=GuaranteeClaim.TWO_SIDED_COVERAGE,
                        currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                        declared_slack=None,
                    ),
                ),
                finite=step == 1,
                side=None,
            )
        }
        for step in range(1, 4)
    }

    decision = dispatch_policy(
        _request(
            _configuration("newsvendor"),
            frame,
            issuances,
            positions={"sku-a": InventoryPosition(1.0, 0.0, 0.0)},
        )
    )[0]

    assert decision.evidence.raw_target == 16.0
    assert decision.evidence.target == 16.0
    assert decision.quantity == 15.0
    assert decision.evidence.source_columns == (lower, upper)
    assert decision.evidence.source_descriptor.window is EmissionScope.PER_STEP


def test_newsvendor_prefers_the_exact_issued_critical_ratio_quantile() -> None:
    lower, upper = interval_columns(0.8)
    quantile = quantile_column(0.75)
    frame = _frame(
        columns={
            lower: [10.0, 10.0, 10.0],
            upper: [18.0, 18.0, 18.0],
            quantile: [20.0, math.inf, math.inf],
        }
    )
    interval_descriptor = replace(
        _descriptor(),
        type=GuaranteeType(
            claim=GuaranteeClaim.TWO_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
    )
    quantile_descriptor = _descriptor(level=0.75, claim=GuaranteeClaim.RISK_CONTROL)
    issuances = {
        _key("sku-a", step): {
            (lower, upper): _issuance(interval_descriptor, side=None),
            (quantile,): _issuance(quantile_descriptor, finite=step == 1),
        }
        for step in range(1, 4)
    }

    decision = dispatch_policy(_request(_configuration("newsvendor"), frame, issuances))[0]

    assert decision.evidence.raw_target == 20.0
    assert decision.evidence.source_columns == (quantile,)
    assert decision.evidence.source_descriptor == quantile_descriptor


def test_newsvendor_does_not_treat_an_unissued_dense_column_as_calibrated() -> None:
    lower, upper = interval_columns(0.8)
    quantile = quantile_column(0.75)
    frame = _frame(
        columns={
            lower: [10.0] * 3,
            upper: [18.0] * 3,
            quantile: [99.0] * 3,
        }
    )
    descriptor = replace(
        _descriptor(),
        type=GuaranteeType(
            claim=GuaranteeClaim.TWO_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
    )
    issuances = {
        _key("sku-a", step): {(lower, upper): _issuance(descriptor, side=None)}
        for step in range(1, 4)
    }

    decision = dispatch_policy(_request(_configuration("newsvendor"), frame, issuances))[0]

    assert decision.evidence.raw_target == 16.0
    assert decision.evidence.source_columns == (lower, upper)


def test_newsvendor_target_is_monotone_in_underage_cost() -> None:
    lower, upper = interval_columns(0.8)
    frame = _frame(columns={lower: [10.0] * 3, upper: [18.0] * 3})
    descriptor = replace(
        _descriptor(),
        type=GuaranteeType(
            claim=GuaranteeClaim.TWO_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
    )
    issuances = {
        _key("sku-a", step): {(lower, upper): _issuance(descriptor, side=None)}
        for step in range(1, 4)
    }
    low = _configuration(
        "newsvendor",
        cost_structure=CostStructure(1.0, 3.0, 0.0, 0.0),
    )
    high = _configuration(
        "newsvendor",
        cost_structure=CostStructure(3.0, 1.0, 0.0, 0.0),
    )

    assert dispatch_policy(_request(low, frame, issuances))[0].evidence.target == 12.0
    assert dispatch_policy(_request(high, frame, issuances))[0].evidence.target == 16.0


def test_newsvendor_refuses_reversed_interval_and_window_sum_scope() -> None:
    lower, upper = interval_columns(0.8)
    frame = _frame(columns={lower: [18.0] * 3, upper: [10.0] * 3})
    descriptor = replace(
        _descriptor(),
        type=GuaranteeType(
            claim=GuaranteeClaim.TWO_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
    )
    issuances = {
        _key("sku-a", step): {(lower, upper): _issuance(descriptor, side=None)}
        for step in range(1, 4)
    }
    with pytest.raises(OrderingInputError, match="lower bound <= upper bound"):
        dispatch_policy(_request(_configuration("newsvendor"), frame, issuances))

    cumulative = replace(descriptor, window=EmissionScope.WINDOW_SUM)
    cumulative_issuances = {
        _key("sku-a", step): {(lower, upper): _issuance(cumulative, side=None)}
        for step in range(1, 4)
    }
    with pytest.raises(OrderingInputError, match="wrong emission scope"):
        dispatch_policy(_request(_configuration("newsvendor"), frame, cumulative_issuances))


def test_rs_distinguishes_per_step_sum_from_terminal_window_bound() -> None:
    upper = interval_columns(0.8)[1]
    per_step = _descriptor(window=EmissionScope.PER_STEP)
    cumulative = _descriptor(window=EmissionScope.WINDOW_SUM)
    per_step_frame = _frame(columns={upper: [7.0, 11.0, math.inf]})
    cumulative_frame = _frame(columns={upper: [math.nan, 15.0, -math.inf]})

    summed = dispatch_policy(
        _request(
            _configuration("rs"),
            per_step_frame,
            _upper_issuances(per_step),
        )
    )[0]
    terminal = dispatch_policy(
        _request(
            _configuration("rs"),
            cumulative_frame,
            _upper_issuances(cumulative, finite=(False, True, False)),
        )
    )[0]

    assert summed.evidence.raw_target == 18.0
    assert terminal.evidence.raw_target == 15.0
    assert summed.evidence.source_descriptor.window is EmissionScope.PER_STEP
    assert terminal.evidence.source_descriptor.window is EmissionScope.WINDOW_SUM


def test_rs_explicit_quantile_always_sums_and_marks_nonengine_provenance() -> None:
    quantile = quantile_column(0.6)
    upper = interval_columns(0.8)[1]
    frame = _frame(
        columns={
            quantile: [4.5, 5.5, math.inf],
            upper: [math.nan, 99.0, math.inf],
        }
    )
    issuances = _upper_issuances(
        _descriptor(window=EmissionScope.WINDOW_SUM),
        finite=(False, True, False),
    )
    configuration = _configuration(
        "rs",
        calibration_coverage=None,
        explicit_quantile=0.6,
    )

    decision = dispatch_policy(_request(configuration, frame, issuances))[0]

    assert decision.evidence.raw_target == 10.0
    assert decision.evidence.source_columns == (quantile,)
    assert decision.evidence.source_descriptor.type.claim is GuaranteeClaim.NONE
    assert decision.evidence.source_descriptor.type.currency is None
    assert decision.evidence.source_descriptor.window is EmissionScope.PER_STEP
    assert decision.evidence.effective_descriptor == decision.evidence.source_descriptor


def test_rs_explicit_quantile_preserves_genuine_engine_issuance() -> None:
    quantile = quantile_column(0.6)
    frame = _frame(columns={quantile: [4.5, 5.5, math.inf]})
    descriptor = _descriptor(level=0.6, claim=GuaranteeClaim.RISK_CONTROL)
    issuance = _issuance(descriptor)
    issuances = {_key("sku-a", step): {(quantile,): issuance} for step in range(1, 4)}
    configuration = _configuration(
        "rs",
        calibration_coverage=None,
        explicit_quantile=0.6,
    )

    decision = dispatch_policy(_request(configuration, frame, issuances))[0]

    assert decision.evidence.raw_target == 10.0
    assert decision.evidence.source_descriptor == descriptor
    assert decision.evidence.effective_descriptor == descriptor
    assert decision.evidence.source_descriptor.type.claim is GuaranteeClaim.RISK_CONTROL


@pytest.mark.parametrize(
    ("position", "expected"),
    [(8.0, 10.0), (9.0, 0.0), (10.0, 0.0)],
)
def test_rss_absolute_gate_is_inclusive(position: float, expected: float) -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(columns={upper: [7.0, 11.0, math.inf]})
    configuration = _configuration("rss", reorder_point=9.0)
    decision = dispatch_policy(
        _request(
            configuration,
            frame,
            _upper_issuances(_descriptor()),
            positions={"sku-a": InventoryPosition(position, 0.0, 0.0)},
        )
    )[0]

    assert decision.evidence.target == 18.0
    assert decision.evidence.reorder_point == 9.0
    assert decision.quantity == expected


@pytest.mark.parametrize(
    ("position", "expected"),
    [(8.0, 10.0), (9.0, 0.0), (10.0, 0.0)],
)
def test_rss_target_scale_gate_is_inclusive_and_not_a_modifier(
    position: float,
    expected: float,
) -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(columns={upper: [7.0, 11.0, math.inf]})
    configuration = _configuration(
        "rss",
        reorder_point=None,
        reorder_point_scale=0.5,
    )
    decision = dispatch_policy(
        _request(
            configuration,
            frame,
            _upper_issuances(_descriptor()),
            positions={"sku-a": InventoryPosition(position, 0.0, 0.0)},
        )
    )[0]

    assert decision.evidence.reorder_point == 9.0
    assert decision.quantity == expected
    assert decision.evidence.bindings == ()
    assert decision.evidence.effective_descriptor == decision.evidence.source_descriptor


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("target_cap", 15.0, 15.0),
        ("target_floor", 20.0, 20.0),
        ("target_scale", 1.1, 19.8),
    ],
)
def test_binding_target_modifier_records_and_voids_claim(
    field: str,
    value: float,
    expected: float,
) -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(columns={upper: [7.0, 11.0, math.inf]})
    configuration = _configuration("rs", **{field: value})

    decision = dispatch_policy(_request(configuration, frame, _upper_issuances(_descriptor())))[0]

    assert decision.evidence.raw_target == 18.0
    assert decision.evidence.target == pytest.approx(expected)
    assert [
        (binding.name, binding.value, binding.bound) for binding in decision.evidence.bindings
    ] == [(field, value, True)]
    assert decision.evidence.effective_descriptor.type.claim is GuaranteeClaim.NONE
    assert decision.evidence.effective_descriptor.type.currency is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("target_cap", 20.0), ("target_floor", 10.0), ("target_scale", 1.0)],
)
def test_nonbinding_modifier_is_recorded_and_preserves_claim(field: str, value: float) -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(columns={upper: [7.0, 11.0, math.inf]})
    source = _descriptor()

    decision = dispatch_policy(
        _request(
            _configuration("rs", **{field: value}),
            frame,
            _upper_issuances(source),
        )
    )[0]

    assert decision.evidence.target == 18.0
    assert [
        (binding.name, binding.value, binding.bound) for binding in decision.evidence.bindings
    ] == [(field, value, False)]
    assert decision.evidence.effective_descriptor == source


def test_cfg6_is_separate_always_bound_and_always_voids_claim() -> None:
    lower, upper = interval_columns(0.8)
    frame = _frame(columns={lower: [10.0] * 3, upper: [18.0] * 3})
    source = replace(
        _descriptor(),
        type=GuaranteeType(
            claim=GuaranteeClaim.TWO_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
    )
    issuances = {
        _key("sku-a", step): {(lower, upper): _issuance(source, side=None)} for step in range(1, 4)
    }

    decision = dispatch_policy(
        _request(
            _configuration("newsvendor", explicit_decision_fractile=0.5),
            frame,
            issuances,
        )
    )[0]

    assert decision.evidence.target == 14.0
    assert [
        (binding.name, binding.value, binding.bound) for binding in decision.evidence.bindings
    ] == [("explicit_decision_fractile", 0.5, True)]
    assert decision.evidence.effective_descriptor.type.claim is GuaranteeClaim.NONE


def test_absent_modifiers_preserve_the_baseline_decision_exactly() -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(columns={upper: [7.0, 11.0, math.inf]})
    issuances = _upper_issuances(_descriptor())

    first = dispatch_policy(_request(_configuration("rs"), frame, issuances))
    second = dispatch_policy(
        _request(
            _configuration(
                "rs",
                target_cap=None,
                target_floor=None,
                target_scale=None,
            ),
            frame,
            issuances,
        )
    )

    assert first == second


def test_dispatch_is_pure_repeatable_and_keeps_series_and_models_independent() -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(
        series=("sku-a", "sku-b"),
        models=("model-a", "model-b"),
        columns={upper: [7.0, 11.0, math.inf] * 4},
    )
    original = frame.copy(deep=True)
    descriptor = _descriptor()
    issuances = _upper_issuances(
        descriptor,
        series=("sku-a", "sku-b"),
        models=("model-a", "model-b"),
    )
    configuration = _configuration(
        "rs",
        series_keys=("sku-b", "sku-a"),
        cost_structure={"sku-a": COST, "sku-b": COST},
    )
    request = _request(
        configuration,
        frame,
        issuances,
        positions={
            "sku-a": InventoryPosition(1.0, 0.0, 0.0),
            "sku-b": InventoryPosition(2.0, 0.0, 0.0),
        },
    )

    first = dispatch_policy(request)
    second = dispatch_policy(request)

    assert first == second
    assert [(value.series_key, value.model_name, value.quantity) for value in first] == [
        ("sku-a", "model-a", 17.0),
        ("sku-a", "model-b", 17.0),
        ("sku-b", "model-a", 16.0),
        ("sku-b", "model-b", 16.0),
    ]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("mutator", "pattern"),
    [
        (lambda frame, _issuances: frame.loc[frame[HORIZON_STEP] != 2], "missing"),
        (
            lambda frame, _issuances: pd.concat([frame, frame.iloc[[2]]], ignore_index=True),
            "duplicate",
        ),
        (
            lambda frame, _issuances: frame.assign(
                **{interval_columns(0.8)[1]: [7.0, math.nan, 1.0]}
            ),
            "finite",
        ),
    ],
)
def test_rs_refuses_missing_duplicate_and_nonfinite_consumed_horizons(
    mutator,
    pattern: str,
) -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(columns={upper: [7.0, 11.0, 1.0]})
    issuances = _upper_issuances(_descriptor())
    changed = mutator(frame, issuances)
    if pattern == "missing":
        issuances.pop(_key("sku-a", 2))

    with pytest.raises(OrderingInputError, match=pattern):
        dispatch_policy(_request(_configuration("rs"), changed, issuances))


def test_every_decision_group_must_cover_the_complete_task_horizon() -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(horizon=2, columns={upper: [7.0, 11.0]})
    issuances = _upper_issuances(_descriptor(), horizon=2)

    with pytest.raises(OrderingInputError, match="complete task horizon"):
        dispatch_policy(_request(_configuration("rs"), frame, issuances))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rs_refuses_every_nonfinite_value_it_would_consume(value: float) -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(columns={upper: [7.0, value, 1.0]})
    issuances = _upper_issuances(_descriptor())

    with pytest.raises(OrderingInputError, match="finite"):
        dispatch_policy(_request(_configuration("rs"), frame, issuances))


def test_cumulative_refuses_a_nonfinite_terminal_without_consuming_null_prefix() -> None:
    upper = interval_columns(0.8)[1]
    descriptor = _descriptor(window=EmissionScope.WINDOW_SUM)
    frame = _frame(columns={upper: [math.nan, math.nan, math.inf]})
    issuances = _upper_issuances(descriptor, finite=(False, False, False))

    with pytest.raises(OrderingInputError, match="terminal.*finite"):
        dispatch_policy(_request(_configuration("rs"), frame, issuances))


def test_refuses_mixed_scope_missing_descriptor_and_coverage_mismatch() -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(columns={upper: [7.0, 11.0, 1.0]})
    per_step = _descriptor()
    mixed = _upper_issuances(per_step)
    mixed[_key("sku-a", 2)] = {(upper,): _issuance(_descriptor(window=EmissionScope.WINDOW_SUM))}
    missing = _upper_issuances(per_step)
    missing[_key("sku-a", 2)] = {}
    mismatch_descriptor = _descriptor(level=0.7)
    mismatch = {
        _key("sku-a", step): {(upper,): _issuance(mismatch_descriptor)} for step in range(1, 4)
    }

    for issuances, pattern in (
        (mixed, "mixed"),
        (missing, "issuance"),
        (mismatch, "level"),
    ):
        with pytest.raises(OrderingInputError, match=pattern):
            dispatch_policy(_request(_configuration("rs"), frame, issuances))


def test_point_forecast_never_substitutes_for_a_missing_explicit_quantile() -> None:
    frame = _frame()
    issuances = {_key("sku-a", step): {} for step in range(1, 4)}
    configuration = _configuration(
        "rs",
        calibration_coverage=None,
        explicit_quantile=0.6,
    )

    with pytest.raises(OrderingInputError, match="quantile"):
        dispatch_policy(_request(configuration, frame, issuances))


def test_refuses_wrong_series_missing_inventory_and_multiple_origins() -> None:
    upper = interval_columns(0.8)[1]
    frame = _frame(columns={upper: [7.0, 11.0, 1.0]})
    issuances = _upper_issuances(_descriptor())

    foreign = frame.copy(deep=True)
    foreign[SERIES_KEY] = "aggregate:all"
    foreign_issuances = {
        ("aggregate:all", origin, step, model): value
        for (_series, origin, step, model), value in issuances.items()
    }
    with pytest.raises(OrderingInputError, match="decision series"):
        dispatch_policy(_request(_configuration("rs"), foreign, foreign_issuances))

    with pytest.raises(OrderingInputError, match="inventory"):
        dispatch_policy(
            _request(
                _configuration("rs"),
                frame,
                issuances,
                positions={"sku-a": cast(Any, None)},
            )
        )

    multiple = frame.copy(deep=True)
    multiple.loc[0, ORIGIN] = ORIGIN_VALUE - pd.Timedelta(days=1)
    with pytest.raises(OrderingInputError, match="one origin"):
        dispatch_policy(_request(_configuration("rs"), multiple, issuances))


def test_rss_gate_configuration_is_exactly_one_and_modifiers_are_unambiguous() -> None:
    with pytest.raises(OrderingConfigError, match="exactly one"):
        _configuration("rss", reorder_point=None, reorder_point_scale=None)
    with pytest.raises(OrderingConfigError, match="exactly one"):
        _configuration("rss", reorder_point=9.0, reorder_point_scale=0.5)
    with pytest.raises(OrderingConfigError, match="simultaneous"):
        _configuration("rs", target_cap=15.0, target_floor=20.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1"])
def test_modifier_values_must_be_finite_reals(value: object) -> None:
    with pytest.raises(OrderingConfigError, match="target_cap"):
        _configuration("rs", target_cap=cast(Any, value))
