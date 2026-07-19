"""Exercise the points-only reconciliation registry and native strategies."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
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
    HierarchyIndex,
    HierarchyNodeKind,
    interval_columns,
    quantile_column,
)
from newcalibre.reconcile import (
    BOTTOM_UP,
    BOTTOM_UP_DECLARATION,
    NONE,
    NONE_DECLARATION,
    MatrixCapability,
    Reconciler,
    ReconciliationContext,
    ReconciliationError,
    ReconciliationInputFamily,
    ReconciliationRegistryError,
    available_strategies,
    coherence_tolerance,
    resolve_strategy,
    strategy_declaration,
)


def _hierarchy() -> HierarchyIndex:
    facts = pd.DataFrame.from_records(
        [
            {SERIES_KEY: "c", "department": "outer", "store": 2},
            {SERIES_KEY: "a", "department": "inner", "store": 1},
            {SERIES_KEY: "b", "department": "inner", "store": 2},
        ]
    )
    return HierarchyIndex.from_facts(facts, bottom_series=("c", "b", "a"))


def _frame(
    points: dict[str, float],
    *,
    model_name: str = "model-a",
    origin: str = "2026-01-05",
    horizon_step: int = 1,
    row_order: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    origin_value = pd.Timestamp(origin)
    rows = [
        {
            SERIES_KEY: series_key,
            TARGET_TIMESTAMP: origin_value + pd.Timedelta(days=horizon_step - 1),
            ACTUAL_VALUE: float(index),
            POINT_FORECAST: points[series_key],
            HORIZON_STEP: horizon_step,
            ORIGIN: origin_value,
            MODEL_NAME: model_name,
            "channel": "retail",
        }
        for index, series_key in enumerate(row_order or tuple(points), start=1)
    ]
    frame = pd.DataFrame.from_records(rows)
    frame[SERIES_KEY] = frame[SERIES_KEY].astype("string")
    frame[MODEL_NAME] = frame[MODEL_NAME].astype("string")
    frame["channel"] = frame["channel"].astype("string")
    frame[ACTUAL_VALUE] = frame[ACTUAL_VALUE].astype("float64")
    frame[POINT_FORECAST] = frame[POINT_FORECAST].astype("float64")
    frame[HORIZON_STEP] = frame[HORIZON_STEP].astype("int64")
    return frame


def _node_for_members(hierarchy: HierarchyIndex, members: tuple[str, ...]) -> str:
    return next(
        node.label
        for node in hierarchy.nodes
        if node.kind is not HierarchyNodeKind.BOTTOM and node.members == members
    )


def test_native_declarations_are_immutable_and_inspectable() -> None:
    assert NONE_DECLARATION.name == NONE
    assert BOTTOM_UP_DECLARATION.name == BOTTOM_UP
    assert NONE_DECLARATION.input_family is ReconciliationInputFamily.SYNTHESIS
    assert BOTTOM_UP_DECLARATION.input_family is ReconciliationInputFamily.SYNTHESIS
    assert not NONE_DECLARATION.requires_fitted_values
    assert not BOTTOM_UP_DECLARATION.requires_fitted_values
    assert NONE_DECLARATION.matrix_capability is MatrixCapability.SPARSE_CAPABLE
    assert BOTTOM_UP_DECLARATION.matrix_capability is MatrixCapability.SPARSE_CAPABLE
    with pytest.raises(FrozenInstanceError):
        cast(Any, BOTTOM_UP_DECLARATION).requires_fitted_values = True


def test_registry_lists_normalizes_and_builds_fresh_native_strategies() -> None:
    assert available_strategies() == (BOTTOM_UP, NONE)
    assert strategy_declaration(" Bottom_UP ") is BOTTOM_UP_DECLARATION
    first = resolve_strategy(" Bottom_UP ")
    second = resolve_strategy("bOtToM_uP")

    assert first is not second
    assert isinstance(first, Reconciler)
    assert isinstance(second, Reconciler)
    assert isinstance(resolve_strategy(" NONE "), Reconciler)


def test_registry_unknown_name_lists_available_strategies() -> None:
    with pytest.raises(
        ReconciliationRegistryError,
        match=r"unknown strategy 'projection'.*bottom_up, none",
    ):
        resolve_strategy("projection")


@pytest.mark.parametrize("strategy_name", available_strategies())
def test_no_hierarchy_and_empty_frame_are_exact_identities(strategy_name: str) -> None:
    strategy = resolve_strategy(strategy_name)
    frame = _frame({"a": 1.0})
    empty = frame.iloc[0:0]
    context = ReconciliationContext()

    assert strategy(frame, None, context) is frame
    assert strategy(empty, _hierarchy(), context) is empty


def test_none_is_a_strict_point_frame_identity() -> None:
    frame = _frame({"a": 1.0, "b": 2.0, "c": 3.0})

    result = resolve_strategy(NONE)(frame, _hierarchy(), ReconciliationContext())

    assert result is frame


@pytest.mark.parametrize("strategy_name", available_strategies())
@pytest.mark.parametrize("bound_kind", ["interval", "quantile"])
def test_every_native_strategy_rejects_distributional_columns(
    strategy_name: str,
    bound_kind: str,
) -> None:
    frame = _frame({"a": 1.0, "b": 2.0, "c": 3.0})
    if bound_kind == "interval":
        lower, upper = interval_columns(0.9)
        frame[lower] = frame[POINT_FORECAST] - 1.0
        frame[upper] = frame[POINT_FORECAST] + 1.0
    else:
        frame[quantile_column(0.5)] = frame[POINT_FORECAST]

    with pytest.raises(ReconciliationError, match="point forecasts only"):
        resolve_strategy(strategy_name)(frame, _hierarchy(), ReconciliationContext())


@pytest.mark.parametrize("strategy_name", available_strategies())
def test_native_strategies_reject_uncovered_duplicate_and_aggregate_rows(
    strategy_name: str,
) -> None:
    hierarchy = _hierarchy()
    strategy = resolve_strategy(strategy_name)
    uncovered = _frame({"foreign": 1.0})
    duplicate = pd.concat(
        [_frame({"a": 1.0}), _frame({"a": 2.0})],
        ignore_index=True,
    )
    aggregate = _frame({"a": 1.0})
    aggregate.loc[aggregate.index[0], SERIES_KEY] = hierarchy.node_labels[-1]

    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*not covered"):
        strategy(uncovered, hierarchy, ReconciliationContext())
    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*duplicate"):
        strategy(duplicate, hierarchy, ReconciliationContext())
    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*bottom-node rows"):
        strategy(aggregate, hierarchy, ReconciliationContext())


def test_bottom_up_suppresses_partial_aggregates_instead_of_summing_them() -> None:
    hierarchy = _hierarchy()
    frame = _frame({"b": 2.0, "a": 1.0}, row_order=("b", "a"))

    result = resolve_strategy(BOTTOM_UP)(frame, hierarchy, ReconciliationContext())

    emitted = dict(zip(result[SERIES_KEY], result[POINT_FORECAST], strict=True))
    inner = _node_for_members(hierarchy, ("a", "b"))
    a_only = _node_for_members(hierarchy, ("a",))
    b_and_c = _node_for_members(hierarchy, ("b", "c"))
    assert tuple(result[SERIES_KEY].iloc[: len(frame)]) == ("b", "a")
    assert emitted[inner] == 3.0
    assert emitted[a_only] == 1.0
    assert b_and_c not in emitted
    assert hierarchy.node_labels[-1] not in emitted


def test_bottom_up_present_nan_poisons_exactly_containing_aggregates() -> None:
    hierarchy = _hierarchy()
    frame = _frame({"a": np.nan, "b": 2.0, "c": 3.0})

    result = resolve_strategy(BOTTOM_UP)(frame, hierarchy, ReconciliationContext())
    emitted = result.set_index(SERIES_KEY)[POINT_FORECAST]

    poisoned = {
        node.label
        for node in hierarchy.nodes
        if node.kind is not HierarchyNodeKind.BOTTOM and "a" in node.members
    }
    finite = {
        node.label: sum({"b": 2.0, "c": 3.0}[member] for member in node.members)
        for node in hierarchy.nodes
        if node.kind is not HierarchyNodeKind.BOTTOM and "a" not in node.members
    }
    assert emitted.loc[list(poisoned)].isna().all()
    assert emitted.loc[list(finite)].to_dict() == finite


def test_bottom_up_isolated_cross_sections_and_appends_in_canonical_order() -> None:
    hierarchy = _hierarchy()
    sections = [
        _frame(
            {"a": 1.0, "b": 2.0, "c": 3.0},
            model_name="z-model",
            origin="2026-01-06",
            horizon_step=2,
            row_order=("c", "a", "b"),
        ),
        _frame(
            {"a": 10.0, "b": 20.0, "c": 30.0},
            model_name="a-model",
            origin="2026-01-05",
            horizon_step=1,
            row_order=("b", "c", "a"),
        ),
    ]
    frame = pd.concat(sections, ignore_index=True)
    before = frame.copy(deep=True)

    result = resolve_strategy(BOTTOM_UP)(frame, hierarchy, ReconciliationContext())

    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result.iloc[: len(frame)].reset_index(drop=True), before)
    aggregate = result.iloc[len(frame) :]
    expected_labels = tuple(
        node.label for node in hierarchy.nodes if node.kind is not HierarchyNodeKind.BOTTOM
    )
    section_keys = list(
        aggregate[[MODEL_NAME, ORIGIN, HORIZON_STEP]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert section_keys == [
        ("a-model", pd.Timestamp("2026-01-05"), 1),
        ("z-model", pd.Timestamp("2026-01-06"), 2),
    ]
    assert tuple(aggregate[SERIES_KEY]) == expected_labels * 2
    assert aggregate[ACTUAL_VALUE].isna().all()
    assert set(aggregate["channel"]) == {"retail"}
    totals = aggregate.loc[aggregate[SERIES_KEY] == hierarchy.node_labels[-1]]
    assert totals[POINT_FORECAST].tolist() == [60.0, 6.0]


@pytest.mark.parametrize("strategy_name", available_strategies())
def test_registered_native_strategy_is_a_deterministic_fixed_point(strategy_name: str) -> None:
    hierarchy = _hierarchy()
    frame = _frame({"a": 1.0, "b": 2.0, "c": 3.0})
    strategy = resolve_strategy(strategy_name)

    first = strategy(frame, hierarchy, ReconciliationContext())
    second = strategy(frame, hierarchy, ReconciliationContext())
    bound = coherence_tolerance(
        reduction_width=len(hierarchy.bottom_series),
        vector_magnitude=float(first[POINT_FORECAST].abs().max()),
    )

    pd.testing.assert_frame_equal(first, second, check_exact=False, atol=bound, rtol=0.0)
