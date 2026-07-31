"""Exercise the points-only reconciliation registry and native strategies."""

from __future__ import annotations

import gc
import weakref
from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from newcalibre.domain import (
    ACTUAL_VALUE,
    FITTED_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    TIMESTAMP,
    FittedValues,
    HierarchyIndex,
    HierarchyNodeKind,
    TargetSupport,
    interval_columns,
    quantile_column,
)
from newcalibre.reconcile import (
    BOTTOM_UP,
    BOTTOM_UP_DECLARATION,
    MINT_SHRINK,
    MINT_SHRINK_DECLARATION,
    NONE,
    NONE_DECLARATION,
    SPARSE_SOLVER_TOLERANCE,
    WLS_STRUCT,
    WLS_STRUCT_DECLARATION,
    WLS_VAR,
    WLS_VAR_DECLARATION,
    MatrixCapability,
    Reconciler,
    ReconcilerRegistry,
    ReconciliationContext,
    ReconciliationError,
    ReconciliationInputFamily,
    ReconciliationRegistryError,
    available_strategies,
    build_bottom_up,
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


def _fitted_context(hierarchy: HierarchyIndex) -> ReconciliationContext:
    timestamps = pd.date_range("2025-01-01", periods=8, freq="D")
    rows: list[dict[str, object]] = []
    for node_index, label in enumerate(hierarchy.node_labels, start=1):
        for period, timestamp in enumerate(timestamps, start=1):
            fitted = float(10 + node_index + period)
            rows.append(
                {
                    SERIES_KEY: label,
                    TIMESTAMP: timestamp,
                    ACTUAL_VALUE: fitted + node_index * period,
                    FITTED_VALUE: fitted,
                    MODEL_NAME: "model-a",
                }
            )
    frame = pd.DataFrame.from_records(rows)
    frame[SERIES_KEY] = frame[SERIES_KEY].astype("string")
    frame[MODEL_NAME] = frame[MODEL_NAME].astype("string")
    frame[ACTUAL_VALUE] = frame[ACTUAL_VALUE].astype("float64")
    frame[FITTED_VALUE] = frame[FITTED_VALUE].astype("float64")
    return ReconciliationContext(fitted_values=FittedValues.from_frame(frame))


def _context_for(strategy_name: str, hierarchy: HierarchyIndex) -> ReconciliationContext:
    return (
        _fitted_context(hierarchy)
        if strategy_declaration(strategy_name).requires_fitted_values
        else ReconciliationContext()
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


def test_registry_lists_normalizes_and_builds_fresh_strategies() -> None:
    expected = (BOTTOM_UP, MINT_SHRINK, NONE, WLS_STRUCT, WLS_VAR)
    assert available_strategies() == expected
    declarations = {
        BOTTOM_UP: BOTTOM_UP_DECLARATION,
        MINT_SHRINK: MINT_SHRINK_DECLARATION,
        NONE: NONE_DECLARATION,
        WLS_STRUCT: WLS_STRUCT_DECLARATION,
        WLS_VAR: WLS_VAR_DECLARATION,
    }
    for name in expected:
        assert strategy_declaration(f" {name.upper()} ") is declarations[name]
        first = resolve_strategy(f" {name.upper()} ")
        second = resolve_strategy(name.swapcase())
        assert first is not second
        assert isinstance(first, Reconciler)
        assert isinstance(second, Reconciler)

    assert WLS_STRUCT_DECLARATION.input_family is ReconciliationInputFamily.PROJECTION
    assert WLS_VAR_DECLARATION.input_family is ReconciliationInputFamily.PROJECTION
    assert MINT_SHRINK_DECLARATION.input_family is ReconciliationInputFamily.PROJECTION
    assert not WLS_STRUCT_DECLARATION.requires_fitted_values
    assert WLS_VAR_DECLARATION.requires_fitted_values
    assert MINT_SHRINK_DECLARATION.requires_fitted_values
    assert WLS_STRUCT_DECLARATION.matrix_capability is MatrixCapability.SPARSE_CAPABLE
    assert WLS_VAR_DECLARATION.matrix_capability is MatrixCapability.SPARSE_CAPABLE
    assert MINT_SHRINK_DECLARATION.matrix_capability is MatrixCapability.DENSE_ONLY


def test_registry_unknown_name_lists_available_strategies() -> None:
    with pytest.raises(
        ReconciliationRegistryError,
        match=(
            r"unknown strategy 'projection'.*bottom_up, mint_shrink, none, "
            r"wls_struct, wls_var"
        ),
    ):
        resolve_strategy("projection")


def test_registry_rejects_mint_cov_before_method_or_matrix_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newcalibre.reconcile.nixtla as nixtla

    def forbidden(*_args, **_kwargs):
        raise AssertionError("mint_cov resolution reached projection construction")

    monkeypatch.setattr(nixtla, "MinTrace", forbidden)
    monkeypatch.setattr(nixtla, "build_dense_summing_matrix", forbidden)
    monkeypatch.setattr(nixtla, "build_sparse_summing_matrix", forbidden)

    with pytest.raises(
        ReconciliationRegistryError,
        match=r"mint_cov.*wls_var.*wls_struct.*bottom_up, mint_shrink, none, wls_struct, wls_var",
    ):
        resolve_strategy("  MiNt_CoV ")


def test_registry_rejects_a_reused_live_reconciler_without_retaining_discarded_ones() -> None:
    registry = ReconcilerRegistry()
    singleton = build_bottom_up()
    registry.register(BOTTOM_UP_DECLARATION, lambda: singleton)

    assert registry.resolve(BOTTOM_UP) is singleton
    with pytest.raises(ReconciliationRegistryError, match="fresh reconciler"):
        registry.resolve(BOTTOM_UP)

    reconciler = resolve_strategy(BOTTOM_UP)
    discarded = weakref.ref(reconciler)
    del reconciler
    gc.collect()
    assert discarded() is None


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


def test_support_validator_canonicalizes_admissible_nonnegative_residue() -> None:
    frame = _frame({"a": -0.0, "b": 2.0, "c": 3.0})
    frame.loc[frame[SERIES_KEY] == "a", POINT_FORECAST] = -0.0
    context = ReconciliationContext(target_support=TargetSupport.NONNEGATIVE)

    result = resolve_strategy(NONE)(frame, _hierarchy(), context)

    assert result.loc[result[SERIES_KEY] == "a", POINT_FORECAST].iat[0] == 0.0


def test_support_validator_rejects_material_negative_with_identity() -> None:
    frame = _frame({"a": -5.1e-2, "b": 2.0, "c": 3.0})
    context = ReconciliationContext(target_support=TargetSupport.NONNEGATIVE)

    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*series='a'"):
        resolve_strategy(NONE)(frame, _hierarchy(), context)


def test_real_target_support_preserves_negative_points() -> None:
    frame = _frame({"a": -5.1e-2, "b": 2.0, "c": 3.0})

    result = resolve_strategy(NONE)(frame, _hierarchy(), ReconciliationContext())

    assert result is frame
    assert result.loc[result[SERIES_KEY] == "a", POINT_FORECAST].iat[0] == -5.1e-2


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

    hierarchy = _hierarchy()
    with pytest.raises(ReconciliationError, match="point forecasts only"):
        resolve_strategy(strategy_name)(frame, hierarchy, _context_for(strategy_name, hierarchy))


@pytest.mark.parametrize("strategy_name", available_strategies())
def test_native_strategies_reject_uncovered_and_duplicate_rows(strategy_name: str) -> None:
    hierarchy = _hierarchy()
    strategy = resolve_strategy(strategy_name)
    context = _context_for(strategy_name, hierarchy)
    uncovered = _frame({"foreign": 1.0})
    duplicate = pd.concat(
        [_frame({"a": 1.0}), _frame({"a": 2.0})],
        ignore_index=True,
    )

    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*not covered"):
        strategy(uncovered, hierarchy, context)
    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*duplicate"):
        strategy(duplicate, hierarchy, context)


def test_none_rejects_aggregate_rows() -> None:
    hierarchy = _hierarchy()
    aggregate = _frame({"a": 1.0})
    aggregate.loc[aggregate.index[0], SERIES_KEY] = hierarchy.node_labels[-1]

    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*bottom-node rows"):
        resolve_strategy(NONE)(aggregate, hierarchy, ReconciliationContext())


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
def test_registered_strategy_is_coherent_and_idempotent_on_applicable_input(
    strategy_name: str,
) -> None:
    hierarchy = _hierarchy()
    bottom = _frame({"a": 1.0, "b": 2.0, "c": 3.0})
    if strategy_declaration(strategy_name).input_family is ReconciliationInputFamily.PROJECTION:
        frame = resolve_strategy(BOTTOM_UP)(bottom, hierarchy, ReconciliationContext())
    else:
        frame = bottom
    strategy = resolve_strategy(strategy_name)
    context = _context_for(strategy_name, hierarchy)

    first = strategy(frame, hierarchy, context)
    second = strategy(first, hierarchy, context)
    magnitude = float(first[POINT_FORECAST].abs().max())
    bound = coherence_tolerance(
        reduction_width=len(hierarchy.bottom_series),
        vector_magnitude=magnitude,
        solver_tolerance=SPARSE_SOLVER_TOLERANCE,
        condition_number=10.0,
    )

    pd.testing.assert_frame_equal(first, second, check_exact=False, atol=bound, rtol=0.0)
