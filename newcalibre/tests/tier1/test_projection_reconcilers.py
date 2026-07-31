"""Exercise Nixtla-backed projection reconciliation and its owned guards."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from reference.reconcile import (
    covariance_projection,
    diagonal_projection,
    shrink_covariance,
    structural_weights,
    variance_weights,
)

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
    TargetSupport,
    interval_columns,
)
from newcalibre.reconcile import (
    NixtlaLayout,
    ProjectionMetadata,
    ProjectionReconciler,
    ReconciliationContext,
    ReconciliationError,
    SparseSummingMatrix,
    build_dense_summing_matrix,
    build_mint_shrink,
    build_sparse_summing_matrix,
    build_wls_struct,
    build_wls_var,
    coherence_tolerance,
    covariance_estimator_tolerance,
    derive_variance_weights,
    preflight_projection,
)
from newcalibre.reconcile.apply import ReconciledValues
from newcalibre.reconcile.nixtla import SPARSE_SOLVER_TOLERANCE

_BASE_FORECAST = np.array(
    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 48.0, 75.0, 95.0, 55.0, 160.0, 200.0]
)
_P1 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
_P2 = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
_P3 = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
_RESIDUAL_COEFFICIENTS = np.array(
    [
        (2.0, 1.0, 0.0),
        (1.0, -2.0, 0.5),
        (-1.0, 1.0, -0.5),
        (0.0, 2.0, 1.0),
        (2.0, 0.0, -1.0),
        (1.0, 1.0, 1.0),
        (4.0, 2.0, 1.0),
        (3.0, -1.0, 2.0),
        (2.0, 3.0, -1.0),
        (6.0, 3.0, 1.5),
        (5.0, -2.0, -2.0),
        (8.0, 4.0, 2.0),
    ],
    dtype=np.float64,
)


def _hierarchy() -> HierarchyIndex:
    facts = pd.DataFrame.from_records(
        [
            {SERIES_KEY: "s1", "channel": "a", "region": "east"},
            {SERIES_KEY: "s2", "channel": "b", "region": "east"},
            {SERIES_KEY: "s3", "channel": "c", "region": "east"},
            {SERIES_KEY: "s4", "channel": "a", "region": "west"},
            {SERIES_KEY: "s5", "channel": "b", "region": "west"},
            {SERIES_KEY: "s6", "channel": "c", "region": "west"},
        ]
    )
    return HierarchyIndex.from_facts(facts, bottom_series=tuple(reversed(range_ids())))


def range_ids() -> tuple[str, ...]:
    return tuple(f"s{index}" for index in range(1, 7))


def _base_by_label(hierarchy: HierarchyIndex) -> dict[str, float]:
    return dict(zip(hierarchy.node_labels, _BASE_FORECAST, strict=True))


def _frame(
    hierarchy: HierarchyIndex,
    *,
    labels: tuple[str, ...] | None = None,
    row_order: tuple[str, ...] | None = None,
    model_name: str = "model-a",
    origin: str = "2026-01-05",
    horizon_step: int = 1,
) -> pd.DataFrame:
    selected = labels or hierarchy.node_labels
    order = row_order or selected
    points = _base_by_label(hierarchy)
    origin_value = pd.Timestamp(origin)
    rows = [
        {
            SERIES_KEY: label,
            TARGET_TIMESTAMP: origin_value + pd.Timedelta(days=horizon_step - 1),
            ACTUAL_VALUE: float(index),
            POINT_FORECAST: points[label],
            HORIZON_STEP: horizon_step,
            ORIGIN: origin_value,
            MODEL_NAME: model_name,
            "channel_name": "retail",
        }
        for index, label in enumerate(order, start=1)
    ]
    frame = pd.DataFrame.from_records(rows)
    frame[SERIES_KEY] = frame[SERIES_KEY].astype("string")
    frame[MODEL_NAME] = frame[MODEL_NAME].astype("string")
    frame["channel_name"] = frame["channel_name"].astype("string")
    frame[ACTUAL_VALUE] = frame[ACTUAL_VALUE].astype("float64")
    frame[POINT_FORECAST] = frame[POINT_FORECAST].astype("float64")
    frame[HORIZON_STEP] = frame[HORIZON_STEP].astype("int64")
    return frame


def _residuals() -> np.ndarray:
    centered = _RESIDUAL_COEFFICIENTS @ np.vstack((_P1, _P2, _P3))
    means = np.arange(1, len(centered) + 1, dtype=np.float64)[:, None] / 10.0
    return centered + means


def _fitted_frame(
    hierarchy: HierarchyIndex,
    *,
    model_name: str = "model-a",
) -> pd.DataFrame:
    residuals = _residuals()
    timestamps = pd.date_range("2025-01-01", periods=residuals.shape[1], freq="D")
    rows: list[dict[str, object]] = []
    for node_index, label in enumerate(hierarchy.node_labels):
        for period, timestamp in enumerate(timestamps):
            fitted = 100.0 + node_index + period
            rows.append(
                {
                    SERIES_KEY: label,
                    TIMESTAMP: timestamp,
                    ACTUAL_VALUE: fitted + residuals[node_index, period],
                    FITTED_VALUE: fitted,
                    MODEL_NAME: model_name,
                }
            )
    result = pd.DataFrame.from_records(rows)
    result[SERIES_KEY] = result[SERIES_KEY].astype("string")
    result[MODEL_NAME] = result[MODEL_NAME].astype("string")
    result[ACTUAL_VALUE] = result[ACTUAL_VALUE].astype("float64")
    result[FITTED_VALUE] = result[FITTED_VALUE].astype("float64")
    return result


def _context(
    hierarchy: HierarchyIndex,
    *,
    frame: pd.DataFrame | None = None,
) -> ReconciliationContext:
    values = _fitted_frame(hierarchy) if frame is None else frame
    return ReconciliationContext(
        fitted_values=FittedValues.from_frame(values),
        target_support=TargetSupport.REAL,
    )


def _canonical_points(result: pd.DataFrame, hierarchy: HierarchyIndex) -> np.ndarray:
    return (
        result.set_index(SERIES_KEY)
        .loc[list(hierarchy.node_labels), POINT_FORECAST]
        .to_numpy(dtype=np.float64)
    )


def _reference_bound(
    expected: np.ndarray,
    *,
    condition_number: float,
    sparse: bool,
    covariance_estimator: bool = False,
) -> float:
    estimator = (
        covariance_estimator_tolerance(n_nodes=12, residual_periods=8)
        if covariance_estimator
        else 0.0
    )
    return coherence_tolerance(
        reduction_width=6,
        vector_magnitude=float(np.abs(expected).max()),
        solver_tolerance=SPARSE_SOLVER_TOLERANCE if sparse else 0.0,
        condition_number=condition_number,
        estimator_tolerance=estimator,
    )


def _assert_reference_agreement(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    condition_number: float,
    sparse: bool,
    covariance_estimator: bool = False,
) -> None:
    bound = _reference_bound(
        expected,
        condition_number=condition_number,
        sparse=sparse,
        covariance_estimator=covariance_estimator,
    )
    assert float(np.max(np.abs(actual - expected))) <= bound


def _applicable_labels(
    hierarchy: HierarchyIndex,
    bottom_labels: tuple[str, ...],
) -> tuple[str, ...]:
    present = set(bottom_labels)
    return tuple(
        node.label for node in hierarchy.nodes if any(member in present for member in node.members)
    )


def test_nixtla_layout_round_trips_vectors_and_both_matrix_representations() -> None:
    hierarchy = _hierarchy()
    dense = build_dense_summing_matrix(hierarchy)
    sparse = build_sparse_summing_matrix(hierarchy)
    layout = NixtlaLayout.from_matrix(dense)
    vector = np.arange(dense.n_nodes, dtype=np.float64)

    nixtla_vector = layout.to_nixtla_vector(vector)

    assert layout.to_project_vector(nixtla_vector).tolist() == vector.tolist()
    assert np.array_equal(layout.dense_matrix(dense)[-dense.n_bottom :], np.eye(6))
    sparse_nixtla = layout.sparse_matrix(sparse)
    assert sparse_nixtla.shape == dense.shape
    assert np.array_equal(sparse_nixtla.toarray(), layout.dense_matrix(dense))


def test_wls_struct_dense_and_sparse_paths_match_independent_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    frame = _frame(hierarchy, row_order=tuple(reversed(hierarchy.node_labels)))
    matrix = build_dense_summing_matrix(hierarchy).to_dense()
    reference = diagonal_projection(matrix, _BASE_FORECAST, structural_weights(matrix))

    dense_result = build_wls_struct()(
        frame,
        hierarchy,
        ReconciliationContext(target_support=TargetSupport.REAL),
    )

    import newcalibre.reconcile.nixtla as nixtla

    def forbidden(_hierarchy: HierarchyIndex):
        raise AssertionError("sparse-required projection invoked the dense producer")

    monkeypatch.setattr(nixtla, "build_dense_summing_matrix", forbidden)
    sparse_result = build_wls_struct(dense_workspace_ceiling_bytes=0)(
        frame,
        hierarchy,
        ReconciliationContext(target_support=TargetSupport.REAL),
    )

    _assert_reference_agreement(
        _canonical_points(dense_result, hierarchy),
        reference.reconciled,
        condition_number=reference.condition_number,
        sparse=False,
    )
    _assert_reference_agreement(
        _canonical_points(sparse_result, hierarchy),
        reference.reconciled,
        condition_number=reference.condition_number,
        sparse=True,
    )
    assert tuple(dense_result[SERIES_KEY]) == tuple(frame[SERIES_KEY])
    pd.testing.assert_frame_equal(
        dense_result.drop(columns=[POINT_FORECAST]),
        frame.drop(columns=[POINT_FORECAST]),
    )


def test_projection_support_validator_canonicalizes_dense_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    frame = _frame(hierarchy)
    residue = -6.661338147750939e-16
    matrix = build_dense_summing_matrix(hierarchy)

    def reconciled(
        self: ProjectionReconciler,
        *args: object,
        **kwargs: object,
    ) -> ReconciledValues:
        del self, args, kwargs
        bottom = _BASE_FORECAST[: matrix.n_bottom].copy()
        bottom[0] = residue
        values = matrix.matvec(bottom)
        return ReconciledValues(values, abs(residue))

    monkeypatch.setattr(ProjectionReconciler, "_reconcile_section", reconciled)

    result = build_wls_struct()(
        frame,
        hierarchy,
        ReconciliationContext(target_support=TargetSupport.NONNEGATIVE),
    )

    points = _canonical_points(result, hierarchy)
    expected_bottom = _BASE_FORECAST[: matrix.n_bottom].copy()
    expected_bottom[0] = 0.0

    assert np.array_equal(points, matrix.matvec(expected_bottom))
    assert points[0] == 0.0
    assert not np.signbit(points[0])


def test_sparse_projection_support_uses_shared_matrix_solver_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    matrix = build_sparse_summing_matrix(hierarchy)
    layout = NixtlaLayout.from_matrix(matrix)
    bottom = np.ones(matrix.n_bottom, dtype=np.float64)
    bound = coherence_tolerance(
        reduction_width=matrix.reduction_width,
        vector_magnitude=float(matrix.n_bottom),
        solver_tolerance=SPARSE_SOLVER_TOLERANCE,
    )
    bottom[0] = -(bound / 2.0)
    reconciled = matrix.matvec(bottom)
    nixtla_values = layout.to_nixtla_vector(reconciled)

    def sparse_result(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        del args, kwargs
        return {"mean": nixtla_values[:, None]}

    monkeypatch.setattr(
        "newcalibre.reconcile.nixtla._CheckedMinTraceSparse.fit_predict",
        sparse_result,
    )

    result = build_wls_struct(dense_workspace_ceiling_bytes=0)(
        _frame(hierarchy),
        hierarchy,
        ReconciliationContext(target_support=TargetSupport.NONNEGATIVE),
    )
    expected_bottom = bottom.copy()
    expected_bottom[0] = 0.0

    assert np.array_equal(_canonical_points(result, hierarchy), matrix.matvec(expected_bottom))


def test_sparse_projection_support_rejects_material_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    matrix = build_sparse_summing_matrix(hierarchy)
    layout = NixtlaLayout.from_matrix(matrix)
    bottom = np.ones(matrix.n_bottom, dtype=np.float64)
    bottom[0] = -0.1
    nixtla_values = layout.to_nixtla_vector(matrix.matvec(bottom))

    def sparse_result(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        del args, kwargs
        return {"mean": nixtla_values[:, None]}

    monkeypatch.setattr(
        "newcalibre.reconcile.nixtla._CheckedMinTraceSparse.fit_predict",
        sparse_result,
    )

    with pytest.raises(ReconciliationError, match=r"model-a.*series='s1'"):
        build_wls_struct(dense_workspace_ceiling_bytes=0)(
            _frame(hierarchy),
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.NONNEGATIVE),
        )


def test_projection_support_validator_rejects_material_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    frame = _frame(hierarchy)
    matrix = build_dense_summing_matrix(hierarchy)

    def reconciled(
        self: ProjectionReconciler,
        *args: object,
        **kwargs: object,
    ) -> ReconciledValues:
        del self, args, kwargs
        bottom = _BASE_FORECAST[: matrix.n_bottom].copy()
        bottom[0] = -5.1e-2
        values = matrix.matvec(bottom)
        return ReconciledValues(values, 1e-12)

    monkeypatch.setattr(ProjectionReconciler, "_reconcile_section", reconciled)

    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*series='s1'"):
        build_wls_struct()(
            frame,
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.NONNEGATIVE),
        )


def test_projection_support_validator_preserves_real_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    frame = _frame(hierarchy)

    def reconciled(
        self: ProjectionReconciler,
        *args: object,
        **kwargs: object,
    ) -> ReconciledValues:
        del self, args, kwargs
        values = _BASE_FORECAST.copy()
        values[0] = -5.1e-2
        return ReconciledValues(values, 0.0)

    monkeypatch.setattr(ProjectionReconciler, "_reconcile_section", reconciled)

    result = build_wls_struct()(
        frame,
        hierarchy,
        ReconciliationContext(target_support=TargetSupport.REAL),
    )

    assert (
        result.loc[result[SERIES_KEY] == hierarchy.node_labels[0], POINT_FORECAST].iat[0] == -5.1e-2
    )


def test_projection_requires_exactly_the_nodes_for_the_present_bottom_subset() -> None:
    hierarchy = _hierarchy()
    bottoms = ("s1", "s4")
    applicable = _applicable_labels(hierarchy, bottoms)
    frame = _frame(hierarchy, labels=applicable)
    strategy = build_wls_struct()

    result = strategy(
        frame,
        hierarchy,
        ReconciliationContext(target_support=TargetSupport.REAL),
    )

    assert tuple(result[SERIES_KEY]) == applicable
    missing = frame.loc[frame[SERIES_KEY] != applicable[-1]].reset_index(drop=True)
    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*missing node rows"):
        strategy(
            missing,
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.REAL),
        )

    outside_label = next(
        label
        for label in hierarchy.node_labels
        if label not in applicable and label not in hierarchy.bottom_series
    )
    outside = pd.concat(
        [frame, _frame(hierarchy, labels=(outside_label,))],
        ignore_index=True,
    )
    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*outside"):
        strategy(
            outside,
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.REAL),
        )

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ReconciliationError, match=r"model-a.*2026-01-05.*1.*duplicate"):
        strategy(
            duplicate,
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.REAL),
        )


def test_projection_rejects_a_section_without_bottom_rows() -> None:
    hierarchy = _hierarchy()
    aggregate_labels = tuple(
        label for label in hierarchy.node_labels if label not in hierarchy.bottom_series
    )

    with pytest.raises(ReconciliationError, match="requires at least one bottom-node row"):
        build_wls_struct()(
            _frame(hierarchy, labels=aggregate_labels),
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.REAL),
        )


def test_projection_rejects_distributional_columns() -> None:
    hierarchy = _hierarchy()
    frame = _frame(hierarchy)
    lower, upper = interval_columns(0.9)
    frame[lower] = frame[POINT_FORECAST] - 1.0
    frame[upper] = frame[POINT_FORECAST] + 1.0

    with pytest.raises(ReconciliationError, match="point forecasts only"):
        build_wls_struct()(
            frame,
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.REAL),
        )


@pytest.mark.parametrize(
    "builder,needs_context",
    [
        (build_wls_struct, False),
        (build_wls_var, True),
        (build_mint_shrink, True),
    ],
)
def test_projection_strategies_are_coherent_fixed_points(
    builder: Callable,
    needs_context: bool,
) -> None:
    hierarchy = _hierarchy()
    frame = _frame(hierarchy)
    context = (
        _context(hierarchy)
        if needs_context
        else ReconciliationContext(target_support=TargetSupport.REAL)
    )
    strategy = builder()

    first = strategy(frame, hierarchy, context)
    second = strategy(first, hierarchy, context)
    points = _canonical_points(first, hierarchy)
    matrix = build_dense_summing_matrix(hierarchy)
    expected = matrix.matvec(points[: matrix.n_bottom])
    bound = coherence_tolerance(
        reduction_width=matrix.reduction_width,
        vector_magnitude=float(np.abs(points).max()),
    )

    assert np.allclose(points, expected, atol=bound, rtol=0.0)
    dense = matrix.to_dense()
    if builder is build_mint_shrink:
        covariance, _ = shrink_covariance(_residuals())
        reference = covariance_projection(dense, _BASE_FORECAST, covariance)
        solver_tolerance = 0.0
    else:
        weights = (
            variance_weights(_residuals())[0]
            if builder is build_wls_var
            else structural_weights(dense)
        )
        reference = diagonal_projection(dense, _BASE_FORECAST, weights)
        solver_tolerance = SPARSE_SOLVER_TOLERANCE if builder is build_wls_var else 0.0
    fixed_point_bound = coherence_tolerance(
        reduction_width=matrix.reduction_width,
        vector_magnitude=float(np.abs(points).max()),
        solver_tolerance=solver_tolerance,
        condition_number=reference.condition_number,
    )
    pd.testing.assert_frame_equal(
        first,
        second,
        check_exact=False,
        atol=fixed_point_bound,
        rtol=0.0,
    )


def test_one_membership_corruption_bites_the_reference_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    correct = build_sparse_summing_matrix(hierarchy)
    indices = list(correct.indices)
    data = list(correct.data)
    del indices[correct.indptr[-2]]
    del data[correct.indptr[-2]]
    indptr = (*correct.indptr[:-1], correct.indptr[-1] - 1)
    corrupted = SparseSummingMatrix(
        tuple(data),
        tuple(indices),
        indptr,
        correct.bottom_ids,
        correct.node_labels,
    )
    dense = build_dense_summing_matrix(hierarchy).to_dense()
    reference = diagonal_projection(dense, _BASE_FORECAST, structural_weights(dense))

    import newcalibre.reconcile.nixtla as nixtla

    monkeypatch.setattr(
        nixtla,
        "build_sparse_summing_matrix",
        lambda _hierarchy, *, bottom_ids=None: corrupted.subset(bottom_ids),
    )
    actual = _canonical_points(
        build_wls_struct(dense_workspace_ceiling_bytes=0)(
            _frame(hierarchy),
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.REAL),
        ),
        hierarchy,
    )
    bound = _reference_bound(
        reference.reconciled,
        condition_number=reference.condition_number,
        sparse=True,
    )

    assert float(np.max(np.abs(actual - reference.reconciled))) > bound


def test_starved_bicgstab_raises_with_cross_section_identity() -> None:
    hierarchy = _hierarchy()

    with pytest.raises(
        ReconciliationError,
        match=r"wls_struct.*model-a.*2026-01-05.*horizon_step=1.*exit_code",
    ):
        build_wls_struct(
            dense_workspace_ceiling_bytes=0,
            bicgstab_maxiter=1,
        )(
            _frame(hierarchy),
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.REAL),
        )


def test_wls_var_requires_context_model_nodes_periods_and_timestamp_alignment() -> None:
    hierarchy = _hierarchy()
    strategy = build_wls_var()
    frame = _frame(hierarchy)

    with pytest.raises(ReconciliationError, match="requires fitted values"):
        strategy(
            frame,
            hierarchy,
            ReconciliationContext(target_support=TargetSupport.REAL),
        )

    wrong_model = _fitted_frame(hierarchy, model_name="other-model")
    with pytest.raises(ReconciliationError, match=r"model-a.*no fitted values"):
        strategy(frame, hierarchy, _context(hierarchy, frame=wrong_model))

    missing_node = _fitted_frame(hierarchy)
    missing_node = missing_node.loc[missing_node[SERIES_KEY] != hierarchy.node_labels[-1]]
    with pytest.raises(ReconciliationError, match=r"model-a.*missing fitted-value nodes"):
        strategy(frame, hierarchy, _context(hierarchy, frame=missing_node))

    one_period = _fitted_frame(hierarchy)
    one_period = one_period.loc[one_period[TIMESTAMP] == one_period[TIMESTAMP].min()]
    with pytest.raises(ReconciliationError, match="at least two aligned periods"):
        strategy(frame, hierarchy, _context(hierarchy, frame=one_period))

    misaligned = _fitted_frame(hierarchy)
    node = hierarchy.node_labels[-1]
    last = misaligned.index[misaligned[SERIES_KEY] == node][-1]
    misaligned.loc[last, TIMESTAMP] = pd.Timestamp("2025-02-01")
    with pytest.raises(ReconciliationError, match="timestamp sets are misaligned"):
        strategy(frame, hierarchy, _context(hierarchy, frame=misaligned))


def test_residual_widening_rejects_non_finite_residuals() -> None:
    hierarchy = _hierarchy()
    fitted = _fitted_frame(hierarchy)
    fitted.loc[fitted.index[0], ACTUAL_VALUE] = np.inf

    with pytest.raises(ReconciliationError, match="residual history must be finite"):
        build_wls_var()(
            _frame(hierarchy),
            hierarchy,
            _context(hierarchy, frame=fitted),
        )


def test_partial_residual_widening_ignores_extra_nodes_but_requires_applicable_ones() -> None:
    hierarchy = _hierarchy()
    bottoms = ("s1", "s4")
    applicable = _applicable_labels(hierarchy, bottoms)
    frame = _frame(hierarchy, labels=applicable)

    result = build_wls_var()(frame, hierarchy, _context(hierarchy))

    assert tuple(result[SERIES_KEY]) == applicable
    missing = _fitted_frame(hierarchy)
    missing = missing.loc[missing[SERIES_KEY] != applicable[-1]]
    with pytest.raises(ReconciliationError, match="missing fitted-value nodes"):
        build_wls_var()(frame, hierarchy, _context(hierarchy, frame=missing))


def test_fitted_value_order_is_irrelevant_and_history_is_reused_across_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newcalibre.reconcile.nixtla as nixtla

    aligned_calls = 0
    matrix_calls = 0
    align = nixtla._aligned_fitted_matrices
    build_matrix = nixtla.build_sparse_summing_matrix

    def count_alignment(*args, **kwargs):
        nonlocal aligned_calls
        aligned_calls += 1
        return align(*args, **kwargs)

    def count_matrix(
        hierarchy: HierarchyIndex,
        *,
        bottom_ids: tuple[str, ...] | None = None,
    ):
        nonlocal matrix_calls
        matrix_calls += 1
        return build_matrix(hierarchy, bottom_ids=bottom_ids)

    monkeypatch.setattr(nixtla, "_aligned_fitted_matrices", count_alignment)
    monkeypatch.setattr(nixtla, "build_sparse_summing_matrix", count_matrix)

    hierarchy = _hierarchy()
    fitted = _fitted_frame(hierarchy)
    shuffled = fitted.sample(frac=1.0, random_state=7).reset_index(drop=True)
    first = _frame(hierarchy, horizon_step=1)
    second = _frame(hierarchy, horizon_step=2)
    combined = pd.concat([second, first], ignore_index=True)
    strategy = build_wls_var()

    ordered_result = strategy(combined, hierarchy, _context(hierarchy, frame=fitted))
    shuffled_result = strategy(combined, hierarchy, _context(hierarchy, frame=shuffled))

    pd.testing.assert_frame_equal(ordered_result, shuffled_result)
    by_horizon = ordered_result.groupby(HORIZON_STEP, sort=True)[POINT_FORECAST].apply(list)
    assert np.allclose(by_horizon.loc[1], by_horizon.loc[2])
    assert aligned_calls == 2
    assert matrix_calls == 2


def test_variance_floor_is_exact_and_keeps_one_zero_variance_node_positive() -> None:
    residuals = _residuals()
    residuals[3] = 7.0
    derived = derive_variance_weights(residuals)
    expected, expected_floor = variance_weights(residuals)

    assert derived.floor == expected_floor
    assert np.array_equal(derived.values, expected)
    assert derived.values[3] == derived.floor
    assert derived.floor > 0.0


def test_wls_var_matches_independent_dense_reference_and_never_builds_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    dense = build_dense_summing_matrix(hierarchy).to_dense()
    weights, _ = variance_weights(_residuals())
    reference = diagonal_projection(dense, _BASE_FORECAST, weights)

    import newcalibre.reconcile.nixtla as nixtla

    def forbidden(_hierarchy: HierarchyIndex):
        raise AssertionError("wls_var invoked the dense producer")

    monkeypatch.setattr(nixtla, "build_dense_summing_matrix", forbidden)
    actual = _canonical_points(
        build_wls_var()(_frame(hierarchy), hierarchy, _context(hierarchy)),
        hierarchy,
    )

    _assert_reference_agreement(
        actual,
        reference.reconciled,
        condition_number=reference.condition_number,
        sparse=True,
    )


def test_mint_shrink_matches_independent_reference_at_and_below_ceiling() -> None:
    hierarchy = _hierarchy()
    dense = build_dense_summing_matrix(hierarchy).to_dense()
    covariance, intensity = shrink_covariance(_residuals())
    assert 0.0 < intensity < 1.0
    reference = covariance_projection(dense, _BASE_FORECAST, covariance)

    actual = _canonical_points(
        build_mint_shrink(dense_workspace_ceiling_bytes=12_288)(
            _frame(hierarchy),
            hierarchy,
            _context(hierarchy),
        ),
        hierarchy,
    )

    _assert_reference_agreement(
        actual,
        reference.reconciled,
        condition_number=reference.condition_number,
        sparse=False,
        covariance_estimator=True,
    )


def test_mint_shrink_preflights_and_builds_only_the_applicable_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    bottoms = ("s1", "s4")
    applicable = _applicable_labels(hierarchy, bottoms)
    subset_estimate = preflight_projection(
        "mint_shrink",
        ProjectionMetadata(
            n_bottom=len(bottoms),
            n_nodes=len(applicable),
            n_attributes=len(hierarchy.attribute_names),
            residual_periods=8,
        ),
    )
    fitted = _fitted_frame(hierarchy)
    outside_label = next(label for label in hierarchy.node_labels if label not in applicable)
    extra = fitted.loc[fitted[SERIES_KEY] == outside_label].iloc[[0] * 20].copy(deep=True)
    extra[TIMESTAMP] = pd.date_range("2024-01-01", periods=len(extra), freq="D")
    fitted = pd.concat([fitted, extra], ignore_index=True)
    full_estimate = preflight_projection(
        "mint_shrink",
        ProjectionMetadata(
            n_bottom=len(hierarchy.bottom_series),
            n_nodes=len(hierarchy.node_labels),
            n_attributes=len(hierarchy.attribute_names),
            residual_periods=int(fitted[TIMESTAMP].nunique()),
        ),
    )
    assert subset_estimate.dense_workspace_bytes < full_estimate.dense_workspace_bytes

    import newcalibre.reconcile.nixtla as nixtla

    original = nixtla.build_dense_summing_matrix
    built_subsets: list[tuple[str, ...] | None] = []

    def build_subset(
        hierarchy: HierarchyIndex,
        *,
        bottom_ids: tuple[str, ...] | None = None,
    ):
        built_subsets.append(bottom_ids)
        return original(hierarchy, bottom_ids=bottom_ids)

    monkeypatch.setattr(nixtla, "build_dense_summing_matrix", build_subset)
    result = build_mint_shrink(
        dense_workspace_ceiling_bytes=subset_estimate.dense_workspace_bytes,
    )(
        _frame(hierarchy, labels=applicable),
        hierarchy,
        _context(hierarchy, frame=fitted),
    )

    assert tuple(result[SERIES_KEY]) == applicable
    assert built_subsets == [bottoms]


def test_mint_shrink_rejects_before_residual_widening_or_matrix_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = _hierarchy()
    import newcalibre.reconcile.nixtla as nixtla

    def forbidden(*_args, **_kwargs):
        raise AssertionError("rejected projection allocated forbidden workspace")

    monkeypatch.setattr(nixtla, "build_dense_summing_matrix", forbidden)
    monkeypatch.setattr(nixtla, "_aligned_fitted_matrices", forbidden)
    monkeypatch.setattr(FittedValues, "frame", property(forbidden))

    with pytest.raises(
        ReconciliationError,
        match=r"mint_shrink.*dense-only.*wls_var.*wls_struct",
    ):
        build_mint_shrink(dense_workspace_ceiling_bytes=0)(
            _frame(hierarchy),
            hierarchy,
            _context(hierarchy),
        )
