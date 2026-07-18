"""Gate B prototype checks: formulation equivalence, witnesses, and preflight.

Every numeric gate pairs with its witness (chapter 50): the dense/sparse
equivalence gate is shown to bite on a one-membership corruption, the solver
guard on a starved solve, and the preflight ceiling on boundary metadata.
Tolerance classes cited per assertion.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from gate_b_proto.fixture import BASE_FORECAST, fixture_lattice, fixture_residuals
from gate_b_proto.formulations import (
    SolverConvergenceError,
    mint_shrink_covariance,
    normal_equation_condition,
    project_dense,
    project_dense_covariance,
    project_sparse,
    validate_residuals,
    wls_struct_weights,
    wls_var_weights,
)
from gate_b_proto.lattice import (
    coherence_residual,
    dense_summing_matrix,
    derived_projection_tolerance,
    sparse_summing_matrix,
)
from gate_b_proto.preflight import (
    DENSE_PERMITTED,
    DENSE_WORKSPACE_CEILING_BYTES,
    FORMULATIONS,
    M5_METADATA,
    REJECTED,
    SPARSE_REQUIRED,
    LatticeMetadata,
    estimate_formulation,
)
from gate_b_proto.record import fixture_result, m5_scale_estimate
from scipy import sparse

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "recorded"


def test_lattice_structure_exact() -> None:
    """Class 1 (exact structural): node-count identity, nnz rule, dense==sparse values."""
    lattice = fixture_lattice()
    assert lattice.n_bottom == 6
    assert lattice.n_nodes == 12
    assert lattice.nnz == 24  # n_bottom * (n_attributes + 2)
    S_dense = dense_summing_matrix(lattice)
    S_sparse = sparse_summing_matrix(lattice)
    # [REC-13]: the two representations of one index are value-identical.
    np.testing.assert_array_equal(S_sparse.toarray(), S_dense)
    assert S_dense.shape == (12, 6)
    # Hand-checkable membership: channel=a aggregates s1+s4; the total sums all.
    np.testing.assert_array_equal(S_dense[6], [1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(S_dense[11], [1, 1, 1, 1, 1, 1])
    # Structural weights are lattice structure, exactly: diag(S S').
    np.testing.assert_array_equal(wls_struct_weights(lattice), [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 6])


def test_dense_sparse_equivalence_within_derived_bound() -> None:
    """Class 3: sparse operator path agrees with the dense closed form.

    The bound is the shared derived tolerance evaluated per formulation:
    fixture conditioning (exact kappa at this scale) plus the solver's
    declared relative tolerance.
    """
    lattice = fixture_lattice()
    S_dense = dense_summing_matrix(lattice)
    S_sparse = sparse_summing_matrix(lattice)
    residuals = fixture_residuals()
    magnitude = float(np.max(np.abs(BASE_FORECAST)))

    for weights in (wls_struct_weights(lattice), wls_var_weights(residuals, 12).values):
        dense = project_dense(S_dense, weights, BASE_FORECAST)
        sparse_result = project_sparse(S_sparse, weights, BASE_FORECAST)
        condition = normal_equation_condition(S_dense, weights)
        bound = derived_projection_tolerance(
            max_members=lattice.n_bottom,
            condition=condition,
            magnitude=magnitude,
            solver_rtol=sparse_result.solver_rtol,
        )
        diff = float(np.max(np.abs(dense.reconciled - sparse_result.reconciled)))
        assert diff <= bound
        assert bound < 1e-6  # the derived bound stays tight at fixture scale


def test_coherence_within_derived_bound() -> None:
    """[REC-12]: every formulation's output satisfies r = S @ r[:n_bottom]."""
    lattice = fixture_lattice()
    S_dense = dense_summing_matrix(lattice)
    residuals = fixture_residuals()
    magnitude = float(np.max(np.abs(BASE_FORECAST)))
    candidates = [
        project_dense(S_dense, wls_struct_weights(lattice), BASE_FORECAST),
        project_dense(S_dense, wls_var_weights(residuals, 12).values, BASE_FORECAST),
    ]
    shrunk = mint_shrink_covariance(residuals, lattice.n_nodes)
    candidates.append(project_dense_covariance(S_dense, shrunk.matrix, BASE_FORECAST))
    for result in candidates:
        residual = coherence_residual(S_dense, result.reconciled, lattice.n_bottom)
        bound = derived_projection_tolerance(
            max_members=lattice.n_bottom,
            condition=result.condition,
            magnitude=magnitude,
            solver_rtol=result.solver_rtol,
        )
        assert float(np.max(np.abs(residual))) <= bound


def test_idempotence() -> None:
    """[REC-22]: reconciling an already-coherent frame is a fixed point."""
    lattice = fixture_lattice()
    S_dense = dense_summing_matrix(lattice)
    weights = wls_struct_weights(lattice)
    once = project_dense(S_dense, weights, BASE_FORECAST)
    twice = project_dense(S_dense, weights, once.reconciled)
    bound = derived_projection_tolerance(
        max_members=lattice.n_bottom,
        condition=once.condition,
        magnitude=float(np.max(np.abs(once.reconciled))),
        solver_rtol=0.0,
    )
    assert float(np.max(np.abs(once.reconciled - twice.reconciled))) <= bound


def test_witness_membership_corruption_breaks_the_gate() -> None:
    """Witness: dropping one S membership must fail the equivalence gate.

    The smallest meaningful drift for this gate is one summing-matrix entry;
    a corrupted membership moves the affected aggregate by a whole bottom
    forecast — orders of magnitude above the derived bound.
    """
    lattice = fixture_lattice()
    S_dense = dense_summing_matrix(lattice)
    S_corrupted = sparse_summing_matrix(lattice)
    S_corrupted = sparse.csr_array(S_corrupted)
    # Zero the total row's membership of s1 (row 11, column 0).
    S_corrupted[11, 0] = 0.0
    weights = wls_struct_weights(lattice)
    dense = project_dense(S_dense, weights, BASE_FORECAST)
    corrupted = project_sparse(S_corrupted, weights, BASE_FORECAST)
    condition = normal_equation_condition(S_dense, weights)
    bound = derived_projection_tolerance(
        max_members=lattice.n_bottom,
        condition=condition,
        magnitude=float(np.max(np.abs(BASE_FORECAST))),
        solver_rtol=corrupted.solver_rtol,
    )
    diff = float(np.max(np.abs(dense.reconciled - corrupted.reconciled)))
    assert diff > bound


def test_witness_starved_solver_raises() -> None:
    """Witness [REC-21]: a solver starved of iterations fails loudly."""
    lattice = fixture_lattice()
    S_sparse = sparse_summing_matrix(lattice)
    weights = wls_struct_weights(lattice)
    with pytest.raises(SolverConvergenceError, match="exit_code"):
        project_sparse(S_sparse, weights, BASE_FORECAST, maxiter=1)


def test_wls_var_floor_handles_zero_variance_node() -> None:
    """Singular behavior: a constant-residual node yields a floored finite weight."""
    residuals = fixture_residuals()
    residuals[3] = 7.0  # zero variance at node s4
    weights = wls_var_weights(residuals, 12)
    assert np.all(np.isfinite(weights.values))
    assert weights.values[3] == weights.floor
    expected_floor = float(np.var(residuals, axis=1, ddof=1).max()) * 8 * np.finfo(float).eps
    assert weights.floor == pytest.approx(expected_floor, rel=1e-12)
    # The floored path still reconciles coherently.
    lattice = fixture_lattice()
    S_dense = dense_summing_matrix(lattice)
    result = project_dense(S_dense, weights.values, BASE_FORECAST)
    residual = coherence_residual(S_dense, result.reconciled, lattice.n_bottom)
    assert float(np.max(np.abs(residual))) < 1e-8


def test_wls_var_rejects_degenerate_residuals() -> None:
    """The fitted-values contract fails loudly for T < 2 or ragged input."""
    with pytest.raises(ValueError, match="at least 2 periods"):
        validate_residuals(np.ones((12, 1)), 12)
    with pytest.raises(ValueError, match="shape"):
        validate_residuals(np.ones((11, 8)), 12)


def test_mint_cov_rank_deficiency_is_why_it_is_rejected() -> None:
    """[REC-10] witness: the raw sample covariance is singular for T < n_nodes."""
    residuals = fixture_residuals()  # T=8 < n_nodes=12
    covariance = np.cov(residuals, ddof=1)
    assert np.linalg.matrix_rank(covariance) < 12
    with pytest.raises(np.linalg.LinAlgError):
        project_dense_covariance(dense_summing_matrix(fixture_lattice()), covariance, BASE_FORECAST)


def test_mint_shrink_stays_positive_definite_where_mint_cov_fails() -> None:
    """The shrinkage target rescues the same residuals: W is symmetric PD."""
    lattice = fixture_lattice()
    shrunk = mint_shrink_covariance(fixture_residuals(), lattice.n_nodes)
    assert 0.0 < shrunk.intensity <= 1.0
    np.linalg.cholesky(shrunk.matrix)  # raises if not positive definite
    result = project_dense_covariance(dense_summing_matrix(lattice), shrunk.matrix, BASE_FORECAST)
    residual = coherence_residual(
        dense_summing_matrix(lattice), result.reconciled, lattice.n_bottom
    )
    bound = derived_projection_tolerance(
        max_members=lattice.n_bottom,
        condition=result.condition,
        magnitude=float(np.max(np.abs(BASE_FORECAST))),
        solver_rtol=0.0,
    )
    assert float(np.max(np.abs(residual))) <= bound


def test_preflight_fixture_permits_dense_and_rejects_mint_cov() -> None:
    """At fixture scale everything dense fits; mint_cov is rejected by name."""
    meta = LatticeMetadata(n_bottom=6, n_nodes=12, n_attributes=2, residual_periods=8)
    for name in ("wls_struct", "wls_var", "mint_shrink"):
        estimate = estimate_formulation(name, meta)
        assert estimate.decision == DENSE_PERMITTED
        assert estimate.dense_workspace_bytes <= DENSE_WORKSPACE_CEILING_BYTES
    assert estimate_formulation("mint_cov", meta).decision == REJECTED


def test_preflight_m5_decisions_from_metadata() -> None:
    """[PRF-21]: at full-M5 metadata, diagonal formulations go sparse; dense-only rejects."""
    struct_estimate = estimate_formulation("wls_struct", M5_METADATA)
    assert struct_estimate.decision == SPARSE_REQUIRED
    assert struct_estimate.dense_components[0] == ("summing_matrix_dense", 8_186_686_960)
    assert struct_estimate.sparse_components is not None
    assert dict(struct_estimate.sparse_components)["summing_matrix_csr"] == 2_695_416
    var_estimate = estimate_formulation("wls_var", M5_METADATA)
    assert var_estimate.decision == SPARSE_REQUIRED
    assert dict(var_estimate.sparse_components or ())["residual_wide"] == 2 * 33_563 * 1_941 * 8
    shrink_estimate = estimate_formulation("mint_shrink", M5_METADATA)
    assert shrink_estimate.decision == REJECTED
    assert "wls_var or wls_struct" in shrink_estimate.reason
    assert estimate_formulation("mint_cov", M5_METADATA).decision == REJECTED


def test_witness_preflight_ceiling_boundary() -> None:
    """Witness: the ceiling bites at exactly one float64 row of drift."""
    meta = LatticeMetadata(n_bottom=6, n_nodes=12, n_attributes=2, residual_periods=8)
    exact = estimate_formulation("wls_struct", meta)
    at_ceiling = estimate_formulation("wls_struct", meta, ceiling_bytes=exact.dense_workspace_bytes)
    assert at_ceiling.decision == DENSE_PERMITTED  # <= is permitted
    below_ceiling = estimate_formulation(
        "wls_struct", meta, ceiling_bytes=exact.dense_workspace_bytes - 8
    )
    assert below_ceiling.decision == SPARSE_REQUIRED
    dense_only_below = estimate_formulation(
        "mint_shrink", meta, ceiling_bytes=exact.dense_workspace_bytes - 8
    )
    assert dense_only_below.decision == REJECTED


def test_preflight_allocates_no_forbidden_structure() -> None:
    """[PRF-2]/[REC-16]: the estimate is O(metadata) — no n^2 allocation, ever."""
    tracemalloc.start()
    try:
        for name in FORMULATIONS:
            estimate_formulation(name, M5_METADATA)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000  # bytes; M5 dense workspace would be ~31 GiB


def test_recorded_files_match_recomputation() -> None:
    """The committed JSON artifacts cannot drift from the prototype's own math."""
    recorded_fixture = json.loads((_RESULTS_DIR / "fixture-result.json").read_text())
    recorded_m5 = json.loads((_RESULTS_DIR / "m5-scale-estimate.json").read_text())
    fresh_fixture = fixture_result()
    fresh_m5 = m5_scale_estimate()
    assert recorded_fixture["lattice"] == fresh_fixture["lattice"]
    assert recorded_fixture["preflight_fixture"] == fresh_fixture["preflight_fixture"]
    for key, value in fresh_fixture["dense_vs_sparse_max_abs_diff"].items():
        assert recorded_fixture["dense_vs_sparse_max_abs_diff"][key] == pytest.approx(
            value, rel=1e-9, abs=1e-15
        )
    assert recorded_fixture["shrinkage_intensity"] == pytest.approx(
        fresh_fixture["shrinkage_intensity"], rel=1e-9
    )
    assert recorded_m5 == fresh_m5  # integer estimates: exact
