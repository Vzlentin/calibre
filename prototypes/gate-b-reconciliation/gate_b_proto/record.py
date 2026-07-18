"""Record the fixture result and the full-M5-scale estimate as reviewable JSON.

Run with ``uv run python -m gate_b_proto.record`` from the prototype root (or
through the recorded-files freshness test). The outputs are prototype
evidence, not baselines: every number is recomputed from the fixture's first
principles on each run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gate_b_proto.fixture import BASE_FORECAST, fixture_lattice, fixture_residuals
from gate_b_proto.formulations import (
    mint_shrink_covariance,
    normal_equation_condition,
    project_dense,
    project_dense_covariance,
    project_sparse,
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
    FORMULATIONS,
    M5_METADATA,
    LatticeMetadata,
    dense_summing_matrix_bytes,
    estimate_formulation,
    summing_matrix_nnz,
)

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "recorded"


def _estimate_dict(estimate: Any) -> dict[str, Any]:
    return {
        "formulation": estimate.formulation,
        "decision": estimate.decision,
        "reason": estimate.reason,
        "dense_components": dict(estimate.dense_components),
        "sparse_components": (
            dict(estimate.sparse_components) if estimate.sparse_components is not None else None
        ),
        "dense_workspace_bytes": estimate.dense_workspace_bytes,
        "sparse_workspace_bytes": estimate.sparse_workspace_bytes,
        "ceiling_bytes": estimate.ceiling_bytes,
    }


def fixture_result() -> dict[str, Any]:
    """Compute every recorded fixture fact from first principles."""
    lattice = fixture_lattice()
    S_dense = dense_summing_matrix(lattice)
    S_sparse = sparse_summing_matrix(lattice)
    residuals = fixture_residuals()
    struct_w = wls_struct_weights(lattice)
    var_w = wls_var_weights(residuals, lattice.n_nodes)
    shrunk = mint_shrink_covariance(residuals, lattice.n_nodes)

    struct_dense = project_dense(S_dense, struct_w, BASE_FORECAST)
    struct_sparse = project_sparse(S_sparse, struct_w, BASE_FORECAST)
    var_dense = project_dense(S_dense, var_w.values, BASE_FORECAST)
    var_sparse = project_sparse(S_sparse, var_w.values, BASE_FORECAST)
    shrink_dense = project_dense_covariance(S_dense, shrunk.matrix, BASE_FORECAST)

    struct_condition = normal_equation_condition(S_dense, struct_w)
    var_condition = normal_equation_condition(S_dense, var_w.values)
    magnitude = float(np.max(np.abs(BASE_FORECAST)))
    max_members = lattice.n_bottom
    struct_sparse_tol = derived_projection_tolerance(
        max_members=max_members,
        condition=struct_condition,
        magnitude=magnitude,
        solver_rtol=struct_sparse.solver_rtol,
    )
    var_sparse_tol = derived_projection_tolerance(
        max_members=max_members,
        condition=var_condition,
        magnitude=magnitude,
        solver_rtol=var_sparse.solver_rtol,
    )

    fixture_meta = LatticeMetadata(
        n_bottom=lattice.n_bottom,
        n_nodes=lattice.n_nodes,
        n_attributes=lattice.n_attributes,
        residual_periods=residuals.shape[1],
    )

    return {
        "lattice": {
            "n_bottom": lattice.n_bottom,
            "n_nodes": lattice.n_nodes,
            "n_attributes": lattice.n_attributes,
            "nnz": lattice.nnz,
            "node_labels": list(lattice.node_labels),
        },
        "base_forecast": BASE_FORECAST.tolist(),
        "structural_weights": struct_w.tolist(),
        "variance_weights": var_w.values.tolist(),
        "variance_floor": var_w.floor,
        "shrinkage_intensity": shrunk.intensity,
        "condition_numbers": {
            "wls_struct": struct_condition,
            "wls_var": var_condition,
            "mint_shrink": shrink_dense.condition,
        },
        "reconciled": {
            "wls_struct_dense": struct_dense.reconciled.tolist(),
            "wls_struct_sparse": struct_sparse.reconciled.tolist(),
            "wls_var_dense": var_dense.reconciled.tolist(),
            "wls_var_sparse": var_sparse.reconciled.tolist(),
            "mint_shrink_dense": shrink_dense.reconciled.tolist(),
        },
        "dense_vs_sparse_max_abs_diff": {
            "wls_struct": float(np.max(np.abs(struct_dense.reconciled - struct_sparse.reconciled))),
            "wls_var": float(np.max(np.abs(var_dense.reconciled - var_sparse.reconciled))),
        },
        "coherence_max_abs_residual": {
            "wls_struct_dense": float(
                np.max(
                    np.abs(coherence_residual(S_dense, struct_dense.reconciled, lattice.n_bottom))
                )
            ),
            "wls_var_dense": float(
                np.max(np.abs(coherence_residual(S_dense, var_dense.reconciled, lattice.n_bottom)))
            ),
            "mint_shrink_dense": float(
                np.max(
                    np.abs(coherence_residual(S_dense, shrink_dense.reconciled, lattice.n_bottom))
                )
            ),
        },
        "derived_tolerances": {
            "wls_struct_sparse": struct_sparse_tol,
            "wls_var_sparse": var_sparse_tol,
        },
        "preflight_fixture": {
            name: _estimate_dict(estimate_formulation(name, fixture_meta)) for name in FORMULATIONS
        },
    }


def m5_scale_estimate() -> dict[str, Any]:
    """Compute the full-M5-scale preflight estimates from checked-in metadata.

    Cross-checks against the spec's published constants (`[PRF-21]`'s ~7.6 GiB
    dense matrix, chapter 07's `nnz = n_bottom * (A + 2)` rule) are recomputed,
    not transcribed; a drift raises instead of recording a false match.
    """
    dense_bytes = dense_summing_matrix_bytes(M5_METADATA)
    nnz = summing_matrix_nnz(M5_METADATA)
    if dense_bytes != 8_186_686_960 or nnz != 213_430:
        raise ValueError(
            f"M5 metadata drifted from the spec's published constants: dense S "
            f"{dense_bytes} B (expect 8,186,686,960), nnz {nnz} (expect 213,430)"
        )
    return {
        "metadata": {
            "n_bottom": M5_METADATA.n_bottom,
            "n_nodes": M5_METADATA.n_nodes,
            "n_attributes": M5_METADATA.n_attributes,
            "residual_periods": M5_METADATA.residual_periods,
            "sources": [
                "docs/spec/21-protocol-m5.md [M5-H2] (30,490 bottom; 33,563 nodes)",
                "docs/spec/21-protocol-m5.md [M5-H1]/[M5-D1] (five attribute columns)",
                "docs/spec/21-protocol-m5.md [M5-D3] (evaluation phase: 1,941 days)",
            ],
        },
        "cross_checks": {
            "dense_summing_bytes_matches_prf21": True,
            "dense_summing_gib": dense_bytes / 1024**3,
            "nnz": nnz,
        },
        "estimates": {
            name: _estimate_dict(estimate_formulation(name, M5_METADATA)) for name in FORMULATIONS
        },
    }


def main() -> None:
    """Write both recorded JSON artifacts, deterministically."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("fixture-result.json", fixture_result()),
        ("m5-scale-estimate.json", m5_scale_estimate()),
    ):
        path = _RESULTS_DIR / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
