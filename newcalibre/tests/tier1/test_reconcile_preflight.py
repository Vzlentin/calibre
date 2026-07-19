"""Exercise metadata-only reconciliation workspace admission."""

from __future__ import annotations

import tracemalloc

import pandas as pd
import pytest

from newcalibre.domain import HierarchyIndex
from newcalibre.reconcile.preflight import (
    DENSE_PERMITTED,
    DENSE_WORKSPACE_CEILING_BYTES,
    REJECTED_AT_SCALE,
    SPARSE_REQUIRED,
    ProjectionMetadata,
    metadata_from_hierarchy,
    preflight_projection,
)

_FIXTURE = ProjectionMetadata(
    n_bottom=6,
    n_nodes=12,
    n_attributes=2,
    residual_periods=8,
    dtype_bytes=8,
)
_M5 = ProjectionMetadata(
    n_bottom=30_490,
    n_nodes=33_563,
    n_attributes=5,
    residual_periods=1_941,
    dtype_bytes=8,
)


def _components(result, *, sparse: bool = False) -> dict[str, int]:
    items = result.sparse_components if sparse else result.dense_components
    assert items is not None
    return {component.name: component.bytes for component in items}


def test_fixture_workspace_components_are_exact() -> None:
    struct = preflight_projection("wls_struct", _FIXTURE)
    variance = preflight_projection("wls_var", _FIXTURE)
    shrink = preflight_projection("mint_shrink", _FIXTURE)

    assert _components(struct) == {
        "summing_matrix_canonical": 576,
        "summing_matrix_conversion_temporary": 576,
        "summing_matrix_nixtla": 576,
        "constraint_matrices": 1_152,
        "weights_dense": 1_152,
        "weighted_constraints": 576,
        "linear_system": 576,
        "factor_temporary": 576,
        "projection_intermediates": 1_440,
        "weights_diagonal": 96,
    }
    assert _components(struct, sparse=True) == {
        "summing_matrix_csr": 340,
        "weights_diagonal": 96,
        "iterative_vectors": 384,
    }
    residual_components = {
        "residual_aligned_project": 1_536,
        "residual_aligned_nixtla": 1_536,
        "residual_upstream": 768,
        "residual_nan_mask": 96,
    }
    assert _components(variance) == {
        **_components(struct),
        **residual_components,
    }
    assert _components(shrink) == {
        **{
            name: value for name, value in _components(struct).items() if name != "weights_diagonal"
        },
        "covariance_estimator_temporary": 1_152,
        **residual_components,
    }
    assert struct.dense_workspace_bytes == 7_296
    assert _components(variance, sparse=True) == {
        **_components(struct, sparse=True),
        "residual_aligned_project": 1_536,
        "residual_variance_temporary": 768,
    }
    assert variance.sparse_workspace_bytes == 3_124
    assert variance.dense_workspace_bytes == 11_232
    assert shrink.dense_workspace_bytes == 12_288
    assert struct.decision == DENSE_PERMITTED
    assert variance.decision == DENSE_PERMITTED
    assert shrink.decision == DENSE_PERMITTED


def test_full_m5_workspace_components_and_sparse_pivot_are_exact() -> None:
    struct = preflight_projection("wls_struct", _M5)
    variance = preflight_projection("wls_var", _M5)
    shrink = preflight_projection("mint_shrink", _M5)

    assert _components(struct)["summing_matrix_canonical"] == 8_186_686_960
    assert _components(struct)["summing_matrix_conversion_temporary"] == 8_186_686_960
    assert _components(struct)["summing_matrix_nixtla"] == 8_186_686_960
    assert _components(struct)["weights_dense"] == 9_011_799_752
    assert _components(struct, sparse=True) == {
        "summing_matrix_csr": 2_695_416,
        "weights_diagonal": 268_504,
        "iterative_vectors": 1_512_688,
    }
    assert struct.dense_workspace_bytes == 62_182_207_344
    assert struct.sparse_workspace_bytes == 4_476_608
    assert struct.decision == SPARSE_REQUIRED

    assert _components(variance)["residual_aligned_project"] == 1_042_332_528
    assert _components(variance)["residual_nan_mask"] == 65_145_783
    assert variance.dense_workspace_bytes == 64_853_184_447
    assert variance.sparse_workspace_bytes == 1_567_975_400
    assert variance.decision == SPARSE_REQUIRED

    assert _components(shrink)["covariance_estimator_temporary"] == 9_011_799_752
    assert shrink.dense_workspace_bytes == 73_864_715_695
    assert shrink.sparse_components is None
    assert shrink.sparse_workspace_bytes is None
    assert shrink.decision == REJECTED_AT_SCALE
    assert "wls_var" in shrink.reason
    assert "wls_struct" in shrink.reason
    assert "summing_matrix_canonical=8186686960 B" in shrink.reason


def test_ceiling_admits_equality_and_eight_bytes_change_the_decision() -> None:
    exact = preflight_projection("wls_struct", _FIXTURE)

    admitted = preflight_projection(
        "wls_struct",
        _FIXTURE,
        ceiling_bytes=exact.dense_workspace_bytes,
    )
    pivoted = preflight_projection(
        "wls_struct",
        _FIXTURE,
        ceiling_bytes=exact.dense_workspace_bytes - 8,
    )

    assert admitted.decision == DENSE_PERMITTED
    assert pivoted.decision == SPARSE_REQUIRED


def test_mint_shrink_is_admitted_at_ceiling_and_rejected_only_above() -> None:
    estimate = preflight_projection("mint_shrink", _FIXTURE)

    at_ceiling = preflight_projection(
        "mint_shrink",
        _FIXTURE,
        ceiling_bytes=estimate.dense_workspace_bytes,
    )
    above_ceiling = preflight_projection(
        "mint_shrink",
        _FIXTURE,
        ceiling_bytes=estimate.dense_workspace_bytes - 8,
    )

    assert at_ceiling.decision == DENSE_PERMITTED
    assert above_ceiling.decision == REJECTED_AT_SCALE
    assert "wls_var" in above_ceiling.reason
    assert "wls_struct" in above_ceiling.reason
    assert "covariance_estimator_temporary=1152 B" in above_ceiling.reason


def test_metadata_is_derived_only_from_the_hierarchy_index() -> None:
    facts = pd.DataFrame.from_records(
        [
            {"series_key": "c", "department": "outer", "store": 2},
            {"series_key": "a", "department": "inner", "store": 1},
            {"series_key": "b", "department": "inner", "store": 2},
        ]
    )
    hierarchy = HierarchyIndex.from_facts(facts, bottom_series=("c", "b", "a"))

    metadata = metadata_from_hierarchy(hierarchy, residual_periods=17)

    assert metadata == ProjectionMetadata(
        n_bottom=len(hierarchy.bottom_series),
        n_nodes=len(hierarchy.node_labels),
        n_attributes=len(hierarchy.attribute_names),
        residual_periods=17,
        dtype_bytes=8,
    )


def test_m5_preflight_is_constant_memory_and_never_builds_a_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newcalibre.reconcile.summing as summing

    def forbidden(*_args, **_kwargs):
        raise AssertionError("metadata preflight invoked a summing-matrix producer")

    monkeypatch.setattr(summing, "build_dense_summing_matrix", forbidden)
    monkeypatch.setattr(summing, "build_sparse_summing_matrix", forbidden)

    tracemalloc.start()
    try:
        for strategy in ("wls_struct", "wls_var", "mint_shrink"):
            preflight_projection(strategy, _M5)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000


def test_preflight_uses_the_one_gibibyte_default() -> None:
    assert DENSE_WORKSPACE_CEILING_BYTES == 1 << 30
