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
        "summing_matrix_dense": 576,
        "weights_diagonal": 96,
        "normal_equations": 288,
        "factor_temporary": 288,
    }
    assert _components(struct, sparse=True) == {
        "summing_matrix_csr": 340,
        "weights_diagonal": 96,
        "iterative_vectors": 384,
    }
    assert _components(variance) == {
        **_components(struct),
        "residual_wide": 1_536,
    }
    assert _components(shrink) == {
        "summing_matrix_dense": 576,
        "covariance_dense": 1_152,
        "normal_equations": 288,
        "factor_temporary": 288,
        "residual_wide": 1_536,
    }
    assert struct.dense_workspace_bytes == 1_248
    assert variance.dense_workspace_bytes == 2_784
    assert shrink.dense_workspace_bytes == 3_840
    assert struct.decision == DENSE_PERMITTED
    assert variance.decision == DENSE_PERMITTED
    assert shrink.decision == DENSE_PERMITTED


def test_full_m5_workspace_components_and_sparse_pivot_are_exact() -> None:
    struct = preflight_projection("wls_struct", _M5)
    variance = preflight_projection("wls_var", _M5)
    shrink = preflight_projection("mint_shrink", _M5)

    assert _components(struct)["summing_matrix_dense"] == 8_186_686_960
    assert _components(struct)["normal_equations"] == 7_437_120_800
    assert _components(struct, sparse=True) == {
        "summing_matrix_csr": 2_695_416,
        "weights_diagonal": 268_504,
        "iterative_vectors": 1_512_688,
    }
    assert struct.dense_workspace_bytes == 23_061_197_064
    assert struct.sparse_workspace_bytes == 4_476_608
    assert struct.decision == SPARSE_REQUIRED

    assert _components(variance)["residual_wide"] == 1_042_332_528
    assert variance.dense_workspace_bytes == 24_103_529_592
    assert variance.sparse_workspace_bytes == 1_046_809_136
    assert variance.decision == SPARSE_REQUIRED

    assert _components(shrink)["covariance_dense"] == 9_011_799_752
    assert shrink.dense_workspace_bytes == 33_115_060_840
    assert shrink.sparse_components is None
    assert shrink.sparse_workspace_bytes is None
    assert shrink.decision == REJECTED_AT_SCALE
    assert "wls_var" in shrink.reason
    assert "wls_struct" in shrink.reason
    assert "summing_matrix_dense=8186686960 B" in shrink.reason


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
    assert "covariance_dense=1152 B" in above_ceiling.reason


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
