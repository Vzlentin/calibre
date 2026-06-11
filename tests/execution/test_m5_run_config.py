from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibre.cli.commands import (
    _load_dataset,
    run_config,
)
from calibre.cli.config import load_config_from_mapping
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y
from calibre.execution.hierarchy_memory import (
    effective_available_memory_bytes,
    estimate_hierarchical_expansion,
    estimated_node_history_peak_bytes,
    read_cgroup_available_memory_bytes,
)
from calibre.execution.task_builder import build_node_history
from calibre.execution.validation import validate_dataset_bundle
from calibre.reconciliation.summing import (
    TOTAL_LABEL,
    build_hierarchy_index,
    build_summing_matrix,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m5"


def test_hierarchical_expansion_estimate_counts_lattice_nodes() -> None:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "B", "B"],
            DS: pd.date_range("2024-01-01", periods=2, freq="D").tolist() * 2,
            Y: [1.0, 2.0, 3.0, 4.0],
        }
    )
    hierarchy = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            "dept_id": ["D1", "D2"],
            "state_id": ["CA", "CA"],
        }
    )

    estimate = estimate_hierarchical_expansion(
        history,
        build_hierarchy_index(hierarchy),
        horizon=3,
        model_count=2,
    )

    assert estimate.bottom_unique_ids == 2
    assert estimate.aggregate_nodes == 4
    assert estimate.node_count == 6
    assert estimate.bottom_rows == 4
    assert estimate.projected_node_count == 6
    assert estimate.projected_node_history_rows == 12
    assert estimate.forecast_partitions == 36


def test_hierarchical_expansion_estimate_matches_ragged_node_history_rows() -> None:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "B", "C", "C"],
            DS: pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-01", "2024-01-02"]
            ),
            Y: [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    hierarchy = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B", "C"],
            "dept_id": ["D1", "D1", "D2"],
            "store_id": ["S1", "S2", "S1"],
        }
    )

    node_history = build_node_history(history, build_hierarchy_index(hierarchy))
    estimate = estimate_hierarchical_expansion(history, build_hierarchy_index(hierarchy), horizon=2)
    naive_full_panel_rows = 2 * len(build_summing_matrix(hierarchy).node_labels)

    assert estimate.projected_node_history_rows == len(node_history) == 12
    assert estimate.projected_node_history_rows < naive_full_panel_rows
    assert estimate.projected_node_count == node_history[UNIQUE_ID].nunique()


def test_hierarchical_expansion_estimate_matches_hierarchy_superset_node_history_rows() -> None:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            DS: pd.date_range("2024-01-01", periods=2, freq="D"),
            Y: [1.0, 2.0],
        }
    )
    hierarchy = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B", "C"],
            "dept_id": ["D_A", "D_B", "D_C"],
            "state_id": ["CA", "CA", "CA"],
        }
    )

    node_history = build_node_history(history, build_hierarchy_index(hierarchy))
    estimate = estimate_hierarchical_expansion(history, build_hierarchy_index(hierarchy), horizon=2)
    naive_full_panel_rows = 2 * len(build_summing_matrix(hierarchy).node_labels)

    assert estimate.projected_node_history_rows == len(node_history) == 4
    assert estimate.projected_node_history_rows < naive_full_panel_rows
    assert estimate.projected_node_count == node_history[UNIQUE_ID].nunique() == 2
    assert TOTAL_LABEL not in set(node_history[UNIQUE_ID])


def test_hierarchical_expansion_estimate_rejects_inconsistent_inputs() -> None:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "C"],
            DS: pd.date_range("2024-01-01", periods=2, freq="D"),
            Y: [1.0, 2.0],
        }
    )
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A", "B"], "dept_id": ["D1", "D1"]})

    with pytest.raises(ValueError, match="not present in hierarchy"):
        estimate_hierarchical_expansion(history, build_hierarchy_index(hierarchy), horizon=1)


def test_hierarchical_expansion_estimate_rejects_empty_hierarchy() -> None:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            DS: [pd.Timestamp("2024-01-01")],
            Y: [1.0],
        }
    )
    hierarchy = pd.DataFrame({UNIQUE_ID: [], "dept_id": []})

    with pytest.raises(ValueError, match="hierarchy has no rows"):
        estimate_hierarchical_expansion(history, build_hierarchy_index(hierarchy), horizon=1)


def test_effective_available_memory_uses_lower_cgroup_budget() -> None:
    assert effective_available_memory_bytes(10_000, 4_000) == 4_000
    assert effective_available_memory_bytes(4_000, 10_000) == 4_000
    assert effective_available_memory_bytes(None, 4_000) == 4_000
    assert effective_available_memory_bytes(4_000, None) == 4_000
    assert effective_available_memory_bytes(None, None) is None


def test_read_cgroup_available_memory_bytes_supports_v2(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("10000\n", encoding="utf-8")
    (tmp_path / "memory.current").write_text("2500\n", encoding="utf-8")
    self_cgroup = tmp_path / "self-cgroup"
    self_cgroup.write_text("0::/\n", encoding="utf-8")

    assert (
        read_cgroup_available_memory_bytes(cgroup_root=tmp_path, self_cgroup=self_cgroup) == 7_500
    )


def test_read_cgroup_available_memory_bytes_supports_v1(tmp_path: Path) -> None:
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "memory.limit_in_bytes").write_text("10000\n", encoding="utf-8")
    (memory_root / "memory.usage_in_bytes").write_text("3000\n", encoding="utf-8")
    self_cgroup = tmp_path / "self-cgroup"
    self_cgroup.write_text("7:memory:/\n", encoding="utf-8")

    assert (
        read_cgroup_available_memory_bytes(cgroup_root=tmp_path, self_cgroup=self_cgroup) == 7_000
    )


def test_estimated_node_history_peak_reserves_downstream_overhead() -> None:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "B", "B"],
            DS: pd.date_range("2024-01-01", periods=2, freq="D").tolist() * 2,
            Y: [1.0, 2.0, 3.0, 4.0],
        }
    )
    hierarchy = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            "dept_id": ["D1", "D2"],
            "state_id": ["CA", "CA"],
        }
    )
    hierarchy_index = build_hierarchy_index(hierarchy)
    one_model = estimate_hierarchical_expansion(history, hierarchy_index, horizon=3, model_count=1)
    two_models = estimate_hierarchical_expansion(history, hierarchy_index, horizon=3, model_count=2)

    assert estimated_node_history_peak_bytes(two_models) > estimated_node_history_peak_bytes(
        one_model
    )
    assert (
        estimated_node_history_peak_bytes(one_model) > one_model.projected_node_history_rows * 128
    )


def test_estimated_node_history_peak_bounds_m5_fixture_expansion() -> None:
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "m5", "path": str(_FIXTURE)},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 1,
                    "config": {"backend": "statsforecast", "season_length": 7},
                }
            ],
            "origins": {"start": "2011-01-30", "end": "2011-01-30", "freq": "D"},
            "reconciliation": {"strategy": "bottom_up"},
            "output": {"ledger_path": "unused.parquet", "streaming": False},
            "execution": {"backend": "local", "seed": 42},
        }
    )
    bundle = _load_dataset(config)
    assert bundle.hierarchy is not None

    node_history = build_node_history(bundle.history, build_hierarchy_index(bundle.hierarchy))
    estimate = estimate_hierarchical_expansion(
        bundle.history,
        build_hierarchy_index(bundle.hierarchy),
        horizon=config.tasks[0].horizon,
        model_count=len(config.tasks),
    )

    assert estimate.projected_node_history_rows == len(node_history)
    assert estimate.projected_node_count == node_history[UNIQUE_ID].nunique()
    assert estimated_node_history_peak_bytes(estimate) > node_history.memory_usage(deep=True).sum()


def test_direct_call_estimate_leaves_dense_s_bytes_zero() -> None:
    """estimate_hierarchical_expansion never sets the dense-S term itself; only
    run preparation's densifying branch does (so direct-call peaks are unchanged)."""
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            DS: pd.date_range("2024-01-01", periods=2, freq="D").tolist(),
            Y: [1.0, 2.0],
        }
    )
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A", "B"], "dept_id": ["D", "D"]})

    estimate = estimate_hierarchical_expansion(history, build_hierarchy_index(hierarchy), horizon=1)

    assert estimate.dense_s_bytes == 0
    peak = estimated_node_history_peak_bytes(estimate)
    # The dense-S term is additive: a non-zero term raises the peak.
    densified = replace(estimate, dense_s_bytes=4 * 2 * 8)
    assert estimated_node_history_peak_bytes(densified) == peak + 4 * 2 * 8


def test_run_config_smoke_on_m5_fixture(tmp_path: Path) -> None:
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "m5", "path": str(_FIXTURE)},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 1,
                    "config": {"backend": "statsforecast", "season_length": 7},
                }
            ],
            "origins": {"start": "2011-01-30", "end": "2011-01-30", "freq": "D"},
            "output": {
                "ledger_path": str(tmp_path / "m5-ledger.parquet"),
                "streaming": False,
            },
            "execution": {"backend": "local", "seed": 42},
        }
    )
    result = run_config(config)
    ledger_df = result.ledger.to_df()
    assert not ledger_df.empty
    assert (tmp_path / "m5-ledger.parquet").exists()

    bundle = _load_dataset(config)
    assert bundle.hierarchy is not None
    validate_dataset_bundle(bundle)
    summing = build_summing_matrix(bundle.hierarchy)
    assert set(ledger_df[UNIQUE_ID]) == set(summing.bottom_ids)
    assert TOTAL_LABEL not in set(ledger_df[UNIQUE_ID])


def test_run_config_with_reconciliation_emits_node_level_m5_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "calibre.execution.hierarchy_memory.read_effective_available_memory_bytes", lambda: 2**60
    )
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "m5", "path": str(_FIXTURE)},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 1,
                    "config": {"backend": "statsforecast", "season_length": 7},
                }
            ],
            "origins": {"start": "2011-01-30", "end": "2011-01-30", "freq": "D"},
            "reconciliation": {"strategy": "bottom_up"},
            "output": {
                "ledger_path": str(tmp_path / "m5-node-ledger.parquet"),
                "streaming": False,
            },
            "execution": {"backend": "local", "seed": 42},
        }
    )

    result = run_config(config)
    ledger_df = result.ledger.to_df()

    bundle = _load_dataset(config)
    assert bundle.hierarchy is not None
    summing = build_summing_matrix(bundle.hierarchy)
    assert set(ledger_df[UNIQUE_ID]) == set(summing.node_labels)
    assert "dept_id=HOBBIES_1" in set(summing.node_labels)
    for _, group in ledger_df.groupby([MODEL_NAME, FORECAST_ORIGIN, H], sort=False):
        values = group.set_index(UNIQUE_ID)[Y_HAT]
        bottom = values.reindex(summing.bottom_ids).to_numpy(dtype=np.float64)
        actual = values.reindex(summing.node_labels).to_numpy(dtype=np.float64)
        np.testing.assert_allclose(actual, summing.S @ bottom, rtol=1e-10, atol=1e-10)
    assert (tmp_path / "m5-node-ledger.parquet").exists()


def test_run_config_rejects_oversized_hierarchy_before_build_node_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "m5", "path": str(_FIXTURE)},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 1,
                    "config": {"backend": "statsforecast", "season_length": 7},
                }
            ],
            "origins": {"start": "2011-01-30", "end": "2011-01-30", "freq": "D"},
            "reconciliation": {"strategy": "ols"},
            "output": {
                "ledger_path": str(tmp_path / "m5-node-ledger.parquet"),
                "streaming": False,
            },
            "execution": {"backend": "local", "seed": 42},
        }
    )
    monkeypatch.setattr(
        "calibre.execution.hierarchy_memory.read_effective_available_memory_bytes", lambda: 1
    )

    def fail_build_node_history(*args, **kwargs):
        raise AssertionError("build_node_history should not run after the preflight guard fails")

    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.build_node_history",
        fail_build_node_history,
    )

    with pytest.raises(ValueError, match="projected node-history rows"):
        run_config(config)


def test_run_config_without_reconciliation_skips_hierarchy_memory_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "calibre.execution.hierarchy_memory.read_effective_available_memory_bytes", lambda: 1
    )
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "m5", "path": str(_FIXTURE)},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 1,
                    "config": {"backend": "statsforecast", "season_length": 7},
                }
            ],
            "origins": {"start": "2011-01-30", "end": "2011-01-30", "freq": "D"},
            "output": {
                "ledger_path": str(tmp_path / "m5-ledger.parquet"),
                "streaming": False,
            },
            "execution": {"backend": "local", "seed": 42},
        }
    )

    result = run_config(config)

    assert not result.ledger.to_df().empty
