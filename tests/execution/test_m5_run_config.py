from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibre.cli.commands import _estimate_hierarchical_expansion, _load_dataset, run_config
from calibre.cli.config import load_config_from_mapping
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y
from calibre.execution.validation import validate_dataset_bundle
from calibre.reconciliation.summing import TOTAL_LABEL, build_summing_matrix

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

    estimate = _estimate_hierarchical_expansion(
        history,
        hierarchy,
        horizon=3,
        model_count=2,
    )

    assert estimate.bottom_unique_ids == 2
    assert estimate.aggregate_nodes == 4
    assert estimate.node_count == 6
    assert estimate.bottom_rows == 4
    assert estimate.periods_per_bottom == 2
    assert estimate.projected_node_history_rows == 12
    assert estimate.forecast_partitions == 36


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
        _estimate_hierarchical_expansion(history, hierarchy, horizon=1)


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
        _estimate_hierarchical_expansion(history, hierarchy, horizon=1)


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
    monkeypatch.setattr("calibre.cli.commands._read_linux_available_memory_bytes", lambda: 2**60)
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
            "reconciliation": {"strategy": "bottom_up"},
            "output": {
                "ledger_path": str(tmp_path / "m5-node-ledger.parquet"),
                "streaming": False,
            },
            "execution": {"backend": "local", "seed": 42},
        }
    )
    monkeypatch.setattr("calibre.cli.commands._read_linux_available_memory_bytes", lambda: 1)

    def fail_build_node_history(*args, **kwargs):
        raise AssertionError("build_node_history should not run after the preflight guard fails")

    monkeypatch.setattr("calibre.cli.commands.build_node_history", fail_build_node_history)

    with pytest.raises(ValueError, match="projected node-history rows"):
        run_config(config)


def test_run_config_without_reconciliation_skips_hierarchy_memory_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("calibre.cli.commands._read_linux_available_memory_bytes", lambda: 1)
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
