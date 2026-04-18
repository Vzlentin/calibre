"""Smoke test for the migrated VN2 winning pipeline.

Verifies the new Calibre-public-API path runs end-to-end on a tiny
synthetic dataset and produces the expected cost columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.vn2.run_winning import run_winning


def _wide_sales(uids: list[str], weeks: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=weeks, freq="W-MON")
    rows: list[dict] = []
    for uid in uids:
        store, product = uid.split("_")
        base = rng.uniform(5, 25)
        row: dict[str, object] = {"Store": int(store), "Product": int(product)}
        for i, d in enumerate(dates):
            seasonal = 3.0 * np.sin(2 * np.pi * i / 52)
            noise = rng.normal(0, 1.0)
            row[d.strftime("%Y-%m-%d")] = max(base + seasonal + noise, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _initial_state_csv(uids: list[str]) -> pd.DataFrame:
    rows = []
    for uid in uids:
        store, product = uid.split("_")
        rows.append(
            {
                "Store": int(store),
                "Product": int(product),
                "End Inventory": 50.0,
                "In Transit W+1": 0.0,
                "In Transit W+2": 0.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_vn2_dir(tmp_path: Path) -> Path:
    """Create a minimal VN2-shaped data directory (week_0 + week_1)."""
    uids = ["1_100", "2_200", "3_300"]
    weeks = 60

    full = _wide_sales(uids, weeks)
    date_cols = [c for c in full.columns if c not in ("Store", "Product")]
    week0 = full[["Store", "Product", *date_cols[:-1]]]
    week1 = full

    week0.to_csv(tmp_path / "week_0_sales.csv", index=False)
    week1.to_csv(tmp_path / "week_1_sales.csv", index=False)
    _initial_state_csv(uids).to_csv(tmp_path / "week_0_initial_state.csv", index=False)

    return tmp_path


def test_run_winning_smoke(synthetic_vn2_dir: Path) -> None:
    summary = run_winning(
        data_dir=synthetic_vn2_dir,
        decision_rounds=1,
        delivery_weeks=0,
        verbose=False,
    )

    assert set(summary.columns) == {
        "unique_id",
        "holding_cost",
        "shortage_cost",
        "total_cost",
    }
    assert len(summary) == 3
    assert (summary["total_cost"] >= 0).all()
