"""Tests for the VN2 dataset adapter's inventory seeding and censoring carry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from calibre.core.forecast_frame import IN_STOCK, UNIQUE_ID
from calibre.execution.vn2_adapter import LEAD_TIME_DEPTH, VN2DatasetAdapter

DATA_DIR = Path(__file__).parents[2] / "data" / "vn2"


def test_inventory_seeded_per_history_uid() -> None:
    bundle = VN2DatasetAdapter().load(DATA_DIR)

    assert bundle.inventory is not None
    history_uids = set(bundle.history[UNIQUE_ID].astype(str).unique())
    assert set(bundle.inventory) == history_uids


def test_inventory_values_match_state_csv() -> None:
    bundle = VN2DatasetAdapter().load(DATA_DIR)
    assert bundle.inventory is not None

    raw = pd.read_csv(DATA_DIR / "week_0_initial_state.csv")
    row = raw.iloc[0]
    uid = f"{int(row['Store'])}_{int(row['Product'])}"

    state = bundle.inventory[uid]
    assert state.unique_id == uid
    assert state.end_inventory == float(row["End Inventory"])
    assert list(state.pipeline) == [
        float(row["In Transit W+1"]),
        float(row["In Transit W+2"]),
    ]
    assert state.lead_time_depth == LEAD_TIME_DEPTH


def test_absent_state_file_leaves_inventory_none(tmp_path: Path) -> None:
    # Stage only a sales file so load() succeeds but finds no state CSV.
    sales = pd.read_csv(DATA_DIR / "week_0_sales.csv")
    sales.to_csv(tmp_path / "week_0_sales.csv", index=False)

    bundle = VN2DatasetAdapter().load(tmp_path)

    assert bundle.inventory is None


def test_period_state_file_falls_back_to_week_0(tmp_path: Path) -> None:
    # A non-zero period with no period-specific state CSV falls back to week_0.
    pd.read_csv(DATA_DIR / "week_0_sales.csv").to_csv(tmp_path / "week_2_sales.csv", index=False)
    pd.read_csv(DATA_DIR / "week_0_initial_state.csv").to_csv(
        tmp_path / "week_0_initial_state.csv", index=False
    )

    bundle = VN2DatasetAdapter().load(tmp_path, period=2)

    assert bundle.inventory is not None


def test_censoring_carried_as_long_instock_frame() -> None:
    bundle = VN2DatasetAdapter().load(DATA_DIR)

    assert bundle.censoring is not None
    assert not bundle.censoring.empty
    assert bundle.censoring[IN_STOCK].dtype == bool


def test_forecast_only_output_invariant_to_inventory_seeding(tmp_path: Path) -> None:
    """Seeding inventory must not change the loaded history (forecast-only stays byte-identical)."""
    with_inventory = VN2DatasetAdapter().load(DATA_DIR)

    # A data dir without the state CSV yields inventory=None but identical history.
    for name in ("week_0_sales.csv", "week_0_in_stock.csv", "week_0_master.csv"):
        src = DATA_DIR / name
        if src.exists():
            pd.read_csv(src).to_csv(tmp_path / name, index=False)
    without_inventory = VN2DatasetAdapter().load(tmp_path)

    assert with_inventory.inventory is not None
    assert without_inventory.inventory is None
    pd.testing.assert_frame_equal(with_inventory.history, without_inventory.history)
