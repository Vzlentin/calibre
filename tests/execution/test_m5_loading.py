from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.execution.m5_loading import build_m5_hierarchy, melt_m5_sales
from calibre.execution.task_builder import build_node_history
from calibre.reconciliation.summing import TOTAL_LABEL, build_summing_matrix

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m5"


@pytest.fixture
def fixture_dir() -> Path:
    return _FIXTURE


def test_melt_m5_sales_shapes_and_values(fixture_dir: Path) -> None:
    sales = pd.read_csv(fixture_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(fixture_dir / "calendar.csv")
    history = melt_m5_sales(sales, calendar)
    assert list(history.columns) == [UNIQUE_ID, DS, Y]
    assert len(history) == 12
    assert history[Y].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(history[DS])

    expected_ids = {
        "HOBBIES_1_001_CA_1",
        "HOBBIES_1_001_CA_2",
        "HOBBIES_2_001_TX_1",
        "HOBBIES_2_001_TX_2",
    }
    assert set(history[UNIQUE_ID]) == expected_ids

    sample = history[
        (history[UNIQUE_ID] == "HOBBIES_1_001_CA_1") & (history[DS] == pd.Timestamp("2011-01-29"))
    ]
    assert sample[Y].iloc[0] == pytest.approx(1.0)


def test_build_m5_hierarchy_one_row_per_series(fixture_dir: Path) -> None:
    sales = pd.read_csv(fixture_dir / "sales_train_evaluation.csv")
    hierarchy = build_m5_hierarchy(sales)
    assert len(hierarchy) == 4
    assert hierarchy[UNIQUE_ID].is_unique
    assert list(hierarchy.columns) == [
        UNIQUE_ID,
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]
    row = hierarchy.loc[hierarchy[UNIQUE_ID] == "HOBBIES_1_001_CA_1"].iloc[0]
    assert row["item_id"] == "HOBBIES_1_001"
    assert row["store_id"] == "CA_1"
    assert row["state_id"] == "CA"


def test_m5_node_history_contains_expected_aggregate_labels(fixture_dir: Path) -> None:
    sales = pd.read_csv(fixture_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(fixture_dir / "calendar.csv")
    history = melt_m5_sales(sales, calendar)
    hierarchy = build_m5_hierarchy(sales)
    node_history = build_node_history(history, hierarchy)
    summing = build_summing_matrix(hierarchy)

    assert node_history[UNIQUE_ID].unique().tolist() == list(summing.node_labels)
    assert "dept_id=HOBBIES_1" in summing.node_labels
    assert "store_id=CA_1" in summing.node_labels
    assert TOTAL_LABEL in summing.node_labels

    first_day = history[DS].min()
    bottom_total = history.loc[history[DS] == first_day, Y].sum()
    aggregate_total = node_history.loc[
        (node_history[UNIQUE_ID] == TOTAL_LABEL) & (node_history[DS] == first_day), Y
    ].iloc[0]
    assert aggregate_total == pytest.approx(bottom_total)
