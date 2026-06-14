"""Tests for dataset loading and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from calibre.core.forecast_frame import DS, IN_STOCK, UNIQUE_ID, Y
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetBundle
from calibre.execution.validation import load_costs, validate_dataset_bundle
from calibre.execution.vn2_adapter import VN2DatasetAdapter


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "B", "B"],
            DS: pd.to_datetime(["2024-01-01", "2024-01-08"] * 2),
            Y: [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_validate_dataset_bundle_accepts_scalar_costs() -> None:
    bundle = DatasetBundle(
        history=_history(),
        future_x=None,
        costs=CostStruct(),
        hierarchy=None,
        censoring=pd.DataFrame(
            {
                UNIQUE_ID: ["A", "A", "B", "B"],
                DS: pd.to_datetime(["2024-01-01", "2024-01-08"] * 2),
                IN_STOCK: [True, False, True, True],
            }
        ),
    )

    validate_dataset_bundle(bundle)


def test_validate_dataset_bundle_rejects_missing_per_sku_costs() -> None:
    bundle = DatasetBundle(
        history=_history(),
        future_x=None,
        costs={"A": CostStruct()},
        hierarchy=None,
        censoring=None,
    )

    with pytest.raises(ValueError, match="costs missing"):
        validate_dataset_bundle(bundle)


def _hierarchy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            "item_id": ["iA", "iB"],
            "dept_id": ["dA", "dB"],
            "cat_id": ["cA", "cB"],
            "store_id": ["sA", "sB"],
            "state_id": ["stA", "stB"],
        }
    )


def test_validate_dataset_bundle_accepts_covering_hierarchy() -> None:
    bundle = DatasetBundle(
        history=_history(),
        future_x=None,
        costs=CostStruct(),
        hierarchy=_hierarchy(),
        censoring=None,
    )
    validate_dataset_bundle(bundle)


def test_validate_dataset_bundle_accepts_none_hierarchy() -> None:
    bundle = DatasetBundle(
        history=_history(),
        future_x=None,
        costs=CostStruct(),
        hierarchy=None,
        censoring=None,
    )
    validate_dataset_bundle(bundle)


def test_validate_dataset_bundle_rejects_missing_unique_id_column() -> None:
    bad = _hierarchy().drop(columns=[UNIQUE_ID])
    bundle = DatasetBundle(
        history=_history(),
        future_x=None,
        costs=CostStruct(),
        hierarchy=bad,
        censoring=None,
    )
    with pytest.raises(ValueError, match="hierarchy missing required column"):
        validate_dataset_bundle(bundle)


def test_validate_dataset_bundle_rejects_duplicate_hierarchy_rows() -> None:
    hierarchy = pd.concat([_hierarchy(), _hierarchy().iloc[[0]]], ignore_index=True)
    bundle = DatasetBundle(
        history=_history(),
        future_x=None,
        costs=CostStruct(),
        hierarchy=hierarchy,
        censoring=None,
    )
    with pytest.raises(ValueError, match="duplicate unique_id"):
        validate_dataset_bundle(bundle)


def test_validate_dataset_bundle_rejects_history_uid_not_in_hierarchy() -> None:
    hierarchy = _hierarchy().iloc[[0]]
    bundle = DatasetBundle(
        history=_history(),
        future_x=None,
        costs=CostStruct(),
        hierarchy=hierarchy,
        censoring=None,
    )
    with pytest.raises(ValueError, match="hierarchy missing unique_id"):
        validate_dataset_bundle(bundle)


def test_vn2_adapter_bundle_passes_validation(data_dir) -> None:
    bundle = VN2DatasetAdapter().load(data_dir, period=0)
    validate_dataset_bundle(bundle)


def test_load_costs_from_csv(tmp_path) -> None:
    path = tmp_path / "costs.csv"
    path.write_text(
        "unique_id,underage_cost,overage_cost,holding_cost,shortage_cost\nA,1.0,0.2,0.2,1.0\n",
        encoding="utf-8",
    )

    costs = load_costs(path)

    assert costs["A"].critical_ratio == pytest.approx(1.0 / 1.2)
