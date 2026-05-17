from __future__ import annotations

import pandas as pd
import pytest

from calibre.core.forecast_frame import DS, IN_STOCK, UNIQUE_ID, Y
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetBundle
from calibre.execution.validation import load_costs, validate_dataset_bundle


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


def test_load_costs_from_csv(tmp_path) -> None:
    path = tmp_path / "costs.csv"
    path.write_text(
        "unique_id,underage_cost,overage_cost,holding_cost,shortage_cost\nA,1.0,0.2,0.2,1.0\n",
        encoding="utf-8",
    )

    costs = load_costs(path)

    assert costs["A"].critical_ratio == pytest.approx(1.0 / 1.2)
