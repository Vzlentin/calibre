"""Dataset adapter that loads the VN2 challenge period data into a bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from calibre.core.io import exists, join_uri
from calibre.core.order_types import CostStruct
from calibre.execution.data_loading import load_master, load_period, melt_wide_instock
from calibre.execution.dataset import DatasetAdapter, DatasetBundle
from calibre.execution.dataset_registry import register_dataset_adapter

HOLDING_COST_RATE: float = 0.2
SHORTAGE_COST_RATE: float = 1.0


@register_dataset_adapter("vn2")
class VN2DatasetAdapter(DatasetAdapter):
    """Load VN2 period sales, master hierarchy, and in-stock censoring into a bundle."""

    def name(self) -> str:
        return "vn2"

    def load(self, path: str | Path, **kwargs: Any) -> DatasetBundle:
        data_dir = str(path)
        period = int(kwargs.get("period", 0))

        history = load_period(data_dir, period)

        master_path = join_uri(data_dir, f"week_{period}_master.csv")
        if not exists(master_path) and period != 0:
            master_path = join_uri(data_dir, "week_0_master.csv")
        hierarchy = load_master(master_path) if exists(master_path) else None

        instock_path = join_uri(data_dir, f"week_{period}_in_stock.csv")
        if not exists(instock_path) and period != 0:
            instock_path = join_uri(data_dir, "week_0_in_stock.csv")
        censoring = melt_wide_instock(instock_path) if exists(instock_path) else None

        costs = CostStruct(
            underage_cost=SHORTAGE_COST_RATE,
            overage_cost=HOLDING_COST_RATE,
            holding_cost=HOLDING_COST_RATE,
            shortage_cost=SHORTAGE_COST_RATE,
        )
        return DatasetBundle(
            history=history,
            future_x=None,
            costs=costs,
            hierarchy=hierarchy,
            censoring=censoring,
        )
