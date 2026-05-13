from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.vn2.simulator import HOLDING_COST_RATE, SHORTAGE_COST_RATE
from calibre.order.types import CostStruct
from calibre.pipeline.dataset import DatasetAdapter, DatasetBundle
from calibre.pipeline.loading import load_master, load_period, melt_wide_instock


class VN2DatasetAdapter(DatasetAdapter):
    def name(self) -> str:
        return "vn2"

    def load(self, path: str | Path, **kwargs: Any) -> DatasetBundle:
        data_dir = Path(path)
        period = int(kwargs.get("period", 0))

        history = load_period(data_dir, period)

        master_path = data_dir / f"week_{period}_master.csv"
        if not master_path.exists() and period != 0:
            master_path = data_dir / "week_0_master.csv"
        hierarchy = load_master(master_path) if master_path.exists() else None

        instock_path = data_dir / f"week_{period}_in_stock.csv"
        if not instock_path.exists() and period != 0:
            instock_path = data_dir / "week_0_in_stock.csv"
        censoring = melt_wide_instock(instock_path) if instock_path.exists() else None

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
