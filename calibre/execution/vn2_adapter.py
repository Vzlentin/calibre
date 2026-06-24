"""Dataset adapter that loads the VN2 challenge period data into a bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from calibre.core.io import exists, join_uri
from calibre.core.order_types import CostStruct
from calibre.execution.data_loading import load_master, load_period, melt_wide_instock
from calibre.execution.dataset import DatasetAdapter, DatasetBundle
from calibre.execution.dataset_registry import register_dataset_adapter
from calibre.ordering.simulation.state import ProductState, make_pipeline

LEAD_TIME_DEPTH: int = 2

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

        state_path = join_uri(data_dir, f"week_{period}_initial_state.csv")
        if not exists(state_path) and period != 0:
            state_path = join_uri(data_dir, "week_0_initial_state.csv")
        inventory = _load_inventory(state_path) if exists(state_path) else None

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
            inventory=inventory,
        )


def _load_inventory(state_path: str) -> dict[str, ProductState]:
    """Seed per-UID :class:`ProductState` from a VN2 ``week_*_initial_state.csv``.

    Mirrors :func:`benchmarks.vn2.simulator.load_initial_states`: each row's
    ``End Inventory`` becomes on-hand stock and ``[In Transit W+1, In Transit
    W+2]`` the two-slot in-transit pipeline at ``LEAD_TIME_DEPTH``.
    """
    df = pd.read_csv(state_path)
    return {
        f"{int(row['Store'])}_{int(row['Product'])}": ProductState(
            unique_id=f"{int(row['Store'])}_{int(row['Product'])}",
            end_inventory=float(row["End Inventory"]),
            pipeline=make_pipeline(
                [float(row["In Transit W+1"]), float(row["In Transit W+2"])],
                LEAD_TIME_DEPTH,
            ),
        )
        for _, row in df.iterrows()
    }
