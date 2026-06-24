"""Dataset adapter that loads the M5 competition sales/calendar into a bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID
from calibre.core.io import exists, join_uri
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetAdapter, DatasetBundle
from calibre.execution.dataset_registry import register_dataset_adapter
from calibre.execution.m5_loading import build_m5_hierarchy, melt_m5_sales
from calibre.ordering.simulation.state import ProductState, make_pipeline

_COST_KEYS = ("underage_cost", "overage_cost", "holding_cost", "shortage_cost")


def _resolve_sales_path(data_dir: str, *, phase: str) -> str:
    for candidate in (f"sales_train_{phase}.csv", f"sales_train_{phase}_1.csv"):
        path = join_uri(data_dir, candidate)
        if exists(path):
            return path
    raise FileNotFoundError(
        f"No M5 sales file for phase={phase!r} under {data_dir} (tried sales_train_{phase}.csv)"
    )


@register_dataset_adapter("m5")
class M5DatasetAdapter(DatasetAdapter):
    """Load M5 sales and calendar CSVs into a long-format :class:`DatasetBundle` with hierarchy."""

    def name(self) -> str:
        return "m5"

    def load(self, path: str | Path, **kwargs: Any) -> DatasetBundle:
        data_dir = str(path)
        phase = str(kwargs.get("phase", "evaluation"))
        try:
            sales_path = _resolve_sales_path(data_dir, phase=phase)
        except FileNotFoundError:
            if phase == "evaluation":
                sales_path = _resolve_sales_path(data_dir, phase="validation")
            else:
                raise
        calendar_path = join_uri(data_dir, "calendar.csv")
        if not exists(calendar_path):
            raise FileNotFoundError(f"M5 calendar.csv not found under {data_dir}")

        sales = pd.read_csv(str(sales_path))
        calendar = pd.read_csv(str(calendar_path))
        history = melt_m5_sales(sales, calendar)
        hierarchy = build_m5_hierarchy(sales)
        costs, inventory = _build_costs_and_inventory(history, kwargs)
        return DatasetBundle(
            history=history,
            future_x=None,
            costs=costs,
            hierarchy=hierarchy,
            censoring=None,
            inventory=inventory,
        )


def _build_costs_and_inventory(
    history: pd.DataFrame, kwargs: dict[str, Any]
) -> tuple[CostStruct, dict[str, ProductState] | None]:
    """Resolve the bundle cost struct and optional initial inventory from kwargs.

    With no cost or inventory kwargs the result is byte-identical to today's
    path: a zero ``CostStruct()`` and no inventory. When any cost kwarg is
    supplied a non-zero ``CostStruct`` is built; when any cost OR inventory kwarg
    is supplied an initial ``ProductState`` per history series is seeded, which
    requires ``lead_time >= 1`` so the settle loop's :class:`LostSalesRule`
    can accept positive orders.
    """
    cost_kwargs = {key: float(kwargs[key]) for key in _COST_KEYS if key in kwargs}
    costs = CostStruct(**cost_kwargs) if cost_kwargs else CostStruct()

    has_inventory_kwarg = "initial_inventory" in kwargs or "lead_time" in kwargs
    if not cost_kwargs and not has_inventory_kwarg:
        return costs, None

    lead_time = int(kwargs.get("lead_time", 0))
    if lead_time < 1:
        raise ValueError(
            "ordering-enabled M5 (cost or inventory kwargs supplied) requires lead_time >= 1 so "
            f"the seeded ProductState can accept orders; got lead_time={lead_time}"
        )
    initial_inventory = float(kwargs.get("initial_inventory", 0.0))
    history_uids = history[UNIQUE_ID].astype(str).unique().tolist()
    inventory = {
        uid: ProductState(
            unique_id=uid,
            end_inventory=initial_inventory,
            pipeline=make_pipeline([], lead_time),
        )
        for uid in history_uids
    }
    return costs, inventory
