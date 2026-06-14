"""Dataset adapter that loads the M5 competition sales/calendar into a bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from calibre.core.io import exists, join_uri
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetAdapter, DatasetBundle
from calibre.execution.dataset_registry import register_dataset_adapter
from calibre.execution.m5_loading import build_m5_hierarchy, melt_m5_sales


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
        return DatasetBundle(
            history=history,
            future_x=None,
            costs=CostStruct(),
            hierarchy=hierarchy,
            censoring=None,
        )
