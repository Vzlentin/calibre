"""Protocols describing the ordering-policy interface."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from calibre.core.order_types import CostStruct


class DecisionRule(Protocol):
    """Maps a forecast frame and cost struct to an order-up-to target."""

    def __call__(self, frame: pd.DataFrame, costs: CostStruct) -> float: ...


class OrderingArithmetic(Protocol):
    """Turns a target and inventory position into an order quantity."""

    def __call__(self, target: float, ip: float, *, reorder_point=None) -> float: ...
