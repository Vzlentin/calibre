from __future__ import annotations

from typing import Protocol

import pandas as pd

from calibre.order.types import CostStruct


class DecisionRule(Protocol):
    def __call__(self, frame: pd.DataFrame, costs: CostStruct) -> float: ...


class OrderingArithmetic(Protocol):
    def __call__(self, target: float, ip: float, *, reorder_point=None) -> float: ...
