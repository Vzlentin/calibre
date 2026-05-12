from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from calibre.contracts.forecast_frame import Y_HAT
from calibre.metrics import METRICS
from calibre.order.protocols import DecisionRule, OrderingArithmetic
from calibre.order.types import INVENTORY_POSITION, REORDER_POINT, CostStruct


class TuningObjective(Protocol):
    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float: ...


def _metric_callable(metric: str | Callable[[np.ndarray, np.ndarray], float]):
    if isinstance(metric, str):
        try:
            return METRICS[metric]
        except KeyError as err:
            raise ValueError(f"Unknown accuracy metric: {metric!r}") from err
    return metric


@dataclass(frozen=True, slots=True)
class Accuracy:
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "mase"

    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        if Y_HAT not in frame.columns:
            raise ValueError(f"frame must include {Y_HAT!r}")
        actual_arr = actuals.to_numpy(dtype=float)
        pred_arr = frame[Y_HAT].to_numpy(dtype=float)
        valid = np.isfinite(actual_arr) & np.isfinite(pred_arr)
        if not valid.any():
            return float("inf")
        metric_fn = _metric_callable(self.metric)
        return float(metric_fn(actual_arr[valid], pred_arr[valid]))


@dataclass(frozen=True, slots=True)
class Cost:
    decision_rule: DecisionRule
    arithmetic: OrderingArithmetic
    costs: CostStruct

    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        target = float(self.decision_rule(frame, self.costs))
        inventory_position = (
            float(frame[INVENTORY_POSITION].iloc[0]) if INVENTORY_POSITION in frame.columns else 0.0
        )
        reorder_point = (
            float(frame[REORDER_POINT].iloc[0]) if REORDER_POINT in frame.columns else None
        )
        order_qty = float(
            self.arithmetic(target, inventory_position, reorder_point=reorder_point)
        )
        demand = float(actuals.dropna().sum())
        overage = max(order_qty - demand, 0.0) * float(self.costs.overage_cost)
        underage = max(demand - order_qty, 0.0) * float(self.costs.underage_cost)
        return float(overage + underage)


@dataclass(frozen=True, slots=True)
class Pareto:
    decision_rule_fn: Callable[[float], DecisionRule]
    arithmetic: OrderingArithmetic
    costs: CostStruct
    lambda_grid: Iterable[float]
    reduction: Literal["min", "mean"] | Callable[[list[float]], float] = "min"

    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        values = [
            Cost(self.decision_rule_fn(float(weight)), self.arithmetic, self.costs).evaluate(
                frame,
                actuals,
            )
            for weight in self.lambda_grid
        ]
        if not values:
            return float("inf")
        if callable(self.reduction):
            return float(self.reduction(values))
        if self.reduction == "min":
            return float(min(values))
        if self.reduction == "mean":
            return float(np.mean(values))
        raise ValueError(f"Unknown Pareto reduction: {self.reduction!r}")
