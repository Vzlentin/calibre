from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import CONFORMAL_MODE, FORECAST_ORIGIN, UNIQUE_ID, Y_HAT, H
from calibre.core.order_types import INVENTORY_POSITION, REORDER_POINT, CostStruct
from calibre.evaluation.point_metrics import METRICS
from calibre.evaluation.regret import compute_regret
from calibre.ordering.policy_protocols import DecisionRule, OrderingArithmetic


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
    mode: Literal["perhorizon", "cumulative"] = field(default="perhorizon", kw_only=True)

    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        mode = self._validated_mode(frame)
        if mode == "cumulative":
            self._validate_single_window(frame)
            return self._evaluate_window(frame, actuals)
        if H not in frame.columns:
            raise ValueError(f"frame must include {H!r} for perhorizon cost evaluation")
        total = 0.0
        for horizon, group in frame.groupby(H, sort=False):
            if len(group) != 1:
                raise ValueError(
                    "perhorizon Cost expects one row per horizon in a single window; "
                    f"h={horizon!r} has {len(group)} rows"
                )
            total += self._evaluate_window(group, actuals.reindex(group.index))
        return float(total)

    def _evaluate_window(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        target = float(self.decision_rule(frame, self.costs))
        inventory_position = (
            float(frame[INVENTORY_POSITION].iloc[0]) if INVENTORY_POSITION in frame.columns else 0.0
        )
        reorder_point = (
            float(frame[REORDER_POINT].iloc[0]) if REORDER_POINT in frame.columns else None
        )
        order_qty = float(self.arithmetic(target, inventory_position, reorder_point=reorder_point))
        demand = float(actuals.dropna().sum())
        overage = max(order_qty - demand, 0.0) * float(self.costs.overage_cost)
        underage = max(demand - order_qty, 0.0) * float(self.costs.underage_cost)
        return float(overage + underage)

    def _validated_mode(self, frame: pd.DataFrame) -> Literal["perhorizon", "cumulative"]:
        if CONFORMAL_MODE not in frame.columns:
            return self.mode
        modes = set(frame[CONFORMAL_MODE].dropna().astype(str).unique())
        if not modes:
            return self.mode
        if len(modes) != 1:
            raise ValueError(f"Cost frame has mixed conformal modes: {sorted(modes)}")
        frame_mode = modes.pop()
        if frame_mode not in {"perhorizon", "cumulative"}:
            raise ValueError(f"Unknown conformal mode in frame: {frame_mode!r}")
        if frame_mode != self.mode:
            raise ValueError(
                f"Cost mode {self.mode!r} disagrees with frame conformal_mode {frame_mode!r}"
            )
        return self.mode

    def _validate_single_window(self, frame: pd.DataFrame) -> None:
        missing = [col for col in (UNIQUE_ID, FORECAST_ORIGIN) if col not in frame.columns]
        if missing:
            raise ValueError(f"cumulative Cost frame missing window key columns: {missing}")
        windows = frame[[UNIQUE_ID, FORECAST_ORIGIN]].drop_duplicates()
        if len(windows) != 1:
            raise ValueError("cumulative Cost expects a single (unique_id, forecast_origin) window")


@dataclass(frozen=True, slots=True)
class Pareto:
    decision_rule_fn: Callable[[float], DecisionRule]
    arithmetic: OrderingArithmetic
    costs: CostStruct
    lambda_grid: Iterable[float]
    reduction: Literal["min", "mean"] | Callable[[list[float]], float] = "min"
    mode: Literal["perhorizon", "cumulative"] = field(default="perhorizon", kw_only=True)

    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        values = [
            Cost(
                self.decision_rule_fn(float(weight)),
                self.arithmetic,
                self.costs,
                mode=self.mode,
            ).evaluate(frame, actuals)
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


@dataclass(frozen=True, slots=True)
class Regret:
    """Positive regret of the realized policy cost against a fixed oracle.

    ``oracle_cost`` is the perfect-foresight benchmark precomputed once
    before the HPO study (e.g. a backtest with ``actuals`` substituted
    for the demand quantile). Each trial only re-evaluates the policy
    via the wrapped ``Cost`` and compares the scalar realized cost
    against ``oracle_cost``.
    """

    decision_rule: DecisionRule
    arithmetic: OrderingArithmetic
    costs: CostStruct
    oracle_cost: float
    mode: Literal["perhorizon", "cumulative"] = field(default="perhorizon", kw_only=True)

    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        realized = Cost(
            self.decision_rule,
            self.arithmetic,
            self.costs,
            mode=self.mode,
        ).evaluate(frame, actuals)
        return compute_regret(
            pd.Series([float(realized)]),
            pd.Series([float(self.oracle_cost)]),
        )
