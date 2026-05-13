from __future__ import annotations

import pandas as pd

from calibre.contracts.forecast_frame import UNIQUE_ID, validate_forecast_frame
from calibre.order._helpers import _decision_columns
from calibre.order.rules import QuantileInterpolationRule, RSArithmetic
from calibre.order.types import (
    INVENTORY_POSITION,
    OVERAGE_COST,
    UNDERAGE_COST,
    CostStruct,
    NewsvendorPolicyParameters,
    normalize_newsvendor_policy_parameters,
)


def apply_newsvendor_policy(
    frame: pd.DataFrame,
    params: pd.DataFrame | list[NewsvendorPolicyParameters],
    coverage: float = 0.9,
    period: int = 1,
) -> pd.DataFrame:
    """Apply a newsvendor critical-ratio policy to a forecast frame."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *_decision_columns(frame),
                INVENTORY_POSITION,
                UNDERAGE_COST,
                OVERAGE_COST,
                "critical_ratio",
                "demand_quantile",
                "order_qty",
            ]
        )

    validate_forecast_frame(frame)
    params_frame = normalize_newsvendor_policy_parameters(params)
    merged = frame.copy().merge(params_frame, on=UNIQUE_ID, how="left", validate="many_to_one")
    missing_uids = merged.loc[
        merged[INVENTORY_POSITION].isna(),
        UNIQUE_ID,
    ].drop_duplicates()
    if not missing_uids.empty:
        raise ValueError(f"Missing policy parameters for unique_id values: {missing_uids.tolist()}")

    rule = QuantileInterpolationRule(coverage=coverage, period=period)
    arithmetic = RSArithmetic()
    decision_columns = _decision_columns(merged)
    outputs: list[dict[str, object]] = []

    for _, group in merged.groupby(decision_columns, sort=False):
        row = group.iloc[0]
        inventory_position = float(row[INVENTORY_POSITION])
        costs = CostStruct(
            underage_cost=float(row[UNDERAGE_COST]),
            overage_cost=float(row[OVERAGE_COST]),
        )
        demand_quantile = rule(group, costs)
        order_qty = arithmetic(demand_quantile, inventory_position)

        result = {column: row[column] for column in decision_columns}
        result[INVENTORY_POSITION] = inventory_position
        result[UNDERAGE_COST] = costs.underage_cost
        result[OVERAGE_COST] = costs.overage_cost
        result["critical_ratio"] = costs.critical_ratio
        result["demand_quantile"] = demand_quantile
        result["order_qty"] = order_qty
        outputs.append(result)

    return pd.DataFrame(outputs)
