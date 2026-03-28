from __future__ import annotations

import pandas as pd

from calibre.contracts.forecast_frame import (
    UNIQUE_ID,
    H,
    validate_forecast_frame,
)
from calibre.order._helpers import (
    _decision_columns,
    _validate_interval_columns,
)
from calibre.order.types import (
    INVENTORY_POSITION,
    OVERAGE_COST,
    UNDERAGE_COST,
    NewsvendorPolicyParameters,
    normalize_newsvendor_policy_parameters,
)


def apply_newsvendor_policy(
    frame: pd.DataFrame,
    params: pd.DataFrame | list[NewsvendorPolicyParameters],
    coverage: float = 0.9,
    period: int = 1,
) -> pd.DataFrame:
    """Apply a newsvendor (critical ratio) policy to a forecast frame.

    The newsvendor problem minimizes expected cost given overage cost (Co) and underage
    cost (Cu). The optimal order quantity is the demand quantile at the critical ratio
    Cu / (Cu + Co).

    With conformal intervals providing lo and hi bounds at the given coverage, the demand
    quantile is approximated by linear interpolation: lo + critical_ratio * (hi - lo).
    This treats the conformal interval as a two-point approximation of the demand
    distribution. The approximation improves at coverage levels close to the true
    demand distribution.

    Args:
        frame: Forecast frame with conformal interval columns.
        params: Policy parameters per unique_id.
        coverage: Conformal interval coverage level.
        period: Horizon step (h value) to use for the newsvendor calculation. Default 1.

    Returns:
        DataFrame with one row per decision group containing order_qty.
    """
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
    lower_col, upper_col = _validate_interval_columns(frame, coverage)
    params_frame = normalize_newsvendor_policy_parameters(params)

    merged = frame.copy().merge(params_frame, on=UNIQUE_ID, how="left", validate="many_to_one")
    missing_uids = merged.loc[
        merged[INVENTORY_POSITION].isna(),
        UNIQUE_ID,
    ].drop_duplicates()
    if not missing_uids.empty:
        raise ValueError(f"Missing policy parameters for unique_id values: {missing_uids.tolist()}")

    period_frame = merged.loc[merged[H] == period]
    if period_frame.empty:
        raise ValueError(f"No rows found for period h={period}")

    outputs: list[dict[str, object]] = []
    decision_columns = _decision_columns(merged)

    for _, group in period_frame.groupby(decision_columns, sort=False):
        row = group.iloc[0]
        inventory_position = float(row[INVENTORY_POSITION])
        underage_cost = float(row[UNDERAGE_COST])
        overage_cost = float(row[OVERAGE_COST])
        critical_ratio = underage_cost / (underage_cost + overage_cost)
        lo = float(row[lower_col])
        hi = float(row[upper_col])
        demand_quantile = lo + critical_ratio * (hi - lo)
        order_qty = max(demand_quantile - inventory_position, 0.0)

        result = {column: row[column] for column in decision_columns}
        result[INVENTORY_POSITION] = inventory_position
        result[UNDERAGE_COST] = underage_cost
        result[OVERAGE_COST] = overage_cost
        result["critical_ratio"] = critical_ratio
        result["demand_quantile"] = demand_quantile
        result["order_qty"] = order_qty
        outputs.append(result)

    return pd.DataFrame(outputs)
