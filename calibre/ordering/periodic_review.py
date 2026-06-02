from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID, H, validate_forecast_frame
from calibre.core.order_types import (
    INVENTORY_POSITION,
    LEAD_TIME,
    REVIEW_PERIOD,
    CostStruct,
    RsPolicyParameters,
    normalize_rs_policy_parameters,
)
from calibre.ordering.decision_frame import decision_columns
from calibre.ordering.decision_rules import RSArithmetic, UpperBoundRule


def apply_rs_policy(
    frame: pd.DataFrame,
    params: pd.DataFrame | list[RsPolicyParameters],
    coverage: float = 0.9,
    quantile: float | None = None,
) -> pd.DataFrame:
    """Apply a periodic-review order-up-to policy to a forecast frame."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *decision_columns(frame),
                INVENTORY_POSITION,
                LEAD_TIME,
                REVIEW_PERIOD,
                "protection_period",
                "target_stock_level",
                "order_qty",
            ]
        )

    validate_forecast_frame(frame)
    params_frame = normalize_rs_policy_parameters(params)
    merged = frame.copy().merge(params_frame, on=UNIQUE_ID, how="left", validate="many_to_one")
    missing_uids = merged.loc[
        merged[INVENTORY_POSITION].isna(),
        UNIQUE_ID,
    ].drop_duplicates()
    if not missing_uids.empty:
        raise ValueError(f"Missing policy parameters for unique_id values: {missing_uids.tolist()}")

    grouping_columns = decision_columns(merged)
    decision_rule = UpperBoundRule(coverage=coverage, quantile=quantile)
    arithmetic = RSArithmetic()
    outputs: list[dict[str, object]] = []

    for _, group in merged.groupby(grouping_columns, sort=False):
        ordered = group.sort_values(H)
        inventory_position = float(ordered[INVENTORY_POSITION].iloc[0])
        lead_time = int(ordered[LEAD_TIME].iloc[0])
        review_period = int(ordered[REVIEW_PERIOD].iloc[0])
        protection_period = lead_time + review_period
        target_stock_level = decision_rule(ordered, CostStruct())
        order_qty = arithmetic(target_stock_level, inventory_position)

        result = {column: ordered[column].iloc[0] for column in grouping_columns}
        result[INVENTORY_POSITION] = inventory_position
        result[LEAD_TIME] = lead_time
        result[REVIEW_PERIOD] = review_period
        result["protection_period"] = protection_period
        result["target_stock_level"] = target_stock_level
        result["order_qty"] = order_qty
        outputs.append(result)

    return pd.DataFrame(outputs)
