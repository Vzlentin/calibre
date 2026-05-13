from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID, H, validate_forecast_frame
from calibre.core.order_types import (
    INVENTORY_POSITION,
    LEAD_TIME,
    REORDER_POINT,
    REVIEW_PERIOD,
    CostStruct,
    RssPolicyParameters,
    normalize_rss_policy_parameters,
)
from calibre.ordering.decision_frame import _decision_columns
from calibre.ordering.decision_rules import RSSArithmetic, UpperBoundRule


def apply_rss_policy(
    frame: pd.DataFrame,
    params: pd.DataFrame | list[RssPolicyParameters],
    coverage: float = 0.9,
) -> pd.DataFrame:
    """Apply a periodic-review (R,s,S) order-up-to policy to a forecast frame."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *_decision_columns(frame),
                INVENTORY_POSITION,
                REORDER_POINT,
                LEAD_TIME,
                REVIEW_PERIOD,
                "protection_period",
                "target_stock_level",
                "order_qty",
            ]
        )

    validate_forecast_frame(frame)
    params_frame = normalize_rss_policy_parameters(params)
    merged = frame.copy().merge(params_frame, on=UNIQUE_ID, how="left", validate="many_to_one")
    missing_uids = merged.loc[
        merged[INVENTORY_POSITION].isna(),
        UNIQUE_ID,
    ].drop_duplicates()
    if not missing_uids.empty:
        raise ValueError(f"Missing policy parameters for unique_id values: {missing_uids.tolist()}")

    decision_columns = _decision_columns(merged)
    decision_rule = UpperBoundRule(coverage=coverage)
    arithmetic = RSSArithmetic()
    outputs: list[dict[str, object]] = []

    for _, group in merged.groupby(decision_columns, sort=False):
        ordered = group.sort_values(H)
        inventory_position = float(ordered[INVENTORY_POSITION].iloc[0])
        reorder_point = float(ordered[REORDER_POINT].iloc[0])
        lead_time = int(ordered[LEAD_TIME].iloc[0])
        review_period = int(ordered[REVIEW_PERIOD].iloc[0])
        protection_period = lead_time + review_period
        target_stock_level = decision_rule(ordered, CostStruct())
        order_qty = arithmetic(
            target_stock_level,
            inventory_position,
            reorder_point=reorder_point,
        )

        result = {column: ordered[column].iloc[0] for column in decision_columns}
        result[INVENTORY_POSITION] = inventory_position
        result[REORDER_POINT] = reorder_point
        result[LEAD_TIME] = lead_time
        result[REVIEW_PERIOD] = review_period
        result["protection_period"] = protection_period
        result["target_stock_level"] = target_stock_level
        result["order_qty"] = order_qty
        outputs.append(result)

    return pd.DataFrame(outputs)
