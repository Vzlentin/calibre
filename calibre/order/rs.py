from __future__ import annotations

import pandas as pd

from calibre.contracts.forecast_frame import (
    UNIQUE_ID,
    H,
    quantile_column,
    validate_forecast_frame,
)
from calibre.order._helpers import (
    _decision_columns,
    _validate_interval_columns,
)
from calibre.order.types import (
    INVENTORY_POSITION,
    LEAD_TIME,
    REVIEW_PERIOD,
    RsPolicyParameters,
    normalize_rs_policy_parameters,
)


def apply_rs_policy(
    frame: pd.DataFrame,
    params: pd.DataFrame | list[RsPolicyParameters],
    coverage: float = 0.9,
    quantile: float | None = None,
) -> pd.DataFrame:
    """Apply a periodic-review order-up-to policy to a forecast frame.

    By default the target stock level is the sum of per-horizon conformal
    upper bounds at ``coverage`` over the protection period. When
    ``quantile`` is provided, the target stock level is instead the sum of
    the per-horizon predicted quantile (column ``q_<p>``); this requires
    the forecast frame to carry that quantile column and bypasses the
    conformal interval requirement.
    """
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *_decision_columns(frame),
                INVENTORY_POSITION,
                LEAD_TIME,
                REVIEW_PERIOD,
                "protection_period",
                "target_stock_level",
                "order_qty",
            ]
        )

    validate_forecast_frame(frame)
    if quantile is not None:
        target_col = quantile_column(quantile)
        if target_col not in frame.columns:
            raise ValueError(f"Missing quantile column for quantile={quantile}: {target_col!r}")
    else:
        _, target_col = _validate_interval_columns(frame, coverage)
    params_frame = normalize_rs_policy_parameters(params)

    merged = frame.copy().merge(params_frame, on=UNIQUE_ID, how="left", validate="many_to_one")
    missing_uids = merged.loc[
        merged[INVENTORY_POSITION].isna(),
        UNIQUE_ID,
    ].drop_duplicates()
    if not missing_uids.empty:
        raise ValueError(f"Missing policy parameters for unique_id values: {missing_uids.tolist()}")

    outputs: list[dict[str, object]] = []
    decision_columns = _decision_columns(merged)

    for _, group in merged.groupby(decision_columns, sort=False):
        ordered = group.sort_values(H)
        inventory_position = float(ordered[INVENTORY_POSITION].iloc[0])
        lead_time = int(ordered[LEAD_TIME].iloc[0])
        review_period = int(ordered[REVIEW_PERIOD].iloc[0])
        protection_period = lead_time + review_period
        horizons = ordered[H].astype(int)
        duplicate_horizons = sorted(horizons[horizons.duplicated()].unique().tolist())
        if duplicate_horizons:
            raise ValueError(f"Duplicate horizons for decision group: {duplicate_horizons}")
        required_horizons = set(range(1, protection_period + 1))
        available_horizons = set(horizons.tolist())

        if not required_horizons.issubset(available_horizons):
            missing_horizons = sorted(required_horizons - available_horizons)
            max_horizon = int(horizons.max())
            if max_horizon >= protection_period and missing_horizons:
                raise ValueError(
                    f"Missing horizons within protection period "
                    f"{protection_period}: {missing_horizons}"
                )
            raise ValueError(
                f"Protection period {protection_period} exceeds available horizon {max_horizon}"
            )

        target_stock_level = float(ordered.loc[horizons <= protection_period, target_col].sum())
        result = {column: ordered[column].iloc[0] for column in decision_columns}
        result[INVENTORY_POSITION] = inventory_position
        result[LEAD_TIME] = lead_time
        result[REVIEW_PERIOD] = review_period
        result["protection_period"] = protection_period
        result["target_stock_level"] = target_stock_level
        result["order_qty"] = max(target_stock_level - inventory_position, 0.0)
        outputs.append(result)

    return pd.DataFrame(outputs)
