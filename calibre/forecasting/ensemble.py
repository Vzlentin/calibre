"""Ensemble aggregators for multi-model forecast ledgers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    is_quantile_column,
    validate_forecast_frame,
)

_GROUP_KEYS = [UNIQUE_ID, FORECAST_ORIGIN, DS, H]


def ensemble_median(
    ledger_df: pd.DataFrame,
    name: str = "ensemble_median",
) -> pd.DataFrame:
    """Aggregate a multi-model forecast ledger by taking the median y_hat."""
    if ledger_df.empty:
        result = pd.DataFrame(columns=REQUIRED_COLUMNS)
        result[Y] = result[Y].astype("float64")
        result[Y_HAT] = result[Y_HAT].astype("float64")
        result[H] = result[H].astype("int64")
        return result

    validate_forecast_frame(ledger_df)

    grouped = ledger_df.groupby(_GROUP_KEYS, sort=False)[Y_HAT].median().reset_index()

    grouped[MODEL_NAME] = name
    grouped[Y] = np.nan

    grouped[DS] = pd.to_datetime(grouped[DS])
    grouped[FORECAST_ORIGIN] = pd.to_datetime(grouped[FORECAST_ORIGIN])
    grouped[Y_HAT] = grouped[Y_HAT].astype("float64")
    grouped[Y] = grouped[Y].astype("float64")
    grouped[H] = grouped[H].astype("int64")
    grouped[UNIQUE_ID] = grouped[UNIQUE_ID].astype("object")
    grouped[MODEL_NAME] = grouped[MODEL_NAME].astype("object")

    result = grouped[REQUIRED_COLUMNS].reset_index(drop=True)
    validate_forecast_frame(result)
    return result


def ensemble_weighted(
    frames: list[pd.DataFrame],
    weights: list[float],
    name: str = "ensemble_weighted",
) -> pd.DataFrame:
    """Aggregate a list of forecast frames by weighted linear combination."""
    if not frames:
        result = pd.DataFrame(columns=REQUIRED_COLUMNS)
        result[Y] = result[Y].astype("float64")
        result[Y_HAT] = result[Y_HAT].astype("float64")
        result[H] = result[H].astype("int64")
        return result

    if len(frames) != len(weights):
        raise ValueError(
            f"frames ({len(frames)}) and weights ({len(weights)}) must have the same length"
        )

    weight_sum = sum(weights)
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"weights must sum to 1.0, got {weight_sum}")

    base = frames[0].copy()
    validate_forecast_frame(base)

    for i, df in enumerate(frames[1:], start=1):
        validate_forecast_frame(df)
        if set(df[_GROUP_KEYS].itertuples(index=False, name=None)) != set(
            base[_GROUP_KEYS].itertuples(index=False, name=None)
        ):
            raise ValueError(f"Frame {i} does not share the same group keys as frame 0")

    merged = base
    for i, df in enumerate(frames[1:], start=1):
        merged = merged.merge(df, on=_GROUP_KEYS, suffixes=("", f"__m{i}"))

    result = merged[_GROUP_KEYS].copy()
    result[Y_HAT] = (
        sum(merged[f"{Y_HAT}__m{i}"] * weights[i] for i in range(1, len(frames)))
        + merged[Y_HAT] * weights[0]
    )

    q_bases: set[str] = set()
    for df in frames:
        q_bases.update(c for c in df.columns if is_quantile_column(c))

    for q_base in sorted(q_bases):
        col_values = []
        for i, _df in enumerate(frames):
            col_name = q_base if i == 0 else f"{q_base}__m{i}"
            if col_name in merged.columns:
                col_values.append(merged[col_name] * weights[i])
            else:
                col_values.append(pd.Series(np.nan, index=merged.index) * weights[i])
        result[q_base] = sum(col_values)

    result[MODEL_NAME] = name
    result[Y] = np.nan

    result[DS] = pd.to_datetime(result[DS])
    result[FORECAST_ORIGIN] = pd.to_datetime(result[FORECAST_ORIGIN])
    result[Y_HAT] = result[Y_HAT].astype("float64")
    result[Y] = result[Y].astype("float64")
    result[H] = result[H].astype("int64")
    result[UNIQUE_ID] = result[UNIQUE_ID].astype("object")
    result[MODEL_NAME] = result[MODEL_NAME].astype("object")

    for q_base in q_bases:
        result[q_base] = result[q_base].astype("float64")

    result = result[REQUIRED_COLUMNS + sorted(q_bases)].reset_index(drop=True)
    validate_forecast_frame(result)
    return result


def ensemble_inverse_error(
    frames: list[pd.DataFrame],
    errors: list[float],
    name: str = "ensemble_inverse_error",
) -> pd.DataFrame:
    """Weighted ensemble where weights are inverse errors."""
    if len(frames) != len(errors):
        raise ValueError(
            f"frames ({len(frames)}) and errors ({len(errors)}) must have the same length"
        )

    if any(not np.isfinite(e) or e <= 0 for e in errors):
        raise ValueError(f"all errors must be finite and > 0, got {errors}")

    inv = [1.0 / e for e in errors]
    total = sum(inv)
    weights = [v / total for v in inv]
    return ensemble_weighted(frames, weights, name=name)
