"""Weighted ensemble aggregators for combining multi-model forecast ledgers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.contracts.forecast_frame import (
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


def ensemble_weighted(
    frames: list[pd.DataFrame],
    weights: list[float],
    name: str = "ensemble_weighted",
) -> pd.DataFrame:
    """Aggregate a list of forecast frames by weighted linear combination.

    All frames must share the same group keys (unique_id, forecast_origin, ds, h)
    and the same column layout.  ``y_hat`` and every ``q_*`` quantile column
    are combined row-wise using the supplied weights.

    Args:
        frames: List of valid forecast-frame DataFrames (one per model).
        weights: List of weights, one per frame. Must sum to 1.
        name: The model_name to assign to the resulting ensemble rows.

    Returns:
        A valid forecast-frame DataFrame with model_name = name and y = NaN.

    Raises:
        ValueError: If frame/weight counts mismatch, weights do not sum to 1,
            or the frames do not share identical group keys.
    """
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

    # Validate and align on a common index
    base = frames[0].copy()
    validate_forecast_frame(base)

    for i, df in enumerate(frames[1:], start=1):
        validate_forecast_frame(df)
        if set(df[_GROUP_KEYS].itertuples(index=False, name=None)) != set(
            base[_GROUP_KEYS].itertuples(index=False, name=None)
        ):
            raise ValueError(
                f"Frame {i} does not share the same group keys as frame 0"
            )

    # Merge all frames on group keys, suffixing columns to keep them distinct.
    merged = base
    for i, df in enumerate(frames[1:], start=1):
        merged = merged.merge(df, on=_GROUP_KEYS, suffixes=("", f"__m{i}"))

    # Weighted combination of y_hat
    result = merged[_GROUP_KEYS].copy()
    result[Y_HAT] = sum(
        merged[f"{Y_HAT}__m{i}"] * weights[i] for i in range(1, len(frames))
    ) + merged[Y_HAT] * weights[0]

    # Quantile columns: each frame may have different q_* columns.
    # We need to ensemble matching quantile columns across frames.
    # Collect all quantile column *bases* (without suffix) present in any frame.
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
                # Frame missing this quantile → treat as NaN (will propagate)
                col_values.append(pd.Series(np.nan, index=merged.index) * weights[i])
        result[q_base] = sum(col_values)

    result[MODEL_NAME] = name
    result[Y] = np.nan

    # Ensure correct dtypes and column order
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
    """Weighted ensemble where weights are inverse errors.

    Computes ``weights = (1 / errors) / sum(1 / errors)`` and delegates to
    :func:`ensemble_weighted`.  Smaller errors receive larger weights.

    Args:
        frames: List of valid forecast-frame DataFrames (one per model).
        errors: List of non-negative error values, one per frame.
        name: The model_name to assign to the resulting ensemble rows.

    Returns:
        A valid forecast-frame DataFrame with model_name = name.

    Raises:
        ValueError: If any error is not finite, <= 0, or if frame/error counts mismatch.
    """
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
