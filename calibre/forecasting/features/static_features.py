"""Static (per-series) categorical feature merging."""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID


def add_static_features(
    df: pd.DataFrame,
    static: pd.DataFrame | None,
    exclude: tuple[str, ...] = ("Store", "Product"),
) -> pd.DataFrame:
    """Merge per-series categorical features keyed on ``unique_id``.

    Any column in ``static`` that is not in ``exclude`` and not ``unique_id``
    is merged in and cast to ``category`` so tree-based models (e.g. LightGBM)
    can use them natively.
    """
    if static is None or static.empty:
        return df

    cat_cols = [c for c in static.columns if c not in exclude and c != UNIQUE_ID]
    if not cat_cols:
        return df

    merge_cols = [UNIQUE_ID] + cat_cols
    static_subset = static[merge_cols].drop_duplicates(subset=[UNIQUE_ID])

    df = df.merge(static_subset, on=UNIQUE_ID, how="left")
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df
