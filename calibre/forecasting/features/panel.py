"""Internal helpers shared by the feature transforms."""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID


def _sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (unique_id, ds)-sorted copy with a fresh contiguous index."""
    return df.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
