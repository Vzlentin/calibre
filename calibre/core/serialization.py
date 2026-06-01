"""Frame <-> JSON-record coercion shared by the API boundary and the stores.

These are the single source of truth for turning a DataFrame into JSON-safe rows
(datetimes -> ISO strings, NaN -> None) and back. The API uses them for request
parsing and response shaping; the lifecycle order store uses them to persist and
reconstruct order rows. Keeping one implementation avoids the two paths drifting.
"""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN


def json_safe_records(frame: pd.DataFrame) -> list[dict]:
    """Coerce a frame to JSON-safe rows: datetimes -> ISO strings, NaN -> None."""
    clean = frame.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    clean = clean.astype(object)
    clean[pd.isna(clean)] = None
    return clean.to_dict(orient="records")


def frame_from_records(records: list[dict]) -> pd.DataFrame:
    """Inverse of ``json_safe_records``: rebuild a frame, re-parsing ds/origin."""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    for col in (DS, FORECAST_ORIGIN):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col])
    return frame
