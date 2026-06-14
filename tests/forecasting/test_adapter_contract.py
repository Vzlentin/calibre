"""Tests for the shared forecasting-adapter contract."""

import pandas as pd
import pytest

from calibre.core.forecast_frame import (
    DS,
    FITTED_Y_HAT,
    MODEL_NAME,
    UNIQUE_ID,
    Y,
    validate_fitted_values_frame,
)


def _fitted_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: pd.Series(["A", "A"], dtype="object"),
            DS: pd.to_datetime(["2024-01-01", "2024-01-02"]),
            Y: [1.0, 2.0],
            MODEL_NAME: pd.Series(["m", "m"], dtype="object"),
            FITTED_Y_HAT: [1.1, 1.9],
        }
    )


def test_fitted_values_frame_validates_unique_numeric_sidecar() -> None:
    validate_fitted_values_frame(_fitted_frame())


def test_fitted_values_frame_rejects_duplicate_model_timestamp_keys() -> None:
    frame = pd.concat([_fitted_frame(), _fitted_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate fitted-value rows"):
        validate_fitted_values_frame(frame)


def test_fitted_values_frame_rejects_non_numeric_fitted_values() -> None:
    frame = _fitted_frame()
    frame[FITTED_Y_HAT] = pd.Series(["bad", "worse"], dtype="object")

    with pytest.raises(ValueError, match=FITTED_Y_HAT):
        validate_fitted_values_frame(frame)
