import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    UNIQUE_ID,
    DS,
    Y,
    Y_HAT,
    H,
    FORECAST_ORIGIN,
    MODEL_NAME,
    REQUIRED_COLUMNS,
)
from calibre.engine.ledger import Ledger


def _make_frame(n: int = 3, origin: str = "2024-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001"] * n,
            DS: pd.date_range("2024-01-07", periods=n, freq="W"),
            Y: np.nan,
            Y_HAT: [10.0, 20.0, 30.0][:n],
            H: list(range(1, n + 1)),
            FORECAST_ORIGIN: pd.Timestamp(origin),
            MODEL_NAME: ["SeasonalNaive"] * n,
        }
    )


def test_empty_ledger():
    ledger = Ledger()
    df = ledger.to_df()
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 0


def test_append_and_to_df():
    ledger = Ledger()
    ledger.append(_make_frame(3))
    df = ledger.to_df()
    assert len(df) == 3


def test_append_multiple():
    ledger = Ledger()
    ledger.append(_make_frame(2, origin="2024-01-01"))
    ledger.append(_make_frame(2, origin="2024-02-01"))
    df = ledger.to_df()
    assert len(df) == 4


def test_append_validates_schema():
    ledger = Ledger()
    bad_df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        ledger.append(bad_df)


def test_update_resolved():
    ledger = Ledger()
    ledger.append(_make_frame(3))
    updated = ledger.to_df().copy()
    updated.loc[0, Y] = 11.0
    updated["error"] = np.nan
    updated.loc[0, "error"] = 1.0
    ledger.update_resolved(updated)
    df = ledger.to_df()
    assert df.loc[0, Y] == 11.0
    assert df.loc[0, "error"] == 1.0


def test_to_parquet(tmp_path):
    ledger = Ledger()
    ledger.append(_make_frame(3))
    path = str(tmp_path / "test.parquet")
    ledger.to_parquet(path)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 3
    assert set(REQUIRED_COLUMNS).issubset(set(loaded.columns))
