"""Tests for the run ledger and its I/O."""

import numpy as np
import pandas as pd
import pytest

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.execution.ledger import InMemoryLedger as Ledger


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
    assert df[Y_HAT].tolist() == [10.0, 20.0, 30.0]
    assert df[H].tolist() == [1, 2, 3]


def test_append_multiple():
    ledger = Ledger()
    ledger.append(_make_frame(2, origin="2024-01-01"))
    ledger.append(_make_frame(2, origin="2024-02-01"))
    df = ledger.to_df()
    assert df[Y_HAT].tolist() == [10.0, 20.0, 10.0, 20.0]
    assert sorted(df[FORECAST_ORIGIN].dt.strftime("%Y-%m-%d").unique()) == [
        "2024-01-01",
        "2024-02-01",
    ]


def test_append_validates_schema():
    ledger = Ledger()
    bad_df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        ledger.append(bad_df)


def test_apply_resolutions_keyed_upsert_in_place():
    # Keyed-upsert contract (replaces the old wholesale update_resolved): a due
    # subset is resolved and applied; resolved values land in place at the row's
    # append position and updates-only columns are created NaN-filled.
    ledger = Ledger()
    ledger.append(_make_frame(3))
    due = ledger.due_frame(pd.Timestamp("2024-01-21"))  # all 3 rows due
    assert list(due.index) == [0, 1, 2]  # RangeIndex contract

    due.loc[0, Y] = 11.0
    due["error"] = np.nan
    due.loc[0, "error"] = 1.0
    # Drop the still-pending rows: apply only the resolved one (mix in the frame
    # is also accepted, but this pins resolved-only application).
    ledger.apply_resolutions(due.loc[[0]])

    df = ledger.to_df()
    assert df.loc[0, Y] == 11.0
    assert df.loc[0, "error"] == 1.0
    # Still-pending rows keep NaN y and append order is preserved.
    assert df[Y].isna().tolist() == [False, True, True]
    assert df[Y_HAT].tolist() == [10.0, 20.0, 30.0]


def test_due_frame_filters_by_origin_and_pending():
    ledger = Ledger()
    ledger.append(_make_frame(3))  # ds = 2024-01-07, -14, -21
    due = ledger.due_frame(pd.Timestamp("2024-01-14"))
    assert due[H].tolist() == [1, 2]  # ds > origin excluded
    # Resolve the first row; it leaves the due set on the next call.
    due.loc[0, Y] = 5.0
    ledger.apply_resolutions(due.loc[[0]])
    again = ledger.due_frame(pd.Timestamp("2024-01-14"))
    assert again[H].tolist() == [2]


def test_apply_resolutions_unknown_key_raises():
    ledger = Ledger()
    ledger.append(_make_frame(2))
    rogue = _make_frame(1, origin="2099-01-01")  # never appended
    rogue.loc[0, Y] = 1.0
    with pytest.raises(ValueError, match="not in the open set"):
        ledger.apply_resolutions(rogue)


def test_append_duplicate_key_raises():
    ledger = Ledger()
    ledger.append(_make_frame(2))
    with pytest.raises(ValueError, match="duplicate forecast keys"):
        ledger.append(_make_frame(2))  # identical 5-tuples


def test_apply_resolutions_empty_is_noop():
    ledger = Ledger()
    ledger.append(_make_frame(2))
    before = ledger.to_df()
    ledger.apply_resolutions(ledger.due_frame(pd.Timestamp("2024-01-21")).iloc[0:0])
    pd.testing.assert_frame_equal(ledger.to_df(), before)


def test_inmemory_to_df_byte_identical_to_wholesale_replacement():
    # Order characterization: the keyed in-place update must reproduce, value-
    # and order-identical, what the old wholesale-replacement path produced.
    ledger = Ledger()
    ledger.append(_make_frame(2, origin="2024-01-01"))
    ledger.append(_make_frame(2, origin="2024-02-01"))

    # Reference: the old path resolved the full frame and replaced it wholesale.
    expected = ledger.to_df().copy()
    expected["error"] = np.nan
    for pos, (y_val, err) in {0: (100.0, 5.0), 2: (200.0, 7.0)}.items():
        expected.loc[pos, Y] = y_val
        expected.loc[pos, "error"] = err

    # New path: resolve the same two rows (h==1 of each origin) via the keyed
    # contract and assert to_df() matches the wholesale-replacement frame.
    due = ledger.due_frame(pd.Timestamp("2024-02-29"))
    resolved = due[due[H] == 1].copy()
    resolved["error"] = np.nan
    jan = resolved[FORECAST_ORIGIN] == pd.Timestamp("2024-01-01")
    feb = resolved[FORECAST_ORIGIN] == pd.Timestamp("2024-02-01")
    resolved.loc[jan, [Y, "error"]] = [100.0, 5.0]
    resolved.loc[feb, [Y, "error"]] = [200.0, 7.0]
    ledger.apply_resolutions(resolved)

    pd.testing.assert_frame_equal(ledger.to_df(), expected)


def test_to_parquet(tmp_path):
    ledger = Ledger()
    ledger.append(_make_frame(3))
    path = str(tmp_path / "test.parquet")
    ledger.to_parquet(path)
    loaded = pd.read_parquet(path)
    assert set(REQUIRED_COLUMNS).issubset(set(loaded.columns))
    assert loaded[Y_HAT].tolist() == [10.0, 20.0, 30.0]
    assert loaded[H].tolist() == [1, 2, 3]
