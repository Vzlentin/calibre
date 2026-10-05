import numpy as np
import pandas as pd
import pytest

from calibre.panel import Panel


def test_from_long_pivots_and_sorts():
    frame = pd.DataFrame(
        {
            "series": ["b", "a", "b", "a"],
            "period": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-01", "2020-01-01"]),
            "value": [4.0, 3.0, 2.0, 1.0],
        }
    )
    panel = Panel.from_long(frame, freq="D")
    assert list(panel.series) == ["a", "b"]
    assert panel.values.dtype == np.float32
    np.testing.assert_array_equal(panel.values, [[1.0, 3.0], [2.0, 4.0]])


def test_panel_rejects_missing_cell_and_gap():
    frame = pd.DataFrame(
        {
            "series": ["a", "a", "b"],
            "period": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
            "value": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="rectangular"):
        Panel.from_long(frame, freq="D")
    periods = pd.to_datetime(["2020-01-01", "2020-01-03"])
    with pytest.raises(ValueError, match="complete"):
        Panel(series=np.array(["a"]), periods=periods, values=np.ones((1, 2)), freq="D")


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_panel_rejects_nonfinite_history(value):
    values = np.ones((1, 2))
    values[0, 0] = value
    periods = pd.to_datetime(["2020-01-01", "2020-01-02"])
    with pytest.raises(ValueError, match="finite"):
        Panel(series=np.array(["a"]), periods=periods, values=values, freq="D")


def test_panel_rejects_empty_input_and_is_read_only():
    with pytest.raises(ValueError, match="at least one"):
        Panel(np.array([]), pd.DatetimeIndex([]), np.zeros((0, 0)), "D")
    source = np.ones((1, 2))
    panel = Panel(np.array(["a"]), pd.date_range("2020-01-01", periods=2), source, "D")
    source[0, 0] = np.nan
    assert np.isfinite(panel.values).all()
    with pytest.raises(ValueError):
        panel.values[0, 0] = np.nan
