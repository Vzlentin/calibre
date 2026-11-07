"""P1, P2, P3, P4: the layout of `Panel`, `Panel.from_frame`, the checks of the constructor,
and `Panel.take`."""

import dataclasses
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from calibre import Panel

from ._panels import Dense, from_panel, make_dense, to_frame, to_panel

IDS = ("s10", "s2", "a", "b7")  # sorted: a, b7, s10, s2
SPANS = [(12, 40), (0, 18), (2, 40), (5, 30)]


def daily(seed: int = 0, censored: bool = False) -> Dense:
    periods = pd.date_range("2024-01-01", periods=40, freq="D")
    rng = np.random.default_rng(seed)
    dense = make_dense(
        rng, IDS, periods, "D", SPANS, coarse={1}, censored_share=0.2 if censored else None
    )
    return dense


def weekly(seed: int = 1) -> Dense:
    periods = pd.date_range("2023-01-02", periods=30, freq="W-MON")
    rng = np.random.default_rng(seed)
    return make_dense(rng, ("w2", "w1", "w3"), periods, "W-MON", [(3, 30), (2, 25), (10, 28)])


def check_layout(panel: Panel) -> None:
    """Every condition of P1."""
    series = list(panel.series)
    assert all(isinstance(s, str) for s in series)
    assert len(set(series)) == len(series)
    assert isinstance(panel.periods, pd.DatetimeIndex)
    expected = pd.date_range(panel.periods[0], periods=len(panel.periods), freq=panel.freq)
    assert panel.periods.equals(expected)
    assert panel.data.dtype == np.float32
    assert panel.indptr.dtype == np.int64
    assert panel.starts.dtype == np.int64
    assert panel.indptr.shape == (len(series) + 1,)
    assert panel.starts.shape == (len(series),)
    assert panel.indptr[0] == 0
    assert np.all(np.diff(panel.indptr) > 0)
    assert panel.indptr[-1] == len(panel.data)
    assert np.all(np.isfinite(panel.data))
    lengths = np.diff(panel.indptr)
    assert np.all(panel.starts >= 0)
    assert np.all(panel.starts + lengths <= len(panel.periods))
    if panel.censored is not None:
        assert panel.censored.dtype == np.bool_
        assert panel.censored.shape == panel.data.shape


def expected_arrays(dense: Dense, order):
    """data, indptr, starts and censored of the sorted panel, from the dense values."""
    rows = [dense.ids.index(s) for s in order]
    finite = np.isfinite(dense.values)
    first = int(np.flatnonzero(finite.any(axis=0))[0])
    data = np.concatenate([dense.values[r][finite[r]] for r in rows]).astype(np.float32)
    indptr = np.concatenate([[0], np.cumsum([finite[r].sum() for r in rows])])
    starts = np.array([dense.starts[r] - first for r in rows])
    censored = None
    if dense.censored is not None:
        censored = np.concatenate([dense.censored[r][finite[r]] for r in rows])
    return data, indptr, starts, censored


@pytest.mark.parametrize("make", [daily, weekly])
def test_from_frame_sorts_and_lays_out(make):
    """P1, P2: from_frame sorts the series by id, keeps each series in time order, lays the
    values out as the ragged array of P1, and builds the calendar from the earliest first
    period of the series to the latest last period."""
    dense = make()
    frame = to_frame(dense)
    frame = frame.iloc[np.random.default_rng(5).permutation(len(frame))].reset_index(drop=True)
    panel = Panel.from_frame(frame, freq=dense.freq)
    check_layout(panel)
    order = sorted(dense.ids)
    assert list(panel.series) == order
    finite = np.isfinite(dense.values).any(axis=0)
    first, last = np.flatnonzero(finite)[[0, -1]]
    assert panel.periods.equals(dense.periods[first : last + 1])
    assert pd.tseries.frequencies.to_offset(panel.freq) == pd.tseries.frequencies.to_offset(
        dense.freq
    )
    data, indptr, starts, _ = expected_arrays(dense, order)
    np.testing.assert_array_equal(panel.data, data)
    np.testing.assert_array_equal(panel.indptr, indptr)
    np.testing.assert_array_equal(panel.starts, starts)
    assert panel.censored is None


def test_from_frame_columns_and_censored():
    """P1, P2: from_frame reads the named columns and the boolean censored column, aligned
    with the data."""
    dense = daily(seed=3, censored=True)
    frame = to_frame(dense, censored_col="flag").rename(
        columns={"unique_id": "item", "ds": "date", "y": "sales"}
    )
    frame = frame.iloc[np.random.default_rng(6).permutation(len(frame))]
    panel = Panel.from_frame(
        frame, freq="D", id_col="item", time_col="date", target_col="sales", censored_col="flag"
    )
    check_layout(panel)
    order = sorted(dense.ids)
    data, indptr, starts, censored = expected_arrays(dense, order)
    np.testing.assert_array_equal(panel.data, data)
    np.testing.assert_array_equal(panel.indptr, indptr)
    np.testing.assert_array_equal(panel.starts, starts)
    np.testing.assert_array_equal(panel.censored, censored)
    assert censored.any() and not censored.all()


def test_from_frame_float64_values():
    """P1, P2: from_frame stores values as float32."""
    dense = daily(seed=4)
    frame = to_frame(dense).astype({"y": np.float64})
    panel = Panel.from_frame(frame, freq="D")
    np.testing.assert_array_equal(from_panel(panel), dense.values[[2, 3, 0, 1]])
    assert panel.data.dtype == np.float32


def bad_frames():
    dense = daily(seed=7, censored=True)
    frame = to_frame(dense, censored_col="cens")
    duplicate = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
    off_hour = frame.copy()
    off_hour.loc[5, "ds"] = off_hour.loc[5, "ds"] + np.timedelta64(12, "h")
    gap = frame.drop(index=frame.index[(frame.unique_id == "a") & (frame.ds == "2024-01-10")])
    nan = frame.copy()
    nan.loc[3, "y"] = np.nan
    inf = frame.copy()
    inf.loc[4, "y"] = np.inf
    as_int = frame.astype({"cens": np.int64})
    as_object = frame.astype({"cens": object})
    as_object.loc[0, "cens"] = None
    return {
        "duplicate id and time": (duplicate, "D"),
        "time off the daily calendar": (off_hour, "D"),
        "gap inside a series": (gap, "D"),
        "NaN value": (nan, "D"),
        "infinite value": (inf, "D"),
        "integer censored column": (as_int, "D"),
        "object censored column": (as_object, "D"),
    }


@pytest.mark.parametrize("case", list(bad_frames()))
def test_from_frame_rejects(case):
    """P2: from_frame rejects a duplicate id and time, a time off the calendar, a gap inside a
    series, a value that is not finite, and a censored column that is not boolean."""
    frame, freq = bad_frames()[case]
    with pytest.raises(ValueError):
        Panel.from_frame(frame, freq=freq, censored_col="cens")


def test_from_frame_rejects_time_off_weekly_calendar():
    """P2: a Tuesday is off the W-MON calendar."""
    frame = to_frame(weekly())
    frame.loc[7, "ds"] = frame.loc[7, "ds"] + np.timedelta64(1, "D")
    with pytest.raises(ValueError):
        Panel.from_frame(frame, freq="W-MON")


def test_from_frame_missing_columns():
    """P2: from_frame raises ValueError when a required default or renamed column is missing."""
    frame = to_frame(daily(seed=5, censored=True), censored_col="cens")
    for col in ("unique_id", "ds", "y"):
        with pytest.raises(ValueError):
            Panel.from_frame(frame.drop(columns=col), freq="D")
    with pytest.raises(ValueError):
        Panel.from_frame(frame.drop(columns="cens"), freq="D", censored_col="cens")
    renamed = frame.rename(
        columns={"unique_id": "item", "ds": "date", "y": "sales", "cens": "flag"}
    )
    for col in ("item", "date", "sales", "flag"):
        with pytest.raises(ValueError):
            Panel.from_frame(
                renamed.drop(columns=col),
                freq="D",
                id_col="item",
                time_col="date",
                target_col="sales",
                censored_col="flag",
            )


def test_from_frame_types():
    """P2: from_frame raises TypeError for a frame that is not a DataFrame, or a freq that is
    not a str."""
    frame = to_frame(daily())
    from_frame: Any = Panel.from_frame  # the arguments below have wrong types on purpose
    with pytest.raises(TypeError):
        from_frame(frame.to_dict("list"), freq="D")
    with pytest.raises(TypeError):
        from_frame(frame, freq=1)


def arrays(dense: Dense) -> dict:
    panel = to_panel(dense)
    return {
        "series": np.array(panel.series),
        "periods": panel.periods,
        "freq": panel.freq,
        "data": np.array(panel.data),
        "indptr": np.array(panel.indptr),
        "starts": np.array(panel.starts),
        "censored": None if panel.censored is None else np.array(panel.censored),
    }


def test_constructor_keeps_order_and_converts():
    """P1, P3: the constructor keeps the order of the series and converts the arrays to the
    stated dtypes."""
    dense = daily(seed=8, censored=True)
    good = arrays(dense)
    panel = Panel(
        series=good["series"],
        periods=good["periods"],
        freq="D",
        data=good["data"].astype(np.float64),
        indptr=good["indptr"].astype(np.int32),
        starts=good["starts"].astype(np.int32),
        censored=good["censored"],
    )
    check_layout(panel)
    assert list(panel.series) == list(IDS)
    np.testing.assert_array_equal(from_panel(panel), dense.values)
    np.testing.assert_array_equal(panel.censored, good["censored"])


def test_constructor_copies_and_freezes():
    """P3: a change of the input arrays after construction does not change the panel, and
    the arrays of the panel are read-only."""
    dense = daily(seed=9, censored=True)
    good = arrays(dense)
    kept = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in good.items()
    }
    panel = Panel(**good)
    good["series"][0] = "zzz"
    good["data"][:] = 1.0
    good["indptr"][1] += 1
    good["starts"][0] += 1
    good["censored"][:] = ~good["censored"]
    for name in ("series", "data", "indptr", "starts", "censored"):
        np.testing.assert_array_equal(getattr(panel, name), kept[name])
        array = getattr(panel, name)
        assert not array.flags.writeable, name
        with pytest.raises(ValueError):
            array[0] = array[1]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, panel).data = kept["data"]


def broken(name: str) -> dict:
    """Valid constructor arguments with one condition of P1 broken."""
    good = arrays(daily(seed=10, censored=True))
    periods = good["periods"]
    if name == "duplicate series":
        good["series"] = np.array(["a", "a", "b", "c"])
    elif name == "calendar with a gap":
        good["periods"] = periods.delete(20)
    elif name == "calendar at another freq":
        good["freq"] = "W-MON"
    elif name == "indptr too short":
        good["indptr"] = good["indptr"][:-1]
    elif name == "indptr not starting at 0":
        good["indptr"] = good["indptr"] + 1
        good["data"] = np.concatenate([[1.0], good["data"]]).astype(np.float32)
        good["censored"] = np.concatenate([[False], good["censored"]])
    elif name == "empty series":
        end = good["indptr"][3]
        good["indptr"][4] = end
        good["data"] = good["data"][:end]
        good["censored"] = good["censored"][:end]
    elif name == "indptr decreasing":
        good["indptr"][1], good["indptr"][2] = good["indptr"][2], good["indptr"][1]
    elif name == "indptr end not the data length":
        good["data"] = np.concatenate([good["data"], [1.0]]).astype(np.float32)
    elif name == "starts too short":
        good["starts"] = good["starts"][:-1]
    elif name == "negative start":
        good["starts"][1] = -1
    elif name == "series past the calendar":
        good["starts"][0] += 1
    elif name == "NaN value":
        good["data"][3] = np.nan
    elif name == "infinite value":
        good["data"][3] = -np.inf
    elif name == "censored of another length":
        good["censored"] = good["censored"][:-1]
    else:
        raise AssertionError(name)
    return good


CASES = [
    "duplicate series",
    "calendar with a gap",
    "calendar at another freq",
    "indptr too short",
    "indptr not starting at 0",
    "empty series",
    "indptr decreasing",
    "indptr end not the data length",
    "starts too short",
    "negative start",
    "series past the calendar",
    "NaN value",
    "infinite value",
    "censored of another length",
]


@pytest.mark.parametrize("case", CASES)
def test_constructor_raises(case):
    """P3: the constructor raises ValueError when a condition of P1 fails."""
    with pytest.raises(ValueError):
        Panel(**broken(case))


def test_constructor_types():
    """P3: the constructor raises TypeError for a freq that is not a str."""
    good = arrays(daily(seed=11))
    good["freq"] = 7
    with pytest.raises(TypeError):
        Panel(**good)


def test_constructor_keeps_a_longer_calendar():
    """P1, P3: a calendar that starts before every series and ends after every series is
    kept as it is."""
    dense = daily(seed=12, censored=True)
    good = arrays(dense)
    periods = good["periods"]
    good["periods"] = pd.date_range(
        periods[0] - np.timedelta64(3, "D"), periods[-1] + np.timedelta64(2, "D"), freq="D"
    )
    good["starts"] = good["starts"] + 3
    panel = Panel(**good)
    check_layout(panel)
    assert panel.periods.equals(good["periods"])
    np.testing.assert_array_equal(panel.starts, dense.starts + 3)
    np.testing.assert_array_equal(from_panel(panel)[:, 3:-2], dense.values)


@pytest.mark.parametrize(
    "index", [[0, 1, 2, 3], [1, 3], [2], [0, 2, 3]], ids=lambda index: "-".join(map(str, index))
)
def test_take(index):
    """P4: take gives the series at the index, in that order, with their values and censored
    values, on the same calendar and freq. A panel with no censored array gives none."""
    dense = daily(seed=13, censored=True)
    panel = to_panel(dense)
    part = panel.take(np.array(index))
    check_layout(part)
    assert list(part.series) == [IDS[i] for i in index]
    assert part.periods.equals(panel.periods)
    assert part.freq == panel.freq
    np.testing.assert_array_equal(part.starts, dense.starts[index])
    np.testing.assert_array_equal(from_panel(part), dense.values[index])
    assert dense.censored is not None and part.censored is not None
    finite = np.isfinite(dense.values)
    for k, i in enumerate(index):
        flags = part.censored[part.indptr[k] : part.indptr[k + 1]]
        np.testing.assert_array_equal(flags, dense.censored[i][finite[i]])
    assert to_panel(daily(seed=13)).take(np.array(index)).censored is None


def test_take_input_conditions():
    """P4: take raises TypeError for an index that is not a numpy array, and ValueError for
    one that does not hold integers, is empty, is not 1-d, holds a position outside 0 to
    S - 1, or does not increase."""
    panel = to_panel(daily(seed=14))
    take: Any = panel.take  # the arguments below have wrong types on purpose
    for index in ([0, 1], (0, 1), range(2)):
        with pytest.raises(TypeError):
            take(index)
    for index in (
        np.array([0.0, 1.0]),
        np.array([True, False, True, False]),
        np.array([], dtype=np.int64),
        np.array([[0, 1]]),
        np.array([0, 4]),
        np.array([-1, 0]),
        np.array([2, 1]),
        np.array([1, 1]),
    ):
        with pytest.raises(ValueError):
            take(index)
