"""F1, F2, F4, F5, F6: `rolling_forecasts`, `Forecasts`, `Forecasts.from_cv` and `select`."""

import dataclasses
from typing import Any, cast

import numpy as np
import pandas as pd
import psutil
import pytest
from statsforecast import StatsForecast
from statsforecast.models import Naive

from calibre import Forecasts, Panel, Window, rolling_forecasts

from . import _checks, _m2, _reference
from ._engines import LastMeanForecaster, engine
from ._expected import RESERVED, assert_bits, window_bounds
from ._panels import make_dense, to_frame, to_panel

SIMPLE = [
    ("Naive", {}),
    ("SeasonalNaive", {"season_length": 7}),
    ("WindowAverage", {"window_size": 5}),
    ("HistoricAverage", {}),
    ("RandomWalkWithDrift", {}),
]
# AutoETS with a multiplicative error raises on a window that holds a zero or a single value,
# so the engine calls the fallback there.
WITH_FALLBACK = [
    ("AutoETS", {"model": "MNN", "alias": "ets_mnn"}),
    ("SimpleExponentialSmoothing", {"alpha": 0.3}),
]
FALLBACK = ("Naive", {})

# (horizon, step_size, n_windows, input_size): every series is longer than the test size.
SETTINGS = {
    "daily": [(5, 1, 12, None), (7, 3, 6, 20), (1, 2, 4, 5), (3, 1, 1, 1)],
    "weekly": [(3, 1, 10, None), (2, 2, 5, 8), (6, 4, 3, None)],
}


def daily_dense():
    """Series that end on the last period, with late launches, ties and zeros."""
    periods = pd.date_range("2024-01-01", periods=120, freq="D")
    ids = ("s10", "s2", "a", "b7", "c", "z1", "m", "k3", "e")
    starts = [0, 0, 3, 10, 25, 40, 47, 0, 5]
    rng = np.random.default_rng(21)
    return make_dense(rng, ids, periods, "D", [(s, 120) for s in starts], coarse={1, 7})


def weekly_dense():
    periods = pd.date_range("2022-01-03", periods=60, freq="W-MON")
    ids = ("w2", "w1", "w3", "w10", "w0", "w5")
    starts = [0, 2, 9, 20, 0, 31]
    rng = np.random.default_rng(22)
    return make_dense(rng, ids, periods, "W-MON", [(s, 60) for s in starts], coarse={0})


def ragged_dense(seed: int = 23):
    """Late launches, early ends, and a series of three periods."""
    periods = pd.date_range("2024-02-05", periods=70, freq="D")
    ids = ("r1", "r0", "q", "late", "early", "mid", "short", "full")
    spans = [(0, 70), (0, 55), (6, 70), (40, 70), (0, 20), (15, 50), (30, 33), (0, 70)]
    rng = np.random.default_rng(seed)
    return make_dense(rng, ids, periods, "D", spans, coarse={1, 5})


DENSE = {"daily": daily_dense, "weekly": weekly_dense}
JOBS = [
    {
        "frame": name,
        "freq": "D" if name == "daily" else "W-MON",
        "models": models,
        "fallback": fallback,
        "h": h,
        "step_size": step_size,
        "n_windows": n_windows,
        "input_size": input_size,
    }
    for name, settings in SETTINGS.items()
    for h, step_size, n_windows, input_size in settings
    for models, fallback in ((SIMPLE, None), (WITH_FALLBACK, FALLBACK))
]


@pytest.fixture(scope="module")
def cv_frames(tmp_path_factory):
    """The cross-validation frame of each job, computed by statsforecast without calibre."""
    panels = {name: make() for name, make in DENSE.items()}
    return _reference.cv_frames("cv", JOBS, panels, tmp_path_factory.mktemp("cv"))


@pytest.mark.parametrize("index", range(len(JOBS)))
def test_rolling_forecasts_equal_cross_validation(cv_frames, index):
    """F1: with the cutoffs of statsforecast's cross-validation as origins and input_size as
    lookback, rolling_forecasts gives the same points, bit for bit, NaN included, the same
    models, series and origins, and equals Forecasts.from_cv of the frame."""
    job, cv = JOBS[index], cv_frames[index]
    dense = DENSE[job["frame"]]()
    panel = Panel.from_frame(to_frame(dense), freq=job["freq"])
    sf = engine(job["models"], job["freq"], job["fallback"])
    origins = pd.DatetimeIndex(np.unique(cv["cutoff"].to_numpy()))
    got = rolling_forecasts(panel, [sf], origins, job["h"], lookback=job["input_size"])
    _checks.assert_equals_cross_validation(got, cv, panel, job["h"])


def test_cross_validation_jobs_hold_nan_points(cv_frames):
    """F1: the jobs hold NaN points, so the comparison covers them."""
    nan_jobs = [
        k
        for k, cv in enumerate(cv_frames)
        if cv[[c for c in cv.columns if c not in RESERVED]].isna().to_numpy().any()
    ]
    assert nan_jobs


def workers() -> list[psutil.Process]:
    """The child processes of this process, other than the helper processes that
    multiprocessing keeps until its parent exits."""
    found = []
    for child in psutil.Process().children(recursive=True):
        try:
            line = " ".join(child.cmdline())
        except psutil.Error:
            continue
        if (
            "multiprocessing.resource_tracker" not in line
            and "multiprocessing.forkserver" not in line
        ):
            found.append(child)
    return found


def assert_no_worker_left() -> None:
    """The call ended its workers before it returned."""
    assert [(child.pid, child.status()) for child in workers()] == []


@pytest.mark.parametrize("lookback", [None, 12])
def test_points_do_not_depend_on_n_jobs(lookback):
    """F2: the points are the same, bit for bit, with one worker and with a pool, and no
    worker process outlives the call."""
    panel = to_panel(ragged_dense())
    origins = panel.periods[[8, 9, 15, 22, 23, 30, 41, 52, 60]]
    runs = []
    for n_jobs in (1, 3):
        engines = [
            engine(SIMPLE, "D", n_jobs=n_jobs),
            engine(WITH_FALLBACK, "D", FALLBACK, n_jobs=n_jobs),
        ]
        runs.append(rolling_forecasts(panel, engines, origins, 6, lookback=lookback))
        assert_no_worker_left()
    assert_bits(runs[1].points, runs[0].points)
    assert list(runs[1].models) == list(runs[0].models)


def window_points(dense, origin_idx, horizon, lookback, models, fallback, directory):
    """[M, S, H, O] float32: each model on the window of each series inside its span, from
    statsforecast without calibre, and NaN outside the spans."""
    inside, first = window_bounds(dense, origin_idx, lookback)
    cells = [(int(s), int(o)) for s, o in np.argwhere(inside)]
    windows = [
        dense.values[s, first[s, o] : origin_idx[o] + 1].astype(np.float32) for s, o in cells
    ]
    offsets = np.concatenate([[0], np.cumsum([len(w) for w in windows])])
    arrays = {
        "jobs": _reference.text([{"model": m, "fallback": fallback, "h": horizon} for m in models])
    }
    for k in range(len(models)):
        arrays[f"{k}/y"] = np.concatenate(windows)
        arrays[f"{k}/offsets"] = offsets
    out = _reference.run("windows", arrays, directory)
    points = np.full((len(models), len(dense.ids), horizon, len(origin_idx)), np.nan, np.float32)
    for k in range(len(models)):
        means = out[f"{k}/mean"]
        for w, (s, o) in enumerate(cells):
            points[k, s, :, o] = means[w].astype(np.float32)
    return points


@pytest.mark.parametrize("lookback", [None, 9])
def test_points_outside_spans_and_model_nan(tmp_path, lookback):
    """F4: a series whose first period is after o, or whose last period is before o, has NaN
    points at o. A model that gives NaN gives NaN points. Inside its span, each series reads
    its values through o, at most the last `lookback`."""
    dense = ragged_dense(seed=24)
    panel = to_panel(dense)
    origin_idx = np.array([3, 5, 10, 19, 20, 29, 32, 33, 40, 49, 54, 55, 60, 69])
    horizon = 4
    models = [
        ("WindowAverage", {"window_size": 6}),
        ("Naive", {}),
        ("WindowAverage", {"window_size": 6, "alias": "wa6_with_fallback"}),
        *WITH_FALLBACK,
    ]
    expected = np.concatenate(
        [
            window_points(dense, origin_idx, horizon, lookback, models[:2], None, tmp_path / "a"),
            window_points(
                dense, origin_idx, horizon, lookback, models[2:], FALLBACK, tmp_path / "b"
            ),
        ]
    )
    engines = [engine(models[:2], "D"), engine(models[2:], "D", FALLBACK)]
    got = rolling_forecasts(panel, engines, panel.periods[origin_idx], horizon, lookback=lookback)
    assert_bits(got.points, expected)
    inside = (dense.starts[:, None] <= origin_idx) & (origin_idx < dense.ends[:, None])
    assert not inside.all()
    by_cell = np.moveaxis(got.points, 3, 2)  # [M, S, O, H]
    assert np.isnan(by_cell[:, ~inside]).all()
    assert np.isnan(by_cell[0][inside]).any()  # WindowAverage on a short window
    assert np.isfinite(by_cell[1][inside]).all()
    assert np.isnan(by_cell[2][inside]).any()  # a NaN is not an error: no fallback


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_model_error_without_fallback_stops_the_call(n_jobs):
    """F5: a model that raises, in an engine with no fallback model, stops the call with its
    own exception, and no worker process outlives the call."""
    periods = pd.date_range("2024-01-01", periods=40, freq="D")
    rng = np.random.default_rng(25)
    ids = ("z0", "z1", "z2")
    dense = make_dense(rng, ids, periods, "D", [(0, 40)] * 3, coarse={0, 1, 2})
    assert (dense.values == 0).any()
    sf = engine([("AutoETS", {"model": "MNN"})], "D", n_jobs=n_jobs)
    with pytest.raises(ValueError, match="zero values"):
        rolling_forecasts(to_panel(dense), [sf], periods[[12, 20, 33]], 3)
    assert_no_worker_left()


def test_duplicate_alias_raises():
    """F5: two models with the same alias in different engines raise. Rows follow the engines
    and their models, named by alias."""
    panel = to_panel(ragged_dense())
    origins = panel.periods[[30, 40]]
    naive = [("Naive", {})]
    with pytest.raises(ValueError):
        rolling_forecasts(panel, [engine(naive, "D"), engine(naive, "D")], origins, 3)
    renamed = [("HistoricAverage", {"alias": "Naive"})]
    with pytest.raises(ValueError):
        rolling_forecasts(panel, [engine(naive, "D"), engine(renamed, "D")], origins, 3)
    other = [("HistoricAverage", {}), ("Naive", {"alias": "naive2"})]
    got = rolling_forecasts(panel, [engine(naive, "D"), engine(other, "D")], origins, 3)
    assert list(got.models) == ["Naive", "HistoricAverage", "naive2"]
    assert_bits(got.points[0], got.points[2])


def test_engine_freq_must_be_the_panel_freq():
    """F5: an engine whose freq is not the panel freq raises. Freqs compare as pandas
    offsets."""
    daily = to_panel(ragged_dense())
    origins = daily.periods[[30, 40]]
    naive = [("Naive", {})]
    with pytest.raises(ValueError):
        rolling_forecasts(daily, [engine(naive, "W-MON")], origins, 3)
    with pytest.raises(ValueError):
        rolling_forecasts(
            daily, [engine(naive, "D"), engine([("HistoricAverage", {})], "2D")], origins, 3
        )
    same = rolling_forecasts(daily, [engine(naive, "1D")], origins, 3)
    assert same.points.shape == (1, len(daily.series), 3, 2)
    weekly = Panel.from_frame(to_frame(weekly_dense()), freq="W-MON")
    with pytest.raises(ValueError):
        rolling_forecasts(weekly, [engine(naive, "W")], weekly.periods[[30, 40]], 3)


class OtherEngine(StatsForecast):
    """A subclass: its type is not StatsForecast."""


def test_rolling_forecasts_input_conditions():
    """F1: rolling_forecasts raises TypeError for an input of the wrong type, and ValueError
    when origins, horizon, lookback or engines break their conditions."""
    dense = ragged_dense()
    panel = to_panel(dense)
    origins = panel.periods[[30, 40]]
    sf = engine([("Naive", {})], "D")
    call: Any = rolling_forecasts  # the arguments below have wrong types on purpose
    wrong_types = [
        ((to_frame(dense), [sf], origins, 3), {}),
        ((panel, sf, origins, 3), {}),
        ((panel, [object()], origins, 3), {}),
        ((panel, [OtherEngine(models=[Naive()], freq="D")], origins, 3), {}),
        ((panel, [sf], list(origins), 3), {}),
        ((panel, [sf], origins, 3.0), {}),
        ((panel, [sf], origins, 3), {"lookback": 2.5}),
    ]
    for args, kwargs in wrong_types:
        with pytest.raises(TypeError):
            call(*args, **kwargs)
    wrong_values = [
        ((panel, [], origins, 3), {}),
        ((panel, [sf], origins[::-1], 3), {}),
        ((panel, [sf], panel.periods[[30, 30]], 3), {}),
        ((panel, [sf], origins + np.timedelta64(12, "h"), 3), {}),
        ((panel, [sf], pd.DatetimeIndex(["2030-01-01"]), 3), {}),
        ((panel, [sf], origins, 0), {}),
        ((panel, [sf], origins, 3), {"lookback": 0}),
    ]
    for args, kwargs in wrong_values:
        with pytest.raises(ValueError):
            call(*args, **kwargs)


def cv_frame(seed: int = 31) -> tuple[pd.DataFrame, list]:
    """A hand-made frame: shuffled rows, unsorted cutoffs, a series absent at one cutoff,
    interval columns, and model values that float32 rounds."""
    cutoffs = pd.to_datetime(["2024-03-10", "2024-03-03", "2024-03-17"])
    rng = np.random.default_rng(seed)
    rows = []
    for uid in ("a", "b", "c"):
        for cutoff in cutoffs:
            if uid == "c" and cutoff == cutoffs[0]:
                continue
            for h in (1, 2, 3):
                m1, m2 = rng.normal(size=2) * 100
                rows.append(
                    {
                        "unique_id": uid,
                        "ds": cutoff + np.timedelta64(h, "D"),
                        "cutoff": cutoff,
                        "y": float(rng.integers(0, 100)),
                        "m1": m1,
                        "m1-lo-90": m1 - 5,
                        "m1-hi-90": m1 + 5,
                        "m2": m2,
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True), sorted(cutoffs)


def test_from_cv_layout():
    """F6: from_cv puts model m, series s, cutoff o and step h at points[m, s, h - 1, o], as
    float32. The origins are the sorted cutoffs, the rows follow `series`, and a series absent
    at a cutoff has NaN points there. models=None skips the target and interval columns."""
    cv, cutoffs = cv_frame()
    series = np.array(["b", "d", "a", "c"])
    got = Forecasts.from_cv(cv, series=series, horizon=3)
    expected = np.full((2, 4, 3, 3), np.nan, np.float32)
    for row in cv.to_dict("records"):
        s = list(series).index(row["unique_id"])
        o = cutoffs.index(row["cutoff"])
        h = (row["ds"] - row["cutoff"]).days
        expected[0, s, h - 1, o] = np.float32(row["m1"])
        expected[1, s, h - 1, o] = np.float32(row["m2"])
    assert_bits(got.points, expected)
    assert list(got.models) == ["m1", "m2"]
    assert list(got.series) == list(series)
    assert np.array_equal(got.origins.values, pd.DatetimeIndex(cutoffs).values)
    assert np.isnan(got.points[:, 1]).all()
    assert np.isnan(got.points[:, 3, :, 1]).all()  # c has no rows at the second cutoff
    values = cv["m1"].to_numpy()
    assert (values != values.astype(np.float32)).all()  # the cast to float32 rounds


def test_from_cv_columns_and_models():
    """F6: `models` picks the columns in its order, and the column names are arguments."""
    cv, _ = cv_frame(seed=32)
    series = np.array(["a", "b", "c"])
    full = Forecasts.from_cv(cv, series=series, horizon=3)
    picked = Forecasts.from_cv(cv, series=series, horizon=3, models=["m2", "m1"])
    assert list(picked.models) == ["m2", "m1"]
    assert_bits(picked.points, full.points[::-1].copy())
    renamed = cv.rename(columns={"unique_id": "item", "ds": "date", "y": "sales"})
    other = Forecasts.from_cv(
        renamed, series=series, horizon=3, id_col="item", time_col="date", target_col="sales"
    )
    assert list(other.models) == ["m1", "m2"]
    assert_bits(other.points, full.points)


def test_from_cv_input_conditions():
    """F6: from_cv raises TypeError for an input of the wrong type, and ValueError for an id
    that is not in `series`, a series and cutoff without exactly `horizon` rows, a cutoff
    column that is missing or not datetime, and an unknown model."""
    cv, _ = cv_frame(seed=33)
    series = np.array(["a", "b", "c"])
    from_cv: Any = Forecasts.from_cv  # the arguments below have wrong types on purpose
    with pytest.raises(TypeError):
        from_cv(cv.to_dict("list"), series=series, horizon=3)
    with pytest.raises(TypeError):
        from_cv(cv, series=series, horizon=3.0)
    short = cv.drop(index=cv.index[(cv.unique_id == "a")][0])
    duplicated = pd.concat([cv, cv.iloc[[0]]], ignore_index=True)
    as_text = cv.assign(cutoff=cv["cutoff"].astype(str))
    cases = [
        (cv, np.array(["a", "b"]), None),
        (short, series, None),
        (duplicated, series, None),
        (cv.drop(columns="unique_id"), series, None),
        (cv.drop(columns="ds"), series, None),
        (cv.drop(columns="cutoff"), series, None),
        (as_text, series, None),
        (cv, series, ["m3"]),
    ]
    for frame, ids, models in cases:
        with pytest.raises(ValueError):
            Forecasts.from_cv(frame, series=ids, horizon=3, models=models)
    renamed = cv.rename(columns={"unique_id": "item", "ds": "date", "y": "sales"})
    for col in ("item", "date", "cutoff"):
        with pytest.raises(ValueError):
            Forecasts.from_cv(
                renamed.drop(columns=col),
                series=series,
                horizon=3,
                id_col="item",
                time_col="date",
                target_col="sales",
            )


def test_select():
    """F6: select returns one model with M = 1, and raises for an unknown model."""
    cv, _ = cv_frame(seed=34)
    full = Forecasts.from_cv(cv, series=np.array(["a", "b", "c"]), horizon=3)
    one = full.select("m2")
    assert list(one.models) == ["m2"]
    assert_bits(one.points, full.points[1:2].copy())
    assert list(one.series) == list(full.series)
    assert one.origins.equals(full.origins)
    with pytest.raises(ValueError):
        full.select("m3")
    with pytest.raises(TypeError):
        cast(Any, full).select(1)


def test_forecasts_constructor():
    """F6: the constructor converts points to float32, checks that the shapes agree and that
    the origins strictly increase, and the class is frozen."""
    points = np.arange(2 * 3 * 2 * 4, dtype=np.float64).reshape(2, 3, 2, 4) / 3
    models, series = np.array(["m1", "m2"]), np.array(["x", "y", "z"])
    origins = pd.date_range("2024-01-01", periods=4, freq="D")
    made = Forecasts(points=points, models=models, series=series, origins=origins)
    assert made.points.dtype == np.float32
    assert_bits(made.points, points.astype(np.float32))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, made).points = points
    broken = [
        dict(points=points[0]),
        dict(models=models[:1]),
        dict(series=series[:2]),
        dict(origins=origins[:3]),
        dict(origins=origins[[0, 2, 1, 3]]),
        dict(origins=origins[[0, 1, 1, 3]]),
    ]
    good = dict(points=points, models=models, series=series, origins=origins)
    make: Any = Forecasts  # the arguments below break the conditions on purpose
    for change in broken:
        with pytest.raises(ValueError):
            make(**{**good, **change})
    with pytest.raises(TypeError):
        make(**{**good, "origins": list(origins)})


def last_mean_points(dense, origin_idx, horizon: int, lookback: int | None) -> np.ndarray:
    """[1, S, H, O] float32: the float64 mean of the values of each series through each origin,
    at most the last `lookback`, by a direct computation. NaN outside the spans."""
    points = np.full((1, len(dense.ids), horizon, len(origin_idx)), np.nan, np.float32)
    inside, first = window_bounds(dense, origin_idx, lookback)
    for s, o in np.argwhere(inside):
        points[0, s, :, o] = np.float32(dense.values[s, first[s, o] : origin_idx[o] + 1].mean())
    return points


def test_window_and_forecaster():
    """F1, F4, F5: Window.at slices each series through the origin, at most the last
    `lookback` values, and rolling_forecasts runs Forecaster engines alongside StatsForecast
    engines, with the points that each Forecaster computes from its windows."""
    dense = ragged_dense()
    panel = to_panel(dense)
    win = Window.at(panel, 25, lookback=10)
    assert win.origin == 25
    inside, first = window_bounds(dense, [25], 10)
    for s in range(len(dense.ids)):
        if inside[s, 0]:
            offset = panel.indptr[s] - dense.starts[s]
            assert (win.start[s], win.stop[s]) == (offset + first[s, 0], offset + 26)
        else:
            assert win.start[s] == win.stop[s]
    assert win.start.dtype == np.int64 and win.stop.dtype == np.int64
    assert not win.start.flags.writeable and not win.stop.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, win).origin = 0
    make_win: Any = Window
    for bad in (
        {"origin": -1, "start": win.start, "stop": win.stop},
        {"origin": 1, "start": win.start[:2], "stop": win.stop},
        {"origin": 1, "start": win.stop + 1, "stop": win.stop},
    ):
        with pytest.raises(ValueError):
            make_win(**bad)
    origins = panel.periods[[25, 35, 45]]
    sf = engine([("Naive", {})], "D")
    custom = LastMeanForecaster("custom_mean")
    got = rolling_forecasts(panel, [sf, custom], origins, 3, lookback=10)
    assert list(got.models) == ["Naive", "custom_mean"]
    expected = last_mean_points(dense, [25, 35, 45], 3, 10)
    assert np.isnan(expected).any() and np.isfinite(expected).any()
    assert_bits(got.points[1:], expected)
    with pytest.raises(ValueError):
        rolling_forecasts(panel, [sf, LastMeanForecaster("Naive")], origins, 3)


def whole_forecasts(seed: int = 35) -> Forecasts:
    rng = np.random.default_rng(seed)
    points = rng.lognormal(3.0, 2.0, (3, 5, 2, 6)).astype(np.float32)
    points[rng.random(points.shape) < 0.1] = np.nan
    return Forecasts(
        points,
        np.array(["m1", "m2", "m3"]),
        np.array(["a", "b", "c", "d", "e"]),
        pd.date_range("2024-01-07", periods=6, freq="W-SUN"),
    )


def part(whole: Forecasts, models=None, series=None, origins=None) -> Forecasts:
    """The cells of `whole` at the given positions, built with the constructor."""
    m = list(range(len(whole.models))) if models is None else models
    s = list(range(len(whole.series))) if series is None else series
    o = list(range(len(whole.origins))) if origins is None else origins
    points = whole.points[np.ix_(m, s, range(whole.points.shape[2]), o)]
    return Forecasts(points, whole.models[m], whole.series[s], whole.origins[o])


def test_concat_joins_by_label():
    """F8: concat joins parts along the one axis whose labels differ. The origins of the
    result are sorted, and the models and the series follow the order of the parts. Each
    value stays at its labels, bit for bit."""
    whole = whole_forecasts()
    cases = [
        ([part(whole, models=[2]), part(whole, models=[0, 1])], [2, 0, 1], None, None),
        ([part(whole, series=[3, 4]), part(whole, series=[0, 1, 2])], None, [3, 4, 0, 1, 2], None),
        ([part(whole, origins=[3, 4, 5]), part(whole, origins=[0, 1, 2])], None, None, None),
        (
            [part(whole, origins=[0, 3]), part(whole, origins=[5]), part(whole, origins=[1, 2, 4])],
            None,
            None,
            None,
        ),
        ([whole], None, None, None),
    ]
    for parts, models, series, origins in cases:
        got = _m2.concat(parts)
        want = part(whole, models, series, origins)
        assert_bits(got.points, want.points)
        assert list(got.models) == list(want.models)
        assert list(got.series) == list(want.series)
        assert got.origins.equals(want.origins)
    assert _m2.concat((part(whole, origins=[1]), part(whole, origins=[0]))).origins.equals(
        whole.origins[:2]
    )


def test_concat_input_conditions():
    """F8: concat raises TypeError when `forecasts` is not a sequence of Forecasts, and ValueError
    for no part, two values for one cell, parts that differ on more than one axis, and parts
    with other numbers of steps."""
    whole = whole_forecasts(seed=36)
    for parts in (whole, "parts", [whole, whole.points], None):
        with pytest.raises(TypeError):
            _m2.concat(parts)
    shorter = Forecasts(whole.points[:, :, :1], whole.models, whole.series, whole.origins)
    early = part(whole, origins=[0, 1, 2])
    wrong_values = [
        [],
        [whole, whole],
        [early, part(whole, origins=[2, 3])],
        [part(whole, models=[0, 1]), part(whole, models=[1, 2])],
        [part(whole, series=[0, 1]), part(whole, series=[1, 2])],
        [early, part(whole, series=[0, 1], origins=[3, 4])],
        [part(whole, models=[0], origins=[0, 1]), part(whole, models=[1], origins=[2, 3])],
        [early, part(whole, series=[4, 3, 2, 1, 0], origins=[3, 4])],
        [part(whole, models=[0]), part(whole, models=[1]), part(whole, origins=[0])],
        [early, Forecasts(shorter.points[..., 3:], whole.models, whole.series, whole.origins[3:])],
    ]
    for parts in wrong_values:
        with pytest.raises(ValueError):
            _m2.concat(parts)
