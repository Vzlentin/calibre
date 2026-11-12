"""L1, L2, L3, L4: `rolling_forecasts` with an `MLForecast` engine."""

import pickle
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mlforecast import MLForecast

import calibre
from calibre import Panel

from . import _checks, _reference
from ._engines import engine, estimator, ml_engine
from ._expected import assert_bits, layout
from ._panels import make_dense, probe_dense, to_frame, to_panel

# Any, so that the type checker accepts MLForecast objects in `engines` before the M2 code.
rolling_forecasts: Any = calibre.rolling_forecasts
HGB = "ensemble.HistGradientBoostingRegressor"


def hgb(alias: str, **kwargs) -> list:
    """A HistGradientBoostingRegressor spec: it accepts the NaN lag features of a short
    series."""
    return [alias, HGB, {"random_state": 0, "max_iter": 8, "min_samples_leaf": 4, **kwargs}]


def spec(freq: str, models: list, **features) -> dict:
    return {"freq": freq, "models": models, **features}


ROLLING = {"1": [["RollingMean", {"window_size": 3}]]}
TWO = [hgb("HistGradientBoostingRegressor"), hgb("hgb_deep", max_depth=3, learning_rate=0.3)]


def daily_dense():
    """Series that end on the last period, with late launches that miss early cutoffs."""
    periods = pd.date_range("2024-01-01", periods=60, freq="D")
    ids = ("s10", "s2", "a", "b7", "late", "k3")
    starts = [0, 0, 3, 10, 51, 25]
    rng = np.random.default_rng(61)
    return make_dense(rng, ids, periods, "D", [(s, 60) for s in starts], coarse={1})


def weekly_dense():
    periods = pd.date_range("2022-01-03", periods=45, freq="W-MON")
    ids = ("w2", "w1", "w3", "w10", "w0")
    starts = [0, 2, 9, 39, 20]
    rng = np.random.default_rng(62)
    return make_dense(rng, ids, periods, "W-MON", [(s, 45) for s in starts], coarse={0})


def ml_ragged_dense(seed: int = 63):
    """Late launches and early ends. Every series starts before period 6."""
    periods = pd.date_range("2024-02-05", periods=60, freq="D")
    ids = ("r1", "r0", "q", "early", "mid", "full")
    spans = [(0, 60), (0, 45), (2, 60), (0, 20), (5, 38), (1, 60)]
    return make_dense(np.random.default_rng(seed), ids, periods, "D", spans, coarse={1})


DENSE = {"daily": daily_dense, "weekly": weekly_dense}
FREQ = {"daily": "D", "weekly": "W-MON"}

# (frame, engine, h, step_size, n_windows, input_size)
SETTINGS = [
    ("daily", spec("D", TWO, lags=[1, 2, 7], lag_transforms=ROLLING), 5, 1, 6, None),
    ("daily", spec("D", TWO[:1], lags=[1, 7], date_features=["dayofweek"]), 3, 4, 4, 20),
    ("daily", spec("D", TWO[1:], lag_transforms={"2": [["ExpandingMean", {}]]}), 1, 2, 5, 12),
    ("weekly", spec("W-MON", TWO, lags=[1, 4], date_features=["month"]), 2, 3, 5, None),
    ("weekly", spec("W-MON", TWO[:1], lags=[1, 2], lag_transforms=ROLLING), 4, 1, 6, 15),
]
CV_JOBS: list[dict[str, Any]] = [
    {"frame": f, "engine": e, "h": h, "step_size": s, "n_windows": n, "input_size": i}
    for f, e, h, s, n, i in SETTINGS
]
# L2: irregular origins on the daily panel. Each origin is the cutoff of a one-window
# cross-validation on the rows through origin + h, where every series reaches origin + h.
IRREGULAR = np.array([12, 13, 19, 30, 44, 52, 54])
IRREGULAR_H = 4
IRREGULAR_LOOKBACK = [None, 9]
IRREGULAR_SPEC = spec("D", TWO, lags=[1, 3], lag_transforms=ROLLING)
ORIGIN_JOBS: list[dict[str, Any]] = [
    {
        "frame": "daily",
        "engine": IRREGULAR_SPEC,
        "h": IRREGULAR_H,
        "step_size": 1,
        "n_windows": 1,
        "input_size": lookback,
        "last": str(daily_dense().periods[t + IRREGULAR_H]),
    }
    for lookback in IRREGULAR_LOOKBACK
    for t in IRREGULAR
]


@pytest.fixture(scope="module")
def cv_frames(tmp_path_factory):
    """The frame of each job, from mlforecast's cross-validation without calibre."""
    panels = {name: make() for name, make in DENSE.items()}
    jobs = CV_JOBS + ORIGIN_JOBS
    return _reference.cv_frames("mlcv", jobs, panels, tmp_path_factory.mktemp("mlcv"))


def model_names(job: dict) -> list[str]:
    return [alias for alias, _, _ in job["engine"]["models"]]


@pytest.mark.parametrize("index", range(len(CV_JOBS)))
def test_rolling_forecasts_equal_cross_validation(cv_frames, index):
    """L1: with the cutoffs of mlforecast's cross-validation as origins and input_size as
    lookback, rolling_forecasts gives the same points, bit for bit, NaN included, the same
    models, series and origins, and equals Forecasts.from_cv of the frame."""
    job, cv = CV_JOBS[index], cv_frames[index]
    panel = Panel.from_frame(to_frame(DENSE[job["frame"]]()), freq=FREQ[job["frame"]])
    origins = pd.DatetimeIndex(np.unique(cv["cutoff"].to_numpy()))
    assert len(origins) == job["n_windows"]
    mlf = ml_engine(job["engine"])
    got = rolling_forecasts(panel, [mlf], origins, job["h"], lookback=job["input_size"])
    assert _checks.assert_equals_cross_validation(got, cv, panel, job["h"]) == model_names(job)


def test_cross_validation_jobs_hold_nan_points(cv_frames):
    """L1: the jobs hold NaN points, from series absent at early cutoffs, so the comparison
    covers them."""
    held = []
    for job, cv in zip(CV_JOBS, cv_frames[: len(CV_JOBS)], strict=True):
        series = DENSE[job["frame"]]().ids
        cutoffs = np.unique(cv["cutoff"].to_numpy())
        held.append(len(cv) < len(series) * len(cutoffs) * job["h"])
    assert sum(held) >= 2


@pytest.mark.parametrize("lookback", IRREGULAR_LOOKBACK)
def test_irregular_origins(cv_frames, lookback):
    """L2: at origins of any spacing, each origin gives the points of a fit on the rows of
    every series through the origin, at most the last `lookback`, then a prediction of h."""
    dense = daily_dense()
    panel = to_panel(dense)
    origins = panel.periods[IRREGULAR]
    got = rolling_forecasts(
        panel, [ml_engine(IRREGULAR_SPEC)], origins, IRREGULAR_H, lookback=lookback
    )
    models = model_names(ORIGIN_JOBS[0])
    expected = np.full((len(models), len(panel.series), IRREGULAR_H, len(origins)), np.nan)
    expected = expected.astype(np.float32)
    first = len(CV_JOBS) + IRREGULAR_LOOKBACK.index(lookback) * len(IRREGULAR)
    for o in range(len(origins)):
        cv = cv_frames[first + o]
        assert (cv["cutoff"] == origins[o]).all()
        cut = layout(cv, panel.series, panel.periods, models, IRREGULAR_H, origins[o : o + 1])
        expected[..., o] = cut[..., 0]
    assert_bits(got.points, expected)
    assert list(got.models) == models
    assert np.isnan(got.points[:, list(dense.ids).index("late"), :, 0]).all()
    assert np.isfinite(got.points[..., -1]).all()


def fitted_state(mlf: MLForecast) -> list[str]:
    """The attributes that fit sets on an MLForecast object and on its TimeSeries."""
    found = [name for name in ("models_", "cv_models_") if hasattr(mlf, name)]
    return found + [name for name in ("ga", "uids", "last_dates") if hasattr(mlf.ts, name)]


def test_engine_object_is_not_changed(cv_frames):
    """L3: the engine changes no attribute of the MLForecast object it gets. The object passed
    twice gives the same bits, and after the call it cross-validates to the same frame."""
    job, cv = CV_JOBS[0], cv_frames[0]
    df = to_frame(daily_dense())
    panel = Panel.from_frame(df, freq="D")
    origins = pd.DatetimeIndex(np.unique(cv["cutoff"].to_numpy()))
    mlf = ml_engine(job["engine"])
    before = pickle.dumps(mlf)
    assert fitted_state(mlf) == []
    runs = [rolling_forecasts(panel, [mlf], origins, job["h"]) for _ in range(2)]
    assert pickle.dumps(mlf) == before
    assert fitted_state(mlf) == []
    assert_bits(runs[1].points, runs[0].points)
    again = mlf.cross_validation(
        df=df, h=job["h"], n_windows=job["n_windows"], step_size=job["step_size"], refit=True
    )
    assert list(again.columns) == list(cv.columns)
    for column in cv.columns:
        assert np.array_equal(again[column].to_numpy(), cv[column].to_numpy())


PROBE_SPEC = spec("D", TWO[:1], lags=[1, 2, 3], lag_transforms=ROLLING)


@pytest.mark.parametrize("lookback", [None, 10])
def test_points_ignore_later_values(lookback):
    """L4 (F3): no point issued at origin o changes when a value after o changes, or when the
    data after o is removed. The probes run at the second, middle and one-before-last
    origin."""
    run = partial(_checks.rolling, engines=[ml_engine(PROBE_SPEC)], horizon=4, lookback=lookback)
    origin_idx = np.array([10, 13, 20, 27, 28, 36, 47, 55])
    rng = np.random.default_rng(65)
    _checks.assert_ignores_later_values(run, probe_dense(64, 60), origin_idx, rng)


@pytest.mark.parametrize("lookback", [None, 9])
def test_points_outside_spans(lookback):
    """L4 (F4): a series whose first period is after o, or whose last period is before o,
    has NaN points at o. The model gives finite points inside each span."""
    dense = ml_ragged_dense()
    origin_idx = np.array([4, 5, 8, 19, 20, 30, 37, 38, 44, 45, 50, 59])
    got = rolling_forecasts(
        to_panel(dense), [ml_engine(PROBE_SPEC)], dense.periods[origin_idx], 3, lookback=lookback
    )
    inside = (dense.starts[:, None] <= origin_idx) & (origin_idx < dense.ends[:, None])
    assert not inside.all()
    by_cell = np.moveaxis(got.points, 3, 2)  # [M, S, O, H]
    assert np.isnan(by_cell[:, ~inside]).all()
    assert np.isfinite(by_cell[:, inside]).all()


def test_duplicate_alias_raises():
    """L4 (F5): two models with the same alias in two engines raise, for an MLForecast with a
    StatsForecast and for two MLForecast objects. Rows follow the engines and their models."""
    panel = to_panel(ml_ragged_dense())
    origins = panel.periods[[30, 40]]
    naive = engine([("Naive", {})], "D")
    clash = ml_engine(spec("D", [hgb("Naive")], lags=[1]))
    with pytest.raises(ValueError):
        rolling_forecasts(panel, [naive, clash], origins, 3)
    with pytest.raises(ValueError):
        rolling_forecasts(panel, [ml_engine(PROBE_SPEC), ml_engine(PROBE_SPEC)], origins, 3)
    other = ml_engine(spec("D", [hgb("hgb_a"), hgb("hgb_b")], lags=[1]))
    got = rolling_forecasts(panel, [naive, other, ml_engine(PROBE_SPEC)], origins, 3)
    assert list(got.models) == ["Naive", "hgb_a", "hgb_b", "HistGradientBoostingRegressor"]
    assert_bits(got.points[1], got.points[2])


def test_engine_freq_must_be_the_panel_freq():
    """L4 (F5): an MLForecast whose freq is not the panel freq raises. Freqs compare as
    pandas offsets."""
    daily = to_panel(ml_ragged_dense())
    origins = daily.periods[[30, 40]]
    models = [hgb("hgb")]
    for freq in ("W-MON", "2D", "h"):
        with pytest.raises(ValueError):
            rolling_forecasts(daily, [ml_engine(spec(freq, models, lags=[1]))], origins, 3)
    same = rolling_forecasts(daily, [ml_engine(spec("1D", models, lags=[1]))], origins, 3)
    assert same.points.shape == (1, len(daily.series), 3, 2)
    weekly = to_panel(weekly_dense())
    for freq in ("W", "7D"):  # 7D gives the same dates as W-MON, but is another offset
        with pytest.raises(ValueError):
            rolling_forecasts(
                weekly, [ml_engine(spec(freq, models, lags=[1]))], weekly.periods[[30]], 3
            )


def test_model_error_stops_the_call():
    """L4 (F5): a model that raises stops the call with its own exception. LinearRegression
    raises on the NaN lag features of a series shorter than its lags."""
    dense = ml_ragged_dense()
    linear = ["linear", "linear_model.LinearRegression", {}]
    mlf = ml_engine(spec("D", [linear], lags=[1, 7]))
    panel = to_panel(dense)
    ok = rolling_forecasts(panel, [mlf], panel.periods[[20, 30]], 3)
    inside = (dense.starts <= 20) & (dense.ends > 20)
    assert np.isfinite(ok.points[:, inside, :, 0]).all()
    with pytest.raises(ValueError, match="Input X contains NaN"):
        rolling_forecasts(panel, [mlf], panel.periods[[8, 30]], 3)


def test_origin_subsets():
    """L4 (F7): a call on some of the origins gives the columns of those origins, bit for
    bit."""
    panel = to_panel(ml_ragged_dense())
    origins = panel.periods[[6, 9, 17, 25, 33, 41, 50]]
    engines = [ml_engine(PROBE_SPEC), engine([("Naive", {})], "D")]
    columns = ([0, 3, 6], [5], [1, 2, 4])
    _checks.assert_splits(panel, engines, origins, 3, 12, (), columns)


class OtherMLForecast(MLForecast):
    """A subclass: its type is not MLForecast."""


def test_subclass_raises():
    """L4: rolling_forecasts raises TypeError for an engine whose type is a subclass of
    MLForecast."""
    panel = to_panel(ml_ragged_dense())
    alias, path, kwargs = hgb("hgb")
    other = OtherMLForecast(models={alias: estimator(path, kwargs)}, freq="D", lags=[1])
    with pytest.raises(TypeError):
        rolling_forecasts(panel, [other], panel.periods[[30, 40]], 3)
