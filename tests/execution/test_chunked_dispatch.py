"""Universal byte-equality locks for chunked-local dispatch (U2 #162).

Chunking the local scope must be invariant: the ledger a run produces cannot
depend on ``chunk_size`` for ANY backend, because the chunk worker fits each
series independently. This is the #149 byte-equality recipe made universal.

The fixture deliberately mixes TWO local configs over a multi-series panel:

* a statsforecast ``SeasonalNaive`` (per-series estimator within a panel), and
* an ``mlforecast`` LightGBM with a ``future_x`` exogenous column (a *pooled*
  adapter — one model over the stacked panel).

The seasonal group also carries a late-starting series whose history is empty
before the first origin: the chunk worker must skip it at that origin (the
cold-start path) without disturbing its chunk-mates, then fit it at the second
origin — at every chunk size.

Running at ``chunk_size`` 1 / 2 / large and asserting frame equality across all
ledgers proves the invariance for both. The mlforecast lock is the one that
would have caught the rejected panel-route design: had chunks been fed through
the global panel worker, mlforecast would fit ONE pooled model per chunk, so
``chunk_size=1`` (one model per series) and ``chunk_size=large`` (one pooled
model over every series) would diverge. Per-series fitting keeps them identical.
"""

from __future__ import annotations

import pandas as pd
import pytest

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine, ExecutionOptions
from calibre.execution.task_builder import partition_tasks

_SEASONAL_UIDS = ("SN_A", "SN_B", "SN_C")
_ML_UIDS = ("ML_A", "ML_B", "ML_C")
_LATE_UID = "SN_LATE"
_ORIGINS = [pd.Timestamp("2024-06-09"), pd.Timestamp("2024-06-16")]


def _dates() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-07", periods=24, freq="W")


def _seasonal_history(uid: str, phase: float) -> pd.DataFrame:
    dates = _dates()
    pattern = [(10.0 + phase), (20.0 + phase), (30.0 + phase), (40.0 + phase)] * 6
    return pd.DataFrame({UNIQUE_ID: uid, DS: dates, Y: pattern})


def _ml_history(uid: str, phase: float) -> pd.DataFrame:
    dates = _dates()
    base = [(5.0 + phase), (12.0 + phase), (19.0 + phase), (26.0 + phase)] * 6
    promo = [0.0, 1.0] * 12
    return pd.DataFrame({UNIQUE_ID: uid, DS: dates, Y: base, "promo": promo})


def _late_history() -> pd.DataFrame:
    # One observation at the first origin: ``ds < origin`` is empty there (the
    # cold-start skip), and the series fits on this single row at origin 2.
    return pd.DataFrame({UNIQUE_ID: _LATE_UID, DS: [_ORIGINS[0]], Y: [7.5]})


def _ml_future_x(uid: str) -> pd.DataFrame:
    # Two horizons beyond every origin in the fixture window.
    future_dates = pd.date_range("2024-06-16", periods=6, freq="W")
    return pd.DataFrame(
        {
            UNIQUE_ID: uid,
            DS: future_dates,
            "promo": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )


def _tasks() -> list[ForecastTask]:
    seasonal = [
        ForecastTask(
            history=_seasonal_history(uid, phase=float(idx)),
            horizon=2,
            model_config={
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "season_length": 4,
            },
        )
        for idx, uid in enumerate(_SEASONAL_UIDS)
    ]
    seasonal.append(
        ForecastTask(
            history=_late_history(),
            horizon=2,
            model_config={
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "season_length": 4,
            },
        )
    )
    ml = [
        ForecastTask(
            history=_ml_history(uid, phase=float(idx)),
            horizon=2,
            model_config={
                "backend": "mlforecast",
                "model": "lightgbm.LGBMRegressor",
                "lags": [1, 2, 3, 4],
                "static_features": [],
                "n_estimators": 15,
                "num_leaves": 7,
                "min_child_samples": 2,
                "verbosity": -1,
                "n_jobs": 1,
                "random_state": 17,
            },
            future_x=_ml_future_x(uid),
        )
        for idx, uid in enumerate(_ML_UIDS)
    ]
    return [*seasonal, *ml]


def _actuals() -> pd.DataFrame:
    frames = [
        _seasonal_history(uid, phase=float(idx))[[UNIQUE_ID, DS, Y]]
        for idx, uid in enumerate(_SEASONAL_UIDS)
    ]
    frames += [
        _ml_history(uid, phase=float(idx))[[UNIQUE_ID, DS, Y]] for idx, uid in enumerate(_ML_UIDS)
    ]
    frames.append(_late_history())
    return pd.concat(frames, ignore_index=True)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values([UNIQUE_ID, FORECAST_ORIGIN, H]).reset_index(drop=True)


def _run(chunk_size: int) -> pd.DataFrame:
    engine = BackendEngine(execution=ExecutionOptions(chunk_size=chunk_size))
    ledger = engine.execute(partition_tasks(_tasks()), _actuals(), _ORIGINS).ledger.to_df()
    return _sorted(ledger)


def test_chunked_local_output_is_invariant_across_chunk_sizes() -> None:
    """The ledger is byte-identical at chunk_size 1 / 2 / large (one chunk)."""
    pytest.importorskip("mlforecast")

    per_series = _run(chunk_size=1)
    paired = _run(chunk_size=2)
    one_chunk = _run(chunk_size=256)

    pd.testing.assert_frame_equal(paired, per_series)
    pd.testing.assert_frame_equal(one_chunk, per_series)

    # The cold-start path actually fired: the late series is skipped at the
    # first origin (empty pre-origin history) and forecast at the second.
    late = per_series[per_series[UNIQUE_ID] == _LATE_UID]
    assert set(late[FORECAST_ORIGIN]) == {_ORIGINS[1]}


def test_mlforecast_chunking_stays_per_series_not_pooled() -> None:
    """The mlforecast forecasts are identical at chunk_size 1 vs. one-chunk.

    This is the lock that would have caught the rejected panel-route design: a
    pooled fit over a whole chunk would make these diverge. Per-series fitting
    inside the chunk worker keeps them equal regardless of chunk membership.
    """
    pytest.importorskip("mlforecast")

    per_series = _run(chunk_size=1)
    one_chunk = _run(chunk_size=256)

    ml_per_series = per_series[per_series[UNIQUE_ID].isin(_ML_UIDS)].reset_index(drop=True)
    ml_one_chunk = one_chunk[one_chunk[UNIQUE_ID].isin(_ML_UIDS)].reset_index(drop=True)

    assert not ml_per_series.empty
    assert set(ml_per_series[MODEL_NAME].unique()) == {"lightgbm.LGBMRegressor"}
    pd.testing.assert_frame_equal(ml_one_chunk, ml_per_series)
