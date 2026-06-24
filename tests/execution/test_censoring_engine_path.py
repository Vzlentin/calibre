"""Censoring reaches the gated fit on the engine staging path (U3·A4).

Slice A wired the censoring-aware fit *fit-side only*; Slice B carries
``ForecastTask.censoring`` across the staging boundary so the gated mlforecast
fit fires on the real engine path. The primary path is GLOBAL (VN2's winning
config is ``scope: global``): ``ForecastTask.to_uri`` writes a censoring parquet,
``ForecastTaskRef.materialize`` reads it back, and ``_process_global_panel``
concatenates it alongside the histories. This proves:

* ``build_tasks(censoring=...)`` attaches the per-task censoring slice (global
  task carries the whole panel; local task carries only its own uid);
* the global ref path round-trips censoring through staging, so with the gate ON
  the frame handed to mlforecast's ``.fit`` carries the uncensored-imputed
  demand, and with the gate OFF (the default) the fit target is the raw ``y``;
* the chunk (local) path slices censoring to the chunk's uids.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from calibre.core.forecast_frame import DS, IN_STOCK, UNIQUE_ID, Y
from calibre.core.forecast_task import ChunkTaskRef, ForecastTask, stage_local_chunk
from calibre.execution.prediction import _process_global_panel, _process_local_chunk
from calibre.execution.task_builder import build_tasks
from calibre.forecasting.features import add_stockout_features

_GLOBAL_CONFIG = {
    "backend": "mlforecast",
    "scope": "global",
    "name": "global_lgbm",
    "model": "lightgbm.LGBMRegressor",
    "lags": [1],
    "n_estimators": 1,
    "verbosity": -1,
}


_OOS_WEEK = pd.Timestamp("2024-01-22")


def _panel() -> pd.DataFrame:
    """A two-series panel where series A's OOS week is depressed (censored low).

    Series A holds a steady demand of ~10/week, but the ``_OOS_WEEK`` row reads
    ``1.0`` — an out-of-stock truncation. ``add_stockout_features`` lifts that
    week toward the rolling in-stock median, so the imputed value exceeds the raw
    one and the gate's effect is observable.
    """
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29", "2024-02-05"]
    )
    rows = []
    for uid, base in (("A", 10.0), ("B", 5.0)):
        for ds in dates:
            value = 1.0 if (uid == "A" and ds == _OOS_WEEK) else base
            rows.append({UNIQUE_ID: uid, DS: ds, Y: value})
    return pd.DataFrame(rows)


def _censoring(panel: pd.DataFrame) -> pd.DataFrame:
    """In-stock panel marking ``_OOS_WEEK`` out of stock for series A."""
    cens = panel[[UNIQUE_ID, DS]].copy()
    cens[IN_STOCK] = True
    oos = (cens[UNIQUE_ID] == "A") & (cens[DS] == _OOS_WEEK)
    cens.loc[oos, IN_STOCK] = False
    return cens


def test_build_tasks_attaches_global_panel_censoring() -> None:
    panel = _panel()
    cens = _censoring(panel)
    groups = build_tasks(panel, [_GLOBAL_CONFIG], horizon=2, censoring=cens)

    assert len(groups.global_) == 1
    task = groups.global_[0]
    assert task.censoring is not None
    # The global task sees the whole panel, so it carries every series' censoring.
    assert set(task.censoring[UNIQUE_ID].astype(str)) == {"A", "B"}
    assert IN_STOCK in task.censoring.columns


def test_build_tasks_slices_local_censoring_to_uid() -> None:
    panel = _panel()
    cens = _censoring(panel)
    local_config = {**_GLOBAL_CONFIG, "scope": "local", "backend": "statsforecast"}
    local_config["model"] = "SeasonalNaive"
    local_config["season_length"] = 2
    groups = build_tasks(panel, [local_config], horizon=2, censoring=cens)

    assert len(groups.local) == 2
    for task in groups.local:
        assert task.censoring is not None
        # Each local task carries only its own series' censoring rows.
        assert set(task.censoring[UNIQUE_ID].astype(str)) == {task.unique_id}


def _capture_fit_df(monkeypatch) -> dict[str, pd.DataFrame]:
    captured: dict[str, pd.DataFrame] = {}
    mock_instance = MagicMock()

    def _record_fit(df, **_kwargs):
        captured["fit_df"] = df

    mock_instance.fit.side_effect = _record_fit
    # predict returns an empty-ish frame; the test only inspects the fit target.
    mock_instance.predict.return_value = pd.DataFrame({UNIQUE_ID: [], DS: [], "global_lgbm": []})
    monkeypatch.setattr(
        "calibre.forecasting.mlforecast_adapter.MLForecast",
        MagicMock(return_value=mock_instance),
    )
    return captured


def test_global_ref_path_carries_censoring_to_gated_fit(tmp_path: Path, monkeypatch) -> None:
    """With the gate ON, the global-ref fit target is the uncensored-imputed y."""
    panel = _panel()
    cens = _censoring(panel)
    origin = pd.Timestamp("2024-02-12")

    groups = build_tasks(
        panel, [{**_GLOBAL_CONFIG, "censoring_fit": True}], horizon=2, censoring=cens
    )
    task = groups.global_[0]
    ref = task.to_uri(str(tmp_path / "stage"))
    # Censoring survived staging: the ref points at a real parquet.
    assert ref.censoring_uri is not None

    captured = _capture_fit_df(monkeypatch)
    _process_global_panel([ref], dict(task.model_config), origin, collect_fitted_values=False)

    fit_df = captured["fit_df"]
    # The fit target equals add_stockout_features' y_uncensored over the
    # ds < origin window — proving censoring reached the adapter, not raw y.
    window = panel[panel[DS] < origin]
    expected = (
        add_stockout_features(window, cens)[[UNIQUE_ID, DS, "y_uncensored"]]
        .rename(columns={"y_uncensored": Y})
        .sort_values([UNIQUE_ID, DS])
        .reset_index(drop=True)
    )
    expected[Y] = expected[Y].astype("float32")
    actual = fit_df[[UNIQUE_ID, DS, Y]].sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    # The OOS week's imputed demand is lifted above the raw censored value.
    raw = window.set_index([UNIQUE_ID, DS])[Y]
    imputed = actual.set_index([UNIQUE_ID, DS])[Y]
    key = ("A", _OOS_WEEK)
    assert float(imputed.loc[key]) > float(raw.loc[key])


def test_global_ref_path_gate_off_uses_raw_y(tmp_path: Path, monkeypatch) -> None:
    """With the gate OFF (default), the fit target is the raw censored y."""
    panel = _panel()
    cens = _censoring(panel)
    origin = pd.Timestamp("2024-02-12")

    groups = build_tasks(panel, [_GLOBAL_CONFIG], horizon=2, censoring=cens)
    task = groups.global_[0]
    ref = task.to_uri(str(tmp_path / "stage"))

    captured = _capture_fit_df(monkeypatch)
    _process_global_panel([ref], dict(task.model_config), origin, collect_fitted_values=False)

    fit_df = captured["fit_df"]
    window = panel[panel[DS] < origin]
    expected = window[[UNIQUE_ID, DS, Y]].sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    expected[Y] = expected[Y].astype("float32")
    actual = fit_df[[UNIQUE_ID, DS, Y]].sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_chunk_path_slices_censoring_to_chunk_uids(tmp_path: Path) -> None:
    """The local chunk path stages censoring filtered to the chunk's uids."""
    panel = _panel()
    cens = _censoring(panel)
    tasks = [
        ForecastTask(
            history=panel[panel[UNIQUE_ID] == uid].reset_index(drop=True),
            horizon=2,
            model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 2},
            censoring=cens[cens[UNIQUE_ID] == uid].reset_index(drop=True),
        )
        for uid in ("A", "B")
    ]
    # Stage only series A's task into a chunk; the chunk censoring must hold A only.
    chunk_ref = stage_local_chunk(
        [tasks[0]],
        str(tmp_path / "chunk"),
        horizon=2,
        model_config=tasks[0].model_config,
        task_group=None,
    )
    assert isinstance(chunk_ref, ChunkTaskRef)
    assert chunk_ref.censoring_uri is not None
    staged = pd.read_parquet(chunk_ref.censoring_uri)
    assert set(staged[UNIQUE_ID].astype(str)) == {"A"}


def test_process_local_chunk_passes_sliced_censoring(tmp_path: Path, monkeypatch) -> None:
    """The chunk worker hands the gated fit the per-uid uncensored target."""
    panel = _panel()
    cens = _censoring(panel)
    origin = pd.Timestamp("2024-02-12")
    a_hist = panel[panel[UNIQUE_ID] == "A"].reset_index(drop=True)
    a_cens = cens[cens[UNIQUE_ID] == "A"].reset_index(drop=True)
    task = ForecastTask(
        history=a_hist,
        horizon=2,
        model_config={**_GLOBAL_CONFIG, "scope": "local", "censoring_fit": True},
        censoring=a_cens,
    )
    chunk_ref = stage_local_chunk(
        [task], str(tmp_path / "chunk"), horizon=2, model_config=task.model_config, task_group=None
    )

    captured = _capture_fit_df(monkeypatch)
    _process_local_chunk(chunk_ref, origin, collect_fitted_values=False)

    fit_df = captured["fit_df"]
    window = a_hist[a_hist[DS] < origin]
    expected = (
        add_stockout_features(window, a_cens)[[UNIQUE_ID, DS, "y_uncensored"]]
        .rename(columns={"y_uncensored": Y})
        .sort_values([UNIQUE_ID, DS])
        .reset_index(drop=True)
    )
    expected[Y] = expected[Y].astype("float32")
    actual = fit_df[[UNIQUE_ID, DS, Y]].sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
