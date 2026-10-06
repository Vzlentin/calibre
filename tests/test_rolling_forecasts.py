import numpy as np
import pandas as pd
import pytest

from calibre.backtest import rolling_forecasts
from calibre.data import Hierarchy, Panel
from calibre.forecast import Covariate, Fitted, Forecaster, Window
from calibre.forecast.models.naive import SeasonalNaive
from calibre.forecast.reconcile import BottomUp, Identity, WlsStruct

ORIGINS = np.array([3, 5, 8])


class MeanAtFit(Forecaster, Fitted):
    """A global model: it learns each series mean at fit time and repeats it."""

    def fit(self, window: Window) -> "MeanAtFit":
        fitted = MeanAtFit()
        fitted.mean = window.y.mean(axis=1)
        return fitted

    def predict(self, window: Window) -> np.ndarray:
        return np.repeat(self.mean[:, None], window.horizon, axis=1)


class Echo(Forecaster, Fitted):
    """Return the last `horizon` visible values of one covariate for every series."""

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, window: Window) -> "Echo":
        return self

    def predict(self, window: Window) -> np.ndarray:
        values = window.x[self.name]
        tail = values[:, -window.horizon :] if values.shape[1] > 1 else values
        return np.broadcast_to(tail, (window.y.shape[0], window.horizon))


class WrongShape(Forecaster, Fitted):
    def fit(self, window: Window) -> "WrongShape":
        return self

    def predict(self, window: Window) -> np.ndarray:
        return np.zeros((1, 1))


class Writer(Forecaster, Fitted):
    def fit(self, window: Window) -> "Writer":
        return self

    def predict(self, window: Window) -> np.ndarray:
        window.y[0, 0] = 0
        return window.y[:, -window.horizon :]


def tiny() -> tuple[Panel, Hierarchy]:
    """Bottoms a, b in group g1 and c in g2. Values a = t, b = 10 + t, c = 100 + t.

    Group g2 has only c, so it is not a separate node.
    """
    series = np.array(["a", "b", "c"])
    values = np.arange(10, dtype=np.float32) + np.array([[0], [10], [100]], dtype=np.float32)
    panel = Panel(series, pd.date_range("2024-01-01", periods=10, freq="D"), values, "D")
    attributes = pd.DataFrame({"group": ["g1", "g1", "g2"]}, index=series)
    return panel, Hierarchy.from_attributes(series, attributes)


def test_bottom_up_points_and_residuals():
    panel, hierarchy = tiny()
    run = rolling_forecasts(panel, hierarchy, SeasonalNaive(1), BottomUp(hierarchy), ORIGINS, 2)
    assert list(hierarchy.nodes) == ["a", "b", "c", "group=g1", "total"]
    # Last value at origin o: a = o, b = 10 + o, c = 100 + o, then sums.
    assert run.points[0, :, 0].tolist() == [3, 13, 103, 16, 119]
    assert run.points.shape == (3, 5, 2) and run.points.dtype == np.float32
    resid = run.residuals(hierarchy.summing @ panel.values)
    # Each bottom grows by 1 per period, so step s misses by s per bottom.
    np.testing.assert_array_equal(resid[0], [[1, 2], [1, 2], [1, 2], [2, 4], [3, 6]])
    # Origin 8, step 2 targets period 10, after the last period: not yet known.
    assert np.isfinite(resid[2, :, 0]).all() and np.isnan(resid[2, :, 1]).all()


def test_wls_on_coherent_base_points_equals_bottom_up():
    panel, hierarchy = tiny()
    bottom_up = rolling_forecasts(
        panel, hierarchy, SeasonalNaive(1), BottomUp(hierarchy), ORIGINS, 2
    )
    wls = rolling_forecasts(panel, hierarchy, SeasonalNaive(1), WlsStruct(hierarchy), ORIGINS, 2)
    np.testing.assert_allclose(wls.points, bottom_up.points, atol=1e-4)


def test_refit_every_controls_when_the_model_learns():
    panel, hierarchy = tiny()
    once = rolling_forecasts(panel, hierarchy, MeanAtFit(), BottomUp(hierarchy), ORIGINS, 1)
    # Fitted at origin 3 only: mean of a over periods 0..3 is 1.5.
    assert once.points[:, 0, 0].tolist() == [1.5, 1.5, 1.5]
    every_two = rolling_forecasts(
        panel, hierarchy, MeanAtFit(), BottomUp(hierarchy), ORIGINS, 1, refit_every=2
    )
    # Fitted at origins 3 and 8: means 1.5 and 4.
    assert every_two.points[:, 0, 0].tolist() == [1.5, 1.5, 4.0]


def test_a_value_after_the_origin_never_changes_its_points():
    panel, hierarchy = tiny()
    changed = panel.values.copy()
    changed[0, 6] = 1000
    other = Panel(panel.series, panel.periods, changed, "D")
    model = MeanAtFit()
    before = rolling_forecasts(panel, hierarchy, model, BottomUp(hierarchy), ORIGINS, 2, 1)
    after = rolling_forecasts(other, hierarchy, model, BottomUp(hierarchy), ORIGINS, 2, 1)
    np.testing.assert_array_equal(after.points[:2], before.points[:2])
    assert not np.array_equal(after.points[2], before.points[2])


def covariates() -> dict[str, Covariate]:
    dynamic = np.arange(12, dtype=np.float32) + np.array([[1000], [2000], [3000]])
    return {
        "promo": Covariate(dynamic, known_ahead=True, aggregate="mean"),
        "weather": Covariate(dynamic, known_ahead=False, aggregate="sum"),
        "calendar": Covariate(np.arange(12)[None], known_ahead=True, aggregate="sum"),
        "size": Covariate([[1], [2], [3]], known_ahead=False, aggregate="sum"),
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Known ahead: periods 4 and 5. Aggregates are means of their bottoms.
        (
            "promo",
            [[1004, 1005], [2004, 2005], [3004, 3005], [1504, 1505], [2004, 2005]],
        ),
        # Not known ahead: the last seen periods 2 and 3. Aggregates are sums.
        (
            "weather",
            [[1002, 1003], [2002, 2003], [3002, 3003], [3004, 3006], [6006, 6009]],
        ),
        # One calendar row is the same for every node.
        ("calendar", [[4, 5]] * 5),
        ("size", [[1, 1], [2, 2], [3, 3], [3, 3], [6, 6]]),
    ],
)
def test_window_shows_covariates_by_availability_and_node(name, expected):
    panel, hierarchy = tiny()
    identity = Identity(hierarchy)
    run = rolling_forecasts(
        panel, hierarchy, Echo(name), identity, ORIGINS, 2, covariates=covariates()
    )
    np.testing.assert_array_equal(run.points[0], expected)


@pytest.mark.parametrize(
    ("origins", "horizon", "refit_every", "match"),
    [
        (np.array([5, 3]), 2, None, "increasing"),
        (np.array([3, 10]), 2, None, r"\[0, 9\]"),
        (np.array([3.0]), 2, None, "period indexes"),
        (ORIGINS, 0, None, "horizon"),
        (ORIGINS, 2, 0, "refit_every"),
    ],
)
def test_rolling_forecasts_rejects_invalid_schedules(origins, horizon, refit_every, match):
    panel, hierarchy = tiny()
    with pytest.raises(ValueError, match=match):
        rolling_forecasts(
            panel, hierarchy, SeasonalNaive(1), BottomUp(hierarchy), origins, horizon, refit_every
        )


def test_rolling_forecasts_rejects_bad_covariates_models_and_reconcilers():
    panel, hierarchy = tiny()
    short = {"promo": Covariate(np.ones((3, 10)), known_ahead=True, aggregate="sum")}
    with pytest.raises(ValueError, match="has 10 periods, needs 11"):
        rolling_forecasts(
            panel, hierarchy, Echo("promo"), BottomUp(hierarchy), ORIGINS, 2, None, short
        )
    rows = {"promo": Covariate(np.ones((2, 12)), known_ahead=True, aggregate="sum")}
    with pytest.raises(ValueError, match="one row per bottom series"):
        rolling_forecasts(
            panel, hierarchy, Echo("promo"), BottomUp(hierarchy), ORIGINS, 2, None, rows
        )
    with pytest.raises(ValueError, match=r"model returned shape \(1, 1\)"):
        rolling_forecasts(panel, hierarchy, WrongShape(), BottomUp(hierarchy), ORIGINS, 2)
    other = BottomUp(Hierarchy.flat(panel.series))
    with pytest.raises(ValueError, match="does not fit the hierarchy"):
        rolling_forecasts(panel, hierarchy, SeasonalNaive(1), other, ORIGINS, 2)
    with pytest.raises(ValueError, match="read-only"):
        rolling_forecasts(panel, hierarchy, Writer(), BottomUp(hierarchy), ORIGINS, 2)
