import numpy as np
import pandas as pd
import pytest
from statsforecast.models import SeasonalNaive as SfSeasonalNaive
from statsforecast.models import SimpleExponentialSmoothing

from calibre.forecast import Covariate, Window
from calibre.forecast.models.naive import SeasonalNaive
from calibre.forecast.models.statsforecast import StatsForecastModel
from tests.synthetic import make_panel


def window(history: np.ndarray, horizon: int) -> Window:
    history = history.view()
    history.flags.writeable = False
    periods = pd.date_range("2020-01-01", periods=history.shape[1] + horizon, freq="D")
    return Window(y=history, x={}, periods=periods, horizon=horizon)


def test_seasonal_naive_repeats_last_season():
    history = np.arange(2 * 10, dtype=np.float32).reshape(2, 10)
    model = SeasonalNaive(season=3)
    points = model.fit(window(history, 7)).predict(window(history, 7))
    assert points.shape == (2, 7)
    np.testing.assert_array_equal(points[0], [7, 8, 9, 7, 8, 9, 7])
    np.testing.assert_array_equal(points[1], [17, 18, 19, 17, 18, 19, 17])


def test_seasonal_naive_horizon_shorter_than_season():
    history = np.arange(10, dtype=np.float32).reshape(1, 10)
    points = SeasonalNaive(season=7).predict(window(history, 3))
    np.testing.assert_array_equal(points[0], [3, 4, 5])


def test_seasonal_naive_rejects_bad_season_and_short_history():
    with pytest.raises(ValueError, match="season"):
        SeasonalNaive(season=0)
    with pytest.raises(ValueError, match="history has 3 periods"):
        SeasonalNaive(season=7).predict(window(np.ones((1, 3), dtype=np.float32), 5))


def test_statsforecast_matches_own_seasonal_naive_on_read_only_history():
    panel, _ = make_panel(n_series=5, n_periods=40)
    ours = SeasonalNaive(season=7)
    theirs = StatsForecastModel(SfSeasonalNaive(season_length=7))
    points = theirs.fit(window(panel.values, 10)).predict(window(panel.values, 10))
    assert points.shape == (5, 10) and points.dtype == np.float32
    np.testing.assert_array_equal(points, ours.predict(window(panel.values, 10)))


def test_statsforecast_fixed_alpha_ses_is_the_exponential_average():
    history = np.array([[10, 20, 30, 40]], dtype=np.float32)
    model = StatsForecastModel(SimpleExponentialSmoothing(alpha=0.5))
    # Level starts at the first value: 10, 15, 22.5, 31.25. The forecast is flat.
    np.testing.assert_allclose(model.predict(window(history, 3)), [[31.25] * 3])


def test_statsforecast_rejects_foreign_model():
    with pytest.raises(ValueError, match="statsforecast.models instance"):
        StatsForecastModel(SeasonalNaive(season=7))


@pytest.mark.parametrize(
    ("values", "aggregate", "match"),
    [
        (np.ones(3), "sum", "2D"),
        ([[1.0, np.nan]], "sum", "finite"),
        ([[1.0, 2.0]], "max", "aggregate"),
    ],
)
def test_covariate_rejects_invalid_inputs(values, aggregate, match):
    with pytest.raises(ValueError, match=match):
        Covariate(np.asarray(values), known_ahead=True, aggregate=aggregate)


def test_covariate_values_are_a_read_only_float32_copy():
    source = np.array([[1, 2]])
    covariate = Covariate(source, known_ahead=False, aggregate="mean")
    source[0, 0] = 9
    assert covariate.values.dtype == np.float32
    assert covariate.values.tolist() == [[1.0, 2.0]]
    with pytest.raises(ValueError):
        covariate.values[0, 0] = 5
