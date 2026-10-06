# Calibre

Calibre is a conformal forecasting library for panels of time series. It makes point
forecasts at many origins, reconciles them over a hierarchy, and calibrates bands and
bounds from out-of-sample residuals.

## Quick start

```python
import numpy as np
import pandas as pd

from calibre import AbsoluteResidual, BottomUp, Conformal, Hierarchy, Panel, SeasonalNaive
from calibre import SplitQuantile, WindowSum, forecast_origins
from calibre.evaluate import coverage

rng = np.random.default_rng(0)
series = np.array(["a", "b", "c"])
periods = pd.date_range("2024-01-01", periods=120, freq="D")
panel = Panel(series, periods, rng.poisson(5.0, (3, 120)), "D")
hierarchy = Hierarchy.from_attributes(
    series, pd.DataFrame({"group": ["x", "x", "y"]}, index=series)
)

origins = np.arange(60, 113)  # each origin is the index of the last observed period
run = forecast_origins(panel, hierarchy, SeasonalNaive(7), BottomUp(hierarchy), origins, 7)
actuals = hierarchy.aggregate(panel.values)

bands = Conformal(AbsoluteResidual(), SplitQuantile(window=28), level=0.9).replay(run, actuals)
bound = Conformal(WindowSum(7), SplitQuantile(window=28), level=0.9).replay(run, actuals)
print(coverage(bands.target, bands.lower, bands.upper))
```

## Concepts

- **Panel**: bottom series values `[B, T]` on a complete calendar, with an optional
  `censored` mask. Validated once, then read-only.
- **Hierarchy**: node labels and the summing matrix `[N, B]`, bottoms first.
  `from_attributes` builds it from attribute columns, crossed when a level lists
  several, and drops an aggregate that repeats another node.
- **Covariate**: an input aligned on the panel calendar. Its shape gives the kind:
  `[B, 1]` static, `[1, L]` calendar, `[B, L]` dynamic. It declares if it is known
  ahead and how aggregate nodes get it, `sum` or `mean`.
- **Window**: what a model can see at one origin. History through the origin,
  covariates through the origin, and known-ahead covariates through the last step.
- **Forecaster**: `fit(window)` returns a **Fitted** model, and `predict(window)`
  returns base points `[S, H]`. Fitting never changes the forecaster.
- **Reconciler**: maps base points `[S, H]` to node points `[N, H]`. `BottomUp`
  forecasts the bottoms, `Identity` and `WlsStruct` forecast every node.
- **forecast_origins**: builds the windows, fits every `refit_every` origins, and
  returns **Forecasts**: points `[O, N, H]`.
- **Score**: how wrong a forecast was, in columns. `AbsoluteResidual` per step for a
  band, `SignedResidual` per step for an upper bound, `WindowSum(steps)` for a bound on
  a multi-step total.
- **Calibrator**: from the scores known so far to a threshold per node and column.
  `SplitQuantile`, and the online `ACI` and `QuantileTracker`.
- **Conformal**: runs a score and a calibrator origin by origin. `step` is one origin,
  `replay` is a backtest. It returns **Calibrated**: thresholds, bounds, targets,
  and scores `[O, N, C]`.
- **State**: a nested dict of numpy arrays. `flatten` gives one named array per key,
  for any store.

```text
Panel ─ Hierarchy ─▶ Window per origin ─ Forecaster.fit / Fitted.predict ─▶ base [S, H]
base ─ Reconciler ─▶ points [O, N, H]
points + actuals ─ Conformal(Score, Calibrator, level).step per origin ─▶ thresholds, bounds
Calibrated ─ calibre.evaluate ─▶ coverage, width, interval score, pinball, cost
```

Two rules hold everywhere. A model reads only its window, so a later value cannot
change a point unless a covariate declares it known ahead. A calibrator sees a score
only once its target is known, and before the origin that knows it issues.

## Write a calibrator

A calibrator is three functions of an explicit state. `Conformal` handles the
origins, the delays, and the indexing. This is the whole of a quantile tracker:

```python
import numpy as np


class Tracker:
    def __init__(self, lr):
        self.lr = lr

    def init(self, n_nodes, n_columns):
        return {"q": np.zeros((n_nodes, n_columns))}

    def update(self, state, feedback, level):
        q = state["q"].copy()
        for row, column in enumerate(feedback.column):  # one row = one origin, one column
            miss = feedback.scores[row] > feedback.issued[row]
            q[:, column] += self.lr * (miss - (1 - level))
        return {"q": q}

    def threshold(self, state, level):
        return state["q"].astype(np.float32)
```

Keep the state in numpy arrays. Then a product can save it after each origin with
`calibre.conformal.state.flatten` and continue from it.

## Models

| Model | Kind | Install |
|---|---|---|
| `SeasonalNaive` | local | core |
| `calibre.forecast.statsforecast.StatsForecastModel` | local, any `statsforecast.models` model | `calibre[stats]` |
| `calibre.forecast.mlforecast.MLForecast` | global regressor with lags and covariates | `calibre[ml]` |
| `calibre.forecast.neuralforecast.NeuralForecast` | global network from `neuralforecast.models` | `calibre[neural]` |

```python
from lightgbm import LGBMRegressor
from calibre import Covariate
from calibre.forecast.mlforecast import MLForecast

model = MLForecast(
    LGBMRegressor(), lags=[7, 14, 28], features=["price"], fit_periods=365, lookback=84
)
price = Covariate(prices, known_ahead=True, aggregate="mean")  # [B, T + H]
run = forecast_origins(
    panel,
    hierarchy,
    model,
    BottomUp(hierarchy),
    origins,
    28,
    refit_every=7,
    covariates={"price": price},
)
```

## Documents

- [Architecture](docs/architecture.md): modules, array contracts, and costs.
- [Semantics](docs/semantics.md): the calibration rules.

## Development

```sh
uv sync --locked --group dev
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check src/calibre/
uv build --no-sources
```
