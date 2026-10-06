# Calibre

Calibre is a conformal forecasting library for panels of time series. It makes point
forecasts at many origins, reconciles them over a hierarchy, and calibrates bands and
bounds from out-of-sample residuals.

## Quick start

```python
import numpy as np
import pandas as pd

from calibre import Absolute, BottomUp, Hierarchy, LeadTime, Panel, SeasonalNaive, Signed
from calibre import SplitQuantile, Step, replay, rolling_forecasts
from calibre.metrics import coverage

rng = np.random.default_rng(0)
series = np.array(["a", "b", "c"])
periods = pd.date_range("2024-01-01", periods=120, freq="D")
panel = Panel(series, periods, rng.poisson(5.0, (3, 120)), "D")
hierarchy = Hierarchy.from_attributes(
    series, pd.DataFrame({"group": ["x", "x", "y"]}, index=series)
)

origins = np.arange(60, 113)  # each origin is the index of the last observed period
run = rolling_forecasts(panel, hierarchy, SeasonalNaive(7), BottomUp(hierarchy), origins, 7)
actuals = hierarchy.aggregate(panel.values)

calibrator = SplitQuantile(0.9, window=28)
bands = replay(run, actuals, target=Step(), score=Absolute(), calibrator=calibrator)
bound = replay(run, actuals, target=LeadTime(7), score=Signed(), calibrator=calibrator)
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
- **rolling_forecasts**: builds the windows, fits every `refit_every` origins, and
  returns **Forecasts**: points `[O, N, H]`.
- **Target**: which quantity each column bounds. `Step` gives one column per step,
  `LeadTime(steps)` one column for the total over the first steps.
- **Score**: how wrong a point was on a target column, and the bound that a threshold
  gives. `Absolute` for a two-sided band, `Signed` for an upper bound.
- **Calibrator**: from the scores known so far to a threshold per node and column, at
  its own target level: a scalar, or `[N, C]` per node and column. `SplitQuantile`, and
  the online `ACI` and `QuantileTracker`. A **QuantileCalibrator** also gives its
  threshold at any other level, which `ACI` needs from its base.
- **step**: one origin of online calibration. It gives the calibrator the scores whose
  targets are now known, then issues an **Issue**: thresholds and bounds `[N, C]`.
- **replay**: `step` over the origins of a backtest. It returns a **Replay**:
  thresholds, bounds, targets, and scores `[O, N, C]`.
- **Decision**: `critical_ratio(holding, shortage)` gives the newsvendor level for a
  calibrator, per node when the costs are per node. `order_up_to` turns an upper bound into an order, and `settle` runs one
  period of lost-sales inventory and returns its cost.
- **State**: a nested dict of numpy arrays. `flatten` gives one named array per key,
  for any store.

```text
Panel ─ Hierarchy ─▶ Window per origin ─ Forecaster.fit / Fitted.predict ─▶ base [S, H]
base ─ Reconciler ─▶ points [O, N, H]
points + actuals ─ step(Target, Score, Calibrator) per origin ─▶ thresholds, bounds
upper bound ─ order_up_to ─▶ order ─ settle ─▶ holding and shortage cost
Replay ─ calibre.metrics ─▶ coverage, width, interval score, pinball, cost
```

Two rules hold everywhere. A model reads only its window, so a later value cannot
change a point unless a covariate declares it known ahead. A calibrator sees a score
only once its target is known, and before the origin that knows it issues.

## Write a calibrator

A calibrator owns its target level and is three functions of an explicit state.
`calibre.online` handles the
origins, the delays, and the indexing. A new method is one file in
`calibre/conformal/calibrators/` that imports only `calibrators.base` and
`calibrators.ranks`. This is the whole of a quantile tracker:

```python
import numpy as np

from calibre.conformal.calibrators.base import Calibrator


class Tracker(Calibrator):
    def __init__(self, level, lr):
        self.level = level
        self.lr = lr

    def init(self, n_nodes, n_columns):
        return {"q": np.zeros((n_nodes, n_columns))}

    def update(self, state, feedback):
        q = state["q"].copy()
        for row, column in enumerate(feedback.column):  # one row = one origin, one column
            miss = feedback.scores[row] > feedback.issued[row]
            q[:, column] += self.lr * (miss - (1 - self.level))
        return {"q": q}

    def threshold(self, state):
        return state["q"].astype(np.float32)
```

Keep the state in numpy arrays. Then a product can save it after each origin with
`calibre.online.state.flatten` and continue from it.

## Models

| Model | Kind | Install |
|---|---|---|
| `calibre.forecast.models.naive.SeasonalNaive` | local | core |
| `calibre.forecast.models.statsforecast.StatsForecastModel` | local, any `statsforecast.models` model | `calibre[stats]` |
| `calibre.forecast.models.mlforecast.MLForecast` | global regressor with lags and covariates | `calibre[ml]` |
| `calibre.forecast.models.neuralforecast.NeuralForecast` | global network from `neuralforecast.models` | `calibre[neural]` |

```python
from lightgbm import LGBMRegressor
from calibre import Covariate
from calibre.forecast.models.mlforecast import MLForecast

model = MLForecast(
    LGBMRegressor(), lags=[7, 14, 28], features=["price"], fit_periods=365, lookback=84
)
price = Covariate(prices, known_ahead=True, aggregate="mean")  # [B, T + H]
run = rolling_forecasts(
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

## Benchmarks

`bench/` is a folder of scripts. They load benchmark data from a directory, run
benchmark protocols, and compare calibrators. They import the installed `calibre`.
Raw data is not in Git.

| Script | Content |
|---|---|
| `vn2.py` | `load(path)`: weekly VN2 sales with the out-of-stock mask, hierarchy, and starting stock. `play(data, bounds)`: six orders up to the bounds, eight settled weeks, holding 0.2 and shortage 1.0 |
| `m5.py` | `load(path)`: daily M5 sales and the 12-level hierarchy (42,840 nodes). `prices(horizon)`: the sell price covariate |
| `compare.py` | `compare(forecasts, actuals, calibrators, target, score, level)`: one row of metrics per method, on the cells where each method was ready |

`vn2_conformal.py` runs the whole VN2 path in about one second:

```sh
uv run --locked python bench/vn2_conformal.py data/vn2
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
uv run --locked ty check src/
uv build --no-sources
```
