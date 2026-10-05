# Calibre

Calibre is a conformal forecasting library for panels of time series. It makes point
forecasts at many origins, reconciles them over a hierarchy, and calibrates bands and
bounds from out-of-sample residuals.

## Quick start

```python
import numpy as np
import pandas as pd

from calibre import BottomUp, Hierarchy, Panel, SeasonalNaive, forecast_origins
from calibre import resolved_rows, width, window_bound

rng = np.random.default_rng(0)
series = np.array(["a", "b", "c"])
periods = pd.date_range("2024-01-01", periods=120, freq="D")
panel = Panel(series, periods, rng.poisson(5.0, (3, 120)), "D")
hierarchy = Hierarchy.from_attributes(
    series, pd.DataFrame({"group": ["x", "x", "y"]}, index=series)
)

horizon = 7
origins = np.arange(60, 113)  # each origin is the index of the last observed period
run = forecast_origins(panel, hierarchy, SeasonalNaive(7), BottomUp(hierarchy), origins, horizon)
resid = run.residuals(hierarchy.summing @ panel.values)

last = len(origins) - 1
rows = resolved_rows(origins, last, horizon)
half_width = width(resid, rows, level=0.9, window=28)
upper = window_bound(run.points[last], resid, int(rows[-1]), 0.9, horizon, window=28)
```

## Concepts

- **Panel**: bottom series values `[B, T]` on a complete calendar. Validated once,
  then read-only.
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
- **forecast_origins**: the one origin loop. It builds the windows, fits every
  `refit_every` origins, and returns **Forecasts**: points `[O, N, H]` and residuals.
- **Calibration**: `width` gives per-step two-sided bands, and `window_bound` gives a
  one-sided bound on a multi-step total. Both read only residuals known at the origin.

```text
Panel ─ Hierarchy ─▶ Window per origin ─ Forecaster.fit / Fitted.predict ─▶ base [S, H]
base ─ Reconciler ─▶ points [O, N, H] ─ actuals ─▶ residuals ─ width / window_bound ─▶ bands
```

Two rules hold everywhere. A model reads only its window, so a later value cannot
change a point unless a covariate declares it known ahead. A band reads only residuals
whose target is known at the origin.

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
