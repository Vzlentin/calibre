# Conformalized quantile regression

CQR calibrates an interval predicted by quantile models. `CQR()` is a score;
`SplitQuantile(level)` computes the correction. There is no separate CQR rank or
calibrator state. [Semantics](semantics.md#cqr-explicit-semantic-additions) defines
the numerical rules and coverage claims.

## Train either backend

Install `libcalibre[ml,neural]` to run both examples. From this repository,
`uv sync --group dev` supplies the two backends and the test tools.

The following example trains each model at origin 49 using only periods 0 through
49, then calibrates on subsequently revealed targets. `Identity` forecasts every
node directly, including aggregates when a hierarchy has them. The quantiles need
not be coherent across nodes.

```python
import numpy as np
import pandas as pd
from neuralforecast.models import MLP
from sklearn.linear_model import QuantileRegressor, Ridge

from calibre import CQR, Hierarchy, Identity, Panel, SplitQuantile, Step
from calibre import replay, rolling_forecasts
from calibre.forecast.models.mlforecast import MLForecast
from calibre.forecast.models.neuralforecast import NeuralForecast

rng = np.random.default_rng(7)
series = np.array(["a", "b", "c"])
periods = pd.date_range("2024-01-01", periods=90, freq="D")
values = rng.normal(10, 2, size=(3, 90)).astype(np.float32)
panel = Panel(series, periods, values, "D")
hierarchy = Hierarchy.flat(series)
origins = np.arange(49, 84)
horizon = 3
actuals = hierarchy.aggregate(panel.values)

models = {
    "ml": MLForecast(
        Ridge(),
        lags=[1, 7],
        quantile_models={
            0.05: QuantileRegressor(quantile=0.05, alpha=0.1),
            0.95: QuantileRegressor(quantile=0.95, alpha=0.1),
        },
    ),
    "neural": NeuralForecast(
        MLP,
        {
            "input_size": 7,
            "hidden_size": 16,
            "num_layers": 1,
            "max_steps": 20,
            "val_check_steps": 20,
            "random_seed": 7,
            "accelerator": "cpu",
        },
        quantiles=(0.05, 0.95),
    ),
}

for name, model in models.items():
    forecasts = rolling_forecasts(panel, hierarchy, model, Identity(hierarchy), origins, horizon)
    intervals = replay(
        forecasts,
        actuals,
        target=Step(),
        score=CQR(),
        calibrator=SplitQuantile(0.9, window=20),
    )
    ready = np.isfinite(intervals.threshold)
    empty = ready & (intervals.lower > intervals.upper)
    print(name, forecasts.interval.shape, intervals.covered[ready].mean(), empty.sum())
```

MLForecast clones the two unfitted estimators and fits a direct regressor for each
horizon. Configure their quantile objectives yourself: the mapping keys label the
models, but do not rewrite estimator parameters. The separate point model retains
its recursive behavior. LightGBM quantile estimators can replace `QuantileRegressor`
when installed and configured with matching quantile objectives.

NeuralForecast configures `MQLoss` for the two requested quantiles and the median.
Its point forecast is that median. Do not also supply `loss` or `valid_loss`.
`import calibre` itself imports neither backend.

## Use precomputed quantiles

`Forecasts(origins, points, interval=raw, quantile_levels=(low, high))` accepts
caller-owned points `[O, N, H]` and finite raw quantiles `[O, N, H, 2]`. It orders a
copy of crossed endpoints. Use the same `replay` call as above. The caller owns
causal fitting, fixed model identity, and alignment of series, horizons, and levels.

`CQR` ignores the point values in its score, but points remain available in the
forecast and replay output. Raw levels such as `(0.05, 0.95)` describe the learned
quantiles; the calibrator's `level=0.9` is the requested calibrated coverage.

## One online origin and saved state

Using `forecasts` and `actuals` from the complete example:

```python
import io

from calibre import initial_state, step
from calibre.online.state import flatten, unflatten

calibrator = SplitQuantile(0.9, window=20)
state = initial_state(Step(), calibrator, len(hierarchy.nodes), horizon, score=CQR())
origin = int(forecasts.origins[0])
state, issued = step(
    state,
    origin,
    forecasts.points[0],
    actuals[:, : origin + 1],
    target=Step(),
    score=CQR(),
    calibrator=calibrator,
    interval=forecasts.interval[0],
)

buffer = io.BytesIO()
np.savez(buffer, allow_pickle=False, **flatten(state))
buffer.seek(0)
with np.load(buffer, allow_pickle=False) as saved:
    restored = unflatten(dict(saved))

next_origin = int(forecasts.origins[1])
restored, next_issued = step(
    restored,
    next_origin,
    forecasts.points[1],
    actuals[:, origin + 1 : next_origin + 1],
    target=Step(),
    score=CQR(),
    calibrator=calibrator,
    interval=forecasts.interval[1],
)
```

The state includes native intervals pending their delayed targets. It contains no
fitted vendor models. Continue with the same frozen model and configuration. You
can also pass the restored state to `replay` with only later forecast origins.

## Limits

- Native rolling forecasts reject `refit_every`, fit-time OOB seeding, `BottomUp`,
  and `WlsStruct`. MLForecast quantile mode also rejects target transforms.
- CQR accepts `Step` only. A sum of per-step quantiles is not the quantile of a
  lead-time total. Train a model for that total before defining such an extension.
- Scores and corrections can be negative. Only raw quantile crossings are ordered;
  corrected endpoints are not. Empty prediction sets count as misses, not missing
  data. Generic width and interval-score metrics have no empty-set convention.
- Insufficient conformal history issues `[-inf, inf]`. With `SplitQuantile(0.9)`,
  nine finite scores are necessary per node/step without pooling. In this example,
  `ready` excludes only this warm-up, not finite empty sets.
- The finite-sample split-CQR claim requires exchangeable calibration/test scores
  and separate model training. A frozen model and causal code alone do not provide
  exchangeability for lagged time series. No conditional, simultaneous-path, or
  hierarchical-coherence guarantee is added.
