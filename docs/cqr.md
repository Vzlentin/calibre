# Conformalized quantile regression examples

[Semantics](semantics.md#cqr) defines the CQR score, fitting constraints, validation,
empty sets and coverage assumptions. This guide shows the array API.

## Train either backend

Install `libcalibre[ml,neural]` to run both examples. From this repository,
`uv sync --group dev` supplies the two backends and development tools.

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
    assert forecasts.quantiles is not None
    intervals = replay(
        forecasts,
        actuals,
        target=Step(),
        score=CQR(),
        calibrator=SplitQuantile(0.9, window=20),
    )
    ready = np.isfinite(intervals.threshold)
    empty = ready & (intervals.lower > intervals.upper)
    print(name, forecasts.quantiles.values.shape, intervals.covered[ready].mean(), empty.sum())
```

## Use precomputed quantiles

Replace the model forecasts above with caller-owned arrays:

```python
from calibre import Forecasts, Quantiles

forecasts = Forecasts(
    origins,
    points,
    quantiles=Quantiles(raw_quantiles, (0.05, 0.95)),
)
```

Use the same `replay` call. The array shapes are listed in
[Architecture](architecture.md#array-contracts).

## One online origin and saved state

Using `forecasts` and `actuals` from the training example:

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
    forecasts.quantiles.values[0],
    actuals[:, : origin + 1],
    target=Step(),
    score=CQR(),
    calibrator=calibrator,
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
    forecasts.quantiles.values[1],
    actuals[:, origin + 1 : next_origin + 1],
    target=Step(),
    score=CQR(),
    calibrator=calibrator,
)
```
