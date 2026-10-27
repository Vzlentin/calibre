# Calibre

[![CI](https://github.com/Vzlentin/calibre/actions/workflows/ci.yml/badge.svg)](https://github.com/Vzlentin/calibre/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Vzlentin/calibre/blob/main/LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Calibre is a conformal forecasting library for panels of time series. It makes point
forecasts at many origins, reconciles them over a hierarchy, and calibrates bands and
bounds from out-of-sample residuals.

## Install

The package is `libcalibre` on PyPI, and the import name is `calibre`:

```sh
pip install libcalibre
# or
uv add libcalibre
```

The vendor model adapters need an extra: `stats` (statsforecast), `ml` (mlforecast), or
`neural` (neuralforecast). `bootstrap` and `spci` both install scikit-learn for their
respective methods; importing `calibre` loads none of these optional libraries.

```sh
pip install "libcalibre[stats]"
```

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

Shapes use these axes: B bottom series, T periods, N nodes with bottoms first,
S forecast series (B for `BottomUp`, N otherwise), O origins, H forecast steps, and
C target columns (H for `Step`, 1 for `LeadTime`).

![Panel and Hierarchy go to the Forecaster, base forecasts to the Reconciler, points to the Calibrator, bounds to order_up_to and settle](https://raw.githubusercontent.com/Vzlentin/calibre/main/docs/pipeline.png)

| Stage | Names |
|---|---|
| Data | `Panel`, `Hierarchy`, `Covariate` |
| Forecast | `Forecaster`, `Reconciler` (`BottomUp`, `Identity`, `WlsStruct`), `rolling_forecasts` |
| Calibrate | `Target` (`Step`, `LeadTime`), `Score` (`Absolute`, `Signed`, `Asymmetric`), `Loss` (`Miss`), `Calibrator` (`SplitQuantile`, `ACI`, `QuantileTracker`, `ConformalPID`, `RiskControl`, `EnbPI`, `SPCI`), `step`, `replay` |
| Decide | `critical_ratio`, `order_up_to`, `settle` |
| Measure | `calibre.metrics`: coverage, width, interval score, pinball, cost |

Two rules hold everywhere. A model reads only its window, so a later value cannot
change a point unless a covariate declares it known ahead. A calibrator sees a score
only once its target is known, and before the origin that knows it issues.

[Architecture](https://github.com/Vzlentin/calibre/blob/main/docs/architecture.md) has the shapes and the modules.

## Write a calibrator

A calibrator owns its target, a level or a loss, and is three functions of an explicit
state. `calibre.online` handles the origins, the delays, and the indexing. A new method
is one file in `calibre/conformal/calibrators/` that imports only `calibrators.base`,
`calibrators.ranks`, and `conformal.losses`. This is the whole of a quantile tracker:

```python
import numpy as np

from calibre.conformal.calibrators.base import Calibrator


class Tracker(Calibrator):
    def __init__(self, level, lr):
        self.level = level
        self.lr = lr

    def initial_state(self, n_nodes, n_columns):
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
| `calibre.forecast.models.bootstrap.BootstrapEnsemble` | local, frozen one-step lag-row ensembles with OOB errors | `libcalibre[bootstrap]` |
| `calibre.forecast.models.statsforecast.StatsForecastModel` | local, any `statsforecast.models` model | `libcalibre[stats]` |
| `calibre.forecast.models.mlforecast.MLForecast` | global regressor with lags and covariates | `libcalibre[ml]` |
| `calibre.forecast.models.neuralforecast.NeuralForecast` | global network from `neuralforecast.models` | `libcalibre[neural]` |

```python
from lightgbm import LGBMRegressor
from calibre import BottomUp, Covariate, rolling_forecasts
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

## EnbPI

Install `libcalibre[bootstrap]` for the scikit-learn bootstrap adapter. Forecasting
and residual calibration are separate: `BootstrapEnsemble` trains frozen learners;
`EnbPI` maintains signed errors; `Asymmetric` turns the two quantiles into bounds.

Using the `panel` and `hierarchy` from the quick start:

```python
from sklearn.linear_model import Ridge

from calibre import Asymmetric, EnbPI, initial_state
from calibre.forecast.models.bootstrap import BootstrapEnsemble

forecasts = rolling_forecasts(
    panel,
    hierarchy,
    BootstrapEnsemble(Ridge(), n_lags=7, random_state=42),
    BottomUp(hierarchy),
    np.arange(60, 113),
    1,
    initial_residuals=True,
)
seed = forecasts.initial_residuals
assert seed is not None
calibrator = EnbPI(level=0.9, window=len(seed.values))
state = initial_state(Step(), calibrator, len(hierarchy.nodes), horizon=1)
state["calibrator"] = calibrator.seed(state["calibrator"], seed.values)
intervals = replay(
    forecasts,
    hierarchy.aggregate(panel.values),
    target=Step(),
    score=Asymmetric(),
    calibrator=calibrator,
    state=state,
)
print(coverage(intervals.target, intervals.lower, intervals.upper))
```

The adapter supports one-step lag regression only and rejects covariates.
OOB errors are initialization data, not chronological forecast errors. Only their
known initial training prefix may seed calibration; no refits are allowed when
collecting these errors. The empirical interval searches both residual tails, with
no split-conformal n+1 correction. Its parameters are [N, C, 2]; bounds remain [N, C].
This does not establish distribution-free conditional coverage for dependent series.
See [Semantics](docs/semantics.md#enbpi-explicit-semantic-additions) for the precise
center, ranks, missing-data policy, and scope. A saved online state can be passed
back to `replay(state=...)` to continue, without retraining the point model.

## SPCI

Install `libcalibre[spci]` for scikit-learn. `SPCI` learns conditional quantiles of
signed forecast errors from their recent history, using a separate quantile forest
per node and target column. It searches both tails for the shortest estimated interval.
Point forecasting stays separate; any existing multi-horizon forecaster can supply
points. `BootstrapEnsemble` remains one-step only.

Using `run` and `actuals` from the quick start:

```python
from calibre import SPCI, Asymmetric

calibrator = SPCI(
    level=0.9,
    window=128,
    n_lags=7,
    ahead=np.arange(1, 8),  # last covered step: 1..H for Step()
    random_state=42,
)
intervals = replay(run, actuals, target=Step(), score=Asymmetric(), calibrator=calibrator)
ready = np.isfinite(intervals.threshold).all(axis=-1)
print(ready.sum())  # keep warm-up visible when assessing coverage and width
```

For `LeadTime(7)`, use `ahead=7`: SPCI then learns the signed error of the total,
not a sum of independently calibrated endpoints. `ahead` is required and must match
each target column; it is measured in calendar periods, including when origins have gaps.
Each historical context contains only residuals whose targets were already known at
that historical issuing origin. At prediction time, only revealed residuals are used.

Without a seed, the existing replay collects errors online. Until at least
`min_train_samples` (default 20) complete training pairs exist, a cell returns
`[-inf, inf]`, with no empirical EnbPI fallback. An optional initialization is explicit:

```python
from calibre import initial_state

# calibration is a historical Forecasts from the same point-model generation.
# Every retained target must be known before the first evaluation origin.
known = calibration.origins + run.points.shape[-1] <= run.origins[0]
state = initial_state(Step(), calibrator, actuals.shape[0], run.points.shape[-1])
state["calibrator"] = calibrator.seed(
    state["calibrator"],
    calibration.residuals(actuals)[known],
    origins=calibration.origins[known],
)
intervals = replay(
    run, actuals, target=Step(), score=Asymmetric(), calibrator=calibrator, state=state
)
```

The seed example assumes the caller has separately produced historical `calibration`
forecasts; it never substitutes in-sample fitted errors. Save and resume both the
calibrator and ledger arrays via `replay(state=...)`. Forests are refitted from arrays
at every ready emission and discarded; a fixed random seed makes continuation exact.
This costs up to N × H forest fits per origin for `Step()`; full-panel scalability is
not established. See [Semantics](docs/semantics.md#spci) for weighted ranks, readiness,
and scope. This is an adaptation of [Xu and Xie (2023)](https://arxiv.org/abs/2212.03463v3),
not a reproduction of their block-emission experiments or an automatic finite-sample,
conditional, or simultaneous-path coverage guarantee.

## Conformal PID

`ConformalPID` combines quantile tracking, a log-scaled tangent integrator, and
rolling regularized AR(1) scorecasting. The anticipation term is a baseline
substitute for the paper's Theta model, not a finite-difference derivative.
It uses nonnegative scalar scores, normally `Absolute()`, independently per node
and target column.

Using the seven-step `run`, `panel`, `hierarchy`, and `actuals` from the quick start:

```python
from calibre import ConformalPID, initial_state

calibration = rolling_forecasts(
    panel, hierarchy, SeasonalNaive(7), BottomUp(hierarchy), np.arange(28, 54), 7
)
# Last calibration target: origin 53 + step 7 = 60, known at the first run origin.
known_actuals = actuals[:, : int(run.origins[0]) + 1]
scores = np.abs(calibration.residuals(known_actuals))
calibrator = ConformalPID(level=0.9, window=256, scorecast_window=64, ahead=np.arange(1, 8))
state = initial_state(Step(), calibrator, len(hierarchy.nodes), horizon=7)
state["calibrator"] = calibrator.seed(state["calibrator"], scores)
intervals = replay(
    run, actuals, target=Step(), score=Absolute(), calibrator=calibrator, state=state
)
print(coverage(intervals.target, intervals.lower, intervals.upper))
```

Seed every node/column with at least one already revealed score. Only the last
`window` finite scores initialize its empirical quantile and fixed learning-rate
scale; seeding invents no past misses. Set `ahead` to the last covered step per
column: 1 through H for `Step()`, or `steps` for `LeadTime(steps)`; the default is 1.
Saved state resumes through `replay(state=...)` with the same configuration.

Negative thresholds give empty sets; `+inf` gives the whole line and `-inf` an
empty set. Do not clip these outputs or exclude saturation dates from coverage.
Use `Replay.covered` or `coverage` on all known targets. The current generic width
and interval-score metrics have no empty-set convention, and `bench/compare.py`
masks infinite thresholds as unready: it is not a PID evaluator.
See [Semantics](docs/semantics.md#conformal-pid-explicit-semantic-additions) for the
exact variant and the limits of its coverage claims.

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
uv run python bench/vn2_conformal.py data/vn2
```

## Documents

- [Architecture](https://github.com/Vzlentin/calibre/blob/main/docs/architecture.md): modules, array contracts, and costs.
- [Semantics](https://github.com/Vzlentin/calibre/blob/main/docs/semantics.md): the calibration rules.

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check src/
uv build --no-sources
```

See [Contributing](https://github.com/Vzlentin/calibre/blob/main/CONTRIBUTING.md).

## License

[MIT](https://github.com/Vzlentin/calibre/blob/main/LICENSE)
