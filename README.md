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

The `stats` extra installs statsforecast, the engine that `rolling_forecasts` runs.
Importing `calibre` does not load it.

```sh
pip install "libcalibre[stats]"
```

## Quick start

```python
import numpy as np
import pandas as pd
from conformalforecast.nonconformity import AbsoluteResidual, SignedResidual
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from calibre import BottomUp, Hierarchy, Panel, Span, Split, Step
from calibre import critical_ratio, reconcile, replay, rolling_forecasts

rng = np.random.default_rng(0)
frame = pd.DataFrame(
    {
        "unique_id": np.repeat(["a", "b", "c"], 120),
        "ds": np.tile(pd.date_range("2024-01-01", periods=120, freq="D"), 3),
        "y": rng.poisson(5.0, 360).astype(float),
    }
)
panel = Panel.from_frame(frame, freq="D")
attributes = pd.DataFrame({"group": ["x", "x", "y"]}, index=panel.series)
hierarchy = Hierarchy.from_attributes(panel.series, attributes, ["group"])

engine = StatsForecast(models=[SeasonalNaive(season_length=7)], freq="D")
origins = panel.periods[60:113]  # each origin is the last period that a forecast reads
forecasts = rolling_forecasts(panel, [engine], origins, 7)
nodes = reconcile(forecasts, hierarchy, BottomUp())
truth = hierarchy.aggregate(panel)

bands = replay(
    nodes, truth, target=Step(), method=Split(score=AbsoluteResidual(), window=28), level=0.9
)
bound = replay(
    nodes,
    truth,
    target=Span(1, 7),
    method=Split(score=SignedResidual(), window=28),
    level=critical_ratio(0.2, 1.0),
)
```

`bands.lower` and `bands.upper` are `[N, C, O]`: one band per node, step and origin.
`bound.upper` is the bound of the 7-day sum, at the cost-calibrated level.

## Concepts

Shapes use these axes: N nodes with the bottoms first, S series, B bottom series,
M models, T periods, O origins, H steps, and C target columns (H for `Step`, 1 for
`Span`).

| Stage | Names |
|---|---|
| Data | `Panel`, `Hierarchy` |
| Forecast | `rolling_forecasts`, `Forecasts`, `Window`, `Forecaster`, `reconcile`, `Reconciler` (`BottomUp`, `Identity`, `WlsStruct`) |
| Calibrate | `Target` (`Step`, `Span`), `Split`, `replay`, `Replay` |
| Decide | `critical_ratio`, `order_up_to`, `settle`, `Settlement` |

A model reads only its window, so a later value cannot change a point. A threshold reads
only the scores that are known at its origin. Scores, quantiles and diagnostics come from
conformalforecast.

`rolling_forecasts` runs `StatsForecast` objects as they are, and any subclass of
`Forecaster`. `Forecasts.from_cv` reads the cross-validation frame of a Nixtla library.

[Architecture](https://github.com/Vzlentin/calibre/blob/main/docs/architecture.md) has the shapes and the modules.

## Benchmarks

`bench/` is a folder of scripts. They load benchmark data from a directory and run
benchmark protocols. They import the installed `calibre`.
Raw data is not in Git. The [benchmark guide](bench/README.md) covers nine additional
datasets, their versions, data-use conditions, and a runnable forecaster example.
All downloaded files and forecast outputs stay in ignored `data/`.

```sh
uv run --no-dev python bench/download_benchmarks.py etth1
```

| Script | Content |
|---|---|
| `download_benchmarks.py` | Pinned, hash-verified downloads for nine datasets; no preparation or cache |
| `lstnet.py` | `load(path)`: Electricity, Traffic, Solar-AL or Exchange matrix with positional dates |
| `ett.py` | `load(path)`: hourly ETTh1 oil temperature |
| `beijing.py` | `load(path)`: station PM2.5, causal forward-fill and original-target mask |
| `bike.py` | `load(path)`: hourly rentals, causal forward-fill and original-target mask |
| `favorita.py` | `load(path)`: store-family sales, causal forward-fill, attributes and hierarchy |
| `freshretail.py` | `load(path)`: store-product sales, censoring, attributes and hierarchy; needs pyarrow |
| `vn2.py` | `read(path)`: weekly VN2 sales with the out-of-stock mask, attributes, starting stock and the VN2 protocol |
| `m5.py` | `read(path)`: daily M5 sales and their attributes. `LEVELS` gives the 42,840 standard series |
| `gate_m5.py`, `gate_vn2.py` | The speed and cost gates of the architecture document |

## Documents

- [Architecture](https://github.com/Vzlentin/calibre/blob/main/docs/architecture.md): modules, array contracts, and costs.
- [Semantics](https://github.com/Vzlentin/calibre/blob/main/docs/semantics.md): the calibration rules.

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build --no-sources
```

See [Contributing](https://github.com/Vzlentin/calibre/blob/main/CONTRIBUTING.md).

## License

[MIT](https://github.com/Vzlentin/calibre/blob/main/LICENSE)
