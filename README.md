# Calibre

Calibre is a demand planning engine that combines probabilistic forecasting
(`statsforecast`, `mlforecast`, `neuralforecast`), conformal prediction
intervals, and ordering policies (e.g. newsvendor) into a single backtestable
pipeline. Benchmarks include the VN2 inventory challenge and adaptive
conformal inference (ACI) parity runs.

## Setup

Requires Python `>=3.11` and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Commands

Always run Python tooling through `uv`.

| Task                | Command                                            |
| ------------------- | -------------------------------------------------- |
| Run all tests       | `uv run pytest`                                    |
| Run a single test   | `uv run pytest tests/test_engine.py::test_name`    |
| Lint                | `uv run ruff check .`                              |
| Format              | `uv run ruff format .`                             |
| Type check          | `uv run mypy calibre/`                             |
| Download VN2 data   | `uv run python benchmarks/vn2/download_vn2_data.py`|
| Run VN2 benchmark   | `uv run python benchmarks/vn2/run_benchmark.py`    |
| Run ACI parity      | `uv run python benchmarks/cp/aci/run_aci_parity.py`|

## Repository structure

```
calibre/
├── contracts/      # Forecast frame schema and shared dtypes
├── engine/         # Backend engine, ledgers, scoring
├── conformal/      # ACI / MSCP / split conformal policies
├── models/         # Adapters: statsforecast, mlforecast, neuralforecast
├── ensemble/       # Ensemble combinators (e.g. median)
├── order/          # Ordering policies (newsvendor, RS, RSS)
├── pipeline/       # Loading, task building, end-to-end runner
├── tasks/          # ForecastTask, TuningTask
└── metrics.py      # MAE / RMSE / SMAPE / WAPE
benchmarks/
├── vn2/            # VN2 inventory challenge benchmark
└── cp/aci/         # Adaptive conformal inference parity benchmark
tests/              # pytest suite
.github/workflows/  # CI: ruff, mypy, pytest
```