# Calibre

Calibre is a demand-planning engine that combines probabilistic forecasting
(`statsforecast`, `mlforecast`, `neuralforecast`), conformal prediction
intervals, and ordering policies (newsvendor, reorder-point, periodic-review)
into a single backtestable pipeline. Benchmarks include the VN2 inventory
challenge and adaptive conformal inference (ACI) parity runs.

## Setup

Requires Python `>=3.11` and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra benchmarks
```

## Commands

Always run Python tooling through `uv`.

| Task | Command |
|---|---|
| Run all tests | `uv run pytest` |
| Run a single test | `uv run pytest tests/test_conformal.py::test_name` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type check | `uv run mypy calibre/` |

### CLI

```bash
# Health check
uv run calibre health

# Run a backtest from a YAML config
uv run calibre run --config benchmarks/vn2/config/smoke.yaml

# Validate a config without executing
uv run calibre validate --config my-config.yaml

# Run a sweep over a directory of configs
uv run calibre run-sweep --configs benchmarks/vn2/config/
```

### Benchmarks

```bash
# Download VN2 data
uv run python benchmarks/vn2/download_vn2_data.py

# Run VN2 benchmark
uv run python benchmarks/vn2/run_benchmark.py

# Run VN2 winning config
uv run calibre run --config benchmarks/vn2/config/winning.yaml

# Run ACI parity
uv run python benchmarks/cp/aci/run_aci_parity.py
```

## API

Calibre exposes a FastAPI service for programmatic access.

```bash
uv run uvicorn calibre.api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | `GET` | Liveness probe |
| `/metrics` | `GET` | Prometheus metrics |
| `/forecasts` | `POST` | Synchronous forecast + order decision (≤30 SKUs, <60s) |
| `/backtests` | `POST` | Asynchronous backtest job (returns `run_id`) |
| `/runs/{run_id}` | `GET` | Poll run status and artifact pointers |

When `CALIBRE_DATABASE_URL` is set, run metadata and conformal calibration state
are persisted in Postgres. See [`docs/deployment.md`](docs/deployment.md) for
Terraform, AWS Batch, Azure Container Instances, and Databricks setup.

## Repository structure

```
calibre/                 # Python package
├── api/                 # FastAPI routes and schemas
├── cli/                 # CLI entrypoint, config loader, commands
├── conformal/           # ACI / MSCP / split conformal policies
├── contracts/           # Forecast frame schema and shared dtypes
├── core/                # ForecastFrame, ForecastTask, metrics, tracing
├── engine/              # Backend engine internals
├── ensemble/            # Ensemble combinators (e.g. median)
├── evaluation/          # Scoring and metric computation
├── eval/                # Legacy evaluation helpers
├── execution/           # BackendEngine, ledger, dataset registry, I/O
├── features/            # Feature engineering for global models
├── forecasting/         # Adapter registry + model adapters
│   └── features/        # Forecasting-specific feature transforms
├── models/              # Model-specific logic and wrappers
├── orchestration/       # Pipeline orchestration helpers
├── order/               # Ordering policies (newsvendor, RS, RSS)
├── ordering/            # Order policy protocols and implementations
│   └── simulation/      # Ordering simulation helpers
├── pipeline/            # Loading, task building, end-to-end runner
├── simulation/          # Demand and lead-time simulators
├── storage/             # Postgres state store, Alembic migrations
│   └── migrations/      # Alembic revision scripts
├── tasks/               # ForecastTask, TuningTask
└── tuning/              # Optuna-based hyper-parameter tuning

benchmarks/              # Benchmark suites
├── vn2/                 # VN2 inventory challenge
└── cp/aci/              # Adaptive conformal inference parity

tests/                   # pytest suite
docs/                    # Deployment guides
infra/                   # Terraform modules
scripts/                 # Databricks notebooks, jobs
.github/workflows/       # CI: ruff, mypy, pytest
```
