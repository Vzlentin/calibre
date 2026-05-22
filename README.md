# Calibre

Calibre is a demand-planning engine that combines probabilistic forecasting
(`statsforecast`, `mlforecast`, `neuralforecast`), conformal prediction
intervals, and ordering policies (newsvendor, reorder-point, periodic-review)
into a single backtestable pipeline. Benchmarks include the VN2 inventory
challenge and adaptive conformal inference (ACI) parity runs.

The engine supports deployment-oriented workflows: deterministic session ids,
per-partition conformal state, restart-safe pending observations, live inventory
snapshots, global-model fan-out, joint model/conformal/ordering tuning, and
promotion what-if prediction overrides.

## Setup

Requires Python `>=3.11` and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra benchmarks
```

When using persistent API state, set `CALIBRE_DATABASE_URL` and apply the
Alembic migrations:

```bash
uv run alembic -c alembic.ini upgrade head
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

The current VN2 winning-config regression baseline is `total_cost=4992.20`.

## API

Calibre exposes a FastAPI service for programmatic access.

```bash
uv run uvicorn calibre.api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | `GET` | Liveness probe |
| `/metrics` | `GET` | Prometheus metrics |
| `/backtests` | `POST` | Asynchronous backtest job (returns `run_id`) |
| `/runs/{run_id}` | `GET` | Poll run status and artifact pointers |
| `/fit` | `POST` | Start a fit lifecycle and return `fit_id` + deterministic `session_id` |
| `/fits/{fit_id}` | `GET` | Poll fit lifecycle status |
| `/predict` | `POST` | Produce forecasts for a fit and origin, with optional `future_x_override` |
| `/calibrate` | `POST` | Apply session-keyed conformal calibration to a forecast frame |
| `/order` | `POST` | Convert calibrated forecasts into an order ledger |
| `/observe` | `POST` | Resolve actuals back into conformal state for the session |
| `/sessions/{tenant}/{uid}` | `GET` | Return state, last forecast, and open orders for a tenant/SKU |
| `/tune` | `POST` | Start multi-SKU HPO for model, conformal, and ordering configs |
| `/studies/{study_id}` | `GET` | Poll tuning status and per-SKU best candidates |

When `CALIBRE_DATABASE_URL` is set, run metadata and conformal calibration state
are persisted in Postgres. Conformal state is keyed by stable `session_id` and
partition, unresolved observations are buffered in `pending_observations`, and
multi-SKU HPO results are stored in `tuning_runs` for partial-completion resume.
See [`docs/deployment.md`](docs/deployment.md) for Terraform, AWS Batch, Azure
Container Instances, and Databricks setup.

### Tuning and What-Ifs

Tuning search spaces return a `TuningCandidate` with separate model, conformal,
and ordering config channels. `/tune` fans out across the requested SKU set,
persists each completed SKU candidate, and skips already-finished rows when a
study is resumed with the same session. `/predict` accepts
`future_x_override` as a `unique_id -> rows` mapping so planners can run
promotion or regressor scenarios without mutating the fit-time baseline.

### Execution Notes

`ForecastTask.task_group` lets the engine preserve grouped scheduling semantics.
Global models are grouped by model config and can fan out through Ray, fitting
one full-panel adapter per distinct global config. Inventory state can be
supplied through `SyntheticInventoryAdapter`, `SnapshotInventoryAdapter`, or a
client-provided ERP adapter implementation.

## Repository structure

```
calibre/                 # Python package
├── api/                 # FastAPI routes and schemas
├── cli/                 # CLI entrypoint, config loader, commands
├── conformal/           # Conformal calibration policies
├── core/                # ForecastFrame, ForecastTask, metrics, tracing
├── evaluation/          # Scoring and metric computation
├── execution/           # BackendEngine, ledger, dataset registry, I/O
├── forecasting/         # Adapter registry + model adapters
│   └── features/        # Forecasting-specific feature transforms
├── ordering/            # Order policy protocols and implementations
│   └── simulation/      # Inventory simulation (costs, rules, state)
├── storage/             # Postgres state store, Alembic migrations
│   └── migrations/      # Alembic revision scripts
└── tuning/              # Ray Tune + OptunaSearch hyper-parameter tuning

benchmarks/              # Benchmark suites
├── vn2/                 # VN2 inventory challenge
└── cp/aci/              # Adaptive conformal inference parity

tests/                   # pytest suite
docs/                    # Deployment guides
infra/                   # Terraform modules
scripts/                 # Databricks notebooks, jobs
.github/workflows/       # CI: ruff, mypy, pytest
```
