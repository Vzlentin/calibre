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

For a deployment that survives restarts and runs multiple workers, also set:

- `LIFECYCLE_STORE=sql` — the `/fit` lifecycle store (fit/tune records,
  session-owned conformal state) defaults to **in-memory** and is otherwise
  lost on restart and invisible across workers.
- `CALIBRE_ARTIFACT_URI` — base URI for fit-frame parquet artifacts and
  trusted server-owned model artifacts. Multi-host workers must point this at a
  **shared** object store (e.g. `s3://bucket/prefix`); a local path is
  single-host only (and warns).

## Commands

Always run Python tooling through `uv`.

| Task | Command |
|---|---|
| Run all tests | `uv run pytest` |
| Run a single test | `uv run pytest tests/conformal/test_conformal.py::test_name` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type check | `uv run ty check calibre/` |

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

### Dataset adapters

Built-in adapters in `calibre.execution` (shipped in the wheel; no `benchmarks`
import): `vn2` (weekly sales + optional master / in-stock) and `m5`
(`sales_train_{phase}.csv` + `calendar.csv`). M5 `unique_id` is
`f"{item_id}_{store_id}"`; `hierarchy` carries product/location taxonomy per
series.

```yaml
dataset:
  adapter: m5
  path: tests/fixtures/m5
  phase: evaluation   # optional; defaults to evaluation, falls back to validation
```

### Hierarchical Reconciliation

When a dataset supplies `hierarchy`, configure point reconciliation with:
`none`, `bottom_up`, `ols`, `wls_struct`, `mint_shrink`, `wls_var`, or `erm`.
Residual-backed strategies (`mint_shrink`, `wls_var`, `erm`) request
horizonless in-sample fitted values from the model adapter, keyed by
`(unique_id, ds, model_name)`, and pass them as explicit reconciliation context;
fitted values are not written as historical rows in the forecast-frame ledger.
`mint_cov` is not exposed because the full M5
lattice produces ill-conditioned covariance estimates. Reconciliation still
applies only to point forecasts before conformal calibration; coherent interval
or quantile reconciliation remains out of scope for this path.

For interim hierarchy-aware intervals, use the separate fused phase:

```yaml
hierarchical_intervals:
  method: nixtla_conformal
  coverage: 0.9
  strategy: bottom_up  # bottom_up, ols, wls_struct, mint_shrink, wls_var, erm
```

This path requires a dataset `hierarchy`, requests horizonless fitted values
keyed by `(unique_id, ds, model_name)`, and runs as
`Predict -> HierarchicalIntervals -> Order -> Commit`. It is mutually exclusive
with `conformal` and non-`none` point `reconciliation`, because Nixtla owns both
the coherent point output and the marginal conformal interval columns for that
run. The emitted bounds use Calibre's normal `lo_<coverage>` / `hi_<coverage>`
column contract for bottom and aggregate node rows. These are marginal
hierarchical conformal intervals: point forecasts are coherent, but published
per-node interval boxes are not additive bands and should not be described as
conditional coverage at a chosen hierarchy level.

### Benchmarks

```bash
# Download VN2 data
uv run python benchmarks/vn2/download_vn2_data.py

# Run VN2 benchmark
uv run python benchmarks/vn2/run_benchmark.py

# Run VN2 winning config (harness entrypoint)
uv run python -m benchmarks.vn2 --config benchmarks/vn2/config/winning.yaml

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
| `/fit` | `POST` | Start a fit lifecycle: history is ingested from `sales_uri` (parquet/SQL, with optional point-in-time `as_of`) and regressors from `future_x_uri`. Returns `fit_id` + deterministic `session_id`; eagerly fits to validate config (incompatible configs land `FAILED` rather than failing later at `/predict`) and persists a trusted server-owned model artifact when compatible |
| `/fits/{fit_id}` | `GET` | Poll fit lifecycle status |
| `/predict` | `POST` | Produce forecasts for a fit and origin, with optional `future_x_override` |
| `/calibrate` | `POST` | Apply session-keyed conformal calibration to a forecast frame |
| `/order` | `POST` | Convert calibrated forecasts into orders, persisted to the durable `orders` ledger keyed by `(session_id, unique_id, forecast_origin, model_name)` |
| `/observe` | `POST` | Resolve actuals back into conformal state for the session |
| `/sessions/{tenant}/{uid}` | `GET` | Return state, last forecast, and open orders (read from the `orders` ledger) for a tenant/SKU |
| `/tune` | `POST` | Start multi-SKU HPO for model, conformal, and ordering configs; history from `sales_uri`, realized actuals from `actuals_uri` |
| `/studies/{study_id}` | `GET` | Poll tuning status and per-SKU best candidates |

Sales ingestion goes through a `SalesAdapter` resolved by URI scheme:
`SnapshotSalesAdapter` reads a parquet/fsspec snapshot, while a `sql://` /
`db://` `sales_uri` reads the project's own Postgres `sales` table
(`SqlSalesAdapter`) — both honour point-in-time `as_of` semantics, mirroring the
inventory adapters.

When `CALIBRE_DATABASE_URL` is set, run metadata and conformal calibration state
are persisted in Postgres. Conformal state is keyed by stable `session_id` and
partition, unresolved observations are buffered in `pending_observations`, and
multi-SKU HPO results are stored in `tuning_runs` for partial-completion resume.
With `LIFECYCLE_STORE=sql`, the `/fit` and `/tune` lifecycle records and their
session-owned conformal state also persist. Fit-frame data planes are written as
parquet under `CALIBRE_ARTIFACT_URI`, and fitted model artifacts are written
under the same root using the native Nixtla persistence APIs. Model artifacts
are trusted server-owned files: requests never provide model bytes or arbitrary
artifact URIs, and `/predict` only loads artifacts addressed by server-computed
cache keys. `/order` writes durable rows to the `orders` table that `/sessions`
reads back, so the API survives restarts and multi-worker deployments. See
[`docs/deployment.md`](docs/deployment.md) for Terraform, AWS Batch, Azure
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
.github/workflows/       # CI: ruff, ty, pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the one-PR-per-item workflow,
and the four CI gates.

## License

Calibre is licensed under the [Apache License 2.0](LICENSE).
