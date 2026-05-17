# Calibre — Single-Tenant MVP Core

**Goal.** Take Calibre from a research engine to a deployable single-tenant cloud service: CLI-driven, containerised, cloud-storage-aware, API-exposed, with persistent state in Postgres. One paying customer must be demoable end-to-end against `data/vn2/`.

**Non-goals.** Multi-tenancy, RLS, white-box packaging, BYOI OIDC, managed Dask/Spark clusters, model registry, real-time inference — all deferred to a later roadmap.

## Context

Calibre at `/home/vzl/Val/calibre` is a Python 3.11+ demand-planning engine: probabilistic forecasting (`statsforecast`, `mlforecast`, `neuralforecast`) + conformal prediction intervals + ordering policies (newsvendor, reorder-point, periodic-review), exercised through a walk-forward `BackendEngine` that partitions per `unique_id` through Fugue. Today: 341 tests green, no CLI, no containers, no DB, in-memory ledgers, local file IO only.

**Architecture is already Protocol-decomposed and stable** (see `calibre/conformal/protocols.py`, `calibre/order/protocols.py`, `calibre/contracts/forecast_frame.py`). The single-tenant MVP is packaging + persistence + dispatch over that core — no algorithmic rewrites.

**Already cloud-clean** (no change): `calibre/ordering/`, `calibre/order/`, `calibre/forecasting/`, `calibre/features/`, `calibre/ensemble/`, `calibre/simulation/`, `calibre/eval/`, `calibre/contracts/`, `calibre/core/forecast_task.py`.

**Operational invariants** (violating any silently produces NaN or biased output):
1. `BackendEngine(freq=...)` must match panel's day-of-week anchor. VN2 = `W-MON`. Bare `"W"` = Sunday.
2. Use `np.fmax`, **never** `np.maximum`, when combining mlforecast targets (mlforecast rejects null `y`).
3. Drivers set `origin = last_observed_ds + 1 step`. Engine's `< origin` filter is strict.
4. Cumulative conformal: `WARMUP_ORIGINS ≥ K + ⌈1/α⌉ − 1` per series, else orders are NaN.
5. Quantile/interval columns: use `quantile_column(p)`, `is_quantile_column(name)`, `interval_column_names(α)` from `calibre/core/forecast_frame.py`. **No** `startswith("q_")` / `lo_` parsing.
6. Conformal calibration is sequential per series. Never parallelise origins within the same `unique_id`.

## Commands (project-wide)

| Task | Command |
|---|---|
| Install deps | `uv sync --extra dev --extra benchmarks` |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type check | `uv run mypy calibre/` |

**Never** invoke `python`, `pytest`, `ruff`, or `mypy` directly. Always `uv run`.

## Constraints (every phase)

- Don't add pydantic in Phase 0. Stdlib `dataclasses` + manual validation. Reconsider at Phase 4 (HTTP boundary).
- Don't change algorithmic code in `calibre/conformal/calibrators.py` beyond adding `set_state()`. Preserve `get_state()` logic exactly.
- Don't break in-memory ledger behaviour: `streaming_output=None` must preserve current `_frames` accumulator bit-for-bit (regression gate).
- Don't rename existing public symbols. Add new ones; deprecate only if they block streaming.
- Use `fsspec` for all new IO paths. Keep local paths backward-compatible.

## Target architecture (single-tenant)

```
┌────────────────────────────────────────────────────────────┐
│  Container: calibre-api (FastAPI + Uvicorn)                │
│   POST /forecasts   (sync, <60s, ≤30 SKUs)                 │
│   POST /backtests   (async, returns run_id)                │
│   GET  /runs/{id}   (status + signed artifact URLs)        │
│   In-process queue (APScheduler / BackgroundTasks)         │
└───────────────────────┬────────────────────────────────────┘
                        │
       ┌────────────────┴────────────────┐
       ▼                                  ▼
┌─────────────────┐              ┌─────────────────────┐
│ Postgres        │              │ Object store        │
│  - runs         │              │ (S3 / GCS / Azure   │
│  - conformal_   │              │  Blob / local fs    │
│    state (jsonb)│              │  via fsspec)        │
└─────────────────┘              │  ledger parquet     │
                                 │  raw inputs         │
                                 └─────────────────────┘
```

No Cloud Tasks, no Redis, no Celery, no K8s required. Single container + Postgres + object store.

---

## Phase 0 · Foundation — CLI + Config + State + Streaming

**Goal:** Calibre runnable from a CLI against a YAML config, with externalisable calibration state and streaming ledger writes. Pure refactor, no heavy deps.

### 0.1 CLI entrypoint
- New package `calibre/cli/`: `main.py`, `commands.py`, `config.py`
- `pyproject.toml` adds `[project.scripts]`: `calibre = "calibre.cli.main:app"`
- Stdlib `argparse` (no Typer dep)
- Subcommands: `calibre run --config <path>`, `calibre validate --config <path>`, `calibre health`, `calibre run-sweep --configs <dir>`

### 0.2 YAML config schema
- `calibre/cli/config.py` — `BackendConfig` frozen dataclass with nested sections
- `load_config(path) -> BackendConfig` with manual validation
- Pin `config_schema: "1.0"` field
- `calibre/execution/dataset_registry.py` maps `dataset.adapter` strings → `DatasetAdapter` classes (decorator registration, mirrors `calibre/forecasting/adapter_registry.py`)

```yaml
# example config
config_schema: "1.0"
dataset: {adapter: vn2, path: data/vn2, period: 8}
tasks: [{model: lgb_global, horizon: 3, config: {...}}]
conformal: {method: mscp, mode: cumulative, coverage: 0.9, protection_period: 3, calibration_window: 100}
ordering: {policy: periodic_review, coverage: 0.95}
origins: {start: 2024-01-01, end: 2024-12-31, freq: W-MON}
output: {ledger_path: results/vn2/ledger.parquet, order_ledger_path: results/vn2/orders.parquet, streaming: true}
execution: {engine: null, seed: 42}
```

### 0.3 Conformal state round-trip

```python
# calibre/conformal/protocols.py
class Calibrator(Protocol):
    def fit(self, scores: dict[str, list[float]]) -> None: ...
    def predict(self, alpha: float, partition: str = "__global__") -> float: ...
    def update(self, new_score: float, partition: str = "__global__") -> None: ...
    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...  # NEW

class Controller(Protocol):
    def observe(self, y_true, y_pred, h: int) -> None: ...
    def get_alpha(self) -> float: ...
    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...  # NEW

# calibre/conformal/runtime.py
class ConformalRuntime:
    @classmethod
    def from_state(cls, config: ConformalPolicyConfig, state_payload: str) -> "ConformalRuntime": ...
```
- Implement `set_state` on `RollingQuantileCalibrator`, `FixedAlphaController`, `AdaptiveAlphaController` — restore private fields currently emitted by `get_state` (`_scores`, `_alpha`, `_alpha_history`, `_error_history`).
- `from_state` calls `deserialize_calibration_state` (dead code at `calibre/conformal/__init__.py:29,63`), rehydrates `_issued_count`, delegates to calibrator/controller `set_state`.
- Round-trip test: run VN2 to origin 3, snapshot `CALIBRATION_STATE` from last resolved row, kill runtime, reconstruct via `from_state`, continue to origin 52, assert identical final output.

### 0.4 Streaming ledger writes

```python
# calibre/execution/ledger.py
class LedgerSink(Protocol):
    def append(self, df: pd.DataFrame) -> None: ...
    def close(self) -> None: ...

class ForecastLedger:
    def stream_to(self, path: str, *, partition_cols: list[str] | None = None) -> None: ...
    def append_streaming(self, df: pd.DataFrame) -> None: ...
    # existing to_df(), to_parquet() unchanged

# calibre/execution/backend.py
class BackendEngine:
    def __init__(
        self,
        freq: str = "W",
        metrics: list[Callable] | None = None,
        engine: Any = None,
        conformal_config: ConformalPolicyConfig | None = None,
        order_config: OrderPolicyConfig | None = None,
        streaming_output: str | None = None,        # NEW
        streaming_order_output: str | None = None,  # NEW
        seed: int | None = None,                    # NEW
    ) -> None: ...
```
- `stream_to(path)` opens lazy `pyarrow.parquet.ParquetWriter`; `append_streaming(df)` writes immediately, releases the frame.
- When `streaming_output` set, `execute()` writes per-origin and never accumulates `_frames`.
- Compaction fix: `update_resolved` today `pd.concat`s on every resolve (O(N²) at `calibre/execution/ledger.py:16-19` and `calibre/execution/backend.py:127,149`). With streaming, resolved rows write as `*.resolved.parquet` siblings, merged on read.
- Backward: `streaming_output=None` preserves current in-memory behaviour exactly.

### 0.5 Seed mechanism
- New `calibre/core/seeding.py`: `set_seed(seed)` sets numpy + python `random`, propagates frozen `Seed` through `ForecastTask.model_config` if not set.
- `BackendEngine(seed=...)` calls `set_seed` and forwards through adapter `model_config`.
- Centralise `optuna.samplers.TPESampler(seed=seed)` in `calibre/tuning/optimizer.py` (today wired inline at `benchmarks/vn2/run_benchmark.py:430,1133`).

### 0.6 Validation at engine boundary
- Call `validate_forecast_frame` at start of `BackendEngine.execute` on `actuals` (input) and on `origin_preds` before `ledger.append` (output). Failure raises with origin timestamp.

### 0.7 Migrate benchmarks to CLI
- One YAML config per benchmark script under `benchmarks/<name>/config/`.
- Replace bodies with thin wrappers invoking `calibre.cli.commands.run(config_path)` — preserves `python -m benchmarks.vn2.run_benchmark` for CI.
- Multi-config sweeps via `calibre run-sweep --configs <dir>`.
- Force benchmarks through `VN2DatasetAdapter` (today bypassed at `benchmarks/vn2/run_winning.py:175`).

### Phase 0 DoD
- `uv run calibre run --config benchmarks/vn2/config/winning.yaml` reproduces VN2 cost <€5,000.
- `uv run pytest` green plus new `tests/cli/`, `tests/test_state_resume.py`, `tests/test_streaming_ledger.py`.
- State round-trip: kill at origin 3, resume via `from_state` → byte-identical final ledger.
- Streaming output bit-for-bit matches in-memory ledger.

---

## Phase 1 · Cloud-Agnostic IO + Working Distributed Backend

**Goal:** every IO boundary `fsspec`-aware AND Fugue's `engine=` seam actually works under Dask/Spark, not just pandas. Single-tenant MVP still runs on one container, but the engine is correct under any backend so customers with larger SKU counts don't require a re-architecture.

### 1.1 fsspec deps + helpers
- `pyproject.toml`: `fsspec` in base. Extras: `s3=["s3fs"]`, `azure=["adlfs"]`, `gcs=["gcsfs"]`, `cloud=["s3fs","adlfs","gcsfs"]`. `moto[s3]` in dev extras for CI.
- New `calibre/execution/io.py`: `open_fs(uri)`, `resolve_path(uri)`.
- Replace `Path(path).exists()` with `fsspec.filesystem(...).exists(path)`. pandas already accepts URI strings for `read_csv`/`read_parquet`.
- VN2 download script gains `--target <uri>` flag, uses `fsspec.open(target, "wb")` instead of local-file writes.

### 1.2 `DatasetAdapter` Protocol + validation

```python
# calibre/execution/dataset.py
@dataclass(frozen=True)
class DatasetBundle:
    history: pd.DataFrame
    future_x: pd.DataFrame | None
    costs: dict[str, Any]
    hierarchy: pd.DataFrame | None
    censoring: pd.DataFrame | None

class DatasetAdapter(Protocol):
    def load(self, config: dict) -> DatasetBundle: ...
```
- `VN2DatasetAdapter` wraps existing VN2 loading.
- `calibre/execution/validation.py`: `validate_dataset_bundle(bundle)` — columns, dtypes, monotonic `ds` per `unique_id`, no nulls in key columns, `future_x` alignment, `censoring.in_stock` bool, cost coverage of every `unique_id`.
- Per-SKU cost loading: `load_costs(uri) -> dict[str, CostStruct]` from CSV/parquet keyed by `unique_id`; scalar `CostStruct` fallback for VN2 parity.

### 1.3 Streaming ledger over fsspec
- Phase 0's `stream_to(uri)` tested via `moto`-mocked S3.

### 1.4 Fix Fugue closure serialisation (`calibre/execution/backend.py`)
- Today `_process_partition` closes over `tasks_by_uid: dict` (broadcast antipattern, `calibre/execution/backend.py:164-194`). Under pandas this is fine; under Dask/Spark it ships the full task dict to every worker.
- Lift `_process_partition` to a module-level function.
- Refactor closure capture to `(uid, model_config, horizon, history_uri)` only; history is materialised to a fsspec URI once at `execute()` start. Workers read their slice on first call and cache per worker.
- Equivalent refactor for `_run_direct` global-model path: history materialised once, all global tasks share the URI.

### 1.5 `ForecastTask` URI materialisation

```python
# calibre/core/forecast_task.py — add method
class ForecastTask:
    def to_uri(self, base_uri: str) -> "ForecastTaskRef":
        """Write history parquet to base_uri/<uid>.parquet, return lightweight ref."""
        ...

# calibre/core/forecast_task_io.py — new module
@dataclass(frozen=True)
class ForecastTaskRef:
    unique_id: str
    model_config: dict
    horizon: int
    forecast_origin: pd.Timestamp | None
    history_uri: str

    def materialize(self) -> ForecastTask:
        """Read history parquet on the worker side, return concrete ForecastTask."""
        ...

# per-worker LRU cache keyed by (uid, history_uri)
```

### 1.6 Dask / Spark engine wiring (`calibre/cli/config.py`, `calibre/execution/backend.py`)
- Config `execution.engine: null | "dask" | "spark"`.
- `dask`: `execution.dask_address: "tcp://...:8786"` (or `null` for in-process `LocalCluster`); construct `DaskExecutionEngine`, pass to `BackendEngine(engine=...)`.
- `spark`: `execution.spark_session: {master, app_name, ...}`; construct `SparkExecutionEngine`.
- New optional extras:
  ```
  dask  = ["dask[distributed]"]
  spark = ["pyspark"]
  ```

### 1.7 Distributed verification (CI-enforced)
- `tests/integration/test_s3_ingestion.py` (`moto`-mocked S3): upload VN2 fixture, run `calibre run` with `s3://` URIs, assert identical results to local.
- `tests/integration/test_dask.py`: spin up Dask `LocalCluster`, run VN2 winning config, assert results within float tolerance of single-node pandas run. **Runs on every PR.**
- `tests/integration/test_spark.py` (`@pytest.mark.spark`, gated to nightly): same against `local[*]` Spark — keeps PR runtime sane.
- New benchmark config `benchmarks/vn2/config/winning_dask.yaml` reproduces VN2 result via Dask backend.

### Phase 1 DoD
- `tests/integration/test_s3_ingestion.py` (`moto`) green.
- `calibre run` with `s3://` / `abfs://` / `gs://` URIs produces identical results to local.
- `DatasetAdapter` validates all bundles; per-SKU cost loading tested.
- **`tests/integration/test_dask.py` green: `calibre run --config benchmarks/vn2/config/winning_dask.yaml` on `dask.distributed.LocalCluster` matches single-node pandas within float tolerance.** Nightly `tests/integration/test_spark.py` green against `local[*]`.
- `uv run mypy calibre/` clean; `uv run pytest -m "not slow"` <90s.

> **Out-of-scope for this MVP** (operations, not code): managed Dask/Spark cluster provisioning, autoscaling, prod-scale load testing on real-tenant traffic profiles, cluster cost dashboards. The single-tenant MVP runs on `LocalCluster` (or a single-process pandas backend) inside its container; the `execution.engine` config can later point at a managed endpoint with no code change.

---

## Phase 2 · Containerisation

**Goal:** one immutable image deployable to any container platform (Cloud Run / Container Apps / Fargate / k3s), runs any config from any URI.

### 2.1 Dockerfile
- Multi-stage: `python:3.11-slim` builder with `uv` + `uv sync --extra cloud --no-dev` → non-root `runtime` user.
- Entrypoint `["calibre"]`, default CMD `["health"]`, `PATH=/app/.venv/bin:$PATH`.

### 2.2 `.dockerignore`
- Excludes: `data/`, `mlruns/`, `lightning_logs/`, `results/`, `tests/`, `benchmarks/*/data`, `.git`, `.venv`, `__pycache__`, `*.ipynb`.

### 2.3 Image variants
- `calibre:slim` — no NeuralForecast (no torch). Target ~150MB.
- `calibre:full` — all backends. Target <600MB.

### 2.4 Health check
- `calibre health` returns 0 on import + version + dry-run schema validation against embedded fixture.
- k8s `livenessProbe: exec: ["calibre", "health"]`.

### 2.5 CI build
- GitHub Actions `docker-build` job per PR; smoke job `docker run calibre:full run --config /app/benchmarks/vn2/config/smoke.yaml` against image-baked fixture.

### 2.6 Deployment docs
- `docs/deployment.md`: K8s job manifest, Azure Container Instance, AWS Batch examples. Note: stateless container, no PVC required when output is `s3://`/`abfs://`.

### Phase 2 DoD
- `docker build .` succeeds.
- `docker run calibre:full run --config benchmarks/vn2/config/winning.yaml` reproduces VN2 result.
- CI green; both images pushed per PR.

---

## Phase 3 · Observability

**Goal:** every run emits structured logs + metrics a client's ops team wires into Grafana/Datadog/Azure Monitor without code changes.

### 3.1 Structured logging
- `calibre/core/logging.py` — stdlib `logging` with custom JSON formatter (no `structlog` dep).
- `setup_logging(level, format)` called from `calibre/cli/main.py`.
- Standard fields: `origin`, `model_name`, `unique_id`, `phase` (fit/predict/conformal/order), `duration_ms`.
- Add loggers in `BackendEngine.execute`, `ConformalRuntime.apply/observe`, adapter `fit/predict`, `cli.commands.run`. Convert existing logger at `calibre/evaluation/point_metrics.py:8`.

### 3.2 Metrics
- `calibre/core/metrics.py` — `prometheus_client`.
- Three series: `calibre_forecast_duration_seconds{model, phase}` histogram, `calibre_conformal_coverage_ratio{model, mode}` gauge, `calibre_order_cost{currency, dataset}` gauge.
- `metrics.serve(port)` starts Prometheus HTTP server in background thread; opt-in via `--metrics-port` flag.

### 3.3 Tracing hooks (no SDK)
- `tracing.span("backtest", origin=...)` context manager around per-origin loop; no-op when OTel SDK absent.

### 3.4 Replace prints
- 53 `print()` calls in `benchmarks/` → `logger.info()`.
- Drop `T20` per-file-ignore in `pyproject.toml:55`.

### Phase 3 DoD
- `calibre run --metrics-port 9090` + `curl localhost:9090/metrics` shows all three series.
- `tests/test_observability.py` asserts JSON log fields.

---

## Phase 4 · API Surface

**Goal:** expose engine over HTTP. One internal tenant. In-process queue for async backtests.

### 4.1 FastAPI app
- New `calibre/api/` package.
- Routes:
  - `POST /forecasts` — sync, <60s, ≤30 SKUs
  - `POST /backtests` — async, returns `run_id`
  - `GET  /runs/{id}` — status + signed URLs to artifacts
  - `GET  /healthz`, `GET /metrics`

```python
# calibre/api/main.py (sketch)
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from calibre.api.schemas import ForecastRequest, ForecastResponse, RunResponse
from calibre.api.runs import create_run, run_backtest_job, get_run

app = FastAPI(title="Calibre", version="0.1")

@app.post("/forecasts", response_model=ForecastResponse)
def forecasts(req: ForecastRequest):
    # build BackendConfig from req, call run_forecast() directly
    ...

@app.post("/backtests", response_model=RunResponse, status_code=202)
def backtests(req: ForecastRequest, bg: BackgroundTasks, idempotency_key: str | None = Header(None)):
    run = create_run(req, idempotency_key=idempotency_key)
    bg.add_task(run_backtest_job, run.id)
    return run

@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run_status(run_id: str):
    run = get_run(run_id)
    if not run:
        raise HTTPException(404)
    return run
```

### 4.2 In-process queue
- `APScheduler` or FastAPI `BackgroundTasks` for long backtests. No Cloud Tasks, no Redis, no Celery.
- Idempotency keys on `POST /backtests` (header `Idempotency-Key`).

### 4.3 Reconsider pydantic
- HTTP boundary justifies pydantic now. If adopted, generate models from the Phase 0 `BackendConfig` dataclass rather than duplicating.

### 4.4 Deploy
- Cheapest container platform (Cloud Run or k3s on OVH VPS). Target idle ≤ €60/mo measured over 7 days.

### Phase 4 DoD
- `POST /forecasts` returns calibrated forecasts for 30-SKU dataset in <60s.
- `POST /backtests` returns `run_id`; `GET /runs/{id}` returns `succeeded` with signed URLs to ledger parquet.
- Idle infra cost measured.

---

## Phase 5 · Persistence

**Goal:** Postgres stores runs + conformal state. Artifact data stays on object store.

### 5.1 SQLAlchemy 2.x models

```python
# calibre/storage/models.py (sketch — single tenant, no tenant_id yet)
class Run(Base):
    __tablename__ = "runs"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    idempotency_key: Mapped[str | None] = mapped_column(unique=True, nullable=True)
    config: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str]  # queued | running | succeeded | failed
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    finished_at: Mapped[datetime | None]
    error: Mapped[str | None]

class ConformalState(Base):
    __tablename__ = "conformal_state"
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    partition: Mapped[str] = mapped_column(primary_key=True)
    state: Mapped[dict] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now())

class ForecastPointer(Base):
    __tablename__ = "forecast_pointers"
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    kind: Mapped[str] = mapped_column(primary_key=True)  # ledger | order_ledger
    uri: Mapped[str]
    byte_size: Mapped[int]
```
- Tenants table omitted in MVP (single-tenant); add `tenant_id` column with default in the multi-tenant phase.

### 5.2 Alembic migrations
- Under `calibre/storage/migrations/`. Initial migration creates the three tables.

### 5.3 Repos + helpers
- `calibre/storage/postgres.py` — session management + repos (`RunRepo`, `ConformalStateRepo`).
- `calibre/storage/objstore.py` — `write_ledger_shard`, `read_run_artifacts`, signed URL generation via fsspec.

### 5.4 Wire `ConformalRuntime` to DB
- On `BackendEngine.execute` start with a `run_id`: load conformal state from `conformal_state` table, instantiate via `ConformalRuntime.from_state(...)`.
- After each origin's `observe`: persist via `get_state()`.

### 5.5 MLflow (optional)
- Self-hosted on container platform, object-store artifacts, Postgres backend. Reuse same Postgres or run a separate small instance.
- Point `MLFLOW_TRACKING_URI` at the containerised instance; no code change in benchmarks (`benchmarks/common/tracking.py` already env-driven).

### 5.6 Terraform
- One root module per provider (start with one — whichever the first paying customer uses): Postgres (Cloud SQL / Azure DB / RDS), object store bucket, container service, container registry, IAM.

### Phase 5 DoD
- `runs`, `conformal_state`, `forecast_pointers` tables exist with Alembic migration.
- Artifact parquet on object store; Postgres holds pointers + state.
- A run interrupted between origins resumes from DB state on retry → byte-identical to uninterrupted run.
- Terraform `apply` provisions everything on the chosen provider.

---

## Cost target (MVP, idle)

| Layer | Choice | Cost |
|---|---|---|
| Compute | Container platform (Cloud Run / k3s on VPS) | €5–15 (scale-to-zero) |
| DB | Postgres small instance, no HA | €10 |
| Object storage | S3/GCS/Azure Blob | €1–5 |
| Queue | In-process | €0 |
| MLflow (optional) | Self-hosted | €5–10 |
| Container registry | GAR/ACR/ECR | €0 (≤0.5GB free) |
| Monitoring | Cloud-native free tier + Sentry free | €0 |
| **Total** | | **~€30–60/mo idle** |

---

## End-to-end verification (run at every phase boundary)

```bash
uv run pytest
uv run mypy calibre/
uv run ruff check .
```

Phase-specific smoke:

| Phase | Smoke |
|---|---|
| 0 | `uv run calibre run --config benchmarks/vn2/config/winning.yaml` → cost <€5,000 |
| 1 | Same against `s3://moto-bucket/...`; plus `calibre run --config benchmarks/vn2/config/winning_dask.yaml` on Dask `LocalCluster` matches single-node pandas within float tolerance |
| 2 | `docker run calibre:full run --config /app/benchmarks/vn2/config/winning.yaml` |
| 3 | Same with `--metrics-port 9090`; `curl localhost:9090/metrics` shows three series |
| 4 | `POST /forecasts` 30-SKU payload → intervals; `POST /backtests` → poll until `succeeded`; signed-URL parquet matches local backtest |
| 5 | `terraform apply` provisions infra; backtest writes to managed Postgres + bucket; interrupted run resumes from DB state |

---

## Out of scope (deferred for later roadmap)

- Multi-tenancy + RLS + tenant context middleware
- API key issuance/rotation + JWT for human users + BYOI OIDC for white-box
- White-box packaging (Helm chart, Compose recipe, swappable queue backends)
- Managed Dask/Spark cluster operations, autoscale, 100-tenant load test (the *code* for distribution ships in Phase 1; only the *operations* of a managed cluster are deferred)
- Model artifact caching, shared Optuna storage, `calibre run-origin` orchestration CLI
- Full OpenTelemetry SDK, model registry, drift alerts
