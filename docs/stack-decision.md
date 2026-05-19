# Calibre Execution and Tuning Stack Decision

Status: implemented
Decision date: 2026-05-19

## Decision

Replace the current Fugue-based execution path and sequential Optuna tuning loop with
Ray Core for execution fan-out and Ray Tune with OptunaSearch plus ASHA for HPO. Keep
pandas, Parquet, fsspec, ForecastTaskRef URI materialization, FastAPI, SQLAlchemy,
Alembic, MLflow, and Prometheus.

This is a targeted scheduler migration, not a platform rewrite. The model adapters in
`calibre/forecasting/{stats,ml,neural}forecast_adapter.py` remain backend-blind.
Conformal calibration remains driver-owned because it is sequential mutable state.
The VN2 cost gate remains exact: `benchmarks/vn2/config/winning.yaml` must reproduce
`total_cost = 4992.20` rounded to cents at every milestone.

## Implementation Evidence

The execution layer now schedules Calibre's natural unit of work directly instead of
adapting it into a relational transform:

- `calibre/execution/backend.py` uses explicit `backend`, `ray_address`,
  `ray_threshold`, and `max_concurrency` execution options.
- Local-scope fan-out materializes `ForecastTaskRef` instances from URI-backed task
  payloads and submits Ray Core tasks only when the selected backend and task count
  require it.
- The local fast path runs in-process below the configured Ray threshold, which keeps
  smoke tests, health checks, and small fixtures free of Ray startup cost.
- Global-scope models always run in-process on the driver.
- The driver still owns conformal interval application, ordering policy, ledger
  appends, and result aggregation.

The tuning layer now delegates trial scheduling to Ray Tune while preserving the Optuna
search-space contract:

- `calibre/tuning/task.py` keeps `search_space: Callable[[optuna.Trial], dict]`.
- `calibre/tuning/optimizer.py` uses Ray Tune `Tuner`, `OptunaSearch`, and
  `ASHAScheduler`.
- Trial trainables stream completed origins through the execution backend and report
  pruning metrics only between origins.
- Trial CPU budgets cap UID fan-out and common library thread counts.

The CLI and packaging expose the Ray/local scheduler fields directly:

- `calibre/cli/config.py` parses `execution.backend`, `execution.ray_address`,
  `execution.ray_threshold`, and `execution.max_concurrency`.
- `calibre/cli/commands.py` constructs a `BackendEngine` from those explicit options.
- `pyproject.toml` includes Ray as the scheduler runtime and no longer exposes legacy
  scheduler extras.

The data hand-off layer is worth keeping:

- `calibre/core/forecast_task.py:40-58` writes task history and future covariates to
  Parquet URIs.
- `forecast_task.py:70-82` materializes those URIs back into a `ForecastTask`.
- `forecast_task.py:12-14` caches Parquet reads per process with `lru_cache`, which
  maps cleanly to worker processes.

## 1. Execution Layer

### Recommendation

Use Ray Core for per-series fit and predict fan-out. Keep global models and small local
runs in-process.

Ray fits Calibre's execution shape better than Fugue, Dask, Spark, or a bare process
pool because Calibre's unit of work is already a Python object plus URI references, not
a relational DataFrame transform. Ray can schedule those Python tasks directly across
cores or nodes, and the same scheduler can also run HPO trials. That removes the current
Fugue-only artifacts: base64 config columns, schema strings, dispatch DataFrames, and
global-model detours.

### Local-scope fan-out

For local-scope models, `BackendEngine` should build one `ForecastTaskRef` per uid and
origin-independent model configuration. At each origin:

- If the local task count is below the fast-path threshold, run the existing fit and
  predict loop in-process.
- Otherwise, submit one Ray task per uid for that origin.
- Each Ray worker materializes the `ForecastTaskRef`, filters `history[ds] < origin`,
  resolves the backend-blind adapter, fits, predicts, finalizes the forecast frame, and
  returns a pandas DataFrame.
- The driver concatenates returned frames, normalizes dtypes, applies conformal intervals,
  applies ordering policy if configured, and appends to the ledger.

This keeps all distributed-framework imports out of model adapters. Workers call the
same adapter registry as the local path.

### Global-scope execution

Global-scope models run in-process on the driver. A global LightGBM model fits once on
the full panel; it is not a per-uid map operation. Sending one global task to a worker
adds scheduling, serialization, logging, and lifecycle cost without creating useful
parallelism. Distributed LightGBM is a separate model-training concern and is skipped
until a real global panel crosses the scale where single-process training is the
bottleneck.

The current `_run_global_distributed` branch should disappear during migration.

### Task hand-off

Use `ForecastTaskRef` as the worker argument. It contains small Python metadata and
URIs to Parquet data. Ray serializes that reference directly; large histories stay in
Parquet and are read by workers through the existing materialization path.

Do not move per-uid history frames into Ray object storage by default. Object storage is
useful when the same large immutable object is reused many times by many tasks. Calibre's
history hand-off is already URI-based, cloud-compatible, and resumable. For current VN2
scale, task refs are small enough to pass by value, while returned forecast frames are
small enough to collect on the driver. Use `ray.put` only after profiling shows that a
shared object is repeatedly serialized and is larger than the task-ref metadata.

### ExecutionOptions and engine resolution

Replace the current `ExecutionOptions.engine: Any` Fugue object with explicit scheduler
options:

- `backend`: `local`, `ray`, or `auto`; default `auto`.
- `ray_address`: optional Ray cluster address; absent means local Ray when needed.
- `ray_threshold`: default `10`; below this count, do not initialize Ray.
- `max_concurrency`: optional cap on concurrent uid tasks for a run or trial.
- `seed`, `freq`, and metrics fields remain conceptually unchanged.

The CLI config now uses `execution.backend: local | ray | auto` plus
`execution.ray_address`. `commands.py` passes explicit scheduler options into the
execution layer. Scheduler lifecycle belongs in the execution layer, not in model
adapters and not in the CLI command body.

### Cluster and worker lifecycle

Local CLI runs:

- If `backend=local`, never initialize Ray.
- If `backend=auto` and local task count is below `ray_threshold`, never initialize Ray.
- If Ray is needed and no address is provided, Calibre starts a local Ray runtime for
  the process and owns its shutdown at the run boundary.

Remote cluster runs:

- If `ray_address` is provided, Calibre connects to that cluster and does not own its
  lifecycle.
- KubeRay, ECS bootstrap scripts, or a platform job runner owns cluster startup and
  shutdown.
- The Ray runtime lifetime should be one Calibre run for CLI jobs and process-scoped for
  long-lived API workers.

This preserves small-run latency while giving cloud jobs a clear ownership boundary.

### Single-node fast path

Keep a pure in-process fast path for fewer than 10 local tasks. This path is not a
fallback for correctness; it is the expected path for smoke tests, health checks,
single-SKU debugging, and small customer fixtures. It avoids Ray import, runtime startup,
task serialization, and dashboard overhead when the work is too small to amortize them.

### BackendEngine API

`BackendEngine.execute(tasks, actuals, origins) -> BackendResult` should survive. It is
the public batch API used by CLI, API, tests, and benchmarks.

Add a separate streaming origin iterator for tuning and pruning. The tuning layer needs
per-origin intermediate metrics; existing callers need a completed `BackendResult`.
Changing `execute()` into a generator would create avoidable migration churn.

### Why not the alternatives?

ProcessPoolExecutor is the right mental model for the fast path but not the cloud
scheduler. It has no cluster story, no dashboard, no HPO scheduler, and no resource
budgeting across trials.

Dask Distributed can submit Python futures and would remove some Fugue schema overhead,
but it does not solve HPO as directly. Calibre would still need a custom ask/tell Optuna
or pruning orchestrator to coordinate trial parallelism, early stopping, and nested
per-uid execution. That is more bespoke scheduler code than Ray Tune.

Spark is the wrong default for this workload. It is strong for SQL, large shuffles, and
JVM-heavy ETL. Calibre needs Python model fits per uid, conditional HPO, and low-overhead
single-node operation. Spark would keep the schema and pandas-UDF tax that Fugue already
exposes.

Keeping Fugue is not justified. Its portability across Dask and Spark is no longer a
constraint, and its abstraction forces non-domain code into `backend.py`.

## 2. Tuning Layer

### Recommendation

Use Ray Tune with OptunaSearch and ASHAScheduler.

The hard requirements are conditional Optuna search spaces, parallel trials, early
stopping between origins, resource budgeting, and MLflow tracking. Ray Tune covers the
scheduler side; OptunaSearch preserves the current search-space API.

### Preserving `TuningTask.search_space`

Keep `TuningTask.search_space: Callable[[optuna.Trial], dict]` unchanged. This API is
load-bearing because Calibre search spaces can be conditional: one sampled value can
determine whether later parameters are sampled at all.

Use Ray Tune's OptunaSearch with the callable form of the search space. Current Ray
documentation supports Optuna define-by-run callables that receive an Optuna trial and
return suggested values. Do not convert these search spaces to Ray Tune's declarative
parameter dictionaries; that would break conditional sampling.

### Trial-level parallelism

Ray Tune should schedule trials as the top-level parallel unit. Each trial receives an
explicit CPU budget. For per-series tuning, the trial usually runs one series and should
not fan out further. For panel-level sweeps, the trial may use Ray Core inside its budget
to fan out uid work, but concurrency must be capped so one trial cannot consume the
whole cluster.

Set `max_concurrent_trials` from available CPUs and the configured CPU budget per trial.
Inside each trial, set model-level thread counts and Calibre uid concurrency to the same
budget. Ray resource requests are scheduling admission control, not hard CPU isolation,
so LightGBM, NumPy, Torch, and other libraries still need explicit thread controls.

### Early stopping and pruning

Prune only between origins. The evaluation loop should report a cumulative objective
after each completed origin. ASHA should use origin index as the monotonic progress
attribute, with `max_t = len(origins)` and a conservative grace period.

Default grace period: 8 origins for VN2 conformal/order-cost searches. This avoids
pruning during the conformal warmup period where early origins are not representative of
later inventory cost. Make the grace period configurable per `TuningTask`.

Do not prune mid-fit, mid-predict, mid-conformal update, or mid-ordering decision.
Conformal state must remain internally consistent for every reported origin.

### MLflow tracking

Keep MLflow as the experiment system.

- Keep `safe_log_metric` and `log_costs_dataframe` for non-HPO benchmark paths.
- Use Ray's MLflow logger or `setup_mlflow` for Ray Tune HPO paths.
- Use a remote MLflow tracking URI for multi-node Tune runs.
- Preserve parent/child run discoverability by tagging Tune trial runs with the parent
  Calibre run id and the Ray trial id.
- Log the best config, trial table, pruning summary, and cost artifacts as MLflow
  artifacts.

Ray's MLflow callback logs from the driver, not from the trainable. If a trainable must
log custom artifacts directly, use Ray's MLflow setup helper inside that trainable rather
than calling plain MLflow APIs without an active session.

### RunStore integration

Keep `RunStore` as run-level state, not as a trial database. The current `RunStore`
contract in `calibre/api/run_store.py:24-31` is about creating, queueing, retrieving,
and running backtest jobs. The SQL implementation records status, row counts, errors,
and artifact pointers. That is the right boundary.

For HPO:

- `RunStore` records the parent run, status, artifact pointers, best config pointer,
  Tune experiment pointer, and Optuna study name or storage URI.
- Ray Tune and Optuna own trial-level persistence and resumption.
- MLflow owns experiment metrics, params, and artifacts.
- `SqlConformalStateStore` remains for resumable backtests, not for sharing mutable
  conformal state across parallel trials.

On resume, a queued HPO run should recover the same Tune experiment directory and Optuna
study identity, then continue unfinished trials. A completed best config is written back
as a run artifact and can be used by normal `BackendEngine.execute()` runs.

### Nested parallelism prevention

Use one scheduler: Ray. Do not run Fugue, Dask, Spark, joblib, or ThreadPoolExecutor
inside Ray Tune trials.

Per trial:

- Request an explicit CPU budget.
- Cap uid fan-out to that budget.
- Set model library thread counts to fit that budget.
- Keep global models single-trial and in-process unless a measured global-training
  bottleneck appears.
- Disable Ray initialization inside code paths that are already running in a Ray worker
  except for submitting nested Ray tasks under the same cluster and resource budget.

## 3. Scheduler Depth

| Layer | What it means | Verdict | Why |
|-------|---------------|---------|-----|
| 0 | Execution framework for per-uid fan-out | Do: Ray Core | Calibre's dominant execution unit is a picklable Python task plus Parquet URI references. Ray schedules that directly across cores or nodes and removes Fugue's schema/config dispatch layer. |
| 1 | Tuning framework for HPO | Do: Ray Tune + OptunaSearch + ASHA | Trial parallelism and origin-level pruning are the biggest runtime gap. OptunaSearch preserves Calibre's conditional `Callable[[optuna.Trial], dict]` search spaces. |
| 2 | Data loading with parallel IO and column pruning | Defer | `fsspec` plus pandas/pyarrow already handles object-store Parquet. Ray Data is useful if IO becomes the bottleneck, but VN2 and near-term panels are model-fit bound, not scan bound. |
| 3 | Stateful actors for caching or shared mutable state | Defer | Conformal runtime state is sequential and should stay on the driver. Named actors may help future multi-tenant cache sharing, but they add lifecycle complexity before there is a measured need. |
| 4 | Distributed model training | Skip | Global LightGBM is one model over the panel and is not the current bottleneck. Distributed LightGBM/XGBoost adds network coordination and failure modes that are unjustified below multi-million-row, multi-minute training jobs. |
| 5 | Serving replacement or execution colocation | Skip | FastAPI already exposes the job surface. Calibre runs batch backtests and planning jobs, not low-latency online inference that needs Ray Serve. |
| 6 | State store replacement | Skip | SQLAlchemy, Alembic, and Postgres are correct for durable business state. Ray checkpoints and Tune trial state are not replacements for run records, idempotency, artifact pointers, or conformal state snapshots. |
| 7 | Orchestration replacement | Skip | CLI, ECS jobs, and K8s jobs match the current flat workflow. KubeRay can own Ray cluster resources, but Prefect/Airflow/Ray Workflows would add DAG machinery Calibre does not need. |

## 4. Data Layer

### fsspec

Keep and extend fsspec. It is already the right boundary for local, S3, Azure, and GCS
URIs. The cloud extras in `pyproject.toml` map cleanly to the backend filesystems.
Replacing it would break the current object-store story without solving the scheduler
problem.

### In-memory frames

Keep pandas as the in-memory frame format. The forecasting adapters, metrics, ordering
logic, and ledger code already speak pandas. Per-uid task frames and returned forecast
frames are small. Moving to Polars, Arrow tables, Ray Data, or Spark DataFrames would
push conversion work into every adapter and validator before a measured bottleneck
exists.

Revisit Polars or Arrow-native internals only if profiling shows pandas ledger
operations or dtype coercion dominating large-panel runs.

### Serialization

Keep Parquet for durable task and ledger serialization. It is columnar, compressed,
cloud-friendly, and already backed by `pyarrow`. Arrow IPC/Feather is faster for local
same-machine interchange, but Calibre needs stateless containers and object-store URIs,
not only local shared memory. Plasma/ObjectRef transfer is optional for measured large
shared objects, not the default task protocol.

### ForecastTaskRef

Keep `ForecastTaskRef` and remove the Fugue-specific dispatch wrapper around it.

`ForecastTaskRef` is the correct abstraction because it is small, picklable, durable,
and cloud-native. It also makes worker retries practical: a retried worker can
materialize the same URI-backed task without relying on driver memory.

Potential extension: add a manifest form for large panels so one object-store prefix can
describe many uid refs and checksums. Do not replace the current URI materialization
contract.

## 5. Observability

### Dashboard

Use separate surfaces with shared run identifiers:

- Ray Dashboard for scheduler state, task durations, worker CPU/memory, failed tasks,
  and Tune trial progress.
- MLflow for experiment parameters, trial metrics, artifacts, and best configs.
- Existing FastAPI and CLI logs for Calibre job status and user-facing errors.

A forced "single pane" would hide the different failure modes. Scheduling failures,
model quality, and business cost are different questions.

### Experiment tracking

Keep MLflow. It is already integrated in benchmark utilities and tests. W&B would add a
second experiment system without replacing a broken one. Ray's native result directory
is useful for Tune resumption but is not enough as the experiment UI and artifact system.

### Metrics

Keep Prometheus for Calibre business and operational metrics. Add Ray's exported
Prometheus metrics when Ray is enabled. Calibre metrics such as forecast duration,
conformal coverage, and order cost remain application-level metrics; Ray metrics explain
cluster utilization and scheduling behavior.

### Logging and tracing

Use structured JSON logs everywhere, including workers. Required fields should include
`run_id`, `trial_id`, `origin`, `unique_id`, `model_name`, `phase`, and duration.

Defer distributed tracing. The current workload is a batch pipeline with clear origin
and uid boundaries. Tracing becomes useful when Calibre adds long-lived, multi-service,
latency-sensitive workflows.

## 6. Packaging

### Extras

Recommended dependency shape after migration:

- Core: keep NumPy, pandas, pyarrow, pyyaml, fsspec, FastAPI, SQLAlchemy, Alembic,
  psycopg, prometheus-client, statsforecast, optuna, and uvicorn.
- Remove from core: `fugue`.
- Legacy optional scheduler extras were removed once migration tests were green.
- Add `ray = ["ray[default,tune]>=2.38,<3"]`.
- Keep `cloud`, `s3`, `azure`, `gcs`, `ml`, `neural`, `xgboost`, `benchmarks`, and
  `dev` extras.
- Keep `optuna-integration[mlflow]` only until old Optuna MLflow callback users are
  removed.

### Slim and full images

Keep the slim/full Docker split.

| Image | Contents | Use |
|-------|----------|-----|
| Slim | Core Calibre, FastAPI, SQL state store, fsspec, pandas/pyarrow, statsforecast, Prometheus client, Ray runtime | API workers, health checks, small CLI runs, Ray worker base image |
| Full | Slim plus `ml`, `neural`, `benchmarks`, cloud filesystem extras, MLflow tooling | VN2 benchmarks, HPO jobs, LightGBM/XGBoost/neural experiments |

Image size is not a decision constraint, so prefer operational consistency over shaving
Ray or ML libraries out of images that need them.

### Databricks compatibility

Do not keep Spark/Fugue solely for Databricks. The Databricks path should be:

- Small and medium panels: run Calibre on the Databricks driver as a normal Python job
  with object-store URIs.
- Large panels: run Ray on a managed Ray/KubeRay cluster adjacent to the lakehouse and
  point Calibre at that Ray address.
- Notebook workflow: use Databricks notebooks to prepare configs, launch Calibre jobs,
  and inspect artifacts, not to force the execution backend through Spark.

This keeps the core engine consistent across ECS, K8s, local CLI, and notebooks.

### Version pinning

Use a broad package constraint and an exact lock:

- `pyproject.toml`: `ray[default,tune]>=2.38,<3`.
- `uv.lock`: exact tested Ray patch version.
- KubeRay Helm/operator: pin to a version compatible with the locked Ray version.
- Upgrade Ray and KubeRay together on a scheduled cadence after running the VN2 cost gate.

Do not use Ray versions between 2.11.0 and 2.37.0 for KubeRay deployments because the
current KubeRay upgrade guide calls out a RayJob readiness/liveness bug in that range.

## 7. Staged Migration Sequence

| Phase | Entry criteria | Work | Exit criteria | Rollback plan | Estimate |
|-------|----------------|------|---------------|---------------|----------|
| 0. Baseline and acceptance | This document reviewed; current worktree clean except planned docs | Record baseline commands and expected VN2 winning cost; agree config fields and API compatibility | Baseline run record shows `benchmarks/vn2/config/winning.yaml` returns `total_cost = 4992.20` rounded to cents; no source changes yet | No rollback needed | 0.5 day |
| 1. Ray Core execution | Phase 0 complete; baseline cost recorded | Replace Fugue dispatch with Ray Core local fan-out; keep global in-process; preserve `BackendEngine.execute`; add fast path and Ray execution config | Unit and integration tests green; ruff and mypy green; no active runtime references to the legacy scheduler path remain in `calibre` or `tests`; VN2 winning cost remains `4992.20` | Revert execution migration commit; old scheduler path was isolated to backend, CLI engine resolution, config, and tests | 2 days |
| 2. Ray Tune HPO | Phase 1 complete; Ray execution stable locally | Add streaming origin evaluation for tuning; move `optimize_task` to Ray Tune with OptunaSearch and ASHA; add trial budgets and grace period | HPO tests show conditional search spaces work, at least one bad trial prunes after grace period, no nested oversubscription in resource tests; VN2 winning cost remains `4992.20` | Revert tuning migration; keep Ray Core execution | 2 to 3 days |
| 3. Observability and persistence | Phase 2 complete; MLflow remote URI available | Wire Tune outputs to MLflow; record Tune/Optuna pointers and best config artifacts through RunStore; document resume path | RunStore tests cover HPO artifact pointers and resume metadata; MLflow trial runs are discoverable under parent run id; VN2 winning cost remains `4992.20` | Disable HPO resume metadata and fall back to MLflow artifacts while keeping execution | 1 day |
| 4. Packaging and deployment | Phase 3 complete; deployment image build available | Remove Fugue/Dask/Spark deps; add Ray extra; update slim/full image install sets; pin Ray/KubeRay versions | Fresh env installs; Docker slim/full build; API health check; local Ray smoke; VN2 winning cost remains `4992.20` | Restore dependency set from previous lockfile and image definitions | 1 day |
| 5. Cleanup and docs | Phase 4 complete | Remove dead Fugue tests/docs; update deployment docs and examples; add troubleshooting notes for Ray startup and Windows dev | Docs match code; no stale Dask/Spark config examples; final full test suite green; VN2 winning cost remains `4992.20` | Revert docs cleanup only | 0.5 day |

Total expected effort: 6 to 7 working days.

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Framework setup cost per CLI invocation | Medium | Medium | Keep `ray_threshold=10` fast path. Do not initialize Ray for health checks, one-off single-series runs, or tiny fixtures. Own local Ray shutdown at CLI run boundary. |
| Early stopping prunes good trials because origin ordering is non-stationary | Medium | High | Report metrics only after completed origins; use conservative grace period of 8 origins for VN2; compare ASHA results against a no-pruning sample before accepting HPO migration; allow disabling pruning per task. |
| Conditional search spaces break under the chosen scheduler | Low | High | Keep OptunaSearch callable search spaces; add tests where early sampled values suppress later parameters; pin Ray and Optuna versions; do not convert to Ray declarative search spaces. |
| Memory pressure on large panels | Medium | High | Keep URI-backed `ForecastTaskRef`; cap uid concurrency; use streaming ledger output for panels above a documented threshold; avoid putting full panels into Ray object storage by default; profile driver ledger growth. |
| Dev experience on actual platforms, especially Windows | Medium | Medium | Keep local fast path independent of Ray; run Ray tests primarily on Linux CI; document that Windows Ray support is for local development only and multi-node Ray should run on Linux containers/KubeRay; keep path handling URI-based. |

## References Checked

- Ray Core tasks and object refs: https://docs.ray.io/en/latest/ray-core/tasks.html
- Ray resource scheduling: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
- Ray Tune OptunaSearch: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html
- Ray Tune ASHA scheduler: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html
- Ray Tune MLflow callback: https://docs.ray.io/en/latest/tune/api/doc/ray.air.integrations.mlflow.MLflowLoggerCallback.html
- Ray metrics and Prometheus: https://docs.ray.io/en/latest/cluster/metrics.html
- Ray on Kubernetes and KubeRay: https://docs.ray.io/en/latest/cluster/kubernetes/
- KubeRay upgrade guide: https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/upgrade-guide.html
- Ray Windows support notes: https://docs.ray.io/en/master/ray-overview/installation.html#windows-support
- Dask delayed best practices: https://docs.dask.org/en/latest/delayed-best-practices.html
- Dask futures: https://docs.dask.org/en/stable/futures.html
- Spark grouped pandas UDF: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html
- Ray Data Parquet loading: https://docs.ray.io/en/latest/data/loading-data.html
- pandas Parquet IO: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html
- Apache Arrow Feather/IPC: https://arrow.apache.org/docs/3.0/python/feather.html
- Optuna RDB storage and resume: https://optuna.readthedocs.io/en/v3.0.3/tutorial/20_recipes/001_rdb.html
- Optuna pruning: https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html
