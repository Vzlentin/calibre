# Calibre Execution + Tuning Stack Decision

> **Status:** accepted · **Date:** 2026-05-18
> **Author:** analysis session · **Canonical plan reference:** `~/.claude/plans/i-want-to-switch-polished-ocean.md`

---

## Executive Summary

Replace Fugue (execution fan-out) and sequential Optuna (HPO) with **Ray Core + Ray Tune
(OptunaSearch + ASHAScheduler)**. Keep everything else: pandas, parquet, fsspec, SQLAlchemy,
FastAPI, MLflow, Prometheus, ForecastTaskRef URI hand-off.

The canonical plan (`i-want-to-switch-polished-ocean.md`) is correct in its overall direction.
This document agrees with it on the fundamental choice and sharpens several implementation
specifics where the plan left gaps: the `execute()` API contract, the single-node fast path,
resource budgeting for nested parallelism, and the `run_streaming()` / `execute()` split.

---

## 1. Execution Layer

### 1.1 Problem with the current stack

`backend.py:_run_parallel` calls `fugue.api.transform(task_df, _process_task_ref_partition, schema=..., partition={"by": UNIQUE_ID}, engine=self.engine)`. This forces four artefacts that are wrong by design:

1. **Schema strings.** `_collect_quantile_columns` (line 138) decodes every model config just to enumerate quantile column names so Fugue can declare a schema string. The schema is a Fugue protocol requirement; it has no value for Calibre.
2. **Base64-pickled model configs as DataFrame columns.** `_encode_model_config` / `_decode_model_config` exist solely to embed model configs in a Fugue-partitioned DataFrame. This is a workaround for Fugue's inability to pass arbitrary Python objects alongside partition data.
3. **Parquet round-trips.** `ForecastTask.to_uri` was introduced to make tasks serializable across Fugue workers. URI hand-off is the right pattern; the problem is that Fugue *also* forces the dispatch frame to be materialized to parquet before it can be transformed.
4. **`_run_direct` bypass.** Line 598: `if self.engine is not None: return self._run_global_distributed(...)`. The global bypass exists because Fugue's partition-by-uid semantics don't compose with global (multi-series) adapters. This is a smell: the execution layer needs a special case to handle a common workflow.

None of these problems exist with Ray.

### 1.2 Proposed design: Ray Core

**Local-scope (per-uid) fan-out.**

```python
@ray.remote(num_cpus=1)
def _ray_fit_predict_uid(ref: ForecastTaskRef, origin: pd.Timestamp) -> pd.DataFrame:
    task = ref.materialize()
    history = task.history[task.history[DS] < origin]
    if history.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    origin_task = ForecastTask(history=history, horizon=task.horizon,
                               model_config=task.model_config, forecast_origin=origin)
    preds = _fit_predict_task(origin_task)
    return _finalize_preds(preds, origin, origin_task.model_name)
```

`_run_parallel` becomes:

```python
def _run_parallel(self, refs: list[ForecastTaskRef], origin: pd.Timestamp) -> pd.DataFrame:
    if not refs:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    if len(refs) < self._ray_threshold:
        # in-process fast path, zero Ray overhead
        return self._run_sequential(refs, origin)
    futures = [_ray_fit_predict_uid.remote(ref, origin) for ref in refs]
    frames = ray.get(futures)
    return _coerce_forecast_frame_dtypes(
        pd.concat([f for f in frames if not f.empty], ignore_index=True)
    )
```

Key differences from current Fugue path:
- `ForecastTaskRef` is pickled directly into the Ray task argument (it is already picklable: two URI strings + a config dict + a timestamp). No schema string, no base64 encoding, no intermediate DataFrame.
- Schema discovery is gone. `_collect_quantile_columns` is deleted. Ray returns real Python `pd.DataFrame` objects; `_coerce_forecast_frame_dtypes` handles dtype normalization on the concat result.
- The `_TaskDispatchRecord` dataclass and `_dispatch_records_to_frame` are deleted. They existed only to satisfy Fugue's partition-by-column contract.

**Global-scope (single model on full panel).**

Global tasks always run in-process on the driver. No Ray hop. This was already the intent of `_run_direct` (line 592–655), but the current code adds a Fugue distributed path when `self.engine is not None`. That branch is deleted; global scope is unconditionally in-process:

```python
def _run_direct(self, refs: list[ForecastTaskRef], origin: pd.Timestamp) -> pd.DataFrame:
    # Global adapters fit on the full panel: no fan-out, always in-process.
    ...
```

This is correct because global models (e.g. LightGBM with `scope="global"`) fit once on the concatenated panel. Distributing one training job across Ray workers would add coordination overhead with no parallelism gain. The panel for VN2 is ~600 series × ~260 weeks × 1 feature = ~156k rows — milliseconds to pass in-process.

**Task hand-off mechanism.**

`ForecastTaskRef` → pickled into `@ray.remote` args. Each ref is ~500 bytes (two file paths + horizon int + timestamp + config dict). At 600 series this is <300 KB total. Plasma store is not needed; plasma is optimized for large NumPy arrays (>100 MB).

Workers read parquet via `_read_parquet_cached` (the existing `lru_cache` wrapper). In Ray workers, the cache is per-process, which is correct — each worker process independently caches the parquet it has read, and the cache is released when the process exits.

**`ExecutionOptions` changes.**

Replace `engine: Any` (Fugue engine object) with:

```python
@dataclass(frozen=True)
class ExecutionOptions:
    freq: str = "W"
    ray_address: str | None = None   # None = local; "ray://host:10001" = cluster
    ray_threshold: int = 10          # tasks below this run in-process (no Ray)
    max_concurrency: int | None = None  # None = num_cpus; used as placement group budget
    seed: int | None = None
    metrics: list[Callable] | None = None
```

`engine: Any = None` is removed. The `ray_threshold` field gives the single-node fast path without a special config flag.

**Cluster and worker lifecycle.**

`ray.init(address=options.ray_address, ignore_reinit_error=True)` is called once at the **process level**, not per `BackendEngine` instance. `BackendEngine.__init__` calls `ray.init(...)` if `options.ray_address is not None` (cluster) or if `len(tasks) >= ray_threshold` (local multicore). For the fast path (below threshold, no address) Ray is never initialized, preserving the zero-startup smoke test path.

`ray.shutdown()` is **not** called by `BackendEngine`. Lifetime is the process. Tests use a session-scoped `ray.init(num_cpus=2, ignore_reinit_error=True)` fixture.

**Single-node fast path (< `ray_threshold` tasks).**

When `len(parallel_tasks) < ray_threshold`, `_run_parallel` calls `_run_sequential` — the same pure-Python loop used in `_run_direct`. This preserves the existing behavior for smoke tests, health checks, and small configs without any Ray overhead (no `ray.init`, no task pickling, no futures).

Default threshold: 10 tasks. Configurable in `ExecutionOptions` for benchmarks that want a lower crossover.

**`BackendEngine.execute()` signature.**

The existing signature `execute(tasks, actuals, origins) -> BackendResult` is **preserved unchanged**. This is the primary public API surface. It is called from `commands.py`, integration tests, and benchmark scripts.

A new `run_streaming(tasks, actuals, origins)` method is added as a **generator** that yields `OriginResult` (one per origin). This is the surface consumed by the Ray Tune objective function for per-origin ASHA reporting. The canonical plan proposes converting `execute()` itself to a generator — this is wrong. It would break every existing call site that does `result = engine.execute(...)`. The generator belongs in `run_streaming()`.

---

## 2. Tuning Layer

### 2.1 Problem with the current stack

`optimizer.py:optimize_task` calls `study.optimize(_objective, n_trials=task.n_trials)`. The `_objective` for the conformal path calls `BackendEngine(...).execute(...)` — a full backtest per trial. Every trial is sequential. The `ThreadPoolExecutor` in `benchmarks/vn2/tuning.py:98–102` parallelizes across *series* but not across *trials within a series*. For VN2 (600 series × 50 trials × ~40 origins = 1.2M fit-predicts), this is the dominant runtime bottleneck.

There is no early stopping. A 50-trial study evaluates all 50 trials to completion even if 40 of them are clearly worse than the incumbent after 8 origins.

### 2.2 Proposed design: Ray Tune + OptunaSearch + ASHAScheduler

**Preserving `TuningTask.search_space: Callable[[optuna.Trial], dict]`.**

`OptunaSearch(space=task.search_space, sampler=TPESampler(seed=task.seed))` is the correct interface. It forwards a real `optuna.Trial` object to the user callback — same `suggest_int`, `suggest_float`, `suggest_categorical`, same `TrialPruned`, same conditional logic. The define-by-run pattern used in `benchmarks/vn2/run_benchmark.py:987–1029` (`_sample_cost_search_crc_config`, `crc_enabled=False` skipping ~10 suggests) is preserved without code changes.

This is the decisive advantage of `OptunaSearch` over Ray Tune's native search: it does not require the search space to be declared upfront.

**Trial-level parallelism.**

```python
tuner = tune.Tuner(
    tune.with_resources(objective_fn, resources={"cpu": K}),
    tune_config=tune.TuneConfig(
        search_alg=OptunaSearch(space=task.search_space, sampler=TPESampler(seed=task.seed)),
        scheduler=ASHAScheduler(
            time_attr="origin_idx",
            grace_period=task.grace_period,
            max_t=len(task.origins),
            reduction_factor=3,
        ),
        num_samples=task.n_trials,
        max_concurrent_trials=max(1, num_cpus // K),
    ),
    run_config=RunConfig(name=task.unique_id),
)
result_grid = tuner.fit()
```

Where `K = max(1, num_cpus // max_concurrent_trials)` is the CPU budget per trial. For a 16-core machine running 4 concurrent trials, each trial gets 4 cores for its per-uid Ray fan-out.

**Preventing nested oversubscription.**

Each trial's `BackendEngine` uses `ExecutionOptions(ray_address=None, max_concurrency=K)`. The per-uid Ray tasks inside a trial are limited to `K` concurrent workers via the placement group budget. This prevents 4 concurrent trials × 600 series each from spawning 2400 simultaneous workers on a 16-core machine.

The canonical plan says `resources_per_trial={"cpu": K}` and `ExecutionOptions.max_uid_concurrency` — the intent is identical. The name `max_concurrency` is clearer.

**Early stopping / pruning via per-origin ASHA.**

The objective function consumes `BackendEngine.run_streaming()`:

```python
def objective_fn(config: dict) -> None:
    engine = BackendEngine(
        execution=ExecutionOptions(freq=task.freq, max_concurrency=K),
        conformal=ConformalOptions(runtime=task.conformal_runtime_factory()),
    )
    running_score = 0.0
    for idx, origin_result in enumerate(engine.run_streaming([forecast_task], actuals, origins)):
        running_score = task.objective.evaluate(origin_result.resolved, origin_result.resolved[Y])
        tune.report({"score": running_score, "origin_idx": idx})
```

ASHA prunes a trial by stopping `tune.report` calls — the objective exits, Ray reclaims the worker CPUs. No trial is killed mid-fit; pruning happens between origins.

**Grace period.**

`task.grace_period` defaults to `WARMUP_ORIGINS` for conformal tasks, where `WARMUP_ORIGINS = max(K, ceil(1/alpha) - 1)` — the minimum number of origins before conformal intervals stabilize. For VN2 (α=0.167, K=3): `WARMUP_ORIGINS = max(3, 5) = 5`. The canonical plan uses 8 — conservative and safe. Calibre should encode the formula and let the user override via `TuningTask.grace_period: int = 8`.

**MLflow experiment tracking.**

Replace `optuna_integration.mlflow.MLflowCallback` with `ray.air.integrations.mlflow.MLflowLoggerCallback`. Parent run semantics:

```python
with mlflow.start_run(run_name=task.unique_id) as parent_run:
    tuner = tune.Tuner(
        ...,
        run_config=RunConfig(
            callbacks=[MLflowLoggerCallback(
                tracking_uri=MLFLOW_TRACKING_URI,
                experiment_name="vn2-tuning",
                tags={"parent_run_id": parent_run.info.run_id},
            )]
        ),
    )
```

Each trial logs as a child run keyed by `parent_run_id`. The existing `safe_log_metric` and `log_costs_dataframe` remain for non-HPO paths.

**`RunStore` integration.**

`RunStore` (PR #31, `calibre/storage/state.py`) stores conformal runtime state. It is not extended for trial persistence — that is MLflow's job. After `tuner.fit()` completes, the calling script writes `result_grid.get_best_result().config` to the run store if a `run_id` is provided. No direct coupling between Ray Tune and SQLAlchemy.

**`TuningTask` field additions.**

```python
@dataclass(frozen=True)
class TuningTask:
    ...
    grace_period: int = 8           # ASHA grace period (minimum origins before pruning)
    resources_per_trial: dict = field(default_factory=lambda: {"cpu": 1})  # Ray resource budget
```

`search_space: Callable[[optuna.Trial], dict]` is unchanged — the load-bearing constraint.

---

## 3. Scheduler Depth: Do / Defer / Skip

| Layer | What it means | Verdict | Why |
|-------|--------------|---------|-----|
| **0. Execution framework (fan-out)** | Ray Core `@ray.remote` replaces `fugue.api.transform` | **DO** | Unified scheduler with tuning layer, no schema strings, picklable task refs work natively, KubeRay address is one config field. Fugue added complexity with no remaining advantage. |
| **1. Tuning framework (HPO)** | Ray Tune + OptunaSearch + ASHAScheduler | **DO** | Fixes the trial-serialization bottleneck. ASHA cuts wasteful trials at the origin level — the natural unit of evaluation. OptunaSearch is the only Ray Tune search algorithm that supports define-by-run conditional search spaces. |
| **2. Data loading (parallel IO, column pruning)** | Ray Data parallel parquet reads | **DEFER** | VN2 has 600 series. Each per-uid parquet is <5 MB. `fsspec` + `_read_parquet_cached` (lru_cache) is fast enough. Ray Data parallelizes parquet reads but adds 100–200 ms scheduling overhead per batch — net negative at VN2 scale. Revisit at 50k+ series where IO is measurably the bottleneck. |
| **3. Stateful actors (caching, shared mutable state)** | Ray named actor for ConformalRuntime | **DEFER** | The driver-hosted `ConformalRuntime` is correct for both local and KubeRay deployments (mutable sequential state, never on the hot path of worker tasks). A named `ConformalRuntimeActor` is useful for multi-tenant server deployments where multiple concurrent requests share a single calibrated runtime. That use case is revenue-gated. |
| **4. Model training (distributed LightGBM/XGBoost)** | Ray Train / distributed tree training | **SKIP** | Global LGBM trains on 600 series × ~260 weeks = 156k rows. Training takes 1–3 seconds. Distributed tree training adds coordination overhead that exceeds the training time. The right scale trigger is >10M rows and >5 minutes of training. Calibre is not there. |
| **5. Serving (colocate with execution or replace FastAPI)** | Ray Serve | **SKIP** | Calibre's serving model is batch inference via ECS Jobs triggered by CLI or API. FastAPI wraps the CLI commands cleanly (PR #31). Ray Serve adds actor lifecycle complexity for a request pattern that is not latency-sensitive. No client SLA requires sub-second response times for a demand planning backtest. |
| **6. State store (replace SQLAlchemy/Alembic)** | Ray's native checkpointing or a document store | **SKIP** | SQLAlchemy + Alembic + Postgres is the correct persistence layer for transactional state (run records, conformal snapshots). Ray's checkpointing is designed for ML training recovery, not durable business state. Migration would cost more than it saves. |
| **7. Orchestration (replace CLI / ECS / K8s Jobs)** | Ray Workflows, Prefect, or Airflow | **SKIP** | Calibre's execution DAG is flat: one `BackendEngine.execute()` call per run. There are no cross-run DAG dependencies, no fan-in aggregations, no conditional branches between jobs. ECS Jobs + CLI covers the orchestration need. KubeRay handles cluster-mode execution when needed. Ray Workflows is designed for DAGs with 10+ steps and inter-task data dependencies — not Calibre's use case. |

---

## 4. Data Layer

**fsspec for object-store IO: keep.**

`fsspec` is the right abstraction. It handles `file://`, `s3://`, `az://`, `gs://` uniformly. PR #31 consolidated all IO through `calibre/execution/io.py`. The `cloud`, `s3`, `azure`, `gcs` extras in `pyproject.toml` map cleanly to `s3fs`, `adlfs`, `gcsfs`. No reason to change.

**pandas as in-memory frame format: keep.**

Per-uid frames at VN2 scale are <1 MB each. The driver accumulates origins into the `ForecastLedger` — at 600 series × 40 origins × ~10 columns × 3 horizon steps, the ledger is ~72k rows, trivially small for pandas. Polars would reduce peak memory by 2–3× and accelerate the `_coerce_forecast_frame_dtypes` coercions, but these are not bottlenecks. The integration cost (adapter compatibility, parquet read types) is not worth it for this scale. Revisit at 100k+ series.

**Parquet as serialization format: keep.**

Parquet is the right format for ForecastTaskRef persistence. It is columnar, compressed, and read by `pd.read_parquet` via pyarrow (already a dependency). Arrow IPC (feather) is faster for in-process round-trips but ForecastTaskRef is written once and read by workers — the IO is not on the critical path relative to model training. Plasma store targets sub-millisecond inter-process transfers of large NumPy arrays; it adds a daemon dependency for no gain at this task size.

**ForecastTaskRef (URI-based materialization): keep.**

`ForecastTaskRef` is already Ray-clean. It is picklable (two string URIs + config dict + timestamp), reads via `fsspec` + pyarrow, and the `lru_cache` on `_read_parquet_cached` is per-process — correct in Ray worker processes. The workaround for multi-origin reads (a single `ForecastTaskRef` per uid, filtering by `history[DS] < origin` in the worker) is correct and already implemented.

The `_TaskDispatchRecord` intermediate layer (lines 148–195 of `backend.py`) is deleted. It existed only to embed task data into a Fugue partition DataFrame.

---

## 5. Observability

**Dashboard: Ray Dashboard.**

Ray Dashboard (included in `ray[default]`) shows task queue depth, CPU/memory per worker, task duration histograms, and HPO trial status (via Ray Tune's built-in Tune tab). This replaces the current opacity between Fugue worker logs and Optuna's local study. The Ray Dashboard runs on port 8265 by default; for ECS/K8s, expose it via an internal load balancer.

Keep the existing structured JSON logs (`logger.info(..., extra={...})`) — they are the per-task audit trail, not a replacement for the dashboard.

**Experiment tracking: MLflow (keep).**

MLflow is already deployed on the Tailscale mesh (`http://404records.tail810e2e.ts.net:5000`). Every benchmark run logs params, metrics, and artifacts. The switch is:

- Non-HPO paths: `safe_log_metric`, `log_costs_dataframe` unchanged.
- HPO paths: `MLflowLoggerCallback` (from `ray.air.integrations.mlflow`) replaces `optuna_integration.mlflow.MLflowCallback`. Nested run semantics (parent = benchmark, child = trial) are preserved by tagging child runs with `parent_run_id`.

W&B and Ray's native tracking are not needed. MLflow covers the use case and is already operational.

**Metrics: Prometheus (keep).**

`prometheus-client` stays for operational metrics (`observe_forecast_duration`, `set_conformal_coverage`, `set_order_cost`). Ray's built-in metrics (task duration, memory, queue depth) complement but do not replace these business-level metrics. Configure Ray to scrape its metrics endpoint from the same Prometheus instance.

**Logging: structured JSON (keep).**

The existing `logger.info(..., extra={...})` pattern produces structured logs. Ray workers inherit the driver's logging config. No distributed tracing is needed at this scale — OpenTelemetry spans would add 50–100 ms overhead per origin for no operational benefit in a batch pipeline. Revisit if Calibre moves to real-time decision APIs.

---

## 6. Packaging

**`pyproject.toml` extras post-migration:**

Remove:
```toml
# DELETE
dask = ["dask[distributed]"]
spark = ["pyspark"]
```

Add:
```toml
ray = ["ray[default,tune]>=2.10,<3"]
```

Keep: `cloud`, `s3`, `azure`, `gcs`, `ml`, `neural`, `xgboost`, `benchmarks`, `dev`.

`benchmarks` extra retains `mlflow>=2.17,<3` and `optuna-integration[mlflow]>=4.0` — the latter is kept until all callers of `optuna_mlflow_callback` are migrated, then removed.

**Slim vs full image:**

| Image | Contents | Use |
|-------|----------|-----|
| `slim` | core deps + `ray` extra | ECS Fargate tasks, health checks, CLI runs without LightGBM |
| `full` | slim + `ml` + `neural` + `benchmarks` | VN2 benchmarks, HPO, neuralforecast experiments |

Image size is not a concern (stated in PLAN.md constraints). The slim/full split from PR #31 is preserved.

**Databricks compatibility.**

Not required for Phase A or B. If a client operates on Databricks, the options are:
1. Run Calibre CLI as a Databricks Job task on a single-driver cluster (no Ray needed for small datasets).
2. Use RayDP (`raydp` package) to mount Ray on top of Databricks' existing Spark cluster for large-scale fan-out.

No Calibre-side code changes are required for either path. The `ray_address` config field points to whatever address Ray is listening on.

**Version pinning.**

`ray>=2.10,<3` — Ray 2.10 is when `OptunaSearch` with define-by-run callables stabilized and `ASHAScheduler` with `time_attr` other than "training_iteration" was formally supported. Pin the major version upper bound to avoid breaking API changes (Ray 3.x is not yet released; add a CI check to catch it when it arrives).

---

## 7. Staged Migration Sequence

### Phase A — Execution backend (Ray Core)

**Entry criteria:** This decision document merged. `benchmarks/vn2/config/winning.yaml` cost verified at ≤ EUR 4,992.20 on the pre-migration stack.

**Work:**
- Delete `_TaskDispatchRecord`, `_dispatch_records_to_frame`, `_collect_quantile_columns`, `_encode_model_config`, `_decode_model_config`, `_process_task_ref_partition` (Fugue version), `_process_global_task_ref_partition` (Fugue version).
- Add `_ray_fit_predict_uid` as `@ray.remote` module-level function.
- Rewrite `_run_parallel` to `ray.get([...])` + concat, with sequential fast path below `ray_threshold`.
- Delete `_run_global_distributed`. `_run_direct` becomes unconditionally in-process.
- Update `ExecutionOptions`: remove `engine: Any`, add `ray_address`, `ray_threshold`, `max_concurrency`.
- Update `commands.py`: replace `_resolve_execution_engine` / `_close_execution_engine` with `_init_ray(config)`.
- Delete Dask/Spark integration tests (`test_dask.py`, `test_dask_quantile.py`, `test_spark.py`).
- Add `tests/integration/test_ray.py`: local Ray cluster, VN2 winning config, cost ≤ EUR 4,992.20.
- Add `tests/integration/test_ray_quantile.py`: quantile columns survive Ray task round-trip.
- Remove `fugue`, add `ray[default,tune]>=2.10,<3` in `pyproject.toml`.

**Exit criteria:**
- `uv run pytest` green.
- `uv run mypy calibre/` clean.
- `uv run ruff check .` clean.
- `tests/integration/test_ray.py` green.
- `rg -i 'fugue|fa\.transform|DaskExecutionEngine|SparkExecutionEngine' calibre/ tests/` → no matches.
- `benchmarks/vn2/config/winning.yaml` cost ≤ EUR 4,992.20 under Ray.

**Rollback:** `git revert` the Phase A commit(s). The Fugue path is in a single file (`backend.py`) and the CLI engine resolver is in `commands.py`. Revert is surgical.

**Effort:** ~2 working days.

---

### Phase B — Tuning (Ray Tune + OptunaSearch + ASHA)

**Entry criteria:** Phase A complete. `test_ray.py` green on CI.

**Work:**
- Add `BackendEngine.run_streaming(tasks, actuals, origins)` generator yielding `OriginResult`.
- Rewrite `optimize_task` in `optimizer.py` to use `tune.Tuner(OptunaSearch, ASHAScheduler)`.
- Add `grace_period: int = 8` and `resources_per_trial: dict` to `TuningTask`.
- Update `benchmarks/vn2/tuning.py`: delete `ThreadPoolExecutor`, `tune_all_series` submits one Tune study per uid (or one multi-uid study — benchmark after Phase A).
- Update `benchmarks/vn2/run_benchmark.py`: `run_hpo` and `run_cost_search` switch to `tune.Tuner(...).fit()`, replace `optuna_mlflow_callback` with `MLflowLoggerCallback`.
- Add `tests/integration/test_ray_tune.py`: small study (10 trials), ASHA pruning observed, conditional search space exercised.

**Exit criteria:**
- `tests/integration/test_ray_tune.py` green.
- At least one trial pruned before completing all origins (ASHA is active).
- `run_hpo` cost within ±5% of pre-migration baseline (variance from ASHA exploration is acceptable).
- Conditional search spaces (`_sample_cost_search_crc_config` with `crc_enabled=False`) covered by integration test.

**Rollback:** Restore `optimizer.py` to sequential `study.optimize(...)`. `TuningTask` field additions are backward-compatible (default values).

**Effort:** ~2 working days.

---

### Phase C — Vault sync

**Entry criteria:** Phase B complete.

**Work:**
- Update `~/obsidian-vault/vault/Val/Projects/calibre/architecture.md`: technology stack row → `Ray Core (local + KubeRay)` for distributed execution, `Ray Tune + OptunaSearch + ASHA` for HPO.
- Append to `lessons.md`: Fugue removed in favour of Ray — schema strings, base64-encoded configs, and dispatch DataFrames deleted. Unified scheduler for series + trials enables ASHA early stopping.
- Flip plan status `i-want-to-switch-polished-ocean.md` → `accepted`.

**Exit criteria:** Vault committed and pushed.

**Effort:** < 1 working day.

---

## 8. Risk Register

### Risk 1: Framework setup cost per CLI invocation
**Likelihood:** Medium. **Impact:** Medium.

`ray.init()` on a cold process takes 1–2 seconds (imports, daemon startup, object store init). For a health check or a 3-series smoke test, this is the majority of runtime.

**Mitigation:** The `ray_threshold` fast path. When `len(parallel_tasks) < ray_threshold` (default: 10), Ray is never initialized — the in-process sequential path runs. Health checks use 2 tasks (single origin, single series). Smoke tests use ≤ 5 series. Both stay below the threshold. For ECS Jobs (600 series × 40 origins), the 2-second overhead is <0.5% of total runtime.

---

### Risk 2: ASHA pruning good trials because origin ordering is non-stationary
**Likelihood:** Medium. **Impact:** High.

VN2 conformal costs are systematically higher in early origins (fewer warmup windows → wider intervals → more holding cost). A trial that looks bad at origin 3 may be excellent at origin 15 after calibration stabilizes. ASHA would prune it if `grace_period` is too small.

**Mitigation:** `grace_period = WARMUP_ORIGINS = 8` for VN2 (α=0.167, K=3: `ceil(1/0.167) - 1 + 3 = 5 + 3 = 8`). This is the minimum number of origins before conformal intervals are statistically stable. Encoded as `TuningTask.grace_period` with a default derived from `ConformalOptions` when present. The `reduction_factor=3` means a trial must be in the bottom 1/3 of all trials *after* the grace period to be pruned — conservative enough to avoid pruning warmup artefacts.

---

### Risk 3: Conditional search spaces breaking under OptunaSearch
**Likelihood:** Low. **Impact:** High.

`OptunaSearch` with define-by-run callables is the primary advertised use case for Ray Tune's Optuna integration. The risk is subtle: if Ray Tune recreates a pruned trial's config for re-evaluation (e.g., after a worker failure), it must replay the same `Trial.suggest_*` sequence. This works because `OptunaSearch` preserves the trial's parameter history in the Optuna study, which is owned by the search algorithm object (not the worker).

**Mitigation:** Integration test in `test_ray_tune.py` that exercises `_sample_cost_search_crc_config` end-to-end: verify that `crc_enabled=False` trials produce configs without the `crc_*` keys, and that `crc_enabled=True` trials include them. Pin `ray>=2.10` where this is stable. Monitor Ray release notes for `OptunaSearch` changes.

---

### Risk 4: Memory pressure on large panels
**Likelihood:** Low (at VN2 scale). **Impact:** Medium (at 10k+ series).

Each Ray worker deserializes its `ForecastTaskRef`, reads a parquet file (~1–2 MB/series), holds it in memory during fit+predict, then returns a DataFrame. At 600 series with `max_concurrency=8`: 8 workers × 2 MB = 16 MB peak working memory. Negligible.

At 10,000 series with `max_concurrency=16`: 16 workers × 2 MB = 32 MB — still fine. The `lru_cache` in `_read_parquet_cached` is per-process; workers do not accumulate across tasks (each task gets a fresh worker if the pool is saturated). The driver's `ForecastLedger` grows to ~10k × 40 origins × 10 columns = 4M rows — ~500 MB for pandas. That is the real pressure point.

**Mitigation:** `LedgerOutputOptions(streaming=True)` (already implemented in PR #31) writes ledger rows to parquet as they complete, keeping driver memory bounded. Ensure streaming mode is documented as the recommended mode for panels > 5k series.

---

### Risk 5: Dev experience on the platforms actually developed on
**Likelihood:** Low. **Impact:** Medium.

Ray on Linux (the primary dev environment) is mature. The risk is the `ray.init()` / `ray.shutdown()` lifecycle in tests: if a test crashes without calling `shutdown()`, subsequent tests may hit a re-init error. `ignore_reinit_error=True` handles this in the session fixture.

A subtler risk: `ray.remote` functions must be defined at module level (not inside functions or closures) to be picklable. `_process_task_ref_partition` is already module-level in `backend.py`. Moving to `@ray.remote` at module level is the same constraint. The existing test `test_fugue_partition_worker_is_module_level_picklable` (retargeted to `test_ray_remote_is_picklable`) enforces this.

**Mitigation:** Session-scoped `ray.init(num_cpus=2, ignore_reinit_error=True)` fixture in `conftest.py`. Static picklability test that asserts `_ray_fit_predict_uid.__wrapped__` pickles without `ConformalRuntime` references.

---

## Appendix A: Where this document disagrees with the canonical plan

The canonical plan (`i-want-to-switch-polished-ocean.md`) is the correct direction. The following are refinements, not contradictions:

1. **`execute()` as generator.** The plan (section 2.3) says "`BackendEngine.execute()` becomes a generator." This document proposes `execute()` stays non-generator; `run_streaming()` is the new generator method. Reason: `execute()` is called from 8+ sites in `commands.py`, tests, and benchmarks — changing its return type breaks all of them. Adding `run_streaming()` as a parallel method is additive and avoids migration churn at every call site.

2. **`_run_global_distributed` deletion.** The plan (section 3, `backend.py:591-655`) says this method is deleted and "global scope always runs in-process on the driver." This document agrees — and removes it from the design without the waffling in the original Fugue path (which had the `if self.engine is not None` branch). The global scope is unconditionally in-process.

3. **`ray_threshold` fast path.** The plan does not specify a threshold mechanism for the single-node fast path. This document adds `ExecutionOptions.ray_threshold: int = 10` to make the crossover explicit and configurable.

4. **Resource budget calculation.** The plan says `resources_per_trial={"cpu": K}` and `ExecutionOptions.max_uid_concurrency`. This document specifies the formula: `K = max(1, num_cpus // max_concurrent_trials)`. The naming `max_concurrency` is preferred over `max_uid_concurrency` because it applies at the engine level, not the uid level.

5. **`MLflowLoggerCallback` nesting.** The plan notes MLflow nested-run semantics may differ. This document specifies the mechanism: start a parent `mlflow.start_run()` before `tune.Tuner.fit()`, pass `parent_run_id` as a tag to `MLflowLoggerCallback`. This is a known pattern in the Ray docs and does not require custom MLflow run management in the objective function.
