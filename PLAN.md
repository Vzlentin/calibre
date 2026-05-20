# /goal: Migrate Calibre execution and tuning to Ray Core + Ray Tune

Status: implemented
Decision date: 2026-05-19
Long-form rationale: `docs/stack-decision.md`

## Decision

Replace Fugue per-series fan-out and sequential Optuna tuning with **Ray Core**
(execution) and **Ray Tune with OptunaSearch + ASHAScheduler** (HPO). Keep
everything else: pandas, Parquet, fsspec, `ForecastTaskRef` URI hand-off,
FastAPI, SQLAlchemy/Alembic, MLflow, Prometheus, driver-owned conformal state.

This is a scheduler migration, not a platform rewrite. Model adapters in
`calibre/forecasting/{stats,ml,neural}forecast_adapter.py` remain backend-blind.

## Non-negotiables

1. **VN2 cost gate.** `benchmarks/vn2/config/winning.yaml` must reproduce
   `total_cost = 4992.20` rounded to cents at every phase exit. If it breaks,
   stop and fix the behavioral regression before continuing.
2. **Backend-blind adapters.** No `ray`, `fugue`, `dask`, or `pyspark` imports
   in `calibre/forecasting/*_adapter.py`.
3. **Driver-owned conformal state.** `ConformalRuntime` and its mutable
   `_issued_count` stay on the driver. Never inside a Ray worker.
4. **Search-space API unchanged.** `TuningTask.search_space:
   Callable[[optuna.Trial], dict]` is load-bearing (conditional sampling).
   Use `OptunaSearch` callable form. Do not convert to Ray Tune declarative
   parameter dicts.
5. **`BackendEngine.execute(tasks, actuals, origins) -> BackendResult`
   survives** as the batch API. Add a separate streaming origin iterator for
   tuning; do not turn `execute()` into a generator.

## Target API shape

`ExecutionOptions` replaces `engine: Any` with explicit scheduler fields:

- `backend`: `"local" | "ray" | "auto"`, default `"auto"`.
- `ray_address`: optional Ray cluster address; `None` means local Ray when needed.
- `ray_threshold`: default `10`; below this, never initialize Ray.
- `max_concurrency`: optional cap on concurrent uid tasks per run/trial.
- `seed`, `freq`, `metrics`: unchanged.

CLI config moves from `execution.engine: null | dask | spark` to
`execution.backend: local | ray | auto` plus `execution.ray_address`.

## Phases

Each phase ends with: `uv run pytest` green, `uv run ruff check .` green,
`uv run mypy calibre/` green, VN2 winning cost still `4992.20`.

### Phase 0 — Baseline and acceptance (0.5 day)

**Entry.** This document reviewed; worktree clean except planned doc changes.

**Work.** Record baseline VN2 winning cost and command. Agree config field
names and `BackendEngine` API compatibility before any source change.

**Exit.** Baseline run record shows `benchmarks/vn2/config/winning.yaml`
returns `total_cost = 4992.20` rounded to cents. No source changes yet.

**Rollback.** None needed.

### Phase 1 — Ray Core execution (2 days)

**Entry.** Phase 0 complete; baseline cost recorded.

**Work.**

- Add the Ray dependency path needed for execution work:
  `ray = ["ray[default,tune]>=2.38,<3"]`. Final dependency cleanup happens
  in Phase 5.
- Replace `fa.transform(...)` per-uid fan-out in `calibre/execution/backend.py`
  with a `@ray.remote` task that materializes `ForecastTaskRef`, filters
  `history[ds] < origin`, calls the existing adapter, and returns a pandas
  frame. Driver concatenates, normalizes dtypes, applies conformal intervals,
  applies ordering policy, appends to ledger.
- Delete `_collect_quantile_columns`, `_encode_model_config` /
  `_decode_model_config`, `_TaskDispatchRecord`, `_dispatch_records_to_frame`.
- Delete `_run_global_distributed`; global-scope models always run in-process
  on the driver.
- Add the in-process fast path: when local task count is below
  `ray_threshold`, run the existing sequential loop. Do not initialize Ray.
- Lifecycle: if Ray is needed and no `ray_address` is provided, start a local
  Ray runtime for the process and own shutdown at the CLI run boundary. If
  `ray_address` is provided, connect and do not own lifecycle.
- Replace `ExecutionOptions.engine: Any` with the field set above. Update
  `calibre/cli/config.py` and `calibre/cli/commands.py` to stop constructing
  Dask/Spark Fugue engines.

**Exit.**

- `rg -n "fugue|fa\.transform|DaskExecutionEngine|SparkExecutionEngine" calibre tests`
  returns no active runtime references.
- Unit and integration tests green.
- VN2 winning cost still `4992.20`.

**Rollback.** Revert the execution migration commit. The Fugue path is
isolated to `backend.py`, `cli/config.py`, `cli/commands.py`, and the related
tests, so revert is a localized operation.

### Phase 2 — Streaming origin API (1 day)

**Entry.** Phase 1 complete; Ray execution stable locally.

**Work.**

- Add a streaming origin iterator on `BackendEngine` for HPO.
- Keep `BackendEngine.execute(...) -> BackendResult` as the batch API.
- Refactor shared origin-loop internals so `execute()` and the streaming
  iterator use the same conformal/order/ledger behavior.
- Add parity tests proving streaming aggregation matches `execute()` output,
  conformal state advances identically, and ordering costs match.

**Exit.**

- Existing batch callers still pass.
- Streaming parity tests pass.
- VN2 winning cost still `4992.20`.

**Rollback.** Remove the streaming iterator and restore the single batch
origin loop. Phase 1 Ray execution stays.

### Phase 3 — Ray Tune HPO (2 to 3 days)

**Entry.** Phase 2 complete; streaming origin API matches batch behavior.

**Work.**

- Per-origin cumulative objective is reported through Tune's
  `train.report(...)`.
- Rewrite `calibre/tuning/optimizer.py::optimize_task` to use
  `tune.Tuner(trainable, search_alg=OptunaSearch(space=task.search_space),
  scheduler=ASHAScheduler(...))`.
- ASHA configuration: progress attribute is origin index, `max_t =
  len(origins)`, default `grace_period = 8` (matches VN2 `WARMUP_ORIGINS`),
  per-`TuningTask` override allowed.
- Pruning only between origins. No mid-fit, mid-predict, mid-conformal-update,
  or mid-ordering pruning.
- Trial resource budget: explicit CPU request per trial. Cap uid fan-out and
  library thread counts (LightGBM, NumPy, Torch) to that budget. No nested
  Fugue/Dask/Spark/joblib/ThreadPool inside trials.
- Set `max_concurrent_trials` from available CPUs and the per-trial budget.

**Exit.**

- Integration test exercises a conditional search space (one sampled value
  suppresses later parameters) and passes.
- At least one trial is pruned after the grace period in the HPO test.
- Resource-budget test confirms no nested oversubscription.
- VN2 winning cost still `4992.20`.

**Rollback.** Revert tuning migration commit. Phase 1 Ray execution and
Phase 2 streaming API stay.

### Phase 4 — Observability and persistence (1 day)

**Entry.** Phase 3 complete; MLflow remote URI available.

**Work.**

- Replace `optuna_mlflow_callback` on HPO paths with
  `ray.air.integrations.mlflow.MLflowLoggerCallback` (or `setup_mlflow` inside
  the trainable for custom artifacts). Keep `safe_log_metric` and
  `log_costs_dataframe` for non-HPO benchmark paths.
- Tag Tune trial runs with the parent Calibre `run_id` and Ray `trial_id` so
  parent/child runs are discoverable in MLflow.
- Log best config, trial table, pruning summary, and cost artifacts to MLflow.
- Extend `RunStore` (`calibre/api/run_store.py`) only with HPO-level
  metadata/artifact pointers: Tune experiment directory, Optuna study name or
  storage URI, best-config artifact pointer. Do not store every trial as SQL
  business state. Resume reuses the same Tune experiment dir and Optuna study
  identity.

**Exit.**

- `RunStore` tests cover HPO artifact pointers and resume metadata.
- MLflow trial runs are discoverable under the parent `run_id`.
- VN2 winning cost still `4992.20`.

**Rollback.** Disable HPO resume metadata and fall back to MLflow artifacts.
Phases 1-3 stay.

### Phase 5 — Packaging and deployment (1 day)

**Entry.** Phase 4 complete; deployment image build available.

**Work.**

- `pyproject.toml`: keep the Ray extra added in Phase 1. Remove `fugue` from
  core. Remove `dask`, `spark`, `fugue_dask`, `fugue_spark` extras. Keep
  `optuna-integration[mlflow]` only until old callback users are removed.
- Update slim/full Docker image install sets. Ray runtime in slim; ML/neural
  extras only in full.
- Pin Ray and KubeRay versions in `uv.lock` and Helm values. **Do not use
  Ray 2.11.0–2.37.0 for KubeRay deployments** (RayJob readiness/liveness bug
  per KubeRay upgrade guide).

**Exit.**

- Fresh `uv sync --extra dev --extra benchmarks` installs cleanly.
- Slim and full Docker images build.
- API health check passes. Local Ray smoke test passes.
- VN2 winning cost still `4992.20`.

**Rollback.** Restore dependency set from previous lockfile and image
definitions.

### Phase 6 — Cleanup and docs (0.5 day)

**Entry.** Phase 5 complete.

**Work.**

- Remove dead Fugue/Dask/Spark tests, doc examples, and config snippets.
- Update deployment docs and CLI examples to the new `backend` field.
- Add troubleshooting notes for Ray startup cost and Windows dev experience
  (multi-node Ray on Linux containers / KubeRay; Windows local-only).

**Exit.**

- Docs match code. No stale Dask/Spark config examples.
- Full test suite green.
- VN2 winning cost still `4992.20`.

**Rollback.** Revert docs cleanup commit only.

**Total expected effort: 7 to 8 working days.**

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ray `init` cost per CLI invocation | Medium | Medium | `ray_threshold=10` fast path skips Ray for small runs. Own local Ray shutdown at CLI run boundary. |
| Local Ray runtime leak after CLI jobs | Medium | Medium | Local CLI-created Ray is owned by the run and explicitly shut down. Remote Ray clusters are externally owned and never shut down by Calibre. |
| ASHA prunes good trials during conformal warmup | Medium | High | Report metrics only after completed origins. `grace_period=8` matches VN2 `WARMUP_ORIGINS`. Compare ASHA results against a no-pruning sample before accepting HPO migration. Allow disabling pruning per `TuningTask`. |
| Conditional search spaces break under `OptunaSearch` | Low | High | Use callable search-space form. Add a test where an early sampled value suppresses later parameters. Pin Ray and Optuna versions. Do not convert to Ray declarative spaces. |
| Nested parallelism oversubscribes CPUs | High | High | Make Tune trial resources the outer budget. Cap uid fan-out inside trials and set LightGBM, NumPy, and Torch thread counts to fit the trial budget. |
| Memory pressure on large panels | Medium | High | Keep URI-backed `ForecastTaskRef`. Cap uid concurrency. Use streaming ledger output above a documented panel size. Do not put full panels into Ray object storage by default. |
| Ray/KubeRay version mismatch | Medium | High | Pin Ray exactly in `uv.lock` and pin KubeRay operator/Helm values compatibly. Avoid Ray 2.11.0-2.37.0 for KubeRay deployments per the upgrade warning. |
| Windows / platform dev experience | Medium | Medium | Local fast path is independent of Ray. Run Ray tests primarily on Linux CI. Document Windows Ray as local-development only; multi-node Ray runs on Linux containers / KubeRay. Path handling stays URI-based. |

## References

- `docs/stack-decision.md` — full rationale, code evidence, scheduler-depth
  verdicts (layers 0–7), data-layer and observability decisions, Databricks
  compatibility, version pinning.
- Key source files touched by this migration:
  - `calibre/execution/backend.py`
  - `calibre/tuning/optimizer.py`, `calibre/tuning/task.py`
  - `calibre/cli/config.py`, `calibre/cli/commands.py`
  - `calibre/api/run_store.py`
  - `pyproject.toml`, `uv.lock`
