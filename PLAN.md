# Calibre Post-Ray: Phased Execution Plan

The architectural reasoning, code citations, and seam audit live in
[`docs/deployment-audit.md`](docs/deployment-audit.md). This file is the
executable plan: phases, files, tests, and DoD. Dispatched one phase at a
time via `/goal implement Phase N of PLAN.md`.

## Conventions

- Every command is invoked through `uv run` (CLAUDE.md).
- Per-phase regression gate is at the bottom of this file. It runs at every
  phase boundary, not between sub-tasks within a phase.
- `PROGRESS.md` (created on first dispatch) records the last completed
  task per phase so a resumed agent picks up at the checkpoint.

---

## Phase 1 · Fix Predict-Then-Optimize HPO

**Goal:** cost-objective HPO becomes correct, distributed, and mode-aware.

**Files:**
- `calibre/tuning/optimizer.py` (`_evaluate_candidate:184–221`,
  `_trainable:275–312`, fallback at `:243–250`)
- `calibre/tuning/objectives.py` (`Cost:45–63`)
- `calibre/storage/state.py` (per-trial snapshot helper)

**Changes:**
- **(a) Unblock Ray-Tune-with-conformal.** Before launching `tune.Tuner`,
  serialise the seed conformal state to an object-store URI keyed by
  `trial_id`. Inside `_trainable`, hydrate via
  `SymmetricIntervalRuntime.from_state(uri)` instead of calling the
  factory directly. Delete the sequential fallback at `optimizer.py:243–250`
  and the `RuntimeWarning` it emits.
- **(b) Accumulate cost across origins, single scale.** Maintain
  `total_cost` across the `iter_origins` loop in both `_evaluate_candidate`
  and `_trainable`. Report cumulative `total_cost` as both the
  per-iteration metric (so ASHA prunes on a monotone-non-decreasing
  signal) and the trial's final objective. Do not use a running mean —
  intermediate and final must share the same scale so the value ASHA
  ranks on is the value Optuna receives.
- **(c) Make `Cost.evaluate` mode-aware.** Add
  `mode: Literal["perhorizon", "cumulative"]` to `Cost`. In `perhorizon`,
  group the frame by `(forecast_origin, h)`, evaluate cost per group, sum.
  In `cumulative`, assert the frame is a single (uid, origin) window and
  keep the current `demand = actuals.sum()` semantics. Raise a clear
  error if `Cost(mode=...)` disagrees with the frame's `conformal_mode`
  column.

**Tests:**
- `tests/tuning/test_ray_tune_with_conformal.py` —
  `test_no_sequential_fallback_when_conformal_in_loop`
- `tests/tuning/test_cost_objective_aggregation.py` —
  `test_total_cost_accumulates_across_origins`,
  `test_intermediate_metric_matches_final`
- `tests/tuning/test_cost_mode_dispatch.py` —
  `test_perhorizon_sums_per_group`,
  `test_cumulative_keeps_single_window_semantics`,
  `test_mode_mismatch_raises`

**DoD:**
- `uv run pytest tests/tuning/test_ray_tune_with_conformal.py
  tests/tuning/test_cost_objective_aggregation.py
  tests/tuning/test_cost_mode_dispatch.py` green.
- VN2 winning-cost number **will move** under the new aggregation. Record
  the post-Phase-1 cost in `PROGRESS.md` as the new baseline; update
  `benchmarks/vn2/config/winning.yaml` if a different config now wins.
  The cross-phase regression gate compares against the recorded baseline,
  not the pre-Phase-1 number.

---

## Phase 2 · Per-Partition State + Session Identity

**Goal:** conformal state moves from one fat JSON blob per run to
per-partition rows keyed by a stable cross-run session.

**Files:**
- `calibre/storage/state.py` (`RUNTIME_PARTITION:8`,
  `SqlConformalStateStore:17–28`)
- `calibre/storage/models.py` (conformal_state schema)
- `calibre/storage/migrations/` (new Alembic migration)
- `calibre/execution/backend.py:243, 282, 496–509` (wire real partition)
- `calibre/conformal/runtime.py` (expose partition keys)
- `calibre/execution/decision_loop.py:135–185` (replace
  `pending: list[pd.DataFrame]` with a `PendingStore`)

**Changes:**
- Replace the hard-coded `RUNTIME_PARTITION = "__runtime__"` literal.
  Thread the real per-`(uid, model, horizon)` partition string through
  `get`/`upsert` so the existing `(run_id, partition)` schema gets used
  as designed.
- Add `session_id: UUID` to `conformal_state` (Alembic migration).
  Derive via `hash(tenant, sku_set, model_config, conformal_config)`.
  `run_id` becomes an audit pointer on `runs`, not a state primary key.
- Add `PendingStore` Protocol and `SqlPendingStore` implementation. Table
  `pending_observations(session_id, uid, origin, h, lo, hi, y_hat)`.
  Observed rows are deleted on `observe`.
- Add a `last_updated_at` column on `conformal_state` and a
  `calibre maint compact-state --older-than 90d` CLI sub-command for
  scheduled compaction.

**Tests:**
- `tests/storage/test_per_partition_state.py` —
  `test_partitions_round_trip_independently`,
  `test_no_single_blob_collision`
- `tests/storage/test_session_keyed_resume.py` —
  `test_same_session_id_resumes_across_runs`,
  `test_different_session_id_starts_fresh`
- `tests/execution/test_pending_store.py` —
  `test_pending_persists_across_process_restart`
- `tests/cli/test_maint_compact.py`

**DoD:**
- `uv run pytest tests/storage/ tests/execution/test_pending_store.py
  tests/cli/test_maint_compact.py` green.
- Alembic migration applies on a fresh database without manual fixup.
- Running the same config twice with the same `session_id` → second run
  hydrates from the first run's last state (byte-identical final ledger
  for the resumed portion).

---

## Phase 3 · InventoryAdapter + Global Fan-Out + API Lifecycle

**Goal:** the engine ingests live inventory state, fans out global model
configs in parallel, and exposes a deployable HTTP lifecycle.

**Files:**
- `calibre/execution/dataset.py` (new `InventoryAdapter` Protocol)
- `calibre/simulation/` (accept injected initial `ProductState`)
- `calibre/core/forecast_task.py` (add `task_group: str | None` field)
- `calibre/execution/backend.py` (per-config Ray fan-out for globals;
  schedule by `task_group`)
- `calibre/api/main.py`, `calibre/api/schemas.py`,
  `calibre/cli/commands.py`

**Changes:**
- **InventoryAdapter.** Protocol with `load_state(unique_id, at_origin) ->
  ProductState` and `load_lead_times() -> dict[str, int]`. Backtests use
  `SyntheticInventoryAdapter` (today's default behaviour);
  `SnapshotInventoryAdapter` reads from a parquet snapshot URI;
  `ErpInventoryAdapter` stub left for client implementations.
- **Global-model fan-out.** Today `global_refs` execute in a driver loop
  (`_execute_origin`). Wrap each *global model config* in its own
  `@ray.remote` task with the full panel; results join the per-uid
  outputs at the per-origin merge.
- **Task grouping.** Add `task_group: str | None` to `ForecastTask`
  (defaults to `unique_id`). `BackendEngine` schedules grouped tasks
  together so a category can be prioritised or a warm-start can be shared
  across SKUs in the same group later (Phase 4 picks this up for the
  artifact cache).
- **API lifecycle split.** Replace the monolithic `POST /forecasts` with:
  - `POST /fit` — async, returns `fit_handle` (artifact URIs + session_id)
  - `POST /predict` — sync, `fit_handle` + origin → forecast frame
  - `POST /calibrate` — sync, `session_id` + forecast frame → calibrated
    frame (lands here because it needs the session key from Phase 2; the
    HPO-side `/tune` ships in Phase 4 alongside the unified search space)
  - `POST /order` — sync, calibrated frame + inventory snapshot + costs
    → order ledger
  - `POST /observe` — async, `session_id` + actuals → new conformal state
  - `GET /sessions/{tenant}/{uid}` — state + last forecast + open orders
  Keep `/backtests` for now; deprecate when the lifecycle endpoints cover
  it.

**Tests:**
- `tests/execution/test_inventory_adapter.py` —
  `test_synthetic_matches_today`,
  `test_snapshot_loads_from_parquet`,
  `test_injected_state_propagates_to_simulator`
- `tests/execution/test_global_fanout.py` —
  `test_multiple_global_configs_run_in_parallel`
- `tests/execution/test_task_grouping.py` —
  `test_group_scheduling_preserves_results`
- `tests/api/test_lifecycle_endpoints.py` —
  `test_fit_predict_calibrate_order_observe_roundtrip`,
  `test_session_state_get`

**DoD:**
- `uv run pytest tests/execution/test_inventory_adapter.py
  tests/execution/test_global_fanout.py
  tests/execution/test_task_grouping.py
  tests/api/test_lifecycle_endpoints.py` green.
- A two-config global ensemble on VN2 runs strictly faster than the
  driver-loop baseline (recorded in `PROGRESS.md`).

---

## Phase 4 · Unified Search Space + `/tune` + Cache + Drift

**Goal:** HPO tunes model + conformal + ordering jointly; pre-trained
artifacts are reused across origins; conformal coverage drift is observable.

**Files:**
- `calibre/tuning/task.py` (`TuningTask.search_space` signature)
- `calibre/tuning/optimizer.py` (consume `TuningCandidate`)
- `calibre/tuning/objectives.py` (`Regret`)
- `calibre/forecasting/cache.py` (new), adapter base
- `calibre/core/metrics.py` (drift gauge)
- `calibre/api/main.py` (`/tune` endpoint)

**Changes:**
- `TuningCandidate(model_config, conformal_config, ordering_config)`
  dataclass; `search_space` returns `TuningCandidate` instead of a model
  dict. The optimiser routes each component to the right runtime object.
- `Regret(decision_rule, arithmetic, costs)` as a sibling
  `TuningObjective` to `Cost` and `Pareto` — wraps `eval/regret.py`.
- `ModelArtifactCache(uri)` keyed by `adapter.cache_key(task)` (hash of
  history rows + config). Adapters check the cache before fitting.
  Conservative: identical-hash hits only.
- `calibre_conformal_coverage_drift{model, partition}` gauge derived from
  `AdaptiveAlphaController.error_history` running mean minus target.
- `POST /tune` — async, returns `study_id`; `GET /studies/{id}` returns
  best `TuningCandidate`. Reuses the existing `RunStore` for job tracking.

**Tests:**
- `tests/tuning/test_unified_candidate.py` —
  `test_search_space_returns_candidate`,
  `test_conformal_params_propagate`
- `tests/tuning/test_regret_objective.py`
- `tests/forecasting/test_model_cache.py` —
  `test_cache_hit_skips_fit`,
  `test_cache_miss_writes`
- `tests/observability/test_coverage_drift.py`
- `tests/api/test_tune_endpoint.py`

**DoD:**
- `uv run pytest tests/tuning/test_unified_candidate.py
  tests/tuning/test_regret_objective.py
  tests/forecasting/test_model_cache.py
  tests/observability/test_coverage_drift.py
  tests/api/test_tune_endpoint.py` green.
- A repeat HPO trial with the same `cache_key` skips fit (asserted via a
  fit-counter in the test).

---

## Cross-phase regression gate

Run at every phase boundary:

```bash
uv run pytest
uv run mypy calibre/
uv run ruff check .
uv run calibre run --config benchmarks/vn2/config/winning.yaml
```

The VN2 backtest is compared against the baseline cost recorded in
`PROGRESS.md`, not a fixed number — Phase 1 intentionally moves the
baseline, and later phases may move it again as caching / fan-out change
which configs win.

---

## Out of scope (deferred to cloud-native roadmap or revenue gate)

- Multi-tenancy, RLS, white-box packaging, BYOI OIDC
- Managed Ray / KubeRay cluster operations
- Full OpenTelemetry SDK
- Real-time inference endpoint
- Per-SKU `CostStruct` loader (owned by cloud-native Phase 1.2)
- fsspec / DatasetAdapter Protocol (owned by cloud-native Phase 1)
