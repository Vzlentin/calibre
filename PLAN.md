# Calibre Post-Ray: Phased Execution Plan

The architectural reasoning, code citations, and seam audit live in
[`docs/deployment-audit.md`](docs/deployment-audit.md). This file is the
executable plan: phases, files, tests, and DoD.

## Execution mode

Single autonomous run: `/goal implement PLAN.md phase by phase`. No human
pause between phases — every signature, path, and dependency is frozen in
this file before dispatch.

**Resume protocol.** Maintain `PROGRESS.md` at the repo root with the
format below. Append one block per task as it completes. On (re)start the
agent reads the tail and resumes from `next_task`.

```yaml
phase: 1
last_completed_task: "1.a unblock Ray-Tune-with-conformal"
next_task: "1.b accumulate cost across origins"
last_commit: "<sha>"
notes: "freed sequential fallback; trial state now passed via ray.put"
```

**Halt protocol.** If a phase's DoD fails after one good-faith attempt,
the agent writes `HALT.md` with `phase`, `failing_test`, `last_error`,
`hypothesis`, `commit_at_halt` and stops. No silent skips, no
`# type: ignore` shims to keep mypy green.

**Per-phase invariants.** Each phase must leave `uv run pytest`,
`uv run mypy calibre/`, and `uv run ruff check .` green before the
agent advances. The VN2 backtest baseline cost is read from
`PROGRESS.md`, not hard-coded.

## Conventions

- Every command is invoked through `uv run` (CLAUDE.md).
- Per-phase regression gate is at the bottom of this file. It runs at every
  phase boundary, not between sub-tasks within a phase.

---

## Phase 1 · Fix Predict-Then-Optimize HPO

**Goal:** cost-objective HPO becomes correct, distributed, and mode-aware.

**Files:**
- `calibre/tuning/optimizer.py` (`_evaluate_candidate:184–221`,
  `_trainable:275–312`, fallback at `:243–250`)
- `calibre/tuning/objectives.py` (`Cost:45–63`)
- `calibre/storage/state.py` (per-trial snapshot helper)

**Changes:**
- **(a) Unblock Ray-Tune-with-conformal.** `SymmetricIntervalRuntime.from_state`
  already takes a `dict` (`conformal/runtime.py:185–198`); pass it through
  Ray's object store, not a file URI. Mechanism:
  1. Before `tune.Tuner.fit()`, call `state_ref = ray.put(seed_runtime.get_state())`.
  2. Bind `state_ref` into the Tune trial config (`tune.with_parameters(...)`
     or a closure capture; do **not** put the `ObjectRef` directly in the
     search space).
  3. Inside `_trainable`, build the runtime via
     `SymmetricIntervalRuntime.from_state(task.conformal_config, ray.get(state_ref))`.
  Delete the sequential fallback at `optimizer.py:243–250` and the
  `RuntimeWarning` it emits. State is sub-10KB per partition; the object
  store is the right medium, not fsspec.
- **(b) Accumulate cost across origins, single scale.** Maintain
  `total_cost` across the `iter_origins` loop in both `_evaluate_candidate`
  and `_trainable`. Report cumulative `total_cost` as both the
  per-iteration metric (so ASHA prunes on a monotone-non-decreasing
  signal) and the trial's final objective. Do not use a running mean —
  intermediate and final must share the same scale so the value ASHA
  ranks on is the value Optuna receives.
- **(c) Make `Cost.evaluate` mode-aware.** Add a **kw-only field with
  default** to preserve `Pareto`'s positional construction
  (`Pareto.evaluate` builds `Cost(decision_rule, arithmetic, costs)` at
  `objectives.py:76`):
  ```python
  @dataclass(frozen=True, slots=True)
  class Cost:
      decision_rule: DecisionRule
      arithmetic: OrderingArithmetic
      costs: CostStruct
      mode: Literal["perhorizon", "cumulative"] = "perhorizon"  # kw-only via dataclass
  ```
  In `perhorizon`, group the frame by `(forecast_origin, h)`, evaluate
  cost per group, sum. In `cumulative`, assert the frame is a single
  `(uid, origin)` window and keep the current `demand = actuals.sum()`
  semantics. Raise a clear error if `Cost(mode=...)` disagrees with the
  frame's `conformal_mode` column. `Pareto.evaluate` is updated to forward
  `mode` so a cumulative tune produces cumulative Pareto evaluations.

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
- Add `session_id: str` (16-char hex) to `conformal_state` (Alembic
  migration). `dict`s are unhashable, so use a deterministic JSON
  serialisation:
  ```python
  import hashlib, json
  def derive_session_id(tenant: str, sku_set: list[str],
                        model_config: dict, conformal_config: dict) -> str:
      payload = json.dumps(
          {"tenant": tenant,
           "sku_set": sorted(sku_set),
           "model_config": model_config,
           "conformal_config": conformal_config},
          sort_keys=True, default=str)
      return hashlib.sha256(payload.encode()).hexdigest()[:16]
  ```
  Lives in `calibre/storage/session.py`. `run_id` becomes an audit
  pointer on `runs`, not a state primary key.
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
- `calibre/ordering/simulation/` (accept injected initial `ProductState`)
- `calibre/core/forecast_task.py` (add `task_group: str | None` field)
- `calibre/execution/backend.py` (per-config Ray fan-out for globals;
  schedule by `task_group`)
- `calibre/api/main.py`, `calibre/api/schemas.py`,
  `calibre/cli/commands.py`

**Changes:**
- **InventoryAdapter.** Protocol with `load_state(unique_id, at_origin) ->
  ProductState` and `load_lead_times() -> dict[str, int]`. **The returned
  `ProductState` is the generic `calibre.ordering.simulation.state.ProductState`,
  not the VN2 dataclass.** The VN2 benchmark already converts via
  `_to_generic()` (`benchmarks/vn2/simulator.py:63`); update the call
  site to take its initial state from the adapter rather than constructing
  a `VN2ProductState` first. Backtests use `SyntheticInventoryAdapter`
  (today's behaviour); `SnapshotInventoryAdapter` reads from a parquet
  snapshot URI; `ErpInventoryAdapter` stub left for client implementations.
- **Global-model fan-out.** Today `global_refs` execute in a driver loop
  (`backend.py:_run_global_scope:597–607`). Do **not** reuse
  `_process_task_ref` — that function takes a single ref. Add a new
  module-level function:
  ```python
  @ray.remote
  def _process_global_panel(
      refs: list[ForecastTaskRef],
      origin: pd.Timestamp,
  ) -> pd.DataFrame:
      """Materialise all refs, concat histories into the full panel,
      fit the global model once, return predictions for all SKUs."""
  ```
  `_run_global_scope` groups `global_refs` by `model_config` hash and
  dispatches one `_process_global_panel.remote(group, origin)` per
  config. Results concat into the per-origin merge alongside local
  outputs.
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
- `calibre/evaluation/regret.py` (**new** — pre-work, no `eval/regret.py`
  exists today)
- `calibre/conformal/controllers.py` (add public `error_history` property
  on `AdaptiveAlphaController`)
- `calibre/tuning/task.py` (`TuningTask.search_space` signature)
- `calibre/tuning/optimizer.py` (consume `TuningCandidate`)
- `calibre/tuning/objectives.py` (`Regret`)
- `calibre/forecasting/cache.py` (new), adapter base
- `calibre/core/metrics.py` (drift gauge)
- `calibre/api/main.py` (`/tune` endpoint)

**Changes:**
- **(pre-work) Create `calibre/evaluation/regret.py`.** Module exposes
  `compute_regret(realized: pd.Series, oracle: pd.Series) -> float`
  returning `(realized - oracle).clip(lower=0).sum()`. Oracle cost is
  the perfect-foresight benchmark; for VN2 this is the simulator run
  with `actuals` as the demand quantile. No `Regret` objective ships
  without this file.
- **`TuningCandidate(model_config, conformal_config, ordering_config)`**
  dataclass in `calibre/tuning/candidate.py`; `search_space` return type
  becomes `Callable[[optuna.Trial], TuningCandidate]`. The optimiser
  routes `model_config` to `ForecastTask`, `conformal_config` to the
  trial's conformal runtime factory (passed via `ray.put` from Phase 1),
  and `ordering_config` to the `Cost` / `Pareto` objective constructors.
- **`Regret(decision_rule, arithmetic, costs, mode="perhorizon")`** as a
  sibling `TuningObjective` to `Cost` and `Pareto`. Implementation
  composes `Cost` (for realized) + an oracle path that replays the
  decision-rule with `actuals` standing in for `y_hat`. Returns
  `compute_regret(realized, oracle)`.
- **`AdaptiveAlphaController.error_history` public property.**
  `_error_history` (private, `controllers.py:54`) gains a read-only
  `property` wrapper. Drift gauge reads the property, not the
  private attribute, so no `AttributeError` and no test mocking
  underscore-prefixed names.
- **`ModelArtifactCache(uri)`** keyed by `adapter.cache_key(task)` (hash
  of history rows + config). Adapters check the cache before fitting.
  Conservative: identical-hash hits only. No warm-start, no partial
  reuse.
- **`calibre_conformal_coverage_drift{model, partition}`** gauge derived
  from `controller.error_history` running mean minus target alpha.
- **`POST /tune`** — async, returns `study_id`; `GET /studies/{id}`
  returns best `TuningCandidate` serialized to JSON. Reuses the existing
  `RunStore` for job tracking.

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

## Phase 5 · Multi-SKU HPO Orchestration + Promo What-If

**Goal:** close the two genuine gaps the audit surfaced that Phases 1–4
don't touch: fan HPO across an SKU set and aggregate best configs into a
deployable bundle; let `/predict` accept a regressor override so planners
can run promotion scenarios.

**Files:**
- `calibre/tuning/orchestrator.py` (**new**)
- `calibre/api/main.py`, `calibre/api/schemas.py` (`/predict` future_x
  override)
- `calibre/storage/models.py` (new `tuning_runs` table for per-SKU best
  configs)
- `calibre/storage/migrations/` (Alembic migration)

**Changes:**
- **`TuningOrchestrator(session_id, sku_set, candidate_space, objective)`**
  in `calibre/tuning/orchestrator.py`. Fans one `optimize_task` per
  `unique_id` (or per `task_group` from Phase 3 when set), runs them
  concurrently on Ray, and aggregates results:
  ```python
  class TuningOrchestrator:
      def run(self) -> dict[str, TuningCandidate]:
          """Return {unique_id: best_candidate} after all trials complete."""
  ```
  Persists per-SKU best configs to `tuning_runs(session_id, unique_id,
  candidate_json, score, finished_at)`. Read by `/fit` so the next
  weekly cycle uses the tuned configs without a human in the loop.
- **`/predict` future_x override.** Add optional `future_x_override:
  dict[str, list[dict]]` field on `ForecastRequest` (uid → list of
  `{ds, regressor_name: value}` rows). Merged onto the loaded `future_x`
  before the engine runs. Enables "what if promo X is on next week"
  scenarios without retraining. No backend changes; only adapter
  forwarding (which already passes `future_x` through).

**Tests:**
- `tests/tuning/test_orchestrator.py` —
  `test_per_sku_best_configs_persisted`,
  `test_orchestrator_resumes_partial_completion`
- `tests/api/test_predict_what_if.py` —
  `test_future_x_override_changes_forecast`,
  `test_override_does_not_persist_across_calls`

**DoD:**
- `uv run pytest tests/tuning/test_orchestrator.py
  tests/api/test_predict_what_if.py` green.
- Running orchestrator over a 5-SKU set produces 5 rows in `tuning_runs`
  keyed by the same `session_id`.
- `/predict` called twice with and without the same `future_x_override`
  returns different forecasts in the first call and the baseline in the
  second.

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
- MLflow model registry with versioning + lineage — `ModelArtifactCache`
  in Phase 4 is hash-keyed blobs only; the registry layer (versions,
  metadata, signed-URL artifact downloads) is owned by cloud-native
  Phase 5
- Hierarchical reconciliation — `DatasetBundle.hierarchy` is typed but
  unconsumed; lands when M5 is wired (architecture.md scope boundary)
