# Pre-flight Review: PLAN.md vs. Codebase

**Date:** 2026-05-21
**Scope:** Verify every file, signature, and dependency referenced in `PLAN.md` against the actual post-Ray codebase at commit `b6e81d0`.
**Status:** BLOCKING ISSUES FOUND — do not dispatch Phase 1 without fixing the stale references below.

---

## 1. Verified Claims (plan matches reality)

### Phase 1 · PnO HPO

| Claim | Location | Status |
|---|---|---|
| Sequential fallback with `RuntimeWarning` exists | `optimizer.py:243-250` | **EXACT MATCH** |
| `_evaluate_candidate` overwrites `value` each origin | `optimizer.py:212-220` | **CONFIRMED** — `value` reassigned every iteration |
| `_trainable` reports per-origin via `tune.report` | `optimizer.py:310` | **CONFIRMED** — no accumulation across origins |
| `Cost.evaluate` sums entire frame into one demand scalar | `objectives.py:60` | **CONFIRMED** — `demand = float(actuals.dropna().sum())` |

### Phase 2 · State + Session Identity

| Claim | Location | Status |
|---|---|---|
| `RUNTIME_PARTITION = "__runtime__"` hard-coded | `storage/state.py:8` | **EXACT MATCH** |
| `SqlConformalStateStore` has `(run_id, partition)` PK | `storage/models.py:30-40` | **CONFIRMED** — no `session_id` column |
| `backend.py` uses single partition for get/upsert | `backend.py:500, 509` | **CONFIRMED** — both pass `RUNTIME_PARTITION` |
| `pending` is in-process `list[pd.DataFrame]` | `decision_loop.py:191` | **CONFIRMED** |
| `BackendEngine` hydrates on start, persists after each origin | `backend.py:496-509` | **CONFIRMED** — `from_state` on `_restore_conformal_state`, `upsert` on `_persist_conformal_state` |

### Phase 3 · InventoryAdapter + API

| Claim | Location | Status |
|---|---|---|
| `DatasetAdapter` Protocol exists | `execution/dataset.py:21-24` | **CONFIRMED** |
| `ForecastTask` has no `task_group` | `core/forecast_task.py:19-24` | **CONFIRMED** |
| `global_refs` run serially on driver | `backend.py:597-607` | **CONFIRMED** — `_run_global_scope` loops over refs synchronously |
| API only has `/forecasts` and `/backtests` | `api/main.py:55-86` | **CONFIRMED** |

### Phase 4 · Unified Search Space

| Claim | Location | Status |
|---|---|---|
| `TuningTask.search_space` returns `dict` | `tuning/task.py:21` | **CONFIRMED** — `Callable[[optuna.Trial], dict]` |
| No `Regret` objective in `tuning/objectives.py` | `tuning/objectives.py` | **CONFIRMED** — only `Accuracy`, `Cost`, `Pareto` |
| No `ModelArtifactCache` exists | `forecasting/` | **CONFIRMED** — directory has no `cache.py` |
| `AdaptiveAlphaController` tracks `_error_history` | `conformal/controllers.py:54` | **CONFIRMED** — private attr `_error_history: list[int]` |

---

## 2. Stale References (plan/audit cites code that does not exist)

### 2.1 `deserialize_calibration_state` — NOT FOUND

- **Audit claim:** "dead `deserialize_calibration_state` symbol at `calibre/conformal/__init__.py:29,63`"
- **Reality:** `calibre/conformal/__init__.py` has no such symbol. Line 29 is `absolute_error_score` in the `__all__` list.
- **Impact:** LOW — nothing to delete, but the agent may search for a ghost symbol.
- **Fix:** Strike this reference from the audit. No action needed.

### 2.2 `eval/regret.py` — NOT FOUND

- **Audit claim:** "`eval/regret.py` computes `cost - cost_oracle` post-hoc"
- **Reality:** No `eval/regret.py` exists anywhere in the repo. No module named `calibre.eval`.
- **Impact:** HIGH for P5 — `Regret` objective is supposed to wrap existing regret computation, but the source file does not exist. The agent will have to implement regret from scratch or skip it.
- **Fix:** Either create `calibre/evaluation/regret.py` as part of P1-P3 groundwork, or re-scope P5 to exclude `Regret` as a `TuningObjective`.

### 2.3 `calibre/simulation/` — PATH MISMATCH

- **Plan claim:** "`calibre/simulation/` (accept injected initial `ProductState`)"
- **Reality:** The simulation code lives at `calibre/ordering/simulation/`, not `calibre/simulation/`.
- **Impact:** MEDIUM — the agent will create files in the wrong place or fail to find existing code.
- **Fix:** Update all plan references to `calibre/ordering/simulation/`.

---

## 3. Signature Mismatches (plan assumes shapes that don't hold)

### 3.1 `session_id = hash(tenant, sku_set, model_config, conformal_config)` — WILL RAISE

- **Plan claim:** "derive via `hash(tenant, sku_set, model_config, conformal_config)`"
- **Reality:** `model_config` and `conformal_config` are `dict`s. `hash((..., dict, ...))` raises `TypeError: unhashable type: 'dict'`.
- **Impact:** BLOCKING for P2 — this line will crash at runtime.
- **Fix:** Use a stable serialization: `json.dumps((tenant, sorted(sku_set), model_config, conformal_config), sort_keys=True)` + `hashlib.sha256(...).hexdigest()[:16]`.

### 3.2 `Pareto` constructs `Cost` internally — will break when `Cost` gains `mode`

- **Plan claim:** "Add `mode: Literal['perhorizon', 'cumulative']` to `Cost`"
- **Reality:** `Pareto.evaluate` at `objectives.py:76` instantiates `Cost(self.decision_rule_fn(...), self.arithmetic, self.costs)` with three positional args. Adding `mode` as a required arg breaks this call.
- **Impact:** BLOCKING for P1(c) — `Pareto` will fail to construct `Cost`.
- **Fix:** Update `Pareto.evaluate` to pass `mode` (default `"perhorizon"`), or make `mode` kw-only with default `"perhorizon"` on `Cost`.

### 3.3 `AdaptiveAlphaController.error_history` is private

- **Plan claim:** "derived from `AdaptiveAlphaController.error_history`"
- **Reality:** The attribute is `_error_history` (private, line 54). There is no public accessor.
- **Impact:** MEDIUM for P4 — the agent will get `AttributeError`.
- **Fix:** Either add a public `error_history` property to `AdaptiveAlphaController`, or access via `get_state()["error_history"]`.

---

## 4. Three Highest-Risk Assumptions

### 4.1 ProductState Has Two Incompatible Shapes

There are **two** `ProductState` classes:

1. `calibre/ordering/simulation/state.py::ProductState` — generic, has `pipeline: deque[float]` (line 27)
2. `benchmarks/vn2/simulator.py::ProductState` — VN2-specific, has `in_transit_w1: float, in_transit_w2: float` (lines 41-42)

The `InventoryAdapter` Protocol (P3) says `load_state(...) -> ProductState`. But which one? The VN2 benchmark uses its own shape with `_to_generic()` conversion (line 63). If `InventoryAdapter` returns the generic shape, the VN2 benchmark needs to be updated to convert. If it returns the VN2 shape, the generic simulator breaks.

**Risk:** The agent will pick one shape, and either VN2 tests or generic simulation tests will fail. The plan does not mention updating the VN2 benchmark.

**Fix:** Explicitly state in P3 that `InventoryAdapter` returns the **generic** `calibre.ordering.simulation.ProductState`, and add a `_to_generic()` adapter in the VN2 benchmark (or make VN2's `ProductState` a subclass of the generic one).

### 4.2 Global Model Fan-Out Requires a New Ray Task Shape

The plan says: "Wrap each global model config in its own `@ray.remote` task with the full panel."

Current code:
- `_process_task_ref(ref, origin, local_scope=...)` processes **one** ref at a time (backend.py:164-189)
- `_run_global_scope` calls it in a driver loop (backend.py:605-606)
- `_run_refs_on_ray` chunks refs and dispatches `_process_task_ref` remotely (backend.py:627-655)

For global models, each task needs the **full panel** (all SKUs' histories), not just its own ref. The current `_process_task_ref` takes a single ref. A new function like `_process_global_task(refs: list[ForecastTaskRef], origin: pd.Timestamp)` is needed, which materializes ALL refs inside the Ray worker, fits the global model on the combined panel, and returns predictions for all SKUs.

**Risk:** The agent will try to add `@ray.remote` to the existing `_process_task_ref` and pass a single global ref, which will not give the global model the full panel. The predictions will be wrong.

**Fix:** Add a new `_process_global_panel` function to the plan, with a clear signature and materialization logic. Do not reuse `_process_task_ref` for global fan-out.

### 4.3 Conformal State Serialization for Ray Tune — URI vs. Ray Object Store

The plan says: "serialise the seed conformal state to an object-store URI keyed by `trial_id`" and "hydrate via `SymmetricIntervalRuntime.from_state(uri)`".

Current code:
- `to_json_safe_state()` returns a `dict` (runtime.py:53-55)
- `SymmetricIntervalRuntime.from_state(config, state)` takes a `dict`, not a URI (runtime.py:185-198)

There are two ways to pass state into a Ray Tune trial:
1. **Ray object store:** `state_ref = ray.put(state_dict)` before tuning, then `state_dict = ray.get(state_ref)` inside `_trainable`.
2. **File URI:** Write JSON to a temp file, pass the path in the trial config, read it inside `_trainable`.

The plan mixes both concepts — it says "object-store URI" but `from_state` doesn't accept URIs.

**Risk:** The agent will invent a URI-loading layer that doesn't exist, or try to pass a URI to `from_state` and get a `TypeError`.

**Fix:** Explicitly choose one mechanism. Recommendation: use `ray.put` + `ray.get` because the state is small (<10KB per partition) and Ray's object store is already in use. Update the plan to say: "Snapshot conformal state dict via `ray.put`, pass the ObjectRef into the trial config, and hydrate via `from_state(ray.get(ref))` inside `_trainable`."

---

## 5. Dependency Checks (imports used in plan)

| Import | Source | Status |
|---|---|---|
| `alembic.command`, `alembic.config` | `alembic` package | **INSTALLED** — used in `test_storage_migrations.py` |
| `optuna` | `optuna` package | **INSTALLED** — used throughout tuning |
| `ray`, `ray.tune` | `ray` package | **INSTALLED** — Ray migration PR already landed |
| `sqlalchemy` | `sqlalchemy` package | **INSTALLED** — used in storage layer |
| `prometheus_client` | `prometheus-client` package | **INSTALLED** — used in metrics.py |
| `fastapi` | `fastapi` package | **INSTALLED** — used in api/main.py |

**All required dependencies are present in the venv.** No new packages needed for P1-P4.

---

## 6. Test Infrastructure

| Test file (plan) | Exists today? | Notes |
|---|---|---|
| `tests/tuning/test_ray_tune_with_conformal.py` | **NO** — new |
| `tests/tuning/test_cost_objective_aggregation.py` | **NO** — new |
| `tests/tuning/test_cost_mode_dispatch.py` | **NO** — new |
| `tests/storage/test_per_partition_state.py` | **NO** — new |
| `tests/storage/test_session_keyed_resume.py` | **NO** — new |
| `tests/execution/test_pending_store.py` | **NO** — new |
| `tests/cli/test_maint_compact.py` | **NO** — new |
| `tests/execution/test_inventory_adapter.py` | **NO** — new |
| `tests/execution/test_global_fanout.py` | **NO** — new |
| `tests/execution/test_task_grouping.py` | **NO** — new |
| `tests/api/test_lifecycle_endpoints.py` | **NO** — new |
| `tests/tuning/test_unified_candidate.py` | **NO** — new |
| `tests/tuning/test_regret_objective.py` | **NO** — new |
| `tests/forecasting/test_model_cache.py` | **NO** — new |
| `tests/observability/test_coverage_drift.py` | **NO** — new |
| `tests/api/test_tune_endpoint.py` | **NO** — new |

All 16 test files are new. The agent must create them. This is fine, but it means ~16 new files across 4 phases, which is a lot of test-writing overhead. The plan should acknowledge this.

---

## 7. Migration Path

Current Alembic setup:
- `alembic.ini` points to `calibre/storage/migrations`
- `env.py` uses `Base.metadata` (autodetects models)
- `versions/0001_initial.py` creates `runs`, `conformal_state`, `forecast_pointers`

P2 adds:
- `session_id` column to `conformal_state`
- `last_updated_at` / TTL column to `conformal_state`
- New `pending_observations` table

This will auto-generate cleanly via `alembic revision --autogenerate`. The plan correctly identifies this.

---

## 8. Verdict

**P1 is dispatchable after 2 fixes:**
1. Make `mode` on `Cost` optional with default `"perhorizon"` (to avoid breaking `Pareto`)
2. Clarify that `total_cost` accumulation replaces the per-origin `value` overwrite, and the intermediate metric for ASHA is `total_cost` (not running mean — running mean would be non-monotone, confusing ASHA)

**P2 is dispatchable after 1 fix:**
1. Replace `hash(tenant, sku_set, model_config, conformal_config)` with a stable JSON-based hash

**P3 is dispatchable after 2 fixes:**
1. Fix `calibre/simulation/` path to `calibre/ordering/simulation/`
2. Add explicit `_process_global_panel` function signature for global fan-out (don't reuse `_process_task_ref`)

**P4 is dispatchable after 1 fix:**
1. Either create `calibre/evaluation/regret.py` as pre-work, or remove `Regret` from P4 scope

**BLOCKING ISSUE for all phases:** The 16 new test files. A 10-hour Codex run that writes 16 test files from scratch risks test-quality drift. Consider whether some tests can be consolidated (e.g., `test_cost_objective_aggregation.py` and `test_cost_mode_dispatch.py` into one file).

---

## 9. Recommended Plan Patches

Apply these patches to `PLAN.md` before dispatch:

### Patch 1: Cost mode default

In Phase 1, change:
```
Add `mode: Literal["perhorizon", "cumulative"]` to `Cost`
```
To:
```
Add `mode: Literal["perhorizon", "cumulative"] = "perhorizon"` to `Cost` (kw-only, default preserves Pareto compatibility)
```

### Patch 2: Session ID hash

In Phase 2, change:
```
Derive via `hash(tenant, sku_set, model_config, conformal_config)`
```
To:
```
Derive via `hashlib.sha256(json.dumps((tenant, sorted(sku_set), model_config, conformal_config), sort_keys=True).encode()).hexdigest()[:16]`
```

### Patch 3: Global fan-out function

In Phase 3, after "Global-model fan-out", add:
```
Implement `_process_global_panel(refs: list[ForecastTaskRef], origin: pd.Timestamp) -> pd.DataFrame` that materializes all refs, concatenates histories into a full panel, fits the global model once, and returns predictions for all SKUs. Wrap this in `@ray.remote`, not `_process_task_ref`.
```

### Patch 4: Path fix

In Phase 3, change all `calibre/simulation/` to `calibre/ordering/simulation/`.

### Patch 5: Regret scope

In Phase 4, either:
- Remove `Regret` from scope, OR
- Add a P0 task: "Create `calibre/evaluation/regret.py` with `compute_regret(cost_frame, oracle_cost_frame) -> float`"

---

*Review complete. Five patches recommended before dispatch. No dependency gaps. Three highest-risk assumptions identified: ProductState dual-shape, global fan-out task shape, and conformal state serialization mechanism.*
