# [ARCHIVED — SUPERSEDED]

**Status:** Archived on 2026-05-21. All blocking issues identified in this review were resolved in the subsequent revision of `docs/PLAN.md` (post-iteration). The current `PLAN.md` supersedes this document entirely. Do not use this review for dispatch decisions.

**What changed:** All 5 recommended patches (Cost `mode` default, session ID SHA256 hash, `_process_global_panel` explicit signature, `calibre/ordering/simulation/` path fix, and `Regret` pre-work) were applied to `PLAN.md`. Additional telemetry protocols (`PROGRESS.md` resume, `HALT.md` halt) were added.

---

     1|# Pre-flight Review: PLAN.md vs. Codebase
     2|
     3|**Date:** 2026-05-21
     4|**Scope:** Verify every file, signature, and dependency referenced in `PLAN.md` against the actual post-Ray codebase at commit `b6e81d0`.
     5|**Status:** BLOCKING ISSUES FOUND — do not dispatch Phase 1 without fixing the stale references below.
     6|
     7|---
     8|
     9|## 1. Verified Claims (plan matches reality)
    10|
    11|### Phase 1 · PnO HPO
    12|
    13|| Claim | Location | Status |
    14||---|---|---|
    15|| Sequential fallback with `RuntimeWarning` exists | `optimizer.py:243-250` | **EXACT MATCH** |
    16|| `_evaluate_candidate` overwrites `value` each origin | `optimizer.py:212-220` | **CONFIRMED** — `value` reassigned every iteration |
    17|| `_trainable` reports per-origin via `tune.report` | `optimizer.py:310` | **CONFIRMED** — no accumulation across origins |
    18|| `Cost.evaluate` sums entire frame into one demand scalar | `objectives.py:60` | **CONFIRMED** — `demand = float(actuals.dropna().sum())` |
    19|
    20|### Phase 2 · State + Session Identity
    21|
    22|| Claim | Location | Status |
    23||---|---|---|
    24|| `RUNTIME_PARTITION = "__runtime__"` hard-coded | `storage/state.py:8` | **EXACT MATCH** |
    25|| `SqlConformalStateStore` has `(run_id, partition)` PK | `storage/models.py:30-40` | **CONFIRMED** — no `session_id` column |
    26|| `backend.py` uses single partition for get/upsert | `backend.py:500, 509` | **CONFIRMED** — both pass `RUNTIME_PARTITION` |
    27|| `pending` is in-process `list[pd.DataFrame]` | `decision_loop.py:191` | **CONFIRMED** |
    28|| `BackendEngine` hydrates on start, persists after each origin | `backend.py:496-509` | **CONFIRMED** — `from_state` on `_restore_conformal_state`, `upsert` on `_persist_conformal_state` |
    29|
    30|### Phase 3 · InventoryAdapter + API
    31|
    32|| Claim | Location | Status |
    33||---|---|---|
    34|| `DatasetAdapter` Protocol exists | `execution/dataset.py:21-24` | **CONFIRMED** |
    35|| `ForecastTask` has no `task_group` | `core/forecast_task.py:19-24` | **CONFIRMED** |
    36|| `global_refs` run serially on driver | `backend.py:597-607` | **CONFIRMED** — `_run_global_scope` loops over refs synchronously |
    37|| API only has `/forecasts` and `/backtests` | `api/main.py:55-86` | **CONFIRMED** |
    38|
    39|### Phase 4 · Unified Search Space
    40|
    41|| Claim | Location | Status |
    42||---|---|---|
    43|| `TuningTask.search_space` returns `dict` | `tuning/task.py:21` | **CONFIRMED** — `Callable[[optuna.Trial], dict]` |
    44|| No `Regret` objective in `tuning/objectives.py` | `tuning/objectives.py` | **CONFIRMED** — only `Accuracy`, `Cost`, `Pareto` |
    45|| No `ModelArtifactCache` exists | `forecasting/` | **CONFIRMED** — directory has no `cache.py` |
    46|| `AdaptiveAlphaController` tracks `_error_history` | `conformal/controllers.py:54` | **CONFIRMED** — private attr `_error_history: list[int]` |
    47|
    48|---
    49|
    50|## 2. Stale References (plan/audit cites code that does not exist)
    51|
    52|### 2.1 `deserialize_calibration_state` — NOT FOUND
    53|
    54|- **Audit claim:** "dead `deserialize_calibration_state` symbol at `calibre/conformal/__init__.py:29,63`"
    55|- **Reality:** `calibre/conformal/__init__.py` has no such symbol. Line 29 is `absolute_error_score` in the `__all__` list.
    56|- **Impact:** LOW — nothing to delete, but the agent may search for a ghost symbol.
    57|- **Fix:** Strike this reference from the audit. No action needed.
    58|
    59|### 2.2 `eval/regret.py` — NOT FOUND
    60|
    61|- **Audit claim:** "`eval/regret.py` computes `cost - cost_oracle` post-hoc"
    62|- **Reality:** No `eval/regret.py` exists anywhere in the repo. No module named `calibre.eval`.
    63|- **Impact:** HIGH for P5 — `Regret` objective is supposed to wrap existing regret computation, but the source file does not exist. The agent will have to implement regret from scratch or skip it.
    64|- **Fix:** Either create `calibre/evaluation/regret.py` as part of P1-P3 groundwork, or re-scope P5 to exclude `Regret` as a `TuningObjective`.
    65|
    66|### 2.3 `calibre/simulation/` — PATH MISMATCH
    67|
    68|- **Plan claim:** "`calibre/simulation/` (accept injected initial `ProductState`)"
    69|- **Reality:** The simulation code lives at `calibre/ordering/simulation/`, not `calibre/simulation/`.
    70|- **Impact:** MEDIUM — the agent will create files in the wrong place or fail to find existing code.
    71|- **Fix:** Update all plan references to `calibre/ordering/simulation/`.
    72|
    73|---
    74|
    75|## 3. Signature Mismatches (plan assumes shapes that don't hold)
    76|
    77|### 3.1 `session_id = hash(tenant, sku_set, model_config, conformal_config)` — WILL RAISE
    78|
    79|- **Plan claim:** "derive via `hash(tenant, sku_set, model_config, conformal_config)`"
    80|- **Reality:** `model_config` and `conformal_config` are `dict`s. `hash((..., dict, ...))` raises `TypeError: unhashable type: 'dict'`.
    81|- **Impact:** BLOCKING for P2 — this line will crash at runtime.
    82|- **Fix:** Use a stable serialization: `json.dumps((tenant, sorted(sku_set), model_config, conformal_config), sort_keys=True)` + `hashlib.sha256(...).hexdigest()[:16]`.
    83|
    84|### 3.2 `Pareto` constructs `Cost` internally — will break when `Cost` gains `mode`
    85|
    86|- **Plan claim:** "Add `mode: Literal['perhorizon', 'cumulative']` to `Cost`"
    87|- **Reality:** `Pareto.evaluate` at `objectives.py:76` instantiates `Cost(self.decision_rule_fn(...), self.arithmetic, self.costs)` with three positional args. Adding `mode` as a required arg breaks this call.
    88|- **Impact:** BLOCKING for P1(c) — `Pareto` will fail to construct `Cost`.
    89|- **Fix:** Update `Pareto.evaluate` to pass `mode` (default `"perhorizon"`), or make `mode` kw-only with default `"perhorizon"` on `Cost`.
    90|
    91|### 3.3 `AdaptiveAlphaController.error_history` is private
    92|
    93|- **Plan claim:** "derived from `AdaptiveAlphaController.error_history`"
    94|- **Reality:** The attribute is `_error_history` (private, line 54). There is no public accessor.
    95|- **Impact:** MEDIUM for P4 — the agent will get `AttributeError`.
    96|- **Fix:** Either add a public `error_history` property to `AdaptiveAlphaController`, or access via `get_state()["error_history"]`.
    97|
    98|---
    99|
   100|## 4. Three Highest-Risk Assumptions
   101|
   102|### 4.1 ProductState Has Two Incompatible Shapes
   103|
   104|There are **two** `ProductState` classes:
   105|
   106|1. `calibre/ordering/simulation/state.py::ProductState` — generic, has `pipeline: deque[float]` (line 27)
   107|2. `benchmarks/vn2/simulator.py::ProductState` — VN2-specific, has `in_transit_w1: float, in_transit_w2: float` (lines 41-42)
   108|
   109|The `InventoryAdapter` Protocol (P3) says `load_state(...) -> ProductState`. But which one? The VN2 benchmark uses its own shape with `_to_generic()` conversion (line 63). If `InventoryAdapter` returns the generic shape, the VN2 benchmark needs to be updated to convert. If it returns the VN2 shape, the generic simulator breaks.
   110|
   111|**Risk:** The agent will pick one shape, and either VN2 tests or generic simulation tests will fail. The plan does not mention updating the VN2 benchmark.
   112|
   113|**Fix:** Explicitly state in P3 that `InventoryAdapter` returns the **generic** `calibre.ordering.simulation.ProductState`, and add a `_to_generic()` adapter in the VN2 benchmark (or make VN2's `ProductState` a subclass of the generic one).
   114|
   115|### 4.2 Global Model Fan-Out Requires a New Ray Task Shape
   116|
   117|The plan says: "Wrap each global model config in its own `@ray.remote` task with the full panel."
   118|
   119|Current code:
   120|- `_process_task_ref(ref, origin, local_scope=...)` processes **one** ref at a time (backend.py:164-189)
   121|- `_run_global_scope` calls it in a driver loop (backend.py:605-606)
   122|- `_run_refs_on_ray` chunks refs and dispatches `_process_task_ref` remotely (backend.py:627-655)
   123|
   124|For global models, each task needs the **full panel** (all SKUs' histories), not just its own ref. The current `_process_task_ref` takes a single ref. A new function like `_process_global_task(refs: list[ForecastTaskRef], origin: pd.Timestamp)` is needed, which materializes ALL refs inside the Ray worker, fits the global model on the combined panel, and returns predictions for all SKUs.
   125|
   126|**Risk:** The agent will try to add `@ray.remote` to the existing `_process_task_ref` and pass a single global ref, which will not give the global model the full panel. The predictions will be wrong.
   127|
   128|**Fix:** Add a new `_process_global_panel` function to the plan, with a clear signature and materialization logic. Do not reuse `_process_task_ref` for global fan-out.
   129|
   130|### 4.3 Conformal State Serialization for Ray Tune — URI vs. Ray Object Store
   131|
   132|The plan says: "serialise the seed conformal state to an object-store URI keyed by `trial_id`" and "hydrate via `SymmetricIntervalRuntime.from_state(uri)`".
   133|
   134|Current code:
   135|- `to_json_safe_state()` returns a `dict` (runtime.py:53-55)
   136|- `SymmetricIntervalRuntime.from_state(config, state)` takes a `dict`, not a URI (runtime.py:185-198)
   137|
   138|There are two ways to pass state into a Ray Tune trial:
   139|1. **Ray object store:** `state_ref = ray.put(state_dict)` before tuning, then `state_dict = ray.get(state_ref)` inside `_trainable`.
   140|2. **File URI:** Write JSON to a temp file, pass the path in the trial config, read it inside `_trainable`.
   141|
   142|The plan mixes both concepts — it says "object-store URI" but `from_state` doesn't accept URIs.
   143|
   144|**Risk:** The agent will invent a URI-loading layer that doesn't exist, or try to pass a URI to `from_state` and get a `TypeError`.
   145|
   146|**Fix:** Explicitly choose one mechanism. Recommendation: use `ray.put` + `ray.get` because the state is small (<10KB per partition) and Ray's object store is already in use. Update the plan to say: "Snapshot conformal state dict via `ray.put`, pass the ObjectRef into the trial config, and hydrate via `from_state(ray.get(ref))` inside `_trainable`."
   147|
   148|---
   149|
   150|## 5. Dependency Checks (imports used in plan)
   151|
   152|| Import | Source | Status |
   153||---|---|---|
   154|| `alembic.command`, `alembic.config` | `alembic` package | **INSTALLED** — used in `test_storage_migrations.py` |
   155|| `optuna` | `optuna` package | **INSTALLED** — used throughout tuning |
   156|| `ray`, `ray.tune` | `ray` package | **INSTALLED** — Ray migration PR already landed |
   157|| `sqlalchemy` | `sqlalchemy` package | **INSTALLED** — used in storage layer |
   158|| `prometheus_client` | `prometheus-client` package | **INSTALLED** — used in metrics.py |
   159|| `fastapi` | `fastapi` package | **INSTALLED** — used in api/main.py |
   160|
   161|**All required dependencies are present in the venv.** No new packages needed for P1-P4.
   162|
   163|---
   164|
   165|## 6. Test Infrastructure
   166|
   167|| Test file (plan) | Exists today? | Notes |
   168||---|---|---|
   169|| `tests/tuning/test_ray_tune_with_conformal.py` | **NO** — new |
   170|| `tests/tuning/test_cost_objective_aggregation.py` | **NO** — new |
   171|| `tests/tuning/test_cost_mode_dispatch.py` | **NO** — new |
   172|| `tests/storage/test_per_partition_state.py` | **NO** — new |
   173|| `tests/storage/test_session_keyed_resume.py` | **NO** — new |
   174|| `tests/execution/test_pending_store.py` | **NO** — new |
   175|| `tests/cli/test_maint_compact.py` | **NO** — new |
   176|| `tests/execution/test_inventory_adapter.py` | **NO** — new |
   177|| `tests/execution/test_global_fanout.py` | **NO** — new |
   178|| `tests/execution/test_task_grouping.py` | **NO** — new |
   179|| `tests/api/test_lifecycle_endpoints.py` | **NO** — new |
   180|| `tests/tuning/test_unified_candidate.py` | **NO** — new |
   181|| `tests/tuning/test_regret_objective.py` | **NO** — new |
   182|| `tests/forecasting/test_model_cache.py` | **NO** — new |
   183|| `tests/observability/test_coverage_drift.py` | **NO** — new |
   184|| `tests/api/test_tune_endpoint.py` | **NO** — new |
   185|
   186|All 16 test files are new. The agent must create them. This is fine, but it means ~16 new files across 4 phases, which is a lot of test-writing overhead. The plan should acknowledge this.
   187|
   188|---
   189|
   190|## 7. Migration Path
   191|
   192|Current Alembic setup:
   193|- `alembic.ini` points to `calibre/storage/migrations`
   194|- `env.py` uses `Base.metadata` (autodetects models)
   195|- `versions/0001_initial.py` creates `runs`, `conformal_state`, `forecast_pointers`
   196|
   197|P2 adds:
   198|- `session_id` column to `conformal_state`
   199|- `last_updated_at` / TTL column to `conformal_state`
   200|- New `pending_observations` table
   201|
   202|This will auto-generate cleanly via `alembic revision --autogenerate`. The plan correctly identifies this.
   203|
   204|---
   205|
   206|## 8. Verdict
   207|
   208|**P1 is dispatchable after 2 fixes:**
   209|1. Make `mode` on `Cost` optional with default `"perhorizon"` (to avoid breaking `Pareto`)
   210|2. Clarify that `total_cost` accumulation replaces the per-origin `value` overwrite, and the intermediate metric for ASHA is `total_cost` (not running mean — running mean would be non-monotone, confusing ASHA)
   211|
   212|**P2 is dispatchable after 1 fix:**
   213|1. Replace `hash(tenant, sku_set, model_config, conformal_config)` with a stable JSON-based hash
   214|
   215|**P3 is dispatchable after 2 fixes:**
   216|1. Fix `calibre/simulation/` path to `calibre/ordering/simulation/`
   217|2. Add explicit `_process_global_panel` function signature for global fan-out (don't reuse `_process_task_ref`)
   218|
   219|**P4 is dispatchable after 1 fix:**
   220|1. Either create `calibre/evaluation/regret.py` as pre-work, or remove `Regret` from P4 scope
   221|
   222|**BLOCKING ISSUE for all phases:** The 16 new test files. A 10-hour Codex run that writes 16 test files from scratch risks test-quality drift. Consider whether some tests can be consolidated (e.g., `test_cost_objective_aggregation.py` and `test_cost_mode_dispatch.py` into one file).
   223|
   224|---
   225|
   226|## 9. Recommended Plan Patches
   227|
   228|Apply these patches to `PLAN.md` before dispatch:
   229|
   230|### Patch 1: Cost mode default
   231|
   232|In Phase 1, change:
   233|```
   234|Add `mode: Literal["perhorizon", "cumulative"]` to `Cost`
   235|```
   236|To:
   237|```
   238|Add `mode: Literal["perhorizon", "cumulative"] = "perhorizon"` to `Cost` (kw-only, default preserves Pareto compatibility)
   239|```
   240|
   241|### Patch 2: Session ID hash
   242|
   243|In Phase 2, change:
   244|```
   245|Derive via `hash(tenant, sku_set, model_config, conformal_config)`
   246|```
   247|To:
   248|```
   249|Derive via `hashlib.sha256(json.dumps((tenant, sorted(sku_set), model_config, conformal_config), sort_keys=True).encode()).hexdigest()[:16]`
   250|```
   251|
   252|### Patch 3: Global fan-out function
   253|
   254|In Phase 3, after "Global-model fan-out", add:
   255|```
   256|Implement `_process_global_panel(refs: list[ForecastTaskRef], origin: pd.Timestamp) -> pd.DataFrame` that materializes all refs, concatenates histories into a full panel, fits the global model once, and returns predictions for all SKUs. Wrap this in `@ray.remote`, not `_process_task_ref`.
   257|```
   258|
   259|### Patch 4: Path fix
   260|
   261|In Phase 3, change all `calibre/simulation/` to `calibre/ordering/simulation/`.
   262|
   263|### Patch 5: Regret scope
   264|
   265|In Phase 4, either:
   266|- Remove `Regret` from scope, OR
   267|- Add a P0 task: "Create `calibre/evaluation/regret.py` with `compute_regret(cost_frame, oracle_cost_frame) -> float`"
   268|
   269|---
   270|
   271|*Review complete. Five patches recommended before dispatch. No dependency gaps. Three highest-risk assumptions identified: ProductState dual-shape, global fan-out task shape, and conformal state serialization mechanism.*
   272|