# Calibre Improvement Wave 1 — Unified Roadmap

**Status:** Open  
**Updated:** 2026-05-22  
**Sources:**
- [PR #35 deployment readiness assessment](pr35-deployment-assessment.md) (2026-05-21, vault)
- [Post-merge code review / thermo-nuclear audit](2026-05-22-codebase-audit.md) (2026-05-22, vault)

**Scope:** Close the gap between *demoable* and *deployable* across the API lifecycle, conformal online recalibration, data-plane edge, and benchmark hygiene.

---

## How to Read This Doc

This is a unified view of two prior assessments that are complementary, not overlapping:

- **[pr35-deployment-assessment.md](pr35-deployment-assessment.md)** — vertical functional review: bugs, deployment blockers, data-plane gaps.
- **[2026-05-22-codebase-audit.md](2026-05-22-codebase-audit.md)** — horizontal code-quality scan: root causes, type-system leaks, architectural boundary erosion.

The causal graph from the audit explains *why* several PR35 bugs exist. Each item below cross-references the source assessment and the root issue (R1–R5) where applicable.

---

## Executive Summary

The codebase is clean in the mathematical pillars (core, ordering, conformal) and accumulates slop at the *integration* boundaries: CLI config parsing, execution backend, tuning optimizer, and benchmark scripts. Slop concentrates wherever the library interfaces with external systems (YAML, Ray, Optuna, fsspec) or where prototypes hardened into production without refactoring.

Five interconnected root issues generate most of the observed symptoms:

- **R1:** Manual config parsing instead of schema validation (`cli/config.py`)
- **R2:** Type-system avoidance at integration boundaries (Ray, Optuna, fsspec)
- **R3:** Benchmark monolith absorbing library responsibilities (`run_benchmark.py`)
- **R4:** Missing protocols for external integrations (conformal state persistence, simulator)
- **R5:** Backend engine encapsulation leaks (private imports, duck typing, direct attribute access)

---

## P0 — Blocking (Deploy & Correct)

These prevent a safe deployment or produce silently wrong results.

### P0.1 /observe cumulative conformal mode is broken
- **Source:** pr35 §3a; AUDIT R4/R5
- **File:** `calibre/api/main.py`
- **Symptom:** `/observe` filters out rows with `NaN` bounds before calling `runtime.observe()`. In cumulative conformal mode, non-terminal rows intentionally carry `NaN` bounds; filtering them prevents the runtime from seeing intermediate actuals needed to update the cumulative residual pool.
- **Fix:** Centralize dispatch through `calibre/execution/decision_loop.py`, which already knows how to route observations to the correct runtime mode without pre-filtering bounds.

### P0.2 LifecycleStore is process-local
- **Source:** pr35 §1; AUDIT R5
- **File:** `calibre/api/main.py`
- **Symptom:** `LifecycleStore` is a global in-memory dict. Only `RunStore` and `ConformalStateStore` are SQL-backed. A second worker process has no visibility into sessions created by the first. A restart drops all in-flight fits, pending observations, and tune studies. Gunicorn/uvicorn with `workers > 1` will corrupt or lose state.
- **Fix:** Mirror the `ConformalStateStore` pattern — give `LifecycleStore` a SQL-backed implementation, or push session metadata into the existing SQL layer and keep only ephemeral coordination in memory.

### P0.3 /fit does not actually fit
- **Source:** pr35 §2; AUDIT R5
- **File:** `calibre/api/main.py`
- **Symptom:** `_run_fit_job` flips status flags but performs no actual model fitting. The fit runs lazily inside `/predict` (`_fit_predict_task`). `/fit` returns `SUCCEEDED` without validating that the model config is compatible with the data (frequency, regressors, horizon). You can POST garbage, get 202 -> SUCCEEDED, and only discover the failure at `/predict` time.
- **Fix:** Make `/fit` actually fit and persist artifacts. Include a config-validation pass so the endpoint only returns `SUCCEEDED` when the model is proven loadable. Wire `ModelArtifactCache` into the lifecycle.

### P0.4 Zero-order fallback exploited by HPO cost search
- **Source:** pr35 §5e; AUDIT R3
- **File:** `benchmarks/vn2/run_benchmark.py`
- **Symptom:** When an order policy errors, the benchmark falls back to ordering zero units. During cost tuning this is dangerous: zero orders produces a deterministic, low-cost outcome the optimizer can exploit. A broken policy looks like a "cheap but valid" configuration. The HPO can converge on the bug instead of the real optimum.
- **Fix:** Fail the trial on policy error. Zero-order fallback should only exist in a dedicated "degraded mode" test, not in the cost search path.

### P0.5 /observe silent return when last_calibrated is empty
- **Source:** pr35 §3b
- **File:** `calibre/api/main.py`
- **Symptom:** `_run_observe_job` silently returns when `last_calibrated` is empty. No log, no metric, no exception. In production this means the online recalibration pipeline can be dead for weeks without alarming.
- **Status:** **FIXED** on `deslop-audit` branch (commit `96348b9` — adds warning logging). Must be cherry-picked or merged before deployment.
- **Fix:** Already done. Verify it lands on main.

---

## P1 — Structural (Scale & Maintain)

These are the difference between a prototype and a production service.

### P1.1 VN2 benchmark monolith
- **Source:** pr35 §5b; AUDIT R3/R5
- **File:** `benchmarks/vn2/run_benchmark.py` (~2,300 LOC)
- **Symptom:** The file owns data prep, HPO, Ray Tune, replay, CRC tuning, diagnostics, logging, and orchestration. Hard to review, test in isolation, or keep in sync with library changes. The module docstring lists 5 documented gaps that the script "works around" — this is a second, untested implementation of the pipeline.
- **Fix:** Split into coordinated modules: `data.py`, `tuning.py`, `replay.py`, `diagnostics.py`, with `run_benchmark.py` as a thin orchestration shell. Target: <800 LOC for the runner.

### P1.2 Deduplicate VN2 tuning logic
- **Source:** pr35 §5c; AUDIT R3/R5
- **File:** `benchmarks/vn2/run_benchmark.py`, `calibre/tuning/optimizer.py`
- **Symptom:** VN2 tuning reimplements canonical tuning behavior from `calibre/tuning/optimizer.py`. Any change to search space shaping, ASHA pruning, or cost evaluation must be manually ported or the benchmark stops measuring the actual runtime.
- **Fix:** Refactor the benchmark to call `calibre.tuning.optimizer` directly, injecting VN2-specific adapters where needed. The benchmark should measure the product, not shadow it. Also extract `_cap_threaded_config` (duplicated in `backend.py` and `optimizer.py`) to a single module (`calibre.execution.threading`).

### P1.3 Cost search swallows infra failures
- **Source:** pr35 §5d; AUDIT R3
- **File:** `benchmarks/vn2/run_benchmark.py`
- **Symptom:** Cost search reaches into `_ot_study` (Optuna private attribute) and converts broad failures (infra misconfig, import errors, Ray worker crashes) into `inf` cost. A broken environment looks like a "bad trial" rather than a hard failure. Upgrades to Ray or Optuna can break the benchmark without a type error.
- **Fix:** Distinguish *search failure* (bad hyperparameters -> high cost) from *infra failure* (exception -> raise / log / metric). Do not swallow exceptions into `inf` indiscriminately.

### P1.4 Replace manual YAML parsing with schema validation
- **Source:** AUDIT R1/R2
- **File:** `calibre/cli/config.py`
- **Symptom:** Heavy use of `Any` in parsing helpers, `# type: ignore[arg-type]` to bypass Literal validation, and manual key whitelisting. Every new config field requires manual validation code. Type checker cannot help. Errors surface at runtime when a user runs a backtest.
- **Fix:** Replace with `pydantic` or `dataclasses` + `cattrs`. This eliminates ~15 `Any` annotations, 3 `# type: ignore[arg-type]`, and the entire class of runtime `ValueError` from malformed configs. Estimated LOC reduction: -80.

### P1.5 Add PartitionedConformalRuntime Protocol
- **Source:** AUDIT R4/R5
- **File:** `calibre/execution/backend.py`
- **Symptom:** `_restore_conformal_state` and `_persist_conformal_state` use `getattr(conformal_runtime, "get_partition_states", None)` and `getattr(self.conformal_state_store, "list_for_run", None)` rather than a protocol. This makes static analysis impossible and invites runtime `AttributeError` if the attribute exists but is not callable.
- **Fix:** Define `PartitionedConformalRuntime` as a Protocol. Make `SymmetricIntervalRuntime` implement it explicitly. Replace `getattr` chains with `isinstance(runtime, PartitionedConformalRuntime)`.

### P1.6 Type the Ray integration
- **Source:** AUDIT R2/R4
- **File:** `calibre/execution/backend.py`, `calibre/tuning/optimizer.py`
- **Symptom:** `Any` used for `self._ray`, remote task refs, and `_ensure_ray` return. Ray has typed stubs (`py.typed`).
- **Fix:** Replace `Any` with `ray.actor.ActorHandle`, `ray.remote_function.RemoteFunction`, etc.

### P1.7 Wire Regret end-to-end
- **Source:** pr35 §4
- **File:** `calibre/api/main.py`, `calibre/tuning/optimizer.py`
- **Symptom:** `Regret` needs `oracle_cost` precomputed once before the study. The PR adds `compute_regret` but no `/tune` plumbing to populate the oracle. The path is sketched, not paved.
- **Fix:** Precompute oracle inside `/tune`, store on `TuneRecord`, surface in `/studies/{id}`.

### P1.8 Data-plane ingestion seam
- **Source:** pr35 §5a, §6
- **Files:** `calibre/api/main.py`, `calibre/execution/`
- **Symptom:** Sales and inventory still arrive as JSON in request bodies. For real volume you need a parquet / SQL ingestion seam. The `DatasetAdapter` / `InventoryAdapter` protocols are the right shape but only `SyntheticInventoryAdapter` is implemented. No persistent order table. No events calendar. Promotions are inference-side only (`future_x_override` on `/predict`).
- **Fix:** Ship a SQL-backed `InventoryAdapter` and a thin `SalesAdapter` for parquet / SQL ingestion. Keep `Synthetic*` for tests. Add persistent `Order` table. Decide whether `tenant` needs auth enforcement (currently an honor-system string in `session_id`).

---

## P2 — Hygiene

Quick wins that improve signal-to-noise in reviews and refactors.

### P2.1 Remove type-system slop in conformal numerics
- **Source:** AUDIT §3.3
- **File:** `calibre/conformal/adaptive.py`, `calibre/conformal/numerics.py`
- **Fix:**
  - Overload `_clip_alpha` return type for array-vs-scalar inputs (removes `type: ignore[assignment]` on lines 153, 162).
  - Fix `_validate_quantile_rule` to return `Literal["conformal", "higher"]` directly or use `typing.cast` instead of `type: ignore[return-value]` (line 56).
  - Add missing type annotations on `MultiStepAdaptiveConformalInference.__init__` params (`alpha`, `gamma`, `initial_alpha`, `initial_radius`).

### P2.2 Narrow broad exceptions in API and evaluation
- **Source:** AUDIT §3.5, §3.8
- **Files:** `calibre/api/main.py`, `calibre/evaluation/point_metrics.py`
- **Fix:**
  - In `api/main.py` (lines 278, 326, 370, 580): verify all four `except Exception` blocks log the full traceback before converting to HTTP 500 or job failure. Narrow to specific exception types where possible.
  - In `evaluation/point_metrics.py` (line 323): narrow `except Exception` to the specific exception the underlying metric raises (likely `ValueError` or `ZeroDivisionError`).

### P2.3 Remove print() from CLI library boundary
- **Source:** AUDIT §3.9
- **File:** `calibre/cli/commands.py`
- **Fix:** Replace `print()` in `run_config`, `validate`, `health` with structured logger calls or return strings for display. `run_config` is called programmatically from `api/main.py`; side effects in library-adjacent code are slop.

### P2.4 Extract deeply nested _trainable
- **Source:** AUDIT §3.10
- **File:** `calibre/tuning/optimizer.py`
- **Fix:** Extract `_trainable` (lines 416-473, nesting depth 6+) to a module-level function with explicit parameters. Also DRY the ~30 lines of identical setup logic shared between `_evaluate_candidate` and `_trainable`.

### P2.5 Fix remaining type ignores
- **Source:** AUDIT §3.6, §3.12
- **Files:** `calibre/execution/backend.py`, `tests/`
- **Fix:**
  - `backend.py:512` — replace `type: ignore[union-attr]` with `assert order_ledger is not None`.
  - `evaluation/forecast_metrics.py:79` — remove `type: ignore[assignment]`; change to `name: str | None = getattr(...)`.
  - `cli/config.py:182,186,258` — replace `type: ignore[arg-type]` with `typing.cast(Literal[...], validated_str)`.
  - Tests — type mocks with `create_autospec(RunStore)` instead of `type: ignore`.

---

## Causal Graph

```
R1 (Manual Config Parsing)
  |
  +--> R2 (Type Avoidance) ..................> cli/config.py, cli/commands.py
  |
R4 (Missing Protocols)
  |
  +--> R2 (Type Avoidance) ..................> backend.py, optimizer.py
  |
R5 (Backend Encapsulation Leaks)
  |
  +--> R3 (Benchmark Monolith) ..............> run_benchmark.py
  |     |
  |     +--> Duplication with tuning/optimizer > _cap_threaded_config, HPO logic
  |
  +--> R4 (Missing Protocols) ...............> conformal state persistence
```

---

## Recommended Fix Order

This order respects dependencies: backend fixes unblock API fixes; API fixes unblock deployment; benchmark fixes come after backend stabilization so the benchmark can call the product instead of shadowing it.

1. **Cherry-pick or merge the `/observe` silent-return fix** from `deslop-audit` (already done there).
2. **Fix `/observe` cumulative path** — centralize dispatch through `decision_loop.py`. Unblocks online recalibration for CRC / ACIC.
3. **Extract `_cap_threaded_config`** to `calibre.execution.threading` — trivial, removes duplication.
4. **Add `PartitionedConformalRuntime` Protocol** and replace `getattr` duck-typing in `backend.py`.
5. **Back `LifecycleStore` with SQL** — session state survives restarts and multi-worker deployments.
6. **Make `/fit` actually fit** — eager training + config validation + `ModelArtifactCache` wiring.
7. **Replace manual YAML parsing** in `cli/config.py` with `pydantic`/`cattrs`. This unlocks downstream type safety for all CLI-driven flows.
8. **Split VN2 benchmark** into `data.py`, `tuning.py`, `replay.py`, `diagnostics.py`. Do this *before* adding more tuning behavior.
9. **Deduplicate VN2 tuning** — refactor benchmark to call `calibre.tuning.optimizer` directly.
10. **Clean up cost-search error handling** — separate infra failures from high-cost trials; remove zero-order fallback from search path.
11. **Ship SQL-backed `InventoryAdapter`** and thin `SalesAdapter` for parquet / SQL ingestion.
12. **Wire `Regret` end-to-end** — precompute oracle inside `/tune`, store on `TuneRecord`.
13. **Type-system sweep** — Ray stubs, conformal numerics, API exception narrowing, test mocks.
14. **Decide whether `tenant` needs auth enforcement**.

---

## Verification

- `uv run ruff check .` — must pass after every batch.
- `uv run pytest` — targeted tests for touched modules must pass.
- `uv run mypy calibre/` — net reduction in `# type: ignore` and `Any` counts is the success metric for P2.

---

## Bottom Line

The bones of a deployable demand-planning service are in the codebase. What remains is boundary work: closing the gap between demoable and deployable by fixing the integration seams where slop accumulated. The mathematical pillars (core, ordering, conformal) are solid — the work is at the edges where the library meets external systems and where prototypes hardened without refactoring.

Small in code, but the difference between *demoable* and *deployable*.
