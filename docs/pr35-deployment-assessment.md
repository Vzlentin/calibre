# PR #35 — Deployment Readiness & Post-Merge Code Review

**Reviews:**
- Deployment readiness assessment (Hermes), 2026-05-21
- Post-merge code review (Cursor thermo-nuclear), 2026-05-22

**Scope:** `calibre/api/main.py`, `benchmarks/vn2/run_benchmark.py`, and related tuning / execution paths at post-PR35 HEAD.

**Status:** Open — two blocking code-level issues, four structural issues, and five deployment gaps. Ruff clean; targeted tests pass.

---

## Score by Ask

| Ask | Coverage |
| --- | --- |
| Scale seams (many SKUs, local + global ensembles, HPO in the loop) | ~70% |
| Train + inference endpoints | ~60% |
| Conformal state + online recalibration | ~75% |
| HPO against cost (predict-then-optimize) | ~80% |

The PR materially advances every axis. The remaining gaps are concentrated on the **data-plane edge** (ingestion, persistent state, eager training) and on **two blocking code-level bugs** in the API observe path and session state.

---

## 1. API Lifecycle — Process-Local Session State

**File:** `calibre/api/main.py`
**Severity:** Blocking for multi-worker / restart-safe deployment.

`LifecycleStore` is a global in-memory dictionary. `/fit`, `/predict`, `/calibrate`, `/observe`, `/tune`, and session resume all depend on it. Only `RunStore` and `ConformalStateStore` are SQL-backed; `FitRecord` and `TuneRecord` evaporate on restart.

- A second worker process has no visibility into sessions created by the first.
- A restart drops all in-flight fits, pending observations, and tune studies.
- Gunicorn/uvicorn with `workers > 1` will corrupt or lose state. Kubernetes pod restart = data loss.

**Fix direction:** Mirror the `ConformalStateStore` pattern — give `LifecycleStore` a SQL-backed implementation, or push session metadata into the existing SQL layer and keep only ephemeral coordination in memory.

---

## 2. Train + Inference Endpoints — `/fit` Is Structurally Wrong

**File:** `calibre/api/main.py`
**Severity:** Deployment gap.

The lifecycle split (`/fit`, `/predict`, `/calibrate`, `/order`, `/observe`, `/tune`, `/sessions/{tenant}/{uid}`) is cleanly carved, and `derive_session_id` gives real content-addressed identity. The problem is what `/fit` actually does.

**Gap:** `_run_fit_job` just flips status flags — **no actual model fitting happens at `/fit` time.** The fit runs lazily inside `/predict` (`_fit_predict_task`). For client deployment you want eager training so prediction latency is bounded and model artifacts are cacheable. `ModelArtifactCache` exists but isn't wired into the lifecycle.

**Worse:** `/fit` returns `SUCCEEDED` without validating that the model config is compatible with the data (frequency, regressors, horizon). You can POST garbage, get 202 → `SUCCEEDED`, and only discover the failure at `/predict` time. The endpoint promises readiness it cannot deliver.

**Fix direction:** Make `/fit` actually fit and persist artifacts. Include a config-validation pass so the endpoint only returns `SUCCEEDED` when the model is proven loadable.

---

## 3. Conformal + Online Recalibration — Two Bugs

### 3a. `/observe` Cumulative Conformal Mode Broken

**File:** `calibre/api/main.py`
**Severity:** Blocking for cumulative (CRC / ACIC) online recalibration.

The `/observe` endpoint filters on interval bounds before calling `runtime.observe()`:

```python
if row["lower"] is not None and row["upper"] is not None:
    runtime.observe(...)
```

In cumulative conformal mode, non-terminal rows intentionally carry `NaN` bounds because the cumulative interval is only valid at the final horizon. Filtering those rows out prevents the runtime from ever seeing the intermediate actuals needed to update the cumulative residual pool.

**Fix direction:** Reuse or centralize the dispatcher logic from `calibre/execution/decision_loop.py`, which already knows how to route observations to the correct runtime mode without pre-filtering bounds.

### 3b. Silent Return When `last_calibrated` Is Empty

**File:** `calibre/api/main.py`
**Severity:** Operational hazard.

`_run_observe_job` silently returns when `last_calibrated` is empty. No log, no metric, no exception. In production this means the online recalibration pipeline can be dead for weeks without alarming. This is a coverage-drift failure mode that won't self-report.

**Fix direction:** Add a loud failure path — metric, log, or exception, not a silent return.

---

## 4. Predict-Then-Optimize HPO — Abstraction Is Right, Rough Edges Remain

- `TuningCandidate` unifies `model_config` + `conformal_config` + `ordering_config` — a real joint search space.
- `Cost`, `Pareto`, and the new `Regret` objective all evaluate the *post-order* cost.
- `Cost` validates `CONFORMAL_MODE` agreement with the frame, catching a real footgun where per-horizon `Cost` would silently double-count cumulative bounds.

**Gap:** `Regret` needs `oracle_cost` precomputed once before the study; the PR adds `compute_regret` but no `/tune` plumbing to populate the oracle. The path is sketched, not paved.

---

## 5. VN2 Benchmark — Monolith, Drift, and Silent Failures

### 5a. Scale Seams — Solid Plumbing, Thin Data Ingress

- `_process_global_panel` fits one global adapter across the multi-SKU panel; the PR cites a 1.25× speedup on the VN2 two-config global-LightGBM run.
- Ray Tune fan-out via `_run_optuna_study` with `_OptunaSearchSpaceAdapter`, ASHA, and `tune.with_parameters(state_ref=…)` for shared conformal state across trials.
- `max_uid_concurrency` knob; per-trial CPU budget capping.

**Gap:** Sales and inventory still arrive as JSON in request bodies. For real volume you need a parquet / SQL ingestion seam. The `DatasetAdapter` / `InventoryAdapter` protocols are the right shape but only `SyntheticInventoryAdapter` is implemented.

### 5b. Monolith Past 1k-Line Bar

**File:** `benchmarks/vn2/run_benchmark.py` (~2,300 lines)
**Severity:** Structural / maintainability.

The file owns data prep, HPO, Ray Tune, replay, CRC tuning, diagnostics, logging, and orchestration. Hard to review, test in isolation, or keep in sync with library changes.

**Fix direction:** Split into coordinated modules: `data.py`, `tuning.py`, `replay.py`, `diagnostics.py`, with `run_benchmark.py` as a thin orchestration shell.

### 5c. Tuning Behavior Duplicates Product Runtime

VN2 tuning reimplements canonical tuning behavior from `calibre/tuning/optimizer.py`. Any change to search space shaping, ASHA pruning, or cost evaluation must be manually ported or the benchmark stops measuring the actual runtime.

**Fix direction:** Refactor the benchmark to call `calibre.tuning.optimizer` directly, injecting VN2-specific adapters where needed. The benchmark should measure the product, not shadow it.

### 5d. Cost Search Depends on Ray / Optuna Private State

Cost search reaches into `_ot_study` (Optuna private attribute) and converts broad failures (infra misconfig, import errors, Ray worker crashes) into `inf` cost. A broken environment looks like a "bad trial" rather than a hard failure. Upgrades to Ray or Optuna can break the benchmark without a type error.

**Fix direction:** Distinguish *search failure* (bad hyperparameters → high cost) from *infra failure* (exception → raise / log / metric). Do not swallow exceptions into `inf` indiscriminately.

### 5e. Order-Policy Errors Can Fall Back to Zero Orders

When an order policy errors, the benchmark can fall back to ordering zero units. During cost tuning this is dangerous:
- Zero orders produces a deterministic, low-cost outcome the optimizer can exploit.
- A broken policy looks like a "cheap but valid" configuration.
- The HPO can converge on the bug instead of the real optimum.

**Fix direction:** Fail the trial on policy error. Zero-order fallback should only exist in a dedicated "degraded mode" test, not in the cost search path.

---

## 6. Tables vs. What Landed

| Asked for | PR has | Status |
| --- | --- | --- |
| Sales | request JSON → `history` | Wire-only, no persisted table |
| Order | `OrderLedger` (memory), `last_orders` on `FitRecord` | No persistent order table |
| Inventory | `InventoryAdapter` protocol, `SyntheticInventoryAdapter` | Protocol-only; no SQL / parquet impl |
| Events calendar | — | Not addressed |
| Promotions | `future_x_override` on `/predict` for what-ifs | Inference-side only, no promo table |

---

## Recommended Fix Order

1. **Fix `/observe` cumulative path first** — centralize dispatch through `decision_loop.py`. This unblocks online recalibration for CRC / ACIC.
2. **Split VN2 benchmark responsibilities** — extract data, tuning, replay, and diagnostics into separate modules before adding more tuning behavior.
3. **Back `LifecycleStore` with SQL** — after the observe fix, so session state survives restarts and multi-worker deployments.
4. **Make `/fit` actually fit** — eager training + config validation + `ModelArtifactCache` wiring.
5. **Clean up cost-search error handling** — separate infra failures from high-cost trials; remove zero-order fallback from the search path.
6. **Add loud failure to `_run_observe_job`** when `last_calibrated` is missing or empty.
7. **Ship a SQL-backed `InventoryAdapter`** and a thin `SalesAdapter` for parquet / SQL ingestion; keep `Synthetic*` for tests.
8. **Deduplicate VN2 tuning** — once split, refactor the benchmark to call the product optimizer.
9. **Wire `Regret` end-to-end** — precompute oracle inside `/tune`, store on `TuneRecord`, surface in `/studies/{id}`.
10. **Decide whether `tenant` needs auth enforcement** (currently an honor-system string in `session_id`).

---

## Verification

- `uv run ruff check .` — passed.
- Targeted tests: VN2 benchmark, lifecycle endpoints, decision loop, pending observations — passed.

These passing tests mean the issues above are structural / design-level, not caught by unit-test coverage. They need code-review or integration-test gates.

---

## Bottom Line

The bones of a deployable demand-planning service are in this PR. What's missing is:
- **Two blocking bugs:** the `/observe` cumulative path and process-local session state.
- **The data-plane edge:** ingestion, persistent state for fits / orders / inventory, eager training, and operational safety on the observability path.
- **Benchmark hygiene:** split the monolith, remove silent failure modes, and stop duplicating product tuning logic.

Small in code, but the difference between *demoable* and *deployable*.
