# Code Review: PR #38 — Cardinal Improvements: Wave 1 Phased Execution

**Reviewer:** Claude (automated)
**Date:** 2026-05-23
**PR:** #38 `cardinal-improvements` → `main`
**Head:** `93e7ab8`
**CI:** All 4 checks green (test, lint-and-type-check, docker-build, s3-ingestion)

---

## Overview

This PR implements a 6-phase "improvement wave" across 75 files (+7,166 / -2,728) in 53 commits. It is a major architectural refactor of the Calibre demand planning engine, touching the API layer, storage, conformal runtime, forecasting adapters, CLI, tuning, and the VN2 benchmark suite.

**Key changes:**
- **API service extraction** — Fit, predict, observe, order, and tune logic moved from `main.py` into dedicated service modules (`observe_service.py`, `order_service.py`, `tune_service.py`, `model_lifecycle.py`)
- **SQL lifecycle persistence** — `SqlLifecycleStore` in `lifecycle_repo.py` with separate `fit_frame_artifacts` and `lifecycle_conformal_state` tables; data frames stored by reference, not inline
- **Adapter persistence opt-in** — New `CacheableAdapter` mixin; only adapters that inherit it get pickle-based `dump_state`/`load_state`
- **Conformal runtime typing** — `PartitionedConformalRuntime` protocol, typed alpha helpers, removal of `type: ignore` comments
- **VN2 benchmark split** — Monolithic `run_benchmark.py` broken into `data.py`, `replay.py`, `diagnostics.py`, `tuning.py`
- **Regret oracle** — `oracle_cost` precomputed before tuning studies, persisted on `TuneRecord`
- **SQL data adapters** — `SqlInventoryAdapter`, `SqlSalesAdapter`, `OrderRepo` with 4 new Alembic migrations
- **CLI Pydantic validation** — Config parsing rewritten with Pydantic models
- **Ray typing** — `_ray`, remote task handles typed as `ModuleType`/`RemoteFunction` instead of `Any`

---

## Critical Issues

### 1. Missing Alembic migrations — schema/model mismatch (Blocker)

**Migration 0005** creates `fit_records` with inline JSON columns: `history`, `future_x`, `last_forecast`, `last_calibrated`, `last_orders`, `conformal_state`. The ORM model (`LifecycleFitRecord` in `calibre/storage/models.py`) defines `history_ref`, `future_x_ref`, `last_forecast_ref`, `last_calibrated_ref`, `last_orders_ref` (String, not JSON) and has no `conformal_state`. Migration 0008 only drops `conformal_state`. **There is no migration that:**

- Renames/replaces the inline JSON columns with `*_ref` String columns
- Creates the `fit_frame_artifacts` table (defined as `LifecycleFitFrame` in `models.py`)

Any production Postgres deployment running `alembic upgrade head` will have a table schema that does not match the ORM, causing `OperationalError` on every lifecycle query. Tests pass because they use `Base.metadata.create_all()` which creates the schema from the ORM, not from migrations.

### 2. `pickle.loads` on adapter state — arbitrary code execution risk

`CacheableAdapter.load_state()` (`calibre/forecasting/adapter_base.py`) calls `pickle.loads(blob)` on bytes from the artifact cache. If the cache directory is shared, writable by other tenants, or backed by object storage with weak access controls, a crafted pickle blob achieves remote code execution. At minimum, document the trust boundary and consider `hmac` signing or a safer serialization format.

### 3. Global state race condition in `_lifecycle_store()` and `_model_artifact_cache()`

Both functions in `calibre/api/main.py` use a check-then-act pattern on module-level globals without a lock. Under Uvicorn's threaded worker model, two concurrent requests can each see `_LIFECYCLE_STORE is None`, both create a new store, and one is discarded — losing any fits stored in it. This is a data-loss bug for `MemoryLifecycleStore` and a connection-leak risk for `SqlLifecycleStore`.

---

## High-Severity Issues

### 4. `cache_key` includes `forecast_origin` — breaks eager-fit cache hits

`ModelAdapter.cache_key()` (`calibre/forecasting/adapter_base.py`) now includes `forecast_origin` in its hash. For eager-fit models (where fit is origin-independent), every prediction with a different origin produces a different key and misses the cache. The key should only include fit-affecting fields; `forecast_origin` should be excluded or handled separately.

### 5. Non-cacheable adapters silently omitted from `artifacts` dict

In `fit_model_artifacts()` (`calibre/execution/model_lifecycle.py`), non-`CacheableAdapter` adapters are fitted but `continue`d without adding an entry to `artifacts`. Callers that expect `artifacts` to cover all labels (e.g., `predict_from_artifacts` checks `record.artifact_urls.get(label)`) will silently fall through to re-fitting, which may be the intent but should be explicitly documented.

### 6. Private imports across module boundaries

- `_cap_threaded_config` imported from `calibre.execution.threading` (underscore-private) in both `backend.py` and `benchmarks/vn2/tuning.py`
- `_log_mlflow_params` imported as private from `benchmarks/vn2/replay` in `tuning.py`
- `_load_instock`, `_model_uses_cumulative_target`, `_prepare_model_history`, etc. imported as private from `data.py` in `replay.py`

These create fragile cross-module dependencies on internal APIs. Make them public or inline them.

### 7. `os.environ` mutation in `_trial_thread_env` — thread-safety bug

`benchmarks/vn2/tuning.py` mutates `os.environ` inside Ray Tune parallel trial workers. With `max_concurrent_trials > 1` in local mode, concurrent threads race on the same environment dictionary. The context manager save/restore pattern is not thread-safe.

### 8. MLflow history artifact regression

In `benchmarks/vn2/replay.py`, `log_cached_replay_run` writes `history.csv` to a temp directory but the `mlflow.log_artifact()` call that existed in the old `run_benchmark.py` is missing. The CSV is written and immediately discarded when the `TemporaryDirectory` context exits. This silently drops MLflow history artifact logging.

---

## Medium-Severity Issues

### 9. `_resolve_history_source` uses `assert` for type narrowing

`assert source is not None` in `calibre/api/main.py` will be stripped under `python -O`. Use `if source is None: raise ValueError(...)`.

### 10. No Pydantic `model_validator` for mutual exclusivity

`FitRequest` and `TuneRequest` (`calibre/api/schemas.py`) accept both `history` and `history_source` as `None`. The XOR check is in `_resolve_history_source`, not in the schema, so Pydantic validation passes and the error message is less clear.

### 11. `health()` CLI output changed from stdout JSON to `logger.info`

`calibre/cli/commands.py` switches `print(json.dumps(payload))` to `logger.info("health=%s", payload)`. Any script or CI pipeline that parsed the JSON from stdout will break silently.

### 12. `point_metrics.py` uses `logger.exception` for expected failures

`ZeroDivisionError` on UMBRAE or `ValueError` on empty arrays are expected edge-case data conditions in `calibre/evaluation/point_metrics.py`. `logger.exception` (with full traceback) will flood production logs for series with zero demand. `logger.warning` was more appropriate.

### 13. Observe "no rows resolved" check is opaque

In `calibre/api/observe_service.py`, the old code used `resolved.empty` (direct). The new code uses `pending_rows >= len(merged)` which is less clear and may have subtle semantic differences depending on how `observe_cumulative`/`observe_per_horizon` return remaining frames.

### 14. `fits_for_tenant_uid` does full-table scan + Python-side filtering

`SqlLifecycleStore.fits_for_tenant_uid()` in `calibre/storage/lifecycle_repo.py` fetches all fits for a tenant and filters `if uid in row.sku_set` in Python. For tenants with many fits, this is expensive. A JSONB `@>` query would be more efficient.

### 15. `_best_tune_result` filters but calls `get_best_result` on unfiltered results

In `benchmarks/vn2/tuning.py`, the pre-filter of valid results is only used for an error message guard; the actual `get_best_result()` call operates on the unfiltered `results` object and could theoretically pick an invalid result.

### 16. `getattr(runtime, "mode", "perhorizon")` silent fallback

In `calibre/api/observe_service.py`, if `mode` is absent or misspelled on the runtime, the code silently falls back to `"perhorizon"` instead of raising. This leaks conformal runtime strategy into the observe service with a hidden default.

---

## Low-Severity / Style Issues

- `import traceback` inside `except` block in `tune_service.py` — should be top-level
- `TuneRunner` type alias defined but not used in `benchmarks/vn2/tuning.py`
- `SqlLifecycleStore` not in `__all__` of `lifecycle.py` despite being accessible via `__getattr__`
- `LifecycleStore` Protocol has `@staticmethod` bodies — confusing pattern for a Protocol; move to module-level functions
- `_lifecycle_store()` re-reads `database_url()` and `LIFECYCLE_STORE` env var on every call — should resolve once at startup
- Downgrade in migration 0008 is silently destructive (adds back `conformal_state` with empty default, discarding data)
- `_METRIC_FAILURE_EXCEPTIONS` includes both `ArithmeticError` and `ZeroDivisionError` — redundant since `ZeroDivisionError` is a subclass
- `_frame_from_records` duplicated between `observe_service.py` and `main.py`
- `_records_from_frame` in `lifecycle_repo.py` leaves numpy scalars unconverted — may cause JSON serialization failures at DB write time
- `OrderRecord.order_id` uses a Python `lambda: uuid4().hex` default but no `server_default` — raw SQL inserts would get `NULL`

---

## What's Good

- **Service extraction is well-structured** — The separation of concerns between HTTP routes and business logic is a significant improvement. Routes now create records, dispatch to services, and serialize responses.
- **Conformal typing is clean** — `PartitionedConformalRuntime` protocol with factory-only restoration is the right pattern. The `overload`-based `_clip_alpha` and removal of 12+ `type: ignore` comments shows real type-safety investment.
- **VN2 split is substantial** — `run_benchmark.py` from ~2000 LOC to ~400 LOC with proper module separation.
- **`CacheableAdapter` opt-in mixin** is the right direction vs. the old `raise NotImplementedError` default.
- **Comprehensive test coverage** — 13 new test files with focused unit and integration tests.
- **CI is fully green** across all checks.
- **Migration chain is well-ordered** — 0005 → 0006 → 0007 → 0008 with clear `down_revision` links and proper downgrade paths (aside from the schema mismatch noted above).
- **Regret oracle precomputation** is a clean design — computing `oracle_cost` before the study starts and persisting it avoids repeated expensive computation during trials.

---

## Verdict

**Do not merge as-is.** The three blockers/critical items are:

1. **Missing migrations** for `fit_frame_artifacts` table and `*_ref` column renames on `fit_records`
2. **`pickle.loads` on untrusted cache data** without signing or documentation of trust boundary
3. **Thread-unsafe global store initialization** with potential data loss under concurrent requests

The remaining high/medium issues (cache key including `forecast_origin`, MLflow regression, private cross-module imports, `os.environ` thread safety) are worth addressing before merge or tracking as immediate follow-up work.
