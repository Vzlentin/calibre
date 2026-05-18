# Calibre Stack Decision — Executive Summary

> **Decision date:** 2026-05-18 · **Full document:** `docs/stack-decision.md`

---

## Recommendation

**Go.** Replace Fugue + sequential Optuna with Ray Core + Ray Tune (OptunaSearch + ASHA). Keep everything else.

---

## Before / After Stack

| Component | Before | After |
|-----------|--------|-------|
| **Per-series fan-out** | `fugue.api.transform(partition_by=UNIQUE_ID, engine=...)` | `@ray.remote _ray_fit_predict_uid` + `ray.get([...])` |
| **Execution engine config** | `ExecutionOptions(engine: Any)` — Dask/Spark object | `ExecutionOptions(ray_address: str \| None)` |
| **Single-node fast path** | `_run_direct` bypass for `engine=None` | `ray_threshold=10`: below threshold runs in-process, no Ray init |
| **Task hand-off** | Base64-pickled config in DataFrame columns + schema strings | `ForecastTaskRef` pickled directly into `@ray.remote` args |
| **Global scope** | `_run_global_distributed` (Fugue or in-process) | Always in-process on driver (one training job, no fan-out) |
| **HPO** | Sequential `study.optimize(n_trials=50)`, no pruning | `tune.Tuner(OptunaSearch, ASHAScheduler)`, parallel trials + per-origin ASHA |
| **Search space API** | `Callable[[optuna.Trial], dict]` | `Callable[[optuna.Trial], dict]` — **unchanged** |
| **Early stopping** | None | ASHA prunes after `grace_period=8` origins |
| **MLflow HPO tracking** | `optuna_integration.mlflow.MLflowCallback` | `ray.air.integrations.mlflow.MLflowLoggerCallback` |
| **Observability** | Fugue worker logs + Optuna local study | Ray Dashboard + MLflow (unchanged) + Prometheus (unchanged) |
| **In-memory frames** | pandas | pandas (unchanged) |
| **Serialization** | Parquet (fsspec) | Parquet (fsspec) — unchanged |
| **State store** | SQLAlchemy + Alembic + Postgres | SQLAlchemy + Alembic + Postgres — unchanged |
| **Serving** | FastAPI | FastAPI — unchanged |
| **Dependencies added** | — | `ray[default,tune]>=2.10,<3` |
| **Dependencies removed** | `fugue`, `dask[distributed]`, `pyspark` | — |

---

## What does not change

- `ForecastTaskRef` URI-based materialization — already Ray-clean
- `pandas` as the in-memory frame format
- `fsspec` for object-store IO
- `FastAPI` + SQLAlchemy + Alembic state store (PR #31)
- `MLflow` experiment tracking (server running on Tailscale)
- `Prometheus` operational metrics
- `ConformalRuntime` on the driver — mutable sequential state, never in a Ray worker
- `TuningTask.search_space: Callable[[optuna.Trial], dict]` — preserved end-to-end via `OptunaSearch`
- VN2 cost gate: `benchmarks/vn2/config/winning.yaml` must reproduce ≤ EUR 4,992.20

---

## Migration Phases

| Phase | Scope | Exit criteria | Effort |
|-------|-------|---------------|--------|
| **A** | Ray Core execution backend | pytest green, no Fugue in codebase, VN2 winning cost ≤ EUR 4,992.20 under Ray | ~2 days |
| **B** | Ray Tune + ASHA tuning | `test_ray_tune.py` green, ≥1 trial pruned, HPO cost within ±5% of baseline | ~2 days |
| **C** | Vault sync | architecture.md updated, lessons.md appended, plan status → accepted | <1 day |

Total: **~5 working days.**

---

## Why not keep Fugue?

Fugue imposes three artefacts with no benefit for Calibre:

1. **Schema strings.** `_collect_quantile_columns` decodes every model config before every origin to produce a schema string Fugue needs but Calibre does not.
2. **Base64-pickled configs in DataFrame columns.** A Fugue protocol workaround. Deleted entirely with Ray.
3. **Dispatch DataFrames.** `_TaskDispatchRecord` and `_dispatch_records_to_frame` exist only to satisfy Fugue's partition-by-column contract. Ray accepts Python objects directly.

Fugue's Dask/Spark portability is not a constraint (stated in PLAN.md: "no deprecation cycles required — nothing is in production").

---

## Why not keep sequential Optuna?

VN2 at full scale: 600 series × 50 trials × 40 origins = 1.2 M fit-predicts, all sequential. There is no early stopping — the worst trial runs all 40 origins. ASHA with `grace_period=8` prunes provably bad trials after 8 origins, cutting wall time by an estimated 3–5× on a typical search.

---

## Top risks

| Risk | Mitigation |
|------|-----------|
| Ray `init` cost per CLI invocation | `ray_threshold=10` fast path skips Ray for small runs |
| ASHA pruning good trials during conformal warmup | `grace_period=8` = `WARMUP_ORIGINS` for VN2; no pruning before calibration stabilizes |
| Conditional search spaces break under `OptunaSearch` | Integration test covers `_sample_cost_search_crc_config` conditional branches |
| Memory pressure at 10k+ series | `LedgerOutputOptions(streaming=True)` bounds driver memory |
| `@ray.remote` not picklable | Static test asserts module-level functions pickle without `ConformalRuntime` refs |
