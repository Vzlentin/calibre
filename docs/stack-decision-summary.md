# Calibre Stack Decision Summary

Decision date: 2026-05-19
Recommendation: Go

## Recommendation

Migrate Fugue execution fan-out to Ray Core and the reusable per-series
`TuningTask` path to Ray Tune with OptunaSearch and ASHA. Keep pandas, Parquet,
fsspec, ForecastTaskRef, FastAPI, SQLAlchemy/Alembic, MLflow, Prometheus, and
driver-owned conformal state. The VN2 benchmark's panel-level HPO and cost search
remain sequential Optuna in this PR and are tracked for a follow-up in
[#33](https://github.com/Vzlentin/calibre/issues/33).

This is worth doing because Calibre's bottleneck is embarrassingly parallel Python work:
per-series fit/predict and repeated HPO trials. Fugue makes that work look like a
DataFrame transform and forces schema strings, base64 config columns, and Dask/Spark
engine objects. Ray schedules the real unit of work directly and gives the tuning layer
the same scheduler.

## Before / After

| Area | Before | After |
|------|--------|-------|
| Per-series execution | Fugue `transform` partitioned by `unique_id` | Ray Core task per uid and origin |
| Small run path | `engine=None` direct bypass | In-process fast path below `ray_threshold=10` |
| Global models | Direct unless a Fugue engine exists | Always in-process on driver |
| Worker hand-off | Dispatch DataFrame, base64-pickled configs, Parquet refs | Pickled `ForecastTaskRef` plus existing Parquet URIs |
| Execution config | `engine: Any`, Dask/Spark Fugue objects | `backend`, `ray_address`, `staging_uri`, `ray_threshold`, `max_concurrency`, `cpu_per_task` |
| Per-series HPO | Sequential `study.optimize` inside `TuningTask` | Ray Tune trials with OptunaSearch |
| VN2 panel HPO/cost search | Sequential `study.optimize` | Deferred to [#33](https://github.com/Vzlentin/calibre/issues/33) |
| Search space API | `Callable[[optuna.Trial], dict]` | Same API, unchanged |
| Pruning | None | ASHA reports after each origin; conservative grace period |
| MLflow | Current benchmark helpers and Optuna callback | Keep benchmark helpers; Tune callbacks are limited to `TuningTask` |
| RunStore | Run status and artifact pointers | Same run status and artifact pointers; HPO metadata is deferred |
| Data frames | pandas | pandas |
| Serialization | Parquet through fsspec | Parquet through fsspec |
| Serving | FastAPI | FastAPI |
| State store | SQLAlchemy/Alembic/Postgres | SQLAlchemy/Alembic/Postgres |
| Metrics | Prometheus app metrics | Prometheus app metrics plus Ray metrics |
| Packaging | Fugue core, Dask/Spark extras | Ray extra for dev/benchmark/full installs; slim stays local-only |

## Effort and Timeline

| Phase | Scope | Exit gate | Estimate |
|-------|-------|-----------|----------|
| 0 | Baseline and acceptance | VN2 winning baseline recorded at `total_cost = 4992.20` | 0.5 day |
| 1 | Ray Core execution | Tests, lint/type checks, no Fugue runtime refs, VN2 cost still `4992.20` | 2 days |
| 2 | Ray Tune HPO | Per-series `TuningTask` tests, pruning after grace period, resource-budget tests, VN2 cost still `4992.20` | 2 to 3 days |
| 3 | Observability and persistence | Existing benchmark logging remains intact; HPO persistence deferred until CLI/API tuning is wired | 1 day |
| 4 | Packaging and deployment | Fresh install, slim/full images build, local Ray smoke, VN2 cost still `4992.20` | 1 day |
| 5 | Cleanup and docs | No stale Fugue/Dask/Spark examples, full suite green, VN2 cost still `4992.20` | 0.5 day |

Total: 6 to 7 working days.

## Go / No-go

Go, with two guardrails.

First, keep `BackendEngine.execute()` as the batch API and add a separate streaming
origin iterator for HPO. Tuning needs intermediate reports; existing callers need the
current completed-result contract.

Second, treat the VN2 cost gate as the migration release gate. Every phase must preserve
`benchmarks/vn2/config/winning.yaml` at `total_cost = 4992.20` rounded to cents. If that
breaks, stop and fix the behavioral regression before continuing.

The main risks are Ray startup cost on small CLI invocations, over-aggressive ASHA
pruning during non-stationary early origins, conditional search-space compatibility,
large-panel memory pressure, and Windows developer experience. The proposed fast path,
1-origin default grace period for short studies, OptunaSearch callable API,
shared-URI task refs, streaming ledger, and Linux-container-first Ray deployment address
those risks directly.
