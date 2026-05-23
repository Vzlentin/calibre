# PR #38 Review Synthesis

This document synthesizes the recurring PR loop reviews, thermo-nuclear code
quality reviews, and roadmap comparison for PR #38, `cardinal-improvements`.
It keeps the original review context while removing duplicate framing between
agents.

## Review Scope

- PR: #38, `cardinal-improvements`
- Pushed PR head reviewed by the recurring loop:
  `1a09116a7e84f3359f37ae2817b524650029da90`
- Thermo-nuclear PR head reviewed:
  `af093b9020dec66d83a6460d99053a8e12226388`
- Roadmap comparison reviewed PR #38 after docs commit:
  `c6e977a6fcc5638ddb12000bb91d57f21cfd9a8b`
- Later local branch progress extends beyond the pushed PR head through Phase 6
  commits.
- CI was green after retrying a transient `docker-build` GHCR push failure.

## Approval Posture

Do not approve PR #38 as-is. The branch addresses much of the improvement-wave
roadmap, but the remaining issues are architectural, not cosmetic:

1. Lifecycle fit records still store full data planes and mutable conformal
   state as JSON.
2. `calibre/api/main.py` still owns fitting, prediction, artifact caching,
   observe dispatch, order persistence, and regret-oracle orchestration.
3. Adapter artifact persistence is still pickle-by-default instead of explicit
   opt-in.
4. VN2 benchmark modules are improved but still coupled through private helper
   re-exports and monkey-patching.
5. The flagship VN2 benchmark still masks order-policy wiring failures by
   returning zero orders.

## Consolidated Findings

### 1. Lifecycle Persistence Stores Data Planes and Mutable State

- Source commit: `54f9979`
- Files: `calibre/api/lifecycle.py`, `calibre/storage/models.py`
- Status: still present in the local branch
- Severity: blocker

`LifecycleFitRecord` stores full DataFrame-derived payloads on the lifecycle
row: `history`, `future_x`, `last_forecast`, `last_calibrated`, and
`last_orders`. This turns fit lifecycle metadata into an unindexed data
warehouse and makes row size, query behavior, retention, and schema evolution
hard to reason about.

The same lifecycle rows also carry conformal state and update it across all fits
for a session. Reads then recover session-level or partition-level mutable state
from a fit row, duplicating state at the wrong ownership level.

Fix direction: lifecycle rows should contain metadata, status, configuration,
and explicit pointers only. Persist large frames through the forecast ledger,
artifact store, or dedicated repository tables. Store conformal state in a
session-owned or partition-owned table/repository, and have fits reference that
state owner instead of carrying copies.

### 2. Lifecycle Module Mixes API Contracts, Storage, and Serialization

- Source commit: `54f9979`
- File: `calibre/api/lifecycle.py`
- Status: still present in the local branch
- Severity: high

`calibre/api/lifecycle.py` contains the lifecycle protocol, memory store, SQL
store, SQL row mapping, DataFrame serialization, and persistence policy. This
turns an API lifecycle module into a combined interface, storage, and
serialization layer.

Fix direction: keep typed lifecycle records and the store contract near the API
boundary, but move SQL row mapping and database querying into a storage or
repository module.

### 3. API Layer Still Owns Execution and Conformal Runtime Strategy

- Source commits: `2b7f9ac`, `08c6cb4`
- File: `calibre/api/main.py`
- Status: still present in the local branch
- Severity: blocker for fit execution; medium for observe dispatch

`calibre/api/main.py` owns model training orchestration, adapter resolution,
validation, cache creation, artifact recording, prediction cache reuse, order
persistence, and regret-oracle setup. It also imports private backend helpers
such as `_fit_predict_task`, `_fit_adapter_for_task`, `_finalize_preds`, and
`_coerce_forecast_frame_dtypes`, making the HTTP/API module act as an execution
service and bypassing the `BackendEngine` boundary.

The same module also makes conformal observation dispatch decisions directly:
`_run_observe_job` imports and dispatches to `observe_cumulative` or
`observe_per_horizon`, using `getattr(runtime, "mode", "perhorizon")` to choose
behavior. That leaks conformal runtime strategy into the API layer and adds a
silent fallback around a core invariant.

Fix direction: move fit execution, artifact production, order persistence,
regret-oracle orchestration, and observe dispatch behind execution or lifecycle
services with typed public boundaries. Routes should create lifecycle records,
prepare request data, invoke one service boundary, and serialize responses.

### 4. Adapter Artifact Persistence Is Too Broad

- Source commit: `2b7f9ac`
- File: `calibre/forecasting/adapter_base.py`
- Status: still present in the local branch
- Severity: high

The base adapter makes every adapter cacheable by pickling `self.__dict__`.
That is too broad for a persistence boundary and hides adapter-specific
serialization invariants.

Fix direction: make cache persistence explicit and opt-in through adapter
implementations or a dedicated cacheable-adapter protocol/mixin. Do not make all
adapters implicitly persistent.

### 5. Artifact URLs Are Persisted but Not Used as the Loading Contract

- Source commit: `2b7f9ac`
- Files: `calibre/api/main.py`, `calibre/forecasting/cache.py`
- Status: still present in the local branch
- Severity: medium

Fit records persist `artifact_urls`, but prediction mainly treats their presence
as a boolean to enable the global model cache. Actual artifact lookup still
depends on recomputed adapter cache keys, so the lifecycle record is not the
source of truth for loading artifacts.

Fix direction: persist explicit artifact keys or URIs and load the selected
artifact through a cache API keyed by that persisted value.

### 6. Conformal Runtime Restoration Still Mutates Existing Instances

- Source commit: `e02708b`
- Files: `calibre/conformal/runtime.py`, `calibre/execution/backend.py`
- Status: partially improved, still present
- Severity: medium

The branch added `from_partition_states(...)`, which is the right direction, but
`set_partition_states(...)` still restores by creating another runtime and
copying selected mutable fields back onto the existing instance.

Fix direction: make restoration a factory or codec boundary that returns a
complete runtime. Avoid requiring partitioned runtimes to support in-place state
mutation.

### 7. Thread Budget Helper Carries Caller-Specific Aliases

- Source commit: `8c12edb`
- File: `calibre/execution/threading.py`
- Status: still present in the local branch
- Severity: medium

`_cap_threaded_config` accepts `cpu_budget`, `cpu_per_task`, and
`cpu_per_trial`, then chooses among non-null values. That makes a private helper
carry multiple caller vocabularies and hides the actual contract.

Fix direction: keep one canonical `cpu_budget` parameter. Backend, tuning, and
benchmark callers should map their domain-specific names before calling the
helper.

### 8. VN2 Split Improved the Monolith but Still Has Facade Coupling

- Source commit: originally reported as `279bf84`; current branch has the split
  at `d55bd78`
- Files: `benchmarks/vn2/run_benchmark.py`, `benchmarks/vn2/tuning.py`
- Status: improved but not fully resolved
- Severity: high/medium

Phase 4 made real progress: `run_benchmark.py` is now about 412 lines, and VN2
tuning reuses shared optimizer helpers such as `optimize_task`,
`create_tpe_sampler`, and `restore_cwd`.

The structural concern remains. `run_benchmark.py` still imports private helpers
from sibling modules, re-exports tuning hooks, and monkey-patches
`_tuning._run_optuna_tune` in `run_cost_search`. `tuning.py` is still about 789
lines and mixes Ray Tune execution, HPO, simulator-cost search, CRC sampling,
MLflow logging, and search adapters.

Fix direction: expose explicit public APIs from the owning VN2 modules, update
callers and tests to import owners directly, and use dependency injection for
the tune runner instead of runtime monkey-patching. Split shared Ray Tune
execution, HPO, cost search, and search-space/config sampling into focused
modules.

### 9. VN2 Benchmark Masks Policy Failures With Zero Orders

- Source commit: `d55bd78`
- File: `benchmarks/vn2/run_benchmark.py`
- Status: still present in the local branch
- Severity: high

The benchmark `_policy` path catches `ValueError` and `KeyError` from order
policy application and returns zero orders. It also returns zero orders when the
expected quantile column is missing. In the flagship benchmark, this can hide
broken conformal/order wiring and produce misleading cost results.

Fix direction: fail by default when the policy frame is structurally invalid or
order computation raises. Keep zero-order behavior only behind an explicit
diagnostic or degraded-mode flag that is clearly reported in logs and outputs.

Important distinction: the original roadmap only required zero-order fallback to
be removed from the HPO cost-search path. PR #38 does that by passing
`policy_error_mode="raise"` during cost search. This review asks for the
stricter benchmark default: fail fast during normal benchmark replay, and keep
zero-order replay only as an explicit diagnostic mode.

## Roadmap Comparison

PR #38 substantially addresses most items from
`docs/2026-05-22-improvement-wave-1.md`, including:

- Cumulative `/observe` behavior and empty-calibration logging
- SQL-backed lifecycle records
- Real `/fit` execution with model artifacts
- Zero-order fallback removal from the HPO cost-search path
- VN2 runner splitting and tuning reuse of shared optimizer helpers
- Cost-search infrastructure failure surfacing
- Pydantic CLI config validation
- `PartitionedConformalRuntime`
- SQL/parquet storage adapters
- Regret-oracle plumbing
- Conformal numeric typing
- CLI logging
- Optimizer trainable extraction
- Local type-ignore sweep

The roadmap-level gaps still not fully addressed are:

1. **P1.6 Ray integration typing is incomplete.** The plan called for replacing
   Ray integration `Any` usage with typed Ray handles and remote-function types.
   PR #38 reduces nearby type-ignore and `Any` usage, but `BackendEngine` still
   keeps `_ray`, remote task handles, and `_ensure_ray()` typed as `Any`.
2. **P1.8 data-plane ingestion is adapter-level, not API-level.** PR #38 adds
   `SqlInventoryAdapter`, `SqlSalesAdapter`, data-plane tables, parquet loading,
   and order persistence. It does not fully replace JSON request-body ingestion
   in `/fit` and `/tune` with an end-to-end SQL/parquet source contract.
3. **P2.2 broad exception narrowing is only partial.** API broad exception
   handlers now log tracebacks before returning job failure or HTTP 500, but
   `calibre/evaluation/point_metrics.py` still catches `Exception` around every
   metric instead of narrowing to expected metric failures.

Deployment note: Phase 6 documents `tenant` as a storage partitioning key, with
authorization expected at the gateway, mesh, or JWT middleware layer. That
satisfies the roadmap's "decide tenant auth" item, but leaves enforcement
outside Calibre.

## Clean Reviews and Non-Issues

The thermo-nuclear review found no material concerns in the plan/progress-only
commits. Repeated loop reviews of pushed PR head commits also found no concerns
when those commits only appended `PROGRESS.md` status blocks, including:

- `ed803e8` (`phase-4: record boundary gate`)
- `1a09116` (`phase-5: record boundary gate`)

`678e795` and `1921634` were also reviewed without thermo-nuclear concerns.

## CI Note

During the loop, `docker-build` failed once in the GHCR push step with
`denied: denied` after other checks had passed. The failed job was retried once
and passed. This looked like a registry permission or transient publish failure,
not a code-quality finding.
