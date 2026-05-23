# Calibre Improvement Wave 1: Phased Execution Plan

The architectural reasoning, code citations, and root-cause analysis live in
[[2026-05-22-improvement-wave-1|2026-05-22-improvement-wave-1.md]]. This file is the
executable plan: phases, files, tests, and DoD.

## Execution mode

**Resume protocol.** Maintain `PROGRESS.md` at the repo root with the
format below. Append one block per task as it completes. On (re)start the
agent reads the tail and resumes from `next_task`.

```yaml
phase: 1
last_completed_task: "1.a cherry-pick observe silent-return fix"
next_task: "1.b fix observe cumulative dispatch"
last_commit: "<sha>"
notes: "<one-line summary of what changed and what was verified>"
```

**Halt protocol.** If a phase's DoD fails after one good-faith attempt,
the agent writes `HALT.md` with `phase`, `failing_test`, `last_error`,
`hypothesis`, `commit_at_halt` and stops. No silent skips, no
`# type: ignore` shims to keep mypy green.

**Per-phase invariants.** Each phase must leave `uv run pytest`,
`uv run mypy calibre/`, and `uv run ruff check .` green before the
agent advances. Any numeric regression baseline is read from `PROGRESS.md`,
not hard-coded.

**Commit cadence.** Commit after each completed task (the unit that produces
one `PROGRESS.md` block). Use conventional-commit style: `phase-N.x: <subject>`.

**Push cadence — phase by phase.** Commits accumulate locally during a phase
and are **pushed only at the phase boundary**, after the cross-phase
regression gate (`ruff`, `mypy`, `pytest`) is green for that phase. Never
push mid-phase, and never bundle commits from multiple phases into a single
push. Each `git push` corresponds to exactly one completed phase on the
`cardinal-improvements` branch, in order (Phase 1 → 2 → 3 → 4 → 5 → 6).
Before each push, run `git pull --rebase origin cardinal-improvements` to
incorporate any ci-fix commits. If rebase produces conflicts you can't
auto-resolve, write `HALT.md` and stop — do not force-push.

## Conventions

- Every command is invoked through `uv run` (CLAUDE.md).
- Per-phase regression gate is at the bottom of this file. It runs at every
  phase boundary, not between sub-tasks within a phase.

---

## Phase 1 · Backend Stabilization

**Goal:** `/observe` is correct in all conformal modes, the backend uses no
untyped duck-typing for conformal state, and `_cap_threaded_config` has a
single canonical home.

**Files:**
- `calibre/api/main.py` (`_run_observe_job:392–449`, `resolved:439`)
- `calibre/execution/decision_loop.py` (`observe_per_horizon:63–89`, `observe_cumulative:92–115`)
- `calibre/execution/backend.py` (`_cap_threaded_config:139`, `_restore_conformal_state:556–576`, `_persist_conformal_state:577–604`, `getattr lines:563,580`)
- `calibre/tuning/optimizer.py` (`_cap_threaded_config:82`)
- `calibre/execution/threading.py` *(new file)*
- `calibre/conformal/runtime.py` *(add PartitionedConformalRuntime Protocol)*

**Changes:**
- **(a) Cherry-pick `/observe` silent-return fix.** The fix already exists
  on branch `deslop-audit` (commit `96348b9`) — it adds warning logging when
  `_run_observe_job` exits early because `last_calibrated` is empty. Either
  cherry-pick that commit or re-apply: at `main.py:404` replace the bare
  `return` with `logger.warning(...)` + `return`. Confirm the branch has no
  other changes that need isolating.

- **(b) Fix `/observe` cumulative dispatch.** At `main.py:439`, `resolved =
  merged.dropna(subset=[Y, lower_col, upper_col])` silently drops non-NaN
  intermediate rows, which are the rows cumulative mode needs to see. Fix:
  detect which conformal mode is active and route accordingly.
  Mechanism: after building `merged`, check
  `getattr(runtime, "mode", "perhorizon")`. If `mode == "cumulative"`, call
  `observe_cumulative(runtime, [merged], actuals_lookup)` from
  `decision_loop.py:92–115` — which handles window completion internally.
  If `mode == "perhorizon"`, keep the current `dropna` path or call
  `observe_per_horizon`. Remove the orphaned `resolved = merged.dropna(...)` +
  `runtime.observe(resolved)` direct call; let the `decision_loop` functions
  own dispatch.

- **(c) Extract `_cap_threaded_config` to `calibre/execution/threading.py`.**
  The function is duplicated verbatim at `backend.py:139` and
  `optimizer.py:82`. Create `calibre/execution/threading.py` with the
  canonical implementation. Update both callers to import from there. No
  behavioural change; this is a mechanical de-duplication.

- **(d) Add `PartitionedConformalRuntime` Protocol and replace `getattr` chains.**
  In `backend.py`, `_restore_conformal_state:563` calls
  `getattr(self.conformal_state_store, "list_for_run", None)` and
  `_persist_conformal_state:580` calls
  `getattr(conformal_runtime, "get_partition_states", None)`. Both are
  statically invisible and will produce a silent no-op if the attribute is
  renamed. Fix:
  1. Define `PartitionedConformalRuntime(Protocol)` in
     `calibre/conformal/runtime.py` with `get_partition_states()`
     and `set_partition_states()`.
  2. Make `SymmetricIntervalRuntime` explicitly implement it.
  3. In `backend.py`, replace both `getattr` blocks with
     `isinstance(runtime, PartitionedConformalRuntime)` guards and direct
     attribute access. Same for the `ConformalStateStore` — add
     `list_for_run` to the Protocol in `calibre/storage/state.py` (or a
     sub-protocol) and replace the `getattr` call.

**Tests:**
- `tests/api/test_observe.py` —
  `test_observe_cumulative_does_not_drop_intermediate_rows`,
  `test_observe_perhorizon_drops_unresolved_rows`
- `tests/execution/test_threading.py` —
  `test_cap_threaded_config_single_source_of_truth`
- `tests/conformal/test_partitioned_protocol.py` —
  `test_symmetric_interval_runtime_implements_protocol`

**DoD:**
- `uv run pytest tests/api/test_observe.py tests/execution/test_threading.py tests/conformal/` green.
- `grep -rn "getattr.*partition\|getattr.*list_for_run" calibre/` returns no results.
- `grep -rn "_cap_threaded_config" calibre/execution/backend.py calibre/tuning/optimizer.py` shows only import, not definition.

---

## Phase 2 · API Lifecycle Correctness

**Goal:** `LifecycleStore` survives restarts and multi-worker deployments;
`/fit` only returns `SUCCEEDED` after real model fitting.

**Files:**
- `calibre/api/lifecycle.py` (`LifecycleStore:42–107`, `FitRecord:12–30`, `TuneRecord:31–39`)
- `calibre/api/main.py` (`_LIFECYCLE_STORE:71`, `_run_fit_job:266–279`)
- `calibre/storage/state.py` (`SqlConformalStateStore:23` — mirror this pattern)
- `calibre/execution/backend.py` (`ModelArtifactCache` references)

**Changes:**
- **(a) SQL-back `LifecycleStore`.** The current implementation at
  `lifecycle.py:42–107` is three in-memory dicts (`_fits`, `_studies`,
  `_conformal_state`). `SqlConformalStateStore` at `storage/state.py:23`
  shows the pattern: Protocol + SQLAlchemy-backed implementation. Apply the
  same approach:
  1. Extract the `LifecycleStore` interface into a `LifecycleStore(Protocol)`
     in a new `calibre/api/store.py` (or keep in `lifecycle.py` as a Protocol
     + concrete class).
  2. Add `SqlLifecycleStore` backed by the project's Postgres/SQLite engine,
     with tables for `fit_records` and `tune_records`. Reuse
     `calibre/storage/postgres.py`'s `make_engine` / `make_session_factory`.
  3. Remove the three in-memory dicts. Keep `LifecycleStore.new_fit_id()` /
     `new_study_id()` as static helpers.
  4. In `main.py:71`, construct `SqlLifecycleStore` (or keep the in-memory
     version selectable via env var `LIFECYCLE_STORE=memory` for tests).

- **(b) Make `/fit` actually fit.** `_run_fit_job:266–279` today only flips
  status flags. Fix:
  1. Parse the `FitRecord`'s `model_config` and call the engine's fit routine
     (reuse the logic currently inside `_fit_predict_task` — extract the fit
     step from it).
  2. Validate config compatibility (frequency, regressors, horizon) before
     marking `SUCCEEDED`; on incompatibility, set `FAILED` with a descriptive
     error.
  3. Persist the fitted artifact via `ModelArtifactCache`. Store the artifact
     URL(s) on `FitRecord.artifact_urls`.
  4. Update `_fit_predict_task` to load the cached artifact instead of
     re-fitting, if present.

**Tests:**
- `tests/api/test_fit_lifecycle.py` —
  `test_fit_actually_trains_model`,
  `test_fit_fails_on_invalid_config`,
  `test_fit_artifact_stored_and_loadable`
- `tests/api/test_lifecycle_store.py` —
  `test_sql_lifecycle_store_survives_reconnect`,
  `test_sql_lifecycle_store_fit_round_trip`

**DoD:**
- `uv run pytest tests/api/` green.
- Posting garbage config to `/fit` returns `FAILED` with a descriptive error
  before `/predict` is called.
- `LIFECYCLE_STORE=sql uv run pytest tests/api/` green (SQL backend selected).

---

## Phase 3 · Config Schema Validation

**Goal:** `calibre/cli/config.py` has no `Any` in parsing helpers and no
`# type: ignore[arg-type]`; malformed configs fail at parse time, not at
backtest time.

**Files:**
- `calibre/cli/config.py` (`type: ignore[arg-type]:182,186,258`, manual key whitelisting throughout)

**Changes:**
- **(a) Replace manual YAML parsing with pydantic models.** The current file
  uses `_require_key`, `_optional_key`, and manual `str(...)` casts with
  `# type: ignore[arg-type]` to satisfy `Literal` fields. Replace with
  `pydantic.BaseModel` (or `dataclasses` + `cattrs` if pydantic is not already
  in the dependency set — prefer pydantic if it is):
  1. Define one `BaseModel` (or `@dataclass`) per top-level config section
     (conformal config, execution config, output config, etc.).
  2. Use `pydantic.Literal` validators for fields like `method`, `mode`,
     `backend` that currently require `# type: ignore[arg-type]` at
     `config.py:182,186,258`.
  3. Delete the `_require_key` / `_optional_key` helpers and the manual
     `ValueError` raises they produce.
  4. Target: eliminate the three `# type: ignore[arg-type]` at lines 182, 186,
     258 and the ~15 `Any` annotations in parsing helpers. Expected LOC
     reduction: ≥ 80 lines.

**Tests:**
- `tests/cli/test_config.py` —
  `test_invalid_method_raises_at_parse_time`,
  `test_invalid_mode_raises_at_parse_time`,
  `test_unknown_key_raises_at_parse_time`

**DoD:**
- `uv run pytest tests/cli/` green.
- `uv run mypy calibre/cli/` shows no `Any` in config parsing helpers and
  zero `# type: ignore[arg-type]`.
- `uv run ruff check calibre/cli/config.py` clean.

---

## Phase 4 · Benchmark Refactor

**Goal:** `benchmarks/vn2/run_benchmark.py` is a thin orchestration shell
(≤ 800 LOC); benchmark calls `calibre.tuning.optimizer` directly; infra
failures are not swallowed into `float("inf")`; zero-order fallback is
removed from the HPO cost path.

**Files:**
- `benchmarks/vn2/run_benchmark.py` (~1,948 LOC — zero-order fallback at `:1837–1862`, cost-search exception at `:1169`)
- `benchmarks/vn2/data.py` *(new)*
- `benchmarks/vn2/tuning.py` *(new)*
- `benchmarks/vn2/replay.py` *(new)*
- `benchmarks/vn2/diagnostics.py` *(new)*
- `calibre/tuning/optimizer.py` (`_evaluate_candidate:294`, `_trainable:416–473`)

**Changes:**
- **(a) Split `run_benchmark.py` into coordinated modules.** Extract the
  five documented internal concerns into separate files:
  - `data.py` — data loading, windowing, forecast cache helpers.
  - `tuning.py` — HPO setup, Optuna study creation, search-space shaping.
  - `replay.py` — `replay_cached_cost`, the decision loop wrapper, order
    policy application.
  - `diagnostics.py` — cost attribution, logging helpers, summary tables.
  - `run_benchmark.py` becomes a ≤ 800 LOC orchestration shell that imports
    from the above.
  Each module gets its own docstring summarising responsibility. The five
  "documented gaps" in the current module docstring must be resolved or
  tracked as issues — not silently carried over.

- **(b) Deduplicate VN2 tuning against `calibre.tuning.optimizer`.** The
  benchmark re-implements search space shaping and ASHA pruning that
  already exist in `optimizer.py`. After splitting (task a), refactor
  `tuning.py` to call `calibre.tuning.optimizer` directly, injecting
  VN2-specific cost adapters. The benchmark should measure the product,
  not shadow it.
  Also remove the `_cap_threaded_config` duplication (already addressed in
  Phase 1.c).

- **(c) Fix cost-search error handling and zero-order fallback.**
  Two distinct fixes:
  1. At `run_benchmark.py:1169` (now in `tuning.py` after split), the
     `except Exception` block sets `trial.set_user_attr("error", repr(exc))`
     and re-raises — this is correct for *search failures* but currently
     mixes infra failures (Ray worker crash, import error) with bad-trial
     errors. Add a discriminator: catch `optuna.TrialPruned` first (already
     done), then distinguish `ValueError`/`KeyError` (bad trial → high cost)
     from other `Exception` (infra failure → raise, log at ERROR, emit
     metric). Do not convert infra failures to `inf`.
  2. At `run_benchmark.py:1837–1862` (now in `replay.py`), the `_policy`
     function falls back to zero orders on `ValueError`/`KeyError`. In the
     HPO cost-search path this is dangerous: a broken policy looks like a
     cheap trial. Fix: in the HPO path, fail the trial on policy error
     (raise, do not return zero orders). Zero-order fallback is acceptable
     only in the dedicated degraded-mode replay path, not in cost search.

**Tests:**
- `tests/benchmarks/test_data.py` — `test_data_loading_round_trip`
- `tests/benchmarks/test_replay.py` — `test_policy_error_fails_trial_in_hpo_path`
- `tests/benchmarks/test_tuning.py` — `test_infra_exception_not_swallowed_as_inf`

**DoD:**
- `uv run pytest tests/benchmarks/` green.
- `wc -l benchmarks/vn2/run_benchmark.py` ≤ 800.
- `grep -n "except Exception" benchmarks/vn2/tuning.py` shows no broad
  swallow to `inf`; infra exceptions re-raise.
- `grep -n "zero orders" benchmarks/vn2/replay.py` appears only in
  degraded-mode replay, not in the HPO objective.

---

## Phase 5 · Data Plane & Regret

**Goal:** A SQL-backed `InventoryAdapter` and thin `SalesAdapter` exist for
parquet/SQL ingestion; `Regret` is wired end-to-end through `/tune`.

**Files:**
- `calibre/execution/dataset.py` (`DatasetAdapter:23`, `InventoryAdapter:29`, `SyntheticInventoryAdapter:35`)
- `calibre/api/main.py` (`/tune` handler at `:465–582`)
- `calibre/api/lifecycle.py` (`TuneRecord:31–39` — add `oracle_cost` field)
- `calibre/tuning/objectives.py` (`Regret:141–166`)
- `calibre/storage/` *(add `SqlInventoryAdapter`)*

**Changes:**
- **(a) Ship `SqlInventoryAdapter` and `SqlSalesAdapter`.** The
  `InventoryAdapter` and `DatasetAdapter` protocols at `dataset.py:29,23`
  are the right shape; only `SyntheticInventoryAdapter` and
  `SnapshotInventoryAdapter` are implemented. Add:
  1. `SqlInventoryAdapter` — reads inventory state from a SQL table (reuse
     the existing Postgres engine). Implements `InventoryAdapter` Protocol.
  2. `SqlSalesAdapter` — reads sales history from a SQL table or parquet
     file via `fsspec`. Keep `Synthetic*` for tests and local development.
  3. Add a persistent `Order` table to track placed orders across sessions.
  Note: hold the `tenant` auth question for Phase 6 — add a `tenant`
  column to the new tables but do not enforce auth yet.

- **(b) Wire `Regret` end-to-end through `/tune`.** `Regret` at
  `objectives.py:141` requires `oracle_cost` precomputed once before the
  study. Currently `/tune` does not compute or store the oracle:
  1. Add `oracle_cost: float | None = None` to `TuneRecord` at
     `lifecycle.py:31`.
  2. In the `/tune` handler (`main.py:465`), if the objective is `regret`,
     run a perfect-foresight pass to compute `oracle_cost` before creating
     the Optuna study. Store it on `TuneRecord`.
  3. Expose `oracle_cost` in the `/studies/{id}` response.

**Tests:**
- `tests/execution/test_sql_adapters.py` —
  `test_sql_inventory_adapter_round_trip`,
  `test_sql_sales_adapter_loads_parquet`
- `tests/api/test_tune_regret.py` —
  `test_tune_computes_oracle_before_study`,
  `test_studies_endpoint_exposes_oracle_cost`

**DoD:**
- `uv run pytest tests/execution/test_sql_adapters.py tests/api/test_tune_regret.py` green.
- `GET /studies/{id}` response includes `oracle_cost` when objective is `regret`.

---

## Phase 6 · Type-System Sweep

**Goal:** Net reduction in `# type: ignore` and `Any` counts across the
codebase; no broad `except Exception` without traceback logging; no
`print()` in library-adjacent CLI code.

**Files:**
- `calibre/conformal/adaptive.py` (`type: ignore[assignment]:153,162`)
- `calibre/conformal/numerics.py` (`type: ignore[return-value]:56`)
- `calibre/api/main.py` (`except Exception:278,326,370,580`)
- `calibre/evaluation/point_metrics.py` (`except Exception:323`)
- `calibre/cli/commands.py` (`print():162,216,218,224,245`)
- `calibre/tuning/optimizer.py` (`_trainable:416–473`, nesting depth 6+)
- `calibre/execution/backend.py` (`type: ignore[union-attr]:512`)
- `calibre/evaluation/forecast_metrics.py` (`type: ignore[assignment]:79`)
- `tests/` (mock typing)

**Changes:**
- **(a) Conformal numeric type fixes.** Three suppressions in `adaptive.py`
  and `numerics.py`:
  - `adaptive.py:153,162` — overload `_clip_alpha` return type so
    array-vs-scalar inputs resolve correctly; remove `type: ignore[assignment]`.
  - `numerics.py:56` — change `_validate_quantile_rule` return to
    `Literal["conformal", "higher"]` directly or use `typing.cast`;
    remove `type: ignore[return-value]`.
  - `adaptive.py` `__init__` params `alpha`, `gamma`, `initial_alpha`,
    `initial_radius` — add missing type annotations.

- **(b) Narrow broad exceptions.** At `main.py:278,326,370,580` and
  `point_metrics.py:323`, each `except Exception` block must:
  1. Log the full traceback (`logger.exception(...)` or `logger.error(...,
     exc_info=True)`) before converting to HTTP 500 or job failure.
  2. Where the underlying exception type is known (e.g. `ValueError` from
     metric computation, `FileNotFoundError` from artifact load), narrow
     the catch accordingly.

- **(c) Replace `print()` in CLI library boundary.** At `commands.py:162,216,
  218,224,245`, replace `print(...)` with `logger.info(...)` or structured
  return values. `run_config` is called programmatically from `api/main.py`;
  side effects (stdout writes) in library-adjacent code break when the caller
  does not expect them.

- **(d) Extract `_trainable` from `optimizer.py`.** `_trainable:416–473` has
  nesting depth 6+ and ~30 lines of setup logic duplicated from
  `_evaluate_candidate:294`. Extract `_trainable` to a module-level function
  with explicit parameters. DRY the shared setup into a helper
  `_build_trial_task(config, ...)` called by both.

- **(e) Fix remaining type ignores.**
  - `backend.py:512` — replace `type: ignore[union-attr]` with
    `assert order_ledger is not None`.
  - `forecast_metrics.py:79` — remove `type: ignore[assignment]`; change to
    `name: str | None = getattr(...)`.
  - Tests — type mocks with `create_autospec(RunStore)` instead of
    `type: ignore`.

- **(f) Decide tenant auth.** `tenant` is currently an honor-system string in
  `session_id`. After Phase 5 adds the `tenant` column to SQL tables, decide:
  enforce (add middleware that validates the tenant claim against an allowlist
  or JWT) or document explicitly as out-of-scope for this wave. Record the
  decision in `PROGRESS.md`; do not leave it undecided.

**Tests:**
- `uv run mypy calibre/` — net reduction in `# type: ignore` and `Any` count
  vs the baseline recorded in `PROGRESS.md` at Phase 5 boundary.
- `uv run pytest tests/` — no regressions.

**DoD:**
- `uv run mypy calibre/` type-ignore count ≤ baseline − 10.
- `grep -rn "print(" calibre/cli/commands.py` returns no results.
- `grep -rn "except Exception" calibre/` — every match has `logger.exception`
  or `exc_info=True` on the next line.

---

## Cross-phase regression gate

Run at every phase boundary:

```bash
uv run ruff check .
uv run mypy calibre/
uv run pytest
```

The `# type: ignore` and `Any` count from `uv run mypy calibre/` is
recorded in `PROGRESS.md` at Phase 1 boundary and used as the baseline for
Phase 6. Phases that intentionally move the baseline must update the recorded
value as part of their DoD.
