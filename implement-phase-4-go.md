# Implement Phase 4 — Orchestration + Remaining Gaps

## Context

Phases 1–3 of the Calibre gap-analysis roadmap (vault plan
`2026-04-14-calibre-gap-analysis.md`) are merged on `main`: global model
support, features + quantile/direct multi-horizon, generic inventory
simulator, and cumulative conformal mode. Phase 4 — the last tranche in
that roadmap — is "not started" per the plan's Open section.

Phase 4 exists because after Phase 3 the VN2 drivers still carry
~500 lines each of boilerplate: rolling decision loop, pending-forecast
bookkeeping, conformal observe lifecycle, manual per-series config
overrides, median-only ensembling. The roadmap's success criterion is
that both VN2 drivers can be expressed in < 100 lines of
benchmark-specific code once Phase 4 lands. Phase 4 also closes three
stray gaps: dead `future_x` plumbing (Gap 7), absent per-series config
overrides (Gap 10), and median-only ensembling (Gap 8).

The user wants Phase 4 planned now and shipped via `/go` once the plan
is approved.

## Scope (user-locked)

**This PR: Workstream 1 only — the decision-loop orchestrator.**
Workstreams 2 (future_x), 3 (per-series overrides), and 4 (weighted
ensembles) are deferred to small follow-up PRs. Rationale: smallest
diff, biggest line-count win, lowest risk. Those items are documented
below for continuity but are not implemented in this PR.

Success bar: `benchmarks/vn2/run_benchmark.py` and `run_seasonal.py`
each drop to < 150 lines of VN2-specific wiring by delegating the
rolling loop + `observe()` dispatch to Calibre. End-to-end cost must
match current main within ±5% on the fast-iteration config
(SeasonalNaive + 6 warmup origins; Phase 3 baseline EUR 9,576).

## Non-goals

- No new conformal methods, no new order policies, no new model
  adapters.
- No VN1 scaffolding (tracked separately in the mlflow plan's follow-ups).
- No API changes to Phase 1–3 modules beyond additive parameters.

---

## Workstream 1 — `calibre/orchestration/decision_loop.py`

### Why

Both VN2 drivers reimplement the same skeleton:

```
for round_num in range(n_rounds):
    origin = advance(origin)
    tasks = build_tasks(history, origin, ...)
    ledger = engine.execute(tasks)
    conformal_frame = runtime.apply(ledger)
    orders = apply_rs_policy(conformal_frame, ...)
    sim.step(orders, actuals)
    runtime.observe(resolved_rows)   # per-horizon OR cumulative dispatch
```

The dispatch between per-horizon and cumulative `observe()` is exactly
the trap documented in `lessons.md:40` ("Per-horizon CP `observe()`
requires interval columns; cumulative does not"). Putting the loop in
one place prevents every future driver from re-hitting that landmine.

### Design

New package `calibre/orchestration/` with:

- `decision_loop.py`
  - `@dataclass RoundResult`: `round_num: int`, `origin: Timestamp`,
    `ledger: DataFrame`, `conformal_frame: DataFrame`, `orders: DataFrame`,
    `sim_records: DataFrame` (from `Simulator.to_dataframe()` for this
    round only).
  - `@dataclass DecisionLoopConfig`:
    - `n_rounds: int`
    - `warmup_origins: int` (pre-loop calibration rounds, no sim step)
    - `freq: str` (panel frequency, forwarded to the engine)
    - `policy: Callable[[DataFrame], DataFrame]` — wraps
      `apply_rs_policy(..., **policy_kwargs)` so cumulative vs
      quantile vs per-horizon paths are caller-owned.
    - `on_round: Callable[[RoundResult], None] | None` — hook for
      per-round logging (used by the MLflow tracking already in
      `benchmarks/common/tracking.py`).
  - `class DecisionLoop`:
    - `__init__(self, engine, runtime, simulator, build_tasks_fn,
      advance_origin_fn, observe_fn, config)`.
    - `run(history_0, actuals_stream) -> list[RoundResult]`.
    - Internals: manages `origin` advancement, calls `build_tasks_fn`,
      `engine.execute`, `runtime.apply`, `config.policy`,
      `simulator.step`, `observe_fn`. `observe_fn` is injected (not
      hard-coded) so the caller selects the per-horizon vs cumulative
      resolution strategy — but we also ship two ready-made helpers:
      - `observe_per_horizon(runtime, ledger_rows) -> None`
      - `observe_cumulative(runtime, ledger_rows, protection_period)`
      Both live next to `DecisionLoop` and encode exactly the rules
      from `lessons.md:40` so drivers don't re-invent them.

- `__init__.py` re-exports `DecisionLoop`, `DecisionLoopConfig`,
  `RoundResult`, `observe_per_horizon`, `observe_cumulative`.

### Reuse (do NOT re-implement)

- `BackendEngine.execute` (`calibre/engine/backend.py`)
- `ConformalRuntime.apply` / `.observe`
  (`calibre/conformal/runtime.py`) — dispatch logic already exists; the
  orchestrator just feeds it the right inputs.
- `apply_rs_policy` (`calibre/order/rs.py`) — already supports all
  three target-stock paths (explicit quantile, cumulative marker, per-
  horizon).
- `Simulator` (`calibre/simulation/simulator.py`) — already has
  `step`, `to_dataframe`, cumulative cost accounting.
- `build_tasks` (`calibre/pipeline/tasks.py`) — extended in
  Workstream 3; otherwise called unchanged.

### Tests

`tests/test_decision_loop.py` (new):
1. Fake engine + fake runtime + fake simulator (pure python, no
   Fugue/Optuna) — smoke test the round skeleton advances origin, calls
   each collaborator once per round, returns `len(results) == n_rounds`.
2. Per-horizon `observe_per_horizon`: partial-window rows are filtered
   (matches `lessons.md:40` rule).
3. Cumulative `observe_cumulative`: incomplete windows stay queued,
   complete ones submit one cumulative score.
4. `on_round` hook fires exactly once per round with a populated
   `RoundResult`.

### Migration

`benchmarks/vn2/run_benchmark.py` and `run_seasonal.py` switch to
`DecisionLoop`. `run_winning.py` stays as-is for now (no conformal, no
incremental need). The MLflow per-round `cost/*` logging wires into
`on_round`.

---

## Deferred — not implemented in this PR

The three sections below document Workstreams 2–4 for vault continuity.
They are **not** part of this PR; each ships in its own follow-up PR
after Workstream 1 lands.

## Workstream 2 (DEFERRED) — `future_x` wiring (Gap 7)

### Why

`ForecastTask.future_x` is a field but `MLForecastAdapter`,
`StatsForecastAdapter`, and `NeuralForecastAdapter` all ignore it. The
engine passes it in at `calibre/engine/backend.py:~147` but nothing
reads it. Dead code today; silently wrong the day someone sets it.

### Design

- `MLForecastAdapter.predict`: forward `future_x` to
  `MLForecast.predict(X_df=...)` when non-empty. mlforecast's public
  API already takes `X_df` for exogenous future regressors; this is a
  direct pass-through.
- `StatsForecastAdapter.predict`: pass `X_df=future_x` to
  `StatsForecast.predict` when non-empty (statsforecast accepts
  `X_df`).
- `NeuralForecastAdapter.predict`: pass `futr_df=future_x` (neuralforecast
  uses the `futr_df` kwarg).
- All three: drop `future_x` silently when empty/None (current
  behavior). Dtype/column alignment is caller's contract — the frame
  schema already matches `history` regressors.

### Reuse

- Forecast-frame column conventions
  (`calibre/contracts/forecast_frame.py`) — no change, regressors flow
  through the same schema.
- No change to `ForecastTask` or `GlobalForecastTask`.

### Tests

Extend `tests/test_mlforecast_adapter.py`,
`tests/test_statsforecast_adapter.py` (and neural if it exists):
1. Build a task with a single regressor column in both `history` and
   `future_x`, fit, predict — assert the regressor influences
   predictions vs a control task with a constant regressor.
2. Empty `future_x` path still works (regression test for current
   behavior).

---

## Workstream 3 (DEFERRED) — Per-series config overrides in `build_tasks()` (Gap 10)

### Why

`build_tasks()` currently applies the same model list to every series.
`benchmarks/vn2/run_benchmark.py` manually constructs per-series
`ForecastTask` objects for tuned series (~50 lines of bespoke loop).

### Design

Extend `build_tasks(history, model_configs, horizon, ...)` in
`calibre/pipeline/tasks.py`:

- New optional kwarg
  `overrides: Mapping[str, list[dict]] | None = None`.
- When `overrides[unique_id]` is present, use that list of model
  configs instead of the default `model_configs` for that series.
- Unknown `unique_id`s in `overrides` raise `ValueError` (typo catch).
- Default `None` preserves current behavior exactly.

### Reuse

- `ForecastTask` constructor (`calibre/tasks/forecast_task.py`) —
  unchanged.
- The existing `model_configs` iteration already expands one
  `ForecastTask` per config; the override path just swaps the list.

### Tests

Extend `tests/test_pipeline_tasks.py`:
1. Override for a single series swaps its model list.
2. Series without an override key keep the default list.
3. Unknown `unique_id` in `overrides` raises `ValueError`.

---

## Workstream 4 (DEFERRED) — Ensemble extensions (Gap 8)

### Why

`calibre/ensemble/median.py` is the only ensemble. No weighted
average, no inverse-error weighting, no selection. Trivial to add, but
also the lowest-value Phase 4 item — ship only if time permits.

### Design

New file `calibre/ensemble/weighted.py`:
- `ensemble_weighted(frames: list[DataFrame], weights: list[float])`
  — validates `len(frames) == len(weights)` and `weights` sums to 1,
  linearly combines `y_hat` (and any `q_*` columns via
  `is_quantile_column` from `forecast_frame.py` — honor the lesson at
  `lessons.md:25` and don't re-implement the startswith check).
- `ensemble_inverse_error(frames, errors)` — compute
  `weights = (1/errors) / sum(1/errors)`, then delegate to
  `ensemble_weighted`.

Export from `calibre/ensemble/__init__.py`. No change to `median.py`.

### Tests

`tests/test_ensemble.py` (new or extended):
1. Weighted with equal weights matches mean.
2. Weighted validates weight-count mismatch / non-unit sum.
3. Quantile columns are ensembled row-wise using `is_quantile_column`.
4. Inverse-error weighting gives larger weight to the smaller error.

---

## Critical files to create / modify (this PR)

| File | Change |
|---|---|
| `calibre/orchestration/__init__.py` | New — re-export `DecisionLoop`, `DecisionLoopConfig`, `RoundResult`, `observe_per_horizon`, `observe_cumulative` |
| `calibre/orchestration/decision_loop.py` | New — orchestrator + two observe helpers |
| `benchmarks/vn2/run_benchmark.py` | Replace manual loop with `DecisionLoop`; per-round `cost/*` wired via `on_round` hook into the existing MLflow tracking |
| `benchmarks/vn2/run_seasonal.py` | Replace manual loop with `DecisionLoop` |
| `tests/test_decision_loop.py` | New |

Files NOT modified in this PR: everything under `calibre/engine/`,
`calibre/conformal/`, `calibre/order/`, `calibre/simulation/`,
`calibre/models/`, `calibre/pipeline/`, `calibre/ensemble/`,
`benchmarks/common/tracking.py`, `benchmarks/vn2/config.py`,
`benchmarks/vn2/simulator.py`, `benchmarks/vn2/tuning.py`,
`benchmarks/vn2/run_winning.py`.

Files NOT modified: `calibre/engine/backend.py` (orchestrator calls
`execute` unchanged), `calibre/conformal/*` (Phase 3 is frozen API),
`calibre/order/rs.py` (the three target-stock paths already cover
everything the loop needs), `calibre/simulation/*`, the MLflow
tracking module.

---

## Verification

1. `uv sync --extra dev --extra benchmarks`
2. `uv run pytest` — full suite green (baseline 266; this PR adds
   ~4 new tests in `tests/test_decision_loop.py`).
3. `uv run ruff check .`
4. `uv run mypy calibre/`
5. Line-count check: `wc -l benchmarks/vn2/run_benchmark.py
   benchmarks/vn2/run_seasonal.py` — both under 150 lines (down from
   ~500 today). Success criterion from
   `2026-04-14-calibre-gap-analysis.md` §Verification.
6. End-to-end: `uv run python benchmarks/vn2/run_benchmark.py` on the
   **fast-iteration config** (SeasonalNaive + 6 warmup origins) —
   total cost within ±5% of the Phase 3 baseline EUR 9,576. MLflow
   run appears under experiment `"vn2"` with per-round `cost/*`
   metrics logged via the new `on_round` hook.
7. `/go` opens a PR titled
   `feat(orchestration): decision-loop orchestrator for rolling benchmarks`
   against `main`.

## Risks

- **Cost regression**: the orchestrator must exactly reproduce the
  current driver's `observe()` dispatch. Mitigated by the two named
  helpers (`observe_per_horizon`, `observe_cumulative`) that encode
  `lessons.md:40`, plus the ±5% cost check in verification.
- **Origin-advancement off-by-one**: `lessons.md:31` warns that
  `BackendEngine` filters history with strict `<`; the loop must
  advance `origin` to `last_observed_ds + 1 step`. The
  `advance_origin_fn` injection makes this a single testable
  function rather than inlined arithmetic.
