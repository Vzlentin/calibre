# Wire `future_x` into forecast adapters (Phase 4 — Workstream 2, Gap 7)

## Context

Phase 4 Workstream 1 (the decision-loop orchestrator) shipped in PR #16. The
next deferred item in `~/.claude/plans/implement-phase-4-go-fluttering-cherny.md`
is Workstream 2: wiring `future_x` through the three forecast adapters.

`ForecastTask.future_x: pd.DataFrame | None` has existed for a while
(`calibre/tasks/forecast_task.py:23`) and the engine threads it through in
both code paths (`calibre/engine/backend.py:167-173` parallel,
`218-224` direct), but all three adapters silently ignore it at `predict`
AND drop regressor columns at `fit`. Net effect: exogenous regressor
support is dead code today and would silently mis-predict the day a user
sets `future_x`. The roadmap (`Vault/calibre/plans/2026-04-14-calibre-gap-analysis.md`
Gap 7) calls this out as a medium-priority correctness gap.

Outcome: regressor columns in `history` reach the underlying library's
`fit`, and non-empty `future_x` reaches the library's `predict` under the
correct kwarg for each (`X_df=` for mlforecast and statsforecast, `futr_df=`
for neuralforecast). Behavior when `future_x` is `None`/empty is byte-
identical to today.

## Scope

**In:** the three adapters + the engine's per-uid slice in `_run_parallel`
+ a small shared helper in the forecast-frame contract + adapter and engine
tests.

**Out:** no new adapters, no new ensemble methods, no VN2 driver changes,
no `ForecastTask` API change, no forecast-frame output schema change, no
dep bumps.

## Design

### 1. Shared helper (one place for "what counts as a regressor")

Add to `calibre/contracts/forecast_frame.py`:

```python
_RESERVED_HISTORY_COLS = frozenset({UNIQUE_ID, DS, Y})

def exogenous_columns(df: pd.DataFrame) -> list[str]:
    """Columns of `df` that are not {unique_id, ds, y} — i.e. regressors."""
    return [c for c in df.columns if c not in _RESERVED_HISTORY_COLS]
```

Three adapters + test assertions will call this. The convention
(regressor = any column beyond `{unique_id, ds, y}`) is a contract-level
decision, not adapter-internal. Order-preserving list keeps library calls
deterministic.

### 2. Adapter edits — same pattern in all three

Stop dropping regressor columns in `fit`; forward `future_x` in `predict`
only when non-empty (so the `None` / empty path is byte-identical).

#### `calibre/models/mlforecast.py`

- `fit` L116: replace `task.history[[UNIQUE_ID, DS, Y]].copy()` with
  `task.history[[UNIQUE_ID, DS, Y, *exogenous_columns(task.history)]].copy()`.
- `predict` L128: build `predict_kwargs = {"h": task.horizon}`; if
  `task.future_x is not None and not task.future_x.empty`, set
  `predict_kwargs["X_df"] = task.future_x`; call
  `self._mlf.predict(**predict_kwargs)`.

#### `calibre/models/statsforecast.py`

- `fit` L28: same exog-preserving projection.
- `predict` L37: same conditional `X_df=` pattern feeding
  `self._sf.predict(**predict_kwargs)`. Some statsforecast models
  (e.g. `SeasonalNaive`) don't accept exogenous regressors — let the
  library raise; do not try/except (core rule).

#### `calibre/models/neuralforecast.py`

- `fit` L38: same exog-preserving projection.
- `predict` L47: build `predict_kwargs = {}`; if `future_x` non-empty set
  `predict_kwargs["futr_df"] = task.future_x`; call
  `self._nf.predict(**predict_kwargs)`.
- `futr_exog_list` already flows through `params` at L29 (it isn't in
  `_RESERVED_KEYS`), so no plumbing change — just note in a docstring that
  users must set `futr_exog_list=[<col>]` inside `model_config` for the
  model to consume the forwarded regressors.

### 3. Engine — slice `future_x` by uid in `_run_parallel` only

`calibre/engine/backend.py:167-173` currently copies the ENTIRE `future_x`
frame into each per-uid Fugue partition. Fix inside `_process_partition`,
immediately before constructing `origin_task`:

```python
task_future_x = task.future_x
if task_future_x is not None and not task_future_x.empty:
    task_future_x = task_future_x[task_future_x[UNIQUE_ID] == uid]
```

then pass `future_x=task_future_x`. Preserve `None` as `None` (so the
adapter's `is not None` branch still short-circuits).

`_run_direct` (L202-235) is global scope; it needs the cross-uid frame
intact and is left unchanged.

### 4. Tests

**Shared pattern:** monkeypatch the library class (`MLForecast`,
`StatsForecast`, `NeuralForecast`) with a `MagicMock`, spy on `fit` and
`predict` calls, and assert column presence + kwarg presence. This keeps
CI fast and deterministic.

#### `tests/test_mlforecast_adapter.py`
- `test_fit_preserves_exogenous_columns`: history with a `promo` column;
  assert the df passed to `MLForecast.fit` contains `promo`.
- `test_predict_forwards_future_x_as_X_df`: assert
  `MLForecast.predict` was called with `X_df=<frame with promo>`.
- `test_predict_without_future_x_omits_X_df` (regression): existing
  baseline; assert `X_df` not in `predict` call kwargs.

#### `tests/test_statsforecast_adapter.py`
- Mirror three tests (use `AutoARIMA`, which accepts exog).

#### `tests/test_neuralforecast_adapter.py`
- Mirror three tests with `futr_df` instead of `X_df`. Include
  `futr_exog_list=["promo"]` in `model_config` to exercise the existing
  `params` pass-through and document the user contract.

#### `tests/test_engine.py`
- `test_run_parallel_slices_future_x_per_uid`: build two tasks
  (uids `A` and `B`), attach a `future_x` with rows for BOTH uids, register
  a stub adapter that records the `future_x` it receives. After
  `BackendEngine.execute`, assert the recorded `future_x` for `A` contains
  only `A` rows (and likewise `B`).
- `test_run_direct_passes_full_future_x`: same setup with a `scope="global"`
  model; assert the stub adapter sees the full multi-uid frame.

## Critical files

| File | Change |
|---|---|
| `calibre/contracts/forecast_frame.py` | Add `exogenous_columns()` helper + `_RESERVED_HISTORY_COLS`. |
| `calibre/models/mlforecast.py` | `fit` keeps exog cols; `predict` forwards `X_df` when non-empty. |
| `calibre/models/statsforecast.py` | Same pattern. |
| `calibre/models/neuralforecast.py` | Same pattern with `futr_df`; docstring about `futr_exog_list`. |
| `calibre/engine/backend.py` | Slice `future_x` by uid inside `_run_parallel._process_partition`. |
| `tests/test_mlforecast_adapter.py` | Three new tests (exog fit, X_df predict, no-X_df regression). |
| `tests/test_statsforecast_adapter.py` | Three analogous tests with `AutoARIMA`. |
| `tests/test_neuralforecast_adapter.py` | Three analogous tests with `futr_df` + `futr_exog_list`. |
| `tests/test_engine.py` | Parallel uid-slice test + direct-path contrast test. |

## Files NOT modified (scope guard)

- `calibre/tasks/forecast_task.py` (field already exists).
- `calibre/models/base.py`, `calibre/models/registry.py`.
- `calibre/conformal/*`, `calibre/order/*`, `calibre/simulation/*`,
  `calibre/orchestration/*` (Workstream 1, frozen).
- `calibre/pipeline/tasks.py` (Workstream 3, follow-up PR).
- `calibre/ensemble/*` (Workstream 4, follow-up PR).
- `benchmarks/**` — VN2 drivers don't use exogenous regressors; no wiring
  change needed here.
- `pyproject.toml` / `uv.lock` — no new deps.

## Reuse — existing code to leverage

- `UNIQUE_ID`, `DS`, `Y` constants from `calibre/contracts/forecast_frame.py`
  (already imported by all three adapters).
- `_build_predict_frame` and `_build_quantile_predict_frame` in
  `calibre/models/base.py` / `mlforecast.py` — output-frame assembly is
  unchanged; we only augment input/predict kwargs.
- `params = {k: v for k, v in self._config.items() if k not in _RESERVED_KEYS}`
  pattern (neural L29) — already forwards `futr_exog_list` correctly.
- Existing test fixtures `repeating_history`, `lgbm_task`, `xgb_task` in
  `tests/test_mlforecast_adapter.py` — extend rather than duplicate.

## Verification

Run from `C:\Users\a933186\Dev\calibre`:

1. `uv sync --extra dev --extra benchmarks` — dep sync (no-op if already synced).
2. Focused adapter tests:
   `uv run pytest tests/test_mlforecast_adapter.py tests/test_statsforecast_adapter.py tests/test_neuralforecast_adapter.py -v`
3. Engine test:
   `uv run pytest tests/test_engine.py -v -k "future_x"`
4. Full suite: `uv run pytest` — all prior tests still green.
5. `uv run ruff check .` and `uv run ruff format --check .`.
6. `uv run mypy calibre/`.
7. End-to-end smoke: `uv run python benchmarks/vn2/run_seasonal.py` on the
   fast-iteration config — cost must match Phase 3/Workstream-1 baseline
   (VN2 has no exog, so the empty-path regression is exercised live).

## Risks

- **Empty/None path drift.** The library-call shape for `future_x is
  None` MUST stay identical. Mitigation: conditional `**predict_kwargs`
  (kwarg only added when non-empty) + explicit "no-X_df" regression test
  per adapter.
- **Library-specific rejection** (e.g. `SeasonalNaive` + `X_df` raises).
  By design we don't swallow it — document in docstring, let caller pick
  a compatible model.
- **Model config vs data mismatch** for NeuralForecast (`futr_exog_list`
  not in sync with `future_x` columns). Library raises at fit/predict;
  acceptable user contract.
- **NaN regressors.** Some libraries tolerate (LightGBM), some don't
  (sklearn linear). Do not pre-validate; surface the library's error.
- **Column ordering.** `exogenous_columns` preserves `df.columns` order
  so library calls are deterministic; tests build history with a fixed
  column order to avoid flakes.
- **Fugue schema.** `_run_parallel`'s schema string only describes output
  columns; `future_x` is closure-captured on the task, not partitioned
  via schema — adding the uid slice requires no schema change.
