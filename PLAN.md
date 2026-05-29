# PR G — De-shadow the VN2 benchmark (P1.1 + panel-tuning lift)

> **Handoff doc for an implementing agent (e.g. Codex).** Self-contained: you do
> **not** need the Obsidian vault or any prior chat to execute this. Line numbers
> are as of the `main` referenced below; re-locate by symbol name if they drift.
>
> Base commit: `066f296` (`main`). Repo conventions in `CLAUDE.md` — **always**
> prefix Python tooling with `uv run` (never bare `python`/`pytest`/`ruff`/`mypy`).

---

## 0. Why this exists (read first)

`benchmarks/vn2/run_benchmark.py` is a **2,312-LOC monolith**. The roadmap item
P1.1 frames it as "split into `data/tuning/replay/diagnostics`," but the **real
problem** is that the benchmark is *a second, untested implementation of the
pipeline*. Its own module docstring lists five "documented gaps that this script
works around." A pure file-split would just relocate that duplication into four
files — exactly the anti-pattern a prior attempt (the reverted 77-file PR #38)
was faulted for.

**Root cause of the duplication:** `calibre/tuning/optimizer.py` already owns the
full Ray-Tune / Optuna / ASHA orchestration, but `calibre.tuning.TuningTask` is
**per-series only** (`unique_id: str`, point `Accuracy` / single-window `Cost`
objectives). The benchmark needs **panel/global** tuning (one global LightGBM
across all series, a cumulative-cost objective over every `(uid, origin)`
window), so it reimplements the entire study runner.

Near-identical clones (benchmark → canonical):

| `run_benchmark.py` (shadow)                | `calibre/tuning/optimizer.py` (canonical) |
| ------------------------------------------ | ----------------------------------------- |
| `_restore_cwd`                             | `restore_cwd` (public, line 80)           |
| `_trial_thread_env`                        | `_trial_thread_env` (line 90)             |
| `_resolve_tune_storage_path`               | `_resolve_tune_storage_path` (line 67)    |
| `_resolve_max_concurrent_trials`           | `_resolved_max_concurrent_trials` (line 50) |
| `_run_optuna_tune` (Tuner/ASHA/OptunaSearch) | `_run_optuna_study` (line 366)          |
| `_best_tune_result`                        | `_best_result_config` (line 193)          |
| `_HpoSearchSpaceAdapter` / `_CostSearchSpaceAdapter` | `_OptunaSearchSpaceAdapter` (line 236) |
| `_cap_threaded_model_config`               | `calibre.execution.threading.cap_threaded_config` |

**Goal:** the benchmark becomes VN2-specific glue + config that *calls into*
`calibre`, with no functional duplication — achieved by making **panel tuning a
first-class `calibre` concept** and refitting the benchmark onto it.

**Decisions already locked (do not relitigate):**
- **Two staged PRs.** G1 = library change (this is where the new capability
  lands). G2 = benchmark consumes it + reorg. Each independently deployable.
- **Push panel tuning into `calibre`** (not merely share an orchestration core).

---

## 1. Invariants that MUST NOT regress

1. **VN2 winning-config baseline `total_cost == 4992.20`** (see `CLAUDE.md`
   Gotchas). The full tuned benchmark on the real dataset must still produce
   exactly this. Do not drift it.
2. **MLflow history artifact** — `log_cached_replay_run` must keep its
   `mlflow.log_artifact(history.csv)` call **inside** the
   `with tempfile.TemporaryDirectory()` block (currently correct at
   `run_benchmark.py:~1234-1245`). The prior reverted attempt dropped it (the CSV
   was written then discarded when the temp dir closed). Don't reintroduce that
   bug.
3. **Public cross-module APIs only.** No re-export shims in the runner, no
   production monkeypatching, no `_private` helper imported across module
   boundaries. (This was the headline complaint about the reverted attempt.)
4. **`conformal/runtime.py` is the stable conformal interface** — don't touch it.
5. **`/tune` endpoint** (`calibre/api/main.py:~623-635`, uses
   `optimize_task_candidate`) must keep working unchanged.

## 2. Out of scope (later roadmap PRs — do NOT bleed in)

- Cost-search swallowing infra failures into `inf` / fail-fast default (a
  separate PR owns this). Keep `run_cost_search`'s error semantics as-is.
- `os.environ` thread-safety beyond what the shared core already handles.
- Deleting `_optimize_task_sequential` (it is currently **dead code** —
  `optimizer.py:325`, referenced by no caller; a test defensively monkeypatches
  it). Leave it untouched.

---

## PR G1 — Library: shared study core + first-class panel tuning

**All changes in `calibre/`.** Behavior-preserving for the existing per-series
path; adds the panel capability with its own tests. This PR is **risky
(library/architecture)** → open green, then a human runs `/code-review ultra`
before merge.

Branch: `git checkout main && git checkout -b g1-panel-tuning-core`.

### G1.1 — Extract a trainable-agnostic study runner

In `calibre/tuning/optimizer.py`, pull the orchestration out of
`_run_optuna_study` (lines 366-506) into a public, task-agnostic core. Suggested
shape:

```python
@dataclass(frozen=True, slots=True)
class StudyOutcome:
    best_config: dict[str, Any]
    results: Any  # ray.tune.ResultGrid — exposed so callers needn't reach into
                  # Optuna's private _ot_study

def run_optuna_study(
    *,
    space: Callable[[optuna.Trial], None],   # OptunaSearch-ready (returns None)
    trainable: Callable[..., None],
    n_trials: int,
    max_t: int,
    seed: int | None,
    asha_grace_period: int,
    cpu_per_trial: float,
    max_concurrent_trials: int,
    ray_address: str | None,
    ray_local_mode: bool,
    tune_storage_path: str,
    metric: str = _OBJECTIVE_METRIC,
    mode: str = "min",
    time_attr: str = _ORIGIN_INDEX,
    experiment_name: str | None = None,
    callbacks: list[Any] | None = None,
    trial_state: Any | None = None,  # ray.put()'d and passed as state_ref kwarg
) -> StudyOutcome:
    ...
```

It owns everything currently duplicated on both sides:
`prepare_ray_environment()`, the `OptunaSearch` + `ASHAScheduler` +
`tune.Tuner` + `tune.with_resources` wiring, the
`TUNE_DISABLE_AUTO_CALLBACK_LOGGERS` / `RAY_CHDIR_TO_TRIAL_DIR` env guards,
`restore_cwd()`, `acquire_ray_runtime`/`release`, the
`tune.with_parameters(..., state_ref=...)` wrap when `trial_state` is set, and
`_best_result_config(results)`.

Reuse `_OptunaSearchSpaceAdapter` (line 236) as the `space` wrapper — both
callers pass `_OptunaSearchSpaceAdapter(their_search_space)`.

**Refactor two helpers from task-based to value-based so both per-series and
panel can share them** (keep behavior identical):
- `_resolved_max_concurrent_trials(task)` → `_resolve_max_concurrent_trials(
  max_concurrent_trials: int | None, cpu_per_trial: float) -> int`.
- ⚠️ `_resolve_tune_storage_path(task)` is **imported directly by
  `tests/test_tuning_task.py:15`** (`_resolve_tune_storage_path` with a
  `TuningTask` arg, asserted in
  `test_default_tune_storage_path_stays_under_results_dir_when_home_unwritable`).
  Either keep that exact signature, or split into
  `_resolve_tune_storage_path(tune_storage_path, results_dir)` **and** update the
  test. Simplest: add a value-based helper and have the task-based one delegate,
  leaving the test import working.

Then **refit** `_run_optuna_study(task)` to: validate (`_validate_task`),
snapshot conformal (`_snapshot_conformal_runtime`), build `worker_task`,
`history = _history_with_uid(worker_task)`, define the per-series `_trainable`
closure (unchanged body, lines 397-454), and call `run_optuna_study(...)`,
returning `outcome.best_config`. **No behavior change** — the existing per-series
tests prove the seam.

### G1.2 — Add a panel/global tuning task + entry point

In `calibre/tuning/task.py` add a frozen `PanelTuningTask`: panel `history` and
`actuals` spanning **all** uids, `origins: list[pd.Timestamp]`, `horizon`,
`base_model_config` (expects `scope="global"`), a define-by-run `search_space:
Callable[[optuna.Trial], TuningCandidate]`, a panel `objective`, plus the same
ray/asha/seed/cpu/storage knobs as `TuningTask` (`n_trials`, `freq`, `seed`,
`asha_grace_period`, `cpu_per_trial`, `max_concurrent_trials`,
`max_uid_concurrency`, `ray_address`, `ray_local_mode`, `tune_storage_path`,
`results_dir`, `tune_experiment_name`, mlflow_* ).

In `optimizer.py` add `optimize_panel_task(task: PanelTuningTask) -> dict`:
builds **one** global `ForecastTask(history=panel_history, horizon, model_config)`
per trial (capped via `cap_threaded_config`), iterates `engine.iter_origins([task],
actuals, origins)`, and scores the objective on the **full per-origin frame** via
the existing `_objective_contribution_with` / `_newly_resolved_frame` machinery
(lines 162-190) — those already dedupe by `_FORECAST_KEY_COLUMNS` and call
`objective.evaluate(frame, frame[Y])`, which works for multi-window panel frames.
Then call `run_optuna_study(...)` and return the best `model_config`.

Export `PanelTuningTask`, `optimize_panel_task`, `run_optuna_study`,
`StudyOutcome`, and `CumulativePinball` from `calibre/tuning/__init__.py`
(and add to `__all__`).

### G1.3 — Add the panel cumulative objective

In `calibre/tuning/objectives.py` add `CumulativePinball` implementing the
`TuningObjective` protocol (`evaluate(frame, actuals: pd.Series) -> float`):

- group `frame` by `(UNIQUE_ID, FORECAST_ORIGIN)`;
- per window sum the actual `y` and the quantile column `q_<alpha>` over `h`;
- return the mean over windows of `pinball_linear(actual_sum, pred_sum, tau)`.

Reuse `calibre.evaluation.point_metrics.pinball_linear`. This generalizes the
benchmark's `_cumulative_pinball` (`run_benchmark.py:531-572`). Fields:
`quantile: float` (selects the `q_*` column), `tau: float` (cost-optimal
cumulative quantile, `Cu/(Cu+Co)`). The objective `Protocol` already fits panel
frames; only the existing `Accuracy`/`Cost` assume single windows.

### G1 — critical files
- `calibre/tuning/optimizer.py` (extract core, refit per-series, add panel entry, value-based helpers)
- `calibre/tuning/task.py` (`PanelTuningTask`)
- `calibre/tuning/objectives.py` (`CumulativePinball`)
- `calibre/tuning/__init__.py` (exports)
- `tests/test_tuning_task.py` (only if you change `_resolve_tune_storage_path`'s signature)

### G1 — test strategy
- Keep the regression net green (these exercise the per-series path, now flowing
  through the extracted core):
  `tests/test_tuning_task.py`, `tests/tuning/`, `tests/storage/test_tuning_runs.py`.
  Note `tests/tuning/test_ray_tune_with_conformal.py` asserts the conformal path
  does **not** fall back to sequential — preserve that (don't call
  `_optimize_task_sequential`).
- New `tests/tuning/test_panel_tuning.py`:
  - `CumulativePinball` unit test on a hand-built 2-series, multi-window frame
    with known sums → known pinball mean.
  - `optimize_panel_task` smoke: tiny global model, `ray_local_mode=True`,
    `n_trials=1`, 2 series, assert a complete best `model_config` dict.
- Verify: `uv run pytest tests/test_tuning_task.py tests/tuning/
  tests/storage/test_tuning_runs.py`, then `uv run mypy calibre/`,
  `uv run ruff check .`, `uv run ruff format .`. Confirm `calibre/api/main.py`
  still imports (`uv run python -c "import calibre.api.main"`).

### G1 — ship
Commit, push, open PR. **Stop at "open PR + green CI"** — the harness/maintainer
policy blocks self-merge to `main`. Flag it as risky so a human runs
`/code-review ultra <PR#>`; address findings, then they merge with
`gh pr merge <#> --squash --delete-branch`.

---

## PR G2 — Benchmark: consume the library, delete the shadow, reorganize

**All changes in `benchmarks/` + `tests/`. No `calibre/` changes.** Do this only
**after G1 is merged to `main`.** Branch off the updated main:
`git checkout main && git pull && git checkout -b g2-vn2-benchmark-reorg`.

### G2.1 — Refit the two search entry points onto `calibre`
- **`run_hpo`** → build a `PanelTuningTask` (VN2 history/actuals/origins + the VN2
  define-by-run space returning a `TuningCandidate` +
  `CumulativePinball(quantile, tau=HPO_COST_OPTIMAL_TAU)`) and call
  `optimize_panel_task`. **Delete** from the benchmark: `_run_optuna_tune`,
  `_HpoSearchSpaceAdapter`, `_best_tune_result`, `_resolve_max_concurrent_trials`,
  `_resolve_tune_storage_path`, `_short_tune_trial_name`, `_restore_cwd`,
  `_trial_thread_env`, `_cap_threaded_model_config` (→
  `calibre.execution.threading.cap_threaded_config`), the env-guard block, the
  `_TUNE_*` constants, and `_cumulative_pinball` (→ `CumulativePinball`).
- **`run_cost_search`** → keep the VN2-specific objective (simulator EUR cost over
  a replay) but drive the search via the public `run_optuna_study` core (supply a
  VN2 trainable that builds/replays the cache and reports cost via
  `tune.report`). Use `StudyOutcome.results` instead of reaching into Optuna's
  private `_ot_study` where free. **Leave the `inf`-swallowing error semantics
  unchanged** (a later PR owns that).

### G2.2 — Lift generic helpers out of the benchmark
Move `_log_mlflow_params`, `_stable_value`, `_stable_config_key`
(`run_benchmark.py:1180-1218`) → `benchmarks/common/tracking.py` as **public**
`log_mlflow_params` / `stable_value` (shared by `replay` + the search glue; fixes
the cross-module private-import complaint).

### G2.3 — Reorganize the VN2-specific remainder
⚠️ **Name collisions:** `benchmarks/vn2/tuning.py` already exists (the *seasonal
per-series tuner*, imported only by `run_seasonal.py:67`) and
`benchmarks/vn2/dataset.py` is the `DatasetAdapter` (imported by
`calibre/execution/dataset_registry.py`). **Leave both untouched.** That is why
the new search glue goes in `search.py`, not `tuning.py`.

New modules — **every cross-module symbol is public (no leading underscore)**:

| New module                       | Holds (current `_private` → public) |
| -------------------------------- | ----------------------------------- |
| `benchmarks/vn2/data.py`         | `prepare_model_history`, `prepare_cumulative_target_history`, `as_cumulative_decision_frame`, `prepare_policy_forecast_frame`, `load_instock`, `model_uses_cumulative_target`, `build_model_config`, `strip_private`, `ROLLING_WINDOWS` (keep `_prepare_history` private — data-internal) |
| `benchmarks/vn2/replay.py`       | `summary_from_simulator`, `build_rs_params`, `round_actuals`, `orders_from_policy_result`, `order_conformal_warmup_frames`, `run_order_conformal_warmup`, dataclasses `CachedRound`/`VN2ReplayCache`/`ReplayResult`, `build_replay_cache`, `replay_cached_cost`, `log_cached_replay_run` (**keep `mlflow.log_artifact`**) |
| `benchmarks/vn2/diagnostics.py`  | `optimal_order_path_for_sku`, `just_in_time_order_path_for_sku`, `simulate_orders`, `cost_diagnostic_tables`, `oracle_diagnostic` (DP helpers `_advance_state_values`, `_round_order_to_step` stay private) |
| `benchmarks/vn2/search.py`       | thin VN2 glue: `run_hpo`, `run_cost_search` (build the `PanelTuningTask` / cost trainable, call `calibre`) |
| `benchmarks/vn2/run_benchmark.py` | **thin shell**: `run_benchmark` + `__main__` only (well under the <800-LOC target) |

Module DAG (acyclic): `data → replay → {diagnostics, search}`;
`run_benchmark → {data, replay, search}`. Keep `benchmarks/vn2/__init__.py` a
bare package marker (no re-exports).

### G2.4 — Callers (mostly unchanged because `run_benchmark` stays put)
- `calibre/cli/commands.py:115` imports `run_benchmark` from
  `benchmarks.vn2.run_benchmark` — **unchanged**.
- `tests/integration/test_ray.py:11` imports `run_benchmark`; `tests/cli/test_cli.py:265`
  monkeypatches `benchmarks.vn2.run_benchmark.run_benchmark` — **unchanged**.
- `README.md:73` runs `python benchmarks/vn2/run_benchmark.py` — keep `__main__`.

### G2.5 — Tests
- Rewrite `tests/test_vn2_benchmark.py`: import the now-public helpers from their
  new homes (`benchmarks.vn2.data` / `replay` / `diagnostics` / `search`); drop
  every private cross-module import. The current file imports these privates
  (lines 21-32): `_as_cumulative_decision_frame`, `_optimal_order_path_for_sku`,
  `_prepare_cumulative_target_history`, `_round_actuals`,
  `_run_order_conformal_warmup` → switch to the new public names. Replace the
  `run_benchmark_module` monkeypatch of `_run_optuna_tune` /
  `_TUNE_OBJECTIVE_METRIC` (deleted) with patching at the `calibre` core seam, or
  restructure around the public `optimize_panel_task`. Keep the equivalence test
  (`test_cached_replay_matches_run_benchmark_on_small_subset`) and the structural
  config tests.
- **Regression guard:** run the tuned benchmark on the full VN2 dataset and
  assert `total_cost == 4992.20`; the small-subset equivalence test backstops it
  cheaply in CI.
- Verify: `uv run pytest tests/test_vn2_benchmark.py tests/integration/test_ray.py
  tests/cli/test_cli.py tests/test_vn2_seasonal.py`; smoke
  `uv run python benchmarks/vn2/run_benchmark.py`; `uv run mypy calibre/`
  (and the benchmark if configured), `uv run ruff check .`, `uv run ruff format .`.

### G2 — ship
Build → local `/code-review` → open PR → stop at green CI; maintainer merges.

---

## 3. Quick reference — `run_benchmark.py` triage (current → destination)

- **Delete (duplicates calibre):** `_run_optuna_tune`, `_HpoSearchSpaceAdapter`,
  `_CostSearchSpaceAdapter` boilerplate, `_best_tune_result`,
  `_resolve_max_concurrent_trials`, `_resolve_tune_storage_path`,
  `_short_tune_trial_name`, `_restore_cwd`, `_trial_thread_env`,
  `_cap_threaded_model_config`, env-guard block, `_TUNE_*` consts,
  `_cumulative_pinball`, `_optuna_study_from_search_alg` (where `StudyOutcome`
  removes the need).
- **Move to `calibre` (G1):** the orchestration core + panel task + cumulative
  objective (above).
- **Move to `benchmarks/common/tracking.py` (G2):** `_log_mlflow_params`,
  `_stable_value`, `_stable_config_key`.
- **Keep, VN2-specific (G2 → data/replay/diagnostics/search):** everything in the
  table in G2.3 — VN2 data layout/target shaping, simulator coupling, replay
  cache, RS-from-simulator-state, the exact finite-horizon **oracle DP** (VN2
  lead-time mechanics), and the two public `run_hpo`/`run_cost_search` wrappers.

---

## 4. Status at handoff
- Branch `g1-panel-tuning-core` was created then removed (no commits). **No code
  changes exist yet** — start G1 from scratch on a fresh branch off `main`.
- This `PLAN.md` is the canonical handoff. (The repo's durable plan home is an
  Obsidian vault the implementing agent won't have; everything needed is inlined
  here.)
