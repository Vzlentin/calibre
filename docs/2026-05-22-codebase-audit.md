# Calibre Codebase Audit

**Branch:** `deslop-audit`  
**Scope:** Full codebase, logical pillar by logical pillar  
**Total Python LOC (excl. venv):** ~25,764 across 179 files  
**Audit method:** Automated slop-scan (type ignores, `Any`, broad exceptions, deep nesting, long functions, commented code, `print` in library code) + manual deep-dive of top-offender files.

---

## 1. Executive Summary

The codebase is generally clean in the *mathematical* pillars (core, conformal, ordering, evaluation) and accumulates slop at the *integration* boundaries: CLI config parsing, execution backend, tuning optimizer, and benchmark scripts. There is a clear pattern: **slop concentrates wherever the library interfaces with external systems (YAML, Ray, Optuna, fsspec) or where prototypes hardened into production without refactoring.**

**Top-level grades (subjective):**

| Pillar | Grade | Primary Issue |
|--------|-------|---------------|
| core | A | None material |
| ordering | A | None material |
| conformal | B+ | `# type: ignore` in adaptive numerics |
| evaluation | B+ | One broad `except Exception` in point metrics |
| forecasting | B+ | Optional-import type ignores |
| api | B | Several broad `except Exception` safety nets; imports private backend internals |
| storage | B | One broad `except Exception` in SQL rollback; some `Any` |
| execution | C+ | Duplicated `_cap_threaded_config`; `Any` Ray types; duck-typing state persistence; `# type: ignore[union-attr]` |
| cli | C | Manual imperative YAML parsing with `Any` soup; `# type: ignore[arg-type]` to bypass Literal validation; `print()` in library boundary |
| tuning | C | `cast(optuna.Trial, ...)` type slop; deep nesting in Ray trainable; duplication with backend |
| benchmarks | D | `run_benchmark.py` is 1,948 LOC monolith absorbing HPO, decision loop, and data prep that belong in library modules |
| tests | B | `# type: ignore` on mocks instead of `Mock[Protocol]` typing; some commented-out code |

---

## 2. Methodology

1. Automated scan excluded `.venv`, `__pycache__`, `.git`, `node_modules`.
2. Scored each file on: `type: ignore` (x3), `Any` casts (x2), bare `except` (x5), broad `except Exception` (x2), deep nesting >5 (x5), functions >80 LOC (x3), commented-out code (x1), `print` in library code (x2).
3. Manually reviewed the 15 highest-scoring non-script files and all files with >2 `# type: ignore` or `Any` misuse.
4. Cross-referenced duplicated logic and architectural leakage.

---

## 3. Pillar-by-Pillar Findings

### 3.1 Core (`calibre/core/`) — CLEAN

- 9 files, 634 LOC.
- Zero type ignores, zero `Any` misuse, zero broad exceptions, zero long functions.
- The pillar holds. The frame/task/metrics abstractions are tight.

### 3.2 Ordering (`calibre/ordering/`) — CLEAN

- 14 files, 854 LOC.
- Zero slop flags.
- Policy implementations are small, focused, and type-complete.

### 3.3 Conformal (`calibre/conformal/`) — MINOR TYPING GAPS

- 14 files, 2,216 LOC.

**Findings:**
- `calibre/conformal/adaptive.py:153,162` — `# type: ignore[assignment]` on `_clip_alpha` returns assigned to `np.ndarray` attributes. The `_clip_alpha` function lacks precise return-type overloads for array-vs-scalar inputs. Fix: overload `_clip_alpha` or narrow its return type.
- `calibre/conformal/adaptive.py:167` — `# type: ignore[union-attr]` on `self._alpha.copy()`. `self._alpha` is typed as `np.ndarray` but the checker sees ambiguity from the `_clip_alpha` call path.
- `calibre/conformal/numerics.py:56` — `# type: ignore[return-value]` in `_validate_quantile_rule`. Returns a `str` but is typed as `Literal["conformal", "higher"]`. Fix: use `typing.cast(Literal[...], ...)` instead of `# type: ignore`, or return a `Literal` directly from the validation.
- `MultiStepAdaptiveConformalInference.__init__` is missing type annotations on `alpha`, `gamma`, `initial_alpha`, `initial_radius` while the single-step `AdaptiveConformalInference.__init__` has full annotations. Inconsistency.

**Verdict:** Not slop in behavior, but slop in *type-system discipline*. The adaptive module is the most mathematically dense; imprecise types here erode confidence during refactors.

### 3.4 Forecasting (`calibre/forecasting/`) — OPTIONAL-DEPENDENCY PATTERN

- 17 files, 950 LOC.

**Findings:**
- `calibre/forecasting/neuralforecast_adapter.py:13-14` — `# type: ignore[assignment]` and `[misc, assignment]` for the optional `neuralforecast` import fallback (`neuralforecast = None`). This is a standard Python pattern for soft dependencies; acceptable but could be isolated in a `_compat.py` module.
- `Any` appears in adapter configuration dicts, which is expected for passthrough model kwargs.

**Verdict:** Acceptable. The optional-import pattern is conventional and contained.

### 3.5 Evaluation (`calibre/evaluation/`) — ONE BROAD CATCH

- 4 files, 495 LOC.

**Findings:**
- `calibre/evaluation/point_metrics.py:323` — `except Exception as err:` swallows an error during metric computation. Without context this could mask real failures. Should be narrowed to the specific exception the underlying metric raises (likely `ValueError` or `ZeroDivisionError`).
- `calibre/evaluation/forecast_metrics.py:79` — `# type: ignore[assignment]` for `name = getattr(metric_fn, "__name__", None)`. Fix: `name: str | None = getattr(...)` is sufficient; the ignore is unnecessary.

### 3.6 Execution (`calibre/execution/`) — INTEGRATION SLOP

- 12 files, 2,275 LOC.

**Findings:**
- **`_cap_threaded_config` duplication**: Identical function exists in `calibre/execution/backend.py:139` and `calibre/tuning/optimizer.py:82`. Divergence risk. Should live in one place (`calibre.execution.threading` or similar).
- **`calibre/execution/backend.py`** (807 LOC):
  - Lines 347, 349, 350, 723: `Any` used for Ray module, remote task refs, and `_ensure_ray` return. Ray has typed stubs (`ray` package includes `py.typed` since recent versions). These could be `ray.actor.ActorHandle` or `ray.remote_function.RemoteFunction`.
  - Line 512: `# type: ignore[union-attr]` on `order_ledger.append(order_result)`. The `order_ledger` is already guarded by `if self.order_config is not None` but type narrowing is lost. Fix: use an explicit `assert order_ledger is not None` or narrow with `typing.assert_never`.
  - Lines 249-257: `_adaptive_controller_drift` uses three chained `getattr` calls (`controller`, `error_history`, `target_alpha`) instead of a protocol or `isinstance` check. This is duck typing slop.
  - Lines 556-575 & 577-592: `_restore_conformal_state` and `_persist_conformal_state` use `getattr(conformal_runtime, "get_partition_states", None)` and `getattr(self.conformal_state_store, "list_for_run", None)` rather than defining a `PartitionedConformalRuntime` protocol. This makes static analysis impossible and invites runtime `AttributeError` if the attribute exists but is not callable.
  - Line 630-652: `_advance_issued_count_from_initial_ledger` accesses `runtime._issued_count` (private attribute) directly. Breaks encapsulation.
- **`calibre/execution/io.py`** (67 LOC): Clean. `fsspec` type ignores are for an untyped third-party library; acceptable.
- **`calibre/execution/decision_loop.py`**: `Any` used for `simulator` and step function. Should be a `Protocol` with `step(period, orders, actual_demand) -> ...`.
- **`calibre/execution/ledger.py`**: `Any` used for file handles in `_partition_value`. Could be `fsspec.core.OpenFile | IO[bytes]`.
- **`calibre/execution/validation.py`**: `Any` used in validation results dict. Acceptable for generic validator output.

### 3.7 Storage (`calibre/storage/`) — ONE BROAD CATCH

- 11 files, 835 LOC.

**Findings:**
- `calibre/storage/postgres.py:53` — `except Exception:` inside `session_scope` context manager. This is the SQLAlchemy rollback pattern and is idiomatic; the exception is re-raised. Not slop.
- `calibre/storage/postgres.py`: `Any` in `values: dict[str, Any]` for bulk insert mapping. Acceptable for ORM passthrough.

### 3.8 API (`calibre/api/`) — SAFETY-NET EXCEPTIONS + LEAKAGE

- 5 files, 1,084 LOC.

**Findings:**
- `calibre/api/main.py:278,326,370,580` — Four `except Exception as exc` blocks. Two are explicitly marked `# pragma: no cover - background task safety net`. The others wrap internal job execution. In a FastAPI app, broad catches prevent unhandled exceptions from crashing the process, but they also swallow stack traces unless carefully re-logged. Verify that all four log the full traceback before converting to HTTP 500 or job failure.
- `calibre/api/main.py` imports private functions from `calibre.execution.backend`: `_coerce_forecast_frame_dtypes`, `_finalize_preds`, `_fit_predict_task`. The backend module prefixing these with `_` indicates they are internal, yet the API layer depends on them. This is an architectural leakage: either promote them to public API in `calibre.execution` or encapsulate them behind a public facade.
- `calibre/api/run_store.py:106,195` — `except Exception as exc` around DB operations. Acceptable for idempotent store transitions, but should log the exception.

### 3.9 CLI (`calibre/cli/`) — CONFIG PARSING SLOP

- 4 files, 619 LOC.

**Findings:**
- **`calibre/cli/config.py`** (306 LOC):
  - Heavy use of `Any` in parsing helpers: `_require_mapping(data: Any, ...)`, `_require_key(mapping: dict[str, Any], ...)`, every `_parse_*` function takes `data: Any`. This is manual YAML unmarshalling when the project already has `pydantic` or could use `dataclasses` + `cattrs`. The `Any` soup makes the parser brittle: a wrong YAML key type causes a `ValueError` at runtime instead of a parse-time type error.
  - Lines 182, 186, 258: `# type: ignore[arg-type]` to coerce `str` values into `Literal` fields (`method`, `mode`, `backend`). This is the type system correctly flagging a design flaw: the parser validates the string *after* assignment, so the type checker sees an invalid literal. Fix: parse to `str`, validate, then `typing.cast(Literal[...], validated)` instead of `# type: ignore`.
  - The `_parse_execution` function (lines 225-265) is 40 lines of manual key whitelisting and validation. This is exactly what a schema validator does better.
- **`calibre/cli/commands.py`** (262 LOC):
  - Lines 162, 216, 218, 224, 245: `print()` statements inside `run_config`, `validate`, `health`. These functions are in the CLI module but `run_config` is also called programmatically (e.g., from `calibre/api/main.py` via the backend engine). `print()` side effects in library-adjacent code are slop; use a structured logger or return the message and let the caller decide how to display it.
  - Line 82: `# type: ignore[arg-type]` for `policy=config.ordering.policy` passed to `OrderPolicyConfig`. Same `Literal` coercion smell as `config.py`.

### 3.10 Tuning (`calibre/tuning/`) — RAY/OPTUNA TYPE SLIP

- 4 files, 763 LOC.

**Findings:**
- **`calibre/tuning/optimizer.py`** (525 LOC):
  - Line 362: `cast(optuna.Trial, optuna.trial.FixedTrial(...))`. `FixedTrial` *is* a `Trial` subclass; the cast is unnecessary if `search_space` is typed to accept `optuna.Trial`. Fix: change the `TuningTask.search_space` callable type hint to `Callable[[optuna.Trial], TuningCandidate]` (it likely already is), then remove the cast.
  - Lines 416-473: `_trainable` function nested inside `_run_optuna_study`. It is deeply nested (function inside function, with `try/finally`, `with`, `for`, `if`). The nesting depth is 6+. This hurts testability. Should be extracted to a module-level function with explicit parameters.
  - Lines 283-291: `_apply_ordering_overrides` takes `objective: Any` and checks `is_dataclass(objective)`. Should accept `DataclassInstance | T` bounded by an ordering objective protocol.
  - `_evaluate_candidate` and `_trainable` share ~30 lines of identical setup logic (build `ForecastTask`, build `ConformalOptions`, build `BackendEngine`). DRY violation.

### 3.11 Benchmarks (`benchmarks/`) — MONOLITH ABSORPTION

- 13 files, 4,161 LOC.

**Findings:**
- **`benchmarks/vn2/run_benchmark.py`** (1,948 LOC):
  - This is the single largest file in the project by an order of magnitude. It contains:
    - Data preparation (`_prepare_history`, `_prepare_cumulative_target_history`, `_load_instock`)
    - Model config building (`_build_model_config`)
    - HPO logic (`_suggest_from_spec`, `_objective_fn`, `_run_panel_hpo`)
    - Decision loop orchestration
    - Cumulative conformal risk runtime setup
    - Result tracking and MLflow logging
  - The module docstring (lines 20-31) explicitly lists **5 documented gaps** that the script "works around":
    1. `MLForecastAdapter` silently drops date/static features.
    2. `TuningTask` is per-series and point-metric only, so panel + quantile + cumulative HPO is inlined.
    3. `ensemble_median` ignores quantile columns.
    4. Decision loop was "now delegated to `calibre.execution.DecisionLoop`" but the script still has its own orchestration.
    5. Exogenous `future_x` is dead end-to-end.
  - **This is the definition of architectural slop**: a benchmark script that has grown into a parallel implementation of library features because the library's abstractions were insufficient. Every item in that workaround list is a missing library feature that forces duplication into the benchmark.
- **`benchmarks/cp/aci/run_aci_parity.py`** (560 LOC): Also large for a parity script. Contains its own data loading, plotting, and metric computation that could be thinner.
- **`benchmarks/common/tracking.py`** (227 LOC): `# type: ignore[assignment]` for mlflow fallback object. Acceptable.

### 3.12 Tests (`tests/`) — MOCK TYPING + COMMENTED CODE

- 68 files, 10,670 LOC.

**Findings:**
- `tests/test_run_store.py:68,69,71,80` — `# type: ignore[attr-defined]` and `[union-attr]` for mock store methods. The test mocks a `RunStore` protocol but doesn't type the mock as `Mock[RunStore]`. Fix: use `unittest.mock.Mock(spec=RunStore)` or `create_autospec`.
- `tests/test_forecast_task.py:37` — `# type: ignore[misc]` for mutating a frozen dataclass field. This is intentional (testing immutability), so the ignore is justified.
- `tests/test_order_dispatch.py:226` — `# type: ignore` for intentionally invalid `OrderPolicyConfig`. Justified.
- Several test files contain commented-out `import` or `print` statements (detected by scan but not material).

### 3.13 Infra / Scripts (`infra/`, `scripts/`) — SCRIPTS

- 3 files, 467 LOC.

**Findings:**
- `scripts/databricks_notebook.py` (92 LOC): Contains `# TODO` and `print` statements. This is expected for a deployment script.
- `infra/aws/...`: Clean.

---

## 4. Root Issue Synthesis

### 4.1 The Five Root Issues

From the pillar findings, five interconnected root issues generate most of the observed slop:

#### R1: Manual Config Parsing Instead of Schema Validation

**Location:** `calibre/cli/config.py`
**Symptoms:** `Any` soup, `# type: ignore[arg-type]`, long parsing functions, runtime `ValueError` instead of parse-time validation.
**Impact:** Every new config field requires manual validation code. Type checker cannot help. Errors surface at runtime when a user runs a backtest.
**Connection:** R1 forces R2 (type-system avoidance) because the parser cannot produce statically typed objects.

#### R2: Type-System Avoidance at Integration Boundaries

**Location:** `calibre/cli/config.py`, `calibre/execution/backend.py`, `calibre/tuning/optimizer.py`
**Symptoms:** `# type: ignore`, `cast(...)`, `Any` for Ray/Optuna/fsspec.
**Impact:** Refactors in the backend or tuning become dangerous because the type checker is blind to half the call graph.
**Connection:** R2 is caused by R1 (config parsing) and R4 (Ray/Optuna interfaces not being properly typed or abstracted).

#### R3: Benchmark Monolith Absorbing Library Responsibilities

**Location:** `benchmarks/vn2/run_benchmark.py`
**Symptoms:** 1,948 LOC script with inlined HPO, decision loop, data prep, and model config building.
**Impact:** The benchmark is not just a benchmark; it is a second, untested implementation of Calibre's pipeline. Bug fixes in the library may not fix the benchmark, and vice versa. The docstring lists 5 library gaps that the script works around.
**Connection:** R3 is caused by R5 (backend engine abstraction gaps). The engine did not expose enough hooks for cumulative-target HPO, so the benchmark reimplemented them.

#### R4: Missing Protocols for External Integrations

**Location:** `calibre/execution/backend.py`, `calibre/execution/decision_loop.py`
**Symptoms:** `getattr(obj, "method", None)` duck typing, `Any` for simulator, `Any` for Ray handles.
**Impact:** Static analysis impossible. Runtime `AttributeError` risk. Hard to test backend logic without importing Ray.
**Connection:** R4 makes R2 worse and prevents cleanup of R3.

#### R5: Backend Engine Encapsulation Leaks

**Location:** `calibre/execution/backend.py`
**Symptoms:** API imports `_private` functions; `_advance_issued_count_from_initial_ledger` pokes at `runtime._issued_count`; state persistence uses duck typing.
**Impact:** The backend engine is both too large (807 LOC) and too porous. It tries to be a universal execution coordinator but leaks its internals to both the API layer and the conformal runtime.
**Connection:** R5 is the architectural root that enables R3 (benchmark monolith). Because the engine is hard to extend cleanly, the benchmark author chose to bypass it.

### 4.2 Causal Graph

```
R1 (Manual Config Parsing)
  |
  +--> R2 (Type Avoidance) ......................> cli/config.py, cli/commands.py
  |
R4 (Missing Protocols)
  |
  +--> R2 (Type Avoidance) ......................> backend.py, optimizer.py
  |
R5 (Backend Encapsulation Leaks)
  |
  +--> R3 (Benchmark Monolith) ..................> run_benchmark.py
  |     |
  |     +--> Duplication with tuning/optimizer ..> _cap_threaded_config, HPO logic
  |
  +--> R4 (Missing Protocols) ...................> conformal state persistence
```

### 4.3 The Positive Counter-Example

The conformal, ordering, and core pillars are clean because they:
1. Have stable, narrow interfaces (no external system integration).
2. Use protocols/abstract base classes (`OnlineConformalController`, `Score`).
3. Do not parse untyped user input.
4. Are unit-tested without heavy mocking.

This proves the slop is not a team-wide discipline failure; it is a **boundary-condition** problem.

---

## 5. Priority Recommendations

### P0 (Fix First)

1. **Replace manual YAML parsing in `calibre/cli/config.py`** with `pydantic` or `dataclasses` + `cattrs`. This eliminates ~15 `Any` annotations, 3 `# type: ignore[arg-type]`, and the entire class of runtime `ValueError` from malformed configs. Estimated LOC reduction: -80.
2. **Extract `_cap_threaded_config`** to a single module (`calibre.execution.threading`). Trivial, removes duplication.
3. **Add `PartitionedConformalRuntime` Protocol** and make `SymmetricIntervalRuntime` implement it explicitly. Replace `getattr` chains in `backend.py` with `isinstance(runtime, PartitionedConformalRuntime)`. Removes duck-typing slop and enables static analysis.

### P1 (Fix Before Next Milestone)

4. **Type the Ray integration**: Replace `Any` for `self._ray`, `self._remote_process_task`, etc., with actual Ray types. Ray stubs exist.
5. **Remove `# type: ignore` from `adaptive.py` and `numerics.py`** by fixing `_clip_alpha` return types and `_validate_quantile_rule` casting. Low risk, high signal.
6. **Audit `benchmarks/vn2/run_benchmark.py`** for functions that belong in library modules. Specifically:
   - `_prepare_cumulative_target_history` -> `calibre.execution.dataset` or `calibre.forecasting.transforms`
   - `_build_model_config` -> `calibre.forecasting.mlforecast_adapter` or a factory
   - `_run_panel_hpo` -> `calibre.tuning.optimizer` or a new `calibre.tuning.panel` module
   Target: reduce `run_benchmark.py` to <800 LOC.
7. **Replace `print()` in `commands.py`** with logger calls or return strings for display.

### P2 (Opportunistic)

8. **Type test mocks** with `create_autospec(RunStore)` instead of `# type: ignore`.
9. **Narrow broad `except Exception` in `api/main.py`** and `evaluation/point_metrics.py` to specific exception types.
10. **Extract `_trainable` in `optimizer.py`** to a module-level function and parameterize it explicitly.

---

## 6. Quick-Win Patch List

If you want to knock out the easiest wins in one commit:

- `calibre/cli/config.py:182,186,258` — Replace `# type: ignore[arg-type]` with `typing.cast(Literal[...], validated_str)`.
- `calibre/evaluation/forecast_metrics.py:79` — Remove `# type: ignore[assignment]`; change to `name: str | None = getattr(...)`.
- `calibre/execution/backend.py:512` — Replace `# type: ignore[union-attr]` with `assert order_ledger is not None`.
- `calibre/execution/backend.py` + `calibre/tuning/optimizer.py` — Extract `_cap_threaded_config` to `calibre.execution.threading`.
- `calibre/cli/commands.py` — Replace `print()` with `logger.info(...)`.

---

## 7. Files Scanned

All `.py` files in the repo, excluding `.venv`, `__pycache__`, `.git`, `.mypy_cache`, `.ruff_cache`, `.pytest_cache`.

| Pillar | Files | LOC |
|--------|-------|-----|
| tests | 68 | 10,670 |
| benchmarks | 13 | 4,161 |
| execution | 12 | 2,275 |
| conformal | 14 | 2,216 |
| api | 5 | 1,084 |
| forecasting | 17 | 950 |
| ordering | 14 | 854 |
| storage | 11 | 835 |
| tuning | 4 | 763 |
| core | 9 | 634 |
| cli | 4 | 619 |
| evaluation | 4 | 495 |
| infra_scripts | 3 | 467 |

---

*Audit completed on branch `deslop-audit`.*
