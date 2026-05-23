# /deslop Horizontal Audit — PR #38 (Cardinal Improvements)

**Date:** 2026-05-23
**Scope:** Full diff of `main..cardinal-improvements` (7,166 additions / 2,728 deletions across 75 files)
**Method:** Automated slop-scan (type ignores, `Any`, broad exceptions, deep nesting, long functions, commented code, `print` in library code) + manual deep-dive of API and lifecycle layers.

---

## Summary

PR #38 introduces a lot of new indirection in the API and lifecycle layer. The core pattern is **premature decomposition**: logic that previously lived in `api/main.py` has been split into 5 new service/lifecycle modules, a Protocol abstraction, and a SQL repository — but the boundaries don't create real separation. Most new "service" modules have only one caller (`main.py`), and the `LifecycleStore` Protocol adds 14 methods of interface weight for what is essentially a dict wrapper plus DataFrame storage.

---

## P1 — Structural Slop

### 1. `LifecycleStore` Protocol is premature abstraction

**What changed:** `LifecycleStore` went from a concrete in-memory class to a 14-method Protocol, with `MemoryLifecycleStore` as one implementation and `SqlLifecycleStore` as another.

**Why it's slop:**
- There are only two implementations, and they are almost identical in semantics. A Protocol with 14 methods is heavy for this.
- `MemoryLifecycleStore` delegates `new_fit_id()` and `new_study_id()` to `LifecycleStore.new_fit_id()` — the Protocol itself — which is circular and weird.
- The `__getattr__` lazy import for `SqlLifecycleStore` in `calibre/api/lifecycle.py` is a code smell. It means circular imports weren't resolved; they were hidden.

**What to do:** If you need both memory and SQL, use an abstract base class or just duck-type. The Protocol isn't buying you anything here — the memory store is the default, SQL is only used when `database_url()` returns something.

---

### 2. `FitRecord` ref indirection adds complexity without clarity

**What changed:** `FitRecord` lost direct DataFrame fields (`history`, `future_x`, `last_forecast`, etc.) and gained string refs (`history_ref`, `future_x_ref`, etc.) plus a separate `put_fit_frame`/`get_fit_frame` API.

**Why it's slop:**
- `_FIT_FRAME_REF_FIELDS` is defined *after* `__all__` in `calibre/api/lifecycle.py`.
- Every caller now does `store.get_fit_frame(fit_id, "history")` instead of `record.history`. That's two lookups instead of one, and the string literal `"history"` is not type-safe (you have `FitFrameKind`, but callers still pass strings in some places).
- The `_fit_frame_ref()` URI scheme (`lifecycle://fits/{fit_id}/frames/{kind}`) is fake — it doesn't resolve like a real URI. It's just a key.

**What to do:** Either keep DataFrames on the record directly, or use a real storage abstraction where the record doesn't expose refs at all. The half-measure of string refs on a dataclass is the worst of both worlds.

---

### 3. Service layer decomposition creates files without boundaries

**New files added:**
- `calibre/api/observe_service.py` (131 LOC) — only caller: `main.py`
- `calibre/api/tune_service.py` (161 LOC) — only caller: `main.py`
- `calibre/api/order_service.py` (49 LOC) — only caller: `main.py`
- `calibre/api/fit_service.py` (25 LOC) — **pure re-export** of `calibre.execution.model_lifecycle`
- `calibre/execution/model_lifecycle.py` (199 LOC) — only caller: `main.py` (via `fit_service.py`)

**Why it's slop:**
- These aren't services. They're functions that `main.py` calls. Splitting them into separate files didn't create reusable boundaries — it just added import overhead and indirection.
- `fit_service.py` is especially egregious: 25 lines to re-export 6 names from another module.
- `observe_service.py` duplicates `_frame_from_records()`, which also exists in `main.py` (line 185). Same logic, same datetime handling.

**What to do:** Keep `observe`, `tune`, and `order` logic in `main.py` until they actually need to be called from multiple places. If `main.py` is too big, split by *capability* (e.g., a real `tuning` module that the CLI also uses), not by HTTP endpoint.

---

## P2 — Code Quality Slop

### 4. `SqlLifecycleStore` is heavy boilerplate

**File:** `calibre/storage/lifecycle_repo.py` (395 LOC)

**Problems:**
- `_fit_from_row()` and `_tune_from_row()` manually map every field (20+ lines each). This is exactly what SQLAlchemy's `dataclass` mapping or `__init__` kwargs avoid.
- `_set_fit_field()` and `_set_tune_field()` are long `if key == "..."` chains. Using `setattr` with string keys bypasses type safety.
- `_expect_mapping()` and `_expect_iterable()` are one-line validators that add noise.
- `_records_from_frame()` serializes DataFrames by hand with `strftime` and `map(lambda ...)`. This is fragile and duplicates any standard serialization you might already have.

**What to do:** Use SQLAlchemy's native dataclass support or `Mapper` with `__init__` to avoid manual from/to mapping. The `update_fit(**fields)` API is inherently unsafe — use typed `replace()` on the dataclass instead.

---

### 5. Global mutable singleton factories

**New in `api/main.py`:**
- `_lifecycle_store()` — switches between memory/SQL via env var, manages global singleton
- `_model_artifact_cache()` — manages global singleton
- `_run_store()` — already existed, but the pattern is repeated

**Why it's slop:**
- These aren't thread-safe. FastAPI runs handlers in a thread pool. Multiple requests can race through the `if _LIFECYCLE_STORE is None:` check.
- The `_LIFECYCLE_STORE_KEY` tuple comparison is a hack to handle reconfiguration.
- Env-var based switching (`LIFECYCLE_STORE`) makes testing harder because it hides dependency injection.

**What to do:** Pass stores into the app at startup. If you need env-based config, do it once in `lifespan` and attach to `app.state`.

---

### 6. Broad `except Exception` in API layer

**Locations:**
- `api/main.py` line 385: `_run_fit_job`
- `api/main.py` line 438: `predict()`
- `api/tune_service.py` line 156: `run_tune_job()`

**Why it's slop:**
- These catch `Exception` and convert to "failed" status or 500s. This swallows programming errors (AttributeError, NameError) that should crash loudly.
- `tune_service.py` even imports `traceback` locally inside the except block.

**What to do:** Catch only the exceptions you expect (ValueError, KeyError, etc.). Let unexpected errors propagate so they appear in logs as real tracebacks.

---

### 7. `AdapterResolver = Callable[[dict[str, Any]], Any]`

**File:** `calibre/execution/model_lifecycle.py`

This is a type alias that uses `Any` twice. The input is a model config dict (okay), but the return type is `Any` when it should be the adapter base type (`ForecastingAdapter` or similar). It makes the type system useless for the module's core dependency.

---

## P3 — Minor / Quick Wins

| Issue | Location | Fix |
|-------|----------|-----|
| `fit_service.py` is pure re-export | `calibre/api/fit_service.py` | Delete it; import from `execution.model_lifecycle` directly |
| Duplicated `_frame_from_records` | `api/main.py` + `observe_service.py` | Share one helper |
| `__getattr__` lazy import | `api/lifecycle.py` | Fix imports at module level; remove `__getattr__` |
| `_FIT_FRAME_REF_FIELDS` after `__all__` | `api/lifecycle.py` | Move constants above `__all__` |
| `candidate_to_payload` does `dict(dict(x))` | `api/tune_service.py` | Unwrap if already dict-like; avoid double wrapping |
| `load_existing_tuning_run` does `dict(candidate.get(...))` | `api/tune_service.py` | Same issue — redundant `dict()` calls |

---

## Root Cause

The PR treats "separation of concerns" as "separation of files." The lifecycle code got split horizontally (API vs execution vs storage) without any of the new modules being independently reusable. The `LifecycleStore` Protocol and `FitRecord` ref system were added to support SQL persistence, but the abstraction leaked all the way up to the HTTP layer. A simpler approach: keep `FitRecord` as a plain dataclass with DataFrames, and let the SQL repo serialize/deserialize at the boundary without changing the record shape or adding a Protocol.

---

## Priority Recommendations

- **P0:** Revert `FitRecord` to hold DataFrames directly. Remove `_fit_frame_ref()` indirection.
- **P0:** Delete `fit_service.py`. Merge `observe_service.py`, `tune_service.py`, `order_service.py` back into `main.py` or into real capability modules that the CLI also uses.
- **P1:** Replace `LifecycleStore` Protocol with a simple abstract base class or just duck-type between memory and SQL.
- **P1:** Fix the `except Exception` blocks in background tasks to catch specific exceptions.
- **P2:** Simplify `SqlLifecycleStore` using SQLAlchemy dataclass mapping; remove `_set_fit_field()` chains.
