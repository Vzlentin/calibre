# Calibre Post-Ray: Deployment Workflow Audit

## Revision history

- **2026-05-20 AM** — initial audit (this document).
- **2026-05-20 PM** — corrections folded in after diffing against an
  independent Hermes audit. Three substantive updates:
  (a) within-run kill-resume already works in `BackendEngine` (was wrongly
  listed as a gap citing a stale roadmap status); the real defect is the
  partition keying;
  (b) the `Cost` objective has two independent bugs separate from the
  Ray-Tune-with-conformal fallback — frame-level demand summation and
  last-origin-only Optuna reporting;
  (c) two new gaps surfaced (no SKU batching/grouping; no
  expiry/compaction on rolling calibrators).
  Affected sections: §3.1, §3.3, §3.4, P1, P2, §5.

## Context

PR #32 just landed: Ray Core handles per-uid local fan-out, Ray Tune handles
per-series HPO with `Accuracy / Cost / Pareto` objectives. Phase 0 of the
cloud-native roadmap is shipped (CLI, config, streaming ledger, `RunStore`,
`ForecastTaskRef`); Phase 1 (fsspec + DatasetAdapter Protocol) is still
open. Within-run conformal kill-resume is wired in `BackendEngine`
(`backend.py:243, 282, 496–509`) — the cloud-native-roadmap "Phase 0.3 still
pending" status is stale and should be struck. The seven Protocols and the
forecast-frame contract are stable.

This document is not an implementation plan. It's an audit of what a
**deployed Calibre** would actually look like against a real client — 
Sales + Orders + Inventory + Events + Promotions — and whether the seams that
landed with Ray (and the seams Phase 1–5 will land) cover the four concerns
raised:

1. Volume — many SKUs × ensemble of local+global models × HPO inside the
   forecasting workflow
2. Endpoints — train vs inference vs observe lifecycle
3. Conformal state + online recalibration as actuals arrive
4. Predict-then-optimize HPO against cost directly

Goal of this turn: identify the seams that hold, the seams that don't, and
the smallest changes that close the gaps. Implementation gets its own plan
after you decide which gaps are in scope.

---

## 1 · Client data model — what's in vs what's needed

| Client table | Current ingestion | Status |
|---|---|---|
| Sales | `DatasetBundle.history` long-panel `[unique_id, ds, y, *regressors]` | ✅ clean fit |
| Promotions | regressor columns in `history` / `future_x` | ✅ as features; ❌ no scenario / what-if surface |
| Events calendar | regressor columns in `future_x` (`add_calendar_features` for global ML) | ✅ |
| Costs | `DatasetBundle.costs: CostStruct \| dict[unique_id, CostStruct]` | ✅ scalar today; per-SKU loader planned in Phase 1.2 |
| Inventory (on-hand) | **not ingested in production path**; synthesised by simulator in backtest | ❌ missing adapter |
| Open orders (in-transit) | absorbed by `ProductState.pipeline: deque(maxlen=lead_time_depth)` in the simulator only | ❌ missing adapter |
| Lead times | `LostSalesRule(lead_time_depth)` — single scalar at engine init | ❌ no per-SKU table seam |
| Hierarchy / censoring | typed in `DatasetBundle` but optional | ✅ typed, ⚠️ engine does not yet consume hierarchy |

**The structural gap.** The current Protocols treat the world as
*historical demand panel → forecast frame → order quantity*. In a deployed
setting the engine has to *enter the loop mid-flight*: at every origin it
needs `(history-so-far, current on-hand, in-transit pipeline, per-SKU lead
time, per-SKU cost, prior conformal state)`. Today the simulator owns the
in-transit pipeline as a private dataclass; in production this state lives
in the client's ERP and has to be loaded by an adapter.

**Smallest fix.** Add a sibling to `DatasetAdapter` for runtime state:

```python
class InventoryAdapter(Protocol):
    def load_state(self, unique_id: str, at_origin: pd.Timestamp) -> ProductState: ...
    def load_lead_times(self) -> dict[str, int]: ...
```

`ProductState` already serialises cleanly (`copy()`, `to_dataframe()`); the
work is making the simulator accept an injected initial state per SKU
rather than constructing one synthetically. Backtests would pass a
`SyntheticInventoryAdapter` (today's behaviour); production passes
`ErpInventoryAdapter` reading from Postgres or a snapshot parquet.

---

## 2 · Deployed workflow — mapped to current code

Weekly cycle for a fashion retailer (one tenant, ~50K SKU-locations):

| Stage | Cadence | Current implementation | Gap |
|---|---|---|---|
| **Tune** — search model/conformal/order configs | Quarterly | `tuning.optimize_task` per series; Ray Tune unless conformal in loop | falls back to sequential when conformal_runtime_factory set (`optimizer.py:243–250`); search space is model-only |
| **Fit** — train models on latest history | Weekly | inlined inside `BackendEngine.execute`; refits every origin | no model artifact cache; `forecasting/cache.py` planned (Phase 3) not landed |
| **Predict** — produce H-step forecast at next origin | Weekly | `engine.iter_origins([task], ..., [next_origin])` | conflated with fit; no `/predict` endpoint distinct from `/forecasts` |
| **Calibrate** — apply prior conformal state | Weekly | `ConformalRuntime.apply()` after engine emits frame | within-run resume works (`backend.py:496–509`); cross-run resume blocked by run-scoped state key (see §3.3) |
| **Order** — combine forecast + inventory + costs | Weekly | `DecisionRule × OrderingArithmetic`; pure function | inventory_position comes from simulator object, not adapter |
| **Observe** — feed last week's actuals back | Weekly (lagged by lead time) | `decision_loop.observe_per_horizon / observe_cumulative` | runs in-process; pending-forecast buffer is `list[pd.DataFrame]` in memory; state keyed by `run_id`, not by stable session |

The center of gravity is `BackendEngine` running a closed walk-forward
loop. In production the same loop runs **across cron invocations**, so
every piece of state that lives inside the loop (`pending`, runtime,
`_frames`) needs an external home.

---

## 3 · Seam audit against the four concerns

### 3.1 Volume: many SKUs × ensemble × HPO

What holds:
- Per-SKU local fan-out is real: `BackendEngine._execute_origin` routes
  `local_refs` through Ray, `global_refs` direct. Chunked by
  `max_concurrency`, threshold-gated by `ray_threshold` (≥10).
- Ensembling on the frame is post-hoc and quantile-preserving
  (`ensemble.py`: `median / weighted / inverse_error`).
- HPO via Ray Tune + ASHA + OptunaSearch, per-series TuningTask.

What doesn't:
- **Multiple global models are serial.** Global tasks run on the driver
  (per the architecture: "multi-series models can't parallelize
  per-origin"). For 3–5 global ensemble members × N origins, the driver is
  the bottleneck. Fix: run *each global model config* in its own Ray task
  with the full panel — fan-out at the model-config level, not the
  unique_id level.
- **HPO of conformal/order params is impossible today.** `TuningTask.search_space`
  returns only a `model_config` dict. A unified search-space contract
  returning `{model, conformal, ordering}` is needed for true
  predict-then-optimize tuning (see §3.4).
- **Model artifact caching is missing.** Every origin refits from scratch
  inside Ray workers; HPO trials refit across all origins for each config.
  `ModelArtifactCache(uri)` keyed by `cache_key(task)` is sketched in the
  cloud-native roadmap (Phase 3) but not in code yet.
- **Flat task list — no SKU batching or grouping.** Tasks fan out
  one-per-unique_id. Ray absorbs the count, but there's no category /
  cluster level above the SKU. This forecloses warm-start sharing across
  similar SKUs, category-aware global models, and any ordering that wants
  to prioritise certain series first. Adding a `task_group: str` field on
  `ForecastTask` (default = uid) and a `BackendEngine` option to schedule
  by group would be the smallest seam.

### 3.2 Endpoints — train vs inference vs observe

Current surface (`calibre/api/main.py`):
- `POST /forecasts` — sync, ≤30 SKUs, runs `run_config` end-to-end
- `POST /backtests` — async, full run
- `GET /runs/{id}`, `/healthz`, `/metrics`

`/forecasts` and `/backtests` are both the *same execution path* with
different scope — there is no separation between fit, predict, calibrate,
and observe at the HTTP surface, and no `/tune` endpoint at all. For a
deployed engine this is the surface that has to exist:

```
POST /tune     {tenant, sku_set, search_space, objective}     → study_id, best_configs
POST /fit      {tenant, sku_set, model_configs}               → fit_handle (artifact URIs)
POST /predict  {tenant, fit_handle, origin}                   → forecast_frame
POST /calibrate{tenant, forecast_frame, session_id}           → calibrated_frame
POST /order    {tenant, calibrated_frame, inventory_snapshot, costs} → order_ledger
POST /observe  {tenant, session_id, actuals}                  → new conformal state
GET  /sessions/{tenant}/{uid}                                 → state + last forecast + open orders
```

`session_id` is the persistent handle across weekly invocations
(§3.3). This split also matches the operational cadence (tune quarterly,
fit weekly, predict weekly, observe per actuals arrival) — the cron jobs
hit different endpoints with different rate budgets.

### 3.3 Conformal state + online recalibration

What holds:
- `get_state` / `set_state` on `RollingQuantileCalibrator`,
  `FixedAlphaController`, `AdaptiveAlphaController` (`calibrators.py`,
  `controllers.py`).
- `SymmetricIntervalRuntime.from_state` rehydrates issued_count + calibrator +
  controller (`runtime.py`).
- `SqlConformalStateStore` persists `(run_id, partition) → JSONB`
  (`storage/state.py:11–28`).
- `decision_loop.observe_per_horizon / observe_cumulative` encodes the
  correct dispatch rule from `lessons.md §40`.

What doesn't:
- **State is one fat blob per run, not per-partition.** The store key is
  `(run_id, partition)` but in practice `partition` is hard-coded to a
  single string `RUNTIME_PARTITION = "__runtime__"` (`storage/state.py:8`,
  `backend.py:500, 509`). All partitions get serialised into one JSON blob
  per run. No per-partition row updates, no compaction, full read/write
  cycle on every origin. The store *schema* allows per-partition rows; the
  *usage* doesn't exercise it. Fix: thread the per-(uid, model, horizon)
  partition into `upsert` calls so the store actually uses its
  multi-partition shape.
- **State key is `run_id`, not a stable cross-run session.** Within-run
  kill-resume is wired (`backend.py:496–509`: `from_state` on start,
  `upsert` after every origin). But every weekly cron is a new run, and
  nothing links last week's `run_id` to this week's. Fix: introduce
  `series_session_id = hash(tenant, sku_set, model_config,
  conformal_config)` and have the store accept `session_id` as the primary
  key alongside `partition`; `run_id` becomes a transient audit pointer
  inside the `runs` table.
- **No state expiry or compaction.** Rolling calibrators bound memory via
  `deque(maxlen=...)`, but there's no archival policy for the JSON
  payloads in Postgres, no snapshot rotation, no merge of stale partitions.
  Fix: per-partition TTL on `conformal_state` rows + scheduled compaction
  that drops partitions inactive past a threshold.
- **Pending forecasts are in-process.** `decision_loop.py` holds
  `pending: list[pd.DataFrame]` of un-resolved (uid, origin, h) rows.
  Between cron invocations this needs to land in Postgres or parquet,
  keyed by `session_id`.
- **No coverage drift alert.** `controllers.AdaptiveAlphaController` keeps
  `alpha_history` and `error_history` — the right surface to derive a
  "running coverage − target" metric. Phase 3 wires
  `calibre_conformal_coverage_ratio` as a gauge but no per-controller drift
  signal yet.

### 3.4 Predict-then-optimize HPO against cost directly

What holds:
- `Cost(decision_rule, arithmetic, costs)` is a concrete `TuningObjective`
  that runs the order policy + simulator inside each trial and returns
  realised inventory cost (`tuning/objectives.py`).
- `Pareto(decision_rule_fn, arithmetic, costs, lambda_grid)` sweeps λ and
  reduces to a scalar (hypervolume / min_regret) so Optuna receives one
  number.

What doesn't:
- **Ray Tune fallback when conformal is in the loop.** When
  `conformal_runtime_factory` is set and `ray_local_mode=False`, the
  optimiser falls back to sequential Optuna with a `RuntimeWarning`
  (`optimizer.py:243–250`). Cost-objective HPO is only meaningful with
  conformal in the loop (cost depends on interval width), so in practice
  every cost-objective tune runs serially today. Fix: snapshot conformal
  state to object store before each trial, hydrate via `from_state` inside
  the trial. JSON state is sub-10KB per partition; the overhead is small
  next to fitting.
- **`Cost.evaluate` collapses the whole frame into one decision.**
  `objectives.py:60`: `demand = float(actuals.dropna().sum())`. If the
  frame has H horizon rows, all H actuals get summed into a single demand
  scalar, paired with a single `order_qty` from one `decision_rule` call,
  for one over/under-age computation. This is defensible for cumulative
  mode (one order serves the protection-period total) but wrong for
  per-horizon mode (H independent decisions each Friday). The objective
  doesn't dispatch on `conformal_mode`, and the semantics aren't
  documented. Fix: in per-horizon mode, evaluate cost per horizon and
  return the sum; in cumulative mode, keep current semantics but assert
  the frame is a single (uid, origin) cumulative window.
- **HPO objective returns the *last* origin's value, not the sum.**
  Sequential path (`optimizer.py:212–220`): `value` is overwritten in
  every iteration of `engine.iter_origins(...)`; the final returned scalar
  is whatever the last origin produced. Ray Tune path
  (`optimizer.py:300–310`): `tune.report({metric: value, origin_idx: ...})`
  fires per origin; ASHA uses early reports for pruning, but Optuna's
  trial score is the final iteration's value. For an inventory-cost
  objective, you want cumulative cost over the backtest window. Fix:
  accumulate `total_cost += value` inside the loop, report
  `total_cost / origin_idx` (running mean) as the intermediate metric for
  ASHA to compare apples-to-apples across origins, and emit the final
  `total_cost` as the trial's objective.
- **Search space is model-only.** A real predict-then-optimize tune varies
  `{model, conformal_window, conformal_alpha, decision_rule, λ}`
  jointly. Fix: change `TuningTask.search_space` from
  `Callable[[Trial], dict[model_config]]` to
  `Callable[[Trial], TuningCandidate]` where `TuningCandidate` is a
  `(model_config, conformal_config, ordering_config)` triple.
- **Regret is not a `TuningObjective`.** `eval/regret.py` computes
  `cost − cost_oracle` post-hoc but isn't wired into HPO. Cost is noisier
  than regret across configs because both share the same demand
  realisation; switching the objective is one wrapper.
- **Single static `CostStruct` per trial.** When costs are per-SKU (Phase
  1.2 `load_costs(uri)`), `Cost.evaluate` needs to fan the cost vector
  over the frame rather than averaging — minor change but easy to forget.

---

## 4 · Recommendations, priority-ordered

The first three are the load-bearing fixes; the rest follow.

### P1 · Fix the predict-then-optimize HPO path end-to-end

Three independent defects, all in `calibre/tuning/`. None of these can ship
alone — together they make cost-objective HPO actually meaningful.

Files: `calibre/tuning/optimizer.py:212–220, 243–310`,
`calibre/tuning/objectives.py:51–63`, `calibre/storage/state.py`.

- **(a) Unblock Ray-Tune-with-conformal.** Snapshot conformal state to a
  per-trial URI before launching Ray Tune; inside `_trainable` call
  `SymmetricIntervalRuntime.from_state` to hydrate from that URI. Remove
  the sequential-Optuna fallback at `optimizer.py:243–250`.
- **(b) Accumulate the objective across origins.** In both
  `_evaluate_candidate` (sequential) and `_trainable` (Ray Tune),
  accumulate `total_cost += value` across the `iter_origins` loop. Report
  `total_cost / origin_idx` (running mean) as the per-iteration metric so
  ASHA prunes consistently across configs; return `total_cost` as the
  trial's final objective. Today's last-origin behaviour silently biases
  the search toward configs whose tail happens to land cheap.
- **(c) Make `Cost.evaluate` mode-aware.** Add a `mode: Literal["perhorizon",
  "cumulative"]` field on `Cost`. In `perhorizon` mode, group the frame
  by `(forecast_origin, h)`, run the decision-rule + arithmetic per group,
  and sum the per-group over/under-age. In `cumulative` mode, assert the
  frame is a single (uid, origin) window and keep current semantics.
  Document the dispatch in `objectives.py` and surface a clear error when
  `Cost` is paired with a `decision_rule` whose mode disagrees with the
  frame's `conformal_mode` column.

These three together turn predict-then-optimize from "structurally there,
silently wrong, single-threaded" into a usable production loop.

### P2 · Per-partition state rows + stable session key

Files: `calibre/storage/state.py`, `calibre/storage/models.py`,
`calibre/execution/backend.py:496–509`, `calibre/conformal/runtime.py`.

The current state path *works within a run* — `BackendEngine` already
hydrates via `from_state` on start and `upsert`s after every origin
(`backend.py:243, 282, 496–509`). Two defects remain:

- **Single-blob keying.** The `partition` argument is hard-coded to
  `RUNTIME_PARTITION = "__runtime__"` (`storage/state.py:8`,
  `backend.py:500, 509`), so every origin reads and writes one fat JSON
  blob containing every partition's calibrator + controller state. Fix:
  thread the real per-(uid, model, horizon) partition string through to
  `upsert`/`get`. The `(run_id, partition)` schema already exists; just
  use it.
- **Run-scoped, not session-scoped.** Every weekly cron is a new
  `run_id`. Introduce `session_id: UUID` derived from `hash(tenant,
  sku_set, model_config, conformal_config)` and add it as a primary-key
  column on `conformal_state`. `run_id` moves to the `runs` table as an
  audit pointer. Existing within-run resume still works because `session_id`
  is stable across runs that share the config tuple.
- **Pending forecasts buffer.** Persist `decision_loop.pending` as a
  sibling table `pending_observations(session_id, uid, origin, h, lo, hi,
  y_hat)`. Observed rows are deleted on `observe`. This is what enables
  cross-cron-invocation lifecycle.
- **Compaction.** Add a TTL column on `conformal_state` rows; a
  scheduled job drops rows untouched past a threshold (e.g. 90 days for
  weekly forecasts). Prevents unbounded growth on long-lived deployments.

### P3 · InventoryAdapter + dead-code cleanup

Files: `calibre/execution/dataset.py` (new `InventoryAdapter` Protocol),
`calibre/simulation/`, `calibre/conformal/__init__.py`.

- Add `InventoryAdapter.load_state / load_lead_times`. Backtests get a
  `SyntheticInventoryAdapter` (today's behaviour); production gets
  `ErpInventoryAdapter` or `SnapshotInventoryAdapter` reading from
  Postgres or a parquet snapshot. `ProductState` already serialises
  cleanly, so the simulator just needs to accept an injected initial
  state per SKU instead of constructing one synthetically.
- Strike the stale "Phase 0.3 still pending" entry from the cloud-native
  roadmap status header. Within-run kill-resume is wired (see P2). The
  remaining state work is the keying refactor in P2, not a new resume
  path.
- Wire (or remove) the dead `deserialize_calibration_state` symbol at
  `calibre/conformal/__init__.py:29,63` — either fold it into the P2 store
  refactor or delete it.

### P4 · Lifecycle endpoints

Files: `calibre/api/main.py`, `calibre/api/schemas.py`,
`calibre/cli/commands.py`.

- Split `POST /forecasts` (today: backtest-shaped) into:
  `/tune`, `/fit`, `/predict`, `/calibrate`, `/order`, `/observe`.
- Each accepts `session_id` (or returns one on first call).
- `/predict` and `/order` are sync; `/tune` and `/fit` are async via the
  existing `RunStore`.
- Reuse existing `cli/commands.py` plumbing — the engine APIs already
  separate fit/predict at the adapter level; the HTTP layer is the
  re-shape.

### P5 · Unified `TuningCandidate` search-space

Files: `calibre/tuning/task.py`, `calibre/tuning/optimizer.py`.

- Change `search_space` return type to a `TuningCandidate(model_config,
  conformal_config, ordering_config)` dataclass.
- Tune over conformal window + α target + decision rule + λ jointly with
  model hyperparameters.
- Add `Regret(decision_rule, arithmetic, costs)` as a sibling
  `TuningObjective` to `Cost` and `Pareto`.

### P6 · Global model fan-out

Files: `calibre/execution/backend.py`.

- Today `global_refs` run on the driver in a loop. Wrap each global task
  in its own `@ray.remote` so 3–5 global ensemble members fan out
  per-origin. Per-uid local fan-out stays unchanged.

### P7 · Model artifact cache + coverage drift metric

Files: `calibre/forecasting/cache.py` (new),
`calibre/core/metrics.py` (Phase 3 stub).

- `ModelArtifactCache(uri)` keyed by `adapter.cache_key(task)`.
- Conservative initial behaviour: cache hits only on identical history
  hash + config. Helps HPO trials more than weekly forecasts.
- Add `calibre_conformal_coverage_drift{model, partition}` gauge derived
  from `AdaptiveAlphaController.error_history` running mean − target.

---

## 5 · Verification — how to know the audit is right

This audit is text, but the assertions are testable:

| Claim | Verification |
|---|---|
| Ray Tune falls back to sequential with conformal in loop | `uv run pytest -k tune_conformal_fallback` — write a test that asserts the `RuntimeWarning` fires today and stops firing after P1(a). |
| HPO objective returns last-origin value, not sum | Write a test that runs `optimize_task` with a `Cost` objective over 5 fabricated origins where origin 5's cost is known-low and origins 1–4 known-high; today's optimiser picks the low-cost-on-tail config. After P1(b), it picks the cumulative winner. |
| `Cost.evaluate` ignores horizon dimension | Fabricate a frame with H=4 horizons; call `Cost.evaluate(frame, frame[Y])`; today's result equals `cost(order_qty=DR(frame), demand=Σ y[0..3])`. After P1(c) in per-horizon mode, result equals `Σ_h cost(order_qty_h, demand_h)`. |
| State is one fat blob per run | `grep -n 'RUNTIME_PARTITION' calibre/execution/backend.py` shows the single-string partition; `SELECT partition, length(state::text) FROM conformal_state GROUP BY partition;` returns one row per run. After P2, multiple partition rows per run. |
| Run-scoped, not session-scoped | Today: run a config twice → two `run_id`s, two state blobs, second run starts from scratch. After P2: same `session_id` → second run hydrates from first run's last state. |
| Pending forecasts are in-memory | Read `decision_loop.py:135–185`; after P2 add a `PendingStore` Protocol and migrate `list[pd.DataFrame]` to it. |
| Global models serial on driver | `grep -n 'global_refs' calibre/execution/backend.py`; after P6, `@ray.remote` decorator visible on global-task dispatch. |
| Inventory adapter missing | `grep -rn 'InventoryAdapter' calibre/` returns nothing today; after P3 the Protocol exists with a `Synthetic` and `Erp` (or `Snapshot`) implementation, exercised by the VN2 backtest and a new fixture test. |
| Within-run kill-resume already works | `grep -nE 'from_state\|state_store\.(get\|upsert)' calibre/execution/backend.py` shows the wiring at `backend.py:243, 282, 496–509`. Kill VN2 at origin 3, restart with the same `run_id`, get byte-identical final ledger — this should already pass today. |

End-to-end smoke after P1–P3:
```
uv run pytest tests/test_state_resume.py tests/integration/test_ray_tune_conformal.py tests/test_cost_objective_aggregation.py
uv run calibre run --config benchmarks/vn2/config/winning.yaml
```

---

## 6 · What this audit is not

- Not a phase plan. The cloud-native roadmap (Phases 0–8) already owns
  packaging + persistence + multi-tenancy. P1–P7 above are seam-level
  fixes that are orthogonal to those phases and unblock client-shaped
  workflows the roadmap assumes will work.
- Not a vision change. The seven Protocols + forecast-frame contract +
  walk-forward semantics hold. The deployed shape just requires their
  state edges to be externalised.
- Not a Ray re-evaluation. The Ray migration was the right call; the
  remaining limitations (global-on-driver, Tune fallback) are
  composition-level fixes inside the same backend.

---

## Related

- `architecture` (vault) — seven Protocols + forecast-frame contract
- `vision` (vault) — Oventi white-box + SaaS positioning
- `cloud-native-strategy` (vault) — Track 2 audit (2026-05-13)
- `2026-05-17-cloud-native-roadmap` (vault) — canonical Phase 0–8 plan
- `2026-05-18-stack-decision-and-ray-migration` (vault) — Ray rationale + PR #32
- `PLAN.md` (repo root) — executable phased plan derived from this audit
