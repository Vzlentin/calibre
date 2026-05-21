# PR #35 — Deployment Readiness Assessment

Reviewing [PR #35 "Implement deployment lifecycle plan"](https://github.com/Vzlentin/calibre/pull/35)
against the question: *how would Calibre work when deployed in a client setting,
with Sales / Order / Inventory tables (and possibly events / promotions),
and do the seams support volume, training+inference endpoints, online
recalibration, and predict-then-optimize HPO?*

## Score by ask

| Ask | Coverage |
| --- | --- |
| Scale seams (many SKUs, local + global ensembles, HPO in the loop) | ~70% |
| Train + inference endpoints | ~85% |
| Conformal state + online recalibration | ~90% |
| HPO against cost (predict-then-optimize) | ~80% |

The PR materially advances every axis. The remaining gaps are concentrated
on the **data-plane edge**: ingestion, persistent state for fits/orders/
inventory, and eager training.

## 1. Scale seams — solid plumbing, thin data ingress

- `_process_global_panel` (`calibre/execution/backend.py:198`) fits one
  global adapter across the multi-SKU panel; the PR cites a 1.25× speedup
  on the VN2 two-config global-LightGBM run.
- Ray Tune fan-out via `_run_optuna_study` with `_OptunaSearchSpaceAdapter`,
  ASHA, and `tune.with_parameters(state_ref=…)` for shared conformal state
  across trials.
- `max_uid_concurrency` knob; per-trial CPU budget capping via
  `_cap_threaded_config` + `_trial_thread_env`.

**Gap:** sales and inventory still arrive as JSON in request bodies. For
real volume (thousands of SKUs × weekly) you need a parquet / SQL ingestion
seam. The `DatasetAdapter` / `InventoryAdapter` protocols are the right
shape but only `SyntheticInventoryAdapter` is implemented.

## 2. Train + inference endpoints — split is right, `/fit` is hollow

- `/fit`, `/predict`, `/calibrate`, `/order`, `/observe`,
  `/tune` + `/studies/{id}`, `/sessions/{tenant}/{uid}` — the lifecycle is
  cleanly carved.
- `derive_session_id(tenant, sku_set, forecaster_config, conformal_config)`
  is a real content-addressed identity; reruns coalesce.

**Gap that surprised me:** `_run_fit_job` (`calibre/api/main.py:267`) just
flips status flags — **no actual model fitting happens at `/fit` time.**
The fit runs lazily inside `/predict` (`_fit_predict_task(task)` at
`calibre/api/main.py:326`). For client deployment you almost certainly
want eager training so prediction latency is bounded and model artifacts
are cacheable. `ModelArtifactCache` exists but isn't wired into the
lifecycle.

**Gap:** `LifecycleStore` is an in-memory dict. `RunStore` has SQL,
conformal state has SQL, but fit / study records don't — they evaporate
on restart.

## 3. Conformal + online recalibration — the strongest part

- Per-`(session_id, partition)` state in `ConformalState`, with
  `SqlConformalStateStore.list_for_run` returning the full partition map.
- `SymmetricIntervalRuntime.get_partition_states` /
  `from_partition_states` lets you tear down and rehydrate per partition —
  important when SKUs partition the residual pool.
- `PendingObservationRepo` makes pending forecasts restart-safe;
  `DecisionLoop` has both in-memory and SQL paths.
- `/observe` background-merges actuals into `last_calibrated` and upserts
  the new state; `compact_old_state` covers retention.

**Watch-out:** `_run_observe_job` silently returns when `last_calibrated`
is empty (`calibre/api/main.py:396`) — easy to miss in operations. Worth
a metric or warning log.

## 4. Predict-then-optimize HPO — abstraction is right, one rough edge

- `TuningCandidate` unifies `model_config` + `conformal_config` +
  `ordering_config` — a real *joint* search space, not three siloed
  studies.
- `Cost`, `Pareto`, and the new `Regret` objective all evaluate the
  *post-order* cost, so HPO is genuinely optimizing the decision
  objective, not surrogate MAE / SMAPE.
- `Cost` validates `CONFORMAL_MODE` agreement with the frame — catches a
  real footgun where perhorizon `Cost` would silently double-count
  cumulative bounds.

**Gap:** `Regret` needs `oracle_cost` precomputed once before the study;
the PR adds `compute_regret` but no `/tune` plumbing to populate the
oracle. The path is sketched, not paved.

## Tables you mentioned vs. what landed

| Asked for | PR has | Status |
| --- | --- | --- |
| Sales | request JSON → `history` | Wire-only, no persisted table |
| Order | `OrderLedger` (memory), `last_orders` on `FitRecord` | No persistent order table |
| Inventory | `InventoryAdapter` protocol, `SyntheticInventoryAdapter` | Protocol-only; no SQL / parquet impl |
| Events calendar | — | Not addressed |
| Promotions | `future_x_override` on `/predict` for what-ifs | Inference-side only, no promo table |

## Recommended next steps before "deployed"

1. **Make `/fit` actually fit** and persist artifacts via
   `ModelArtifactCache`. Bounded `/predict` latency falls out for free.
2. **Back `LifecycleStore` with SQL** (mirror the `ConformalState`
   pattern) so sessions survive restarts.
3. **Ship a SQL-backed `InventoryAdapter`** and a thin `SalesAdapter` for
   the parquet / SQL ingestion path; keep `Synthetic*` for tests.
4. **Decide whether `tenant` needs auth enforcement** (currently it's an
   honor-system string in the session_id).
5. **Wire `Regret` end-to-end:** precompute oracle inside `/tune`, store
   on `TuneRecord`, surface in `/studies/{id}`.

## Bottom line

The bones of a deployable demand-planning service are in this PR. What's
missing is the data-plane edge — ingestion, persistent state for fits /
orders / inventory, and eager training. Small in code, but the difference
between *demoable* and *deployable*.
