---
title: "Forecasting plugins — the model-adapter protocol"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 04 — Forecasting plugins

This chapter owns one contract: the **model adapter** — the concrete form of a
forecasting plugin (chapter 02). An adapter is the engine's only conduit to a
backing model library. Everything a model needs arrives in a forecast task;
everything a model produces leaves as forecast-frame rows or the fitted-values
sidecar. The chapter specifies the adapter protocol, registration and
discovery, the local-vs-global scope axis, artifact persistence, the
fitted-values side channel, and the censoring-aware fit input contract. All
vocabulary is chapter 02's, used without redefinition. No gated decisions are
consumed; this chapter carries no seam hooks.

## Define the adapter protocol

An adapter is constructed from a **model configuration** and exposes four
operations: `fit(task)`, `predict(task)`, `fitted_values(task)`, and the
artifact pair `dump_state() / load_state(bytes)`.

**Inputs.** The forecast task supplies everything: a history (a panel slice,
strictly pre-origin `[TSK-2]`), a horizon `H`, the model configuration,
optional future exogenous regressors `[TSK-3]`, and optional censoring facts
`[PAN-3]`. Exogenous regressor columns are numeric `[PAN-4]`.

**Outputs.** `predict` emits point forecasts in forecast-frame form: exactly
one row per `(series key, horizon step)` for every series in the task, horizon
steps `1..H`, point forecast typed float. When the model configuration
requests quantile levels, the adapter additionally emits one numeric column
per level per `[FRA-2]` (**native quantiles** — produced by the model itself,
e.g. per-quantile loss objectives, as distinct from intervals added later by
calibration). A model configuration requesting native quantile columns in a
run with an active hierarchy is rejected at validation time; the rejection
rule and its gating live in chapter 07 (`[REC-24]`). `fitted_values` emits
the sidecar of `[FRA-5]` (see "Emit the fitted-values side channel").

Protocol invariants:

- `[ADA-1]` **Task-closed.** `fit` and `predict` consume only the task and the
  model configuration — no ambient I/O, no engine state, no data source other
  than the task's fields. This is what makes tasks the unit of distribution.
- `[ADA-2]` **Deterministic under seed.** Given the same task and the same
  model configuration (including any seed parameter), an adapter produces
  identical outputs. Stochastic libraries must expose their seed as a
  configuration parameter.
- `[ADA-3]` **Output normalization is the adapter's job.** Whatever shape the
  backing library returns, the adapter converts it to the forecast-frame
  column contract before returning; the engine never contains
  per-library shape knowledge.
- `[ADA-4]` **Quantile/point consistency.** When native quantiles are emitted,
  the point forecast is the 0.5-level column if requested, otherwise the
  requested level closest to 0.5. A frame never carries a point forecast
  inconsistent with its own quantile columns.
- `[ADA-5]` **Declared capabilities, loud failures.** Fitted values, native
  quantiles, censoring-aware fit, incremental update (an *update* verb that
  extends fitted state by one period, distinct from refit — required of this
  protocol by chapter 30, `[PRF-12]`; the engine uses it when declared and
  falls back to refit otherwise), and artifact persistence are optional
  *declared* capabilities. An adapter asked for a capability it does not
  declare fails with a capability error; it never silently degrades (no empty
  sidecar, no point-only frame when quantiles were requested).
- `[ADA-6]` **Lifecycle ordering.** `predict`, `fitted_values`, and
  `dump_state` before a successful `fit` (or `load_state`) are errors, not
  empty results.

## Register and discover adapters

A **registry** maps a *backend identifier* — a required string field of the
model configuration — to an adapter implementation.

- `[REG-1]` Every model configuration names its backend explicitly. An unknown
  backend is rejected with the list of available backends; there is no default
  backend.
- `[REG-2]` Registration is data and extensible: adapters register into the
  mapping (built-ins at import, third-party via an explicit registration hook
  or packaging entry point) without modifying engine code. Backend modules
  load lazily, so optional heavy dependencies cost nothing unless configured.
- `[REG-3]` **Capability validation precedes execution.** Resolving a model
  configuration checks every requested capability (`[ADA-5]`) against the
  backend's declarations and rejects mismatches at validation time (the
  `validate` verb, chapter 10) — before any data is loaded or any fit runs.
- `[REG-4]` The baseline adapter set wraps the Nixtla libraries (chapter 03's
  forecasting substrate): a classical-statistical family, a gradient-boosted
  ML family (dotted-path estimator selection, e.g. any scikit-learn-compatible
  regressor; native quantiles via per-level objectives), and a neural family.
  Each is one adapter class; model choice *within* a library is configuration.

Feature engineering splits along the same line: transforms the backing library
executes natively (lags, lag transforms, target transforms) are declared
inside the model configuration and resolved by the adapter; transforms that
produce exogenous regressor columns (calendar, scaling, static attributes) run
engine-side before task construction, so `[ADA-1]` holds — the adapter sees
finished columns, never a feature pipeline.

## Select scope by configuration, not adapter code

The **local-vs-global axis** is task scope `[TSK-1]`: *local* fits one adapter
instance per series; *global* fits one adapter instance over the whole panel.

- `[SCO-1]` Scope is a model-configuration field with values `local` |
  `global` and a documented default. It is data, never adapter code.
- `[SCO-2]` **Adapter code is scope-blind.** The same adapter class serves
  both scopes; the only difference it observes is whether the task's history
  contains one series or many. No adapter may branch on scope.
- `[SCO-3]` Scope is resolved exactly once, at task construction, yielding a
  partition of tasks (local set, global set). Downstream stages consume the
  partition; nothing re-derives scope.
- `[SCO-4]` **Fan-out is invisible in results.** Local tasks fan out across
  the distribution substrate (Ray, chapter 03). The executor may batch
  same-configuration local tasks into bounded chunks for staging efficiency,
  but per-series fits remain independent and outputs are row-identical to
  unbatched execution; batch identity never reaches forecast or ledger rows,
  which always carry the real series key.
- `[SCO-5]` A serialized task reference (per `[TSK-4]`) carries the history,
  future exogenous regressors, and censoring facts by storage reference, so a
  worker materializes the identical task on the far side of a process
  boundary.

## Persist artifacts via native APIs

A fitted adapter's state is an **artifact**: opaque bytes the engine stores
and transports without interpretation.

- `[ART-1]` Serialization is the adapter's job, implemented through the
  backing library's own save/load API (native persistence). The engine never
  generically pickles adapter objects — native APIs own cross-version model
  fidelity.
- `[ART-2]` The engine-facing contract is `dump_state() → bytes` /
  `load_state(bytes)`. Libraries that save to a directory are packed into a
  single deterministic archive; unpacking rejects absolute or parent-escaping
  member paths.
- `[ART-3]` **Round-trip fidelity.** For any fitted adapter,
  `load_state(dump_state())` yields an adapter whose `predict` output is
  identical for the same task — including quantile column identity, which
  travels inside the artifact if the library does not preserve it natively.
- `[ART-4]` **Exact-hit reuse only.** Fitted-state caching is keyed by a
  content hash over (history, horizon, model configuration); adapters may
  extend the key with adapter-specific state. Identical key → restore, else
  fresh fit. No warm starts, no partial reuse.
- `[ART-5]` Artifacts carry predictive state only: the fitted-values sidecar
  is *not* part of an artifact, and a consumer that needs fitted values must
  trigger a collecting fit (`[FIT-1]`) rather than restore from cache.
- `[ART-6]` Artifacts are server-owned: produced and consumed by the engine,
  addressed by engine-controlled storage references. Clients never supply
  model bytes or artifact locations (chapter 11).

## Emit the fitted-values side channel

**Fitted values** are in-sample historical predictions, delivered as the
sidecar of `[FRA-5]`: keyed by `(series key, timestamp, model name)`, never
written as forecast rows. Their consumers are residual-based reconciliation
strategies (chapter 07), which estimate residual second moments from in-sample
errors, and any conformal method whose manifest declares an in-sample
residual requirement (chapter 05, `[CNF-23]`).

- `[FIT-1]` Collection is **opt-in at fit time** via a fit flag, because it is
  materially more expensive than a plain fit. The engine sets the flag exactly
  when a configured downstream stage declares it requires fitted values; a fit
  without the flag owes no sidecar.
- `[FIT-2]` The sidecar validates against `[FRA-5]` before leaving the
  adapter: unique key, numeric actual and fitted columns, rows with a missing
  actual or fitted value dropped.
- `[FIT-3]` When the backing library produces fitted values at multiple
  horizon steps for one timestamp (direct multi-step strategies), the sidecar
  keeps the one-step-ahead value — one row per `(series key, timestamp, model
  name)`, always.
- `[FIT-4]` The sidecar travels alongside the predict output within the same
  origin pass (a prediction result is *forecast frame + optional sidecar*), so
  the reconcile stage receives residuals computed by exactly the model
  instance that produced the points. It is never persisted to the ledger.
- `[FIT-5]` Temporal hygiene is inherited, not re-checked: fitted values are
  functions of the task's history, which is strictly pre-origin `[TSK-2]`, so
  the sidecar is admissible at the origin `[INV-TEMPORAL]`.

## Accept censoring-aware fit as a first-class input contract

Recorded sales during out-of-stock periods bound demand from below; training
on them biases forecasts downward. The seam: **when the dataset provides
censoring facts `[PAN-3]` and the model configuration enables censoring-aware
fitting, an uncensored-demand column reaches model fitting** — the training
target, not the recorded value, is what the model sees for censored periods.

- `[CEN-1]` Censoring facts are a first-class task field: they ride the
  forecast task, survive serialization `[TSK-4]`, and reach the adapter
  wherever the fit runs. They are per-`(series key, timestamp)` availability
  status flags, joined to the history with absent flags meaning *available*,
  plus an optional numeric **availability bound** — the supply level that
  truncated the observation — riding the task exactly like the flags. The
  bound is optional per observation and present only where the dataset
  supplies it.
- `[CEN-2]` **No silent fallback.** Enabling censoring-aware fit on a dataset
  that declares no censoring source is rejected at validation `[REG-3]`; a fit
  invocation whose task carries no censoring facts while the flag is on is an
  error. The failure mode "quietly trained on censored sales" must be
  unreachable.
- `[CEN-3]` The uncensored target for a censored period is
  `max(recorded value, estimate)` — the recorded value is a lower bound on
  demand and is never reduced. The estimate derives only from the same
  series' *available*-period observations at or before that timestamp
  (default estimator: expanding median of available-period values; when no
  such history exists yet, the recorded value stands). Available periods keep
  their recorded value untouched.
- `[CEN-4]` Censoring-aware fitting alters the training target **only**. It
  must not smuggle availability flags or imputation artifacts into the
  exogenous regressor set — the regressor set is derived from the
  pre-imputation history, and any availability feature a user wants is
  declared explicitly like every other regressor.
- `[CEN-5]` Censoring-aware fit is a declared adapter capability `[ADA-5]`,
  checked per backend at validation — a registry-level special case
  hard-coding which backend supports it is non-conforming.
- `[CEN-6]` The predict path is unchanged: a censoring-aware model forecasts
  *demand*, while ledger actuals record what was observed. How any benchmark
  scores that distinction is owned by the protocol chapters (20, 21), not
  here.

## Conformance

A conforming implementation must be able to demonstrate, by test:

1. Resolving a model configuration with an unknown backend fails naming the
   available backends; a configuration requesting an undeclared capability
   fails at validation, before any fit `[REG-1]`, `[REG-3]`.
2. Flipping one series set between `scope: local` and `scope: global` changes
   no adapter code and yields schema-identical forecast frames `[SCO-1]`,
   `[SCO-2]`.
3. Batched and unbatched local fan-out produce row-identical forecast frames,
   with only true series keys in the output `[SCO-4]`.
4. `load_state(dump_state())` reproduces `predict` output exactly, including
   quantile columns, for every baseline adapter `[ART-3]`.
5. A fitted-values sidecar from any capable adapter passes `[FRA-5]`
   validation and contains one-step values only `[FIT-2]`, `[FIT-3]`.
6. With censoring-aware fit enabled: a task without censoring facts errors
   `[CEN-2]`; with facts, the training target equals the recorded value on
   available periods and is ≥ it on censored periods `[CEN-3]`; the regressor
   set is identical with the feature on and off `[CEN-4]`.
7. A quantile-emitting configuration produces `[FRA-2]` columns whose
   0.5-level (or nearest) column equals the point forecast `[ADA-4]`.

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the old engine: `calibre/forecasting/adapter_base.py` (fit/predict/
fitted-values/dump-load protocol, prediction result = forecast + optional
sidecar, exact-hit cache keying), `calibre/forecasting/adapter_registry.py`
(backend map, lazy import, `scope` as config), `calibre/forecasting/
native_persistence.py` (directory-save packing with unsafe-path rejection),
`calibre/forecasting/statsforecast_adapter.py` and `mlforecast_adapter.py`
(native save/load, per-quantile estimator naming with quantile identity
persisted beside the native artifact, point-from-median rule),
`calibre/forecasting/features/stockout_features.py` (lower-bound imputation,
regressors derived pre-imputation), `calibre/core/forecast_task.py`
(serialized task refs carrying censoring; chunked local staging with
per-series semantics preserved). Negative space: the old registry was a
hard-coded map with no third-party extension point (`[REG-2]` adds one) and
special-cased censoring support to a single named backend (`[CEN-5]` replaces
that with declared capabilities); capability absence surfaced as runtime
exceptions rather than validation-time rejection (`[REG-3]` moves it).
