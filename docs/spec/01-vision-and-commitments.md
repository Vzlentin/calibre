---
title: "Vision and commitments"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 01 — Vision and commitments

This chapter is the **commitments register**: it restates every product-vision
element as a testable architectural commitment with a stable tag (`[VIS-n]`),
a one-line acceptance criterion, and its owning chapter(s) per the chapter 00
vision-coverage matrix. The register owns no mechanics itself; its contract is
completeness — no vision element without an owner, no commitment without a
test. All domain terms (series, panel, forecast frame, forecast task, origin,
horizon, aggregation lattice, cost structure, order, session, calibration
state, partition, ledger) are chapter 02 vocabulary, used verbatim.

## Read the register discipline

- A **commitment** is a normative "the architecture must" statement that
  stands without reference to any prior engine.
- An **acceptance criterion** is one line a reviewer or CI job can execute;
  the owning chapter expands it into a full conformance section.
- The **owner** is the chapter whose contract realizes the commitment. This
  register never overrides an owner: on conflict, the owning chapter wins and
  this register is corrected.
- Spec review fails if any vision-coverage-matrix row lacks a `[VIS-n]` entry
  here, or any entry names an owner that does not cite the tag back. The
  reciprocal citation is a ratification-time obligation: a pre-gate draft may
  not yet cite its owning `[VIS-n]` tags, but no owner chapter reaches
  `ratified` without them.

## State the commitments

### `[VIS-1]` Build on a Nixtla/Ray core

**Commitment.** The forecasting substrate is the Nixtla library ecosystem and
the distribution substrate is Ray. The engine core carries no bespoke
forecasting algorithms and no second distributed-execution framework: model
fitting executes forecast tasks against Nixtla interfaces, and any fan-out
beyond one process runs on Ray.
**Acceptance.** A CI dependency audit of the engine core finds Ray as the sole
distributed-execution dependency and every bundled forecasting plugin
implemented against Nixtla interfaces.
**Owner.** Chapter 03.

### `[VIS-2]` Make forecasting models plugins

**Commitment.** A forecasting model is a plugin behind a registry. It
implements the chapter 04 model-adapter protocol — consume a forecast task
(history, horizon, optional exogenous regressors), emit forecast-frame rows
(point forecasts, optionally native quantile columns and a fitted-values
sidecar per `[FRA-5]`) — and is added, swapped, or removed without touching
engine core.
**Acceptance.** A new adapter reaches a runnable backtest via one new module
plus one registry entry, zero engine-core diff, and passes the shared adapter
conformance suite run against every registered adapter.
**Owner.** Chapter 04.

### `[VIS-3]` Make conformal methods plugins

**Commitment.** Conformal calibration methods coexist behind one stable
runtime interface with a registry. Each method maintains calibration state
keyed by `(session, partition)` `[CAL-1]` and declares its statistical
assumptions (e.g. exchangeability) at registration. Low-level statistical
building blocks may evolve freely; the runtime seam does not.
**Acceptance.** Every registered method passes the shared protocol suite —
state round-trip `[CAL-2]`, admissible-update rule `[CAL-3]`, observable
readiness `[CAL-4]` — without engine-core changes.
**Owner.** Chapter 05.

### `[VIS-4]` Drive ordering by cost, and make cost a tuning objective

**Commitment.** An ordering policy consumes a calibrated forecast, the
inventory position, and a cost structure and emits orders (`[ORD-1..3]`);
backtests realize
cost through inventory simulation, booked to the ledger as settlement
records. Realized cost is an optimization objective, not just a report: it is
computable per candidate inside a tuning loop, and the tuning layer binds to
the chapter 08 objective symbolically, never to a formula of its own.
**Acceptance.** A tuning study declares realized cost — a pure sum over the
ledger's settlement records — as its objective and ranks candidates by it
inside the study loop, with no post-hoc scoring step.
**Owner.** Chapters 08 (cost and policies), 09 (objective binding).

### `[VIS-5]` Run cloud-native on Kubernetes

**Commitment.** The engine deploys on Kubernetes as stateless API replicas
over a shared durable store and a shared artifact store, with Ray providing
in-cluster scaling. Every durable fact — run metadata, calibration state,
artifacts — lives in a store, never in process memory.
**Acceptance.** Killing any single replica mid-run and restarting loses no
durable fact: the run resumes and its ledger resolves identically
(restart-safety test).
**Owner.** Chapter 12.

### `[VIS-6]` Expose the full lifecycle over the API

**Commitment.** Fit, predict, calibrate, order, observe, session
introspection, backtest jobs, and tuning studies are all API verbs, and every
verb is a thin projection of the chapter 03 engine: no behavior exists only
behind an HTTP handler. Sessions addressed over the API follow deterministic
identity `[SES-1]`.
**Acceptance.** For every lifecycle verb, an API-driven call and a
driver-invoked call with the same defining inputs address the same session and
append identical ledger rows.
**Owner.** Chapter 11.

### `[VIS-7]` Make pipeline authoring easy, clean, and fast

**Commitment.** A full pipeline — dataset adapter, model configuration,
reconciler, conformal method, cost structure, ordering policy, tuning block —
is declared as data, maps 1:1 onto chapter 02 domain objects, validates before
execution (`validate` is a first-class verb), and ships sane defaults. A sweep
is a directory of configs; a tuning run is a config plus a search space.
**Acceptance.** A scripted onboarding test: a new user authors and validates a
runnable backtest without reading engine source.
**Owner.** Chapter 10.

### `[VIS-8]` Offer local and global modelling and tuning

**Commitment.** Local-versus-global is a configuration axis, never a code
axis: forecast-task scope `[TSK-1]` selects one adapter instance per series or
one full-panel adapter fanned out on Ray, and tuning studies declare per-series
or panel-level scope through the same study machinery.
**Acceptance.** Flipping a pipeline and its tuning study between local and
global scope is a config-only change, and both scopes pass the same protocol
tests.
**Owner.** Chapters 04 (modelling axis), 09 (tuning axis).

### `[VIS-9]` Serve backtesting and inference with one engine

**Commitment.** One code path, two drivers: a time-loop driver replaying a
sequence of origins for backtests, and an event driver fed by the API for live
inference, over the same stage spine. Session semantics are identical in both
`[SES-3]`, and temporal hygiene `[INV-TEMPORAL]` is enforced structurally in
both.
**Acceptance.** Driver-equivalence test: identical defining inputs and
identical actuals streams produce ledgers equal row-for-row across the two
drivers.
**Owner.** Chapter 03.

### `[VIS-10]` Recalibrate online

**Commitment.** The observe loop is a first-class runtime contract: actuals
resolve into the ledger `[LED-2]` and update calibration state only under the
admissible-update rule `[CAL-3]`; late and out-of-order actuals buffer as
pending observations `[LED-3]`; the contract is restart-safe and applies to
any conformal plugin satisfying chapter 05.
**Acceptance.** Replaying the same actuals with the same per-origin
availability and canonical delivery sequence, with restarts interleaved,
yields identical calibration state and identical resolved ledgers. Late or
out-of-order submissions never retroactively change committed issuance;
replaying the same arrival schedule resolves and delivers each eligible row
exactly once.
**Owner.** Chapter 06.

### `[VIS-11]` Reconcile hierarchies as a pipeline stage

**Commitment.** Hierarchical reconciliation is a dedicated pipeline stage
behind a reconciler protocol with a strategy registry. The aggregation lattice
is declared from hierarchy facts and static per run `[HIE-1]`, and the stage
remains feasible at retail scale (order of 30k bottom series), which mandates
a sparse summing-matrix representation as the default. Strategy choice is
configuration.
**Acceptance.** Swapping the reconciliation strategy is a config-only change
verified by a registry test, and the stage completes on a ~30k-bottom-series
lattice using the sparse representation within the memory budget chapter 30
will set.
**Owner.** Chapter 07.

## Bind the flagship metric

**BOUND — chapter 42 (40-gated-seams/).** The flagship is the two-axis claim
— a coverage certificate (gated) and a price ratio (tracked, not gated),
published together — owned by chapter 42 and bound to this register by
`[FLG-2]`. This register points to that binding and designates no other
number as flagship (`[ANNEX:01-flagship-metric-decision]`). Standing
consequences:

- protocol chapters 20/21 report their protocol-defined scoring surfaces in
  full, and the flagship figures are the only headline numbers;
- engine-specific regression constants remain internal tripwires, never
  product claims.

## Audit the register

A conforming spec tree demonstrates:

1. **Completeness** — every row of the chapter 00 vision-coverage matrix maps
   to exactly one `[VIS-n]` entry whose owners match the matrix.
2. **Reciprocity** — each owning chapter cites its `[VIS-n]` tag(s) in its own
   conformance section (whatever that chapter titles it — e.g. an
   acceptance-criteria section).
3. **Binding discipline** — this chapter carries no marked slot; its
   flagship binding names chapter 42 and no candidate number, and
   `[ANNEX:01-flagship-metric-decision]` remains listed in
   `90-annex-registry.md`.
4. **Public safety** — no acceptance criterion in this register depends on
   gated material.

| Tag | Vision element | Owner |
|---|---|---|
| `[VIS-1]` | Nixtla/Ray core | 03 |
| `[VIS-2]` | Pluggable forecasting models | 04 |
| `[VIS-3]` | Pluggable conformal methods | 05 |
| `[VIS-4]` | Cost-driven ordering; cost as first-class tuning objective | 08, 09 |
| `[VIS-5]` | Cloud-native, scales on K8S | 12 |
| `[VIS-6]` | Strong API | 11 |
| `[VIS-7]` | Easy, clean, fast pipeline authoring | 10 |
| `[VIS-8]` | Local and global modelling and tuning | 04, 09 |
| `[VIS-9]` | One engine for backtesting and inference | 03 |
| `[VIS-10]` | Online recalibration | 06 |
| `[VIS-11]` | Hierarchical reconciliation | 07 |

## Provenance

For spec authors only; the register above stands without these. Positive
space from the old engine: registry-based model adapters
(`calibre/forecasting/`), the stable conformal runtime seam
(`calibre/conformal/runtime.py`) whose lesson `[VIS-3]` generalizes, realized
cost via inventory simulation (`calibre/ordering/` + `simulation/`), Ray
Tune/Optuna studies (`calibre/tuning/`), and the Postgres state store
(`calibre/storage/`). Negative space: the old orchestrator
(`calibre/execution/backend.py`) conflated orchestration with I/O — `[VIS-9]`'s
one-engine/two-drivers contract replaces it rather than porting it; and the
old repo's benchmark regression suite hard-codes an engine-specific cost
tripwire — the flagship-metric binding (chapter 42) exists precisely so no
apparatus constant is ever promoted to a product claim.
