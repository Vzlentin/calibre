---
title: "Engine core — one engine, two drivers"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 03 — Engine core

This chapter owns the runtime contract of the single engine that serves both
backtesting and live inference. It fixes the pipeline spine, the two drivers
that run over one code path, the state that crosses driver boundaries, the
determinism requirements, and the settle hook. All vocabulary is chapter 02's,
used verbatim; invariants carry stable tags (`[ENG-*]`, `[SPN-*]`, `[DRV-*]`,
`[STA-*]`, `[DET-*]`, `[SET-*]`) for citation by other chapters and tests.

## State the one-engine contract

There is exactly one engine. A backtest is not a separate evaluation harness
that approximates production behavior; it is the production engine driven by a
replayed clock.

- `[ENG-1]` **One code path.** Every forecasting, reconciliation, calibration,
  ordering, settlement, and ledger computation lives in the engine. Backtest
  and live inference invoke the same engine functions; neither carries a
  private variant of any stage.
- `[ENG-2]` **Drivers sequence, the engine computes.** A driver contains no
  domain math: it may only construct engine inputs, invoke engine verbs in a
  legal order, and marshal outputs. Any logic that changes a forecast, an
  interval, an order, or a booked cost belongs to the engine.
- `[ENG-3]` **Orchestration is I/O-free.** The engine core sequences phases
  over abstract ports — panel source, actuals source, artifact store,
  calibration-state store, ledger sink, dispatch backend. All filesystem,
  object-store, and database access lives behind those ports in adapters. The
  engine core must be exercisable end-to-end with in-memory port
  implementations only.
- `[ENG-4]` **No-op composability.** An unconfigured stage is the identity:
  no reconciler configured ⇒ Reconcile returns its input unchanged; no
  conformal method ⇒ Calibrate is a pass-through; no ordering policy ⇒ Order
  emits nothing. Disabling one stage never changes another stage's behavior.

## Choose the substrates

- **Forecasting substrate: the Nixtla library family** (statistical, ML, and
  neural forecasters), reached exclusively through the forecasting-plugin
  protocol of chapter 04. The engine never imports a model library directly;
  model artifacts persist through each plugin's native persistence API
  (`[STA-4]`).
- **Distribution substrate: Ray**, reached exclusively through the dispatch
  port. Distribution is an execution detail, never a semantic one: results are
  identical whether work runs in-process or fanned out (`[DET-3]`, `[DET-4]`).
  The dispatch port decides *where* a unit of work runs; it is forbidden from
  influencing *what* the unit computes.

## Specify the pipeline spine

At run level the spine is: **load** (a dataset adapter yields a panel, plus
optional hierarchy facts, censoring facts, and a cost structure) → **task
build** (forecast tasks constructed per origin, enforcing `[TSK-2]` at
construction) → the **per-origin decision cycle** below, repeated over the
origin sequence → the **ledger** as the single scoring surface.

For each origin, the engine runs a fixed phase cycle:

1. **Resolve** — realize the *observe* verb: resolve due pending ledger rows
   `[LED-1]`–`[LED-3]` against actuals admissible at this origin
   (`[INV-TEMPORAL]`: strictly before it), updating calibration state
   `[CAL-3]`. Resolve runs *before* Predict so this origin's intervals reflect
   every observation admissible at it.
2. **Predict** — obtain fitted models and emit point forecasts as forecast
   frame rows (with the fitted-values sidecar `[FRA-5]` when a downstream
   stage requires it). A fitted model is obtained either by fitting on the
   task's history or by loading a persisted model artifact whose training
   window is admissible at this origin; the fit cadence (refit per origin vs.
   fit once and reuse) is configuration, and either choice must satisfy
   `[FRA-4]`.
3. **Reconcile** — rewrite point forecasts over the aggregation lattice
   (chapter 07 owns the protocol). Identity without hierarchy facts.
4. **Calibrate** — turn point forecasts into interval/quantile columns
   `[FRA-2]` from calibration state (chapter 05 owns the method protocol).
5. **Order** — at decision origins (`[SET-7]`), apply the ordering policy to
   the calibrated forecast, the inventory position, and the cost structure,
   emitting orders `[ORD-1]`–`[ORD-3]` (chapter 08 owns the policy protocol).
6. **Commit** — validate `[FRA-3]`, append this origin's rows to the ledger as
   pending `[LED-1]`, and persist calibration state — exactly once per origin,
   only here. Resolution belongs to Resolve alone: rows that become due after
   this origin resolve at a later origin's Resolve phase (`[SET-5]` forbids a
   second resolution path).

Spine rules:

- `[SPN-1]` The phase order is fixed and identical in both drivers.
- `[SPN-2]` Each phase is a function of its declared inputs and declared state
  only; no phase communicates with another through hidden channels.
- `[SPN-3]` A phase failure propagates naming the phase and the origin;
  per-phase timing is observable even on the failure path.
- `[SPN-4]` Mutation-then-persistence of calibration state happens exactly
  once per origin, at Commit. Intermediate phases may mutate in-memory state
  but never persist.
- `[SPN-5]` A committed origin is durable: re-running the same session skips
  origins already committed and continues from the first uncommitted one.

## Specify the two drivers

- **Time-loop driver (backtest).** Replays a historical panel as an ordered
  sequence of origins. Actuals are revealed progressively — each origin sees
  only the strictly-before-origin slice — so a replay is
  information-equivalent to having run live over the same period.
- **Event driver (live inference).** Engine verbs arrive as external events
  through the service surface (chapter 11): fit, predict, calibrate, order,
  observe. Each event names its session; origins arrive in real time and
  actuals may arrive late or out of order.

Driver rules:

- `[DRV-1]` **Observational equivalence.** Given the same session-defining
  inputs and the same sequence of resolved actuals, the two drivers produce
  identical ledger rows and identical orders. `[SES-3]` is the identity half
  of this rule; `[DRV-1]` is the behavioral half.
- `[DRV-2]` **A closed verb surface.** The engine exposes exactly the verbs
  fit, predict, reconcile, calibrate, order, observe, settle, commit; the
  per-origin cycle is their canonical composition, and both drivers (and every
  API route) compose only these verbs. No driver-only or API-only logic.
- `[DRV-3]` **Out-of-order tolerance.** The event driver must accept late and
  out-of-order actuals via pending-observation buffering (chapter 06 owns that
  contract); the time-loop driver is its in-order special case, not a separate
  mechanism.

## Specify the state crossing driver boundaries

Three state classes must survive process exit and cross between drivers:
**calibration state**, **open orders**, and **model artifacts** (ledger rows
are durable too, but the ledger is its own contract in chapter 02; storage
topology is chapter 12's).

- `[STA-1]` Every cross-boundary fact is durable and keyed by session (plus
  partition or series key as its term requires); at any phase boundary, no
  engine fact exists only in process memory.
- `[STA-2]` Calibration state restorability is unconditional: *every*
  conformal method's state round-trips `[CAL-2]`, so a session warmed by the
  time-loop driver is continuable by the event driver — and vice versa — with
  no state translation and no method-family exceptions.
- `[STA-3]` The **open-order** set (chapter 02) is derivable from the ledger's
  order rows and settlement records alone `[ORD-2]`, `[ORD-3]`; the engine
  never keeps it as free process state.
- `[STA-4]` Model artifacts are engine-owned: written through the forecasting
  plugin's native persistence API, addressed by deterministic engine-computed
  keys, and never supplied by callers as bytes or arbitrary URIs.

## Require determinism

- `[DET-1]` Session identity is deterministic per `[SES-1]`; the engine never
  mints random identity for anything a caller must re-address.
- `[DET-2]` **Reproducible task ordering.** The set of forecast tasks for an
  origin, and their ordering, is a pure function of (panel, configuration,
  origin). Dispatch grouping — batching, chunking, config grouping — is
  derived deterministically from that ordering; duplicate tasks (same series
  key, model configuration, horizon) collapse deterministically to one.
- `[DET-3]` **Batch-placement invariance.** A task's forecast is invariant to
  dispatch placement: which batch or chunk it lands in, batch/chunk size,
  worker count, and local vs. distributed backend. For a *local* task the
  forecast is a function of that task alone — co-batched series must not
  influence it. For a *global* task the function is of the whole panel and
  configuration `[TSK-1]`, still never of dispatch grouping.
- `[DET-4]` **Schedule-order independence.** Any parallel execution schedule
  commits origins in the serial reference order and yields a ledger
  byte-identical to serial execution. State-mutating steps for origin *n+1*
  never run before origin *n* commits; only state-independent work may
  overlap.
- `[DET-5]` **Explicit thread budgets.** Numeric kernels run under a
  configuration-derived thread budget so floating-point results do not vary
  with concurrency level or with serial-vs-parallel dispatch.
- `[DET-6]` **Seeded randomness.** Every stochastic component draws from an
  RNG seeded deterministically from configuration. Two runs with identical
  inputs on the same platform are bit-identical. Bit-identity across hardware
  or numeric-library stacks is *not* promised; protocol chapters (20, 21) own
  cross-platform tolerances.
- `[DET-7]` **Resume determinism.** A run interrupted after origin *k* and
  resumed produces remaining ledger rows and orders byte-identical to the
  uninterrupted run.

## Contract the settle hook

The settle hook is the engine's rolling decision loop over inventory time. Two
chapter 02 configuration facts drive it: the **lead time** `L` and the
**review period** `R`. At each decision origin (`[SET-7]`) the engine
forecasts, orders, and settles realized demand — one loop, both drivers. The
decision terms — open order, inventory position, settlement record, stock-out
transition rule, protection window — are chapter 02 vocabulary; this section
owns only their runtime contract.

- `[SET-1]` **Arrival law.** An order committed at origin `t` becomes
  available to serve demand at `t` advanced `L` periods on the series'
  calendar — never earlier, never later. Between commit and arrival it is an
  open order `[STA-3]` and counts in inventory position.
- `[SET-2]` **Settlement step.** For each (series key, period), settlement
  books one settlement record (chapter 02): it credits arrivals due that
  period; consumes realized demand against on-hand stock; applies the
  configured stock-out transition rule to unmet demand — configuration, not
  engine code; and updates the inventory position (chapter 02), which is what
  an ordering policy reads at decision time.
- `[SET-3]` **Costs booked exactly once.** Each (series key, period) books its
  holding and shortage cost exactly once, at that period's settlement, priced
  from the cost structure `[CST-3]`. No path — resume `[DET-7]`, replay,
  re-observation — may book the same (series key, period) twice, and an
  interruption before booking must book it on resume. Realized cost totals
  are pure sums of settlement records.
- `[SET-4]` **Drain.** After the final decision origin the loop runs `L`
  additional zero-order settlement periods so every committed order arrives
  and settles. Realized demand for the full drain window must be available;
  if it is not, the run fails at construction — silently settling missing
  periods at zero demand understates shortage cost and is forbidden.
- `[SET-5]` **Single observation.** Each resolved (series key, origin, horizon
  step, model name) row feeds calibration state exactly once `[CAL-3]`.
  Exactly one component owns the observe verb for a session at any time; a
  configuration that would double-observe (e.g. two resolution paths on one
  session) is rejected at construction.
- `[SET-6]` **Driver symmetry.** The time-loop driver realizes settlement by
  inventory simulation over revealed history; the event driver realizes it
  from observed actuals and inventory snapshots arriving as events. Policy
  inputs, the arrival law, and cost accounting are identical in both.
- `[SET-7]` **Decision cadence.** Decision origins are a function of the
  review period: with review period `R`, every `R`-th origin of the run's
  origin sequence — starting at the first — is a **decision origin** and runs
  Order; the origins between are non-decision origins, which resolve,
  predict, reconcile, and calibrate but emit no orders. The default is
  `R = 1`: every origin decides. The cadence comes from the configured review
  period, never inferred from data.

## Acceptance criteria

A conforming implementation must demonstrate, by test:

1. No-op composition: disabling reconciliation, calibration, or ordering
   leaves every other stage's output byte-identical `[ENG-4]`.
2. Two-driver equivalence: a panel replayed by the time-loop driver and the
   same origins fed as events through the verb surface produce identical
   ledger rows and orders `[DRV-1]`, `[DRV-2]` — the `[VIS-9]`
   one-engine-two-drivers commitment.
3. Resume: kill after origin *k*, resume, and the remaining ledger and total
   booked cost equal the uninterrupted run's — no re-booked (series key,
   period) `[DET-7]`, `[SET-3]`, `[SPN-5]`.
4. Batch-placement invariance: permuting series order, varying chunk/batch
   size, and switching local vs. distributed dispatch leaves every task's
   forecast rows identical `[DET-2]`, `[DET-3]`.
5. Serial/parallel identity: parallel dispatch of the same run yields a
   byte-identical ledger to serial execution `[DET-4]`, `[DET-5]`.
6. Boundary crossing: a session warmed by a backtest and continued by the
   event driver behaves identically to the uninterrupted session, for every
   registered conformal method `[STA-2]`, `[SES-3]`.
7. Arrival-law property test: an order committed at `t` first affects on-hand
   at exactly `t + L` periods, and each period's holding/shortage cost appears
   exactly once in the settlement records `[SET-1]`, `[SET-3]`.
8. Drain guard: a run whose realized-demand history cannot cover the `L`-period
   drain window fails at construction with a diagnostic naming the shortfall
   `[SET-4]`.
9. Port isolation: the full per-origin cycle runs under in-memory port
   implementations with no filesystem, network, or database access `[ENG-3]`.
10. Substrate audit: a dependency audit of the engine core finds Ray as the
    sole distributed-execution dependency, reached only through the dispatch
    port, and every bundled forecasting plugin implemented against Nixtla
    interfaces — the `[VIS-1]` Nixtla/Ray-core commitment.

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the old engine: `calibre/execution/backend.py::BackendEngine` (the fixed
per-origin phase cycle ResolveOpen → Predict → Reconcile → Calibrate → Order →
Commit; resolve-before-predict; persist-exactly-once at Commit; resume by
skipping completed origins; parallel dispatch draining head-first to stay
byte-identical with serial; chunking guaranteed never to change per-series
math; shared thread budgets to keep serial/parallel floats equal),
`calibre/execution/decision_loop.py` + `calibre/ordering/simulation/` (the
rolling forecast → order → settle → observe loop, lead-time drain rounds,
cumulative cost owned by the simulator and recorded once),
`calibre/cli/commands.py` (drain-window history guard; the single-observer
rule avoiding double-observed residuals). Negative space: `BackendEngine`
conflates orchestration with I/O — staging URIs, parquet materialization, Ray
handles, and threadpool control sit inline in the orchestrator — which
`[ENG-3]` separates; the settle loop lives in the CLI layer rather than the
engine, which the settle-hook contract moves inside; and only one conformal
runtime family is resume-restorable, which `[STA-2]` makes unconditional.
