---
title: "Domain model — the ubiquitous language"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 02 — Domain model

This chapter is the vocabulary contract for the entire spec: every term is
defined exactly once, with its invariants, and later chapters use these terms
verbatim without redefinition (a chapter needing a new term adds it *here*).
Invariants carry stable tags (e.g. `[FRA-2]`) so other chapters and the test
suite can cite them. Two invariants are cross-cutting and get their own
sections: **temporal hygiene** and **coherence**.

## Define the data terms

### Series and panel

A **series** is a single univariate demand time series: an ordered set of
**observations** `(timestamp, value)` identified by a **series key** — an
opaque, non-empty string unique within a run. A **panel** is a set of series
in long format: one row per `(series key, timestamp)` carrying the observed
value, plus optional per-row metadata.

- `[SER-1]` A series key is stable for the life of a run and across sessions
  referencing the same data.
- `[SER-2]` Each series declares a calendar frequency; its timestamps lie on
  that calendar in strictly increasing order.
- `[PAN-1]` Within a panel, `(series key, timestamp)` is unique.
- `[PAN-2]` Observed values are numeric; an unobserved period is an absent
  row or an explicit missing value — never a fabricated zero.
- `[PAN-3]` A panel may carry **censoring facts**: per-`(series key,
  timestamp)` metadata marking periods where the recorded value is bounded
  by availability rather than demand (e.g. out-of-stock). A censoring fact
  is a **status** — the declared vocabulary is two-valued, *censored* /
  *uncensored*, with *undeclared* as the recorded default — plus an optional
  numeric **availability bound**: the observed supply level that truncated
  the observation. The bound is a companion field, never a third status; it
  may be carried per observation, including uncensored ones, wherever the
  dataset supplies it. Censoring facts are optional metadata; their use is
  owned by the consuming chapter.
- `[PAN-4]` Columns beyond the key, timestamp, value, and declared metadata
  are **exogenous regressors** and must be numeric.

### Forecast frame

The **forecast frame** is the canonical long-format table through which every
pipeline stage communicates. One row asserts: *for this series, from this
origin, at this horizon step, this model predicted this.* Required, typed
columns: series key (string), target timestamp, actual value (float; missing
while unresolved), point forecast (float), horizon step (integer), origin
(timestamp), model name (string).

- `[FRA-1]` One row per `(series key, origin, horizon step, model name)`; the
  target timestamp is derived: origin advanced `h − 1` periods on the series'
  calendar, so step 1 targets the origin period itself. Admissible data
  remains strictly pre-origin (`[INV-TEMPORAL]`); step 1 is a nowcast of the
  decision period, not a past observation.
- `[FRA-2]` Interval forecasts appear as numeric column *pairs* (lower, upper)
  suffixed by nominal coverage level — a frame may carry several levels at
  once; quantile forecasts as numeric columns suffixed by quantile level.
- `[FRA-3]` A frame either passes schema validation (columns present, types
  exact) or is rejected before any stage consumes it. There is no partially
  valid frame.
- `[FRA-4]` Every forecast value in a row is a function only of information
  admissible at that row's origin (see temporal hygiene).
- `[FRA-5]` In-sample **fitted values** (historical predictions consumed by
  residual-based reconcilers) are a separate sidecar table keyed by
  `(series key, timestamp, model name)`, never written as forecast rows.

### Forecast task

A **forecast task** is the unit of fit-and-predict work: a history (a panel
slice), a horizon, and a **model configuration** — the declarative parameters
selecting and parameterizing a forecasting plugin, whose stable label in
frame rows is the **model name**. A task may carry future exogenous
regressors and censoring facts. Its invariants:

- `[TSK-1]` Task **scope** is configuration, not model code: a *local* task
  covers exactly one series; a *global* task covers the whole panel with one
  model instance.
- `[TSK-2]` Every timestamp in a task's history is strictly before the task's
  origin (temporal hygiene applied at construction, not left to the model).
- `[TSK-3]` Future exogenous regressors may cover target timestamps only with
  values legitimately known at the origin (calendar facts, planned prices).
- `[TSK-4]` A task is immutable once built and serializable across process
  boundaries; materializing a serialized task reproduces it exactly.

### Origin and horizon

An **origin** is a timestamp at which the pipeline issues forecasts and makes
decisions. A backtest is a sequence of origins replayed in order; live
inference is a stream of origins arriving in real time — same definitions.
The **horizon** `H` is the number of periods the decision spans from the
origin forward; the **horizon step** `h ∈ {1, …, H}` indexes those periods:
step `h` targets the origin's timestamp advanced `h − 1` periods, so step 1
is the origin period itself and step `H` is `H − 1` periods after it. Every
step is forecast from data strictly before the origin (`[TSK-2]`).

## Temporal hygiene invariant

`[INV-TEMPORAL]` **An origin never sees data at or after itself.** The
information admissible at origin `o` is exactly: observations with timestamp
strictly before `o`, plus facts known in advance of `o` (calendar structure,
planned exogenous values, hierarchy facts, configuration). An observation
stamped `o` itself is *not* admissible at `o`.

This binds every stage: task histories `[TSK-2]`, forecast values `[FRA-4]`,
calibration-state updates `[CAL-3]`, and order decisions. It is enforced
structurally — by construction of the inputs each stage receives, not by
per-model discipline — and holds identically in backtesting and live
inference.

## Define the hierarchy terms

### Hierarchy node and aggregation lattice

**Hierarchy facts** are per-series attribute assignments: each bottom series
carries a value for each column of a fixed attribute set (e.g. product
category, location). A **hierarchy node** is either a *bottom node* — one
series — or an *aggregate node* — the set of all bottom series sharing one
attribute value, plus a single *total node* containing every bottom series.
The **aggregation lattice** is the full set of nodes induced by the hierarchy
facts. Every node is itself addressable as a series (its values formed under
the coherence invariant), so frames and ledgers carry aggregate rows
uniformly.

- `[HIE-1]` The lattice is static per run: declared from hierarchy facts
  before execution, never mutated by it. Each aggregate's member set is a
  pure function of the hierarchy facts.
- `[HIE-2]` Every bottom series has a non-missing value for every attribute
  column; membership is total and per-node expected member counts are fixed,
  run-constant facts.
- `[HIE-3]` Node identity is label-based and collision-free: a bottom label
  is its series key; an aggregate label encodes its attribute column and
  value; the total node carries a single reserved label denoting the
  all-series total. No two node labels collide — in particular, neither the
  total label nor any aggregate label collides with any series key.

### Coherence invariant

`[INV-COHERENCE]` **An aggregate equals the sum of its members when all
members are observed.** For an aggregate node `A` and timestamp `t`: if every
member's value at `t` is observed, the aggregate's value at `t` is defined
and equals the members' sum; if *any* member is unobserved, it is undefined —
never zero, never a partial sum. This all-members-present rule governs
aggregate history construction and aggregate actuals resolution, and every
component that computes aggregates must agree with it row-for-row.

Stated here structurally for *observed* quantities only. For forecast
quantities the question is settled: reconciled point forecasts satisfy
additivity as chapter 07's own contract (`[REC-12]`), and no non-point
forecast quantity carries any additivity requirement (chapter 41,
`[SEAM-3]`).

## Define the decision terms

### Cost structure

A **cost structure** is a first-class configuration value carrying the
economic parameters of a decision problem: **underage cost** and **overage
cost** (per unit short / over, for a single decision period) and **holding
cost** and **shortage cost** (per unit per period, realized by inventory
simulation over time).

- `[CST-1]` All four components are non-negative floats.
- `[CST-2]` The **critical ratio** `underage / (underage + overage)` is
  defined only when the denominator is positive; a consumer that needs the
  ratio but cannot form it must reject the cost structure.
- `[CST-3]` A cost structure is data, not code: declared in configuration,
  attached per dataset or per series, consumed by ordering policies, tuning
  objectives, and conformal methods declaring cost-coupled configuration
  (chapters 08, 09, 05) without reinterpretation.
- `[CST-4]` The **cost-pair mapping** — how the per-decision pair (underage,
  overage) and the per-period pair (holding, shortage) relate under a lead
  time `L` and review period `R` — is owned by chapter 08. This chapter
  defines both pairs; no other chapter derives one pair from the other.

### Lead time, review period, and protection window

The **lead time** `L` is the whole number of periods, on the series'
calendar, between an order's commit origin and its **arrival** — the first
period from which the ordered quantity can serve demand. The **review
period** `R` is the decision cadence: order decisions are taken every `R`
periods. Both are configuration facts. The **protection window** of a
decision is the span of `L + R` periods starting at its origin (horizon steps
`1 … L + R`) — the horizon span whose demand sum the decision covers, because
the quantity committed now must carry the series until the next decision's
order can itself arrive.

Two **emission-scope** terms are canonical vocabulary, cited verbatim by any
chapter declaring what a forecast or calibrated bound targets: **per-step** —
the value of a single horizon step — and **window-sum (over the protection
window)** — the sum of demand over the protection window. A stage declares
which scope it emits; the two are never mixed silently.

### Order

An **order** is a decision fact: a quantity committed at an origin for a
series, produced by an ordering policy from a calibrated forecast, the
inventory position, and a cost structure.

- `[ORD-1]` An order quantity is non-negative.
- `[ORD-2]` Orders are keyed uniquely by `(session, series key, origin,
  model name)`.
- `[ORD-3]` A recorded order is immutable: decisions are historical facts.

### Open order, inventory position, and settlement record

An **open order** is an order whose arrival period has not yet been settled.
The **inventory position** of a series is, in its general form, on-hand stock
plus on-order quantity (the open orders) minus backorders; it is what an
ordering policy reads at decision time. The **stock-out transition rule** is
the configured rule mapping unmet demand to the next period's state (lost
sales, backorder, …): configuration, never engine code — it determines which
inventory-position components can be non-zero. A **settlement record** is the
durable per-`(session, series key, period)` fact booked when a period
settles: arrivals credited, realized demand consumed, the stock-out
transition applied, and the period's holding and shortage cost. Chapter 03
owns the settlement runtime contract (when settlement runs in the loop and
its exactly-once discipline); chapter 08 owns the policy protocol that
consumes inventory position and the interpretation of the booked costs.

## Define the runtime terms

### Session

A **session** is the durable identity tying one forecasting lifecycle
together across calls — fit, predict, calibrate, order, observe — and across
restarts. A **tenant** is a session's isolation namespace; all durable state
is tenant-scoped.

- `[SES-1]` Session identity is **deterministic**: a pure function of its
  defining inputs — tenant, series set, calendar frequency, horizon, model
  configuration, conformal configuration, and the decision-side
  configuration (ordering policy and cost structure). Identical inputs yield
  the same session, never random; changing any defining input mints a new
  session. This keeps decision facts immutable: orders are keyed by session
  `[ORD-2]` and immutable `[ORD-3]`, so no configuration change can rewrite
  an existing session's decisions.
- `[SES-2]` A session owns its calibration state and its orders: two sessions
  never share mutable state.
- `[SES-3]` A backtest and a live deployment with the same defining inputs
  address the same session semantics — this is what makes "one engine, two
  drivers" (chapter 03) coherent.

### Calibration state

**Calibration state** is the mutable state a conformal method maintains to
turn point forecasts into calibrated intervals or quantiles, keyed by
`(session, partition)`. A **partition** is an equivalence class of
forecast-frame rows under a configured partition key — e.g. all rows, one
series, one `(series, horizon step)` — so state granularity is
configuration, not code.

- `[CAL-1]` State is keyed by `(session, partition)` and nothing else; a
  state row is addressable and restorable independently of any other.
- `[CAL-2]` State round-trips: serialize → persist → restore yields behavior
  identical to the uninterrupted state (restart safety).
- `[CAL-3]` State is updated only by (a) resolution of actuals admissible
  under temporal hygiene and (b) issuance bookkeeping; it never embeds
  future observations.
- `[CAL-4]` Every method declares a **calibration requirement**: the minimum
  resolved-score condition a partition must satisfy before the method emits
  finite bounds. A method's **readiness** — whether a partition currently
  meets that requirement — is part of its state and must be externally
  observable (see `[LED-6]`, `[LED-7]`).

## Define the ledger

The **ledger** is the run's single durable record of decisions and outcomes,
and the single scoring surface: coverage, cost, and diagnostic metrics come
from the ledger and nowhere else. It carries three row families:

- **Forecast rows** — one per issued forecast-frame row, keyed `(series key,
  origin, horizon step, model name)` `[FRA-1]`, resolving against actuals per
  `[LED-1]`–`[LED-3]` below. Coverage and forecast diagnostics are computed
  from these rows.
- **Order rows** — one per order, keyed `(session, series key, origin,
  model name)` `[ORD-2]`, immutable `[ORD-3]`. Together with settlement
  records they make the open-order set derivable from the ledger alone.
- **Settlement records** — one per settled `(session, series key, period)`,
  booking that period's arrivals, demand consumption, stock-out transition,
  and holding and shortage cost. Realized cost is a pure sum over settlement
  records; chapter 03 owns the settlement runtime contract.

Write discipline: the ledger is append-only at issuance with **one-shot
monotone resolution** — no row is ever deleted; a forecast row mutates
exactly once, pending → resolved `[LED-2]`, never backward; order rows and
settlement records never mutate at all.

### Pending versus resolved rows

- `[LED-1]` A row enters the ledger **pending** at issue time: all forecast
  columns populated, the actual value missing.
- `[LED-2]` A row becomes **resolved** when its actual value is set to a
  finite number. Resolution happens at most once per row and never nulls or
  degrades a previously populated column — a resolved row is a full row.
- `[LED-3]` Rows whose target timestamp has passed but whose actual has not
  arrived remain pending (late/out-of-order actuals are normal; chapter 06
  owns the buffering contract).

### Scored-row predicate

Scoring is evaluated per nominal coverage level, using the level's bound
column pair `[FRA-2]`. Each resolved row is scored under the predicate its
guarantee descriptor type selects (`[LED-8]`); band-coverage scoring
requires interval columns: a resolved row carrying only quantile columns
and no interval pair at the level under evaluation is unscored at that
level, and its cause must be attributable per `[LED-7]` like any other
unscored mass:

- `[LED-4]` For band-coverage types (the `[LED-8]` two-sided predicate):
  **resolved** ⇔ the actual value is finite.
  **scored** ⇔ resolved AND both interval bounds are finite.
  **covered** ⇔ scored AND lower ≤ actual ≤ upper.
  **unscored** ⇔ resolved AND NOT scored.
- `[LED-8]` The ledger scores each resolved row under the predicate selected
  by the row's guarantee descriptor type `[GRT-2]`: at most one registered
  predicate per admissible `(claim, currency)` pair. One-sided coverage scores
  bound-exceedance indicator events; two-sided coverage scores
  band-containment events (the `[LED-4]` predicate); risk-control scores the
  realized declared-loss average against the claimed level;
  class-conditional coverage scores per-class indicator events; the
  long-run-pathwise and approximate-with-declared-slack currencies score as
  trajectories and against the declared slack respectively, not as
  finite-sample events. A row whose descriptor claim is none (not
  engine-calibrated) is never scored as calibration evidence; a row whose
  descriptor type has no registered predicate is unscored, its cause
  attributable per `[LED-7]`.
- `[LED-5]` Denominator discipline: coverage ratios use scored rows as the
  denominator; pending and unscored rows never enter it, and unscored counts
  are always reported alongside coverage so calibration gaps cannot
  masquerade as (mis)coverage.

### Warm-up and unscored attribution

- `[LED-6]` **warm-up** ⇔ an unscored row issued before its calibration
  partition met the method's calibration requirement: ordering the rows of
  the row's configured calibration partition `[CAL-1]` by origin, the
  resolved scores available prior to the row's issue origin do not satisfy
  the method's declared calibration requirement `[CAL-4]`.
- `[LED-7]` Every unscored row must be attributable to a cause from the
  ledger alone: the engine persists the calibrator's readiness/finiteness
  decision per issued row `[CAL-4]`; no unscored mass is ever "undetermined".

## Define the guarantee descriptor

The **guarantee descriptor** is the statement of claim a calibrated decision
bound carries: what kind of guarantee it asserts, at what level, scored
against which series, over which emission scope, at which decision scope.
It exists so that claims are declared, never inferred.

- `[GRT-1]` The descriptor is the tuple `{type, level, scored series,
  window, scope}`, attached to every calibrated decision bound at issuance.
  It states what claim the bound makes; no consumer ever infers one.
- `[GRT-2]` **type** is a pair `(claim, currency)`. Admissible claims:
  **one-sided coverage** (the default), **two-sided coverage** (diagnostic),
  **risk-control** (expected declared loss at most alpha),
  **class-conditional coverage**, and **none (not engine-calibrated)**.
  The claim `none` means *no engine claim is stated* — it covers bounds the
  engine never calibrated and bounds whose claim was voided (a clamp,
  chapter 41 `[SEAM-8]`; an explicit-fractile override, `[CFG-6]`; a
  reference-tuned bound, `[TUN-24]`); its currency field is not applicable.
  Admissible currencies: **finite-sample-marginal** (the default),
  **long-run-pathwise**, and **approximate-with-declared-slack** (the slack
  declared numerically). The vocabulary is closed: a claim or currency
  outside it is unregistrable.
- `[GRT-3]` **level** is the claimed level — for the default claim, the
  cost-derived fractile from the declared cost structure; **scored series**
  is the series the claim is scored against; **window** is the emission
  scope the claim attaches to — per-step or window-sum (over the protection
  window) — as declared in the chapter 05 manifest; **scope** is the
  decision scope (chapter 41 (40-gated-seams/) `[SEAM-5]`). For a
  class-conditional coverage claim, the scope field additionally names the
  finite-dimensional class system the claim conditions on (`[SEAM-6]`).
- `[GRT-4]` The descriptor is populated at issuance from the method manifest
  (chapter 05) and the run configuration; the normative force of the
  mandatory-carriage rule is chapter 41 (`[SEAM-1]`), and of the scope,
  `[SEAM-5]`. A mutation
  of a bound (e.g. a clamp) must rewrite the descriptor, never leave it
  stale.

## Structurally defined terms — normative force bound

Two terms are defined here *structurally*, so all chapters share their
shape; their normative semantics are bound by the chapters cited with each
term.

**Coherent cost** — a realized-cost quantity attached to hierarchy nodes.
Bound by chapter 08 and chapter 41 `[SEAM-4]`: the term denotes the
per-decision-node realized-cost family and its bookkeeping totals — sums of
realized cost over series are bookkeeping, not a lattice-level cost object;
no lattice-level aggregate cost functional exists.

**Hierarchical coverage** — a family of coverage statistics indexed by
hierarchy node and by lattice level. Bound by chapters 05/07 and chapter 41
`[SEAM-5]`/`[SEAM-6]`: the term denotes per-node and per-level diagnostic
statistics; no member of the family is a joint or simultaneous lattice-wide
claim, and the only admissible conditional claim form is class-conditional
coverage per `[SEAM-6]`.

## Conformance

A conforming implementation must be able to demonstrate, by test:

1. Frame validation rejects any frame missing a required column or carrying
   a mistyped required, interval, or quantile column `[FRA-3]`.
2. A property test shows no history timestamp ≥ origin ever reaches a model
   `[TSK-2]`, `[INV-TEMPORAL]`.
3. Aggregate construction yields undefined (not partial) values whenever any
   member is unobserved, exact member sums otherwise `[INV-COHERENCE]`.
4. The scored-row predicate is one shared per-type predicate registry
   consumed by every metric surface, never re-derived `[LED-4]`, `[LED-8]`.
5. Calibration state round-trips behavioral equality `[CAL-2]`; every
   unscored row in a completed run is attributed to a cause `[LED-7]`.
6. Session derivation is a pure function: same inputs, same identity, across
   processes `[SES-1]`.

## Provenance

For spec authors only; the chapter above stands without these. Positive
space from the old engine: `calibre/core/forecast_frame.py` (column contract,
interval/quantile columns, fitted-values sidecar),
`calibre/core/forecast_task.py` (task immutability, serialized refs,
local/global partition), `calibre/core/order_types.py` (cost structure),
`calibre/evaluation/forecast_metrics.py` (resolved/scored/covered masks),
`calibre/execution/actuals.py` + `calibre/reconciliation/summing.py`
(all-members-present rule, static hierarchy index). Negative space: the old
engine enforced `ds < origin` inside prediction
(`calibre/execution/prediction.py`) rather than at task construction —
`[TSK-2]` moves it; and its ledger did not persist the calibrator's readiness
decision, leaving part of the unscored mass unattributable — `[LED-7]` and
`[CAL-4]` close that gap.
