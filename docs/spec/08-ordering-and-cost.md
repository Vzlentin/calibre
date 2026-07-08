---
title: "Ordering and cost — policies, inventory simulation, realized-cost objective"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 08 — Ordering and cost

This chapter owns the decision layer: how a calibrated forecast, an inventory
state, and a cost structure become an order, how orders are replayed against
resolved actuals to realize cost, and the contract that realized cost is an
optimization objective — computable per candidate inside a tuning loop — not
just a report. It uses chapter 02 vocabulary verbatim (series key, origin,
horizon step `h`, forecast frame, cost structure, critical ratio, order,
ledger, hierarchy node) and chapter 07's placement of reconciliation.
Invariants carry stable tags (`[POL-*]`, `[SIM-*]`, `[OBJ-*]`, `[CFG-*]`).

## Position the decision layer

The decision layer runs after conformal calibration, consuming the calibrated
forecast frame. Three placement rules:

- `[POL-1]` **Decision origins.** Ordering runs at decision origins
  (`[SET-7]`, chapter 03). Under periodic review, consecutive decision
  origins for a series are one review period apart. One order per **decision
  group** — `(series key, origin, model name)` — per decision origin, keyed
  per `[ORD-2]` (chapter 02).
- `[POL-2]` **Decision nodes only.** Orders exist at decision nodes — the
  bottom nodes of the aggregation lattice, where orders are placed; ordering
  policies consume bottom-node rows exclusively. Aggregate nodes carry no
  orders and no cost objective. This decision-node-only scope is ruled, not
  interim (chapter 41 (40-gated-seams/), `[SEAM-4]`; the cost-scope binding
  below).
- `[POL-3]` **Additivity.** Enabling the decision layer changes only decision
  outputs. Point forecasts and calibration state are bit-identical with the
  layer on or off; with the layer off, no decision bound or order exists.

## Make the cost structure first-class configuration

The cost structure is defined in chapter 02 (`[CST-1]`–`[CST-3]`): underage,
overage, holding, and shortage costs, all non-negative, with the critical
ratio `underage / (underage + overage)` defined only for a positive
denominator. This chapter gives it normative force:

- `[CFG-1]` A cost structure is declared in configuration — per dataset or
  per series — before execution, and is never inferred from data. A consumer
  requiring one shared fractile across a panel of heterogeneous per-series
  cost structures must reject the panel, not average it.
- `[CFG-2]` **Fractile derivation.** For cost-driven policies the decision
  fractile is exactly the critical ratio; it derives exclusively from the
  cost structure. Supplying an explicit target quantile to a cost-driven
  policy is a configuration error, rejected before execution — the only
  sanctioned deviation is the `[CFG-6]` override field. A consumer requiring
  a fractile strictly inside `(0, 1)` must reject boundary values.
- `[CFG-3]` **Coverage sync.** The nominal coverage level a policy consumes
  and the level calibration produces are one configured fact, declared once.
  An omitted policy coverage inherits the calibration coverage; an explicit
  mismatch is rejected before execution. Explicit-quantile consumption is
  exempt (it names its column directly).
- `[CFG-4]` **Protection window coupling.** `lead time` and `review period`
  are declared together; the **protection period** `P = lead time + review
  period` is an exact integer identity, and the **protection window** is the
  horizon steps `1 ≤ h ≤ P` (inclusive at both ends). Validation before
  execution rejects: a task horizon `H < P`, and a window-bound policy with
  no configured bound source.
- `[CFG-5]` **Domain.** Coverage and quantile levels are valid strictly
  inside `(0, 1)`; boundary and outside values are rejected. The explicit
  quantile-column knob is invalid for policies whose target derives from the
  cost structure (newsvendor) or that gate on a reorder point; `[CFG-6]` is
  the one sanctioned exception, and it is a different field.
- `[CFG-6]` **Explicit-fractile override.** Exactly one sanctioned override
  field exists, for what-if studies: an explicit decision fractile that
  replaces the critical-ratio derivation of a cost-driven policy. It is
  distinct from the explicit quantile-column knob (which names a frame
  column; `[CFG-2]`/`[CFG-5]` govern that knob unchanged), off by default,
  named in configuration, and its binding is recorded on every decision it
  affects. The guarantee descriptor (chapter 02, `[GRT-*]`) of any result
  computed under the override carries the claim none (not engine-calibrated)
  (`[GRT-2]`); the override is configuration, never a search dimension in
  default tuning (chapter 09, `[TUN-7]`; the sole sanctioned relaxation is
  reference-tuning mode, `[TUN-24]`).

## Specify the order-policy protocol

An ordering policy is a pure function:

    policy(decision frame, inventory state, cost structure, parameters) -> order quantity

where the *decision frame* is the calibrated forecast-frame rows of one
decision group, and **inventory state** is: on-hand inventory, plus an
in-transit pipeline (a vector of quantities indexed by periods until
arrival). The **inventory position** `IP` this chapter consumes is on-hand
plus the pipeline sum — the lost-sales specialization of chapter 02's
general form, on-hand plus on-order minus backorders, with backorders
identically zero (`[SIM-2]`).

- `[POL-4]` **Purity.** A policy is deterministic, performs no I/O, and never
  mutates its inputs. Per-series parameters and decisions are independent: a
  decision for one series key never reads another series' state.
- `[POL-5]` **Order-up-to skeleton.** Every policy in this chapter computes a
  target `T` and emits `order = max(T − IP, 0)` — zero when `IP ≥ T`
  (inclusive), never negative (`[ORD-1]`, chapter 02).
- `[POL-6]` **Refuse, never degrade.** On insufficient or malformed input a
  policy raises; it never emits a zero, default, or NaN order. Refusal
  conditions (minimum set): protection period exceeding the available
  horizon; missing horizon steps inside the protection window; duplicate
  horizon steps within a decision group; absent interval columns for the
  requested nominal coverage level; absent quantile column when one is
  explicitly requested; missing decision-period row (newsvendor); any
  non-finite bound consumed by the target arithmetic (including a NaN
  terminal window bound). A policy exception during a backtest or replay
  aborts and propagates — it is never converted to a zero or default order,
  because silent zero-ordering corrupts the cost signal the run exists to
  measure.
- `[POL-13]` **Bounds consumed unmodified.** A policy consumes calibrated
  bounds exactly as issued. Beyond the arithmetic this chapter specifies,
  any cap, floor, or scale applied to a bound, to the target `T`, or to the
  order quantity is a named, off-by-default configuration act whose binding
  is recorded per affected decision — the chapter 05 no-silent-clamp rule
  (`[CNF-19]`–`[CNF-21]`) extended to the decision layer. A bound modified
  between calibration and decision loses the coverage claim of its guarantee
  descriptor (chapter 02, `[GRT-*]`); the descriptor must reflect the
  modification.

## Specify policy-family mechanics (per decision origin)

### Newsvendor

For a single decision period: the target is the critical-ratio fractile of
the calibrated demand distribution for that period.

- `[POL-7]` `CR = underage / (underage + overage)`; where the frame exposes
  only an interval column pair at nominal coverage level `c`, the sanctioned
  fractile estimator is linear interpolation, `T = lower + CR · (upper −
  lower)` on the decision-period row. If a richer calibrated distribution
  (dense quantiles) is available, the true CR-quantile replaces
  interpolation; the CR definition, the `[POL-5]` clamp, and monotonicity
  (`T` non-decreasing in underage cost) are invariant across estimators.

### Order-up-to (R,S)

The target stock level is a **fractile of the window sum**: an upper bound,
at the configured nominal coverage level, on cumulative demand over the
protection window, consumed as the order-up-to level. Two sanctioned sources:

- `[POL-8]` **Per-step summation.** `T = Σ_{h=1..P}` of the coverage-selected
  upper bound column (or of an explicitly requested quantile column) over
  the protection window. Every summand must be finite (`[POL-6]`); a
  non-finite bound inside the window is a refusal, never skipped.
- `[POL-9]` **Terminal window bound.** When calibration runs in cumulative
  mode, it emits a single bound on the window sum located exactly on the
  terminal row `h = P`; rows `h < P` carry no bound. The target is exactly
  that terminal bound — never a sum of per-step bounds. A present-but-NaN
  terminal bound means the window is uncalibrated: refusal (`[POL-6]`).
- `[POL-10]` **Mode discrimination.** Frames marked cumulative take the
  `[POL-9]` path; per-step or unmarked frames take `[POL-8]`; an explicit
  quantile request always sums the named quantile column regardless of mode.
  Conformance fixtures must make the paths distinguishable (window sum ≠
  terminal bound).
- `[POL-14]` **Non-engine bounds are marked.** A policy consuming a quantile
  or band not issued by the engine's calibration stage (an explicit quantile
  request under `[POL-8]`/`[POL-10]` may name such a column) records a
  guarantee descriptor whose claim is `none (not engine-calibrated)`
  (chapter 02, `[GRT-2]`; chapter 41, `[SEAM-1]` — claims are declared,
  never inferred; cf. `[SEAM-8]`); such a bound states no engine coverage
  claim, and scoring surfaces treat it accordingly.

### Gated order-up-to (R,s,S)

- `[POL-11]` Same target arithmetic as (R,S), plus a reorder gate: order is
  `0` if and only if `IP ≥ s` — **boundary inclusive**: equality means no
  order. This inclusivity is a pinned specification decision, not a derived
  fact. The gate may be declared as an absolute reorder point `s` or as a
  scale on the target (`s = scale · T`); both are inclusive.

### Integer order units

- `[POL-12]` The committed order the execution layer records is whole units:
  every requested series key receives `max(ceil(q), 0)` units, where `q` is
  the policy's real-valued quantity. Ceiling never under-orders a fractional
  need; negative quantities clamp to zero; a series key absent from the
  policy output receives `0`; an empty policy output yields all zeros.

## Specify the inventory simulator

The simulator realizes cost in backtests: it replays committed orders and
resolved actuals period by period, per series key, and emits an auditable
**simulation trace**. Per-series state: on-hand inventory, an in-transit
pipeline vector, and cumulative cost components. `[SIM-1]`–`[SIM-6]` are the
time-loop realization of the chapter 03 settle contract `[SET-1]`–`[SET-6]`,
with pipeline depth `d` equal to the lead time `L`.

- `[SIM-1]` **Event order within period `t`.** (1) Arrivals land:
  `start(t) = end(t−1) + arrivals(t)`. (2) Demand draws against available
  stock: `sales(t) = min(start(t), demand(t))`;
  `missed(t) = demand(t) − sales(t)`; `end(t) = start(t) − sales(t)`.
  (3) An order placed at `t` enters the pipeline and can never serve period
  `t` demand. (4) Costs accrue on the period's resulting attributes.
- `[SIM-2]` **Conservation (lost-sales configuration).** `start(t) −
  sales(t) = end(t)` and `sales(t) = min(start(t), demand(t))` hold in
  *every* period, not only terminally. Unmet demand is lost — never
  backordered, never carried: this simulator is the **lost-sales
  configuration** of the chapter 03 stock-out rule (`[SET-2]`, where the
  transition rule is configuration, not engine code). Under it backorders
  are identically zero, so chapter 02's general inventory position — on-hand
  plus on-order minus backorders — specializes to this chapter's
  `IP = on-hand + pipeline sum`.
- `[SIM-3]` **Pipeline.** An order placed at `t` with pipeline depth `d ≥ 1`
  arrives exactly at `t + d`; depth `0` is rejected. A supplied in-transit
  vector shorter than `d` is zero-padded; longer is rejected. `IP = end +
  Σ in-transit` at all times.
- `[SIM-4]` **Linear cost model, recomputable from parts.** Each cost
  component is declared as (name, rate, named per-period attribute); the
  canonical pair is holding × ending inventory and shortage × missed sales.
  A component naming an unknown attribute is rejected at construction. The
  trace exposes per-period, per-component values so that every component
  total and the grand total are recomputable as `rate × attribute` sums from
  the trace alone; `total = Σ components = Σ periods`, exactly.
- `[SIM-5]` **Purity.** The simulator never mutates caller-supplied state;
  two simulations from the same inputs produce identical traces.
- `[SIM-6]` **Refusal.** A cost-settlement window that requires demand beyond
  the resolved history is rejected before simulation — never settled against
  fabricated zero demand, which would silently undercount shortage.

## Make realized cost an optimization objective

Realized cost is the default objective of the tuning loop. The objective
this chapter **exports as its default** — the symbol chapter 09 binds to as
"the chapter 08 objective" — is `[OBJ-2]`, the simulator-accrued settle-path
cost: it matches the cost accounting a full run reports. `[OBJ-1]` is the
per-decision form, applicable to diagnostic and single-period evaluation,
never the exported default. The contract:

- `[OBJ-1]` **Per-decision cost (diagnostic/single-period).** For an order
  quantity `q` against demand `d`: `cost = overage · (q − d)⁺ + underage ·
  (d − q)⁺`. Two evaluation forms, matching the calibration mode: *per-step*
  — one decision per horizon row, summed over rows; *cumulative-window* —
  one decision against the window-summed demand, where the evaluated frame
  must be exactly one `(series key, origin)` window (more than one is
  rejected). An objective whose mode mismatches the frame's calibration
  mode, or a frame mixing modes, is rejected.
- `[OBJ-2]` **Settle-path cost (the exported default objective).** For
  multi-period replay, realized cost is the simulator's total under
  `[SIM-4]` — holding and shortage rates over the trace.
- `[OBJ-3]` **Aggregation.** A candidate's objective value is the sum of
  per-origin (and per-series) realized costs. The running value after `k`
  origins is the partial sum, monotone non-decreasing for non-negative
  costs — so early pruning of a partially evaluated candidate is sound.
- `[OBJ-4]` **Per-candidate computability.** The objective is a pure function
  of (candidate configuration, data, seed): no cross-candidate state, no
  side effects on durable state, evaluable inside a single tuning trial.
- `[OBJ-5]` **Failure semantics.** A statistically infeasible or degenerate
  candidate scores `+inf`; an engine or infrastructure failure propagates as
  an error — never `+inf`, never a silent zero.
- `[OBJ-6]` **Demand semantics.** Cost is defined against *demand*, and the
  binding is enforced, not merely labeled. When the dataset declares
  censoring facts (`[PAN-3]`, chapter 02), the realized-cost objective binds
  its actuals to the demand-honest series or refuses with an attributable
  cause — matching the calibration chapters' demand-honest scoring
  requirement. A censored-sales surrogate is permitted only under an
  explicit surrogate label, and that label is carried into every number
  derived from the objective value (aggregates, regret, study results); an
  unlabeled surrogate cost is non-conforming.
- `[OBJ-7]` **Regret.** `regret = Σ max(realized − oracle, 0)` over decisions
  aligned by key (never by position), with the oracle cost stream fixed
  independently of the candidate (e.g. a hindsight-optimal order stream on
  known demand); an empty alignment scores exactly `0`.
- `[OBJ-8]` **Objective integrity and pinned identity.** Two distinct
  rationales ban search dimensions, and chapter 09 (`[TUN-8]`) owns the
  single normative list of banned names — this clause defers to it wholly.
  Cost-structure components are banned for *objective integrity*: they
  parameterize the objective itself, so searching them lets the optimizer
  redefine what it is minimizing. The decision fractile and the
  policy-consumed coverage are banned from default tuning as *pinned
  policy-class identity*: the critical ratio is the definitional identity of
  the newsvendor policy class, derived from the cost structure — a pin on
  what the policy *is*, not a claim that this fractile is the multi-period
  optimum (see the scope limit below). The sanctioned explicit override is
  `[CFG-6]`.

## State the scope limit

The policies in this chapter are **per-decision-origin rules** (`[SET-7]`,
chapter 03): each maps one decision origin's calibrated forecast and
inventory state to one order via closed-form arithmetic. This chapter's claims are exactly: correct fractile and
order-up-to arithmetic, explicit boundary conventions, refusal on
insufficient data, and exact, recomputable cost accounting. It makes **no
multi-period system-optimality claim**: no assertion that any policy here
minimizes realized cost over the full horizon of a run. Realized cost
*measures* a policy; it does not certify it optimal. Multi-period optimality
analysis is out of scope for this chapter.

A second exclusion is explicit: **decision-outcome feedback** — feeding the
executed decision's realized outcome back into calibration state — is out
of scope. Calibration state updates remain limited to actuals resolution
and issuance bookkeeping (`[CAL-3]`, chapter 02); analyses over resolved
ledger rows, which carry the issued bound and the actual, are unaffected.
The exclusion is deliberate; revisiting it reopens this chapter.

## Bind the cost scope

The slot this chapter reserved for a coherent hierarchical cost objective is
bound by chapter 41, `[SEAM-4]`: realized cost attaches at the **decision
nodes** — the nodes where orders are placed. The engine defines no
lattice-level aggregate cost functional and makes no optimality claim above
the decision nodes; sums of realized cost over series (`[OBJ-3]`) are
bookkeeping totals, not a coherent-cost object. Chapter 02's structural term
**coherent cost** binds to exactly this: the per-decision-node realized-cost
family and its bookkeeping totals, nothing further. The derivation and
decision record are private spec-author material — the annex never ships
with the spec: [ANNEX:08-cost-objective-derivation].

## Conformance

A conforming implementation must demonstrate, by test:

1. Newsvendor: `CR` recomputed from the cost structure; interpolated target;
   `order = 0` when `IP ≥ T`; target monotone in underage cost (`[POL-7]`,
   `[POL-5]`).
2. (R,S): `P = lead time + review period` exact; per-step target sums bounds
   over `1..P` inclusive; cumulative frames use exactly the `h = P` terminal
   bound on fixtures where sum ≠ terminal bound (`[POL-8]`–`[POL-10]`).
3. (R,s,S): gate probed strictly below, at, and above `s`; equality yields no
   order (`[POL-11]`).
4. Integer units: fixtures probe fractional quantities away from integer
   boundaries (ceiling is discontinuous); negatives clamp; missing series
   score `0` (`[POL-12]`).
5. Every `[POL-6]` refusal condition raises before any order is emitted, and
   a policy exception propagates out of a replay uncaught.
6. Simulator: conservation asserted per period; an order placed at `t`
   arrives exactly `t + d`; cost breakdown recomputed from the trace equals
   the reported totals; caller state unmutated (`[SIM-1]`–`[SIM-5]`).
7. A settlement window extending past resolved history is rejected before
   simulation (`[SIM-6]`).
8. The objective evaluated twice on the same (candidate, data, seed) is
   identical; partial sums are monotone; mode mismatch is rejected
   (`[OBJ-1]`, `[OBJ-3]`, `[OBJ-4]`).
9. With no cap/floor/scale and no `[CFG-6]` override configured, decisions
   are byte-identical to a build without those features; with either on,
   every affected decision records the binding and the guarantee descriptor
   reflects the modification (`[POL-13]`, `[CFG-6]`).

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the old engine: `calibre/ordering/decision_rules.py` (`UpperBoundRule`,
`QuantileInterpolationRule`, `RSArithmetic`, `RSSArithmetic`, the NaN
terminal-bound refusal), `calibre/ordering/periodic_review.py` (protection
window assembly), `calibre/core/order_types.py` (`CostStruct`, critical
ratio), `calibre/ordering/simulation/` (simulator step, pipeline, linear
cost model, deep-copied state), `calibre/tuning/` objectives (per-step and
cumulative cost forms, regret). Negative space: the old engine's per-step
summation silently skipped NaN bounds (NaN-skipping sums), so a missing bound
inside the window could shrink the target — `[POL-8]`/`[POL-6]` rule that a
refusal; and its newsvendor target interpolated between interval bounds only
because richer calibrated distributions were not exposed — `[POL-7]` keeps
the invariants estimator-independent.
