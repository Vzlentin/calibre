---
title: "External protocol — the VN2 inventory challenge"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 20 — External protocol: VN2

This chapter restates the VN2 inventory-planning challenge as a complete,
engine-independent protocol: the data contract, the decision cadence, the
inventory dynamics, the cost accounting, and the outputs a replication must
produce. Any implementation — including one written from scratch against this
chapter alone — must be able to run and score the protocol from these
statements. Vocabulary is chapter 02's, used without redefinition. Facts carry
stable tags (`[VN2-*]`) for citation by tests and later chapters.

## Scope the protocol, not the solution

This chapter owns *rules of the game* only. Which forecasting plugin, conformal
method, or ordering policy a pipeline uses to play it is owned by chapters
04/05/08; how the pipeline is declared is chapter 10. One normative bridge
binds them:

- `[VN2-0]` **Protocol as data.** Every constant defined below — round count,
  lead time, review period, cost rates, initial state — enters an
  implementation as configuration (the cost rates specifically as a cost
  structure per `[CST-3]`), never as a hard-coded literal in engine or policy
  code. A replication with different rates or cadence is a different
  configuration of the same machinery.

## Define the data contract

The challenge dataset is a fixed set of tables covering **599 series**, each a
weekly demand series for one store–product pair.

- `[VN2-D0]` **Provenance.** The dataset originates from the VN2
  inventory-planning challenge; the challenge distribution is the sole
  conformant source. Every conformance run records file-level integrity
  facts alongside its outputs: an inventory of the input files consumed and
  a content digest per file. This chapter pins no specific hashes — digests
  are facts of a run record, not protocol constants.

### Sales panel — progressive reveal

- `[VN2-D1]` The sales history is a wide table: one row per series, key columns
  `Store` and `Product`, and one column per week-start date (Monday calendar).
  Series identity is the `(Store, Product)` pair itself; the string rendering
  `"<store>_<product>"` is an engine-internal spelling, non-normative to the
  protocol. Values are non-negative floats: units sold that week.
- `[VN2-D2]` The reveal is progressive. The round-0 table carries 157 weekly
  columns (2021-04-12 through 2024-04-08). Each subsequent reveal `r`
  (`r = 1..8`) appends **exactly one** new week column relative to reveal
  `r−1`; that new column is week `r`'s realized sales. A reveal differing from
  its predecessor by anything other than one appended column is invalid input.
- `[VN2-D3]` An absent or missing value in a revealed column means zero sales
  for that series-week. This is a rule of this protocol's data (the panel is
  dense by construction), stated explicitly so it is a declared protocol fact
  and not a silent imputation in tension with `[PAN-2]`.
- `[VN2-D4]` Recorded sales are **availability-censored**: a week where the
  series was out of stock records what could be sold, not what was demanded.
  The protocol's demand stream *is* the revealed sales stream (see
  `[VN2-S2]`); censoring awareness is a fitting-side concern only.

### Master table — hierarchy facts

- `[VN2-D5]` An optional static table assigns each series values for six
  categorical attributes: product group, division, department, department
  group, store format, format. These are hierarchy facts in the chapter 02
  sense (`[HIE-1]`–`[HIE-3]`): fixed before execution, usable to induce an
  aggregation lattice or as model features. The protocol itself never
  aggregates; scoring is per-series.

### In-stock table — censoring facts

- `[VN2-D6]` An optional static table gives a boolean per (series, week) —
  `True` when the series was available for sale — covering history up to
  challenge start. These are censoring facts per `[PAN-3]`: admissible at
  every origin, usable for censoring-aware fitting (e.g. imputing demand for
  out-of-stock weeks before training). When the dataset supplies a numeric
  availability level, it rides the censoring fact as the optional numeric
  availability bound (chapter 02 `[PAN-3]`) alongside the status flag; the
  flag-only form stays valid. They never alter the demand stream the
  simulation consumes.

### Initial inventory state

- `[VN2-D7]` A per-series starting state is given as data: on-hand end
  inventory plus two in-transit quantities — one arriving in week 1, one in
  week 2. These seed the two-slot pipeline of `[VN2-S1]` and stand in for
  orders notionally placed in the two weeks before round 1.

## Define the decision cadence

- `[VN2-C1]` The challenge runs **6 decision rounds**, `r = 1..6`, one per
  week (**review period = 1 week**). Round `r`'s origin is the Monday
  following the last revealed week: the participant observes sales through
  week `r−1` (reveal `r−1`) and decides for week `r`. In the challenge data
  the six origins are 2024-04-15 through 2024-05-20.
- `[VN2-C2]` At each round the participant emits one order per series: a
  non-negative quantity (`[ORD-1]`), committed before week `r`'s column is
  revealed. Temporal hygiene (`[INV-TEMPORAL]`) binds exactly here: the
  information admissible at round `r`'s origin is reveals `0..r−1` plus the
  static tables `[VN2-D5]`–`[VN2-D7]` — never week `r`'s sales.
- `[VN2-C3]` **Lead time = 2 weeks.** An order committed at round `r` is in
  transit during weeks `r` and `r+1` and becomes available to serve demand at
  the start of week `r+2`.
- `[VN2-C4]` The **protection window** is a derived structural fact, not an
  extra rule: the round-`r` decision is the last one that can affect
  availability before week `r+3` (the earliest arrival of round `r+1`'s
  order), so it protects demand over weeks `r..r+2` — lead time + review
  period = **3 weeks**. Per-origin forecasts therefore need horizon `H = 3`,
  and the decision-relevant quantity is the demand *sum* over horizon steps
  1..3, not any single step.

## Define the inventory dynamics

State per series: on-hand end inventory plus a two-slot in-transit pipeline
`(w+1, w+2)`, seeded from `[VN2-D7]`.

- `[VN2-S1]` **Weekly transition.** For each series, week `w` advances in this
  order:
  1. *Arrive*: the `w+1` slot empties into on-hand:
     `start_inventory = end_inventory_prev + arrivals`.
  2. *Sell*: `sales = min(start_inventory, demand_w)`;
     `missed_sales = demand_w − sales`;
     `end_inventory = start_inventory − sales`.
  3. *Shift and commit*: the `w+2` slot moves to `w+1`; the order committed
     this week (zero in drain weeks) enters the `w+2` slot.
- `[VN2-S2]` `demand_w` is the value of reveal `w`'s new column for the
  series (zero when absent, `[VN2-D3]`). The protocol scores against revealed
  sales as demand; a replication must not substitute an uncensored estimate
  into the transition.
- `[VN2-S3]` **Lost sales, no backorders.** Unmet demand vanishes; it never
  carries into a later week's demand. Inventory and orders are non-negative
  floats throughout; `end_inventory ≥ 0` always holds by construction of
  step 2.
- `[VN2-S4]` The horizon simulated is **8 weeks**: the 6 decision weeks plus
  **2 drain weeks** (weeks 7–8, in the challenge data 2024-05-27 and
  2024-06-03) with orders forced to zero, so the round-5 and round-6 orders
  arrive and are exposed to realized demand. Costs accrue in all 8 weeks.

## Define the cost accounting

- `[VN2-K1]` Two linear cost components, with rates that are protocol inputs
  carried by the cost structure (`[VN2-0]`, `[CST-1]`): **holding at 0.20 per
  unit** of end-of-week on-hand inventory, and **shortage at 1.00 per unit**
  of missed sales. Units are currency (EUR in the source challenge).
- `[VN2-K2]` Holding is charged on `end_inventory` only — in-transit
  quantities accrue **no** holding cost. Shortage is charged on
  `missed_sales` only. There are no ordering, purchase, or salvage terms.
- `[VN2-K3]` The implied critical ratio (`[CST-2]`) is
  `1.00 / (1.00 + 0.20) = 5/6 ≈ 0.833` — the protocol's cost-optimal decision
  fractile for protection-window demand. This is a fact about the rates, cited
  here because measurement (`[VN2-R3]`) refers to it; how a policy uses it is
  chapter 08's concern.
- `[VN2-K4]` Total cost is the sum of both components over all 599 series and
  all 8 simulated weeks. The accounting identities
  `holding_w = 0.20 × end_inventory_w` and
  `shortage_w = 1.00 × missed_sales_w` hold row-exactly.

## Require the replication outputs

A run of the protocol must produce three artifacts:

- `[VN2-R1]` **Order stream** — one non-negative quantity per (series, round),
  6 rounds × 599 series, each an order fact per `[ORD-1]`–`[ORD-3]` recorded
  before the corresponding reveal.
- `[VN2-R2]` **Cost ledger** — one record per (series, week) for all 8 weeks:
  start inventory, arrivals, demand, sales, missed sales, end inventory,
  holding cost, shortage cost. The ledger is the sole scoring surface; the
  final figures are recomputable from it and must match it row-for-row.
- `[VN2-R3]` **Final triple** — `(holding_total, shortage_total, total_cost)`
  with `total_cost = holding_total + shortage_total` exactly.

Measurement runs (benchmark or evidence runs, beyond bare scoring) must
additionally record:

- `[VN2-R4]` Cumulative cost per decision round (rounds 1–6) and the drain
  remainder, so cost trajectories are comparable across runs, not only
  endpoints.
- `[VN2-R5]` **Realized protection-window coverage** per (series, origin).
  Conditional: this measurement applies when the measured run exposes a
  calibrated decision bound consumed as an order-up-to level; a run without
  such a bound owes the cost trajectory (`[VN2-R4]`) only. The measurement is
  the binary event that the realized 3-week demand sum over the protection
  window (`[VN2-C4]`) did not exceed that bound, evaluated at the fractile
  the run targeted — 599 × 6 = 3,594 events — reported per origin and pooled.
  Any pooled figure must state its pooling window explicitly, because
  early-round calibration transients can dominate a naive 6-round pool.
  Because the realized sum is built from revealed sales (`[VN2-S2]`), the
  default measurement is **sales-coverage by construction**; on evaluation
  windows containing no stockouts, sales equal demand week by week, so raw
  and demand-honest coverage provably coincide there. Whenever the run's
  data carries stockout indicators — VN2 supplies an in-stock signal
  (`[VN2-D6]`) — a censoring-aware companion measurement is **required**
  alongside the raw figure, under chapter 50's censoring doctrine: coverage
  properties are stated over demand, and sales-scored numbers are labeled as
  such.

## Retire old totals — non-carry rule

- `[VN2-N1]` No cost total produced by the previous engine is a protocol
  constant, a target, or a comparison baseline for the rewrite. Such figures
  were regression tripwires for that engine's code paths and retire with it.
  A replication is judged by conformance to `[VN2-D*]`/`[VN2-C*]`/`[VN2-S*]`/
  `[VN2-K*]`, never by proximity to an inherited number.
- `[VN2-N2]` The rewrite mints its own reference totals fresh, on its own
  pinned toolchain **and CPU architecture** — floating-point results of
  gradient-boosted model paths are known to diverge across architectures, so
  a total is comparable only under a pinned (config, toolchain, architecture)
  triple, and any frozen figure must record all three.

## Bind the flagship figures

Bound by chapter 42 (40-gated-seams/), `[FLG-1]`/`[FLG-2]`: the product
reports the two-axis flagship claim on this protocol — the **certificate**
(realized coverage of the decision bound at the cost-derived fractile within
a pre-registered acceptance band over a declared post-warmup window) and the
**price ratio** (guarantee-on total cost against an engine-fresh cost-tuned
reference) — measured on the surfaces this chapter fixes
(`[VN2-R3]`–`[VN2-R5]`) under the `[VN2-N2]` environment pin. This chapter
continues to fix the measurement *procedure* only.
`[ANNEX:20-vn2-replication-notes]`

## Conformance

A conforming implementation must demonstrate, by test:

1. Reveal validation rejects a sales table that appends other than exactly one
   new week column, and treats missing values as zero sales
   (`[VN2-D2]`, `[VN2-D3]`).
2. On hand-checkable fixtures, one weekly transition reproduces `[VN2-S1]`
   exactly — arrival before sale, lost sales, order committed to the far
   slot — including the seeded-state rounds 1–2 (`[VN2-D7]`).
3. An order committed at round `r` first affects sales in week `r+2` and
   never earlier (`[VN2-C3]`).
4. No round-`r` order is a function of reveal `r` or later
   (`[VN2-C2]`, property test).
5. The 8-week run books costs satisfying `[VN2-K2]`/`[VN2-K4]` row-exactly,
   and the final triple equals the ledger's column sums (`[VN2-R2]`,
   `[VN2-R3]`).
6. All protocol constants are supplied as configuration and changing them
   requires no code change (`[VN2-0]`).

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the old repo: `benchmarks/vn2/config.py` (rounds/lead-time/review constants,
cost rates, critical-ratio derivation), `benchmarks/vn2/simulator.py` +
`calibre/ordering/simulation/rules.py` (transition order, two-slot pipeline,
lost-sales rule, linear cost model), `benchmarks/vn2/replay.py` (round-reveal
pairing, drain weeks), `benchmarks/vn2/config/vn2-winning-loop.yaml` (the
protocol instantiated as pure configuration — evidence `[VN2-0]` is
achievable), and the guarantee-on measurement memo under
`benchmarks/vn2/results/` (the `[VN2-R4]`/`[VN2-R5]` measurement shape).
Negative space: the old engine's frozen cost total is exactly what `[VN2-N1]`
retires, and its arch-specific divergence motivated `[VN2-N2]`.
