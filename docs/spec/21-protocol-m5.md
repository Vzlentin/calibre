---
title: "External protocol — the M5 hierarchical retail benchmark"
status: draft
invalidation-tags: []
date: 2026-07-20
---

# 21 — External protocol: M5

This chapter restates the M5 forecasting benchmark, as this program uses it,
as a complete, engine-independent protocol: the data contract, the aggregation
lattice, the rolling-origin backtest shape, and diagnostic scoring with exact
acceptance validity over a resolved ledger. Any implementation — including one
written from scratch
against this chapter and the public M5 release alone — must be able to run and
score the protocol from these statements. Vocabulary is chapter 02's, used
without redefinition; facts carry stable tags (`[M5-*]`). The
scoring-exemption ruling stated below is ratified; its decision record is
`[ANNEX:21-m5-scoring-exemption-record]`.

## Scope the protocol, not the solution

This chapter owns *rules of the game* only. Which forecasting plugin,
reconciliation strategy, or conformal method a pipeline uses to play it is
owned by chapters 04/07/05; how the pipeline is declared is chapter 10. One
normative bridge binds them:

- `[M5-0]` **Protocol as data.** Every protocol input defined below — dataset
  phase, target coverage, origin window, horizon, and method readiness
  requirement — enters an implementation as configuration, never as a
  hard-coded literal in engine or scoring code. Only the horizon (28) and the lattice construction
  rule are protocol-fixed structural facts.

## Define the data contract

The benchmark dataset is the public *M5 Forecasting — Accuracy* release:
Walmart daily unit sales for 3,049 products across 10 stores in 3 US states.

- `[M5-D1]` **Sales table.** A wide table with one row per bottom series — a
  product sold at a store, key `item×store` — five categorical attribute
  columns (item, department, category, store, state), and one column per
  calendar day labelled `d_1, d_2, …` in day order. Values are non-negative
  unit-sales counts, so the compiled canonical panel declares
  `NONNEGATIVE` target support `[PAN-5]`; this is a protocol fact, not a YAML
  field. The bottom series key is the `(item, store)` pair rendered as a
  single label.
- `[M5-D2]` **Calendar table.** A companion table mapping each day label to a
  calendar date (plus event/SNAP fields). The label-to-date mapping is
  **positional**: `d_N` is the N-th day of a contiguous daily calendar. A
  release variant that omits the explicit label column is admissible only
  when its dates form a contiguous daily sequence, from which the labels are
  re-derived positionally; otherwise the input is invalid.
- `[M5-D3]` **Phases.** Two release variants exist: *validation* (1,913 days,
  2011-01-29 through 2016-04-24) and *evaluation* (1,941 days, through
  2016-05-22 — 28 additional final days). The phase is a declared protocol
  input; **evaluation is the canonical full-run phase**. A run must state
  its phase and re-derive its origin window (`[M5-B4]`) from that phase's
  actual history end, never assume the other variant's.
- `[M5-D4]` **Censoring is real and unindicated.** Recorded sales are
  availability-censored — an out-of-stock day records what could be sold, not
  what was demanded — and the release carries **no stockout indicator and no
  uncensored demand stream**. In chapter 02 terms the panel carries *no*
  censoring facts (`[PAN-3]`): demand ground truth is unrecoverable from this
  dataset. Section `[M5-X*]` states the scoring consequence.
- `[M5-D5]` **Auxiliary inputs.** The weekly sell-prices table and the
  calendar's event/SNAP fields are admissible exogenous model inputs under
  temporal hygiene (`[INV-TEMPORAL]`). The protocol never requires them and
  scoring never consumes them.

## Define the aggregation lattice

- `[M5-H1]` The five attribute columns are hierarchy facts in the chapter 02
  sense (`[HIE-1]`–`[HIE-3]`): fixed before execution, complete (no missing
  attribute on any bottom series), and the sole source of the lattice.
- `[M5-H2]` **Marginal lattice.** The evaluated node set is: the bottom
  identity block (one node per bottom series), one aggregate node per distinct
  value of each attribute column (the sum of its member bottom series), and a
  single grand total. For the canonical release:

  | level class            | nodes      |
  | ---------------------- | ---------- |
  | bottom (`item×store`)  | 30,490     |
  | item                   | 3,049      |
  | department             | 7          |
  | category               | 3          |
  | store                  | 10         |
  | state                  | 3          |
  | grand total            | 1          |
  | **all**                | **33,563** |

- `[M5-H3]` **Narrower than the competition's set — deliberately.** The
  public competition scores 12 aggregation levels (42,840 series), including
  cross-classified levels such as state×category and item×state. This
  protocol's scored node set is the **marginal lattice of `[M5-H2]` only**. A
  replication may compute the cross-classified levels, but the diagnostic and
  validity contract (`[M5-A*]`) is defined over the marginal lattice and its
  seven level classes.
- `[M5-H4]` **Level identity from node identity.** Every node label must
  encode its level class recoverably from the label alone (e.g. bare series
  keys at bottom, `attribute=value` aggregates, a reserved total label), so
  per-level scoring is derivable from a resolved ledger with no side table,
  and no aggregate label can collide with a bottom key (`[HIE-3]`).
- `[M5-H5]` Every node in the lattice — bottom and aggregate alike — is
  forecast, calibrated, and scored. Aggregate actuals are exact member sums,
  defined only when every member is observed (`[INV-COHERENCE]`).

## Define the backtest loop

- `[M5-B1]` **Rolling daily origins.** Evaluation is a rolling-origin
  backtest over a contiguous daily origin window at the tail of the declared
  phase's history. At each origin, models see history strictly before the
  origin (`[TSK-2]`, `[INV-TEMPORAL]`) and issue forecast rows for horizon
  steps 1..28 per node; rows enter the ledger pending (`[LED-1]`) and resolve
  as the corresponding days' actuals become admissible (`[LED-2]`).
- `[M5-B2]` **Horizon = 28 days**, the competition's forecast task. The full
  multi-step profile is calibrated and scored — never step 1 alone.
- `[M5-B3]` **Origin-window readiness invariant.** With `h` the horizon,
  `n_origins` the origin count, and `n_first` the calibration method's
  declared readiness requirement (`[CAL-4]`) — the number of resolved
  calibration scores a partition needs before the method emits a finite bound
  at the target coverage — the window must satisfy:

  ```
  n_first + 2 · (h − 1) ≤ n_origins
  ```

  Reading: at the deepest horizon step, (i) accumulating the first `n_first`
  resolved scores costs `n_first + (h − 1)` origins, because a step-`h` score
  settles only `h − 1` days after its origin; (ii) the first finite step-`h`
  interval then needs `h − 1` further days for its own target to elapse
  inside the window. Only then does every horizon step contribute scored
  rows — every origin past the readiness point can issue and eventually score
  a full 28-step calibrated window. A violating window produces zero scored
  rows at the deepest steps and must be rejected at configuration validation,
  not discovered as a silent completeness failure.
- `[M5-B4]` The inequality above instantiates for per-`(series, horizon-step)`
  calibration partitions, where each origin contributes exactly one score per
  partition. Coarser partitions reach readiness sooner; the readiness term is
  always recomputed from the method's declared rule under the run's actual
  partition scheme and target coverage — never copied. *Example only, not a
  protocol constant:* at coverage 0.9 under a split-conformal higher-quantile
  rule, `n_first = 10` (the smallest `n` with `α > 1/(n+1)`), so `h = 28`
  requires at least `10 + 54 = 64` daily origins; one prior configuration ran
  exactly 64 (2016-03-20 through 2016-05-22 on the evaluation phase).

## Define diagnostic scoring and acceptance validity

The resolved ledger is the sole scoring surface: one row per
`(node, origin, horizon step, model)`, scored/resolved/covered per the shared
predicate `[LED-4]`, with denominator discipline `[LED-5]` throughout. All
coverage quantities below carry the **sales-coverage** label per `[M5-X2]`.
Coverage values are mandatory diagnostics. They never affect protocol
acceptance or Gate C status; acceptance force in this section attaches only
to exact scoring completeness and artifact validity.

- `[M5-A1]` **Population sales-coverage diagnostic.** Report the share of
  covered rows among all scored rows, pooled over every node, origin, and
  horizon step, with the target, signed deviation (`estimate − target`), and
  scored/resolved/total counts. The value is descriptive and never receives
  a pass, fail, or undetermined status.
- `[M5-A2]` **Per-level sales-coverage diagnostics.** Compute per-node
  coverage per `(node, model)` over that node's scored rows. For every level
  class, report the unweighted mean of scored-node coverages, the pooled
  per-level rate, the target and signed deviations, and
  scored/resolved/total counts. Every level statistic is descriptive and
  never gates acceptance.
- `[M5-A3]` **Exact scoring completeness.** Before reading covered/uncovered
  outcomes, derive a deterministic eligibility mask from the ledger's
  resolution state and the run's declared method readiness requirement, target coverage, partition
  scheme, origin window, and horizon. The scored mask must equal that
  eligibility mask exactly at population level and within every level class.
  A missing eligible row or an early ineligible scored row makes the artifact
  invalid and blocks acceptance. Every excluded row remains attributable
  from the ledger alone (`[LED-7]`).
- `[M5-A4]` **No inferential acceptance criterion.** The protocol neither
  derives nor consumes tolerance bands, confidence intervals, bootstrap
  intervals, power criteria, or any other statistical threshold over M5
  sales-coverage diagnostics. Separately declared research may analyze
  diagnostic uncertainty, but that analysis stays outside the protocol
  status and cannot affect Gate C or configuration selection.
- `[M5-A5]` **Per-node sales-coverage diagnostics.** Emit one row per
  `(node, model)` with coverage, target deviation, level identity, and
  scored/resolved/total counts. Per-node rows make no conditional,
  coherent-box, joint, or simultaneous claim. Rankings or descriptive flags
  may aid investigation, but no per-node value or threshold affects Gate C.
- `[M5-A6]` **Outputs and validity logic.** A scoring run produces a
  machine-readable summary, a per-node table, and a human-readable report.
  Each carries the sales-coverage label, declared reconciler, phase,
  conformal method, partition scheme, origin window, target, diagnostic
  estimates, counts, and exact-mask result. The machine status is `VALID`
  only when schemas, labels, identities, and exact completeness hold;
  otherwise it is `INVALID` and blocks acceptance. No coverage estimate
  contributes to that status, and no coverage-based pass/fail verdict is
  emitted.

## State the sales-coverage exemption

Ruling, ratified and dataset-scoped (`[ANNEX:21-m5-scoring-exemption-record]`):

- `[M5-X1]` **M5 diagnostics are sales-coverage by definition.** Every
  ledger row resolves against recorded unit sales — availability-censored,
  with no stockout indicator and no recoverable demand ground truth
  (`[M5-D4]`). Imputing demand to score against would score one model with
  another model's output; the protocol therefore scores sales, and says so.
- `[M5-X2]` **Label discipline.** The machine-readable summary and every
  piece of derived prose carry the "sales-coverage" label. A sales-coverage
  figure is never quoted as calibration honesty or as a service-level
  guarantee.
- `[M5-X3]` **The scoring seam stays demand-first.** The exemption lives in
  the M5 *dataset binding*, not in the metric definition: the engine's
  scoring interface remains defined over demand, and binding it to sales here
  is a declared property of this dataset, not a change to what coverage
  means.
- `[M5-X4]` **Honesty claims live elsewhere.** Calibration-honesty claims
  require datasets where demand is known or recoverable — synthetic fixtures,
  or censoring-indicated datasets whose panels carry censoring facts
  (`[PAN-3]`).
- `[M5-X5]` **Scoped and revisitable, never a license.** The exemption is
  scoped to this dataset and open to revision (e.g. should a
  censoring-indicated M5 variant appear). It never licenses sales-scoring on
  a dataset that *does* carry censoring indicators.

## Declare the reconciler with every figure

- `[M5-R1]` **Reconciliation strategy is a coverage lever, not
  coverage-neutral.** Evidence-grade observation from full-scale runs of the
  prior engine at a 90% target, same models and conformal settings:
  structural-weights reconciliation landed population sales-coverage near
  target (~91.0%) where bottom-up aggregation over-covered (~94.9%).
  Consequently any quoted M5 coverage figure must declare its reconciliation
  strategy — alongside its phase, conformal method, partition scheme, and
  origin window — or it is not a reproducible claim. This is a reporting
  obligation, not a strategy prescription. A Gate C configuration is
  selected independently of observed M5 sales-coverage; diagnostic values
  never tune or select the strategy.

## Bound the protocol's role

- `[M5-N1]` **Apparatus, never flagship.** M5 is protocol apparatus: a scale
  and hierarchy exercise — ~30.5k bottom series, ~33.6k lattice nodes,
  28-step daily calibration — proving an engine's hierarchical machinery and
  coverage accounting at retail scale. By `[M5-X2]`/`[M5-X4]` and chapter
  01's flagship discipline, no flagship or honesty headline may bind to M5
  sales-coverage; decision-grade claims live on censoring-honest measurements.
- `[M5-N2]` **Non-carry rule.** No coverage figure, statistical interval,
  diagnostic threshold, or sales-scored baseline produced by a previous
  engine is a protocol constant, a target, or a comparison baseline for a
  replication. A replication is judged by conformance to
  `[M5-D*]`/`[M5-H*]`/`[M5-B*]`/
  `[M5-A*]`, never by proximity to an inherited number.

## Conformance

A conforming implementation must demonstrate, by test:

1. Data validation rejects a sales table with no day columns, a calendar not
   covering every day label, and a label-less calendar whose dates are not
   contiguous daily (`[M5-D1]`, `[M5-D2]`).
2. Lattice construction satisfies the node-count identity
   `n_nodes = n_bottom + Σ_attr (distinct values) + 1`, produces exact member
   sums for aggregates, and rejects label collisions (`[M5-H2]`, `[M5-H4]`,
   `[M5-H5]`).
3. Configuration validation rejects any origin window violating
   `n_first + 2(h − 1) ≤ n_origins` under the run's declared method,
   partition scheme, and coverage (`[M5-B3]`, `[M5-B4]`).
4. Every reported coverage quantity carries scored/resolved/total counts, and
   the scored-row predicate is the shared one, never re-derived (`[M5-A1]`,
   `[LED-4]`, `[LED-5]`).
5. Exact completeness, by property test: exact eligibility/scored-mask
   equality is valid; removing one eligible scored row or scoring one
   ineligible row makes the artifact invalid (`[M5-A3]`, `[M5-A6]`).
6. Coverage values do not affect validity: otherwise-identical synthetic
   artifacts with 0%, target, and 100% sales-coverage retain the same
   `VALID` status when their schemas and masks are valid (`[M5-A4]`,
   `[M5-A6]`).
7. The machine-readable summary of every M5 scoring run carries the
   sales-coverage label; its absence is a test failure (`[M5-X2]`).
8. All protocol inputs are supplied as configuration and changing them
   requires no code change (`[M5-0]`).

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the old repo: `benchmarks/m5/README.md` (runbook: acceptance-run procedure,
origin-window derivation, gate shape, artifact layout),
`benchmarks/m5/config/full-wls-struct.yaml` and siblings (`full.yaml`,
`full-cumulative.yaml`, `ca-subset-streaming.yaml` — the protocol instantiated
as pure configuration, evidence `[M5-0]` is achievable),
`tests/benchmarks/test_m5_config.py` (the readiness inequality pinned as a
test, with 64/28/10 as that configuration's values),
`calibre/execution/m5_adapter.py` + `calibre/execution/m5_loading.py` (data
contract: wide-to-long melt, positional day-label derivation with the
contiguity guard, phase resolution, hierarchy attribute extraction),
`calibre/reconciliation/summing.py` (marginal lattice, node-label scheme,
sparse representation), and `calibre/evaluation/m5_coverage.py` (the scorer,
relocated engine-side from the benchmark directory: per-level averaging,
legacy completeness-as-undetermined verdict logic, outlier diagnostics, and
report shape). Negative space: that engine's gate constants (±3.0 pp population,
±5.0 pp per-level, 0.50 scored-ratio floor, 0.10 outlier tolerance) are
retired value-based gate inputs; `[M5-A4]` makes the whole statistical gate
inadmissible. Its summary carried no sales-coverage label — `[M5-X2]` closes
that gap; its full-scale runs supplied the reconciler-as-lever evidence
behind `[M5-R1]`.
