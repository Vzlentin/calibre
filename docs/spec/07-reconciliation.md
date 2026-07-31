---
title: "Reconciliation — the point-forecast coherence stage"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 07 — Reconciliation

This chapter owns hierarchical reconciliation as a pipeline stage: the
reconciler protocol, the strategy registry, summing-matrix construction from
hierarchy facts, sparse-versus-dense feasibility at retail scale, the input
contracts, and the coherence/idempotence properties every strategy must
satisfy. It uses chapter 02 vocabulary verbatim: series key, panel, forecast
frame, origin, horizon step, model name, hierarchy facts, hierarchy node
(bottom / aggregate / total), aggregation lattice, fitted values, and the
invariants `[HIE-1..3]`, `[INV-COHERENCE]`, `[FRA-2]`, `[FRA-5]`. Normative
statements carry `[REC-n]` tags so tests can cite them. ADR 0002 records
why target support is a domain fact enforced at this seam.

Scope boundary: this chapter specifies **point-forecast** reconciliation.
The stage's contract does not extend beyond points, and the pipeline demands
no additivity of non-point forecast quantities across the lattice; both
contracts are bound by chapter 41 (40-gated-seams/) and stated in the two
bound sections at the end of this chapter.

## Place reconciliation in the pipeline

- `[REC-1]` Reconciliation runs per origin, **between predict and
  calibrate**: it consumes the forecast frame the forecasting stage emitted
  for that origin and rewrites the point-forecast column so that, for every
  cross-section, aggregate points relate to bottom points coherently
  (`[REC-12]`). Conformal calibration (chapter 05) then consumes the
  reconciled points.
- `[REC-2]` A **cross-section** is the set of forecast-frame rows sharing one
  `(model name, origin, horizon step)`. Reconciliation is applied
  cross-section by cross-section; it never mixes rows across models, origins,
  or horizon steps.
- `[REC-3]` The stage rewrites the point-forecast column **only**. It must
  reject, not ignore, a frame that already carries interval or quantile
  columns `[FRA-2]`: rewriting points underneath existing distributional
  columns would silently decouple them. (Such columns are rejected, never
  reconciled — the output-column contract below, bound by chapter 41
  `[SEAM-2]`.)
- `[REC-24]` **Validation-time rejection.** A configuration pairing native
  model quantile columns (chapter 04) with an active hierarchy is rejected at
  validation time (the `validate` verb, chapter 10) — never discovered
  mid-run. This is the validation-time face of the output-column contract
  (chapter 41 `[SEAM-2]`, bound below): no configuration may place quantile
  columns on a reconciliation path. It deliberately strengthens the
  reference engine, which rejected such frames only when the reconcile stage
  ran; the resulting behavior-oracle divergence (validation-time versus
  mid-run failure) is expected, not a regression.
- `[REC-4]` The stage preserves the frame contract: same required columns and
  dtypes, no reordering surprises. A strategy that synthesizes aggregate rows
  appends them in canonical node-label order after the unchanged bottom rows.
- `[REC-5]` Strategies that need in-sample residuals receive **fitted
  values** through an explicit per-origin reconciliation context, keyed by
  `(series key, timestamp, model name)` per `[FRA-5]` — never through
  forecast rows. The same context carries the panel target support `[PAN-5]`.
  Callers pass the context unconditionally, even to strategies that ignore
  fitted values; a residual-requiring strategy given no fitted values fails
  loudly before reconciling anything.

## Define the reconciler protocol

- `[REC-6]` A **reconciler** is a callable with the fixed signature: forecast
  frame + optional prebuilt hierarchy index + reconciliation context → forecast
  frame. The **hierarchy index** is the validated, canonical form of the
  hierarchy facts (bottom series keys, attribute columns, node labels,
  per-node expected member counts) built exactly once by run preparation and
  threaded to every consumer — no reconciler re-derives it `[HIE-1]`.
- `[REC-7]` An empty frame is a no-op pass-through. A `None` hierarchy index
  disables reconciliation math, but the stage still validates finite point
  values and enforces the context target support `[REC-25]`. A no-hierarchy
  `NONNEGATIVE` frame must be point-only because enforcement may rewrite its
  points; orchestration retains any native distributional columns outside the
  reconciliation seam and restores them only after the point rows return with
  identical keys.
- `[REC-8]` Every reconciler declares, as inspectable metadata (not
  behavior discovered by failure): (a) whether it requires fitted values;
  (b) its **input family** — *synthesis* (consumes bottom-node rows only and
  synthesizes aggregate rows itself) or *projection* (requires an independent
  base forecast at every node of the applicable lattice subset); and (c)
  which summing-matrix representation it consumes (sparse-capable or
  dense-only). Run preparation reads these declarations to decide whether
  aggregate-node forecast tasks must be built, whether the fitted-values
  sidecar must be produced, and how to size the memory preflight — all before
  execution starts. Every reconciler output must also satisfy the target
  support carried in the reconciliation context.

## Define the strategy registry

- `[REC-9]` Strategies are resolved by name through a registry: name →
  builder → reconciler instance. Names are normalized (trimmed,
  case-insensitive); resolving an unknown name is an error that lists the
  available names; the available-strategy listing is deterministic. The
  registry must include a no-op strategy (explicit "reconcile nothing") and a
  bottom-up synthesis strategy; projection strategies (least-squares and
  trace-minimization families) register through the same seam, whether native
  or adapter-backed.
- `[REC-10]` A strategy known to be numerically unusable at the engine's
  target scale (e.g. a full-covariance estimator that is ill-conditioned on
  retail-sized lattices) is rejected **by name at resolution time** with a
  message naming usable alternatives — not left to fail mid-run.

Substrate boundary: adapter-backed strategies wrap a substrate library's
**point** reconcilers. The library's off-the-shelf conformal-interval
reconciliation is deliberately not used — interval transformation at this
stage is excluded by chapter 41 `[SEAM-2]`.

## Construct the summing matrix from hierarchy facts

- `[REC-11]` The **summing matrix** `S` is derived generically from the
  hierarchy index — never hard-coded to a particular tree. Layout: columns
  are the bottom series in deterministic sorted-key order; rows are, in
  order, (1) a bottom identity block, one row per bottom series, (2) one row
  per distinct value of each attribute column, labelled by that column and
  value per `[HIE-3]`, (3) a single total row. Entries are exactly 0 or 1;
  `S @ b` for a bottom vector `b` reproduces every aggregate as the sum of
  its members and the total as the full sum. Overlapping attribute
  memberships make this a lattice, not a single tree — the construction must
  not assume unique parents.
- `[REC-12]` **Coherence property.** A reconciled cross-section vector `r`
  (aligned to the node labels, bottom block first) satisfies
  `r = S @ r[:n_bottom]` within a **derived tolerance**: the comparison
  tolerance must be computed from the problem instance — floating-point
  precision, lattice dimensions, and the magnitude of the vector under test
  (and, for iterative solvers, the solver's own convergence tolerance) — and
  the derivation must be a single shared function used by the runtime check
  and the test suite alike. A literal tolerance constant copied between
  verification sites is a defect: it silently encodes one lattice's
  conditioning as a universal truth.
- `[REC-13]` **Determinism and representation equality.** The same hierarchy
  facts produce the identical node-label order, column order, and matrix
  values on every build, and the sparse and dense representations of the same
  index are value-identical (densifying the sparse matrix reproduces the
  dense one exactly). This equality is a named behavior-oracle surface for
  the test strategy chapter.

## Select sparse or dense representation

Feasibility fact, restated engine-independently: at retail scale — on the
order of 30k bottom series and 34k lattice nodes — a dense float64 summing
matrix costs `n_nodes x n_bottom x 8` bytes ≈ **7.6 GiB for the matrix
alone**, before any per-cross-section copies; that is infeasible on commodity
memory. The same matrix stored sparse costs `nnz = n_bottom x (A + 2)`
entries (each bottom series appears once in the identity block, once per
attribute column `A`, once in the total row) — **megabytes, not gibibytes**.
The summing matrix is the memory pivot of the whole hierarchical pipeline at
this scale.

- `[REC-14]` The engine provides both representations behind one shared
  label-indexed interface (same node labels, bottom ids, and subset
  operation), so strategy math is representation-blind wherever the
  underlying solver permits.
- `[REC-15]` Representation selection is a **producer seam owned by the
  strategy's declaration** (`[REC-8]`c) and exercised *before* any
  per-strategy math runs: a sparse-capable strategy must never cause the
  dense matrix to be materialized on its behalf by shared harness code.
  Dense-only strategies keep a documented dense memory ceiling.
- `[REC-16]` **Memory preflight.** Before any eager expansion or matrix
  allocation, the engine estimates the run's peak memory from run-constant
  facts (bottom row counts, projected node-history rows, lattice
  cardinalities, and the summing-matrix bytes the *selected* representation
  will allocate) and refuses to start a run whose estimate exceeds detected
  available memory, reporting the estimate's components. The preflight is a
  deterministic stop, not a scalability mechanism: it prevents doomed runs;
  it does not make infeasible configurations feasible.

## Enforce input contracts

Hierarchy-index build time (once per run):

- `[REC-17]` Hierarchy facts must assign every attribute column a
  non-missing value for every bottom series `[HIE-2]`; series keys must be
  unique **after key normalization** — two keys that collide only under
  stringification (e.g. numeric `1` and string `"1"`) are duplicates and are
  rejected; aggregate node labels must not collide with any series key
  `[HIE-3]`. All violations fail the run before execution.
- `[REC-18]` The hierarchy covers every series: a panel or forecast frame
  presenting a series key absent from the hierarchy facts is an error, not a
  silently unreconciled row.

Per cross-section (every origin):

- `[REC-19]` Duplicate node rows within a cross-section are rejected. A
  synthesis-family strategy rejects any non-bottom row in its input; a
  projection-family strategy, given the subset of bottom series present,
  requires a forecast row for **every** node of the corresponding lattice
  subset — missing required nodes and aggregate rows outside the present
  subset both fail loudly with the cross-section's identity in the error.
- `[REC-20]` **Completeness alignment.** The member set used to form an
  aggregate's forecast must agree, per node and timestamp, with the member
  set used to resolve that aggregate's actual under `[INV-COHERENCE]`'s
  all-members-present rule. Concretely: a synthesis strategy synthesizes an
  aggregate only when every member is present in the cross-section (a partial
  member sum resolved against a complete-member actual silently undercounts
  the forecast — suppression, never partial sums). A present row with a
  non-finite point is rejected by the common frame validator rather than
  dropped from a sum or propagated downstream.

## Guarantee numerical honesty and idempotence

- `[REC-21]` **Solver convergence is a first-class signal.** For projection
  strategies whose output is coherent *by construction* (a projection of the
  base vector through `S`), the coherence check `[REC-12]` is an alignment
  guard only — it cannot detect a bad solve. Any iterative solver inside a
  strategy must therefore surface its convergence status explicitly, and
  non-convergence is an error carrying the cross-section identity (model
  name, origin, horizon step) — never a silently returned best-effort
  iterate.
- `[REC-22]` **Idempotence.** Reconciliation is a fixed-point transform: for
  every registered strategy, reconciling a frame whose points already satisfy
  `[REC-12]` returns the same point values within the derived tolerance. For
  synthesis strategies this reads as: re-deriving aggregate values from the
  (unchanged) bottom block reproduces the synthesized values.
- `[REC-25]` **Target-support postcondition.** After each cross-section is
  reconciled, the common application layer enforces the context target
  support `[PAN-5]` before calibration sees the points. For `REAL`, finite
  point values pass unchanged. For `NONNEGATIVE`, an adapter supplies the
  absolute numerical-error bound for its output; values in `[-bound, 0)` are
  canonicalized to exactly `0.0`, and values below `-bound` are rejected with
  the model name, origin, horizon step, and series key. Projection
  canonicalization corrects the bottom block and re-synthesizes every node so
  the returned vector still satisfies `[REC-12]`; the common coherence check
  runs again on that returned vector. Native strategies use the same validator
  rather than private clipping logic.

## Treat strategy choice as an experimental knob

Engine-observed fact, restated: reconciliation strategy is **not
coverage-neutral downstream** — with the conformal configuration held fixed,
the choice of reconciler measurably moves realized coverage (measured
magnitudes: Provenance). The reconciler is therefore a declared, first-class
knob, never a fixed implementation detail.

- `[REC-23]` The rewrite must treat the reconciliation strategy as a
  first-class experimental knob: selectable purely by configuration
  (chapter 10), sweepable and tunable alongside model and conformal choices
  (chapter 09), with per-node coverage diagnostics (chapter 02's
  hierarchical-coverage statistics) reported per strategy so a coverage miss
  is investigated against the reconciler as well as the conformal
  configuration. No coverage figure ships without naming the reconciler that
  produced it.

## Bind the output-column contract

Bound by chapter 41 `[SEAM-2]`. The reconciliation stage's **output-column
contract** is points only: the stage emits reconciled point columns and never
emits, adjusts, or consumes interval or quantile columns. Calibrated bounds
are computed downstream from the reconciled point base and are never
transformed by this stage. The calibrate stage asserts per-node marginal
bounds as specified in this chapter — nothing joint. Decision record at
[ANNEX:07-coherence-decision].

## Bind the non-additivity position

Bound by chapter 41 `[SEAM-3]`. Beyond reconciled points, no forecast
quantity — bound, quantile, or interval width — is required to sum across the
aggregation lattice. Ledger and scoring surfaces never assume additivity of
non-point quantities and read per-node bounds independently.
`[INV-COHERENCE]` remains scoped to observed quantities and reconciled
points. Decision record at [ANNEX:07-coherence-decision].

## Conformance

A conforming implementation must demonstrate, by test:

1. A configured reconciler runs after predict and before calibrate; the
   calibration stage observably consumes reconciled points `[REC-1]`.
2. A frame carrying interval or quantile columns is rejected by every
   point strategy `[REC-3]`.
3. Hierarchy validation rejects: a stringification-colliding duplicate key,
   a missing attribute value, an aggregate-label/series-key collision, and a
   forecast series absent from the hierarchy `[REC-17]`, `[REC-18]`.
4. Sparse and dense summing matrices built from one index densify to equal
   values with identical labels `[REC-13]`; a sparse-capable strategy's run
   never allocates the dense matrix `[REC-15]`.
5. The coherence check and its test use one shared, instance-derived
   tolerance function; no literal tolerance appears at a verification site
   `[REC-12]`.
6. A solver starved of iterations fails the run with cross-section identity
   in the error, despite its output passing the coherence check `[REC-21]`.
7. Reconciling an already-coherent frame is a fixed point for every
   registered strategy `[REC-22]`.
8. A partial cross-section yields suppressed (absent) aggregates for a
   synthesis strategy — never partial sums — and any present non-finite point
   fails the common frame validator `[REC-20]`.
9. The memory preflight blocks a run whose estimated peak exceeds detected
   memory and its report itemizes the summing-matrix term of the selected
   representation `[REC-16]`.
10. A configuration pairing native quantile columns with an active hierarchy
    fails the `validate` verb before any run starts `[REC-24]`.

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the old engine: `calibre/reconciliation/protocols.py` (fixed reconciler
signature, threaded hierarchy index, context-carried fitted values),
`calibre/reconciliation/summing.py` (generic lattice construction, dual
sparse/dense representation behind one label interface, stringified-key
collision rejection), `calibre/reconciliation/apply.py` (cross-section
harness, producer-selection seam, missing-node rejection),
`calibre/reconciliation/bottom_up.py` (bottom-only synthesis,
all-members-present suppression, NaN poisoning),
`calibre/reconciliation/nixtla_adapter.py` (checked iterative-solver guard —
upstream discarded the solver exit code and the coherent-by-construction
output masked it), `calibre/execution/hierarchy_memory.py` +
`benchmarks/m5/README.md` (preflight estimate; ~7.6 GiB dense ceiling at
full-M5 scale, sparse pivot). The measurement behind `[REC-23]`: at retail
scale with the conformal configuration held fixed at a 90% nominal level, a
structurally weighted projection strategy landed population coverage
approximately on-target (~91%) where bottom-up synthesis over-covered
(~95%) — a ~4-point swing attributable to the reconciler alone. Negative
space: the old engine verified
coherence against literal `1e-6` tolerances copied between sites —
`[REC-12]` replaces that with a derived, shared tolerance; and its preflight
was an explicit deterministic stop standing in for missing memory-efficient
paths — `[REC-16]` keeps the stop but names it a guard, not a scalability
mechanism.
