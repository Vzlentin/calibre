---
title: "Gated seams — the decision-calibration seam contract"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 41 — The decision-calibration seams

This chapter states the ratified seam contract: where calibration,
reconciliation, decision, and scoring meet, and what claim each surface may
make. It gives normative force to the chapter 02 structural terms *coherent
cost* and *hierarchical coverage*, and binds the marked slots that chapters
05, 07, 08, 09, and 50 reserved for it. It states **interfaces and
claims only**; every derivation, alternative, and rationale sits behind
`[ANNEX:41-seam-decision-record]`. Requirements carry `[SEAM-n]` tags for
citation by tests and later chapters.

The seam statement between the dividers below is the ratified text, carried
verbatim; the binding requirements that follow attach it to the surfaces the
rest of the spec reserved.

---

## The four seams

1. **Calibration seam.** The one-sided decision-bound calibration attaches
   after reconciliation, to the consumed window-sum at each decision node.
   Input: the reconciled point base and resolved calibration scores for
   the protection-window functional. Output: one decision bound per
   (node, decision origin), carrying a **guarantee descriptor** —
   `{type, level, scored series, window, scope}` — so the harness knows
   what claim each bound makes (default type: one-sided coverage at the
   cost fractile). The calibration method behind the seam is pluggable;
   the descriptor is not optional.
2. **Reconciliation seam.** Coherence is enforced at the point layer, as a
   constraint on the forecast object — never as a transformation of
   calibrated bounds. The reconciliation strategy is a declared,
   first-class configuration choice: strategy selection measurably moves
   realized coverage, so no coverage number ships without naming its
   reconciler.
3. **Decision seam.** The mapping from calibrated bound to order:
   protection-window assembly, the cost-derived fractile tau = Cu/(Cu+Co)
   from the declared cost structure, and the order-up-to consumption of
   the bound. Clamps on the decision bound are forbidden by default; any
   clamp is an explicit, named configuration act that changes the
   guarantee descriptor (a clamped bound no longer carries a coverage
   claim).
4. **Scoring seam.** Observation and scoring are demand-first: the scored
   series is demand, with censoring handled explicitly at the dataset
   binding. Coverage scored against a series other than demand carries a
   label saying so and is never quoted as calibration honesty.

## Scope declaration

Cost is defined at the decision nodes — the nodes where orders are placed;
aggregate levels constrain and inform the forecast object but do not
aggregate cost. Coverage claims attach at that same decision scope: the
per-decision one-sided bound is the guaranteed object. No claim of
cross-node coherence of calibrated bounds is made; the two-sided band that
analysts read is diagnostic and is not the decision object.

---

## Bind the seams into the tree

- `[SEAM-1]` **The guarantee descriptor is normative.** Every calibrated
  decision bound the engine issues carries the guarantee descriptor defined
  in chapter 02, populated at issuance from the method manifest and the
  run configuration. The default claim is one-sided coverage at the
  cost-derived fractile in the finite-sample-marginal currency; any other
  claim is declared, never inferred. A bound without a descriptor is
  non-conforming; a consumer that cannot read the descriptor must reject
  the bound, not assume a claim.
- `[SEAM-2]` **Coherence binds at the point layer only.** The
  reconciliation stage's output-column contract is points only: the stage
  emits reconciled point columns and never emits, adjusts, or consumes
  interval or quantile columns. Calibrated bounds are computed from the
  reconciled point base downstream of the stage and are never transformed
  by it. This binds the coherent-distributional-outputs slot in chapter 07.
- `[SEAM-3]` **No non-point quantity is required to be additive.** Beyond
  reconciled points, the engine requires no forecast quantity — bound,
  quantile, or interval width — to sum across the aggregation lattice, and
  no ledger or scoring surface may assume such additivity. Per-node bounds
  are read independently per node. This binds the non-additivity slot in
  chapter 07; chapter 02's `[INV-COHERENCE]` remains scoped to observed
  quantities, and reconciled-point additivity is chapter 07's own contract
  (`[REC-12]`).
- `[SEAM-4]` **Cost attaches at the decision nodes.** Realized cost is
  defined where orders are placed. The engine defines no lattice-level
  aggregate cost functional and makes no claim about what "optimal" means
  above the decision nodes; sums of realized cost over series are
  bookkeeping totals, not a coherent-cost object. This binds the
  coherent-hierarchical-cost-objective slot in chapter 08 and, through
  chapter 08's exported objective, the tuning-objective slot in chapter 09:
  the chapter 02 structural term *coherent cost* denotes the per-decision-node
  realized-cost family and its bookkeeping totals, nothing further.
- `[SEAM-5]` **Coverage claims attach at decision scope.** The guaranteed
  object is the per-(node, decision origin) one-sided bound. The engine
  makes no cross-node coherence claim about calibrated bounds; the
  two-sided band is diagnostic. The chapter 02 structural term
  *hierarchical coverage* denotes the per-node and per-level diagnostic
  statistics; no member of that family is a joint or simultaneous claim
  across the lattice. This binds the joint-claims slot in chapter 05 and
  the lattice-claim rule in chapter 50.
- `[SEAM-6]` **`joint_claim` admissible values.** The chapter 05 manifest
  field `joint_claim` takes exactly two values: `none` (default) and
  `class-conditional`. `class-conditional` declares coverage conditional on
  membership in a **finite-dimensional class system**: a declared, finite
  set of series groups derived from the hierarchy facts of the run's
  aggregation lattice (levels and aggregate-node memberships), fixed at
  calibration time. The claim attaches at decision scope per class; its
  descriptor carries the claim `class-conditional coverage`, with the class
  system named in the descriptor's scope field (`[GRT-3]`).
  Lattice-wide joint or simultaneous coverage claims are inadmissible under
  every value of this field; no third value may be registered without
  reopening this chapter.
- `[SEAM-7]` **The calibration context is a declared channel.** A conformal
  method may consume a **calibration context**: the hierarchy facts of the
  rows it calibrates, applies to, or observes — series key, lattice level,
  and aggregate-node memberships, derived from the same hierarchy index
  reconciliation strategies receive (chapter 07, `[REC-6]`); it is
  engine-supplied, read-only, and declared in the method manifest
  (`consumes-calibration-context`), absent by default, satisfying chapter
  03's declared-channel rule (`[SPN-2]`). A method declaring
  `joint_claim: class-conditional` must declare the context; the class
  system is formable from the context alone.
- `[SEAM-8]` **A clamp voids the claim.** Consistent with the decision
  seam: a clamp or override on the decision bound is an explicit, named
  configuration act, and the issued descriptor of a clamped bound carries
  the claim `none (not engine-calibrated)` (its currency field is then not
  applicable, `[GRT-2]`) — it states no coverage claim, and scoring
  surfaces score it as such.
- `[SEAM-9]` **Demand-first scoring is claim-bearing.** The scored-series
  field of the descriptor is the surface that carries the scoring seam:
  coverage scored against any series other than demand is labeled by that
  field per the score-input contract (chapter 05, `[CNF-27]`–`[CNF-28]`)
  and is never quoted as calibration honesty (chapter 50 owns the honesty
  scoring stance; chapter 21 owns the ratified dataset-level exemption).

## Conformance

A conforming implementation must demonstrate, by test:

1. Every issued decision bound carries a complete guarantee descriptor;
   issuance of a bound with a missing or unregistered claim type is
   rejected (`[SEAM-1]`).
2. The reconciliation stage rejects any frame carrying interval or
   quantile columns (chapter 07's rejection test, `[REC-3]`), and on a
   two-origin fixture the bounds issued at the first origin are
   byte-identical in ledger and state before and after the second origin's
   reconciliation runs — the stage never touches previously issued bounds
   (`[SEAM-2]`).
3. Per-node scoring of a fixture lattice matches scoring each node in
   isolation, and a witness variant that injects a cross-node bound sum
   into the scoring surface makes that equivalence check fail (`[SEAM-3]`).
4. Registry validation rejects a manifest whose `joint_claim` is neither
   `none` nor `class-conditional`, and rejects `class-conditional` without
   `consumes-calibration-context` (`[SEAM-6]`, `[SEAM-7]`).
5. A class-conditional method receives a context whose class system is
   derivable from the run's declared lattice alone; feeding the same rows
   with a permuted context yields the permuted class assignment and
   nothing else (no hidden channel) (`[SEAM-7]`).
6. Enabling any bound clamp rewrites the issued descriptor's claim to
   `none (not engine-calibrated)` on exactly the rows where the clamp
   modified the raw bound, per the per-row binding record (chapter 05,
   `[CNF-21]`) (`[SEAM-8]`).

## Provenance

The seam statement between the dividers is the ratified Stage 1 seam
contract, carried verbatim; its decision record, derivations, and rejected
alternatives are private spec-author material at
`[ANNEX:41-seam-decision-record]`. This chapter consumes no old-engine
behavior.
