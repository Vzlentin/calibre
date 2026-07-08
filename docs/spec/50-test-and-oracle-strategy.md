---
title: "Test and oracle strategy"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 50 — Test and oracle strategy

This chapter owns one contract: how the rewrite proves correctness and
behavioral fidelity. It fixes the behavior-oracle stance toward the frozen
predecessor engine, the taxonomy every inherited assertion is classified
under, the tolerance doctrine that decides what "equal" means in every test,
the portability rules for cross-engine checkpoints, the lifecycle rules for
numeric gates, and the CI gate structure.

It binds to the test-suite curation ruling, which is **ratified**. The
per-item corpus (every inherited
assertion classified, consolidated into a canonical oracle-property set with
per-property tolerance classes and derivations) lives behind
`[ANNEX:50-oracle-curation-record]`. This chapter states the doctrine the
ratified ruling applies.

## Adopt the behavior-oracle stance

The predecessor engine is frozen and serves as a **behavior oracle**: a
running reference whose *behavior* — not whose code, design, or output bytes
— the rewrite is checked against on the surfaces the curation ruling carries
forward.

- The oracle is pinned to **one immutable reference tag**. No oracle capture
  is ever taken from a moving branch head; "the oracle said X" is meaningful
  only at the pinned tag.
- Every captured oracle output ships a **capture manifest**: oracle tag,
  platform (CPU architecture, OS, numeric-library provenance), dependency
  lockfile digest, configuration digest, and input-data digest. A capture
  missing any manifest field is not oracle evidence.
- The oracle is a behavior source, **not a design source**: the rewrite's
  architecture is specified by this spec's other chapters; oracle tests
  constrain observable behavior only.
- **Cutover** is the point at which the rewrite replaces the frozen engine.
  At cutover the oracle retires: oracle captures become historical artifacts,
  and every gate defined against them is deleted, not loosened.

## Classify every inherited assertion

Every assertion inherited from the frozen engine's test estate is ruled into
exactly one class before it may influence the rewrite's suite:

| Class | Meaning | Disposition |
|---|---|---|
| **Oracle property** | Component behavior the rewrite must reproduce | Carried, restated behaviorally, tolerance per doctrine below |
| **Apparatus** | Number-continuity tripwires of the frozen engine (pinned scalars, sha-pinned baselines, config pins) | Retired at cutover; never a cross-engine target, never a headline number |
| **Plumbing** | Engine-idiosyncratic interface tests (schemas, wiring, registries, transport) | Retired; design lessons may route to spec chapters as guidance, never into the equivalence contract |
| **Artifact** | Non-test continuity data (frozen baselines, evidence records) | Archived or retired, with named salvage of any definition they carry |
| **Mixed** | A behavioral kernel wrapped in an engine-idiosyncratic harness | Kernel extracted as an oracle property; shell retired |

**Extraction is the default posture.** Mixed is the largest inherited class;
the contract carries kernels, not test files. A test is never carried
"wholesale because it exists" — the ruled corpus in
`[ANNEX:50-oracle-curation-record]` is the exhaustive list, and anything not
carried there is retired.

The carried behavioral spine covers: ordering-policy
arithmetic and its refusal contracts, inventory-simulation conservation and
pipeline physics, reconciliation mathematics against a closed-form reference,
conformal calibration semantics (readiness, window accounting, structural
interval validity), metric definitions over the ledger and the scored-row
predicate `[LED-4]`, dataset ingestion contracts, and the resume /
state-continuity protocol for calibration state `[CAL-2]`.

## Apply the tolerance doctrine

Every carried property names exactly one tolerance class. There is no
default tolerance and no shared global epsilon.

1. **Exact structural/integer.** Sets, counts, presence/absence,
   raise-vs-proceed, discrete indices: tolerance-free.
2. **Closed-form recomputation.** The test independently recomputes the
   defining formula from shared inputs; tolerance covers only that
   recomputation's float rounding (summation bounds of the form
   n·eps·magnitude; a few ulps for short expressions). The comparand is
   never a stored engine output.
3. **Reference-implementation agreement.** Both engines are compared to one
   engine-independent reference — a closed-form dense formula, or a published
   third-party implementation at a pinned commit. The bound is the engine's
   declared solver tolerance plus fixture conditioning (a κ·eps term computed
   per fixture), re-derived per engine, never copied from the other engine.
4. **Per-engine self-consistency.** Exact equality between two executions of
   the *same* engine in the same environment. Never a cross-engine
   comparison.
5. **Statistical band.** Coverage-style claims receive binomial/Wilson
   acceptance intervals derived at the test's own sample size and seeds;
   realized values from the frozen engine are directional evidence only.
6. **Measured-variance noise floor.** Where a cross-engine numeric comparison
   is unavoidable, its band is at least the frozen engine's *measured*
   cross-environment spread (see below). Nothing model-mediated is pinned
   tighter.

Two corollaries the suite enforces:

- **Fixture-arithmetic numbers are not apparatus.** An exact constant that is
  hand-derivable from a fixture's first principles without running any engine
  is a legitimate exact assertion — the test re-derives it (the derivation
  travels with the assertion); it never copies a measured engine output.
- **Engine-measured tolerances are re-derived, never copied.** Any inherited
  numeric tolerance (a solver rtol, a float atol, a sanity threshold) is
  replaced by its derivation — solver-declared tolerance, conditioning bound,
  statistical band, or data-derived extremes — evaluated for the new engine.

## Never compare floats across engines

**Standing rule: no oracle test asserts float equality — at any tolerance
minted from same-engine behavior — between the rewrite's outputs and the
frozen engine's outputs.**

The standing reason is measured, not theoretical: the frozen engine, at one
pinned tag, one configuration, and one dependency set, produces end-to-end
cost scalars that differ by roughly 0.4% across CPU architectures —
instruction-set-dependent fused-multiply-add and SIMD paths, libm variants,
and numeric-backend differences compound through model fitting into every
downstream number. Same-engine floats do not survive an architecture change;
they cannot define a cross-engine target. Consequences:

- Oracle properties are **behavioral**: structural identities, refusal
  contracts, monotonicity, set membership, closed-form recomputation on
  shared fixtures — never "matches the old number within eps".
- Any unavoidable cross-engine numeric sanity band is at least the ~0.4%
  measured cross-environment spread (tolerance class 6), and is a sanity
  check, never an equivalence claim.
- Byte- and bit-level anchors never cross engines under any tolerance.

## Keep same-engine exactness exact, per engine

Zero-tolerance equality remains the right assertion *within* one engine, one
build, one environment — and is re-instantiated inside the rewrite as
self-consistency templates (tolerance class 4):

- **Resumed == uninterrupted**: a run killed after any origin and resumed
  from persisted calibration state and partial ledger yields a final ledger
  identical to the uninterrupted run.
- **Distributed == sequential**: fixed tasks, data, origins, and seed yield
  an identical ledger across any backend, batching, or concurrency
  configuration.
- **Serialized == never-serialized**: calibration state round-tripped through
  persistence yields bit-identical subsequent bounds `[CAL-2]` — any
  round-trip that changes an output bit is a bug regardless of magnitude.
- **Same seed == same bytes**: seeded reruns of one build on one platform are
  bitwise reproducible.

These templates transfer as *claim shapes*; their exactness is never
inherited by any cross-engine comparison. If the rewrite adopts a
non-deterministic reduction anywhere, the affected template degrades to that
engine's own measured single-node repeat variance — measured, not assumed.

## Classify checkpoints as protocol-portable or engine-internal

Every cross-engine checkpoint is ruled into one of two kinds before it is
built:

- **Protocol-portable** — a statement about an external protocol or input
  data, true of any correct engine: dataset integrity facts, reveal/anchor
  timing of a backtest protocol, cost-accounting identities, closed-form
  transform definitions. Exact, tolerance-free, directly reusable.
- **Engine-internal** — equality between two code paths, physical row
  orders, byte layouts, or float trajectories of one engine. Never crosses
  engines; at most re-instantiated inside the rewrite as class-4 templates.

The designated mechanism for cross-engine end-to-end checking is
**conditional replay**: capture the frozen engine's *decision stream* (the
orders it committed, per round) at the pinned tag, feed the identical stream
plus revealed actuals to the rewrite's settlement arithmetic, and require
the resulting cost trajectory to match an expected trajectory
**independently recomputed from the same shared inputs** — the decision
stream plus revealed actuals pushed through the protocol's cost-accounting
identities, a tolerance-class-2 closed-form recomputation. The expectation
is never read from the frozen engine's stored ledger; the match is exact up
to summation rounding. The model-in-the-loop part (which orders get
produced) is exactly the non-portable part; unconditioned end-to-end totals
are never compared tighter than the class-6 floor.

## Refuse bug-for-bug fidelity

Bug-for-bug fidelity is a **non-goal**. Where a frozen-engine behavior has
been ruled a defect, the rewrite implements the ruled-correct behavior, and
the equivalence harness *expects divergence* on that surface — cross-engine
disagreement there is the fix working, and agreement is grounds for
suspicion, not comfort.

The governing instance: the metric layer's actual value denotes **demand**.
The frozen engine resolves ledger actuals from recorded values that on real
retail data are censored sales, so its realized coverage numbers on such data
are sales-scored by construction. Therefore: oracle properties about
calibration honesty are stated over demand; equivalence fixtures supply true
or synthetic demand, never raw censored sales; every sales-scored real-data
number from the frozen engine is apparatus; and cross-engine agreement on a
sales-resolved ledger certifies accounting mechanics only, never coverage
honesty. Where a dataset cannot supply demand, the scoring label is owned by
that dataset's protocol chapter (chapter 21 states the ratified
sales-coverage exemption, `[M5-X1]`–`[M5-X5]`).

Coverage acceptance at scale asserts the per-node and per-level statistics
defined structurally in chapter 02 ("hierarchical coverage"), with bands
derived from sampling variance at the run's own scored-row count `[LED-5]`
and a minimum scored-ratio floor. Chapter 41 (40-gated-seams/) rules
lattice-wide joint or simultaneous coverage claims inadmissible
(`[SEAM-5]`, `[SEAM-6]`); no oracle property or gate mints one. The only
admissible conditional claim is class-conditional coverage, scored per its
registered ledger predicate (chapter 02) at decision scope; per-node and
per-level statistics remain diagnostics.

## Ship a witness with every numeric gate

Every numeric gate the rewrite mints — any assertion of the form "value
within band" — ships a **witness test**: a cheap companion that perturbs the
input or output by the smallest drift the gate exists to catch (one cost-rate
quantum on one period, one unit of demand, one miscounted row) and proves the
gate *fails*. A tolerance that cannot reject the smallest meaningful drift is
not a gate; it is decoration. Witnesses are collected and run with the gates
they guard, and a gate merged without its witness fails review.

## Stop without a pre-change baseline

Binding stop condition for physical-equivalence checks: a check asserting
byte-level or physical-row-order identity across a change is **binding only
if a pre-change baseline captured from the same engine in the same
environment exists**. If no such baseline exists: stop; never infer
equivalence; never encode the check as a collected-but-skipped test that
green-washes the suite. The equivalence question is then answered
structurally (behavioral properties over the same surface) or not at all.

## Structure the CI gates

| Tier | Content | Tolerance classes | Cadence | Fate at cutover |
|---|---|---|---|---|
| 0 | Static gates: lint, types, frame schema validation `[FRA-3]` | — | Every commit | Permanent |
| 1 | Oracle-property suite on synthetic fixtures: the carried corpus as fast unit/property tests | 1, 2, 3 | Every commit | Permanent — outlives the oracle |
| 2 | Self-consistency suite: resume, serialization, distribution invariance, seeded determinism | 4 | Every merge | Permanent, per-engine |
| 3 | Cross-engine equivalence harness: conditional replay against pinned-tag oracle captures; the frozen engine's side of reference-implementation comparisons | 2, 3, 6 | Scheduled / pre-release | Deleted with the oracle |
| 4 | Protocol acceptance at scale (statistical coverage bands, scored-ratio floor, machine-readable summary, nonzero exit on non-PASS); the rewrite's side of reference-implementation gates (engine-independent references — closed-form, or third-party implementations at pinned commits) | 3, 5 | Scheduled | Permanent — statistical bands re-derived per run scale, reference pins immutable |

Lifecycle rules:

- Tier 1 is the suite's center of gravity: it must run without the oracle,
  without network access, and without any capture manifest — it is what
  outlives cutover.
- Tier 3 consumes only manifest-complete captures at the pinned oracle tag;
  it is the *only* tier allowed to touch frozen-engine outputs.
- No apparatus of the frozen engine (pinned scalars, sha-pinned baselines,
  config pins) is imported into any tier. Retired means deleted.
- Every statistical tier-4 gate re-derives its bands from the run's own
  sample sizes at execution time; no band constant is committed as a literal
  without its derivation alongside. A tier-4 reference-implementation gate
  instead carries the engine's declared solver tolerance plus the
  per-fixture conditioning term (class 3), re-derived for the engine under
  test.
- Reference-implementation gates are per-engine instances of one
  engine-independent reference: the rewrite's scheduled-gate instance is
  permanent (tier 4; fast closed-form instances run as tier-1 property
  tests); the frozen engine's instance exists only as build-time
  triangulation evidence (tier 3) and retires with the oracle. The
  reference itself — a closed-form formula, or a third-party implementation
  at a pinned commit — is never frozen-engine apparatus.
- Every numeric gate in any tier ships its witness (above) in the same tier.

## State the acceptance criteria

A conforming test estate demonstrates:

1. Every test in the repository is traceable to a ruled class; no assertion
   compares rewrite floats to numbers minted from frozen-engine runs at any
   tolerance (the conditional-replay expectation is independently recomputed
   from shared inputs, never read from a stored frozen-engine ledger).
2. Each carried oracle property names its tolerance class, and every
   non-exact tolerance in the suite carries its derivation in-source.
3. The class-4 templates (resumed==uninterrupted, distributed==sequential,
   serialized==never-serialized, same seed==same bytes) pass on the rewrite.
4. The conditional-replay checkpoint passes: the captured oracle decision
   stream replayed through the rewrite's settlement reproduces the
   independently recomputed cost trajectory to summation rounding.
5. On a censoring-ruled surface, the harness asserts *divergence* from the
   frozen engine under a demand-scored fixture — the fix-is-working check.
6. Every numeric gate's witness fails the gate under its smallest meaningful
   drift; deleting the witness makes CI fail.
7. Tier 1 passes on a clean checkout with the oracle and its captures absent.
8. No physical-equivalence check exists without a manifest-complete
   pre-change baseline.

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the frozen engine: `tests/` (the classified assertion estate, seventeen
families), `tests/infra.py` (the extracted closed-form reconciliation
reference), `tests/fixtures/m5/` (designated shared input corpus),
`tests/benchmarks/test_vn2_regression.py` (the gate-bites witness pattern,
carried; its pinned scalars, retired), `benchmarks/cp/aci/` (the
reference-implementation gate pattern: third-party published trace at a
pinned commit). Negative space: `tests/benchmarks/test_vn2_cli_parity.py`
(the checkpoint-decomposition template — most of its checkpoints ruled
engine-internal), the streaming byte-check runbook in `benchmarks/m5/`
(local-only never-committed baseline — the stop rule exists because of it),
and `tests/baselines/m5/` (frozen continuity artifact whose coverage figures
are engine-measured *and* sales-scored — doubly non-oracle).
