---
title: "Conformal plugins — the method protocol"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 05 — Conformal plugins

This chapter owns the **conformal-method protocol**: the contract a calibration
method implements to turn point forecasts in a forecast frame into calibrated
bounds, maintain calibration state per partition, and learn from resolved
actuals. It uses the chapter 02 vocabulary verbatim (forecast frame, partition,
calibration state, session, ledger, scored/unscored, readiness) and cites its
invariant tags. Everything specified here is a **per-partition, marginal**
contract; claims that are joint across partitions or hierarchy nodes are
bound by chapter 41 (40-gated-seams/) — see "Declare joint claims" below.
Requirements carry `[CNF-n]` tags for citation by tests and later chapters.

## Expose one stable runtime seam

The engine's calibration stage depends on exactly one interface: the
**conformal runtime seam** defined in this chapter. Conformal methods are
plugins behind that seam; the internals a method composes — score functions,
quantile rules, window structures, level controllers — are method-private,
carry no stability promise, and may churn freely without touching the engine.

- `[CNF-1]` The engine imports the seam and the registry, and nothing else,
  from the conformal layer. No engine or authoring code may reach past the
  seam into a method's internals; a method's internals may not import engine
  orchestration.

## Define the method lifecycle

A conformal method implements three verbs over calibration state:

| Verb | Signature (semantic) | Role |
|---|---|---|
| `calibrate` | `(scores, config) → state` | Seed state from historical nonconformity scores. Optional: with no scores, state starts empty (cold start) and warm-up governs issuance. |
| `apply` | `(frame, state) → (frame′, state′)` | Wrap each admissible frame row's point forecast in bounds and stamp issuance metadata. |
| `observe` | `(resolved rows, state) → (state′, annotations)` | Ingest resolved actuals: update score windows and level state; annotate each consumed row with its nonconformity score. |

- `[CNF-2]` `apply` consumes a schema-valid frame `[FRA-3]`, emits bounds as
  the nominal-level column pair `[FRA-2]`, and never reads actual values. Its
  only state change is issuance bookkeeping `[CAL-3]`; it never advances a
  score window.
- `[CNF-3]` `observe` consumes only resolved rows admissible under temporal
  hygiene `[INV-TEMPORAL]`, `[CAL-3]`. It must be deterministic given the
  sequence of rows it is fed; the engine feeds resolved rows in one canonical
  order (chapter 06 owns resolution ordering and buffering). A method whose
  state is order-sensitive (rolling windows, feedback controllers) declares
  so in its manifest; it never sorts internally to compensate. Each resolved
  row arrives with its censoring status attached (chapter 06,
  `[OBS-30]`–`[OBS-32]`); which values may enter a score window is owned by
  the score-input contract `[CNF-26]`–`[CNF-28]`.
- `[CNF-4]` Every issued row is stamped with: method name, emission form and
  scope (below), the working level at issue time, the partition label, a
  state reference encoding `(method, issue counter, partition)`, and the
  readiness decision `[CAL-4]` — so every unscored row is attributable from
  the ledger alone `[LED-7]`.
- `[CNF-5]` The lifecycle is identical in backtesting and live inference:
  same verbs, same state, two drivers (chapter 03; `[SES-3]`).
- `[CNF-29]` **Calibration context.** The engine supplies a read-only
  **calibration context** to `calibrate`, `apply`, and `observe` when the
  manifest declares `consumes-calibration-context` (chapter 41, `[SEAM-7]`);
  it is absent by default. The context carries the hierarchy facts of the
  rows in hand — series key, lattice level, and aggregate-node memberships,
  derived from the run's static aggregation lattice — the same hierarchy
  index reconciliation strategies receive (chapter 07, `[REC-6]`). A
  method declaring `joint_claim: class-conditional` must declare the
  context, and its class system must be formable from the context alone.

## Lay out per-partition state

- `[CNF-6]` Calibration state is keyed by `(session, partition)` and nothing
  else `[CAL-1]`; each state row is self-contained, independently restorable,
  and round-trips to behavioral equality `[CAL-2]`.
- `[CNF-7]` State is split by scope. **Partition-scope** fields (score
  windows, per-partition readiness) live in that partition's row.
  **Method-scope** fields (working level, issue counter, controller state)
  live once, under a reserved partition label — never duplicated into every
  partition row. The parenthesized examples are illustrative, not
  exhaustive: a per-partition level controller is conforming
  partition-scope state. Method-scope state under the reserved label may
  legitimately update on any observe cycle and is exempt from the
  partition-scoped persistence isolation of chapter 06 (`[OBS-21]`).
  Duplication forces a merge heuristic on restore
  ("take the most-advanced copy"), which can resurrect stale state; it is
  forbidden.
- `[CNF-8]` Label hygiene: partition labels are derived from `(model name,
  partition value, horizon scope)` by an injective encoding, and reserved
  labels are structurally disjoint from every data-derivable label — a series
  key must be unable to forge a reserved label, and a reserved label must be
  unable to collide with a partition.
- `[CNF-9]` Restoration is factory-only: persisted state rows construct a
  fresh method instance. A live instance is never mutated into a restored
  one, so persistence and the running runtime can never alias.

## Declare hosted sub-models

A method may host an internal predictive sub-model — a score forecaster, a
residual regressor, an ensemble sidecar — among its method-private
internals.

- `[CNF-30]` A hosted sub-model is declared in the method's manifest
  `[CNF-23]`. It is method-private: it may not reach the chapter 04
  adapter spine, and no engine or authoring surface addresses it directly —
  a private model stack inside the plugin is the granted route.
- `[CNF-31]` Stochastic internals take their seed as configuration, subject
  to parity `[CNF-16]` — mirroring the forecasting-adapter rule (chapter
  04, `[ADA-2]`). `calibrate` is deterministic given `(scores, config)`; a
  hosted sub-model does not weaken the `observe` determinism contract
  `[CNF-3]`.

## Emit bounds: declared form and scope

A method declares its **emission form** and **emission scope** in its
manifest; consumers (ordering, scoring) read the declaration, never guess.

- `[CNF-10]` **Two-sided** form: both bound columns of the nominal coverage
  level `[FRA-2]` are populated; the default claim is
  `P(lower ≤ actual ≤ upper) ≥ 1 − α` per partition, in the
  finite-sample-marginal currency, valid under the method's
  declared assumption class, where `α` is the miscoverage rate
  (`α = 1 −` nominal coverage level).
- `[CNF-11]` **One-sided** form: the method declares its guaranteed side
  (upper or lower) and the default claim, likewise finite-sample-marginal,
  attaches to that side only (upper case:
  `P(actual ≤ upper) ≥ 1 − α`). The undeclared side is filled with the
  declared support bound of the target (for non-negative demand, `0.0`) —
  finite, so the row also evaluates under the two-sided diagnostic
  predicate `[LED-4]`, and non-binding, so that predicate coincides with
  the row's own one-sided predicate (`[LED-8]`, chapter 02) on the declared
  support. Filling the undeclared side with any value that can bind (the
  point forecast, a running minimum) is non-conforming: it silently converts
  a one-sided guarantee into an unverifiable two-sided score.
- `[CNF-32]` The `[CNF-10]`/`[CNF-11]` probability statements are the
  **default claims**, stated in the finite-sample-marginal currency. A
  method whose mathematics delivers a different currency —
  *long-run-pathwise*, or *approximate-with-declared-slack* with the slack
  declared numerically — declares that currency in its manifest guarantee
  declaration `[CNF-23]`, and every descriptor it issues restates the claim
  in that currency. The assumption-class label `[CNF-13]` qualifies the
  claim; it never substitutes for the claim statement. A
  sequential-adaptive method whose mathematics cannot deliver the
  finite-sample-marginal claim must not declare it.
- `[CNF-12]` **Emission scope**: *per-step* (one bound pair per frame row —
  per horizon step) or *window-sum (over the protection window)* (one bound
  pair on the sum of actuals over the leading `P` horizon steps — the
  protection window, whose derivation from decision cadence is owned by
  chapter 08 — emitted on the window's terminal row). *Per-step* /
  *window-sum* is the canonical term pair for this manifest field, used
  verbatim wherever the scope is named; the decision chapters consume the
  window-sum form. A window-sum method's non-terminal rows carry no bounds;
  they are unscored with attributed cause *emission scope* (distinct from
  warm-up `[LED-6]`), recorded via the manifest declaration.

## Declare assumptions and readiness

- `[CNF-13]` Every method's manifest declares its **assumption class**, one
  of: *exchangeable* (scores exchangeable within a partition),
  *weighted* (a declared reweighting of scores — e.g. recency — with the
  matching finite-sample correction), or *sequential-adaptive* (no
  exchangeability assumed; the target is pursued by feedback on realized
  errors). The `[CNF-10]`/`[CNF-11]` claims are conditional on the declared
  class holding; the declaration is what a run report cites.
- `[CNF-14]` Every method declares a **calibration requirement**: the minimum
  count of resolved scores `n_min(α, config)` a partition's window must hold
  before a finite bound may be issued. The requirement must equal the
  method's finite-sample mathematics — the smallest window for which the raw
  bound is finite — never a looser constant. Reference case, restated: a conservative split-conformal
  upper rank is finite iff `α > 1/(n+1)`, so a 90%-nominal bound requires at
  least 10 resolved scores in the partition.
- `[CNF-15]` Before readiness, `apply` issues the row with non-finite bounds:
  the row is unscored, attributed *warm-up* `[LED-6]`, and the readiness
  decision is persisted per issued row `[CAL-4]`, `[LED-7]`. No finite
  fallback value is ever substituted unless an explicit clamp (below) is
  configured — a silent fallback manufactures coverage the method never
  earned.

## Reach every field from the authoring layer (config parity)

- `[CNF-16]` **Full parity**: every field of a method's runtime configuration
  is reachable from the authoring layer (chapter 10) under the same name.
  Defaults are defined exactly once, on the runtime config; the authoring
  layer passes fields through without re-defaulting.
- `[CNF-17]` No omitted-versus-null divergence: where "absent" and "explicitly
  disabled" mean different things, the disabled meaning is an explicit enum
  value, never a null whose presence must be distinguished from omission.
- `[CNF-18]` Parity is machine-checked: a test enumerates the runtime config
  fields and the authoring schema fields for every registered method and
  fails on any mismatch, so parity cannot drift silently.

Negative space this closes: the previous engine's authoring layer threaded
only the fields its execution loop happened to need, leaving four runtime
fields of one bundled method unreachable from any config file. Unreachable
fields are untunable, invisible in run records, and silently pin behavior
to defaults nobody chose.

## Treat clamps and buffers as opt-in, guarantee-affecting config

A **clamp** is any post-hoc modification of a method's raw calibrated bound:
floors, caps, non-negativity clips, inflation factors, or substitution of a
fallback value where the raw bound is non-finite.

- `[CNF-19]` Default configuration applies **no clamp**. Every clamp is a
  named, per-method config field, off by default, subject to parity
  `[CNF-16]`.
- `[CNF-20]` A clamp that binds changes the guarantee, direction-aware, and
  the manifest documents the impact per clamp: a cap binding on the
  guaranteed side voids the `≥ 1 − α` claim for that row; a floor on the
  guaranteed upper side widens the bound conservatively. What the issued
  descriptor states is owned by chapter 41: any clamp on the decision bound
  rewrites the descriptor's claim to none (not engine-calibrated)
  (`[SEAM-8]`) — the direction-aware impact documented here qualifies the
  mathematics of the clamped bound, never the descriptor it carries.
- `[CNF-21]` Binding is recorded: each issued row records, per active clamp,
  whether it bound; diagnostics report the binding rate per partition. A
  coverage figure reported without its clamp-binding rate is non-conforming.
- `[CNF-22]` A non-finite raw bound is a readiness failure, not a clamping
  opportunity: the row goes out unscored and attributed `[CNF-15]`. A method
  whose mathematics can legitimately yield a non-finite bound *after* its
  calibration requirement is met (e.g. a weighted method whose requested
  level falls into held-out weight mass) declares that post-warm-up emission
  mode in its manifest; an undeclared post-warm-up non-finite emission is
  defective. A declaring method must not require a cap to be consumable;
  requiring one makes the clamp mandatory in practice, which contradicts
  `[CNF-19]`.

## Declare joint claims

Every guarantee in this chapter is per-partition and marginal. The manifest
field `joint_claim` is bound by chapter 41 (`[SEAM-5]`, `[SEAM-6]`): its
admissible values are exactly `none` (default) and `class-conditional`;
lattice-wide joint or simultaneous claims are inadmissible under every
value of the field. A `class-conditional` claim declares coverage
conditional on membership in a **finite-dimensional class system** — a
declared, finite set of series groups derived from the run's aggregation
lattice, fixed at calibration time — and its issued descriptors carry the
class-conditional coverage claim with that class system named. A method
declaring `joint_claim: class-conditional` must declare the calibration
context `[CNF-29]`, from which its class system must be formable
`[SEAM-7]`.

## Register methods

- `[CNF-23]` Methods register by name in a **method registry**:
  `name → (manifest, config schema, factory)`. The manifest is the single
  declaration surface: method name, emission form `[CNF-10]`/`[CNF-11]`,
  emission scope `[CNF-12]`, assumption class `[CNF-13]`, **guarantee
  declaration** (the descriptor claims the method can emit — each a claim
  plus a currency in the chapter 02 `[GRT-2]` vocabulary — defaulting to
  one-sided coverage in the finite-sample-marginal currency, a
  risk-control claim additionally naming its declared loss), calibration
  requirement `[CNF-14]`, order-sensitivity `[CNF-3]`, censoring policy
  `[CNF-26]`, calibration-context consumption `[CNF-29]`, hosted
  sub-models `[CNF-30]`, in-sample residual requirement (whether
  `calibrate` seeds from the fitted-values sidecar `[FRA-5]` — the
  declaration chapter 04's fit flag `[FIT-1]` reads), post-warm-up
  non-finite emission `[CNF-22]`, clamp
  inventory with guarantee impact `[CNF-20]`, `joint_claim` (chapter 41,
  `[SEAM-6]`), and a state schema version. A method may not emit a bound
  whose descriptor states a claim its manifest does not declare.
- `[CNF-24]` Three method families coexist behind the seam from day one:
  split-conformal (fixed-level, exchangeable), weighted (reweighted scores
  with finite-sample correction), and sequential-adaptive (feedback-driven
  level control). Wrapping a substrate library's marginal conformal
  machinery inside a method is conforming method-private internals;
  transformation of calibrated bounds at the reconciliation stage is
  excluded by chapter 41 `[SEAM-2]` (chapter 07 carries the boundary
  note). The candidate-method survey per family is private:
  [ANNEX:05-method-families-survey].
- `[CNF-25]` Adding a method touches the registry and the method's own module
  only; engine and authoring layers pick it up generically (registration
  data plus config schema), with no engine diff.

## Contract the score-input series

A nonconformity score compares an issued bound to a realized value; *which*
series supplies that value is a declared contract, never an ambient
assumption. Chapter 06 delivers every resolution with its censoring status
attached and keeps the recorded series, the censoring facts, and any
demand-honest resolution distinct (`[OBS-30]`–`[OBS-32]`); this section owns
what a method may score.

- `[CNF-26]` Every method's manifest declares its **censoring policy**, one
  of: *requires-uncensored* (scores are meaningful only against values
  admissible as demand), *consumes-censoring-facts* (the method ingests
  censoring facts natively — the status plus the optional numeric
  availability bound, chapter 02 `[PAN-3]` — and its mathematics accounts
  for lower-bound observations), or *imputation-consumer* (the method
  scores a series produced by a named imputation policy, itself an explicit
  configuration act).
- `[CNF-27]` **Exclusion is the default.** A delivered resolution whose
  censoring status is *censored* is excluded from the score window, with a
  machine-readable attributable cause recorded through the ledger's
  unscored-cause mechanism (`[LED-6]`, `[LED-7]`); an excluded observation
  contributes no score toward the calibration requirement `[CNF-14]`. A row
  excluded with attributable cause contributes no score and does not advance
  delivered-score accounting (chapter 06, `[OBS-15]`). The only alternatives
  are explicit configuration acts matched to the manifest:
  native handling under a *consumes-censoring-facts* declaration, or a named
  imputation policy for an *imputation-consumer* method. Feeding a censored
  recorded value into a score window as if it were demand is a spec
  violation, whatever the configuration. The default applies to composite
  scored quantities too: a window-sum observation `[CNF-12]` whose protection
  window contains a censored member is itself censored — its summed recorded
  value is a lower bound. An observation of *undeclared* status is not
  excluded — it scores at its recorded value — but it taints the
  scored-series label `[CNF-28]`.
- `[CNF-28]` **Scored-series labeling.** Every calibration state row and
  every coverage figure derived from it records which series it scored:
  *demand-honest* (every scored observation was declared uncensored, consumed
  natively under a *consumes-censoring-facts* policy, or produced by a named
  imputation policy) or *recorded-sales* (the window scored at least one
  observation of undeclared censoring status at its recorded value). The
  label rides state persistence and every derived report; the protocol
  chapters' sales-coverage labeling (chapters 20, 21) and the chapter 50
  censoring doctrine consume it — presentation and repair theory live there,
  not here.

## Acceptance criteria

1. A protocol-conformance suite runs against **every** registered method:
   `apply` on a frame with a poisoned actuals column never reads it
   `[CNF-2]`; state round-trip yields behavioral equality `[CAL-2]`,
   `[CNF-6]`; restore is factory-only `[CNF-9]`; `observe` replayed in
   canonical order is deterministic `[CNF-3]`; `calibrate` re-run on
   identical `(scores, config)` yields behaviorally equal state `[CNF-31]`.
2. For each method, issuing below its declared `n_min` yields non-finite
   bounds attributed *warm-up*, and at exactly `n_min` yields finite bounds
   `[CNF-14]`, `[CNF-15]`.
3. A one-sided method's undeclared side equals the declared support bound,
   and on synthetic data the one-sided predicate (`[LED-8]`) agrees with
   the `[LED-4]` band predicate on the declared support `[CNF-11]`.
4. The parity test enumerates runtime-config versus authoring-schema fields
   for every registered method and passes `[CNF-16]`–`[CNF-18]`.
5. With all clamps off, emitted bounds are byte-identical to a build with the
   clamp code absent; with a clamp on, every issued row carries its binding
   record and diagnostics report the per-partition binding rate
   `[CNF-19]`–`[CNF-21]`.
6. Registering a trivial fixture method requires no engine or authoring diff
   and is immediately runnable from a config file `[CNF-25]`.
7. Registry validation rejects a manifest whose `joint_claim` is neither
   `none` nor `class-conditional`, and rejects `class-conditional` without
   `consumes-calibration-context` (chapter 41, `[SEAM-6]`, `[SEAM-7]`);
   issuance of a bound whose descriptor states a claim outside the
   manifest's guarantee declaration is rejected `[CNF-23]`, `[CNF-32]`.
8. A delivered resolution declared censored is excluded from every score
   window by default, carries its attributable unscored cause, and adds
   nothing to the calibration requirement; a window-sum observation whose
   protection window contains a censored member is likewise excluded; every
   calibration state row carries its scored-series label, and a window that
   scored an undeclared-status observation is labeled *recorded-sales*
   `[CNF-26]`–`[CNF-28]`.

## Provenance

For spec authors only; the chapter stands without these. **Positive space**
from the old engine: `calibre/conformal/runtime.py` (the single stable
pipeline-facing seam — apply/observe/resume-state protocol, per-partition
state rows, factory-only restoration, issue-counter state references — while
the rest of `calibre/conformal/` stayed experimental; the lesson behind
`[CNF-1]`, `[CNF-9]`); `calibre/conformal/protocols.py` and `calibrators.py`
(score/calibrator/controller decomposition; the finite-sample readiness test
`α ≤ 1/(n+1)` behind `[CNF-14]`); the old engine's one-sided window-sum
method (emission behind `[CNF-11]`/`[CNF-12]`). **Negative
space**: the authoring layer reached only a subset of that method's runtime
config, leaving several runtime fields unreachable from
YAML (→ `[CNF-16]`–`[CNF-18]`) and used field-was-set introspection to
distinguish null from omitted (→ `[CNF-17]`); one bound cap was a de facto
mandatory clamp for one mode, with no per-row
binding record (→ `[CNF-19]`–`[CNF-22]`); the one-sided runtime filled its
undeclared lower bound with a value that can bind
under two-sided scoring (→ `[CNF-11]`); shared controller state was
duplicated into every partition row and merged back by a most-advanced-row
heuristic (→ `[CNF-7]`); and no method declared a censoring policy —
`observe` scored whatever resolved value arrived, with no censoring field
and no scored-series label on state or coverage (→ `[CNF-26]`–`[CNF-28]`).
