---
title: "Tuning — joint hyper-parameter search across model, conformal, and ordering"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 09 — Tuning

This chapter is the contract for hyper-parameter optimization: what a search
candidate is, what objective a study minimizes, which quantities may never be
searched, how trials are evaluated and distributed, and how a search space is
declared. It consumes chapter 02 vocabulary verbatim, the forecasting-plugin
protocol (chapter 04), the conformal-method protocol (chapter 05), and the
ordering/cost contract (chapter 08). Invariants carry `[TUN-n]` tags for
citation by tests and later chapters.

## Define the tuning candidate

A **tuning candidate** is the unit of search: one immutable, serializable
value holding proposed parameter overrides for all three pipeline channels.

- `[TUN-1]` A candidate has exactly three channels, each a mapping of
  parameter names to values:
  1. **model channel** — overrides merged into the forecast task's model
     configuration (chapter 04);
  2. **conformal channel** — overrides applied to the conformal method's
     configuration (chapter 05);
  3. **ordering channel** — overrides applied to the ordering policy's
     parameters (chapter 08).
  A channel may be empty; the candidate object always carries all three.
- `[TUN-2]` One trial evaluates one candidate as a single joint point in the
  product space of the three channels. The engine provides no default
  machinery that tunes a channel in isolation: cross-channel interactions
  (a model change shifting calibration widths shifting order quantities) are
  observable only under joint evaluation, so joint evaluation is the primitive
  and any per-channel study is a degenerate case (two empty channels).
- `[TUN-3]` A candidate is reconstructible from its recorded trial parameters
  alone: replaying the study's search-space declaration against the recorded
  parameter assignment yields a value-equal candidate. A study's best result
  is therefore durable and auditable without the study process.

A **study** is one optimization run: a search-space declaration, an objective,
a trial budget, a seed, and an evaluation window (a list of origins). Its
result is the best candidate plus a trial table recording, per trial, the
candidate parameters, the objective value, and the completion status.

## Bind the objective

- `[TUN-4]` A **tuning objective** is a pure function from a trial's resolved
  ledger rows (and the applicable cost structure) to a single finite or
  infinite scalar, lower is better. Objectives consume only resolved rows per
  `[LED-2]`/`[LED-4]`; pending rows never contribute. A study record carries
  the dataset's censoring status and the objective's actuals binding —
  demand-honest or labeled surrogate. When the dataset declares censoring
  facts, the default objective binds its actuals to the demand-honest series
  or refuses with an attributable cause (chapter 08, `[OBJ-6]`).
- `[TUN-5]` **The default objective is the chapter 08 realized-cost
  objective** — the symbol chapter 08 exports as its default, the
  settle-path realized cost `[OBJ-2]`. The binding is symbolic — this
  chapter cites the exported symbol and states no formula; chapter 08 owns
  its definition and guarantees it is computable per candidate inside a
  tuning loop (cost as an optimization objective, not just a report). If
  chapter 08's exported default changes, the tuning default follows with no
  change to this chapter.
- `[TUN-6]` Objective selection is configuration in the authoring layer
  (chapter 10). Realized cost must be reachable purely from configuration —
  no code required — and alternative objectives (point-accuracy metrics,
  quantile losses) are selectable but never silently substituted for the
  default. A study result must record which objective it minimized.

## Bind the default objective

The exact functional the default objective evaluates is owned by chapter 08
and bound by chapter 41 (`40-gated-seams/`) `[SEAM-4]`: realized cost is
defined per decision node, and the engine defines no lattice-level aggregate
cost functional. This chapter's dependence stays symbolic — it consumes the
scalar realized-cost objective chapter 08 exports, lower-is-better, evaluable
per candidate — and nothing here restates or constrains what that objective
computes.

## Never tune decision numbers

In default tuning, decision-bearing quantities are **derived from the cost
structure, never searched**. Two distinct rationales apply, and they must
not be conflated. Cost-structure components are banned for **objective
integrity**: they parameterize the objective itself, so a searched component
lets the optimizer move its own goalposts — the "best" trial wins by
changing what the objective measures. The decision fractile and the
policy-consumed coverage are banned from default tuning as **pinned
policy-class identity**: the critical ratio is the definitional identity of
the newsvendor policy class, derived from the cost structure — pinning it
asserts what the policy *is*, and carries no claim that this fractile is the
multi-period optimum.

- `[TUN-7]` The **cost fractile** used by a newsvendor-family decision rule is
  the cost structure's critical ratio `[CST-2]`. It may be overridden only
  through the single sanctioned explicit-fractile override field chapter 08
  defines (`[CFG-6]`, what-if studies); it is never a search dimension in
  default tuning. A degenerate derivation (`[CST-2]` denominator not
  positive, or a ratio outside the open interval (0, 1)) rejects the study at
  validation rather than producing a collapsed objective.
- `[TUN-8]` Search-space validation rejects, before any execution, any
  dimension whose name denotes a decision-bearing or derived quantity — at
  minimum: the cost fractile, the critical ratio, and any decision coverage
  level consumed by an ordering policy (pinned policy-class identity), and
  the cost-structure components themselves (objective integrity). Matching
  is on the final dotted segment of the dimension name, case-insensitively,
  so aliases and namespacing do not evade the guard. This clause owns the
  single normative forbidden-dimension list; chapter 08 (`[OBJ-8]`) defers
  to it. The only sanctioned relaxation is the reference-tuning mode
  `[TUN-24]`, and it relaxes only the identity-pinned names, never the
  objective-integrity ones.
- `[TUN-9]` Search-space validation also rejects dimensions naming
  **structural identity keys** — task scope (local/global), model identity,
  execution backend, horizon, calendar frequency. These define the study, not
  a hyper-parameter; sampling them would let a trial silently change what is
  being compared.
- `[TUN-10]` The guard is defense-in-depth, not the only barrier: derived
  quantities are sourced from the cost structure by construction and are
  never read out of the search space, so even a name that evades `[TUN-8]`
  cannot reach the decision rule.
- `[TUN-24]` **Reference-tuning mode.** One sanctioned exception to the
  identity-pinned half of `[TUN-8]` exists: an explicitly labeled study mode
  that may search decision-channel dimensions — the decision fractile and
  the policy-consumed coverage — to mint **cost-tuned reference
  configurations** for comparison studies. The mode is off by default and
  named in the study configuration; the cost-structure components remain
  banned in every mode (objective integrity is absolute). Its outputs are
  reference configurations, never certified: the guarantee descriptor
  (chapter 02, `[GRT-*]`) of any bound produced under a reference
  configuration carries the claim none (not engine-calibrated) — the
  `[GRT-2]` value stating that no engine claim is made — and the study
  result and every number derived from it are labeled reference-tuned.
  Default tuning keeps the full `[TUN-8]` ban.

## Evaluate a trial as a backtest

- `[TUN-11]` A trial is a real backtest of its candidate through the same
  engine as chapter 03 — task build, fit, predict, calibrate, order, observe,
  ledger — over the study's origins in order. There is no separate "tuning
  scorer": temporal hygiene `[INV-TEMPORAL]`, frame validation `[FRA-3]`, and
  ledger semantics hold inside a trial exactly as in a standalone run.
- `[TUN-12]` The objective accumulates incrementally: after each origin, the
  trial scores the rows newly resolved at that origin — each resolved row
  contributes exactly once per study (no double counting across origins) —
  and reports the running objective with the origin index as the progress
  measure. This enables budget-aware early stopping: a scheduler may halt
  underperforming trials after a configured grace period measured in
  *origins*, which must be strictly less than the study's origin count.
- `[TUN-13]` Failure semantics distinguish candidate failure from
  infrastructure failure. A candidate that cannot produce a finite objective
  (no resolvable rows, degenerate parameters) completes its trial reporting
  the worst possible objective value; an infrastructure error (cluster loss,
  I/O failure) aborts the study loudly rather than being recorded as one
  quietly-errored trial among many. A study whose trials produced no finite
  objective fails with a diagnosable error; the best trial must have a finite
  objective.
- `[TUN-14]` A trial that tunes the conformal channel starts from the study's
  seed calibration state (the state snapshot taken when the study was built)
  and mutates only its own trial-local copy; trials never share mutable
  calibration state, and a study never writes to any session's durable state
  `[SES-2]`.

## Specify the local/global axis

Tuning scope mirrors task scope `[TSK-1]`: it is configuration, not code.

- `[TUN-15]` A **local study** tunes one series: one study per series key,
  its candidate's model channel configuring a single-series forecast task.
  Tuning a panel locally is a fan-out of independent studies over series
  keys, each with its own best candidate.
- `[TUN-16]` A **global study** tunes one panel-scoped model configuration:
  one study whose trials fit a single model instance over the whole panel and
  whose best candidate applies panel-wide.
- `[TUN-17]` Both scopes use the same study machinery, candidate shape,
  objective protocol, and search-space declaration; the axis is selected in
  the authoring layer. Both scopes must be reachable purely from
  configuration.

## Fan out and resume on Ray

Studies distribute trials on the Ray substrate (chapter 03).

- `[TUN-18]` Every trial declares its resource requirement; study concurrency
  defaults to the cluster budget divided by the per-trial requirement and is
  explicitly cappable. Trials additionally bound their own intra-process
  thread usage to their declared share, so co-scheduled trials do not
  oversubscribe a node.
- `[TUN-19]` Shared read-only trial inputs — the history panel, actuals, the
  seed calibration state — are shipped to the cluster once and referenced by
  every trial, never serialized per trial.
- `[TUN-20]` **Partial-completion resume.** A study is durable: identified by
  a stable study name and persisted incrementally to a configured storage
  location. Resuming an interrupted study with the same name and storage
  location re-runs only trials that did not complete; completed trials'
  results are never re-evaluated. Study results survive process and cluster
  death.
- `[TUN-21]` Reproducibility: the search sampler is seeded from the study
  configuration; identical study configuration, data, and seed produce the
  same candidate sequence in sequential execution, and every recorded trial
  is individually replayable via `[TUN-3]` regardless of concurrency.

## Declare the search space in the authoring layer

- `[TUN-22]` A search space is data, not code: a mapping from dimension names
  to declarative specs, written in the same configuration surface as the rest
  of the pipeline (chapter 10). A dimension name addresses a channel and a
  parameter; each spec is one of: **categorical** over explicit choices,
  **integer range** (low, high, optional step), **float range** (low, high,
  optional step, optional log scaling). An unknown spec type or a malformed
  spec is rejected at validation, never silently ignored.
- `[TUN-23]` The study block (budget, seed, objective selection, grace
  period, search space, scope) validates as part of the pipeline `validate`
  verb, with `[TUN-7]`–`[TUN-9]` enforced there — an invalid study is
  rejected before any data loads or cluster resources are acquired. A config
  carrying a study block remains a valid backtest config; the block is inert
  unless a tuning run is requested.

## Acceptance criteria

A conforming implementation must demonstrate, by test:

1. A candidate round-trips: search-space declaration + recorded trial
   parameters reproduce the identical three-channel candidate `[TUN-3]`.
2. A configuration-only study minimizes the chapter 08 realized-cost
   objective with no code authored, and its result records that objective
   `[TUN-5]`, `[TUN-6]`.
3. Search-space validation rejects a dimension named for the cost fractile,
   the critical ratio, a decision coverage level, or a structural identity
   key — including dotted-alias and case variants — before execution
   `[TUN-8]`, `[TUN-9]`. The same study declared in reference-tuning mode
   accepts the decision-channel dimensions, still rejects a cost-structure
   component, and its result is labeled reference-tuned `[TUN-24]`.
4. A trial's ledger obeys temporal hygiene: a property test shows no
   objective contribution derives from data at or after its origin
   `[TUN-11]`.
5. Per-origin objective accumulation counts each resolved row exactly once;
   an early-stopped trial's partial objective equals the full trial's
   objective truncated at the stopping origin `[TUN-12]`.
6. A candidate failure yields a completed worst-objective trial; an injected
   infrastructure error aborts the study; a study with zero finite-objective
   trials fails diagnosably `[TUN-13]`.
7. Killing a study mid-flight and resuming with the same name and storage
   completes only the unfinished trials, leaving completed trial results
   byte-identical `[TUN-20]`.
8. The same declared study runs in local scope (fan-out over series keys)
   and global scope (one panel study) with no change outside the scope
   setting `[TUN-15]`–`[TUN-17]`.

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the old engine: `calibre/tuning/task.py` (`TuningCandidate` three-channel
split — model/conformal/ordering — and local/global study tasks),
`calibre/tuning/optimizer.py` (Ray Tune + Optuna study, seeded sampler,
origin-indexed incremental reporting with ASHA, shared conformal seed state
shipped once via the object store, candidate replay from recorded trial
params, worst-objective-on-candidate-failure vs. raise-on-infrastructure
semantics), `calibre/tuning/search_space.py` (declarative
categorical/int/float spec), `calibre/cli/config.py::HpoConfig` (the
never-tune guardrail: forbidden decision-number names matched on the final
dotted segment case-insensitively, plus reserved structural keys, enforced at
config validation). Negative space this chapter closes: the old CLI hardcoded
the tuning objective to a cumulative quantile loss at the derived fractile —
the library's realized-cost objective existed but was unreachable from
configuration (`calibre/cli/commands.py::run_tune`), local studies were
reachable only from library code, only one conformal runtime family could be
tuned on the Ray path, and interrupted studies could not resume.
