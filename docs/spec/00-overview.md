---
title: "New-Calibre architecture spec — overview and map (two-layer)"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 00 — Overview

This file is the map of the architecture spec for the greenfield Calibre
rewrite: its purpose, layout, conventions, and reading order. The spec is
**two-layer**:

- **Public layer** — everything under `docs/spec/`. Standalone-
  readable, public-safe, no private substance. This is the layer specified
  below.
- **Private rationale annex** — a maintainers' knowledge base outside the repo.
  Public files never inline annex content; they reference it **by opaque
  pointer only** (see "Annex pointer convention" below). The annex holds
  design rationale, gated Stage 1 decision records, and evidence not yet
  released.

**Vocabulary — "Stage 1" (defined once, here).** "Stage 1" is an opaque gate
label: the pre-publication decision gate behind the `40-gated-seams/`
chapters. The gate has landed: its decisions are ratified and bound into the
tree, and the `stage1-*` invalidation tags that marked the chapters it could
invalidate are cleared. The label carries no further public meaning; readers
need only know that the gate existed and which chapters carry its outcomes.

The old engine (the current `calibre` repo) is a **behavior reference**, not a
design source. Where a chapter below cites old-repo paths, those are provenance
footnotes for spec authors — the chapters themselves must restate every fact so
a reader without the old repo understands them.

## Design rules for the spec tree

1. **One chapter, one contract.** Each numbered file owns exactly one
   architectural surface (a protocol, a runtime concern, or an external
   protocol restatement). Cross-cutting decisions get an ADR under
   `docs/spec/adr/`, referenced from the chapters they bind.
2. **Slots are files, not gaps.** Gated or evidence-pending chapters exist from
   day one as stub files carrying the same frontmatter contract as this
   file (`status`, `invalidation-tags`), so the tree is complete and the
   gate-dependencies are machine-scannable.
3. **Every file carries frontmatter**: `title`, `status`
   (`pre-gate-draft` | `gated-slot` | `evidence-pending` | `draft` |
   `ratified`), `invalidation-tags` (possibly empty), `date`. `draft` is the
   plain post-gate status: a `gated-slot` or `evidence-pending` file moves to
   `draft` once its gate or evidence lands, and onward to `ratified`.
4. **Numbering encodes reading order**, not dependency order; `00` is the map.
5. **No positioning language.** Spec chapters state contracts; no chapter
   claims novelty, priority, or differentiation for any method or
   mechanism. Positioning language fails review.
6. **Requirement-ID prefixes are chapter-owned.** Each chapter owns the
   requirement-ID prefixes it declares; no two chapters share a prefix, and a
   chapter introducing a new prefix registers it in the **prefix registry**
   (below) in the same change. The registry is the complete prefix→owner index
   and the surface on which the no-collision invariant is checked.

## Annex pointer convention

Public files reference the private annex with inline markers of the form
`[ANNEX:<chapter>-<slug>]`, e.g. `[ANNEX:08-cost-objective-derivation]`.
A single registry file (`90-annex-registry.md`) lists every pointer in use with
a one-line public-safe description of what kind of material sits behind it
(e.g. "derivation", "gated decision record", "benchmark evidence").
The registry is the leak-review surface: a pointer not in the registry fails
review; annex material inlined into a public file fails review.

---

## The tree

```
docs/spec/
├── 00-overview.md
├── 01-vision-and-commitments.md
├── 02-domain-model.md
├── 03-engine-core.md
├── 04-forecasting-plugins.md
├── 05-conformal-plugins.md
├── 06-online-recalibration.md
├── 07-reconciliation.md
├── 08-ordering-and-cost.md
├── 09-tuning.md
├── 10-pipeline-authoring.md
├── 11-api.md
├── 12-cloud-native.md
├── 20-protocol-vn2.md
├── 21-protocol-m5.md
├── 30-performance.md
├── 40-gated-seams/
│   ├── README.md
│   ├── 41-decision-calibration-seams.md
│   └── 42-flagship-metric.md
├── 50-test-and-oracle-strategy.md
├── 60-onboarding.md
├── 90-annex-registry.md
└── adr/
    └── README.md
```

---

## Chapter scopes

### `00-overview.md`

The map: spec purpose, the two-layer contract, the frontmatter/status legend,
the annex pointer convention, reading order, and a **vision-coverage matrix**
(the table below, kept current) proving every vision element has an owning
chapter. Consumes: nothing — this file is the map itself. No gated
content.

### `01-vision-and-commitments.md`

Restates the eleven product-vision elements as testable architectural
commitments — Nixtla/Ray core; pluggable forecasting; pluggable conformal;
cost-driven ordering with cost as a first-class tuning objective; cloud-native
K8S scaling; a strong API; easy/clean/fast pipeline authoring; local and
global modelling and tuning; one engine for backtesting and inference; online
recalibration; hierarchical reconciliation — each with a one-line acceptance
criterion and a pointer to its owning chapter. The flagship metric that
headlines the product is bound: chapter 42 (`40-gated-seams/`) owns the
two-axis claim, and this chapter's register points to it. Consumes: the
ratified vision statement; `[ANNEX:01-flagship-metric-decision]`.
Invalidation-tags: none.

### `02-domain-model.md`

The ubiquitous language: series/panel, forecast frame, forecast task, origin,
horizon, hierarchy node, cost structure, session, calibration state, order,
ledger — plus the decision-time vocabulary: lead time, review period,
protection window, open order, inventory position, settlement record,
stock-out transition rule, the per-step / window-sum emission-scope pair, and
the calibration requirement. Defines each term once with its invariants (e.g.
an origin never sees data at or after itself; a hierarchy is a static
aggregation lattice over series keys). All later chapters use these terms
without redefinition: chapter 03 states the settlement *runtime* contract
over these terms, and chapter 08 owns the mapping between the per-decision
(underage/overage) and per-period (holding/shortage) cost pairs.
Also owns the **guarantee descriptor** (`[GRT-*]`): the typed claim every
calibrated decision bound carries. The structural terms coherent cost and
hierarchical coverage are defined by shape here; their normative force is
bound — 08 and chapter 41 for coherent cost, 05/07 and chapter 41 for
hierarchical coverage. Consumes: the chapter 41 bindings.
Provenance footnote: old repo `calibre/core/` (`ForecastFrame`,
`ForecastTask`).

### `03-engine-core.md`

The single engine that serves both backtesting and live inference: Nixtla
libraries as the forecasting substrate, Ray as the distribution substrate, and
the pipeline spine — load → task build → then, per origin: resolve/observe →
predict → reconcile → calibrate → order → commit — with the ledger as the
single scoring surface (actuals resolution runs *before* prediction, so each
origin's intervals reflect everything admissible at it). Specifies why
backtest and inference are one code path with two drivers (a time-loop driver
replaying history vs. an event driver fed by the API), what state crosses
driver boundaries (calibration state, open orders, model artifacts),
determinism requirements (stable session ids, reproducible task ordering),
and the settle hook as a runtime contract (`[SET-*]`, `[STA-*]`): when
settlement happens in the loop, what a settlement record books, and the
stock-out transition rule as configuration rather than engine code — over the
chapter 02 decision terms, never redefining them. Consumes: no gated
decisions.
Provenance footnote: old repo `calibre/execution/backend.py::BackendEngine`,
which conflated orchestration with I/O and is cited only as negative space.

### `04-forecasting-plugins.md`

The model-adapter protocol: what a forecasting plugin must accept (panel,
exogenous regressors, horizon) and emit (point forecasts, optionally in-sample
fitted values and native quantiles), registration/discovery, and the
**local-vs-global axis** — one adapter instance per series versus one
full-panel adapter fanned out over Ray, selected by config rather than by
adapter code. Specifies artifact persistence via native Nixtla persistence
APIs and the fitted-values side channel that residual-based reconcilers
consume. Consumes: chapter 02 vocabulary; no gated decisions.
Provenance footnote: old repo `calibre/forecasting/` registry.

### `05-conformal-plugins.md`

The conformal-method protocol: calibrate(scores, config) → interval/quantile
machinery, per-partition state layout, exchangeability assumptions each method
declares, and the plugin registry so split-conformal, weighted, and
sequential-adaptive families coexist behind one stable runtime interface (the
old engine's lesson: expose one stable runtime seam, keep low-level building
blocks experimental). Claims joint or simultaneous across partitions or
hierarchy nodes are bound by chapter 41 (`40-gated-seams/`): the
`joint_claim` vocabulary, the calibration context, and the decision-scope
rule; this chapter specifies per-partition mechanics and carries the bound
declarations. Consumes: chapter 02; chapter 41;
`[ANNEX:05-method-families-survey]`. Invalidation-tags: none.
Provenance footnote: old repo `calibre/conformal/runtime.py`.

### `06-online-recalibration.md`

The observe loop as a first-class runtime contract: actuals resolution into
calibration state, pending-observation buffering for late/out-of-order
actuals, restart safety, state keying by session and partition, and the
recalibration cadence contract (what a method promises when fed a stream
rather than a batch). Split from chapter 05 because it is a *runtime/state*
contract, not a *method* contract — any conformal plugin gets online behavior
by satisfying it. Consumes: chapters 03 and 05; no gated decisions.
Provenance footnote: old repo `/observe` route + `pending_observations`
persistence.

### `07-reconciliation.md`

Hierarchical reconciliation as a pipeline stage: the reconciler protocol,
strategy registry, summing-matrix construction from hierarchy facts, sparse
vs. dense feasibility at retail scale (the old engine established that a
sparse summing matrix is the memory pivot at ~30k-series scale, with a dense
ceiling near 7.6 GiB for the matrix alone), and where reconciliation sits
relative to calibration. Its two formerly gated interfaces are bound by
chapter 41 (`40-gated-seams/`): the stage's output-column contract is points
only (`[SEAM-2]`), and no non-point forecast quantity is required to be
additive across the lattice (`[SEAM-3]`); the bound statements live in this
chapter. Consumes: chapter 02 hierarchy vocabulary; chapter 41; full-M5
memory evidence (restated, engine-independent);
`[ANNEX:07-coherence-decision]`.
Invalidation-tags: none.
Provenance footnote: old repo `calibre/reconciliation/` and README
reconciliation section.

### `08-ordering-and-cost.md`

Cost-driven ordering: the cost-structure object (holding, shortage, ordering
frictions) as a first-class config citizen; the order-policy protocol
(calibrated forecast + inventory position + cost structure → order) and the
cost *interpretation* — including the chapter 02-declared mapping between the
per-decision (underage/overage) and per-period (holding/shortage) cost pairs
under lead time and review period; the inventory simulation that realizes the
chapter 03 settlement contract (`[SET-*]`) in backtests; and the contract
that **realized cost is an optimization objective, not just a report** (it
must be computable per-candidate inside a tuning loop). Boundary: chapter 02
defines the decision terms, chapter 03 owns the settle runtime contract, this
chapter owns the policy protocol and what the booked costs mean. The cost
scope is bound by chapter 41 (`40-gated-seams/`, `[SEAM-4]`): realized cost
attaches at the decision nodes, no lattice-level aggregate cost functional
exists, and this chapter's newsvendor-family mechanics carry the bound
statement. Consumes: chapters 02, 07; chapter 41;
`[ANNEX:08-cost-objective-derivation]`.
Invalidation-tags: none.
Provenance footnote: old repo `calibre/ordering/` + `simulation/`.

### `09-tuning.md`

Joint hyper-parameter search across the three channels — model, conformal,
ordering — as one candidate object, with realized cost from chapter 08 as the
default objective. Covers the local/global tuning axis (per-series studies
vs. panel-level studies), fan-out and partial-completion resume on Ray, and
search-space declaration in the authoring layer. The default objective is
bound through chapter 08 (`[SEAM-4]`); the chapter binds to "the chapter 08
objective" symbolically, never to a formula.
Consumes: chapters 04, 05, 08. Invalidation-tags: none.
Provenance footnote: old repo `calibre/tuning/` (Ray Tune + Optuna),
`TuningCandidate`.

### `10-pipeline-authoring.md`

The authoring surface: how a user declares a full pipeline (dataset adapter,
model, reconciler, conformal method, cost structure, policy, tuning block) in
minutes, with validation before execution (`validate` as a first-class verb),
config-as-data (YAML or equivalent) mapping 1:1 onto the chapter 02 domain
objects, sane defaults, and composability (a sweep is a directory of configs;
a tuning run is a config plus a search space). The "easy, clean, fast" vision
element lives here and is judged by a concrete acceptance script: a new user
authors and validates a runnable backtest without reading engine source.
Consumes: chapters 02–09 (it is their user-facing projection); no gated
decisions.
Provenance footnote: old repo `calibre/cli/` YAML loader.

### `11-api.md`

The service surface: the lifecycle routes (fit / predict / calibrate / order /
observe / session introspection / backtest jobs / tuning studies), async job
semantics, deterministic session identity, tenancy keying, trusted
server-owned artifacts (clients never supply model bytes or arbitrary
artifact URIs), what-if prediction overrides, and the guarantee that every
API verb is a thin projection of the chapter 03 engine (no API-only logic).
Consumes: chapters 03, 06, 08, 09; no gated decisions.
Provenance footnote: old repo `calibre/api/` route table in README.

### `12-cloud-native.md`

Scaling and state on Kubernetes: Ray-on-K8S topology, stateless API replicas
against a shared Postgres state store and shared object store for artifacts,
migration discipline, health/liveness/metrics surfaces, and the
restart/multi-worker invariants (every durable fact lives in the store, never
in process memory). Distinguishes the three state classes — run metadata,
calibration state, artifacts — and their consistency requirements. Consumes:
chapters 03, 06, 11; no gated decisions.
Provenance footnote: old repo `calibre/storage/`, `LIFECYCLE_STORE=sql`,
`CALIBRE_ARTIFACT_URI` semantics.

### `20-protocol-vn2.md`

Engine-independent restatement of the VN2 inventory-challenge protocol: data
shape (weekly sales panel, optional master/in-stock), the decision cadence,
lead-time and review-period structure, the cost accounting (holding +
shortage), and what a submission/replication must produce — written so any
engine (including a from-scratch one) can implement and score it. Explicitly
**does not** carry the old engine's regression number: that figure is an
old-engine tripwire retired at cutover, not a protocol constant. The headline
figures the rewrite reports on VN2 are bound by chapter 42
(`40-gated-seams/`, `[FLG-1]`/`[FLG-2]`): the two-axis flagship claim,
measured on the surfaces this chapter fixes. Consumes: public VN2 challenge
materials; chapter 42; `[ANNEX:20-vn2-replication-notes]`.
Invalidation-tags: none.
Provenance footnote: old repo `benchmarks/vn2/`.

### `21-protocol-m5.md`

Engine-independent restatement of the M5 protocol as Calibre uses it: the
sales/calendar data contract, the `item×store` bottom level and the
product/location aggregation lattice (~30.5k bottom series, ~33.6k total
nodes), phase semantics (validation/evaluation), the origin-window invariant
for streaming evaluation, and per-node coverage scoring over a resolved
ledger. **Drafted**: the chapter exists and is bound to the ratified
scoring-exemption ruling (which rows may be legitimately excluded from
scoring, and why); the decision record sits behind
`[ANNEX:21-m5-scoring-exemption-record]`. Consumes: public M5 competition
materials; `[ANNEX:21-m5-scoring-exemption-record]`. Invalidation-tags: none.
Provenance footnote: old repo `benchmarks/m5/README.md`, `score-m5-coverage`.

### `30-performance.md`

Performance targets and budgets for the rewrite: fit/predict throughput at
full-M5 scale, memory ceilings per pipeline stage, Ray fan-out efficiency,
and the reconciliation memory pivot quantified. **Drafted**: the full-M5
profile of the old engine (the behavior oracle's cost baseline) landed
2026-07-07, and the chapter is bound to it — the measured baseline restated
engine-independently, the rewrite's performance budget, and the architectural
requirements the gap between them forces; raw telemetry sits behind
`[ANNEX:30-profile-raw-data]`. Consumes: full-M5 profiling evidence;
`[ANNEX:30-profile-raw-data]`. Invalidation-tags: none.

### `40-gated-seams/`

The chapters that carry the ratified Stage 1 decisions, stated as
*interfaces and claims* only, with all derivations behind annex pointers.
`41-decision-calibration-seams.md` carries the seam contract — the four
decision-calibration seams and the scope declaration — and binds it into the
tree (`[SEAM-*]`): the guarantee descriptor made mandatory, the points-only
reconciliation output contract, the non-additivity position, the
decision-node cost scope, the decision-scope coverage rule, the
`joint_claim` vocabulary, and the calibration context.
`42-flagship-metric.md` carries the flagship metric — the two-axis headline
claim (certificate gated, price tracked) — and binds its measurement to
chapter 20 and its reporting to chapter 01 (`[FLG-*]`). The directory
`README.md` records the chapter set, the bound hook table, and the drafting
rules (annex discipline; the spec-wide no-positioning rule). A change to
the chapter set
reopens every binding, per the README rule. Consumes: ratified Stage 1
decision records via `[ANNEX:41-seam-decision-record]` and
`[ANNEX:42-flagship-metric-record]`. Invalidation-tags: none.

### `50-test-and-oracle-strategy.md`

How the rewrite proves behavioral fidelity and correctness: the old engine as
a **behavior oracle** (golden outputs captured from it before retirement),
which oracle surfaces are worth freezing versus which old behaviors are
explicitly *not* carried (bug-for-bug fidelity is a non-goal), property-based
and protocol-level tests that outlive the oracle, and CI gate structure.
**Drafted**: the test-suite curation ruling is ratified, and the chapter is
bound to it — the assertion taxonomy,
the tolerance doctrine, the cross-engine checkpoint rules, and the CI gate
tiers; the per-item ruled corpus sits behind
`[ANNEX:50-oracle-curation-record]`. Consumes: the ratified curation ruling;
`[ANNEX:50-oracle-curation-record]`. Invalidation-tags: none.

### `60-onboarding.md`

The first-contributor path: the named **first brick** — the seasonal-naive
forecasting adapter, implementable from chapters 02 + 04 alone, touching no
gated seam and green in under a day — the acceptance walkthrough, the first
runnable test, and the minimal reading order (00 → 02 → 04). Doubles as the
spec's standalone-readability check (the brick is buildable from its named
chapters without the old repo or the annex). Consumes: chapters 02, 04; no
gated decisions. Invalidation-tags: none.

### `90-annex-registry.md`

The complete list of `[ANNEX:*]` pointers used anywhere in `docs/spec/`, each
with a one-line public-safe description of the material class behind it. The
leak-review checklist lives here: every pointer registered, no annex content
inlined, no private paths anywhere in the public layer. Consumes: all
chapters (mechanical aggregation); no gated decisions.

### `adr/README.md`

Index and template for architecture decision records that bind multiple
chapters (e.g. "config-as-data over programmatic pipeline construction",
"Postgres as the single durable store"). ADRs follow the same frontmatter
contract; gated decisions do NOT become ADRs in the public layer — they live
behind `40-gated-seams/` and the annex.

---

## Vision-coverage matrix

| Vision element | Owning chapter(s) |
|---|---|
| Nixtla/Ray core | 03 |
| Pluggable forecasting models | 04 |
| Pluggable conformal methods | 05 |
| Cost-driven ordering; cost as first-class tuning objective | 08, 09 |
| Cloud-native, scales on K8S | 12 |
| Strong API | 11 |
| Easy, clean, fast pipeline authoring | 10 |
| Local and global modelling and tuning | 04, 09 |
| One engine for backtesting and inference | 03 |
| Online recalibration | 06 |
| Hierarchical reconciliation | 07 |

## Requirement-ID prefix registry

Every requirement-ID prefix in use, with its owning chapter (design rule 6).
No two chapters share a prefix; a chapter that introduces a prefix adds its
row here in the same change. Chapter 50 references other chapters' tags and
owns no prefix of its own.

| Chapter | Prefixes |
|---|---|
| 01 | `[VIS-*]` |
| 02 | `[SER-*]` `[PAN-*]` `[FRA-*]` `[TSK-*]` `[HIE-*]` `[CST-*]` `[ORD-*]` `[SES-*]` `[CAL-*]` `[INV-*]` `[LED-*]` `[GRT-*]` |
| 03 | `[ENG-*]` `[DRV-*]` `[SET-*]` `[STA-*]` `[SPN-*]` `[DET-*]` |
| 04 | `[ADA-*]` `[REG-*]` `[SCO-*]` `[ART-*]` `[FIT-*]` `[CEN-*]` |
| 05 | `[CNF-*]` |
| 06 | `[OBS-*]` |
| 07 | `[REC-*]` |
| 08 | `[CFG-*]` `[POL-*]` `[OBJ-*]` `[SIM-*]` |
| 09 | `[TUN-*]` |
| 10 | `[AUT-*]` `[VAL-*]` `[CMP-*]` |
| 11 | `[API-*]` `[JOB-*]` |
| 12 | `[TOP-*]` `[DUR-*]` `[RST-*]` `[MIG-*]` `[MON-*]` |
| 20 | `[VN2-*]` |
| 21 | `[M5-*]` |
| 30 | `[PRF-*]` |
| 41 | `[SEAM-*]` |
| 42 | `[FLG-*]` |

## Gate/evidence dependency summary

| File | Blocked by |
|---|---|
| All | Nothing — the Stage 1 gate landed and the onboarding brick is named; every formerly gated interface is bound in its owning chapter and `40-gated-seams/` |
