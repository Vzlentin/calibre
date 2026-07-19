---
title: "Online recalibration — the observe loop"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 06 — Online recalibration

This chapter is the runtime/state contract that makes any conformal method
(chapter 05) *online*: how realized actuals — and the censoring facts that
ride them — flow back into calibration state, in what order, with what
durability, and what a method may rely on when it is fed a stream of
observations rather than a batch. It is deliberately split
from chapter 05: chapter 05 owns *method* contracts (how scores become
intervals); this chapter owns the *loop* — a conformal plugin gets online
behavior by satisfying this contract, without loop code of its own. It
consumes the engine spine of chapter 03 and the vocabulary of chapter 02
(`session`, `partition`, `calibration state`, `ledger`, `origin`,
`horizon step`, `series key`, `model name`), used here without redefinition.
Contract clauses carry stable tags `[OBS-n]` for citation by tests and later
chapters. Nothing in this chapter is gated: what joint claims may be made
across hierarchy nodes from these mechanics is owned by chapters 05 and 07,
not here. One further exclusion is deliberate: decision-endogenous censoring
facts at the test point — the decision-time supply level that will truncate
the observation being predicted — are out of scope for this contract.
Calibration receives the censoring facts of resolved observations only, and
no channel supplies test-point-side availability to any verb; revisiting this
exclusion reopens this chapter.

## Position the observe loop

An **observe cycle** is one pass of: accept actuals → resolve pending ledger
rows → deliver the newly resolved material to the calibrator → persist state.
Both drivers of chapter 03 run the *same* observe-loop code path:

- The **time-loop driver** (backtesting) runs one cycle per origin, feeding
  actuals from the replayed dataset's reveal schedule.
- The **event driver** (live inference) runs one cycle per accepted actuals
  submission on the observe verb (chapter 11 is its HTTP projection).

`[OBS-1]` Driver equivalence: given identical resolution schedules (the same
actuals becoming available in the same order relative to the same origins),
the two drivers produce identical calibration state. There is no
backtest-only or API-only observe logic. Changing whether an actual is
available before an origin changes the resolution schedule and is not an
equivalence case. A different cross-cycle delivery order is likewise not
required to be equivalent for a method that declares order sensitivity
`[CNF-3]`. Out-of-order tolerance means durable acceptance, no retroactive
change to committed issuance, and exactly-once eventual delivery under
`[OBS-17]`; it does not mean permutation invariance.

Two durable state surfaces participate, both session-owned:

- the **observed-actuals history**: every accepted actual with its censoring
  status (`[OBS-30]`), keyed by `(series key, timestamp)`;
- the **pending-observation set**: the pending ledger rows `[LED-1]` awaiting
  actuals, keyed per row by `(series key, origin, horizon step, model name)`
  `[FRA-1]` and carrying the target timestamp, point forecast, and issued
  bounds.

## Resolve actuals into calibration state

`[OBS-2]` Submission validation is atomic: an actuals submission is validated
as a unit, and a malformed record — non-numeric or non-finite value, unknown
series key, a censoring declaration outside the declared status set
(`[OBS-30]`), a non-numeric or non-finite availability bound where one is
supplied (`[OBS-30]`), duplicate `(series key, timestamp)` within the
submission — rejects the whole submission synchronously, before any state
mutation. No record is ever silently dropped.

`[OBS-3]` Every accepted actual is durably recorded — with its censoring
status `[OBS-30]` — in the observed-actuals history whether or not it matches
a pending row. Matching is a separate step:
an actual with no matching pending row is not an error and is not discarded —
it may complete an aggregate (see below) or simply stand as history.

`[OBS-4]` Resolution is a keyed join: an actual `(series key, timestamp,
value)` resolves exactly the pending rows whose series key and target
timestamp match, across all origins, horizon steps, and model names.
Resolution carries the actual's censoring status onto the row along with its
value. It fills only unresolved rows; a resolved row's actual is never
mutated `[LED-2]`.

`[OBS-5]` Resolution is idempotent: re-submitting an already-recorded actual
with the identical value and censoring status is a no-op; a *conflicting*
value — or a conflicting censoring status — for an already-recorded
`(series key, timestamp)` is rejected with an error naming the key. In both
cases persisted state is unchanged.

`[OBS-6]` Due rule: a pending row is **due** at origin `o` when its target
timestamp is admissible at `o` under temporal hygiene `[INV-TEMPORAL]` —
strictly before `o`. A due row whose actual has not arrived stays pending
without error `[LED-3]`; lateness is normal, not exceptional.

## Gate aggregate actuals on completeness

Aggregate nodes are addressable as series, so their ledger rows resolve too —
but their actuals are **derived, never posted**:

`[OBS-7]` An actuals submission may address bottom series only; a value posted
directly against an aggregate node label is rejected (it could contradict the
derived sum).

`[OBS-8]` Completeness gate `[INV-COHERENCE]`: an aggregate node's actual at
timestamp `t` exists only when *every* member's actual at `t` is present in
the observed-actuals history, and then equals the members' sum exactly. While
any member is unobserved, the aggregate's pending rows at `t` remain pending —
never a partial sum, never a zero-fill, never an eviction. The derived actual
inherits member censoring: it is censored when any member's actual at `t` is
censored — the derived sum is then a lower bound on aggregate demand —
uncensored only when every member is declared uncensored, and undeclared
otherwise `[OBS-30]`.

`[OBS-9]` Deferred aggregate resolution: when the last missing member's actual
lands, the aggregate's actual comes into existence in that same observe cycle,
and the aggregate's pending rows at `t` resolve and deliver in that cycle.
A member actual arriving for a series with no pending rows of its own still
counts toward every containing aggregate's completeness.

## Order observation before issuance

`[OBS-10]` Within a session, origins are processed in chronological order
(the driver's obligation, chapter 03).

`[OBS-11]` Observe-before-issue: before intervals for origin `o` are issued,
the engine runs an observe cycle that delivers every resolution due at `o`
(rows due per `[OBS-6]` whose actuals are available). The calibrator that
issues at `o` has therefore already ingested everything the timeline admits.
Behavioral consequence, testable without value pins: perturbing one actual
that resolves before `o` changes the band issued at `o` for any method with
nonzero sensitivity to its most recent score, while an unperturbed control
run's band is unchanged.

`[OBS-12]` Snapshot issuance: all rows issued at one origin are calibrated
from a single calibration-state snapshot. No state update lands between the
first and last row of one origin's issuance.

## Guarantee cold-start liveness

A cold calibrator issues NaN bounds until its declared calibration
requirement is met (`[LED-6]`, chapter 05). The loop must not let that NaN
state starve the calibrator of the very observations it needs:

`[OBS-13]` Delivery readiness is independent of issued bounds: a pending row
is deliverable when its actual is resolved and its point forecast is finite —
the finiteness of its issued bound columns is irrelevant. Filtering
NaN-bound rows out of delivery is forbidden: it deadlocks cold partitions
(the calibrator never receives a score, so it never warms, so bounds stay
NaN forever).

`[OBS-14]` A pending row without a finite point forecast never delivers (there
is no residual to extract); it remains pending and is excluded from scoring
with its cause recorded `[LED-7]`.

`[OBS-15]` Escape from NaN: once a partition's delivered-score count meets the
method's declared calibration requirement, subsequent issuance for that
partition emits finite bounds. Readiness accounting counts delivered scores
only: an observation excluded with attributable cause under the score-input
contract (`[CNF-27]`, chapter 05) contributes no score and does not advance
the delivered-score count. A calibrator that has met its declared
requirement and still emits non-finite bounds is defective unless its
manifest declares that post-warm-up emission mode (`[CNF-22]`, chapter 05);
an undeclared post-warm-up non-finite emission is a defect, a declared one
goes out unscored with its cause attributed. Conversely, before the
requirement is met, bounds are NaN — never fabricated, never zero-width by
default `[CAL-4]`.

## Buffer late and out-of-order actuals

`[OBS-16]` The pending-observation set is durable state under the same store
discipline as calibration state, and unbounded in time: no timeout, no
eviction. A row pends until its actual (or, for aggregates, its last member's
actual) lands — whether that is the next cycle or many origins later.

`[OBS-17]` Exactly-once delivery: each pending row (per-step methods) or each
protection window (window-sum methods) is delivered exactly once — in the
cycle it becomes ready — then leaves the pending set. It is never
redelivered, and never lost: when every actual has landed, the pending set
drains to empty.

`[OBS-18]` Window-sum gating: for a window-sum method, a protection window
`(series key, model name, origin)` delivers only when every row of the window
is resolved. A partial window delivers *nothing* — no call reaches the
calibrator, not an empty or padded one — and delivers exactly one composite
observation when complete. The score's definition is chapter 05's; the
gating is this loop's.

## Key state by session and partition

`[OBS-19]` Calibration state is addressed by `(session, partition)` and
nothing else `[CAL-1]`; the pending-observation set and observed-actuals
history are addressed by session. Two sessions never share any of the three
surfaces `[SES-2]`; tenancy isolation follows from session identity
`[SES-1]`.

`[OBS-20]` Partition routing is exact: each delivered score is routed to
exactly one partition — the partition of its ledger row under the session's
configured partition key. Any pooling or blending across partitions is a
method-declared behavior inside chapter 05, invisible to this loop.

`[OBS-21]` Partition-scoped writes: persisting an observe cycle touches only
the partitions that received deliveries in that cycle; every other
partition's persisted **partition-scope** state rows are byte-identical
before and after. State under the reserved method-scope label (`[CNF-7]`,
chapter 05) is outside this clause: it may legitimately update on any observe
cycle, including one that delivers to no partition, and routing `[OBS-20]`
never delivers to the reserved label.

## Survive restarts

`[OBS-22]` Cycle atomicity: for one observe cycle, the calibration-state
upsert, the pending-set removals, and the observed-actuals appends commit
atomically. A failure at any point leaves all three surfaces exactly as they
were before the cycle — no partial ingestion, no double-count on retry.

`[OBS-23]` Resume equality: a process killed between observe cycles and
restarted from persisted state produces, for the remainder of the run,
deliveries and issued bounds identical to the uninterrupted run
(a per-engine exact-equality property, per `[CAL-2]`). A kill mid-way through
a window-sum method's protection window leaves its rows pending; the resumed
run completes the window and delivers it exactly once.

`[OBS-24]` Bounded state: a partition's persisted calibration state has size
bounded by the method's declared window, independent of stream length; the
serialized payload restores everything issuance needs. The pending set may
grow only with genuinely unresolved rows.

## Contract the recalibration cadence

What a method is promised — and must promise — when fed a stream rather than
a batch:

`[OBS-25]` Continuous cadence: the loop never defers a deliverable
resolution. There is no separate "recalibrate" verb and no schedule;
recalibration is the emergent effect of every cycle, visible at the next
issuance. A deployment wanting a slower cadence batches its submissions; the
engine never delays ingestion on its own.

`[OBS-26]` Deterministic delivery order: within one cycle, rows are delivered
in the canonical total order — ascending `(origin, horizon step, series key,
model name)`, total by row uniqueness `[FRA-1]`. Across cycles, delivery
follows resolution order: everything delivered in an earlier cycle precedes
everything delivered in a later one. Each partition's delivery sequence is
the induced subsequence.

`[OBS-27]` Chunk invariance: delivering the same delivery sequence in one
cycle or split across many consecutive cycles yields identical calibration
state. (Permutation invariance is *not* promised: order-sensitive methods
are legitimate, which is why `[OBS-26]` makes the order deterministic.)

`[OBS-28]` Method declarations: a conformal method participating in the loop
declares (a) its emission scope — *per-step*, or *window-sum (over the
protection window)* — read from its chapter 05 manifest field (`[CNF-12]`),
never re-declared on a second surface; (b) its calibration requirement
(minimum delivered scores per partition before finite bounds); (c) whether
its state is order-sensitive beyond the multiset of scores; (d) its state
bound `[OBS-24]`. The loop enforces the scope's gating and readiness; the
method owns everything inside `calibrate(scores, config)`.

`[OBS-29]` Progress: while ready resolutions keep arriving for a partition,
its delivered-score count grows monotonically. A loop that silently stops
delivering to a live partition is defective, whatever bounds it emits.

## Carry censoring facts end-to-end

Recorded values during stockout periods bound demand from below. The loop's
obligation is to carry that fact intact — never to repair it:

`[OBS-30]` Censoring facts ride the submission: each actuals record carries a
censoring declaration — *censored* (stocked out; the recorded value is a
lower bound on demand) or *uncensored*. A record carrying no declaration is
accepted and recorded with censoring status **undeclared**: a durable third
state, never silently coerced to uncensored. A record may additionally carry
an optional numeric **availability bound** — the supply level that truncated
the observation — present where the dataset supplies it, including on
uncensored records; the bound is stored in the observed-actuals history
alongside the censoring status. The bound is optional everywhere: a record
without one is complete, and its absence changes no behavior of this loop.
The event driver takes declarations — and the bound, where supplied — from
the submission; the time-loop driver derives them from the replayed dataset's
censoring facts `[PAN-3]` — driver equivalence `[OBS-1]` covers censoring
status like any other delivered fact.

`[OBS-31]` Three series, never collapsed: the recorded series (values as
observed), the censoring facts (per-observation status), and any
demand-honest resolution (a value admissible as demand for scoring, produced
only under chapter 05's score-input contract `[CNF-26]`–`[CNF-28]`) stay
distinct end-to-end. The observed-actuals history stores recorded values with
censoring status; no repair, imputation, or substitution ever overwrites a
recorded value, and no surface presents a repaired value as recorded.

`[OBS-32]` Delivery attaches status: every delivered resolution reaches the
calibrator with its actual's censoring status — for a derived aggregate, the
inherited status of `[OBS-8]`. Where the observed-actuals history records an
availability bound for the actual (`[OBS-30]`), delivery attaches the bound
to methods that declare censoring-fact consumption (`[CNF-26]`, chapter 05);
a resolution without a bound delivers unchanged — the bound is optional
everywhere. The loop neither filters nor repairs on censoring: whether a
censored or undeclared resolution may enter a score window, and against which
series, is chapter 05's score-input contract. A delivery path that drops the
status field recreates the collapsed-series defect this section forbids.

## Acceptance criteria

A conforming implementation demonstrates, by test:

1. **Late-member aggregate**: three bottom members, one actual arriving two
   origins late ⇒ the aggregate row stays pending with no partial sum, then
   resolves in the cycle the last member lands, equal to the exact member sum
   on integer fixtures `[OBS-8]`, `[OBS-9]`.
2. **Cold-start liveness**: a three-origin run with a cold calibrator —
   origin 1 issues NaN bounds; its actual resolves before origin 2; finite
   bounds emerge at the first origin after the declared requirement is met.
   An implementation that filters delivery on finite bounds fails this test
   `[OBS-13]`, `[OBS-15]`.
3. **Observe-before-issue**: the single-perturbation contrast of `[OBS-11]`,
   asserted as changed-versus-unchanged bands, never as pinned values.
4. **Exactly-once under restart**: kill between cycles, resume; per-partition
   delivered-score counts and final state equal the uninterrupted run; a
   mid-window kill delivers the window exactly once `[OBS-17]`, `[OBS-23]`.
5. **Chunk invariance**: one cycle versus N consecutive cycles over the same
   sequence ⇒ identical state `[OBS-27]`.
6. **Idempotent resubmission**: identical re-post is a no-op; a conflicting
   value is rejected naming the key; state unchanged in both `[OBS-5]`.
7. **Partition isolation**: a cycle delivering to one partition leaves all
   other partitions' persisted partition-scope state rows byte-identical
   `[OBS-21]`.
8. **Atomic cycle**: an injected failure mid-cycle leaves state, pending set,
   and history exactly as before the cycle `[OBS-22]`.
9. **Drain**: after all actuals land, the pending set is empty `[OBS-17]`.
10. **Censoring round-trip**: one submission mixing declared-censored,
    declared-uncensored, and undeclared records persists three distinct
    statuses `[OBS-30]`; an aggregate with one censored member derives a
    censored actual `[OBS-8]`; every delivered resolution carries its status
    to the calibrator `[OBS-32]`; a record carrying an availability bound
    persists it and delivers it to a censoring-fact-consuming method, while
    records without one resolve and deliver unchanged `[OBS-30]`, `[OBS-32]`;
    no surface reports a repaired value as recorded `[OBS-31]`.

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the old engine: `calibre/execution/decision_loop.py` (mode-keyed
`observe_pending`; per-horizon readiness = finite actual AND finite point
forecast with NaN bounds explicitly non-blocking; cumulative outer
completeness gate), `calibre/execution/backend.py` (a ResolveOpen phase
running before Predict — observe-before-issue), `calibre/execution/actuals.py`
(all-members-present aggregate resolution with complete-only caching),
`calibre/api/main.py::run_observe_job` (state upserted only on full success),
and `calibre/storage/models.py` (`pending_observations` keyed by session +
row key; conformal state keyed by session + partition). Negative space: an
earlier API observe path filtered NaN-bound rows and deadlocked cold
calibrators — `[OBS-13]` forbids that class outright; the old due comparison
was inclusive of the origin timestamp, harmless only because its reveal
protocols kept revealed actuals strictly before each origin — `[OBS-6]` pins
the strict boundary from `[INV-TEMPORAL]` instead; the old engine never
persisted why a row went unscored, which `[OBS-14]` (with `[LED-7]`) closes;
and its observe path carried recorded values only — submissions had no
censoring field, so a score computed on a stocked-out period's sales was
indistinguishable from a demand score — which `[OBS-30]`–`[OBS-32]` close by
making censoring status a first-class recorded, delivered fact.
