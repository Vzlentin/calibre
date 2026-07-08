---
title: "API — the service surface"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 11 — API

This chapter is the contract for the HTTP service surface: the verb set, their
async semantics, session identity, tenancy, artifact trust, and what-if
overrides. It consumes chapters 03 (engine and drivers), 06 (observe loop),
08 (ordering and cost), and 09 (tuning), and uses the chapter 02 vocabulary
without redefinition. It carries no gated decisions and no seam hooks.

## State the projection rule

`[API-1]` **Every API verb is a thin projection of the chapter 03 engine.**
The API is the transport for the engine's *event driver*: it decodes and
validates requests, resolves tenancy, invokes an engine operation, and encodes
the result. No API-only logic exists — no forecast computation, no calibration
arithmetic, no override-merge semantics, no readiness dispatch implemented in
the route layer. Any behavior observable through a route must be reproducible
by invoking the same engine operation directly, and the backtest time-loop
driver and the API event driver exercise the same operations — `[SES-3]`
(chapter 02) is the session-identity half of that rule, `[DRV-1]`
(chapter 03) the behavioral half.

`[API-2]` Layering is one-directional: the API depends on the engine; the
engine never depends on the API, and the API never depends on any other
front-end (CLI, notebook, authoring tooling). Configuration types the API
accepts are engine-owned types, not front-end types.

Acceptance: for every route, a test demonstrates an equivalent engine-level
call producing the same durable state transition and the same result payload
(modulo transport encoding); a dependency check shows no import path from the
API package into any sibling front-end package.

## Enumerate the verbs

The verb set is normative; concrete path spellings may vary, but each verb
below must exist with the stated method, mode, and semantics.

| Verb | Method/mode | Semantics |
|---|---|---|
| fit | POST, async (202 + handle) | Ingest history for a series set, derive the session, fit the model configuration, persist a server-owned model artifact |
| fit status | GET, sync | Poll a fit handle: status, session identity, artifact references, structured error |
| predict | POST, sync | Produce a forecast frame from a succeeded fit at a supplied origin, with optional what-if override |
| calibrate | POST, sync | Apply the session's conformal configuration to a forecast frame, persisting updated calibration state |
| order | POST, sync | Apply the session's ordering policy and cost structure to a calibrated forecast frame; persist the resulting orders |
| observe | POST, async (202 + handle) | Resolve actuals into the session's calibration state per the chapter 06 observe loop |
| session introspection | GET, sync | Read-only projection of a session: calibration state, latest forecast frame, open orders |
| backtest job | POST, async (202 + handle) | Run a full pipeline configuration under the time-loop driver |
| backtest status | GET, sync | Poll a backtest handle: status, result artifact references, structured error |
| tuning study | POST, async (202 + handle) | Run a chapter 09 study: joint model/conformal/ordering search over a declared search space and objective |
| study status | GET, sync | Poll a study handle: status, scope-shaped best result — per-series best candidates for a local-scope study, one panel-wide best candidate for global scope (chapter 09) — each candidate spanning the three config channels, structured error |

Liveness, readiness, and metrics endpoints exist but are operational surfaces
owned by chapter 12; this chapter only requires that they impose no
dependency on any durable store for liveness.

`[API-3]` Synchronous verbs (predict, calibrate, order) complete within the
request and return their full result payload. Asynchronous verbs return
`202` with a **handle** and never block on the work.

## Define async job semantics

`[JOB-1]` Every async submission returns a handle carrying at minimum: a
unique job identifier, the session identity where one applies, and a status.
Status values form a fixed lifecycle: `QUEUED → RUNNING → SUCCEEDED | FAILED`.
Terminal states are terminal; a failed job carries a structured error.

`[JOB-2]` **Every handle is pollable.** Each async verb has a corresponding
status verb addressing the job by identifier, returning current status,
results or result references on success, and the structured error on failure.
This includes observe: an accepted observation's outcome must be observable,
never fire-and-forget.

`[JOB-3]` **Validate before accepting.** Every precondition checkable at
request time is checked at request time, and a violation is rejected with a
client error — a `202` is a commitment that the job's preconditions held at
submission, not a maybe. In particular: an empty series set, an unreadable
client-referenced data source, an unknown search space or objective, a
session with no conformal configuration (for observe), and actuals with no
usable resolved value are all synchronous rejections. Conversely, failures
only discoverable by doing the work (e.g. a model configuration incompatible
with the data) land the job in `FAILED` with the cause — never a later,
unrelated verb. A fit must validate its model configuration against the data
inside the fit job, so an incompatible configuration fails at fit, never
surfacing first at predict.

`[JOB-4]` Submission is idempotent under a client-supplied idempotency key:
re-submitting with the same key returns the existing job's handle rather
than creating a second job. Jobs and their results are addressable by handle
from any API replica (durable job records; chapter 12 owns store topology).

## Derive session identity deterministically

`[API-4]` Session identity is computed **server-side** as a pure,
deterministic function of the session's defining inputs per `[SES-1]`
(chapter 02): tenant, series set, horizon, calendar frequency, model
configuration, conformal configuration, and the decision-side configuration
(ordering policy and cost structure). The decision-side configuration enters
the derivation at fit time — fit accepts it alongside the conformal
configuration, both optional, and an absent configuration enters the
derivation as explicitly empty, so two fits differing only in whether one
was supplied mint different sessions. The requirement
is determinism and the input set, not any particular algorithm. Binding
rules:

- Clients never mint or supply a session identity at creation; the fit verb
  derives it and returns it in the handle. Subsequent verbs (calibrate,
  observe, introspection) address the session by that identity.
- Derivation is canonical: independent of series-set ordering and of key
  ordering within configuration mappings. Identical defining inputs yield the
  same identity across processes, replicas, and restarts.
- Any change to a defining input is a different session; sessions never share
  mutable state `[SES-2]`. Re-fitting the *same* defining inputs re-enters
  the same session: calibration state accumulated by the session carries
  across re-fits, while each fit remains an individually addressable job.

Acceptance: a property test derives the identity for permuted series sets and
permuted configuration key orders and gets one value; deriving in two
separate processes gets the same value; changing any single defining input
changes it.

## Key everything by tenant

`[API-5]` Every state-touching request carries a tenant, and all durable
state — sessions, calibration state, orders, fit and job records, artifacts —
is tenant-scoped `[SES-2]`. Reads and writes resolve only within the caller's
tenant. A request addressing state outside its tenant receives the same
response as for state that does not exist (no cross-tenant existence oracle).
Because tenant is a defining input of session identity, identical
configurations under different tenants are distinct sessions with disjoint
state. Authentication and credential mapping to tenants is deployment policy
(chapter 12); this chapter binds the keying discipline.

## Own artifacts server-side

`[API-6]` **Fitted-model artifacts are trusted, server-owned objects.**
Clients never supply model bytes, and no request field accepts an artifact
URI to load. The engine persists artifacts under a server-configured artifact
root, addressed by server-computed keys; predict resolves the artifact for a
fit exclusively through those server-computed keys. Artifact references
returned to clients (in fit handles or job results) are read-only pointers —
possibly time-limited signed URLs — and are never accepted back as request
inputs. Deserializing a model artifact is a trusted operation precisely
because this boundary guarantees every artifact was produced by this service.

`[API-7]` Client-referenced URIs exist only on the **data plane**: history,
future exogenous regressors, and realized actuals may be passed by reference.
Such references are resolved through the engine's dataset-adapter seam
(chapter 03) under a deployment-configured scheme allowlist; an unresolvable
reference is a client error, not a server fault. Data references may carry a
point-in-time cutoff (`as_of`) so ingested history respects `[INV-TEMPORAL]`
with respect to later data revisions.

## Merge what-if overrides statelessly

`[API-8]` Predict accepts an optional **what-if override**: a mapping from
series key to future-exogenous-regressor rows. Semantics are a **stateless
merge**: the override is combined with the fit-time baseline regressors,
override rows taking precedence per `(series key, target timestamp)`;
baseline rows not overridden pass through unchanged. The merge exists only
for the duration of the predict call — the stored fit-time baseline is never
mutated, and no override persists. Two predict calls against the same fit,
one with and one without an override, are fully independent; issuing an
override predict then a plain predict returns the un-overridden baseline
forecast. Override rows are future exogenous regressors and obey `[TSK-3]`:
values legitimately known at the origin (planned prices, promotion flags).
A malformed override (missing timestamps, unmergeable schema) is a client
error before any engine work.

## Bind each verb to the engine

Per-verb contracts, each a projection of an engine operation:

- **fit** — inputs: tenant, series set, horizon, calendar frequency, history
  reference, optional future-regressor reference, model configuration,
  optional conformal configuration, optional decision-side configuration
  (ordering policy and cost structure). Horizon and calendar frequency are
  session-defining inputs (`[SES-1]`, chapter 02): two fits differing only
  in horizon, or only in frequency, mint different sessions. Effects:
  derives the session `[API-4]`, ingests the panel, runs fit validation
  `[JOB-3]`, persists the model artifact `[API-6]`. Handle carries fit
  identifier + session identity.
- **predict** — inputs: fit identifier, origin, optional override `[API-8]`.
  Requires a `SUCCEEDED` fit (conflict error otherwise). The engine builds
  the forecast task with history strictly before the origin — temporal
  hygiene enforced structurally at task construction (`[TSK-2]`,
  `[INV-TEMPORAL]`), not by route-layer filtering. Returns forecast-frame
  rows valid under `[FRA-3]`.
- **calibrate** — inputs: session identity, forecast-frame rows. The engine
  restores the session's calibration state (`[CAL-1]`, `[CAL-2]`), applies
  the conformal method, persists updated state, and returns the frame with
  interval columns per `[FRA-2]`. A session lacking a conformal
  configuration is a client error.
- **order** — inputs: calibrated forecast-frame rows, inventory state,
  session identity. The ordering policy and its cost structure (`[CST-3]`,
  chapter 08) are the session's decision-side configuration, resolved from
  the session — never supplied per request, because they are session-defining
  inputs (`[SES-1]`): an order under a different decision-side configuration
  belongs to a different session. A session lacking a decision-side
  configuration is a client error (mirroring calibrate).
  Inventory state is explicit: the request carries
  the inventory state the policy reads — which may be explicitly zero
  on-hand with an empty pipeline — or the verb refuses with a client error.
  An absent inventory state is never silently defaulted to zero (the
  chapter 08 refuse-never-degrade rule, `[POL-6]`): a fabricated zero state
  corrupts the inventory position and with it the order. Orders satisfy
  `[ORD-1]`, are persisted as immutable decision facts keyed by `[ORD-2]`
  (`[ORD-3]`), and are returned in the response. Persistence is not
  best-effort: an order request naming an unknown session is an error, never
  a silent skip.
- **observe** — inputs: session identity, actual observations
  (series key, timestamp, value). Projects the chapter 06 observe loop:
  resolution updates calibration state under `[CAL-3]`; late or out-of-order
  actuals buffer per chapter 06; the job is pollable `[JOB-2]`.
- **session introspection** — read-only; returns the session's calibration
  state per partition, the latest forecast frame, and open orders, all read
  from durable stores (never process memory), so any replica answers
  identically.
- **backtest job** — input: a complete pipeline configuration (the chapter 10
  authoring document as data). Runs the identical engine under the time-loop
  driver; results (ledger, metrics, artifact references) are addressable via
  the handle.
- **tuning study** — inputs: tenant, series set, history and actuals
  references, origins, base model configuration, search-space and objective
  identifiers, trial budget, local/global scope. Projects chapter 09: one
  candidate object spanning the three config channels; a study result shaped
  by the declared scope — per-series best candidates for a local-scope study
  (one study per series key), one panel-wide best candidate for a global
  study; and partial-completion resume — re-submitting the same study
  resumes rather than restarts, skipping completed work (completed series
  studies under local scope, completed trials under global scope).

## Standardize error semantics

`[API-9]` Errors are structured (machine-readable code + human-readable
detail) and consistently classed: malformed payloads and unreadable
client-referenced data are client errors; unknown identifiers are not-found;
verbs invoked against state in the wrong lifecycle stage (predict on an
unfinished fit, observe before any calibrate) are conflicts; semantically
invalid but well-formed inputs are validation errors. Engine faults surface
as server errors whose payloads never leak internals (no stack traces, no
store URIs). No verb ever converts a checkable client mistake into a
server error, and none converts an engine fault into a silent success.

## Conformance

A conforming implementation must demonstrate, by test:

1. Route-vs-engine equivalence and front-end import isolation `[API-1]`,
   `[API-2]`.
2. Every async verb returns a pollable handle; a forced job failure is
   observable via its status verb with a structured error `[JOB-1]`,
   `[JOB-2]`.
3. Checkable precondition violations are rejected synchronously; an
   incompatible model configuration fails the fit job, and predict against
   that fit returns a conflict, not the incompatibility `[JOB-3]`.
4. Idempotent re-submission returns the original handle `[JOB-4]`.
5. Session-identity determinism under permutation and across processes
   `[API-4]`, `[SES-1]`.
6. Cross-tenant addressing is indistinguishable from absent state `[API-5]`.
7. No request schema accepts model bytes or artifact URIs; predict resolves
   artifacts only via server-computed keys `[API-6]`.
8. Override predict then plain predict: the second returns the baseline
   forecast; the stored fit baseline is bit-identical before and after
   `[API-8]`.
9. Predict at origin `o` never exposes any observation stamped at or after
   `o` to the model, enforced at task construction `[INV-TEMPORAL]`.
10. An order request naming an unknown session fails; a successful order is
    readable back through session introspection `[ORD-2]`, `[ORD-3]`.

## Provenance

For spec authors only; the chapter stands without these. Positive space from
the old engine: the README route table and `calibre/api/main.py` (verb set,
202-handle pattern, synchronous precondition validation before 202, eager fit
validation, deterministic session derivation in
`calibre/storage/session.py::derive_session_id`, server-computed artifact
cache keys, `future_x_override` stateless merge, idempotency key on
backtests). Negative space: the route module imported CLI-layer configuration
types and private execution helpers and implemented lifecycle logic
(override merging, observe dispatch guards) in the route layer — the
api-to-CLI inversion `[API-1]`/`[API-2]` forbids; its observe job had no
pollable status (`[JOB-2]` closes this); and its order route silently skipped
persistence when the session lookup missed (the order verb contract forbids
best-effort persistence).
