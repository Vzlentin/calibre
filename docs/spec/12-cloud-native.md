---
title: "Cloud-native runtime — scaling and state on Kubernetes"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 12 — Cloud-native runtime

This chapter is the deployment contract: how the engine (chapter 03), the
observe loop (chapter 06), and the API (chapter 11) run on Kubernetes with
elastic compute and multiple replicas *without changing observable behavior*.
Its one governing invariant: **every durable fact lives in a shared store,
never in process memory** — a deployment is correct exactly when no
kill-and-replace of any process changes what the system subsequently answers.
It uses chapter 02 vocabulary verbatim (session, tenant, calibration state,
partition, ledger, order, forecast task, origin) and carries no gated
decisions and no seam hooks.

## Deploy three planes

A deployment consists of three planes with distinct lifecycles:

1. **Serving plane** — N identical, stateless API replicas (a Kubernetes
   Deployment behind one Service).
2. **Compute plane** — a Ray cluster (one head, elastic autoscaled workers),
   provisioned on Kubernetes (e.g. via a Ray operator such as KubeRay).
   Batch backtests may alternatively run as plain Kubernetes Jobs using the
   same engine; both are the chapter 03 engine under the same time-loop
   driver, on different dispatch substrates.
3. **Durable plane** — a relational state store (Postgres) and an object
   store (S3-compatible or equivalent), both living *outside* pod lifecycles
   (managed services or independently operated stateful infrastructure).

Topology invariants:

- `[TOP-1]` **API replicas are interchangeable.** Any request may land on any
  replica with identical results; no sticky sessions, no replica-local
  mutable state that a second request could observe. Horizontal scaling is a
  replica-count change and nothing else.
- `[TOP-2]` **Compute workers are ephemeral.** A Ray worker holds no durable
  fact; losing any worker at any time costs only recomputation. Autoscaling
  and preemption are normal operation, not failure modes.
- `[TOP-3]` **The durable plane outlives everything.** Relational store and
  object store survive full cluster replacement; a fresh cluster pointed at
  the same stores resumes every session and run.
- `[TOP-4]` **One engine version per run.** API replicas, Ray workers, and
  batch Jobs participating in the same work run the same engine build
  (enforced by image digest). Version skew across driver and worker is a
  deployment error, not a supported state.
- `[TOP-5]` **Work ships as serialized forecast tasks.** Fan-out relies on
  task immutability and serializability `[TSK-4]`: the driver stages task
  payloads (panel slices, future exogenous regressors) in the shared object
  store under a run-scoped prefix and passes references, not bulk data,
  through the scheduler. Ray's in-memory object transfer is a transport
  optimization, never a system of record; staged prefixes are deleted on
  best effort at run end and are reconstructible from inputs at any time.
- `[TOP-6]` **Compute tasks are retry-safe.** A distributed task is a pure
  function of its serialized immutable inputs; re-execution (retry,
  speculation, worker loss) is always safe. Durable writes happen only in
  the driver, or as idempotent keyed upserts, so replays cannot double-apply.

## Classify durable state

- `[DUR-1]` Every durable fact belongs to exactly one of three state classes;
  a fact with no class has no right to exist.

### Run metadata (relational, strongly consistent)

The control-plane record: session registry, run/job/study lifecycle records
with status and error, orders `[ORD-2]`, `[ORD-3]`, tuning results keyed for
reuse, and **artifact pointers** (URI + size per artifact kind per run).
All rows are tenant-scoped.

- `[DUR-2]` Run metadata is transactional and strongly consistent: status
  transitions are atomic; a status observed by one replica is the status,
  not a stale replica-local view.
- `[DUR-3]` Job submission is idempotent under a client-supplied idempotency
  key: resubmission returns the existing record, never a duplicate run.
- `[DUR-4]` Concurrent writers on the same natural key resolve by atomic
  conditional upsert at the store, never by read-then-insert races.

### Calibration state (relational, per-key serialized)

The chapter 06 runtime state: calibration state rows keyed by
`(session, partition)` `[CAL-1]`, plus the pending-observation buffer for
issued-but-unresolved rows awaiting actuals.

- `[DUR-5]` Updates are atomic per `(session, partition)` key; concurrent
  updates to the same key from different replicas serialize to some total
  order — no lost update. No transaction ever needs to span keys, because
  each row is independently addressable and restorable `[CAL-1]`.
- `[DUR-6]` Persisted state round-trips to behavioral equality `[CAL-2]`:
  serialize → store → restore on a different process yields a calibrator
  indistinguishable from the uninterrupted one.
- `[DUR-7]` State is written only after the computation that produced it
  succeeds; a failed observe or calibrate leaves prior durable state intact.

### Artifacts (object store, immutable, pointer-referenced)

The bulk plane: ledger shards, panel and forecast-frame parquet, fitted model
artifacts, staged task payloads — everything large.

- `[DUR-8]` Artifacts are **write-once immutable**: a URI, once readable,
  never changes content. Corrections write a new URI and re-point.
- `[DUR-9]` **Pointer-after-blob ordering:** the run-metadata pointer to an
  artifact is committed only after the blob is durably written. A crash
  between the two leaves an orphan blob (garbage-collectable) — never a
  dangling pointer. Every pointer a reader can see resolves.
- `[DUR-10]` The artifact root is a **shared** object store with
  read-after-write visibility for every replica and worker. A process-local
  filesystem root is valid only in a single-process development deployment;
  a multi-replica or multi-worker deployment configured with a non-shared
  root must fail at boot, not degrade with a warning.
- `[DUR-11]` Artifact roots are server-owned trusted storage: clients never
  supply artifact URIs or model bytes (chapter 11); only engine processes
  hold write credentials; model artifacts load only from server-computed
  keys under the configured root.

## Enforce restart and multi-worker invariants

- `[RST-1]` **Store, never memory.** Any fact required to answer an API call
  is readable from the durable plane. Fit records, calibration state,
  pending observations, orders, run statuses, artifact pointers: all durable
  at the moment they become observable, none reconstructed from process
  memory.
- `[RST-2]` **Fail at boot, not per request.** A serving process resolves its
  store configuration once at startup and refuses to start on an
  unserveable configuration (durable mode with no database URL, multi-worker
  mode with a non-shared artifact root). In-memory store implementations
  are test fixtures; a serving deployment cannot select one.
- `[RST-3]` **Kill-any-time equivalence.** Killing any replica between any
  two API calls and routing the next call to a different replica yields the
  same response the uninterrupted single replica would have given. Session
  identity makes this addressable: it is a pure function of defining inputs
  `[SES-1]`, so every replica derives the same key to the same state.
- `[RST-4]` **Jobs are crash-visible.** Every async job's status transitions
  are durable, and an executing job maintains a lease (or heartbeat) so a
  job orphaned by process death is detectable and transitions to a terminal
  failed state — never a perpetual "running" that no process owns.
  Idempotent submission `[DUR-3]` makes re-running safe.
- `[RST-5]` **Caches are transparent.** A replica may cache immutable
  artifacts `[DUR-8]` freely; every mutable fact (status, calibration state,
  pending buffer) is read through the store. No cache may change any
  observable response.

## Discipline schema migrations

- `[MIG-1]` One linear, forward-only migration lineage, versioned in the
  same repository as the code. Every schema change is a migration script;
  serving processes never auto-create or alter schema at runtime.
- `[MIG-2]` **Parity gate in CI:** the declared ORM/schema model diffed
  against the migration head is empty. A model change without its migration
  fails the build.
- `[MIG-3]` Migrations run as an explicit pre-rollout step (a Kubernetes Job
  or release hook), never concurrently from serving replicas; the runner
  takes a store-level lock so accidental concurrent invocation is safe.
- `[MIG-4]` **Expand/contract for zero downtime:** a rolling deployment
  implies old and new code briefly coexist against one schema, so additive
  changes ship first and destructive changes (drop, rename, retype) ship
  only in a release whose predecessor no longer reads the old shape.
- `[MIG-5]` A serving process verifies at startup that the store schema is
  at the revision its code expects, and reports not-ready otherwise.

## Expose health, readiness, and metrics surfaces

- `[MON-1]` **Liveness** (`/healthz`): process-local only — returns healthy
  whenever the process can serve the endpoint, with no dependency calls, so
  a store outage never triggers cascading pod restarts.
- `[MON-2]` **Readiness** (`/readyz`): gates traffic — verifies configuration
  resolved, relational store reachable, schema revision as expected
  `[MIG-5]`, artifact root resolvable. Not-ready removes the replica from
  the Service without killing it.
- `[MON-3]` **Metrics** (`/metrics`, Prometheus text exposition format), with
  at minimum: request count and latency per route; async job counts by
  terminal status; pending-observation backlog size and oldest-row age per
  tenant (the observe loop's lag signal, chapter 06); calibration-state
  write failures; compute-task retry count.
- `[MON-4]` Logs are structured and carry the correlating identities on
  every record: tenant, session, and run/job id, so one lifecycle is
  traceable across replicas and workers.

## Conformance

A conforming deployment must demonstrate, by test:

1. **Kill-and-resume:** run fit → kill the serving process → calibrate,
   order, and observe against a fresh process; results equal the
   uninterrupted sequence (`[RST-1]`, `[RST-3]`, `[DUR-6]`).
2. **Two-replica equivalence:** interleave one lifecycle's calls across two
   replicas sharing one relational store and artifact root; every response
   equals the single-replica trace (`[TOP-1]`, `[SES-1]`).
3. **Serialized state updates:** two replicas concurrently observing into
   the same `(session, partition)` end in a state equal to some serial
   order of the two updates — no lost update (`[DUR-5]`).
4. **No dangling pointer:** a crash injected between artifact write and
   pointer commit leaves no reader-visible pointer that fails to resolve
   (`[DUR-9]`).
5. **Boot-time fail-fast:** a serving configuration missing its database
   URL, or multi-worker with a non-shared artifact root, exits at startup
   with a diagnostic; it never serves and fails per-request (`[RST-2]`,
   `[DUR-10]`).
6. **Migration parity and gate:** CI fails on a model/migration-head diff
   (`[MIG-2]`); a replica booted against a stale schema reports not-ready
   and receives no traffic (`[MIG-5]`, `[MON-2]`).
7. **Orphaned-job detection:** killing a process mid-job yields a terminal
   failed status within the lease interval, observable from any replica
   (`[RST-4]`).
8. **Liveness isolation:** with the relational store down, liveness stays
   healthy while readiness reports not-ready (`[MON-1]`, `[MON-2]`).

## Provenance

For spec authors only; the chapter above stands without these. Positive space
from the old engine: `calibre/storage/models.py` + `postgres.py` (the three
classes already separated in practice: lifecycle/run/order/tuning rows,
`conformal_state` keyed `(session_id, partition)`, `forecast_pointers` as
URI+size pointer rows), `calibre/storage/lifecycle_repo.py` (dialect-native
atomic upserts → `[DUR-4]`; frames by reference, never inline),
`calibre/storage/session.py` (deterministic session hash → `[RST-3]`),
`calibre/api/main.py` (boot-time fail-fast on `LIFECYCLE_STORE=sql` without a
database URL → `[RST-2]`; `/healthz` + `/metrics` → `[MON-1]`, `[MON-3]`),
Alembic lineage under `calibre/storage/migrations/`. Negative space: durable
storage was *opt-in* (`LIFECYCLE_STORE=sql`, default in-memory — lost on
restart, invisible across workers) where `[RST-2]` makes durable the only
serving mode; `CALIBRE_ARTIFACT_URI` fell back to a local temp path with only
a logged warning where `[DUR-10]` demands boot failure; there was no
readiness probe distinct from liveness (`[MON-2]` adds it); background jobs
ran in-process with no lease, so a crash left statuses stuck (`[RST-4]`
closes that); and migration head/model parity was unenforced (`[MIG-2]`).
