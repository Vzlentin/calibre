---
title: "ADR 0001 — Pin the reference environments: Gate-A evidence on a versioned x86_64 Linux CI runner, the [PRF-1] headline on a workstation-class profile"
status: ratified
invalidation-tags: []
date: 2026-07-09
---

# ADR 0001 — Pin the reference environments: Gate-A evidence on a versioned x86_64 Linux CI runner, the [PRF-1] headline on a workstation-class profile

## Status

ratified (2026-07-09).

## Context

Two distinct obligations need an environment pin, and they pull in different
directions:

- **Comparability of minted numbers.** `[VN2-N2]` (chapter 20) makes a cost
  total comparable only under a pinned (config, toolchain, architecture)
  triple — floating-point results of model paths are known to diverge across
  CPU architectures and numeric-library builds. Every tracked total must
  therefore be minted in an environment whose identity is recorded and
  reproducible on demand.
- **The performance budget.** `[PRF-1]` (chapter 30) binds a 15-minute
  full-M5 wall-clock bar to "the pinned reference environment". Chapter 30
  assesses that bar plausible on **workstation-class hardware only**; the
  74.7-minute measured baseline was produced on a laptop-class machine
  (Dell Precision 3480, Intel Core i7-1370P, 32 GB RAM, Windows 11,
  Python 3.12, single process) and remains directional context — never a
  comparison surface — with laptop-class feasibility of the bar explicitly
  not established.

Functional acceptance evidence (does the engine produce row-exact protocol
artifacts; do the gates bite) needs availability, auditability, and a
stable public identity more than it needs raw speed. Headline performance
measurement needs the hardware class the bar was assessed against. One
environment cannot serve both roles well, and leaving either unpinned lets
evidence bind to whatever machine happened to run it.

## Decision

Pin **two environment roles**, one per obligation:

1. **Acceptance-evidence role (Gate A and all skeleton-era tracking
   records).** All acceptance evidence and every tracked cost-regression
   record minted before the performance architecture lands runs on a
   GitHub-hosted CI runner with the **explicitly versioned label
   `ubuntu-24.04`** — never `ubuntu-latest` — on **x86_64**, with the
   interpreter pinned to **Python 3.12** and all packages installed from the
   successor's committed lockfile (`--locked`; numeric libraries are the
   locked manylinux wheels and their bundled BLAS). Every evidence run
   records an environment manifest: runner-image build, CPU model, OS
   release, Python patch version, numeric-library/BLAS provenance,
   thread policy (declared explicitly in the workflow environment, never
   inherited), and lockfile sha256. Within that manifest, the
   **comparability key** is the `[VN2-N2]` triple — CPU architecture, OS
   release, lockfile sha256 — plus the config/input/capture digests and
   `actuals_semantics` of the run; the remaining facts are per-run
   provenance, recorded but never matched. Development machines (including
   the maintainer's Windows workstation) are convenience venues only and
   mint no evidence.

2. **Headline-performance role (`[PRF-1]`, measured at Gate C).** The
   15-minute bar binds to a **workstation-class x86_64 Linux profile**: at
   minimum 16 physical cores and 64 GB RAM, explicitly versioned OS release,
   Python 3.12, locked toolchain, declared thread policy. The concrete
   instance's full facts (CPU model, core/RAM configuration, OS release,
   BLAS provenance, thread policy, lockfile sha256) are recorded in the
   standard profile deliverables of the first budget run and appended to
   this ADR's "Concrete instance" section when that machine is stood up —
   recording facts under an already-made class decision, not reopening it.
   No performance claim is minted on the acceptance-evidence runner, and no
   functional gate depends on the workstation profile.

Testable consequences: Stage 3 evidence workflows contain no
`ubuntu-latest` runner label; a non-x86_64 runner fails the evidence
preflight; an evidence artifact missing any manifest field is refused at
promotion; chapter 30 carries no pending environment decision.

## Binds

- `20-protocol-vn2.md` — `[VN2-N2]` totals mint under role 1 until Gate C;
  the flagship measurement environment is role-resolved here.
- `30-performance.md` — the pending reference-environment decision resolves
  to role 2 for `[PRF-1]`; the laptop baseline attribution stands as
  directional context.
- `50-test-and-oracle-strategy.md` — tier venues: all tiers' CI evidence
  runs on role 1; tolerance-class-6 cross-environment spread is measured
  against these pinned roles.

## Concrete instance (role 2)

Pending — filled when the Gate C machine is stood up; the class decision
above does not wait for it.

## Consequences

- Easier: evidence is auditable (public CI logs, recorded manifests),
  available (no single physical machine on the critical path), and
  comparable (one key, mechanically matched).
- Impossible: minting tracked numbers on unversioned or local
  environments; comparing totals across environments whose comparability
  keys differ; smuggling a performance claim out of a functional CI run.
- Revisit: if GitHub retires the `ubuntu-24.04` label, a superseding ADR
  pins the next explicitly versioned label; the OS-release change ends the
  comparability window and new reference totals are minted fresh, exactly
  as `[VN2-N2]` prescribes. If workstation-class procurement fails by Gate
  C, the `[PRF-1]` bar is re-scoped by ADR, never silently rebound.

## Alternatives considered

- **Pin the laptop class for `[PRF-1]`.** Rejected: chapter 30 explicitly
  declines to claim the 5× gap closes on that machine, and a personal
  Windows laptop is not a reproducible measurement venue.
- **One environment for both roles (self-hosted workstation for
  everything).** Rejected: couples day-to-day acceptance evidence to a
  single physical machine's uptime and trust story; hosted runners give
  functional evidence better availability and a public audit trail.
- **`ubuntu-latest` for CI.** Rejected: the alias migrates silently,
  changing the OS release under the comparability key mid-window.
