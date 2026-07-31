---
title: "ADR 0002 — Enforce target support at the reconciliation seam"
status: draft
invalidation-tags: []
date: 2026-07-31
---

# ADR 0002 — Enforce target support at the reconciliation seam

## Status

draft (2026-07-31).

## Context

Some protocols define targets whose support is narrower than the real line.
M5 sales are non-negative unit counts, while generic demand-planning panels
may be real-valued. Least-squares reconciliation can produce tiny negative
roundoff residue even when the mathematical projection is zero. If support is
left implicit, conformal calibration can consume numerically invalid points or
adapters can invent local clipping policies that differ across strategies.

## Decision

Make target support an explicit panel fact and enforce it once after point
reconciliation, before conformal calibration.

- A canonical panel declares exactly one target support: `REAL` or
  `NONNEGATIVE`. Protocol compilers own protocol-specific declarations; run
  configuration does not author support.
- The engine passes the panel's target support through the reconciliation
  context. Every reconciler output must respect that context.
- Adapter-backed projection strategies return reconciled point values together
  with a finite non-negative absolute numerical-error bound derived from the
  summing matrix width, output magnitude, floating-point precision, and solver
  facts when a sparse solver is used.
- For `NONNEGATIVE`, values in `[-bound, 0)` canonicalize to exactly `0.0`.
  Values below `-bound` fail with an error naming the model, origin, horizon
  step, and series key. For `REAL`, finite values are preserved unchanged.
- Conformal calibration consumes only support-valid reconciled points. Any
  one-sided claim remains a claim of the conformal method; reconciliation does
  not add a conformal clamp binding.

## Binds

- `02-domain-model.md` — panels declare target support as canonical domain
  data.
- `07-reconciliation.md` — reconcilers receive target support and enforce the
  postcondition centrally.
- `21-protocol-m5.md` — M5 sales compile as non-negative-supported targets.

## Consequences

Support handling becomes uniform across native and adapter-backed strategies,
and tiny solver residue no longer aborts non-negative protocols. Materially
negative outputs are no longer silently repaired by conformal calibration,
configuration floors, or adapter-local clipping. If a future protocol needs a
different support, it must extend the canonical support vocabulary explicitly
rather than infer it from sample values.

## Alternatives considered

- **Configure an M5 or conformal floor.** Rejected because support is a domain
  fact, not a run knob, and conformal calibration should not repair invalid
  point forecasts.
- **Clip inside each adapter.** Rejected because it would duplicate policy and
  make native and Nixtla-backed strategies diverge.
- **Reject the representative M5 pipeline.** Rejected because the residue is a
  numerical artifact of an otherwise valid projection.
- **Author a new non-negative reconciler.** Rejected because the existing
  reconciler family is adequate once its support postcondition is explicit.
