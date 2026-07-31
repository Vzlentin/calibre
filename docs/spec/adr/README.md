---
title: "ADR index and template"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# ADRs — index and template

## Scope an ADR

An ADR records an architecture decision that **binds multiple chapters** —
e.g. "config-as-data over programmatic pipeline construction" or "a single
durable relational store for all run state". A decision owned by exactly one
architectural surface is recorded in its owning chapter, not here. Every ADR
is referenced from each chapter it binds.

## Rule: gated decisions never become public ADRs

Gated (Stage 1) decisions are never recorded as ADRs in the public
layer, in any form — not as context, not as "alternatives considered". They
live behind `40-gated-seams/` and the private annex, referenced only by
`[ANNEX:*]` pointers registered in `90-annex-registry.md`. An ADR that
states, implies, or narrows a gated ruling fails review.

## Index

| ADR | Title | Status | Binds |
|---|---|---|---|
| [0001](0001-reference-environment.md) | Pin the reference environments: Gate-A evidence on a versioned x86_64 Linux CI runner, the `[PRF-1]` headline on a workstation-class profile | ratified | 20, 30, 50 |
| [0002](0002-target-support-reconciliation.md) | Enforce target support at the reconciliation seam | draft | 02, 07, 21 |

## Name ADR files

`adr/NNNN-short-slug.md` — `NNNN` zero-padded, monotonically increasing,
never reused. A superseded ADR keeps its number and gains a "Superseded by
ADR NNNN" line in its Status section; it is never deleted.

## Use the template

Every ADR carries the spec-wide frontmatter contract — `title`, `status`
(`pre-gate-draft` | `gated-slot` | `evidence-pending` | `draft` |
`ratified`), `invalidation-tags` (possibly empty), `date` — and this body
shape:

```markdown
---
title: "ADR NNNN — <imperative decision statement>"
status: draft
invalidation-tags: []
date: YYYY-MM-DD
---

# ADR NNNN — <imperative decision statement>

## Status

pre-gate-draft | ratified | ratified, superseded by ADR MMMM.

## Context

The forces in tension, stated engine-independently in chapter 02 vocabulary.

## Decision

One imperative statement of what is decided, plus its testable consequences.

## Binds

The chapters this decision constrains, by file name.

## Consequences

What becomes easier, what becomes impossible, what must be revisited if the
listed invalidation-tags fire.

## Alternatives considered

Public-safe alternatives only; any private rationale stays behind a
registered [ANNEX:*] pointer.
```

## Provenance

None. This index restates only the ADR contract stated in
`00-overview.md`; it consumes no
old-engine behavior.
