---
title: "Gated seams — chapter directory"
status: draft
invalidation-tags: []
date: 2026-07-20
---

# 40 — Gated seams

## State the directory

This directory carries the chapters that bind the gated decisions of the
rewrite, ratified at the Stage 1 gate (the opaque gate label is defined
once, in `00-overview.md`). The chapter set is fixed:

| File | Contract |
|---|---|
| `41-decision-calibration-seams.md` | The four decision-calibration seams, the scope declaration, and their bindings into the tree (`[SEAM-*]`). |
| `42-flagship-metric.md` | The flagship metric — the two-axis headline claim and its reporting discipline (`[FLG-*]`). |

A change to this chapter set — adding, removing, or re-scoping a file —
reopens every binding these chapters state and fails review unless it lands
together with a new decision record behind a registered `[ANNEX:*]` pointer
and a re-review of every chapter that cites the affected `[SEAM-*]`/`[FLG-*]`
requirements.

## Record the inbound seam hooks — all bound

Each pre-gate chapter that touched a gated surface carried a marked slot
deferring here. Every hook is now bound; the table records where.

| From chapter | Interface the slot bound | Bound by |
|---|---|---|
| `01-vision-and-commitments.md` | The flagship metric that headlines the product. | Chapter 42, `[FLG-2]`; binding stated in chapter 01. |
| `02-domain-model.md` | Normative force of the structural terms **coherent cost** and **hierarchical coverage**. | Chapter 41, `[SEAM-4]` and `[SEAM-5]`/`[SEAM-6]`; bound statements in chapter 02. |
| `05-conformal-plugins.md` | Claims joint or simultaneous across partitions or hierarchy nodes; the `joint_claim` manifest field. | Chapter 41, `[SEAM-5]`–`[SEAM-7]`; bound in chapter 05, including the calibration context. |
| `07-reconciliation.md` | (a) Output-column contract of the reconciliation stage; (b) the non-additivity position on forecast quantities. | Chapter 41, `[SEAM-2]` and `[SEAM-3]`; bound in chapter 07. |
| `08-ordering-and-cost.md` | The cost functional above the bottom level — what "optimal" means there. | Chapter 41, `[SEAM-4]`; bound in chapter 08. |
| `09-tuning.md` | The default tuning objective, bound symbolically to the chapter 08 objective. | Chapter 41, `[SEAM-4]`, through chapter 08's exported objective; bound in chapter 09. |
| `20-protocol-vn2.md` | Which headline figure the rewrite reports on VN2. | Chapter 42, `[FLG-1]`/`[FLG-2]`; bound in chapter 20. |
| `21-protocol-m5.md` | The sales-scoring role for M5 hierarchical diagnostics. | Bound in chapter 21 (`[M5-X*]`) — ratified ahead of the gate; no slot remained. |
| `50-test-and-oracle-strategy.md` | Which joint or simultaneous coverage claims the test oracle may state. | Chapter 41, `[SEAM-5]`/`[SEAM-6]`; bound in chapter 50. |

## State the drafting rules

1. Chapters in this directory state **interfaces and claims only**; every
   derivation, alternative, and rationale sits behind an `[ANNEX:*]` pointer
   registered in `90-annex-registry.md`. Annex material inlined into a
   public file fails review.
2. **No positioning language** — the spec-wide rule (`00-overview.md`,
   design rule 5) applies here as everywhere: contracts only, no novelty,
   priority, or differentiation claims for any method or mechanism.
3. Chapters here follow the spec-wide frontmatter contract (`title`,
   `status`, `invalidation-tags`, `date`) and the status legend in
   `00-overview.md`.

## Provenance

The chapters in this directory carry the ratified Stage 1 decision texts
verbatim and bind them to the tree; decision records sit behind their
registered annex pointers. This directory consumes no old-engine behavior.
