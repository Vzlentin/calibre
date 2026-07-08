---
title: "Annex registry — the public-to-private pointer index"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 90 — Annex registry

This file is the complete index of `[ANNEX:*]` pointers used anywhere in the
public spec layer, and the leak-review surface for the two-layer contract
stated in `00-overview.md`.

## State the registry rules

1. **Every pointer is registered.** Every `[ANNEX:*]` pointer appearing in
   any public spec file is listed here with its owning chapter and a one-line
   public-safe description of the material class behind it. A pointer in use
   but not registered fails review; a registered pointer no longer in use is
   removed in the same change that drops its last use.
2. **Pointers are one-way, public to private.** A public file may name an
   annex pointer; no annex content is ever inlined into a public file, and
   the annex never links back into the public layer or is required to read
   it.
3. **The annex never ships.** No private path, repository, or storage
   location appears anywhere in the public layer; the opaque pointer is the
   only admissible reference.

New chapters — including post-gate `40-gated-seams/` chapters — register their
pointers here in the same change that introduces them. The registry is
mechanical aggregation, kept current; it decides nothing.

## Register the pointers

| Pointer | Owning chapter | Material class |
|---|---|---|
| `[ANNEX:01-flagship-metric-decision]` | 01 | Gated decision record (Stage 1 flagship metric). |
| `[ANNEX:05-method-families-survey]` | 05 | Survey (conformal method families and their assumptions). |
| `[ANNEX:07-coherence-decision]` | 07 | Gated decision record (coherent reconciliation scope and non-additivity position). |
| `[ANNEX:08-cost-objective-derivation]` | 08 | Derivation (coherent cost objective). |
| `[ANNEX:20-vn2-replication-notes]` | 20 | Replication notes (VN2 protocol). |
| `[ANNEX:21-m5-scoring-exemption-record]` | 21 | Ratified decision record (M5 scoring exemptions). |
| `[ANNEX:30-profile-raw-data]` | 30 | Benchmark evidence (full-M5 performance profile raw data). |
| `[ANNEX:41-seam-decision-record]` | 41 | Gated decision record (seam placement and positions; derivations and rejected alternatives). |
| `[ANNEX:42-flagship-metric-record]` | 42 | Gated decision record (flagship-metric decision and its sub-decisions). |
| `[ANNEX:50-oracle-curation-record]` | 50 | Decision record (test-oracle corpus curation). |

## Provenance

None. This registry mechanically aggregates pointers already declared by
their owning chapters; it consumes no old-engine behavior.
