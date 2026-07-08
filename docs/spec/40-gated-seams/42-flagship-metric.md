---
title: "Gated seams — the flagship metric"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 42 — The flagship metric

This chapter binds the flagship-metric slot: the single headline claim by
which the product states its value, its metric family, the protocol chapter
that computes it, and the reporting discipline that keeps it honest. It
binds the marked slots in chapters 01 and 20. The metric statement between
the dividers below is the ratified text, carried verbatim; the binding
requirements that follow attach it to the tree. The decision record sits
behind `[ANNEX:42-flagship-metric-record]`. Requirements carry `[FLG-n]`
tags for citation by tests and later chapters.

---

## The flagship metric

The engine's flagship claim is a joint statement with two axes, published
together and never quoted separately:

1. **Certificate (gated).** On the flagship benchmark run, realized
   coverage of the decision bound at the cost-derived fractile
   tau = Cu/(Cu+Co) lies within a pre-registered finite-sample acceptance
   band (binomial/Wilson interval computed at the actual coverage-event
   count) over a declared post-warmup window. Coverage is scored
   demand-honestly: the scored series is demand, not raw sales, and the
   run names its scored series explicitly.
2. **Price (tracked, not gated).** Total realized cost of the
   guarantee-on configuration, published as a ratio to a cost-tuned
   reference configuration tuned fresh on the same engine. The ratio is
   evidence with a target direction, tracked run-over-run by
   cost-regression tracking; it is not an acceptance gate.

## Measurement procedure

- **Protocol.** The VN2 multi-round ordering loop, treated as
  protocol-as-data: rounds, lead time, review period, and the holding and
  shortage cost rates are competition inputs, not engine choices.
- **Coverage event.** One event per (series, decision origin): the
  realized demand sum over the protection window is at or below the
  calibrated one-sided bound that was consumed as the order-up-to level.
- **Guarantee-on configuration.** Coverage target equal to the critical
  ratio in every decision slot; no clamp on the decision bound; the
  unweighted split-conformal branch (the exchangeability-carrying one);
  runtime defaults guarantee-correct. The configuration is minted fresh
  on this engine — no constants ported from any prior engine.
- **Pooling window.** Declared in advance of the run: post-warmup rounds
  only, with the full per-round coverage trajectory always published
  alongside the pooled number. Pooling across all rounds is rejected — it
  buries the calibrated steady state under the warmup transient.
- **Cost accounting.** Simulator-booked holding plus shortage over the
  settle path, identically accounted for the flagship and the reference
  configuration; the delta is reported as a ratio.
- **Reference discipline.** Historical totals from any prior engine are
  reference points only — never targets, never denominators. The
  reference configuration is re-tuned on the engine under measurement.
- **Environment.** The flagship numbers are pinned to one named
  environment (architecture, OS, dependency lockfile), recorded with the
  run.

## Explicit non-claims

- Coverage at tau is a **calibration property of the per-decision bound**,
  not a multi-period cost-optimality claim. The cost fractile is
  single-period-exact and multi-period-myopic; the certificate never
  certifies that ordering at tau minimizes multi-period system cost — the
  price axis exists precisely to report that gap honestly.
- The price ratio validates nothing about the guarantee, and the
  certificate implies nothing about cost. The two axes are independent
  and must be published together.
- No calibration-honesty claim is made on datasets whose demand ground
  truth is unrecoverable; coverage numbers on such datasets are labeled
  sales-coverage and never quoted as service-level or honesty evidence.

## Revisit checkpoint

On evaluation windows containing no stockouts, demand-scored and
sales-scored coverage coincide by construction, so the honesty of the
scored series is undiscriminated there. A discriminating measurement —
an evaluation window containing stockouts, or a censoring-indicated
dataset — is a declared checkpoint before the metric is frozen.

---

## Bind the metric into the tree

- `[FLG-1]` **The flagship protocol is chapter 20.** The flagship benchmark
  run is the VN2 protocol as restated there; the certificate's coverage
  events, pooling window, and run configuration bind to the measurement
  surfaces chapter 20 fixes (`[VN2-R3]`–`[VN2-R5]`). The certificate scores
  the demand-resolved coverage surface of `[VN2-R5]`; where the evaluation
  window's demand is unrecoverable, no certificate is minted — the
  non-claims and the revisit checkpoint above govern. The recorded
  environment pin is the union (run configuration, architecture, OS,
  dependency lockfile and toolchain), which satisfies `[VN2-N2]`'s
  comparability triple. This binds the flagship-figure slot in chapter 20.
- `[FLG-2]` **One flagship, two axes, one publication.** The certificate
  and the price ratio are the product's only headline figures; they are
  published together with the per-round trajectory, and no other number is
  designated flagship by any spec chapter or external material. This binds
  the flagship-metric slot in chapter 01.
- `[FLG-3]` **The acceptance band is pre-registered.** The band's interval
  type and confidence level are fixed in a recorded pre-registration
  before any certified run, at the event count and environment that run
  declares; they are never chosen or adjusted at run time. A certified
  number names its pre-registration.
- `[FLG-4]` **The certificate scores the descriptor's claim.** The
  coverage event realizes the one-sided decision-bound claim of chapter 41
  (`[SEAM-1]`, `[SEAM-5]`): certification requires descriptors whose claim
  is one-sided coverage in the finite-sample-marginal currency at the
  cost-derived fractile level (`[GRT-2]`/`[GRT-3]`), scored under the
  descriptor's registered predicate, with the scored-series label
  demand-honest per the score-input contract (chapter 05,
  `[CNF-27]`–`[CNF-28]`). A run whose scored-series label is not
  demand-honest, or whose bounds carry any clamp (`[SEAM-8]`), is
  ineligible for certification.
- `[FLG-5]` **The reference configuration is engine-fresh.** The price
  ratio's denominator is re-tuned on the engine under measurement in
  reference-tuning mode (chapter 09, `[TUN-24]`) against the chapter 09
  default objective — a labeled, never-certified mode whose bounds carry
  no coverage claim; no historical total from any prior engine appears as
  a target or denominator in any published figure.

## Conformance

A conforming implementation must demonstrate, by test:

1. The certificate computation rejects a run whose descriptor claim is
   not one-sided coverage in the finite-sample-marginal currency, whose
   level is not the cost-derived fractile, whose scored-series label is
   not demand-honest, or whose bounds were clamped (`[FLG-4]`).
2. The pooled coverage number is computed over exactly the
   declared-in-advance post-warmup window (`[FLG-1]`), and the published
   artifact carries the full per-round trajectory beside it.
3. The price ratio's two totals are produced by the same settle-path
   accounting on the same protocol inputs, differing only in
   configuration (`[FLG-5]`).

## Provenance

The metric statement between the dividers is the ratified Stage 1 flagship
decision, carried verbatim; the decision record and its sub-decisions are
private spec-author material at `[ANNEX:42-flagship-metric-record]`. This
chapter consumes no old-engine behavior.
