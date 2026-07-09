---
title: "Performance — baseline, budget, and the architecture it forces"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 30 — Performance

This chapter was an evidence-pending slot; the evidence has landed. A complete
full-M5 backtest of the old engine (the behavior oracle) was profiled on
2026-07-07 with stage-level telemetry, and this chapter now states three
things: (1) the measured baseline, restated engine-independently as the
rewrite's **cost baseline**; (2) the rewrite's **performance budget**; (3) the
**architectural requirements** the gap between them forces. Raw telemetry,
per-origin distributions, and run logs sit behind
`[ANNEX:30-profile-raw-data]`.

The chapter uses chapter 02 vocabulary verbatim: origin, horizon, panel,
forecast task, forecast frame, calibration state, partition, ledger,
pending/resolved rows.

## State the baseline workload

The baseline is one full-M5 backtest — the largest workload the spec commits
to — with this shape:

- **Panel**: 30,490 bottom series; a 59.2M-row long-format bottom panel; an
  aggregation lattice of 33,563 hierarchy nodes whose node panel is 65.1M
  rows.
- **Origins**: 64 consecutive daily origins, replayed in order, horizon 28.
- **Pipeline**: one *global* forecast task (a seasonal-naive baseline model —
  deliberately trivial fit); structural-weights (WLS) point reconciliation
  over the lattice; a sequential-adaptive conformal method with per-horizon-
  step partitions; ordering enabled; a streaming ledger resolved as actuals
  arrive.
- **Machine**: laptop-class reference — Dell Precision 3480, Intel Core
  i7-1370P (14 cores / 20 threads), 32 GB RAM, Windows 11, Python 3.12,
  single process. Most cores were idle throughout: the run exercised
  essentially one core.
- **Ledger artifacts**: 1.01 GB pending ledger + 1.59 GB resolved ledger
  (columnar files).

Single run, machine otherwise lightly loaded; no variance estimate. Peak RSS
was **not** captured — a known gap this chapter closes for all future
profiles (see "Require the standard profile deliverables").

## State the measured baseline

**Total wall clock: 4,479.7 s = 74.7 minutes**, 99.8% attributed.

| Segment | Seconds | % of wall |
|---|---:|---:|
| Pre-origin (load → validate → node panel → task build) | 238.8 | 5.3% |
| Origin loop (64 origins) | 4,073.7 | 90.9% |
| Ledger close (final resolve + artifact write) | 159.9 | 3.6% |

Origin durations: min 38.3 s, mean 63.7 s, max 190.6 s.

Per-origin pipeline phases (disjoint; shares are of total wall):

| Phase | Total s | % wall |
|---|---:|---:|
| Resolve open ledger rows (actuals resolution + observe) | 1,213.2 | 27.1% |
| Predict (fit + predict + panel staging) | 1,363.9 | 30.4% |
| Reconcile | 163.0 | 3.6% |
| Calibrate | 253.9 | 5.7% |
| Order | 0.0 | 0.0% |
| Commit (append pending rows, resolve same-origin rows) | 1,075.9 | 24.0% |

Cross-cutting span totals (spans nest inside phases; **not** additive with
the phase table):

- **Actuals lookup — 1,686.6 s, 37.7% of wall** (128 calls, 2 per origin):
  deriving which resolved observations match which pending ledger rows, by
  re-scanning the 65.1M-row node panel and the growing ledger at every
  origin.
- **Ledger maintenance — 503.8 s, 11.2%**: append, resolved-row updates,
  resolution-frame assembly, final close.
- **Model refit — 833.1 s, 18.6%**: the global model is refit from scratch at
  every origin, *even though the model is a trivial seasonal-naive*. Predict
  itself is 29.2 s (0.7%).
- **Panel staging — ~502 s, 11.2%**: per-origin forecast-frame/history
  assembly around the model, rebuilt each origin from the same immutable
  panel.
- **Conformal state apply + observe — 476.3 s, 10.6%**: per-partition
  calibration-state updates, applied row-set by row-set.

Pre-origin decomposition, from a standalone re-measurement of the pre-origin
stages (246.6 s total, agreeing with the 238.8 s in-run pre-origin gap to
~4%): wide→long panel reshape 11.8 s, bundle validation 7.5 s, memory
preflight estimate 94.0 s, node-panel construction 112.8 s, task build
20.5 s.

### Read the profile: data movement dominates, math is nearly free

The pipeline's mathematical core — reconciliation (3.6%) plus calibration
(5.7%) — is **under 10% of wall**. Roughly 49% is actuals lookup plus ledger
maintenance: bookkeeping that re-derives, at every origin, state that changed
only incrementally since the previous origin. Another ~11% re-stages
immutable history, and ~19% refits a model whose fit is analytically trivial.
The baseline engine's cost is **data movement, not modeling**. This is the
single most important input to the rewrite's engine-core design (chapter 03):
naive per-origin recomputation over a panel of this size costs ~64 s/origin
before any real model runs.

## Set the performance budget

- `[PRF-1]` **A full-M5 backtest (the workload above) completes in ≤ 15
  minutes wall clock on the pinned reference environment.** Against the
  74.7-minute laptop baseline this demands ~5×. The bar is assessed
  **plausible on workstation-class hardware**; feasibility on the laptop
  class that produced the baseline is **not established**, and this chapter
  makes no claim that the 5× gap closes on that machine. No
  micro-optimization of the baseline design reaches the bar on any hardware
  class; the architectural requirements below are what close the gap,
  together with the parallel headroom the single-process baseline left
  untapped. Which hardware class the bar binds to is resolved by the
  reference-environment pinning decision below.
- `[PRF-2]` Pre-origin overhead (everything before the first origin) is ≤ 60 s
  for the full-M5 workload. In particular, any memory-preflight guard must be
  O(metadata), not a scan of the panel (the baseline spent 94 s estimating
  expansion by group-by).
- `[PRF-3]` Budget compliance is measured by the standard profile deliverables
  (below), produced by the rewrite's own benchmark harness — the budget is a
  CI-checkable fact, not a claim.

**Reference-environment pinning — resolved by
[ADR 0001](adr/0001-reference-environment.md).** The budget in `[PRF-1]`
binds to the environment roles that ADR pins: acceptance evidence and
skeleton-era tracking records run on an explicitly versioned x86_64 Linux
CI runner, while the 15-minute bar of `[PRF-1]` binds to a
workstation-class x86_64 Linux profile whose concrete instance facts are
appended to the ADR when the Gate C machine is stood up. The laptop-class
machine above remains the provenance of this chapter's measured numbers —
directional context, never a comparison surface. This is an engineering
decision, not a gated seam; no material for `40-gated-seams/`.

## Derive the architectural requirements

Each requirement names the baseline cost it eliminates. These bind chapter 03
(engine core) and the plugin chapters; this chapter owns the numbers.

- `[PRF-10]` **Incremental actuals and ledger indexing.** Between consecutive
  origins, the admissible-observation set and the pending-row set change only
  by the rows the elapsed period added or resolved. Actuals resolution and
  ledger maintenance must therefore cost O(newly admissible observations +
  newly resolved rows) per origin — never a rescan of the full panel or full
  ledger. Eliminates the ~49% resolve/commit block.
- `[PRF-11]` **Staging reuse.** The historical panel is immutable within a
  run; per-origin task histories are views/slices over one staged
  representation, built once. Per-origin staging cost must be O(slice
  boundary), not O(panel). Eliminates the ~11% staging block.
- `[PRF-12]` **Incremental fit.** The forecasting-plugin protocol (chapter
  04) must let a plugin declare an *update* path (extend fitted state by one
  period) distinct from *refit*; the engine must use it when declared, and a
  trivial model's per-origin cost must be near zero. Refit-per-origin remains
  the fallback for plugins without an update path. Eliminates most of the
  ~19% refit block for cheap models.
- `[PRF-13]` **Vectorized calibration-state updates.** Applying and observing
  calibration state across partitions must be a batch/vectorized operation
  over the partition axis, not per-partition iteration. Targets the ~11%
  conformal apply+observe block. Round-trip and keying invariants `[CAL-1]`,
  `[CAL-2]` are unchanged.
- `[PRF-14]` **Parallelism along the permitted axes.** Under `[INV-TEMPORAL]`
  and `[CAL-3]` the dependency structure is: fit/predict at origin `o`
  depends only on the immutable panel before `o` (parallelizable across
  tasks, series, and even ahead across origins), while
  calibrate → order → observe is sequential *per partition* along the origin
  order. The engine must exploit series/task parallelism within an origin and
  may pipeline state-free work across origins; it must never reorder
  state-bearing updates within a partition. The baseline used ~1 of 14 cores;
  this is the largest untapped lever.

## Set the memory budget

- `[PRF-20]` The full-M5 workload fits in **32 GB RAM single-node** (the
  baseline machine class ran it; the rewrite must not regress this).
- `[PRF-21]` The reconciliation memory pivot: at ~30k bottom series the dense
  summing-matrix representation costs ~7.6 GiB for the matrix alone
  (33,563 lattice nodes × 30,490 bottom series × 8 bytes per float64); the
  lattice-shaped operators of chapter 07 must be sparse-first at and above
  this scale. Dense paths are permitted only below a configured series-count
  threshold.
- `[PRF-22]` Ledger artifacts at this scale are gigabyte-class (1.01 GB
  pending / 1.59 GB resolved in the baseline); ledger I/O must stream —
  appending an origin's rows must not require materializing the full ledger
  in memory.
- `[PRF-23]` Peak RSS per pipeline stage is a mandatory profile deliverable;
  no memory ceiling in this chapter can tighten past `[PRF-20]` until the
  first rewrite profile delivers it (the baseline profile did not capture
  RSS).

## Require the standard profile deliverables

Every future performance profile (baseline re-runs and all rewrite profiles)
must deliver, machine-readably:

- `[PRF-30]` **Stage-level wall time**: per-origin duration for every spine
  stage (resolve, fit, predict, reconcile, calibrate, order, commit) plus
  pre-origin and close segments, with totals reconciling to ≥ 99% of
  end-to-end wall clock.
- `[PRF-31]` **Peak RSS per stage**, same granularity as `[PRF-30]`.
- `[PRF-32]` **Scaling curve vs. series count**: wall clock and peak RSS at
  no fewer than three panel sizes (order of 1k / 10k / full ~30k bottom
  series), same config otherwise, so super-linear stages are visible.
- `[PRF-33]` **Parallel efficiency** once distributed execution exists:
  speedup vs. worker count for the fan-out stages, reported alongside the
  single-process number.

## Acceptance criteria

1. The rewrite's benchmark harness runs the full-M5 workload on the reference
   environment and reports wall clock ≤ 15 min `[PRF-1]`, pre-origin ≤ 60 s
   `[PRF-2]`, within 32 GB `[PRF-20]`.
2. Per-origin resolve+commit cost is demonstrated flat (not growing with
   ledger size) across the 64-origin run `[PRF-10]`.
3. A trivial-model run shows per-origin fit cost near zero via the update
   path `[PRF-12]`.
4. A profile artifact containing `[PRF-30]`–`[PRF-32]` (and `[PRF-33]` when
   applicable) is produced by the same harness invocation that produces the
   benchmark result — never assembled by hand.

## Provenance

For spec authors only; the chapter stands without these. The baseline numbers
restate the old-repo profiling evidence at `benchmarks/m5/profile/` (MEMO
plus `phase-aggregates.json`, `preorigin-timings.json`, `run-log.jsonl`),
measured on the old engine's full-M5 config at a pinned commit. Positive
space: the old engine's per-origin phase telemetry made the attribution
possible and the rewrite keeps that discipline (`[PRF-30]`). Negative space:
its per-origin actuals lookup, ledger rewrites, staging rebuilds, and
refit-per-origin are the anti-patterns `[PRF-10]`–`[PRF-12]` exist to
eliminate; its memory preflight (94 s group-by scan) motivates `[PRF-2]`; its
missing RSS capture motivates `[PRF-31]`.
