# À chaud: the first full M5 gate run, and the performance cascade it exposed

**Date:** 2026-06-10
**Trigger:** making the full M5 benchmark (30,490 series, 64 daily origins, h=28,
bottom_up reconciliation, MSCP per-horizon conformal at `partition: series`) the
Wave 2 quality gate instead of VN2.
**Host:** Windows 10 laptop, 16 GB RAM, 4 physical cores.

**Status of this document:** this is the *à chaud* analysis — measurements,
mechanics, and candidate directions captured the same day, while the evidence
was live (py-spy dumps, phase logs, run restarts). It deliberately makes **no
architectural decisions.** A separate *à froid* post-mortem and architecture
refactor planning session, run with fresh context, owns the decisions; this
document is its primary input.

## Headline

The first attempt to actually run our own acceptance benchmark end-to-end
failed four independent ways before producing a single resolved ledger row.
Every failure was invisible at unit-test and VN2 scale and obvious within
minutes at M5 scale. Forecasting 30,490 series with SeasonalNaive is ~0.5 s of
math per origin; the engine was spending **4.5+ minutes** per origin around it.

| Stage | Run state | Per-origin wall | Projected total |
| --- | --- | --- | --- |
| Calendar `d` bug (#146) | crashed at load | — | could not run at all |
| Dense summing matrix (PR #145 fix 2) | ~2 h in first reconcile, 12 GB RSS, zero output | unbounded | days, if it survived memory |
| Per-series Ray dispatch (#149) | ran | ~270 s | ~5 h |
| iterrows conformal runtime (#150) | ran | ~160 s | ~2.8 h |
| After all fixes (gate run 3) | ran | ~45 s warmup, **~150 s steady state** | **~2.5 h** |

The remaining ~130 s/origin is the next unfixed layer (see "Open bottlenecks").

## The failure cascade, in order of discovery

### 1. The harness could not load its own data (PR #146)

`benchmarks/m5/download_m5_data.py` downloads via the `datasetsforecast`
mirror, whose `calendar.csv` has no `d` column; `melt_m5_sales` required one.
The M5 loader had never been run against the data the M5 download script
produces — the smoke fixture was hand-built in Kaggle shape. **Fix:** derive
`d_N` positionally from the date-sorted calendar with a daily-contiguity guard.

> Lesson: a benchmark harness that has never been executed end-to-end is
> documentation, not a gate. The data-loading path is part of the benchmark.

### 2. Dense summing matrix at hierarchy scale (fixed inside PR #145, in flight at time of writing)

`build_summing_matrix` materializes a dense `(33,563 × 30,490)` float64 `S`
(~8.2 GB), and `SummingMatrix.subset` fancy-index-copied it once per
`(model, origin, h)` cross-section — 28 copies per origin. py-spy showed the
driver pinned inside `subset` ~2 h into the run with 12 GB RSS and nothing
written. **Fix:** `BottomUpReconciler` now does grouped member sums per
hierarchy attribute (`S` entries are 0/1 memberships, so `S @ bottom` *is* a
groupby-sum) — 3.4 s per origin, no matrix at all.

> Lesson: dense hierarchy algebra is O(nodes × bottoms) memory. At M5 scale
> that is gigabytes. Any path that "just multiplies by S" must justify itself.
> The eager Nixtla paths (`apply.py`, `hierarchical_intervals.py`) still use
> dense S and remain guarded by the #136 memory preflight — they have the same
> wall waiting behind them if they ever need to run at full M5 scale.

### 3. Per-series task dispatch (PR #149 — a workaround, not the fix)

The benchmark config ran SeasonalNaive at default local scope: **30,490
ForecastTasks per origin**. The mechanics, from the code: `execute()` stages
**one parquet file per (series, model) task** (`ForecastTask.to_uri`,
30,490 files); every origin then makes 30,490 `ray.remote()` submissions,
each worker **re-reads its per-series parquet** (`ForecastTaskRef
.materialize()`), truncates, fits a ~2 ms model, and returns a tiny frame the
driver deserializes single-threaded (`ray.get` over 30,490 refs). Over 64
origins: ~2M scheduling round-trips and ~2M small file reads around ~0.5 s of
actual math per origin. py-spy showed the driver pinned in
`ray deserialize_objects`.

**Why it exists:** the URI-backed task-ref design is rational for *few large*
tasks (big histories stay out of the object store; workers pull from staging,
including S3). The defect is that the **dispatch unit is conflated with the
model-semantics unit** — "local model" (fit per series) silently became "one
distributed task per series", multiplying a few-big-tasks design by the
series count.

**Shipped mitigation (#149):** `scope: global` — one vectorized statsforecast
panel call per origin (0.5 s), byte-identical forecasts for SeasonalNaive
(verified with strict `assert_frame_equal` before switching). This is a
config workaround that happens to be correct for a single homogeneous local
model; it is **not** the architecture for Calibre's core use case of many
series × many models ensembled locally.

> Lesson: task granularity is a first-class architectural decision,
> independent of model scope. Per-series dispatch is never the right
> scheduling unit for panel-capable backends.

### 4. Row-wise conformal runtime (PR #150)

At `partition: series` M5 emits 939,764 interval rows per origin.
`_apply_perhorizon` looped 33,563 groups with `iterrows()` inside (~106
µs/row → **99.8 s/origin**) and `_observe_perhorizon` matched it (~750 µs/row
→ 25 s). The per-row work is a quantile over a ≤10-score window and a deque
append. **Fix:** vectorized both paths (batched `np.quantile` by window
length, factorize-based group accounting, compact replay loop): apply 6.5 s,
observe 5.6 s at ~900k rows. Equivalence pinned by a predict-vs-predict_batch
test plus the existing 66-test conformal suite; the VN2 baseline never enters
this code (no `conformal:` section in winning.yaml).

> Lesson: `iterrows`/per-group pandas in a hot loop costs two to three orders
> of magnitude. Anything that runs per ledger row at M5 scale (~60M rows per
> full run) must be vectorized by construction.

## Open bottlenecks (measured, not yet fixed)

1. **Ledger ResolveOpen/Commit plumbing: ~130 s/origin at steady state.**
   Once h=28 settles, every origin resolves ~906k newly-due rows through
   `HierarchyActualsSource` against a ~26M-row open ledger, then streams ~1.9M
   parquet rows. There is a measured ~91 s gap inside ResolveOpen between the
   conformal observe log line and the next fit, with no instrumentation in
   between. Likely the same per-row/reindex disease. Fixing this takes the
   full gate from ~2.5 h to ~25 min. **This is the first target of the
   post-mortem optimization session.**
2. **Phase instrumentation gap.** ~85% of origin wall time today had no log
   line attributing it. Reconcile, ResolveOpen internals (actuals lookup vs
   ledger mutation vs parquet append), and Commit need duration logs like fit
   /predict/conformal already have — the unaccounted gap is where every new
   bottleneck hides.
3. **`adapter fit` 5.5 s/origin** rebuilds the truncated panel and
   statsforecast state every origin. Bounded, but ~6 min over a full run.
4. **Eager dense-S Nixtla paths** (see §2 lesson) — blocked behind the memory
   guard, unusable at full M5.

## Why VN2 never caught any of this

VN2 is ~30 series, no hierarchy expansion pressure, no `conformal.partition:
series` cardinality, no 60M-row ledger. Every one of today's failures scales
with node count × horizon × origins. The unit suite pins *semantics* at toy
scale; only a full-scale run pins *feasibility*. That is exactly the argument
for keeping the full M5 run as the recurring gate — it found four real
defects in one day.

## Process notes

- Each fix went through a headless Opus adversarial review before merge
  (#145 ×3 passes, #146, #149, #150); the reviews caught one silent-metric
  correctness bug (partial-aggregate vs complete-actual scoring) that the
  benchmark alone would not have surfaced as a failure.
- py-spy dump against the live driver was the decisive diagnostic every time;
  log-based progress signals (parquet mtime, console-wrapped JSON lines) were
  both misleading. Phase-duration logs in JSON are the reliable signal.

## Candidate directions for the à froid session

Captured while the evidence was hot; none of these are decided. The à froid
session should confirm, amend, or reject each with the measurements above as
the forcing function:

**D1 — Decouple model scope from dispatch granularity.**
Local (per-series) model *semantics* must not imply per-series *scheduling*.
Execution should group local tasks by `(backend, resolved config)` and
dispatch **chunks of series as panel calls**: statsforecast fits every
classical local model per-series inside one vectorized call, so "local" is
preserved exactly while the scheduling unit becomes ~dozens of chunk tasks
instead of 30,490 series tasks. Per-series overrides and local ensembles
(workstreams 3–4) group by distinct config: *k* configs ≈ *k* panel calls per
origin — the ensemble case costs `m_models × 0.5 s`, not
`m_models × 30,490 × dispatch overhead`. Target budget: **a local-model M5
origin costs seconds, ensembles included.**

**D2 — Stage the panel once, not per task.**
Replace 30,490 per-series staged parquet files with one shared panel artifact
(or a single `ray.put` object ref); workers slice their chunk. Staging cost
becomes O(1) per run instead of O(series × models).

**D3 — Chunk-level result return.**
Workers return one frame per chunk; the driver concatenates dozens of frames,
not 30,490 — eliminating the single-threaded `ray.get` deserialization wall.

**D4 — Vectorize the ledger resolve/commit path** (~130 s/origin → seconds;
same per-row disease as the conformal runtime, see "Open bottlenecks").

**D5 — Instrument every phase.** Reconcile, ResolveOpen sub-steps (actuals
lookup / ledger mutation / parquet append), and Commit get duration logs.
~85% of origin wall time was unattributed today; unmeasured phases are where
the next regression hides.

**D6 — Decide the fate of dense-S eager reconciliation paths.** Grouped/
sparse aggregation or an explicit scale ceiling, but not silent gigabytes.

## Follow-ups

- [ ] Run the à froid post-mortem + architecture refactor planning session
      (fresh context, this document as primary input); land D1–D6 verdicts
      there and convert them into issues.
- [ ] `HierarchyActualsSource` member counting on raw dtypes vs stringified
      source (review finding ADV-1, crash-not-corruption, low severity).
