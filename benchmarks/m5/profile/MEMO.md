# Full-M5 profile — where the wall clock goes (#289)

Phase attribution of a complete full-M5 backtest
(`benchmarks/m5/config/full-wls-struct.yaml`), from the engine's own JSON
telemetry plus a standalone re-measurement of the untimed pre-origin stages.
Zero engine code was changed: the origin loop was already fully instrumented
(`BackendEngine._phase` per-origin `duration_ms` logs and tracing spans), so
the optional spans PR was not needed.

## Run provenance

- Config: `benchmarks/m5/config/full-wls-struct.yaml` (SeasonalNaive global
  scope, wls_struct reconciliation, MSCP per-horizon, 64 daily origins
  2016-03-20 → 2016-05-22, streaming ledger). Tree at `main@686a1b2`.
- Command: `uv run calibre run --config benchmarks/m5/config/full-wls-struct.yaml`
  with JSON logs captured to `run-log.jsonl` (committed here verbatim).
- Timing anchors: a first attempt launched 09:03:09 local died unrecorded (its
  logs were truncated by the relaunch); the measured run is the 10:40:24 local
  relaunch (evidence: `stdout.log` LastWriteTime 10:40:24 at 0 bytes — a
  truncating re-open) through the final log line 11:55:03 local. Single run,
  machine otherwise lightly loaded; no variance estimate.
- Outputs: `results/m5/full-mscp-wls-struct/forecast-ledger.parquet` (1.01 GB)
  and `forecast-ledger.resolved.parquet` (1.59 GB).

## Machine

Dell Precision 3480 laptop; Intel Core i7-1370P (14C/20T), 31.7 GB RAM,
Windows 11 Enterprise 10.0.22631, Python 3.12.13. Single process,
`execution.backend: auto` → local (one global task group, below the Ray
threshold); most cores idle throughout. The OOM preflight enforce step is
inert off Linux; the run fits this machine's memory.

## Wall-clock attribution — 74.7 min total, 99.8% accounted

| Segment | Seconds | % of wall |
|---|---:|---:|
| Pre-origin (launch → first span) | 238.8 | 5.3% |
| Origin loop (64 origins) | 4,073.7 | 90.9% |
| Ledger close (final resolve + artifact write) | 159.9 | 3.6% |
| Loop overhead outside phases | ~10 | 0.2% |
| **Total wall (10:40:24 → 11:55:03 local)** | **4,479.7** | **100%** |

### Origin loop by phase (share of total wall)

| Phase | Total s | % wall | Mean s/origin | Max s |
|---|---:|---:|---:|---:|
| ResolveOpen | 1,213.2 | 27.1% | 18.96 | 48.6 |
| Predict | 1,363.9 | 30.4% | 21.31 | 81.8 |
| Reconcile | 163.0 | 3.6% | 2.55 | 10.0 |
| Calibrate | 253.9 | 5.7% | 3.97 | 7.4 |
| Order | 0.0 | 0.0% | 0.00 | 0.0 |
| Commit | 1,075.9 | 24.0% | 16.81 | 42.6 |

Within Predict: adapter fit 833.1 s (18.6% of wall), adapter predict 29.2 s
(0.7%); the remaining ~502 s (11.2%) is panel staging/frame assembly around
the adapter. Cross-phase span totals (spans nest inside phases; not additive
with the table above): `actuals_lookup` 1,686.6 s — **37.7% of wall** in 128
calls (2/origin, in ResolveOpen and Commit); ledger maintenance
(`ledger_append` 215.9 + `ledger_update_resolved` 105.4 +
`ledger_resolution_frame` 22.6 + `ledger_close` 159.9) 503.8 s — 11.2%;
conformal apply+observe 476.3 s — 10.6%.

### Pre-origin decomposition (standalone re-measurement)

The pre-origin stages are untimed in the engine, so they were re-measured by a
standalone script replicating the exact pre-loop call path (config load →
`M5DatasetAdapter.load` unrolled → `validate_dataset_bundle` →
`build_hierarchy_index` → OOM preflight → `build_node_history` →
`build_tasks`) on the same machine, config, and commit. Standalone total
249.4 s vs the 238.8 s in-run gap — a 4% match, so the decomposition is
representative (`preorigin-timings.json`):

| Stage | Seconds |
|---|---:|
| imports + config load | 0.8 |
| CSV reads (sales + calendar) | 1.8 |
| `melt_m5_sales` (wide → 59.2M-row long) | 11.8 |
| `validate_dataset_bundle` | 7.5 |
| `build_hierarchy_index` | 0.0 |
| OOM preflight (`estimate_hierarchical_expansion`) | 94.0 |
| `build_node_history` (node panel: 65.1M rows, 33,563 nodes) | 112.8 |
| `build_tasks` (1 global task) | 20.5 |

## Implied bottleneck (input to the spec's performance chapter)

The engine spends its time on **data movement, not modeling**:

1. **Per-origin re-scanning of actuals and ledger state** is the single
   largest cost center: `actuals_lookup` alone is 37.7% of wall, and with
   ledger maintenance the resolve/commit machinery accounts for ~49%. Each
   origin re-derives lookups over the full 65.1M-row node panel and a
   growing ledger instead of consulting an incremental index.
2. **Panel staging around the adapter** (~11%) rebuilds per-origin frames
   from the same immutable history each iteration.
3. **Model fit is 18.6% even for SeasonalNaive** — the global panel is refit
   from scratch at every origin; an incremental/persistent-model design
   removes most of this for cheap models.
4. **The OOM preflight costs 94 s** (2% of wall) of group-by estimation —
   guard overhead worth a cheaper cardinality estimate.
5. Reconciliation (3.6%) and calibration (5.7%) — the pipeline's mathematical
   core — are nearly free at this scale. Order is free.

The historical "~4 hours" figure did not reproduce here: this laptop runs the
full config in 74.7 min. The config's own comments attribute ~3 min/origin to
pre-global-scope task dispatch (64 × 3 min ≈ 3.2 h) — the global-scope change
already removed that; what remains is the data-movement profile above.

**R13 sanity check (15-minute bar):** needs a further ~5× on this class of
machine. No micro-optimization gets there; the levers are architectural —
incremental actuals/ledger indexing (~49% of wall), staging reuse across
origins (~11%), incremental fitting (~19%), vectorized conformal state
updates (~11%) — plus the parallelism headroom of 14 mostly-idle cores. That
combination makes the bar plausible on a workstation, which is exactly what
the spec's performance chapter should design for.

## Files

- `run-log.jsonl` — the run's JSON telemetry, verbatim (source of every
  origin-loop number).
- `phase-aggregates.json` — machine-readable aggregation of the log (phase
  and span totals, origin stats, wall-clock anchors).
- `preorigin-timings.json` — standalone pre-origin stage timings + run facts
  (row/node/task counts).
