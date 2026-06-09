---
title: Wave 2 completion — hierarchy support on M5, benchmark extraction
type: feat
status: active
date: 2026-06-10
origin: /goal session — finish Wave 2 (issues #138, #139, #140) + de-benchmark calibre src + M5 data
---

# Goal

Finish Wave 2 (Engine credibility): wire hierarchy support end-to-end using M5
data, and remove VN2/M5-specific code from `calibre/` (calibre is generic;
benchmarks are benchmarks).

## Work items (dependency order)

1. **M5 data** — DONE. Downloaded via `datasetsforecast.m5.M5.download` and
   placed under `data/m5/` (`sales_train_evaluation.csv`, `calendar.csv`, …).
2. **#138 ActualsSource** (lazy hierarchy actual resolution)
   - New `calibre/execution/actuals.py`: `ActualsSource` protocol
     (`resolve(ledger_df, current_origin) -> (updated, newly_resolved)`),
     `FrameActualsSource` (wraps existing `resolve_actuals`), and
     `HierarchyActualsSource` (bottom history + hierarchy facts; resolves
     bottom / `col=value` aggregate / `__total__` rows on demand with the same
     completeness rule as `build_node_history`).
   - Engine: `execute` / `iter_origins` accept `pd.DataFrame | ActualsSource`;
     `_resolve_due` calls `source.resolve(...)`. Delayed-feedback timing stays
     in the engine.
   - Tests: sparse aggregate resolution, unknown node diagnostics, duplicate
     keys, partial due windows, parity with eager `build_node_history`.
3. **#139 native point bottom_up**
   - `calibre/reconciliation/bottom_up.py`: `BottomUpReconciler` consumes
     bottom-only rows per `(model_name, forecast_origin, h)` cross-section and
     synthesizes aggregate node rows = `S @ bottom`; deterministic node order;
     preserves frame columns; rejects quantile columns.
   - Registry: `bottom_up` -> native; `NixtlaReconciler` rejects `bottom_up`
     for point reconciliation (fused `hierarchical_intervals.strategy:
     bottom_up` stays Nixtla).
   - `prepare_run`: point `bottom_up` builds bottom-only tasks + lazy
     `HierarchyActualsSource`; MinT-style strategies keep eager node history.
4. **#140 fold preflight facts into run preparation**
   - One source of truth for node/bottom/aggregate counts and projected
     partition counts (`hierarchy_memory.estimate_hierarchical_expansion`
     becomes the shared facts computation consumed by `prepare_run`); the
     memory guard and the conformal partition limit consume the same facts;
     bottom-only `bottom_up` still estimates ledger partitions over the full
     node set. Diagnostics stay deterministic and at least as specific as #136.
5. **Extraction — calibre is calibre, benchmarks are benchmarks**
   - Move `vn2_adapter.py` + `data_loading.py` -> `benchmarks/vn2/`;
     `m5_adapter.py` + `m5_loading.py` -> `benchmarks/m5/`;
     `evaluation/m5_coverage.py` -> `benchmarks/m5/coverage.py` (generic
     interval-coverage summaries stay in `evaluation/forecast_metrics`).
   - Dataset registry: resolve dotted-path adapters
     (`benchmarks.m5.adapter:M5DatasetAdapter`) so benchmark adapters plug in
     without calibre importing benchmarks (layering test stays green).
   - CLI: drop `score-m5-coverage` (becomes `python -m benchmarks.m5.coverage`);
     `health` stops loading the VN2 fixture; `runner.py` loses the `"vn2"`
     default dataset.
   - Update benchmark YAMLs, Docker smoke configs, tests
     (tests/execution/test_m5_* / test_dataset_registry / cli tests move or
     update).
6. **Verify** — full pytest/ruff/ty; M5 smoke + a real `data/m5` hierarchy run
   (bounded origins) exercising native bottom_up + lazy actuals; PRs close
   #138/#139/#140; CI green; squash-merge.

## Process

One branch + PR per issue, sequential (each builds on the prior); babysit CI;
squash-merge with `closes #N`.
