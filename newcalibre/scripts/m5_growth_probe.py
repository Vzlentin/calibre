"""Probe per-origin M5 cost growth with counted store, state, and GC facts.

Diagnostic only. This script reduces the M5 population so the Gate C growth
shape can be reproduced off the reference host, so nothing it writes is
acceptance evidence: it publishes no five-file result, no environment
manifest, and no budget verdict.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import fields
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newcalibre.benchmarking import LifecycleCollector, LifecycleRecord  # noqa: E402
from newcalibre.engine import Phase, PhaseEvent, PhaseStatus, RunStoreAudit  # noqa: E402
from newcalibre.protocols.m5 import load_m5_config, run_m5  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "benchmarks" / "m5" / "gate-c.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "m5" / "growth-probe"
_PROFILE_SALT = "calibre-gate-c-profile-v1"
_PHASES = tuple(Phase)
_AUDIT_FIELDS = tuple(field.name for field in fields(RunStoreAudit))
_GC_FIELDS = ("collections", "collected", "uncollectable")

type _Audit = tuple[pd.Timestamp, RunStoreAudit, tuple[int, int]]
type _GcStats = Sequence[dict[str, int]]


def build_probe_config(config_path: Path, directory: Path, *, bottom_count: int) -> Path:
    """Clone the Gate C configuration at a reduced digest-rank population.

    Keeps every other strict value — horizon, origin count, reconciler, conformal
    method, and the 16-worker execution budget — so population is the only knob.
    """
    base = load_m5_config(config_path)
    if base.population.kind != "full":
        raise ValueError("the growth probe clones the committed full-M5 configuration")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("full-M5 configuration must be a mapping")
    try:
        diagnostic_root = directory.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("probe work directory must be inside the project root") from error
    payload = copy.deepcopy(raw)
    payload["protocol"]["population"] = {
        "kind": "digest_rank",
        "bottom_count": bottom_count,
        "salt": _PROFILE_SALT,
    }
    payload["output_dir"] = str(diagnostic_root / f"diagnostics-{bottom_count}")
    path = directory / f"growth-probe-{bottom_count}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    load_m5_config(path)
    return path


def collect_growth(
    *,
    config_path: Path,
    bottom_count: int,
    freeze_gc: bool = False,
    clock: Callable[[], float] = time.perf_counter,
    runner: Callable[..., object] = run_m5,
    freezer: Callable[[], None] = gc.freeze,
    progress: Callable[[str], None] = lambda _message: None,
) -> dict[str, object]:
    """Run one probe population and assemble its per-origin growth record.

    Args:
        config_path: Cloned probe configuration to run.
        bottom_count: Bottom-series population the configuration selects.
        freeze_gc: Move everything alive when the first phase starts out of
            generational bookkeeping, isolating live-set collection cost.
        clock: Monotonic source owned by the harness, never by the engine.
        runner: The ``run_m5`` seam, replaced by fakes under test.
        freezer: The ``gc.freeze`` seam, replaced by fakes under test.
        progress: Receives one human-readable line per committed origin.
    """
    collector = LifecycleCollector(clock=clock)
    audits: list[_Audit] = []
    stats: list[_GcStats] = []
    frozen = False

    def report(event: PhaseEvent) -> None:
        nonlocal frozen
        if freeze_gc and not frozen:
            freezer()
            frozen = True
        collector(event)

    def audit_sink(
        origin: pd.Timestamp,
        audit: RunStoreAudit,
        footprint: tuple[int, int],
    ) -> None:
        audits.append((origin, audit, footprint))
        stats.append(gc.get_stats())
        progress(
            f"origin {len(audits)} {origin.date()} "
            f"state_rows={footprint[0]} state_bytes={footprint[1]} "
            f"pending_rows_examined={audit.pending_rows_examined}"
        )

    baseline = gc.get_stats()
    wall_start = clock()
    result = runner(config_path, reporter=report, audit_sink=audit_sink)
    wall_end = clock()
    records = collector.records
    origins = tuple(dict.fromkeys(record.event.origin for record in records))
    if collector.failure:
        raise ValueError(f"probe lifecycle reporter failed: {collector.failure}")
    if getattr(result, "forecast_origin_count", None) != len(origins):
        raise ValueError("probe run result has the wrong forecast origin count")
    if [origin for origin, _audit, _footprint in audits] != list(origins):
        raise ValueError("probe audit sink did not fire once per origin in origin order")
    per_origin = _origin_records(
        origins,
        records,
        audits=audits,
        stats=stats,
        baseline=baseline,
    )
    totals = {phase.value: 0.0 for phase in _PHASES}
    for record in per_origin:
        for name, duration in cast(dict[str, float], record["stages"]).items():
            totals[name] += duration
    return {
        "schema": "calibre-m5-growth-probe",
        "schema_version": 1,
        "acceptance_evidence": False,
        "bottom_count": bottom_count,
        "node_count": getattr(result, "node_count", 0),
        "origin_count": len(origins),
        "freeze_gc": freeze_gc,
        "wall_seconds": wall_end - wall_start,
        "pre_origin_seconds": records[0].timestamp - wall_start,
        "close_seconds": wall_end - records[-1].timestamp,
        "phase_totals": totals,
        "origins": per_origin,
    }


def _origin_records(
    origins: Sequence[pd.Timestamp],
    records: Sequence[LifecycleRecord],
    *,
    audits: Sequence[_Audit],
    stats: Sequence[_GcStats],
    baseline: _GcStats,
) -> list[dict[str, object]]:
    """Assemble one record per origin from disjoint timing and counter streams."""
    expected = tuple(
        (origin, phase, status)
        for origin in origins
        for phase in _PHASES
        for status in (PhaseStatus.STARTED, PhaseStatus.FINISHED)
    )
    observed = tuple(
        (record.event.origin, record.event.phase, record.event.status) for record in records
    )
    if observed != expected:
        raise ValueError("probe lifecycle events are incomplete, foreign, or out of order")
    payloads: list[dict[str, object]] = []
    previous_audit = RunStoreAudit(**dict.fromkeys(_AUDIT_FIELDS, 0))
    previous_stats = baseline
    offset = 0
    for index, origin in enumerate(origins):
        stages: dict[str, float] = {}
        for phase in _PHASES:
            stages[phase.value] = records[offset + 1].timestamp - records[offset].timestamp
            offset += 2
        _timestamp, audit, footprint = audits[index]
        payloads.append(
            {
                "index": index,
                "origin": origin.isoformat(),
                "stages": stages,
                "origin_seconds": sum(stages.values()),
                "audit_total": {name: getattr(audit, name) for name in _AUDIT_FIELDS},
                "audit_delta": {
                    name: getattr(audit, name) - getattr(previous_audit, name)
                    for name in _AUDIT_FIELDS
                },
                "state_rows": footprint[0],
                "state_bytes": footprint[1],
                **_gc_delta(stats[index], previous_stats),
            }
        )
        previous_audit = audit
        previous_stats = stats[index]
    return payloads


def _gc_delta(current: _GcStats, previous: _GcStats) -> dict[str, list[int]]:
    """Difference one generational collector snapshot against its predecessor."""
    return {
        f"gc_{name}_delta": [
            later[name] - earlier[name] for earlier, later in zip(previous, current, strict=True)
        ]
        for name in _GC_FIELDS
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the growth-probe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bottom-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-gc", action="store_true")
    return parser


def main() -> int:
    """Run one probe population from parsed command-line arguments."""
    args = build_parser().parse_args()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Scoring refuses a pre-existing destination, so every invocation owns a
    # private one; the per-origin JSON, not the diagnostics, is the probe result.
    with tempfile.TemporaryDirectory(prefix="attempt-", dir=DEFAULT_OUTPUT_DIR) as work:
        config = build_probe_config(args.config, Path(work), bottom_count=args.bottom_count)
        payload = collect_growth(
            config_path=config,
            bottom_count=args.bottom_count,
            freeze_gc=args.freeze_gc,
            progress=_stderr_progress,
        )
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def _stderr_progress(message: str) -> None:
    """Stream one probe progress line so a long run can be monitored."""
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    sys.exit(main())
