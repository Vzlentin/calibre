"""Exercise diagnostic growth-probe config cloning and per-origin assembly."""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from newcalibre.engine import Phase, PhaseEvent, PhaseStatus, RunStoreAudit
from newcalibre.protocols.m5 import load_m5_config

_PROJECT_ROOT = Path(__file__).parents[2]
_GATE_C = _PROJECT_ROOT / "benchmarks" / "m5" / "gate-c.yaml"
_SCRIPT = _PROJECT_ROOT / "scripts" / "m5_growth_probe.py"
_AUDIT_FIELDS = (
    "origin_opens",
    "actuals_opens",
    "source_rows_examined",
    "target_buckets_examined",
    "pending_rows_examined",
    "history_rows_examined",
    "commits",
    "history_rows_appended",
    "forecast_rows_appended",
    "resolution_rows_applied",
    "staged_rows_validated",
    "due_targets_indexed",
    "checkpoint_indexes_decoded",
)


class Clock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _namespace(root: Path) -> dict[str, Any]:
    namespace = runpy.run_path(str(_SCRIPT))
    for value in namespace.values():
        if callable(value) and getattr(value, "__module__", None) == "<run_path>":
            value.__globals__["PROJECT_ROOT"] = root
    return namespace


def _audit(**counts: int) -> RunStoreAudit:
    return RunStoreAudit(**{**dict.fromkeys(_AUDIT_FIELDS, 0), **counts})


def _fake_runner(
    origins: tuple[pd.Timestamp, ...],
    *,
    audits: dict[pd.Timestamp, tuple[RunStoreAudit, tuple[int, int]]],
) -> Callable[..., object]:
    class Result:
        forecast_origin_count = len(origins)

    def run(
        _config_path: Path,
        *,
        reporter: Callable[[PhaseEvent], None],
        audit_sink: Callable[[pd.Timestamp, RunStoreAudit, tuple[int, int]], None],
    ) -> Result:
        for origin in origins:
            for phase in Phase:
                reporter(PhaseEvent(phase, origin, PhaseStatus.STARTED))
                reporter(PhaseEvent(phase, origin, PhaseStatus.FINISHED))
                if phase is Phase.COMMIT:
                    audit_sink(origin, *audits[origin])
        return Result()

    return run


def test_probe_config_reduces_only_the_population(tmp_path: Path) -> None:
    """Clone Gate C at a digest-rank population without touching any other value."""
    directory = tmp_path / "results" / "m5" / "growth-probe"
    directory.mkdir(parents=True)
    build_probe_config = _namespace(tmp_path)["build_probe_config"]

    path = build_probe_config(_GATE_C, directory, bottom_count=100)

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    committed = yaml.safe_load(_GATE_C.read_text(encoding="utf-8"))
    config = load_m5_config(path)
    baseline = load_m5_config(_GATE_C)

    assert payload["protocol"]["population"] == {
        "kind": "digest_rank",
        "bottom_count": 100,
        "salt": "calibre-gate-c-profile-v1",
    }
    assert payload["output_dir"] == "results/m5/growth-probe/diagnostics-100"
    assert (
        payload
        | {
            "protocol": committed["protocol"],
            "output_dir": committed["output_dir"],
        }
        == committed
    )
    assert (
        payload["protocol"] | {"population": committed["protocol"]["population"]}
        == (committed["protocol"])
    )
    assert config.execution == baseline.execution
    assert config.reconciliation_strategy == baseline.reconciliation_strategy
    assert config.conformal_partition == baseline.conformal_partition
    assert config.origin_count == baseline.origin_count == 64


def test_probe_config_refuses_an_already_reduced_configuration(tmp_path: Path) -> None:
    """Refuse to clone anything but the committed full-M5 configuration."""
    directory = tmp_path / "results" / "m5" / "growth-probe"
    directory.mkdir(parents=True)
    namespace = _namespace(tmp_path)
    reduced = namespace["build_probe_config"](_GATE_C, directory, bottom_count=100)

    with pytest.raises(ValueError, match="full-M5"):
        namespace["build_probe_config"](reduced, directory, bottom_count=50)


def test_probe_config_refuses_a_destination_outside_the_project(tmp_path: Path) -> None:
    """Keep probe output inside the project root the config paths resolve against."""
    outside = tmp_path / "outside"
    outside.mkdir()
    build_probe_config = _namespace(tmp_path / "root")["build_probe_config"]

    with pytest.raises(ValueError, match="project root"):
        build_probe_config(_GATE_C, outside, bottom_count=100)


def test_per_origin_records_pair_phase_durations_with_counter_deltas(tmp_path: Path) -> None:
    """Assemble stage durations, audit deltas, footprints, and GC deltas per origin."""
    origins = (pd.Timestamp("2016-03-20"), pd.Timestamp("2016-03-21"))
    audits = {
        origins[0]: (_audit(commits=1, pending_rows_examined=10), (4, 40)),
        origins[1]: (_audit(commits=2, pending_rows_examined=35), (7, 91)),
    }
    collect_growth = _namespace(tmp_path)["collect_growth"]
    clock = Clock(*(float(value) for value in range(0, 60)))

    payload = collect_growth(
        config_path=tmp_path / "growth-probe-100.yaml",
        bottom_count=100,
        clock=clock,
        runner=_fake_runner(origins, audits=audits),
    )

    first, second = payload["origins"]
    assert payload["bottom_count"] == 100
    assert payload["origin_count"] == 2
    assert payload["freeze_gc"] is False
    assert payload["acceptance_evidence"] is False
    assert payload["pre_origin_seconds"] == 1.0
    assert payload["phase_totals"] == {phase.value: 2.0 for phase in Phase}
    assert first["origin"] == "2016-03-20T00:00:00"
    assert first["stages"] == {phase.value: 1.0 for phase in Phase}
    assert first["origin_seconds"] == 7.0
    assert first["audit_total"]["pending_rows_examined"] == 10
    assert first["audit_delta"]["pending_rows_examined"] == 10
    assert (first["state_rows"], first["state_bytes"]) == (4, 40)
    assert second["index"] == 1
    assert second["audit_total"]["commits"] == 2
    assert second["audit_delta"]["commits"] == 1
    assert second["audit_delta"]["pending_rows_examined"] == 25
    assert (second["state_rows"], second["state_bytes"]) == (7, 91)
    assert all(
        len(record[f"gc_{name}_delta"]) == 3
        and all(isinstance(value, int) and value >= 0 for value in record[f"gc_{name}_delta"])
        for record in (first, second)
        for name in ("collections", "collected", "uncollectable")
    )


def test_frozen_probe_freezes_once_before_the_first_phase(tmp_path: Path) -> None:
    """Move the pre-loaded panel out of generational bookkeeping exactly once."""
    origins = (pd.Timestamp("2016-03-20"), pd.Timestamp("2016-03-21"))
    audits = {origin: (_audit(commits=index + 1), (1, 1)) for index, origin in enumerate(origins)}
    collect_growth = _namespace(tmp_path)["collect_growth"]
    freezes: list[int] = []

    payload = collect_growth(
        config_path=tmp_path / "growth-probe-100.yaml",
        bottom_count=100,
        freeze_gc=True,
        clock=Clock(*(float(value) for value in range(0, 60))),
        runner=_fake_runner(origins, audits=audits),
        freezer=lambda: freezes.append(1),
    )

    assert freezes == [1]
    assert payload["freeze_gc"] is True


def test_probe_refuses_a_run_whose_sink_and_lifecycle_disagree(tmp_path: Path) -> None:
    """Refuse to publish counters that cannot be paired with their origin."""
    origins = (pd.Timestamp("2016-03-20"),)
    audits = {origins[0]: (_audit(commits=1), (1, 1))}
    namespace = _namespace(tmp_path)
    silent = namespace["collect_growth"]

    def runner(
        _config_path: Path,
        *,
        reporter: Callable[[PhaseEvent], None],
        audit_sink: Callable[[pd.Timestamp, RunStoreAudit, tuple[int, int]], None],
    ) -> object:
        _fake_runner(origins, audits=audits)(_config_path, reporter=reporter, audit_sink=audit_sink)
        audit_sink(origins[0], *audits[origins[0]])

        class Result:
            forecast_origin_count = 1

        return Result()

    with pytest.raises(ValueError, match="once per origin"):
        silent(
            config_path=tmp_path / "growth-probe-100.yaml",
            bottom_count=100,
            clock=Clock(*(float(value) for value in range(0, 60))),
            runner=runner,
        )
