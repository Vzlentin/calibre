"""Accept a synthetic standard M5 profile through the production publisher."""

from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from newcalibre.benchmarking import MemorySample, ProcessResidentSample, validate_profile
from newcalibre.engine import Phase, PhaseEvent, PhaseStatus
from newcalibre.protocols.m5 import load_m5_config, score_m5
from tier1.test_m5_scorer import _Reader, _rows
from tier1.test_profile_artifacts import _environment

pytestmark = pytest.mark.tier4

PROJECT_ROOT = Path(__file__).parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "m5_benchmark.py"
CONFIG = PROJECT_ROOT / "benchmarks" / "m5" / "gate-c.yaml"
_THREAD_POLICY = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class _Clock:
    """Repeat one fully reconciled 64-origin attempt clock."""

    def __init__(self) -> None:
        lifecycle = [
            value
            for phase_index in range(448)
            for value in (float(phase_index + 1), float(phase_index + 2))
        ]
        self._values = (0.0, *lifecycle, 450.0)
        self._index = 0

    def __call__(self) -> float:
        value = self._values[self._index % len(self._values)]
        self._index += 1
        return value


class _Monitor:
    """Supply one complete deterministic process/cgroup observation."""

    def __init__(self) -> None:
        processes = (
            ProcessResidentSample(10, "driver", 100),
            ProcessResidentSample(11, "ray-control", 200),
            ProcessResidentSample(12, "object-store", 300),
            ProcessResidentSample(13, "worker", 400),
        )
        self.samples = (
            MemorySample(
                processes,
                peak_job_memory_bytes=900,
                cgroup_identity="/gate-c",
            ),
        )
        self.failure = None

    def start(self) -> None:
        """Start the synthetic sampler."""

    def finish_process_inventory(self) -> None:
        """Finish the synthetic process inventory."""

    def stop(self) -> None:
        """Stop the synthetic sampler."""


def test_synthetic_profile_runs_all_intent_and_emits_exact_valid_pair(
    tmp_path: Path,
) -> None:
    """Recompute timing, memory, scaling, concurrency, and budget facts."""
    namespace = runpy.run_path(str(SCRIPT))
    run_standard_profile = namespace["run_standard_profile"]
    ordinary: list[tuple[int, int]] = []
    diagnostic_destinations: list[Path] = []
    concurrency: list[tuple[int, int]] = []

    def runner(config_path: Path, *, reporter) -> object:
        config = load_m5_config(config_path)
        count = config.population.bottom_count or 30490
        ordinary.append((count, config.execution.workers))
        diagnostic_destinations.append(config.output_dir)
        for origin in pd.date_range("2026-01-01", periods=config.origin_count, freq="D"):
            for phase in Phase:
                reporter(PhaseEvent(phase, origin, PhaseStatus.STARTED))
                reporter(PhaseEvent(phase, origin, PhaseStatus.FINISHED))
        score_m5(config, _Reader(_rows()), output_dir=PROJECT_ROOT / config.output_dir)
        return SimpleNamespace(forecast_origin_count=config.origin_count)

    def concurrency_runner(config_path: Path, value: int) -> tuple[int, float, int, dict[str, str]]:
        config = load_m5_config(config_path)
        concurrency.append((config.population.bottom_count or 0, value))
        return value, 4.0 if value == 1 else 1.0, 16, dict(_THREAD_POLICY)

    lock = tmp_path / "uv.lock"
    lock.write_text("fixture-lock")
    output = tmp_path / "profile"

    def environment_capture(**_kwargs: object) -> dict[str, object]:
        payload = deepcopy(_environment())
        payload["attempt_id"] = "synthetic-attempt"
        return payload

    run_standard_profile(
        config_path=CONFIG,
        output_dir=output,
        attempt_id="synthetic-attempt",
        lock_path=lock,
        candidate_sha="9" * 40,
        azure_image="fixture-image",
        sampling_interval_seconds=0.1,
        clock=_Clock(),
        runner=runner,
        monitor_factory=_Monitor,
        concurrency_runner=concurrency_runner,
        environment_capture=environment_capture,
    )

    assert ordinary == [(1000, 16), (10000, 16), (30490, 16)]
    assert len(set(diagnostic_destinations)) == 3
    assert all(".profile-attempts" in destination.parts for destination in diagnostic_destinations)
    assert concurrency == [(1000, 1), (1000, 16)]
    assert {path.name for path in output.iterdir()} == {
        "coverage-summary.json",
        "coverage-by-node.parquet",
        "report.md",
        "profile.json",
        "environment.json",
    }
    profile = json.loads((output / "profile.json").read_text())
    environment = json.loads((output / "environment.json").read_text())
    assert validate_profile(profile, environment=environment) == profile
    assert profile["valid"] is True
    assert profile["timing"]["reconciliation_percent"] == 100.0
    assert profile["budgets"]["peak_job_memory_bytes"]["passed"] is True
