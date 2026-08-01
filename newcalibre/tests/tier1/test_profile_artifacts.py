"""Exercise strict standard-performance artifact validation and publication."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from newcalibre.benchmarking import (
    EnvironmentError,
    LinuxMemoryReader,
    ProfileError,
    publish_profile_artifacts,
    validate_environment,
    validate_profile,
)

_PERFORMANCE_SPEC = Path(__file__).parents[3] / "docs" / "spec" / "30-performance.md"


def _environment() -> dict[str, object]:
    return {
        "schema": "calibre-performance-environment",
        "schema_version": 1,
        "attempt_id": "attempt-a",
        "cpu": {"logical_count": 16, "model": "fixture-cpu"},
        "memory": {"total_bytes": 64 * 1024**3},
        "os": {"machine": "x86_64", "release": "fixture", "system": "Linux"},
        "python": {"implementation": "CPython", "version": "3.12.0"},
        "provenance": {
            "lock_sha256": "a" * 64,
            "numeric_libraries": {"numpy": "2.0.0"},
            "ray_version": "2.0.0",
        },
        "execution": {
            "logical_shards": 16,
            "numeric_threads_per_worker": 1,
            "thread_policy": {
                "BLIS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "RAYON_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
            "workers": 16,
        },
        "cgroup": {"identity": "/gate-c", "version": 2},
        "sampler": {"complete": True, "interval_seconds": 0.1},
    }


def _profile() -> dict[str, object]:
    stages = {
        "Resolve": 1.0,
        "Fit": 2.0,
        "Predict": 1.0,
        "Reconcile": 1.0,
        "Calibrate": 1.0,
        "Order": 1.0,
        "Commit": 1.0,
    }
    return {
        "schema": "calibre-performance-profile",
        "schema_version": 1,
        "attempt_id": "attempt-a",
        "valid": True,
        "invalid_reasons": [],
        "timing": {
            "wall_seconds": 10.0,
            "pre_origin_seconds": 1.0,
            "close_seconds": 1.0,
            "origins": [{"origin": "2026-01-01T00:00:00", "stages": stages}],
            "attributed_seconds": 10.0,
            "reconciliation_percent": 100.0,
        },
        "memory": {
            "sampling_interval_seconds": 0.1,
            "complete": True,
            "processes": [
                {
                    "pid": 10,
                    "role": "driver",
                    "peak_process_resident_bytes": 100,
                },
                {
                    "pid": 11,
                    "role": "ray-control",
                    "peak_process_resident_bytes": 200,
                },
                {
                    "pid": 12,
                    "role": "object-store",
                    "peak_process_resident_bytes": 300,
                },
                {
                    "pid": 13,
                    "role": "worker",
                    "peak_process_resident_bytes": 400,
                },
            ],
            "peak_process_resident_bytes": 1000,
            "peak_job_memory_bytes": 900,
            "cgroup_identity": "/gate-c",
        },
        "dispatch": {
            "logical_shards": 16,
            "numeric_threads_per_worker": 1,
            "origin_count": 1,
            "shard_dispatch_count": 16,
            "workers": 16,
        },
        "scaling": [
            {
                "dispatch_count": 16,
                "peak_job_memory_bytes": 900,
                "series_count": 1000,
                "wall_seconds": 10.0,
                "workers": 16,
            },
            {
                "dispatch_count": 16,
                "peak_job_memory_bytes": 1800,
                "series_count": 10000,
                "wall_seconds": 20.0,
                "workers": 16,
            },
            {
                "dispatch_count": 16,
                "peak_job_memory_bytes": 2700,
                "series_count": 30490,
                "wall_seconds": 30.0,
                "workers": 16,
            },
        ],
        "concurrency": {
            "series_count": 1000,
            "origin_count": 1,
            "logical_shards": 16,
            "phases": ["Fit", "Predict"],
            "timeout_seconds": 60.0,
            "runs": [
                {"concurrency": 1, "dispatch_count": 16, "wall_seconds": 4.0},
                {"concurrency": 16, "dispatch_count": 16, "wall_seconds": 1.0},
            ],
        },
        "budgets": {
            "wall_seconds": {"limit": 900.0, "passed": True},
            "pre_origin_seconds": {"limit": 60.0, "passed": True},
            "peak_job_memory_bytes": {"limit": 32 * 1024**3, "passed": True},
            "reconciliation_percent": {"minimum": 99.0, "passed": True},
        },
    }


def test_validators_recompute_derived_facts_and_reject_tampering() -> None:
    """Reject hand-assembled timing, memory, and budget verdicts."""
    profile = _profile()
    environment = _environment()
    assert validate_profile(profile, environment=environment) == profile
    assert validate_environment(environment) == environment

    for path, changed in (
        (("timing", "attributed_seconds"), 9.0),
        (("memory", "peak_process_resident_bytes"), 999),
        (("budgets", "peak_job_memory_bytes", "passed"), False),
    ):
        tampered = deepcopy(profile)
        target = tampered
        for key in path[:-1]:
            target = target[key]  # type: ignore[index,assignment]
        target[path[-1]] = changed  # type: ignore[index]
        with pytest.raises(ProfileError):
            validate_profile(tampered, environment=environment)


def test_profile_job_memory_verdict_does_not_use_process_sum() -> None:
    """Use the cgroup-v2 peak alone for the 32 GB memory verdict."""
    profile = _profile()
    profile["memory"]["peak_process_resident_bytes"] = 1000  # type: ignore[index]
    profile["memory"]["peak_job_memory_bytes"] = 33 * 1024**3  # type: ignore[index]
    profile["budgets"]["peak_job_memory_bytes"]["passed"] = False  # type: ignore[index]
    assert validate_profile(profile, environment=_environment()) == profile


def test_publish_emits_exact_deterministic_pair_and_refuses_dirty_root(
    tmp_path: Path,
) -> None:
    """Publish exactly two canonical files without exposing a partial pair."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    profile = _profile()
    environment = _environment()
    publish_profile_artifacts(first, profile=profile, environment=environment)
    publish_profile_artifacts(second, profile=profile, environment=environment)

    assert {path.name for path in first.iterdir()} == {"profile.json", "environment.json"}
    assert (first / "profile.json").read_bytes() == (second / "profile.json").read_bytes()
    assert (first / "environment.json").read_bytes() == (second / "environment.json").read_bytes()
    assert json.loads((first / "profile.json").read_text()) == profile

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "run.log").write_text("not a performance artifact")
    with pytest.raises(ProfileError, match="dirty"):
        publish_profile_artifacts(dirty, profile=profile, environment=environment)


def test_invalid_profile_never_emits_passing_artifacts(tmp_path: Path) -> None:
    """Refuse publication when any attempt validity fact is false."""
    profile = _profile()
    profile["valid"] = False
    profile["invalid_reasons"] = ["sampler failed"]
    for budget in profile["budgets"].values():  # type: ignore[union-attr]
        budget["passed"] = False
    assert validate_profile(profile, environment=_environment()) == profile
    with pytest.raises(ProfileError, match="invalid"):
        publish_profile_artifacts(
            tmp_path / "profile",
            profile=profile,
            environment=_environment(),
        )
    assert not (tmp_path / "profile").exists()


def test_atomic_replacement_failure_restores_the_previous_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore both prior artifacts when publication cannot install the new pair."""
    target = tmp_path / "profile"
    publish_profile_artifacts(target, profile=_profile(), environment=_environment())
    previous = {path.name: path.read_bytes() for path in target.iterdir()}
    real_replace = os.replace

    def fail_install(source: Path, destination: Path) -> None:
        if (
            destination == target
            and source.name.startswith(".profile.")
            and ".old." not in source.name
        ):
            raise OSError("fixture install failed")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_install)
    with pytest.raises(OSError, match="install failed"):
        publish_profile_artifacts(target, profile=_profile(), environment=_environment())

    assert {path.name: path.read_bytes() for path in target.iterdir()} == previous


def test_linux_memory_reader_classifies_processes_and_reads_cgroup_peak(
    tmp_path: Path,
) -> None:
    """Read resident bytes by role and charged peak bytes from cgroup v2."""
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("0::/profile-job\n")
    (cgroup / "profile-job").mkdir(parents=True)
    (cgroup / "profile-job" / "memory.peak").write_text("4096\n")
    commands = {
        10: "profile-driver",
        11: "gcs_server",
        12: "raylet",
        13: "default_worker.py",
    }
    for pid, command in commands.items():
        process = proc / str(pid)
        process.mkdir()
        (process / "cmdline").write_bytes(command.encode() + b"\0")
        (process / "status").write_text("Name:\tfixture\nVmRSS:\t2 kB\n")

    sample = LinuxMemoryReader(
        proc_root=proc,
        cgroup_root=cgroup,
        driver_pid=10,
    ).sample()

    assert [(value.pid, value.role, value.resident_bytes) for value in sample.processes] == [
        (10, "driver", 2048),
        (12, "object-store", 2048),
        (11, "ray-control", 2048),
        (13, "worker", 2048),
    ]
    assert sample.peak_job_memory_bytes == 4096
    assert sample.cgroup_identity == "/profile-job"


def test_linux_memory_reader_rejects_unreadable_cgroup_peak(tmp_path: Path) -> None:
    """Invalidate a memory observation when cgroup-v2 counters are malformed."""
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup" / "profile-job"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("0::/profile-job\n")
    cgroup.mkdir(parents=True)
    (cgroup / "memory.peak").write_text("max\n")

    with pytest.raises(EnvironmentError, match="malformed"):
        LinuxMemoryReader(proc_root=proc, cgroup_root=tmp_path / "cgroup").sample()


def test_performance_spec_uses_consistent_resident_memory_contract() -> None:
    """Pin public resident-memory, cgroup-v2, and bounded scaling terminology."""
    source = _PERFORMANCE_SPEC.read_text(encoding="utf-8")

    assert "RSS" not in source
    assert source.count("peak_process_resident_bytes") >= 2
    assert source.count("peak_job_memory_bytes") >= 3
    assert "cgroup-v2 charged job peak" in source
    assert "full-M5 serial path" in source
    assert "exactly `profile.json` and `environment.json`" in source
