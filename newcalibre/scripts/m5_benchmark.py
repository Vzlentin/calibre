"""Run the standard M5 profile and publish one five-file Gate C result."""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import queue
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newcalibre.benchmarking import (  # noqa: E402
    LifecycleCollector,
    LifecycleRecord,
    LinuxMemoryReader,
    MemoryMonitor,
    MemorySample,
    ProfileError,
    aggregate_profile,
    capture_environment,
    publish_m5_gate_c_result,
)
from newcalibre.engine import (  # noqa: E402
    RAY_WORKER_THREAD_POLICY,
    Phase,
    PhaseEvent,
    PhaseStatus,
)
from newcalibre.protocols.m5 import load_m5_config, run_m5  # noqa: E402
from newcalibre.protocols.m5.runner import run_m5_fit_predict  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "benchmarks" / "m5" / "gate-c.yaml"
DEFAULT_LOCK = PROJECT_ROOT / "uv.lock"
_POPULATIONS = (1000, 10000, 30490)
_PROFILE_SALT = "calibre-gate-c-profile-v1"


class _Monitor(Protocol):
    @property
    def samples(self) -> tuple[MemorySample, ...]: ...

    @property
    def failure(self) -> str | None: ...

    def start(self) -> None: ...

    def finish_process_inventory(self) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ScalingObservation:
    """Retain one ordinary 16-worker scaling observation."""

    series_count: int
    wall_start: float
    wall_end: float
    origins: tuple[pd.Timestamp, ...]
    lifecycle: tuple[LifecycleRecord, ...]
    memory_samples: tuple[MemorySample, ...]
    sampler_failure: str | None
    reporter_failure: str | None
    dispatch_count: int

    @property
    def wall_seconds(self) -> float:
        """Return end-to-end wall duration for this scaling point."""
        return self.wall_end - self.wall_start

    @property
    def peak_job_memory_bytes(self) -> int:
        """Return the greatest charged cgroup-v2 job-memory peak."""
        if not self.memory_samples:
            raise ProfileError("memory sampling produced no observations")
        return max(sample.peak_job_memory_bytes for sample in self.memory_samples)


def build_scaling_configs(config_path: Path, directory: Path) -> tuple[tuple[int, Path], ...]:
    """Clone all scaling configs with invocation-owned diagnostic destinations."""
    base = load_m5_config(config_path)
    if base.population.kind != "full":
        raise ProfileError("standard profiling requires the committed full-M5 config")
    if (
        base.execution.logical_shards != 16
        or base.execution.workers != 16
        or base.execution.numeric_threads_per_worker != 1
    ):
        raise ProfileError("standard profiling requires the fixed 16-worker budget")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProfileError("full-M5 configuration must be a mapping")
    try:
        diagnostic_root = directory.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ProfileError("profiling work directory must be inside the project root") from error
    generated: list[tuple[int, Path]] = []
    for count in _POPULATIONS:
        payload = copy.deepcopy(raw)
        if count != _POPULATIONS[-1]:
            payload["protocol"]["population"] = {
                "kind": "digest_rank",
                "bottom_count": count,
                "salt": _PROFILE_SALT,
            }
        payload["output_dir"] = str(diagnostic_root / f"diagnostics-{count}")
        path = directory / f"gate-c-{count}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        load_m5_config(path)
        generated.append((count, path))
    return tuple(generated)


def measure_scaling_point(
    *,
    series_count: int,
    config_path: Path,
    origin_count: int,
    clock: Callable[[], float],
    monitor: _Monitor,
    runner: Callable[..., object],
) -> ScalingObservation:
    """Measure one ordinary scaling point around the sole M5 run path."""
    collector = LifecycleCollector(clock=clock)
    finished_commits = 0

    def report(event: PhaseEvent) -> None:
        nonlocal finished_commits
        collector(event)
        if event.phase is Phase.COMMIT and event.status is PhaseStatus.FINISHED:
            finished_commits += 1
            if finished_commits == origin_count:
                monitor.finish_process_inventory()

    wall_start = clock()
    monitor.start()
    try:
        result = runner(config_path, reporter=report)
    finally:
        monitor.stop()
    wall_end = clock()
    result_origins = tuple(dict.fromkeys(record.event.origin for record in collector.records))
    if len(result_origins) != origin_count:
        raise ProfileError("M5 scaling run emitted the wrong lifecycle origin count")
    forecast_origins = getattr(result, "forecast_origin_count", None)
    if forecast_origins != origin_count:
        raise ProfileError("M5 scaling result has the wrong forecast origin count")
    return ScalingObservation(
        series_count=series_count,
        wall_start=wall_start,
        wall_end=wall_end,
        origins=result_origins,
        lifecycle=collector.records,
        memory_samples=monitor.samples,
        sampler_failure=monitor.failure,
        reporter_failure=collector.failure,
        dispatch_count=origin_count * 16,
    )


def run_bounded_concurrency(
    config_path: Path,
    *,
    concurrency: int,
    timeout_seconds: float = 60.0,
    clock: Callable[[], float] = time.perf_counter,
    termination_grace_seconds: float = 1.0,
) -> tuple[int, float, int, dict[str, str]]:
    """Run one Fit/Predict-only comparison in a killable child process."""
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_fit_predict_process,
        args=(config_path, concurrency, results),
    )
    wall_start = clock()
    previous = {name: os.environ.get(name) for name in RAY_WORKER_THREAD_POLICY}
    try:
        os.environ.update(RAY_WORKER_THREAD_POLICY)
        process.start()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(termination_grace_seconds)
        if process.is_alive():
            process.kill()
            process.join(termination_grace_seconds)
        if process.is_alive():
            raise ProfileError("M5 Fit/Predict concurrency child resisted forced cleanup")
        raise ProfileError("M5 Fit/Predict concurrency run exceeded 60 seconds")
    try:
        kind, value = results.get(timeout=1.0)
    except queue.Empty as error:
        raise ProfileError("M5 Fit/Predict concurrency run returned no result") from error
    if kind == "error":
        raise ProfileError(f"M5 Fit/Predict concurrency run failed: {value}")
    wall_seconds = clock() - wall_start
    if wall_seconds > timeout_seconds:
        raise ProfileError("M5 Fit/Predict concurrency run exceeded 60 seconds")
    dispatch_count, thread_policy = value
    return concurrency, wall_seconds, dispatch_count, thread_policy


def run_standard_profile(
    *,
    config_path: Path,
    output_dir: Path,
    attempt_id: str,
    lock_path: Path,
    candidate_sha: str,
    azure_image: str,
    sampling_interval_seconds: float,
    clock: Callable[[], float] = time.perf_counter,
    runner: Callable[..., object] = run_m5,
    monitor_factory: Callable[[], _Monitor] | None = None,
    concurrency_runner: (
        Callable[[Path, int], tuple[int, float, int, dict[str, str]]] | None
    ) = None,
    environment_capture: Callable[..., dict[str, object]] = capture_environment,
) -> None:
    """Run, validate, and atomically publish one complete five-file result."""
    work_parent = PROJECT_ROOT / "results" / "m5" / ".profile-attempts"
    work_parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gate-c-result-", dir=output_dir.parent) as staged:
        diagnostics_copy = Path(staged) / "diagnostics"
        with tempfile.TemporaryDirectory(prefix="attempt-", dir=work_parent) as temporary:
            configs = build_scaling_configs(config_path, Path(temporary))
            observations: list[ScalingObservation] = []
            for series_count, scaling_config in configs:
                config = load_m5_config(scaling_config)
                monitor = (
                    MemoryMonitor(
                        LinuxMemoryReader(),
                        interval_seconds=sampling_interval_seconds,
                    )
                    if monitor_factory is None
                    else monitor_factory()
                )
                observations.append(
                    measure_scaling_point(
                        series_count=series_count,
                        config_path=scaling_config,
                        origin_count=config.origin_count,
                        clock=clock,
                        monitor=monitor,
                        runner=runner,
                    )
                )
            concurrency_config = configs[0][1]
            if concurrency_runner is None:
                concurrency = tuple(
                    run_bounded_concurrency(concurrency_config, concurrency=value, clock=clock)
                    for value in (1, 16)
                )
            else:
                concurrency = tuple(
                    concurrency_runner(concurrency_config, value) for value in (1, 16)
                )
            full_config = load_m5_config(configs[-1][1])
            shutil.copytree(PROJECT_ROOT / full_config.output_dir, diagnostics_copy)
        full = observations[-1]
        scaling = tuple(
            (
                observation.series_count,
                16,
                observation.wall_seconds,
                observation.peak_job_memory_bytes,
                observation.dispatch_count,
            )
            for observation in observations
        )
        failures = tuple(
            reason for observation in observations for reason in _observation_failures(observation)
        )
        if failures:
            raise ProfileError("standard profile measurement failed: " + "; ".join(failures))
        profile = aggregate_profile(
            attempt_id=attempt_id,
            expected_origins=full.origins,
            wall_start=full.wall_start,
            wall_end=full.wall_end,
            lifecycle=full.lifecycle,
            memory_samples=full.memory_samples,
            sampling_interval_seconds=sampling_interval_seconds,
            dispatch={
                "logical_shards": 16,
                "numeric_threads_per_worker": 1,
                "origin_count": len(full.origins),
                "shard_dispatch_count": full.dispatch_count,
                "workers": 16,
            },
            scaling=scaling,
            concurrency=concurrency,
            reporter_failure=full.reporter_failure,
            sampler_failure=full.sampler_failure,
        )
        cgroup_identity = str(profile["memory"]["cgroup_identity"])  # type: ignore[index]
        config = load_m5_config(config_path)
        environment = environment_capture(
            attempt_id=attempt_id,
            execution={
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "gpu_count": 0,
                "logical_shards": 16,
                "numeric_threads_per_worker": 1,
                "thread_policy": dict(RAY_WORKER_THREAD_POLICY),
                "workers": 16,
            },
            sampling_interval_seconds=sampling_interval_seconds,
            cgroup_identity=cgroup_identity,
            lock_path=lock_path,
            config_path=config_path,
            inventory_path=PROJECT_ROOT / config.inventory_path,
            candidate_sha=candidate_sha,
            reference={
                "provider": "azure",
                "image": azure_image,
                "instance_class": "Standard_F16as_v6",
                "region": "westeurope",
            },
            sampler_complete=bool(profile["memory"]["complete"]),  # type: ignore[index]
        )
        publish_m5_gate_c_result(
            output_dir,
            diagnostics=diagnostics_copy,
            profile=profile,
            environment=environment,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the standard M5 profiling CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--azure-image", required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--sampling-interval", type=float, default=0.1)
    return parser


def main() -> int:
    """Run the standard profile from parsed command-line arguments."""
    args = build_parser().parse_args()
    try:
        run_standard_profile(
            config_path=args.config,
            output_dir=args.output,
            attempt_id=args.attempt_id,
            lock_path=args.lock,
            candidate_sha=args.candidate_sha,
            azure_image=args.azure_image,
            sampling_interval_seconds=args.sampling_interval,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


def _fit_predict_process(config_path: Path, concurrency: int, results: Any) -> None:
    try:
        result = run_m5_fit_predict(config_path, concurrency=concurrency)
    except Exception as error:
        results.put(("error", str(error)))
        return
    results.put(("result", (result.dispatch_count, dict(result.thread_policy))))


def _observation_failures(observation: ScalingObservation) -> tuple[str, ...]:
    """Return every invalidating fact retained for one scaling point."""
    reasons: list[str] = []
    if observation.reporter_failure:
        reasons.append(
            f"{observation.series_count}: lifecycle reporter failed: {observation.reporter_failure}"
        )
    if observation.sampler_failure:
        reasons.append(
            f"{observation.series_count}: memory sampler failed: {observation.sampler_failure}"
        )
    expected_lifecycle = tuple(
        (origin, phase, status)
        for origin in observation.origins
        for phase in Phase
        for status in (PhaseStatus.STARTED, PhaseStatus.FINISHED)
    )
    actual_lifecycle = tuple(
        (record.event.origin, record.event.phase, record.event.status)
        for record in observation.lifecycle
    )
    if actual_lifecycle != expected_lifecycle:
        reasons.append(f"{observation.series_count}: lifecycle evidence was incomplete")
    if not observation.memory_samples:
        reasons.append(f"{observation.series_count}: memory sampling produced no observations")
        return tuple(reasons)
    identities = {sample.cgroup_identity for sample in observation.memory_samples}
    if len(identities) != 1:
        reasons.append(f"{observation.series_count}: cgroup identity changed")
    if any(not sample.complete for sample in observation.memory_samples):
        reasons.append(f"{observation.series_count}: memory sampling was incomplete")
    if any(sample.vanished_pids for sample in observation.memory_samples):
        reasons.append(f"{observation.series_count}: required process vanished")
    roles = {process.role for sample in observation.memory_samples for process in sample.processes}
    required_roles = {"driver", "ray-control", "object-store", "worker"}
    if roles != required_roles:
        reasons.append(f"{observation.series_count}: required process roles were incomplete")
    return tuple(reasons)


if __name__ == "__main__":
    sys.exit(main())
