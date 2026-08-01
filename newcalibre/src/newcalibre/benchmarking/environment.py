"""Capture standard profiling environment and resident-memory facts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast, runtime_checkable

REQUIRED_PROCESS_ROLES: Final = frozenset({"driver", "ray-control", "object-store", "worker"})
_PROCESS_ROLES: Final = REQUIRED_PROCESS_ROLES
_SHA256: Final = re.compile(r"[0-9a-f]{64}")


class EnvironmentError(ValueError):
    """Report unavailable or malformed profiling environment evidence."""


@dataclass(frozen=True, slots=True)
class ProcessResidentSample:
    """Record one process's resident bytes at one sampling instant."""

    pid: int
    role: str
    resident_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.pid, int) or isinstance(self.pid, bool) or self.pid < 1:
            raise EnvironmentError("process sample pid must be positive")
        if self.role not in _PROCESS_ROLES:
            raise EnvironmentError("process sample role is unknown")
        if (
            not isinstance(self.resident_bytes, int)
            or isinstance(self.resident_bytes, bool)
            or self.resident_bytes < 0
        ):
            raise EnvironmentError("process resident bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class MemorySample:
    """Record one complete process inventory and cgroup-v2 memory reading."""

    processes: tuple[ProcessResidentSample, ...]
    peak_job_memory_bytes: int
    cgroup_identity: str
    complete: bool = True
    vanished_pids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        processes = tuple(self.processes)
        if any(not isinstance(value, ProcessResidentSample) for value in processes):
            raise EnvironmentError("memory sample processes must contain process samples")
        identities = tuple((value.pid, value.role) for value in processes)
        if len(set(identities)) != len(identities):
            raise EnvironmentError("memory sample process identities must be unique")
        if (
            not isinstance(self.peak_job_memory_bytes, int)
            or isinstance(self.peak_job_memory_bytes, bool)
            or self.peak_job_memory_bytes < 0
        ):
            raise EnvironmentError("job memory peak must be non-negative")
        if not isinstance(self.cgroup_identity, str) or not self.cgroup_identity.startswith("/"):
            raise EnvironmentError("memory sample requires an absolute cgroup identity")
        if not isinstance(self.complete, bool):
            raise EnvironmentError("memory sample completeness must be boolean")
        vanished = tuple(self.vanished_pids)
        if any(not isinstance(pid, int) or isinstance(pid, bool) or pid < 1 for pid in vanished):
            raise EnvironmentError("vanished process ids must be positive")
        object.__setattr__(self, "processes", processes)
        object.__setattr__(self, "vanished_pids", vanished)


@runtime_checkable
class MemoryReader(Protocol):
    """Read one process inventory and cgroup-v2 memory observation."""

    def sample(self) -> MemorySample:
        """Return one complete memory observation."""
        ...


class MemoryMonitor:
    """Sample resident memory on a bounded background interval."""

    def __init__(self, reader: MemoryReader, *, interval_seconds: float) -> None:
        if not isinstance(reader, MemoryReader):
            raise TypeError("memory monitor reader must expose sample()")
        if (
            not isinstance(interval_seconds, (int, float))
            or isinstance(interval_seconds, bool)
            or float(interval_seconds) <= 0.0
        ):
            raise EnvironmentError("memory monitor interval must be positive")
        self._reader = reader
        self._interval = float(interval_seconds)
        self._samples: list[MemorySample] = []
        self._failure: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracked_pids: frozenset[int] = frozenset()

    @property
    def samples(self) -> tuple[MemorySample, ...]:
        """Return the immutable observations collected so far."""
        return tuple(self._samples)

    @property
    def failure(self) -> str | None:
        """Return the sampler failure that invalidates this attempt, if any."""
        return self._failure

    def start(self) -> None:
        """Take the initial observation and start periodic sampling."""
        if self._thread is not None:
            raise RuntimeError("memory monitor has already started")
        self._take_sample()
        if self._failure is not None:
            return
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="calibre-memory-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop periodic sampling without taking a post-run observation."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._interval):
            self._take_sample()
            if self._failure is not None:
                self._stop.set()

    def _take_sample(self) -> None:
        try:
            sample = self._reader.sample()
            if not isinstance(sample, MemorySample):
                raise TypeError("memory reader returned a non-MemorySample value")
            current = frozenset(process.pid for process in sample.processes)
            vanished = tuple(sorted(self._tracked_pids - current))
            roles = {process.role for process in sample.processes}
            if roles == REQUIRED_PROCESS_ROLES:
                self._tracked_pids = current
            if vanished:
                sample = MemorySample(
                    sample.processes,
                    peak_job_memory_bytes=sample.peak_job_memory_bytes,
                    cgroup_identity=sample.cgroup_identity,
                    complete=sample.complete,
                    vanished_pids=tuple(sorted((*sample.vanished_pids, *vanished))),
                )
            self._samples.append(sample)
        except Exception as error:
            self._failure = str(error)


class LinuxMemoryReader:
    """Read process resident memory and authoritative cgroup-v2 job memory."""

    def __init__(
        self,
        *,
        proc_root: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        driver_pid: int | None = None,
    ) -> None:
        self._proc_root = Path(proc_root)
        self._cgroup_root = Path(cgroup_root)
        self._driver_pid = os.getpid() if driver_pid is None else driver_pid

    def sample(self) -> MemorySample:
        """Read one complete required-role inventory and cgroup peak."""
        identity = _cgroup_identity(self._proc_root / "self" / "cgroup")
        peak = _nonnegative_integer_file(
            self._cgroup_root / identity.removeprefix("/") / "memory.peak",
            name="cgroup memory peak",
        )
        processes: list[ProcessResidentSample] = []
        for entry in self._proc_root.iterdir():
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
                resident = _resident_bytes(entry / "status")
            except (FileNotFoundError, ProcessLookupError):
                continue
            role = _process_role(pid, command=command, driver_pid=self._driver_pid)
            if role is not None:
                processes.append(ProcessResidentSample(pid, role, resident))
        return MemorySample(
            tuple(sorted(processes, key=lambda value: (value.role.encode(), value.pid))),
            peak_job_memory_bytes=peak,
            cgroup_identity=identity,
        )


def capture_environment(
    *,
    attempt_id: str,
    execution: dict[str, object],
    sampling_interval_seconds: float,
    cgroup_identity: str,
    lock_path: Path,
    sampler_complete: bool,
) -> dict[str, object]:
    """Capture one public, deterministic-shape execution environment record."""
    _text(attempt_id, name="attempt identity")
    if not isinstance(lock_path, Path):
        raise EnvironmentError("lock path must be a pathlib.Path")
    try:
        lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    except OSError as error:
        raise EnvironmentError("lock file is unreadable") from error
    if not isinstance(sampling_interval_seconds, (int, float)) or not (
        float(sampling_interval_seconds) > 0.0
    ):
        raise EnvironmentError("sampling interval must be positive")
    if not isinstance(sampler_complete, bool):
        raise EnvironmentError("sampler completeness must be boolean")
    total_memory = _total_memory_bytes()
    return {
        "schema": "calibre-performance-environment",
        "schema_version": 1,
        "attempt_id": attempt_id,
        "cpu": {
            "logical_count": os.cpu_count() or 1,
            "model": platform.processor() or "unknown",
        },
        "memory": {"total_bytes": total_memory},
        "os": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "provenance": {
            "lock_sha256": lock_sha256,
            "numeric_libraries": {
                name: _distribution_version(name) for name in ("numpy", "scipy", "statsforecast")
            },
            "ray_version": _distribution_version("ray"),
        },
        "execution": dict(execution),
        "cgroup": {"identity": cgroup_identity, "version": 2},
        "sampler": {
            "complete": sampler_complete,
            "interval_seconds": float(sampling_interval_seconds),
        },
    }


def validate_environment(value: object) -> dict[str, object]:
    """Validate one exact standard environment artifact."""
    root = _mapping(
        value,
        keys={
            "schema",
            "schema_version",
            "attempt_id",
            "cpu",
            "memory",
            "os",
            "python",
            "provenance",
            "execution",
            "cgroup",
            "sampler",
        },
        name="environment",
    )
    if root["schema"] != "calibre-performance-environment" or root["schema_version"] != 1:
        raise EnvironmentError("environment schema is unsupported")
    _text(root["attempt_id"], name="environment attempt identity")
    cpu = _mapping(root["cpu"], keys={"logical_count", "model"}, name="environment CPU")
    _positive_integer(cpu["logical_count"], name="logical CPU count")
    _text(cpu["model"], name="CPU model")
    memory = _mapping(root["memory"], keys={"total_bytes"}, name="environment memory")
    _positive_integer(memory["total_bytes"], name="total memory")
    operating = _mapping(root["os"], keys={"machine", "release", "system"}, name="environment OS")
    for key in ("machine", "release", "system"):
        _text(operating[key], name=f"OS {key}")
    python = _mapping(root["python"], keys={"implementation", "version"}, name="environment Python")
    _text(python["implementation"], name="Python implementation")
    _text(python["version"], name="Python version")
    provenance = _mapping(
        root["provenance"],
        keys={"lock_sha256", "numeric_libraries", "ray_version"},
        name="environment provenance",
    )
    if (
        not isinstance(provenance["lock_sha256"], str)
        or _SHA256.fullmatch(provenance["lock_sha256"]) is None
    ):
        raise EnvironmentError("environment lock digest must be SHA-256")
    libraries = provenance["numeric_libraries"]
    if not isinstance(libraries, dict) or not libraries:
        raise EnvironmentError("numeric-library provenance must be non-empty")
    for name, version in libraries.items():
        _text(name, name="numeric-library name")
        _text(version, name="numeric-library version")
    _text(provenance["ray_version"], name="Ray version")
    execution = _mapping(
        root["execution"],
        keys={
            "logical_shards",
            "numeric_threads_per_worker",
            "thread_policy",
            "workers",
        },
        name="environment execution",
    )
    if _positive_integer(execution["logical_shards"], name="logical shards") != 16:
        raise EnvironmentError("environment requires 16 logical shards")
    if _positive_integer(execution["workers"], name="workers") != 16:
        raise EnvironmentError("environment requires 16 workers")
    if (
        _positive_integer(
            execution["numeric_threads_per_worker"], name="numeric threads per worker"
        )
        != 1
    ):
        raise EnvironmentError("environment requires one numeric thread per worker")
    thread_policy = execution["thread_policy"]
    required_thread_keys = {
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
    if not isinstance(thread_policy, dict) or set(thread_policy) != required_thread_keys:
        raise EnvironmentError("environment thread policy is incomplete")
    if set(thread_policy.values()) != {"1"}:
        raise EnvironmentError("environment thread policy must cap every pool at one")
    cgroup = _mapping(root["cgroup"], keys={"identity", "version"}, name="environment cgroup")
    if cgroup["version"] != 2:
        raise EnvironmentError("environment requires cgroup v2")
    if not isinstance(cgroup["identity"], str) or not cgroup["identity"].startswith("/"):
        raise EnvironmentError("environment cgroup identity must be absolute")
    sampler = _mapping(
        root["sampler"], keys={"complete", "interval_seconds"}, name="environment sampler"
    )
    if not isinstance(sampler["complete"], bool):
        raise EnvironmentError("sampler completeness must be boolean")
    _positive_number(sampler["interval_seconds"], name="sampling interval")
    return root


def _process_role(pid: int, *, command: str, driver_pid: int) -> str | None:
    if pid == driver_pid:
        return "driver"
    lowered = command.lower()
    if "raylet" in lowered or "plasma_store" in lowered:
        return "object-store"
    if any(value in lowered for value in ("gcs_server", "ray dashboard", "monitor.py")):
        return "ray-control"
    if any(value in lowered for value in ("ray::", "default_worker.py", "ray/workers")):
        return "worker"
    return None


def _resident_bytes(path: Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdecimal():
                break
            return int(fields[1]) * 1024
    raise EnvironmentError("process resident-memory counter is unreadable")


def _cgroup_identity(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EnvironmentError("cgroup identity is unreadable") from error
    matches = [line.split("::", maxsplit=1)[1] for line in lines if line.startswith("0::")]
    if len(matches) != 1 or not matches[0].startswith("/"):
        raise EnvironmentError("cgroup-v2 identity is unavailable")
    return matches[0]


def _nonnegative_integer_file(path: Path, *, name: str) -> int:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise EnvironmentError(f"{name} is unreadable") from error
    if not value.isdecimal():
        raise EnvironmentError(f"{name} is malformed")
    return int(value)


def _total_memory_bytes() -> int:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return int(page_size) * int(pages)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _mapping(value: object, *, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise EnvironmentError(f"{name} must contain exact fields")
    if any(not isinstance(key, str) for key in value):
        raise EnvironmentError(f"{name} fields must be strings")
    return cast(dict[str, object], value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnvironmentError(f"{name} must be non-empty text")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise EnvironmentError(f"{name} must be a positive integer")
    return value


def _positive_number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise EnvironmentError(f"{name} must be positive")
    return float(value)


__all__ = [
    "EnvironmentError",
    "LinuxMemoryReader",
    "MemoryMonitor",
    "MemoryReader",
    "MemorySample",
    "ProcessResidentSample",
    "REQUIRED_PROCESS_ROLES",
    "capture_environment",
    "validate_environment",
]
