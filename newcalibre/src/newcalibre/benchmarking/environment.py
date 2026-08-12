"""Capture standard profiling environment and resident-memory facts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast, runtime_checkable

from newcalibre.engine.ray import RAY_WORKER_THREAD_POLICY

REQUIRED_PROCESS_ROLES: Final = frozenset({"driver", "ray-control", "object-store", "worker"})
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_COMMIT_SHA: Final = re.compile(r"[0-9a-f]{40}")


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
        if self.role not in REQUIRED_PROCESS_ROLES:
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

    def reset_peak(self) -> None:
        """Reset the authoritative cgroup peak for one scaling point."""

    def sample(self) -> MemorySample:
        """Return one complete memory observation."""
        ...

    def sample_job_peak(self) -> MemorySample:
        """Return a cgroup-only observation after process inventory closes."""
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
        self._inventory_finished = False

    @property
    def samples(self) -> tuple[MemorySample, ...]:
        """Return the immutable observations collected so far."""
        return tuple(self._samples)

    @property
    def failure(self) -> str | None:
        """Return the sampler failure that invalidates this attempt, if any."""
        return self._failure

    def start(self) -> None:
        """Reset the cgroup peak, observe it, and start periodic sampling."""
        if self._thread is not None:
            raise RuntimeError("memory monitor has already started")
        try:
            self._reader.reset_peak()
        except Exception as error:
            self._failure = str(error)
            return
        self._take_sample()
        if self._failure is not None:
            return
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="calibre-memory-monitor",
            daemon=True,
        )
        self._thread.start()

    def finish_process_inventory(self) -> None:
        """Freeze required-role tracking after the final engine commit."""
        if not self._tracked_pids and self._failure is None:
            self._failure = "required process inventory was never complete"
        self._inventory_finished = True

    def stop(self) -> None:
        """Stop periodic sampling and take one synchronous final observation."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._failure is None:
            self._take_job_peak()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._interval):
            if self._inventory_finished:
                self._take_job_peak()
            else:
                self._take_sample()
            if self._failure is not None:
                self._stop.set()

    def _take_sample(self) -> None:
        try:
            sample = self._reader.sample()
            if not isinstance(sample, MemorySample):
                raise TypeError("memory reader returned a non-MemorySample value")
            if not self._inventory_finished:
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

    def _take_job_peak(self) -> None:
        try:
            sample = self._reader.sample_job_peak()
            if not isinstance(sample, MemorySample) or sample.processes:
                raise TypeError("final memory reading must be a cgroup-only MemorySample")
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
        self._reset_identity: str | None = None

    def reset_peak(self) -> None:
        """Reset and synchronously verify the cgroup-v2 peak counter."""
        identity = _cgroup_identity(self._proc_root / "self" / "cgroup")
        path = self._cgroup_root / identity.removeprefix("/") / "memory.peak"
        try:
            path.write_text("0", encoding="ascii")
        except OSError as error:
            raise EnvironmentError("cgroup memory peak cannot be reset") from error
        _nonnegative_integer_file(path, name="reset cgroup memory peak")
        self._reset_identity = identity

    def sample(self) -> MemorySample:
        """Read one complete required-role inventory and cgroup peak."""
        identity = _cgroup_identity(self._proc_root / "self" / "cgroup")
        if self._reset_identity != identity:
            raise EnvironmentError("cgroup memory peak was not reset for this scaling point")
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
                process_identity = _cgroup_identity(entry / "cgroup")
                if not _is_same_job_cgroup(process_identity, identity):
                    continue
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
            except (FileNotFoundError, ProcessLookupError):
                continue
            role = _process_role(pid, command=command, driver_pid=self._driver_pid)
            if role is None:
                continue
            try:
                resident = _resident_bytes(entry / "status")
            except (FileNotFoundError, ProcessLookupError):
                continue
            processes.append(ProcessResidentSample(pid, role, resident))
        return MemorySample(
            tuple(sorted(processes, key=lambda value: (value.role.encode(), value.pid))),
            peak_job_memory_bytes=peak,
            cgroup_identity=identity,
        )

    def sample_job_peak(self) -> MemorySample:
        """Read the authoritative cgroup peak without inventorying processes."""
        identity = _cgroup_identity(self._proc_root / "self" / "cgroup")
        if self._reset_identity != identity:
            raise EnvironmentError("cgroup memory peak was not reset for this scaling point")
        peak = _nonnegative_integer_file(
            self._cgroup_root / identity.removeprefix("/") / "memory.peak",
            name="cgroup memory peak",
        )
        return MemorySample((), peak_job_memory_bytes=peak, cgroup_identity=identity)


def capture_environment(
    *,
    attempt_id: str,
    execution: dict[str, object],
    sampling_interval_seconds: float,
    cgroup_identity: str,
    lock_path: Path,
    config_path: Path,
    inventory_path: Path,
    candidate_sha: str,
    reference: dict[str, str],
    sampler_complete: bool,
) -> dict[str, object]:
    """Capture one public, deterministic-shape execution environment record."""
    _text(attempt_id, name="attempt identity")
    candidate = _commit(candidate_sha, name="candidate SHA")
    reference_payload = _mapping(
        reference,
        keys={"provider", "image", "instance_class", "region"},
        name="reference environment",
    )
    lock_sha256 = _file_digest(lock_path, name="lock")
    config_sha256 = _file_digest(config_path, name="configuration")
    inventory_sha256 = _file_digest(inventory_path, name="input inventory")
    if not isinstance(sampling_interval_seconds, (int, float)) or not (
        float(sampling_interval_seconds) > 0.0
    ):
        raise EnvironmentError("sampling interval must be positive")
    if not isinstance(sampler_complete, bool):
        raise EnvironmentError("sampler completeness must be boolean")
    total_memory = _total_memory_bytes()
    operating = _operating_system()
    return {
        "schema": "calibre-performance-environment",
        "schema_version": 2,
        "attempt_id": attempt_id,
        "reference": reference_payload,
        "candidate_sha": candidate,
        "cpu": {
            "logical_count": os.cpu_count() or 1,
            "physical_count": _physical_core_count(),
            "model": platform.processor() or "unknown",
        },
        "memory": {"total_bytes": total_memory, "usable_bytes": total_memory},
        "os": operating,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "provenance": {
            "blas": _blas_provenance(),
            "lock_sha256": lock_sha256,
            "numeric_libraries": {
                name: _distribution_version(name) for name in ("numpy", "scipy", "statsforecast")
            },
            "ray_version": _distribution_version("ray"),
            "uv_version": _uv_version(),
        },
        "input": {
            "config_sha256": config_sha256,
            "dataset": "m5",
            "inventory_sha256": inventory_sha256,
            "inventory_verified": True,
        },
        "execution": dict(execution),
        "cgroup": {"identity": cgroup_identity, "reset_verified": True, "version": 2},
        "sampler": {
            "complete": sampler_complete,
            "interval_seconds": float(sampling_interval_seconds),
        },
        "output": {"persistent": True, "placement": "persistent-root"},
    }


def validate_environment(value: object) -> dict[str, object]:
    """Validate one exact standard environment artifact."""
    root = _mapping(
        value,
        keys={
            "schema",
            "schema_version",
            "attempt_id",
            "reference",
            "candidate_sha",
            "cpu",
            "memory",
            "os",
            "python",
            "provenance",
            "input",
            "execution",
            "cgroup",
            "sampler",
            "output",
        },
        name="environment",
    )
    if root["schema"] != "calibre-performance-environment" or root["schema_version"] != 2:
        raise EnvironmentError("environment schema is unsupported")
    _text(root["attempt_id"], name="environment attempt identity")
    reference = _mapping(
        root["reference"],
        keys={"provider", "image", "instance_class", "region"},
        name="environment reference",
    )
    if reference["provider"] != "azure":
        raise EnvironmentError("environment provider must equal azure")
    _text(reference["image"], name="resolved Azure image")
    if reference["instance_class"] != "Standard_F16as_v6":
        raise EnvironmentError("environment instance class is not the pinned role-2 class")
    if reference["region"] != "westeurope":
        raise EnvironmentError("environment region is not the pinned role-2 region")
    _commit(root["candidate_sha"], name="environment candidate SHA")
    cpu = _mapping(
        root["cpu"],
        keys={"logical_count", "physical_count", "model"},
        name="environment CPU",
    )
    _positive_integer(cpu["logical_count"], name="logical CPU count")
    if _positive_integer(cpu["physical_count"], name="physical CPU count") < 16:
        raise EnvironmentError("environment requires at least 16 physical cores")
    _text(cpu["model"], name="CPU model")
    memory = _mapping(
        root["memory"], keys={"total_bytes", "usable_bytes"}, name="environment memory"
    )
    _positive_integer(memory["total_bytes"], name="total memory")
    if _positive_integer(memory["usable_bytes"], name="usable memory") < 64 * 1000**3:
        raise EnvironmentError("environment requires at least 64 GB-class usable memory")
    operating = _mapping(
        root["os"],
        keys={"id", "machine", "pretty_name", "release", "system", "version_id"},
        name="environment OS",
    )
    for key in ("id", "machine", "pretty_name", "release", "system", "version_id"):
        _text(operating[key], name=f"OS {key}")
    if (
        operating["id"] != "ubuntu"
        or operating["version_id"] != "24.04"
        or operating["machine"] != "x86_64"
        or operating["system"] != "Linux"
    ):
        raise EnvironmentError("environment requires x86_64 Ubuntu 24.04")
    python = _mapping(root["python"], keys={"implementation", "version"}, name="environment Python")
    _text(python["implementation"], name="Python implementation")
    _text(python["version"], name="Python version")
    provenance = _mapping(
        root["provenance"],
        keys={"blas", "lock_sha256", "numeric_libraries", "ray_version", "uv_version"},
        name="environment provenance",
    )
    if (
        not isinstance(provenance["lock_sha256"], str)
        or _SHA256.fullmatch(provenance["lock_sha256"]) is None
    ):
        raise EnvironmentError("environment lock digest must be SHA-256")
    blas = _mapping(
        provenance["blas"], keys={"configuration", "name", "version"}, name="BLAS provenance"
    )
    for key in ("configuration", "name", "version"):
        _text(blas[key], name=f"BLAS {key}")
    libraries = provenance["numeric_libraries"]
    if not isinstance(libraries, dict) or not libraries:
        raise EnvironmentError("numeric-library provenance must be non-empty")
    for name, version in libraries.items():
        _text(name, name="numeric-library name")
        _text(version, name="numeric-library version")
    _text(provenance["ray_version"], name="Ray version")
    _text(provenance["uv_version"], name="uv version")
    inputs = _mapping(
        root["input"],
        keys={"config_sha256", "dataset", "inventory_sha256", "inventory_verified"},
        name="environment input",
    )
    for key in ("config_sha256", "inventory_sha256"):
        _digest(inputs[key], name=f"environment {key}")
    if inputs["dataset"] != "m5" or inputs["inventory_verified"] is not True:
        raise EnvironmentError("environment requires verified M5 inputs")
    execution = _mapping(
        root["execution"],
        keys={
            "logical_shards",
            "cuda_visible_devices",
            "gpu_count",
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
    if not isinstance(thread_policy, dict) or set(thread_policy) != set(RAY_WORKER_THREAD_POLICY):
        raise EnvironmentError("environment thread policy is incomplete")
    if set(thread_policy.values()) != {"1"}:
        raise EnvironmentError("environment thread policy must cap every pool at one")
    if execution["gpu_count"] != 0 or execution["cuda_visible_devices"] != "":
        raise EnvironmentError("environment requires no GPU visibility")
    cgroup = _mapping(
        root["cgroup"], keys={"identity", "reset_verified", "version"}, name="environment cgroup"
    )
    if cgroup["version"] != 2:
        raise EnvironmentError("environment requires cgroup v2")
    if not isinstance(cgroup["identity"], str) or not cgroup["identity"].startswith("/"):
        raise EnvironmentError("environment cgroup identity must be absolute")
    if cgroup["reset_verified"] is not True:
        raise EnvironmentError("environment cgroup peak reset must be verified")
    sampler = _mapping(
        root["sampler"], keys={"complete", "interval_seconds"}, name="environment sampler"
    )
    if not isinstance(sampler["complete"], bool):
        raise EnvironmentError("sampler completeness must be boolean")
    _positive_number(sampler["interval_seconds"], name="sampling interval")
    output = _mapping(root["output"], keys={"persistent", "placement"}, name="environment output")
    if output != {"persistent": True, "placement": "persistent-root"}:
        raise EnvironmentError("environment output must use persistent-root storage")
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


def _is_same_job_cgroup(candidate: str, job: str) -> bool:
    """Accept only the profiled job cgroup or one of its descendants."""
    if candidate == job:
        return True
    prefix = job.rstrip("/") + "/"
    return candidate.startswith(prefix)


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


def _physical_core_count() -> int:
    if sys.platform.startswith("linux"):
        try:
            blocks = Path("/proc/cpuinfo").read_text(encoding="ascii").split("\n\n")
            identities: set[tuple[str, str]] = set()
            for block in blocks:
                facts = {
                    key.strip(): value.strip()
                    for line in block.splitlines()
                    if ":" in line
                    for key, value in (line.split(":", maxsplit=1),)
                }
                if "physical id" in facts and "core id" in facts:
                    identities.add((facts["physical id"], facts["core id"]))
            if identities:
                return len(identities)
        except OSError:
            pass
    return os.cpu_count() or 1


def _operating_system() -> dict[str, str]:
    facts: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", maxsplit=1)
            facts[key] = value.strip().strip('"')
    except OSError:
        pass
    return {
        "id": facts.get("ID", "unknown"),
        "machine": platform.machine(),
        "pretty_name": facts.get("PRETTY_NAME", "unknown"),
        "release": platform.release(),
        "system": platform.system(),
        "version_id": facts.get("VERSION_ID", "unknown"),
    }


def _blas_provenance() -> dict[str, str]:
    try:
        import numpy as np

        configuration = np.__config__.show(mode="dicts")
        dependencies = cast(dict[str, object], configuration.get("Build Dependencies", {}))
        blas = cast(dict[str, object], dependencies.get("blas", {}))
    except Exception as error:
        raise EnvironmentError("BLAS provenance is unavailable") from error
    name = str(blas.get("name", ""))
    version = str(blas.get("version", ""))
    detail = str(blas.get("openblas configuration", blas.get("pc file directory", "")))
    if not name or not version or not detail:
        raise EnvironmentError("BLAS provenance is incomplete")
    return {"configuration": detail, "name": name, "version": version}


def _uv_version() -> str:
    try:
        result = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EnvironmentError("uv version is unavailable") from error
    return _text(result.stdout.strip(), name="uv version")


def _file_digest(path: Path, *, name: str) -> str:
    if not isinstance(path, Path):
        raise EnvironmentError(f"{name} path must be a pathlib.Path")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise EnvironmentError(f"{name} file is unreadable") from error


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


def _commit(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_SHA.fullmatch(value) is None:
        raise EnvironmentError(f"{name} must be a lowercase full commit SHA")
    return value


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EnvironmentError(f"{name} must be a lowercase sha256 digest")
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
