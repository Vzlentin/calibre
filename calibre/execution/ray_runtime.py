from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from types import ModuleType

_LOCK = threading.RLock()
_LOCAL_REFCOUNT = 0
_LOCAL_RAY: ModuleType | None = None


def prepare_ray_environment() -> None:
    """Disable Ray's uv working_dir hook before importing or initializing Ray."""
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    try:
        import ray._private.ray_constants as ray_constants

        ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    except ImportError:
        return


@dataclass
class RayRuntimeHandle:
    ray: ModuleType
    owns_local_runtime: bool = False

    def release(self) -> None:
        release_ray_runtime(self)


def acquire_ray_runtime(
    *,
    address: str | None = None,
    local_mode: bool = False,
) -> RayRuntimeHandle:
    """Initialize or attach to Ray with explicit local ownership accounting."""
    global _LOCAL_RAY, _LOCAL_REFCOUNT

    with _LOCK:
        prepare_ray_environment()

        import ray

        if ray.is_initialized():
            if address is None and _LOCAL_RAY is ray:
                _LOCAL_REFCOUNT += 1
                return RayRuntimeHandle(ray=ray, owns_local_runtime=True)
            return RayRuntimeHandle(ray=ray, owns_local_runtime=False)

        _LOCAL_RAY = None
        _LOCAL_REFCOUNT = 0

        if address is not None:
            ray.init(address=address, ignore_reinit_error=True)
            return RayRuntimeHandle(ray=ray, owns_local_runtime=False)

        ray.init(
            include_dashboard=False,
            ignore_reinit_error=True,
            local_mode=local_mode,
            _skip_env_hook=True,
        )
        _LOCAL_RAY = ray
        _LOCAL_REFCOUNT = 1
        return RayRuntimeHandle(ray=ray, owns_local_runtime=True)


def release_ray_runtime(handle: RayRuntimeHandle | None) -> None:
    """Release a local Ray runtime acquired through this module."""
    global _LOCAL_RAY, _LOCAL_REFCOUNT

    if handle is None or not handle.owns_local_runtime:
        return

    with _LOCK:
        _LOCAL_REFCOUNT = max(0, _LOCAL_REFCOUNT - 1)
        if _LOCAL_REFCOUNT > 0:
            return
        if handle.ray.is_initialized():
            handle.ray.shutdown()
        _LOCAL_RAY = None
