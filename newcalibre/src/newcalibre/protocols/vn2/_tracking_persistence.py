"""Private race-safe proposal publication for VN2 tracking."""

from __future__ import annotations

import contextlib
import os
import stat
from pathlib import Path

from newcalibre.protocols.vn2._tracking_contracts import (
    TrackingError,
    VN2TrackingRecord,
)


def _reject_tracked_path(path: Path) -> None:
    if path.name == "series.jsonl" or ("stage3" in path.parts and "tracking" in path.parts):
        raise TrackingError("tracking writers never accept the tracked history path")


def _successor_root(destination: Path) -> tuple[Path, Path]:
    absolute = destination if destination.is_absolute() else Path.cwd() / destination
    absolute = absolute.absolute()
    _reject_tracked_path(absolute)
    candidates: list[Path] = []
    for ancestor in (absolute.parent, *absolute.parents):
        if ancestor.name != "newcalibre":
            continue
        if any(parent.is_symlink() for parent in (ancestor, *ancestor.parents)):
            raise TrackingError("proposal path contains a symbolic ancestor")
        artifacts = ancestor / "artifacts"
        if (ancestor / "pyproject.toml").is_file() and not artifacts.is_symlink():
            candidates.append(ancestor)
    if not candidates:
        raise TrackingError("proposal path must be beneath the successor newcalibre/artifacts root")
    root = max(candidates, key=lambda path: len(path.parts))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise TrackingError(
            "proposal path must be beneath the successor newcalibre/artifacts root"
        ) from error
    if not relative.parts or relative.parts[0] != "artifacts":
        raise TrackingError("proposal path must be beneath the successor newcalibre/artifacts root")
    return root, relative


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise TrackingError("proposal path contains an unreadable or symbolic directory") from error
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISDIR(mode):
            raise TrackingError("proposal path contains a non-directory ancestor")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_child_directory(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise TrackingError("proposal path contains an unreadable or symbolic directory") from error
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise TrackingError("proposal path contains a non-directory ancestor")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _descend(root: Path, relative_parent: tuple[str, ...]) -> tuple[int, list[int]]:
    root_fd = _open_directory(root)
    fds = [root_fd]
    current = root_fd
    try:
        for name in relative_parent:
            if name in {"", ".", ".."} or "/" in name or "\\" in name:
                raise TrackingError("proposal path must use canonical relative components")
            try:
                child_fd = _open_child_directory(current, name)
            except TrackingError:
                try:
                    os.mkdir(name, mode=0o755, dir_fd=current)
                except FileExistsError:
                    pass
                except OSError as error:
                    raise TrackingError("proposal directory creation failed") from error
                child_fd = _open_child_directory(current, name)
            fds.append(child_fd)
            current = child_fd
        return current, fds
    except BaseException:
        for fd in reversed(fds):
            with contextlib.suppress(OSError):
                os.close(fd)
        raise


def _read_existing(parent_fd: int, name: str) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise TrackingError("tracking proposal destination is unreadable") from error
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise TrackingError("tracking proposal destination must be a regular non-symlink file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        raise TrackingError("tracking proposal destination is unreadable") from error
    finally:
        os.close(fd)


def _publish_unix(parent_fd: int, name: str, payload: bytes) -> bool:
    existing = _read_existing(parent_fd, name)
    if existing is not None:
        if existing == payload:
            return False
        raise TrackingError("tracking proposal destination conflicts with the proposed record")
    temporary_name: str | None = None
    temporary_fd: int | None = None
    try:
        for counter in range(100):
            candidate = f".{name}.{os.getpid()}.{counter}.tmp"
            try:
                temporary_fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o644,
                    dir_fd=parent_fd,
                )
                temporary_name = candidate
                break
            except FileExistsError:
                continue
        if temporary_fd is None or temporary_name is None:
            raise TrackingError("tracking proposal temporary publication failed")
        os.write(temporary_fd, payload)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raced = _read_existing(parent_fd, name)
            if raced == payload:
                return False
            raise TrackingError(
                "tracking proposal destination conflicts with the proposed record"
            ) from error
        os.fsync(parent_fd)
        return True
    except TrackingError:
        raise
    except OSError as error:
        raise TrackingError("tracking proposal publication failed") from error
    finally:
        if temporary_fd is not None:
            with contextlib.suppress(OSError):
                os.close(temporary_fd)
        if temporary_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_fd)


def _write_tracking_record(record: VN2TrackingRecord, path: Path) -> bool:
    root, relative = _successor_root(Path(path))
    if len(relative.parts) < 2:
        raise TrackingError("proposal path must be beneath newcalibre/artifacts")
    parent_fd, fds = _descend(root, tuple(relative.parts[:-1]))
    try:
        name = relative.parts[-1]
        if name in {"", ".", ".."} or "/" in name or "\\" in name:
            raise TrackingError("proposal path must use canonical relative components")
        payload = record.to_bytes()
        if os.name == "posix":
            return _publish_unix(parent_fd, name, payload)
        destination = root / relative
        if destination.exists() or destination.is_symlink():
            existing = destination.read_bytes()
            if existing == payload:
                return False
            raise TrackingError("tracking proposal destination conflicts with the proposed record")
        destination.write_bytes(payload)
        return True
    finally:
        for fd in reversed(fds):
            with contextlib.suppress(OSError):
                os.close(fd)


def write_proposal_record(record: VN2TrackingRecord, path: Path) -> bool:
    """Publish a proposal only beneath the real successor artifacts root."""
    if not isinstance(record, VN2TrackingRecord):
        raise TrackingError("proposal writer requires a VN2TrackingRecord")
    return _write_tracking_record(record, Path(path))
