"""Filesystem-agnostic URI helpers and parquet I/O built on fsspec."""

from __future__ import annotations

import posixpath
from contextlib import suppress
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
from fsspec.spec import AbstractFileSystem


def open_fs(uri: str | Path, **storage_options: Any) -> tuple[AbstractFileSystem, str]:
    """Resolve a URI to its ``(filesystem, path)`` pair via fsspec."""
    return fsspec.core.url_to_fs(str(uri), **storage_options)


def resolve_path(uri: str | Path, **storage_options: Any) -> str:
    """Return the filesystem-local path for a URI, dropping the protocol."""
    _, path = open_fs(uri, **storage_options)
    return path


def exists(uri: str | Path, **storage_options: Any) -> bool:
    """Return whether the object at ``uri`` exists."""
    fs, path = open_fs(uri, **storage_options)
    return bool(fs.exists(path))


def fs_protocols(fs: AbstractFileSystem) -> set[str]:
    """Return the protocol name(s) a filesystem responds to as a set."""
    protocol = fs.protocol
    return {protocol} if isinstance(protocol, str) else set(protocol)


def is_local_fs(fs: AbstractFileSystem) -> bool:
    """True for the on-disk filesystem only; remote protocols and memory return False."""
    return not fs_protocols(fs).isdisjoint({"file", "local"})


def ensure_parent_dir(uri: str | Path, **storage_options: Any) -> None:
    """Create the parent directory of ``uri`` for local/memory filesystems.

    Remote object stores need no directories, so they are skipped.
    """
    fs, path = open_fs(uri, **storage_options)
    if fs_protocols(fs).isdisjoint({"file", "local", "memory"}):
        return
    parent = posixpath.dirname(path.replace("\\", "/"))
    if parent and parent != ".":
        fs.mkdirs(parent, exist_ok=True)


def join_uri(base: str | Path, *parts: str) -> str:
    """Join path segments onto ``base``, preserving any ``scheme://`` prefix."""
    base_text = str(base)
    if "://" not in base_text:
        return str(Path(base_text).joinpath(*parts))
    return "/".join([base_text.rstrip("/"), *(part.strip("/") for part in parts)])


def write_parquet(frame: pd.DataFrame, uri: str | Path) -> None:
    """Write a DataFrame to ``uri`` as parquet, creating parent dirs as needed."""
    ensure_parent_dir(uri)
    with fsspec.open(str(uri), "wb") as handle:
        frame.to_parquet(handle, index=False)


def read_parquet(uri: str | Path) -> pd.DataFrame:
    """Read a parquet object at ``uri`` into a DataFrame."""
    with fsspec.open(str(uri), "rb") as handle:
        return pd.read_parquet(handle)


def rm(uri: str | Path, *, recursive: bool = True) -> None:
    """Remove the object at ``uri``, ignoring a missing target."""
    fs, path = open_fs(uri)
    with suppress(FileNotFoundError):
        fs.rm(path, recursive=recursive)
