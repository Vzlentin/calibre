from __future__ import annotations

import posixpath
from contextlib import suppress
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
from fsspec.spec import AbstractFileSystem


def open_fs(uri: str | Path, **storage_options: Any) -> tuple[AbstractFileSystem, str]:
    return fsspec.core.url_to_fs(str(uri), **storage_options)


def resolve_path(uri: str | Path, **storage_options: Any) -> str:
    _, path = open_fs(uri, **storage_options)
    return path


def exists(uri: str | Path, **storage_options: Any) -> bool:
    fs, path = open_fs(uri, **storage_options)
    return bool(fs.exists(path))


def fs_protocols(fs: AbstractFileSystem) -> set[str]:
    protocol = fs.protocol
    return {protocol} if isinstance(protocol, str) else set(protocol)


def is_local_fs(fs: AbstractFileSystem) -> bool:
    """True for the on-disk filesystem only; remote protocols and memory return False."""
    return not fs_protocols(fs).isdisjoint({"file", "local"})


def ensure_parent_dir(uri: str | Path, **storage_options: Any) -> None:
    fs, path = open_fs(uri, **storage_options)
    if fs_protocols(fs).isdisjoint({"file", "local", "memory"}):
        return
    parent = posixpath.dirname(path.replace("\\", "/"))
    if parent and parent != ".":
        fs.mkdirs(parent, exist_ok=True)


def join_uri(base: str | Path, *parts: str) -> str:
    base_text = str(base)
    if "://" not in base_text:
        return str(Path(base_text).joinpath(*parts))
    return "/".join([base_text.rstrip("/"), *(part.strip("/") for part in parts)])


def write_parquet(frame: pd.DataFrame, uri: str | Path) -> None:
    ensure_parent_dir(uri)
    with fsspec.open(str(uri), "wb") as handle:
        frame.to_parquet(handle, index=False)


def read_parquet(uri: str | Path) -> pd.DataFrame:
    with fsspec.open(str(uri), "rb") as handle:
        return pd.read_parquet(handle)


def rm(uri: str | Path, *, recursive: bool = True) -> None:
    fs, path = open_fs(uri)
    with suppress(FileNotFoundError):
        fs.rm(path, recursive=recursive)
