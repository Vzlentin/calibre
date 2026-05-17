from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Any

import fsspec  # type: ignore[import-untyped]
from fsspec.spec import AbstractFileSystem  # type: ignore[import-untyped]


def _as_uri(uri: str | Path) -> str:
    return str(uri)


def open_fs(uri: str | Path, **storage_options: Any) -> tuple[AbstractFileSystem, str]:
    return fsspec.core.url_to_fs(_as_uri(uri), **storage_options)


def resolve_path(uri: str | Path, **storage_options: Any) -> str:
    _, path = open_fs(uri, **storage_options)
    return path


def exists(uri: str | Path, **storage_options: Any) -> bool:
    fs, path = open_fs(uri, **storage_options)
    return bool(fs.exists(path))


def ensure_parent_dir(uri: str | Path, **storage_options: Any) -> None:
    fs, path = open_fs(uri, **storage_options)
    protocol = fs.protocol
    protocols = {protocol} if isinstance(protocol, str) else set(protocol)
    if protocols.isdisjoint({"file", "local", "memory"}):
        return
    normalized = path.replace("\\", "/")
    parent = posixpath.dirname(normalized)
    if parent and parent != ".":
        fs.mkdirs(parent, exist_ok=True)


def join_uri(base: str | Path, *parts: str) -> str:
    base_text = _as_uri(base)
    if "://" not in base_text:
        return str(Path(base_text).joinpath(*parts))
    return "/".join([base_text.rstrip("/"), *(part.strip("/") for part in parts)])
