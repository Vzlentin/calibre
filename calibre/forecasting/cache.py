"""Model artifact cache: keyed reuse of fitted adapter state across origins."""

from __future__ import annotations

import posixpath
from pathlib import Path

import fsspec


def _join_uri(base: str, *parts: str) -> str:
    if "://" not in base:
        return str(Path(base).joinpath(*parts))
    return "/".join([base.rstrip("/"), *(part.strip("/") for part in parts)])


def _ensure_parent_dir(uri: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    protocol = fs.protocol
    protocols = {protocol} if isinstance(protocol, str) else set(protocol)
    if protocols.isdisjoint({"file", "local", "memory"}):
        return
    parent = posixpath.dirname(path.replace("\\", "/"))
    if parent and parent != ".":
        fs.mkdirs(parent, exist_ok=True)


class ModelArtifactCache:
    """fsspec-backed blob store keyed by ``ModelAdapter.cache_key(task)``.

    Conservative: identical-key hits only. No warm-start, no partial reuse.
    The cache stores opaque bytes; serialization is the adapter's job via native
    library ``save`` / ``load`` calls in ``ModelAdapter.dump_state`` /
    ``load_state``.
    """

    def __init__(self, uri: str) -> None:
        root = str(uri)
        self._root = root if root.endswith("://") else root.rstrip("/\\")
        if not self._root:
            raise ValueError("ModelArtifactCache uri must not be empty")

    def get(self, key: str) -> bytes | None:
        uri = self.uri_for_key(key)
        fs, path = fsspec.core.url_to_fs(uri)
        if not fs.exists(path):
            return None
        with fs.open(path, "rb") as handle:
            return handle.read()

    def put(self, key: str, blob: bytes) -> None:
        uri = self.uri_for_key(key)
        _ensure_parent_dir(uri)
        fs, path = fsspec.core.url_to_fs(uri)
        with fs.open(path, "wb") as handle:
            handle.write(blob)

    def uri_for_key(self, key: str) -> str:
        self._validate_key(key)
        if self._root.endswith("://"):
            return f"{self._root}{key}.bin"
        return _join_uri(self._root, f"{key}.bin")

    @staticmethod
    def _validate_key(key: str) -> None:
        if not key or "/" in key or "\\" in key:
            raise ValueError(f"Invalid cache key: {key!r}")
