"""Model artifact cache: keyed reuse of fitted adapter state across origins."""

from __future__ import annotations

from calibre.execution.io import ensure_parent_dir, exists, join_uri, open_fs


class ModelArtifactCache:
    """fsspec-backed blob store keyed by ``ModelAdapter.cache_key(task)``.

    Conservative: identical-key hits only. No warm-start, no partial reuse.
    The cache stores opaque bytes; serialization is the adapter's job via native
    library ``save`` / ``load`` calls in ``ModelAdapter.dump_state`` /
    ``load_state``.
    """

    def __init__(self, uri: str) -> None:
        self._root = str(uri).rstrip("/\\")
        if not self._root:
            raise ValueError("ModelArtifactCache uri must not be empty")

    def get(self, key: str) -> bytes | None:
        uri = self.uri_for_key(key)
        if not exists(uri):
            return None
        fs, path = open_fs(uri)
        with fs.open(path, "rb") as handle:
            return handle.read()

    def put(self, key: str, blob: bytes) -> None:
        uri = self.uri_for_key(key)
        ensure_parent_dir(uri)
        fs, path = open_fs(uri)
        with fs.open(path, "wb") as handle:
            handle.write(blob)

    def uri_for_key(self, key: str) -> str:
        self._validate_key(key)
        return join_uri(self._root, f"{key}.bin")

    @staticmethod
    def _validate_key(key: str) -> None:
        if not key or "/" in key or "\\" in key:
            raise ValueError(f"Invalid cache key: {key!r}")
