"""Model artifact cache: keyed reuse of fitted adapter state across origins."""

from __future__ import annotations

from pathlib import Path


class ModelArtifactCache:
    """Filesystem-backed blob store keyed by ``ModelAdapter.cache_key(task)``.

    Conservative: identical-key hits only. No warm-start, no partial reuse.
    The cache stores opaque bytes; serialization is the adapter's job via
    ``ModelAdapter.dump_state`` / ``load_state``.
    """

    def __init__(self, uri: str) -> None:
        self._root = Path(uri).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> bytes | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        return path.read_bytes()

    def put(self, key: str, blob: bytes) -> None:
        self._path_for(key).write_bytes(blob)

    def uri_for(self, key: str) -> str:
        return self._path_for(key).as_uri()

    def _path_for(self, key: str) -> Path:
        if not key or "/" in key or "\\" in key:
            raise ValueError(f"Invalid cache key: {key!r}")
        return self._root / f"{key}.bin"
