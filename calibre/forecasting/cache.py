"""Model artifact cache: keyed reuse of fitted adapter state across origins."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


class ModelArtifactCache:
    """Filesystem-backed blob store keyed by ``ModelAdapter.cache_key(task)``.

    Conservative: identical-key hits only. No warm-start, no partial reuse.
    The cache stores opaque bytes; serialization is the adapter's job via
    ``CacheableAdapter.dump_state`` / ``load_state``.

    Two read APIs are exposed: ``get(key)`` for callers that derive the key
    locally from a task, and ``load_by_uri(uri)`` for callers that persisted
    the URI returned from ``uri_for(key)`` and want the URI to be the
    loading contract (no recomputation of the cache key).
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

    def load_by_uri(self, uri: str) -> bytes | None:
        """Load the blob previously stored at ``uri`` (from ``uri_for``).

        Returns ``None`` if the underlying file is missing. Raises
        ``ValueError`` if the URI is not a ``file://`` URI inside this
        cache's root — guards against callers feeding arbitrary paths.
        """
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"Unsupported artifact uri scheme: {parsed.scheme!r}")
        path = Path(url2pathname(unquote(parsed.path))).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(f"Artifact uri outside cache root: {uri}") from exc
        if not path.exists():
            return None
        return path.read_bytes()

    def _path_for(self, key: str) -> Path:
        if not key or "/" in key or "\\" in key:
            raise ValueError(f"Invalid cache key: {key!r}")
        return self._root / f"{key}.bin"
