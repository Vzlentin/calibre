from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import UUID

import fsspec  # type: ignore[import-untyped]
import pandas as pd

from calibre.execution.io import ensure_parent_dir
from calibre.execution.ledger import resolved_ledger_uri


class _Pointer(Protocol):
    uri: str


class _PointerRepo(Protocol):
    def get(self, run_id: UUID, kind: str) -> _Pointer | None: ...


def artifact_pointer(uri: str) -> dict[str, int | str]:
    fs, path = fsspec.core.url_to_fs(uri)
    info = fs.info(path)
    return {"uri": uri, "byte_size": int(info.get("size", 0))}


def _exists(uri: str) -> bool:
    fs, path = fsspec.core.url_to_fs(uri)
    return bool(fs.exists(path))


def canonical_ledger_uri(uri: str) -> str:
    resolved_uri = resolved_ledger_uri(uri)
    return resolved_uri if _exists(resolved_uri) else uri


def write_ledger_shard(df: pd.DataFrame, uri: str) -> dict[str, int | str]:
    ensure_parent_dir(uri)
    with fsspec.open(uri, "wb") as handle:
        df.to_parquet(handle, index=False)
    return artifact_pointer(uri)


def read_run_artifacts(uris: dict[str, str]) -> dict[str, pd.DataFrame]:
    return {kind: pd.read_parquet(uri) for kind, uri in uris.items()}


def read_initial_ledger(pointer_repo: _PointerRepo, run_id: UUID) -> pd.DataFrame | None:
    pointer = pointer_repo.get(run_id, "ledger")
    if pointer is None:
        return None
    uri = canonical_ledger_uri(str(pointer.uri))
    if not _exists(uri):
        return None
    return pd.read_parquet(uri)


def signed_url(uri: str, *, expires: int = 3600) -> str:
    fs, path = fsspec.core.url_to_fs(uri)
    sign = getattr(fs, "sign", None)
    if callable(sign):
        try:
            return str(sign(path, expiration=expires))
        except NotImplementedError:
            pass
    return str(Path(uri)) if "://" not in uri else uri
