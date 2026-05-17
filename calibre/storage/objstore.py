from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import UUID

import fsspec  # type: ignore[import-untyped]
import pandas as pd

from calibre.execution.io import ensure_parent_dir


class _Pointer(Protocol):
    uri: str


class _PointerRepo(Protocol):
    def get(self, run_id: UUID, kind: str) -> _Pointer | None: ...


def artifact_pointer(uri: str) -> dict[str, int | str]:
    fs, path = fsspec.core.url_to_fs(uri)
    info = fs.info(path)
    return {"uri": uri, "byte_size": int(info.get("size", 0))}


def write_ledger_shard(df: pd.DataFrame, uri: str) -> dict[str, int | str]:
    ensure_parent_dir(uri)
    df.to_parquet(uri, index=False)
    return artifact_pointer(uri)


def read_run_artifacts(uris: dict[str, str]) -> dict[str, pd.DataFrame]:
    return {kind: pd.read_parquet(uri) for kind, uri in uris.items()}


def read_initial_ledger(pointer_repo: _PointerRepo, run_id: UUID) -> pd.DataFrame | None:
    pointer = pointer_repo.get(run_id, "ledger")
    if pointer is None:
        return None
    return pd.read_parquet(str(pointer.uri))


def signed_url(uri: str, *, expires: int = 3600) -> str:
    fs, path = fsspec.core.url_to_fs(uri)
    sign = getattr(fs, "sign", None)
    if callable(sign):
        try:
            return str(sign(path, expiration=expires))
        except NotImplementedError:
            pass
    return str(Path(uri)) if "://" not in uri else uri
