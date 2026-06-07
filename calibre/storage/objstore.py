from __future__ import annotations

import logging
import os
import tempfile
from typing import Protocol
from uuid import UUID

import pandas as pd

from calibre.core.io import exists, open_fs, write_parquet
from calibre.execution.ledger import resolved_ledger_uri

logger = logging.getLogger(__name__)


def artifact_base_uri() -> str:
    """Base URI for lifecycle frame parquet and model artifacts.

    Reads ``CALIBRE_ARTIFACT_URI``. Falls back to a local directory, which is
    correct for single-host use only: multi-worker / multi-host deployments
    must point this at a shared object store (e.g. ``s3://bucket/prefix``) or
    workers cannot read each other's fit frames.
    """
    uri = os.environ.get("CALIBRE_ARTIFACT_URI")
    if uri:
        return uri
    fallback = os.path.join(tempfile.gettempdir(), "calibre-artifacts")
    logger.warning(
        "CALIBRE_ARTIFACT_URI is unset; using local path %s. Multi-worker "
        "deployments must set a shared object store (CALIBRE_ARTIFACT_URI=s3://...).",
        fallback,
    )
    return fallback


class _Pointer(Protocol):
    uri: str


class _PointerRepo(Protocol):
    def get(self, run_id: UUID, kind: str) -> _Pointer | None: ...


def artifact_pointer(uri: str) -> dict[str, int | str]:
    fs, path = open_fs(uri)
    info = fs.info(path)
    return {"uri": uri, "byte_size": int(info.get("size", 0))}


def canonical_ledger_uri(uri: str) -> str:
    resolved_uri = resolved_ledger_uri(uri)
    return resolved_uri if exists(resolved_uri) else uri


def write_ledger_shard(df: pd.DataFrame, uri: str) -> dict[str, int | str]:
    write_parquet(df, uri)
    return artifact_pointer(uri)


def read_run_artifacts(uris: dict[str, str]) -> dict[str, pd.DataFrame]:
    return {kind: pd.read_parquet(uri) for kind, uri in uris.items()}


def read_initial_ledger(pointer_repo: _PointerRepo, run_id: UUID) -> pd.DataFrame | None:
    pointer = pointer_repo.get(run_id, "ledger")
    if pointer is None:
        return None
    uri = canonical_ledger_uri(str(pointer.uri))
    if not exists(uri):
        return None
    return pd.read_parquet(uri)


def signed_url(uri: str, *, expires: int = 3600) -> str:
    fs, path = open_fs(uri)
    sign = getattr(fs, "sign", None)
    if callable(sign):
        try:
            return str(sign(path, expiration=expires))
        except NotImplementedError:
            pass
    return uri
