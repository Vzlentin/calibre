from __future__ import annotations

from pathlib import Path

import fsspec  # type: ignore[import-untyped]
import pandas as pd


def write_ledger_shard(df: pd.DataFrame, uri: str) -> dict[str, int | str]:
    df.to_parquet(uri, index=False)
    fs, path = fsspec.core.url_to_fs(uri)
    info = fs.info(path)
    return {"uri": uri, "byte_size": int(info.get("size", 0))}


def read_run_artifacts(uris: dict[str, str]) -> dict[str, pd.DataFrame]:
    return {kind: pd.read_parquet(uri) for kind, uri in uris.items()}


def signed_url(uri: str, *, expires: int = 3600) -> str:
    fs, path = fsspec.core.url_to_fs(uri)
    sign = getattr(fs, "sign", None)
    if callable(sign):
        try:
            return str(sign(path, expiration=expires))
        except NotImplementedError:
            pass
    return str(Path(uri)) if "://" not in uri else uri
