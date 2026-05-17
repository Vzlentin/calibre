from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from uuid import uuid4

import pandas as pd

from calibre.api.schemas import RunResponse
from calibre.cli.commands import run_config
from calibre.cli.config import load_config_from_mapping
from calibre.storage.objstore import signed_url


@dataclass
class RunRecord:
    id: str
    config: dict
    status: str = "queued"
    artifact_urls: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def response(self) -> RunResponse:
        return RunResponse(
            id=self.id,
            status=self.status,  # type: ignore[arg-type]
            artifact_urls=dict(self.artifact_urls),
            error=self.error,
        )


_RUNS: dict[str, RunRecord] = {}
_IDEMPOTENCY: dict[str, str] = {}


def create_run(config: dict, *, idempotency_key: str | None = None) -> RunResponse:
    if idempotency_key is not None and idempotency_key in _IDEMPOTENCY:
        return _RUNS[_IDEMPOTENCY[idempotency_key]].response()
    run_id = str(uuid4())
    record = RunRecord(id=run_id, config=config)
    _RUNS[run_id] = record
    if idempotency_key is not None:
        _IDEMPOTENCY[idempotency_key] = run_id
    return record.response()


def get_run(run_id: str) -> RunResponse | None:
    record = _RUNS.get(run_id)
    return record.response() if record is not None else None


def run_backtest_job(run_id: str) -> None:
    record = _RUNS[run_id]
    record.status = "running"
    try:
        config = load_config_from_mapping(record.config)
        result = run_config(config)
        if isinstance(result, pd.DataFrame):
            rows = len(result)
        else:
            rows = len(result.ledger.to_df())
            if config.output.ledger_path is not None:
                record.artifact_urls["ledger"] = signed_url(config.output.ledger_path)
            if config.output.order_ledger_path is not None:
                record.artifact_urls["order_ledger"] = signed_url(config.output.order_ledger_path)
        record.artifact_urls["rows"] = str(rows)
        record.status = "succeeded"
    except Exception as exc:  # pragma: no cover - exercised through API test as status
        record.status = "failed"
        record.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
