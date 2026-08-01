"""Export one disposable successor ledger in the frozen M5 scorer schema."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from urllib.parse import unquote

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from newcalibre.domain import (
    AGGREGATE_NODE_PREFIX,
    TOTAL_NODE_LABEL,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    ScoredSeries,
    interval_columns,
)
from newcalibre.engine import (
    LedgerBatch,
    LedgerBoundIssuance,
    LedgerColumn,
    LedgerForecastKey,
    LedgerReader,
    LedgerResolution,
    LedgerSelection,
    LedgerSessionMetadata,
)
from newcalibre.protocols.m5.config import M5ProtocolConfig

_COVERAGE = 0.9
_BATCH_SIZE = 4096
_FROZEN_COLUMNS = ("unique_id", "h", "model_name", "y", "lo_0p9", "hi_0p9")
_FROZEN_LEVELS = {
    "item": "item_id",
    "department": "dept_id",
    "category": "cat_id",
    "store": "store_id",
    "state": "state_id",
}
_SCHEMA = pa.schema(
    (
        pa.field("unique_id", pa.string(), nullable=False),
        pa.field("h", pa.int64(), nullable=False),
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("y", pa.float64()),
        pa.field("lo_0p9", pa.float64()),
        pa.field("hi_0p9", pa.float64()),
    )
)


class FrozenM5ExportError(ValueError):
    """Report a malformed intent, ledger row, or disposable parquet export."""


def export_frozen_m5_ledger(
    config: M5ProtocolConfig,
    ledger: LedgerReader,
    output: Path,
    *,
    batch_size: int = _BATCH_SIZE,
) -> None:
    """Stream one closed ledger into the exact six-column frozen scorer schema."""
    model_name = _validate_intent(config)
    if not isinstance(output, Path):
        raise FrozenM5ExportError("frozen M5 export path must be a pathlib.Path")
    if output.exists() or output.is_symlink():
        raise FrozenM5ExportError("frozen M5 export path must not already exist")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise FrozenM5ExportError("frozen M5 export batch size must be a positive integer")
    try:
        metadata = ledger.metadata
    except Exception as error:
        raise FrozenM5ExportError("frozen M5 export reader metadata is unavailable") from error
    if not isinstance(metadata, LedgerSessionMetadata):
        raise FrozenM5ExportError("frozen M5 export requires LedgerSessionMetadata")

    partial = output.with_suffix(f"{output.suffix}.partial")
    if partial.exists() or partial.is_symlink():
        raise FrozenM5ExportError("frozen M5 partial export path must not already exist")
    selection = LedgerSelection(
        metadata.session,
        (LedgerColumn.ISSUANCES, LedgerColumn.RESOLUTION),
        batch_size,
    )
    seen: set[LedgerForecastKey] = set()
    row_count = 0
    try:
        with pq.ParquetWriter(partial, _SCHEMA) as writer:
            for batch in ledger.scan(selection):
                table, keys = _export_batch(
                    batch,
                    metadata=metadata,
                    model_name=model_name,
                    seen=seen,
                )
                writer.write_table(table)
                seen.update(keys)
                row_count += table.num_rows
        if row_count == 0:
            raise FrozenM5ExportError("frozen M5 export ledger must not be empty")
        partial.replace(output)
    except FrozenM5ExportError:
        partial.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise
    except Exception as error:
        partial.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise FrozenM5ExportError("frozen M5 parquet export failed") from error


def _validate_intent(config: M5ProtocolConfig) -> str:
    if not isinstance(config, M5ProtocolConfig):
        raise FrozenM5ExportError("frozen M5 export requires M5ProtocolConfig")
    conformal = config.conformal_config
    if conformal.get("coverage") != _COVERAGE:
        raise FrozenM5ExportError("frozen M5 export supports only 0.9 scoring intent")
    if config.conformal_partition != "series-horizon":
        raise FrozenM5ExportError("frozen M5 export requires series-horizon scoring intent")
    model_name = config.model_config.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise FrozenM5ExportError("frozen M5 export requires one configured model name")
    return model_name


def _export_batch(
    batch: object,
    *,
    metadata: LedgerSessionMetadata,
    model_name: str,
    seen: set[LedgerForecastKey],
) -> tuple[pa.Table, tuple[LedgerForecastKey, ...]]:
    if not isinstance(batch, LedgerBatch):
        raise FrozenM5ExportError("frozen M5 export reader yielded a non-LedgerBatch value")
    if batch.session != metadata.session:
        raise FrozenM5ExportError("frozen M5 export batch has the wrong session")
    if set(batch.columns) != {"issuances", "resolution"}:
        raise FrozenM5ExportError("frozen M5 export batch has the wrong projection")

    values: dict[str, list[object]] = {column: [] for column in _FROZEN_COLUMNS}
    batch_seen: set[LedgerForecastKey] = set()
    issuances = batch.columns[LedgerColumn.ISSUANCES.value]
    resolutions = batch.columns[LedgerColumn.RESOLUTION.value]
    for key, raw_issuances, raw_resolution in zip(
        batch.keys,
        issuances,
        resolutions,
        strict=True,
    ):
        if key in seen or key in batch_seen:
            raise FrozenM5ExportError(f"frozen M5 export contains duplicate row {key!r}")
        _validate_key(key, metadata=metadata, model_name=model_name)
        issuance = _validate_issuance(raw_issuances, key=key)
        resolution = _validate_resolution(raw_resolution, key=key)
        upper = cast(float | None, issuance.bound_values[0])
        actual = None if resolution is None else resolution.actual_value
        lower = actual if actual is not None and upper is not None else None
        values["unique_id"].append(_frozen_label(key.series_key))
        values["h"].append(key.horizon_step)
        values["model_name"].append(key.model_name)
        values["y"].append(actual)
        values["lo_0p9"].append(lower)
        values["hi_0p9"].append(upper)
        batch_seen.add(key)
    return pa.Table.from_pydict(values, schema=_SCHEMA), tuple(batch.keys)


def _validate_key(
    key: LedgerForecastKey,
    *,
    metadata: LedgerSessionMetadata,
    model_name: str,
) -> None:
    if key.series_key not in metadata.series_keys:
        raise FrozenM5ExportError(f"frozen M5 export contains unexpected node {key.series_key!r}")
    if key.model_name != model_name:
        raise FrozenM5ExportError(f"frozen M5 export contains unexpected model {key.model_name!r}")
    if key.horizon_step > 28:
        raise FrozenM5ExportError(
            f"frozen M5 export contains unexpected horizon {key.horizon_step}"
        )


def _validate_issuance(
    value: object,
    *,
    key: LedgerForecastKey,
) -> LedgerBoundIssuance:
    if not isinstance(value, tuple) or len(value) != 1:
        raise FrozenM5ExportError(f"frozen M5 export row {key!r} must contain exactly one issuance")
    issuance = value[0]
    if not isinstance(issuance, LedgerBoundIssuance):
        raise FrozenM5ExportError(f"frozen M5 export row {key!r} has a malformed issuance")
    descriptor = issuance.descriptor
    expected_upper = interval_columns(_COVERAGE)[1]
    if (
        descriptor.level != _COVERAGE
        or descriptor.type.claim is not GuaranteeClaim.ONE_SIDED_COVERAGE
        or (issuance.bounds_finite and descriptor.scored_series is not ScoredSeries.RECORDED_SALES)
        or descriptor.window is not EmissionScope.PER_STEP
        or descriptor.scope.kind is not DecisionScopeKind.PER_DECISION_NODE
        or descriptor.scope.class_system_name is not None
        or issuance.guaranteed_side != "upper"
        or issuance.bound_key != (expected_upper,)
        or len(issuance.bound_values) != 1
        or issuance.calibration_ready != issuance.bounds_finite
    ):
        raise FrozenM5ExportError(
            f"frozen M5 export row {key!r} has unexpected 0.9 scoring issuance"
        )
    return issuance


def _validate_resolution(
    value: object,
    *,
    key: LedgerForecastKey,
) -> LedgerResolution | None:
    if value is None:
        return None
    if not isinstance(value, LedgerResolution):
        raise FrozenM5ExportError(f"frozen M5 export row {key!r} has malformed resolution")
    expected_target = key.origin + pd.Timedelta(days=key.horizon_step - 1)
    if value.target_timestamp != expected_target:
        raise FrozenM5ExportError(f"frozen M5 export row {key!r} has unexpected resolution target")
    if value.censoring_assertion is not None or value.availability_bound is not None:
        raise FrozenM5ExportError(f"frozen M5 export row {key!r} has unexpected censoring facts")
    return value


def _frozen_label(label: str) -> str:
    if label == TOTAL_NODE_LABEL:
        return label
    prefix = f"{AGGREGATE_NODE_PREFIX}:"
    if not label.startswith(prefix):
        if label.startswith(AGGREGATE_NODE_PREFIX):
            raise FrozenM5ExportError(f"malformed canonical M5 node label {label!r}")
        return label
    tokens = label.split(":", maxsplit=3)
    if len(tokens) != 4 or tokens[2] != "s":
        raise FrozenM5ExportError(f"malformed canonical M5 aggregate label {label!r}")
    level = unquote(tokens[1])
    value = unquote(tokens[3])
    frozen_level = _FROZEN_LEVELS.get(level)
    if frozen_level is None or not value:
        raise FrozenM5ExportError(f"unknown canonical M5 aggregate label {label!r}")
    return f"{frozen_level}={value}"


__all__ = ["FrozenM5ExportError", "export_frozen_m5_ledger"]
