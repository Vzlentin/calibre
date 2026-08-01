"""Centralize checkpoint-bound adapter fit, update, load, and prediction."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Final, cast

import pandas as pd

from newcalibre.domain import CycleToken, ForecastTask, HistoryCursor, SessionIdentity
from newcalibre.domain._canonical_json import (
    CanonicalJsonError,
    canonical_json_bytes,
)
from newcalibre.engine.ports import ArtifactStore
from newcalibre.forecasting import AdapterCapability, ForecastAdapter

_CHECKPOINT_SCHEMA: Final = "newcalibre.forecast-checkpoint/v1"
_CHECKPOINT_INDEX_SCHEMA: Final = "newcalibre.forecast-checkpoint-index/v1"

type AdapterResolver = Callable[[Mapping[str, object]], ForecastAdapter]
type ForecastLifecycleItem = tuple[SessionIdentity, ForecastTask, CycleToken]


class ForecastLifecycleError(RuntimeError):
    """Report invalid checkpoint metadata or adapter lifecycle composition."""


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    session_value: str
    task_identity: str
    lineage_identity: str
    config_digest: str
    cursor: HistoryCursor
    capabilities: tuple[str, ...]
    fit_time_bound: int
    native_state: bytes


@dataclass(slots=True)
class _Prepared:
    adapter: ForecastAdapter
    key: str | None
    index_key: str | None
    lineage_identity: str
    config_digest: str
    fit_time_bound: int


@dataclass(frozen=True, slots=True)
class _PendingCheckpoint:
    key: str
    value: bytes
    index_key: str
    index_value: bytes


@dataclass(frozen=True, slots=True)
class ForecastLifecycleResult:
    """Stage one adapter prediction and its unpublished checkpoint effect."""

    frame: pd.DataFrame
    checkpoint: _PendingCheckpoint | None


class ForecastLifecycle:
    """Own one same-instance adapter lifecycle and checkpoint publication seam."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        adapter_resolver: AdapterResolver,
    ) -> None:
        if not isinstance(artifact_store, ArtifactStore):
            raise TypeError("forecast lifecycle artifact store must satisfy ArtifactStore")
        if not callable(adapter_resolver):
            raise TypeError("forecast lifecycle adapter resolver must be callable")
        self._artifact_store = artifact_store
        self._adapter_resolver = adapter_resolver

    def _prepare_adapter(
        self,
        *,
        session: SessionIdentity,
        task: ForecastTask,
    ) -> _Prepared:
        adapter = self._adapter_resolver(task.model_config)
        capabilities = _capability_names(adapter)
        lineage = _lineage_identity(task)
        config_digest = _config_digest(task.model_config)
        persistent = AdapterCapability.ARTIFACT_PERSISTENCE in adapter.capabilities
        key = (
            _checkpoint_key(
                session=session,
                lineage_identity=lineage,
                config_digest=config_digest,
                cursor=task.cursor,
            )
            if persistent
            else None
        )
        index_key = (
            _checkpoint_index_key(
                session=session,
                lineage_identity=lineage,
                config_digest=config_digest,
            )
            if persistent
            else None
        )
        fit_time_bound = task.cursor.time_bound

        if persistent:
            assert key is not None
            exact = self._artifact_store.load(key)
            if exact is not None:
                checkpoint = _decode_checkpoint(exact)
                _require_checkpoint(
                    checkpoint,
                    session=session,
                    task=task,
                    expected_cursor=task.cursor,
                    expected_task_identity=task.identity,
                    lineage_identity=lineage,
                    config_digest=config_digest,
                    capabilities=capabilities,
                )
                adapter.load_state(checkpoint.native_state)
                fit_time_bound = checkpoint.fit_time_bound
            else:
                prior = self._latest_checkpoint(
                    session=session,
                    task=task,
                    index_key=index_key,
                    lineage_identity=lineage,
                    config_digest=config_digest,
                    capabilities=capabilities,
                )
                if prior is None:
                    adapter.fit(task)
                else:
                    checkpoint = prior
                    adapter.load_state(checkpoint.native_state)
                    fit_time_bound = checkpoint.fit_time_bound
                    if AdapterCapability.INCREMENTAL_UPDATE in adapter.capabilities:
                        if task.delta.start_cursor != checkpoint.cursor:
                            raise ForecastLifecycleError(
                                "forecast task delta does not start at its loaded checkpoint"
                            )
                        adapter.update(task.delta)
                    else:
                        cadence = _refit_cadence(task.model_config)
                        if cadence is None:
                            raise ForecastLifecycleError(
                                "a non-updatable adapter requires explicit refit_cadence"
                            )
                        if task.cursor.time_bound - fit_time_bound >= cadence:
                            adapter.fit(task)
                            fit_time_bound = task.cursor.time_bound
        else:
            adapter.fit(task)

        if (
            AdapterCapability.INCREMENTAL_UPDATE in adapter.capabilities
            and _refit_cadence(task.model_config) is not None
        ):
            raise ForecastLifecycleError(
                "refit_cadence is only valid for adapters without incremental update"
            )
        return _Prepared(
            adapter=adapter,
            key=key,
            index_key=index_key,
            lineage_identity=lineage,
            config_digest=config_digest,
            fit_time_bound=fit_time_bound,
        )

    def run_item(self, item: ForecastLifecycleItem) -> ForecastLifecycleResult:
        """Run one adapter lifecycle atomically in its dispatched placement."""
        session, task, token = item
        _require_cycle(session=session, task=task, token=token)
        prepared = self._prepare_adapter(session=session, task=task)
        return self._predict_result(session=session, task=task, prepared=prepared)

    def _predict_result(
        self,
        *,
        session: SessionIdentity,
        task: ForecastTask,
        prepared: _Prepared,
    ) -> ForecastLifecycleResult:
        frame = prepared.adapter.predict(task)
        pending = None
        if prepared.key is not None:
            checkpoint = _Checkpoint(
                session_value=session.value,
                task_identity=task.identity,
                lineage_identity=prepared.lineage_identity,
                config_digest=prepared.config_digest,
                cursor=task.cursor,
                capabilities=_capability_names(prepared.adapter),
                fit_time_bound=prepared.fit_time_bound,
                native_state=prepared.adapter.dump_state(),
            )
            assert prepared.index_key is not None
            pending = _PendingCheckpoint(
                key=prepared.key,
                value=_encode_checkpoint(checkpoint),
                index_key=prepared.index_key,
                index_value=_encode_checkpoint_index(
                    cursor=task.cursor,
                    checkpoint_key=prepared.key,
                ),
            )
        return ForecastLifecycleResult(frame=frame, checkpoint=pending)

    def publish(self, results: tuple[ForecastLifecycleResult, ...]) -> None:
        """Publish a fully accepted forecast batch's staged checkpoints."""
        checkpoints = tuple(
            result.checkpoint for result in results if result.checkpoint is not None
        )
        self._artifact_store.publish(
            {checkpoint.key: checkpoint.value for checkpoint in checkpoints},
            {checkpoint.index_key: checkpoint.index_value for checkpoint in checkpoints},
        )

    def previous_cursors(
        self,
        *,
        session: SessionIdentity,
        tasks: tuple[ForecastTask, ...],
    ) -> dict[tuple[str, ...], HistoryCursor]:
        """Restore indexed predecessor cursors before final task construction."""
        restored: dict[tuple[str, ...], HistoryCursor] = {}
        for task in tasks:
            adapter = self._adapter_resolver(task.model_config)
            if AdapterCapability.ARTIFACT_PERSISTENCE not in adapter.capabilities:
                continue
            lineage = _lineage_identity(task)
            config_digest = _config_digest(task.model_config)
            index_key = _checkpoint_index_key(
                session=session,
                lineage_identity=lineage,
                config_digest=config_digest,
            )
            encoded = self._artifact_store.load_index(index_key)
            if encoded is None:
                continue
            cursor, checkpoint_key = _decode_checkpoint_index(encoded)
            if (
                cursor.panel_identity != task.cursor.panel_identity
                or cursor.series_start != task.cursor.series_start
                or cursor.series_stop != task.cursor.series_stop
                or cursor.time_bound > task.cursor.time_bound
            ):
                raise ForecastLifecycleError("forecast checkpoint index is stale or foreign")
            expected_key = _checkpoint_key(
                session=session,
                lineage_identity=lineage,
                config_digest=config_digest,
                cursor=cursor,
            )
            if checkpoint_key != expected_key:
                raise ForecastLifecycleError("forecast checkpoint index names an invalid artifact")
            restored[task.series_keys] = cursor
        return restored

    def _latest_checkpoint(
        self,
        *,
        session: SessionIdentity,
        task: ForecastTask,
        index_key: str | None,
        lineage_identity: str,
        config_digest: str,
        capabilities: tuple[str, ...],
    ) -> _Checkpoint | None:
        assert index_key is not None
        encoded_index = self._artifact_store.load_index(index_key)
        if encoded_index is None:
            return None
        cursor, checkpoint_key = _decode_checkpoint_index(encoded_index)
        if (
            cursor.panel_identity != task.cursor.panel_identity
            or cursor.series_start != task.cursor.series_start
            or cursor.series_stop != task.cursor.series_stop
            or cursor.time_bound >= task.cursor.time_bound
        ):
            raise ForecastLifecycleError("forecast checkpoint index is stale or foreign")
        expected_key = _checkpoint_key(
            session=session,
            lineage_identity=lineage_identity,
            config_digest=config_digest,
            cursor=cursor,
        )
        if checkpoint_key != expected_key:
            raise ForecastLifecycleError("forecast checkpoint index names an invalid artifact")
        encoded = self._artifact_store.load(checkpoint_key)
        if encoded is None:
            raise ForecastLifecycleError("forecast checkpoint index names a missing artifact")
        checkpoint = _decode_checkpoint(encoded)
        _require_checkpoint(
            checkpoint,
            session=session,
            task=task,
            expected_cursor=cursor,
            expected_task_identity=None,
            lineage_identity=lineage_identity,
            config_digest=config_digest,
            capabilities=capabilities,
        )
        return checkpoint


def _require_cycle(
    *,
    session: SessionIdentity,
    task: ForecastTask,
    token: CycleToken,
) -> None:
    if not isinstance(session, SessionIdentity):
        raise TypeError("forecast lifecycle session must be a SessionIdentity")
    if not isinstance(task, ForecastTask):
        raise TypeError("forecast lifecycle task must be a ForecastTask")
    if not isinstance(token, CycleToken):
        raise TypeError("forecast lifecycle token must be a CycleToken")
    if token.session != session or token.origin != task.origin:
        raise ForecastLifecycleError("forecast lifecycle token does not match its task")


def _refit_cadence(model_config: Mapping[str, object]) -> int | None:
    value = model_config.get("refit_cadence")
    if value is None:
        return None
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ForecastLifecycleError("refit_cadence must be a positive integer")
    return int(value)


def _lineage_identity(task: ForecastTask) -> str:
    payload = canonical_json_bytes(
        {
            "panel_identity": task.cursor.panel_identity,
            "scope": task.scope.value,
            "series_keys": list(task.series_keys),
            "series_start": task.cursor.series_start,
            "series_stop": task.cursor.series_stop,
        },
        path="forecast checkpoint lineage",
    )
    return hashlib.sha256(b"newcalibre.forecast-lineage/v1\0" + payload).hexdigest()


def _config_digest(model_config: Mapping[str, object]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(dict(model_config), path="forecast checkpoint configuration")
    ).hexdigest()


def _capability_names(adapter: ForecastAdapter) -> tuple[str, ...]:
    return tuple(sorted((capability.value for capability in adapter.capabilities), key=str.encode))


def _checkpoint_key(
    *,
    session: SessionIdentity,
    lineage_identity: str,
    config_digest: str,
    cursor: HistoryCursor,
) -> str:
    payload = canonical_json_bytes(
        {
            "config_digest": config_digest,
            "cursor": _cursor_record(cursor),
            "lineage_identity": lineage_identity,
            "session": session.value,
        },
        path="forecast checkpoint key",
    )
    return f"forecast-checkpoint:{hashlib.sha256(payload).hexdigest()}"


def _checkpoint_index_key(
    *,
    session: SessionIdentity,
    lineage_identity: str,
    config_digest: str,
) -> str:
    payload = canonical_json_bytes(
        {
            "config_digest": config_digest,
            "lineage_identity": lineage_identity,
            "session": session.value,
        },
        path="forecast checkpoint index key",
    )
    return f"forecast-checkpoint-index:{hashlib.sha256(payload).hexdigest()}"


def _encode_checkpoint(checkpoint: _Checkpoint) -> bytes:
    return canonical_json_bytes(
        {
            "capabilities": list(checkpoint.capabilities),
            "config_digest": checkpoint.config_digest,
            "cursor": _cursor_record(checkpoint.cursor),
            "fit_time_bound": checkpoint.fit_time_bound,
            "lineage_identity": checkpoint.lineage_identity,
            "native_state": base64.b64encode(checkpoint.native_state).decode("ascii"),
            "schema": _CHECKPOINT_SCHEMA,
            "session": checkpoint.session_value,
            "task_identity": checkpoint.task_identity,
        },
        path="forecast checkpoint",
    )


def _encode_checkpoint_index(*, cursor: HistoryCursor, checkpoint_key: str) -> bytes:
    return canonical_json_bytes(
        {
            "checkpoint_key": checkpoint_key,
            "cursor": _cursor_record(cursor),
            "schema": _CHECKPOINT_INDEX_SCHEMA,
        },
        path="forecast checkpoint index",
    )


def _decode_checkpoint_index(encoded: bytes) -> tuple[HistoryCursor, str]:
    if not isinstance(encoded, bytes):
        raise TypeError("forecast checkpoint index must be bytes")
    try:
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise TypeError
        if canonical_json_bytes(payload, path="forecast checkpoint index") != encoded:
            raise ValueError("not canonical")
        if payload.get("schema") != _CHECKPOINT_INDEX_SCHEMA:
            raise ValueError("unsupported schema")
        checkpoint_key = payload["checkpoint_key"]
        if not isinstance(checkpoint_key, str) or not checkpoint_key:
            raise TypeError("checkpoint key")
        return _cursor_from_record(payload["cursor"]), checkpoint_key
    except (
        CanonicalJsonError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise ForecastLifecycleError(f"forecast checkpoint index is malformed: {error}") from error


def _decode_checkpoint(encoded: bytes) -> _Checkpoint:
    if not isinstance(encoded, bytes):
        raise TypeError("forecast checkpoint must be bytes")
    try:
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise TypeError
        if canonical_json_bytes(payload, path="forecast checkpoint") != encoded:
            raise ValueError("not canonical")
        if payload.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("unsupported schema")
        raw_capabilities = payload["capabilities"]
        if not isinstance(raw_capabilities, list) or any(
            not isinstance(value, str) for value in raw_capabilities
        ):
            raise TypeError
        capabilities = tuple(raw_capabilities)
        if capabilities != tuple(sorted(set(capabilities), key=str.encode)):
            raise ValueError("capabilities are not canonical")
        native_state = base64.b64decode(payload["native_state"], validate=True)
        checkpoint = _Checkpoint(
            session_value=_require_digest(payload["session"]),
            task_identity=_require_digest(payload["task_identity"]),
            lineage_identity=_require_digest(payload["lineage_identity"]),
            config_digest=_require_digest(payload["config_digest"]),
            cursor=_cursor_from_record(payload["cursor"]),
            capabilities=capabilities,
            fit_time_bound=cast(int, payload["fit_time_bound"]),
            native_state=native_state,
        )
        if (
            not isinstance(checkpoint.fit_time_bound, int)
            or isinstance(checkpoint.fit_time_bound, bool)
            or checkpoint.fit_time_bound < 0
            or checkpoint.fit_time_bound > checkpoint.cursor.time_bound
        ):
            raise ValueError("fit time bound")
        return checkpoint
    except (
        CanonicalJsonError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise ForecastLifecycleError(f"forecast checkpoint is malformed: {error}") from error


def _require_checkpoint(
    checkpoint: _Checkpoint,
    *,
    session: SessionIdentity,
    task: ForecastTask,
    expected_cursor: HistoryCursor,
    expected_task_identity: str | None,
    lineage_identity: str,
    config_digest: str,
    capabilities: tuple[str, ...],
) -> None:
    if checkpoint.session_value != session.value:
        raise ForecastLifecycleError("forecast checkpoint belongs to another session")
    if checkpoint.cursor != expected_cursor:
        raise ForecastLifecycleError("forecast checkpoint cursor does not match its key")
    if expected_task_identity is not None and checkpoint.task_identity != expected_task_identity:
        raise ForecastLifecycleError("forecast checkpoint task identity does not match")
    if checkpoint.lineage_identity != lineage_identity:
        raise ForecastLifecycleError("forecast checkpoint belongs to another task lineage")
    if checkpoint.config_digest != config_digest:
        raise ForecastLifecycleError("forecast checkpoint model configuration does not match")
    if checkpoint.capabilities != capabilities:
        raise ForecastLifecycleError("forecast checkpoint adapter capabilities do not match")
    if checkpoint.cursor.panel_identity != task.cursor.panel_identity:
        raise ForecastLifecycleError("forecast checkpoint belongs to another staged panel")


def _cursor_record(cursor: HistoryCursor) -> dict[str, object]:
    return {
        "panel_identity": cursor.panel_identity,
        "series_start": cursor.series_start,
        "series_stop": cursor.series_stop,
        "time_bound": cursor.time_bound,
    }


def _cursor_from_record(value: object) -> HistoryCursor:
    if not isinstance(value, dict) or set(value) != {
        "panel_identity",
        "series_start",
        "series_stop",
        "time_bound",
    }:
        raise ValueError("cursor")
    record = cast(dict[str, object], value)
    return HistoryCursor(
        cast(str, record["panel_identity"]),
        cast(int, record["series_start"]),
        cast(int, record["series_stop"]),
        cast(int, record["time_bound"]),
    )


def _require_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("digest")
    return value


__all__ = [
    "ForecastLifecycle",
    "ForecastLifecycleError",
    "ForecastLifecycleItem",
    "ForecastLifecycleResult",
]
