"""Parse M5 protocol constants as strict, immutable configuration data."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import cast

import yaml
from pydantic import ValidationError

from newcalibre.conformal import SPLIT_PER_STEP, method_config_schema, resolve_method
from newcalibre.conformal.manifest import MethodManifestError
from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.forecasting import SEASONAL_NAIVE_BACKEND, resolve_adapter
from newcalibre.forecasting.protocol import AdapterError
from newcalibre.reconcile import WLS_STRUCT, strategy_declaration
from newcalibre.reconcile.registry import ReconciliationRegistryError

_TOP_LEVEL_KEYS = frozenset(
    {"schema", "dataset", "protocol", "pipeline", "execution", "output_dir"}
)
_DATASET_KEYS = frozenset({"name", "phase", "data_dir", "inventory"})
_PROTOCOL_KEYS = frozenset({"horizon", "origin_count", "population"})
_PIPELINE_KEYS = frozenset({"model", "reconciliation", "conformal"})
_MODEL_KEYS = frozenset(
    {
        "backend",
        "censoring_aware",
        "m",
        "model_name",
        "quantile_levels",
        "scope",
    }
)
_RECONCILIATION_KEYS = frozenset({"strategy"})
_CONFORMAL_KEYS = frozenset(
    {
        "method",
        "coverage",
        "calibration_window",
        "partition",
        "upper_floor",
        "upper_cap",
    }
)
_EXECUTION_KEYS = frozenset(
    {"backend", "logical_shards", "workers", "numeric_threads_per_worker", "retries"}
)


class M5ConfigError(ValueError):
    """Report an incomplete, ambiguous, or inconsistent M5 configuration."""


class _DuplicateYAMLKey(yaml.YAMLError):
    """Retain one duplicate mapping key refused during YAML construction."""

    def __init__(self, key: object) -> None:
        super().__init__(key)
        self.key = key


class _UniqueKeyLoader(yaml.SafeLoader):
    """Construct safe YAML while refusing duplicate mappings recursively."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise _DuplicateYAMLKey(key)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class M5PopulationConfig:
    """Declare one validated bottom-population selection."""

    kind: str
    bottom_count: int | None
    salt: str | None


@dataclass(frozen=True, slots=True)
class M5ExecutionConfig:
    """Record fixed logical execution intent without dispatching work."""

    backend: str
    logical_shards: int
    workers: int
    numeric_threads_per_worker: int
    retries: int


@dataclass(frozen=True, slots=True, init=False)
class M5ProtocolConfig:
    """Carry one fully validated M5 protocol configuration."""

    dataset: str
    phase: str
    data_dir: Path
    inventory_path: Path
    horizon: int
    origin_count: int
    population: M5PopulationConfig
    model_scope: str
    reconciliation_strategy: str
    conformal_partition: str
    minimum_calibration_scores: int
    execution: M5ExecutionConfig
    output_dir: Path
    _model_config_json: bytes = field(repr=False)
    _conformal_config_json: bytes = field(repr=False)

    def __init__(self) -> None:
        raise TypeError("M5ProtocolConfig must be created with load_m5_config()")

    @property
    def model_config(self) -> dict[str, object]:
        """Return an isolated registered forecast-configuration snapshot."""
        return cast(dict[str, object], json.loads(self._model_config_json))

    @property
    def conformal_config(self) -> dict[str, object]:
        """Return an isolated registered conformal-configuration snapshot."""
        return cast(dict[str, object], json.loads(self._conformal_config_json))


def load_m5_config(path: Path) -> M5ProtocolConfig:
    """Load one strict M5 YAML declaration without executing the engine."""
    if not isinstance(path, Path):
        raise M5ConfigError("configuration path must be a pathlib.Path")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise M5ConfigError(f"M5 configuration is unreadable: {path}") from error
    return _load_m5_config_bytes(payload, path=path)


def _load_m5_config_bytes(payload: bytes, *, path: Path) -> M5ProtocolConfig:
    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=_UniqueKeyLoader)
    except _DuplicateYAMLKey as error:
        raise M5ConfigError(
            f"M5 configuration contains duplicate YAML key {error.key!r}"
        ) from error
    except (UnicodeError, yaml.YAMLError) as error:
        raise M5ConfigError(f"M5 configuration is unreadable: {path}") from error

    top = _exact_mapping(raw, keys=_TOP_LEVEL_KEYS, surface="top-level configuration")
    if top["schema"] != 1:
        raise M5ConfigError("configuration schema must equal 1")
    dataset, phase, data_dir, inventory_path = _parse_dataset(top["dataset"])
    horizon, origin_count, population = _parse_protocol(top["protocol"])
    model_scope, model_config, reconciler, partition, conformal, minimum = _parse_pipeline(
        top["pipeline"]
    )
    required_origins = minimum + 2 * (horizon - 1)
    if origin_count < required_origins:
        raise M5ConfigError(
            "protocol origin_count violates registered method readiness: "
            f"requires at least {required_origins}, found {origin_count}"
        )
    execution = _parse_execution(top["execution"])
    output_dir = _safe_relative_path(top["output_dir"], name="output_dir")

    config = object.__new__(M5ProtocolConfig)
    object.__setattr__(config, "dataset", dataset)
    object.__setattr__(config, "phase", phase)
    object.__setattr__(config, "data_dir", data_dir)
    object.__setattr__(config, "inventory_path", inventory_path)
    object.__setattr__(config, "horizon", horizon)
    object.__setattr__(config, "origin_count", origin_count)
    object.__setattr__(config, "population", population)
    object.__setattr__(config, "model_scope", model_scope)
    object.__setattr__(config, "reconciliation_strategy", reconciler)
    object.__setattr__(config, "conformal_partition", partition)
    object.__setattr__(config, "minimum_calibration_scores", minimum)
    object.__setattr__(config, "execution", execution)
    object.__setattr__(config, "output_dir", output_dir)
    object.__setattr__(config, "_model_config_json", _canonical_snapshot(model_config))
    object.__setattr__(config, "_conformal_config_json", _canonical_snapshot(conformal))
    return config


def _parse_dataset(value: object) -> tuple[str, str, Path, Path]:
    payload = _exact_mapping(value, keys=_DATASET_KEYS, surface="dataset")
    if payload["name"] != "m5":
        raise M5ConfigError("dataset name must equal 'm5'")
    if payload["phase"] != "evaluation":
        raise M5ConfigError("dataset phase must explicitly equal 'evaluation'")
    data_dir = _safe_relative_path(payload["data_dir"], name="dataset data_dir")
    inventory = _safe_relative_path(payload["inventory"], name="dataset inventory")
    if inventory.suffix != ".json":
        raise M5ConfigError("dataset inventory path must name one JSON file")
    return "m5", "evaluation", data_dir, inventory


def _parse_protocol(value: object) -> tuple[int, int, M5PopulationConfig]:
    payload = _exact_mapping(value, keys=_PROTOCOL_KEYS, surface="protocol")
    horizon = _positive_integer(payload["horizon"], name="protocol horizon")
    if horizon != 28:
        raise M5ConfigError("protocol horizon must equal 28")
    origins = _positive_integer(payload["origin_count"], name="protocol origin_count")
    return horizon, origins, _parse_population(payload["population"])


def _parse_population(value: object) -> M5PopulationConfig:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise M5ConfigError("protocol population must be a string-keyed mapping")
    payload = dict(cast(Mapping[str, object], value))
    kind = payload.get("kind")
    if kind == "full":
        _require_exact_keys(payload, keys=frozenset({"kind"}), surface="full population")
        return M5PopulationConfig("full", None, None)
    if kind == "digest_rank":
        _require_exact_keys(
            payload,
            keys=frozenset({"kind", "bottom_count", "salt"}),
            surface="digest_rank population",
        )
        count = _positive_integer(payload["bottom_count"], name="population bottom_count")
        salt = _string(payload["salt"], name="population salt")
        return M5PopulationConfig("digest_rank", count, salt)
    raise M5ConfigError("protocol population kind must equal 'full' or 'digest_rank'")


def _parse_pipeline(
    value: object,
) -> tuple[str, dict[str, object], str, str, dict[str, object], int]:
    payload = _exact_mapping(value, keys=_PIPELINE_KEYS, surface="pipeline")
    scope, model = _parse_model(payload["model"])
    reconciler = _parse_reconciliation(payload["reconciliation"])
    partition, conformal, minimum = _parse_conformal(payload["conformal"])
    return scope, model, reconciler, partition, conformal, minimum


def _parse_model(value: object) -> tuple[str, dict[str, object]]:
    payload = _exact_mapping(value, keys=_MODEL_KEYS, surface="pipeline model")
    if payload.pop("scope") != "global":
        raise M5ConfigError("pipeline model scope must equal 'global'")
    if payload["backend"] != SEASONAL_NAIVE_BACKEND:
        raise M5ConfigError(f"pipeline model backend must equal {SEASONAL_NAIVE_BACKEND!r}")
    if payload["m"] != 7:
        raise M5ConfigError("pipeline model season length m must equal 7")
    if payload["model_name"] != SEASONAL_NAIVE_BACKEND:
        raise M5ConfigError(f"pipeline model model_name must equal {SEASONAL_NAIVE_BACKEND!r}")
    if payload["censoring_aware"] is not False:
        raise M5ConfigError("pipeline model censoring_aware must be false")
    if payload["quantile_levels"] != []:
        raise M5ConfigError("pipeline model quantile_levels must be an empty list")
    try:
        resolve_adapter(payload)
    except AdapterError as error:
        raise M5ConfigError(f"invalid pipeline model: {error}") from error
    return "global", payload


def _parse_reconciliation(value: object) -> str:
    payload = _exact_mapping(
        value,
        keys=_RECONCILIATION_KEYS,
        surface="pipeline reconciliation",
    )
    strategy = payload["strategy"]
    if strategy != WLS_STRUCT:
        raise M5ConfigError(f"pipeline reconciliation strategy must equal {WLS_STRUCT!r}")
    try:
        strategy_declaration(WLS_STRUCT)
    except ReconciliationRegistryError as error:
        raise M5ConfigError(f"invalid pipeline reconciliation: {error}") from error
    return WLS_STRUCT


def _parse_conformal(value: object) -> tuple[str, dict[str, object], int]:
    payload = _exact_mapping(value, keys=_CONFORMAL_KEYS, surface="pipeline conformal")
    if payload.pop("method") != SPLIT_PER_STEP:
        raise M5ConfigError(f"pipeline conformal method must equal {SPLIT_PER_STEP!r}")
    if payload.pop("partition") != "series-horizon":
        raise M5ConfigError("pipeline conformal partition must equal 'series-horizon'")
    if payload["coverage"] != 0.9:
        raise M5ConfigError("pipeline conformal coverage must equal 0.9")
    if payload["calibration_window"] != 10:
        raise M5ConfigError("pipeline conformal calibration_window must equal 10")
    if payload["upper_floor"] is not None or payload["upper_cap"] is not None:
        raise M5ConfigError("pipeline conformal clamps must be explicit null")
    runtime_payload = {"method": SPLIT_PER_STEP, "partition_by": "series", **payload}
    schema = method_config_schema(SPLIT_PER_STEP)
    try:
        validated = schema.model_validate(
            {key: item for key, item in runtime_payload.items() if key != "method"},
            strict=True,
        )
        runtime = resolve_method(runtime_payload)
        minimum = runtime.manifest.minimum_calibration_scores(validated)
    except (MethodManifestError, ValidationError, ValueError) as error:
        raise M5ConfigError(f"invalid pipeline conformal declaration: {error}") from error
    normalized = cast(dict[str, object], validated.model_dump(mode="python"))
    normalized["method"] = SPLIT_PER_STEP
    return "series-horizon", normalized, minimum


def _parse_execution(value: object) -> M5ExecutionConfig:
    payload = _exact_mapping(value, keys=_EXECUTION_KEYS, surface="execution")
    if payload["backend"] != "ray":
        raise M5ConfigError("execution backend must equal 'ray'")
    shards = _positive_integer(payload["logical_shards"], name="execution logical_shards")
    workers = _positive_integer(payload["workers"], name="execution workers")
    threads = _positive_integer(
        payload["numeric_threads_per_worker"],
        name="execution numeric_threads_per_worker",
    )
    retries = _nonnegative_integer(payload["retries"], name="execution retries")
    if shards != 16:
        raise M5ConfigError("execution logical_shards must equal 16")
    if workers != 16:
        raise M5ConfigError("execution workers must equal 16")
    if threads != 1:
        raise M5ConfigError("execution numeric_threads_per_worker must equal 1")
    if retries != 0:
        raise M5ConfigError("execution retries must equal zero")
    return M5ExecutionConfig("ray", shards, workers, threads, retries)


def _exact_mapping(
    value: object,
    *,
    keys: frozenset[str],
    surface: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise M5ConfigError(f"{surface} must be a string-keyed mapping")
    payload = dict(cast(Mapping[str, object], value))
    _require_exact_keys(payload, keys=keys, surface=surface)
    return payload


def _require_exact_keys(
    payload: Mapping[str, object],
    *,
    keys: frozenset[str],
    surface: str,
) -> None:
    actual = set(payload)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise M5ConfigError(f"{surface} must contain exact keys: missing={missing} extra={extra}")


def _safe_relative_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str):
        raise M5ConfigError(f"{name} path must be a string")
    text = _string(value, name=name)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or pure.as_posix() != text
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in text
    ):
        raise M5ConfigError(f"{name} must be one normalized safe relative POSIX path")
    return Path(*pure.parts)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise M5ConfigError(f"{name} must be a non-empty trimmed string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise M5ConfigError(f"{name} must be valid UTF-8") from error
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result < 1:
        raise M5ConfigError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise M5ConfigError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise M5ConfigError(f"{name} must be a non-negative integer")
    return result


def _canonical_snapshot(value: Mapping[str, object]) -> bytes:
    try:
        return canonical_json_bytes(dict(value), path="M5 configuration snapshot")
    except CanonicalJsonError as error:
        raise M5ConfigError(str(error)) from error
