"""Resolve explicitly named conformal methods without a default."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel, ValidationError

from newcalibre.conformal.manifest import MethodManifest
from newcalibre.conformal.runtime import ConformalRuntime
from newcalibre.conformal.state import StateCodecError, validate_state_blob

RuntimeFactory = Callable[[BaseModel, Mapping[str, bytes]], ConformalRuntime]


class ConformalRegistryError(ValueError):
    """Report an invalid method registration, configuration, or restoration."""


@dataclass(slots=True)
class _Registration:
    manifest: MethodManifest
    config_schema: type[BaseModel]
    factory: RuntimeFactory
    last_runtime: ConformalRuntime


class ConformalRegistry:
    """Map explicit method identifiers to manifests, schemas, and factories."""

    def __init__(self) -> None:
        """Create an empty registry with no implicit method."""
        self._registrations: dict[str, _Registration] = {}

    @property
    def available_methods(self) -> tuple[str, ...]:
        """Return registered method identifiers in deterministic order."""
        return tuple(sorted(self._registrations))

    def register(
        self,
        method_name: str,
        manifest: MethodManifest,
        config_schema: type[BaseModel],
        factory: RuntimeFactory,
    ) -> None:
        """Atomically register and probe one complete conformal extension."""
        _require_method_name(method_name)
        if method_name in self._registrations:
            raise ConformalRegistryError(f"method {method_name!r} is already registered")
        if not isinstance(manifest, MethodManifest):
            raise ConformalRegistryError("registered manifest must be a MethodManifest")
        if manifest.name != method_name:
            raise ConformalRegistryError(
                f"manifest name {manifest.name!r} must equal registry key {method_name!r}"
            )
        _validate_config_schema(config_schema)
        if not callable(factory):
            raise ConformalRegistryError(f"factory for method {method_name!r} must be callable")
        try:
            default_config = config_schema.model_validate({}, strict=True)
        except ValidationError as error:
            raise ConformalRegistryError(
                "conformal config schemas must define valid runtime defaults for every field"
            ) from error

        empty_states: Mapping[str, bytes] = MappingProxyType({})
        first = _call_factory(
            factory,
            default_config,
            empty_states,
            method_name=method_name,
        )
        _validate_runtime(
            first,
            manifest=manifest,
            config_schema=config_schema,
            expected_config=default_config,
        )
        second = _call_factory(
            factory,
            default_config,
            empty_states,
            method_name=method_name,
        )
        _validate_runtime(
            second,
            manifest=manifest,
            config_schema=config_schema,
            expected_config=default_config,
        )
        if first is second:
            raise ConformalRegistryError("factory must return a fresh runtime instance")

        self._registrations[method_name] = _Registration(
            manifest=manifest,
            config_schema=config_schema,
            factory=factory,
            last_runtime=second,
        )

    def config_schema(self, method_name: str) -> type[BaseModel]:
        """Return the exposed configuration schema for an explicit method."""
        return self._registration_for(method_name).config_schema

    def resolve(
        self,
        configuration: Mapping[str, object],
        *,
        states: Mapping[str, bytes] | None = None,
    ) -> ConformalRuntime:
        """Validate configuration and state before constructing a fresh runtime."""
        if not isinstance(configuration, Mapping):
            raise ConformalRegistryError("conformal configuration must be a mapping")
        available = self._available_text()
        if "method" not in configuration:
            raise ConformalRegistryError(
                "conformal configuration requires an explicit 'method'; "
                f"available methods: {available}"
            )
        method_name = configuration["method"]
        if not isinstance(method_name, str) or not method_name:
            raise ConformalRegistryError(
                "conformal configuration 'method' must be a non-empty string; "
                f"available methods: {available}"
            )
        try:
            registration = self._registrations[method_name]
        except KeyError as error:
            raise ConformalRegistryError(
                f"unknown method {method_name!r}; available methods: {available}"
            ) from error

        payload = dict(configuration)
        del payload["method"]
        try:
            config = registration.config_schema.model_validate(payload, strict=True)
        except ValidationError as error:
            raise ConformalRegistryError(
                f"invalid configuration for method {method_name!r}: {error}"
            ) from error
        state_snapshot = _validated_states(states, manifest=registration.manifest)
        runtime = _call_factory(
            registration.factory,
            config,
            MappingProxyType(state_snapshot),
            method_name=method_name,
        )
        _validate_runtime(
            runtime,
            manifest=registration.manifest,
            config_schema=registration.config_schema,
            expected_config=config,
        )
        if runtime is registration.last_runtime:
            raise ConformalRegistryError("factory must return a fresh runtime instance")
        registration.last_runtime = runtime
        return runtime

    def _registration_for(self, method_name: str) -> _Registration:
        if not isinstance(method_name, str) or not method_name:
            raise ConformalRegistryError("method identifier must be a non-empty string")
        try:
            return self._registrations[method_name]
        except KeyError as error:
            raise ConformalRegistryError(
                f"unknown method {method_name!r}; available methods: {self._available_text()}"
            ) from error

    def _available_text(self) -> str:
        return ", ".join(self.available_methods) or "<none>"


def _validate_config_schema(config_schema: object) -> None:
    if not isinstance(config_schema, type) or not issubclass(config_schema, BaseModel):
        raise ConformalRegistryError("config schema must be a Pydantic BaseModel class")
    if config_schema.model_config.get("frozen") is not True:
        raise ConformalRegistryError("config schema must be frozen")
    if config_schema.model_config.get("extra") != "forbid":
        raise ConformalRegistryError("config schema must set extra='forbid'")


def _call_factory(
    factory: RuntimeFactory,
    config: BaseModel,
    states: Mapping[str, bytes],
    *,
    method_name: str,
) -> ConformalRuntime:
    try:
        return factory(config, states)
    except Exception as error:
        raise ConformalRegistryError(
            f"factory for method {method_name!r} failed: {error}"
        ) from error


def _validate_runtime(
    runtime: object,
    *,
    manifest: MethodManifest,
    config_schema: type[BaseModel],
    expected_config: BaseModel,
) -> None:
    if not isinstance(runtime, ConformalRuntime):
        raise ConformalRegistryError("factory must return a conforming ConformalRuntime")
    if runtime.manifest != manifest:
        raise ConformalRegistryError("runtime manifest must equal the registered manifest")
    if type(runtime.config) is not config_schema:
        raise ConformalRegistryError(
            "runtime configuration schema must equal the exposed config schema"
        )
    if runtime.config != expected_config:
        raise ConformalRegistryError(
            "runtime configuration must equal the registry-validated configuration"
        )


def _validated_states(
    states: object,
    *,
    manifest: MethodManifest,
) -> dict[str, bytes]:
    if states is None:
        return {}
    if not isinstance(states, Mapping):
        raise ConformalRegistryError("restored conformal states must be a mapping")
    snapshot = dict(states)
    for label, state in snapshot.items():
        if not isinstance(label, str):
            raise ConformalRegistryError("restored state labels must be strings")
        if not isinstance(state, bytes):
            raise ConformalRegistryError("restored state values must be immutable bytes")
        try:
            validate_state_blob(
                state,
                method_name=manifest.name,
                schema_version=manifest.state_schema_version,
                expected_label=label,
            )
        except StateCodecError as error:
            raise ConformalRegistryError(f"invalid state for label {label!r}: {error}") from error
    return snapshot


def _require_method_name(value: object) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConformalRegistryError("method identifier must be a non-empty trimmed string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ConformalRegistryError("method identifier must be valid UTF-8") from error
