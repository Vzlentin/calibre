"""Resolve explicitly named forecasting backends without a default."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from newcalibre.forecasting.protocol import (
    AdapterCapabilityError,
    AdapterExecutionMode,
    ForecastAdapter,
)

AdapterFactory = Callable[[Mapping[str, object]], ForecastAdapter]


class AdapterRegistryError(ValueError):
    """Report an invalid backend registration or lookup."""


class AdapterRegistry:
    """Map explicit backend identifiers to adapter factories."""

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    @property
    def available_backends(self) -> tuple[str, ...]:
        """Return registered backend identifiers in deterministic order."""
        return tuple(sorted(self._factories))

    def register(self, backend: str, factory: AdapterFactory) -> None:
        """Register one backend and reject identifier replacement."""
        if not isinstance(backend, str) or not backend or backend != backend.strip():
            raise AdapterRegistryError("backend identifier must be a non-empty trimmed string")
        if backend in self._factories:
            raise AdapterRegistryError(f"backend {backend!r} is already registered")
        if not callable(factory):
            raise AdapterRegistryError(f"factory for backend {backend!r} must be callable")
        self._factories[backend] = factory

    def resolve(self, model_config: Mapping[str, object]) -> ForecastAdapter:
        """Construct and validate the explicitly configured backend adapter."""
        if not isinstance(model_config, Mapping):
            raise AdapterRegistryError("model configuration must be a mapping")

        available = self._available_text()
        if "backend" not in model_config:
            raise AdapterRegistryError(
                "model configuration requires an explicit 'backend'; "
                f"available backends: {available}"
            )
        backend = model_config["backend"]
        if not isinstance(backend, str) or not backend:
            raise AdapterRegistryError(
                "model configuration 'backend' must be a non-empty string; "
                f"available backends: {available}"
            )
        try:
            factory = self._factories[backend]
        except KeyError as error:
            raise AdapterRegistryError(
                f"unknown backend {backend!r}; available backends: {available}"
            ) from error
        adapter = factory(model_config)
        if not isinstance(getattr(adapter, "execution_mode", None), AdapterExecutionMode):
            raise AdapterRegistryError(
                f"backend {backend!r} must declare one valid adapter execution mode"
            )
        unsupported = adapter.requested_capabilities - adapter.capabilities
        if unsupported:
            capability_text = ", ".join(sorted(capability.value for capability in unsupported))
            raise AdapterCapabilityError(
                f"backend {backend!r} does not declare requested capabilities: {capability_text}"
            )
        return adapter

    def _available_text(self) -> str:
        return ", ".join(self.available_backends) or "<none>"
