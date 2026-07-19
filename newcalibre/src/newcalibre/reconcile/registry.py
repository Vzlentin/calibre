"""Resolve normalized reconciliation strategy names to fresh instances."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from weakref import WeakValueDictionary

from newcalibre.reconcile.protocol import Reconciler, ReconcilerDeclaration

ReconcilerFactory = Callable[[], Reconciler]


class ReconciliationRegistryError(ValueError):
    """Report an invalid strategy registration or normalized lookup."""


def _issued_instances() -> WeakValueDictionary[int, Reconciler]:
    return WeakValueDictionary()


@dataclass(slots=True)
class _Registration:
    declaration: ReconcilerDeclaration
    factory: ReconcilerFactory
    issued: WeakValueDictionary[int, Reconciler] = field(default_factory=_issued_instances)


class ReconcilerRegistry:
    """Map canonical strategy names to declarations and instance factories."""

    def __init__(self) -> None:
        self._registrations: dict[str, _Registration] = {}

    @property
    def available_strategies(self) -> tuple[str, ...]:
        """Return canonical strategy names in deterministic order."""
        return tuple(sorted(self._registrations))

    def register(
        self,
        declaration: ReconcilerDeclaration,
        factory: ReconcilerFactory,
    ) -> None:
        """Register one immutable declaration without replacing an existing name."""
        if not isinstance(declaration, ReconcilerDeclaration):
            raise ReconciliationRegistryError(
                "registered declaration must be a ReconcilerDeclaration"
            )
        name = _normalized_name(declaration.name)
        if name != declaration.name:
            raise ReconciliationRegistryError("registered strategy name must be canonical")
        if name in self._registrations:
            raise ReconciliationRegistryError(f"strategy {name!r} is already registered")
        if not callable(factory):
            raise ReconciliationRegistryError(f"factory for strategy {name!r} must be callable")
        self._registrations[name] = _Registration(declaration, factory)

    def declaration(self, name: str) -> ReconcilerDeclaration:
        """Return one normalized strategy's inspectable declaration."""
        return self._registration(name).declaration

    def resolve(self, name: str) -> Reconciler:
        """Construct and validate a fresh instance for one normalized name."""
        registration = self._registration(name)
        try:
            reconciler = registration.factory()
        except Exception as error:
            raise ReconciliationRegistryError(
                f"factory for strategy {registration.declaration.name!r} failed: {error}"
            ) from error
        if not isinstance(reconciler, Reconciler):
            raise ReconciliationRegistryError(
                f"factory for strategy {registration.declaration.name!r} "
                "must return a conforming Reconciler"
            )
        if reconciler.declaration != registration.declaration:
            raise ReconciliationRegistryError(
                f"factory for strategy {registration.declaration.name!r} "
                "returned a mismatched declaration"
            )
        identity = id(reconciler)
        if registration.issued.get(identity) is reconciler:
            raise ReconciliationRegistryError("factory must return a fresh reconciler instance")
        try:
            registration.issued[identity] = reconciler
        except TypeError as error:
            raise ReconciliationRegistryError(
                "factory must return a weak-reference-capable reconciler"
            ) from error
        return reconciler

    def _registration(self, name: str) -> _Registration:
        available = self._available_text()
        try:
            normalized = _normalized_name(name)
        except ReconciliationRegistryError as error:
            raise ReconciliationRegistryError(
                f"{error}; available strategies: {available}"
            ) from error
        if normalized == "mint_cov":
            raise ReconciliationRegistryError(
                "strategy 'mint_cov' is not supported because raw sample covariance is "
                "rank-deficient or ill-conditioned at target scales; use wls_var or "
                f"wls_struct; available strategies: {available}"
            )
        try:
            return self._registrations[normalized]
        except KeyError as error:
            raise ReconciliationRegistryError(
                f"unknown strategy {normalized!r}; available strategies: {available}"
            ) from error

    def _available_text(self) -> str:
        return ", ".join(self.available_strategies) or "<none>"


def _normalized_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReconciliationRegistryError("strategy name must be a non-empty string")
    normalized = value.strip().casefold()
    try:
        normalized.encode("utf-8")
    except UnicodeError as error:
        raise ReconciliationRegistryError("strategy name must be valid UTF-8") from error
    return normalized
