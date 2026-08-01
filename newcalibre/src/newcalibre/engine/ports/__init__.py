"""Define the panel-loading and dispatch boundaries used by the engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from newcalibre.domain import Panel

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


@runtime_checkable
class PanelSource(Protocol):
    """Load the immutable panel that defines a run."""

    def load(self) -> Panel:
        """Return the run's panel."""
        ...


@runtime_checkable
class DispatchBackend(Protocol):
    """Place work without changing its values or deterministic order."""

    def map(
        self,
        function: Callable[[_Input], _Output],
        items: Sequence[_Input],
    ) -> tuple[_Output, ...]:
        """Apply ``function`` in input order and return results in that order."""
        ...


__all__ = ["DispatchBackend", "PanelSource"]
