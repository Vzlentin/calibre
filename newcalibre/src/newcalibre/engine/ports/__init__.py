"""Define the panel-loading boundary used by the engine."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from newcalibre.domain import Panel


@runtime_checkable
class PanelSource(Protocol):
    """Load the immutable panel that defines a run."""

    def load(self) -> Panel:
        """Return the run's panel."""
        ...


__all__ = ["PanelSource"]
