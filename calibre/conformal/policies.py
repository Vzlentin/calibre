"""Abstract base class for online conformal controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class OnlineConformalController(ABC):
    """Protocol for online conformal controllers."""

    @abstractmethod
    def predict_interval(self, point_forecast):
        """Construct an interval prediction from a point forecast."""

    @abstractmethod
    def observe(self, *args, **kwargs):
        """Consume realized data and advance the controller state."""

    @abstractmethod
    def update(self, error):
        """Update control inputs from observed errors."""

    @abstractmethod
    def get_diagnostics(self) -> dict[str, Any]:
        """Return serializable controller diagnostics."""
