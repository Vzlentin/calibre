"""Define immutable evidence carried by one ordering decision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from newcalibre.domain.descriptor import GuaranteeDescriptor


class DecisionEvidenceError(ValueError):
    """Report malformed ordering-decision evidence."""


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DecisionEvidenceError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise DecisionEvidenceError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise DecisionEvidenceError(f"{name} must be a finite real number")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True, slots=True)
class AppliedBinding:
    """Record one named configuration value and whether it modified a decision."""

    name: str
    value: float
    bound: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DecisionEvidenceError("binding name must be a non-empty string")
        try:
            self.name.encode("utf-8")
        except UnicodeError as error:
            raise DecisionEvidenceError("binding name must be valid UTF-8") from error
        if not isinstance(self.bound, bool):
            raise DecisionEvidenceError("binding bound must be a boolean")
        object.__setattr__(self, "value", _finite_real(self.value, name="binding value"))


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    """Carry target arithmetic, provenance, and bindings for one decision."""

    raw_target: float
    target: float
    source_columns: tuple[str, ...]
    source_descriptor: GuaranteeDescriptor
    effective_descriptor: GuaranteeDescriptor
    bindings: tuple[AppliedBinding, ...] = ()
    reorder_point: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_target", _finite_real(self.raw_target, name="raw target"))
        object.__setattr__(self, "target", _finite_real(self.target, name="target"))
        columns = tuple(self.source_columns)
        if not columns or any(not isinstance(column, str) or not column for column in columns):
            raise DecisionEvidenceError("source columns must be non-empty strings")
        if len(set(columns)) != len(columns):
            raise DecisionEvidenceError("source columns must be unique")
        if not isinstance(self.source_descriptor, GuaranteeDescriptor):
            raise DecisionEvidenceError("source descriptor must be a GuaranteeDescriptor")
        if not isinstance(self.effective_descriptor, GuaranteeDescriptor):
            raise DecisionEvidenceError("effective descriptor must be a GuaranteeDescriptor")
        bindings = tuple(self.bindings)
        if any(not isinstance(binding, AppliedBinding) for binding in bindings):
            raise DecisionEvidenceError("bindings must contain AppliedBinding values")
        bindings = tuple(
            binding
            if type(binding) is AppliedBinding
            else AppliedBinding(name=binding.name, value=binding.value, bound=binding.bound)
            for binding in bindings
        )
        if self.reorder_point is not None:
            object.__setattr__(
                self,
                "reorder_point",
                _finite_real(self.reorder_point, name="reorder point"),
            )
        object.__setattr__(self, "source_columns", columns)
        object.__setattr__(self, "bindings", bindings)


__all__ = ["AppliedBinding", "DecisionEvidence", "DecisionEvidenceError"]
