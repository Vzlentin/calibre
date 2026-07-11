"""Name the supported meanings of observations used by decision accounting."""

from enum import StrEnum


class ActualsSemantics(StrEnum):
    """Label the meaning of every observation used for realized cost."""

    DEMAND = "demand"
    CENSORED_SALES_SURROGATE = "censored_sales_surrogate"


__all__ = ["ActualsSemantics"]
