from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

INVENTORY_POSITION = "inventory_position"
LEAD_TIME = "lead_time"
REVIEW_PERIOD = "review_period"

RS_PARAMETER_COLUMNS = [INVENTORY_POSITION, LEAD_TIME, REVIEW_PERIOD]


@dataclass(frozen=True, slots=True)
class RsPolicyParameters:
    unique_id: str
    inventory_position: float
    lead_time: int
    review_period: int

    def __post_init__(self) -> None:
        if not self.unique_id:
            raise ValueError("unique_id must be non-empty")
        if int(self.lead_time) < 0:
            raise ValueError("lead_time must be non-negative")
        if int(self.review_period) < 1:
            raise ValueError("review_period must be at least 1")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_rs_policy_parameters(
    params: pd.DataFrame | Iterable[RsPolicyParameters],
) -> pd.DataFrame:
    if isinstance(params, pd.DataFrame):
        params_frame = params.copy()
    else:
        params_frame = pd.DataFrame([param.as_dict() for param in params])

    required_columns = {"unique_id", *RS_PARAMETER_COLUMNS}
    missing_columns = required_columns - set(params_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required policy parameter columns: {sorted(missing_columns)}")

    duplicates = params_frame[params_frame["unique_id"].duplicated()]["unique_id"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate policy parameters for unique_id values: {duplicates}")

    normalized = params_frame[["unique_id", *RS_PARAMETER_COLUMNS]].copy()
    normalized[INVENTORY_POSITION] = normalized[INVENTORY_POSITION].astype(float)
    normalized[LEAD_TIME] = normalized[LEAD_TIME].astype(int)
    normalized[REVIEW_PERIOD] = normalized[REVIEW_PERIOD].astype(int)

    if (normalized[LEAD_TIME] < 0).any():
        raise ValueError("lead_time must be non-negative")
    if (normalized[REVIEW_PERIOD] < 1).any():
        raise ValueError("review_period must be at least 1")

    return normalized
