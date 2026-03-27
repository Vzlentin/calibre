from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

INVENTORY_POSITION = "inventory_position"
LEAD_TIME = "lead_time"
REVIEW_PERIOD = "review_period"
REORDER_POINT = "reorder_point"
UNDERAGE_COST = "underage_cost"
OVERAGE_COST = "overage_cost"

RS_PARAMETER_COLUMNS = [INVENTORY_POSITION, LEAD_TIME, REVIEW_PERIOD]
RSS_PARAMETER_COLUMNS = [INVENTORY_POSITION, REORDER_POINT, LEAD_TIME, REVIEW_PERIOD]
NEWSVENDOR_PARAMETER_COLUMNS = [UNDERAGE_COST, OVERAGE_COST, INVENTORY_POSITION]


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


@dataclass(frozen=True, slots=True)
class RssPolicyParameters:
    unique_id: str
    inventory_position: float
    reorder_point: float
    lead_time: int
    review_period: int

    def __post_init__(self) -> None:
        if not self.unique_id:
            raise ValueError("unique_id must be non-empty")
        if int(self.lead_time) < 0:
            raise ValueError("lead_time must be non-negative")
        if int(self.review_period) < 1:
            raise ValueError("review_period must be at least 1")
        if float(self.reorder_point) < 0:
            raise ValueError("reorder_point must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NewsvendorPolicyParameters:
    unique_id: str
    underage_cost: float
    overage_cost: float
    inventory_position: float

    def __post_init__(self) -> None:
        if not self.unique_id:
            raise ValueError("unique_id must be non-empty")
        if float(self.underage_cost) <= 0:
            raise ValueError("underage_cost must be positive")
        if float(self.overage_cost) <= 0:
            raise ValueError("overage_cost must be positive")

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


def normalize_rss_policy_parameters(
    params: pd.DataFrame | Iterable[RssPolicyParameters],
) -> pd.DataFrame:
    if isinstance(params, pd.DataFrame):
        params_frame = params.copy()
    else:
        params_frame = pd.DataFrame([param.as_dict() for param in params])

    required_columns = {"unique_id", *RSS_PARAMETER_COLUMNS}
    missing_columns = required_columns - set(params_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required policy parameter columns: {sorted(missing_columns)}")

    duplicates = params_frame[params_frame["unique_id"].duplicated()]["unique_id"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate policy parameters for unique_id values: {duplicates}")

    normalized = params_frame[["unique_id", *RSS_PARAMETER_COLUMNS]].copy()
    normalized[INVENTORY_POSITION] = normalized[INVENTORY_POSITION].astype(float)
    normalized[REORDER_POINT] = normalized[REORDER_POINT].astype(float)
    normalized[LEAD_TIME] = normalized[LEAD_TIME].astype(int)
    normalized[REVIEW_PERIOD] = normalized[REVIEW_PERIOD].astype(int)

    if (normalized[LEAD_TIME] < 0).any():
        raise ValueError("lead_time must be non-negative")
    if (normalized[REVIEW_PERIOD] < 1).any():
        raise ValueError("review_period must be at least 1")
    if (normalized[REORDER_POINT] < 0).any():
        raise ValueError("reorder_point must be non-negative")

    return normalized


def normalize_newsvendor_policy_parameters(
    params: pd.DataFrame | Iterable[NewsvendorPolicyParameters],
) -> pd.DataFrame:
    if isinstance(params, pd.DataFrame):
        params_frame = params.copy()
    else:
        params_frame = pd.DataFrame([param.as_dict() for param in params])

    required_columns = {"unique_id", *NEWSVENDOR_PARAMETER_COLUMNS}
    missing_columns = required_columns - set(params_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required policy parameter columns: {sorted(missing_columns)}")

    duplicates = params_frame[params_frame["unique_id"].duplicated()]["unique_id"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate policy parameters for unique_id values: {duplicates}")

    normalized = params_frame[["unique_id", *NEWSVENDOR_PARAMETER_COLUMNS]].copy()
    normalized[UNDERAGE_COST] = normalized[UNDERAGE_COST].astype(float)
    normalized[OVERAGE_COST] = normalized[OVERAGE_COST].astype(float)
    normalized[INVENTORY_POSITION] = normalized[INVENTORY_POSITION].astype(float)

    if (normalized[UNDERAGE_COST] <= 0).any():
        raise ValueError("underage_cost must be positive")
    if (normalized[OVERAGE_COST] <= 0).any():
        raise ValueError("overage_cost must be positive")

    return normalized
