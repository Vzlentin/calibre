from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from calibre.contracts.forecast_frame import (
    CONFORMAL_MODE,
    H,
    interval_column_names,
    quantile_column,
)
from calibre.order._helpers import _validate_interval_columns
from calibre.order.types import (
    LEAD_TIME,
    REORDER_POINT,
    REVIEW_PERIOD,
    CostStruct,
)


def _protection_period(frame: pd.DataFrame) -> int:
    return int(frame[LEAD_TIME].iloc[0]) + int(frame[REVIEW_PERIOD].iloc[0])


def _validate_protection_horizons(ordered: pd.DataFrame, protection_period: int) -> pd.Series:
    horizons = ordered[H].astype(int)
    duplicate_horizons = sorted(horizons[horizons.duplicated()].unique().tolist())
    if duplicate_horizons:
        raise ValueError(f"Duplicate horizons for decision group: {duplicate_horizons}")
    required_horizons = set(range(1, protection_period + 1))
    available_horizons = set(horizons.tolist())

    if not required_horizons.issubset(available_horizons):
        missing_horizons = sorted(required_horizons - available_horizons)
        max_horizon = int(horizons.max())
        if max_horizon >= protection_period and missing_horizons:
            raise ValueError(
                f"Missing horizons within protection period {protection_period}: "
                f"{missing_horizons}"
            )
        raise ValueError(
            f"Protection period {protection_period} exceeds available horizon {max_horizon}"
        )
    return horizons


@dataclass(frozen=True, slots=True)
class UpperBoundRule:
    coverage: float = 0.9
    quantile: float | None = None

    def __call__(self, frame: pd.DataFrame, costs: CostStruct) -> float:
        del costs
        ordered = frame.sort_values(H)
        protection_period = _protection_period(ordered)
        horizons = _validate_protection_horizons(ordered, protection_period)
        cumulative_mode = (
            self.quantile is None
            and CONFORMAL_MODE in ordered.columns
            and (ordered[CONFORMAL_MODE] == "cumulative").all()
        )

        if self.quantile is not None:
            target_col = quantile_column(self.quantile)
            if target_col not in ordered.columns:
                raise ValueError(
                    f"Missing quantile column for quantile={self.quantile}: {target_col!r}"
                )
            return float(ordered.loc[horizons <= protection_period, target_col].sum())

        _, target_col = _validate_interval_columns(ordered, self.coverage)
        if cumulative_mode:
            cumulative_rows = ordered.loc[horizons == protection_period, target_col]
            if cumulative_rows.empty:
                raise ValueError(
                    f"Cumulative conformal frame missing terminal h={protection_period} row"
                )
            return float(cumulative_rows.iloc[0])
        return float(ordered.loc[horizons <= protection_period, target_col].sum())


@dataclass(frozen=True, slots=True)
class CumulativeBoundRule:
    coverage: float = 0.9

    def __call__(self, frame: pd.DataFrame, costs: CostStruct) -> float:
        del costs
        ordered = frame.sort_values(H)
        protection_period = _protection_period(ordered)
        horizons = _validate_protection_horizons(ordered, protection_period)
        _, upper_col = _validate_interval_columns(ordered, self.coverage)
        terminal = ordered.loc[horizons == protection_period, upper_col]
        if terminal.empty:
            raise ValueError(
                f"Cumulative conformal frame missing terminal h={protection_period} row"
            )
        return float(terminal.iloc[0])


@dataclass(frozen=True, slots=True)
class QuantileInterpolationRule:
    coverage: float = 0.9
    period: int = 1

    def __call__(self, frame: pd.DataFrame, costs: CostStruct) -> float:
        lower_col, upper_col = interval_column_names(self.coverage)
        missing_columns = [
            column for column in (lower_col, upper_col) if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing conformal interval columns for coverage {self.coverage}: "
                f"{missing_columns}"
            )
        period_frame = frame.loc[frame[H] == self.period]
        if period_frame.empty:
            raise ValueError(f"No rows found for period h={self.period}")
        row = period_frame.iloc[0]
        lo = float(row[lower_col])
        hi = float(row[upper_col])
        return lo + costs.critical_ratio * (hi - lo)


@dataclass(frozen=True, slots=True)
class RSArithmetic:
    def __call__(self, target: float, ip: float, *, reorder_point=None) -> float:
        del reorder_point
        return max(float(target) - float(ip), 0.0)


@dataclass(frozen=True, slots=True)
class RSSArithmetic:
    def __call__(self, target: float, ip: float, *, reorder_point=None) -> float:
        if reorder_point is None:
            raise ValueError("RSSArithmetic requires reorder_point")
        if float(ip) >= float(reorder_point):
            return 0.0
        return max(float(target) - float(ip), 0.0)


def reorder_point_from_frame(frame: pd.DataFrame) -> float | None:
    if REORDER_POINT not in frame.columns:
        return None
    return float(frame[REORDER_POINT].iloc[0])
