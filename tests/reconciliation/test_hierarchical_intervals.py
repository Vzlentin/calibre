from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from calibre.core.forecast_frame import (
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    DS,
    FITTED_Y_HAT,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)
from calibre.reconciliation import HierarchicalIntervalContext, HierarchicalIntervalOptions
from calibre.reconciliation import hierarchical_intervals as hi
from calibre.reconciliation.hierarchical_intervals import NixtlaHierarchicalIntervalPhase
from calibre.reconciliation.summing import build_summing_matrix


class _FakeReconciliation:
    calls: list[dict[str, Any]] = []

    def __init__(self, method: Any) -> None:
        del method

    def reconcile(
        self,
        *,
        Y_hat_df: pd.DataFrame,
        S_df: pd.DataFrame,
        tags: dict[str, np.ndarray],
        Y_df: pd.DataFrame,
        level: list[float],
        intervals_method: str,
        seed: int,
        is_balanced: bool,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "Y_hat_df": Y_hat_df.copy(),
                "S_df": S_df.copy(),
                "tags": tags,
                "Y_df": Y_df.copy(),
                "level": level,
                "intervals_method": intervals_method,
                "seed": seed,
                "is_balanced": is_balanced,
            }
        )
        model_col = hi._NIXTLA_MODEL_COL
        rec_col = f"{model_col}/Fake"
        lower_col = f"{rec_col}-lo-{level[0]:.12g}"
        upper_col = f"{rec_col}-hi-{level[0]:.12g}"
        bottom_ids = [column for column in S_df.columns if column != UNIQUE_ID]
        s_matrix = S_df.set_index(UNIQUE_ID)[bottom_ids].to_numpy(dtype=np.float64)
        rows = []
        for ds, ds_group in Y_hat_df.groupby(DS, sort=False):
            values = ds_group.set_index(UNIQUE_ID)[model_col]
            bottom = values.reindex(bottom_ids).to_numpy(dtype=np.float64)
            coherent = s_matrix @ bottom
            for uid, y_hat in zip(S_df[UNIQUE_ID].astype(str), coherent, strict=True):
                rows.append(
                    {
                        UNIQUE_ID: uid,
                        DS: ds,
                        model_col: values.get(uid, np.nan),
                        rec_col: y_hat,
                        lower_col: y_hat - 1.0,
                        upper_col: y_hat + 1.0,
                    }
                )
        return pd.DataFrame(rows)


def _patch_fake_nixtla(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeReconciliation.calls = []
    monkeypatch.setattr(hi, "make_nixtla_method", lambda strategy: object())
    monkeypatch.setattr(hi, "_make_reconciliation", _FakeReconciliation)


def _hierarchy() -> pd.DataFrame:
    return pd.DataFrame({UNIQUE_ID: ["a", "b"], "dept": ["D", "D"]})


def _forecast_frame(*, models: tuple[str, ...] = ("m",), horizon: int = 1) -> pd.DataFrame:
    rows = []
    for model_name in models:
        for h in range(1, horizon + 1):
            ds = pd.Timestamp("2024-01-02") + pd.Timedelta(days=h)
            rows.extend(
                [
                    (model_name, "a", ds, h, 2.0 + h),
                    (model_name, "b", ds, h, 3.0 + h),
                    (model_name, "dept=D", ds, h, 20.0 + h),
                    (model_name, "__total__", ds, h, 30.0 + h),
                ]
            )
    return pd.DataFrame(
        {
            MODEL_NAME: pd.Series([row[0] for row in rows], dtype="object"),
            UNIQUE_ID: pd.Series([row[1] for row in rows], dtype="object"),
            DS: pd.to_datetime([row[2] for row in rows]),
            Y: np.array([np.nan] * len(rows), dtype=np.float64),
            Y_HAT: np.array([row[4] for row in rows], dtype=np.float64),
            H: np.array([row[3] for row in rows], dtype=np.int64),
            FORECAST_ORIGIN: pd.to_datetime(["2024-01-02"] * len(rows)),
        }
    )


def _fitted_values(*, models: tuple[str, ...] = ("m",)) -> pd.DataFrame:
    rows = []
    values_by_ds = [
        ("2024-01-01", {"a": 1.0, "b": 2.0, "dept=D": 3.0, "__total__": 3.0}),
        ("2024-01-02", {"a": 2.0, "b": 3.0, "dept=D": 5.0, "__total__": 5.0}),
    ]
    for model_name in models:
        for ds, values in values_by_ds:
            for uid, y in values.items():
                rows.append(
                    {
                        UNIQUE_ID: uid,
                        DS: pd.Timestamp(ds),
                        Y: y,
                        MODEL_NAME: model_name,
                        FITTED_Y_HAT: y - 0.25,
                    }
                )
    return pd.DataFrame(rows)


def test_valid_sidecar_produces_calibre_interval_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_nixtla(monkeypatch)
    phase = NixtlaHierarchicalIntervalPhase(
        HierarchicalIntervalOptions(coverage=0.9, strategy="bottom_up")
    )

    out = phase.apply(
        _forecast_frame(),
        _hierarchy(),
        HierarchicalIntervalContext(fitted_values=_fitted_values()),
    )

    lower_col, upper_col = interval_column_names(0.9)
    assert lower_col in out.columns
    assert upper_col in out.columns
    assert out[CONFORMAL_METHOD].eq("nixtla_conformal").all()
    assert out[CONFORMAL_MODE].eq("hierarchical_marginal").all()
    summing = build_summing_matrix(_hierarchy())
    values = out.set_index(UNIQUE_ID)[Y_HAT].reindex(summing.node_labels).to_numpy(np.float64)
    np.testing.assert_allclose(values, summing.S @ values[: summing.n_bottom])


def test_missing_aggregate_fitted_key_fails_before_nixtla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_nixtla(monkeypatch)
    fitted = _fitted_values()
    fitted = fitted[~((fitted[UNIQUE_ID] == "dept=D") & (fitted[DS] == pd.Timestamp("2024-01-02")))]
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions())

    with pytest.raises(ValueError, match="dept=D.*2024-01-02.*m"):
        phase.apply(_forecast_frame(), _hierarchy(), HierarchicalIntervalContext(fitted))

    assert _FakeReconciliation.calls == []


def test_duplicate_fitted_keys_fail_before_nixtla(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_nixtla(monkeypatch)
    fitted = pd.concat([_fitted_values(), _fitted_values().iloc[[0]]], ignore_index=True)
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions())

    with pytest.raises(ValueError, match="Duplicate fitted-value rows"):
        phase.apply(_forecast_frame(), _hierarchy(), HierarchicalIntervalContext(fitted))

    assert _FakeReconciliation.calls == []


def test_multi_model_sidecars_do_not_mix_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_nixtla(monkeypatch)
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions())

    with pytest.raises(ValueError, match="model_name='m2'"):
        phase.apply(
            _forecast_frame(models=("m1", "m2")),
            _hierarchy(),
            HierarchicalIntervalContext(_fitted_values(models=("m1",))),
        )


def test_horizonless_sidecar_is_reused_for_multiple_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_nixtla(monkeypatch)
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions())

    out = phase.apply(
        _forecast_frame(horizon=2),
        _hierarchy(),
        HierarchicalIntervalContext(_fitted_values()),
    )

    assert set(out[H]) == {1, 2}
    assert len(_FakeReconciliation.calls) == 1
    assert H not in _FakeReconciliation.calls[0]["Y_df"].columns
