"""Tests for hierarchical conformal intervals."""

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
from calibre.reconciliation.summing import build_hierarchy_index, build_summing_matrix


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
        build_hierarchy_index(_hierarchy()),
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


def test_nixtla_failure_includes_fused_phase_context(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingReconciliation:
        def __init__(self, method: Any) -> None:
            del method

        def reconcile(self, **kwargs: Any) -> pd.DataFrame:
            del kwargs
            raise ValueError("nixtla boom")

    monkeypatch.setattr(hi, "make_nixtla_method", lambda strategy: object())
    monkeypatch.setattr(hi, "_make_reconciliation", _ExplodingReconciliation)
    phase = NixtlaHierarchicalIntervalPhase(
        HierarchicalIntervalOptions(coverage=0.9, strategy="bottom_up")
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "strategy='bottom_up'.*model_name='m'.*"
            "forecast_origin=Timestamp\\('2024-01-02 00:00:00'\\).*coverage=0.9"
        ),
    ):
        phase.apply(
            _forecast_frame(),
            build_hierarchy_index(_hierarchy()),
            HierarchicalIntervalContext(fitted_values=_fitted_values()),
        )


def test_missing_aggregate_fitted_key_fails_before_nixtla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_nixtla(monkeypatch)
    fitted = _fitted_values()
    fitted = fitted[~((fitted[UNIQUE_ID] == "dept=D") & (fitted[DS] == pd.Timestamp("2024-01-02")))]
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions())

    with pytest.raises(ValueError, match="dept=D.*2024-01-02.*m"):
        phase.apply(
            _forecast_frame(),
            build_hierarchy_index(_hierarchy()),
            HierarchicalIntervalContext(fitted),
        )

    assert _FakeReconciliation.calls == []


def test_duplicate_fitted_keys_fail_before_nixtla(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_nixtla(monkeypatch)
    fitted = pd.concat([_fitted_values(), _fitted_values().iloc[[0]]], ignore_index=True)
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions())

    with pytest.raises(ValueError, match="Duplicate fitted-value rows"):
        phase.apply(
            _forecast_frame(),
            build_hierarchy_index(_hierarchy()),
            HierarchicalIntervalContext(fitted),
        )

    assert _FakeReconciliation.calls == []


def test_multi_model_sidecars_do_not_mix_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_nixtla(monkeypatch)
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions())

    with pytest.raises(ValueError, match="model_name='m2'"):
        phase.apply(
            _forecast_frame(models=("m1", "m2")),
            build_hierarchy_index(_hierarchy()),
            HierarchicalIntervalContext(_fitted_values(models=("m1",))),
        )


def test_horizonless_sidecar_is_reused_for_multiple_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_nixtla(monkeypatch)
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions())

    out = phase.apply(
        _forecast_frame(horizon=2),
        build_hierarchy_index(_hierarchy()),
        HierarchicalIntervalContext(_fitted_values()),
    )

    assert set(out[H]) == {1, 2}
    assert len(_FakeReconciliation.calls) == 1
    assert H not in _FakeReconciliation.calls[0]["Y_df"].columns


# ---------------------------------------------------------------------------
# Sparse S_df emission for the sparse-capable roster (#168). Sparse value
# columns are the zero-copy precondition for hierarchicalforecast's
# `.sparse.to_coo()` -> csr path: a plain dense DataFrame silently densifies
# the full S transiently before re-sparsifying.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["bottom_up", "ols", "wls_struct", "wls_var"])
def test_sparse_roster_emits_sparse_s_df_columns(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    _patch_fake_nixtla(monkeypatch)
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions(strategy=strategy))

    phase.apply(
        _forecast_frame(),
        build_hierarchy_index(_hierarchy()),
        HierarchicalIntervalContext(fitted_values=_fitted_values()),
    )

    s_df = _FakeReconciliation.calls[0]["S_df"]
    value_columns = [column for column in s_df.columns if column != UNIQUE_ID]
    assert all(str(dtype).startswith("Sparse[float64") for dtype in s_df[value_columns].dtypes)
    # The zero-copy precondition hierarchicalforecast checks before to_coo.
    coo = s_df[value_columns].sparse.to_coo()
    assert coo.shape == (4, 2)


@pytest.mark.parametrize("strategy", ["mint_shrink", "erm"])
def test_dense_roster_keeps_dense_s_df_columns(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    _patch_fake_nixtla(monkeypatch)
    phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions(strategy=strategy))

    phase.apply(
        _forecast_frame(),
        build_hierarchy_index(_hierarchy()),
        HierarchicalIntervalContext(fitted_values=_fitted_values()),
    )

    s_df = _FakeReconciliation.calls[0]["S_df"]
    value_columns = [column for column in s_df.columns if column != UNIQUE_ID]
    assert all(str(dtype) == "float64" for dtype in s_df[value_columns].dtypes)


def test_sparse_s_df_values_and_row_order_match_dense_emission() -> None:
    index = build_hierarchy_index(_hierarchy())
    dense_df = hi._to_s_df(hi.summing_matrix_from_index(index))
    sparse_df = hi._to_s_df(hi.sparse_summing_matrix_from_index(index))

    assert list(sparse_df[UNIQUE_ID]) == list(dense_df[UNIQUE_ID])
    value_columns = [column for column in sparse_df.columns if column != UNIQUE_ID]
    np.testing.assert_array_equal(
        sparse_df[value_columns].to_numpy(dtype=np.float64),
        dense_df[value_columns].to_numpy(dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Fused completeness fast-path (#215). The count-vs-grid shortcut must stay
# byte-identical to the cartesian set-build it bypasses, on both branches.
# ---------------------------------------------------------------------------


def _slow_missing_fitted_keys(
    fitted_values: pd.DataFrame,
    summing: Any,
    model_name: str,
) -> list[tuple[str, str, str]]:
    """Reconstruct the pre-fast-path cartesian set-build for parity assertions."""
    fitted_values = fitted_values.copy()
    fitted_values[DS] = pd.to_datetime(fitted_values[DS]).astype("datetime64[ns]")
    observed_ds = sorted(fitted_values[DS].dropna().unique())
    observed = {
        (str(row.unique_id), pd.Timestamp(str(row.ds)))
        for row in fitted_values[[UNIQUE_ID, DS]].itertuples(index=False)
    }
    return [
        (node, str(pd.Timestamp(str(ds))), model_name)
        for ds in observed_ds
        for node in summing.node_labels
        if (node, pd.Timestamp(str(ds))) not in observed
    ]


def test_missing_fitted_keys_fast_path_matches_slow_path() -> None:
    summing = build_summing_matrix(_hierarchy())
    complete = _fitted_values()

    # Complete frame: the fast-path returns immediately and the slow path agrees.
    assert hi._missing_fitted_keys(complete.copy(), summing, "m") == []
    assert _slow_missing_fitted_keys(complete.copy(), summing, "m") == []

    # Incomplete frame: drop one (node, ds) so the cartesian path runs.
    incomplete = complete[
        ~((complete[UNIQUE_ID] == "dept=D") & (complete[DS] == pd.Timestamp("2024-01-02")))
    ]
    expected = [("dept=D", "2024-01-02 00:00:00", "m")]
    assert hi._missing_fitted_keys(incomplete.copy(), summing, "m") == expected
    assert _slow_missing_fitted_keys(incomplete.copy(), summing, "m") == expected


def test_to_nixtla_fitted_df_rejects_unknown_node_before_completeness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summing = build_summing_matrix(_hierarchy())
    fitted = pd.concat(
        [
            _fitted_values(),
            pd.DataFrame(
                [
                    {
                        UNIQUE_ID: "ghost",
                        DS: pd.Timestamp("2024-01-01"),
                        Y: 1.0,
                        MODEL_NAME: "m",
                        FITTED_Y_HAT: 0.75,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    fitted_by_model = {"m": fitted}

    # A sidecar id absent from the subset's node_labels must trip the unknown-node
    # guard, which runs before _missing_fitted_keys can count a now-mismatched grid.
    sentinel = {"called": False}

    def _fail_if_called(*args: Any, **kwargs: Any) -> list[tuple[str, str, str]]:
        sentinel["called"] = True
        raise AssertionError("_missing_fitted_keys ran before the unknown-node guard")

    monkeypatch.setattr(hi, "_missing_fitted_keys", _fail_if_called)

    with pytest.raises(ValueError, match="not present in the .*summing matrix.*ghost"):
        hi._to_nixtla_fitted_df(fitted_by_model, summing, "m")
    assert sentinel["called"] is False


def test_fused_phase_input_identical_across_completeness_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _run(force_slow: bool) -> pd.DataFrame:
        _patch_fake_nixtla(monkeypatch)
        if force_slow:
            # Skip the count shortcut so the cartesian set-build feeds reconcile.
            monkeypatch.setattr(
                hi,
                "_missing_fitted_keys",
                lambda fitted, summing, model_name: _slow_missing_fitted_keys(
                    fitted, summing, model_name
                ),
            )
        phase = NixtlaHierarchicalIntervalPhase(HierarchicalIntervalOptions())
        phase.apply(
            _forecast_frame(),
            build_hierarchy_index(_hierarchy()),
            HierarchicalIntervalContext(fitted_values=_fitted_values()),
        )
        return _FakeReconciliation.calls[0]["Y_df"]

    fast_y_df = _run(force_slow=False)
    slow_y_df = _run(force_slow=True)
    pd.testing.assert_frame_equal(fast_y_df, slow_y_df)
