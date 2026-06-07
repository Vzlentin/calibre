from __future__ import annotations

import pandas as pd
import pytest

from calibre.reconciliation import (
    NixtlaReconciler,
    NoOpReconciler,
    ReconciliationContext,
    available_reconcilers,
    resolve_reconciler,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "ds": pd.to_datetime(["2024-01-07", "2024-01-07"]),
            "y_hat": [1.0, 2.0],
            "h": [1, 1],
        }
    )


def test_none_resolves_to_identity_passthrough() -> None:
    reconciler = resolve_reconciler("none")
    frame = _frame()
    out = reconciler(frame, None, ReconciliationContext())
    pd.testing.assert_frame_equal(out, frame)


def test_none_is_identity_even_with_non_none_hierarchy() -> None:
    """The no-op reconciler ignores the hierarchy and is a strict identity."""
    reconciler = resolve_reconciler("none")
    frame = _frame()
    hierarchy = pd.DataFrame({"unique_id": ["A", "B"], "store_id": ["S1", "S2"]})
    out = reconciler(frame, hierarchy, ReconciliationContext())
    pd.testing.assert_frame_equal(out, frame)


def test_resolved_noop_is_a_reconciler_instance() -> None:
    reconciler = resolve_reconciler("none")
    assert isinstance(reconciler, NoOpReconciler)


@pytest.mark.parametrize(
    "name",
    ["bottom_up", "ols", "wls_struct", "mint_shrink", "wls_var", "erm"],
)
def test_nixtla_strategies_resolve_to_adapter(name: str) -> None:
    reconciler = resolve_reconciler(name)
    assert isinstance(reconciler, NixtlaReconciler)
    assert reconciler.strategy == name


def test_unknown_strategy_lists_available_names() -> None:
    with pytest.raises(ValueError, match=r"Unknown reconciliation strategy: 'bogus'") as excinfo:
        resolve_reconciler("bogus")
    message = str(excinfo.value)
    for name in available_reconcilers():
        assert name in message


def test_resolution_is_case_insensitive_and_trimmed() -> None:
    """Mirror the dataset registry's ``str.strip().lower()`` key normalization."""
    assert isinstance(resolve_reconciler("  NONE "), NoOpReconciler)


def test_none_is_registered_and_available() -> None:
    assert "none" in available_reconcilers()


def test_available_reconcilers_contains_only_runnable_set() -> None:
    assert available_reconcilers() == [
        "bottom_up",
        "erm",
        "mint_shrink",
        "none",
        "ols",
        "wls_struct",
        "wls_var",
    ]


def test_mint_cov_is_not_on_generic_runnable_surface() -> None:
    with pytest.raises(ValueError, match="Unknown reconciliation strategy"):
        resolve_reconciler("mint_cov")
