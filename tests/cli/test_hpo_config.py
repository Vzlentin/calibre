"""Tests for the ``hpo`` config block and its search-space guardrail."""

from __future__ import annotations

import pytest

from calibre.cli.config import HpoConfig, load_config_from_mapping


def _base_config(hpo: dict) -> dict:
    return {
        "config_schema": "1.0",
        "dataset": {"adapter": "unit_cli", "path": "ignored"},
        "tasks": [{"model": "stub_model", "horizon": 1, "config": {"backend": "stub"}}],
        "origins": {"start": "2024-02-04", "end": "2024-02-04", "freq": "W-SUN"},
        "output": {"ledger_path": "ignored.parquet", "streaming": False},
        "execution": {"backend": "local", "seed": 123},
        "hpo": hpo,
    }


_VALID_SEARCH_SPACE = {
    "quantile_alpha": {"type": "categorical", "choices": [0.45, 0.51, 0.59]},
    "n_estimators": {"type": "int", "low": 200, "high": 800, "step": 50},
    "learning_rate": {"type": "float", "low": 0.02, "high": 0.10, "log": True},
}


def test_valid_hpo_block_loads() -> None:
    config = load_config_from_mapping(
        _base_config({"budget": 8, "seed": 0, "search_space": _VALID_SEARCH_SPACE})
    )

    assert isinstance(config.hpo, HpoConfig)
    assert config.hpo.budget == 8
    assert config.hpo.seed == 0
    assert config.hpo.cost_fractile is None
    assert config.hpo.asha_grace_period == 1
    assert config.hpo.search_space["quantile_alpha"]["choices"] == [0.45, 0.51, 0.59]


def test_cost_fractile_override_loads() -> None:
    config = load_config_from_mapping(
        _base_config({"budget": 4, "search_space": _VALID_SEARCH_SPACE, "cost_fractile": 0.7})
    )

    assert config.hpo is not None
    assert config.hpo.cost_fractile == 0.7


@pytest.mark.parametrize(
    "forbidden",
    [
        "coverage",
        "tau",
        "cost_fractile",
        "order_conformal.coverage",
        # Matched on the final dotted segment, case-insensitively: aliases and
        # case/whitespace variants of a decision/derived number trip the guard too.
        "Tau",
        "critical_ratio",
        "ordering.coverage",
        "objective.tau",
    ],
)
def test_search_space_rejects_forbidden_keys(forbidden: str) -> None:
    space = {**_VALID_SEARCH_SPACE, forbidden: {"type": "float", "low": 0.5, "high": 0.9}}

    with pytest.raises(ValueError, match=f"may not target '{forbidden}'"):
        load_config_from_mapping(_base_config({"budget": 4, "search_space": space}))


@pytest.mark.parametrize(
    "reserved", ["scope", "model", "name", "backend", "horizon", "freq", "Scope"]
)
def test_search_space_rejects_reserved_structural_keys(reserved: str) -> None:
    # Structural model_config/study fields are not hyperparameters; a search key
    # naming one would override the forced study config (e.g. flipping scope).
    space = {**_VALID_SEARCH_SPACE, reserved: {"type": "categorical", "choices": ["x"]}}

    with pytest.raises(ValueError, match="reserved structural key"):
        load_config_from_mapping(_base_config({"budget": 4, "search_space": space}))


def test_search_space_rejects_malformed_spec_type() -> None:
    space = {"quantile_alpha": {"type": "boolean", "choices": [True, False]}}

    with pytest.raises(ValueError, match="type must be one of"):
        load_config_from_mapping(_base_config({"budget": 4, "search_space": space}))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_cost_fractile_outside_open_unit_interval_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        load_config_from_mapping(
            _base_config({"budget": 4, "search_space": _VALID_SEARCH_SPACE, "cost_fractile": bad})
        )


def test_budget_must_be_positive() -> None:
    with pytest.raises(ValueError):
        load_config_from_mapping(_base_config({"budget": 0, "search_space": _VALID_SEARCH_SPACE}))


def test_unknown_hpo_key_rejected() -> None:
    with pytest.raises(ValueError):
        load_config_from_mapping(
            _base_config({"budget": 4, "search_space": _VALID_SEARCH_SPACE, "bogus": 1})
        )
