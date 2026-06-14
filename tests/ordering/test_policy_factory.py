"""Tests for ordering-policy construction from config."""

from __future__ import annotations

import pandas as pd
import pytest

from calibre.ordering import (
    NewsvendorConfig,
    RsConfig,
    RssConfig,
    build_order_policy,
)

_RS_PARAMS = [{"unique_id": "A", "inventory_position": 5.0, "lead_time": 1, "review_period": 1}]
_NEWSVENDOR_PARAMS = [
    {"unique_id": "A", "underage_cost": 3.0, "overage_cost": 1.0, "inventory_position": 5.0}
]


class TestBuildOrderPolicy:
    def test_rs_spec_builds_rs_config(self) -> None:
        config = build_order_policy({"policy": "rs", "params": _RS_PARAMS, "coverage": 0.95})
        assert isinstance(config, RsConfig)
        assert config.coverage == 0.95
        assert config.quantile is None

    def test_rs_spec_with_quantile_builds_rs_config_with_quantile(self) -> None:
        config = build_order_policy({"policy": "rs", "params": _RS_PARAMS, "quantile": 0.833})
        assert isinstance(config, RsConfig)
        assert config.quantile == pytest.approx(0.833)

    def test_rss_spec_builds_rss_config(self) -> None:
        config = build_order_policy({"policy": "rss", "params": _RS_PARAMS})
        assert isinstance(config, RssConfig)

    def test_newsvendor_spec_builds_newsvendor_config(self) -> None:
        config = build_order_policy(
            {"policy": "newsvendor", "params": _NEWSVENDOR_PARAMS, "period": 2}
        )
        assert isinstance(config, NewsvendorConfig)
        assert config.period == 2

    def test_newsvendor_spec_rejects_quantile_knob(self) -> None:
        with pytest.raises(
            ValueError, match="ordering.quantile is not a valid knob for the newsvendor"
        ):
            build_order_policy(
                {"policy": "newsvendor", "params": _NEWSVENDOR_PARAMS, "quantile": 0.9}
            )

    def test_rss_spec_rejects_quantile_knob(self) -> None:
        with pytest.raises(ValueError, match="ordering.quantile is not a valid knob for the rss"):
            build_order_policy({"policy": "rss", "params": _RS_PARAMS, "quantile": 0.9})

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown order policy: 'bogus'"):
            build_order_policy({"policy": "bogus", "params": _RS_PARAMS})

    def test_newsvendor_period_defaults_to_dataclass_default_when_absent(self) -> None:
        config = build_order_policy({"policy": "newsvendor", "params": _NEWSVENDOR_PARAMS})
        assert isinstance(config, NewsvendorConfig)
        assert config.period == 1

    def test_newsvendor_period_none_defaults_to_dataclass_default(self) -> None:
        config = build_order_policy(
            {"policy": "newsvendor", "params": _NEWSVENDOR_PARAMS, "period": None}
        )
        assert isinstance(config, NewsvendorConfig)
        assert config.period == 1

    def test_params_single_dict_is_wrapped_to_one_row(self) -> None:
        config = build_order_policy({"policy": "rs", "params": _RS_PARAMS[0]})
        assert isinstance(config, RsConfig)
        assert isinstance(config.params, pd.DataFrame)
        assert len(config.params) == 1
        assert config.params.loc[0, "unique_id"] == "A"

    def test_params_dataframe_is_passed_through(self) -> None:
        params_frame = pd.DataFrame(_RS_PARAMS)
        config = build_order_policy({"policy": "rs", "params": params_frame})
        assert isinstance(config, RsConfig)
        assert config.params is params_frame

    def test_params_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="ordering.params is required"):
            build_order_policy({"policy": "rs"})

    def test_params_none_raises(self) -> None:
        with pytest.raises(ValueError, match="ordering.params is required"):
            build_order_policy({"policy": "rs", "params": None})

    def test_omitted_coverage_defaults(self) -> None:
        config = build_order_policy({"policy": "rs", "params": _RS_PARAMS})
        assert config.coverage == 0.9
