from __future__ import annotations

from calibre.execution import backend
from calibre.execution.threading import _cap_threaded_config
from calibre.tuning import optimizer


def test_cap_threaded_config_single_source_of_truth() -> None:
    assert backend._cap_threaded_config is _cap_threaded_config
    assert optimizer._cap_threaded_config is _cap_threaded_config

    capped = _cap_threaded_config(
        {"model": "lightgbm.LGBMRegressor", "n_jobs": -1, "num_threads": 16},
        cpu_budget=2.0,
    )

    assert capped["n_jobs"] == 2
    assert capped["num_threads"] == 2
