from __future__ import annotations

from calibre.execution import threading as threading_mod


def test_cap_threaded_config_single_source_of_truth():
    """backend and optimizer must import _cap_threaded_config from threading, not define it."""
    import calibre.execution.backend as backend_mod
    import calibre.tuning.optimizer as optimizer_mod

    assert backend_mod._cap_threaded_config is threading_mod._cap_threaded_config
    assert optimizer_mod._cap_threaded_config is threading_mod._cap_threaded_config


def test_cap_threaded_config_caps_n_jobs():
    from calibre.execution.threading import _cap_threaded_config

    result = _cap_threaded_config({"n_jobs": 16}, cpu_per_task=2.0)
    assert result["n_jobs"] == 2


def test_cap_threaded_config_none_passes_through():
    from calibre.execution.threading import _cap_threaded_config

    cfg = {"n_jobs": 8}
    assert _cap_threaded_config(cfg, cpu_per_task=None) is cfg
