"""StatsForecast objects built from the same specs that `references.py` reads."""

import statsforecast.models
from statsforecast import StatsForecast


def model(spec):
    if spec is None:
        return None
    name, kwargs = spec
    return getattr(statsforecast.models, name)(**kwargs)


def engine(models, freq: str, fallback=None, n_jobs: int = 1) -> StatsForecast:
    return StatsForecast(
        models=[model(spec) for spec in models],
        freq=freq,
        fallback_model=model(fallback),
        n_jobs=n_jobs,
    )
