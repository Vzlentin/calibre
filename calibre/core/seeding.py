from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Seed:
    value: int


def set_seed(seed: int | Seed) -> Seed:
    resolved = seed if isinstance(seed, Seed) else Seed(int(seed))
    random.seed(resolved.value)
    np.random.seed(resolved.value)
    return resolved


def seed_model_config(model_config: dict, seed: Seed | int | None) -> dict:
    if seed is None:
        return dict(model_config)

    resolved = seed if isinstance(seed, Seed) else Seed(int(seed))
    config = dict(model_config)
    backend = str(config.get("backend", ""))

    if backend == "mlforecast" and "random_state" not in config and "seed" not in config:
        config["random_state"] = resolved.value
    elif backend == "neuralforecast" and "random_seed" not in config:
        config["random_seed"] = resolved.value

    return config
