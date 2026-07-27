"""Expose M5 loading, input verification, and independent scoring."""

from newcalibre.protocols.m5.config import load_m5_config
from newcalibre.protocols.m5.inventory import verify_m5_inputs
from newcalibre.protocols.m5.scorer import M5Diagnostics, score_m5

__all__ = ["M5Diagnostics", "load_m5_config", "score_m5", "verify_m5_inputs"]
