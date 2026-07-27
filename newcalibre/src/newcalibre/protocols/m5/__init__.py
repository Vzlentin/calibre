"""Expose only M5 configuration loading and public input verification."""

from newcalibre.protocols.m5.config import load_m5_config
from newcalibre.protocols.m5.inventory import verify_m5_inputs

__all__ = ["load_m5_config", "verify_m5_inputs"]
