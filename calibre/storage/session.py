"""Deterministic session-id derivation for the lifecycle store."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID


def derive_session_id(
    tenant: str,
    sku_set: list[str],
    model_config: dict,
    conformal_config: dict,
) -> str:
    """Derive a stable session id from the tenant, SKUs, and configs.

    The id is a SHA-256 hash of the canonicalized inputs, so identical
    configurations map to the same session and can reuse cached state.
    """
    payload = json.dumps(
        {
            "tenant": tenant,
            "sku_set": sorted(sku_set),
            "model_config": model_config,
            "conformal_config": conformal_config,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def legacy_session_id(run_id: UUID) -> str:
    """Return the legacy run-scoped session id for a pre-session-keying run."""
    return f"legacy-{run_id.hex}"
