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
    return f"legacy-{run_id.hex}"
