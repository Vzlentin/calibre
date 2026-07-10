"""Own strict canonical JSON shared by domain identity and transport codecs."""

from __future__ import annotations

import json


class CanonicalJsonError(ValueError):
    """Report a value outside the finite, transport-safe JSON subset."""


def require_json_value(
    value: object,
    *,
    path: str,
    _ancestors: set[int] | None = None,
) -> None:
    """Reject values outside null, scalar, list, and string-keyed object JSON."""
    ancestors = set() if _ancestors is None else _ancestors
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not -float("inf") < value < float("inf"):
            raise CanonicalJsonError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in ancestors:
            raise CanonicalJsonError(f"{path} contains a cyclic value")
        ancestors.add(identity)
        try:
            for index, item in enumerate(value):
                require_json_value(
                    item,
                    path=f"{path}[{index}]",
                    _ancestors=ancestors,
                )
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in ancestors:
            raise CanonicalJsonError(f"{path} contains a cyclic value")
        ancestors.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise CanonicalJsonError(f"{path} contains a non-string object key")
                require_json_value(
                    item,
                    path=f"{path}.{key}",
                    _ancestors=ancestors,
                )
        finally:
            ancestors.remove(identity)
        return
    raise CanonicalJsonError(f"{path} contains non-JSON value {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return the single UTF-8-oriented textual form for an admitted value."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: object, *, path: str) -> bytes:
    """Validate a value and return its canonical UTF-8 bytes."""
    require_json_value(value, path=path)
    try:
        return canonical_json(value).encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as error:
        raise CanonicalJsonError(f"{path} must contain finite UTF-8 JSON values") from error
