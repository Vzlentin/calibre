"""Structured JSON logging setup for the engine."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TextIO

_RESERVED_ATTRS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    """Formatter that renders each log record as a single JSON line.

    Standard fields (timestamp, level, logger, message) are emitted alongside
    any non-reserved ``extra`` attributes, each coerced to a JSON-safe value.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _RESERVED_ATTRS:
                continue
            payload[key] = _json_safe(value)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _json_safe(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def setup_logging(
    level: str | int = "INFO",
    *,
    format: str = "json",
    stream: TextIO | None = None,
) -> None:
    """Configure root logging at ``level`` with a ``json`` or ``text`` formatter.

    Raises ValueError for an unknown ``format``.
    """
    handler = logging.StreamHandler(stream or sys.stderr)
    if format == "json":
        handler.setFormatter(JsonFormatter())
    elif format == "text":
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    else:
        raise ValueError("log format must be 'json' or 'text'")
    logging.basicConfig(level=level, handlers=[handler], force=True)
