"""Structured (JSON) logging with per-request correlation IDs.

One JSON object per line is easy to ship to Loki, CloudWatch or ELK and to
filter by ``request_id``. Set ``LOG_FORMAT=text`` for human-readable output
during development.
"""

import contextvars
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from typing import Optional

from thesis_assistant import config

# Correlates every log line produced while handling one user request.
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def new_request_id() -> str:
    """Create a short ID and bind it to the current context."""
    request_id = uuid.uuid4().hex[:12]
    request_id_var.set(request_id)
    return request_id


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON, including any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: Optional[str] = None) -> None:
    """Configure the root logger once (idempotent)."""
    root = logging.getLogger()
    if getattr(root, "_thesis_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    if config.LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.handlers[:] = [handler]
    root.setLevel(level or config.LOG_LEVEL)
    root._thesis_configured = True  # type: ignore[attr-defined]
