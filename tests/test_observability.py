import json
import logging

from prometheus_client import REGISTRY

from thesis_assistant import metrics
from thesis_assistant.logging_config import JsonFormatter, new_request_id, request_id_var


def _record(msg="hello", **extra):
    record = logging.LogRecord("t", logging.INFO, __file__, 1, msg, (), None)
    record.__dict__.update(extra)
    return record


def test_json_formatter_emits_valid_json_with_extras():
    request_id_var.set(None)
    line = JsonFormatter().format(_record(channel="web", latency_ms=12))
    data = json.loads(line)
    assert data["msg"] == "hello" and data["level"] == "INFO"
    assert data["channel"] == "web" and data["latency_ms"] == 12
    assert "request_id" not in data


def test_request_id_is_attached_to_logs():
    rid = new_request_id()
    data = json.loads(JsonFormatter().format(_record()))
    assert data["request_id"] == rid


def test_metrics_record_request_updates_counters():
    before = REGISTRY.get_sample_value(
        "thesis_requests_total", {"channel": "test", "kind": "text", "status": "ok"}
    ) or 0
    metrics.record_request("test", "text", ok=True, latency_s=0.3, sources=2)
    after = REGISTRY.get_sample_value(
        "thesis_requests_total", {"channel": "test", "kind": "text", "status": "ok"}
    )
    assert after == before + 1
