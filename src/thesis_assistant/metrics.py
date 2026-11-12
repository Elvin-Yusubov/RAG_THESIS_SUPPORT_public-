"""Prometheus metrics for request volume, latency and failures.

The Telegram bot exposes them on ``/metrics`` when ``METRICS_PORT`` is set.
Scrape with Prometheus and alert on a rising failure ratio or p95 latency.
"""

import logging
from typing import Optional

from prometheus_client import Counter, Histogram, start_http_server

logger = logging.getLogger(__name__)

REQUESTS = Counter(
    "thesis_requests_total", "Handled user requests", ["channel", "kind", "status"]
)
LATENCY = Histogram(
    "thesis_request_latency_seconds",
    "End-to-end request latency",
    ["channel"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80),
)
RATE_LIMITED = Counter("thesis_rate_limited_total", "Requests rejected by the rate limiter", ["channel"])
SOURCES = Histogram(
    "thesis_sources_per_answer", "Retrieved chunks per answer", buckets=(0, 1, 2, 3, 5, 10)
)


def record_request(channel: str, kind: str, ok: bool, latency_s: Optional[float] = None, sources: Optional[int] = None) -> None:
    """Record one finished request."""
    REQUESTS.labels(channel=channel, kind=kind, status="ok" if ok else "error").inc()
    if latency_s is not None:
        LATENCY.labels(channel=channel).observe(latency_s)
    if sources is not None:
        SOURCES.observe(sources)


def start_metrics_server(port: int) -> None:
    """Expose metrics at ``http://0.0.0.0:<port>/metrics``."""
    start_http_server(port)
    logger.info("Metrics server listening", extra={"port": port})
