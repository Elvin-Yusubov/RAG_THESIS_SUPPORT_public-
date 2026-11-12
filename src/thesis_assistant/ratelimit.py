"""In-memory sliding-window rate limiter."""

import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Hashable


class SlidingWindowRateLimiter:
    """Allow at most ``max_requests`` per ``window_seconds`` for each key.

    State lives in process memory, which suits a single bot or web process.
    Use a shared store such as Redis if you run several replicas.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._events: Dict[Hashable, Deque[float]] = defaultdict(deque)

    def allow(self, key: Hashable) -> bool:
        """Record a request for ``key`` and return whether it is within the limit."""
        now = self._clock()
        events = self._events[key]
        while events and now - events[0] >= self.window_seconds:
            events.popleft()
        if len(events) >= self.max_requests:
            return False
        events.append(now)
        return True

    def retry_after(self, key: Hashable) -> float:
        """Seconds until ``key`` may send another request (0 if allowed now)."""
        events = self._events.get(key)
        if not events or len(events) < self.max_requests:
            return 0.0
        return max(0.0, self.window_seconds - (self._clock() - events[0]))
