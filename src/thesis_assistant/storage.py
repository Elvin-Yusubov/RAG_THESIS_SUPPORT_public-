"""SQLite persistence for Telegram conversation history and usage accounting.

Uses the standard library only. A new connection is opened per call, which is
safe across threads and cheap for SQLite at this scale. For multiple bot
replicas, replace this module with Redis or Postgres behind the same methods.
"""

import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_history_chat ON chat_history (chat_id, id);

CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    channel TEXT NOT NULL,
    kind TEXT NOT NULL,
    latency_ms INTEGER,
    sources INTEGER,
    ok INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_log (user_key, created_at);
"""


class StateStore:
    """Chat history and usage log backed by a single SQLite file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: Optional[sqlite3.Connection] = None
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # ":memory:" databases vanish when the connection closes; keep one open.
        if str(self.path) == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return _NonClosing(self._memory_conn)  # type: ignore[return-value]
        return sqlite3.connect(self.path)

    # --- chat history ---------------------------------------------------------------

    def add_turn(self, chat_id: int, user_text: str, assistant_text: str, keep_turns: int) -> None:
        """Store one exchange and prune the chat to its latest ``keep_turns`` exchanges."""
        now = time.time()
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO chat_history (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [(chat_id, "user", user_text, now), (chat_id, "assistant", assistant_text, now)],
            )
            conn.execute(
                """DELETE FROM chat_history WHERE chat_id = ? AND id NOT IN (
                       SELECT id FROM chat_history WHERE chat_id = ? ORDER BY id DESC LIMIT ?)""",
                (chat_id, chat_id, keep_turns * 2),
            )

    def get_history(self, chat_id: int) -> List[Dict[str, str]]:
        """Return the chat's messages oldest first as ``{"role", "content"}`` dicts."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content FROM chat_history WHERE chat_id = ? ORDER BY id", (chat_id,)
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def clear_history(self, chat_id: int) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))

    # --- usage accounting ---------------------------------------------------------------

    def log_usage(
        self,
        user_key: str,
        channel: str,
        kind: str,
        latency_ms: Optional[int] = None,
        sources: Optional[int] = None,
        ok: bool = True,
    ) -> None:
        """Record one request (who, where, what, how long, how many sources, success)."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO usage_log (user_key, channel, kind, latency_ms, sources, ok, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_key, channel, kind, latency_ms, sources, int(ok), time.time()),
            )

    def usage_summary(self) -> List[Dict[str, object]]:
        """Per user and channel: request count, failures and average latency."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT user_key, channel, COUNT(*), SUM(1 - ok), AVG(latency_ms)
                   FROM usage_log GROUP BY user_key, channel ORDER BY COUNT(*) DESC"""
            ).fetchall()
        return [
            {"user": u, "channel": ch, "requests": n, "failures": int(f or 0), "avg_latency_ms": round(a or 0)}
            for u, ch, n, f, a in rows
        ]


class _NonClosing:
    """Wrap a shared in-memory connection so ``closing()`` does not destroy it."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, *args):
        return self._conn.__exit__(*args)

    def close(self) -> None:  # intentionally a no-op
        pass
