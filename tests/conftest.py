"""Shared fixtures: keep tests offline and isolated from real state."""

import pytest

from thesis_assistant import config
from thesis_assistant.storage import StateStore


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "state.db")


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Point the Telegram bot at a temp database and clear module-level state."""
    try:
        from thesis_assistant import telegram_bot
    except ImportError:  # private module absent (public clone): nothing to isolate
        yield
        return

    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "bot.db")
    monkeypatch.setattr(config, "TELEGRAM_ALLOWED_USER_IDS", set())  # ignore the developer's real .env
    telegram_bot.get_store.cache_clear()
    telegram_bot.pending_documents.clear()
    yield
    telegram_bot.get_store.cache_clear()
