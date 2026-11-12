import pytest

pytest.importorskip("thesis_assistant.telegram_bot", reason="private module not present")

from thesis_assistant import config, telegram_bot


def test_open_access_when_allowlist_empty(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_ALLOWED_USER_IDS", set())
    assert telegram_bot._is_authorized(123)


def test_allowlist_is_enforced(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_ALLOWED_USER_IDS", {1, 2})
    assert telegram_bot._is_authorized(1)
    assert not telegram_bot._is_authorized(3)




# --- Persistence and conversation flow --------------------------------------------------

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402


def test_history_is_capped_and_persisted():
    for i in range(telegram_bot.MAX_STORED_TURNS + 5):
        telegram_bot._remember_turn(1, f"q{i}", f"a{i}")
    history = telegram_bot.get_store().get_history(1)
    assert len(history) == telegram_bot.MAX_STORED_TURNS * 2
    assert history[-1]["content"] == f"a{telegram_bot.MAX_STORED_TURNS + 4}"

    # A fresh store object on the same file still sees the history (survives restart).
    telegram_bot.get_store.cache_clear()
    assert telegram_bot.get_store().get_history(1)[-1]["content"] == history[-1]["content"]


def _update(text: str, user_id: int = 7, chat_id: int = 7):
    placeholder = MagicMock(edit_text=AsyncMock())
    message = MagicMock(text=text, reply_text=AsyncMock(return_value=placeholder))
    return SimpleNamespace(
        message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
    ), placeholder


def test_handle_message_streams_answer_and_saves_history(monkeypatch):
    docs = [object(), object()]

    def fake_stream(query, chat_history=None, document_text=None):
        yield "Salam, "
        yield "dünya"
        yield {"answer": "Salam, dünya", "context": docs}

    monkeypatch.setattr(telegram_bot, "stream_llm", fake_stream)
    monkeypatch.setattr(telegram_bot, "rate_limiter", telegram_bot.SlidingWindowRateLimiter(100, 60))
    update, placeholder = _update("Salam")
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(telegram_bot.handle_message(update, context))

    placeholder.edit_text.assert_awaited_with("Salam, dünya")
    assert telegram_bot.get_store().get_history(7) == [
        {"role": "user", "content": "Salam"},
        {"role": "assistant", "content": "Salam, dünya"},
    ]
    usage = telegram_bot.get_store().usage_summary()
    assert usage[0]["requests"] == 1 and usage[0]["failures"] == 0


def test_handle_message_rate_limited(monkeypatch):
    monkeypatch.setattr(telegram_bot, "rate_limiter", telegram_bot.SlidingWindowRateLimiter(1, 60))
    monkeypatch.setattr(telegram_bot, "stream_llm", lambda *a, **k: iter([{"answer": "ok", "context": []}]))
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    first, _ = _update("one")
    second, _ = _update("two")
    asyncio.run(telegram_bot.handle_message(first, context))
    asyncio.run(telegram_bot.handle_message(second, context))

    reply = second.message.reply_text.await_args.args[0]
    assert "saniyə" in reply


def test_handle_message_reports_llm_failure(monkeypatch):
    def broken_stream(*args, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    monkeypatch.setattr(telegram_bot, "stream_llm", broken_stream)
    monkeypatch.setattr(telegram_bot, "rate_limiter", telegram_bot.SlidingWindowRateLimiter(100, 60))
    update, placeholder = _update("Salam")
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(telegram_bot.handle_message(update, context))

    assert "went wrong" in placeholder.edit_text.await_args.args[0]
    assert telegram_bot.get_store().usage_summary()[0]["failures"] == 1
