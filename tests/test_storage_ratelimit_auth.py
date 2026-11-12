import pytest

from thesis_assistant import auth
from thesis_assistant.ratelimit import SlidingWindowRateLimiter
from thesis_assistant.storage import StateStore


def test_store_keeps_only_latest_turns(store):
    for i in range(5):
        store.add_turn(1, f"q{i}", f"a{i}", keep_turns=2)
    history = store.get_history(1)
    assert [m["content"] for m in history] == ["q3", "a3", "q4", "a4"]


def test_store_isolates_chats_and_clears(store):
    store.add_turn(1, "q", "a", keep_turns=5)
    store.add_turn(2, "x", "y", keep_turns=5)
    store.clear_history(1)
    assert store.get_history(1) == []
    assert len(store.get_history(2)) == 2


def test_usage_summary_counts_failures(store):
    store.log_usage("u", "web", "text", latency_ms=100, sources=2)
    store.log_usage("u", "web", "text", ok=False)
    summary = store.usage_summary()[0]
    assert summary["requests"] == 2 and summary["failures"] == 1


def test_in_memory_store_works():
    store = StateStore(":memory:")
    store.add_turn(1, "q", "a", keep_turns=1)
    assert len(store.get_history(1)) == 2


def test_rate_limiter_blocks_then_recovers():
    now = [0.0]
    limiter = SlidingWindowRateLimiter(2, 10, clock=lambda: now[0])
    assert limiter.allow("u") and limiter.allow("u")
    assert not limiter.allow("u")
    assert limiter.retry_after("u") == pytest.approx(10)
    assert limiter.allow("other")
    now[0] = 10.1
    assert limiter.allow("u")


def test_password_hash_roundtrip_and_salting():
    stored = auth.hash_password("secret")
    assert auth.verify_password("secret", stored)
    assert not auth.verify_password("wrong", stored)
    assert auth.hash_password("secret") != stored  # random salt


def test_authenticate_with_parsed_users():
    users = auth.parse_users(f"alice:{auth.hash_password('pw1')}; bob:{auth.hash_password('pw2')}")
    assert set(users) == {"alice", "bob"}
    assert auth.authenticate("alice", "pw1", users)
    assert not auth.authenticate("alice", "pw2", users)
    assert not auth.authenticate("nobody", "pw1", users)


def test_verify_rejects_malformed_hash():
    assert not auth.verify_password("x", "no-separator")
