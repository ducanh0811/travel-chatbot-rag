"""
tests/test_session_store.py
============================
Unit tests cho InMemorySessionStore — không cần Redis.
"""
import time
import sys
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from session_store import InMemorySessionStore, _new_session_data


# ─── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def store():
    return InMemorySessionStore(ttl=60)


# ─── Tests ───────────────────────────────────────────────────────────────────
class TestGetOrCreate:
    def test_creates_new_session(self, store):
        data = store.get_or_create("user-1")
        assert data["id"] == "user-1"
        assert data["history"] == []
        assert data["summary"] == ""

    def test_returns_existing_session(self, store):
        store.get_or_create("user-2")
        store.add_message("user-2", "user", "Xin chào")
        data = store.get_or_create("user-2")
        assert len(data["history"]) == 1

    def test_expired_session_is_recreated(self, store):
        # TTL rất ngắn
        short_store = InMemorySessionStore(ttl=1)
        short_store.get_or_create("user-exp")
        short_store.add_message("user-exp", "user", "Tin nhắn cũ")
        time.sleep(1.1)
        data = short_store.get_or_create("user-exp")
        assert data["history"] == []  # Session mới — không có history cũ


class TestAddMessage:
    def test_adds_user_message(self, store):
        store.get_or_create("s1")
        store.add_message("s1", "user", "Quán cafe Sơn Trà")
        history = store.get_recent_history("s1", last_n=5)
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Quán cafe Sơn Trà"

    def test_adds_assistant_message(self, store):
        store.get_or_create("s2")
        store.add_message("s2", "assistant", "Đây là gợi ý...")
        history = store.get_recent_history("s2")
        assert history[0]["role"] == "assistant"

    def test_history_has_timestamp(self, store):
        store.get_or_create("s3")
        store.add_message("s3", "user", "Test")
        history = store.get_recent_history("s3")
        assert "timestamp" in history[0]

    def test_history_trimmed_at_max(self, store):
        """Kiểm tra history bị giới hạn khi vượt MAX_HISTORY*2."""
        from session_store import MAX_HISTORY
        store.get_or_create("s4")
        for i in range(MAX_HISTORY * 2 + 10):
            store.add_message("s4", "user", f"msg {i}")
        data = store.get("s4")
        assert len(data["history"]) <= MAX_HISTORY * 2


class TestIsolation:
    def test_two_users_are_isolated(self, store):
        """Hai session khác nhau không chia sẻ lịch sử."""
        store.get_or_create("alice")
        store.get_or_create("bob")
        store.add_message("alice", "user", "Tôi muốn cafe Sơn Trà")
        store.add_message("bob", "user", "Khách sạn 5 sao Hải Châu")

        alice_hist = store.get_recent_history("alice")
        bob_hist = store.get_recent_history("bob")

        assert alice_hist[0]["content"] == "Tôi muốn cafe Sơn Trà"
        assert bob_hist[0]["content"] == "Khách sạn 5 sao Hải Châu"
        assert len(alice_hist) == 1
        assert len(bob_hist) == 1


class TestSummary:
    def test_update_and_get_summary(self, store):
        store.get_or_create("sum-1")
        store.update_summary("sum-1", "Người dùng tìm cafe Sơn Trà, thích view biển.")
        data = store.get("sum-1")
        assert "Sơn Trà" in data["summary"]


class TestDelete:
    def test_clear_removes_session(self, store):
        store.get_or_create("del-1")
        store.clear("del-1")
        assert store.get("del-1") is None

    def test_clear_nonexistent_is_safe(self, store):
        store.clear("nonexistent-xyz")  # Không raise exception


class TestCleanup:
    def test_cleanup_removes_expired(self):
        short_store = InMemorySessionStore(ttl=1)
        short_store.get_or_create("exp-1")
        short_store.get_or_create("exp-2")
        time.sleep(1.1)
        cleaned = short_store.cleanup_expired()
        assert cleaned == 2

    def test_cleanup_keeps_active(self):
        active_store = InMemorySessionStore(ttl=60)
        active_store.get_or_create("active-1")
        cleaned = active_store.cleanup_expired()
        assert cleaned == 0
        assert active_store.get("active-1") is not None


class TestStats:
    def test_stats_returns_correct_backend(self, store):
        stats = store.get_stats()
        assert stats["backend"] == "InMemorySessionStore"
        assert "active_sessions" in stats
        assert "ttl_seconds" in stats
