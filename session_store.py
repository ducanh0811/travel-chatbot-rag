"""
session_store.py
================
Abstraction layer cho session storage với hai backend:
- RedisSessionStore  : Redis (persist qua restart)
- InMemorySessionStore : fallback khi không có Redis

Tự động chọn backend dựa trên biến môi trường REDIS_URL.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_TTL = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))
MAX_HISTORY = int(os.environ.get("MAX_SESSION_HISTORY", "20"))
REDIS_KEY_PREFIX = "travel_chatbot:session:"


# ─── Data helpers ─────────────────────────────────────────────────────────────
def _new_session_data(session_id: str) -> dict:
    now = time.time()
    return {
        "id": session_id,
        "history": [],                # list of {role, content, timestamp}
        "summary": "",                # rolling summary từ ConversationSummarizer
        "summary_message_count": 0,   # history length tại lần summary gần nhất
        "created_at": now,
        "last_access": now,
    }


# ─── Abstract base ────────────────────────────────────────────────────────────
class BaseSessionStore(ABC):
    """Interface chung cho mọi backend."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[dict]:
        ...

    @abstractmethod
    def set(self, session_id: str, data: dict, ttl: int = DEFAULT_TTL) -> None:
        ...

    @abstractmethod
    def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    def list_all_ids(self) -> List[str]:
        ...

    # ── Convenience helpers shared by all backends ────────────────────────────

    def get_or_create(self, session_id: str) -> dict:
        """Lấy session hoặc tạo mới nếu chưa tồn tại / đã hết hạn."""
        data = self.get(session_id)
        if data is None:
            data = _new_session_data(session_id)
        data["last_access"] = time.time()
        self.set(session_id, data)
        return data

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Thêm message vào history của session."""
        data = self.get_or_create(session_id)
        data["history"].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Giới hạn số lượng message lưu trữ
        if len(data["history"]) > MAX_HISTORY * 2:
            data["history"] = data["history"][-(MAX_HISTORY * 2):]
        self.set(session_id, data)

    def get_recent_history(self, session_id: str, last_n: int = 5) -> List[dict]:
        """Lấy `last_n` cặp message gần nhất."""
        data = self.get(session_id)
        if not data:
            return []
        return data["history"][-(last_n * 2):]

    def update_summary(self, session_id: str, summary: str) -> None:
        """Cập nhật rolling summary."""
        data = self.get_or_create(session_id)
        data["summary"] = summary
        data["summary_message_count"] = len(data.get("history", []))
        self.set(session_id, data)

    def clear(self, session_id: str) -> None:
        self.delete(session_id)

    def get_stats(self) -> dict:
        ids = self.list_all_ids()
        return {
            "active_sessions": len(ids),
            "ttl_seconds": DEFAULT_TTL,
            "max_history_per_session": MAX_HISTORY,
            "backend": self.__class__.__name__,
        }


# ─── In-Memory backend ────────────────────────────────────────────────────────
class InMemorySessionStore(BaseSessionStore):
    """
    Backend in-memory với TTL check thủ công.
    Không persist qua restart — dùng khi không có Redis.
    """

    def __init__(self, ttl: int = DEFAULT_TTL):
        self._store: Dict[str, dict] = {}
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            # TTL check
            if time.time() - entry["last_access"] > self._ttl:
                del self._store[session_id]
                return None
            return entry

    def set(self, session_id: str, data: dict, ttl: int = DEFAULT_TTL) -> None:
        with self._lock:
            self._store[session_id] = data

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def list_all_ids(self) -> List[str]:
        with self._lock:
            now = time.time()
            active = [
                sid
                for sid, data in self._store.items()
                if now - data["last_access"] <= self._ttl
            ]
            return active

    def cleanup_expired(self) -> int:
        """Xóa các session đã hết hạn. Trả về số session đã xóa."""
        with self._lock:
            now = time.time()
            expired = [
                sid
                for sid, data in self._store.items()
                if now - data["last_access"] > self._ttl
            ]
            for sid in expired:
                del self._store[sid]
            return len(expired)


# ─── Redis backend ────────────────────────────────────────────────────────────
class RedisSessionStore(BaseSessionStore):
    """
    Backend Redis — session persist qua restart.
    Yêu cầu REDIS_URL trong .env, ví dụ: redis://localhost:6379/0
    """

    def __init__(self, redis_url: str, ttl: int = DEFAULT_TTL):
        import redis  # type: ignore

        self._client = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl
        # Ping để kiểm tra kết nối
        self._client.ping()
        logger.info("RedisSessionStore connected: %s", redis_url)

    def _key(self, session_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}{session_id}"

    def get(self, session_id: str) -> Optional[dict]:
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    def set(self, session_id: str, data: dict, ttl: int = DEFAULT_TTL) -> None:
        self._client.setex(
            self._key(session_id),
            ttl or self._ttl,
            json.dumps(data, ensure_ascii=False),
        )

    def delete(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))

    def list_all_ids(self) -> List[str]:
        prefix_len = len(REDIS_KEY_PREFIX)
        keys = self._client.keys(f"{REDIS_KEY_PREFIX}*")
        return [k[prefix_len:] for k in keys]

    def cleanup_expired(self) -> int:
        """Redis tự xử lý TTL — trả về 0."""
        return 0


# ─── Factory ──────────────────────────────────────────────────────────────────
def create_session_store() -> BaseSessionStore:
    """
    Tự động chọn backend:
    - Redis nếu REDIS_URL có trong .env và kết nối thành công
    - InMemory fallback còn lại
    """
    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url:
        try:
            store = RedisSessionStore(redis_url)
            logger.info("✅ Sử dụng RedisSessionStore")
            return store
        except Exception as exc:
            logger.warning(
                "⚠️ Không thể kết nối Redis (%s), fallback sang InMemorySessionStore: %s",
                redis_url,
                exc,
            )
    logger.info("ℹ️ Sử dụng InMemorySessionStore (session mất khi restart)")
    return InMemorySessionStore()
