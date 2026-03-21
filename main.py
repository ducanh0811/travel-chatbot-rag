"""
main.py — Travel Chatbot RAG API
=================================
FastAPI server với:
- Multi-user session management (Redis hoặc in-memory fallback)
- Context-aware conversation history (ContextBuilder)
- Multi-agent routing (Supervisor → weather / travel)
"""
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
import logging
import time
import re
import uuid
import threading

from Supervisor import (
    get_supervisor_instance,
    get_weather_agent_instance,
    get_travel_agent_instance,
    reset_supervisor,
    health_check,
    classify_query,
)
from summarizer import ConversationSummarizer
from session_store import BaseSessionStore, create_session_store
from context_builder import ContextBuilder, get_context_builder

logger = logging.getLogger("travel-chatbot")
logging.basicConfig(level=logging.INFO)

MAX_QUERY_LEN = 500
SESSION_CLEANUP_INTERVAL = 900  # 15 phút

app = FastAPI(
    title="Travel Chatbot RAG API",
    description="API cho chatbot du lịch Đà Nẵng với RAG và multi-agent",
    version="3.0.0",
)

# ─── Global singletons ────────────────────────────────────────────────────────
session_store: BaseSessionStore = create_session_store()
context_builder: ContextBuilder = get_context_builder()
conversation_summarizer: ConversationSummarizer = ConversationSummarizer()


# ─── Background cleanup ───────────────────────────────────────────────────────
def _schedule_cleanup():
    """Định kỳ dọn session hết hạn (15 phút/lần)."""
    while True:
        time.sleep(SESSION_CLEANUP_INTERVAL)
        try:
            from session_store import InMemorySessionStore
            if isinstance(session_store, InMemorySessionStore):
                cleaned = session_store.cleanup_expired()
                if cleaned:
                    logger.info("🧹 Auto-cleaned %s expired sessions", cleaned)
        except Exception as exc:
            logger.warning("Cleanup error: %s", exc)


_cleanup_thread = threading.Thread(target=_schedule_cleanup, daemon=True)


# ─── Request / Response models ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    use_context: bool = True      # Có inject conversation history không


class QueryResponse(BaseModel):
    result: str
    session_id: str
    has_context: bool


class SessionRequest(BaseModel):
    session_id: str


# ─── Helpers ──────────────────────────────────────────────────────────────────
def normalize_query(raw: str) -> str:
    return (raw or "").strip()


def validate_query(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query không được để trống")
    if len(query) > MAX_QUERY_LEN:
        raise HTTPException(
            status_code=400, detail=f"Query quá dài (>{MAX_QUERY_LEN} ký tự)"
        )


def _update_summary_async(session_id: str, session_data: dict):
    """Cập nhật rolling summary sau khi trả lời (non-blocking)."""
    try:
        history = session_data.get("history", [])
        last_count = session_data.get("summary_message_count", 0)
        if len(history) - last_count < 4:
            return
        recent = history[-conversation_summarizer.max_recent_messages:]
        prev_summary = session_data.get("summary", "")
        new_summary = conversation_summarizer.summarize(recent, prev_summary)
        session_store.update_summary(session_id, new_summary)
    except Exception as exc:
        logger.warning("Summary update failed for %s: %s", session_id, exc)


def _extract_final_response(messages: list, query: str, enhanced_query: str) -> str:
    """Lọc lấy câu trả lời cuối cùng có nội dung từ danh sách message."""
    for msg in reversed(messages):
        content = getattr(msg, "content", msg)
        if not content:
            continue
        content_str = str(content)
        # Bỏ qua transfer messages
        if any(
            kw in content_str.lower()
            for kw in ["transferred to", "transferring", "successfully transfer"]
        ):
            continue
        # Bỏ qua echo của query
        if content_str in (query, enhanced_query):
            continue
        # Xử lý <internal> tags
        if "<internal>" in content_str.lower():
            blocks = re.findall(
                r"<internal>(.*?)</internal>",
                content_str,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if blocks:
                content_str = "\n\n".join(b.strip() for b in blocks if b.strip())
            else:
                content_str = re.sub(
                    r"</?internal>", "", content_str, flags=re.IGNORECASE
                )
        content_str = content_str.strip()
        if content_str:
            return content_str
    return "❌ Không có phản hồi nội dung từ agent."


# ─── Startup / Shutdown ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Travel Chatbot RAG API v3.0...")
    get_supervisor_instance()      # Pre-warm supervisor
    _cleanup_thread.start()        # Bắt đầu background cleanup
    logger.info("✅ API is ready.")


# ─── Main endpoints ───────────────────────────────────────────────────────────
@app.post("/ask", response_model=QueryResponse)
async def ask_agent(data: QueryRequest, background_tasks: BackgroundTasks):
    """
    Endpoint chính — hỏi chatbot với conversation memory.

    - Tự tạo session_id nếu không có
    - Inject lịch sử hội thoại và context hint vào query
    - Context hint giúp RAG hiểu follow-up questions ("ở đó", "loại đó")
    """
    query = normalize_query(data.query)
    validate_query(query)

    # ── Session ──────────────────────────────────────────────────────────────
    session_id = data.session_id or str(uuid.uuid4())
    session_data = session_store.get_or_create(session_id)

    # ── Build context ─────────────────────────────────────────────────────────
    agent_ctx = context_builder.build(session_data, query) if data.use_context else None

    has_context = agent_ctx.has_context if agent_ctx else False
    enhanced_query = agent_ctx.enhanced_query if agent_ctx else query

    # Nếu có RAG hint, nhúng vào query để travel agent dùng khi gọi rag_tool
    if agent_ctx and not agent_ctx.rag_hint.is_empty():
        enhanced_query = context_builder.enrich_rag_query(
            agent_ctx.enhanced_query, agent_ctx.rag_hint
        )

    # ── Routing & invoke ───────────────────────────────────────────────────────
    start = time.perf_counter()
    try:
        route = classify_query(query)
        if route == "weather":
            agent = get_weather_agent_instance()
        elif route == "travel":
            agent = get_travel_agent_instance()
        else:
            agent = get_supervisor_instance()

        messages_payload = []
        if agent_ctx and agent_ctx.summary_message:
            messages_payload.append(
                {"role": "assistant", "content": agent_ctx.summary_message}
            )
        messages_payload.append({"role": "user", "content": enhanced_query})

        result = agent.invoke({"messages": messages_payload})
        messages = result.get("messages", [])
        final_response = _extract_final_response(messages, query, enhanced_query)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Persist & update summary ───────────────────────────────────────────────
    session_store.add_message(session_id, "user", query)
    session_store.add_message(session_id, "assistant", final_response)
    # Lấy lại session data mới nhất để truyền cho summary task
    updated_session = session_store.get(session_id) or session_data
    background_tasks.add_task(_update_summary_async, session_id, updated_session)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "route=%s session=%s ctx=%s ms=%s", route, session_id, has_context, elapsed_ms
    )

    return QueryResponse(
        result=final_response,
        session_id=session_id,
        has_context=has_context,
    )


@app.post("/ask/simple")
async def ask_simple(data: QueryRequest):
    """
    Endpoint đơn giản — không có conversation memory.
    Tương thích ngược với v2.
    """
    query = normalize_query(data.query)
    validate_query(query)

    start = time.perf_counter()
    try:
        route = classify_query(query)
        if route == "weather":
            agent = get_weather_agent_instance()
        elif route == "travel":
            agent = get_travel_agent_instance()
        else:
            agent = get_supervisor_instance()

        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", [])
        final_response = _extract_final_response(messages, query, query)

    except Exception as exc:
        return {"error": str(exc)}

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("route=%s session=none ctx=false ms=%s", route, elapsed_ms)
    return {"result": final_response}


# ─── Session endpoints ────────────────────────────────────────────────────────
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Lấy toàn bộ lịch sử hội thoại của session."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' không tồn tại")
    return {
        "session_id": session_id,
        "history": session.get("history", []),
        "message_count": len(session.get("history", [])),
        "created_at": datetime.fromtimestamp(session["created_at"]).isoformat(),
        "last_access": datetime.fromtimestamp(session["last_access"]).isoformat(),
    }


@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Lấy rolling summary của hội thoại — hữu ích để hiển thị cho user."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' không tồn tại")
    return {
        "session_id": session_id,
        "summary": session.get("summary", ""),
        "message_count": len(session.get("history", [])),
        "summary_covers_messages": session.get("summary_message_count", 0),
    }


@app.get("/session/{session_id}/context")
async def get_session_context(session_id: str):
    """
    Debug endpoint — trả về context hint được trích xuất từ lịch sử.
    Hữu ích để kiểm tra xem chatbot đang hiểu ngữ cảnh đúng không.
    """
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' không tồn tại")

    agent_ctx = context_builder.build(session, "[debug]")
    hint = agent_ctx.rag_hint

    return {
        "session_id": session_id,
        "detected_districts": hint.districts,
        "detected_categories": hint.categories,
        "detected_star_ratings": hint.star_ratings,
        "detected_price_prefs": hint.price_prefs,
        "has_context": agent_ctx.has_context,
        "summary": session.get("summary", ""),
        "recent_history": session.get("history", [])[-6:],
    }


@app.post("/session/clear")
async def clear_session(data: SessionRequest):
    """Xóa session và toàn bộ history."""
    session_store.clear(data.session_id)
    return {"message": f"Đã xóa session '{data.session_id}'"}


# ─── Admin endpoints ──────────────────────────────────────────────────────────
@app.post("/admin/reset-supervisor")
async def admin_reset_supervisor():
    """Reset và tái khởi tạo supervisor agent."""
    reset_supervisor()
    get_supervisor_instance()
    return {"message": "Supervisor đã được reset và khởi tạo lại"}


@app.post("/admin/cleanup-sessions")
async def admin_cleanup_sessions():
    """Dọn dẹp sessions hết hạn (chỉ áp dụng với in-memory backend)."""
    from session_store import InMemorySessionStore
    if isinstance(session_store, InMemorySessionStore):
        cleaned = session_store.cleanup_expired()
        return {"message": f"Đã dọn dẹp {cleaned} sessions hết hạn"}
    return {"message": "Redis backend tự quản lý TTL, không cần cleanup thủ công"}


@app.get("/admin/sessions")
async def admin_list_sessions():
    """Liệt kê tất cả session đang hoạt động."""
    ids = session_store.list_all_ids()
    return {
        "active_session_count": len(ids),
        "session_ids": ids,
    }


# ─── Health & Root ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint."""
    supervisor_status = health_check()
    store_stats = session_store.get_stats()
    return {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "supervisor": supervisor_status,
        "session_store": store_stats,
    }


@app.get("/")
async def root():
    """Root endpoint — danh sách API."""
    return {
        "name": "Travel Chatbot RAG API",
        "version": "3.0.0",
        "endpoints": {
            "POST /ask": "Hỏi chatbot (có conversation memory & context)",
            "POST /ask/simple": "Hỏi chatbot (không memory, tương thích ngược)",
            "GET  /health": "Health check",
            "GET  /session/{id}/history": "Lịch sử hội thoại",
            "GET  /session/{id}/summary": "Tóm tắt hội thoại",
            "GET  /session/{id}/context": "Debug: context hints đã nhận biết",
            "POST /session/clear": "Xóa session",
            "GET  /admin/sessions": "Danh sách sessions đang hoạt động",
            "POST /admin/reset-supervisor": "Reset supervisor agent",
            "POST /admin/cleanup-sessions": "Dọn session hết hạn",
        },
    }
