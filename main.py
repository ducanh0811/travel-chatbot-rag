from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
import logging
import time
import re
from Supervisor import (
    get_supervisor_instance,
    get_weather_agent_instance,
    get_travel_agent_instance,
    reset_supervisor,
    health_check,
    classify_query,
)
from summarizer import ConversationSummarizer
import uuid
import threading

logger = logging.getLogger("travel-chatbot")
logging.basicConfig(level=logging.INFO)

MAX_QUERY_LEN = 500

app = FastAPI(
    title="Travel Chatbot RAG API",
    description="API cho chatbot du lịch Đà Nẵng với RAG và multi-agent",
    version="2.0.0"
)

# ============ CONVERSATION MEMORY ============
class ConversationMemory:
    """
    Quản lý memory cho các cuộc hội thoại.
    Lưu trữ lịch sử chat theo session_id.
    """
    def __init__(self, max_history: int = 10, ttl_seconds: int = 3600):
        self.sessions: Dict[str, Dict] = {}
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
    
    def get_or_create_session(self, session_id: str) -> Dict:
        """Lấy hoặc tạo session mới"""
        with self._lock:
            now = datetime.now().timestamp()
            
            if session_id in self.sessions:
                session = self.sessions[session_id]
                # Kiểm tra TTL
                if now - session["last_access"] > self.ttl_seconds:
                    # Session hết hạn, tạo mới
                    session = self._create_new_session(session_id)
                else:
                    session["last_access"] = now
            else:
                session = self._create_new_session(session_id)
            
            return session
    
    def _create_new_session(self, session_id: str) -> Dict:
        """Tạo session mới"""
        session = {
            "id": session_id,
            "history": [],
            "summary": "",
            "summary_message_count": 0,
            "created_at": datetime.now().timestamp(),
            "last_access": datetime.now().timestamp()
        }
        self.sessions[session_id] = session
        return session
    
    def add_message(self, session_id: str, role: str, content: str):
        """Thêm message vào history"""
        with self._lock:
            if session_id in self.sessions:
                history = self.sessions[session_id]["history"]
                history.append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                })
                # Giới hạn history
                if len(history) > self.max_history * 2:
                    self.sessions[session_id]["history"] = history[-self.max_history * 2:]
    
    def get_context(self, session_id: str, last_n: int = 5) -> str:
        """Lấy context từ history gần nhất"""
        with self._lock:
            if session_id not in self.sessions:
                return ""
            
            history = self.sessions[session_id]["history"][-last_n * 2:]
            if not history:
                return ""
            
            context_parts = []
            for msg in history:
                role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                context_parts.append(f"{role}: {msg['content']}")
            
            return "\n".join(context_parts)

    def get_summary(self, session_id: str) -> str:
        """Lấy tóm tắt hội thoại của session"""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return ""
            return session.get("summary", "")

    def update_summary(self, session_id: str, summarizer: ConversationSummarizer, min_new_messages: int = 4) -> str:
        """Cập nhật tóm tắt hội thoại bằng summarizer"""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return ""
            history = list(session.get("history", []))
            previous_summary = session.get("summary", "")
            last_count = session.get("summary_message_count", 0)

        if len(history) - last_count < min_new_messages:
            return previous_summary

        recent_messages = history[-summarizer.max_recent_messages :]
        try:
            new_summary = summarizer.summarize(recent_messages, previous_summary)
        except Exception:
            return previous_summary

        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return previous_summary
            session["summary"] = new_summary
            session["summary_message_count"] = len(history)
        return new_summary
    
    def clear_session(self, session_id: str):
        """Xóa session"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
    
    def cleanup_expired(self):
        """Dọn dẹp các session hết hạn"""
        with self._lock:
            now = datetime.now().timestamp()
            expired = [
                sid for sid, session in self.sessions.items()
                if now - session["last_access"] > self.ttl_seconds
            ]
            for sid in expired:
                del self.sessions[sid]
            return len(expired)
    
    def get_stats(self) -> Dict:
        """Lấy thống kê memory"""
        with self._lock:
            return {
                "active_sessions": len(self.sessions),
                "max_history": self.max_history,
                "ttl_seconds": self.ttl_seconds
            }

# Global memory instance
conversation_memory = ConversationMemory()
conversation_summarizer = ConversationSummarizer()

# ============ REQUEST/RESPONSE MODELS ============
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    use_context: bool = True  # Có sử dụng context từ history không

class QueryResponse(BaseModel):
    result: str
    session_id: str
    has_context: bool

class SessionRequest(BaseModel):
    session_id: str

def normalize_query(raw: str) -> str:
    return (raw or "").strip()

def validate_query(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query không được để trống")
    if len(query) > MAX_QUERY_LEN:
        raise HTTPException(status_code=400, detail=f"Query quá dài (>{MAX_QUERY_LEN} ký tự)")

# ============ STARTUP ============
@app.on_event("startup")
async def startup_event():
    """Khởi tạo supervisor khi server start"""
    print("Starting Travel Chatbot RAG API...")
    # Pre-warm supervisor
    get_supervisor_instance()
    print("API is ready.")

# ============ ENDPOINTS ============
@app.post("/ask", response_model=QueryResponse)
async def ask_agent(data: QueryRequest):
    """
    Endpoint chính để hỏi chatbot.
    Hỗ trợ conversation memory qua session_id.
    """
    query = normalize_query(data.query)
    validate_query(query)
    
    # Tạo hoặc lấy session
    session_id = data.session_id or str(uuid.uuid4())
    session = conversation_memory.get_or_create_session(session_id)
    
    # Xây dựng query với context nếu cần
    enhanced_query = query
    has_context = False
    
    summary_message = None
    if data.use_context:
        summary = conversation_memory.get_summary(session_id)
        context = conversation_memory.get_context(session_id, last_n=3)
        if summary:
            summary_message = f"Tóm tắt hội thoại trước đó:\n{summary}"
        if context:
            enhanced_query = f"""
Lịch sử hội thoại gần đây:
{context}

Câu hỏi mới: {query}

Hãy trả lời câu hỏi mới, có thể tham khảo lịch sử hội thoại nếu liên quan.
"""
            has_context = True
    
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
        if summary_message:
            messages_payload.append({"role": "assistant", "content": summary_message})
        messages_payload.append({"role": "user", "content": enhanced_query})

        result = agent.invoke({"messages": messages_payload})
        messages = result.get("messages", [])
        final_response = None

        for msg in reversed(messages):
            content = getattr(msg, "content", msg)
            if content and not any(kw in str(content).lower() for kw in ["transferred to", "transferring", "successfully transfer"]):
                if content != query and content != enhanced_query:
                    cleaned = str(content)
                    if "<internal>" in cleaned.lower():
                        blocks = re.findall(
                            r"<internal>(.*?)</internal>",
                            cleaned,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        if blocks:
                            cleaned = "\n\n".join(block.strip() for block in blocks if block.strip())
                        else:
                            cleaned = cleaned.replace("<internal>", "").replace("</internal>", "")
                    cleaned = cleaned.strip()
                    if cleaned:
                        final_response = cleaned
                        break

        if not final_response:
            final_response = "❌ Không có phản hồi nội dung từ agent."
        
        # Lưu vào memory
        conversation_memory.add_message(session_id, "user", query)
        conversation_memory.add_message(session_id, "assistant", final_response)
        conversation_memory.update_summary(session_id, conversation_summarizer)

        response = QueryResponse(
            result=final_response,
            session_id=session_id,
            has_context=has_context
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("route=%s session=%s ctx=%s ms=%s", route, session_id, has_context, elapsed_ms)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/simple")
async def ask_simple(data: QueryRequest):
    """
    Endpoint đơn giản (không memory) - tương thích ngược.
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
        final_response = None

        for msg in reversed(messages):
            content = getattr(msg, "content", msg)
            if content and not any(kw in str(content).lower() for kw in ["transferred to", "transferring", "successfully transfer"]):
                if content != query:
                    cleaned = str(content)
                    if "<internal>" in cleaned.lower():
                        blocks = re.findall(
                            r"<internal>(.*?)</internal>",
                            cleaned,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        if blocks:
                            cleaned = "\n\n".join(block.strip() for block in blocks if block.strip())
                        else:
                            cleaned = cleaned.replace("<internal>", "").replace("</internal>", "")
                    cleaned = cleaned.strip()
                    if cleaned:
                        final_response = cleaned
                        break

        if not final_response:
            return {"result": "❌ Không có phản hồi nội dung từ agent."}

        response = {"result": final_response}
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("route=%s session=none ctx=false ms=%s", route, elapsed_ms)
        return response

    except Exception as e:
        return {"error": str(e)}

@app.post("/session/clear")
async def clear_session(data: SessionRequest):
    """Xóa session và history"""
    conversation_memory.clear_session(data.session_id)
    return {"message": f"Đã xóa session {data.session_id}"}

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Lấy lịch sử hội thoại của session"""
    session = conversation_memory.get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "history": session["history"],
        "created_at": datetime.fromtimestamp(session["created_at"]).isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    supervisor_status = health_check()
    memory_stats = conversation_memory.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "supervisor": supervisor_status,
        "memory": memory_stats
    }

@app.post("/admin/reset-supervisor")
async def admin_reset_supervisor():
    """Reset supervisor (admin only)"""
    reset_supervisor()
    # Re-initialize
    get_supervisor_instance()
    return {"message": "Supervisor đã được reset và khởi tạo lại"}

@app.post("/admin/cleanup-sessions")
async def admin_cleanup_sessions():
    """Dọn dẹp sessions hết hạn"""
    cleaned = conversation_memory.cleanup_expired()
    return {"message": f"Đã dọn dẹp {cleaned} sessions hết hạn"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Travel Chatbot RAG API",
        "version": "2.0.0",
        "endpoints": {
            "POST /ask": "Hỏi chatbot (có memory)",
            "POST /ask/simple": "Hỏi chatbot (không memory)",
            "GET /health": "Health check",
            "POST /session/clear": "Xóa session",
            "GET /session/{id}/history": "Lấy lịch sử session"
        }
    }
