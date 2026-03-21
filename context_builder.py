"""
context_builder.py
==================
Xây dựng context thông minh từ lịch sử hội thoại để:
1. Tăng chất lượng trả lời agent (hiểu follow-up questions)
2. Cải thiện RAG retrieval bằng cách nhúng hint về khu vực/loại địa điểm
   mà người dùng đã đề cập trước đó.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ─── Keyword maps (mirror của rag.py để detect context) ───────────────────────
_DISTRICT_PATTERNS = {
    "sơn trà": "Sơn Trà",
    "son tra": "Sơn Trà",
    "hải châu": "Hải Châu",
    "hai chau": "Hải Châu",
    "thanh khê": "Thanh Khê",
    "thanh khe": "Thanh Khê",
    "liên chiểu": "Liên Chiểu",
    "lien chieu": "Liên Chiểu",
    "ngũ hành sơn": "Ngũ Hành Sơn",
    "ngu hanh son": "Ngũ Hành Sơn",
    "cẩm lệ": "Cẩm Lệ",
    "cam le": "Cẩm Lệ",
    "hòa vang": "Hòa Vang",
    "hoa vang": "Hòa Vang",
}

_CATEGORY_PATTERNS = {
    "cafe": "cafe",
    "cà phê": "cafe",
    "coffee": "cafe",
    "quán cafe": "cafe",
    "quán cà phê": "cafe",
    "khách sạn": "hotel",
    "hotel": "hotel",
    "resort": "hotel",
    "homestay": "hotel",
    "nhà hàng": "restaurant",
    "quán ăn": "restaurant",
    "restaurant": "restaurant",
    "địa điểm": "destination",
    "bãi biển": "destination",
    "biển": "destination",
    "chùa": "destination",
    "bảo tàng": "destination",
    "sự kiện": "event",
    "lễ hội": "event",
    "festival": "event",
}

_STAR_PATTERN = re.compile(r"\b([2-5])\s*sao\b")
_PRICE_KEYWORDS = ["ngân sách", "giá rẻ", "bình dân", "tiết kiệm", "cao cấp", "luxury", "sang trọng"]


# ─── Dataclass ────────────────────────────────────────────────────────────────
@dataclass
class ContextHint:
    """Thông tin ngữ cảnh trích xuất từ lịch sử hội thoại."""
    districts: List[str] = field(default_factory=list)     # Khu vực đã nhắc
    categories: List[str] = field(default_factory=list)    # Loại địa điểm đã hỏi
    star_ratings: List[str] = field(default_factory=list)  # Hạng sao đã đề cập
    price_prefs: List[str] = field(default_factory=list)   # Yêu cầu giá
    raw_topics: List[str] = field(default_factory=list)    # Từ khóa địa điểm cụ thể

    def is_empty(self) -> bool:
        return not any([self.districts, self.categories, self.star_ratings, self.price_prefs])

    def to_hint_text(self) -> str:
        """Chuyển hint thành text ngắn để nhúng vào RAG query."""
        parts = []
        if self.districts:
            parts.append(f"khu vực {', '.join(self.districts)}")
        if self.categories:
            parts.append(f"loại {', '.join(self.categories)}")
        if self.star_ratings:
            parts.append(", ".join(self.star_ratings))
        if self.price_prefs:
            parts.append(self.price_prefs[0])
        return ", ".join(parts) if parts else ""


@dataclass
class AgentContext:
    """Context đầy đủ để gửi vào agent."""
    original_query: str
    enhanced_query: str              # Query + lịch sử hội thoại
    rag_hint: ContextHint            # Hint cho RAG retrieval
    summary_message: Optional[str]   # Rolling summary (làm system message)
    has_context: bool                # Có context hay không


# ─── Core logic ───────────────────────────────────────────────────────────────
class ContextBuilder:
    """
    Xây dựng AgentContext từ session data và query hiện tại.

    Luồng hoạt động:
    1. Đọc `summary` (rolling) và `history` gần nhất từ session
    2. Scan toàn bộ history để trích xuất ContextHint (district, category,...)
    3. Ghép lịch sử + query thành enhanced_query cho agent
    4. Trả về AgentContext đủ thông tin
    """

    def __init__(self, max_recent_turns: int = 3):
        """
        Args:
            max_recent_turns: Số lượt hội thoại gần nhất đưa vào enhanced_query.
        """
        self.max_recent_turns = max_recent_turns

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, session_data: dict, query: str) -> AgentContext:
        """
        Xây dựng AgentContext từ session data và query mới.

        Args:
            session_data: Dict trả về từ SessionStore.get_or_create()
            query: Câu hỏi mới của người dùng

        Returns:
            AgentContext đầy đủ
        """
        history: list = session_data.get("history", [])
        summary: str = session_data.get("summary", "")

        # 1. Trích hint từ toàn bộ lịch sử
        rag_hint = self._extract_hint_from_history(history)

        # 2. Lấy lịch sử gần nhất để đưa vào context
        recent = history[-(self.max_recent_turns * 2):]
        has_context = bool(recent or summary)

        # 3. Xây enhanced_query
        enhanced_query = self._build_enhanced_query(query, recent, rag_hint)

        # 4. Summary message (chỉ khi có summary thực chất)
        summary_message = (
            f"Tóm tắt hội thoại trước:\n{summary}" if summary.strip() else None
        )

        return AgentContext(
            original_query=query,
            enhanced_query=enhanced_query,
            rag_hint=rag_hint,
            summary_message=summary_message,
            has_context=has_context,
        )

    def enrich_rag_query(self, query: str, hint: ContextHint) -> str:
        """
        Nhúng ContextHint vào RAG query để cải thiện retrieval.
        Agent sẽ gọi rag_tool với query này thay vì raw query.

        Ví dụ:
            query = "Có chỗ nào view biển không?"
            hint  = ContextHint(districts=["Sơn Trà"], categories=["cafe"])
            →     = "Có chỗ nào view biển không? [Ngữ cảnh: khu vực Sơn Trà, loại cafe]"
        """
        hint_text = hint.to_hint_text()
        if not hint_text:
            return query
        return f"{query} [Ngữ cảnh hội thoại: {hint_text}]"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_hint_from_history(self, history: list) -> ContextHint:
        """Quét lịch sử để trích xuất các signal ngữ cảnh."""
        hint = ContextHint()
        seen_districts: set = set()
        seen_categories: set = set()

        for msg in history:
            text = msg.get("content", "").lower()
            if not text:
                continue

            # Districts
            for kw, district in _DISTRICT_PATTERNS.items():
                if kw in text and district not in seen_districts:
                    hint.districts.append(district)
                    seen_districts.add(district)

            # Categories
            for kw, cat in sorted(_CATEGORY_PATTERNS.items(), key=lambda x: -len(x[0])):
                if kw in text and cat not in seen_categories:
                    hint.categories.append(cat)
                    seen_categories.add(cat)

            # Star ratings
            for m in _STAR_PATTERN.finditer(text):
                rating = f"{m.group(1)} sao"
                if rating not in hint.star_ratings:
                    hint.star_ratings.append(rating)

            # Price preferences (lấy đầu tiên tìm thấy)
            if not hint.price_prefs:
                for kw in _PRICE_KEYWORDS:
                    if kw in text:
                        hint.price_prefs.append(kw)
                        break

        return hint

    def _build_enhanced_query(
        self, query: str, recent_history: list, hint: ContextHint
    ) -> str:
        """
        Ghép lịch sử gần nhất vào query để agent hiểu follow-up questions.
        """
        if not recent_history:
            return query

        lines = []
        for msg in recent_history:
            role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
            content = msg.get("content", "").strip()
            if content:
                # Cắt bớt nếu quá dài (tránh blow context window)
                if len(content) > 300:
                    content = content[:300] + "..."
                lines.append(f"{role}: {content}")

        history_block = "\n".join(lines)

        enhanced = (
            f"Lịch sử hội thoại gần đây:\n"
            f"{history_block}\n\n"
            f"Câu hỏi mới: {query}\n\n"
            f"Hãy trả lời câu hỏi mới. Nếu câu hỏi mới là follow-up "
            f"(ví dụ: 'ở đó', 'chỗ đó', 'loại đó'), hãy tham chiếu "
            f"ngữ cảnh lịch sử để hiểu đúng ý người dùng."
        )
        return enhanced


# ─── Singleton ────────────────────────────────────────────────────────────────
_default_builder: Optional[ContextBuilder] = None


def get_context_builder() -> ContextBuilder:
    global _default_builder
    if _default_builder is None:
        _default_builder = ContextBuilder(max_recent_turns=3)
    return _default_builder
