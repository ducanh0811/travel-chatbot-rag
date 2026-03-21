"""
tests/test_context_builder.py
==============================
Unit tests cho ContextBuilder — không cần OpenAI key.
"""
import sys
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from context_builder import ContextBuilder, ContextHint


# ─── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def builder():
    return ContextBuilder(max_recent_turns=3)


def _make_session(messages: list) -> dict:
    """Tạo session data giả từ danh sách (role, content)."""
    import time
    history = [
        {"role": r, "content": c, "timestamp": "2026-01-01T00:00:00"}
        for r, c in messages
    ]
    now = time.time()
    return {
        "id": "test-session",
        "history": history,
        "summary": "",
        "summary_message_count": 0,
        "created_at": now,
        "last_access": now,
    }


# ─── Tests: ContextHint extraction ───────────────────────────────────────────
class TestExtractHint:
    def test_detects_district_son_tra(self, builder):
        session = _make_session([("user", "Tôi muốn tìm cafe ở Sơn Trà")])
        ctx = builder.build(session, "Chỗ nào đẹp?")
        assert "Sơn Trà" in ctx.rag_hint.districts

    def test_detects_district_hai_chau(self, builder):
        session = _make_session([("user", "Khách sạn nào đẹp ở Hải Châu?")])
        ctx = builder.build(session, "Gợi ý thêm")
        assert "Hải Châu" in ctx.rag_hint.districts

    def test_detects_category_cafe(self, builder):
        session = _make_session([("user", "Gợi ý quán cafe đẹp")])
        ctx = builder.build(session, "Chỗ có wifi không?")
        assert "cafe" in ctx.rag_hint.categories

    def test_detects_category_hotel(self, builder):
        session = _make_session([("user", "Tìm khách sạn 5 sao gần biển")])
        ctx = builder.build(session, "Giá bao nhiêu?")
        assert "hotel" in ctx.rag_hint.categories
        assert "5 sao" in ctx.rag_hint.star_ratings

    def test_detects_star_rating_from_regex(self, builder):
        session = _make_session([("user", "Resort 4 sao ở Ngũ Hành Sơn")])
        ctx = builder.build(session, "Gợi ý thêm")
        assert "4 sao" in ctx.rag_hint.star_ratings

    def test_detects_price_preference(self, builder):
        session = _make_session([("user", "Tôi có ngân sách thấp, cafe nào giá rẻ?")])
        ctx = builder.build(session, "Ở đâu gần trung tâm?")
        assert len(ctx.rag_hint.price_prefs) > 0

    def test_empty_history_gives_empty_hint(self, builder):
        session = _make_session([])
        ctx = builder.build(session, "Gợi ý cafe")
        assert ctx.rag_hint.is_empty()
        assert not ctx.has_context


class TestHintDeduplication:
    def test_same_district_not_duplicated(self, builder):
        session = _make_session([
            ("user", "Cafe ở Sơn Trà"),
            ("assistant", "Đây là các quán cafe Sơn Trà..."),
            ("user", "Còn quán nào ở Sơn Trà nữa không?"),
        ])
        ctx = builder.build(session, "Loại nào có view đẹp?")
        assert ctx.rag_hint.districts.count("Sơn Trà") == 1

    def test_multiple_districts_detected(self, builder):
        session = _make_session([
            ("user", "Cafe ở Sơn Trà"),
            ("user", "Còn ở Hải Châu thì sao?"),
        ])
        ctx = builder.build(session, "So sánh hai khu")
        assert "Sơn Trà" in ctx.rag_hint.districts
        assert "Hải Châu" in ctx.rag_hint.districts


# ─── Tests: Enhanced query ────────────────────────────────────────────────────
class TestEnhancedQuery:
    def test_empty_history_returns_original(self, builder):
        session = _make_session([])
        ctx = builder.build(session, "Gợi ý cafe")
        assert ctx.enhanced_query == "Gợi ý cafe"

    def test_with_history_wraps_query(self, builder):
        session = _make_session([
            ("user", "Tôi muốn cafe view biển"),
            ("assistant", "Dưới đây là gợi ý..."),
        ])
        ctx = builder.build(session, "Chỗ nào có wifi?")
        assert "Lịch sử hội thoại" in ctx.enhanced_query
        assert "Chỗ nào có wifi?" in ctx.enhanced_query

    def test_summary_message_set_when_summary_exists(self, builder):
        import time
        session = {
            "id": "x",
            "history": [],
            "summary": "Người dùng tìm cafe view biển ở Sơn Trà.",
            "summary_message_count": 2,
            "created_at": time.time(),
            "last_access": time.time(),
        }
        ctx = builder.build(session, "Có chỗ nào còn trống không?")
        assert ctx.summary_message is not None
        assert "Sơn Trà" in ctx.summary_message

    def test_no_summary_message_when_empty(self, builder):
        session = _make_session([])
        ctx = builder.build(session, "Gợi ý cafe")
        assert ctx.summary_message is None


# ─── Tests: enrich_rag_query ─────────────────────────────────────────────────
class TestEnrichRagQuery:
    def test_appends_hint_to_query(self, builder):
        hint = ContextHint(
            districts=["Sơn Trà"],
            categories=["cafe"],
        )
        result = builder.enrich_rag_query("Có chỗ nào view biển không?", hint)
        assert "Sơn Trà" in result
        assert "cafe" in result
        assert "Ngữ cảnh hội thoại" in result

    def test_empty_hint_returns_original(self, builder):
        hint = ContextHint()
        result = builder.enrich_rag_query("Gợi ý cafe", hint)
        assert result == "Gợi ý cafe"


# ─── Tests: ContextHint.to_hint_text ─────────────────────────────────────────
class TestContextHintToText:
    def test_all_fields(self):
        hint = ContextHint(
            districts=["Sơn Trà"],
            categories=["cafe"],
            star_ratings=["5 sao"],
            price_prefs=["ngân sách"],
        )
        text = hint.to_hint_text()
        assert "Sơn Trà" in text
        assert "cafe" in text

    def test_empty_hint_returns_empty_string(self):
        hint = ContextHint()
        assert hint.to_hint_text() == ""
        assert hint.is_empty()
