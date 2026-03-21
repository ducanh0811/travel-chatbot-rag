"""
document_builder.py
====================
Chuyển đổi một JSON item thành 2–3 Document chunks theo semantic aspect,
với stable SHA1 ID để hỗ trợ upsert vào ChromaDB.

Chunking strategy:
  Chunk 1 — Identity & Description  : tên, loại, khu vực, mô tả
  Chunk 2 — Practical Info           : giờ mở/đóng, thời gian gợi ý
  Chunk 3 — Tags / Audience          : tags, phù hợp với ai (nếu có tags)

Mỗi chunk đều chứa tên địa điểm để không mất ngữ cảnh khi retrieve.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _v(value, fallback: str = "") -> str:
    """Trả về giá trị string sạch, bỏ qua None và chuỗi rỗng."""
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s and s.lower() not in ("none", "null", "nan") else fallback


def _make_id(name: str, category: str, district: str, chunk_index: int) -> str:
    """Stable SHA1 ID — deterministic, dùng được cho Chroma upsert."""
    raw = f"{name.strip()}|{category}|{district.strip()}|{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _duration_text(minutes) -> str:
    if minutes is None:
        return ""
    try:
        m = int(minutes)
        if m >= 60:
            h, rem = divmod(m, 60)
            return f"{h} giờ {rem} phút" if rem else f"{h} giờ"
        return f"{m} phút"
    except (ValueError, TypeError):
        return ""


# ─── Builder ──────────────────────────────────────────────────────────────────
class DocumentBuilder:
    """
    Chuyển đổi một dict item JSON thành danh sách Document chunks.

    Usage:
        builder = DocumentBuilder()
        docs, ids = builder.build(item, category="cafe")
    """

    def build(self, item: dict, category: str) -> tuple[List[Document], List[str]]:
        """
        Args:
            item    : dict từ JSON data
            category: "hotel" | "restaurant" | "cafe" | "destination" | "event"

        Returns:
            (docs, ids) — parallel lists; ids là stable SHA1
        """
        name     = _v(item.get("name"), "Không rõ tên")
        district = _v(item.get("district"), "")
        ward     = _v(item.get("ward"), "")
        item_type = _v(item.get("type"), category)
        desc     = _v(item.get("description"), "")
        tags     = [t for t in item.get("tags", []) if isinstance(t, str) and t.strip()]
        tags_str = ", ".join(t.strip() for t in tags) if tags else ""

        # Base metadata — tất cả string, không None (Chroma yêu cầu)
        base_meta: dict = {
            "category": category,
            "name":     name,
            "type":     item_type,
            "district": district,
            "ward":     ward,
            "tags_str": tags_str,
        }

        if category == "event":
            return self._build_event(item, name, district, ward, item_type,
                                     desc, tags_str, base_meta)
        else:
            return self._build_place(item, name, category, district, ward,
                                     item_type, desc, tags_str, base_meta)

    # ── Place (hotel / restaurant / cafe / destination) ───────────────────────
    def _build_place(self, item, name, category, district, ward,
                     item_type, desc, tags_str, base_meta) -> tuple:
        open_t  = _v(item.get("open_time"), "")
        close_t = _v(item.get("close_time"), "")
        dur     = _duration_text(item.get("duration_suggested_min"))

        # ── Chunk 1: Identity & Description ──────────────────────────────────
        location_str = f"phường {ward}, quận {district}" if ward and district \
                       else (f"quận {district}" if district else "")
        chunk1_parts = [f"{name} là {item_type}"]
        if location_str:
            chunk1_parts.append(f"tại {location_str}")
        chunk1_parts.append(".")
        if desc:
            chunk1_parts.append(f" {desc}")

        doc1 = Document(
            page_content=" ".join(chunk1_parts).strip(),
            metadata={**base_meta, "chunk": "identity"},
        )

        # ── Chunk 2: Practical Info ───────────────────────────────────────────
        practical_parts = [f"{name} —"]
        has_practical = False

        if open_t or close_t:
            time_str = f"mở cửa {open_t}" if open_t else ""
            if close_t:
                time_str += f", đóng cửa {close_t}" if time_str else f"đóng cửa {close_t}"
            if category == "hotel":
                time_str = time_str.replace("mở cửa", "check-in").replace("đóng cửa", "check-out")
            practical_parts.append(time_str + ".")
            has_practical = True

        if dur:
            practical_parts.append(f"Thời gian tham quan gợi ý: {dur}.")
            has_practical = True

        if has_practical:
            doc2 = Document(
                page_content=" ".join(practical_parts).strip(),
                metadata={**base_meta, "chunk": "practical",
                          "open_time": open_t, "close_time": close_t},
            )
        else:
            doc2 = None

        # ── Chunk 3: Tags / Audience ──────────────────────────────────────────
        if tags_str:
            tag_content = (
                f"{name} phù hợp cho: {tags_str}. "
                f"Loại: {item_type}. "
                f"{'Khu vực: quận ' + district + '.' if district else ''}"
            ).strip()
            doc3 = Document(
                page_content=tag_content,
                metadata={**base_meta, "chunk": "tags"},
            )
        else:
            doc3 = None

        docs, ids = [], []
        for i, doc in enumerate([doc1, doc2, doc3]):
            if doc is not None:
                docs.append(doc)
                ids.append(_make_id(name, category, district, i))

        return docs, ids

    # ── Event ─────────────────────────────────────────────────────────────────
    def _build_event(self, item, name, district, ward, item_type,
                     desc, tags_str, base_meta) -> tuple:
        start   = _v(item.get("start_date"), "")
        end     = _v(item.get("end_date"), "")
        time_   = _v(item.get("time"), "")
        month   = item.get("month")
        recurring = _v(item.get("recurring"), "")

        month_meta = str(month) if month is not None else ""
        extra_meta = {
            **base_meta,
            "start_date": start,
            "end_date":   end,
            "month":      month_meta,
        }

        # Chunk 1: Identity
        location_str = f"phường {ward}, quận {district}" if ward and district \
                       else (f"quận {district}" if district else "")
        c1 = (
            f"Sự kiện: {name} — {item_type}. "
            f"{'Địa điểm: ' + location_str + '. ' if location_str else ''}"
            f"{desc}"
        ).strip()
        doc1 = Document(page_content=c1, metadata={**extra_meta, "chunk": "identity"})

        # Chunk 2: Timing
        timing_parts = [f"{name} —"]
        if start and end:
            timing_parts.append(f"diễn ra từ {start} đến {end}")
            if time_:
                timing_parts.append(f"lúc {time_}")
            timing_parts.append(".")
        if month:
            timing_parts.append(f"Tháng diễn ra: tháng {month}.")
        if recurring:
            timing_parts.append(f"Tính chất: {recurring}.")

        doc2 = Document(
            page_content=" ".join(timing_parts).strip(),
            metadata={**extra_meta, "chunk": "timing"},
        )

        docs = [doc1, doc2]
        ids  = [_make_id(name, "event", district, i) for i in range(len(docs))]

        # Chunk 3: Tags (nếu có)
        if tags_str:
            c3 = f"Sự kiện {name} gắn với: {tags_str}. Loại: {item_type}."
            doc3 = Document(page_content=c3, metadata={**extra_meta, "chunk": "tags"})
            docs.append(doc3)
            ids.append(_make_id(name, "event", district, 2))

        return docs, ids


# ─── Standalone smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    builder = DocumentBuilder()

    sample_cafe = {
        "name": "Ibasho Coffee",
        "type": "quán cà phê",
        "description": "Không gian tối giản phong cách Hàn, có gác lửng, rất chill.",
        "district": "Thanh Khê",
        "open_time": "07:00",
        "close_time": "22:00",
        "tags": ["tối giản", "hiện đại"],
        "duration_suggested_min": 60,
        "ward": "Thanh Khê Đông",
    }

    sample_hotel = {
        "name": "Hilton Da Nang",
        "type": "5 sao",
        "description": "Khách sạn cao cấp, hồ bơi, view sông Hàn",
        "district": "Hải Châu",
        "open_time": None,
        "close_time": None,
        "tags": ["luxury", "riverview", "pool"],
        "duration_suggested_min": None,
        "ward": "Phường Bình Thuận",
    }

    sample_event = {
        "name": "Lễ hội pháo hoa quốc tế Đà Nẵng (DIFF)",
        "type": "lễ hội",
        "description": "Lễ hội pháo hoa lớn nhất Việt Nam.",
        "district": "Hải Châu",
        "start_date": "2025-06-01",
        "end_date": "2025-07-06",
        "time": "20:00",
        "tags": ["pháo hoa", "quốc tế"],
        "ward": "Phường Phước Ninh",
        "recurring": "yearly",
        "month": 6,
    }

    for item, cat in [(sample_cafe, "cafe"), (sample_hotel, "hotel"), (sample_event, "event")]:
        docs, ids = builder.build(item, cat)
        print(f"\n{'='*60}")
        print(f"[{cat.upper()}] {item['name']} → {len(docs)} chunks")
        for i, (doc, doc_id) in enumerate(zip(docs, ids)):
            print(f"\n  Chunk {i+1} [{doc.metadata.get('chunk')}] id={doc_id[:12]}...")
            print(f"  Content: {doc.page_content[:120]}")
            print(f"  Metadata keys: {list(doc.metadata.keys())}")
