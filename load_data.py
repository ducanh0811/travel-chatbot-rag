"""
load_data.py — Professional Vector Store Loader
=================================================
Pipeline nạp dữ liệu vào ChromaDB:
1. Đọc JSON từ thư mục data/ (absolute path)
2. Validate & deduplicate theo (name, district)
3. Dùng DocumentBuilder để tạo 2-3 semantic chunks / item
4. Upsert ChromaDB với stable SHA1 ID (chỉ update doc đã thay đổi)
5. Watchdog auto-reload khi file JSON thay đổi
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from dotenv import load_dotenv

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from document_builder import DocumentBuilder

# ─── Config ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("load_data")

BASE_DIR    = Path(__file__).parent.resolve()
DATA_DIR    = BASE_DIR / "data"
CHROMA_DIR  = str(BASE_DIR / "chromadb")
MAX_BACKUPS = 3   # Số backup giữ lại

DATA_FILES: Dict[str, Path] = {
    "hotel":       DATA_DIR / "hotel.json",
    "restaurant":  DATA_DIR / "restaurant.json",
    "destination": DATA_DIR / "destination.json",
    "cafe":        DATA_DIR / "cafe.json",
}
EVENT_FILE = DATA_DIR / "events.json"

# ─── API key ──────────────────────────────────────────────────────────────────
load_dotenv()

def _require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("⚠️  Thiếu OPENAI_API_KEY trong .env")
    return key


# ─── Data loading + validation ────────────────────────────────────────────────
def _load_json(path: Path) -> list:
    """Load JSON file với validation cơ bản."""
    if not path.exists():
        logger.warning("File không tồn tại: %s", path)
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("File %s phải là JSON array", path.name)
            return []
        return data
    except json.JSONDecodeError as e:
        logger.error("JSON lỗi trong %s: %s", path.name, e)
        return []


def _deduplicate(items: list, category: str) -> list:
    """Xóa duplicate theo (name, district). Log các entry bị bỏ."""
    seen: set = set()
    result: list = []
    for item in items:
        key = (
            str(item.get("name", "")).strip(),
            str(item.get("district", "")).strip(),
        )
        if key in seen:
            logger.debug("Duplicate bỏ qua [%s]: %s / %s", category, *key)
        else:
            seen.add(key)
            result.append(item)
    return result


def _validate_item(item: dict, category: str) -> bool:
    """Kiểm tra required fields. Trả False nếu thiếu name."""
    name = str(item.get("name", "")).strip()
    if not name or name.lower() in ("", "none", "null"):
        logger.warning("Item không có name trong category '%s', bỏ qua.", category)
        return False
    return True


def _load_category(category: str, path: Path) -> Tuple[List[Document], List[str]]:
    """Load 1 file JSON → Documents + IDs (parallel-safe)."""
    raw = _load_json(path)
    raw = _deduplicate(raw, category)

    builder = DocumentBuilder()
    all_docs, all_ids = [], []

    for item in raw:
        if not _validate_item(item, category):
            continue
        try:
            docs, ids = builder.build(item, category)
            all_docs.extend(docs)
            all_ids.extend(ids)
        except Exception as exc:
            logger.error("Lỗi build doc [%s] %s: %s",
                         category, item.get("name", "?"), exc)

    return all_docs, all_ids


def load_all_data() -> Tuple[List[Document], List[str]]:
    """
    Load song song tất cả data files.
    Returns (documents, stable_ids) — parallel lists.
    """
    all_docs: List[Document] = []
    all_ids:  List[str]      = []

    tasks: Dict[str, Path] = {**DATA_FILES, "event": EVENT_FILE}

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(_load_category, cat, path): cat
            for cat, path in tasks.items()
        }
        for future in as_completed(future_map):
            cat = future_map[future]
            try:
                docs, ids = future.result()
                all_docs.extend(docs)
                all_ids.extend(ids)
                logger.info("  %-12s → %d documents", cat, len(docs))
            except Exception as exc:
                logger.error("Load thất bại [%s]: %s", cat, exc)

    return all_docs, all_ids


# ─── ChromaDB ─────────────────────────────────────────────────────────────────
def _rotate_backups(persist_dir: str):
    """Giữ tối đa MAX_BACKUPS bản backup, xóa bản cũ nhất nếu quá."""
    parent = Path(persist_dir).parent
    stem   = Path(persist_dir).name
    backups = sorted(
        parent.glob(f"{stem}_backup_*"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(backups) >= MAX_BACKUPS:
        oldest = backups.pop(0)
        shutil.rmtree(oldest, ignore_errors=True)
        logger.info("🗑️  Xóa backup cũ: %s", oldest.name)


def save_chromadb(
    documents: List[Document],
    ids: List[str],
    persist_dir: str = CHROMA_DIR,
) -> Chroma:
    """
    Upsert documents vào ChromaDB.
    - Lần đầu: tạo mới + backup nếu đã có từ trước
    - Lần sau: chỉ add/update docs có ID mới hoặc thay đổi
    """
    _require_api_key()
    embedding = OpenAIEmbeddings()

    if os.path.exists(persist_dir):
        # Backup lần đầu (roll-based)
        _rotate_backups(persist_dir)
        backup_name = f"{persist_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(persist_dir, backup_name)
        logger.info("📦 Backup → %s", Path(backup_name).name)

    chroma = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
    )

    # Upsert theo batch để tránh rate limit
    BATCH = 100
    total = len(documents)
    for start in range(0, total, BATCH):
        batch_docs = documents[start : start + BATCH]
        batch_ids  = ids[start : start + BATCH]
        chroma.add_documents(documents=batch_docs, ids=batch_ids)
        logger.info("  Upserted %d/%d", min(start + BATCH, total), total)

    logger.info("✅ ChromaDB saved → %s (%d docs)", persist_dir, total)
    return chroma


# ─── Watchdog auto-reload ──────────────────────────────────────────────────────
class _DataFileHandler(FileSystemEventHandler):
    def __init__(self, callback, debounce: float = 5.0):
        self.callback        = callback
        self.debounce        = debounce
        self._timer: threading.Timer | None = None
        self._lock           = threading.Lock()

    def _trigger(self):
        with self._lock:
            self._timer = None
        logger.info("🔄 Phát hiện thay đổi, đang rebuild ChromaDB...")
        self.callback()

    def _schedule(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce, self._trigger)
            self._timer.start()

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            self._schedule()

    def on_created(self, event):
        self.on_modified(event)


def rebuild_chromadb():
    """Full rebuild ChromaDB từ dữ liệu mới nhất."""
    try:
        docs, ids = load_all_data()
        logger.info("📄 Tổng: %d documents (%d items)", len(docs), len(ids))
        save_chromadb(docs, ids)
        logger.info("🎉 Rebuild xong!")
    except Exception as exc:
        logger.error("❌ Rebuild thất bại: %s", exc)


def start_watcher(data_dir: str = str(DATA_DIR)) -> Observer:
    handler  = _DataFileHandler(rebuild_chromadb)
    observer = Observer()
    observer.schedule(handler, data_dir, recursive=False)
    observer.start()
    logger.info("👀 Watching: %s", data_dir)
    return observer


def stop_watcher(observer: Observer):
    if observer:
        observer.stop()
        observer.join()
        logger.info("🛑 Watcher stopped")


# ─── Stats ────────────────────────────────────────────────────────────────────
def print_stats():
    docs, ids = load_all_data()
    cats: Dict[str, int]   = {}
    dists: Dict[str, int]  = {}
    chunks: Dict[str, int] = {}

    for doc in docs:
        cat   = doc.metadata.get("category", "?")
        dist  = doc.metadata.get("district", "?")
        chunk = doc.metadata.get("chunk", "?")
        cats[cat]  = cats.get(cat, 0) + 1
        dists[dist] = dists.get(dist, 0) + 1
        chunks[chunk] = chunks.get(chunk, 0) + 1

    print("\n📊 Thống kê Documents:")
    print("-" * 45)
    print("Theo category:")
    for k, v in sorted(cats.items()):
        print(f"  • {k}: {v} chunks")
    print("\nTheo chunk type:")
    for k, v in sorted(chunks.items()):
        print(f"  • {k}: {v}")
    print("\nTheo quận (top 10):")
    for k, v in sorted(dists.items(), key=lambda x: -x[1])[:10]:
        print(f"  • {k}: {v}")
    print("-" * 45)
    print(f"Tổng: {len(docs)} chunks từ ~{len(ids)//2} địa điểm (est.)")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"

    if cmd == "watch":
        logger.info("🚀 Watch mode — rebuild on JSON change")
        rebuild_chromadb()
        observer = start_watcher()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_watcher(observer)

    elif cmd == "stats":
        print_stats()

    elif cmd == "build":
        logger.info("🔄 Building ChromaDB...")
        docs, ids = load_all_data()
        logger.info("📄 %d documents sẽ được upsert", len(docs))
        print_stats()
        save_chromadb(docs, ids)

    else:
        print(f"Lệnh không hợp lệ: {cmd}")
        print("Hỗ trợ: build | watch | stats")
