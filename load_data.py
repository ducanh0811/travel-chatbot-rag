import os
import json
import shutil
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import threading

# ====== Load API Key ======
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("⚠️ Vui lòng đặt biến môi trường OPENAI_API_KEY trong file .env của bạn.")

# ====== Load dữ liệu địa điểm (cafe, hotel, restaurant, destination) ======
def load_place_data(data_type: str, path: str) -> list:
    """Load dữ liệu địa điểm từ file JSON"""
    if not os.path.exists(path):
        print(f"⚠️ Không tìm thấy file: {path}")
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    docs = []
    for item in items:
        name = item.get("name", "Không rõ")
        desc = item.get("description", "")
        district = item.get("district", "")
        item_type = item.get("type", "")
        ward = item.get("ward", "")
        
        if data_type == "hotel":
            open_time = item.get("Check-in hour", item.get("checkin_time", "Không rõ"))
            close_time = item.get("Check-out hour", item.get("checkout_time", "Không rõ"))
        else:
            open_time = item.get("open_time", "")
            close_time = item.get("close_time", "")
        
        tags = item.get("tags", [])
        suggested = item.get("duration_suggested_min", "Không rõ")
        
        content = (
            f"{name} ({data_type}) - phường {ward}, quận {district}. "
            f"Loại: {item_type}. "
            f"Mô tả: {desc}. "
            f"Giờ mở: {open_time}, đóng: {close_time}. "
            f"Thời gian gợi ý: {suggested} phút. "
            f"Tags: {', '.join(tags) if tags else 'Không có'}"
        )
        
        doc = Document(
            page_content=content,
            metadata={
                "type": item_type or data_type,
                "name": name,
                "district": district or None,
                "ward": ward or None,
                "category": data_type  # Thêm category để phân biệt
            }
        )
        docs.append(doc)
    
    return docs

# ====== Load dữ liệu sự kiện ======
def load_event_data(path: str = "data/events.json") -> list:
    """Load dữ liệu sự kiện từ file JSON"""
    if not os.path.exists(path):
        print(f"⚠️ Không tìm thấy file sự kiện: {path}")
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    docs = []
    for item in items:
        name = item.get("name", "Không rõ")
        desc = item.get("description", "")
        district = item.get("district", "")
        event_type = item.get("type", "sự kiện")
        ward = item.get("ward", "")
        start_date = item.get("start_date", "")
        end_date = item.get("end_date", "")
        event_time = item.get("time", "")
        tags = item.get("tags", [])
        recurring = item.get("recurring", "")
        month = item.get("month", "")
        
        content = (
            f"Sự kiện: {name} - {event_type}. "
            f"Địa điểm: phường {ward}, quận {district}. "
            f"Mô tả: {desc}. "
            f"Thời gian: từ {start_date} đến {end_date}, bắt đầu lúc {event_time}. "
            f"Tháng diễn ra: tháng {month}. "
            f"Tính chất: {recurring if recurring else 'một lần'}. "
            f"Tags: {', '.join(tags) if tags else 'Không có'}"
        )
        
        doc = Document(
            page_content=content,
            metadata={
                "type": event_type,
                "name": name,
                "district": district or None,
                "ward": ward or None,
                "category": "event",
                "month": month,
                "start_date": start_date,
                "end_date": end_date
            }
        )
        docs.append(doc)
    
    return docs

# ====== Load tất cả dữ liệu ======
def load_all_data() -> list:
    """Load tất cả dữ liệu từ các file JSON (song song)"""
    file_paths = {
        "hotel": "data/hotel.json",
        "restaurant": "data/restaurant.json",
        "destination": "data/destination.json",
        "cafe": "data/cafe.json"
    }
    
    documents = []
    
    with ThreadPoolExecutor() as executor:
        # Load địa điểm song song
        place_futures = [
            executor.submit(load_place_data, data_type, path) 
            for data_type, path in file_paths.items()
        ]
        # Load sự kiện
        event_future = executor.submit(load_event_data)
        
        for future in place_futures:
            documents.extend(future.result())
        
        documents.extend(event_future.result())
    
    return documents

# ====== Tạo và Lưu ChromaDB ======
def save_chromadb(documents: list, persist_dir: str = "chromadb") -> Chroma:
    """
    Tạo vector store ChromaDB từ danh sách Document và lưu vào persist_dir.
    Áp dụng filter_complex_metadata để loại bỏ metadata phức tạp.
    """
    # Backup thư mục cũ nếu tồn tại
    if os.path.exists(persist_dir):
        backup_dir = f"{persist_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(persist_dir, backup_dir)
        print(f"📦 Đã backup ChromaDB cũ vào: {backup_dir}")
        shutil.rmtree(persist_dir)
    
    # Loại bỏ metadata phức tạp (list, dict)
    simple_docs = filter_complex_metadata(documents)
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    chroma_store = Chroma.from_documents(
        documents=simple_docs,
        embedding=embedding,
        persist_directory=persist_dir
    )
    
    print(f"✅ Đã lưu ChromaDB vào thư mục: {persist_dir}")
    return chroma_store

# ====== Auto-reload khi JSON thay đổi ======
class DataFileHandler(FileSystemEventHandler):
    """Handler để theo dõi thay đổi file JSON"""
    
    def __init__(self, callback, debounce_seconds: float = 5.0):
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._timer = None
        self._lock = threading.Lock()
    
    def _debounced_callback(self):
        with self._lock:
            self._timer = None
        print("🔄 Phát hiện thay đổi dữ liệu, đang rebuild ChromaDB...")
        self.callback()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.json'):
            with self._lock:
                if self._timer:
                    self._timer.cancel()
                self._timer = threading.Timer(self.debounce_seconds, self._debounced_callback)
                self._timer.start()
    
    def on_created(self, event):
        self.on_modified(event)

def rebuild_chromadb():
    """Rebuild ChromaDB từ dữ liệu mới"""
    try:
        docs = load_all_data()
        print(f"📄 Tổng số Document: {len(docs)}")
        save_chromadb(docs)
        print("🎉 Rebuild ChromaDB hoàn thành!")
    except Exception as e:
        print(f"❌ Lỗi khi rebuild ChromaDB: {e}")

def start_watcher(data_dir: str = "data"):
    """Bắt đầu theo dõi thư mục data để auto-reload"""
    if not os.path.exists(data_dir):
        print(f"⚠️ Thư mục {data_dir} không tồn tại")
        return None
    
    event_handler = DataFileHandler(rebuild_chromadb)
    observer = Observer()
    observer.schedule(event_handler, data_dir, recursive=False)
    observer.start()
    print(f"👀 Đang theo dõi thư mục: {data_dir}")
    return observer

def stop_watcher(observer):
    """Dừng theo dõi"""
    if observer:
        observer.stop()
        observer.join()
        print("🛑 Đã dừng theo dõi thư mục data")

# ====== CLI Commands ======
def print_stats():
    """In thống kê dữ liệu"""
    docs = load_all_data()
    
    categories = {}
    districts = {}
    
    for doc in docs:
        cat = doc.metadata.get("category", "unknown")
        dist = doc.metadata.get("district", "unknown")
        
        categories[cat] = categories.get(cat, 0) + 1
        if dist:
            districts[dist] = districts.get(dist, 0) + 1
    
    print("\n📊 Thống kê dữ liệu:")
    print("-" * 40)
    print("Theo loại:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count}")
    print("\nTheo quận:")
    for dist, count in sorted(districts.items()):
        print(f"  • {dist}: {count}")
    print("-" * 40)
    print(f"Tổng: {len(docs)} documents")

# ====== Main ======
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "watch":
            # Chế độ watch - tự động rebuild khi có thay đổi
            print("🚀 Khởi động chế độ watch...")
            rebuild_chromadb()
            observer = start_watcher()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                stop_watcher(observer)
        
        elif command == "stats":
            # In thống kê
            print_stats()
        
        else:
            print(f"❌ Lệnh không hợp lệ: {command}")
            print("Các lệnh hỗ trợ: watch, stats")
    else:
        # Chế độ mặc định - build một lần
        print("🔄 Bắt đầu tải dữ liệu...")
        docs = load_all_data()
        print(f"📄 Tổng số Document: {len(docs)}")
        print_stats()
        print("\n💾 Lưu ChromaDB...")
        save_chromadb(docs)
        print("🎉 Hoàn thành!")
