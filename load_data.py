import os
import json
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings

# ====== Load API Key ======
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("⚠️ Vui lòng đặt biến môi trường OPENAI_API_KEY trong file .env của bạn.")

# ====== Load & Gộp dữ liệu từ 4 file JSON ======
def load_all_data():
    """
    Đọc các file JSON của hotel, restaurant, destination, cafe và trả về list Document.
    Metadata tags để dưới dạng list ban đầu, sẽ lọc sau.
    """
    file_paths = {
        "hotel": "data/hotel.json",
        "restaurant": "data/restaurant.json",
        "destination": "data/destination.json",
        "cafe": "data/cafe.json"
    }
    documents = []
    for data_type, path in file_paths.items():
        if not os.path.exists(path):
            print(f"⚠️ Không tìm thấy file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            name = item.get("name", "Không rõ")
            desc = item.get("description", "")
            district = item.get("district", "")
            ward = item.get("ward", "")
            if data_type == "hotel":
                open_time = item.get("checkin_time", "Không rõ")
                close_time = item.get("checkout_time", "Không rõ")
            else:
                open_time = item.get("open_time", "")
                close_time = item.get("close_time", "")
            tags = item.get("tags", [])
            suggested = item.get("duration_suggested_min", "Không rõ")
            content = (
                f"{name} ({data_type}) - phường {ward}, quận {district}. "
                f"Mô tả: {desc}. "
                f"Giờ mở: {open_time}, đóng: {close_time}. "
                f"Thời gian gợi ý: {suggested} phút. "
                f"Tags: {', '.join(tags)}"
            )
            doc = Document(
                page_content=content,
                metadata={
                    "type": data_type,
                    "name": name,
                    "district": district or None,
                    "ward": ward or None,
                    "tags": tags
                }
            )
            documents.append(doc)
    return documents

# ====== Tạo và Lưu ChromaDB ======
def save_chromadb(documents, persist_dir: str = "chromadb"):
    """
    Tạo vector store ChromaDB từ danh sách Document và lưu vào persist_dir.
    Áp dụng filter_complex_metadata để loại bỏ metadata phức tạp.
    """
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

# ====== Main ======
if __name__ == "__main__":
    print("🔄 Bắt đầu tải dữ liệu...")
    docs = load_all_data()
    print(f"📄 Tổng số Document: {len(docs)}")
    print("💾 Lưu ChromaDB...")
    save_chromadb(docs)
    print("🎉 Hoàn thành!")
