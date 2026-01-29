# Travel Chatbot RAG - Đà Nẵng

Chatbot du lịch thông minh sử dụng RAG (Retrieval-Augmented Generation) và Multi-Agent Architecture.

## 🚀 Tính năng

### Core Features
- **RAG với ChromaDB**: Tìm kiếm địa điểm, khách sạn, nhà hàng, cafe thông minh
- **Multi-Agent System**: Supervisor điều phối weather_agent và travel_information_agent
- **Semantic Routing**: Phân loại ý định người dùng thông minh thay vì keyword matching
- **Conversation Memory**: Giữ ngữ cảnh nhiều lượt hội thoại

### Nâng cấp v2.0
- ✅ **Smart Filter**: Tự động filter theo type/district từ câu hỏi
- ✅ **Event Database**: Dữ liệu sự kiện, lễ hội nội bộ
- ✅ **In-memory Cache**: Cache kết quả RAG với TTL
- ✅ **Output Guardrail**: Kiểm tra và làm sạch output
- ✅ **Singleton Supervisor**: Tối ưu khởi tạo, thread-safe
- ✅ **Auto-reload Chroma**: Tự động rebuild khi dữ liệu thay đổi
- ✅ **Session Management**: Quản lý session và lịch sử hội thoại

## 📁 Cấu trúc

```
travel-chatbot-rag/
├── main.py                    # FastAPI server với memory
├── Supervisor.py              # Supervisor agent (singleton)
├── travel_information_agent.py # Agent xử lý địa điểm, sự kiện
├── weather_agent.py           # Agent xử lý thời tiết
├── load_data.py               # Load và build ChromaDB
├── mytools/
│   ├── rag.py                 # RAG tool với filter và cache
│   ├── tavily.py              # Tavily search tool
│   └── weather.py             # Weather API tool
├── data/
│   ├── cafe.json              # Dữ liệu quán cafe
│   ├── hotel.json             # Dữ liệu khách sạn
│   ├── restaurant.json        # Dữ liệu nhà hàng
│   ├── destination.json       # Dữ liệu địa điểm
│   └── events.json            # Dữ liệu sự kiện (MỚI)
├── chromadb/                  # Vector database
├── requirements.txt
└── Dockerfile
```

## 🛠️ Cài đặt

```bash
# Clone và cd vào thư mục
cd travel-chatbot-rag

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Tạo file .env
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "TAVILY_API_KEY=your_tavily_key_here" >> .env
```

## 📊 Build Database

```bash
# Build ChromaDB một lần
python load_data.py

# Xem thống kê
python load_data.py stats

# Chế độ watch (auto-rebuild khi data thay đổi)
python load_data.py watch
```

## 🚀 Chạy Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📡 API Endpoints

### Hỏi Chatbot (có memory)
```bash
POST /ask
{
    "query": "Gợi ý quán cafe đẹp ở Sơn Trà",
    "session_id": "optional-session-id",
    "use_context": true
}

# Response
{
    "result": "...",
    "session_id": "abc123",
    "has_context": false
}
```

### Hỏi Chatbot (không memory)
```bash
POST /ask/simple
{
    "query": "Khách sạn 5 sao gần biển"
}
```

### Lấy lịch sử session
```bash
GET /session/{session_id}/history
```

### Xóa session
```bash
POST /session/clear
{
    "session_id": "abc123"
}
```

### Health Check
```bash
GET /health
```

### Admin Endpoints
```bash
POST /admin/reset-supervisor    # Reset supervisor
POST /admin/cleanup-sessions    # Dọn sessions hết hạn
```

## 🔧 Cấu hình

### RAG Filter Keywords

File `mytools/rag.py` chứa mapping từ khóa → filter:

```python
TYPE_KEYWORDS = {
    "cafe": "quán cà phê",
    "khách sạn": "hotel",
    "nhà hàng": "restaurant",
    ...
}

DISTRICT_KEYWORDS = {
    "sơn trà": "Sơn Trà",
    "hải châu": "Hải Châu",
    ...
}
```

### Cache Settings

```python
CACHE_TTL_SECONDS = 3600  # 1 giờ
MAX_CACHE_SIZE = 100
```

### Memory Settings

```python
ConversationMemory(
    max_history=10,      # Số lượt hội thoại tối đa
    ttl_seconds=3600     # Session timeout
)
```

## 🎯 Ví dụ Query

```python
# Địa điểm với filter
"Quán cafe ở Sơn Trà"           # → filter: type=quán cà phê, district=Sơn Trà
"Khách sạn 5 sao Hải Châu"      # → filter: type=5 sao, district=Hải Châu

# Sự kiện
"Lễ hội tháng 6 có gì?"         # → rag_tool (events trong DB)
"Sự kiện tuần này?"             # → tavily_search_deep (real-time)

# Follow-up (cần session)
"Cho tôi thêm thông tin về cái đầu tiên"  # → sử dụng context từ history
```

## 🐳 Docker

```bash
# Build
docker build -t travel-chatbot .

# Run
docker run -p 8000:8000 --env-file .env travel-chatbot
```

## 📈 Monitoring

### Health Check Response
```json
{
    "status": "healthy",
    "timestamp": "2025-01-01T12:00:00",
    "supervisor": {
        "supervisor_initialized": true,
        "responsive": true
    },
    "memory": {
        "active_sessions": 5,
        "max_history": 10,
        "ttl_seconds": 3600
    }
}
```

## 🔄 Auto-reload Chroma

Khi chạy `python load_data.py watch`, hệ thống sẽ:
1. Build ChromaDB ban đầu
2. Theo dõi thư mục `data/`
3. Tự động rebuild khi có file JSON thay đổi
4. Debounce 5 giây để tránh rebuild liên tục

## 📝 Thêm dữ liệu mới

### Thêm địa điểm
Thêm vào file JSON tương ứng (`cafe.json`, `hotel.json`, etc.):
```json
{
    "name": "Tên địa điểm",
    "type": "loại",
    "description": "Mô tả",
    "district": "Quận",
    "ward": "Phường",
    "open_time": "08:00",
    "close_time": "22:00",
    "tags": ["tag1", "tag2"],
    "duration_suggested_min": 60
}
```

### Thêm sự kiện
Thêm vào `events.json`:
```json
{
    "name": "Tên sự kiện",
    "type": "lễ hội",
    "description": "Mô tả",
    "district": "Quận",
    "ward": "Phường",
    "start_date": "2025-06-01",
    "end_date": "2025-06-03",
    "time": "20:00",
    "tags": ["tag1", "tag2"],
    "recurring": "yearly",
    "month": 6
}
```

Sau đó chạy `python load_data.py` để rebuild database.

## 🤝 Contributing

1. Fork repo
2. Tạo branch feature
3. Commit changes
4. Push và tạo PR

## 📄 License

MIT License

