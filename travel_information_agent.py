import os
from datetime import datetime
from dotenv import load_dotenv
from mytools.rag import rag_tool
from mytools.tavily import tavily_search_deep
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from concurrent.futures import ThreadPoolExecutor

def load_env():
    load_dotenv()

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def get_tools():
    # Khởi tạo các tool song song (chuẩn bị cho mở rộng)
    with ThreadPoolExecutor() as executor:
        future_rag = executor.submit(lambda: rag_tool)
        future_tavily = executor.submit(lambda: tavily_search_deep)
        rag = future_rag.result()
        tavily = future_tavily.result()
    return [rag, tavily]

# ============ SEMANTIC ROUTING PROMPT ============
TRAVEL_AGENT_PROMPT = """
Bạn là agent chuyên cung cấp thông tin Du Lịch Đà Nẵng.

THÔNG TIN HỆ THỐNG:
- Ngày hiện tại: {current_date}
- Phạm vi: Chỉ hỗ trợ thông tin về du lịch tại Đà Nẵng và vùng lân cận

CÁC CÔNG CỤ CÓ SẴN:

1. **rag_tool** - Sử dụng khi:
   - Tìm kiếm địa điểm du lịch: bãi biển, núi, chùa, cầu, bảo tàng, công viên
   - Tìm kiếm quán cafe, nhà hàng, quán ăn
   - Tìm kiếm khách sạn, resort, homestay
   - Tìm kiếm sự kiện, lễ hội (đã lưu trong database)
   - Câu hỏi về địa điểm cụ thể hoặc gợi ý địa điểm
   - Ví dụ: "Quán cafe đẹp ở Sơn Trà", "Khách sạn 5 sao", "Bãi biển nào đẹp?"

2. **tavily_search_deep** - Sử dụng khi:
   - Cần thông tin SỰ KIỆN MỚI NHẤT, tin tức thời sự
   - Tìm kiếm lễ hội, festival đang diễn ra hoặc sắp tới
   - Câu hỏi có yếu tố thời gian ngắn hạn như "hôm nay", "tuần này", "cuối tuần", "sắp tới"
   - Thông tin về giá vé, chương trình khuyến mãi mới
   - Cần cập nhật real-time mà database chưa có
   - Ví dụ: "Sự kiện tuần này có gì?", "Lễ hội pháo hoa năm nay khi nào?"

QUY TẮC XỬ LÝ:

1. **Phân tích ý định trước khi chọn tool**:
   - Xác định người dùng cần loại thông tin gì
   - Thông tin tĩnh (địa điểm, khách sạn) → rag_tool
   - Thông tin động (sự kiện mới, tin tức) → tavily_search_deep
   - Nếu không chắc → thử rag_tool trước

2. **Kết hợp tools khi cần**:
   - Có thể gọi cả 2 tools nếu câu hỏi phức tạp
   - Ví dụ: "Lễ hội tháng 6 và khách sạn gần đó" → tavily + rag

3. **Xử lý kết quả**:
   - Giữ nguyên format output từ tools
   - Nếu không tìm thấy → thông báo rõ ràng
   - Có thể bổ sung gợi ý liên quan

4. **Kiểm tra thời gian**:
   - Với sự kiện, kiểm tra xem còn diễn ra không
   - Ưu tiên sự kiện sắp tới hoặc đang diễn ra

5. **Giới hạn phạm vi**:
   - Chỉ trả lời về Đà Nẵng và vùng lân cận
   - Nếu hỏi ngoài phạm vi → từ chối lịch sự

HƯỚNG DẪN FORMAT TRẢ LỜI:

Với địa điểm:
- Tên, loại, khu vực
- Mô tả ngắn gọn
- Giờ mở cửa (nếu có)
- Gợi ý thời gian tham quan

Với sự kiện:
- Tên sự kiện
- Thời gian diễn ra
- Địa điểm
- Mô tả và điểm nổi bật

User hỏi: {{user_input}}
"""

def create_information_agent():
    """Tạo travel information agent với semantic routing"""
    llm = get_llm()
    tools = get_tools()
    
    prompt = TRAVEL_AGENT_PROMPT.format(
        current_date=datetime.now().strftime('%Y-%m-%d %H:%M')
    )
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        name="travel_information_agent",
    )
    return agent

def run_agent_query(query: str):
    """Chạy query qua travel information agent"""
    load_env()
    agent = create_information_agent()
    response = agent.invoke({"messages": {"role": "user", "content": query}})
    
    if "messages" in response:
        for msg in response["messages"]:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if content:
                print(content)
    else:
        print(response)

if __name__ == "__main__":
    test_queries = [
        "Gợi ý quán cafe đẹp ở Sơn Trà",
        "Lễ hội tháng 6 có gì hay?",
        "Khách sạn 5 sao gần biển Mỹ Khê",
        "Sự kiện tuần này ở Đà Nẵng"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"📝 Query: {query}")
        print('='*50)
        run_agent_query(query)
