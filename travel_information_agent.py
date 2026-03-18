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
<instructions>
- Role: travel sub-agent specialized in Da Nang travel information.
- Audience: supervisor only; do NOT speak directly to the user.
- Output wrapper: always use <internal>...</internal>.
- Current date: {current_date}
- Scope: only provide travel information about Da Nang and nearby areas.

- AVAILABLE TOOLS:

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

 - PROCESSING RULES:

1. **Identify intent before tool selection**:
   - Determine what information the user needs
   - Static info (places, hotels) → rag_tool
   - Dynamic info (latest events/news) → tavily_search_deep
   - If unsure → try rag_tool first

2. **Combine tools when needed**:
   - You may call both tools for complex questions
   - Example: "June festival and nearby hotels" → tavily + rag

3. **Handle results**:
   - Keep the output format from tools
   - If not found → clearly say so
   - You may add related suggestions
   - All output must be inside <internal>...</internal>
   - Return ONLY the tool output (verbatim). No extra summaries, no reformatting.

4. **Check timing**:
   - For events, verify if they are still ongoing
   - Prefer upcoming or ongoing events

5. **Scope limit**:
   - Only answer about Da Nang and nearby areas
   - If out of scope → politely refuse

 - INTERNAL OUTPUT FORMAT (Vietnamese):

For places:
- Tên, loại, khu vực
- Mô tả ngắn gọn
- Giờ mở cửa (nếu có)
- Gợi ý thời gian tham quan

For events:
- Tên sự kiện
- Thời gian diễn ra
- Địa điểm
- Mô tả và điểm nổi bật

 - Language: Vietnamese.
 - Return only the tool output inside <internal>...</internal>. Do not add assistant commentary.
</instructions>
<user>{{user_input}}</user>
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
