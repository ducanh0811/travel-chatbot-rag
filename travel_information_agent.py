import os
from dotenv import load_dotenv
from mytools.rag import rag_tool
from mytools.tavily import tavily_search_deep
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
load_dotenv()
def create_information_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
    tools = [rag_tool, tavily_search_deep]

    # system prompt có biến user_input
    prompt = """
Bạn là agent cung cấp thông tin về Du Lịch Đà Nẵng trong ChromaDB.
- Ngày hôm nay : 10/7/2025
- Chỉ giới hạn trong du lịch, địa phận tại Đà Nẵng.
- Khi user đề cập tới sự kiện thì bạn phải hiểu là sự kiện về du lịch.
- Dùng rag_tool khi người dùng muốn hỏi về các địa điểm du lịch, cafe, nhà hàng, khách sạn, và đừng gộp các địa điểm này trong câu trả lời.
- Dùng tavily_search_deep khi người dùng muốn hỏi về các sự kiện du lịch và lễ hội tại Đà Nẵng.
- Domains for searching: https://danangfantasticity.com/category/le-hoi-amp-su-kien
- Giữ nguyên output của các tool
- Trả về giữ liệu cho supervisor
User hỏi: {user_input}
"""

    # dùng InMemoryCheckpointer để lưu history
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,            # gắn prompt
        version="v2",
        name="travel_information_agent"
    )
    return agent


# Ví dụ sử dụng:
if __name__ == "__main__":
    load_dotenv()
    weather_agent = create_information_agent()
    user_query = "Gợi ý các sự kiện hay lễ hội vào tháng 7"
    response = weather_agent.invoke({"messages": {"role": "user", "content": user_query}})
    # In kết quả
    for msg in response["messages"]:
        msg.pretty_print()