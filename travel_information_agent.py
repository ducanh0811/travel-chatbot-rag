import os
from dotenv import load_dotenv
from mytools.rag import rag_tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
load_dotenv()
def create_information_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    tools = [rag_tool]

    # system prompt có biến user_input
    prompt = """
Bạn là agent cung cấp thông tin về Du Lịch Đà Nẵng trong ChromaDB.
- Dùng rag_tool khi khách hàng muốn hỏi về các địa điểm du lịch, cafe, nhà hàng, khách sạn, và đừng gộp các địa điểm này trong câu trả lời.
- Giữ nguyên output của tool
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
    user_query = "Gợi ý nhà hàng ở Quận Sơn Trà"
    response = weather_agent.invoke({"messages": {"role": "user", "content": user_query}})
    # In kết quả
    for msg in response["messages"]:
        msg.pretty_print()