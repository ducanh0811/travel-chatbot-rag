import os
from dotenv import load_dotenv
from mytools.weather import get_weather, get_weather_forecast
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from mytools.weather import get_weather, get_weather_forecast

def create_weather_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
    tools = [get_weather, get_weather_forecast]

    # system prompt có biến user_input
    prompt = """
Bạn là weather agent chuyên cho Đà Nẵng.
- Nếu user hỏi về đề tài ngoài thời tiết: từ chối “Xin lỗi, tôi chỉ hỗ trợ về thời tiết Đà Nẵng.”
- Nếu user hỏi nơi khác: từ chối “Xin lỗi, tôi chỉ hỗ trợ Đà Nẵng.”
- Dùng tool get_weather cho việc lấy thông tin thời tiết hiện tại
- Dùng tool get_weather_forecast lấy thông tin thời tiết tương lai
- Với Đà Nẵng: gọi get_weather / get_weather_forecast, chỉ output nhiệt độ, xác suất mưa, lời khuyên.
User hỏi: {user_input}
"""

    # dùng InMemoryCheckpointer để lưu history
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,            # gắn prompt
        version="v2",
        name="weather_agent"
    )
    return agent


# Ví dụ sử dụng:
if __name__ == "__main__":
    weather_agent = create_weather_agent()
    # user_query = "Tìm hiểu về thời tiết hôm nay tại Đà Nẵng"
    user_query = "Tìm hiểu về sự kiện hôm nay tại Đà Nẵng"
    response = weather_agent.invoke({"messages": {"role": "user", "content": user_query}})
    # In kết quả
    for msg in response["messages"]:
        msg.pretty_print()
