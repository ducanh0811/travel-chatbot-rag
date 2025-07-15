from dotenv import load_dotenv
from mytools.weather import get_weather, get_weather_forecast
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from concurrent.futures import ThreadPoolExecutor

def load_env():
    load_dotenv()

def get_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def get_tools():
    # Khởi tạo các tool song song (chuẩn bị cho mở rộng)
    with ThreadPoolExecutor() as executor:
        future_weather = executor.submit(lambda: get_weather)
        future_forecast = executor.submit(lambda: get_weather_forecast)
        weather = future_weather.result()
        forecast = future_forecast.result()
    return [weather, forecast]

def create_weather_agent():
    llm = get_llm()
    tools = get_tools()
    prompt = """
Bạn là weather agent chuyên cho Đà Nẵng.
- Nếu user hỏi về đề tài ngoài thời tiết: từ chối “Xin lỗi, tôi chỉ hỗ trợ về thời tiết Đà Nẵng.”
- Nếu user hỏi nơi khác: từ chối “Xin lỗi, tôi chỉ hỗ trợ Đà Nẵng.”
- Dùng tool get_weather cho việc lấy thông tin thời tiết hiện tại
- Dùng tool get_weather_forecast lấy thông tin thời tiết tương lai
- Với Đà Nẵng: gọi get_weather / get_weather_forecast, chỉ output nhiệt độ, xác suất mưa, lời khuyên.
User hỏi: {user_input}
"""
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        version="v2",
        name="weather_agent",
        
    )
    return agent

def run_agent_query(query):
    load_env()
    agent = create_weather_agent()
    response = agent.invoke({"messages": {"role": "user", "content": query}})
    if "messages" in response:
        for msg in response["messages"]:
            print(msg.content)
    else:
        print(response)

if __name__ == "__main__":
    user_query = "Tìm hiểu về sự kiện hôm nay tại Đà Nẵng"
    run_agent_query(user_query)
