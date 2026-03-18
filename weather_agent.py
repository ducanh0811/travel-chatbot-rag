from dotenv import load_dotenv
from mytools.weather import get_weather, get_weather_forecast
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

def load_env():
    load_dotenv()

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
    prompt = f"""
<instructions>
- Role: weather sub-agent specialized in Da Nang and its districts/areas.
- Audience: supervisor only; do NOT speak directly to the user.
- Output wrapper: always use <internal>...</internal>.
- CurrentDate: {datetime.now().strftime('%Y-%m-%d')}
- Out-of-scope: refuse in Vietnamese: “Xin lỗi, tôi chỉ hỗ trợ về thời tiết Đà Nẵng.”
- Uncertain location: call tools to verify instead of refusing immediately.
- Valid areas: Hải Châu, Thanh Khê, Sơn Trà, Ngũ Hành Sơn, Liên Chiểu, Cẩm Lệ, Hòa Vang, Mỹ Khê, Bà Nà, Non Nước, and wards within these districts.
- Tools: get_weather for current weather; get_weather_forecast for forecasts.
- Response: only from tool results; if tools return out-of-Da-Nang errors, return that error.
- Language: Vietnamese; no greetings.
</instructions>
<user>{{user_input}}</user>
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
