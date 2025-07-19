# supervisor.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from weather_agent import create_weather_agent
from travel_information_agent import create_information_agent
from concurrent.futures import ThreadPoolExecutor

def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY trong .env")
    return api_key

def get_llm():
    # Khởi tạo LLM dùng chung
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def create_agents():
    # Khởi tạo các agent con
    with ThreadPoolExecutor() as executor:
        future_weather = executor.submit(create_weather_agent)
        future_travel = executor.submit(create_information_agent)
        weather_agent = future_weather.result()
        travel_information_agent = future_travel.result()
    return weather_agent, travel_information_agent

def create_supervisor_agent():
    load_api_key()
    llm = get_llm()
    weather_agent, travel_information_agent = create_agents()
    prompt = f"""
Bạn là một Supervisor Agent, có nhiệm vụ phân công:
- CurrentDate: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}
- Nếu câu hỏi có yêu cầu về thời tiết về Đà Nẵng (từ khoá: “thời tiết”, “weather”, tên thành phố…) → delegate cho weather_agent.
- Nếu câu hỏi có yêu cầu về thông tin các địa danh, nhà hàng, quán ăn, khách sạn, cafe, địa danh, sự kiện,...  (từ khoá: “Nhà hàng”, "Khách sạn", "Sự kiện",...) → delegate cho travel_information_agent.
User hỏi: {{user_input}}
"""
    graph = create_supervisor(
        model=llm,
        agents=[weather_agent, travel_information_agent],
        prompt=prompt,
    )
    supervisor = graph.compile()
    return supervisor

def run_supervisor_query(query):
    supervisor = create_supervisor_agent()
    response = supervisor.invoke({"messages": {"role": "user", "content": query}})
    if not response:
        print("❌ Không nhận được phản hồi từ supervisor.")
        return
    if "messages" in response and response["messages"]:
        for i, msg in enumerate(response["messages"], 1):
            content = msg.content if hasattr(msg, 'content') else msg
            if content and not any(s in content.lower() for s in ["transferred to", "transferring", "successfully transfer"]):
                if content != query:
                    print(content)
    else:
        print("⚠️ Không có trường 'messages' hoặc không có kết quả từ sub-agent. Toàn bộ response:")

# Ví dụ chạy thử
if __name__ == "__main__":
    query = "Tìm restaurant ở Hải Châu"
    run_supervisor_query(query)

