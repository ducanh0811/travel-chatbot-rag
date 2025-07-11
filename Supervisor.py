# supervisor.py

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

# import các factory tạo sub-agent
from weather_agent import create_weather_agent
from travel_information_agent import create_information_agent
def create_supervisor_agent():
    # 1. Load API key
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Thiếu OPENAI_API_KEY trong .env")

    # 2. Khởi tạo chung LLM (chung cho tất cả các agent)
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 3. Tạo từng sub-agent
    weather_agent  = create_weather_agent() 
    travel_information_agent = create_information_agent()
    # 4. Định nghĩa prompt cho supervisor
    prompt="""
Ngày hôm nay: 10/07/2025
Bạn là một Supervisor Agent, có nhiệm vụ phân công:
- Nếu user hỏi về thời tiết về Đà Nẵng (từ khoá: “thời tiết”, “weather”, tên thành phố…) → delegate cho weather_agent.
- Nếu user hỏi về thông tin các địa danh, nhà hàng, quán ăn, khách sạn, cafe, địa danh, sự kiện,...  (từ khoá: “Nhà hàng”, "Khách sạn", "Sự kiện",...) → delegate cho travel_information_agent.
User hỏi: {user_input}
"""


    # 6. Tạo supervisor graph
    graph = create_supervisor(
        model=llm,
        agents=[weather_agent,travel_information_agent],
        prompt=prompt,
    )
    # …then compile it into a runnable
    supervisor = graph.compile()
    return supervisor


# Ví dụ chạy thử
if __name__ == "__main__":
    supervisor = create_supervisor_agent()
    query = "Cho tôi thông tin về thời tiết 2-3 ngày tới"
    # Chạy supervisor; nó sẽ tự động gọi weather_agent
    response = supervisor.invoke({"messages": {"role": "user", "content": query}})
    # In kết quả
    if "messages" in response:
    # In thông tin chi tiết từ travel_information_agent (nếu có)
        for msg in response["messages"]:
            print(msg.content)

