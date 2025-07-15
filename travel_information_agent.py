import os
from dotenv import load_dotenv
from mytools.rag import rag_tool
from mytools.tavily import tavily_search_deep
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
        future_rag = executor.submit(lambda: rag_tool)
        future_tavily = executor.submit(lambda: tavily_search_deep)
        rag = future_rag.result()
        tavily = future_tavily.result()
    return [rag, tavily]

def create_information_agent():
    llm = get_llm()
    tools = get_tools()
    prompt = """
Bạn là agent cung cấp thông tin về Du Lịch Đà Nẵng trong ChromaDB.
- Chỉ giới hạn trong du lịch, địa phận tại Đà Nẵng.
- Khi user đề cập tới sự kiện thì bạn phải hiểu là sự kiện về du lịch.
- Dùng rag_tool khi người dùng muốn hỏi về các địa điểm du lịch, cafe, nhà hàng, khách sạn, và đừng gộp các địa điểm này trong câu trả lời.
- Dùng tavily_search_deep khi người dùng muốn hỏi về các sự kiện du lịch và lễ hội tại Đà Nẵng. LƯU Ý: Xem xét kỹ câu hỏi để xác định thời gian và nội dung sự kiện.
- Giữ nguyên output của các tool
User hỏi: {user_input}
"""
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        version="v2",
        name="travel_information_agent",
        
    )
    return agent

def run_agent_query(query):
    load_env()
    agent = create_information_agent()
    response = agent.invoke({"messages": {"role": "user", "content": query}})
    if "messages" in response:
        for msg in response["messages"]:
            print(msg.content)
    else:
        print(response)

if __name__ == "__main__":
    user_query = "Gợi ý các sự kiện hay lễ hội vào tháng 7"
    run_agent_query(user_query)