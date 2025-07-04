import getpass
import os
from dotenv import load_dotenv

load_dotenv()

# Import weather tools
from mytools.weather import get_weather, get_weather_forecast, list_available_locations

# Import web search tools (thay thế events tools)
from mytools.web_search import web_search

#define tool
from langchain_tavily import TavilySearch
assert os.getenv("TAVILY_API_KEY")
# Tạo tool tìm kiếm Tavily
search = TavilySearch(max_results=2)

# Gộp vào danh sách nếu muốn dùng agent - bao gồm weather và web search tools
tools = [search, get_weather, get_weather_forecast, list_available_locations,web_search]

# Kiểm tra xem đã có key chưa
assert os.getenv("OPENAI_API_KEY")
# Gọi mô hình OpenAI từ LangChain
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo")  # hoặc "gpt-4" nếu bạn muốn dùng bản mạnh hơn

# model_with_tools = model.bind_tools(tools)

#Create agent
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)

# Ví dụ sử dụng weather và web search tools
input_message = {"role": "user", "content": "các địa điểm du lịch ở Hà Nội"}
response = agent_executor.invoke({"messages": [input_message]})

for message in response["messages"]:
    message.pretty_print()
