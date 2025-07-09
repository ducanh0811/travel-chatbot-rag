import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load OpenAI API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Vui lòng đặt biến môi trường OPENAI_API_KEY để sử dụng mô hình OpenAI.")

assert os.getenv("OPENAI_API_KEY")

# Import weather tools
from mytools.weather import get_weather, get_weather_forecast

# Create tools list
tools = [get_weather, get_weather_forecast]

# Initialize OpenAI model
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo")  # Using gpt-4 as closest to gpt-4.1

# Create research agent using langgraph
from langgraph.prebuilt import create_react_agent

research_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt = (
    "You are a research agent.\n\n"
    "INSTRUCTIONS:\n"
    "- ALWAYS respond in Vietnamese (Tiếng Việt)\n"
    "- ONLY assist with research-related tasks — DO NOT perform any math or numerical computation\n\n"
    "WEATHER TASKS:\n"
    "- You are ONLY allowed to respond to weather queries **if the location is either 'Đà Nẵng' or 'Hội An'** (case-insensitive, match exactly)\n"
    "- If the location is Đà Nẵng or Hội An, you MAY call available tools to retrieve:\n"
    "    • Nhiệt độ (temperature)\n"
    "    • Xác suất mưa (rain probability)\n"
    "    • Một lời khuyên ngắn (short advice in Vietnamese)\n"
    "- DO NOT include humidity, wind speed, or any other data\n"
    "- If the user asks about **any other location**, DO NOT run any tool or return data. Instead, reply exactly:\n"
    "    \"Xin lỗi, tôi chỉ cung cấp thông tin thời tiết cho Đà Nẵng và Hội An.\"\n\n"
    "RESEARCH TASKS:\n"
    "- For all non-weather research questions, you may use the web search tool\n"
    "- DO NOT perform math calculations\n\n"
    "RESPONSE FORMAT:\n"
    "- Return only the result — do NOT explain or mention tools\n"
    "- After completing your task, report directly to the supervisor"
),
    name="research_agent",
)

# Test the agent
if __name__ == "__main__":
    input_message = {"role": "user", "content": "Tìm hiểu về thời tiết hôm nay tại Hội An và  2 ngày tới"}
    response = research_agent.invoke({"messages": [input_message]})
    
    for message in response["messages"]:
        message.pretty_print()
