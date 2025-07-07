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
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- ALWAYS respond in Vietnamese (Tiếng Việt)\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- For weather information: ONLY provide weather data for ĐÀ NẴNG and HỘI AN locations\n"
        "- If user asks for weather in other locations, politely decline and suggest Đà Nẵng or Hội An instead\n"
        "- For weather responses, provide ONLY: temperature, rain probability, and advice (skip humidity, wind speed, etc.)\n"
        "- For general research: Use web search tools for any other research needs\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)

# Test the agent
if __name__ == "__main__":
    input_message = {"role": "user", "content": "Tìm hiểu về thời tiết hôm nay tại Đà Nẵng 2 ngày tới"}
    response = research_agent.invoke({"messages": [input_message]})
    
    for message in response["messages"]:
        message.pretty_print()
