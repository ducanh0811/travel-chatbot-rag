import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load OpenAI API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Vui lòng đặt biến môi trường OPENAI_API_KEY để sử dụng mô hình OpenAI.")

assert os.getenv("OPENAI_API_KEY")

# Import Tavily search tools
from mytools.tavily import tavily_search, tavily_search_deep

# Create tools list
tools = [tavily_search, tavily_search_deep]

# Initialize OpenAI model
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo")

# Create event search agent using langgraph
from langgraph.prebuilt import create_react_agent

event_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=(
        "You are an event search agent specialized in Đà Nẵng and Hội An events.\n\n"
        "INSTRUCTIONS:\n"
        "- ALWAYS respond in Vietnamese (Tiếng Việt)\n"
        "- ONLY search for events, activities, festivals, and happenings in ĐÀ NẴNG and HỘI AN\n"
        "- If user asks about events in other locations, politely decline and suggest searching for Đà Nẵng or Hội An events instead\n"
        "- Focus on: festivals, cultural events, food events, tourism activities, entertainment, concerts, exhibitions\n"
        "- Prioritize current and upcoming events (within next 30 days)\n"
        "- Include event details: date, time, location, description, ticket info (if available)\n"
        "- Use tavily_search for general events and tavily_search_deep for detailed information\n"
        "- Format responses clearly with emoji and organized sections\n"
        "- After completing your search, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="event_agent",
)

# Test the agent
if __name__ == "__main__":
    input_message = {"role": "user", "content": "Tìm kiếm các sự kiện, lễ hội đang diễn ra tại Đà Nẵng và Hội An trong tháng này"}
    response = event_agent.invoke({"messages": [input_message]})
    
    for message in response["messages"]:
        message.pretty_print() 