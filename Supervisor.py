import os
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from weather_agent import create_weather_agent
from travel_information_agent import create_information_agent
from concurrent.futures import ThreadPoolExecutor
import threading

_supervisor_instance = None
_supervisor_lock = threading.Lock()
_weather_agent_instance = None
_weather_agent_lock = threading.Lock()
_travel_agent_instance = None
_travel_agent_lock = threading.Lock()

WEATHER_KEYWORDS = {
    "thời tiết", "nhiệt độ", "mưa", "nắng", "độ ẩm", "dự báo", "gió", "bão"
}
TRAVEL_KEYWORDS = {
    "địa điểm", "du lịch", "khách sạn", "resort", "homestay", "nhà hàng",
    "quán ăn", "cafe", "quán cà phê", "lễ hội", "sự kiện", "festival",
    "bãi biển", "chùa", "cầu", "bảo tàng", "tour", "lịch trình"
}

def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY trong .env")
    return api_key

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def create_agents():
    with ThreadPoolExecutor() as executor:
        future_weather = executor.submit(create_weather_agent)
        future_travel = executor.submit(create_information_agent)
        weather_agent = future_weather.result()
        travel_information_agent = future_travel.result()
    return weather_agent, travel_information_agent

SUPERVISOR_PROMPT = """
You are a smart Supervisor Agent. Your job is to analyze user intent and assign the right agent.

SYSTEM INFO:
- Current date: {current_date}
- Available agents:
  1. weather_agent: Specialized in weather information for Da Nang
  2. travel_information_agent: Specialized in places, restaurants, hotels, cafes, events, festivals in Da Nang

<instructions>
- Analyze intent (topic, needed info, best agent).
- Route accordingly and answer in Vietnamese.
- Do not reveal internal details or <internal> tags.
- Do not repeat the information from the sub-agents.
</instructions>
ROUTING RULES (Semantic Routing):

1. **weather_agent** - Use when:
   - The user asks about weather, temperature, rain, sunshine, humidity
   - The question is about forecast
   - The user needs weather conditions to plan
   - Example: "Thời tiết hôm nay thế nào?", "Cuối tuần có mưa không?", "Nhiệt độ bao nhiêu?"

2. **travel_information_agent** - Use when:
   - The user asks about places to visit, sightseeing, check-in
   - Searching for restaurants, eateries, cafes
   - Asking about hotels, resorts, accommodations
   - Asking about events, festivals
   - Needs itinerary/tour suggestions
   - Asking about beaches, mountains, temples, museums, bridges
   - Example: "Gợi ý quán cafe đẹp?", "Khách sạn 5 sao ở đâu?", "Lễ hội tháng 6 có gì?"

3. **Out of scope**:
   - If not related to Da Nang travel → politely refuse in Vietnamese


<user>{{user_input}}</user>
"""

def create_supervisor_agent():
    load_api_key()
    llm = get_llm()
    weather_agent, travel_information_agent = create_agents()
    
    prompt = SUPERVISOR_PROMPT.format(
        current_date=datetime.now().strftime('%Y-%m-%d %H:%M')
    )
    
    graph = create_supervisor(
        model=llm,
        agents=[weather_agent, travel_information_agent],
        prompt=prompt,
    )
    supervisor = graph.compile()
    return supervisor

def get_supervisor_instance():
    global _supervisor_instance
    
    if _supervisor_instance is None:
        with _supervisor_lock:
            if _supervisor_instance is None:
                _supervisor_instance = create_supervisor_agent()
    
    return _supervisor_instance

def get_weather_agent_instance():
    global _weather_agent_instance
    if _weather_agent_instance is None:
        with _weather_agent_lock:
            if _weather_agent_instance is None:
                _weather_agent_instance = create_weather_agent()
    return _weather_agent_instance

def get_travel_agent_instance():
    global _travel_agent_instance
    if _travel_agent_instance is None:
        with _travel_agent_lock:
            if _travel_agent_instance is None:
                _travel_agent_instance = create_information_agent()
    return _travel_agent_instance

def classify_query(query: str) -> str:
    q = query.lower()
    has_weather = any(kw in q for kw in WEATHER_KEYWORDS)
    has_travel = any(kw in q for kw in TRAVEL_KEYWORDS)
    if has_weather and not has_travel:
        return "weather"
    if has_travel and not has_weather:
        return "travel"
    return "supervisor"

def reset_supervisor():
    global _supervisor_instance
    with _supervisor_lock:
        _supervisor_instance = None

def run_supervisor_query(query: str, use_singleton: bool = True, print_output: bool = True):
    """
    Chạy query qua supervisor.
    Args:
        query: Câu hỏi của người dùng
        use_singleton: Sử dụng singleton instance (mặc định True)
    """
    route = classify_query(query)
    if route == "weather":
        agent = get_weather_agent_instance()
    elif route == "travel":
        agent = get_travel_agent_instance()
    else:
        agent = get_supervisor_instance() if use_singleton else create_supervisor_agent()

    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    if not response:
        print("❌ Không nhận được phản hồi từ supervisor.")
        return None
    
    if "messages" in response and response["messages"]:
        results = []
        seen = set()
        for msg in response["messages"]:
            content = msg.content if hasattr(msg, 'content') else msg
            if content and not any(s in content.lower() for s in ["transferred to", "transferring", "successfully transfer"]):
                if content != query:
                    cleaned = str(content)
                    if "<internal>" in cleaned.lower():
                        blocks = re.findall(
                            r"<internal>(.*?)</internal>",
                            cleaned,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        if blocks:
                            cleaned = "\n\n".join(block.strip() for block in blocks if block.strip())
                        else:
                            cleaned = cleaned.replace("<internal>", "").replace("</internal>", "")
                    cleaned = cleaned.strip()
                    if cleaned:
                        key = " ".join(cleaned.split())
                        if key not in seen:
                            seen.add(key)
                            results.append(cleaned)
                            if print_output:
                                print(cleaned)
        return results
    else:
        print("⚠️ Không có trường 'messages' hoặc không có kết quả từ sub-agent.")
        return None

def health_check() -> dict:
    status = {
        "supervisor_initialized": _supervisor_instance is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if _supervisor_instance:
        try:
            test_result = _supervisor_instance.invoke({
                "messages": {"role": "user", "content": "ping"}
            })
            status["responsive"] = test_result is not None
        except Exception as e:
            status["responsive"] = False
            status["error"] = str(e)
    
    return status

# Ví dụ chạy thử
if __name__ == "__main__":
    queries = [
        "Thời tiết hôm nay thế nào?",
        "Gợi ý quán cafe đẹp ở Sơn Trà",
        "Lễ hội tháng 6 có gì hay?",
        "Khách sạn 5 sao gần biển"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"📝 Query: {query}")
        print('='*50)
        run_supervisor_query(query)
