import sys
from travel_information_agent import create_information_agent, load_env


def run_test(query: str) -> None:
    # Ensure Unicode prints correctly on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    load_env()
    agent = create_information_agent()
    response = agent.invoke({"messages": {"role": "user", "content": query}})

    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    if "messages" in response:
        for msg in response["messages"]:
            content = msg.content if hasattr(msg, "content") else str(msg)
            role = getattr(msg, "type", getattr(msg, "role", ""))
            name = getattr(msg, "name", "")
            header = f"[{role}{' ' + name if name else ''}]"
            if content:
                print(header)
                print(content)
    else:
        print(response)


if __name__ == "__main__":
    # Câu hỏi thời gian thực để buộc agent gọi Tavily
    realtime_query = (
        "Vụ kẹt xe ở cầu Rồng lúc nãy là sao vậy? "
        "Sự việc mới xảy ra trong 1-2 giờ qua, "
        "hãy tìm thông tin mới nhất."
    )
    run_test(realtime_query)
