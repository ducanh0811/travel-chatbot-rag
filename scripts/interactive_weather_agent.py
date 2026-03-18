import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weather_agent import create_weather_agent, load_env


def _get_role(msg):
    if isinstance(msg, dict):
        return msg.get("role") or msg.get("type")
    return getattr(msg, "role", None) or getattr(msg, "type", None)


def _extract_text_messages(messages, user_input: str):
    texts = []
    for msg in messages:
        role = _get_role(msg)
        content = getattr(msg, "content", msg)
        if not content:
            continue
        if role in {"user", "system"}:
            continue
        if content.strip() == user_input.strip():
            continue
        texts.append(content)
    return texts


def main():
    load_env()
    agent = create_weather_agent()

    print("Interactive Weather Agent (Đà Nẵng)")
    print("Gõ 'exit' để thoát.")
    print("- Ví dụ: 'Thời tiết Sơn Trà hôm nay' hoặc 'Dự báo 3 ngày tới ở Đà Nẵng'")
    print()

    while True:
        user_input = input("Bạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        messages = response.get("messages", [])
        texts = _extract_text_messages(messages, user_input)

        if not texts:
            print("Trợ lý: (không có phản hồi)")
        else:
            print("Trợ lý:")
            print("\n".join(texts))
        print()


if __name__ == "__main__":
    main()
