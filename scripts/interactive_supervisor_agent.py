import sys
import pathlib

from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Supervisor as supervisor_module
from summarizer import ConversationSummarizer


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
        if isinstance(content, str) and content.strip() == user_input.strip():
            continue
        if isinstance(content, str) and any(
            s in content.lower()
            for s in ["transferred to", "transferring", "successfully transfer"]
        ):
            continue
        texts.append(content)
    return texts


def _build_context(history, last_n: int = 3) -> str:
    if not history:
        return ""
    recent = history[-last_n * 2 :]
    lines = []
    for msg in recent:
        role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def main():
    load_dotenv()
    summarizer = ConversationSummarizer()
    history = []
    summary = ""
    summary_message_count = 0

    print("Interactive Supervisor Agent (Sự kiện + Thời tiết Đà Nẵng)")
    print("Gõ 'exit' để thoát.")
    print("- Ví dụ: 'Sự kiện hôm nay ở Đà Nẵng' hoặc 'Thời tiết Sơn Trà hôm nay'")
    print()

    while True:
        user_input = input("Bạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        context = _build_context(history, last_n=3)
        summary_message = f"Tóm tắt hội thoại trước đó:\n{summary}" if summary else ""
        enhanced_query = user_input

        if context:
            enhanced_query = (
                "Lịch sử hội thoại gần đây:\n"
                f"{context}\n\n"
                f"Câu hỏi mới: {user_input}\n\n"
                "Hãy trả lời câu hỏi mới, có thể tham khảo lịch sử hội thoại nếu liên quan."
            )

        if summary_message:
            enhanced_query = f"{summary_message}\n\n{enhanced_query}"

        texts = supervisor_module.run_supervisor_query(
            enhanced_query, use_singleton=True, print_output=True
        )

        if texts:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": "\n".join(texts)})
            if len(history) != summary_message_count:
                summary = summarizer.summarize(
                    history[-summarizer.max_recent_messages :], summary
                )
                summary_message_count = len(history)


if __name__ == "__main__":
    main()
