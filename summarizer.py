from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer that creates context for an assistant.\n"
    "Update the summary using the previous summary and new messages.\n"
    "Keep only key facts: user goals, constraints, locations, timing, preferences,"
    " decisions, and agreed results.\n"
    "Ignore greetings and small talk.\n"
    "Respond in the user's detected language.\n"
    "Output 3-7 concise bullet points."
)


class ConversationSummarizer:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        load_dotenv()
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.max_recent_messages = 8

    def summarize(self, messages: List[Dict], previous_summary: str = "") -> str:
        rendered_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            role_name = "user" if role == "user" else "assistant"
            rendered_messages.append(f"[{role_name}] {content}")

        prompt = (
            "<instructions>\n"
            f"{SUMMARY_SYSTEM_PROMPT}\n"
            "Summarize ONLY assistant information and new user information.\n"
            "</instructions>\n"
            "<assistant>\n"
            f"Previous summary:\n{previous_summary or '(none)'}\n"
            "</assistant>\n"
            "<user>\n"
            "New messages:\n"
            + "\n".join(rendered_messages)
            + "\n</user>"
        )

        response = self.llm.invoke(prompt)
        return response.content.strip()
