from fastapi import FastAPI, Request
from pydantic import BaseModel
from Supervisor import create_supervisor_agent

app = FastAPI()
supervisor_agent = create_supervisor_agent()  # Khởi tạo supervisor 1 lần duy nhất khi server start

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_agent(data: QueryRequest):
    query = data.query
    try:
        result = supervisor_agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", [])
        response_texts = []

        for msg in messages:
            content = getattr(msg, "content", msg)
            if content and not any(kw in content.lower() for kw in ["transferred to", "transferring", "successfully transfer"]):
                if content != query:
                    response_texts.append(content)

        if not response_texts:
            return {"result": "❌ Không có phản hồi nội dung từ agent."}

        return {"result": "\n\n".join(response_texts)}

    except Exception as e:
        return {"error": str(e)}