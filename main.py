from fastapi import FastAPI, Request
from pydantic import BaseModel
from Supervisor import create_supervisor_agent
import uvicorn

app = FastAPI()
supervisor = create_supervisor_agent()  

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    query = request.query
    try:
        response = supervisor.invoke({"messages": {"role": "user", "content": query}})
        if not response or "messages" not in response:
            return {"error": "Không có phản hồi từ agent"}

        messages = response["messages"]
        result = []
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else msg
            if content and not any(s in content.lower() for s in ["transferred to", "transferring", "successfully transfer"]):
                result.append(content)

        return {"response": result or ["Không có nội dung phù hợp để trả về."]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
