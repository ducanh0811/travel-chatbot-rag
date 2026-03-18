import requests
import os
from typing import Optional
from langchain_core.tools import tool
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

def load_env():
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY không được tìm thấy trong file .env")
    return api_key

BASE_URL = "https://api.tavily.com"

def tavily_search_request(query: str, search_depth: str = "basic", max_results: int = 5, domain: Optional[str] = None) -> dict:
    """Gửi request tới Tavily API để tìm kiếm với tùy chọn giới hạn domain"""
    api_key = load_env()
    url = f"{BASE_URL}/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False,
        "max_results": max_results
    }
    if domain:
        payload["domain"] = domain
    response = requests.post(url, json=payload, timeout=30)
    return response.json()

@tool
def tavily_search_deep(query: str, max_results: int = 5, domain: Optional[str] = None) -> str:
    """Tìm kiếm sâu với thông tin chi tiết hơn về các sự kiện lễ hội tại Đà Nẵng sử dụng Tavily API (không giới hạn domain)"""
    try:
        # Gửi yêu cầu tìm kiếm sâu
        data = tavily_search_request(query, "advanced", max_results, domain)
        
        if "error" in data:
            return f"❌ Lỗi từ Tavily API: {data['error']}"
        
        if not data.get("results"):
            return f"❌ Không tìm thấy kết quả nào cho: {query}"
        
        result = f"🔍 **Tìm kiếm sâu cho:** {query}\n\n"
        
        if data.get("answer"):
            result += f"💡 **Phân tích chi tiết:** {data['answer']}\n\n"
        
        result += "📚 **Thông tin chi tiết:**\n"
        for i, item in enumerate(data["results"][:max_results], 1):
            title = item.get("title", "Không có tiêu đề")
            url = item.get("url", "")
            content = item.get("content", "")
            
            result += f"\n**{i}. {title}**\n"
            if content:
                # Giữ nhiều content hơn cho tìm kiếm sâu
                short_content = content[:400] + "..." if len(content) > 400 else content
                result += f"   {short_content}\n"
            if url:
                result += f"   🔗 {url}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Lỗi khi tìm kiếm sâu: {str(e)}"
