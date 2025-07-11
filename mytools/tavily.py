import requests
import os
from typing import Optional, List, Dict
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("TAVILY_API_KEY")
if not API_KEY:
    raise ValueError("TAVILY_API_KEY không được tìm thấy trong file .env")

BASE_URL = "https://api.tavily.com"

def tavily_search_request(query: str, search_depth: str = "basic", max_results: int = 5, domain: Optional[str] = None) -> dict:
    """Gửi request tới Tavily API để tìm kiếm với tùy chọn giới hạn domain"""
    url = f"{BASE_URL}/search"
    payload = {
        "api_key": API_KEY,
        "query": query,
        "search_depth": search_depth,
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False,
        "max_results": max_results
    }

    if domain:  # Nếu có domain, thêm vào payload
        payload["domain"] = domain
    
    response = requests.post(url, json=payload, timeout=30)
    return response.json()

@tool
def tavily_search_deep(query: str, max_results: int = 3, domain: Optional[str] = "danangfantasticity.com") -> str:
    """Tìm kiếm sâu với thông tin chi tiết hơn về các sự kiện lễ hội tại Đà Nẵng sử dụng Tavily API và giới hạn domain."""
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
            
            # Chỉ lấy kết quả từ domain "danangfantasticity.com"
            if domain in url:
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
