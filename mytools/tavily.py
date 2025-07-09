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

def tavily_search_request(query: str, search_depth: str = "basic", max_results: int = 5) -> dict:
    """Gửi request tới Tavily API để tìm kiếm"""
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
    
    response = requests.post(url, json=payload, timeout=30)
    return response.json()

@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """Tìm kiếm thông tin trên internet sử dụng Tavily AI search engine."""
    try:
        data = tavily_search_request(query, "basic", max_results)
        
        if "error" in data:
            return f"❌ Lỗi từ Tavily API: {data['error']}"
        
        if not data.get("results"):
            return f"❌ Không tìm thấy kết quả nào cho: {query}"
        
        # Lấy answer nếu có
        result = f"🔍 **Kết quả tìm kiếm cho:** {query}\n\n"
        
        if data.get("answer"):
            result += f"💡 **Tóm tắt:** {data['answer']}\n\n"
        
        # Hiển thị kết quả tìm kiếm
        result += "📖 **Chi tiết:**\n"
        for i, item in enumerate(data["results"][:max_results], 1):
            title = item.get("title", "Không có tiêu đề")
            url = item.get("url", "")
            content = item.get("content", "")
            
            result += f"\n**{i}. {title}**\n"
            if content:
                # Cắt ngắn content nếu quá dài
                short_content = content[:200] + "..." if len(content) > 200 else content
                result += f"   {short_content}\n"
            if url:
                result += f"   🔗 {url}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Lỗi khi tìm kiếm: {str(e)}"


@tool
def tavily_search_deep(query: str, max_results: int = 3) -> str:
    """Tìm kiếm sâu với thông tin chi tiết hơn sử dụng Tavily API."""
    try:
        data = tavily_search_request(query, "advanced", max_results)
        
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