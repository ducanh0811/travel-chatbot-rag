import os
import re
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import time

# ============ CONFIG ============
CACHE_TTL_SECONDS = 3600  # Cache 1 giờ
MAX_CACHE_SIZE = 100

def load_env():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("⚠️ Vui lòng thiết lập biến môi trường OPENAI_API_KEY trong .env")
    return api_key

def get_embedding_and_vectorstore():
    # Khởi tạo embedding và vectorstore song song
    with ThreadPoolExecutor() as executor:
        future_embedding = executor.submit(OpenAIEmbeddings)
        embedding = future_embedding.result()
        future_vectorstore = executor.submit(
            lambda: Chroma(
                persist_directory="chromadb",
                embedding_function=embedding,
            )
        )
        vectorstore = future_vectorstore.result()
    return embedding, vectorstore

# ============ QUERY FILTER EXTRACTION ============
# Mapping từ khóa → category trong metadata
CATEGORY_KEYWORDS = {
    "khách sạn": "hotel",
    "hotel": "hotel",
    "resort": "hotel",
    "homestay": "hotel",
    "nhà nghỉ": "hotel",
    "nhà hàng": "restaurant",
    "quán ăn": "restaurant",
    "restaurant": "restaurant",
    "cafe": "cafe",
    "cà phê": "cafe",
    "coffee": "cafe",
    "quán cafe": "cafe",
    "quán cà phê": "cafe",
    "địa điểm": "destination",
    "bãi biển": "destination",
    "biển": "destination",
    "chùa": "destination",
    "bảo tàng": "destination",
    "công viên": "destination",
    "cầu": "destination",
    "sự kiện": "event",
    "lễ hội": "event",
    "festival": "event",
}

# Mapping từ khóa → type trong metadata
TYPE_KEYWORDS = {
    # Cafe
    "cafe": "quán cà phê",
    "cà phê": "quán cà phê",
    "coffee": "quán cà phê",
    "quán cafe": "quán cà phê",
    "quán cà phê": "quán cà phê",
    
    # Hotel
    "khách sạn": "hotel",
    "hotel": "hotel",
    "resort": "hotel",
    "homestay": "hotel",
    "nhà nghỉ": "hotel",
    "5 sao": "5 sao",
    "4 sao": "4 sao",
    "3 sao": "3 sao",
    "2 sao": "2 sao",
    
    # Restaurant
    "nhà hàng": "restaurant",
    "quán ăn": "local_eatery",
    "restaurant": "restaurant",
    "quán": "local_eatery",
    "ăn uống": "restaurant",
    
    # Destination
    "bãi biển": "bãi biển",
    "biển": "bãi biển",
    "beach": "bãi biển",
    "chùa": "chùa",
    "đền": "đền",
    "nhà thờ": "nhà thờ",
    "cầu": "cầu",
    "công viên": "công viên",
    "bảo tàng": "bảo tàng",
    "chợ": "chợ",
    "núi": "núi",
    "địa điểm": None,  # Không filter cụ thể
}

# Mapping từ khóa → district
DISTRICT_KEYWORDS = {
    "sơn trà": "Sơn Trà",
    "son tra": "Sơn Trà",
    "hải châu": "Hải Châu",
    "hai chau": "Hải Châu",
    "thanh khê": "Thanh Khê",
    "thanh khe": "Thanh Khê",
    "liên chiểu": "Liên Chiểu",
    "lien chieu": "Liên Chiểu",
    "ngũ hành sơn": "Ngũ Hành Sơn",
    "ngu hanh son": "Ngũ Hành Sơn",
    "cẩm lệ": "Cẩm Lệ",
    "cam le": "Cẩm Lệ",
    "hòa vang": "Hòa Vang",
    "hoa vang": "Hòa Vang",
}

def _find_first_keyword_match(query_lower: str, keywords: Dict[str, Any]) -> Optional[Any]:
    # Ưu tiên keyword dài hơn để tránh match chung chung
    for keyword in sorted(keywords.keys(), key=len, reverse=True):
        if keyword in query_lower:
            return keywords[keyword]
    return None

def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """
    Phân tích câu hỏi để trích xuất filter cho retriever.
    Trả về dict với các key: type, district (nếu tìm thấy)
    """
    query_lower = query.lower()
    filters = {}
    
    # Tìm category
    detected_category = _find_first_keyword_match(query_lower, CATEGORY_KEYWORDS)
    if detected_category:
        filters["category"] = detected_category

    # Tìm type (ưu tiên "x sao" nếu có)
    detected_type = None
    star_match = re.search(r"\b([2-5])\s*sao\b", query_lower)
    if star_match:
        detected_type = f"{star_match.group(1)} sao"
    else:
        detected_type = _find_first_keyword_match(query_lower, TYPE_KEYWORDS)
        if detected_type is None:
            detected_type = None
    
    if detected_type:
        generic_types = {"hotel", "restaurant"}
        if not (filters.get("category") in {"hotel", "restaurant"} and detected_type in generic_types):
            filters["type"] = detected_type
    
    # Tìm district
    detected_district = None
    for keyword, district_value in DISTRICT_KEYWORDS.items():
        if keyword in query_lower:
            detected_district = district_value
            break
    
    if detected_district:
        filters["district"] = detected_district

    # Tìm month cho sự kiện (ví dụ: "tháng 6")
    if filters.get("category") == "event":
        month_match = re.search(r"tháng\s*(\d{1,2})", query_lower)
        if month_match:
            month_val = int(month_match.group(1))
            if 1 <= month_val <= 12:
                filters["month"] = month_val
    
    return filters

def build_chroma_filter(filters: Dict[str, Any]) -> Optional[Dict]:
    """
    Chuyển đổi filters thành format Chroma where clause.
    """
    if not filters:
        return None
    
    conditions = []
    for key, value in filters.items():
        conditions.append({key: {"$eq": value}})
    
    if len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

# ============ CACHE SYSTEM ============
class RAGCache:
    """Simple in-memory cache với TTL"""
    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def _hash_key(self, query: str, filters: Optional[Dict]) -> str:
        key_str = f"{query}|{str(filters)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, filters: Optional[Dict]) -> Optional[str]:
        key = self._hash_key(query, filters)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["result"]
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, filters: Optional[Dict], result: str):
        key = self._hash_key(query, filters)
        # Evict oldest if cache full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def clear(self):
        self.cache.clear()

# Global cache instance
rag_cache = RAGCache()

# ============ PROMPT ============
QA_PROMPT = PromptTemplate.from_template("""
    Bạn là một hướng dẫn viên du lịch thông minh. Trả lời câu hỏi của người dùng bằng tiếng Việt, dựa trên thông tin từ dữ liệu địa điểm bên dưới.

    QUAN TRỌNG:
    - Chỉ trả lời đúng loại địa điểm mà người dùng hỏi (ví dụ: chỉ khách sạn nếu hỏi khách sạn, chỉ quán cafe nếu hỏi cafe, v.v.).
    - Ưu tiên các địa điểm có tên hoặc khu vực trùng khớp với truy vấn của người dùng (ví dụ: nếu hỏi về Sơn Trà thì ưu tiên các địa điểm ở khu vực Sơn Trà hoặc có tên Sơn Trà).
    - Không gộp các loại địa điểm khác nhau vào cùng một câu trả lời.
    - Nếu không tìm thấy địa điểm phù hợp, hãy trả lời rõ ràng là không có kết quả phù hợp.
    - Với sự kiện: ưu tiên sự kiện sắp tới/đang diễn ra; nếu có tháng trong câu hỏi thì chỉ chọn sự kiện trong tháng đó.
    
    Mỗi gợi ý trình bày theo mẫu:
                                       
    Tên: (Tên địa điểm)  
    Loại: (Loại địa điểm: quán ăn, cafe, khách sạn...)  
    Khu vực: (Phường, Quận)  
    Mô tả: (Mô tả ngắn gọn, hấp dẫn, nổi bật)  
    Thời gian mở đóng: (Open-Close)/(Checkin/Checkout)
    Thời gian gợi ý ở lại: (Duration_suggested_min) <Hotel thì bỏ qua>
    -----------------------------------------------

    Nếu người dùng yêu cầu chi tiết hơn, bạn có thể mô tả sâu hơn về các dịch vụ, giờ mở cửa, phù hợp với nhóm nào, v.v.

    Câu hỏi của người dùng: {question}  
    Dữ liệu địa điểm truy xuất được:  
    {context}

    Trả lời:
    """)

# ============ OUTPUT GUARDRAIL ============
def validate_and_clean_output(result: str, query: str) -> str:
    """
    Kiểm tra và làm sạch output từ RAG.
    - Đảm bảo format đúng
    - Loại bỏ thông tin không liên quan
    - Thêm disclaimer nếu cần
    """
    if not result or len(result.strip()) < 10:
        return "❌ Xin lỗi, tôi không tìm thấy thông tin phù hợp với yêu cầu của bạn. Vui lòng thử lại với từ khóa khác."
    
    # Kiểm tra nếu kết quả quá ngắn hoặc không có nội dung hữu ích
    if "không tìm thấy" in result.lower() or "không có kết quả" in result.lower():
        return result
    
    # Thêm footer nếu có nhiều kết quả
    if result.count("Tên:") >= 3:
        result += "\n\n💡 *Mẹo: Bạn có thể hỏi chi tiết hơn về một địa điểm cụ thể!*"
    
    return result

# ============ KHỞI TẠO ============
load_env()
embedding, vectorstore = get_embedding_and_vectorstore()

@tool
def rag_tool(query: str) -> str:
    """
    Trả lời câu hỏi dựa trên vectorstore đã khởi tạo từ ChromaDB.
    Tự động phân tích query để áp dụng filter theo type/district.
    Có cache để tối ưu hiệu năng.
    """
    global vectorstore, rag_cache
    
    if vectorstore is None:
        raise ValueError("Vectorstore chưa được gán. Vui lòng khởi tạo trong Agent.")
    
    # Extract filters từ query
    filters = extract_filters_from_query(query)
    chroma_filter = build_chroma_filter(filters)
    
    # Check cache
    cached_result = rag_cache.get(query, filters)
    if cached_result:
        return cached_result
    
    # Build retriever với filter
    search_kwargs = {"k": 5}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
    
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs=search_kwargs
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    result = qa_chain.invoke(query)["result"]
    
    # Apply guardrail
    result = validate_and_clean_output(result, query)
    
    # Cache result
    rag_cache.set(query, filters, result)
    
    return result

# ============ UTILITY FUNCTIONS ============
def reload_vectorstore():
    """Hot-reload vectorstore khi dữ liệu thay đổi"""
    global embedding, vectorstore
    embedding, vectorstore = get_embedding_and_vectorstore()
    rag_cache.clear()
    return True

def get_cache_stats():
    """Lấy thống kê cache"""
    return {
        "size": len(rag_cache.cache),
        "max_size": rag_cache.max_size,
        "ttl": rag_cache.ttl
    }

if __name__ == "__main__":
    test_queries = [
        "Gợi ý quán cafe ở Sơn Trà",
        "Khách sạn 5 sao ở Hải Châu",
        "Nhà hàng ngon ở Thanh Khê",
        "Bãi biển đẹp ở Đà Nẵng"
    ]
    
    for test_query in test_queries:
        print(f"\n⚙️ Query: {test_query}")
        filters = extract_filters_from_query(test_query)
        print(f"📋 Filters: {filters}")
        try:
            result = rag_tool(test_query)
            print(f"🤖 Kết quả:\n{result[:500]}...")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
