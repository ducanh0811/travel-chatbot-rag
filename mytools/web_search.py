import os
import re
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# --- Phần 1: Kiểm tra API Key ---
# Đảm bảo bạn đã đặt biến môi trường TAVILY_API_KEY
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("Vui lòng đặt biến môi trường TAVILY_API_KEY để sử dụng công cụ tìm kiếm.")

# --- Phần 2: Keywords và địa điểm được hỗ trợ ---
# Keywords du lịch được phép tìm kiếm
TOURISM_KEYWORDS = {
    'accommodation': ['khách sạn', 'resort', 'homestay', 'villa', 'nhà nghỉ', 'lưu trú'],
    'attractions': ['điểm đến', 'danh lam', 'thắng cảnh', 'du lịch', 'tham quan', 'check in', 'checkin'],
    'food': ['ẩm thực', 'món ăn', 'quán ăn', 'nhà hàng', 'đặc sản', 'street food'],
    'activities': ['hoạt động', 'trải nghiệm', 'tour', 'vui chơi', 'giải trí', 'festival', 'lễ hội'],
    'transportation': ['di chuyển', 'giao thông', 'xe', 'máy bay', 'tàu hỏa'],
    'shopping': ['mua sắm', 'chợ', 'siêu thị', 'shopping', 'quà lưu niệm'],
    'culture': ['văn hóa', 'lịch sử', 'truyền thống', 'chùa', 'đền', 'bảo tàng']
}

# Các địa điểm trong khu vực Đà Nẵng - Quảng Nam
DANANG_QUANGNAM_LOCATIONS = [
    'đà nẵng', 'da nang', 'danang',
    'hội an', 'hoi an', 'hoian',
    'quảng nam', 'quang nam',
    'mỹ sơn', 'my son', 'myson',
    'bà nà', 'ba na', 'bana',
    'sơn trà', 'son tra', 'linh ứng',
    'hấp dẫn', 'cù lao chàm', 'cu lao cham',
    'tam kỳ', 'tam ky',
    'thừa thiên huế', 'thua thien hue', 'huế', 'hue'  # Khu vực lân cận
]

# --- Phần 3: Helper Functions ---
def is_tourism_related(query: str) -> bool:
    """
    Kiểm tra xem query có liên quan đến du lịch không.
    
    Args:
        query: Query cần kiểm tra
        
    Returns:
        True nếu query liên quan đến du lịch, False nếu không
    """
    query_lower = query.lower()
    
    # Kiểm tra có từ khóa du lịch
    for category, keywords in TOURISM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                return True
            
    return False

def has_danang_location(query: str) -> bool:
    """
    Kiểm tra xem query có nhắc đến khu vực Đà Nẵng - Quảng Nam không.
    
    Args:
        query: Query cần kiểm tra
        
    Returns:
        True nếu có địa điểm Đà Nẵng/Quảng Nam, False nếu không
    """
    query_lower = query.lower()
    
    # Kiểm tra có địa điểm trong khu vực
    for location in DANANG_QUANGNAM_LOCATIONS:
        if location in query_lower:
            return True
            
    return False

def has_other_location(query: str) -> bool:
    """
    Kiểm tra xem query có nhắc đến địa điểm KHÁC khu vực Đà Nẵng - Quảng Nam không.
    
    Args:
        query: Query cần kiểm tra
        
    Returns:
        True nếu có địa điểm khác, False nếu không
    """
    query_lower = query.lower()
    
    # Danh sách các địa điểm khác thường gặp
    other_locations = [
        'hà nội', 'hanoi', 'ha noi',
        'hồ chí minh', 'tp hcm', 'sài gòn', 'saigon', 'ho chi minh',
        'nha trang', 'phú quốc', 'phu quoc', 'phuquoc',
        'đà lạt', 'da lat', 'dalat',
        'hạ long', 'ha long', 'halong',
        'cần thơ', 'can tho',
        'vũng tàu', 'vung tau',
        'phan thiết', 'phan thiet',
        'quy nhon', 'quy nhơn',
        'singapore', 'thailand', 'thái lan',
        'malaysia', 'bali', 'tokyo', 'seoul',
        'paris', 'london', 'new york', 'bangkok',
        'cambodia', 'campuchia', 'myanmar',
        'philippines', 'indonesia'
    ]
    
    # Kiểm tra có địa điểm khác
    for location in other_locations:
        if location in query_lower:
            return True
            
    return False

def enhance_query_with_location(query: str) -> str:
    """
    Thêm keywords về Đà Nẵng - Quảng Nam vào query để tăng độ chính xác.
    
    Args:
        query: Query gốc
        
    Returns:
        Query đã được enhance với location keywords
    """
    query_lower = query.lower()
    
    # Kiểm tra đã có địa điểm chưa
    has_location = any(location in query_lower for location in DANANG_QUANGNAM_LOCATIONS)
    
    if not has_location:
        # Thêm "Đà Nẵng" nếu chưa có địa điểm cụ thể
        enhanced_query = f"{query} Đà Nẵng"
    else:
        enhanced_query = query
    
    # Thêm keyword "du lịch" nếu chưa có
    tourism_found = any(keyword in query_lower for category in TOURISM_KEYWORDS.values() for keyword in category)
    if not tourism_found:
        enhanced_query = f"du lịch {enhanced_query}"
    
    return enhanced_query

def filter_tourism_results(search_results: str, original_query: str, auto_added_danang: bool = False) -> str:
    """
    Filter kết quả tìm kiếm để chỉ giữ lại thông tin du lịch liên quan.
    
    Args:
        search_results: Kết quả tìm kiếm gốc
        original_query: Query ban đầu
        auto_added_danang: Có tự động thêm Đà Nẵng không
        
    Returns:
        Kết quả đã được filter
    """
    # Nếu kết quả quá ngắn, trả về nguyên văn
    if len(search_results) < 100:
        return search_results
    
    # Thêm lời khuyên về scope của tool
    if auto_added_danang:
        filtered_note = f"""
🎯 Tôi chỉ cung cấp thông tin du lịch ở Đà Nẵng - Quảng Nam.

Kết quả tìm kiếm cho "{original_query}" tại Đà Nẵng:
---
{search_results}
---
💡 Lần sau bạn có thể hỏi trực tiếp: "{original_query} ở Đà Nẵng" để rõ ràng hơn.
"""
    else:
        filtered_note = f"""
🎯 Kết quả tìm kiếm du lịch cho "{original_query}":
---
{search_results}
---
💡 Để có kết quả tốt nhất, hãy hỏi về: địa điểm du lịch, ẩm thực, khách sạn, hoạt động vui chơi, văn hóa trong khu vực Đà Nẵng - Quảng Nam.
"""
    
    return filtered_note

# --- Phần 4: Khởi tạo Tool tìm kiếm ---
tavily_search = TavilySearch(max_results=3)

# --- Phần 5: Định nghĩa tool `web_search` theo chuẩn của Agent ---
@tool
def web_search(query: str) -> str:
    """
    Tìm kiếm thông tin du lịch trên web cho khu vực Đà Nẵng - Quảng Nam.
    Tool này chỉ tập trung vào các thông tin liên quan đến du lịch, ẩm thực, 
    lưu trú, điểm tham quan, hoạt động vui chơi trong khu vực Đà Nẵng - Quảng Nam.
    
    Sử dụng tool này khi bạn cần thông tin cập nhật về:
    - Địa điểm du lịch, danh lam thắng cảnh
    - Khách sạn, resort, homestay
    - Ẩm thực, món ăn đặc sản  
    - Hoạt động vui chơi, trải nghiệm
    - Văn hóa, lịch sử địa phương
    - Sự kiện, lễ hội địa phương
    
    Args:
        query: Truy vấn tìm kiếm về du lịch Đà Nẵng - Quảng Nam. 
               Ví dụ: "khách sạn 5 sao gần biển Mỹ Khê", "món ăn đặc sản Hội An"
    
    Returns:
        Thông tin du lịch được filter cho khu vực Đà Nẵng - Quảng Nam.
    """
    
    # Bước 1: Kiểm tra query có liên quan đến du lịch không
    if not is_tourism_related(query):
        return f"""❌ Xin lỗi, tool này chỉ hỗ trợ tìm kiếm thông tin du lịch trong khu vực Đà Nẵng - Quảng Nam.

🎯 Bạn có thể hỏi về:
• 🏨 Khách sạn, resort, homestay
• 🏖️ Điểm du lịch, danh lam thắng cảnh  
• 🍜 Ẩm thực, món ăn đặc sản
• 🎭 Hoạt động vui chơi, văn hóa
• 🛍️ Mua sắm, chợ đêm
• 🚗 Phương tiện di chuyển

💡 Ví dụ: "khách sạn gần biển Mỹ Khê", "món ăn ngon ở Hội An", "tour Bà Nà Hills"

Query của bạn: "{query}" không thuộc phạm vi hỗ trợ."""

    # Bước 2: Kiểm tra có địa điểm khác Đà Nẵng không
    if has_other_location(query):
        return f"""❌ Xin lỗi, tôi chỉ cung cấp thông tin du lịch ở khu vực Đà Nẵng - Quảng Nam.

🎯 Bạn đã hỏi về "{query}" - có vẻ thuộc địa điểm khác.

💡 Thay vào đó, bạn có thể hỏi:
• "khách sạn 5 sao ở Đà Nẵng"
• "món ăn đặc sản Hội An"  
• "tour Bà Nà Hills"
• "điểm tham quan Sơn Trà"
• "lễ hội đèn lồng Hội An"

🌟 Khu vực Đà Nẵng - Quảng Nam có rất nhiều điều thú vị để khám phá!"""

    # Bước 3: Kiểm tra có nhắc đến Đà Nẵng không
    has_danang = has_danang_location(query)
    
    # Enhance query với location keywords
    enhanced_query = enhance_query_with_location(query)
    
    print(f"\n---> [Tourism Search] Original query: '{query}'")
    print(f"---> [Tourism Search] Enhanced query: '{enhanced_query}'")
    print(f"---> [Tourism Search] Has Da Nang location: {has_danang}")
    print(f"---> [Tourism Search] Auto-add Da Nang: {not has_danang}")
    
    try:
        # Gọi công cụ Tavily với enhanced query
        search_results = tavily_search.invoke(enhanced_query)
        
        if not search_results:
             return f"⚠️ Không tìm thấy thông tin du lịch nào cho '{query}' trong khu vực Đà Nẵng - Quảng Nam."

        # Filter kết quả - nếu không có location thì auto_added_danang = True
        filtered_results = filter_tourism_results(search_results, query, not has_danang)
        
        print("---> [Tourism Search] Tourism-focused search completed.")
        return filtered_results

    except Exception as e:
        print(f"---> [Tourism Search Error] {e}")
        return f"❌ Đã xảy ra lỗi khi tìm kiếm thông tin du lịch: {e}"

# --- Phần 6: Schema cập nhật ---
WEB_SEARCH_TOOL_SCHEMA = {
    "name": "web_search",
    "description": (
        "Tìm kiếm thông tin du lịch trên web cho khu vực Đà Nẵng - Quảng Nam. "
        "Tool này chỉ tập trung vào các thông tin liên quan đến du lịch, ẩm thực, "
        "lưu trú, điểm tham quan, hoạt động vui chơi trong khu vực này."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Truy vấn tìm kiếm về du lịch Đà Nẵng - Quảng Nam. Ví dụ: 'khách sạn 5 sao gần biển Mỹ Khê', 'món ăn đặc sản Hội An'."
            }
        },
        "required": ["query"]
    }
}