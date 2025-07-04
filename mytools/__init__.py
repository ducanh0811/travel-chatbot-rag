"""
MyTools - Bộ công cụ cho Gemini Agent với Weather và Web Search
"""

# Import weather tools
# Lưu ý: Bạn chưa có get_weather_forecast trong __all__, có thể bạn muốn thêm vào
from .weather import get_weather, get_weather_forecast, list_available_locations

# Import web search tools
# SỬA Ở ĐÂY: Đổi tên 'search_web' thành 'web_search' để khớp với file web_search.py
from .web_search import web_search 

# __all__ định nghĩa những gì sẽ được import khi dùng 'from mytools import *'
# Đây là một good practice để kiểm soát namespace.
__all__ = [
    'get_weather',
    'get_weather_forecast', # Thêm vào để có thể dùng được
    'list_available_locations', 
    'web_search' # SỬA Ở ĐÂY: Đổi tên cho khớp
]