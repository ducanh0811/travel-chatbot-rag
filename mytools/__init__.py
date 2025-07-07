"""
MyTools - Bộ công cụ cho Gemini Agent với Weather và Web Search
"""

# Import weather tools
# Lưu ý: Bạn chưa có get_weather_forecast trong __all__, có thể bạn muốn thêm vào
from .weather import get_weather, get_weather_forecast


# __all__ định nghĩa những gì sẽ được import khi dùng 'from mytools import *'
# Đây là một good practice để kiểm soát namespace.
__all__ = [
    'get_weather',
    'get_weather_forecast', # Thêm vào để có thể dùng được
]