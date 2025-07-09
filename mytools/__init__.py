"""
MyTools - Bộ công cụ cho Gemini Agent với Weather và Tavily Search
"""

# Import weather tools
from .weather import get_weather, get_weather_forecast

# Import tavily search tools  
from .tavily import tavily_search, tavily_search_deep

# __all__ định nghĩa những gì sẽ được import khi dùng 'from mytools import *'
# Đây là một good practice để kiểm soát namespace.
__all__ = [
    'get_weather',
    'get_weather_forecast',
    'tavily_search',
    'tavily_search_deep',
]