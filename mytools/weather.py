import requests
from typing import Dict, Any, Optional, List
import json
from langchain_core.tools import tool

API_KEY = "ea4de82e443d888676a2250ef1c58aef"
BASE_URL = "http://api.openweathermap.org/data/2.5"

# Danh sách các địa điểm phục vụ - chỉ Đà Nẵng và Hội An
VALID_LOCATIONS = {
    # Đà Nẵng
    "đà nẵng": "Da Nang",
    "da nang": "Da Nang",
    "danang": "Da Nang",
    
    # Hội An  
    "hội an": "Hoi An, Quang Nam",
    "hoi an": "Hoi An, Quang Nam",
    "hoian": "Hoi An, Quang Nam"
}
#Hàm chuẩn hóa tên địa điểm
def validate_location(location: str) -> Optional[str]:
    """
    Kiểm tra và chuẩn hóa tên địa điểm trong khu vực Đà Nẵng mới.
    
    Args:
        location: Tên địa điểm được nhập
        
    Returns:
        Tên địa điểm chuẩn hóa để gọi API hoặc None nếu không hợp lệ
    """
    location_lower = location.lower().strip()
    
    # Tìm kiếm chính xác
    if location_lower in VALID_LOCATIONS:
        return VALID_LOCATIONS[location_lower]
    
    # Tìm kiếm gần đúng
    for key, value in VALID_LOCATIONS.items():
        if location_lower in key or key in location_lower:
            return value
    
    return None

@tool
def get_weather(location: str, units: str = "metric", lang: str = "vi") -> str:
    """
    Lấy thông tin thời tiết hiện tại tại Đà Nẵng hoặc Hội An.
    
    Args:
        location: Tên địa điểm ('đà nẵng' hoặc 'hội an')
        units: Đơn vị đo (metric, imperial, kelvin) - mặc định metric
        lang: Ngôn ngữ hiển thị - mặc định tiếng Việt
        
    Returns:
        Thông tin thời tiết chi tiết hoặc thông báo lỗi nếu địa điểm không được hỗ trợ
    """
    # Kiểm tra địa điểm hợp lệ
    validated_location = validate_location(location)
    if not validated_location:
        return f"""❌ Địa điểm '{location}' không được hỗ trợ.

🌍 Chỉ cung cấp thời tiết cho 2 địa điểm:
• 🏙️ Đà Nẵng (Da Nang, DaNang)
• 🏮 Hội An (Hoi An, HoiAn)

💡 Vui lòng nhập 'đà nẵng' hoặc 'hội an'."""
    
    url = f"{BASE_URL}/weather?appid={API_KEY}&q={validated_location}&units={units}&lang={lang}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code == 401:
            return "❌ Lỗi xác thực API Key. Vui lòng kiểm tra lại API Key."

        elif data.get("cod") == 200:
            main = data["main"]
            weather = data["weather"][0]
            wind_data = data.get("wind", {})
            clouds = data.get("clouds", {})
            sys = data.get("sys", {})
            
            # Extract data
            temperature = main["temp"]
            feels_like = main["feels_like"]
            humidity = main["humidity"]
            pressure = main["pressure"]
            temp_min = main.get("temp_min")
            temp_max = main.get("temp_max")
            
            weather_desc = weather["description"]
            weather_main = weather["main"]
            
            wind_speed = wind_data.get("speed", "N/A")
            wind_deg = wind_data.get("deg")
            wind_gust = wind_data.get("gust")
            
            cloudiness = clouds.get("all", 0)
            
            sunrise = sys.get("sunrise")
            sunset = sys.get("sunset")
            
            # Wind direction helper
            def wind_direction_from_deg(deg):
                if deg is None:
                    return "Không có thông tin"
                directions = ["Bắc", "Đông Bắc", "Đông", "Đông Nam", "Nam", "Tây Nam", "Tây", "Tây Bắc"]
                idx = round(deg / 45) % 8
                return directions[idx]

            # Format time
            def format_time(timestamp):
                if timestamp:
                    from datetime import datetime
                    return datetime.fromtimestamp(timestamp).strftime("%H:%M")
                return "N/A"

            # Weather emoji
            weather_emoji = {
                "Clear": "☀️",
                "Clouds": "☁️",
                "Rain": "🌧️",
                "Drizzle": "🌦️",
                "Thunderstorm": "⛈️",
                "Snow": "❄️",
                "Mist": "🌫️",
                "Fog": "🌫️",
                "Haze": "🌫️"
            }.get(weather_main, "🌤️")

            # Xác suất có mưa - từ tình trạng thời tiết
            rain_probability = "0%"
            if weather_main in ["Rain", "Drizzle"]:
                rain_probability = "90-100%"
            elif weather_main == "Thunderstorm":
                rain_probability = "95-100%"
            elif weather_main == "Clouds" and cloudiness > 70:
                rain_probability = "20-40%"
            elif weather_main == "Clouds" and cloudiness > 50:
                rain_probability = "10-20%"

            result = f"{weather_emoji} Thời tiết hiện tại tại {location.title()}:\n"
            result += f"- Nhiệt độ: {temperature}°C\n"
            result += f"- Tình trạng: {weather_desc.capitalize()}\n"
            result += f"- Độ che phủ mây: {cloudiness}%\n"
            result += f"- Xác suất có mưa: {rain_probability}\n"
            
            if wind_speed != "N/A":
                result += f"- Tốc độ gió: {wind_speed} m/s"
                if wind_gust:
                    result += f" (giật tới {wind_gust} m/s)"
                result += f" - Hướng: {wind_direction_from_deg(wind_deg)}\n"
                
            # Weather advice
            if temperature > 30:
                result += "\n💡 Lời khuyên: Thời tiết khá nóng, nên mang theo nước và kem chống nắng!"
            elif temperature < 15:
                result += "\n💡 Lời khuyên: Thời tiết mát, nên mang theo áo ấm!"
            elif weather_main == "Rain":
                result += "\n💡 Lời khuyên: Có mưa, nhớ mang theo ô hoặc áo mưa!"
            elif humidity > 80:
                result += "\n💡 Lời khuyên: Độ ẩm cao, có thể cảm thấy oi bức!"

            # Location specific advice
            if "da nang" in validated_location.lower():
                result += f"\n🏙️ Thông tin thời tiết cho thành phố Đà Nẵng"
            elif "hoi an" in validated_location.lower():
                result += f"\n🏮 Thông tin thời tiết cho phố cổ Hội An"

            return result

        elif data.get("cod") == "404":
            return f"❌ Không tìm thấy thông tin thời tiết cho {location} trong hệ thống."
        else:
            return f"⚠️ Có lỗi xảy ra: {data.get('message', 'Không rõ lỗi.')}"
            
    except requests.exceptions.Timeout:
        return "❌ Hết thời gian chờ khi kết nối API thời tiết."
    except requests.exceptions.ConnectionError:
        return "❌ Không thể kết nối tới dịch vụ thời tiết."
    except Exception as e:
        return f"❌ Lỗi khi lấy thông tin thời tiết: {str(e)}"

@tool
def get_weather_forecast(location: str, days: int = 3) -> str:
    """
    Lấy dự báo thời tiết cho 2-3 ngày tới (từ ngày mai) tại Đà Nẵng hoặc Hội An.
    
    Chú ý: Tool này dự báo thời tiết cho những ngày sắp tới, KHÔNG thể lấy thời tiết 
    của một ngày cụ thể trong tương lai. Sử dụng khi user hỏi về "dự báo", "những ngày tới", 
    "tuần này", "ngày mai", v.v.
    
    Args:
        location: Tên địa điểm ('đà nẵng' hoặc 'hội an')
        days: Số ngày dự báo (2 hoặc 3) - mặc định 3 ngày
        
    Returns:
        Dự báo thời tiết chi tiết cho các ngày tới hoặc thông báo lỗi
    """
    # Kiểm tra số ngày hợp lệ
    if days not in [2, 3]:
        return "❌ Chỉ hỗ trợ dự báo cho 2 hoặc 3 ngày tới."
    
    # Kiểm tra địa điểm hợp lệ
    validated_location = validate_location(location)
    if not validated_location:
        return f"""❌ Địa điểm '{location}' không được hỗ trợ.

🌍 Chỉ cung cấp dự báo thời tiết cho 2 địa điểm:
• 🏙️ Đà Nẵng (Da Nang, DaNang)
• 🏮 Hội An (Hoi An, HoiAn)

💡 Vui lòng nhập 'đà nẵng' hoặc 'hội an'."""
    
    url = f"{BASE_URL}/forecast?appid={API_KEY}&q={validated_location}&units=metric&lang=vi"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code == 401:
            return "❌ Lỗi xác thực API Key. Vui lòng kiểm tra lại API Key."

        elif data.get("cod") == "200":
            forecasts = data["list"]
            
            # Group forecasts by day
            from datetime import datetime, timedelta
            today = datetime.now().date()
            daily_forecasts = {}
            
            for forecast in forecasts:
                forecast_date = datetime.fromtimestamp(forecast["dt"]).date()
                
                # Chỉ lấy dự báo cho các ngày tiếp theo
                if forecast_date > today:
                    if forecast_date not in daily_forecasts:
                        daily_forecasts[forecast_date] = []
                    daily_forecasts[forecast_date].append(forecast)
            
            # Chỉ lấy số ngày được yêu cầu
            sorted_dates = sorted(daily_forecasts.keys())[:days]
            
            if not sorted_dates:
                return "❌ Không có dữ liệu dự báo cho các ngày tới."
            
            result = f"📅 Dự báo thời tiết {days} ngày tới tại {location.title()}:\n\n"
            
            for i, date in enumerate(sorted_dates, 1):
                day_forecasts = daily_forecasts[date]
                
                # Lấy dự báo trung bình trong ngày
                temps = [f["main"]["temp"] for f in day_forecasts]
                weathers = [f["weather"][0]["main"] for f in day_forecasts]
                clouds = [f["clouds"]["all"] for f in day_forecasts]
                
                avg_temp = sum(temps) / len(temps)
                min_temp = min(temps)
                max_temp = max(temps)
                avg_clouds = sum(clouds) / len(clouds)
                
                # Xác suất có mưa dựa trên tần suất dự báo mưa trong ngày
                rain_count = sum(1 for w in weathers if w in ["Rain", "Drizzle", "Thunderstorm"])
                rain_probability = (rain_count / len(weathers)) * 100
                
                # Format ngày
                date_str = date.strftime("%d/%m")
                day_name = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"][date.weekday()]
                
                result += f"🗓️ **Ngày {i} ({day_name}, {date_str})**\n"
                result += f"   - Nhiệt độ: {avg_temp:.1f}°C ({min_temp:.1f}°C - {max_temp:.1f}°C)\n"
                result += f"   - Xác suất có mưa: {rain_probability:.0f}%\n\n"
            
            # Location specific note
            if "da nang" in validated_location.lower():
                result += f"🏙️ Dự báo cho thành phố Đà Nẵng"
            elif "hoi an" in validated_location.lower():
                result += f"🏮 Dự báo cho phố cổ Hội An"

            return result

        elif data.get("cod") == "404":
            return f"❌ Không tìm thấy dữ liệu dự báo thời tiết cho {location}."
        else:
            return f"⚠️ Có lỗi xảy ra: {data.get('message', 'Không rõ lỗi.')}"
            
    except requests.exceptions.Timeout:
        return "❌ Hết thời gian chờ khi kết nối API dự báo thời tiết."
    except requests.exceptions.ConnectionError:
        return "❌ Không thể kết nối tới dịch vụ dự báo thời tiết."
    except Exception as e:
        return f"❌ Lỗi khi lấy dự báo thời tiết: {str(e)}"

@tool
def list_available_locations(query: str = "") -> str:
    """
    Hiển thị danh sách các địa điểm có thể xem thời tiết hiện tại.
    
    Args:
        query: Không cần thiết, chỉ để tương thích với LangChain tool calling
    
    Returns:
        Danh sách 2 địa điểm được hỗ trợ
    """
    result = "🌍 Các địa điểm có thể xem thời tiết:\n\n"
    
    result += "🏙️ ĐÀ NẴNG\n"
    result += "   • Cách gọi: 'đà nẵng', 'da nang', 'danang'\n"
    result += "   • Mô tả: Thành phố trực thuộc trung ương\n\n"
    
    result += "🏮 HỘI AN\n"
    result += "   • Cách gọi: 'hội an', 'hoi an', 'hoian'\n"
    result += "   • Mô tả: Phố cổ di sản thế giới UNESCO\n\n"
    
    result += "🔧 Chức năng có sẵn:\n"
    result += "   • ☀️ Thời tiết hiện tại (nhiệt độ, tình trạng, mây, mưa, gió)\n"
    result += "   • 📅 Dự báo 2-3 ngày tới (nhiệt độ, xác suất mưa)\n\n"
    
    result += "📊 Tổng cộng: 2 địa điểm chính"
    
    return result


# Weather tool schema - chỉ thời tiết hiện tại cho Đà Nẵng và Hội An
WEATHER_TOOL_SCHEMA = {
    "name": "get_weather",
    "description": "Lấy thông tin thời tiết hiện tại cho Đà Nẵng hoặc Hội An",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Tên địa điểm: 'đà nẵng' hoặc 'hội an' (hỗ trợ cả tiếng Anh: 'da nang', 'hoi an')"
            },
            "units": {
                "type": "string",
                "description": "Đơn vị đo nhiệt độ (metric cho Celsius, imperial cho Fahrenheit)",
                "enum": ["metric", "imperial", "kelvin"],
                "default": "metric"
            },
            "lang": {
                "type": "string", 
                "description": "Ngôn ngữ hiển thị (vi cho tiếng Việt, en cho tiếng Anh)",
                "default": "vi"
            }
        },
        "required": ["location"]
    }
}
