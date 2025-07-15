import requests
from typing import Optional
from langchain_core.tools import tool
from concurrent.futures import ThreadPoolExecutor

API_KEY = "ea4de82e443d888676a2250ef1c58aef"
BASE_URL = "http://api.openweathermap.org/data/2.5"

def get_weather_data(location: str, endpoint: str) -> dict:
    url = f"{BASE_URL}/{endpoint}?appid={API_KEY}&q={location}&units=metric&lang=vi"
    response = requests.get(url, timeout=10)
    return response.json()

@tool
def get_weather(location: str, units: str = "metric", lang: str = "vi") -> str:
    """Lấy thông tin thời tiết hiện tại/hôm nay cho Đà Nẵng, nếu là địa điểm khác thì từ chối."""
    # Chỉ cho phép Đà Nẵng
    if location.strip().lower() not in ["đà nẵng", "da nang", "danang"]:
        return "❌ Xin lỗi, tôi chỉ hỗ trợ thông tin thời tiết cho Đà Nẵng."
    try:
        data = get_weather_data(location, "weather")
        if data.get("cod") != 200:
            return f"❌ Không tìm thấy thông tin thời tiết cho {location}."
        main, weather, wind, clouds = data["main"], data["weather"][0], data.get("wind", {}), data.get("clouds", {})
        temp, humidity, desc, weather_type = main["temp"], main["humidity"], weather["description"], weather["main"]
        wind_speed, cloudiness = wind.get("speed", "N/A"), clouds.get("all", 0)
        emoji = {"Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Drizzle": "🌦️", "Thunderstorm": "⛈️", "Snow": "❄️", "Mist": "🌫️", "Fog": "🌫️"}.get(weather_type, "🌤️")
        rain_prob = "90-100%" if weather_type in ["Rain", "Drizzle"] else "95-100%" if weather_type == "Thunderstorm" else "20-40%" if weather_type == "Clouds" and cloudiness > 70 else "0%"
        result = f"{emoji} Thời tiết hiện tại tại {location.title()}:\n- Nhiệt độ: {temp}°C\n- Tình trạng: {desc.capitalize()}\n- Độ ẩm: {humidity}%\n- Xác suất có mưa: {rain_prob}\n"
        if wind_speed != "N/A": result += f"- Tốc độ gió: {wind_speed} m/s\n"
        if temp > 30: result += "\n💡 Thời tiết khá nóng, nên mang theo nước!"
        elif weather_type == "Rain": result += "\n💡 Có mưa, nhớ mang theo ô!"
        return result
    except Exception as e:
        return f"❌ Lỗi khi lấy thông tin thời tiết: {str(e)}"

@tool
def get_weather_forecast(location: str, days: int = 3) -> str:
    """Lấy dự báo thời tiết cho 2-3 ở Đà Nẵng, nếu là địa điểm khác thì từ chối."""
    if location.strip().lower() not in ["đà nẵng", "da nang", "danang"]:
        return "❌ Xin lỗi, tôi chỉ hỗ trợ dự báo thời tiết cho Đà Nẵng."
    if days not in [2, 3]:
        return "❌ Chỉ hỗ trợ dự báo cho 2 hoặc 3 ngày tới."
    try:
        data = get_weather_data(location, "forecast")
        if data.get("cod") != "200":
            return f"❌ Không tìm thấy dữ liệu dự báo cho {location}."
        from datetime import datetime
        today, daily_forecasts = datetime.now().date(), {}
        for forecast in data["list"]:
            date = datetime.fromtimestamp(forecast["dt"]).date()
            if date > today:
                daily_forecasts.setdefault(date, []).append(forecast)
        sorted_dates = sorted(daily_forecasts.keys())[:days]
        if not sorted_dates:
            return "❌ Không có dữ liệu dự báo."
        result = f"📅 Dự báo thời tiết {days} ngày tới tại {location.title()}:\n\n"
        day_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
        # Song song: tổng hợp nhiệt độ và xác suất mưa cho từng ngày
        def summarize_day(forecasts):
            temps = [f["main"]["temp"] for f in forecasts]
            weathers = [f["weather"][0]["main"] for f in forecasts]
            rain_count = sum(1 for w in weathers if w in ["Rain", "Drizzle", "Thunderstorm"])
            return temps, weathers, rain_count
        with ThreadPoolExecutor() as executor:
            day_summaries = list(executor.map(summarize_day, [daily_forecasts[date] for date in sorted_dates]))
        for i, (date, (temps, weathers, rain_count)) in enumerate(zip(sorted_dates, day_summaries), 1):
            result += f"🗓️ **Ngày {i} ({day_names[date.weekday()]}, {date.strftime('%d/%m')})**\n"
            result += f"   - Nhiệt độ: {sum(temps)/len(temps):.1f}°C ({min(temps):.1f}°C - {max(temps):.1f}°C)\n"
            result += f"   - Xác suất có mưa: {(rain_count/len(weathers)*100):.0f}%\n\n"
        return result
    except Exception as e:
        return f"❌ Lỗi khi lấy dự báo thời tiết: {str(e)}"