import os
import requests
from typing import Optional, Tuple, Dict
from langchain_core.tools import tool
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY không được tìm thấy trong file .env")
    return api_key

BASE_URL = "https://api.openweathermap.org/data/2.5"

DA_NANG_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "đà nẵng": (16.0544, 108.2022), "da nang": (16.0544, 108.2022), "danang": (16.0544, 108.2022),
    "hải châu": (16.0544, 108.2214), "hai chau": (16.0544, 108.2214),
    "thanh khê": (16.0601, 108.1856), "thanh khe": (16.0601, 108.1856),
    "sơn trà": (16.0967, 108.2709), "son tra": (16.0967, 108.2709),
    "ngũ hành sơn": (16.0125, 108.2514), "ngu hanh son": (16.0125, 108.2514),
    "liên chiểu": (16.0940, 108.1360), "lien chieu": (16.0940, 108.1360),
    "cẩm lệ": (16.0180, 108.2050), "cam le": (16.0180, 108.2050),
    "hòa vang": (16.0360, 108.0850), "hoa vang": (16.0360, 108.0850),
    "hoàng sa": (16.5400, 111.6000), "hoang sa": (16.5400, 111.6000),

    "bà nà": (15.9955, 107.9950), "ba na": (15.9955, 107.9950), "bà nà hills": (15.9955, 107.9950),
    "sân bay đà nẵng": (16.0439, 108.1994), "san bay": (16.0439, 108.1994), "airport": (16.0439, 108.1994),
    "cầu rồng": (16.0611, 108.2255), "cau rong": (16.0611, 108.2255), "dragon bridge": (16.0611, 108.2255),
    "cầu sông hàn": (16.0718, 108.2250), "cau song han": (16.0718, 108.2250),
    "chùa linh ứng": (16.0994, 108.2778), "chua linh ung": (16.0994, 108.2778), "linh ung": (16.0994, 108.2778),
    "núi thần tài": (15.9680, 108.0200), "nui than tai": (15.9680, 108.0200),
    "đèo hải vân": (16.1905, 108.1328), "deo hai van": (16.1905, 108.1328), "hai van pass": (16.1905, 108.1328),
    "biển mỹ khê": (16.0594, 108.2460), "bien my khe": (16.0594, 108.2460), "my khe": (16.0594, 108.2460),
    "non nước": (16.0020, 108.2630), "non nuoc": (16.0020, 108.2630), "ngũ hành sơn danh thắng": (16.0020, 108.2630),
    "bán đảo sơn trà": (16.1210, 108.2650), "ban dao son tra": (16.1210, 108.2650),
    "suối mơ": (16.0750, 108.0400), "suoi mo": (16.0750, 108.0400),

    "hải châu 1": (16.0670, 108.2220), "hai chau 1": (16.0670, 108.2220), "hải châu i": (16.0670, 108.2220),
    "hải châu 2": (16.0630, 108.2180), "hai chau 2": (16.0630, 108.2180), "hải châu ii": (16.0630, 108.2180),
    "thạch thang": (16.0750, 108.2180), "thach thang": (16.0750, 108.2180),
    "thanh bình": (16.0720, 108.2100), "thanh binh": (16.0720, 108.2100),
    "thuận phước": (16.0820, 108.2180), "thuan phuoc": (16.0820, 108.2180),
    "hòa thuận đông": (16.0480, 108.2100), "hoa thuan dong": (16.0480, 108.2100),
    "hòa thuận tây": (16.0450, 108.2000), "hoa thuan tay": (16.0450, 108.2000),
    "nam dương": (16.0580, 108.2180), "nam duong": (16.0580, 108.2180),
    "phước ninh": (16.0600, 108.2200), "phuoc ninh": (16.0600, 108.2200),
    "bình hiên": (16.0570, 108.2200), "binh hien": (16.0570, 108.2200),
    "bình thuận": (16.0530, 108.2170), "binh thuan": (16.0530, 108.2170),
    "hòa cường bắc": (16.0400, 108.2180), "hoa cuong bac": (16.0400, 108.2180),
    "hòa cường nam": (16.0300, 108.2200), "hoa cuong nam": (16.0300, 108.2200),

    "tam thuận": (16.0680, 108.1900), "tam thuan": (16.0680, 108.1900),
    "thanh khê tây": (16.0650, 108.1750), "thanh khe tay": (16.0650, 108.1750),
    "thanh khê đông": (16.0680, 108.1850), "thanh khe dong": (16.0680, 108.1850),
    "xuân hà": (16.0720, 108.1900), "xuan ha": (16.0720, 108.1900),
    "tân chính": (16.0680, 108.2000), "tan chinh": (16.0680, 108.2000),
    "chính gián": (16.0650, 108.1950), "chinh gian": (16.0650, 108.1950),
    "vĩnh trung": (16.0620, 108.2050), "vinh trung": (16.0620, 108.2050),
    "thạc gián": (16.0630, 108.2000), "thac gian": (16.0630, 108.2000),
    "an khê": (16.0550, 108.1800), "an khe": (16.0550, 108.1800),
    "hòa khê": (16.0580, 108.1850), "hoa khe": (16.0580, 108.1850),

    "thọ quang": (16.1050, 108.2450), "tho quang": (16.1050, 108.2450),
    "nại hiên đông": (16.0850, 108.2350), "nai hien dong": (16.0850, 108.2350),
    "mân thái": (16.0900, 108.2450), "man thai": (16.0900, 108.2450),
    "an hải bắc": (16.0750, 108.2350), "an hai bac": (16.0750, 108.2350),
    "an hải đông": (16.0650, 108.2400), "an hai dong": (16.0650, 108.2400),
    "an hải tây": (16.0680, 108.2300), "an hai tay": (16.0680, 108.2300),
    "phước mỹ": (16.0650, 108.2480), "phuoc my": (16.0650, 108.2480),

    "mỹ an": (16.0450, 108.2400), "my an": (16.0450, 108.2400),
    "khuê mỹ": (16.0350, 108.2450), "khue my": (16.0350, 108.2450),
    "hòa hải": (15.9950, 108.2600), "hoa hai": (15.9950, 108.2600),
    "hòa quý": (15.9900, 108.2300), "hoa quy": (15.9900, 108.2300),

    "hòa minh": (16.0650, 108.1650), "hoa minh": (16.0650, 108.1650),
    "hòa khánh nam": (16.0750, 108.1500), "hoa khanh nam": (16.0750, 108.1500),
    "hòa khánh bắc": (16.0900, 108.1400), "hoa khanh bac": (16.0900, 108.1400),
    "hòa hiệp nam": (16.1050, 108.1300), "hoa hiep nam": (16.1050, 108.1300),
    "hòa hiệp bắc": (16.1250, 108.1200), "hoa hiep bac": (16.1250, 108.1200),

    "khuê trung": (16.0250, 108.2100), "khue trung": (16.0250, 108.2100),
    "hòa phát": (16.0400, 108.1900), "hoa phat": (16.0400, 108.1900),
    "hòa an": (16.0500, 108.1800), "hoa an": (16.0500, 108.1800),
    "hòa thọ tây": (16.0200, 108.1800), "hoa tho tay": (16.0200, 108.1800),
    "hòa thọ đông": (16.0250, 108.2000), "hoa tho dong": (16.0250, 108.2000),
    "hòa xuân": (16.0050, 108.2150), "hoa xuan": (16.0050, 108.2150),

    "hòa châu": (15.9900, 108.2000), "hoa chau": (15.9900, 108.2000),
    "hòa tiến": (15.9700, 108.1800), "hoa tien": (15.9700, 108.1800),
    "hòa phước": (15.9500, 108.2000), "hoa phuoc": (15.9500, 108.2000),
    "hòa nhơn": (16.0100, 108.1300), "hoa nhon": (16.0100, 108.1300),
    "hòa phong": (16.0000, 108.1500), "hoa phong": (16.0000, 108.1500),
    "hòa khương": (15.9600, 108.1200), "hoa khuong": (15.9600, 108.1200),
    "hòa sơn": (16.0600, 108.1200), "hoa son": (16.0600, 108.1200),
    "hòa liên": (16.0800, 108.1100), "hoa lien": (16.0800, 108.1100),
    "hòa ninh": (16.0400, 108.0600), "hoa ninh": (16.0400, 108.0600),
    "hòa phú": (15.9800, 108.0300), "hoa phu": (15.9800, 108.0300),
    "hòa bắc": (16.1300, 108.0300), "hoa bac": (16.1300, 108.0300),
}

DA_NANG_BBOX = (15.80, 16.25, 107.80, 108.45)

def _is_within_da_nang(lat: float, lon: float) -> bool:
    lat_min, lat_max, lon_min, lon_max = DA_NANG_BBOX
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

def _resolve_location(location: Optional[str], lat: Optional[float], lon: Optional[float]) -> Tuple[Optional[Tuple[float, float]], str]:
    if lat is not None and lon is not None:
        if not _is_within_da_nang(lat, lon):
            return None, "❌ Xin lỗi, tọa độ này nằm ngoài phạm vi Đà Nẵng."
        return (lat, lon), f"tọa độ {lat:.4f}, {lon:.4f}"

    if not location:
        return None, "❌ Vui lòng cung cấp địa điểm hoặc tên khu vực."

    key = location.strip().lower()

    if key in DA_NANG_LOCATIONS:
        coords = DA_NANG_LOCATIONS[key]
        return coords, location.title()

    matched_name = None
    matched_coords = None
    sorted_keys = sorted(DA_NANG_LOCATIONS.keys(), key=len, reverse=True)

    generic_names = {"đà nẵng", "da nang", "danang"}
    for name in sorted_keys:
        if name in generic_names:
            continue
        if name in key:
            matched_name = name
            matched_coords = DA_NANG_LOCATIONS[name]
            break
            
    if matched_coords:
        return matched_coords, matched_name.title()

    if "đà nẵng" in key or "da nang" in key or "danang" in key:
        return DA_NANG_LOCATIONS["đà nẵng"], "Trung tâm Đà Nẵng"

    return None, f"❌ Xin lỗi, không tìm thấy địa điểm '{location}' tại Đà Nẵng."

def get_weather_data(endpoint: str, location: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None, units: str = "metric", lang: str = "vi") -> dict:
    api_key = load_env()
    if lat is not None and lon is not None:
        url = f"{BASE_URL}/{endpoint}?appid={api_key}&lat={lat}&lon={lon}&units={units}&lang={lang}"
    else:
        url = f"{BASE_URL}/{endpoint}?appid={api_key}&q={location}&units={units}&lang={lang}"
    response = requests.get(url, timeout=10)
    return response.json()

@tool
def get_weather(location: Optional[str] = "Đà Nẵng", lat: Optional[float] = None, lon: Optional[float] = None, units: str = "metric", lang: str = "vi") -> str:
    """Lấy thời tiết hiện tại cho Đà Nẵng. Hỗ trợ truyền tọa độ (lat/lon) hoặc tên khu vực Đà Nẵng."""
    try:
        coords, display_name_or_error = _resolve_location(location, lat, lon)
        if coords is None:
            return display_name_or_error
        lat_val, lon_val = coords
        data = get_weather_data("weather", location=location, lat=lat_val, lon=lon_val, units=units, lang=lang)
        if data.get("cod") != 200:
            return f"❌ Không tìm thấy thông tin thời tiết cho {display_name_or_error}."
        main, weather, wind, clouds = data["main"], data["weather"][0], data.get("wind", {}), data.get("clouds", {})
        temp, humidity, desc, weather_type = main["temp"], main["humidity"], weather["description"], weather["main"]
        wind_speed, cloudiness = wind.get("speed", "N/A"), clouds.get("all", 0)
        emoji = {"Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Drizzle": "🌦️", "Thunderstorm": "⛈️", "Snow": "❄️", "Mist": "🌫️", "Fog": "🌫️"}.get(weather_type, "🌤️")
        rain_prob = "90-100%" if weather_type in ["Rain", "Drizzle"] else "95-100%" if weather_type == "Thunderstorm" else "20-40%" if weather_type == "Clouds" and cloudiness > 70 else "0%"
        result = f"{emoji} Thời tiết hiện tại tại {display_name_or_error}:\n- Nhiệt độ: {temp}°C\n- Tình trạng: {desc.capitalize()}\n- Độ ẩm: {humidity}%\n- Xác suất có mưa: {rain_prob}\n"
        if wind_speed != "N/A": result += f"- Tốc độ gió: {wind_speed} m/s\n"
        if temp > 30: result += "\n💡 Thời tiết khá nóng, nên mang theo nước!"
        elif weather_type == "Rain": result += "\n💡 Có mưa, nhớ mang theo ô!"
        return result
    except Exception as e:
        return f"❌ Lỗi khi lấy thông tin thời tiết: {str(e)}"

@tool
def get_weather_forecast(location: Optional[str] = "Đà Nẵng", lat: Optional[float] = None, lon: Optional[float] = None, days: int = 3, units: str = "metric", lang: str = "vi") -> str:
    """Lấy dự báo 2-3 ngày tới cho Đà Nẵng. Hỗ trợ truyền tọa độ (lat/lon) hoặc tên khu vực Đà Nẵng."""
    if days not in [2, 3]:
        return "❌ Chỉ hỗ trợ dự báo cho 2 hoặc 3 ngày tới."
    try:
        coords, display_name_or_error = _resolve_location(location, lat, lon)
        if coords is None:
            return display_name_or_error
        lat_val, lon_val = coords
        data = get_weather_data("forecast", location=location, lat=lat_val, lon=lon_val, units=units, lang=lang)
        if data.get("cod") != "200":
            return f"❌ Không tìm thấy dữ liệu dự báo cho {display_name_or_error}."
        from datetime import datetime
        today, daily_forecasts = datetime.now().date(), {}
        for forecast in data["list"]:
            date = datetime.fromtimestamp(forecast["dt"]).date()
            if date > today:
                daily_forecasts.setdefault(date, []).append(forecast)
        sorted_dates = sorted(daily_forecasts.keys())[:days]
        if not sorted_dates:
            return "❌ Không có dữ liệu dự báo."
        result = f"📅 Dự báo thời tiết {days} ngày tới tại {display_name_or_error}:\n\n"
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