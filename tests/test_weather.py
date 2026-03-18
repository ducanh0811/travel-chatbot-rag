import os
import sys
import pathlib
import types

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stub langchain_core.tools.tool to avoid requiring langchain_core in tests
if "langchain_core.tools" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")
    tools_mod = types.ModuleType("langchain_core.tools")
    tools_mod.tool = lambda f=None, **kwargs: f
    langchain_core.tools = tools_mod
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.tools"] = tools_mod

from mytools import weather as weather_module


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _mock_requests_get(monkeypatch, payload):
    def _get(*args, **kwargs):
        return DummyResponse(payload)

    monkeypatch.setattr(weather_module.requests, "get", _get)


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")


def test_get_weather_with_gps_success(monkeypatch):
    payload = {
        "cod": 200,
        "main": {"temp": 30, "humidity": 70},
        "weather": [{"description": "nắng đẹp", "main": "Clear"}],
        "wind": {"speed": 3.5},
        "clouds": {"all": 0},
    }
    _mock_requests_get(monkeypatch, payload)

    result = weather_module.get_weather(lat=16.06, lon=108.25)

    assert "Thời tiết hiện tại" in result
    assert "Nhiệt độ: 30" in result
    assert "Độ ẩm: 70" in result


def test_get_weather_with_gps_outside_da_nang():
    result = weather_module.get_weather(lat=10.0, lon=106.0)
    assert "chỉ hỗ trợ thông tin thời tiết cho Đà Nẵng" in result


def test_get_weather_with_location_alias(monkeypatch):
    payload = {
        "cod": 200,
        "main": {"temp": 28, "humidity": 65},
        "weather": [{"description": "nhiều mây", "main": "Clouds"}],
        "wind": {"speed": 2.0},
        "clouds": {"all": 90},
    }
    _mock_requests_get(monkeypatch, payload)

    result = weather_module.get_weather(location="Sơn Trà")

    assert "Sơn Trà" in result
    assert "Nhiệt độ: 28" in result


def test_get_weather_forecast_days_invalid():
    result = weather_module.get_weather_forecast(days=4)
    assert "Chỉ hỗ trợ dự báo cho 2 hoặc 3 ngày tới" in result


def test_get_weather_forecast_success(monkeypatch):
    from datetime import datetime, timedelta
    base = datetime.now().date() + timedelta(days=1)
    ts_day1 = int(datetime.combine(base, datetime.min.time()).timestamp())
    ts_day1_later = ts_day1 + 3600
    ts_day2 = int(datetime.combine(base + timedelta(days=1), datetime.min.time()).timestamp())
    payload = {
        "cod": "200",
        "list": [
            {
                "dt": ts_day1,
                "main": {"temp": 27},
                "weather": [{"main": "Clear"}],
            },
            {
                "dt": ts_day1_later,
                "main": {"temp": 29},
                "weather": [{"main": "Clouds"}],
            },
            {
                "dt": ts_day2,
                "main": {"temp": 26},
                "weather": [{"main": "Rain"}],
            },
        ],
    }
    _mock_requests_get(monkeypatch, payload)

    result = weather_module.get_weather_forecast(location="Đà Nẵng", days=2)

    assert "Dự báo thời tiết 2 ngày tới" in result
    assert "Nhiệt độ:" in result
