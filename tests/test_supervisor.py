import pathlib
import sys
import types

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stub langchain/langgraph modules to avoid hard dependencies in tests.
if "langchain_openai" not in sys.modules:
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = object
    sys.modules["langchain_openai"] = langchain_openai

if "langgraph.prebuilt" not in sys.modules:
    langgraph = types.ModuleType("langgraph")
    prebuilt = types.ModuleType("langgraph.prebuilt")
    prebuilt.create_react_agent = lambda *args, **kwargs: object()
    langgraph.prebuilt = prebuilt
    sys.modules["langgraph"] = langgraph
    sys.modules["langgraph.prebuilt"] = prebuilt

if "langgraph_supervisor" not in sys.modules:
    langgraph_supervisor = types.ModuleType("langgraph_supervisor")
    langgraph_supervisor.create_supervisor = lambda *args, **kwargs: object()
    sys.modules["langgraph_supervisor"] = langgraph_supervisor

import Supervisor as supervisor_module


class DummyMsg:
    def __init__(self, content):
        self.content = content


class DummyAgent:
    def __init__(self, response):
        self.response = response
        self.last_payload = None

    def invoke(self, payload):
        self.last_payload = payload
        return self.response


def test_classify_query_routing():
    assert supervisor_module.classify_query("Thời tiết hôm nay thế nào?") == "weather"
    assert supervisor_module.classify_query("Gợi ý khách sạn gần biển") == "travel"
    assert supervisor_module.classify_query("Thời tiết cuối tuần và gợi ý quán cafe") == "supervisor"


def test_run_supervisor_query_routes_weather(monkeypatch):
    agent = DummyAgent({"messages": [DummyMsg("OK weather")]})

    monkeypatch.setattr(supervisor_module, "get_weather_agent_instance", lambda: agent)
    monkeypatch.setattr(
        supervisor_module,
        "get_travel_agent_instance",
        lambda: (_ for _ in ()).throw(AssertionError("Travel agent should not be used")),
    )
    monkeypatch.setattr(
        supervisor_module,
        "get_supervisor_instance",
        lambda: (_ for _ in ()).throw(AssertionError("Supervisor agent should not be used")),
    )

    result = supervisor_module.run_supervisor_query("Thời tiết hôm nay thế nào?")

    assert result == ["OK weather"]
    assert agent.last_payload == {
        "messages": [{"role": "user", "content": "Thời tiết hôm nay thế nào?"}]
    }


def test_run_supervisor_query_routes_travel(monkeypatch):
    agent = DummyAgent({"messages": [DummyMsg("OK travel")]})

    monkeypatch.setattr(supervisor_module, "get_travel_agent_instance", lambda: agent)
    monkeypatch.setattr(
        supervisor_module,
        "get_weather_agent_instance",
        lambda: (_ for _ in ()).throw(AssertionError("Weather agent should not be used")),
    )
    monkeypatch.setattr(
        supervisor_module,
        "get_supervisor_instance",
        lambda: (_ for _ in ()).throw(AssertionError("Supervisor agent should not be used")),
    )

    result = supervisor_module.run_supervisor_query("Gợi ý quán cafe đẹp ở Sơn Trà")

    assert result == ["OK travel"]


def test_run_supervisor_query_filters_transfer_messages(monkeypatch):
    agent = DummyAgent(
        {
            "messages": [
                DummyMsg("Transferring to weather_agent"),
                DummyMsg("Thời tiết hôm nay thế nào?"),
                DummyMsg("Trời nắng đẹp"),
            ]
        }
    )

    monkeypatch.setattr(supervisor_module, "get_supervisor_instance", lambda: agent)

    result = supervisor_module.run_supervisor_query(
        "Thời tiết hôm nay thế nào?", use_singleton=True
    )

    assert result == ["Trời nắng đẹp"]
