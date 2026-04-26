"""Provider-adapter tests. These use stub clients to avoid network calls."""

import pytest

from loop_core import LLMEngine, get_engine
from loop_core.api import _ChatCompletionEngine


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


def test_generate_strips_known_prefixes():
    engine = _ChatCompletionEngine(_FakeClient("Assistant: hello there"), "m", 0.5)
    assert engine.generate("hi") == "hello there"


def test_generate_falls_back_when_empty():
    engine = _ChatCompletionEngine(_FakeClient("   "), "m", 0.5)
    assert engine.generate("hi") == "I'm not sure how to respond."


def test_generate_passes_runtime_through():
    client = _FakeClient("ok")
    engine = _ChatCompletionEngine(client, "my-model", 0.42)
    engine.generate("prompt", max_tokens=64)
    call = client.chat.completions.calls[0]
    assert call["model"] == "my-model"
    assert call["temperature"] == 0.42
    assert call["max_tokens"] == 64
    assert call["messages"] == [{"role": "user", "content": "prompt"}]


def test_engines_satisfy_protocol():
    engine = _ChatCompletionEngine(_FakeClient("ok"), "m", 0.5)
    assert isinstance(engine, LLMEngine)


def test_get_engine_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        get_engine("nonexistent", api_key="k")


def test_get_engine_passes_runtime_model(monkeypatch):
    captured = {}

    class _StubEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("loop_core.api", fromlist=["ENGINE_REGISTRY"]).ENGINE_REGISTRY, "stub", _StubEngine)
    get_engine("stub", api_key="abc", runtime={"model": "custom", "temperature": 0.9})
    assert captured == {"api_key": "abc", "model": "custom", "temperature": 0.9}
