from types import SimpleNamespace

from google.genai.errors import ClientError

from app import llm as module


def test_gemini_retries_short_per_minute_quota(monkeypatch):
    calls = []
    waits = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ClientError(429, {"error": {"message": "Please retry in 0.1s"}})
            return SimpleNamespace(text='{"ok": true}')

    client = object.__new__(module.LLM)
    client.provider = "gemini"
    client.model = "test-model"
    client.client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(module.time, "sleep", waits.append)
    assert client.json("Return JSON", "ping") == {"ok": True}
    assert len(calls) == 2
    assert waits == [1.1]
