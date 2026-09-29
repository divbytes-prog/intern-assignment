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


def test_gemini_retries_temporary_provider_failure(monkeypatch):
    calls, waits = [], []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            if len(calls) < 3:
                raise ClientError(503, {"error": {"message": "temporarily unavailable"}})
            return SimpleNamespace(text='{"ok": true}')

    client = object.__new__(module.LLM)
    client.provider = "gemini"
    client.model = "test-model"
    client.client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(module.time, "sleep", waits.append)
    assert client.json("Return JSON", "ping") == {"ok": True}
    assert len(calls) == 3
    assert waits == [2, 4]


def test_groq_uses_compatible_json_api(monkeypatch):
    settings = {}

    class FakeCompletions:
        def create(self, **kwargs):
            settings["request"] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])

    def client(**kwargs):
        settings["client"] = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    monkeypatch.setattr(module, "OpenAI", client)
    llm = module.LLM("test-key", "openai/gpt-oss-20b", "groq")
    assert llm.json("Return JSON", "ping") == {"ok": True}
    assert settings["client"]["base_url"] == "https://api.groq.com/openai/v1"
    assert settings["request"]["response_format"] == {"type": "json_object"}
