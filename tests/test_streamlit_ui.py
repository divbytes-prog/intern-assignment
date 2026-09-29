from pathlib import Path
from urllib.parse import urlparse
from unittest.mock import patch

import pytest
import requests

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.payload


class FakeAPI:
    def __init__(self):
        self.documents = [{
            "document_id": "doc1",
            "source": "https://fastapi.tiangolo.com/tutorial/path-params/",
            "title": "Path parameters",
            "chunks": 2,
        }]
        self.calls = []

    def __call__(self, method, url, timeout=45, **kwargs):
        path = urlparse(url).path
        self.calls.append((method, path, kwargs))
        if method == "GET" and path == "/documents":
            return FakeResponse(200, {"documents": list(self.documents)})
        if method == "POST" and path == "/query":
            question = kwargs["json"]["question"]
            if question == "bad":
                return FakeResponse(422, {"detail": "Question must contain at least three non-space characters"})
            return FakeResponse(200, {
                "answer_id": "answer-1",
                "session_id": "11111111-1111-1111-1111-111111111111",
                "answer": "FastAPI validates typed path parameters [S1].",
                "sources": [{
                    "marker": "S1",
                    "title": "Path parameters",
                    "source": "https://fastapi.tiangolo.com/tutorial/path-params/",
                    "chunk_id": "doc1:0",
                }],
                "status": "answered",
                "retrieval_attempts": 1,
                "retrieval_mode": "local",
                "failure_reason": None,
            })
        if method == "POST" and path == "/feedback":
            return FakeResponse(200, {"saved": True, "answer_id": kwargs["json"]["answer_id"]})
        if method == "POST" and path == "/ingest":
            self.documents.append({
                "document_id": "upload1",
                "source": "upload:extra.md",
                "title": "extra.md",
                "chunks": 1,
            })
            return FakeResponse(201, self.documents[-1])
        raise AssertionError(f"Unexpected request: {method} {path}")


def button(at, label: str):
    return next(item for item in at.button if item.label == label)


def test_streamlit_chat_followup_feedback_upload_and_reset():
    api = FakeAPI()
    with patch("requests.request", side_effect=api):
        at = AppTest.from_file(APP, default_timeout=10).run()
        assert not at.exception
        assert at.title[0].value == "Technical Documentation Assistant"
        assert button(at, "New conversation")

        at.chat_input[0].set_value("How are typed path parameters validated?").run()
        assert not at.exception
        assert at.session_state["session_id"] == "11111111-1111-1111-1111-111111111111"
        assert at.session_state["messages"][-1]["role"] == "assistant"
        assert any("Path parameters" in item.value for item in at.markdown)

        button(at, "Helpful").click().run()
        assert any(path == "/feedback" for _, path, _ in api.calls)

        at.chat_input[0].set_value("What about invalid values?").run()
        query_calls = [kwargs for method, path, kwargs in api.calls if method == "POST" and path == "/query"]
        assert query_calls[-1]["json"]["session_id"] == "11111111-1111-1111-1111-111111111111"

        at.file_uploader[0].set_value(("extra.md", b"# Extra\nMore docs", "text/markdown")).run()
        button(at, "Index file").click().run()
        assert any(path == "/ingest" for _, path, _ in api.calls)
        assert any("Document indexed" in item.value for item in at.success)

        button(at, "New conversation").click().run()
        assert at.session_state["session_id"] is None
        assert at.session_state["messages"] == []


def test_streamlit_surfaces_backend_validation_message():
    api = FakeAPI()
    with patch("requests.request", side_effect=api):
        at = AppTest.from_file(APP, default_timeout=10).run()
        at.chat_input[0].set_value("bad").run()
        assert not at.exception
        assert any("Question must contain at least three" in item.value for item in at.error)
