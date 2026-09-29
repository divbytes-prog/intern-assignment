import sqlite3
import uuid
from fastapi.testclient import TestClient

from app.main import Service, create_app, query_error_detail
from app.workflow import RAGWorkflow
from tests.test_workflow import FakeLLM, FakeStore, CHUNK


class FakeService:
    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("CREATE TABLE answers (id TEXT PRIMARY KEY, question TEXT)")
        self.db.execute("CREATE TABLE feedback (answer_id TEXT PRIMARY KEY, rating TEXT, comment TEXT)")
        self.db.execute("CREATE TABLE turns (id INTEGER PRIMARY KEY, session_id TEXT, question TEXT, answer TEXT, status TEXT)")
        self.workflow = RAGWorkflow(FakeStore([CHUNK]), FakeLLM())
        self.store = self
        self.documents = []

    def add_document(self, content, source, title):
        self.documents.append({"document_id": "fake", "source": source, "title": title, "chunks": 1})
        return self.documents[-1]

    def list_documents(self):
        return self.documents

    def save_answer(self, answer_id, question):
        self.db.execute("INSERT INTO answers VALUES (?, ?)", (answer_id, question))
        self.db.commit()

    def recent_history(self, session_id):
        rows = self.db.execute("SELECT question, answer FROM turns WHERE session_id=? ORDER BY id DESC LIMIT 4", (str(session_id),)).fetchall()
        return [{"question": q, "answer": a} for q, a in reversed(rows)]

    def save_turn(self, session_id, question, answer, status):
        self.db.execute("INSERT INTO turns(session_id, question, answer, status) VALUES (?, ?, ?, ?)",
                        (str(session_id), question, answer, status))
        self.db.commit()

    def save_feedback(self, data):
        if not self.db.execute("SELECT 1 FROM answers WHERE id=?", (data.answer_id,)).fetchone():
            raise KeyError(data.answer_id)
        self.db.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?)",
                        (data.answer_id, data.rating, data.comment))
        self.db.commit()


def test_query_ingest_documents_feedback():
    with TestClient(create_app(lambda: FakeService())) as client:
        landing = client.get("/")
        assert landing.status_code == 200
        assert "Ask the docs" in landing.text
        assert client.get("/health").json() == {"status": "ok"}
        bad = client.post("/query", json={"question": "x"})
        assert bad.status_code == 422
        assert client.post("/query", json={"question": "   "}).status_code == 422
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/query"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        ingested = client.post("/ingest", files={"file": ("note.md", b"# A note\nSome text", "text/markdown")})
        assert ingested.status_code == 201
        assert client.get("/documents").json()["documents"][0]["title"] == "note.md"
        response = client.post("/query", json={"question": "How do I validate item_id?"})
        assert response.status_code == 200
        answer = response.json()
        assert answer["sources"][0]["marker"] == "S1"
        assert client.post("/feedback", json={"answer_id": answer["answer_id"], "rating": "up"}).status_code == 200
        assert client.post("/feedback", json={"answer_id": "missing", "rating": "down"}).status_code == 404
        assert client.post("/feedback", json={"answer_id": answer["answer_id"], "rating": "maybe"}).status_code == 422


def test_rejects_unsafe_urls_and_file_types():
    with TestClient(create_app(lambda: FakeService())) as client:
        assert client.post("/ingest", json={"url": "http://127.0.0.1/admin"}).status_code == 422
        assert client.post("/ingest", json={"url": "https://fastapi.tiangolo.com:444/tutorial/"}).status_code == 422
        assert client.post("/ingest", json={"url": "https://fastapi.tiangolo.com:notaport/tutorial/"}).status_code == 422
        assert client.post("/ingest", json=["not an object"]).status_code == 422
        assert client.post("/ingest", json={"url": 42}).status_code == 422
        assert client.post("/ingest", files={"file": ("data.pdf", b"%PDF", "application/pdf")}).status_code == 422
        invalid_text = client.post("/ingest", files={"file": ("bad.md", b"\xff\xfe", "text/markdown")})
        assert invalid_text.status_code == 422
        assert invalid_text.json()["detail"] == "Uploaded file must contain valid UTF-8 text"


def test_rejects_malformed_empty_and_oversized_ingest_payloads():
    with TestClient(create_app(lambda: FakeService())) as client:
        malformed = client.post(
            "/ingest", content=b"{", headers={"content-type": "application/json"}
        )
        assert malformed.status_code == 422
        assert malformed.json()["detail"] == "Request body must be valid JSON"

        empty = client.post(
            "/ingest", files={"file": ("empty.md", b"", "text/markdown")}
        )
        assert empty.status_code == 422
        assert empty.json()["detail"] == "Document is empty"

        oversized = client.post(
            "/ingest", files={"file": ("large.md", b"x" * 1_000_001, "text/markdown")}
        )
        assert oversized.status_code == 422
        assert oversized.json()["detail"] == "Document exceeds the 1 MB limit"


def test_real_local_index_with_stubbed_provider(tmp_path, monkeypatch):
    from app import main, store
    from tests.test_store import FakeEmbedding
    monkeypatch.setattr(store, "Embeddings", FakeEmbedding)
    monkeypatch.setattr(main, "LLM", lambda *args: FakeLLM())
    with TestClient(create_app(lambda: Service(tmp_path, "test-key"))) as client:
        ingested = client.post("/ingest", files={"file": (
            "path.md", b"# Path parameters\n\nDeclare item_id: int on the route function.",
            "text/markdown",
        )})
        assert ingested.status_code == 201, ingested.text
        response = client.post("/query", json={"question": "How do I validate item_id?"})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "answered"
        assert response.json()["sources"][0]["source"] == "upload:path.md"
        assert client.get("/documents").json()["documents"][0]["chunks"] == 1


def test_provider_failure_diagnostics_do_not_include_secrets():
    class ProviderError(Exception):
        code = 429

    detail = query_error_detail(ProviderError("secret key and request details"))
    assert detail == {"error": "quota_or_rate_limit", "provider_status": 429}
    assert "secret" not in str(detail)

    class FailingService(FakeService):
        def __init__(self):
            super().__init__()
            self.workflow = self

        def invoke(self, question, history=None):
            raise ProviderError("secret key and request details")

    with TestClient(create_app(lambda: FailingService())) as client:
        response = client.post("/query", json={"question": "A valid question?"})
        assert response.status_code == 429
        assert response.json()["detail"] == detail


def test_sessions_are_isolated_and_follow_up_is_forwarded():
    class RecordingWorkflow:
        def __init__(self):
            self.calls = []

        def invoke(self, question, history=None):
            self.calls.append((question, history))
            return {"answer": "A supported response", "sources": [], "status": "insufficient_context",
                    "attempts": 1, "retrieval_mode": "none"}

    service = FakeService()
    service.workflow = RecordingWorkflow()
    with TestClient(create_app(lambda: service)) as client:
        first = client.post("/query", json={"question": "First question?"}).json()
        session = first["session_id"]
        assert str(uuid.UUID(session)) == session
        second = client.post("/query", json={"question": "What about this?", "session_id": session}).json()
        assert second["session_id"] == session
        assert service.workflow.calls[1][1] == [{"question": "First question?", "answer": "A supported response"}]
        client.post("/query", json={"question": "Separate session?"})
        assert service.workflow.calls[2][1] == []
        assert client.post("/query", json={"question": "A question?", "session_id": "invalid"}).status_code == 422


def test_service_keeps_only_four_turns(tmp_path, monkeypatch):
    from app import main, store
    from tests.test_store import FakeEmbedding
    monkeypatch.setattr(store, "Embeddings", FakeEmbedding)
    monkeypatch.setattr(main, "LLM", lambda *args: FakeLLM())
    service = Service(tmp_path, "test-key")
    session_a, session_b = uuid.uuid4(), uuid.uuid4()
    for i in range(7):
        service.save_turn(session_a, f"q{i}", f"a{i}", "answered")
    service.save_turn(session_b, "other", "answer", "answered")
    assert [row["question"] for row in service.recent_history(session_a)] == ["q3", "q4", "q5", "q6"]
    assert [row["question"] for row in service.recent_history(session_b)] == ["other"]
    service.db.close()
