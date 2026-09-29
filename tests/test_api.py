import sqlite3
from fastapi.testclient import TestClient

from app.main import Service, create_app
from app.workflow import RAGWorkflow
from tests.test_workflow import FakeLLM, FakeStore, CHUNK


class FakeService:
    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("CREATE TABLE answers (id TEXT PRIMARY KEY, question TEXT)")
        self.db.execute("CREATE TABLE feedback (answer_id TEXT PRIMARY KEY, rating TEXT, comment TEXT)")
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

    def save_feedback(self, data):
        if not self.db.execute("SELECT 1 FROM answers WHERE id=?", (data.answer_id,)).fetchone():
            raise KeyError(data.answer_id)
        self.db.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?)",
                        (data.answer_id, data.rating, data.comment))
        self.db.commit()


def test_query_ingest_documents_feedback():
    with TestClient(create_app(lambda: FakeService())) as client:
        assert client.get("/health").json() == {"status": "ok"}
        bad = client.post("/query", json={"question": "x"})
        assert bad.status_code == 422
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
        assert client.post("/ingest", files={"file": ("data.pdf", b"%PDF", "application/pdf")}).status_code == 422


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
