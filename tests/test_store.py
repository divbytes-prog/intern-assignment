from pathlib import Path

from app import store as module


class FakeEmbedding:
    def __init__(self, **kwargs):
        pass

    def __call__(self, input):
        return [[float(len(text) % 13), float(text.count("path")), 1.0] for text in input]

    def embed_query(self, input):
        return self(input)

    def name(self):
        return "test-embedding"


def test_persistent_index_and_replacement(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(module, "OpenAIEmbeddingFunction", FakeEmbedding)
    store = module.DocumentStore(tmp_path, "test-key", "test-model")
    first = store.add_document("# Path\n\nA path parameter.", "doc:one", "First")
    assert first["chunks"] == 1
    assert store.search("path", 3)[0]["source"] == "doc:one"
    store.add_document("# Path\n\nA new path note.", "doc:one", "Updated")
    assert store.list_documents() == [
        {"document_id": first["document_id"], "source": "doc:one", "title": "Updated", "chunks": 1}
    ]
    reopened = module.DocumentStore(tmp_path, "test-key", "test-model")
    assert reopened.list_documents()[0]["title"] == "Updated"
