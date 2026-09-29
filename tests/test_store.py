from pathlib import Path

from app import store as module


class FakeEmbedding:
    def __init__(self, *args, **kwargs):
        pass

    def embed(self, input):
        return [[float(len(text) % 13), float(text.count("path")), 1.0] for text in input]


def test_persistent_index_and_replacement(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(module, "Embeddings", FakeEmbedding)
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


def test_bundled_corpus_seeds_four_documents_idempotently(tmp_path: Path, monkeypatch):
    from scripts import seed

    monkeypatch.setattr(module, "Embeddings", FakeEmbedding)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    seed.main()
    first = module.DocumentStore(tmp_path, "test-key", "test-model").list_documents()
    assert len(first) == 4
    assert sum(doc["chunks"] for doc in first) > 4
    seed.main()
    assert module.DocumentStore(tmp_path, "test-key", "test-model").list_documents() == first


def test_embedding_two_returns_one_vector_per_chunk(monkeypatch):
    class Client:
        @property
        def models(self):
            return self

        def embed_content(self, *, model, contents):
            assert model == "gemini-embedding-2"
            assert len(contents) == 2
            assert all(isinstance(item, module.types.Content) for item in contents)
            class Result:
                embeddings = [type("E", (), {"values": [1.0, 0.0]})(),
                              type("E", (), {"values": [0.0, 1.0]})()]
            return Result()

    monkeypatch.setattr(module.genai, "Client", lambda **kwargs: Client())
    assert module.Embeddings("test", "gemini-embedding-2", "gemini").embed(["a", "b"]) == [
        [1.0, 0.0], [0.0, 1.0],
    ]


def test_local_embeddings_need_no_provider_key(monkeypatch):
    import fastembed
    import numpy as np

    class LocalModel:
        def __init__(self, *, model_name):
            assert model_name == "BAAI/bge-small-en-v1.5"

        def embed(self, texts):
            for i, _ in enumerate(texts):
                yield np.array([float(i), 1.0])

    monkeypatch.setattr(fastembed, "TextEmbedding", LocalModel)
    assert module.Embeddings(None, "BAAI/bge-small-en-v1.5", "local").embed(["one", "two"]) == [
        [0.0, 1.0], [1.0, 1.0],
    ]
