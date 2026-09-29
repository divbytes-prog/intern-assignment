"""Persistent Chroma vector store with document-level replacement."""
from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


def split_markdown(text: str, max_chars: int = 1200, overlap: int = 160) -> list[str]:
    """Keep headings and paragraphs together where possible; overlap long sections."""
    blocks = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    current = ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(current) + len(block) + 2 <= max_chars:
            current = f"{current}\n\n{block}".strip()
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(block) > max_chars:
            cut = block.rfind(" ", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            chunks.append(block[:cut].strip())
            block = block[max(0, cut - overlap):].strip()
        current = block
    if current:
        chunks.append(current)
    return chunks


class DocumentStore:
    def __init__(self, data_dir: Path, api_key: str, embedding_model: str):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        embedding = OpenAIEmbeddingFunction(api_key=api_key, model_name=embedding_model)
        self.embedding = embedding
        self.collection = client.get_or_create_collection(
            name="technical_documentation", embedding_function=embedding,
        )

    def add_document(self, content: str, source: str, title: str) -> dict:
        chunks = split_markdown(content)
        if not chunks:
            raise ValueError("Document contains no indexable text")
        document_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
        ids = [f"{document_id}:{i}" for i in range(len(chunks))]
        metadatas = [
            {"document_id": document_id, "source": source, "title": title, "chunk": i}
            for i in range(len(chunks))
        ]
        # Embed first. An embedding failure does not erase the previous version.
        vectors = self.embedding(chunks)
        with self.lock:
            self.collection.delete(where={"document_id": document_id})
            self.collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=vectors)
        return {"document_id": document_id, "source": source, "title": title, "chunks": len(chunks)}

    def search(self, query: str, limit: int) -> list[dict]:
        with self.lock:
            if self.collection.count() == 0:
                return []
            result = self.collection.query(query_texts=[query], n_results=min(limit, self.collection.count()),
                                           include=["documents", "metadatas", "distances"])
        return [
            {"id": rid, "text": doc, "source": meta["source"], "title": meta["title"],
             "distance": dist}
            for rid, doc, meta, dist in zip(result["ids"][0], result["documents"][0],
                                            result["metadatas"][0], result["distances"][0])
        ]

    def list_documents(self) -> list[dict]:
        with self.lock:
            data = self.collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for metadata in data["metadatas"]:
            item = docs.setdefault(metadata["document_id"], {
                "document_id": metadata["document_id"], "source": metadata["source"],
                "title": metadata["title"], "chunks": 0,
            })
            item["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["title"])
