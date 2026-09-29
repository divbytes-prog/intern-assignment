"""Index the four self-authored FastAPI reference notes."""
import json
import os
from pathlib import Path

from app.store import DocumentStore


def main() -> None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY first")
    root = Path(__file__).resolve().parents[1]
    store = DocumentStore(Path(os.getenv("RAG_DATA_DIR", root / "data")), key,
                          os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    manifest = json.loads((root / "corpus" / "index.json").read_text(encoding="utf-8"))
    for doc in manifest:
        text = (root / "corpus" / doc["file"]).read_text(encoding="utf-8")
        print(store.add_document(text, doc["url"], doc["title"]))


if __name__ == "__main__":
    main()
