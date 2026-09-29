"""Index the four self-authored FastAPI reference notes."""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from app.store import DocumentStore


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    if provider not in {"local", "gemini", "openai"}:
        raise SystemExit("EMBEDDING_PROVIDER must be local, gemini, or openai")
    key_name = {"local": None, "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}[provider]
    key = os.getenv(key_name) if key_name else None
    if key_name and not key:
        raise SystemExit(f"Set {key_name} first")
    model = {"local": os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
             "gemini": os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
             "openai": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")}[provider]
    store = DocumentStore(Path(os.getenv("RAG_DATA_DIR", root / "data-local")), key,
                          model, provider)
    manifest = json.loads((root / "corpus" / "index.json").read_text(encoding="utf-8"))
    for doc in manifest:
        text = (root / "corpus" / doc["file"]).read_text(encoding="utf-8")
        print(store.add_document(text, doc["url"], doc["title"]))


if __name__ == "__main__":
    main()
