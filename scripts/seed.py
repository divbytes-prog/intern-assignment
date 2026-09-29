"""Index the four self-authored FastAPI reference notes."""
import json
import os
from pathlib import Path

from app.store import DocumentStore


def main() -> None:
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    if provider not in {"gemini", "openai"}:
        raise SystemExit("AI_PROVIDER must be gemini or openai")
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    key = os.getenv(key_name)
    if not key:
        raise SystemExit(f"Set {key_name} first")
    root = Path(__file__).resolve().parents[1]
    model = (os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2") if provider == "gemini"
             else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    store = DocumentStore(Path(os.getenv("RAG_DATA_DIR", root / "data")), key,
                          model, provider)
    manifest = json.loads((root / "corpus" / "index.json").read_text(encoding="utf-8"))
    for doc in manifest:
        text = (root / "corpus" / doc["file"]).read_text(encoding="utf-8")
        print(store.add_document(text, doc["url"], doc["title"]))


if __name__ == "__main__":
    main()
