"""FastAPI entry point. Start with: uvicorn app.main:app --reload."""
from __future__ import annotations

import os
import json
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from starlette.datastructures import UploadFile

from app.ingest import MAX_BYTES, fetch_document
from app.llm import LLM, ModelResponseError
from app.store import DocumentStore
from app.workflow import RAGWorkflow
from app.web_search import TavilySearch

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def query_error_detail(exc: Exception) -> dict:
    """Return only safe diagnostic codes; provider messages can contain secrets."""
    status = None
    current: BaseException | None = exc
    while current is not None:
        code = getattr(current, "code", None) or getattr(current, "status_code", None)
        if isinstance(code, int) and 400 <= code < 600:
            status = code
            break
        current = current.__cause__
    category = {
        400: "invalid_model_request", 401: "invalid_api_key", 403: "model_access_denied",
        404: "model_not_found", 429: "quota_or_rate_limit",
    }.get(status)
    if category is None:
        category = ("provider_unavailable" if status and status >= 500 else
                    "invalid_model_json" if isinstance(exc, (json.JSONDecodeError, ModelResponseError))
                    else "retrieval_or_model_error")
    return {"error": category, "provider_status": status}


class QueryInput(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    session_id: uuid.UUID | None = None

    @field_validator("question")
    @classmethod
    def meaningful_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Question must contain at least three non-space characters")
        return value


class SourceOutput(BaseModel):
    marker: str
    title: str
    source: str
    chunk_id: str


class QueryOutput(BaseModel):
    answer_id: str
    session_id: uuid.UUID
    answer: str
    sources: list[SourceOutput]
    status: str
    retrieval_attempts: int
    retrieval_mode: str
    failure_reason: str | None = None


class DocumentOutput(BaseModel):
    document_id: str
    source: str
    title: str
    chunks: int


class DocumentsOutput(BaseModel):
    documents: list[DocumentOutput]


class FeedbackOutput(BaseModel):
    saved: bool
    answer_id: str


class FeedbackInput(BaseModel):
    answer_id: str
    rating: str
    comment: str | None = Field(default=None, max_length=2000)


class Service:
    def __init__(self, data_dir: Path, api_key: str):
        provider = os.getenv("AI_PROVIDER", "groq").lower()
        if provider not in {"groq", "gemini", "openai"}:
            raise ValueError("AI_PROVIDER must be groq, gemini, or openai")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
        if embedding_provider not in {"local", "gemini", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be local, gemini, or openai")
        embedding_model = {
            "local": os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            "gemini": os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
            "openai": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        }[embedding_provider]
        chat_model = {
            "groq": os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b"),
            "gemini": os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview"),
            "openai": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        }[provider]
        embedding_key = None if embedding_provider == "local" else os.getenv(
            {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}[embedding_provider]
        )
        if embedding_provider != "local" and not embedding_key:
            raise ValueError(f"Set the {embedding_provider.upper()} embedding API key")
        self.store = DocumentStore(data_dir, embedding_key, embedding_model, embedding_provider)
        self.workflow = RAGWorkflow(
            self.store, LLM(api_key, chat_model, provider),
            top_k=int(os.getenv("RAG_TOP_K", "4")), max_retries=int(os.getenv("RAG_MAX_RETRIES", "2")),
            web_search=TavilySearch(os.environ["TAVILY_API_KEY"]) if os.getenv("TAVILY_API_KEY") else None,
        )
        self.lock = threading.RLock()
        self.db = sqlite3.connect(data_dir / "feedback.sqlite3", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS answers (id TEXT PRIMARY KEY, question TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS feedback (answer_id TEXT PRIMARY KEY, rating TEXT NOT NULL, comment TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, status TEXT NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id)")
        self.db.commit()

    def recent_history(self, session_id: uuid.UUID) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT question, answer FROM turns WHERE session_id=? "
                                   "ORDER BY id DESC LIMIT 4", (str(session_id),)).fetchall()
        return [{"question": q, "answer": a} for q, a in reversed(rows)]

    def save_turn(self, session_id: uuid.UUID, question: str, answer: str, status: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO turns(session_id, question, answer, status) VALUES (?, ?, ?, ?)",
                            (str(session_id), question, answer, status))
            self.db.execute("DELETE FROM turns WHERE session_id=? AND id NOT IN "
                            "(SELECT id FROM turns WHERE session_id=? ORDER BY id DESC LIMIT 4)",
                            (str(session_id), str(session_id)))
            self.db.commit()

    def save_answer(self, answer_id: str, question: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO answers VALUES (?, ?)", (answer_id, question))
            self.db.commit()

    def save_feedback(self, data: FeedbackInput) -> None:
        with self.lock:
            found = self.db.execute("SELECT 1 FROM answers WHERE id=?", (data.answer_id,)).fetchone()
            if not found:
                raise KeyError(data.answer_id)
            self.db.execute(
                "INSERT INTO feedback(answer_id, rating, comment) VALUES (?, ?, ?) "
                "ON CONFLICT(answer_id) DO UPDATE SET rating=excluded.rating, comment=excluded.comment",
                (data.answer_id, data.rating, data.comment),
            )
            self.db.commit()


def create_app(service_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        provider = os.getenv("AI_PROVIDER", "groq").lower()
        key_name = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
                    "openai": "OPENAI_API_KEY"}.get(provider)
        if key_name is None:
            raise RuntimeError("AI_PROVIDER must be groq, gemini, or openai")
        key = os.getenv(key_name)
        if service_factory is None and not key:
            raise RuntimeError(f"Set {key_name} before starting the application")
        app.state.service = service_factory() if service_factory else Service(
            Path(os.getenv("RAG_DATA_DIR", "./data-local")), key
        )
        yield
        if hasattr(app.state.service, "db"):
            app.state.service.db.close()

    api = FastAPI(title="Express Docs RAG", version="1.0.0", lifespan=lifespan)

    @api.get("/", response_class=FileResponse, include_in_schema=False)
    def home():
        return FileResponse(Path(__file__).parent / "static" / "index.html", media_type="text/html")

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.post("/query", response_model=QueryOutput)
    def query(data: QueryInput, request: Request):
        service = request.app.state.service
        session_id = data.session_id or uuid.uuid4()
        try:
            history = service.recent_history(session_id)
            result = service.workflow.invoke(data.question, history=history)
        except Exception as exc:
            detail = query_error_detail(exc)
            raise HTTPException(status_code=429 if detail["provider_status"] == 429 else 502,
                                detail=detail) from exc
        answer_id = str(uuid.uuid4())
        service.save_answer(answer_id, data.question)
        service.save_turn(session_id, data.question, result["answer"], result["status"])
        return {"answer_id": answer_id, "answer": result["answer"], "sources": result["sources"],
                "session_id": session_id, "status": result["status"],
                "retrieval_attempts": result["attempts"], "retrieval_mode": result["retrieval_mode"],
                "failure_reason": result.get("failure_reason") or None}

    @api.post("/ingest", status_code=201, response_model=DocumentOutput)
    async def ingest(request: Request):
        content_type = request.headers.get("content-type", "").lower()
        service = request.app.state.service
        try:
            if "application/json" in content_type:
                try:
                    data = await request.json()
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError("Request body must be valid JSON") from exc
                if not isinstance(data, dict) or not isinstance(data.get("url"), str):
                    raise ValueError("Provide a JSON object with a documentation URL")
                url = data["url"].strip()
                content, title = fetch_document(url)
                source = url
            elif "multipart/form-data" in content_type:
                form = await request.form()
                upload = form.get("file")
                url = str(form.get("url") or "")
                if bool(upload) == bool(url):
                    raise ValueError("Provide exactly one file or URL")
                if url:
                    content, title = fetch_document(url)
                    source = url
                else:
                    if not isinstance(upload, UploadFile):
                        raise ValueError("A file upload is required")
                    title = Path(upload.filename or "").name
                    if Path(title).suffix.lower() not in {".md", ".txt", ".html"}:
                        raise ValueError("Only .md, .txt, and .html files are supported")
                    raw = await upload.read(MAX_BYTES + 1)
                    if len(raw) > MAX_BYTES:
                        raise ValueError("Document exceeds the 1 MB limit")
                    try:
                        content = raw.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError("Uploaded file must contain valid UTF-8 text") from exc
                    if title.endswith(".html"):
                        soup = BeautifulSoup(content, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer", "header"]):
                            tag.decompose()
                        content = soup.get_text("\n", strip=True)
                    source = f"upload:{title}"
            else:
                raise ValueError("Use application/json with url or multipart/form-data with file or url")
            if not content.strip():
                raise ValueError("Document is empty")
            return service.store.add_document(content, source, title)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Could not fetch, embed, or index the document") from exc

    @api.get("/documents", response_model=DocumentsOutput)
    def documents(request: Request):
        return {"documents": request.app.state.service.store.list_documents()}

    @api.post("/feedback", response_model=FeedbackOutput)
    def feedback(data: FeedbackInput, request: Request):
        if data.rating not in {"up", "down"}:
            raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'")
        try:
            request.app.state.service.save_feedback(data)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown answer_id") from exc
        return {"saved": True, "answer_id": data.answer_id}

    return api


app = create_app()
