"""Manual credentialed smoke test; never records provider keys or answer text."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.web_search import TavilySearch


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required for this live smoke test")
    os.environ["AI_PROVIDER"] = "gemini"
    os.environ["RAG_MAX_RETRIES"] = "0"  # Conserve the free-tier model quota.
    with tempfile.TemporaryDirectory() as directory:
        os.environ["RAG_DATA_DIR"] = directory
        with TestClient(create_app()) as client:
            # Report only model decision shapes, never questions, excerpts, or keys.
            llm = client.app.state.service.workflow.llm
            original_json = llm.json

            def observed_json(system: str, user: str) -> dict:
                result = original_json(system, user)
                if "relevance grader" in system:
                    grades = result.get("grades")
                    summary = ({str(key): str(value)[:12] for key, value in grades.items()}
                               if isinstance(grades, dict) else type(grades).__name__)
                    types = {key: type(value).__name__ for key, value in result.items()}
                    print(f"Grading decision: keys={list(result)}, grades={summary}, "
                          f"value_types={types}", flush=True)
                return result

            llm.json = observed_json
            document = Path("corpus/path-parameters.md")
            ingested = client.post("/ingest", files={"file":
                (document.name, document.read_bytes(), "text/markdown")})
            assert ingested.status_code == 201, (
                f"Ingest failed: HTTP {ingested.status_code}, {ingested.text[:400]}"
            )
            listed = client.get("/documents")
            assert listed.status_code == 200 and listed.json()["documents"]
            question = "How does FastAPI validate an integer path parameter?"
            first = client.post("/query", json={"question": question})
            assert first.status_code == 200, f"Query failed: HTTP {first.status_code}, {first.json().get('detail')}"
            answer = first.json()
            assert answer["status"] == "answered", (
                f"Query abstained: {answer['status']}, reason={answer['failure_reason']}, "
                f"mode={answer['retrieval_mode']}, attempts={answer['retrieval_attempts']}"
            )
            assert answer["sources"] and all(s["marker"] in answer["answer"] for s in answer["sources"])
            assert answer["retrieval_mode"] == "local"
            feedback = client.post("/feedback", json={"answer_id": answer["answer_id"], "rating": "up"})
            assert feedback.status_code == 200
            print("PASS: live embed, ingest, local retrieval, LLM grading, cited generation, support check, and feedback")

            # The default free-tier Gemini model may permit only five calls per minute.
            time.sleep(65)
            followup = client.post("/query", json={"question": "How does it report an invalid value?",
                                                   "session_id": answer["session_id"]})
            assert followup.status_code == 200, f"Follow-up failed: HTTP {followup.status_code}, {followup.json().get('detail')}"
            next_answer = followup.json()
            assert next_answer["session_id"] == answer["session_id"]
            assert next_answer["status"] == "answered" and next_answer["sources"]
            print("PASS: live session follow-up with cited answer")

    if os.getenv("TAVILY_API_KEY"):
        results = TavilySearch(os.environ["TAVILY_API_KEY"]).search("FastAPI path parameter validation")
        assert results, "Tavily returned no allowlisted documentation excerpts"
        print("PASS: live Tavily search and official-host filtering")
    else:
        print("SKIP: optional live Tavily request (no TAVILY_API_KEY secret)")


if __name__ == "__main__":
    main()
