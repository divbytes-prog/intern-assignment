"""Small live evaluation suite for the documentation RAG."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


ROOT = Path(__file__).resolve().parents[1]


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation cases must be a non-empty JSON list")
    return data


def markdown_report(rows: list[dict]) -> str:
    passed = sum(row["passed"] for row in rows)
    answered = [row for row in rows if row["expected_status"] == "answered"]
    abstentions = [row for row in rows if row["expected_status"] == "insufficient_context"]
    source_hits = sum(row["source_ok"] for row in answered)
    grounded = sum(row["grounded_ok"] for row in rows)
    lines = [
        "# Latest live RAG evaluation",
        "",
        f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Overall cases | {passed}/{len(rows)} passed |",
        f"| Answered cases with expected source | {source_hits}/{len(answered)} |",
        f"| Grounding / abstention behavior | {grounded}/{len(rows)} |",
        f"| Out-of-domain abstention | {sum(row['passed'] for row in abstentions)}/{len(abstentions)} |",
        "",
        "| Case | Expected | Observed | Source check | Grounding | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['expected_status']} | {row['status']} | "
            f"{'PASS' if row['source_ok'] else 'FAIL'} | "
            f"{'PASS' if row['grounded_ok'] else 'FAIL'} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## What this checks",
        "",
        "- Direct questions across all four bundled FastAPI notes.",
        "- A follow-up question that reuses the API session ID.",
        "- Expected-source selection for answerable questions.",
        "- Explicit cited sources for answered questions.",
        "- Abstention with no sources for an out-of-domain question.",
        "",
        "This is a compact project regression set, not a general-purpose RAG benchmark. "
        "Model-provider behavior can vary between runs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "evaluation" / "cases.json"))
    parser.add_argument("--output", default=str(ROOT / "evaluation" / "latest.md"))
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is required for the live evaluation")
    os.environ["AI_PROVIDER"] = "groq"
    os.environ["EMBEDDING_PROVIDER"] = "local"
    os.environ.pop("RAG_MAX_RETRIES", None)  # Use the application default retry policy.
    os.environ.pop("TAVILY_API_KEY", None)

    cases = load_cases(Path(args.cases))
    sessions: dict[str, str] = {}
    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as directory:
        os.environ["RAG_DATA_DIR"] = directory
        with TestClient(create_app()) as client:
            manifest = json.loads((ROOT / "corpus" / "index.json").read_text(encoding="utf-8"))
            for doc in manifest:
                content = (ROOT / "corpus" / doc["file"]).read_text(encoding="utf-8")
                client.app.state.service.store.add_document(content, doc["url"], doc["title"])

            for case in cases:
                payload = {"question": case["question"]}
                session_name = case.get("session")
                if session_name and session_name in sessions:
                    payload["session_id"] = sessions[session_name]
                response = None
                for request_attempt, delay in enumerate((0, 5, 15)):
                    if delay:
                        time.sleep(delay)
                    response = client.post("/query", json=payload)
                    if response.status_code not in {429, 502}:
                        break
                    print(
                        f"EVAL {case['id']}: transient HTTP {response.status_code}; "
                        f"retry {request_attempt + 1}/3",
                        flush=True,
                    )
                if response.status_code != 200:
                    rows.append({
                        "id": case["id"], "expected_status": case["expected_status"],
                        "status": f"http_{response.status_code}", "source_ok": False,
                        "grounded_ok": False, "passed": False,
                    })
                    continue

                result = response.json()
                if session_name:
                    sessions[session_name] = result["session_id"]

                expected_source = case.get("expected_source")
                sources = result.get("sources") or []
                source_ok = (
                    any(source.get("source") == expected_source for source in sources)
                    if expected_source else not sources
                )
                grounded_ok = (
                    bool(sources) if case["expected_status"] == "answered" else not sources
                )
                status_ok = result.get("status") == case["expected_status"]
                row = {
                    "id": case["id"],
                    "expected_status": case["expected_status"],
                    "status": result.get("status"),
                    "source_ok": source_ok,
                    "grounded_ok": grounded_ok,
                    "passed": status_ok and source_ok and grounded_ok,
                }
                rows.append(row)
                print(
                    f"EVAL {case['id']}: {'PASS' if row['passed'] else 'FAIL'} "
                    f"status={row['status']} source_ok={source_ok} "
                    f"grounded={grounded_ok} attempts={result.get('retrieval_attempts')} "
                    f"mode={result.get('retrieval_mode')}",
                    flush=True,
                )

    report = markdown_report(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)

    passed = sum(row["passed"] for row in rows)
    if passed != len(rows):
        raise SystemExit(f"Evaluation failed: {passed}/{len(rows)} cases passed")


if __name__ == "__main__":
    main()
