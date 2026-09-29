# RAG evaluation and verification report

This report intentionally separates **checks that were actually executed** from
the provider-dependent evaluation suite that can be rerun with `scripts/evaluate`.
No unexecuted case is counted as a pass.

## Verified checks

| Check | Verified result |
| --- | --- |
| Core/API regression suite | 31/31 passed on Python 3.12 and 31/31 passed on Python 3.14 |
| Optional Streamlit interaction suite | 2/2 passed |
| Credentialed end-to-end provider smoke | PASS |
| Live local embedding + ingestion + retrieval | PASS |
| Live Groq relevance grading + cited generation | PASS |
| Live support verification | PASS |
| Live feedback endpoint | PASS |
| Live follow-up session with cited answer | PASS |

The regression results come from GitHub Actions run
`36610329693`. The credentialed provider checks come from run
`36602704098`. Those checks exercise the real provider path without storing
keys or generated answer text in the repository.

## Human-authored RAG evaluation set

`evaluation/cases.json` contains seven reviewer-readable cases covering:

- integer path-parameter validation;
- Pydantic request-body models;
- dependency injection;
- a session follow-up about dependency caching;
- response-model field filtering;
- route-order behavior; and
- an out-of-domain Kubernetes question that should abstain.

For answerable cases, the runner requires the expected FastAPI source URL to be
present in the returned citations. For the out-of-domain case, it requires
`insufficient_context` with no supporting sources.

Run it with:

```bash
python -m scripts.evaluate --output evaluation/latest.md
```

A Groq key is required because this suite intentionally evaluates the same
LLM-backed grading, generation, and verification path used by the application.
Provider output and quota can vary between runs, so a case score is only written
when the suite is actually executed.

## Why this evaluation exists

The goal is not to present a large benchmark. It is to make the important RAG
failure modes visible: wrong-document retrieval, missing citations, unsupported
answers, broken follow-up context, and failure to abstain when the corpus does
not contain the answer.
