# Latest live RAG evaluation

Run: 2026-09-29 19:08 UTC

| Metric | Result |
| --- | ---: |
| Overall cases | 7/7 passed |
| Answered cases with expected source | 6/6 |
| Grounding / abstention behavior | 7/7 |
| Out-of-domain abstention | 1/1 |

| Case | Expected | Observed | Source check | Grounding | Result |
| --- | --- | --- | --- | --- | --- |
| path-integer-validation | answered | answered | PASS | PASS | PASS |
| request-body-model | answered | answered | PASS | PASS | PASS |
| dependency-basics | answered | answered | PASS | PASS | PASS |
| dependency-follow-up | answered | answered | PASS | PASS | PASS |
| response-filtering | answered | answered | PASS | PASS | PASS |
| path-route-order | answered | answered | PASS | PASS | PASS |
| out-of-domain-abstention | insufficient_context | insufficient_context | PASS | PASS | PASS |

## What this checks

- Direct questions across all four bundled FastAPI notes.
- A follow-up question that reuses the API session ID.
- Expected-source selection for answerable questions.
- Explicit cited sources for answered questions.
- Abstention with no sources for an out-of-domain question.

This is a compact project regression set, not a general-purpose RAG benchmark. Model-provider behavior can vary between runs.
