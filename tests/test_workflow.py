import re

from app.store import split_markdown
from app.workflow import RAGWorkflow


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query, limit):
        self.calls.append(query)
        return self.chunks[:limit]


class FakeLLM:
    def __init__(self, relevant=True, supported=True):
        self.relevant = relevant
        self.supported = supported

    def json(self, system, user):
        if "query_type" in system:
            return {"search_query": "FastAPI path parameter integer", "query_type": "how-to"}
        if "relevance grader" in system:
            return {"grades": {i: self.relevant for i in re.findall(r"(?m)^ID (\d+):", user)}}
        if "substantially different" in system:
            return {"search_query": "route URL variable type validation"}
        if "Answer ONLY" in system:
            return {"answer": "Declare item_id: int on the route function [S1].", "citations": ["S1"]}
        if "Check whether EVERY" in system:
            return {"supported": self.supported}
        raise AssertionError("Unexpected model call")


CHUNK = {"id": "one:0", "text": "Declare item_id: int for a typed path parameter.",
         "title": "Path parameters", "source": "https://fastapi.tiangolo.com/tutorial/path-params/"}


def test_relevant_chunk_is_cited_and_no_retry():
    store = FakeStore([CHUNK])
    result = RAGWorkflow(store, FakeLLM(), max_retries=2).invoke("How do I validate item_id?")
    assert result["status"] == "answered"
    assert result["attempts"] == 1
    assert result["sources"][0]["chunk_id"] == "one:0"
    assert "[S1]" in result["answer"]


def test_irrelevant_results_retry_then_abstain():
    store = FakeStore([CHUNK])
    result = RAGWorkflow(store, FakeLLM(relevant=False), max_retries=2).invoke("How do I validate item_id?")
    assert result["status"] == "insufficient_context"
    assert result["attempts"] == 3
    assert result["sources"] == []
    assert len(store.calls) == 3


def test_empty_store_still_has_bounded_retry():
    result = RAGWorkflow(FakeStore([]), FakeLLM(), max_retries=1).invoke("What is the answer?")
    assert result["status"] == "insufficient_context"
    assert result["attempts"] == 2


def test_split_preserves_text_and_limits_chunks():
    text = "# Title\n\n" + "data " * 300
    chunks = split_markdown(text, max_chars=300, overlap=40)
    assert chunks
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert "data" in chunks[-1]


def test_invalid_model_citations_do_not_leak():
    class BadLLM(FakeLLM):
        def json(self, system, user):
            if "Answer ONLY" in system:
                return {"answer": "Unsupported claim [S99].", "citations": ["S99"]}
            return super().json(system, user)

    result = RAGWorkflow(FakeStore([CHUNK]), BadLLM()).invoke("How do I validate item_id?")
    assert result["status"] == "insufficient_context"


def test_malformed_citation_list_abstains_instead_of_crashing():
    class BadLLM(FakeLLM):
        def json(self, system, user):
            if "Answer ONLY" in system:
                return {"answer": "An unsupported claim [S1].", "citations": [{"marker": "S1"}]}
            return super().json(system, user)

    result = RAGWorkflow(FakeStore([CHUNK]), BadLLM()).invoke("How do I validate item_id?")
    assert result["status"] == "insufficient_context"


def test_mixed_relevance_filters_irrelevant_chunk():
    class MixedLLM(FakeLLM):
        def json(self, system, user):
            if "relevance grader" in system:
                return {"grades": {"0": False, "1": True}}
            return super().json(system, user)

    irrelevant = dict(CHUNK, id="other:0", text="unrelated weather data")
    result = RAGWorkflow(FakeStore([irrelevant, CHUNK]), MixedLLM()).invoke("How do I validate item_id?")
    assert result["status"] == "answered"
    assert result["sources"][0]["chunk_id"] == "one:0"
    assert len(result["relevant"]) == 1


def test_grading_multiple_chunks_uses_one_model_call():
    class CountingLLM(FakeLLM):
        grade_calls = 0

        def json(self, system, user):
            if "relevance grader" in system:
                self.grade_calls += 1
            return super().json(system, user)

    llm = CountingLLM()
    RAGWorkflow(FakeStore([CHUNK, dict(CHUNK, id="two:0")]), llm).invoke("Path validation?")
    assert llm.grade_calls == 1


def test_support_check_rejects_unsupported_answer():
    result = RAGWorkflow(FakeStore([CHUNK]), FakeLLM(supported=False)).invoke(
        "How do I validate item_id?"
    )
    assert result["status"] == "insufficient_context"
    assert result["sources"] == []
