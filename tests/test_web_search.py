from app.web_search import TavilySearch


def test_tavily_filters_untrusted_domains_and_bounds_excerpts(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [
                {"url": "https://fastapi.tiangolo.com/tutorial/path-params/",
                 "title": "Path parameters", "content": "x" * 2500},
                {"url": "https://fastapi.tiangolo.com.evil.test/fake", "content": "bad"},
                {"url": "http://docs.python.org/3/", "content": "bad"},
            ]}

    def fake_post(url, *, headers, json, timeout):
        assert url == "https://api.tavily.com/search"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["include_domains_mode"] == "restrict"
        assert timeout == 12
        return Response()

    monkeypatch.setattr("app.web_search.requests.post", fake_post)
    chunks = TavilySearch("test-key").search("FastAPI path")
    assert len(chunks) == 1
    assert chunks[0]["id"].startswith("web:")
    assert len(chunks[0]["text"]) == 1800
