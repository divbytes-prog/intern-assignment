"""Optional, bounded Tavily fallback for official technical documentation."""
from __future__ import annotations

import hashlib
from urllib.parse import urlparse

import requests

from app.ingest import ALLOWED_HOSTS


class TavilySearch:
    def __init__(self, api_key: str):
        if not api_key.strip():
            raise ValueError("A Tavily key is required")
        self.api_key = api_key

    def search(self, query: str) -> list[dict]:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"query": query[:500], "search_depth": "basic", "max_results": 3,
                  "include_domains": sorted(ALLOWED_HOSTS), "include_domains_mode": "restrict",
                  "include_answer": False, "include_raw_content": False,
                  "auto_parameters": False},
            timeout=12,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not isinstance(results, list):
            return []
        chunks = []
        for item in results[:3]:
            if not isinstance(item, dict):
                continue
            url, content = item.get("url"), item.get("content")
            if not isinstance(url, str) or not isinstance(content, str) or not content.strip():
                continue
            parsed = urlparse(url)
            if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
                    or parsed.username or parsed.password or parsed.port not in {None, 443}):
                continue
            chunks.append({"id": "web:" + hashlib.sha256(url.encode()).hexdigest()[:20],
                           "source": url, "title": str(item.get("title") or parsed.hostname)[:200],
                           "text": content[:1800]})
        return chunks
