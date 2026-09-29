"""Safe, bounded document ingestion."""
from __future__ import annotations

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

ALLOWED_HOSTS = {"fastapi.tiangolo.com", "docs.pydantic.dev", "docs.python.org"}
MAX_BYTES = 1_000_000


def fetch_document(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Only HTTPS URLs on the supported official documentation hosts are allowed") from exc
    if (parsed.scheme != "https" or host not in ALLOWED_HOSTS or parsed.username or parsed.password
            or port not in {None, 443}):
        raise ValueError("Only HTTPS URLs on the supported official documentation hosts are allowed")
    with requests.get(url, timeout=15, stream=True, allow_redirects=False) as response:
        if 300 <= response.status_code < 400:
            raise ValueError("Redirects are not followed; provide the final documentation URL")
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].lower()
        if content_type not in {"text/html", "text/plain", "text/markdown"}:
            raise ValueError("URL must return HTML, Markdown, or plain text")
        raw = bytearray()
        for part in response.iter_content(chunk_size=16_384):
            raw.extend(part)
            if len(raw) > MAX_BYTES:
                raise ValueError("Document exceeds the 1 MB limit")
    text = raw.decode("utf-8", errors="replace")
    if content_type == "text/html":
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        return soup.get_text("\n", strip=True), title
    return text, url
