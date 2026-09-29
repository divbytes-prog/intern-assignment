"""Optional Streamlit chat frontend; start the FastAPI server first."""
from __future__ import annotations

import os
from urllib.parse import urlparse

import requests
import streamlit as st

API = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Technical Documentation Assistant", page_icon="📚")
st.title("Technical Documentation Assistant")
st.caption("Ask the indexed docs, see citations, and follow up within this session.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None


def request(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, API + path, timeout=45, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 429:
            raise RuntimeError("Model quota reached. Wait and try again.") from exc
        raise RuntimeError(f"API request failed ({status or 'connection'}). Is FastAPI running?") from exc


def show_answer(message: dict):
    st.write(message["answer"])
    st.caption(f"Source: {message.get('retrieval_mode', 'local')}")
    for source in message.get("sources", []):
        url = source["source"]
        if urlparse(url).scheme == "https":
            st.markdown(f"[{source['marker']}] [{source['title']}]({url})")
        else:
            st.text(f"[{source['marker']}] {source['title']} ({url})")


with st.sidebar:
    if st.button("New conversation"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()
    st.subheader("Indexed documents")
    try:
        for doc in request("GET", "/documents")["documents"]:
            st.write(f"{doc['title']} · {doc['chunks']} chunks")
    except RuntimeError as exc:
        st.warning(str(exc))
    file = st.file_uploader("Add a .md, .txt, or .html document", type=["md", "txt", "html"])
    if file and st.button("Index file"):
        try:
            request("POST", "/ingest", files={"file": (file.name, file.getvalue())})
            st.success("Document indexed. You can ask about it now.")
        except RuntimeError as exc:
            st.error(str(exc))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            show_answer(msg)
        else:
            st.write(msg["content"])

if question := st.chat_input("Ask a documentation question"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            payload = {"question": question}
            if st.session_state.session_id:
                payload["session_id"] = st.session_state.session_id
            result = request("POST", "/query", json=payload)
            st.session_state.session_id = result["session_id"]
            message = {"role": "assistant", **result}
            st.session_state.messages.append(message)
            show_answer(message)
        except RuntimeError as exc:
            st.error(str(exc))

last = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
if last:
    st.caption("Was the most recent answer useful?")
    left, right = st.columns(2)
    for column, label, rating in [(left, "Helpful", "up"), (right, "Not helpful", "down")]:
        if column.button(label):
            try:
                request("POST", "/feedback", json={"answer_id": last["answer_id"], "rating": rating})
                st.toast("Feedback saved")
            except RuntimeError as exc:
                st.error(str(exc))
