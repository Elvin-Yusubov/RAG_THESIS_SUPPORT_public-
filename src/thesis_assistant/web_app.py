"""Streamlit web front-end for the thesis-support assistant.

Features: optional shared-password gate, streaming chat, a collapsible list of
cited sources for each answer, and upload of a student's .docx draft that is
attached to the next question only (never added to the shared index).
Run with: streamlit run src/thesis_assistant/web_app.py
"""

import hmac
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import streamlit as st

from thesis_assistant import auth, config, metrics
from thesis_assistant.logging_config import new_request_id, setup_logging
from thesis_assistant.rag import extract_docx_text, stream_llm
from thesis_assistant.ratelimit import SlidingWindowRateLimiter
from thesis_assistant.storage import StateStore


def _format_sources(context_docs: List[Any]) -> List[str]:
    """Turn retrieved documents into readable labels like ``[category] file (p.3)``."""
    sources = []
    for doc in context_docs or []:
        meta = getattr(doc, "metadata", None) or {}
        source = meta.get("source", "Unknown")
        category = meta.get("category")
        page = meta.get("page_number")
        label = source
        if category:
            label = f"[{category}] {label}"
        if page:
            label = f"{label} (p.{page})"
        sources.append(label)
    return sources


def _build_chat_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Reduce session messages to the plain {role, content} dicts the backend expects."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]


def _text_chunks(
    query: str,
    chat_history: List[Dict[str, str]],
    result_holder: dict,
    document_text: Optional[str] = None,
):
    """Adapt stream_llm for st.write_stream.

    Yields only text chunks; the final {"answer", "context"} dict (not a
    string, so write_stream cannot display it) is stored in ``result_holder``.
    """
    for item in stream_llm(query, chat_history=chat_history, document_text=document_text):
        if isinstance(item, str):
            yield item
        else:
            result_holder["result"] = item


setup_logging()
logger = logging.getLogger("web_app")

st.set_page_config(page_title="Thesis Support Assistant", layout="centered")

# --- Access control -----------------------------------------------------------
# Per-user accounts (APP_USERS, salted PBKDF2 hashes) take precedence. The legacy
# shared password (APP_PASSWORD) is a fallback. With neither set the app is open.
USERS = auth.parse_users(config.APP_USERS)
AUTH_ENABLED = bool(USERS or config.APP_PASSWORD)


@st.cache_resource
def get_rate_limiter() -> SlidingWindowRateLimiter:
    """One limiter shared by every Streamlit session in this process."""
    return SlidingWindowRateLimiter(config.RATE_LIMIT_MAX_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS)


@st.cache_resource
def get_store() -> StateStore:
    return StateStore(config.STATE_DB_PATH)


def _login_form() -> None:
    """Render the login form and set session state on success."""
    st.title("Thesis Support Assistant")
    username = st.text_input("İstifadəçi adı", key="username_input") if USERS else ""
    password = st.text_input("Şifrə", type="password", key="password_input")
    if st.button("Daxil ol"):
        if USERS:
            valid = auth.authenticate(username, password, USERS)
        else:
            valid = hmac.compare_digest(password, config.APP_PASSWORD or "")
        if valid:
            st.session_state.authenticated = True
            st.session_state.user = username or "shared"
            st.rerun()
        st.error("İstifadəçi adı və ya şifrə yanlışdır." if USERS else "Şifrə yanlışdır.")


if AUTH_ENABLED and not st.session_state.get("authenticated"):
    _login_form()
    st.stop()  # nothing below this line renders until the user is signed in

USER_KEY = st.session_state.get("user", "anonymous")

st.title("Thesis Support Assistant")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("uploaded_document_text", None)
        st.session_state.pop("uploaded_document_name", None)
        st.rerun()

    if AUTH_ENABLED:
        st.caption(f"Daxil olub: {USER_KEY}")
        if st.button("Çıxış (log out)", use_container_width=True):
            st.session_state.pop("authenticated", None)
            st.session_state.pop("user", None)
            st.rerun()

    st.subheader("Sənədini yüklə")
    uploaded_file = st.file_uploader(
        "Buraxılış işi/dissertasiya (.docx)", type=["docx"], key="docx_uploader"
    )
    if uploaded_file is not None:
        # Streamlit reruns the script on every interaction; extract only when the
        # uploaded file (name + size) changes.
        file_fingerprint = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("uploaded_document_fingerprint") != file_fingerprint:
            with st.spinner("Sənəd oxunur…"):
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    st.session_state.uploaded_document_text = extract_docx_text(tmp_path)
                    st.session_state.uploaded_document_name = uploaded_file.name
                    st.session_state.uploaded_document_fingerprint = file_fingerprint
                except Exception as e:
                    st.error(f"Sənəd oxunarkən xəta baş verdi: {e}")
                finally:
                    os.unlink(tmp_path)

    if st.session_state.get("uploaded_document_name"):
        st.success(f"Yükləndi: {st.session_state.uploaded_document_name}")
        if st.button("Sənədi sil", use_container_width=True):
            st.session_state.pop("uploaded_document_text", None)
            st.session_state.pop("uploaded_document_name", None)
            st.session_state.pop("uploaded_document_fingerprint", None)
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me about thesis structure, methodology, university templates, or submission rules — or ask for help drafting or improving a section of your thesis. I can also show you the sources behind my answers.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question, or ask for help with your thesis…")
if prompt and not get_rate_limiter().allow(USER_KEY):
    metrics.RATE_LIMITED.labels(channel="web").inc()
    wait = max(1, round(get_rate_limiter().retry_after(USER_KEY)))
    st.warning(f"Çox sorğu göndərmisiniz. {wait} saniyə sonra yenidən cəhd edin.")
elif prompt:
    new_request_id()
    # History excludes the new prompt so it is not sent twice.
    chat_history = _build_chat_history(st.session_state.messages)

    # The uploaded document is attached to the first question after upload and
    # then cleared. The model's reply carries what it learned forward in the
    # history; re-upload if the student needs the document again later.
    document_text = st.session_state.pop("uploaded_document_text", None)
    document_name = st.session_state.pop("uploaded_document_name", None)
    st.session_state.pop("uploaded_document_fingerprint", None)

    display_prompt = prompt
    if document_name:
        display_prompt = f"📎 {document_name}\n\n{prompt}"

    st.session_state.messages.append({"role": "user", "content": display_prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(display_prompt)

    with st.chat_message("assistant"):
        started = time.monotonic()
        try:
            result_holder: Dict[str, Any] = {}
            answer = st.write_stream(
                _text_chunks(prompt, chat_history, result_holder, document_text=document_text)
            )
            answer = str(answer or "").strip() or "(No answer returned.)"

            result = result_holder.get("result", {})
            sources = _format_sources(result.get("context", []))

            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            latency = time.monotonic() - started
            kind = "text+docx" if document_name else "text"
            get_store().log_usage(USER_KEY, "web", kind, latency_ms=round(latency * 1000), sources=len(sources))
            metrics.record_request("web", kind, ok=True, latency_s=latency, sources=len(sources))
            logger.info("answered", extra={"channel": "web", "kind": kind, "latency_ms": round(latency * 1000)})
        except Exception:
            logger.exception("Failed to generate a response for user=%s", USER_KEY)
            get_store().log_usage(USER_KEY, "web", "text", ok=False)
            metrics.record_request("web", "text", ok=False)
            st.error("Failed to generate a response. Please try again.")