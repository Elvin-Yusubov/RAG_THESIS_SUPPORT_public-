"""Unit tests for pure helpers in the RAG backend (no network access)."""

from langchain_core.documents import Document

from thesis_assistant import config
from thesis_assistant.rag import build_user_message, format_retrieved_documents, trim_history


def test_trim_history_keeps_latest_turns():
    history = [{"role": "user", "content": str(i)} for i in range(20)]
    trimmed = trim_history(history)
    assert len(trimmed) == config.MAX_HISTORY_TURNS * 2
    assert trimmed[-1]["content"] == "19"


def test_trim_history_handles_empty():
    assert trim_history(None) == []
    assert trim_history([]) == []


def test_build_user_message_without_document_is_unchanged():
    assert build_user_message("salam") == "salam"


def test_build_user_message_truncates_long_document():
    long_text = "x" * (config.MAX_DOCUMENT_CHARS + 10)
    message = build_user_message("question", long_text)
    assert "truncated" in message
    assert message.endswith("question")


def test_format_retrieved_documents_includes_citation_metadata():
    doc = Document(
        page_content="x" * (config.MAX_CHARS_PER_CHUNK + 50),
        metadata={"source": "rules.pdf", "category": "DB-MASTER", "page_number": 3},
    )
    text = format_retrieved_documents([doc])
    assert "rules.pdf" in text and "DB-MASTER" in text and "page: 3" in text
    assert text.count("x") == config.MAX_CHARS_PER_CHUNK
