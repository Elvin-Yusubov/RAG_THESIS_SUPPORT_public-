"""Agent-level tests with a fake agent and fake vector store (no network)."""

from langchain.messages import AIMessage, ToolMessage
from langchain_core.documents import Document

from thesis_assistant import config, rag


class FakeAgent:
    def __init__(self, doc):
        self.doc = doc
        self.last_messages = None

    def invoke(self, payload):
        self.last_messages = payload["messages"]
        tool = ToolMessage(content="ctx", tool_call_id="1", artifact=[self.doc])
        return {"messages": [tool, AIMessage(content="final answer")]}

    def stream(self, payload, stream_mode):
        self.last_messages = payload["messages"]
        yield ToolMessage(content="ctx", tool_call_id="1", artifact=[self.doc]), {}
        yield AIMessage(content="Hel"), {}
        yield AIMessage(content="lo"), {}


def _patch_agent(monkeypatch):
    doc = Document(page_content="x", metadata={"source": "s.pdf"})
    agent = FakeAgent(doc)
    monkeypatch.setattr(rag, "get_agent", lambda: agent)
    return agent, doc


def test_run_llm_returns_answer_and_sources(monkeypatch):
    agent, doc = _patch_agent(monkeypatch)
    result = rag.run_llm("sual", chat_history=[{"role": "user", "content": "old"}] * 30)
    assert result == {"answer": "final answer", "context": [doc]}
    # history trimmed, then the new question appended
    assert len(agent.last_messages) == config.MAX_HISTORY_TURNS * 2 + 1


def test_stream_llm_yields_chunks_then_final_dict(monkeypatch):
    _, doc = _patch_agent(monkeypatch)
    items = list(rag.stream_llm("sual"))
    assert items[:2] == ["Hel", "lo"]
    assert items[-1] == {"answer": "Hello", "context": [doc]}


def test_uploaded_document_is_included_in_message(monkeypatch):
    agent, _ = _patch_agent(monkeypatch)
    rag.run_llm("rəy yaz", document_text="Mənim işim")
    assert "Mənim işim" in agent.last_messages[-1]["content"]


class FakeStore:
    """Mimics the vector store: returns fixed candidates for any query."""

    def __init__(self, docs):
        self.docs = docs
        self.requested_k = None

    def as_retriever(self, search_kwargs):
        self.requested_k = search_kwargs["k"]
        return self

    def invoke(self, query):
        return self.docs[: self.requested_k]


def test_search_documents_dense_uses_top_k(monkeypatch):
    store = FakeStore([Document(page_content=str(i), metadata={}) for i in range(10)])
    monkeypatch.setattr(rag, "get_vectorstore", lambda: store)
    assert len(rag.search_documents("q", mode="dense")) == config.RETRIEVAL_TOP_K
    assert store.requested_k == config.RETRIEVAL_TOP_K


def test_search_documents_hybrid_overfetches_then_reranks(monkeypatch):
    docs = [Document(page_content=t, metadata={"source": t}) for t in ["aaa", "bbb", "DAK 12 qayda", "ccc", "ddd"]]
    store = FakeStore(docs)
    monkeypatch.setattr(rag, "get_vectorstore", lambda: store)
    monkeypatch.setattr(config, "RETRIEVAL_FETCH_K", 5)
    result = rag.search_documents("DAK 12", mode="hybrid", top_k=2)
    assert store.requested_k == 5 and len(result) == 2
    assert "DAK 12 qayda" in [d.page_content for d in result]
