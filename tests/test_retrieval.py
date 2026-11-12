from langchain_core.documents import Document

from thesis_assistant.retrieval import bm25_scores, fuse_rankings, rerank_hybrid, tokenize


def _docs(*texts):
    return [Document(page_content=t, metadata={"source": f"d{i}"}) for i, t in enumerate(texts)]


def test_tokenize_keeps_azerbaijani_letters():
    assert tokenize("Dövlət Attestasiya Komissiyası") == ["dövlət", "attestasiya", "komissiyası"]


def test_bm25_prefers_documents_with_query_terms():
    docs = _docs("buraxılış işi qaydaları", "tamamilə başqa mövzu")
    scores = bm25_scores("buraxılış qaydaları", docs)
    assert scores[0] > scores[1] == 0


def test_fuse_rankings_promotes_strong_lexical_match():
    # Dense order puts candidate 2 last; BM25 strongly prefers it.
    order = fuse_rankings([0, 1, 2], [0.0, 0.1, 5.0])
    assert order[0] in (2, 0) and order.index(2) < 2


def test_fuse_rankings_keeps_dense_order_without_lexical_signal():
    assert fuse_rankings([0, 1, 2], [0.0, 0.0, 0.0]) == [0, 1, 2]


def test_rerank_hybrid_returns_top_k():
    docs = _docs("alpha beta", "gamma delta", "DAK qaydaları 123")
    result = rerank_hybrid("DAK 123", docs, top_k=2)
    assert len(result) == 2
    assert result[0].metadata["source"] == "d2" or result[1].metadata["source"] == "d2"
