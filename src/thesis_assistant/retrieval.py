"""Retrieval strategies: dense similarity search with optional lexical fusion.

"dense" returns the vector store's top-k. "hybrid" over-fetches candidates
and fuses two rankings with Reciprocal Rank Fusion (RRF):

    1. the dense (semantic) order returned by Pinecone
    2. a BM25 (lexical) order computed over the candidate chunks

Lexical scoring helps with exact terms that embeddings blur: form numbers,
abbreviations such as "DAK", or specific regulation names. It only re-orders
the candidates the dense search already produced, so it improves precision
but cannot recover documents the dense search missed. Use
``python -m thesis_assistant.evaluation`` to compare modes on your own data.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List

from langchain_core.documents import Document

RRF_K = 60  # standard RRF damping constant
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens (Unicode-aware, so Azerbaijani letters are kept)."""
    return _TOKEN_RE.findall(text.casefold())


def bm25_scores(query: str, documents: List[Document]) -> List[float]:
    """BM25 score of each document for ``query``, computed over the given set."""
    tokenized = [tokenize(doc.page_content) for doc in documents]
    if not tokenized:
        return []
    avg_len = sum(len(t) for t in tokenized) / len(tokenized) or 1.0
    n_docs = len(tokenized)
    doc_freq = Counter(term for tokens in tokenized for term in set(tokens))

    scores = []
    for tokens in tokenized:
        term_freq = Counter(tokens)
        score = 0.0
        for term in set(tokenize(query)):
            if term not in term_freq:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            tf = term_freq[term]
            score += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / avg_len))
        scores.append(score)
    return scores


def fuse_rankings(dense_order: List[int], lexical_scores: List[float]) -> List[int]:
    """Reciprocal Rank Fusion of the dense order and the BM25 ranking.

    Args:
        dense_order: Candidate indices in dense-similarity order (best first).
        lexical_scores: BM25 score per candidate, indexed by candidate.

    Returns:
        Candidate indices, best first.
    """
    lexical_order = sorted(range(len(lexical_scores)), key=lambda i: lexical_scores[i], reverse=True)
    fused: Dict[int, float] = defaultdict(float)
    for order in (dense_order, lexical_order):
        for rank, index in enumerate(order):
            fused[index] += 1 / (RRF_K + rank + 1)
    # Ties keep the dense order because sorted() is stable.
    return sorted(dense_order, key=lambda i: fused[i], reverse=True)


def rerank_hybrid(query: str, candidates: List[Document], top_k: int) -> List[Document]:
    """Re-order dense candidates with lexical fusion and return the best ``top_k``."""
    scores = bm25_scores(query, candidates)
    order = fuse_rankings(list(range(len(candidates))), scores)
    return [candidates[i] for i in order[:top_k]]
