"""Retrieval evaluation: recall@k and MRR over a labelled question set.

Each line of the evaluation file is a JSON object::

    {"question": "...", "relevant_sources": ["file1.pdf", "file2.docx"]}

``relevant_sources`` lists the document names (the ``source`` metadata) that
contain the answer. A retrieved chunk is a hit if its source is listed.

Usage::

    python -m thesis_assistant.evaluation evaluation/eval_set.jsonl --k 3 --compare

``--compare`` runs dense and hybrid retrieval side by side so the retrieval
mode can be chosen from measurements instead of intuition. The question set
must be written against your own indexed corpus; the bundled
``eval_set.example.jsonl`` only shows the format.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

Sources = List[str]


def recall_at_k(retrieved: Sources, relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant sources found among the first ``k`` retrieved."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(relevant_set & set(retrieved[:k])) / len(relevant_set)


def hit_at_k(retrieved: Sources, relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant source appears in the first ``k`` results, else 0.0."""
    return float(bool(set(relevant) & set(retrieved[:k])))


def reciprocal_rank(retrieved: Sources, relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant result (0.0 if none was retrieved)."""
    relevant_set = set(relevant)
    for rank, source in enumerate(retrieved, start=1):
        if source in relevant_set:
            return 1 / rank
    return 0.0


def load_eval_set(path: Path) -> List[Dict]:
    """Read a JSONL evaluation file, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(examples: List[Dict], search: Callable[[str], Sources], k: int) -> Dict[str, float]:
    """Average hit@k, recall@k and MRR of ``search`` over the examples.

    Args:
        examples: Items with ``question`` and ``relevant_sources``.
        search: Maps a question to retrieved source names, best first.
        k: Cut-off for hit@k and recall@k.
    """
    if not examples:
        return {"hit@k": 0.0, "recall@k": 0.0, "mrr": 0.0, "n": 0}
    hits = recalls = rr = 0.0
    for example in examples:
        retrieved = search(example["question"])
        relevant = example["relevant_sources"]
        hits += hit_at_k(retrieved, relevant, k)
        recalls += recall_at_k(retrieved, relevant, k)
        rr += reciprocal_rank(retrieved, relevant)
    n = len(examples)
    return {"hit@k": hits / n, "recall@k": recalls / n, "mrr": rr / n, "n": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("eval_file", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--mode", choices=["dense", "hybrid"], help="default: RETRIEVAL_MODE")
    parser.add_argument("--compare", action="store_true", help="evaluate dense and hybrid")
    args = parser.parse_args()

    from thesis_assistant.rag import search_documents  # imported lazily: needs API keys

    examples = load_eval_set(args.eval_file)
    modes = ["dense", "hybrid"] if args.compare else [args.mode]

    print(f"{len(examples)} questions, k={args.k}")
    for mode in modes:
        def search(question: str, mode=mode) -> Sources:
            docs = search_documents(question, mode=mode, top_k=args.k)
            return [doc.metadata.get("source", "") for doc in docs]

        scores = evaluate(examples, search, args.k)
        label = mode or "configured"
        print(f"{label:>8}  hit@{args.k}={scores['hit@k']:.3f}  recall@{args.k}={scores['recall@k']:.3f}  mrr={scores['mrr']:.3f}")


if __name__ == "__main__":
    main()
