"""Answer-quality checks: citation accuracy and LLM-judged faithfulness.

Retrieval metrics (see evaluation.py) say whether the right chunks were found.
These checks say whether the *answer* used them honestly:

* ``citation_accuracy`` is deterministic. It extracts the sources an answer
  cites and checks they were really retrieved, which catches invented citations.
* ``judge_faithfulness`` asks an LLM judge whether every claim in the answer is
  supported by the retrieved context. The judge is injected as a callable, so
  it is easy to test and to swap models.
"""

import re
from typing import Callable, Dict, Iterable, List, Set

FAITHFULNESS_PROMPT = (
    "You are a strict evaluator. Given CONTEXT and an ANSWER, decide whether every factual "
    "claim in the ANSWER is supported by the CONTEXT. Reply with JSON only: "
    '{{"faithful": true|false, "unsupported_claims": ["..."]}}.\n\n'
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}"
)


def cited_sources(answer: str, known_sources: Iterable[str]) -> Set[str]:
    """Known source names that the answer mentions (case-insensitive substring match)."""
    lowered = answer.casefold()
    return {s for s in known_sources if s and s.casefold() in lowered}


_FILE_LIKE = re.compile(r"[^\s,;:\"'«»]+\.(?:pdf|docx)", re.IGNORECASE)


def mentioned_files(answer: str) -> Set[str]:
    """File-like tokens (``*.pdf`` / ``*.docx``) in the answer.

    Tokens stop at whitespace, so a name containing spaces yields only its last
    word (``Rules 2022.pdf`` -> ``2022.pdf``). Matching therefore uses suffixes.
    """
    return {m.group(0).strip() for m in _FILE_LIKE.finditer(answer)}


def citation_accuracy(answer: str, retrieved_sources: Iterable[str]) -> Dict[str, float]:
    """Check that file names cited in the answer were actually retrieved.

    Returns ``precision`` (share of cited files that were retrieved; 1.0 when
    nothing is cited) and ``n_cited``. A value below 1.0 means the model
    referenced a document it never saw, i.e. a hallucinated citation.
    """
    retrieved = {s.casefold() for s in retrieved_sources}
    cited = mentioned_files(answer)
    if not cited:
        return {"precision": 1.0, "n_cited": 0}
    valid = sum(1 for c in cited if any(r.endswith(c.casefold()) for r in retrieved))
    return {"precision": valid / len(cited), "n_cited": len(cited)}


def judge_faithfulness(
    answer: str,
    context_chunks: List[str],
    judge: Callable[[str], str],
) -> Dict[str, object]:
    """Ask ``judge(prompt) -> str`` whether the answer is grounded in the context.

    Returns ``{"faithful": bool, "unsupported_claims": [...]}``. Unparseable
    judge output is reported as not faithful so failures are never silent.
    """
    import json

    prompt = FAITHFULNESS_PROMPT.format(context="\n---\n".join(context_chunks), answer=answer)
    raw = judge(prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
        return {
            "faithful": bool(data["faithful"]),
            "unsupported_claims": list(data.get("unsupported_claims", [])),
        }
    except (KeyError, ValueError, TypeError):
        return {"faithful": False, "unsupported_claims": ["judge output could not be parsed"]}
