from thesis_assistant.evaluation import evaluate, hit_at_k, recall_at_k, reciprocal_rank


def test_metrics_on_known_ranking():
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, ["b", "z"], k=3) == 0.5
    assert recall_at_k(retrieved, ["b"], k=1) == 0.0
    assert hit_at_k(retrieved, ["c"], k=3) == 1.0
    assert reciprocal_rank(retrieved, ["b"]) == 0.5
    assert reciprocal_rank(retrieved, ["z"]) == 0.0


def test_evaluate_averages_over_examples():
    examples = [
        {"question": "q1", "relevant_sources": ["a"]},
        {"question": "q2", "relevant_sources": ["x"]},
    ]
    scores = evaluate(examples, lambda q: ["a", "b"], k=2)
    assert scores["hit@k"] == 0.5 and scores["mrr"] == 0.5 and scores["n"] == 2


def test_evaluate_handles_empty_set():
    assert evaluate([], lambda q: [], k=3)["n"] == 0
