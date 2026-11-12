from thesis_assistant.answer_eval import citation_accuracy, judge_faithfulness, mentioned_files


def test_mentioned_files_finds_pdf_and_docx():
    answer = "Bax: Rules 2022.pdf və template (final).docx səhifə 3."
    assert mentioned_files(answer) == {"2022.pdf", "(final).docx"}


def test_citation_accuracy_flags_invented_sources():
    result = citation_accuracy("Bax rules.pdf və fake.pdf", ["rules.pdf"])
    assert result == {"precision": 0.5, "n_cited": 2}


def test_citation_accuracy_without_citations_is_neutral():
    assert citation_accuracy("no files here", ["a.pdf"]) == {"precision": 1.0, "n_cited": 0}


def test_judge_faithfulness_parses_json():
    judge = lambda prompt: 'Result: {"faithful": false, "unsupported_claims": ["deadline is May 1"]}'  # noqa: E731
    result = judge_faithfulness("deadline is May 1", ["no dates here"], judge)
    assert result == {"faithful": False, "unsupported_claims": ["deadline is May 1"]}


def test_judge_faithfulness_treats_garbage_as_unfaithful():
    result = judge_faithfulness("a", ["b"], lambda p: "I cannot say")
    assert result["faithful"] is False


def test_citation_accuracy_matches_names_with_spaces():
    result = citation_accuracy("Bax Rules 2022.pdf", ["Rules 2022.pdf"])
    assert result["precision"] == 1.0
