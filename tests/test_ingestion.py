"""Tests for manifest handling and incremental-ingestion filtering."""

import pytest

pytest.importorskip("thesis_assistant.ingestion", reason="private module not present")

from thesis_assistant.ingestion import _filter_new_or_changed, load_manifest, save_manifest


def test_missing_manifest_is_empty(tmp_path):
    assert load_manifest(tmp_path / "missing.json") == {"pdfs": {}, "docx": {}, "urls": {}}


def test_corrupt_manifest_is_treated_as_first_run(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json")
    assert load_manifest(path)["pdfs"] == {}


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    save_manifest({"pdfs": {"a": "1"}, "docx": {}, "urls": {}}, path)
    assert load_manifest(path)["pdfs"] == {"a": "1"}


def test_filter_detects_new_unchanged_and_edited_files(tmp_path):
    file = tmp_path / "doc.pdf"
    file.write_bytes(b"v1")

    new, hashes, skipped = _filter_new_or_changed([file], {})
    assert new == [file] and skipped == 0

    new, _, skipped = _filter_new_or_changed([file], hashes)
    assert new == [] and skipped == 1

    file.write_bytes(b"v2")
    new, _, skipped = _filter_new_or_changed([file], hashes)
    assert new == [file] and skipped == 0
