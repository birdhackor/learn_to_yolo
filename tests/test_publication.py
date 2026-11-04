"""Formal publication metadata must identify clean sources and the actual reused execution."""
import pytest

from scripts import publication


@pytest.fixture
def prepared(monkeypatch):
    head = "a" * 40
    receipt = {"key": "execution-key", "executed_revision": "b" * 40, "executed_at": "2026-01-01T00:00:00+00:00"}
    monkeypatch.setattr(publication, "validate_bundle", lambda *args: receipt)
    monkeypatch.setattr(publication, "lessons", lambda *args: [{"id": "one"}])
    monkeypatch.setattr(publication.importlib.metadata, "version", lambda *args: "fixed-builder")
    monkeypatch.setattr(publication.subprocess, "check_output",
                        lambda command, **kwargs: head if command[1] == "rev-parse" else "")
    return head, receipt


def test_publication_and_execution_revisions_remain_distinct(tmp_path, prepared):
    head, receipt = prepared
    info = publication.build_info(tmp_path, tmp_path / "outputs/execution", head)
    assert info["revision"] == head
    assert info["execution_revision"] == receipt["executed_revision"]
    assert info["execution_date"] == receipt["executed_at"]
    assert info["execution_key"] == receipt["key"] and info["publication_ready"] is True


@pytest.mark.parametrize("revision", ["main", "short-sha", "c" * 40])
def test_mismatched_or_non_commit_publication_revision_is_rejected(tmp_path, prepared, revision):
    with pytest.raises(ValueError, match="SHA|differs"):
        publication.build_info(tmp_path, tmp_path / "outputs/execution", revision)


def test_uncommitted_sources_cannot_be_labeled_as_a_published_commit(tmp_path, monkeypatch, prepared):
    head, _ = prepared
    monkeypatch.setattr(publication.subprocess, "check_output",
                        lambda command, **kwargs: head if command[1] == "rev-parse" else " M README.md\n")
    with pytest.raises(ValueError, match="uncommitted"):
        publication.build_info(tmp_path, tmp_path / "outputs/execution", head)
