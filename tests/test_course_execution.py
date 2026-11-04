"""Exercise cache reuse, invalidation and failure handling without rerunning training in fixtures."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import ci_scope, course_execution as execution


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def course(tmp_path, monkeypatch):
    case = tmp_path / "lesson_cases/one.py"
    case.parent.mkdir()
    case.write_text("print(1)\nassert 1 == 1\n")
    notebook = {"metadata": {"kernelspec": {"name": "python3"}}, "cells": [
        {"cell_type": "markdown", "source": ["Old explanation"], "metadata": {}},
        {"cell_type": "code", "source": ["# setup"], "metadata": {}},
        {"cell_type": "code", "source": [case.read_text()], "metadata": {}}]}
    write_json(tmp_path / "notebooks/one.ipynb", notebook)
    write_json(tmp_path / "section-map.json", {"sections": [
        {"kind": "lesson", "id": "one", "notebook": "notebooks/one.ipynb"}]})
    monkeypatch.setattr(execution, "runtime", lambda: {"python": "test", "packages": [("torch", "fixed")]})
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        report = {"device": "cpu", "passed": 1, "total": 1, "results": [
            {"id": "one", "passed": True, "exit_code": 0, "stdout": "1\n", "stderr": ""}]}
        write_json(Path(command[-1]), report)

    monkeypatch.setattr(execution.subprocess, "run", run)
    return tmp_path, notebook, calls


def test_complete_bundle_reuses_results_after_prose_or_markdown_position_changes(course):
    root, notebook, calls = course
    receipt = execution.ensure(root, revision="original")
    notebook["cells"][0]["source"] = ["Current explanation"]
    notebook["cells"].append(notebook["cells"].pop(0))
    write_json(root / "notebooks/one.ipynb", notebook)
    (root / "README.md").write_text("New prose")
    (root / "data/downloads").mkdir(parents=True)
    (root / "data/downloads/optional.json").write_text("Downloaded optional data")
    assert execution.ensure(root, revision="new") == receipt
    assert receipt["executed_revision"] == "original" and len(calls) == 1


@pytest.mark.parametrize("change", ["program", "setup", "kernel", "shared-code", "manifest", "asset", "record", "requirements", "runtime"])
def test_execution_inputs_invalidate_the_complete_bundle(course, monkeypatch, change):
    root, notebook, calls = course
    before = execution.ensure(root)
    if change == "program":
        notebook["cells"][-1]["source"] = ["print(2)\nassert 2 == 2\n"]
        (root / "lesson_cases/one.py").write_text("".join(notebook["cells"][-1]["source"]))
    elif change == "setup":
        notebook["cells"][1]["source"] = ["# different setup"]
    elif change == "kernel":
        notebook["metadata"]["kernelspec"]["name"] = "other"
    elif change == "runtime":
        monkeypatch.setattr(execution, "runtime", lambda: {"packages": [("torch", "changed")]})
    else:
        names = {"shared-code": "miniyolo/helper.py", "manifest": "data/manifest.json",
                 "asset": "docs/assets/diagram.svg", "record": "artifacts/checks/result.json",
                 "requirements": "requirements-model.txt"}
        path = root / names[change]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed")
    write_json(root / "notebooks/one.ipynb", notebook)
    after = execution.ensure(root)
    assert after["key"] != before["key"] and len(calls) == 2


def test_unsynchronized_notebook_cannot_reuse_or_create_a_success(course):
    root, _, calls = course
    (root / "lesson_cases/one.py").write_text("print(99)")
    with pytest.raises(ValueError, match="differs"):
        execution.ensure(root)
    assert not calls and not (root / "outputs/lesson-execution/execution.json").exists()


@pytest.mark.parametrize("damage", ["missing-report", "corrupt-receipt", "tampered-report", "missing-row", "failed-row", "wrong-id"])
def test_damaged_or_incomplete_cache_is_rejected_and_failed_rerun_has_no_receipt(course, monkeypatch, damage):
    root, _, _ = course
    execution.ensure(root)
    output = root / "outputs/lesson-execution"
    if damage == "missing-report":
        (output / "report.json").unlink()
    elif damage == "corrupt-receipt":
        (output / "execution.json").write_text("corrupt")
    else:
        report = json.loads((output / "report.json").read_text())
        if damage == "missing-row":
            report["results"] = []
        elif damage == "failed-row":
            report["results"][0]["passed"] = False
        elif damage == "wrong-id":
            report["results"][0]["id"] = "another"
        else:
            report["results"][0]["stdout"] = "tampered"
        write_json(output / "report.json", report)
        if damage != "tampered-report":
            receipt = json.loads((output / "execution.json").read_text())
            receipt["report_sha256"] = execution.sha((output / "report.json").read_bytes())
            write_json(output / "execution.json", receipt)

    def fail(*args, **kwargs):
        assert not (output / "execution.json").exists()
        raise subprocess.CalledProcessError(1, "checker")

    monkeypatch.setattr(execution.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        execution.ensure(root)
    assert not (output / "execution.json").exists()


def test_site_environment_can_verify_matching_sources_without_model_packages(course, monkeypatch):
    root, _, _ = course
    expected = execution.ensure(root)
    monkeypatch.setattr(execution, "runtime", lambda: pytest.fail("Site verification must not inspect its model runtime"))
    assert execution.validate_bundle(root, root / "outputs/lesson-execution") == expected


def test_cache_cannot_remove_outputs_root_or_an_arbitrary_directory(course):
    root, _, _ = course
    for path in (root, root / "outputs", root.parent):
        with pytest.raises(ValueError, match="child directory"):
            execution.ensure(root, path)
    assert (root / "section-map.json").is_file()


@pytest.mark.parametrize("name", ["README.md", "docs/preparation/publish.md"])
def test_prose_changes_need_only_linux(name):
    assert not ci_scope.runtime_changed([name], lambda _: False)


@pytest.mark.parametrize("name", ["scripts/helper.py", "requirements-model.txt", ".github/workflows/pages.yml", "docs/assets/diagram.svg", "data/manifest.json"])
def test_runtime_changes_include_all_platforms(name):
    assert ci_scope.runtime_changed([name], lambda _: False)


def test_unknown_diff_runs_all_platforms(tmp_path, monkeypatch):
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("BASE_SHA", "missing")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

    def missing(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(ci_scope.subprocess, "check_output", missing)
    ci_scope.main()
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert len(json.loads(values["matrix"])) == 3
