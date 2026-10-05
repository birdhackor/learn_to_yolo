"""Regression checks for notebook output as saved on disk, without nbformat normalization."""

import json
from pathlib import Path

import pytest

from scripts.verify_release import notebook_stdout


@pytest.mark.parametrize("text", ["first line\nsecond line\n", ["first line\n", "second line\n"]])
def test_stdout_accepts_both_ipynb_multiline_representations(text):
    cell = {"outputs": [
        {"output_type": "stream", "name": "stdout", "text": text},
        {"output_type": "stream", "name": "stderr", "text": ["a warning\n"]},
        {"output_type": "display_data", "data": {"text/plain": "a figure"}},
        {"output_type": "stream", "name": "stdout", "text": "last line\n"},
    ]}
    assert notebook_stdout(cell) == "first line\nsecond line\nlast line\n"


def test_empty_cell_has_no_stdout():
    assert notebook_stdout({"outputs": []}) == ""


def test_every_published_notebooks_raw_json_stdout_matches_its_execution_record():
    root = Path(__file__).resolve().parents[1]
    sections = json.loads((root / "section-map.json").read_text())["sections"]
    notebooks = [root / section["notebook"] for section in sections if section["kind"] == "lesson"]
    assert len(notebooks) == 42
    for path in notebooks:
        notebook = json.loads(path.read_text())
        record = json.loads((root / "artifacts/checks/curriculum" / (path.stem + ".json")).read_text())
        assert notebook_stdout(notebook["cells"][-1]) == record["stdout"], path.name
