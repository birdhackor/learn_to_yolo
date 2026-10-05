"""Regression checks for notebook output as saved on disk, without nbformat normalization."""

import json
import os
from pathlib import Path
import subprocess
import sys

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
    assert notebooks
    assert {path.stem for path in notebooks} == {path.stem for path in (root / "lesson_cases").glob("*.py")}
    for path in notebooks:
        notebook = json.loads(path.read_text())
        record = json.loads((root / "artifacts/checks/curriculum" / (path.stem + ".json")).read_text())
        assert notebook_stdout(notebook["cells"][-1]) == record["stdout"], path.name


def test_vit_training_cell_runs_with_notebook_kernel_arguments():
    """A real cell has kernel arguments and no __file__; keep training and restore working."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/21-training.ipynb").read_text())
    source = "".join(notebook["cells"][-1]["source"])
    wrapper = ("import sys\n"
               "sys.argv = ['ipykernel_launcher.py', '-f', 'kernel.json']\n"
               "exec(compile(sys.stdin.read(), '<notebook-cell>', 'exec'), {'__name__': '__main__'})\n")
    execution = subprocess.run([sys.executable, "-c", wrapper], input=source, cwd=root,
                               env={**os.environ, "PYTHONPATH": str(root), "OMP_NUM_THREADS": "2"},
                               capture_output=True, text=True, timeout=30)
    assert execution.returncode == 0, execution.stderr
    report = json.loads(execution.stdout)
    assert report["steps"] == 60 and report["resume"]["model_max_abs_error"] == 0
    assert report["resume"]["optimizer_max_abs_error"] == 0
    assert report["source_hashes"]["lesson_cases/21-training.py"]
