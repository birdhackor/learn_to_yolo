"""Run every notebook's experiment cell on CPU and record actual outcomes."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", action="append")
    args = parser.parse_args()
    registry = json.loads((ROOT / "section-map.json").read_text())
    results = []
    for section in registry["sections"]:
        if section["kind"] != "lesson" or (args.section and section["id"] not in args.section):
            continue
        lesson_id = section["id"]
        notebook_path = ROOT / section["notebook"]
        notebook = json.loads(notebook_path.read_text())
        source = "".join(notebook["cells"][-1]["source"])
        case = ROOT / "lesson_cases" / f"{lesson_id}.py"
        assert source == case.read_text(), f"Notebook differs from tested source: {lesson_id}"
        assert any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(source))), lesson_id
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(ROOT), MPLBACKEND="Agg", OMP_NUM_THREADS="2")
        started = time.perf_counter()
        try:
            execution = subprocess.run([sys.executable, "-c", source], cwd=ROOT, env=environment,
                                       capture_output=True, text=True, timeout=90)
            result = {"id": lesson_id, "passed": execution.returncode == 0,
                      "exit_code": execution.returncode,
                      "seconds": round(time.perf_counter() - started, 3),
                      "stdout": execution.stdout, "stderr": execution.stderr}
        except subprocess.TimeoutExpired:
            result = {"id": lesson_id, "passed": False, "reason": "CPU experiment exceeded 90 seconds"}
        results.append(result)
        if result["passed"]:
            code_cell = notebook["cells"][-1]
            code_cell["execution_count"] = 1
            code_cell["outputs"] = [{"output_type": "stream", "name": "stdout",
                                     "text": result["stdout"].splitlines(keepends=True)}]
            notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        print(f"{lesson_id}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
        if not result["passed"]:
            print(result.get("stderr", result.get("reason")), flush=True)
    import torch
    report = {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu",
              "scope": "Notebook experiment cells; bootstrap validated separately from a fresh public clone. Not full GPU training.",
              "passed": sum(r["passed"] for r in results), "total": len(results), "results": results}
    destination = ROOT / "artifacts" / "checks" / "lesson-runtime.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not results or not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
