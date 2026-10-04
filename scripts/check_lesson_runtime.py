"""Run every notebook's experiment cell on this computer's CPU and compare with the recorded run.

Nothing in the repository changes: the report goes to artifacts/runs/lesson-runtime.json (or --report).
Timing lines and low-order digits can differ from the recorded run on another machine; the report
says, per section, whether the printed output is identical to artifacts/checks/curriculum/<section>.json.
"""

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
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", action="append", help="check only this section (repeatable)")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/runs/lesson-runtime.json")
    args = parser.parse_args()
    registry = json.loads((ROOT / "section-map.json").read_text())
    results = []
    for section in registry["sections"]:
        if section["kind"] != "lesson" or (args.section and section["id"] not in args.section):
            continue
        lesson_id = section["id"]
        notebook = json.loads((ROOT / section["notebook"]).read_text())
        source = "".join(notebook["cells"][-1]["source"])
        case = ROOT / "lesson_cases" / f"{lesson_id}.py"
        assert source == case.read_text(), f"Notebook differs from the lesson program: {lesson_id}"
        assert any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(source))), lesson_id
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(ROOT), MPLBACKEND="Agg", OMP_NUM_THREADS="2")
        started = time.perf_counter()
        try:
            execution = subprocess.run([sys.executable, "-c", source], cwd=ROOT, env=environment,
                                       capture_output=True, text=True, timeout=120)
            result = {"id": lesson_id, "passed": execution.returncode == 0,
                      "exit_code": execution.returncode,
                      "seconds": round(time.perf_counter() - started, 3),
                      "stdout": execution.stdout, "stderr": execution.stderr}
        except subprocess.TimeoutExpired:
            result = {"id": lesson_id, "passed": False, "reason": "CPU experiment exceeded 120 seconds"}
        recorded = ROOT / "artifacts/checks/curriculum" / f"{lesson_id}.json"
        if result["passed"] and recorded.is_file():
            result["same_output_as_record"] = result["stdout"] == json.loads(recorded.read_text())["stdout"]
        results.append(result)
        note = "" if not result["passed"] else (
            "; output identical to the recorded run" if result.get("same_output_as_record")
            else "; output differs from the recorded run (timing or machine-dependent digits)")
        print(f"{lesson_id}: {'PASS' if result['passed'] else 'FAIL'}{note}", flush=True)
        if not result["passed"]:
            print(result.get("stderr", result.get("reason")), flush=True)
    from miniyolo.provenance import machine_info
    report = {"machine": machine_info(2), "device": "cpu",
              "scope": "Notebook experiment cells run on this computer; compared with the recorded run, not a GPU or full-training test.",
              "passed": sum(r["passed"] for r in results), "total": len(results), "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {args.report}")
    if not results or not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
