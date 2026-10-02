"""Check that saved execution evidence belongs to the current lesson code.

This checks consistency, not the scientific validity of a conclusion. Independent
reader and source reviews remain necessary and are recorded separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def finite_json(value):
    if isinstance(value, float):
        assert math.isfinite(value), "Nonfinite number in execution evidence"
    elif isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    lessons = [s for s in json.loads((ROOT / "section-map.json").read_text())["sections"]
               if s["kind"] == "lesson"]
    folder = ROOT / "artifacts/checks/curriculum"
    index = json.loads((folder / "index.json").read_text())
    assert index["total"] == index["passed"] == len(lessons)
    indexed = {row["id"]: row for row in index["results"]}
    assert len(indexed) == len(lessons)
    results = []
    for lesson in lessons:
        sid = lesson["id"]
        case = ROOT / f"lesson_cases/{sid}.py"
        record = json.loads((folder / f"{sid}.json").read_text())
        finite_json(record)
        digest = hashlib.sha256(case.read_bytes()).hexdigest()
        assert record["id"] == sid and record["passed"] is True and record["exit_code"] == 0, sid
        assert record["case_sha256"] == digest == indexed[sid]["case_sha256"], sid
        assert record["executed_at_utc"] == indexed[sid]["executed_at_utc"], sid
        assert datetime.fromisoformat(record["executed_at_utc"]).utcoffset().total_seconds() == 0, sid
        assert isinstance(record["torch"], str) and record["process_wall_seconds"] >= 0, sid
        notebook = json.loads((ROOT / lesson["notebook"]).read_text())
        cell = notebook["cells"][-1]
        assert cell["cell_type"] == "code" and "".join(cell["source"]) == case.read_text(), sid
        assert not any(o["output_type"] == "error" for o in cell.get("outputs", [])), sid
        stdout = "".join("".join(o["text"]) for o in cell.get("outputs", [])
                         if o["output_type"] == "stream" and o["name"] == "stdout")
        assert stdout == record["stdout"] and stdout.strip(), sid
        page = (ROOT / lesson["page"]).read_text()
        assert page.count("<!-- curriculum-evidence:start -->") == 1, sid
        assert page.count("<!-- curriculum-evidence:end -->") == 1, sid
        assert record["executed_at_utc"][:10] in page and record["torch"] in page, sid
        results.append({"id": sid, "case_sha256": digest, "evidence_matches": True})
    report = {"scope": "Current case SHA-256, notebook code/stdout, per-section JSON, index and page evidence markers; correctness reviews are separate",
              "lesson_count": len(results), "passed": len(results), "results": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(results)} current lesson cases, notebook outputs and execution records: consistent")


if __name__ == "__main__":
    main()
