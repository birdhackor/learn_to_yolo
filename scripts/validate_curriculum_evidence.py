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
    extensions = []
    custom = json.loads((folder / "custom-data-learning.json").read_text())
    finite_json(custom)
    assert custom["code"]["script_sha256"] == hashlib.sha256(
        (ROOT / "scripts/run_custom_data_learning.py").read_bytes()).hexdigest()
    for filename, digest in custom["code"]["core_sha256"].items():
        assert digest == hashlib.sha256((ROOT / "miniyolo" / filename).read_bytes()).hexdigest(), filename
    assert custom["steps_completed"] == len(custom["loss_history"])
    assert custom["all_step_gradients_finite_nonzero"] and custom["weight_delta_l2"] > 0
    assert custom["loss_decreased_on_same_full_train_set"]
    assert all(custom["reload_checks"][key] for key in
               ("raw_predictions_exact", "decoded_predictions_exact", "optimizer_state_exact", "rng_state_exact"))
    extensions.append({"id": "custom-data-learning", "source_and_evidence_match": True})
    video = json.loads((folder / "video-file.json").read_text())
    finite_json(video)
    assert video["script_sha256"] == hashlib.sha256((ROOT / "scripts/verify_video_file.py").read_bytes()).hexdigest()
    for sid, digest in video["lesson_case_sha256"].items():
        assert digest == hashlib.sha256((ROOT / f"lesson_cases/{sid}.py").read_bytes()).hexdigest(), sid
    assert video["rgb_pixels_exact"] and video["detector_predictions_exact"] and video["overlays_exact"]
    assert all(video["capture_released"].values())
    assert len(video["tracking"]["ids"]) == video["video"]["frames"]
    extensions.append({"id": "video-file-and-tracking", "source_and_evidence_match": True})
    closure = json.loads((ROOT / "artifacts/checks/curriculum-closure.json").read_text())
    assert closure["required_issues_open"] == 0
    assert len(closure["closure_review_reports"]) == 4
    for review in closure["closure_review_reports"]:
        assert review["required_issues_open"] == 0
        assert review["report_sha256"] == hashlib.sha256((ROOT / review["report"]).read_bytes()).hexdigest()
    report = {"scope": "Current case SHA-256, notebook code/stdout, per-section JSON, index and page evidence markers; correctness reviews are separate",
              "lesson_count": len(results), "passed": len(results), "results": results,
              "closure_extensions": extensions}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(results)} current lesson cases and {len(extensions)} closure extensions: source and execution evidence consistent")


if __name__ == "__main__":
    main()
