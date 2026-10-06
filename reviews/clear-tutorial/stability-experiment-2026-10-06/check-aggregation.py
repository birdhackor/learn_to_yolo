#!/usr/bin/env python3
"""Synthetic checks of aggregation; never changes experimental judgments."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value))


def main():
    scratch_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parents[3] / "work" / "stability-experiment-2026-10-06"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tutorial-score-", dir=scratch_root) as directory:
        scratch = Path(directory)
        (scratch / "private").mkdir()
        (scratch / "grading").mkdir()
        shutil.copyfile(ROOT / "collect.py", scratch / "collect.py")
        cases = [{"case_id": "test_gap", "severity": "blocker"}, {"case_id": "test_complete", "severity": "none"}, {"case_id": "test_name", "severity": "optional"}]
        write(scratch / "private/gold.json", {"cases": cases})
        write(scratch / "private/author-expectations.json", {"cases": [{"case_id": c["case_id"], "kind": "synthetic", "topic_id": c["case_id"]} for c in cases]})
        identities, grades, items = [], [], []
        for case in cases:
            for role in ["a", "b1", "b2", "c", "ca"]:
                eid = f"{case['case_id']}_{role}"
                identities.append({"evaluation_id": eid, "reader_id": role, "role": role, "cohort": "synthetic", "repeat": 1, "case_id": case["case_id"]})
                grade = {"evaluation_id": eid, "case_id": case["case_id"], "verdict": "pass", "target_hit": False, "severity_match": False, "off_target_necessary": False, "timely_target_hit": False, "first_target_unit": None, "premature_necessary_flag": False, "record_concerns": [], "reason": "synthetic aggregation check"}
                if case["case_id"] == "test_gap":
                    if role in ["b1", "b2", "ca"]:
                        grade.update(verdict="needs_revision", target_hit=True, severity_match=role != "b1", timely_target_hit=role != "b1", off_target_necessary=role == "b1", first_target_unit="u2" if role != "b1" else "u3")
                    if role == "c":
                        grade["verdict"] = "undetermined"
                if case["case_id"] == "test_complete" and role == "b2":
                    grade.update(verdict="needs_revision", off_target_necessary=True)
                if case["case_id"] == "test_name" and role == "b2":
                    grade.update(verdict="optional_only", target_hit=True, severity_match=True, first_target_unit="u1")
                grades.append(grade)
                items.append({"evaluation_id": eid, "case_id": grade["case_id"], "verdict": grade["verdict"], "timeline_issues": [{"unit_id": u, "issues": []} for u in ["u1", "u2", "u3"]]})
        write(scratch / "private/grading-identities.json", {"items": identities})
        write(scratch / "grading/grades-1.json", {"grades": grades})
        write(scratch / "grading/bundle-1.json", {"items": items})
        subprocess.run([sys.executable, str(scratch / "collect.py"), "score"], check=True, capture_output=True, text=True)
        result = json.loads((scratch / "results.json").read_text())
        metrics = {(m["arm"], m["scope"]): m for m in result["metrics"]}
        assert metrics[("A", "all")]["essential_wrong_pass"] == 1
        assert metrics[("C_first", "all")]["essential_undetermined"] == 1
        assert metrics[("B", "all")]["essential_hits"] == 1
        assert metrics[("B", "all")]["essential_timely_hits"] == 1
        assert metrics[("B", "all")]["controls_necessary_false_positives"] == 1
        assert metrics[("B", "all")]["names_optional_hit"] == 1
        b = next(row for row in result["pipelines"] if row["arm"] == "B" and row["case_id"] == "test_gap")
        assert b["off_target_necessary"] and len(b["component_ids"]) == 2
        assert b["first_target_unit"] == "u2" and b["component_first_target_units"] == ["u3", "u2"]
        grades[0]["verdict"] = "needs_revision"
        write(scratch / "grading/grades-1.json", {"grades": grades})
        rejection = subprocess.run([sys.executable, str(scratch / "collect.py"), "score"], capture_output=True, text=True)
        assert rejection.returncode != 0 and "grader changed case identity or original verdict" in rejection.stderr
    report = {"status": "passed", "synthetic_only": True, "checks": ["B preserves both hit and false positive", "B preserves component timing and selects earliest target", "optional, wrong pass and undetermined count separately", "altered original verdict rejected"], "limitations": "Aggregation checks only; reader semantics and experimental outcomes are not tested."}
    (ROOT / "scoring-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
