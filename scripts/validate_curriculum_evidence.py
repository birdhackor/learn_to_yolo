"""Check that every saved execution record belongs to the current code.

Each record names, by SHA-256, the files whose code produced it: the lesson case or script and every
repository module they import (miniyolo, scripts). A record passes only if all of them are unchanged,
its notebook and page show exactly its output, and its own consistency checks hold. This checks
consistency, not the scientific validity of a conclusion; independent reviews are recorded separately.
Runs without PyTorch.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from evidence_records import CPU_RECORDS, GPU_RECORDS, is_current, provenance

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


def check_machine(record, name):
    machine = record.get("machine") or record.get("hardware") or record.get("runtime", {}).get("machine")
    assert isinstance(machine, dict) and machine.get("cpu") and machine.get("os", machine.get("python")), \
        f"{name}: record does not say which machine produced it"


def check_gpu_records() -> list[dict]:
    checked = []
    for name, (entries, workflow) in GPU_RECORDS.items():
        record = json.loads((ROOT / name).read_text())
        finite_json(record)
        assert is_current(name), f"{name}: a module it ran changed; rerun .github/workflows/{workflow}"
        assert record["status"] == "passed" and record["gpu_calls_finished"], name
        assert record["modal_stop_confirmation"]["verified"], f"{name}: the Modal app was not confirmed stopped"
        checked.append({"record": name, "bound_to": list(entries), "current": True, "workflow": workflow})
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--scope", choices=("all", "cpu", "gpu"), default="all",
                        help="cpu: lesson and CPU records only; gpu: GPU records only (for the two regeneration paths)")
    args = parser.parse_args()
    if args.scope == "gpu":
        gpu = check_gpu_records()
        print(f"{len(gpu)} GPU records match the current code")
        return
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
        assert record["dependencies_sha256"] == provenance.repo_dependencies(case), \
            f"{sid}: the lesson case or a module it imports changed after the record was made"
        check_machine(record, sid)
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
        assert record["machine"]["cpu"] in page, f"{sid}: the page does not name the recorded machine"
        results.append({"id": sid, "case_sha256": digest, "evidence_matches": True})
    supplementary = []
    for name, entries in CPU_RECORDS.items():
        record = json.loads((ROOT / name).read_text())
        finite_json(record)
        assert is_current(name), f"{name}: its script, lesson case or an imported module changed"
        check_machine(record, name)
        supplementary.append({"record": name, "bound_to": list(entries), "current": True})
    if args.scope == "all":
        supplementary += check_gpu_records()
    custom = json.loads((folder / "custom-data-learning.json").read_text())
    assert custom["steps_completed"] == len(custom["loss_history"])
    assert custom["all_step_gradients_finite_nonzero"] and custom["weight_delta_l2"] > 0
    assert custom["loss_decreased_on_same_full_train_set"]
    assert all(custom["reload_checks"][key] for key in
               ("raw_predictions_exact", "decoded_predictions_exact", "optimizer_state_exact", "rng_state_exact"))
    video = json.loads((folder / "video-file.json").read_text())
    assert video["rgb_pixels_exact"] and video["detector_predictions_exact"] and video["overlays_exact"]
    assert all(video["capture_released"].values())
    assert len(video["tracking"]["ids"]) == video["video"]["frames"]
    fashion = json.loads((folder / "fashion-mnist-learning.json").read_text())
    assert (fashion["steps"], fashion["train_samples"], fashion["validation_samples"], fashion["test_samples"]) == (40, 64, 128, 128)
    assert fashion["weights_changed"] and len(fashion["loss_history"]) == fashion["steps"]
    report = {"scope": "Every record's SHA-256 binding to its code and imported repository modules, notebook stdout, "
                       "page evidence blocks and recorded machine; correctness reviews are separate",
              "lesson_count": len(results), "passed": len(results), "results": results,
              "supplementary_records": supplementary}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(results)} lesson records and {len(supplementary)} supplementary records match the current code"
          + ("" if args.scope == "all" else " (GPU records not checked)"))


if __name__ == "__main__":
    main()
