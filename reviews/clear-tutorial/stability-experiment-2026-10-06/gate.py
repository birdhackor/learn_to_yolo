#!/usr/bin/env python3
"""Instruction-gated disclosure; NOT a filesystem security boundary."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def visible(run, state, case_id):
    case = next(c for c in run["cases"] if c["case_id"] == case_id)
    result = {"p": case["prerequisite"]["text"]}
    for i, unit in enumerate(case["units"]):
        if i <= state["round"]:
            result[f"u{i + 1}"] = unit["text"]
    return result


def validate_refs(obj, sources):
    if isinstance(obj, dict):
        if "source" in obj and "quote" in obj:
            if obj["source"] not in sources:
                raise ValueError("reference to an unseen source")
            quote = obj["quote"]
            if not quote or quote not in sources[obj["source"]]:
                raise ValueError("quotation must be a literal substring of visible source")
        for value in obj.values():
            validate_refs(value, sources)
    elif isinstance(obj, list):
        for value in obj:
            validate_refs(value, sources)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["next", "record"])
    parser.add_argument("reader_id")
    parser.add_argument("file", nargs="?")
    args = parser.parse_args()
    run = read(ROOT / "packets" / f"{args.reader_id}.json")
    state_path = ROOT / "state" / f"{args.reader_id}.json"
    state = read(state_path) if state_path.exists() else {"round": 0, "awaiting": False, "finished": False}
    if state["finished"]:
        emit({"done": True, "raw_record": f"raw/{args.reader_id}.jsonl"})
        return
    r = state["round"]
    is_audit = run["role"] == "auditor"
    source_rounds = max(len(c["units"]) for c in run["cases"])
    comparison = is_audit and r == source_rounds
    if args.action == "next":
        if state["awaiting"]:
            raise ValueError("record the current batch before requesting more")
        if comparison:
            primary_path = ROOT / "raw" / f"{run['primary_id']}.jsonl"
            primary_state = read(ROOT / "state" / f"{run['primary_id']}.json")
            if not primary_state["finished"]:
                emit({"pending_primary": True})
                return
            rows = [json.loads(line) for line in primary_path.read_text().splitlines()]
            batch = {"stage": "compare_after_independent_source_inventory", "primary_records": rows}
        else:
            batch = {"stage": "independent_source" if is_audit else "first_read", "round": r + 1, "cases": []}
            for case in run["cases"]:
                if r >= len(case["units"]):
                    continue
                u = case["units"][r]
                item = {"case_id": case["case_id"], "unit_id": f"u{r + 1}", "text": u["text"], "checkpoint": u.get("checkpoint", False), "page_end": r == len(case["units"]) - 1, "images": u.get("images", [])}
                if r == 0:
                    item["prerequisite"] = case["prerequisite"]
                    item["title"] = case["title"]
                batch["cases"].append(item)
        state["awaiting"] = True
        state["last_batch"] = batch
        state["revealed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(state_path, state)
        emit(batch)
        return
    if not state["awaiting"]:
        raise ValueError("request a source batch first")
    obj = read(Path(args.file))
    if comparison:
        notes = obj["case_results"]
        if {n["case_id"] for n in notes} != {c["case_id"] for c in run["cases"]}:
            raise ValueError("audit must resolve every case")
        for note in notes:
            for key in ["final_verdict", "issues", "reader_claim_audit", "unknown_audit", "explanation"]:
                if key not in note:
                    raise ValueError(f"missing auditor field {key}")
            validate_refs(note, visible(run, state, note["case_id"]))
    else:
        notes = obj["notes"]
        expected = {c["case_id"]: c for c in state["last_batch"]["cases"]}
        if {n["case_id"] for n in notes} != set(expected):
            raise ValueError("one note required for each visible case")
        for note in notes:
            item = expected[note["case_id"]]
            for key in ["unit_id", "understanding", "issues", "unknowns", "visual_status"]:
                if key not in note:
                    raise ValueError(f"missing note field {key}")
            if note["unit_id"] != item["unit_id"]:
                raise ValueError("incorrect unit id")
            if is_audit:
                for key in ["independent_inventory", "independent_judgment"]:
                    if key not in note:
                        raise ValueError(f"missing independent auditor field {key}")
            elif item["checkpoint"]:
                answers = note.get("answers", [])
                if [a["q"] for a in answers] != [1, 2, 3, 4]:
                    raise ValueError("four answers required at checkpoint")
                for answer in answers:
                    if not all(k in answer for k in ["focus", "answer", "claims"]):
                        raise ValueError("focus, answer, claims are required")
                if run["structured"] and not all(k in note for k in ["concept_inventory", "unknown_dispositions"]):
                    raise ValueError("structured reader inventory/dispositions missing")
                if run["structured"]:
                    for unknown in note["unknowns"]:
                        dispositions = [d for d in note["unknown_dispositions"] if d.get("missing") == unknown.get("missing")]
                        if not dispositions and note.get("final_verdict") != "undetermined":
                            raise ValueError("every listed unknown needs an explicit disposition, or an undetermined verdict")
                    for disposition in note["unknown_dispositions"]:
                        if not all(k in disposition for k in ["missing", "needed_now", "reason", "evidence", "disposition"]):
                            raise ValueError("unknown disposition requires missing/need/reason/evidence/action")
                        if disposition["disposition"] not in ["already_taught", "reasonable_deferral", "issue"]:
                            raise ValueError("invalid unknown disposition")
            if item["page_end"] and not is_audit and "final_verdict" not in note:
                raise ValueError("page end needs a final verdict")
            validate_refs(note, visible(run, state, note["case_id"]))
    for note in notes:
        for issue in note.get("issues", []):
            if not all(k in issue for k in ["severity", "location", "already_taught", "missing", "needed_now", "evidence"]):
                raise ValueError("issue requires severity, location, known portion, missing portion, need, evidence")
            if issue["severity"] not in ["blocker", "burden", "optional"]:
                raise ValueError("invalid severity")
        if "final_verdict" in note and note["final_verdict"] not in ["pass", "optional_only", "needs_revision", "undetermined"]:
            raise ValueError("invalid verdict")
    if "access_declaration" not in obj:
        raise ValueError("access declaration required")
    raw_path = ROOT / "raw" / f"{args.reader_id}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"reader_id": args.reader_id, "revealed_at_utc": state["revealed_at_utc"], "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "round": r + 1, "stage": state["last_batch"]["stage"], "visible_batch_sha256": hashlib.sha256(json.dumps(state["last_batch"], ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "record": obj}
    with raw_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    state["round"] += 1
    state["awaiting"] = False
    state.pop("last_batch", None)
    state.pop("revealed_at_utc", None)
    state["finished"] = state["round"] >= source_rounds + int(is_audit)
    write(state_path, state)
    emit({"recorded": True, "done": state["finished"]})


if __name__ == "__main__":
    main()
