#!/usr/bin/env python3
"""Normalize final findings for arm-blinded grading; score without editing raw notes."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def normalize():
    runs = read(ROOT / "private/run-manifest.json")["cohorts"]
    material = {c["case_id"]: c for c in read(ROOT / "materials.json")["cases"]}
    gold = {c["case_id"]: c for c in read(ROOT / "private/gold.json")["cases"]}
    items = []
    identities = []
    rng = random.Random(2026100603)
    opaque = rng.sample(range(100000, 999999), 300)
    for run in runs:
        for reader in run["readers"]:
            rid = reader["reader_id"]
            state = read(ROOT / "state" / f"{rid}.json")
            if not state["finished"]:
                raise ValueError(f"incomplete reader {rid}")
            rows = [json.loads(l) for l in (ROOT / "raw" / f"{rid}.jsonl").read_text().splitlines()]
            final = []
            if reader["role"] == "ca":
                final = rows[-1]["record"]["case_results"]
            else:
                for row in rows:
                    final += [n for n in row["record"]["notes"] if "final_verdict" in n]
            if {n["case_id"] for n in final} != set(run["case_ids"]):
                raise ValueError("missing final case")
            for note in final:
                eid = f"E{opaque.pop()}"
                cid = note["case_id"]
                # Common schema avoids exposing treatment-specific inventory/audit fields.
                timeline = [{"unit_id": n["unit_id"], "issues": n["issues"]} for row in rows for n in row["record"].get("notes", []) if n["case_id"] == cid]
                item = {"evaluation_id": eid, "case_id": cid, "verdict": note["final_verdict"], "issues": note["issues"], "timeline_issues": timeline}
                items.append(item)
                identities.append({"evaluation_id": eid, "reader_id": rid, "role": reader["role"], "cohort": run["cohort"], "repeat": run["repeat"], "case_id": cid})
    rng.shuffle(items)
    dump(ROOT / "private/grading-identities.json", {"items": identities})
    case_order = list(material)
    rng.shuffle(case_order)
    for i in range(3):
        cids = set(case_order[i::3])
        selected = [it for it in items if it["case_id"] in cids]
        dump(ROOT / "grading" / f"bundle-{i + 1}.json", {"items": selected, "materials": [material[cid] for cid in sorted(cids)], "gold": [gold[cid] for cid in sorted(cids)]})
    print(f"Prepared {len(items)} normalized case judgments in 3 shuffled bundles.")


def score():
    identities = read(ROOT / "private/grading-identities.json")["items"]
    ids = {x["evaluation_id"]: x for x in identities}
    gold = {c["case_id"]: c for c in read(ROOT / "private/gold.json")["cases"]}
    mapping = {c["case_id"]: c for c in read(ROOT / "private/author-expectations.json")["cases"]}
    grades = []
    for path in sorted((ROOT / "grading").glob("grades-*.json")):
        grades += read(path)["grades"]
    if len(grades) != len(ids) or {g["evaluation_id"] for g in grades} != set(ids):
        raise ValueError("grading missing or duplicate judgments")
    originals = {it["evaluation_id"]: it for path in (ROOT / "grading").glob("bundle-*.json") for it in read(path)["items"]}
    for grade in grades:
        original = originals[grade["evaluation_id"]]
        if grade["case_id"] != original["case_id"] or grade["verdict"] != original["verdict"]:
            raise ValueError("grader changed case identity or original verdict")
        for field in ["target_hit", "severity_match", "off_target_necessary", "timely_target_hit", "premature_necessary_flag"]:
            if not isinstance(grade[field], bool):
                raise ValueError(f"non-boolean grade {field}")
        if not isinstance(grade["record_concerns"], list):
            raise ValueError("record_concerns must be a list")
        first_unit = grade["first_target_unit"]
        if first_unit is not None and first_unit not in {n["unit_id"] for n in original["timeline_issues"]}:
            raise ValueError("first_target_unit is not a recorded unit")
    by_cohort_case = defaultdict(dict)
    individual = []
    for grade in grades:
        ident = ids[grade["evaluation_id"]]
        row = {**ident, **grade, "kind": mapping[ident["case_id"]]["kind"], "topic_id": mapping[ident["case_id"]]["topic_id"], "gold_severity": gold[ident["case_id"]]["severity"]}
        individual.append(row)
        by_cohort_case[(ident["cohort"], ident["case_id"])][ident["role"]] = row
    pipelines = []
    for (cohort, cid), roles in sorted(by_cohort_case.items()):
        for arm, role in [("A", "a"), ("C_first", "c"), ("C", "ca")]:
            pipelines.append({**roles[role], "arm": arm})
        b1, b2 = roles["b1"], roles["b2"]
        verdicts = {b1["verdict"], b2["verdict"]}
        verdict = "needs_revision" if "needs_revision" in verdicts else "undetermined" if "undetermined" in verdicts else "optional_only" if "optional_only" in verdicts else "pass"
        target_hit = b1["target_hit"] or b2["target_hit"]
        # Both records are preserved; conflicting false positives are not erased.
        b = {**b1, "arm": "B", "verdict": verdict, "target_hit": target_hit, "timely_target_hit": b1["timely_target_hit"] or b2["timely_target_hit"], "severity_match": (b1["target_hit"] and b1["severity_match"]) or (b2["target_hit"] and b2["severity_match"]), "off_target_necessary": b1["off_target_necessary"] or b2["off_target_necessary"], "premature_necessary_flag": b1["premature_necessary_flag"] or b2["premature_necessary_flag"], "record_concerns": b1["record_concerns"] + b2["record_concerns"], "component_ids": [b1["evaluation_id"], b2["evaluation_id"]]}
        b["component_first_target_units"] = [b1["first_target_unit"], b2["first_target_unit"]]
        first_units = [r["first_target_unit"] for r in [b1, b2] if r["target_hit"] and r["first_target_unit"] is not None]
        b["first_target_unit"] = min(first_units, key=lambda u: int(u[1:])) if first_units else None
        pipelines.append(b)
    metrics = []
    for arm in ["A", "B", "C_first", "C"]:
        for scope in ["new_cases", "known_dino", "all"]:
            rows = [r for r in pipelines if r["arm"] == arm and (scope == "all" or (r["topic_id"] == "original_dino") == (scope == "known_dino"))]
            essential = [r for r in rows if r["gold_severity"] in ["burden", "blocker"]]
            controls = [r for r in rows if r["gold_severity"] == "none"]
            names = [r for r in rows if r["gold_severity"] == "optional"]
            metrics.append({"arm": arm, "scope": scope, "n": len(rows), "essential_n": len(essential), "essential_hits": sum(r["target_hit"] and r["severity_match"] for r in essential), "essential_timely_hits": sum(r["timely_target_hit"] for r in essential), "essential_wrong_pass": sum(r["verdict"] in ["pass", "optional_only"] for r in essential), "essential_undetermined": sum(r["verdict"] == "undetermined" for r in essential), "controls_n": len(controls), "controls_necessary_false_positives": sum(r["off_target_necessary"] for r in controls), "controls_clean_pass": sum(r["verdict"] == "pass" and not r["off_target_necessary"] for r in controls), "names_n": len(names), "names_wrong_escalation": sum(r["off_target_necessary"] for r in names), "names_optional_hit": sum(r["target_hit"] and r["severity_match"] for r in names), "undetermined": sum(r["verdict"] == "undetermined" for r in rows)})
    dump(ROOT / "results.json", {"metrics": metrics, "pipelines": pipelines, "individual": individual})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def integrity():
    frozen = read(ROOT / "freeze.json")
    bad = []
    for entry in frozen["files"]:
        if hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            bad.append(entry["path"])
    report = {"freeze_mismatches": bad, "readers": [], "claims": 0, "audited_claim_entries": 0, "chronology_errors": [], "audit_coverage": [], "workload_by_role": {}}
    workloads = defaultdict(Counter)
    for run in read(ROOT / "private/run-manifest.json")["cohorts"]:
        for reader in run["readers"]:
            rid = reader["reader_id"]
            rows = [json.loads(l) for l in (ROOT / "raw" / f"{rid}.jsonl").read_text().splitlines()]
            role = reader["role"]
            workloads[role]["contexts"] += 1
            workloads[role]["record_characters"] += sum(len(json.dumps(row["record"], ensure_ascii=False)) for row in rows)
            workloads[role]["source_note_count"] += sum(len(row["record"].get("notes", [])) for row in rows)
            workloads[role]["checkpoint_count"] += sum(bool(n.get("answers")) for row in rows for n in row["record"].get("notes", []))
            workloads[role]["active_segment_seconds"] += sum((datetime.fromisoformat(row["recorded_at_utc"]) - datetime.fromisoformat(row["revealed_at_utc"])).total_seconds() for row in rows)
            report["readers"].append({"reader_id": rid, "rounds": len(rows), "first_reveal": rows[0]["revealed_at_utc"], "last_record": rows[-1]["recorded_at_utc"], "declarations": [r["record"]["access_declaration"] for r in rows]})
            previous = frozen["frozen_at_utc"]
            for index, row in enumerate(rows):
                if row["round"] != index + 1 or row["revealed_at_utc"] < previous or row["recorded_at_utc"] < row["revealed_at_utc"]:
                    report["chronology_errors"].append({"reader_id": rid, "round": row["round"]})
                previous = row["recorded_at_utc"]
            for row in rows:
                for note in row["record"].get("notes", []):
                    report["claims"] += sum(len(a["claims"]) for a in note.get("answers", []))
                for note in row["record"].get("case_results", []):
                    report["audited_claim_entries"] += len(note["reader_claim_audit"])
            if role == "ca":
                primary = [json.loads(l) for l in (ROOT / "raw" / f"{run['cohort']}_c.jsonl").read_text().splitlines()]
                if rows[-1]["revealed_at_utc"] < primary[-1]["recorded_at_utc"] or rows[-1]["revealed_at_utc"] < rows[-2]["recorded_at_utc"]:
                    report["chronology_errors"].append({"reader_id": rid, "reason": "comparison occurred before own source inventory or primary freeze"})
                for result in rows[-1]["record"]["case_results"]:
                    cid = result["case_id"]
                    needed = {(n["unit_id"], a["q"], i) for row in primary for n in row["record"]["notes"] if n["case_id"] == cid for a in n.get("answers", []) for i, claim in enumerate(a["claims"])}
                    audits = [(a.get("unit_id"), a.get("q"), a.get("claim_index")) for a in result["reader_claim_audit"]]
                    found = set(audits)
                    report["audit_coverage"].append({"reader_id": rid, "case_id": cid, "required_claims": len(needed), "audit_entries": len(audits), "missing_claim_keys": sorted(needed - found), "extra_claim_keys": sorted(found - needed, key=str), "duplicate_keys": len(audits) - len(found)})
    report["workload_by_role"] = dict(workloads)
    dump(ROOT / "integrity.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "readers"}, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    {"normalize": normalize, "score": score, "integrity": integrity}[sys.argv[1]]()
