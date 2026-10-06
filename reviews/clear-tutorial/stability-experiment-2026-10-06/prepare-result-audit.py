#!/usr/bin/env python3
"""Create the deterministic post-grading audit packet specified in the audit plan."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = "result-audit-2026100604"


def read(path):
    return json.loads(path.read_text())


def rank(value):
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def rows(reader_id):
    return [json.loads(line) for line in (ROOT / "raw" / f"{reader_id}.jsonl").read_text().splitlines()]


def main():
    identities = {item["evaluation_id"]: item for item in read(ROOT / "private/grading-identities.json")["items"]}
    sources = {case["case_id"]: case for case in read(ROOT / "materials.json")["cases"]}
    gold = {case["case_id"]: case for case in read(ROOT / "private/gold.json")["cases"]}
    items = {item["evaluation_id"]: item for path in (ROOT / "grading").glob("bundle-*.json") for item in read(path)["items"]}
    grades = {grade["evaluation_id"]: grade for path in (ROOT / "grading").glob("grades-*.json") for grade in read(path)["grades"]}
    if set(items) != set(identities) or set(grades) != set(items):
        raise ValueError("incomplete grading or identities")
    essential = defaultdict(list)
    none, optional = [], []
    flagged = set()
    for eid, item in items.items():
        severity = gold[item["case_id"]]["severity"]
        grade = grades[eid]
        if severity in ["blocker", "burden"]:
            essential[(item["case_id"], identities[eid]["role"])].append(eid)
            if not grade["target_hit"] or not grade["severity_match"]:
                flagged.add(eid)
        else:
            (none if severity == "none" else optional).append(eid)
        if grade["off_target_necessary"] or grade["verdict"] == "undetermined" or grade["record_concerns"]:
            flagged.add(eid)
    baseline = {min(eids, key=rank) for eids in essential.values()}
    baseline.update(sorted(none, key=rank)[:5])
    baseline.update(sorted(optional, key=rank)[:5])
    selected = baseline | flagged
    judgment_packet = [{"item": items[eid], "grade": grades[eid], "selection": "baseline+flag" if eid in baseline and eid in flagged else "baseline" if eid in baseline else "flag"} for eid in sorted(selected, key=rank)]
    claim_packet, unknown_packet = [], []
    for cohort in read(ROOT / "private/run-manifest.json")["cohorts"]:
        rid = f"{cohort['cohort']}_ca"
        primary = rows(f"{cohort['cohort']}_c")
        primary_notes = {(note["case_id"], note["unit_id"]): note for row in primary for note in row["record"]["notes"]}
        claims = []
        for case in rows(rid)[-1]["record"]["case_results"]:
            cid = case["case_id"]
            for audit in case["reader_claim_audit"]:
                key = f"{rid}:{cid}:{audit['unit_id']}:{audit['q']}:{audit['claim_index']}"
                note = primary_notes[(cid, audit["unit_id"])]
                answer = next(answer for answer in note["answers"] if answer["q"] == audit["q"])
                claims.append({"audit_id": rank(key)[:12], "case_id": cid, "unit_id": audit["unit_id"], "q": audit["q"], "claim_index": audit["claim_index"], "claim": answer["claims"][audit["claim_index"]], "audit": audit})
            for index, audit in enumerate(case["unknown_audit"]):
                if audit.get("reader_disposition_valid") is False:
                    unit = audit["unit_id"]
                    note = primary_notes.get((cid, unit), {})
                    unknown_packet.append({"audit_id": rank(f"{rid}:{cid}:unknown:{index}")[:12], "case_id": cid, "unit_id": unit, "audit": audit, "primary_unknowns": note.get("unknowns", []), "primary_dispositions": note.get("unknown_dispositions", []), "primary_unknown_claims": [{"q": answer["q"], "claim": claim} for answer in note.get("answers", []) for claim in answer["claims"] if claim["state"] == "unknown"]})
        required = {claim["audit_id"] for claim in sorted(claims, key=lambda claim: rank(claim["audit_id"]))[:4]}
        claim_packet += [claim for claim in claims if claim["audit_id"] in required or claim["audit"].get("supported") is False]
    packet = {"selection_rule": "result-audit-plan.md; SHA256 deterministic baseline plus all flagged judgments and unsupported/invalid C audit items", "baseline_judgments": len(baseline), "flagged_judgments": len(flagged), "selected_judgments": len(selected), "judgments": judgment_packet, "claim_audits": claim_packet, "unknown_audits": unknown_packet, "materials": list(sources.values()), "gold": list(gold.values())}
    destination = ROOT / "result-audit/packet.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"baseline": len(baseline), "flags": len(flagged), "selected": len(selected), "claims": len(claim_packet), "unknowns": len(unknown_packet)}))


if __name__ == "__main__":
    main()
