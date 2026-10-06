#!/usr/bin/env python3
"""Freeze anonymous materials, then create matched instruction-gated runs."""
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FROZEN = REPO / "reviews/clear-tutorial/full-review-2026-10-06/frozen"


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def prior_with_quotes(prior):
    prior = dict(prior)
    prior["text"] += "\n\n實際前文來源：\n" + "\n".join(f"{s['path']} {s['lines_or_anchor']}：{s['quote']}" for s in prior["source_refs"])
    return prior


def dino_material():
    refs = []
    selections = {"docs/index.md": [25], "docs/learning-path.md": [23], "docs/lessons/21-patches.md": [5, 11], "docs/lessons/21-transformer.md": [], "docs/lessons/21-training.md": [5, 23]}
    # Select by literal content, not fragile pre-known line numbers.
    needles = {"docs/index.md": ["學過 CNN、ResNet"], "docs/learning-path.md": ["- **ViT／DINO 選讀支線"], "docs/lessons/21-patches.md": ["[第 15.1 節]", "沿用第 1 章的答案規則"], "docs/lessons/21-transformer.md": ["紅矩形回答 0", "CLS", "分類 head"], "docs/lessons/21-training.md": ["[上一節]", "這次不是只更新 logits"]}
    for path in selections:
        lines = (FROZEN / path).read_text().split("<!-- curriculum-evidence:start -->")[0].splitlines()
        chosen = []
        for i, line in enumerate(lines):
            if line and any(n in line for n in needles[path]) and not line.lstrip().startswith(("![", "[上一節：", "#", "def ", "return ", "self.", "model.", "```")):
                chosen.append((i + 1, line))
        for line_no, quote in chosen:
            refs.append({"path": path, "lines_or_anchor": f"L{line_no}", "quote": quote})
    prior = prior_with_quotes({"text": "這次局部閱讀實際提供下列原版導覽與前文節錄；其他完整頁面沒有揭露。基本 Python、NN/CNN 及數學為指定背景。不得使用尚未提供的22.2以後內容。", "source_refs": refs})
    original = (FROZEN / "docs/lessons/22-views.md").read_text().splitlines()
    ranges = [(1, 6), (7, 20), (21, 28), (29, 38), (39, 55), (56, 70)]
    units = []
    for i, (start, end) in enumerate(ranges):
        u = {"text": "\n".join(original[start - 1:end]), "checkpoint": i in [1, 3, 5], "source_ref": {"path": "docs/lessons/22-views.md", "start": start, "end": end}, "images": []}
        if i == 1:
            origin = REPO / "reviews/clear-tutorial/full-review-2026-10-06/previews/docs__assets__diagrams__22-views.svg.png"
            dst = ROOT / "images" / "views.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, dst)
            u["images"] = [{"path": str(dst), "status": "原始SVG預覽；網站未驗證"}]
        units.append(u)
    return {"title": "22.1 沒有紅藍標籤，同一張圖的兩個 view 能教什麼？", "prerequisite": prior, "units": units}


def materials():
    candidate = json.loads((ROOT / "private/corpus-candidate.json").read_text())
    rng = random.Random(2026100601)
    ids = rng.sample(range(10000, 99999), 17)
    cases, mapping = [], []
    for t in candidate["topics"]:
        for v in t["variants"]:
            cid = f"M{ids.pop()}"
            case = {"case_id": cid, "title": t["title"], "prerequisite": prior_with_quotes(t["prerequisite"]), "units": [{**u, "checkpoint": True} for u in v["units"]]}
            cases.append(case)
            mapping.append({"case_id": cid, "topic_id": t["topic_id"], "kind": v["kind"], "expected": v["expected"]})
    cid = f"M{ids.pop()}"
    cases.append({"case_id": cid, **dino_material()})
    mapping.append({"case_id": cid, "topic_id": "original_dino", "kind": "known_regression", "expected": {"severity": "burden", "target_gap": {"missing": "DINO 自我蒸餾方法名稱含義、兩模型分工與保留特徵後用的基本串接，不把投影頭細節後教擴為基本介紹皆可後教", "already_taught": "無標籤動機、同圖關係、ViT/CLS、兩模型目標/回答與投影頭角色、僅生成view已存在；導覽也說影像特徵方法。", "needed_now": "u4已把命名DINO引入機制與後續塌縮問題；尚欠基本方法定位與ViT/局部view的分工銜接。"}}})
    rng.shuffle(cases)
    dump(ROOT / "materials.json", {"cases": cases, "adapted": True, "original_dino_only": True})
    dump(ROOT / "private/author-expectations.json", {"cases": mapping})


def runs():
    mats = json.loads((ROOT / "materials.json").read_text())["cases"]
    mapping = json.loads((ROOT / "private/author-expectations.json").read_text())["cases"]
    gold = json.loads((ROOT / "private/gold.json").read_text())
    if {c["case_id"] for c in gold["cases"]} != {c["case_id"] for c in mats}:
        raise ValueError("frozen gold must cover all cases")
    by_id = {c["case_id"]: c for c in mats}
    ids = {(c["topic_id"], c["kind"]): c["case_id"] for c in mapping}
    topics = ["architecture_csp", "algorithm_nms", "evaluation_ap50", "concept_dfl"]
    kinds = ["complete", "essential_gap", "name_only", "legitimate_deferral"]
    rng = random.Random(2026100602)
    manifest = []
    for rep in range(3):
        for batch in range(4):
            cohort = f"r{rep + 1}b{batch + 1}"
            chosen = [ids[(topic, kinds[(batch + i + rep) % 4])] for i, topic in enumerate(topics)]
            if batch == 0:
                chosen.append(ids[("original_dino", "known_regression")])
            rng.shuffle(chosen)
            row = {"cohort": cohort, "repeat": rep + 1, "batch": batch + 1, "case_ids": chosen, "readers": []}
            for tag in ["a", "b1", "b2", "c", "ca"]:
                rid = f"{cohort}_{tag}"
                run = {"reader_id": rid, "role": "auditor" if tag == "ca" else "reader", "structured": tag == "c", "primary_id": f"{cohort}_c" if tag == "ca" else None, "cases": [by_id[cid] for cid in chosen]}
                dump(ROOT / "packets" / f"{rid}.json", run)
                row["readers"].append({"reader_id": rid, "role": tag})
            manifest.append(row)
    dump(ROOT / "private/run-manifest.json", {"cohorts": manifest, "seed": 2026100602})
    paths = [ROOT / "materials.json", ROOT / "private/gold.json", ROOT / "protocol.md", ROOT / "reader-instructions.md", ROOT / "structured-addendum.md", ROOT / "auditor-instructions.md", ROOT / "frozen-skill.md", ROOT / "gate.py", ROOT / "prepare.py", ROOT / "private/run-manifest.json"]
    paths += [ROOT / "private/corpus-candidate.json", ROOT / "private/author-expectations.json", ROOT / "private/independent-gold.json", ROOT / "private/dino-boundary.json"]
    paths += sorted((ROOT / "packets").glob("*.json"))
    dump(ROOT / "freeze.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "code_commit": "516c257ef450a09c185a0d797f334cb52fe30c91", "files": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]})
    print(f"Frozen {len(mats)} unique texts, {len(manifest)} cohorts, {5 * len(manifest)} fresh contexts.")


if __name__ == "__main__":
    import sys
    {"materials": materials, "runs": runs}[sys.argv[1]]()
