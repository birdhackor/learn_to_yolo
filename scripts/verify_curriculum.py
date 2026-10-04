"""Run lesson cases, record their actual output and machine, and attach the output to each page.

A record is bound by SHA-256 to the exact bytes of its lesson case and of every repository file the
case imports (miniyolo modules, scripts), so it stays valid until one of them changes. By default only
sections without a current record are executed; the others keep their record. Each record also stores
the machine that produced it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MARKER = "\n<!-- curriculum-evidence:start -->"
FOLDER = ROOT / "artifacts/checks/curriculum"
THREADS = "2"
# Lesson programs write their figures under artifacts/; the recorded run copies them to the site.
SITE_FIGURES = {
    "17-capstone": ("artifacts/lesson-17/validation.svg", "docs/assets/diagrams/17-capstone.svg"),
    "18-video": ("artifacts/lesson-18/panel.svg", "docs/assets/diagrams/18-video.svg"),
    "19-tracking": ("artifacts/lesson-19/ids.svg", "docs/assets/diagrams/19-tracking.svg"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def machine() -> dict:
    sys.path.insert(0, str(ROOT))
    from miniyolo.provenance import machine_info
    return machine_info(THREADS)


def evidence_block(record: dict) -> str:
    stdout = record["stdout"].strip()
    shown = stdout if len(stdout) <= 6000 else stdout[:2500] + "\n…（完整輸出見 notebook 與 JSON 紀錄）…\n" + stdout[-1500:]
    computer = record["machine"]
    return (MARKER + "\n\n## 實際執行紀錄\n\n"
            f"本節的完整程式於 {record['executed_at_utc'][:10]} 在 {computer['cpu']}（{computer['threads']} 個執行緒）上"
            f"用 PyTorch {record['torch']} 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；"
            "輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。"
            f"[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/{record['id']}.json)\n\n"
            "??? example \"展開本次實際輸出\"\n\n"
            + "    ```text\n" + "\n".join("    " + line for line in shown.splitlines()) + "\n    ```\n"
            "\n<!-- curriculum-evidence:end -->\n")


def dependencies(section: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    from miniyolo.provenance import repo_dependencies
    return repo_dependencies(ROOT / f"lesson_cases/{section['id']}.py")


def current_record(section: dict) -> dict | None:
    """The saved record if the lesson case and every repository file it imports are unchanged, else None."""
    path = FOLDER / f"{section['id']}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    case = ROOT / f"lesson_cases/{section['id']}.py"
    current = record.get("passed") and record.get("case_sha256") == sha256(case) \
        and record.get("dependencies_sha256") == dependencies(section)
    return record if current else None


def attach(section: dict, record: dict, write: bool = True) -> bool:
    """Put a record's stdout into the notebook and its evidence block into the page; True if anything changed."""
    notebook_path = ROOT / section["notebook"]
    notebook = json.loads(notebook_path.read_text())
    case = ROOT / f"lesson_cases/{section['id']}.py"
    assert "".join(notebook["cells"][-1]["source"]) == case.read_text(), f"Rebuild notebooks first: {section['id']}"
    notebook["cells"][-1].update(execution_count=1, outputs=[
        {"output_type": "stream", "name": "stdout", "text": record["stdout"].splitlines(keepends=True)}])
    notebook_text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    page = ROOT / section["page"]
    page_text = page.read_text().split(MARKER)[0].rstrip() + "\n" + evidence_block(record)
    changed = notebook_text != notebook_path.read_text() or page_text != page.read_text()
    if write:
        notebook_path.write_text(notebook_text)
        page.write_text(page_text)
    return changed


def run(section: dict, torch) -> dict:
    lesson_id = section["id"]
    case = ROOT / f"lesson_cases/{lesson_id}.py"
    environment = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg", OMP_NUM_THREADS=THREADS)
    start = time.perf_counter()
    execution = subprocess.run([sys.executable, str(case)], cwd=ROOT, env=environment,
                               text=True, capture_output=True, timeout=120)
    return {"id": lesson_id, "passed": execution.returncode == 0,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu",
            "machine": machine(),
            "case_sha256": sha256(case), "dependencies_sha256": dependencies(section),
            "process_wall_seconds": round(time.perf_counter() - start, 4),
            "timing_scope": "Entire subprocess including imports, calculations, logging, figures and files; not a model benchmark",
            "stdout": execution.stdout, "stderr": execution.stderr, "exit_code": execution.returncode}


def write_index(lessons: list[dict]) -> None:
    results = [json.loads((FOLDER / f"{s['id']}.json").read_text()) for s in lessons
               if (FOLDER / f"{s['id']}.json").is_file()]
    (FOLDER / "index.json").write_text(json.dumps({"passed": sum(r["passed"] for r in results),
        "total": len(results), "results": [{k: v for k, v in r.items() if k not in {"stdout", "stderr"}} for r in results]},
        ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", action="append", help="run this section even if its record is current (repeatable)")
    parser.add_argument("--all", action="store_true", help="run every section")
    parser.add_argument("--check", action="store_true", help="list sections without a current record and exit 1 if any")
    parser.add_argument("--render-only", action="store_true",
                        help="run nothing; rewrite every notebook output and page evidence block from the saved records")
    args = parser.parse_args()
    lessons = [s for s in json.loads((ROOT / "section-map.json").read_text())["sections"] if s["kind"] == "lesson"]
    unknown = set(args.section or []) - {s["id"] for s in lessons}
    if unknown:
        parser.error(f"unknown section: {', '.join(sorted(unknown))}")
    stale = [s["id"] for s in lessons if current_record(s) is None]
    if args.check:
        print("\n".join(f"{sid}: no current record" for sid in stale) or "All 42 records match the current code.")
        raise SystemExit(1 if stale else 0)
    if args.render_only:
        if stale:
            raise SystemExit(f"STOP: run these sections first: {', '.join(stale)}")
        changed = [s["id"] for s in lessons if attach(s, current_record(s))]
        write_index(lessons)
        print(f"Rendered 42 evidence blocks from saved records; changed: {', '.join(changed) or 'none'}")
        return
    selected = {s["id"] for s in lessons} if args.all else set(stale) | set(args.section or [])
    FOLDER.mkdir(parents=True, exist_ok=True)
    import torch
    for section in lessons:
        lesson_id = section["id"]
        if lesson_id not in selected:
            continue
        result = run(section, torch)
        if not result["passed"]:
            print(result["stderr"], flush=True)
            raise SystemExit(f"STOP: {lesson_id} failed; its record was not written and later sections were not run.")
        if lesson_id in SITE_FIGURES:
            produced, site = SITE_FIGURES[lesson_id]
            shutil.copyfile(ROOT / produced, ROOT / site)
        # Update THIS section before the next one starts; the record is written last, so a section
        # whose figure, notebook or page could not be updated stays out of date and runs again.
        attach(section, result)
        (FOLDER / f"{lesson_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"{lesson_id}: PASS; record, notebook and page evidence updated", flush=True)
    write_index(lessons)
    kept = len(lessons) - len(selected)
    print(f"Ran {len(selected)} section(s); kept {kept} current record(s).")


if __name__ == "__main__":
    main()
