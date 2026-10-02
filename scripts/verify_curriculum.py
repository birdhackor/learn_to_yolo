"""Execute lessons in reading order and immediately attach their actual evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MARKER = "\n<!-- curriculum-evidence:start -->"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", action="append")
    args = parser.parse_args()
    registry = json.loads((ROOT / "section-map.json").read_text())
    folder = ROOT / "artifacts/checks/curriculum"
    folder.mkdir(parents=True, exist_ok=True)
    import torch
    for section in registry["sections"]:
        if section["kind"] != "lesson" or (args.section and section["id"] not in args.section):
            continue
        lesson_id = section["id"]
        case = ROOT / f"lesson_cases/{lesson_id}.py"
        notebook_path = ROOT / section["notebook"]
        notebook = json.loads(notebook_path.read_text())
        assert "".join(notebook["cells"][-1]["source"]) == case.read_text()
        environment = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg", OMP_NUM_THREADS="2")
        start = time.perf_counter()
        execution = subprocess.run([sys.executable, str(case)], cwd=ROOT, env=environment,
                                   text=True, capture_output=True, timeout=120)
        result = {"id": lesson_id, "passed": execution.returncode == 0,
                  "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu",
                  "case_sha256": hashlib.sha256(case.read_bytes()).hexdigest(),
                  "process_wall_seconds": round(time.perf_counter() - start, 4),
                  "timing_scope": "Entire subprocess including imports, calculations, logging, figures and files; not a model benchmark",
                  "stdout": execution.stdout, "stderr": execution.stderr, "exit_code": execution.returncode}
        (folder / f"{lesson_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if not result["passed"]:
            print(execution.stderr, flush=True)
            raise SystemExit(f"STOP: {lesson_id} failed; subsequent sections were not run.")
        notebook["cells"][-1].update(execution_count=1, outputs=[
            {"output_type": "stream", "name": "stdout", "text": execution.stdout.splitlines(keepends=True)}])
        notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        # Update THIS section before the next one starts. Large JSON output remains
        # available in the notebook and the unabridged machine-readable report.
        stdout = execution.stdout.strip()
        shown = stdout if len(stdout) <= 6000 else stdout[:2500] + "\n…（完整輸出見 notebook 與 JSON 紀錄）…\n" + stdout[-1500:]
        evidence = (MARKER + "\n\n## 本輪實際執行紀錄\n\n"
                    f"本節範例已於 {result['executed_at_utc'][:10]} 使用 PyTorch {result['torch']} 在 CPU 執行，"
                    "程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。"
                    f"[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/{lesson_id}.json)\n\n"
                    "??? example \"展開本次實際輸出\"\n\n"
                    + "    ```text\n" + "\n".join("    " + line for line in shown.splitlines()) + "\n    ```\n"
                    "\n<!-- curriculum-evidence:end -->\n")
        page = ROOT / section["page"]
        page.write_text(page.read_text().split(MARKER)[0].rstrip() + "\n" + evidence)
        print(f"{lesson_id}: PASS; notebook and lesson evidence updated", flush=True)
    results = [json.loads((folder / f"{s['id']}.json").read_text()) for s in registry["sections"]
               if s["kind"] == "lesson" and (folder / f"{s['id']}.json").is_file()]
    (folder / "index.json").write_text(json.dumps({"passed": sum(r["passed"] for r in results),
        "total": len(results), "results": [{k: v for k, v in r.items() if k not in {"stdout", "stderr"}} for r in results]},
        ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
