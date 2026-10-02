"""Pair each authored lesson with a standalone, version-pinned Colab notebook."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "birdhackor/learn_to_yolo"


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def bootstrap(ref: str, deployment: bool) -> str:
    dependencies = ["numpy==2.3.5", "pillow==12.0.0", "matplotlib==3.10.7"]
    if deployment:
        dependencies += ["onnx==1.19.1", "onnxruntime==1.23.2"]
    return f'''# 全新 runtime 可由此格開始。只取得小型程式，不自動下載 LFS 資料。
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

REF = {ref!r}
base = Path("/content") if Path("/content").is_dir() else Path.cwd()
repository = base / ("learn_to_yolo_" + REF.replace(".", "_").replace("-", "_"))
if not repository.exists():
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF,
                    "https://github.com/{REPOSITORY}.git", str(repository)],
                   env=environment, check=True)
actual_ref = subprocess.check_output(
    ["git", "-C", str(repository), "describe", "--tags", "--exact-match"], text=True
).strip()
if actual_ref != REF:
    raise RuntimeError("程式版本不同，請使用新的 runtime 或移除舊 clone 後重試。")

# 本輪只驗證 CPU。已裝相同 PyTorch 版本時保留其 CPU/CUDA build。
try:
    torch_version = importlib.metadata.version("torch").split("+")[0]
except importlib.metadata.PackageNotFoundError:
    torch_version = None
if torch_version != "2.9.1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.9.1",
                    "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    if "torch" in sys.modules:
        raise RuntimeError("PyTorch 已更新：請從選單重新啟動工作階段，再由第一格執行。")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                *{dependencies!r}], check=True)
os.chdir(repository)
sys.path.insert(0, str(repository))
print("固定教材版本：", REF)
print("目前目錄：", repository)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="lessons-v0.1.0")
    args = parser.parse_args()
    registry_path = ROOT / "section-map.json"
    registry = json.loads(registry_path.read_text())
    for section in registry["sections"]:
        if section["kind"] != "lesson":
            continue
        lesson_id = section["id"]
        page = ROOT / "docs/lessons" / f"{lesson_id}.md"
        case = ROOT / "lesson_cases" / f"{lesson_id}.py"
        if not page.is_file() or not case.is_file():
            raise RuntimeError(f"Incomplete authored lesson: {lesson_id}")
        notebook_path = ROOT / "notebooks" / f"{lesson_id}.ipynb"
        notebook = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
                "lesson_id": lesson_id, "source_ref": args.ref,
            },
            "cells": [
                cell("markdown", f"# {section['title']}\n\n"
                     f"[完整圖文教材](https://birdhackor.github.io/learn_to_yolo/lessons/{lesson_id}/)\n\n"
                     "先執行環境格，再閱讀、執行下方完整實驗。各節互相獨立，不需要上一節的 runtime。\n\n"
                     "本版使用 CPU 驗證機制及短步參數更新；長訓練的辨識效果、AP 與 GPU 效率仍待補測。"
                     "這些範例沒有預訓練權重，也不需下載資料。"),
                cell("code", bootstrap(args.ref, lesson_id == "20-deployment")),
                cell("markdown", "## 本節可修改的完整實驗\n\n先預測結果，再執行；確認輸出後，試做網頁的自主練習。"),
                cell("code", case.read_text()),
            ],
        }
        for index, item in enumerate(notebook["cells"]):
            item["id"] = f"{lesson_id}-{index}"
        notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        url = f"https://colab.research.google.com/github/{REPOSITORY}/blob/{args.ref}/notebooks/{lesson_id}.ipynb"
        text = page.read_text().replace("{{COLAB:" + lesson_id + "}}", url)
        text = re.sub(r"https://colab\.research\.google\.com/github/" + REPOSITORY
                      + r"/blob/[^/]+/notebooks/" + re.escape(lesson_id) + r"\.ipynb", url, text)
        if not re.search(r"\[[^\]]+\]\(" + re.escape(url) + r"\)", text):
            first, rest = text.split("\n", 1)
            text = first + f"\n\n[在 Colab 執行本節]({url}){{ .md-button }}\n" + rest
        page.write_text(text)
        section.update(status="ready", page=str(page.relative_to(ROOT)),
                       notebook=str(notebook_path.relative_to(ROOT)), source_ref=args.ref)
    registry["note"] = "ready means authored files exist; CPU execution and review evidence are recorded separately. GPU training quality is pending."
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    print("Paired 42 authored lessons with version-pinned notebooks.")


if __name__ == "__main__":
    main()
