"""Pair each authored lesson with a standalone, version-pinned Colab notebook."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "birdhackor/learn_to_yolo"


def optional_experiment(lesson_id: str) -> str:
    if lesson_id in {"01-small-cnn", "03-comparison", "04-localization"}:
        command = f"!python scripts/run_learning_extensions.py --section {lesson_id}"
        return ("\n\n### 可選：40步學習實驗\n\n完成下面的完整案例後，另開code cell執行：\n\n"
                f"```python\n{command}\nfrom IPython.display import SVG, display\n"
                f"display(SVG(filename='docs/assets/diagrams/{lesson_id}-learning.svg'))\n```\n\n"
                "資料、optimizer與結果的限制見網頁；這是獨立補充，不取代下方三步檢查。")
    if lesson_id == "10-multiscale":
        return ("\n\n### 可選：兩尺度40步與實際預測\n\n完整案例後另開code cell：\n\n"
                "```python\n!python scripts/run_multiscale_learning.py\nfrom IPython.display import SVG, display\n"
                "display(SVG(filename='docs/assets/diagrams/10-multiscale-learning.svg'))\n```\n\n"
                "這只訓練一張圖；小框可能未達IoU .5，應按實際report解讀。")
    if lesson_id == "07-training":
        return ("\n\n### 可選：160步合成圖與held-out評估\n\n下方主例只跑三步。完成後另開code cell：\n\n"
                "```python\n!python -m miniyolo.train --steps 160 --samples 32 --device cpu\n"
                "from IPython.display import display\nfrom PIL import Image\n"
                "display(Image.open('artifacts/runs/grid-learning/loss.png'))\n```\n\n"
                "另存checkpoint.pt、history.json與validation PNG；完整指標在artifacts/checks/grid-learning.json。")
    return ""


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

# 逐節範例預設 CPU；L4 訓練與部署的實測另見網站驗證頁。保留相同版本的 CPU/CUDA build。
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
    parser.add_argument("--ref", default="lessons-v0.2.0")
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
                     "本節在 CPU 逐節執行並保存輸出；合成資料短訓練、L4 與部署紀錄見網站驗證頁。真實場景長訓練仍待安排。"
                     "這些範例沒有預訓練權重，也不需下載資料。"),
                cell("code", bootstrap(args.ref, lesson_id == "20-deployment")),
                cell("markdown", "## 本節可修改的完整實驗\n\n先預測結果，再執行；確認輸出後，試做網頁的自主練習。"
                     + optional_experiment(lesson_id)),
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
    registry["note"] = "ready means authored files exist; CPU execution and review evidence are recorded separately. Bounded L4 training/checkpoint evidence and per-section results are recorded separately; full real-world training is pending."
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    print("Paired 42 authored lessons with version-pinned notebooks.")


if __name__ == "__main__":
    main()
