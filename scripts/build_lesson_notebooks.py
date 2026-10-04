"""Pair each authored lesson with a standalone, version-pinned Colab notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "birdhackor/learn_to_yolo"


def optional_experiment(lesson_id: str) -> str:
    if lesson_id in {"01-small-cnn", "03-comparison", "04-localization"}:
        return ("\n\n### 可選：40 步學習實驗\n\n跑完下面的完整實驗後，另開一個 code cell 執行：\n\n"
                f"```python\n!python scripts/run_learning_extensions.py --section {lesson_id}\n"
                "from IPython.display import SVG, display\n"
                f"display(SVG(filename='artifacts/runs/learning/{lesson_id}/learning.svg'))\n```\n\n"
                "資料、optimizer 與結果的限制見網頁；這是另外的補充實驗，不取代下面的完整實驗。")
    if lesson_id == "10-multiscale":
        return ("\n\n### 可選：兩尺度 40 步與實際預測\n\n跑完完整實驗後，另開一個 code cell：\n\n"
                "```python\n!python scripts/run_multiscale_learning.py\nfrom IPython.display import SVG, display\n"
                "display(SVG(filename='artifacts/runs/multiscale-learning/learning.svg'))\n```\n\n"
                "這只訓練一張圖；小框不一定達到 IoU 0.5，請照實際的報告解讀。")
    if lesson_id == "07-training":
        return ("\n\n### 可選：160 步合成圖與 held-out 評估\n\n下面的主例只更新三步。跑完後另開一個 code cell：\n\n"
                "```python\n!python -m miniyolo.train --steps 160 --samples 32 --device cpu\n"
                "from IPython.display import display\nfrom PIL import Image\n"
                "display(Image.open('artifacts/runs/grid-learning/loss.png'))\n```\n\n"
                "checkpoint.pt、history.json、validation 圖與完整指標 report.json 都存在 artifacts/runs/grid-learning/。")
    if lesson_id == "08-own-data":
        return ("\n\n### 可選：JSON 資料的完整短訓練\n\n主例只檢查一步更新；跑完後另開一個 code cell：\n\n"
                "```python\n!python scripts/run_custom_data_learning.py --fixture --steps 1600 --fixture-test-seed 7001\n"
                "from IPython.display import SVG, display\n"
                "display(SVG(filename='artifacts/runs/custom-data-learning/learning.svg'))\n```\n\n"
                "三類 PNG／JSON、checkpoint 與結果都存在 artifacts/runs/custom-data-learning/。"
                "換成自己的資料時改用 --annotations 與 --root；格式與指標的解讀見網頁。")
    if lesson_id in {"18-video", "19-tracking"}:
        return ("\n\n### 可選：真正的影片檔案與 tracking 接線\n\n跑完主例後另開一個 code cell：\n\n"
                "```python\n!python -m pip install -r requirements-video.txt\n"
                "!python scripts/verify_video_file.py\n```\n\n"
                "它自行產生 12 幀的無損 AVI，實際讀檔、模型推論、只替 class 0 配 track ID，"
                "並檢查 RGB 與框一致、影片資源有釋放；結果寫在 artifacts/runs/video-file/。沒有測實體相機。")
    return ""


def recorded_output(lesson_id: str, case: Path) -> str | None:
    """The saved stdout of this exact case, so rebuilding a notebook keeps its recorded run."""
    path = ROOT / "artifacts/checks/curriculum" / f"{lesson_id}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    digest = hashlib.sha256(case.read_bytes()).hexdigest()
    return record["stdout"] if record.get("passed") and record.get("case_sha256") == digest else None


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

# 各節實驗都在 CPU 上執行，執行紀錄用的是 PyTorch 2.9.1。已經是 2.9.1 就沿用（CPU 或 CUDA 版都可以）；
# 其他版本會改裝成 2.9.1 的 CPU 版，之後這個工作階段就用不到 GPU，但各節實驗本來就不需要。
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
    parser.add_argument("--ref", required=True, help="the release tag the notebooks and Colab links pin, e.g. lessons-v0.4.0")
    args = parser.parse_args()
    registry_path = ROOT / "section-map.json"
    registry = json.loads(registry_path.read_text())
    missing = []
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
                     "先執行下一格的環境格，再閱讀、執行最後的完整實驗。各節互相獨立，不需要先跑其他節。\n\n"
                     "最後一格存著這段程式在 CPU 上實際執行的輸出；執行環境與其他實驗紀錄見網站的〈驗證範圍與後續實驗〉。"
                     "範例不用預訓練權重，也不需要下載資料。"),
                cell("code", bootstrap(args.ref, lesson_id == "20-deployment")),
                cell("markdown", "## 本節可修改的完整實驗\n\n先預測結果，再執行；確認輸出後，試做網頁的自主練習。"
                     + optional_experiment(lesson_id)),
                cell("code", case.read_text()),
            ],
        }
        record = recorded_output(lesson_id, case)
        if record is None:
            missing.append(lesson_id)
        else:
            notebook["cells"][-1].update(execution_count=1, outputs=[
                {"output_type": "stream", "name": "stdout", "text": record.splitlines(keepends=True)}])
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
    registry["note"] = ("status ready means the page, lesson program and notebook exist; execution records are in "
                        "artifacts/checks/curriculum/ and reviews in reviews/.")
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    print(f"Paired 42 authored lessons with notebooks pinned to {args.ref}.")
    if missing:
        print("No current execution record (run scripts/verify_curriculum.py): " + ", ".join(missing))


if __name__ == "__main__":
    main()
