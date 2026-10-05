# lessons-v0.4.0 發布清單

這份清單把 lessons-v0.4.0 從現在的 main 一路做到公開發布，全部在**產生既有執行紀錄的那台 Linux 電腦**上完成，不需要換電腦。教材內容、審查紀錄與驗證工具都已經在 main 上；還沒做的只有：在這台電腦重產執行紀錄、核對正文引用的數字、重審因此改動的頁面，以及發布與發布後的驗證。

這份檔案是發布前的暫存清單，不屬於教材：第 5 步打 tag 前會把它從 repo 刪掉。所以開始前先把它複製到 repo 外面，之後照副本做。

## 0. 開始前確認

需要：

- 這台電腦有 repo 的 clone，`.venv-model`（Python 3.12、`torch 2.9.1+cpu` 與固定版本的套件）與 `.venv-docs`（Zensical）。缺哪個就照 `README.md` 建。
- GitHub CLI 已登入、能 push 到 `birdhackor/learn_to_yolo`（第 5–7 步要推送、啟動 workflow、下載結果）。
- 能連網（pip、約 31 MB 的 Fashion-MNIST）。
- 第 2、3 步要用 AI 審查：在這台電腦開 Claude Code（或其他能開 subagent、會跑指令的 AI），把附錄 B 的指示貼給它。

```bash
cd <repo>
git fetch origin
git checkout main
git pull --ff-only
cp RELEASE-lessons-v0.4.0.md ~/lessons-v0.4.0-release.md   # 之後照這份副本做
git status --short                                         # 應該是空的
.venv-model/bin/python -c "import sys, torch; print(sys.version.split()[0], torch.__version__)"   # 3.12.x 2.9.1+cpu
.venv-docs/bin/zensical --version
gh auth status
```

## 1. 重產執行紀錄

```bash
.venv-model/bin/python -m pip install -r requirements-video.txt
.venv-model/bin/python scripts/record_evidence.py
.venv-model/bin/python scripts/record_evidence.py --run
```

- 第二行只列出要重跑的紀錄，不執行任何實驗。應該是 42 節的逐節紀錄，加上 9 份補充紀錄：`grid-learning`、`custom-data-160-step`、`custom-data-learning`、四份 `*-learning`（第 1、3、4、10 章）、`video-file` 與 `fashion-mnist-learning`。沒有 GPU 紀錄：兩份 GPU 紀錄綁定的程式沒變，不用重跑。
- 第三行重跑上面這些紀錄。它會先核對 Python 與套件版本，不符就停下（不要加 `--allow-version-mismatch`，改照 README 重建 `.venv-model`）。跑完會更新紀錄、各節頁尾的執行紀錄區塊、notebook 最後一格的輸出，以及程式畫的實驗圖，最後印出 `Changed files:`。逐節實驗加起來只要一兩分鐘，補充實驗與下載另外花一點時間。
- 下載 Fashion-MNIST 失敗時，改用 repo 的 LFS 備份，再重跑第三行：

  ```bash
  git lfs pull --include="data/curated/fashion-mnist-v1.tar" --exclude=""
  tar -xf data/curated/fashion-mnist-v1.tar -C data/downloads
  ```

先不要 commit：第 2 步要拿新紀錄和上一個 commit 的紀錄比較。

## 2. 核對正文引用的數字

頁尾的執行紀錄區塊會自動更新，正文裡引用的數字不會。先把附錄 A 的小工具存成 `/tmp/compare_records.py`（整段複製貼到終端機即可），再執行：

```bash
python3 /tmp/compare_records.py > /tmp/record-changes.txt
less /tmp/record-changes.txt
```

它逐份列出這次重產後，數字或輸出有變的紀錄（日期、計時、版本與雜湊這類每次都會變的欄位不列），以及引用那份紀錄的頁面。

- 每份都寫「數字與輸出都沒變」：正文不用改，直接到第 3 步。
- 有變的：引用它的頁面要逐一核對，數字變了就改正文，據此下的結論（例如「下降到」「沒有學會」）也要跟著成立。這一步和第 3 步交給 AI 一起做，指示在附錄 B。

## 3. 重審改動的頁面

```bash
python3 scripts/review_coverage.py
```

它列出審查已對不上目前內容的頁面：第 2 步改過正文的頁面，以及顯示重畫的圖、而圖確實變了的頁面。每一頁都要照〈發布與帳號設定〉第 7 步審查，在 `reviews/` 對應的紀錄最後補上這次的檢查，再執行 `python3 scripts/review_coverage.py --write <這些頁面>`。附錄 B 的指示也包含這一步。沒有列出任何頁面時，這一步就跳過。

## 4. 跑完所有檢查

```bash
python3 scripts/validate_preparation.py
python3 scripts/validate_lessons.py
python3 scripts/validate_curriculum_evidence.py
.venv-model/bin/python -m pytest tests/
.venv-docs/bin/zensical build --clean --strict
python3 scripts/validate_site.py
```

六項都要通過。`validate_curriculum_evidence.py` 在第 1 步之前會失敗，重產後應該通過；這一步任何一項失敗，都不要往下做。

## 5. commit、刪掉這份清單、建立 tag、推送

```bash
git rm RELEASE-lessons-v0.4.0.md
git add -A
git status --short     # 應該只有紀錄、docs/assets/diagrams 的圖、頁面、notebook 與 reviews/
git commit -m "Record execution evidence and review the changed pages for lessons-v0.4.0"
git tag -a lessons-v0.4.0 -m "Rewritten lessons with recorded runs, source checks and per-page reviews"
git push origin main
git push origin lessons-v0.4.0
```

- `artifacts/checks/curriculum-release-bootstrap.json` 已經不在 repo 裡，不用再刪；`artifacts/checks/curriculum-publication.json` 要留著（第 7 步的檢查要從新 tag 讀它）。
- tag 推上去之後就不移動、不重建；之後發現 tag 內容有問題，要改用新的 tag，見附錄 C。

## 6. 發布網站

```bash
gh workflow run pages.yml --ref main
sleep 15
gh run list --workflow pages.yml --limit 1      # 確認最上面那筆是剛剛啟動的，記下編號
gh run watch <編號> --exit-status
```

build 與 deploy 都成功後，打開 <https://birdhackor.github.io/learn_to_yolo/>，任選一節按「在 Colab 執行本節」，確認網址裡是 `lessons-v0.4.0`。

## 7. 驗證公開網站與新 tag

```bash
gh workflow run verify-release.yml -f tag=lessons-v0.4.0
sleep 15
gh run list --workflow verify-release.yml --limit 1   # 記下編號
gh run watch <編號> --exit-status
python3 scripts/verify_release.py save --run <編號>
```

最後一行把兩份結果存成 `artifacts/checks/curriculum-publication.json` 與 `artifacts/checks/curriculum-release-bootstrap.json`，並印出各自的 `passed`。兩份都是 `passed=True` 時：

```bash
git add artifacts/checks/curriculum-publication.json artifacts/checks/curriculum-release-bootstrap.json
git commit -m "Record the published verification of lessons-v0.4.0"
git push origin main
```

這兩個檔案不在網站裡，不必重新部署。有一份沒通過時，照附錄 C 處理，兩份都不要 commit。

## 8. 收尾

```bash
git status --short          # 應該是空的
git branch -r               # 只剩 origin/main（和 origin/HEAD）
rm ~/lessons-v0.4.0-release.md
```

## 附錄 A：比對紀錄的小工具

整段貼到終端機，會寫出 `/tmp/compare_records.py`：

```bash
cat > /tmp/compare_records.py <<'PYEOF'
"""List execution records whose numbers or output changed since the last commit, and the pages that cite them.

Run from the repository root after `record_evidence.py --run`, before committing:
    python3 /tmp/compare_records.py > /tmp/record-changes.txt
"""
import json
import re
import subprocess
from pathlib import Path

PAGES = {
    'curriculum/00-warmup.json': ['docs/lessons/00-warmup.md', 'docs/status.md'],
    'curriculum/01-small-cnn-learning.json': ['README.md', 'docs/index.md', 'docs/learning-path.md', 'docs/lessons/01-small-cnn.md', 'docs/planning/outline.md'],
    'curriculum/01-small-cnn.json': ['docs/lessons/01-small-cnn.md'],
    'curriculum/02-diagnostics.json': ['docs/lessons/02-diagnostics.md'],
    'curriculum/03-comparison-learning.json': ['README.md', 'docs/index.md', 'docs/learning-path.md', 'docs/lessons/03-comparison.md', 'docs/planning/outline.md', 'docs/status.md'],
    'curriculum/03-comparison.json': ['docs/lessons/03-comparison.md'],
    'curriculum/03-identity.json': ['docs/lessons/03-identity.md'],
    'curriculum/03-projection.json': ['docs/lessons/03-projection.md'],
    'curriculum/04-coordinates.json': ['docs/lessons/04-coordinates.md'],
    'curriculum/04-localization-learning.json': ['README.md', 'docs/index.md', 'docs/learning-path.md', 'docs/lessons/04-localization.md', 'docs/planning/outline.md'],
    'curriculum/04-localization.json': ['docs/lessons/04-localization.md'],
    'curriculum/05-assignment.json': ['docs/lessons/05-assignment.md'],
    'curriculum/06-decode-nms.json': ['docs/lessons/06-decode-nms.md'],
    'curriculum/06-evaluation.json': ['docs/lessons/06-evaluation.md'],
    'curriculum/07-data.json': ['docs/lessons/07-data.md'],
    'curriculum/07-heldout.json': ['docs/lessons/07-heldout.md', 'docs/lessons/17-capstone.md'],
    'curriculum/07-inference.json': ['docs/lessons/07-inference.md'],
    'curriculum/07-loss.json': ['docs/lessons/07-loss.md'],
    'curriculum/07-targets.json': ['docs/lessons/07-targets.md'],
    'curriculum/07-training.json': ['docs/index.md', 'docs/learning-path.md', 'docs/lessons/07-training.md', 'docs/planning/outline.md', 'docs/status.md'],
    'curriculum/08-own-data.json': ['docs/lessons/08-own-data.md'],
    'curriculum/08-own-images.json': ['docs/lessons/08-own-images.md'],
    'curriculum/09-anchor-clustering.json': ['docs/lessons/09-anchor-clustering.md'],
    'curriculum/09-anchors.json': ['docs/lessons/09-anchors.md'],
    'curriculum/10-multiscale-learning.json': ['README.md', 'docs/index.md', 'docs/learning-path.md', 'docs/lessons/10-multiscale.md', 'docs/planning/outline.md'],
    'curriculum/10-multiscale.json': ['docs/lessons/10-multiscale.md'],
    'curriculum/11-augmentation.json': ['docs/lessons/11-augmentation.md'],
    'curriculum/11-csp.json': ['docs/lessons/11-csp.md'],
    'curriculum/11-fusion.json': ['docs/lessons/11-fusion.md'],
    'curriculum/11-iou-loss.json': ['docs/lessons/11-iou-loss.md'],
    'curriculum/12-anchor-free.json': ['docs/lessons/12-anchor-free.md'],
    'curriculum/12-assignment.json': ['docs/lessons/12-assignment.md'],
    'curriculum/12-decoupled-head.json': ['docs/lessons/12-decoupled-head.md'],
    'curriculum/12-dfl.json': ['docs/lessons/12-dfl.md'],
    'curriculum/13-dual-assignment.json': ['docs/lessons/13-dual-assignment.md'],
    'curriculum/13-nms-free.json': ['docs/lessons/13-nms-free.md'],
    'curriculum/14-feature-module.json': ['docs/lessons/14-feature-module.md'],
    'curriculum/15-area-attention.json': ['docs/lessons/15-area-attention.md'],
    'curriculum/15-attention-bridge.json': ['docs/lessons/15-attention-bridge.md'],
    'curriculum/16-dfl-free.json': ['docs/lessons/16-dfl-free.md'],
    'curriculum/16-inference-head.json': ['docs/lessons/16-inference-head.md'],
    'curriculum/16-training.json': ['docs/lessons/16-training.md'],
    'curriculum/17-capstone.json': ['docs/lessons/17-capstone.md'],
    'curriculum/18-video.json': ['docs/lessons/18-video.md'],
    'curriculum/19-tracking.json': ['docs/lessons/19-tracking.md'],
    'curriculum/20-deployment.json': ['docs/lessons/20-deployment.md'],
    'curriculum/custom-data-160-step.json': ['README.md', 'docs/index.md', 'docs/learning-path.md', 'docs/lessons/08-own-data.md', 'docs/planning/outline.md', 'docs/status.md'],
    'curriculum/custom-data-learning.json': ['README.md', 'docs/glossary.md', 'docs/index.md', 'docs/learning-path.md', 'docs/lessons/08-own-data.md', 'docs/planning/outline.md', 'docs/status.md'],
    'curriculum/deployment-gpu.json': ['README.md', 'docs/index.md', 'docs/learning-path.md', 'docs/lessons/20-deployment.md', 'docs/planning/outline.md', 'docs/status.md'],
    'curriculum/fashion-mnist-learning.json': ['README.md', 'docs/preparation/architecture.md', 'docs/preparation/data.md'],
    'curriculum/video-file.json': ['README.md', 'docs/lessons/18-video.md', 'docs/lessons/19-tracking.md'],
    'gpu-smoke.json': ['README.md', 'docs/index.md', 'docs/learning-path.md', 'docs/planning/outline.md', 'docs/status.md', 'docs/validation/gpu-smoke.md'],
    'grid-learning.json': ['README.md', 'docs/index.md', 'docs/learning-path.md', 'docs/lessons/07-heldout.md', 'docs/lessons/07-inference.md', 'docs/lessons/07-training.md', 'docs/lessons/17-capstone.md', 'docs/planning/outline.md', 'docs/preparation/publish.md'],
}
VOLATILE = {"executed_at", "executed_at_utc", "created_at", "machine", "hardware", "torch", "python", "platform", "threads",
            "stderr", "exit_code", "case_sha256", "dependencies_sha256", "data_sha256", "sha256", "bytes"}


def volatile(key):
    """Fields that change with every run or machine: dates, timings, versions, hashes."""
    tokens = [t for t in re.split(r"[.\[\]]", key) if t]
    return any(t in VOLATILE or "seconds" in t or t == "ms" or t.endswith("_ms") for t in tokens)


def leaves(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def committed(name):
    result = subprocess.run(["git", "show", f"HEAD:{name}"], capture_output=True, text=True)
    return json.loads(result.stdout) if result.returncode == 0 else {}


names = subprocess.run(["git", "status", "--porcelain", "--", "artifacts/checks"], capture_output=True, text=True,
                       check=True).stdout.splitlines()
names = sorted(line[3:] for line in names if line.endswith(".json") and not line.endswith("index.json"))
for name in names:
    old, new = committed(name), json.loads(Path(name).read_text())
    before, after = dict(leaves(old)), dict(leaves(new))
    changes = [f"  {key}: {before.get(key)!r} -> {after.get(key)!r}" for key in sorted(set(before) | set(after))
               if key != "stdout" and not volatile(key) and before.get(key) != after.get(key)]
    old_out, new_out = str(old.get("stdout", "")).splitlines(), str(new.get("stdout", "")).splitlines()
    output = [f"  輸出 - {a}\n  輸出 + {b}" for a, b in zip(old_out, new_out) if a != b]
    if len(old_out) != len(new_out):
        output.append(f"  輸出行數 {len(old_out)} -> {len(new_out)}")
    pages = PAGES.get(name.removeprefix("artifacts/checks/"), [])
    print(f"== {name}  （引用它的頁面：{'、'.join(pages) or '沒有'}）")
    if not changes and not output:
        print("  數字與輸出都沒變")
    for line in changes[:60] + output[:60]:
        print(line)
    if len(changes) > 60 or len(output) > 60:
        print(f"  ……另有 {max(0, len(changes) - 60) + max(0, len(output) - 60)} 項，請直接比對這份紀錄")
PYEOF
```

## 附錄 B：交給 AI 的指示

第 1 步跑完、還沒 commit 時，在 repo 根目錄開 Claude Code，把下面整段貼給它：

```text
這是 learn_to_yolo 的 main，正在準備 lessons-v0.4.0。執行紀錄剛用 scripts/record_evidence.py --run 重產完，還沒 commit。請完成下面三件事，不要 commit、push 或建立 tag。

1. 核對正文的數字。執行 python3 /tmp/compare_records.py > /tmp/record-changes.txt 並讀它：它列出數字或輸出有變的紀錄，以及引用每份紀錄的頁面（含 README.md）。對每份有變化的紀錄，打開每個引用它的頁面，找出正文（<!-- curriculum-evidence:start --> 以上）裡取自這份紀錄的數字、比較與據此下的結論，和新紀錄逐一核對；不一致就改正文，讓數字和結論都對得上新紀錄。頁面列表只是起點：也用紀錄的檔名與舊數字在 docs/ 與 README.md 搜尋，找出沒列到的引用。只改正文；頁尾自動區塊、程式、notebook 與程式畫的圖都不要手改。寫法照 docs 既有的規範：正體中文（台灣）、現在式、不寫修改經過、中文與英數之間留空格。

2. 重審改動的頁面。執行 python3 scripts/review_coverage.py，對它列出的每一頁，照 docs/preparation/publish.md 第 7 步審查：派一個獨立的 subagent，在 repo 的暫存副本（rsync 出去，排除 .git、site、.venv*）裡執行該節程式，逐句對照正文、新紀錄、重畫的圖（docs/assets/diagrams/ 裡、頁面顯示的那幾張）與手算，回報問題；你修正後，再派另一個 subagent 檢查你的修正，直到沒有必要的問題。每一頁在 reviews/ 對應的紀錄最後加一節「## 紀錄重產後的檢查」，寫明查核方法、每個發現與它的處理（課程頁是 reviews/<節>.md；其他頁取 docs/ 底下的路徑、把 / 換成 -，例如 docs/status.md 是 reviews/status.md）。全部寫好後，執行 python3 scripts/review_coverage.py --write <這些頁面>，再執行一次不加參數的 python3 scripts/review_coverage.py，確認沒有頁面被列出。

3. 跑完全部檢查：python3 scripts/validate_preparation.py、python3 scripts/validate_lessons.py、python3 scripts/validate_curriculum_evidence.py、.venv-model/bin/python -m pytest tests/、.venv-docs/bin/zensical build --clean --strict、python3 scripts/validate_site.py。最後回報：哪些紀錄的數字變了、改了哪些頁、每頁審查的結果，以及六項檢查的結果。
```

它做完後，自己看過改動（`git diff`），再從第 4 步繼續。

## 附錄 C：出問題時

- **第 1 步版本不符而停下**：照 `README.md` 重建 `.venv-model`，再跑一次。不要用 `--allow-version-mismatch`，否則紀錄記下的環境和教材寫的不一致。
- **第 4 步有檢查失敗**：照錯誤訊息修正，修的是正文就回到第 3 步重審那一頁；修的是程式（`lesson_cases/`、它們 import 的模組或補充實驗的腳本），就從第 1 步重來，因為紀錄會因此過期。
- **第 6 步部署失敗**：`gh run view <編號> --log-failed` 看原因，修好後在 main 上重跑第 6 步；tag 不用動。
- **第 7 步有一份沒通過**：兩份都不要 commit。用 `git restore --staged --worktree artifacts/checks/curriculum-publication.json` 還原上一版的網站驗證，並刪掉 `artifacts/checks/curriculum-release-bootstrap.json`。原因在 tag 以外（部署還沒完成、網路或套件下載暫時失敗）時，確認部署成功後重跑第 7 步。原因在 tag 的內容（頁面、notebook、README 或程式）時，已推上去的 tag 不移動：修正後改用新的 tag（例如 `lessons-v0.4.1`），先執行 `python3 scripts/build_lesson_notebooks.py --ref <新 tag>` 讓 notebook 與 Colab 連結指向新 tag，正文裡寫死 tag 的地方（`README.md`、〈驗證範圍〉、〈全套實驗與審查〉與第 0 章〈第一次用 Colab〉）也一起改，再從本清單第 1 步做起。完整說明在〈發布與帳號設定〉第 11 步。
