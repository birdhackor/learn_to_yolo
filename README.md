# Learn to YOLO

用小型 PyTorch 模型，從 VGG 風格 CNN、ResNet，一路學會圖片偵測與 YOLO 各分支的設計選擇。中文為主：每次改了什麼、為什麼、帶來什麼好處，又付出什麼。

**[閱讀教材](https://birdhackor.github.io/learn_to_yolo/)** · [完整課綱](docs/planning/outline.md) · [驗證範圍](docs/status.md)

42 節網頁各有獨立 Colab notebook，純閱讀也能學；程式與 Colab 固定為 `lessons-v0.2.0`。本輪另以六位陌生讀者與六位技術 reviewer 逐節審查，必要修改經獨立複查；[全套執行與審查](docs/validation/curriculum.md)保留結果與原始意見。模型是教學用簡化模型，不是完整原版的重現。

## CPU 本機執行

使用 Python 3.12，在 repository 根目錄：

```bash
python3 -m venv .venv-model
.venv-model/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv-model/bin/python -m pip install -r requirements-model.txt
PYTHONPATH=. .venv-model/bin/python lesson_cases/00-warmup.py
.venv-model/bin/python -m pytest tests/test_core.py tests/test_checkpoint.py
.venv-model/bin/python scripts/check_lesson_runtime.py
```

以上安裝後，若照網頁的`python ...`命令執行，先用`source .venv-model/bin/activate`啟用環境；也可一直使用`.venv-model/bin/python`的完整路徑。Colab環境格已使用目前runtime的直譯器。

案例不需要下載、GPU、torchvision 或預訓練權重。Windows 可用 `.venv-model\Scripts\python.exe`，並在 PowerShell 先設定 `$env:PYTHONPATH='.'`。Notebook 的實驗格有完整可修改程式；初始化格只取得固定版本與依賴。

## 從零訓練小偵測器

```bash
.venv-model/bin/python -m miniyolo.train --steps 160 --samples 32 --device cpu
```

此命令產生固定的紅／藍矩形，走過 data → target → loss → update → decode → NMS → held-out AP50。設定、曲線、範例圖與 checkpoint 存在 `artifacts/runs/grid-learning/`；報告預設在 `artifacts/checks/grid-learning.json`。這次固定 CPU 實驗得到 validation mAP50 約 .804、test 約 .775，只代表這個受控合成任務，不能外推到照片。

較完整的預算可使用 `--epochs 20 --samples 1024`，並指定不同 `--output`／`--report` 保留對照。GPU 環境先依 [PyTorch 官方安裝選擇器](https://pytorch.org/get-started/locally/) 安裝相容的 CUDA build，確認 `torch.cuda.is_available()`，再使用 `--device cuda`。

已完成一次 [Modal L4／checkpoint 實測](docs/validation/gpu-smoke.md)：沿用本專案模型與訓練步驟，共 80 次 optimizer 更新；跨 container Volume、私有 HF 上下載校驗，以及恢復模型／optimizer／排程／RNG 均通過。[Actions 紀錄](https://github.com/birdhackor/learn_to_yolo/actions/runs/37032967155)保留 JSON artifact。工作流程僅接受手動觸發，不因 push 或 PR 啟動 GPU；正式 GPU 效率比較與真實資料完整訓練仍待實驗。

## 真實分類資料的短步檢查

Fashion-MNIST 四個原始 gzip 已驗證 SHA-256，完整 MIT 授權封裝已發布到 Git LFS。預設 lesson 保持免下載；若要接真實圖片：

```bash
python3 scripts/download_data.py fetch fashion-mnist
.venv-model/bin/python scripts/run_fashion_cnn.py --steps 2
```

Loader 會核對 bytes、SHA-256 與 IDX 格式；28×28 灰階圖複製到三個 channel，再交給同樣的 RGB TinyCNN。固定 54k train／6k validation 與官方 10k test，短步只取小 subset，驗證權重更新。`--train-steps 1000 --eval-samples 0` 可供之後的完整比較；兩步輸出不代表模型已學好。偵測資料的來源、大小與再散佈限制見 [資料規劃](docs/preparation/data.md)。

## 網站預覽與出版

網站使用 **Zensical 0.0.67** 的 modern 主題，原生設定放在 `zensical.toml`。中文導覽、全文搜尋、深淺色模式與快速換頁由 Zensical 提供；公式與 SVG 隨網站發布。

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements-docs.txt
python3 scripts/validate_preparation.py
python3 scripts/validate_lessons.py
python3 scripts/validate_curriculum_evidence.py
.venv-docs/bin/zensical build --clean --strict
python3 scripts/validate_site.py
.venv-docs/bin/zensical serve
```

網站建置只需要文件依賴，不需要 PyTorch、資料或 GPU。後續新版教材以 `python3 scripts/build_lesson_notebooks.py --ref <新release-tag>` 配對，執行 CPU 檢查後再發布新 tag 與 Pages；不要覆寫已發布 tag。GitHub Pages／Colab／LFS 維護步驟見 [發布操作](docs/preparation/publish.md)。

## 程式結構

- `miniyolo/`：資料、模型、targets、loss、解碼、幾何、AP 與可選訓練 CLI。
- `lesson_cases/`：每節可獨立執行的小實驗，notebook 直接包含同份程式。
- `docs/lessons/`、`docs/assets/diagrams/`：教學文字、SVG 與實測圖。
- `artifacts/checks/`：實測結果；大型或臨時產物在 ignored `artifacts/runs/`。
- `section-map.json`：42 節網頁／notebook／版本對應；`ready` 是檔案狀態，效果驗證另記。
