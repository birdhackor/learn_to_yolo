# Learn to YOLO

用小型 PyTorch 模型，從 VGG 風格 CNN、ResNet，一路學到物件偵測與 YOLO 各版本的設計選擇。以中文講解，每次只引入一個主要變化：它解決什麼問題、帶來什麼好處，又付出什麼代價。

**[閱讀教材](https://birdhackor.github.io/learn_to_yolo/)** · [課程大綱](docs/planning/outline.md) · [驗證範圍與後續實驗](docs/status.md)

42 節網頁各有一本獨立的 Colab notebook，只讀網頁也能學；Colab 與 notebook 用的程式固定在 `lessons-v0.4.0` 這個 tag。每節都有一份審查紀錄（[`reviews/`](reviews/)），記下兩位審查者的意見與每個發現的處理：一位扮演初學讀者（高中程度、數學好、程式新手），一位對照論文與固定 commit 的官方程式做技術查核。審查者都是 AI，沒有真人學生測試；執行紀錄與審查的總覽見[全套實驗與審查](docs/validation/curriculum.md)。模型是教學用的簡化版，不是原版的完整重現。

## CPU 本機執行

使用 Python 3.12，在 repository 根目錄執行：

```bash
python3 -m venv .venv-model
.venv-model/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv-model/bin/python -m pip install -r requirements-model.txt
PYTHONPATH=. .venv-model/bin/python lesson_cases/00-warmup.py
.venv-model/bin/python -m pytest tests/
.venv-model/bin/python scripts/check_lesson_runtime.py
```

最後一行在這台電腦依序執行 42 節 notebook 的實驗格，逐節和執行紀錄比對輸出（計時與部分小數換一台電腦可能不同）；報告寫到 git 不追蹤的 `artifacts/runs/lesson-runtime.json`，repo 追蹤的檔案不會改變。

要照網頁上的 `python ...` 命令執行，先用 `source .venv-model/bin/activate` 啟用環境，或一律改用完整路徑 `.venv-model/bin/python`。Windows 改用 `.venv-model\Scripts\python.exe`，並先在 PowerShell 設定 `$env:PYTHONPATH='.'`。各節實驗不需要下載資料、GPU、torchvision 或預訓練權重。

Notebook 的最後一格是完整、可修改的實驗程式，與 `lesson_cases/` 的同名檔逐字相同。前面的環境格直接使用 Colab runtime 的 Python，取得固定 tag 的程式並安裝固定版本的套件（PyTorch 不是 2.9.1 時改裝 2.9.1 的 CPU 版）。

## 從零訓練小偵測器

```bash
.venv-model/bin/python -m miniyolo.train --steps 160 --samples 32 --device cpu
```

這個命令用程式畫出固定的紅／藍矩形圖，走過 data → target → loss → update → decode → NMS → held-out AP50。loss 曲線、validation 範例圖、checkpoint 與完整報告 `report.json`（含設定與指標）都存在 `artifacts/runs/grid-learning/`。repo 保存的執行紀錄 [`artifacts/checks/grid-learning.json`](artifacts/checks/grid-learning.json) 用同樣的設定產生，檔內記有執行的電腦與 validation、test 的 mAP50；這份紀錄的 loss 曲線、指標與預測圖，在 [7.4 三步訓練與診斷](docs/lessons/07-training.md)的 160 步補充實驗有逐項解說。換一台電腦重跑，數字可能略有不同；這些結果只代表這個受控的合成任務，不能外推到照片。

較大的預算可用 `--epochs 20 --samples 1024`；另指定 `--output` 目錄，就能保留前一次的結果對照。GPU 環境先依 [PyTorch 官方安裝選擇器](https://pytorch.org/get-started/locally/) 安裝相容的 CUDA build，確認 `torch.cuda.is_available()` 為 `True`，再改用 `--device cuda`。

GPU 上的檢查只有兩項，都由手動啟動的 GitHub Actions 工作流程在 Modal 的一張 NVIDIA L4 上執行：一項是上面這個偵測器（GridDetector）訓練 40 步、中途存 checkpoint、換新的 container 讀回後續訓，結果和不中斷的訓練一致（[GPU／checkpoint 實測](docs/validation/gpu-smoke.md)）；另一項是第 20 章 ONNX／TensorRT 的 FP32、FP16 engine 與 PyTorch 輸出的一致性。push 或 PR 不會觸發這兩個工作流程。兩項都不是正式的速度比較，也沒有用真實資料完整訓練。

## 真實分類資料的短步檢查

這項檢查要另外下載資料：用 Fashion-MNIST（28×28 灰階衣物圖、10 類，MIT 授權）確認分類管線接得上真實圖片。

```bash
python3 scripts/download_data.py fetch fashion-mnist
.venv-model/bin/python scripts/run_fashion_cnn.py --steps 2
```

第一行從官方來源下載四個原始 gzip 到 git 不追蹤的 `data/downloads/`，並用 `data/manifest.json` 記錄的大小與 SHA-256 核對；含授權檔的同一份資料也打包存在 Git LFS（`data/curated/`）。

Loader 讀取時會再核對大小、SHA-256 與 IDX 格式，把 28×28 灰階圖複製成三個相同的 channel，交給 `miniyolo` 的 TinyCNN（RGB 輸入的 VGG 風格小分類器）。資料固定切成 54k train／6k validation，test 用官方的 10k。預設的兩步只用 64 張訓練圖，validation 與 test 各評 128 張，確認權重有更新；兩步的結果不代表模型學好了。`--train-steps 1000 --eval-samples 0` 改用完整 54k 訓練圖跑 1000 步，並評估完整的 validation 與 test。結果寫在 `artifacts/runs/fashion-cnn/`。偵測資料的來源、大小與再散佈限制見[資料規劃](docs/preparation/data.md)。

## 網站建置與發布

網站使用 **Zensical 0.0.67** 的 modern 主題，設定在 `zensical.toml`。中文導覽、全文搜尋、深淺色模式與快速換頁由 Zensical 提供；公式與 SVG 隨網站發布。

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

建置前的三項檢查核對資料清單與 notebook 配對、頁面裡的程式摘錄與審查紀錄，以及每份執行紀錄是否對得上目前的程式、notebook 與頁面；建置後的 `validate_site.py` 檢查連結、Colab 配對、搜尋與 Markdown 轉換。這些步驟只需要文件依賴，不需要 PyTorch、資料或 GPU；手動觸發的 GitHub Pages 工作流程依同樣順序執行。

發布新版教材時使用新的 tag，不覆寫已發布的 tag。步驟依序如下：

1. 執行 `python3 scripts/build_lesson_notebooks.py --ref <新版 tag>`：重建 notebook，Colab 連結改指向新 tag。
2. 執行 `.venv-model/bin/python scripts/record_evidence.py`：列出程式有變而過期的執行紀錄，不執行任何實驗。程式沒變的紀錄沿用，保留原本的日期與電腦。
3. 有過期的 GPU 紀錄時，先把目前的修改 commit 並推到一個分支，再執行 `.venv-model/bin/python scripts/record_evidence.py --gpu <分支>`。它用 GitHub CLI（`gh`）啟動 GitHub Actions 工作流程，等它跑完再把結果存成紀錄；工作流程跑的是推上去的程式，所以本機的 commit 要和該分支相同。
4. 用 `.venv-model/bin/python -m pip install -r requirements-video.txt` 加裝影片紀錄要用的 OpenCV，再執行 `.venv-model/bin/python scripts/record_evidence.py --run`，在這台電腦重產過期的 CPU 紀錄。`--run` 會先確認 Python 與套件符合固定版本；最好在產生既有紀錄的同一種環境執行（每份紀錄都記有它的電腦），沒有改到的計算才會得到相同的數字。
5. 重新審查改過的頁面。不加參數執行 `python3 scripts/review_coverage.py`，會列出還沒有審查、或審查對不上目前內容的頁面：頁面文字、頁面上的 SVG 圖，或課程頁的程式與它 import 的模組有變，都算改過；頁尾的執行紀錄區塊與 Colab 連結裡的 tag 不算。第 4 步會重畫部分實驗圖，所以審查放在紀錄之後。課程頁的審查同樣要有模擬初學讀者與技術查核兩位 AI 的意見，以及每個發現的處理。審查紀錄放在 `reviews/`（課程頁是 `reviews/<節>.md`；其他頁用路徑命名，例如 `docs/validation/curriculum.md` 是 `reviews/validation-curriculum.md`），寫好後執行 `python3 scripts/review_coverage.py --write <頁面路徑>`，記下它涵蓋的內容。還有頁面沒有審查、或審查對不上目前內容時，`validate_lessons.py` 會失敗。
6. 本節開頭的網站檢查全部通過後，提交變更並推到 main，建立並推送新 tag，再從 main 執行 Pages 工作流程（Publish Learn to YOLO）。

在另一台電腦記錄 CPU 紀錄的做法、GPU 工作流程需要的設定與失敗時的處理、部署後的公開網站與 Colab 驗證，以及 GitHub Pages／LFS 的其他操作，見[發布與帳號設定](docs/preparation/publish.md)。

## 目錄結構

- `miniyolo/`：資料、模型、targets、loss、解碼、幾何、AP 與可選的訓練 CLI。
- `lesson_cases/`、`notebooks/`：每節可獨立執行的實驗；notebook 的最後一格與同名程式逐字相同。
- `docs/`：網站內容；各節教材在 `docs/lessons/`，SVG 圖在 `docs/assets/diagrams/`。
- `artifacts/checks/`：執行與檢查紀錄。每節一份的執行紀錄在 `artifacts/checks/curriculum/`，記有執行的電腦，並以 SHA-256 綁定產生它的程式；其他輸出寫在 git 不追蹤的 `artifacts/runs/` 等位置。
- `reviews/`：網站導覽裡每一頁的 AI 審查紀錄；`coverage.json` 以 SHA-256 記下每份審查涵蓋的頁面文字、圖與程式。
- `scripts/`：資料下載、補充實驗、執行紀錄與網站檢查的工具。
- `section-map.json`：42 節的網頁、notebook 與固定 tag 對應表；`status: ready` 只表示網頁、程式與 notebook 都在。
