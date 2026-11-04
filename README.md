# Learn to YOLO

用小型 PyTorch 模型，從 VGG 風格 CNN、ResNet，一路學到物件偵測與 YOLO 各版本的設計選擇。以中文講解，每次只引入一個主要變化：它解決什麼問題、帶來什麼好處，又付出什麼代價。

**[閱讀教材](https://birdhackor.github.io/learn_to_yolo/)** · [課程大綱](docs/planning/outline.md) · [驗證範圍](docs/status.md)

本次工作版安排 52 節：第 0–20 章的原有 42 節主線，加上第 21–23 章的 10 節 ViT／DINO 選讀支線。主線順序不變；學過 CNN、ResNet 與 [15.1 attention](docs/lessons/15-attention-bridge.md) 後，可分岔讀 [21.1 圖片切成 patch](docs/lessons/21-patches.md)，理解圖片如何交給 Transformer、自監督特徵如何學習，再接回單物件定位。只讀網頁也能學；每節配對一本獨立 notebook。

本版實驗程式與 notebook 固定到 `lessons-v0.6.1`；網頁文字與圖解持續修訂，公開入口驗證以各次保存紀錄中的版本與結果為準。既有 `lessons-v0.5.0` 的發布驗證與依 [clear-tutorial skill](.agents/skills/clear-tutorial/SKILL.md) 做的主線審查保留原本範圍；[原始閱讀紀錄與處理](https://github.com/birdhackor/learn_to_yolo/tree/lessons-v0.6.1/reviews/clear-tutorial/16f6910)保留當時的卡點和協調者前提安排的限制。先前完整頁面審查保存在[固定封存版本](reviews/README.md)，不改標成盲讀，也不當成新支線的審查。審查者都是 AI，沒有真人學生學習效果測試；執行與審查範圍見[驗證範圍](docs/status.md)。模型是教學用的簡化版，不是原版的完整重現。

2026-10-06 已依新版四題重新審閱全部 52 節，保留分段閱讀、修正、非作者技術核對與前後銜接紀錄，見[本輪總覽](https://github.com/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/reviews/clear-tutorial/full-review-2026-10-06/README.md)。四題仍有 DINO 介紹漏抓與逐項引用缺陷，不能把保存答案當成嚴格盲讀全面合格；原始紀錄及獨立裁定一併保留。

2026-10-08 從 `7a8b9d7` 重寫 B–E 的 46 節，以當下的學習問題串起材料、做法、結果與理由；開頭及 A 保留。本輪紀錄分開保存作者的舊新對照、獨立新稿閱讀、技術與銜接審閱、必要及可選建議的處置、實頁與公開驗證，存在不公開的審查 repo（見 [`reviews/README.md`](reviews/README.md)）。原實驗程式未改，Notebook 程式碼仍對應上述 tag。

## CPU 本機執行

新取得專案可用 `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-tags https://github.com/birdhackor/learn_to_yolo.git`。使用 Python 3.12，在 repository 根目錄執行（第一行的 `python3` 要是 3.12 版；不是的話，換成 Python 3.12 直譯器的路徑）：

```bash
python3 -m venv .venv-model
.venv-model/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv-model/bin/python -m pip install -r requirements-model.txt
PYTHONPATH=. .venv-model/bin/python lesson_cases/00-warmup.py
.venv-model/bin/python scripts/check_lesson_runtime.py
```

最後一行在這台電腦依序執行配對表中的 52 節 notebook 實驗格，逐節和執行紀錄比對輸出。計時與部分小數換一台電腦可能不同，這不算失敗；有任何一節報錯，這個命令就回報失敗（結束代碼不是 0）。報告寫到 git 不追蹤的 `artifacts/runs/lesson-runtime.json`，repo 追蹤的檔案不會改變。

要照網頁上的 `python ...` 命令執行，先用 `source .venv-model/bin/activate` 啟用環境，或一律改用完整路徑 `.venv-model/bin/python`。Windows 改用 `.venv-model\Scripts\python.exe`，並先在 PowerShell 設定 `$env:PYTHONPATH='.'`。各節實驗不需要下載資料、GPU、torchvision 或預訓練權重。

Notebook 的最後一格是完整、可修改的實驗程式，與 `lesson_cases/` 的同名檔逐字相同。前面的環境格直接使用 Colab runtime 的 Python，取得固定 tag 的程式並安裝固定版本的套件（PyTorch 不是 2.9.1 時改裝 2.9.1 的 CPU 版）。

## ViT／DINO 選讀支線

[第 21 章](docs/lessons/21-patches.md)從 32×32 RGB 圖的 patch、位置與 CLS 開始，組成 2 個 Transformer blocks、23,970 個參數的小 ViT，再實際更新全部參數。固定設定的 CPU 60 步實驗，train 分類 loss 從 0.710059 降到 0.007155；獨立 seed 的 validation 與 test 各答對 64／64 張，checkpoint 續訓也和連續訓練一致。這只支持紅／藍矩形這個受控分類任務。

[第 22 章](docs/lessons/22-views.md)介紹 DINO（2021）的自監督影像特徵方法：訓練時不看類別標籤，讓 student 對同一張圖的不同視圖，學習 teacher 提供的目標。這裡用小模型重現核心機制，並非原版規模；CPU 160 步後，凍結的自監督特徵與隨機特徵接上相同線性分類器，test 都是 64／64，這次沒有觀察到自監督的分類優勢。DINO（2021）與同名的 DETR 系列偵測器是不同方法。

[第 23 章](docs/lessons/23-dino-versions.md)區分 DINO、DINOv2、DINOv3 的來源與適用範圍，再把凍結 patch 特徵接到可訓練的單物件定位 head。CPU 範例的 test 平均 IoU 為 0.605699，類別正確且 IoU≥0.5 的圖片有 54／64 張；這不是多物件偵測的 AP。支線預設不下載資料或權重、不需 GPU；官方預訓練特徵是另選的操作，沒有新增 GPU 訓練，也沒有實作 DINOv3。

## 從零訓練小偵測器

```bash
.venv-model/bin/python -m miniyolo.train --steps 160 --samples 32 --device cpu
```

這個命令用程式畫出固定的紅／藍矩形圖，走過 data → target → loss → update → decode → NMS → held-out AP50。checkpoint、loss 曲線圖、validation 範例圖、每一步的 loss（`history.json`）與完整報告 `report.json`（設定、指標與每一步的 loss），都存在 git 不追蹤的 `artifacts/runs/grid-learning/`。換一台電腦重跑，數字可能略有不同；這些結果只代表這個受控的合成任務，不能外推到照片。

repo 保存的執行紀錄 [`artifacts/checks/grid-learning.json`](artifacts/checks/grid-learning.json)，是同一個命令另加 `--report artifacts/checks/grid-learning.json` 寫出的報告；只有明確指定 `--report` 才會寫到這裡，照上面的命令重跑不會改動它。檔內記有執行的電腦、每一步的 loss 與 validation、test 的 mAP50，並以 SHA-256 綁定產生它的程式。紀錄過期時，維護工具 `record_evidence.py --run` 就這樣重產它，再由 `scripts/render_learning_evidence.py` 依紀錄重畫網站上的 loss 曲線與預測圖；這些結果在 [7.4 三步訓練與診斷](docs/lessons/07-training.md)的 160 步補充實驗有逐項解說。

要用較大的預算，把 `--steps 160 --samples 32` 換成 `--epochs 20 --samples 1024`：1024 張訓練圖各用過 20 次，共 2560 次參數更新。`--epochs` 與 `--steps` 只能擇一，同時指定會直接報錯。再用 `--output` 另指定目錄，前一次的結果就會留著對照，例如 `.venv-model/bin/python -m miniyolo.train --epochs 20 --samples 1024 --device cpu --output artifacts/runs/grid-learning-20-epochs`。GPU 環境先依 [PyTorch 官方安裝選擇器](https://pytorch.org/get-started/locally/) 安裝相容的 CUDA build，確認 `torch.cuda.is_available()` 為 `True`，再把 `--device cpu` 改成 `--device cuda`；這個指令沒有在 GPU 上實際跑過，GPU 上做過的檢查見下一段。

GPU 上的檢查只有兩項，都由手動啟動的 GitHub Actions 工作流程在 Modal 的一張 NVIDIA L4 上執行：一項是上面這個偵測器（GridDetector）訓練 40 步、中途存 checkpoint、換新的 container 讀回後續訓，結果和不中斷的訓練一致（[GPU／checkpoint 實測](docs/validation/gpu-smoke.md)）；另一項是第 20 章 ONNX／TensorRT 的 FP32、FP16 engine 與 PyTorch 輸出的一致性。push 或 PR 不會觸發這兩個工作流程。兩項都不是正式的速度比較，也沒有用真實資料完整訓練。

## 真實分類資料的短步檢查

這項檢查要另外下載資料：用 Fashion-MNIST（28×28 灰階、10 類的服飾商品圖，有衣服、鞋與包等；MIT 授權）確認分類管線接得上真實圖片。

```bash
python3 scripts/download_data.py fetch fashion-mnist
.venv-model/bin/python scripts/run_fashion_cnn.py --steps 2
```

第一行從官方來源下載四個原始 gzip 到 git 不追蹤的 `data/downloads/fashion-mnist/`，並用 `data/manifest.json` 記錄的大小與 SHA-256 核對；含授權檔的同一份資料也打包存在 Git LFS（`data/curated/`）。

Loader 讀取時會再核對大小、SHA-256 與 IDX 格式，把 28×28 灰階圖複製成三個相同的 channel，交給 `miniyolo` 的 TinyCNN（RGB 輸入的 VGG 風格小分類器）。資料固定切成 54k train／6k validation，test 用官方的 10k。預設的兩步只用 64 張訓練圖，validation 與 test 各評 128 張，確認權重有更新；兩步的結果不代表模型學好了。`--train-steps 1000 --eval-samples 0` 改用完整 54k 訓練圖跑 1000 步，並評估完整的 validation 與 test。結果寫在 `artifacts/runs/fashion-cnn/`。偵測資料的來源、大小與再散布限制見[資料規劃](docs/preparation/data.md)。

## 網站建置與發布

網站使用 Zensical，網站工具固定在 `requirements-docs.txt`。只預覽或檢查網站不需要模型套件、訓練資料或 GPU：

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

維護者另安裝 `requirements-test.txt` 後執行 pytest；一般 CPU 練習只需前述模型環境。

```bash
.venv-model/bin/python -m pip install -r requirements-test.txt
.venv-model/bin/python -m pytest tests/
```

教材修改按影響範圍核對易讀性、正確性與連貫性，PR 留發現、修正、測試和未驗證範圍；詳細閱讀、凍結稿與截圖使用短期 artifact。歷史審閱保留於[固定封存版本](reviews/README.md)，網站不再讀逐頁審閱指紋。來源同步、必要科學結果、嚴格建置、連結和下載入口仍須通過。

正式發布由手動 Pages workflow 執行。它先在 CPU 工作中核對或重跑完整 52 節實驗，再由只安裝網站工具的工作建站；整套成功結果保存在忽略的 `outputs/lesson-execution/`，以有序程式、共用輸入、套件與環境識別。純文字改動可沿用相同執行輸入的結果；缺少、過期或不完整就重跑，並保留真正的執行版本和日期。CI 在 PR/main 執行，依實際執行輸入決定跨平台範圍。

程式修改需要新的 Notebook 發布 tag：先執行 `build_lesson_notebooks.py --ref <新 tag>`，再用 `record_evidence.py` 檢查及更新受影響的科學紀錄與圖，核對正文數值並審閱修改。既有 tag 不移動，已發布 Notebook 保留原內容；新的環境格會使用淺 clone、略過其他 tag 與 LFS 資料。文字與發布工具維護可以沿用現有教材 tag。

發布後檢查公開 `build-info.json` 的網站版本、實際執行版本與 CPU 範圍，點開受影響頁面及 Colab 入口。新版 tag 另以 Verify published lessons 核對公開內容；`verify_release.py save --run <編號>` 將結果存入忽略的 `outputs/release-verification/`，不再覆寫或提交整包發布 log。已發布 tag 的完整 commit 清單獨立保存在 `lesson-releases.json`，不依賴上一輪 log。操作與科學紀錄重建細節見[發布與帳號設定](docs/preparation/publish.md)。

## 目錄結構

- `miniyolo/`：資料、模型、targets、loss、解碼、幾何、AP 與可選的訓練 CLI。
- `lesson_cases/`、`notebooks/`：每節可獨立執行的實驗；各節 notebook 的最後一格與同名程式逐字相同。`notebooks/00_environment_check.ipynb` 另用來檢查 Colab 環境。
- `docs/`：網站內容；各節教材在 `docs/lessons/`，示意圖與實際結果圖在 `docs/assets/diagrams/`。
- `artifacts/checks/`：執行與檢查紀錄。每節一份的執行紀錄在 `artifacts/checks/curriculum/`。這些紀錄和補充實驗、GPU 的紀錄都記下執行的電腦或 GPU，並以 SHA-256 綁定各自的程式；補充與 GPU 紀錄的清單在 `scripts/evidence_records.py`。其他輸出寫在 git 不追蹤的 `artifacts/runs/` 等位置。
- `reviews/`：歷史封存說明及必要科學來源；新的詳細審閱使用短期 artifact，PR 留摘要。
- `scripts/`：資料下載、補充實驗、執行紀錄與網站檢查的工具。
- `section-map.json`：52 節的網頁、notebook 與固定 tag 對應表；`status: ready` 只表示網頁、程式與 notebook 都在，不代表已審查或發布。`lesson-releases.json` 另外保存已發布 tag 的固定 commit。
