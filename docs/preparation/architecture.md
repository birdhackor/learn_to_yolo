# 網頁與 Colab 的前置規格

## 發布架構

採用 **Zensical** 的 modern 主題：Markdown 與小型圖片建成純靜態 GitHub Pages。`zensical.toml` 管理中文導覽、內建搜尋、深淺色模式與快速換頁；`requirements-docs.txt` 固定建置版本。獨立 `.ipynb` 放在同一個 repository，網頁以 Colab URL 連過去。網站依賴與模型依賴分開，建置網站不需要 GPU、PyTorch 或訓練資料。

本次配對範圍是 52 節：原第 0–20 章的 42 節主線，以及從 15.1 attention 分岔的第 21–23 章 10 節 ViT／DINO 選讀支線。本版教材與 notebook 固定為 `lessons-v0.6.1`；公開入口驗證以保存紀錄中的 tag 與結果為準，既有 `lessons-v0.5.0` 紀錄保留原本範圍。

Pages 由手動啟動的 `.github/workflows/pages.yml` 建置：先跑 `scripts/` 裡的 `validate_preparation.py`、`validate_lessons.py` 與 `validate_curriculum_evidence.py`，再執行 `zensical build --clean --strict`，最後用 `validate_site.py` 檢查產生的網頁；任何一步失敗就不上傳。網站發布後，手動啟動的 `.github/workflows/verify-release.yml`（`scripts/verify_release.py`）從公開的 tag（已發布的固定版本標籤）再檢查網站、notebook 環境格與 README 的指令，檢查範圍與結果見〈[全套實驗與審查](../validation/curriculum.md)〉。

```text
zensical.toml                 網站設定、主題與小節導覽
requirements-docs.txt         網站建置的固定版本
overrides/                    主題覆寫（404 頁）
docs/                         網頁文字、圖與保存下來的實驗結果
docs/lessons/                 52 節閱讀頁，頁尾附實際執行紀錄
docs/assets/                  示意圖、實驗結果圖與公式用的 MathJax
docs/validation/              全套實驗與審查、GPU／checkpoint 實測
docs/preparation/             資料規劃、發布與帳號設定、網頁與 Colab 規格
docs/planning/                課程大綱、公開課程研究、讀者與學習心得
docs/research/                基礎資料、偵測資料、Git LFS、Pages 與 Colab、版本來源查證
notebooks/                    每節一本可獨立開啟的 Colab notebook，另有一本環境檢查
lesson_cases/                 各節完整實驗程式，與 notebook 最後一格逐字相同
miniyolo/                     資料、模型、targets、loss、推論、評估與訓練套件
tests/                        核心計算、checkpoint 續訓、8.2 節結果圖的 TP／FP 判定與圖說，以及 notebook 環境格的測試
requirements-model.txt        模型與實驗的固定版本
requirements-video.txt        選用的影片檔讀寫套件（OpenCV），18、19 節讀寫真實影片時用
scripts/                      資料下載與 LFS 封裝、notebook 配對、圖片推論、補充實驗、執行紀錄、審查涵蓋、網站檢查與發布後驗證
section-map.json              小節 ID 與閱讀頁、notebook、固定版本的配對
data/manifest.json            資料來源、校驗碼與 LFS 封裝資訊
data/hosted-lfs-run.json      發布 LFS 封裝的工作流程各步驟的結果
data/*-lfs-verification.json  LFS 在本地與 GitHub 上的驗證紀錄
data/curated/                 授權清楚的精選資料，使用 LFS
data/licenses/                精選資料的原授權文字
data/downloads/               本地或 Colab 下載快取，不進 Git
artifacts/checks/             執行紀錄、發布驗證與其他查核紀錄
artifacts/checks/curriculum/  每節一份執行紀錄 <節>.json、彙總 index.json，另有多數補充紀錄與第 20 章 GPU 紀錄
artifacts/*.png               部分小節程式畫出的圖（例如 01-small-cnn.png），不進 Git
artifacts/lesson-*/           部分小節程式寫出的檔案（例如 lesson-17/），不進 Git
artifacts/runs/               補充實驗、訓練、推論與檢查工具的輸出，不進 Git
artifacts/checkpoints/        .gitattributes 指定給 LFS 的模型權重路徑，目錄不存在
artifacts/exports/            .gitattributes 指定給 LFS 的匯出模型路徑，目錄不存在
reviews/                      網站每一頁的審查紀錄；coverage.json 記下每份審查涵蓋的內容
.github/workflows/            手動啟動的 Pages 發布、發布後驗證、GPU 紀錄與 LFS 資料工作流程
requirements-modal.txt        GPU 工作流程在 GitHub Actions 上安裝的 Modal 用戶端，不含 PyTorch
requirements-gpu.in           GPU 紀錄的套件：沿用 requirements-model.txt，PyTorch 改用 CUDA 版
requirements-gpu.lock         由 requirements-gpu.in 產生、附雜湊的完整清單，Modal 的 GPU 環境照它安裝（第 20 章另加 TensorRT）
```

補充紀錄（各節的補充實驗，以及〈[資料規劃](data.md)〉的 Fashion-MNIST 40 步核對）與兩份 GPU 紀錄各在哪個路徑、綁定哪些程式，列在 `scripts/evidence_records.py`；紀錄過期時怎麼重產，見〈[發布與帳號設定](publish.md)〉。

## 每個小節的出版約定

1. 網頁先寫本節問題、前置知識與完成條件。
2. 說明、公式的符號／shape、示意圖與關鍵程式片段放在網頁。從 `lesson_cases/`、`miniyolo/` 或 `scripts/` 逐字摘錄的程式區塊要標出來源檔：把區塊開頭的 `python` 換成 `{ .python data-excerpt="lesson_cases/07-loss.py" }`，引號裡寫從 repository 根目錄算起的路徑。`validate_lessons.py` 檢查區塊裡的每行程式（不計註解、空行與 `...` 省略行）都出自那個檔；開頭仍是 `python` 的區塊若有三行以上程式、每行都能在這三個目錄的同一個檔裡找到，檢查也會失敗，要求補上標記。
    這是 Markdown 的程式區塊標記，不是修改 Python 程式。最小例子如下；三個反引號包住程式，來源路徑寫在開頭那一行：

    ````text
    ``` { .python data-excerpt="lesson_cases/07-loss.py" }
    prediction = torch.zeros(1, 4, 4, 7, requires_grad=True)
    ```
    ````

3. 保存固定實驗的設定、實際輸出與觀察，說明結果能支持什麼；網頁能直接閱讀這些成果。比較優化效果的對照需使用相同資料與訓練預算；版本機制的小示範不是完整模型的效果比較。
4. 頁首附「在 Colab 執行本節」按鈕；頁尾的「實際執行紀錄」寫明那次執行的日期、CPU 與執行緒數、PyTorch 版本，並附那次印出的輸出。各節範例的資料都由程式當場產生，不必下載。
5. notebook 設計成可在全新 runtime 從第一格依序執行，不依賴上一節的隱藏狀態或自己的 Google Drive 路徑：環境格（每本 notebook 的第二格）先取得固定版本的程式與套件。
6. 正式教材的說明與 notebook 實驗對應同一個固定的 release tag。網站主題與建置工具可以獨立更新，不覆寫教材 tag；修改實驗或教學結果時才配對新版本。下載的原始資料固定 checksum 與 split，程式產生的資料固定 seed；不讓讀者看到新版說明卻執行另一版程式。

Colab URL 格式：

```text
https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/<ref>/notebooks/<section>.ipynb
```

`section-map.json` 記錄 52 個教學小節 ID，各自配對網頁、獨立 notebook 與固定的 release tag（`source_ref`）；另有一筆首頁連結的 Colab 環境檢查 notebook，指向 `main`。一節標成 `ready`，只表示閱讀頁、實驗程式與 notebook 都存在；執行紀錄在 `artifacts/checks/curriculum/`，審查紀錄在 `reviews/`，`ready` 本身不代表審查、發布、GPU 訓練或效果驗證。網址與檔名都用小節 ID，不用標題，改標題不會讓外部連結失效。

`scripts/build_lesson_notebooks.py --ref <tag>` 產生 52 本 notebook，並讓各頁的 Colab 連結與 `section-map.json` 都指向這個 tag。每本 notebook 有四格：說明、環境格、實驗說明、完整實驗程式；最後一格與 `lesson_cases/<節>.py` 逐字相同，並存著執行紀錄的輸出。環境格 clone 這個 tag 並確認版本相符；PyTorch 不是 2.9.1 時改裝 2.9.1 的 CPU 版（已經是 2.9.1 就沿用，CPU 或 CUDA 版都可以），若改裝前 torch 已載入，改裝後就停下，要求重新啟動工作階段。各節實驗都只用 CPU。環境格在這幾種情況下的行為有單元測試（`tests/test_notebook_bootstrap.py`，pip 與 git 都換成假的，不實際安裝或 clone）。

支線共用的小 ViT 與合成資料在 `miniyolo/vision_transformer.py`、`miniyolo/vision_data.py`；自監督核心在 `miniyolo/self_distillation.py`，不改既有 CNN、ResNet 或 MiniYOLO 的架構與訓練入口。預設課堂資料由程式生成，不下載權重；23.1 的官方預訓練特徵操作另選，不是 notebook 啟動的必要條件。支線目前有 CPU 實測，沒有新增 GPU 訓練或 DINOv3 實作。

## 純閱讀模式如何保留成果

網站不在讀者瀏覽時執行 Python。訓練曲線、框圖、shape 表與結果說明，都在固定實驗跑完後存成文字、表格與 SVG、PNG 等圖，再建置進網頁；各節頁尾的「實際執行紀錄」由 `scripts/verify_curriculum.py` 從 `artifacts/checks/curriculum/<節>.json` 寫入。notebook 只在最後一格存執行紀錄的文字輸出，不存圖片的 base64，避免 Git 與 Colab 變慢。

大權重、原始資料、影片與 notebook 都不放進 Pages artifact。LFS 檔在 Git 裡只是 pointer（只記雜湊與大小的小文字檔），不能當圖片或下載內容；需要閱讀的圖使用普通 Git 的小型資源。`validate_preparation.py` 確認 `docs/` 裡每個檔案都小於 5 MiB，沒有權重檔（`.pt`、`.pth`、`.onnx`）、打包檔（`.zip`、`.tar`、`.gz`）或 LFS pointer；`validate_site.py` 在建置結果裡再擋一次這些類型與 `.ipynb`。發布流程只上傳 `site/`。

## Colab 取得資料與 LFS 的約定

各節 notebook 不下載資料，也不用預訓練權重，所以不需要 LFS：環境格 clone 時設 `GIT_LFS_SKIP_SMUDGE=1`，LFS 檔只留下 pointer，不下載內容。讀者不必設定 LFS，也不需要付費帳號。

需要真實資料時（例如 `scripts/run_fashion_cnn.py` 的 Fashion-MNIST 分類檢查），`scripts/download_data.py fetch <資料集 ID>` 從 `data/manifest.json` 記錄的已查核來源下載到快取（預設 `data/downloads/`，不進 Git），並核對 bytes 與 SHA-256；`--asset` 只下載指定檔案，`--output` 可改到 Colab 的 `/content/data` 等位置。下載的檔案存在 `<output>/<資料集 ID>/`；`run_fashion_cnn.py` 預設讀 `data/downloads/fashion-mnist/`，所以用 `--output /content/data` 下載時，執行它要加 `--data-root /content/data/fashion-mnist`。Colab runtime 中斷後可能要重新下載，所以只取自己需要的部分。原始資料與衍生 split 分開記錄。

repository 裡唯一的 LFS 檔是 `data/curated/fashion-mnist-v1.tar`（Fashion-MNIST 四個原始檔加 MIT 授權），52 節教材沒有用到。要用時才裝好 `git-lfs`，以 `git lfs pull --include="data/curated/fashion-mnist-v1.tar" --exclude=""` 只取這一個檔；完整步驟見〈[資料規劃](data.md)〉。

## 官方參考

- [GitHub Pages 自訂 workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [Colab 官方 GitHub notebook 範例](https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Zensical](https://zensical.org/docs/get-started/)
- [Zensical 導覽設定](https://zensical.org/docs/setup/navigation/)
