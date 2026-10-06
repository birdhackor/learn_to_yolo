# Zensical、GitHub Pages 與 Colab

本站使用 **Zensical + Markdown 閱讀頁 + 獨立 `.ipynb`**。這頁記錄網站的配置、建置與發布流程，以及 Colab notebook 怎麼固定到教材版本。網站設定只放在 `zensical.toml`，不使用 `mkdocs.yml`（`validate_site.py` 會檢查）。

## 閱讀與執行分開

52 節教材以 Markdown 保存直覺、公式、SVG、關鍵程式與實驗結果：第 0–20 章保留 42 節主線，第 21–23 章新增 10 節 ViT／DINO 選讀支線，從 15.1 attention 分岔。GitHub Pages 提供純靜態閱讀；每節另有固定版本的 Colab 連結。讀者不執行 Python，也能看到例子與答案。

建置網站不執行 notebook、不安裝 PyTorch、不下載 dataset。原始資料、模型權重與完整訓練產物放在 `docs/` 外；網頁只包含選出的文字與小型結果圖。

## 版本與原生設定

文件依賴固定為 `zensical==0.0.67`，Python 下限由套件 metadata 宣告為 3.10；GitHub Actions 用 Python 3.12，本機也用 3.12，與 CI 相同。本頁〈建置與預覽〉一節的 `python3` 既建立 `.venv-docs`，也直接執行檢查腳本，所以先用 `python3 --version` 確認它是 3.12（不是的話，把這些指令裡的 `python3` 換成 3.12 的直譯器，例如 `python3.12`）：Zensical 在 3.10 就裝得起來，但 `validate_lessons.py` 與 `validate_site.py` 用到 Python 3.11 才加入的標準函式庫 `tomllib`，在更舊的 Python 會停在 `ModuleNotFoundError: No module named 'tomllib'`。這些版本只決定網站的建置與檢查，不代表 CUDA 或訓練程式的相容性。

```text
zensical.toml                   網站設定與完整小節導覽
requirements-docs.txt           固定文件建置版本
.github/workflows/pages.yml     GitHub Pages artifact 發布流程
.github/workflows/verify-release.yml
                                發布後從公開 tag 驗證網站、notebook 與 README 指令
overrides/                      主題覆寫（中文 404 頁）
docs/lessons/                   52 節閱讀頁
docs/assets/                    SVG 圖（含結果圖）、mathjax.js，以及 MathJax 與其字型
notebooks/                      各節獨立的 Colab notebook 與環境檢查 notebook
section-map.json                閱讀頁與 notebook 的配對，以及各節固定的 tag（source_ref）
site/                           產生的靜態網站，不進 Git
.cache/                         本地建置快取，不進 Git
```

`zensical.toml` 使用 `[project]`、`[project.theme]` 與 `[project.markdown_extensions]`。本站採用 modern 主題、`zh-TW`、深淺色模式、內建搜尋與快速換頁；沒有安裝 MkDocs 或 Material 外掛。

來源：[Zensical 安裝](https://zensical.org/docs/get-started/)、[原生設定](https://zensical.org/docs/setup/basics/)、[PyPI metadata](https://pypi.org/pypi/zensical/0.0.67/json)、[MkDocs 相容性](https://zensical.org/docs/compatibility/mkdocs/)。

## 搜尋與公式

Zensical 使用自己的搜尋引擎，搜尋語言由 `theme.language` 決定；MkDocs 與 Material for MkDocs 搜尋外掛專屬的選項，例如 `lang` 與 Lunr 的 `pipeline`，在這裡都不支援。`validate_site.py` 只確認搜尋索引的語言是 `zh-TW`、52 節都在索引裡，不會實際查詢；所以仍要在預覽網站搜尋幾個中文詞與英文詞，確認結果連到正確的教材頁。

官方文件說明搜尋介面目前沒有在地化，所以搜尋視窗裡有些文字是英文；這不影響多語搜尋。網站導覽設定為繁體中文，英文術語也保留在教材中。

公式使用 Arithmatex 標記與本地 MathJax 3.2.2 資產。快速換頁不會重新載入整個網站，所以 `mathjax.js` 訂閱 Zensical 在每次載入新頁時發出的 `document$`，對新頁內容重新排版。`validate_site.py` 只檢查正文裡的 `\(`、`\)`、`\[`、`\]` 都在 Arithmatex 標記內，不會在瀏覽器裡排版；瀏覽器檢查要包含直接開頁、點擊內部連結換頁與深淺色切換。

來源：[Zensical 搜尋](https://zensical.org/docs/setup/search/)、[快速換頁](https://zensical.org/docs/setup/navigation/)、[MathJax 與快速換頁的整合](https://zensical.org/docs/authoring/math/)。

## 建置與預覽

在 repository 根目錄：

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

從 `validate_preparation.py` 到 `validate_site.py` 的五個指令，就是 Pages workflow 的五個檢查與建置步驟，順序也相同（各步驟檢查什麼，見下一節）；推送前在本機跑過，就能先發現 Pages 建置會擋下的問題。最後的 `zensical serve` 建置並啟動本機預覽，預設網址是 `localhost:8000`。

`--clean` 清除建置快取，CI 每次從目前來源建站；`--strict` 在警告時中止。`site_url` 固定為 `https://birdhackor.github.io/learn_to_yolo/`，各節網址是其下的 `lessons/<id>/`。圖片與本地連結使用相對路徑，適用 Pages 的 `/learn_to_yolo/` 子路徑；`validate_site.py` 會檢查每個本地連結都指到產生的網站裡。

## Pages 發布

`.github/workflows/pages.yml`（workflow 名稱 **Publish Learn to YOLO**）只能手動觸發（`workflow_dispatch`），推送到 GitHub 不會自動發布。它的 `concurrency` 群組是 `github-pages`：同一時間只跑一次發布，進行中的不會被取消。

build job 在 `ubuntu-latest` 上執行，逾時上限 10 分鐘。它先用 `actions/checkout@v4` 取出程式，設定 `lfs: false`，不下載 LFS 物件；再用 `actions/setup-python@v5` 安裝 Python 3.12（pip 快取以 `requirements-docs.txt` 為準），執行 `python -m pip install -r requirements-docs.txt`。接著依序執行下面五步；任何一步失敗，後面的步驟都不會執行，網站也不會部署。

1. **Validate data manifests and notebook pairs**：`python scripts/validate_preparation.py`。`data/manifest.json` 的資料集 ID 不能重複，狀態為 `download-ready` 的資料集要有授權、檔名、大小、SHA-256 與 HTTPS 網址。`section-map.json` 列出 52 節課程，另有一筆 `kind` 為 `preparation` 的環境檢查 notebook（〈首頁〉連到的 `notebooks/00_environment_check.ipynb`）；每一筆的 ID 都不能重複、引用的資料集都要在 manifest 裡，就緒的還要有閱讀頁、notebook 與 `source_ref`。每本 notebook 都要是 nbformat 4 的 Python notebook，程式格都要能被 Python 解析。`docs/` 裡每個檔案都要小於 5 MiB，而且不能是權重（`.pt`、`.pth`、`.onnx`）、壓縮封裝（`.zip`、`.tar`、`.gz`）或 LFS pointer。
2. **Validate lessons and reader reviews**：`python scripts/validate_lessons.py`。`section-map.json` 裡 `kind` 為 `lesson` 的項目要與 `lesson_cases/` 的 ID 完全對應，而且都已就緒。每個課程頁的 Colab 連結與 notebook metadata 都要指向該節的 `source_ref`；notebook 最後一格要與 `lesson_cases/<節>.py` 逐字相同；頁面引用的 SVG 都要存在；標了 `data-excerpt` 的程式摘錄，標明的原檔要是 `lesson_cases/`、`miniyolo/` 或 `scripts/` 裡的 Python 檔，摘錄的每一行程式（不計註解、空行與 `...` 省略行，也不計縮排與空白多寡）都要出現在那個檔裡；沒有標記、卻有三行以上全部照抄這三個資料夾裡某個檔的 Python 區塊會被擋下。導覽裡的每一頁都要有審查紀錄，而且紀錄涵蓋該頁目前的文字與頁面上的 SVG 圖，課程頁還包括該節程式與它 import 的模組；頁尾的執行紀錄區塊與 Colab 連結裡的 tag 不算在內。`docs/assets/diagrams/` 的每張 SVG 都要有 `viewBox` 與 `<title>`。
3. **Validate current lesson execution evidence**：`python scripts/validate_curriculum_evidence.py`。52 節的執行紀錄都要標示通過，而且用 SHA-256 綁定的程式（該節程式與它 import 的 repo 檔案）都沒有改過；紀錄要寫明產生它的電腦。notebook 最後一格不能有錯誤輸出，印出的文字要與紀錄的 stdout 一字不差；課程頁上執行紀錄區塊的開始標記與結束標記都要恰好一個，紀錄的日期、PyTorch 版本與 CPU 型號也要出現在頁面上（比對的是整頁文字，不限區塊內）。補充紀錄（各節的補充實驗與〈資料規劃〉的 Fashion-MNIST 核對在 CPU 上的紀錄）與 GPU 紀錄的清單在 `scripts/evidence_records.py`，它們綁定的程式也都不能改過；補充紀錄同樣要寫明電腦，其中 8.2 的 1600 步訓練、影片檔與 Fashion-MNIST 三份還要通過各自的內容檢查（例如 loss 的筆數等於步數）；GPU 紀錄要標示通過，並記載 Modal 上的 app 已確認停止。所有紀錄裡的數字都不能是 NaN 或無限大。這一步只比對一致性，不重跑實驗，也不需要 PyTorch。
4. **Build Zensical pages**：`zensical build --clean --strict`，產生 `site/`。
5. **Validate generated pages and search index**：`python scripts/validate_site.py`。repository 裡不能有 `mkdocs.yml`；每頁都要由 `requirements-docs.txt` 固定的 Zensical 版本產生；本地連結、圖片與 script 都要指到 `site/` 裡存在的檔案，連到的錨點也要存在。搜尋索引的語言要是 `zh-TW`，52 節都要在索引裡；每節頁面都要有指向該節 `source_ref` 的 Colab 連結。Markdown 要照原意轉換：正文裡不能有被誤當成 HTML 標籤的文字，也不能留下沒轉換的表格、摺疊區塊（`???`），或 Arithmatex 標記外的數學式符號；網址都要成為連結；`docs/` 裡每個表格的各列欄數都要與表頭相同；編號清單裡，項目底下空一行之後的段落要比編號多縮排 4 格，否則清單會在那裡斷開，下一項重新從 1 編號。最後，`site/` 裡不能有符號連結、LFS pointer，或 `.pt`、`.pth`、`.onnx`、`.ipynb`、`.zip`、`.tar`、`.gz` 檔。

五步都通過後，`actions/upload-pages-artifact@v3` 把 `site/` 上傳成 Pages artifact。deploy job 要等 build 成功才開始（`needs: build`），逾時上限 5 分鐘；它在 `github-pages` environment 用 `actions/deploy-pages@v4` 部署，environment 顯示的網址取自這一步輸出的 `page_url`。

build job 的 `GITHUB_TOKEN` 只有 workflow 層級設定的 `contents: read`；deploy job 自己宣告權限，只有 `pages: write` 與 `id-token: write`。網站使用 repository 已設定的 Pages 來源（GitHub Actions）與 `github-pages` environment，不需要新增 PAT 或 secret，也不使用 gh-pages 分支。推送與部署是分開的：推送後要到 Actions 手動執行這個 workflow，確認 build 與 deploy 都成功，再打開公開網址檢查。

公開 Pages 與 repository 的存取條件見 GitHub 官方文件；將 repository 改為 private 前，要先核對方案與網站 visibility。GitHub Pages 不能使用 Git LFS，大型資料與 LFS 物件也不經 Pages 發布：workflow 不下載 LFS 物件；第 1 步在 `docs/`、第 5 步在 `site/` 擋下權重、壓縮封裝與 LFS pointer，第 5 步也不允許符號連結。

來源：[Zensical 發布指南](https://zensical.org/docs/publish-your-site/)、[GitHub Pages 自訂 workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)、[Pages 限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)、[Git LFS 與 Pages](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

## Colab 與教材版本

各節頁首的 Colab 按鈕與環境格（各節 notebook 的第二格）都固定到同一個發布 tag，也就是 `section-map.json` 裡該節的 `source_ref`。`scripts/build_lesson_notebooks.py --ref <tag>` 會一起寫入 `source_ref`、notebook metadata、環境格要 clone 的 tag 與頁面上的 Colab 網址；`validate_lessons.py` 檢查頁面的 Colab 連結與 notebook metadata 都對上 `source_ref`，`validate_site.py` 再檢查產生的頁面。Colab 從 GitHub 讀取這個 tag 上的 notebook，所以 tag 要先推到 GitHub。

本版 52 節與 notebook 固定到 `lessons-v0.6.1`；公開入口驗證以保存紀錄中的 tag 與結果為準。既有 `lessons-v0.5.0` 的公開網站與環境格驗證保留原本範圍，不表示新增的 10 節已在 Colab 或公開網站測過。支線的預設實驗有 CPU 實測，沒有新增 GPU 訓練或 DINOv3 實作；23.1 的官方預訓練特徵操作另選，第一次需下載約 84.2 MiB 官方權重；它沿用 notebook 的 PyTorch／NumPy／Pillow，預設實驗不自動取得官方模型。

各節 notebook 的最後一格是該節的完整實驗程式（與 `lesson_cases/<節>.py` 逐字相同），存著執行紀錄 `artifacts/checks/curriculum/<節>.json` 裡的 CPU 輸出。`zensical build` 不讀 `notebooks/`，不會改動這些輸出。

只改網站主題、導覽或建置設定時，不必發新的 tag，重新建置並部署網站。各節 notebook 與它執行的程式都取自發布 tag，所以改到其中任何一部分都要發新的 tag，例如環境格與它固定的套件版本（由 `build_lesson_notebooks.py` 的 `bootstrap()` 產生）、`lesson_cases/`、它們 import 的模組，或可選實驗執行的程式。發新 tag 時，用 `scripts/build_lesson_notebooks.py --ref <新 tag>` 重建 notebook 與各節的 Colab 連結，並用 `scripts/record_evidence.py` 列出並重產過期的執行紀錄。已發布的 tag 不移動、不覆寫。

讀者要有 Google 帳號才能在 Colab 執行。各節實驗只用 CPU，不必選 GPU；Colab 會不會分配 GPU，依當時的配額與可用性而定，不保證有免費 GPU。從 GitHub 開啟的 notebook 是一份可以編輯的檢視，不會覆寫 repository；要保留修改，選 **File → Save a copy in Drive**。副本只有 notebook 本身，不含 runtime 裡的檔案（例如可選實驗寫在 `artifacts/runs/` 的 checkpoint 與圖）；runtime 被回收後這些檔案就不見了，需要的話要先下載。

環境格用 `git clone --depth 1 --branch <tag>` 取得固定版本的程式。clone 時設定 `GIT_LFS_SKIP_SMUDGE=1`，不下載 LFS 資料，之後再確認取得的正是這個 tag。PyTorch 不是 2.9.1 時改裝 2.9.1 的 CPU 版；已經是 2.9.1 就沿用，CPU 或 CUDA 版都可以。若這個工作階段在改裝前已經載入 torch，環境格改裝完就會停下，提示重新啟動工作階段，再從第一格執行。接著安裝固定版本的 numpy、Pillow 與 matplotlib；第 20 章另裝 onnx 與 onnxruntime。各節範例不用預訓練權重、不下載資料，也不依賴其他節的 runtime 或作者的 Google Drive。

環境格的這些分支由 `tests/test_notebook_bootstrap.py` 測試：測試執行 `bootstrap()` 產生的環境格，把 pip、git 與套件版本查詢換成假的替代品（stub），不連網，也不在 Colab 上執行。網站發布後，手動啟動的 workflow **Verify published lessons**（`.github/workflows/verify-release.yml`）在 GitHub 的 Linux runner 上，從公開 tag 的全新 clone 實際執行：每種情況各用一個新的虛擬環境，照原樣執行環境格與最後一格，第 0 章試沒有 PyTorch、裝著其他版本、已是 2.9.1、其他版本已經載入四種情況，另加第 20 章。它也核對公開網站每節的 Colab 連結都開這個 tag 的 notebook，而那本 notebook 的環境格固定在這個 tag、最後一格就是該節程式。兩項結果在發布後存進 main 的 `artifacts/checks/curriculum-release-bootstrap.json` 與 `artifacts/checks/curriculum-publication.json`。各節的 CPU 執行紀錄是在 Colab 以外的電腦上直接執行 `lesson_cases/` 的程式產生的；notebook 沒有在 Google Colab 的託管 runtime 上逐節執行過。這些紀錄、測試與發布後的驗證都不涵蓋 Google 登入、Colab 分配的 GPU，或在 GPU 上的完整訓練。

來源：[Colab 官方 GitHub notebook 示範](https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)、[Colab FAQ](https://research.google.com/colaboratory/faq.html)。操作步驟見〈[發布與帳號設定](../preparation/publish.md)〉；各種執行紀錄能支持哪些結論、哪些事沒有測，見〈[驗證範圍](../status.md)〉。
