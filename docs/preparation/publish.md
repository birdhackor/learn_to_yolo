# GitHub Pages、Colab 與 Git LFS：操作步驟

Learn to YOLO 的 52 節教材各配一本 Colab notebook；網站用 Zensical 建成靜態網頁，由 GitHub Actions 發布到 GitHub Pages。repository `birdhackor/learn_to_yolo` 是 public，預設分支是 main。本頁依序說明本地預覽、推送、Pages 設定、Colab 入口與 Git LFS，最後是發布新版教材的完整流程。

## 1. 本地預覽與授權選擇

網站設定在 `zensical.toml`，建置工具的版本固定在 `requirements-docs.txt`。用 Python 3.12（和 Pages workflow 相同），在 repo 根目錄建立文件環境並預覽：

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements-docs.txt
.venv-docs/bin/zensical serve
```

用瀏覽器打開終端機顯示的本地網址。Windows 把 `.venv-docs/bin/` 換成 `.venv-docs\Scripts\`。預覽不需要 GPU、PyTorch 或資料集。只建置不預覽時執行 `.venv-docs/bin/zensical build --clean --strict`：`--clean` 清除建置快取，`--strict` 遇到警告就中止。和 Pages 相同的整套檢查，指令見第 6 節的來源、證據與網站檢查。

本頁其他直接用 `python3` 執行的指令，也要用 Python 3.12。`validate_lessons.py`、`validate_site.py` 與 `verify_release.py` 用到 Python 3.11 才加入的標準庫 `tomllib`，在更舊的 Python 會停在 `ModuleNotFoundError: No module named 'tomllib'`。Zensical 只要求 Python 3.10 以上，所以預覽成功，不代表這些檢查跑得動。

repo 沒有 LICENSE 檔。程式與原創教材用什麼條款開放，由 owner 決定，例如程式用 MIT、原創教材用 CC BY 4.0；決定後在 repo 根目錄加上 LICENSE。第三方資料一律遵照各自來源的授權。

## 2. 推送到 GitHub

### 推送認證

用你自己的 GitHub 認證推送：SSH key、Git 的 credential helper，或 fine-grained personal access token（PAT）都可以。用 PAT 推送被拒時，檢查它的權限：repository access 要包含 `learn_to_yolo`，Contents 要是 Read and write；推送 `.github/workflows/` 裡的修改還需要 Workflows: Read and write；透過 API 或 GitHub CLI 啟動 workflow 需要 Actions: Read and write。在 GitHub 網頁上按 Run workflow，用的是你登入帳號的權限。

能讀取 repo、API 回 200 或有 push 權限，都不能證明 LFS 檔案已經上傳；要照第 5 節下載回來核對 checksum 才算數。

**選用：`scripts/github_auth.py`。** 環境只能用一個完整的 HTTP Authorization header（例如 `Basic …`）提供認證時，把 header 的值設成環境變數 `GIT_LFS_AUTHORIZATION`，再用這個 helper 包住 git 指令：

```bash
python3 scripts/github_auth.py git ls-remote origin HEAD
python3 scripts/github_auth.py git push origin HEAD:main
```

helper 透過子程序的暫時 Git 設定（`GIT_CONFIG_*` 環境變數）把 header 交給 Git 與 Git LFS，不寫進檔案、Git config 或命令列參數，並遮蔽輸出裡的 token、Authorization header 與網址裡的帳密。header 只套用在 `https://github.com/birdhackor/learn_to_yolo.git`，所以 `origin` 要是這個 HTTPS 網址；設了這個變數時，helper 也會停用 credential helper。沒設時，它照常執行 git、沿用你平常的 Git 認證，只是不會跳出帳密提示。

PAT 與 `GIT_LFS_AUTHORIZATION` 的值都不要寫進 remote URL（例如 `https://<token>@github.com/…`）、repo 裡的檔案或 Git config：remote URL 與 Git config 會以明文留在設定檔裡，repo 裡的檔案則可能隨 commit 公開。

### commit 與 push

確認 diff 後 commit 並推送。下面的指令不會強制覆寫遠端；即使目前的分支不是 main，也會把目前的 commit 推到遠端 main：

```bash
git status --short
git diff --check
git add -A
git diff --cached --check
git diff --cached
git commit -m "Update Learn to YOLO site"
git push origin HEAD:main
```

`work/`、`outputs/`、`data/downloads/`、`data/processed/`、`artifacts/runs/`、`site/` 與建置快取都列在 `.gitignore`，不會被 commit。`git diff --cached` 顯示即將 commit 的實際修改；commit 前看一遍已暫存的檔案，不要夾帶無關的修改。遠端 main 有新的 commit 時，照一般 Git 流程同步、解決衝突，不要 force push。

## 3. 啟用 GitHub Pages

repo 的 Pages 來源已設為 GitHub Actions。第一次設定（或重新設定）時：

1. 開啟 repository：`https://github.com/birdhackor/learn_to_yolo`。
2. **Settings → Pages → Build and deployment → Source** 選 **GitHub Actions**。
3. Actions 被停用時，到 **Settings → Actions → General** 允許 workflow 使用的 GitHub 官方 actions。workflow 自己宣告需要的權限，不必為它另外建立 PAT，也不必把全部 workflow 的權限設成 write。
4. **Settings → Environments → github-pages**：已有部署分支規則時，確認 main 可以部署；也可以把部署來源限制為 main。
5. **Actions → Publish Learn to YOLO → Run workflow**，選 main 後執行。
6. 等 execute、build 與 deploy 都成功，從 deployment URL 或 Settings → Pages 開啟網站。

網址是 **<https://birdhackor.github.io/learn_to_yolo/>**。設定好之後，每次發布只需要第 5、6 步。

CPU 工作先核對或執行完整實驗 bundle；build job 再核對 bundle、寫入版本資訊，並依序執行下面各項，任何一項失敗都不會部署：

- `validate_preparation.py`：資料 manifest、頁面與 notebook 的配對、notebook 格式、網站檔案的大小與類型。
- `validate_lessons.py`：課程頁、頁面裡的程式摘錄與程式一致、固定版本的 notebook、SVG。
- `validate_curriculum_evidence.py`：每份執行紀錄都對得上目前的程式、notebook 與頁面。
- `zensical build --clean --strict`。
- `publication.py`：核對完整成功的 CPU bundle，寫入網站與實際執行版本。
- `validate_site.py --require-publication`：Zensical 版本與 `requirements-docs.txt` 相符、連結與錨點、Colab 配對、搜尋語言、Markdown 是否照原意轉成表格／編號清單／摺疊區塊／數學／連結、有沒有文字被誤當成 HTML 標籤，以及 artifact 邊界（`site/` 裡沒有 symlink 與 LFS pointer，也沒有 `.pt`、`.pth`、`.onnx`、`.ipynb`、`.zip`、`.tar`、`.gz` 檔）。

建置只上傳 `site/`，不下載 LFS 資料、不安裝 PyTorch；部署用 Actions artifact，不需要 gh-pages 分支。公式用的 MathJax 3.2.2 與它的字型放在 `docs/assets/vendor/mathjax/`，隨網站提供，並附上 MathJax 的 Apache 2.0 授權（來源校驗見 `artifacts/checks/mathjax-vendor.json`）。

Pages workflow 整體只有 `contents: read` 權限，只有 deploy job 另外宣告 `pages: write` 與 `id-token: write`。網站、LFS 資料、兩項 GPU 檢查與發布後驗證仍手動啟動。另有 CI 在 PR/main 執行離線測試及建站；push 不會發布網站或啟動 GPU。

public repo 用免費方案就能使用 Pages，也不需要自訂網域。repo 若改成 private，要另外核對帳號方案與網站的可見範圍。

## 4. 驗證 Colab 入口

1. 開啟環境檢查 notebook：
   `https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/main/notebooks/00_environment_check.ipynb`。
2. 登入 Google 帳號；開啟公開的 notebook 不需要授權 GitHub 寫入。
3. 可以先用 CPU 執行；要測 GPU 時，選 **Runtime → Change runtime type → GPU**，實際選項以 Colab 當下的介面為準。
4. 執行全部 cell，查看 Python、PyTorch、CUDA 與 git 的狀態。免費 GPU 不一定分配得到；這份 notebook 不訓練模型。
5. 要保存自己的修改，選 **File → Save a copy in Drive**。notebook 副本不包含 runtime 裡的資料檔與權重，這些要另外保存。

每節網頁的 Colab 按鈕開啟發布 tag 的 notebook，讀者同樣可以照第 5 步另存副本。Pages 建置前的檢查涵蓋 notebook 的格式、最後一格與 `lesson_cases/` 逐字相同，以及存著的輸出與執行紀錄一致；環境格的各個分支（沒裝 PyTorch、版本相符、版本不同、需要重新啟動工作階段、clone 到別的版本）有單元測試（`tests/test_notebook_bootstrap.py`）；發布後，第 6 節第 11 步的 workflow 也會從公開的 tag，在 GitHub 的 Linux runner 上實際執行第 0 章與第 20 章 notebook 的環境格與最後一格。這些檢查都不經過 Colab：Google 登入、Colab 的託管 runtime 與它可能分配的 GPU，只有實際在 Colab 開啟才測得到。

## 5. Git LFS

`.gitattributes` 只把三個路徑交給 LFS：

```text
data/curated/**
artifacts/checkpoints/**
artifacts/exports/**
```

一般圖片、notebook、Markdown 與 manifest 都放在一般 Git。`data/curated/` 只放可再散布的封裝，不放說明文件；授權與索引放在 `data/licenses/` 與 `data/manifest.json`。後兩個路徑保留給精選的 checkpoint 與匯出模型，repo 裡沒有這類檔案。

LFS 裡放著 Fashion-MNIST 的封裝 `data/curated/fashion-mnist-v1.tar`（四個原始 gzip 加上游的 MIT 授權）。它由 GitHub Actions 上傳，再從空的 LFS 快取下載回來核對（`data/remote-lfs-verification.json`）。另外在隔離的本地 repository，用 Fashion-MNIST 的 **26,421,880 bytes** 訓練圖與 Penn-Fudan 的 **53,723,336 bytes** 封裝測過 add → LFS pointer → object fsck → checkout 還原，SHA-256 都相符（`data/local-lfs-verification.json`）；Penn-Fudan 只用於這項本地測試，不放進公開 repo。

### 首次使用遠端 LFS

1. 安裝 [Git LFS](https://git-lfs.com/)，用 `git lfs version` 確認。
2. 照第 2 節設定好推送認證，再在 repo 裡執行 `git lfs install --local`。`.gitattributes` 已指定上面三個路徑；不要把全部 PNG／JPG 改成 LFS。
3. 到 GitHub 帳號的 **Settings → Billing & licensing**（介面也可能顯示 Billing & plans）查看 LFS 用量與 budget。LFS 檔案計入 repo owner 的 storage，讀者下載也計入 owner 的頻寬。
4. Free／Pro 帳號包含 **10 GiB storage＋10 GiB 下載頻寬**，單檔上限 2 GB。這是整個帳號的額度，不是每個 repo 各一份；超額時怎麼處理，由帳號的付款方式與 budget 設定決定。GitHub 的規則會調整，以帳單頁為準。先決定預算；在包含額度內使用 LFS，不必先開通付費。
5. 要發布新的 asset 時（授權必須允許再散布），先在 `data/manifest.json` 的 `lfs_assets` 照既有 `fashion-mnist-v1` 項目的格式加一筆，至少要有 `id`、`path`、`bytes`（位元組數）、`sha256` 與 `license`，再把檔案放進上述路徑。`git add` 後用 `git lfs ls-files` 確認清單裡有它，再用 `git show :<檔案路徑>` 查看 index：內容應以 `version https://git-lfs.github.com/spec/v1` 開頭，表示存進 Git 的是 pointer。確認後再 commit、push。
6. 推送後，另開一個乾淨的 checkout，只 pull 那一個 asset 並比對 manifest 的 checksum，才算遠端上傳與下載都成功。`python3 scripts/verify_remote_lfs.py` 也會把 `lfs_assets` 列的每個檔案下載到空的 LFS 快取，核對大小與 SHA-256。

一般 Git 的檔案超過 50 MiB 會被警告、超過 100 MiB 會被拒絕。LFS 沒有「依檔案大小自動追蹤」的功能，只能按路徑或副檔名指定。LFS 只放精選封裝，不全量上傳完整的公開 dataset。大檔的每個新版本都會再占一份完整的 storage，所以資料包少改，checkpoint 也只保存精選的。

### 重新發布 LFS 資料

Fashion-MNIST 封裝要重新上傳時，在 GitHub Actions 執行 **Publish and verify LFS dataset**（在網頁上按 Run workflow，或用具 Actions: Read and write 權限的 token 透過 API 觸發）。它在 GitHub 提供的 runner 上依序下載官方的四個原始檔並核對 SHA-256、組成固定 SHA-256 的封裝（含上游授權）、上傳到 LFS、從空的 LFS 快取下載回來核對封裝與其中每個檔案，最後才把 pointer 的 commit 推到 main。這個 workflow 用 GitHub Actions 自動提供的 `GITHUB_TOKEN`，只宣告 `contents: write`，不需要把你的 PAT 存成 Actions secret。

### Colab 只取得指定 LFS asset

需要在 Colab 取得某個 LFS 檔時，先跳過全部 LFS 下載，再只取那一個檔。以下整段適用於尚未執行課程環境格的全新 Colab runtime。以 Fashion-MNIST 封裝為例，在 Colab 開一個 code cell，貼上下面整段（第一行 `%%bash` 讓整格在同一個 shell 裡執行，`cd` 才會生效）：

```bash
%%bash
apt-get -qq update
apt-get -qq install git-lfs
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/birdhackor/learn_to_yolo.git
cd learn_to_yolo
git lfs install --local --skip-smudge
git lfs pull --include="data/curated/fashion-mnist-v1.tar" --exclude=""
sha256sum data/curated/fashion-mnist-v1.tar
```

已跑過課程環境格時，它已下載固定版本的 repo，並把 notebook 的目前目錄切到那裡。目錄名包含教材 tag，不是固定的 `/content/learn_to_yolo`。這時省略上面整格的 clone 與 `cd learn_to_yolo` 兩行，在目前目錄直接做安裝、`git lfs install`、指定路徑的 `pull` 與校驗；可先加 `git rev-parse --show-toplevel`，確認印出的是課程 repo 根目錄。這一步只取資料，不切換教材版本。

最後一行印出的 SHA-256，應等於 `data/manifest.json` 的 `lfs_assets` 記的值。教材的 notebook 要用時，改成 clone 固定的發布 tag（`git clone --branch <tag>`）。每次下載都計入 owner 的 LFS 頻寬。完整的來源 dataset 走下載器（`scripts/download_data.py`）或來源的官方下載，不走 Pages。

GitHub 明列 **Git LFS 不能用於 GitHub Pages**。網站只說明取得方式，資料在 Colab 裡下載；網頁要顯示的結果圖用一般 Git 裡的小檔。

## 6. 發布新版教材

先分清這次修改的是文稿、實驗還是發布工具。文字和網站維護可沿用現有 Notebook tag；實驗程式或環境格改動才建立新的 tag。既有 tag 的內容與指向都保留。

1. 修改正文、程式或圖。需要新 Notebook 時執行 `python3 scripts/build_lesson_notebooks.py --ref <新 tag>`；只改工具不用重建已發布 Notebook。
2. 執行 `.venv-model/bin/python scripts/record_evidence.py` 列出過期科學紀錄。程式沒改的紀錄保留原日期與電腦。受影響的 CPU 紀錄用 `--run` 重產；先安裝 `requirements-model.txt`、需要的 `requirements-video.txt`，並依提示核對固定套件版本。涉及真實資料時先明確執行 `python3 scripts/download_data.py fetch fashion-mnist`，再用 `python3 scripts/verify_data.py fashion-mnist` 離線核對；不把下載藏進教材或測試。GPU 紀錄仍須作者授權，依原有手動 GPU workflow 執行。
3. 紀錄重產後，同步圖與正文中的數字、比較條件和限制。`verify_curriculum.py --render-only` 只將有效紀錄接回目前頁面和 Notebook，不執行實驗，也不重畫圖。從別台電腦帶回結果時，圖也要一起帶回；保留本機修改，避免直接解包覆寫整份教材。
4. 依 clear-tutorial 核對受影響內容的易讀性、技術正確性與前後銜接；圖或版面改動時看實際頁面。PR 簡述範圍、發現、修正、驗證與限制。詳細工作資料放 `work/`、`outputs/` 或短期 artifact，不提交全套 session，不要求歷史指紋或原審閱者回叫。
5. 執行來源、證據、測試及網站檢查：

    ```bash
    python3 scripts/validate_preparation.py
    python3 scripts/validate_lessons.py
    python3 scripts/validate_curriculum_evidence.py
    .venv-model/bin/python -m pip install -r requirements-test.txt
    .venv-model/bin/python -m pytest tests/
    .venv-docs/bin/zensical build --clean --strict
    python3 scripts/validate_site.py
    ```

6. 檢查 diff，commit 並推送教材與必要科學紀錄。有新 tag 時建立並推送它，不覆寫舊 tag。保留 `lesson-releases.json` 的既有完整 SHA；新發布經公開核對後，把新的 tag/commit 加入這份小清單。舊驗證 log 不作發布輸入，也不需要先刪 log 才能發版。
7. 手動執行 Pages。它以程式、資產、資料清單、科學紀錄與實際環境識別整包 CPU 結果；相同則核對後重用，缺少或不符則執行全部 52 節。只執行實驗格，沒有執行會 clone/install 的 Colab 環境格；後者另由公開驗證工作核對。網站工作只裝 Zensical，並寫入網站與實際執行版本、執行日期及範圍。
8. 部署完成後核對公開 `build-info.json` 的 `revision`，點開修改頁面、圖與 Colab 入口。有新 tag 時手動執行 Verify published lessons；它從乾淨公開 clone 比對網站、圖、Notebook、環境格和 README 指令，並以固定 release 清單核對舊 tag。這是在 Linux runner 上驗證，不能宣稱已驗證 Colab 託管 runtime。
9. 用 `python3 scripts/verify_release.py save --run <編號>` 下載兩份結果到 `outputs/release-verification/<編號>/`。失敗會回報非零狀態並保留診斷資料；已有不同內容不覆寫。成功後在 PR 留結論與 workflow URL，需要長期支持主張的資料另挑選保存，無須再次 commit 整份 log。

本機要核對正式發布所需的 CPU bundle，可在模型環境執行：

```bash
.venv-model/bin/python scripts/course_execution.py ensure --revision "$(git rev-parse HEAD)"
.venv-model/bin/python scripts/course_execution.py verify
```

結果位於忽略的 `outputs/lesson-execution/`，不覆寫教材的歷史實測輸出。要強制重跑，移走這個快取目錄後再執行。新的執行輸入（包括被程式讀取的 Markdown）要加入 `course_execution.py`；選用下載的存在與否不影響一般 CPU 教材。

要在本機產生正式版本資訊，先完成上述檢查並提交發布來源，再執行 `.venv-docs/bin/python scripts/publication.py --revision "$(git rev-parse HEAD)"` 和 `.venv-docs/bin/python scripts/validate_site.py --require-publication`。CI 的 Linux 工作保留完整歷史並執行 `verify_release.py pins`；其他平台與學生安裝使用淺 clone，查歷史才另行取得完整版本。

## 官方參考與研究紀錄

- [GitHub Pages 自訂 workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [Git LFS 帳單](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [Git LFS 與 Pages 限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Pages／Colab 完整研究](../research/pages-colab.md)
- [LFS 完整研究](../research/lfs.md)
