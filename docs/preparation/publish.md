# GitHub Pages、Colab 與 Git LFS：操作步驟

Learn to YOLO 的 42 節教材各配一本 Colab notebook；網站用 Zensical 建成靜態網頁，由 GitHub Actions 發布到 GitHub Pages。repository `birdhackor/learn_to_yolo` 是 public，預設分支是 main。本頁依序說明本地預覽、推送、Pages 設定、Colab 入口與 Git LFS，最後是發布新版教材的完整流程。

## 1. 本地預覽與授權選擇

網站設定在 `zensical.toml`，建置工具的版本固定在 `requirements-docs.txt`。用 Python 3.12（和 Pages workflow 相同），在 repo 根目錄建立文件環境並預覽：

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements-docs.txt
.venv-docs/bin/zensical serve
```

用瀏覽器打開終端機顯示的本地網址。Windows 把 `.venv-docs/bin/` 換成 `.venv-docs\Scripts\`。預覽不需要 GPU、PyTorch 或資料集。只建置不預覽時執行 `.venv-docs/bin/zensical build --clean --strict`：`--clean` 清除建置快取，`--strict` 遇到警告就中止。和 Pages 相同的整套檢查，指令見第 6 節第 8 步〈跑完所有檢查〉。

本頁其他直接用 `python3` 執行的指令，也要用 Python 3.12。`review_coverage.py`、`validate_lessons.py`、`validate_site.py` 與 `verify_release.py` 用到 Python 3.11 才加入的標準庫 `tomllib`，在更舊的 Python 會停在 `ModuleNotFoundError: No module named 'tomllib'`。Zensical 只要求 Python 3.10 以上，所以預覽成功，不代表這些檢查跑得動。

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
git commit -m "Update Learn to YOLO site"
git push origin HEAD:main
```

`data/downloads/`、`data/processed/`、`artifacts/runs/`、`site/` 與建置快取都列在 `.gitignore`，不會被 commit。commit 前看一遍已暫存的檔案，不要夾帶無關的修改。遠端 main 有新的 commit 時，照一般 Git 流程同步、解決衝突，不要 force push。

## 3. 啟用 GitHub Pages

repo 的 Pages 來源已設為 GitHub Actions。第一次設定（或重新設定）時：

1. 開啟 repository：`https://github.com/birdhackor/learn_to_yolo`。
2. **Settings → Pages → Build and deployment → Source** 選 **GitHub Actions**。
3. Actions 被停用時，到 **Settings → Actions → General** 允許 workflow 使用的 GitHub 官方 actions。workflow 自己宣告需要的權限，不必為它另外建立 PAT，也不必把全部 workflow 的權限設成 write。
4. **Settings → Environments → github-pages**：已有部署分支規則時，確認 main 可以部署；也可以把部署來源限制為 main。
5. **Actions → Publish Learn to YOLO → Run workflow**，選 main 後執行。
6. 等 build 與 deploy 都成功，從 deployment URL 或 Settings → Pages 開啟網站。

網址是 **<https://birdhackor.github.io/learn_to_yolo/>**。設定好之後，每次發布只需要第 5、6 步。

build job 依序執行下面五項，任何一項失敗都不會部署：

- `validate_preparation.py`：資料 manifest、頁面與 notebook 的配對、notebook 格式、網站檔案的大小與類型。
- `validate_lessons.py`：課程頁、頁面裡的程式摘錄與程式一致、固定版本的 notebook、每一頁的審查紀錄對應目前內容、SVG。
- `validate_curriculum_evidence.py`：每份執行紀錄都對得上目前的程式、notebook 與頁面。
- `zensical build --clean --strict`。
- `validate_site.py`：Zensical 版本與 `requirements-docs.txt` 相符、連結與錨點、Colab 配對、搜尋語言、Markdown 是否照原意轉成表格／編號清單／摺疊區塊／數學／連結、有沒有文字被誤當成 HTML 標籤，以及 artifact 邊界（`site/` 裡沒有 symlink 與 LFS pointer，也沒有 `.pt`、`.pth`、`.onnx`、`.ipynb`、`.zip`、`.tar`、`.gz` 檔）。

建置只上傳 `site/`，不下載 LFS 資料、不安裝 PyTorch；部署用 Actions artifact，不需要 gh-pages 分支。公式用的 MathJax 3.2.2 與它的字型放在 `docs/assets/vendor/mathjax/`，隨網站提供，並附上 MathJax 的 Apache 2.0 授權（來源校驗見 `artifacts/checks/mathjax-vendor.json`）。

Pages workflow 整體只有 `contents: read` 權限，只有 deploy job 另外宣告 `pages: write` 與 `id-token: write`。repo 的五個 workflow（網站、LFS 資料、兩項 GPU 檢查、發布後的公開驗證）都只能手動啟動：push 本身不會發布網站，也不會啟動 GPU。

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

需要在 Colab 取得某個 LFS 檔時，先跳過全部 LFS 下載，再只取那一個檔。以 Fashion-MNIST 封裝為例，在 Colab 開一個 code cell，貼上下面整段（第一行 `%%bash` 讓整格在同一個 shell 裡執行，`cd` 才會生效）：

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

最後一行印出的 SHA-256，應等於 `data/manifest.json` 的 `lfs_assets` 記的值。教材的 notebook 要用時，改成 clone 固定的發布 tag（`git clone --branch <tag>`）。每次下載都計入 owner 的 LFS 頻寬。完整的來源 dataset 走下載器（`scripts/download_data.py`）或來源的官方下載，不走 Pages。

GitHub 明列 **Git LFS 不能用於 GitHub Pages**。網站只說明取得方式，資料在 Colab 裡下載；網頁要顯示的結果圖用一般 Git 裡的小檔。

## 6. 發布新版教材

42 節頁面的 Colab 按鈕與 notebook 固定在同一個發布 tag：按鈕開啟這個 tag 的 notebook，notebook 的環境格也 clone 這個 tag（`section-map.json` 各節的 `source_ref` 記著它）。已發布的 `lessons-v*` tag 一律不移動、不覆寫；教材的程式或實驗結果改了，就發布一個新 tag。

發布前，每份執行紀錄都要對得上目前的程式。逐節的紀錄是 `artifacts/checks/curriculum/<節>.json`，記著執行的日期（UTC）、產生它的電腦與 stdout，並用 SHA-256 綁定產生它的程式：該節程式，加上它直接或間接 import 的每個 repo 檔案。補充實驗的 CPU 紀錄與兩份 GPU 紀錄也以同樣方式綁定各自的程式，清單在 `scripts/evidence_records.py`。綁定的檔案都沒變，紀錄就一直有效，保留它的日期與電腦；任何一個改了，那份紀錄才過期。所以發布新版時只重跑過期的紀錄。

以下指令都在 repo 根目錄執行。`.venv-model` 是照 README 建立的 CPU 模型環境，`.venv-docs` 是第 1 節的文件環境；直接寫 `python3` 的指令，照第 1 節用 Python 3.12。

1. **把 notebook 配對到新 tag。** 教材的頁面與程式都改好後執行：

    ```bash
    python3 scripts/build_lesson_notebooks.py --ref <新 tag>
    ```

    它重建 42 本 notebook，並把各節頁面的 Colab 連結與 `section-map.json` 的 `source_ref` 改成新 tag。正文裡直接寫出 tag 的地方不會跟著改：`README.md`、〈驗證範圍〉、〈全套實驗與審查〉與第 0 章〈一次學習的超短暖身〉（環境格印出的版本）都寫著目前的 tag，要另外搜尋 `lessons-v`，逐一改成新 tag；審查涵蓋只略過 Colab 連結裡的 tag，所以改了這些地方的 tag 之後，`README.md` 以外的三頁都要重新審查。`lesson_cases/<節>.py` 沒變的節，最後一格直接填回紀錄裡的輸出；改過的節列在輸出的最後。紀錄是否過期（包括 import 的模組有沒有改），由下一步判斷。

2. **列出過期的紀錄。**

    ```bash
    .venv-model/bin/python scripts/record_evidence.py
    ```

    這只列出，不重產紀錄，也不改動 git 追蹤的檔案：哪些 GPU 紀錄要重跑、哪些 CPU 紀錄（逐節紀錄與補充紀錄）過期，以及這台電腦的規格。`grid-learning.json` 的兩張圖（`grid-learning-curve.svg`、`grid-learning-predictions.svg`）由 `scripts/render_learning_evidence.py` 依紀錄畫出，而這支腳本不在紀錄綁定的程式裡，所以紀錄沒過期時，`record_evidence.py` 會用它在暫存目錄重畫一次；和網站上的圖不同，就列出 `grid-learning figures (renderer changed)`，第 5 步的 `--run` 只重畫這兩張圖，不重新訓練。

3. **把修改推到一個分支。** 第 4 步的 GPU workflow 跑的是 GitHub 上的程式；第 5 步若在另一台電腦記錄，那台電腦也要從 GitHub 取得這些修改。要做其中任何一件事，先照第 2 節確認 diff，再把到目前為止的修改（教材、程式與第 1 步重建的 notebook）commit 到一個分支並推上去：

    ```bash
    git switch -c <分支>
    git add -A
    git commit -m "<這一版的修改>"
    git push -u origin <分支>
    ```

    已經在要推的分支上時，省略第一行。`-u` 讓之後在這個分支上直接執行 `git pull`，就能取回別處推到它的 commit。沒有過期的 GPU 紀錄，CPU 紀錄也就在這台電腦記錄時，跳過這一步。

4. **重產 GPU 紀錄。** 沒有過期的 GPU 紀錄就跳過這一步。在第 3 步的分支上執行：

    ```bash
    .venv-model/bin/python scripts/record_evidence.py --gpu <分支>
    ```

    workflow 跑的是推上去的程式，所以本機的 commit 必須和 `origin/<分支>` 相同，否則它會停下。它用 GitHub CLI（`gh`，先執行 `gh auth login`）逐一啟動過期紀錄對應的 workflow（`deployment-gpu.yml`、`gpu-smoke.yml`，在 Modal 的一張 NVIDIA L4 上執行），等它跑完，再下載結果存成紀錄。workflow 需要 repository 的 Modal 設定（至少 `MODAL_TOKEN_ID` 與 `MODAL_TOKEN_SECRET`），清單見〈[GPU／checkpoint 實測](../validation/gpu-smoke.md)〉。成功產生的 GPU 紀錄留在工作區，commit 後推到同一個分支。

    兩種紀錄分開檢查：`--gpu` 結束前用 `validate_curriculum_evidence.py --scope gpu`，只檢查 GPU 紀錄；下一步的 `--run` 結束前用 `--scope cpu`，只檢查逐節與補充的 CPU 紀錄。另一種紀錄還過期，不會讓這兩項檢查失敗；全部紀錄到第 8 步才一起檢查。`--gpu` 失敗在這項檢查時，GPU 紀錄已經存好；`--run` 失敗在這裡時，不會執行後面的 `--push` 或 `--bundle`。

    workflow 失敗時它會停下；下載得到結果時，仍把結果存進紀錄檔供你檢查。看完後先換掉這份結果，不要 commit 它。原因是結果在 workflow 一開始就寫上了它綁定的程式的 SHA-256，而 `--gpu` 只比對 SHA-256，不看結果有沒有通過：留著它，下次 `--gpu` 會把它當成現行紀錄而跳過，`validate_curriculum_evidence.py` 卻會因為它沒有通過而一直失敗。換的方法是用 `git status --short` 看停下時訊息指名的紀錄檔：顯示 `M`（git 已追蹤）就執行 `git restore --staged --worktree <紀錄檔>`，還原成最近一次 commit 的版本；顯示 `??`（git 還沒追蹤）就刪除。然後修正原因：只改 repo 以外的設定（例如 repository 的 Modal 設定）時，改好後重跑這一步；改了 repo 裡的程式時，從第 1 步重來。

5. **重產 CPU 紀錄。** 既有 CPU 紀錄是在哪台電腦產生的，各紀錄裡都有記載（逐節紀錄在 `machine` 欄：作業系統、CPU、Python、PyTorch 等）。盡量在同一台電腦重跑，沒改到的計算才會得到和舊紀錄相同的數字。在那台電腦 checkout 第 3 步的分支（做了第 4 步時，要包含推上去的 GPU 紀錄），照 README 建好 `.venv-model`，再用 `.venv-model/bin/python -m pip install -r requirements-video.txt` 加裝影片套件，然後執行：

    ```bash
    .venv-model/bin/python scripts/record_evidence.py --run --push <分支>
    ```

    `--run` 先核對 Python 3.12 與 `requirements-model.txt`、`requirements-video.txt` 的固定版本，不符就停下；接著只重跑過期的部分：逐節程式交給 `verify_curriculum.py`（它同時更新 notebook 的輸出與頁尾的執行紀錄區塊，並把第 17–19 章實驗畫的圖複製到 `docs/assets/diagrams/`），補充紀錄交給各自的實驗腳本（除了影片檔與 Fashion-MNIST 的紀錄，各腳本也會重畫 `docs/assets/diagrams/` 裡對應的圖）。重產 Fashion-MNIST 的紀錄時，`--run` 會先執行 `scripts/download_data.py fetch fashion-mnist`，在 `data/downloads/fashion-mnist/` 備齊四個原始檔（共 30.88 MB）：已有、而且大小與 SHA-256 都和 `data/manifest.json` 相符的檔就沿用，缺的才下載，這時記錄的電腦要能連網；有不符的舊檔則會停下，訊息會指名那個檔，移走後重新執行上面的指令。最後用 `validate_curriculum_evidence.py --scope cpu` 檢查，並在輸出最後的 `Changed files:` 列出改動的檔案。`--push <分支>` 把這些檔案 commit 並推到該分支；不方便推送時，改用 `--bundle <檔名>.tar.gz` 把它們打包。兩種的帶回方法見下一步。就在平常編輯的電腦記錄時，只要 `--run`。換一台電腦也能重產，新紀錄會記下新的電腦，但計時與訓練後的小數可能和舊紀錄不同。

6. **從別處帶回紀錄時，連重畫的圖一起帶回。** 第 5 步改動的檔案都列在 `Changed files:` 底下：`artifacts/checks/` 的紀錄、`docs/assets/diagrams/` 裡重畫的圖，以及頁面與 notebook（頁尾的執行紀錄區塊與最後一格的輸出）。用了 `--push` 時，在平常編輯的電腦執行 `git pull --no-rebase`，就帶回整份清單；兩邊都有新的 commit 時，`--no-rebase` 指定把它們合併起來。這台電腦在第 3 步之後若又改過同一個檔，git 不會直接蓋掉，而是合併兩邊的修改；合併不了就停下，列出有問題的檔案。

    `--bundle` 的包裡也是整份清單，但 `tar -xzf` 會直接覆寫同名檔，不合併也不提示。包裡的頁面與 notebook 是記錄那台電腦上的版本：第 3 步之後在這台電腦改過其中一頁，解開後那個修改就不見了。所以只解開紀錄與圖，頁面與 notebook 改在這台電腦依紀錄重寫：

    ```bash
    tar --exclude='docs/lessons/*' --exclude='notebooks/*' -xzf <檔名>.tar.gz
    .venv-model/bin/python scripts/verify_curriculum.py --render-only
    ```

    第二行的 `--render-only` 不執行任何實驗，只依紀錄重寫 42 節頁尾的執行紀錄區塊、notebook 最後一格的輸出，以及紀錄索引 `artifacts/checks/curriculum/index.json`；有任何一節沒有現行紀錄就停下。用其他方式帶回紀錄，或 `git pull` 合併時保留了這邊的頁面或 notebook，也執行這一行。

    紀錄與圖則一定要從記錄的那台電腦帶回：圖只在那台電腦上重畫，`--render-only` 不會重畫、也不會複製任何圖。少帶了圖，網站會在新的輸出旁邊顯示舊圖，而 Pages 建置前的檢查都不會發現：`validate_curriculum_evidence.py` 不檢查圖；圖檔沒變，`review_coverage.py` 也不會要求重新審查。

7. **審查改過的頁面。** 網站導覽裡的每一頁，都要有一份對應目前內容的審查紀錄。課程頁由 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序；頁面引用原始論文、官方程式或函式庫文件的說法，另打開原始來源逐句核對：論文看原文，官方程式看固定的 commit 或 tag（頁面連到固定版本時就用那一版），函式庫看官方文件，並在審查紀錄列出查閱的來源與版本。其他頁對照 repo 的程式、指令、紀錄與頁面引用的來源查核，頁面上有指令的，也實際執行其中一部分。查到的問題修正後，由另一位 AI 檢查修正；審查紀錄要記下查核方法、每個發現和它的處理方式。〈[驗證範圍](../status.md)〉與〈[全套實驗與審查](../validation/curriculum.md)〉向讀者說明這些審查方式，`README.md` 也寫著這些審查方式，〈[課程大綱](../planning/outline.md)〉則寫著審查者都是 AI；改用別的方式審查時，這幾處的說明都要跟著改。

    審查涵蓋頁面文字、頁面上的 SVG，以及課程頁的程式與它 import 的模組；第 5 步會重畫部分實驗圖，所以審查放在紀錄之後。Colab 連結裡的 tag 與頁尾的執行紀錄區塊不算在內。審查紀錄放在 `reviews/`：課程頁是 `reviews/<節>.md`；其他頁取 `docs/` 底下的路徑，把 `/` 換成 `-`，例如 `docs/preparation/publish.md` 的審查是 `reviews/preparation-publish.md`。寫好後記下它涵蓋的內容；頁面路徑從 repo 根目錄算起，可以一次給好幾頁：

    ```bash
    python3 scripts/review_coverage.py --write docs/preparation/publish.md
    ```

    不加參數執行 `python3 scripts/review_coverage.py`，會列出還沒有審查、或審查已對不上目前內容的頁面，列出的路徑可以直接接在 `--write` 後面；有這種頁面時，`validate_lessons.py` 會失敗。

    紀錄裡的數字不在審查涵蓋的範圍裡。課程頁引用自己那一節的紀錄不會漏掉：那份紀錄過期，表示那一節的程式或它 import 的模組改了，審查也跟著失效；重畫後內容變了的圖，也會讓顯示它的頁面被列出。其餘引用紀錄數字的正文就不一定會被列出：引用別份紀錄的課程頁（例如第 18、19 章引用影片檔的紀錄、第 20 章引用 L4 的紀錄），以及不是課程頁、也沒有顯示重畫的圖的頁面（例如〈驗證範圍〉、〈[資料規劃](data.md)〉與〈[GPU／checkpoint 實測](../validation/gpu-smoke.md)〉），紀錄重產後都不會因此被列出；不在網站導覽裡的 `README.md` 則沒有審查紀錄。所以第 4、5 步重產了紀錄時，要在 `docs/` 與 `README.md` 找出引用這些紀錄的地方（可以搜尋紀錄的檔名與舊的數字），對照新紀錄逐一核對；數字變了就改正文，再重審改過的頁。

    審查之後又改了內容時，看改的是什麼。改了程式（`lesson_cases/` 的程式、它們 import 的模組，或補充實驗的腳本），從第 1 步重來，因為 notebook 與紀錄都可能因此過期；改了 `lesson_cases/` 卻略過第 1 步時，`validate_lessons.py` 會因為 notebook 最後一格和程式不一致而失敗，但只顯示 `AssertionError`，不說明原因。只改了頁面文字或圖，就重新審查改過的頁面，再對它執行一次 `review_coverage.py --write`。

8. **跑完所有檢查。**

    ```bash
    python3 scripts/validate_preparation.py
    python3 scripts/validate_lessons.py
    python3 scripts/validate_curriculum_evidence.py
    .venv-model/bin/python -m pytest tests/
    .venv-docs/bin/zensical build --clean --strict
    python3 scripts/validate_site.py
    ```

    除了 pytest，其餘就是 Pages workflow 部署前跑的檢查。

9. **commit、建立 tag、推送。** 先用 `git rm artifacts/checks/curriculum-release-bootstrap.json` 刪掉上一版的環境格與 README 驗證：它驗證的是上一個 tag，沒有程式讀它，第 11 步會為新 tag 重建。`artifacts/checks/curriculum-publication.json` 則要留著：第 11 步的網站檢查會從新 tag 讀這份上一版的網站驗證，核對舊 tag 的 commit；少了它，網站檢查會在讀檔時中止。接著照第 2 節的方式 commit 所有教材與紀錄，再建立並推送新 tag：

    ```bash
    git tag -a <新 tag> -m "<這一版的說明>"
    git push origin HEAD:main
    git push origin <新 tag>
    ```

    已發布的 `lessons-v*` tag 不重建、不移動。Colab 按鈕與環境格都指向這個 tag；tag 推上 GitHub 之前，新版的 Colab 入口打不開。

10. **發布網站。** 到 **Actions → Publish Learn to YOLO → Run workflow**，選 `main`，等 build 與 deploy 都成功。

11. **驗證公開網站與新 tag。** 到 **Actions → Verify published lessons → Run workflow**，`tag` 填新 tag 後執行（也可以用 `gh workflow run verify-release.yml -f tag=<新 tag>`）。這個 workflow 在 GitHub 提供的 Linux runner 上用 Python 3.12，從公開的 repository 重新 clone 這個 tag（不下載 LFS 資料），分兩項檢查：

    - **網站**：在 runner 上用 tag 的內容嚴格建置網站，再和公開網站比對。導覽裡的每一頁都要回應 HTTP 200，`<article>` 元素（頁面的內文）和建置結果相同；`assets/diagrams/` 的每張圖也要逐 byte 相同。它也確認每節頁面的 Colab 按鈕指向這個 tag 的 notebook，GitHub 上這本 notebook 的環境格固定在這個 tag、最後一格就是該節程式；已發布的舊 tag 仍指向上一份驗證記下的 commit；並記下最近一次成功的 Pages workflow run、它部署的 commit，以及那是不是 tag 的 commit。
    - **環境格與 README**：每種情況都用一個空資料夾與新的虛擬環境，像 Colab 的一個工作階段那樣，在同一個命名空間裡依序執行第 0 章 notebook 的環境格與最後一格。情況有四種：沒有 PyTorch、裝著別的 PyTorch 版本、已經是固定的版本（應該沿用），以及別的版本已經 import（應該要求重新啟動工作階段，接著用新的程序重跑兩格）；最後一格的輸出都要和 notebook 存著的輸出相同。另外執行第 20 章的兩格，它的環境格還會安裝 ONNX 套件。接著在另一份新 clone 裡，依序執行 `README.md` 的 bash 區塊裡的每個指令（`zensical serve` 不會自己結束，所以略過）。

    兩項都不經過 Colab：Google 登入、Colab 的託管 runtime 與它可能分配的 GPU，只有照第 4 節實際在 Colab 開啟才看得到，那樣的結果也不在下面的紀錄裡。

    workflow 跑完後（有檢查沒通過，結果照樣會上傳），用 GitHub CLI 把結果存進 repo。run 的編號在它的網址 `…/actions/runs/<編號>` 裡：

    ```bash
    python3 scripts/verify_release.py save --run <編號>
    ```

    它下載這次 run 的兩份結果，存成 `artifacts/checks/curriculum-publication.json`（網站，覆寫上一版的驗證）與 `artifacts/checks/curriculum-release-bootstrap.json`（環境格與 README），並印出兩份各自的 `passed`。兩份都印出 `passed=True` 時，commit 並推到 main；這兩個檔案不在網站裡，不必重新部署。上一版的驗證留在 git 歷史裡，可以用 `git log -p -- artifacts/checks/curriculum-publication.json` 查；新 tag 裡的這個檔案也是上一版的驗證（見第 9 步），因為驗證在建立 tag 之後才 commit。

    有一份沒通過，就兩份都不要 commit，改用重新驗證的結果：〈[全套實驗與審查](../validation/curriculum.md)〉連到這兩份紀錄，向讀者說明它們確認了哪些事；網站驗證還會隨下一個 tag 成為下次核對舊 tag 的依據，沒通過的網站驗證可能少記或記錯舊 tag 的 commit。紀錄裡逐項記著結果與每個指令的輸出，可以從中找出是哪一頁、哪種情況或哪個指令；看完後用 `git restore --staged --worktree artifacts/checks/curriculum-publication.json` 還原成上一版的驗證，並刪除 `artifacts/checks/curriculum-release-bootstrap.json`。

    原因在 tag 以外時（例如部署還沒完成、網路或套件下載暫時失敗），照第 10 步確認部署成功後重新執行這個 workflow，再用新的 run 編號執行 `save`。原因在 tag 的內容時（頁面、notebook、README 或程式），已推上 GitHub 的 tag 不移動，修正後從第 1 步改用另一個新 tag 重來；這時 `curriculum-release-bootstrap.json` 已經不在 git 裡，第 9 步不必再刪。

只更新 Zensical 主題、導覽或建置工具時，不需要新 tag：Colab 按鈕沿用目前的 tag，程式沒變，執行紀錄也都仍然有效。跑完第 8 步的檢查後，照第 10 步重新部署即可。這類修改要另外在瀏覽器裡看過閱讀頁、搜尋、快速換頁後的公式、SVG 與手機版面。

## 官方參考與研究紀錄

- [GitHub Pages 自訂 workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [Git LFS 帳單](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [Git LFS 與 Pages 限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Pages／Colab 完整研究](../research/pages-colab.md)
- [LFS 完整研究](../research/lfs.md)
