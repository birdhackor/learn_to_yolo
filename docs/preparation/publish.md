# GitHub Pages、Colab 與 Git LFS：操作步驟

本專案目前只有前置規劃、資料與基礎配置，沒有正式教材。遠端查核時 `birdhackor/learn_to_yolo` 為 public、預設分支 main，Pages 已設為 GitHub Actions 模式；實測狀態見 `data/remote-lfs-verification.json` 與 GitHub Actions。

## 1. 先確認準備內容與授權選擇

本次新增：MkDocs 網站、手動 Pages workflow、環境檢查 notebook、資料 manifest／下載器、LFS 路徑規則，以及大綱／研究。網站目前只包含前置規劃。

你可先本地預覽：

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r requirements-docs.txt
python scripts/validate_preparation.py
mkdocs serve
```

Windows 使用 `.venv-docs\Scripts\activate`。瀏覽終端機顯示的本地網址。只建置則執行 `mkdocs build --strict`。

專案還沒有 LICENSE。公開釋出時請選自己的程式／教材條款，例如程式 MIT、原創教材 CC BY 4.0；這是待 owner 決定的選項，本輪未替你宣告。第三方資料仍遵照各自來源授權。

## 2. 把準備檔推到 GitHub

### 先確認推送認證

本環境使用你提供的 `GIT_LFS_AUTHORIZATION` 秘密欄位，內容為完整的 `Basic …` Authorization header；允許 HTTPS 主機為 `github.com`、`api.github.com`。值只在平台的秘密設定介面填入，不寫進 repo、remote URL、Git config 或聊天。

`github_auth.py` 將 header 透過子程序的暫時 Git 設定提供給 Git／LFS，並遮蔽敏感輸出。原始 Authorization 值不會成為命令列參數，也不會保存到檔案。使用方式：

```bash
python scripts/github_auth.py git ls-remote origin HEAD
python scripts/github_auth.py git push origin HEAD:main
```

若這個秘密欄位未提供，腳本沿用既有平台／Git 認證。Git 讀取、API 回 200 與 repository 的 push 角色，不能單獨證明 LFS object 已上傳；還要下載回來核對 checksum。

本輪實測：LFS batch upload 認證回 200，但這個雲端的直接 S3 傳輸路徑回 501，S3 指出不支援 `Transfer-Encoding`；包含 29 KB 小檔的診斷也相同。因此準備 `Publish and verify LFS dataset` workflow，在 GitHub-hosted runner 下載官方 Fashion-MNIST、校驗來源、組成固定封裝、上傳 LFS、從空快取下載驗證，最後才提交 pointer 到 main。

這個 workflow 使用 GitHub Actions 自己的 `GITHUB_TOKEN`，只需要該 job 的 `contents: write`，無需將你的 PAT 再存到 GitHub Actions secrets。Pages job 另宣告 `pages: write` 與 `id-token: write`。

若原始碼推送被拒，才檢查 PAT 的 repository access 是否包含 `learn_to_yolo`、Contents 是否為 Read and write；新增 workflow 也需要 Workflows: Read and write。透過 API 手動觸發 workflow 需要 Actions: Read and write；在 GitHub 網頁按 Run workflow 可用你的登入權限。

### 推送準備檔

確認 diff、忽略下載快取後，以你現有的 GitHub 登入方式 commit／push。以下不會強制覆寫遠端；工作區分支即使不是 main，也將目前提交推到遠端 main：

```bash
git status --short
git diff --check
git add README.md .gitignore .gitattributes .github mkdocs.yml requirements-docs.txt section-map.json docs notebooks scripts data/manifest.json data/local-lfs-verification.json data/remote-lfs-verification.json data/licenses
git commit -m "Prepare dataset sources, Pages site and Colab entry"
python scripts/github_auth.py git push origin HEAD:main
```

`data/downloads/` 與 `data/processed/` 被忽略，不會因上面的指令進 Git。若遠端 main 已有新提交，先按正常 Git 流程同步／解決，不用 force push。

## 3. 啟用 GitHub Pages

1. 開啟 repository：`https://github.com/birdhackor/learn_to_yolo`。
2. **Settings → Pages → Build and deployment → Source**，選 **GitHub Actions**。
3. 如 Actions 被停用，在 **Settings → Actions → General** 允許本 workflow 使用的官方 actions。workflow 已自行宣告所需權限，不需為它新增 PAT 或將全部 workflow 設成 write。
4. **Settings → Environments → github-pages**：如已有部署分支規則，確保 main 可部署；可將部署來源限制為 main。
5. **Actions → Publish preparation site → Run workflow**，選 main 後執行。
6. 等 build 與 deploy 都成功，從 deployment URL 或 Settings → Pages 開站。

預期網址：**https://birdhackor.github.io/learn_to_yolo/**。

Pages 與資料發布 workflow 目前都接受手動觸發，推送本身不發布。它只上傳 `site/`，不下載訓練資料或 LFS。採 Actions artifact 流程，不用建立 gh-pages 分支或執行 `mkdocs gh-deploy`。

Pages 經由 public repo 可用免費方案。此網站公開，不需要自訂網域；若以後改 private repo，另核對方案與網站 visibility。你看到的雲端環境「儲存並發布」不是 GitHub Pages 的發布流程。

## 4. 驗證 Colab 入口

1. 從網站首頁點「在 Colab 開啟環境檢查」，或直接開：
   `https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/main/notebooks/00_environment_check.ipynb`。
2. 登入 Google 帳號；公開 notebook 不需另外授權 GitHub 寫入。
3. 可以先用 CPU 執行；若要測 GPU，選 **Runtime → Change runtime type → GPU**，實際選項依 Colab 當時介面為準。
4. 執行全部 cells，查看 Python／PyTorch／CUDA；免費 GPU 可能未分配，這項檢查不訓練模型。
5. 要保存自己的修改，選 **File → Save a copy in Drive**。runtime 中的資料檔與權重需另行保存，notebook 副本不包含它們。

本地已檢查 notebook 格式與診斷程式；Google 登入、實際 Colab GPU、Pages 公開網址需要上述發布後操作才能驗證。

## 5. Git LFS：已準備與你需要設定的部分

已設定三個專用二進位路徑：

```text
data/curated/**
artifacts/checkpoints/**
artifacts/exports/**
```

普通圖片、notebook、Markdown、manifest 仍在一般 Git。`data/curated/` 專放可再散布的封裝，不放說明文件；授權與索引放在 `data/licenses/`、`data/manifest.json`。

已用 Fashion-MNIST 的 **26,421,880 bytes** 訓練圖，以及 Penn-Fudan 的 **53,723,336 bytes** 封裝，在隔離本地 repository 進行 add → LFS pointer → object fsck → checkout 還原，SHA-256 相符；紀錄在 `data/local-lfs-verification.json`。Penn-Fudan 只是本地測試，未將受限照片放進公開 repo。此外，Fashion-MNIST 完整封裝已透過 GitHub Actions 真正上傳並下載校驗；本地測試不包含把 Penn-Fudan 照片重新公開。

### 首次使用遠端 LFS

1. 安裝 [Git LFS](https://git-lfs.com/)，執行 `git lfs version` 確認。
2. 先完成第 2 節的推送認證，再在 checkout 執行 `git lfs install --local`。`.gitattributes` 已備妥，不需要把全部 PNG／JPG 再 track 成 LFS。
3. 到 GitHub 帳號 **Settings → Billing & licensing**（介面可能顯示 Billing & plans），查看 LFS usage 與 budgets。檔案計入 repo owner 的 storage，讀者下載亦計入 owner bandwidth。
4. 目前 Free／Pro 的包含額度為 **10 GiB storage＋10 GiB download bandwidth**；Free／Pro 單檔上限 2 GB。這是帳號額度，非每個 repo 各一份；超額行為由帳號付款／budget 設定決定。先決定你的預算，不必為前置工作開通付費。
5. 有確定要發布的授權清楚 asset 後，先補入大小與 SHA-256 manifest，放入上述路徑，再執行 `git add`、`git lfs ls-files`，確認清單含該檔。用 `git show :實際檔案路徑` 查看 index：應以 `version https://git-lfs.github.com/spec/v1` 開頭，再 commit／push。
6. 另開乾淨 checkout，只 pull 那一個 asset 並比對 manifest checksum，才算遠端上傳／下載驗證成功。

Git 普通檔案 >50 MiB 警告、>100 MiB 阻擋；LFS 本身沒有「按大小自動 track」功能。目前只是準備規則，不將完整公開 dataset 全量上傳。每次大檔新版本也會占完整 storage，因此資料包少改、只保存精選 checkpoint。

### 本雲端需要重新發布資料時

先在 GitHub Actions 執行 **Publish and verify LFS dataset**，或以本環境的秘密認證觸發該 workflow。它下載已查核的官方檔案、重建固定 SHA-256 封裝、上傳並從空快取下載核對，成功後才提交 pointer。直接從此雲端 PUT 到 S3 的路徑曾回 Transfer-Encoding 501，重複改 token 不會修正這項傳輸問題。

### Colab 只取得指定 LFS asset

下面是未來已有 `artifacts/checkpoints/miniyolo-demo.pt` 時的示例，不是本輪新增的模型：

```bash
apt-get -qq update
apt-get -qq install git-lfs
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/birdhackor/learn_to_yolo.git
cd learn_to_yolo
git lfs install --local --skip-smudge
git lfs pull --include="artifacts/checkpoints/miniyolo-demo.pt" --exclude=""
```

此範例先跳過全量下載。正式小節再替換成固定的 release／commit 與確實存在的 asset，並核對 SHA-256。來源 dataset 則走下載器或來源官方下載，不走 Pages。

GitHub 明列 **Git LFS 不能用於 GitHub Pages**。網站可提供取得方式，資料由 Colab 取得；網頁需要的結果圖使用小型普通 Git 資源。

## 官方參考與研究紀錄

- [GitHub Pages 自訂 workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [Git LFS 帳單](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [Git LFS 與 Pages 限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Pages／Colab 完整研究](../research/pages-colab.md)
- [LFS 完整研究](../research/lfs.md)

## 教材的新版本發布

目前教材與42份Colab固定為 `lessons-v0.1.0`。修訂教材時使用新的tag，不覆寫原tag：

1. 完成各節網頁、CPU案例及陌生讀者審查，再用 `python scripts/build_lesson_notebooks.py --ref lessons-v0.2.0` 配對新版本。
2. 執行 `python scripts/validate_preparation.py`、`python scripts/validate_lessons.py`、`.venv-model/bin/python scripts/check_lesson_runtime.py`、`.venv-model/bin/python -m pytest tests/test_core.py`，最後以 `.venv-docs/bin/mkdocs build --strict` 建站。
3. 提交所有教材與證據，再建立 `git tag lessons-v0.2.0`。使用前文的認證helper推送 `main` 與此tag；Colab所用的tag必須先能在GitHub讀到。
4. 到 Actions → **Publish Learn to YOLO** → Run workflow，選 `main`。等 build與deploy都成功，再打開網站、抽查新tag的notebook。

網站建置不會下載LFS資料或安裝PyTorch。公式與字型資產隨網站提供，MathJax 3.2.2保留原Apache 2.0授權與來源校驗紀錄。
