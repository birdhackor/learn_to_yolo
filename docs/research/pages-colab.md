# GitHub Pages + Colab 前置配置研究

> 這是研究代理在子任務當時的查核報告。後續整合已套用前置配置並下載部分資料；目前狀態與操作步驟請以「前置準備」頁為準。

核查日期：2026-10-02。目標儲存庫：`birdhackor/learn_to_yolo`；預定發佈來源：`main`。此文件是配置研究與後續操作清單，不含正式教材，也不代表已啟用 Pages、已發佈或已完成 Colab 登入測試。

## 建議選擇

採用 **MkDocs Material + Markdown 閱讀頁 + 獨立 `.ipynb`**。GitHub Pages 只放靜態閱讀內容；每個未來的小節對應一個 notebook，頁面放「Open in Colab」按鈕。閱讀頁另外保留精簡的圖片、指標、文字與程式片段，讓讀者無須取得 GPU 或執行完整訓練也能理解結果。網站 build 不執行 notebook、不安裝 YOLO、不下載資料集。

這個配置僅需一組 Python 文件建置工具。Material 自帶導航、搜尋與 Markdown 功能，安裝 `mkdocs-material` 時會帶入相容的 MkDocs、Markdown、Pygments 與 Markdown extensions，初期不用增加 notebook 轉換、執行或同步插件。[Material 安裝說明](https://squidfunk.github.io/mkdocs-material/getting-started/)

| 選項 | 適用情況 | 本案取捨 |
| --- | --- | --- |
| MkDocs Material + Markdown + 獨立 notebooks | 小節閱讀頁與可執行 notebook 明確分工 | 首選；建置輕量、樣式與導航現成、可避免把 GPU 執行加入網站 CI |
| 原生 MkDocs 主題 + 獨立 notebooks | 只需最基本閱讀與導航 | 依賴更少；Material 的內建呈現與搜尋更符合教學網站需求 |
| Jupyter Book / notebook 自動轉網站 | 以 notebooks 作教材主要來源，較重視引用、交叉參照與輸出自動匯入 | 可以後續重新評估；現在不用增加執行、轉換與格式同步流程 |

閱讀頁與 notebook 是兩個文件，需明確配對與共同版本管理。不要把相同的長篇教材複製到兩處；Markdown 放主要閱讀說明，notebook 放簡短上下文、執行步驟和回到閱讀頁的連結。是否將來改成單一來源自動產生兩種格式，待教材實際形成後再決定。

## 最小目錄規劃

以下僅為後續配置的檔名與目錄安排，不是本輪已新增的教材：

```text
mkdocs.yml                      # 網站設定與小節導航
requirements-site.txt           # 文件工具版本；與訓練依賴分開
.github/workflows/pages.yml     # 後续使用 Pages artifact 流程
docs/                           # 只放要發佈的閱讀內容
  index.md                      # 未來入口頁
  sections/<section-id>.md       # 未來各小節閱讀頁
  assets/<section-id>/           # 經選擇、壓縮的閱讀用圖片與文字結果
notebooks/<section-id>.ipynb     # 未來 Colab notebook；放在 docs 外
section-map.json                # 小節、notebook、結果、資料版本的配對清單
site/                           # 建置結果，忽略於 Git
data/                           # 大型資料，忽略於 Git，且不可放進 docs
runs/                           # 完整訓練輸出，忽略於 Git
```

`section-map.json` 可用簡單標準庫腳本讀取，記錄 `section_id`、閱讀頁路徑、notebook 路徑、對應結果路徑、程式碼版本與資料版本。初期不用加 MkDocs 巨集或 notebook 插件。

建議 `docs_dir: docs`、`site_dir: site`，並設定 `site_url: https://birdhackor.github.io/learn_to_yolo/`。專案 Pages 帶有 `/learn_to_yolo/` 子路徑；Markdown 的本地圖片、內部連結盡量用相對路徑，避免寫 `/assets/...` 導致瀏覽器到網域根目錄找檔案。[GitHub Pages 站型與網址](https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages)、[Material 的 site_url 說明](https://squidfunk.github.io/mkdocs-material/creating-your-site/)

## Python / pip 支援與建置環境

以本次 PyPI 查詢結果為準：

| 套件 | 當時版本 | 套件 metadata 的 Python 下限 |
| --- | --- | --- |
| `mkdocs` | `1.6.1` | `>=3.8` |
| `mkdocs-material` | `9.7.7` | `>=3.8` |

Material metadata 要求 `mkdocs>=1.6,<2`。以上是套件宣告的最低版本；Python 3.8 已過上游支援期限，不建議新建環境採用。建議文件環境固定 Python 3.11 或 3.12，依目前工作區可用版本驗證；網站 CI 與本地工具保持同版。訓練／Colab 的 Python、PyTorch、Ultralytics 版本另行管理，不能由這個表推論相容性。

可在 repo 外的虛擬環境安裝一組精確版本，例如 `mkdocs==1.6.1`、`mkdocs-material==9.7.7`，完成 build 後保存完整解析的依賴鎖定結果供後續審核。初期不裝 Material 的 `git`、`imaging` 或 `recommended` extras；本案的基本導航、搜尋和靜態圖片不需它們。不要把未限定版本的 `pip install mkdocs-material` 當成長期可重現配置。

來源：[MkDocs PyPI metadata](https://pypi.org/pypi/mkdocs/json)、[Material PyPI metadata](https://pypi.org/pypi/mkdocs-material/json)、[Material 安裝與固定版本](https://squidfunk.github.io/mkdocs-material/getting-started/)。本次僅核查 metadata，未在此研究子任務安裝或驗證套件。

## GitHub Pages 方案、設定、URL

1. **GitHub Free / Free for organizations：Pages 可用於公開儲存庫。** 公開教材最直接的免費方案是 public repo。
2. GitHub Pro、Team、Enterprise 可從 public 或 private repo 建置 Pages。一般 Pages 網站仍對網路公開；**private 原始 repo 不等於 private 網站**。
3. 限制網站讀者權限的 private Pages 是 organization 的 Enterprise Cloud 功能，與個人免費公開教材方案不同。
4. 未使用自訂網域時，此 repo 的預期專案 URL 是 `https://birdhackor.github.io/learn_to_yolo/`。這是網址規則推導，並非已存在／本次成功部署的網址。
5. 設定發佈來源需 repo 的 admin 或 maintainer 權限。

來源：[Pages 可用方案](https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages)、[來源設定與公開性](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)、[private Pages 與 Enterprise Cloud](https://docs.github.com/en/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site)。本研究未登入 GitHub、未查證帳號付費方案或 repo Pages 的現有設定。

### 建議 Pages Actions 部署方式

採 GitHub 官方 Pages artifact 流程，之後可以保留 `main` 為唯一教材原始分支：

1. build job：checkout → setup Python → 安裝文件鎖定依賴 → 核對小節配對 → `mkdocs build --strict` → 檢查 `site/` → `actions/upload-pages-artifact`，且 `path: site`。
2. deploy job：`needs: build`，部署剛完成的 artifact；設定 environment `github-pages`，將 `steps.deployment.outputs.page_url` 接到 environment URL。
3. build 只需 `contents: read`。deploy job 宣告 `pages: write` 與 `id-token: write`，不需要對原始分支的 `contents: write`，也不需要額外 PAT。`id-token` 用於 Pages 的部署驗證；不是讀者要填的密鑰。
4. 只對 `push` 到 `main` 和後續明確手動觸發進行部署。若之後加 PR 工作流程，PR 只建置與檢查，不部署。
5. deployment environment 限制只允許 `main`；避免其他分支發佈。
6. 目前 GitHub docs 範例採 `actions/checkout@v6`、`actions/configure-pages@v5`、`actions/upload-pages-artifact@v4`、`actions/deploy-pages@v4`。準備可執行草稿時再確認所用 runner 與 action 版本，不要混用舊第三方部署方法。

注意 Material 官網另有 `mkdocs gh-deploy --force` 範例：它會寫入 `gh-pages` 分支、要求 `contents: write`，對應 Pages 設定「Deploy from a branch」。這是另一種部署路徑；若採上述 artifact 流程，Pages 的 Source 選「GitHub Actions」，不用建立 `gh-pages` 分支，也不執行 `gh-deploy`。

來源：[GitHub 官方 custom workflow / permissions / artifact](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)、[來源與 environment 設定](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)、[Material 另一種 gh-pages 部署範例](https://squidfunk.github.io/mkdocs-material/publishing-your-site/)。

## 根代理現在可備妥的工作，以及使用者稍後要做的設定

| 階段 | 根代理可在目前工作區備妥 | 使用者／repo 管理者稍後在網頁操作 |
| --- | --- | --- |
| 1. 前置配置 | 在 repo 外保存目錄規劃、`mkdocs.yml` 草稿、文件依賴鎖定、Pages workflow 草稿、小節配對格式及檢查說明；不新增正式教材 | 此階段無需登入或啟用部署 |
| 2. 本地驗證 | 在 repo 外以清楚標示的配置檢查頁驗證 build、輸出 HTML、導航、子路徑連結和 artifact 範圍；檢查 checkout 前後 Git 狀態 | 無需 GitHub 或 Google 權限 |
| 3. 套用準備 | 把已測試的配置與所需檔案列成可檢閱清單；目前不修改受保護的 tracked repo 檔案 | 後續正式套用／推送是另一個授權範圍 |
| 4. Pages 啟用 | 提供精確 UI 路徑與部署行為；本次不替使用者改設定 | repo **Settings → Pages → Build and deployment → Source → GitHub Actions**；確認方案與 repo visibility |
| 5. Actions 與 environment | 準備最小 token permissions 的 YAML，無需要求 PAT | 如政策停用 Actions，至 **Settings → Actions → General** 設定允許所需官方 actions；於 **Settings → Environments → github-pages** 限制部署來源 `main` |
| 6. 正式發佈 | 授權並套用後檢查 workflow；不把本地 build 稱為已部署 | 授權並推送／執行 workflow；在 Actions 查看 build + deploy 成功，再從 Settings → Pages 或 deployment 的 `page_url` 開站 |
| 7. Colab 驗證 | 檢查 notebook JSON、badge URL、安裝／下載步驟的結構；本機 CPU 檢查能證明的能力另行標示 | Google 帳號登入 Colab，實際開啟 notebook，選擇硬體，執行、保存個人副本；本研究無此登入與實際 GPU 證據 |

此表中的「需要使用者 UI」是本次未登入、不改設定、不發佈的範圍限制，不是聲稱 GitHub API 永遠不能設定這些項目。若 repo 或 organization 設有額外審核規則，正式部署時依實際設定操作。

## 網站 artifact 的內容界線

MkDocs 會把 `docs_dir` 中的非 Markdown 資產複製到 `site_dir`，因此只上傳 `site/` 仍不足以避免 `docs/` 裡的意外大檔。將大型資料、weights、完整 runs、archives 及 notebooks 放在 `docs/` 外。閱讀用圖與少量文字結果選擇性匯入 `docs/assets/`，不要將整個訓練資料目錄加入網站。

後續 build 前後用標準庫或 shell 檢查，不需新加外掛：

- `site/` 有入口 HTML、預期的內頁及閱讀圖，沒有 `data/`、`runs/`、`.git/`、venv、憑證或權重檔；上傳路徑明確固定 `site/`，不用 `.`。
- 明確限制 `docs/assets/` 的個別檔案與總量；網站以遠低於 1 GB 為設計目標。此案不應將影片、zip、模型權重打包進 Pages。
- `docs/` 和 `site/` 均不可出現以 `version https://git-lfs.github.com/spec/v1` 起頭的 Git LFS pointer。它是文字參照，不能當成圖片或資料發布。
- 不依賴 Pages 解開 LFS。GitHub docs 明列 Git LFS 不可用於 GitHub Pages；若精選閱讀圖原先存於 LFS，後續以經核對的實體小圖匯入發布內容，不讓 pointer 混進 artifact。
- artifact 不可含 symbolic links 或 hard links；不把 repo 根目錄壓縮交給 Pages。

Pages 已發布網站的上限 **1 GB**，部署超過 **10 分鐘**會逾時，頻寬軟限制 **100 GB/月**。artifact 工具提及 tar 必須小於 10 GB，但它也明示官方支援網站上限仍為 1 GB，不能把 10 GB 誤當本站容量。[Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)、[upload-pages-artifact 驗證](https://github.com/actions/upload-pages-artifact)、[Git LFS 與 Pages 限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)、[MkDocs 配置與資產](https://www.mkdocs.org/user-guide/configuration/)

## Colab 開啟、分享、保存

未來每小節的 badge URL 模板如下；`<section-id>` 是規劃用占位符，目前 repo 未有對應 notebook，因此此模板不能視為現在已可執行的連結：

```markdown
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/main/notebooks/<section-id>.ipynb)
```

Google 官方示範確認公開 GitHub notebook 可以直接載入 Colab，讀取公開檔不需 GitHub 授權。私人 repo 則需在 Colab 的 GitHub 選檔器選「Include Private Repos」並另行登入／授權 GitHub。執行 managed runtime 仍由讀者的 Google／Colab 帳號提供；不能承諾網站訪客零登入便能訓練。

從 GitHub 開啟 notebook 是新的可編輯 view，執行與修改不會自動覆蓋原始 repo。讀者要保留修改，使用 **File → Save a copy in Drive**；如果需要把內容寫回自己的 GitHub，使用 **Save a copy to GitHub** 並按提示授權推送。教學網站應以「保存個人副本」作一般操作，不要求讀者取得原教材 repo 的寫入權限。

Colab 的 Share 分享 notebook 的文字、程式碼、輸出與評論，不包含 runtime 的自訂檔案、已安裝套件或 VM 狀態。`.ipynb` 的保存也不等於權重、下載資料或訓練結果已保存。重要 results／checkpoints 需在中斷前下載或另存讀者自己的 Drive。公開教材發布的 notebook 不保留個人路徑、私密輸出或大型 base64 圖片；網站選用的小圖另外保留。

來源：[Google 官方 GitHub / badge / 保存 notebook 示範](https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)、[Colab FAQ：儲存、分享及 runtime](https://research.google.com/colaboratory/faq.html)。

## Colab runtime、GPU 與資料下載的現實限制

- 免費 GPU 不保證可用；硬體類型、配額、idle timeout 和 VM 壽命會隨時間與使用情況變動。FAQ 對免費 notebook 的描述是「最多 12 小時，依可用性與使用方式」，不是保證連續可跑 12 小時；也不應承諾固定 T4、RAM 或磁碟容量。
- 選了 GPU runtime 不表示程式自動使用 GPU。需要在 notebook 檢查 CUDA／framework 是否實際使用 GPU。無 GPU 時可保留閱讀與小型 CPU 檢查路徑；完整訓練的需求與時間待實際教材驗證。
- VM 會被刪除，`/content` 中下載的資料、安裝的套件與未備份輸出都會消失。每個 notebook 的前置 cells 要能從乾淨 runtime 安裝指定依賴、下載所需資料，不能依賴作者在前一小節留下的檔案。
- 第一次執行只下載當節所需的小型資料、模型與程式。公開資料用可直接下載的 HTTPS endpoint，記錄檔案大小、dataset version、SHA-256 與 license；不要以 GitHub HTML `blob` 網頁作為下載 URL。大型資料不要走 Pages。
- 若後續使用 repo 內的小型 Python helpers，依閱讀頁版本取得相同 commit 的 helpers；不要一部分 notebook 固定版，另一部分 `git clone` 最新 `main`。
- 下載步驟检查 HTTP 失敗、實際大小與 checksum，再解壓；已取得正確版本可重用本次 runtime 的快取。不要把無驗證的下載成功訊息當作資料正確。
- Google Drive 有存取次數、頻寬和容量配額，熱門共用檔可能觸發限制。Drive 網頁共享 URL 不是保證可無認證／無 quota 的大量教學分發服務。若使用 Drive 備份，讀者自行授權；不把作者個人 Drive mount 當所有人可重現的必需步驟。
- Google 建議將 zip/tar archive 複製至 VM 並在 VM 本地解壓，以避免大量小檔從 mounted Drive 逐筆讀取。訓練資料留在本地 runtime，必要結果與 checkpoint 才回存；斷線後能重新下載與恢復。

資料託管應依實際來源與大小選擇：原資料提供者的穩定公開下載、合適的 release asset 或物件儲存均可評估。網站只連結與記錄版本，這一階段不創建 release、不搬移資料、不下載大型資料集，不額外引入 `gdown` 等工具，直到資料來源確定。

來源：[Colab FAQ：resource limits、12 小時、Drive quota、本地解壓](https://research.google.com/colaboratory/faq.html)。

## 閱讀頁、notebook、程式、資料的版本配對

開發時可先用 `main` 的 badge；要發布可重現小節時，建議將同一次網站 build 的 git commit 寫入配對清單，並由簡單 prebuild 腳本產生指向該 commit 的 Colab 連結。若採 release tag，則確認 tag 不移動且 notebook、helpers、網站來源都對應同一版本。正式部署後需真人從該頁點擊驗證一次對應的 Colab URL；JSON 與 URL 結構檢查不能代替 Colab 的實際開啟與執行測試。

每小節保留：網站來源 commit、notebook path、執行所用程式／套件版本、資料 version 與 checksum、選用結果的生成版本。notebook 安裝 cell 不可無條件升級至最新套件；weights 與 dataset 也不要只有可變的 `latest` URL。Colab 自帶 Python/CUDA 會變，首次實際驗證時記錄當時版本，再依相容性固定可固定的依賴。

網站 build 的最低檢查是 Markdown／nav／本地圖片連結正確，小節 notebook 存在且可解析，輸出圖文存在，artifact 無資料／weights／LFS pointers／links。Colab 能否取得 GPU、完成下載、保存結果與中途恢復，只能在對應 runtime 實測後宣稱通過。

## 本次研究證據與尚未驗證項目

已用匿名 HTTPS 讀取上列 GitHub、Google、MkDocs、Material 官方文件與 PyPI metadata；原始來源快照保存於 `/workspace/prep-research/sources/`。本次研究子任務未修改 `/workspace/learn_to_yolo`，未推送、發佈、變更 GitHub 設定或登入 Google。

現有 checkout 的本地分支為 `work`，可見檔案僅空白 `README.md`；此觀察不是對遠端 `main`、repo visibility、Pages 狀態的完整檢查，根代理需按其已取得的遠端證據處理。沒有教材頁或 notebook，故目前不存在可宣稱已完成的逐節 Colab 測試。
