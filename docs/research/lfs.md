# learn_to_yolo 的 Git LFS 準備建議

> 這是研究代理在子任務當時的查核報告。後續整合已套用前置配置並下載部分資料；目前狀態與操作步驟請以「前置準備」頁為準。

查核日期：2026-10-02 UTC。研究範圍只使用 GitHub 與 Git LFS 專案官方文件。本文件是建議，未修改 `birdhackor/learn_to_yolo`、Git 設定、LFS tracking 或遠端狀態；未登入、push、發佈或下載 dataset。檢查時 checkout 只有 `.git` 與空的 `README.md`，尚無資產需要遷移。

## 結論與課程資產分工

若某章確實需要較大的、低頻更新、需和教材版本一起固定的二進位檔，適合先試 Git LFS。課程預設仍應是「clone 程式與說明 → 只下載本章資料」。幾何資料從 seed 生成；完整 VOC／COCO 等公開資料、上游預訓練權重、大型影片和大量實驗 checkpoint 採外部按需下載。Git LFS 本地可用不代表 GitHub LFS 上傳權限、配額或遠端下載已驗證。

| 資產 | 建議位置／方式 | 理由 |
| --- | --- | --- |
| Python、notebook、配置、labels、固定 split、class mapping、manifest、下載來源與 SHA-256 | 一般 Git | 可 review、diff 與重現；notebook 不保存巨大的 output 或嵌入影片 |
| 幾何訓練圖 | Git 保存生成程式與 seed，runtime 生成 | 不需要重複保存可再生資料 |
| 課程小圖、架構圖、網頁縮圖與少量小樣本 | 一般 Git，先壓縮 | GitHub Pages 直接使用；避免整體套用 `*.png`／`*.jpg` LFS |
| 已確認可散布、固定版本的小型真實資料子集 archive | 可試 LFS；以章節／子集拆成可獨立下載的檔案 | 與教材版本一致；讀者不需取得整份 dataset |
| 少量自有的短示範影片、指定章節的 reference checkpoint／ONNX | 可試 LFS，明確標為可選 | 二進位較大，更新頻率低；checkpoint 不取代從零訓練路線 |
| 完整 VOC／COCO、第三方完整資料、上游模型、長影片、多版訓練輸出 | 官方來源／適合的外部資料儲存，按需下載 | 避免課程 clone 大量下載，也避免 owner 的 LFS storage／bandwidth 快速消耗 |

選擇 LFS 的工作門檻可先訂為「單檔約 20–50 MiB 以上、且需要版本固定」，這是專案建議，不是 GitHub 規則，也不表示小於門檻的所有檔案都適合一般 Git。Git LFS **不能依檔案大小自動 tracking**；官方 FAQ 說 `.gitattributes` 沒有大小條件，需按路徑／副檔名選擇。[S8]

## GitHub 的大小、額度與帳單規則

| 項目 | 官方目前規則 | 對本課程的影響 |
| --- | --- | --- |
| 一般 Git 單檔 | 大於 **50 MiB** 警告；大於 **100 MiB** 阻擋；瀏覽器上傳單檔上限 **25 MiB** | 100 MiB 以上不可當普通 Git blob push；避免為了沒達上限而把很多二進位塞入 Git |
| Repo 建議大小 | 理想低於 **1 GB**；強烈建議低於 **5 GB** | 這是 repo 建議，不能當成「每檔低於 100 MiB 就可無限制放資料」 |
| LFS 每檔 | Free／Pro **2 GB**；Team **4 GB**；Enterprise Cloud **5 GB** | 帳號方案未查詢；先以 Free 的 2 GB 上限規劃，每章檔案盡量遠低於此上限 |
| LFS 免費 storage／下載 bandwidth | Free、Pro、Free for organizations 各 **10 GiB storage + 10 GiB bandwidth**；Team／Enterprise Cloud 各 **250 GiB + 250 GiB** | 是 owner 帳號包含額度，不是每份 repo／每位讀者各得一份 |
| LFS 帳單形式 | 預付 data packs 已改為按使用量計費；bandwidth 的免費額度每 billing cycle 重置；storage 依每小時占用累計 | 舊文常見的 1 GB + 1 GB／預購 packs 不應用於新規劃 |
| LFS storage | 每個完整檔案版本計入 owner storage；500 MB 檔案即使只改 1 byte，新版本仍再占約 500 MB | 把資料 pack 做成低頻更新的不可變版本；不要每次訓練都推 checkpoint |
| LFS bandwidth | 下載計入 owner；上傳不計 bandwidth；Actions 下載及包含 LFS 的 archive 下載也計 bandwidth；fork／pull 仍影響 parent owner | 公開課程的讀者數，比單次資料大小更容易形成負擔 |
| 超額／budget | `$0` budget 超額會阻止 LFS 使用；刪除 budget 意味沒有 spending limit；無有效付款方式在額度用完後阻止使用 | 不自動調整帳單或預算；首次遠端試驗前由 owner 核對目前額度與 budget |

上述分別見 [S1]–[S3]。單檔上限原文使用 **GB**，LFS 專用 billing 頁的免費額度使用 **GiB**，此處保留原文單位。一般「included product usage」表顯示 GB，本建議以 LFS 專用頁為額度來源。

GitHub 官方計算器目前列額外 storage **US$0.07／GiB**、額外 data transfer out **US$0.0875／GiB**；storage 的月份成本依 billing 文件按 GiB-hours 換算 GiB-months。[S3][S10] 計算器同頁仍有「In the future ... switch to metered billing」舊措辭；是否已採按量計費以目前 billing 文件的明確說明為準。實際付款與 budget 設定應以 owner 的帳單頁為準，本研究沒有開啟或修改帳單設定。

例：如果一個子集為 **500 MiB**，10 GiB 免費下載額度只約等於 **20 次**完整下載；200 位讀者各下載一次約為 97.66 GiB。這還沒算其他 repo、CI 或重複下載。分章、只下載選定路徑，以及在讀者自己的 runtime／Drive 保留快取都有實際意義。

刪掉目前工作樹檔案或增加 `.gitignore`，不會移除 Git 歷史內的 blob，也不會保證清除歷史 LFS storage。LFS 官方 FAQ 明確說只對既有檔案加入 tracking 並重新 stage，不會修改歷史；完整 migration 會重寫歷史。[S8] 本 repo 目前沒有需要 migration 的資產，不應為準備工作先執行 `git lfs migrate --everything` 或 force push。

## GitHub Pages 的界線

GitHub 官方明文：**“Git LFS cannot be used with GitHub Pages sites.”**[S2] Pages 提供教材 HTML、圖片、Colab 按鈕與資料下載說明；dataset／模型由 Colab 透過選定 LFS 路徑或外部來源取得。不要把 LFS pointer 路徑當作 Pages 的 dataset URL，也不要把 LFS 當成 Pages 的資料 CDN。

Pages 已發佈站點還有 **1 GB** 上限、**100 GB／月** soft bandwidth limit 等限制。[S9] 自訂 Actions build 可自行下載 LFS 成實體檔，但下載仍計 LFS bandwidth，輸出仍受 Pages 限制；不能因此宣稱 Pages 直接支援 LFS，亦不建議用這條路發布 dataset。若需要大檔發行，GitHub 官方也提出 Releases 作為散布二進位檔的選項。[S1] 本研究未建立 Release 或上傳資產。

## 建議 `.gitattributes` 模式（尚未套用）

先保留明確的小範圍資產目錄，只針對已選定的用途追蹤。例如：

```gitattributes
data/lfs/mini-voc/*.tar.gz filter=lfs diff=lfs merge=lfs -text
data/lfs/video/*.mp4 filter=lfs diff=lfs merge=lfs -text
artifacts/reference/*.pt filter=lfs diff=lfs merge=lfs -text
artifacts/reference/*.onnx filter=lfs diff=lfs merge=lfs -text
```

這些是規劃目錄，不是 repo 中已存在的路徑。只加入真的會使用的行，不必為未選修的部署或影片預先放資產。若一章只有一個確定的 archive，也可把 pattern 限定到確切檔名。避免全 repo 的 `*.jpg`、`*.png`、`*.zip`、`*.pt`，以免把網站圖、其他小檔或使用者生成輸出一起變成 LFS。路徑名稱統一小寫；matching 有大小寫區別。[S8]

官方建議將 `.gitattributes` 一起 commit，讓 fresh clone／fork 可重現相同 tracking，而不依賴作者的 global attributes。[S4] 同時需要 `.gitignore` 忽略下載的外部完整 dataset、runtime cache、個人訓練輸出；具體 ignore 路徑待教材目錄確定後再設定。

## 本地到遠端：按步驟完成的建議

以下是後續實作建議，**此研究未執行**：

1. 先列出首章實際需要的檔案、bytes、來源、可散布條件、SHA-256、與教材版本的關係。先挑一個小而真實的 asset 作端到端試驗，避免用完整 dataset 驗證 LFS。
2. 檢查 `git lfs version`。Git LFS 是另行安裝的程式，只有 Git 並不足夠。[S5] 若需初始化，明確選 `git lfs install --local`，不用全域設定；若 repo 已有 hooks，按 Git LFS 的 `--manual` 指引整合，不盲目 `--force` 覆蓋。[S6]
3. 對選定路徑執行如 `git lfs track 'data/lfs/mini-voc/*.tar.gz'`，檢查生成的 `.gitattributes`；stage `.gitattributes` 與該測試 asset。`git show :path/to/file` 應看到 `version https://git-lfs.github.com/spec/v1`、`oid sha256:...`、`size ...` 的 pointer，而不是檔案內容。[S2][S4]
4. 本地檢查 `git lfs status`、`git lfs ls-files`，在完整持有本次測試 objects 的測試 repo 中執行 `git lfs fsck --dry-run`，並比較原檔 checksum。這能證明 pointer／本地 object 一致；若另在隔離本地 repo 測 clone／還原，也仍只證明本地環節。
5. 遠端上傳是另一項檢查：需要實際 owner／collaborator 的寫入權限、GitHub LFS endpoint 可達、未用盡配額與合適 budget。現有 HTTPS Git read 或本地 LFS 成功都不能取代此項驗證；本研究沒有 push，遠端上傳未驗證。
6. 在未來已授權的上傳完成後，使用乾淨 clone、跳過 smudge、只 `pull` 該測試路徑，確認大小與 SHA-256，再測最小讀檔／影像解碼。這才證明 GitHub LFS 實際可供讀者使用。Git commit 可見而 LFS object 缺失仍會造成讀者只拿到 pointer 或下載失敗。
7. 確認小試驗後才納入首個資料 pack；在教材中標示 archive 大小、下載量與可選性。持續保存可重現的資料 manifest，避免為小修改重製大型 archive。

## Colab 的選定下載範本（尚未執行）

目前 repo 沒有 `mini-voc-v1.tar.gz` 或對應 manifest，下列為未來 notebook 的範本。每章替換成實際路徑，**不要直接把不存在的路徑當成已驗證 setup**：

```bash
%%bash
set -euo pipefail
if ! git lfs version >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git-lfs
fi

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/birdhackor/learn_to_yolo.git /content/learn_to_yolo
cd /content/learn_to_yolo
git lfs install --local --skip-smudge
git lfs pull \
  --include='data/lfs/mini-voc/mini-voc-v1.tar.gz' \
  --exclude=''
sha256sum -c data/manifests/mini-voc-v1.sha256
```

`GIT_LFS_SKIP_SMUDGE=1` 在 clone 時保留 pointer；`--depth 1` 只縮短 Git 歷史，**不會自動限制當前 commit 的 LFS 資料量**。在 Colab checkout 中使用 `--local --skip-smudge` 後，後續 checkout／pull 不會自動下載全部 objects，需按章執行明確的 `git lfs pull --include=...`。`--exclude=''` 清除該次呼叫的既有 exclude 設定，避免排除掉本次明確要的檔案。[S6][S7][S8]

沒有 Git LFS 時，clone 只能得到 pointer，不能把小文字檔誤當已下載圖片／權重。[S11] checksum manifest 需由作者事先以真實 archive 計算並保存在一般 Git；此驗證可同時排除 pointer、截斷檔或錯誤版本。

**不要**使用無 include 的 `git lfs pull` 或 `git lfs fetch --all` 作每次 Colab 初始化；前者會下載 current ref 的全部 LFS 資料，後者會下載指定 refs 可達各 commit 的 objects，未指定 refs 時涵蓋所有 refs，官方把它定位為 backup／migration 用途。[S7][S13] 也不要在刻意只下載一章之後直接以全 repo 的 `git lfs fsck --objects` 作必過檢查：官方說此命令預設檢查 HEAD／index 的 objects 是否存在，而其他章尚未下載是預期狀態。[S12] 使用本章 checksum 與實際讀檔檢查即可。

## 原文來源與日期

GitHub Docs 頁面本身沒有列 `effectiveDate`，此處用查核日期 **2026-10-02**。Git LFS manuals 使用官方專案 `main`，查核時 commit 為 `0043a645047926f4bd7f7091299095528253d575`，commit 日期 **2026-09-22**；下列 manuals 連結固定到該 commit。另查得 GitHub Docs billing 頁來源最近修改 commit 為 `7c06b1d8a446c632b1df2cf4dac7a1da70627bbb`（**2026-07-09**），about-LFS 頁為 `6c39b7b7dee4ce3fdcabe563f6bb8e3b15b342ac`（**2026-07-08**）；這是來源檔修改日期，不代表政策生效日期。

- **[S1]** GitHub Docs, [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。原文：“larger than 50 MiB ... warning”；“GitHub blocks files larger than 100 MiB.”；repo 大小建議、瀏覽器 25 MiB、Releases 選項。
- **[S2]** GitHub Docs, [About Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。各方案單檔上限、pointer 格式，以及 “Git LFS cannot be used with GitHub Pages sites.”
- **[S3]** GitHub Docs, [Git Large File Storage billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)。原文：“pre-paid data packs ... removed and replaced with metered billing”；Free／Pro／Free org 10 GiB、Team／Enterprise 250 GiB；完整新版本 storage、owner bandwidth、budget、超額和 GiB-hours。
- **[S4]** GitHub Docs, [Configuring Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)。原文：“We strongly suggest that you commit your local .gitattributes file into your repository.”
- **[S5]** GitHub Docs, [Installing Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)。原文：“a new program that's separate from Git.”
- **[S6]** Git LFS 官方, [git-lfs-install(1)](https://github.com/git-lfs/git-lfs/blob/0043a645047926f4bd7f7091299095528253d575/docs/man/git-lfs-install.adoc)。`--local`、`--manual`、`--skip-smudge`；原文：“Skips automatic downloading of objects on clone or pull.”
- **[S7]** Git LFS 官方, [git-lfs-pull(1)](https://github.com/git-lfs/git-lfs/blob/0043a645047926f4bd7f7091299095528253d575/docs/man/git-lfs-pull.adoc) 與 [git-lfs-config(1)](https://github.com/git-lfs/git-lfs/blob/0043a645047926f4bd7f7091299095528253d575/docs/man/git-lfs-config.adoc)。`--include`／`--exclude`、空字串清除該次設定、`GIT_LFS_SKIP_SMUDGE`。
- **[S8]** Git LFS 官方, [git-lfs-faq(7)](https://github.com/git-lfs/git-lfs/blob/0043a645047926f4bd7f7091299095528253d575/docs/man/git-lfs-faq.adoc)。不支援按檔案大小自動 tracking、pointer、clone smudge、既有檔案／history migration、pattern 大小寫。
- **[S9]** GitHub Docs, [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)。1 GB site 上限、100 GB／月 soft bandwidth。
- **[S10]** GitHub 官方, [Pricing calculator — Git LFS](https://github.com/pricing/calculator?feature=lfs)。額外 storage／data transfer out 單價。
- **[S11]** GitHub Docs, [Collaboration with Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage)。原文：“If collaborators ... don't have Git LFS installed ... they will only fetch the pointer files.”
- **[S12]** Git LFS 官方, [git-lfs-fsck(1)](https://github.com/git-lfs/git-lfs/blob/0043a645047926f4bd7f7091299095528253d575/docs/man/git-lfs-fsck.adoc)。HEAD／index objects 檢查範圍、`--dry-run`；有意只下載一章不宜對全 repo 要求 objects 全部存在。
- **[S13]** Git LFS 官方, [git-lfs-fetch(1)](https://github.com/git-lfs/git-lfs/blob/0043a645047926f4bd7f7091299095528253d575/docs/man/git-lfs-fetch.adoc)。`--all` 的範圍；原文：“This is primarily for backup and migration purposes.”
