# 審查紀錄：Git LFS

審查範圍：`docs/research/lfs.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

通過。沒有必要問題，有 3 項建議。

1. 清單項目：查核指示單列的 13 條全部處理了，也照額外指示在頁首加了一句指向〈發布與帳號設定〉。我掃過整頁的製作或修訂敘述、規劃語氣與平台細節，都沒有殘留。還留著的「帳號方案未查詢」與「研究範圍只使用官方文件」是照實寫的範圍限制，適合有日期的研究紀錄。

2. 事實核對：本頁所有關於 repo 現況的句子都對得上。
- `.gitattributes` 只有 `data/curated/**`、`artifacts/checkpoints/**`、`artifacts/exports/**` 三個 LFS 路徑。
- 三個 ignore 路徑各自有實際用途：`data/downloads/` 是 `scripts/download_data.py:65` 的預設下載位置；`.cache/` 是 `miniyolo/train.py` 第 127–130 行設定的快取目錄（HEAD 是第 124–127 行）；`artifacts/runs/` 是 `miniyolo/train.py:205` 的預設輸出。
- 一般 Git 歷史沒有大檔：查了 main、origin/main 與三個 `lessons-v*` tag（repo 不是 shallow），最大的 blob 是 1,173,007 bytes（MathJax）。`data/curated/fashion-mnist-v1.tar` 在 Git 裡確實是 pointer。
- 示意路徑 `data/lfs`、`artifacts/reference`、`data/manifests` 不在檔案系統也不在 git 裡。
- 兩份驗證紀錄都存在。本地那份註明只在隔離的本地 repo 測過；遠端那份記錄 Actions 上傳成功，並從空的 LFS 快取下載核對通過。
- 42 本教材 notebook 的環境格都用 `GIT_LFS_SKIP_SMUDGE=1` 加 `--branch REF` clone（目前 REF 是 `lessons-v0.3.0`），沒有任何 `git lfs pull` 或 `fetch`。
- publish.md 現在的 `## 5. Git LFS` 一節涵蓋 LFS 路徑、Fashion-MNIST 封裝與兩份驗證紀錄；連結文字「發布與帳號設定」與 `zensical.toml:78` 的導覽標題一致。

3. 刪掉的句子：被刪的都是對 owner 的回報、空 checkout 的快照，以及已被現況推翻的狀態句。真正的範圍限制還在（研究範圍只用官方文件、文件是建議）。唯一的退步是舊頁首「操作步驟以前置準備頁為準」這層優先順序不見了，列為建議第 1 項。

4. 建置：在暫存副本跑 `zensical build --clean --strict` 與 `scripts/validate_site.py`，兩者都 exit 0，連結與錨點全部通過，頁首引言的連結正確轉成 `../../preparation/publish/`。

5. 改動範圍：跟 LFS 有關的檔案（`.gitattributes`、`.gitignore`、`data/*-lfs-verification.json`、`scripts/verify_remote_lfs.py`）都沒改。編輯者暫存副本裡的 `lfs.md` 和 repo 裡的完全相同。其他修改中的檔案屬於別的單位，例如 `data/manifest.json` 的 synthetic-rgb 條目、`zensical.toml` 的課程編號。

另外一點不列為問題：publish.md 的標題已經是 `## 5. Git LFS`，之後若不再改名，頁首連結可以加上錨點；就算加錯，`validate_site` 也會抓到。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/research/lfs.md 第 3 行（頁首引言） | 舊引言說「目前狀態與操作步驟請以「前置準備」頁為準」，新引言把「操作步驟以該頁為準」拿掉了，只說「實際採用的 LFS 路徑、封裝與驗證結果」在〈發布與帳號設定〉。但本頁第 64–74 行的 7 個步驟是一般性建議，缺了本 repo 的必要步驟：新增 asset 要先寫進 `data/manifest.json` 的 `lfs_assets`（`scripts/verify_remote_lfs.py` 只核對這份清單裡的檔），重新發布要跑 Publish and verify LFS dataset workflow。第 66 行又緊接著提到「本 repo 的…驗證紀錄」，讀起來像本 repo 的做法。維護者照本頁新增 asset 時，可能漏寫 `lfs_assets`，那個檔就不會被遠端核對，也沒有任何檢查會發現。 |
| 2 | 建議 | docs/research/lfs.md 第 78 行與範本第 88–89 行（Colab 的選定下載範本） | 新加的第一句說教材 notebook 的環境格 clone 固定的 tag，接著說「需要某個 LFS 檔時，可照下列範本只取那一個檔」。但範本的 `git clone --depth 1` 沒有 `--branch`，拿到的是預設分支 main，不是 notebook 用的 tag，還會在環境格的 clone 之外再 clone 一份。本頁第 9 行說 LFS 適合「需和教材版本一起固定」的檔，照範本取檔卻會和教材版本脫鉤。〈發布與帳號設定〉第 133 行的同一種範本已經註明「教材的 notebook 要用時，改成 clone 固定的發布 tag（`git clone --branch <tag>`）」，本頁沒有這句。 |
| 3 | 建議 | docs/research/lfs.md 第 41 行末句 | 原句是「不應…執行 `git lfs migrate --everything` 或 force push」，清單提議的是「不要…」。改寫後成了「不必為了改用 LFS 執行…」，禁止的語氣變成「沒有必要」。前一句才說完整 migration 會重寫歷史，而這是一個已有公開發布 tag 的 repo，禁止語氣比較貼切。〈發布與帳號設定〉另有「不要 force push」，所以這不算遺失唯一的安全知識，只是語氣變弱。 |

建議事項的處理（處理後由下一次複查檢查）：

- should, line 3 (intro does not say where the operation steps are): fixed. The intro now says the repo's LFS paths, packaging, verification results and operation steps (操作步驟) are in the Git LFS section of 〈發布與帳號設定〉. That section really does hold the steps: first-time remote LFS use, the lfs_assets registration, the Publish and verify LFS dataset workflow and the Colab single-asset pull. Because a maintainer can land directly on the steps section, I also added a line to the 「本地到遠端」 lead-in (line 66): when adding a new LFS asset to this repo, first register it in data/manifest.json lfs_assets, because scripts/verify_remote_lfs.py checks size and SHA-256 only for files in that list. That line links to the same section. Verified against the code: the script loops over manifest['lfs_assets'], and nothing in scripts/ or tests/ checks that LFS files are registered.
- should, line 78 (Colab template clones the default branch, not the release tag): fixed. Added: 範本 clone 的是預設分支；在教材的 notebook 裡用時，改成 clone 固定的發布 tag（`git clone --branch <tag>`），才能確定取得的 LFS 檔和該版教材一致. This matches the wording on 〈發布與帳號設定〉. The template is unchanged. Verified: the environment cell in scripts/build_lesson_notebooks.py runs `git clone --depth 1 --branch REF` with GIT_LFS_SKIP_SMUDGE=1, and origin/HEAD is main. I wrote 「才能確定…一致」 ('only then is a match guaranteed') rather than 「才會一致」 ('only then will it match'), because the latter is false when main and the tag hold the same pointer.
- should, line 41 (prohibition weakened from 不要 to 不必): fixed. The sentence now reads 本 repo 的一般 Git 歷史沒有需要遷移的大檔；不要為了改用 LFS 執行 `git lfs migrate --everything` 或 force push. I re-checked the stated reason: the largest blob reachable from any ref is about 1.17 MB (the MathJax bundle), and the working tree has no file over 2 MB outside the ignored and LFS paths, so the reason still holds at release.

### 第 2 次複查：通過

結論：沒有必要問題，判定通過。另有兩個小的建議，都只是精確度問題，不會讓人做錯操作。

【先說明比對基準】工作樹是乾淨的，`git diff -- docs/research/lfs.md` 沒有輸出。本版對這頁的改動（含上一輪的三項修正）都已在 HEAD 463d3f5，所以我拿 HEAD 對照 lessons-v0.3.0 檢查。

【(5) 建置】依維護者的 rsync 方式（排除 .git、.venv-*、site、.cache）複製到暫存副本，兩項都通過：
- `zensical build --clean --strict` 結束碼 0，輸出 No issues found。
- `python3 scripts/validate_site.py` 結束碼 0，連結與錨點、Markdown 轉換、artifact 邊界都通過。

【(2) 上一輪三項建議都已修正，修法正確】
- 第 3 行引言已加回「操作步驟」。
- 第 66 行新增的 lfs_assets 說明屬實：scripts/verify_remote_lfs.py 只對 manifest['lfs_assets'] 列出的檔核對 bytes 與 SHA-256；publish.md 第 111 行也有同樣的步驟。
- 第 78 行的 `--branch <tag>` 說明和 publish.md 第 135 行一致。範本 clone 的是預設分支，遠端 HEAD 確實是 main。
- 第 41 行已改回「不要」。理由仍成立：所有 ref 可達的最大 blob 是 1,173,007 bytes 的 MathJax，Actions 發布 pointer 的 commit 7b5f1db 也在本地歷史中。

【(1) repo 事實逐項查證，都屬實】
- .gitattributes 只對三個路徑啟用 LFS。
- .gitignore 的三個路徑：data/downloads/ 是 download_data.py 的預設輸出；.cache/ 是 train.py 的 matplotlib／XDG 快取，也是 Zensical 的建置快取；artifacts/runs/ 是訓練的預設輸出。
- 示意路徑 data/lfs、artifacts/reference、data/manifests 在 repo 裡都不存在。
- 本地與遠端兩份 LFS 驗證紀錄都在，內容和頁面描述相符。
- 42 本課程 notebook 的環境格都用 `GIT_LFS_SKIP_SMUDGE=1` 加 `--depth 1 --branch REF` clone，REF 是 lessons-v0.4.0。
- notebook 最大 22 KB，沒有嵌入圖片。
- ShapeDataset 每個索引各有自己的 seed。

【外部來源抽查，都相符】
- git-lfs 的 0043a645 確實是 2026-10-02 當時的 main，commit 日期 2026-09-22。
- FAQ、install、pull、fetch、fsck、config 各 man page 的引述與說法都相符。
- GitHub Docs 的來源 commit 7c06b1d（2026-07-09）與 6c39b7b（2026-07-08）屬實。
- 數字都對：LFS 額度 10 GiB／250 GiB；LFS 單檔 2／2／4／5 GB；一般 Git 50／100／25 MiB；repo 建議 1 GB／5 GB；Pages 1 GB 與每月 100 GB；Pages 不能使用 LFS 的原文。
- 唯一沒驗到的是定價計算器的單價：那是動態頁，抓不到內容。

【(3)(4)】沒有修訂經過的敘述，也沒有計畫語氣：掃到的「目前」「仍」「改成」等字，都是在描述規則或下指示。頁名〈發布與帳號設定〉和導覽一致。頁面沒有宣稱 Colab 實際跑過、GPU 結果或審查範圍。

【附帶說明，不算本頁問題】在暫存副本跑 `scripts/review_coverage.py` 會列出 `docs/research/lfs.md: no review`（結束碼 1），因為工作樹裡沒有 reviews/research-lfs.md。這是審查用的事實與寫作規範清單列為「發布時才會完成」的暫時狀態，發布前要補上審查，再執行 `review_coverage.py --write docs/research/lfs.md`。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/research/lfs.md 第 35 行「上述分別見 [S1]–[S3]。單檔上限原文使用 **GB**，…」 | 「單檔上限原文使用 GB」沒有說是 LFS 的單檔上限。這半句緊接在「上述分別見 [S1]–[S3]」之後，可是上表第一列「一般 Git 單檔」來自 [S1]，原文單位是 MiB（50／100／25 MiB），表格也照原文寫 MiB。原文用 GB 的只有「LFS 每檔」那一列（[S2]）。照字面讀，這半句和表格第一列互相矛盾。它不會讓人做錯操作，只會讓核對單位的讀者困惑。 |
| 2 | 建議 | docs/research/lfs.md 第 41 行末句「不要為了改用 LFS 執行 `git lfs migrate --everything` 或 force push」 | `git lfs migrate` 一定要接一個 mode（`info`、`import` 或 `export`）。固定 commit 0043a645 的 git-lfs-migrate(1) 寫的用法是 `git lfs migrate <mode> [options]`，[S8] FAQ 裡會重寫歷史的指令是 `git lfs migrate import --everything`，所以頁面寫的這條指令實際上不存在。這句是禁令，不會有人照著它執行錯的指令，因此只是精確度問題；但本頁其他地方的指令都寫得精確。 |

### 第 3 次查核：通過

（這一輪同時查核：`docs/research/pages-colab.md`、`docs/research/lfs.md`、`docs/research/detection-data.md`、`docs/research/foundation-data.md`、`docs/planning/course-research.md`；下表只列和本頁有關的發現。）

結論：通過。沒有必要等級的問題，只有兩條建議，都在 pages-colab.md。

**這次查核的頁面改了哪些檔**
- 只改了 pages-colab、lfs、detection-data、foundation-data 四頁。
- course-research.md 與 HEAD 相同，HEAD 已經是 4 格縮排。
- 受程式改動影響的段落沒有這五個檔的條目。

**遺留意見的判斷：9 條都正確**
- pages-colab #1–#3、lfs #1–#2、foundation-data：已改，改法和程式一致。
- pages-colab #4：判為「事實已改變」是對的。依據有三：
  - verify_release.py 的 scope 明寫 "Colab itself was not opened"。
  - 審查用的事實與寫作規範清單寫的是 Verify published lessons workflow。
  - publish.md 第 6 節第 11 步現在就是這個 workflow，所以編輯者擔心的 publish.md 落差已經不存在。

**detection-data 的 must（COCO 的 HTTPS 網址）**
- 官方 download.htm 列的是 http:// 網址，頁面連結也是這組。
- 我在 2026-10-04 19:21 UTC 用 HEAD／range 重查：三個檔的官方 HTTP 網址與 S3 HTTPS 路徑都回 200／206，Content-Length、ETag、Last-Modified 兩邊相同；https://images.cocodataset.org 則是 curl exit 60。
- 其他說法也都對得上：
  - DNS 的 CNAME 指向 images.cocodataset.org.s3.amazonaws.com。
  - 憑證是 CN=s3.amazonaws.com，SAN 不含 images.cocodataset.org；urllib 回報 Hostname mismatch。
  - 單張圖片用 S3 HTTPS 也回 200。
- 暫存副本的 coco-probe-final.log（16:43:27Z）支持頁面寫的 16:43 UTC 結果。
- data/manifest.json 的 coco2017 是三個 S3 HTTPS 網址，status 是 candidate，與頁面一致。

**逐條核對程式**
- 對照的程式：validate_preparation、validate_lessons（code_lines、SOURCES）、review_coverage、validate_curriculum_evidence（finite_json、check_machine，以及 custom 1600 步、video、fashion 的內容檢查）、evidence_records、validate_site（含 markdown_list_problems）、build_lesson_notebooks 的 bootstrap()、pages.yml、verify-release.yml、verify_release.py、tests/test_notebook_bootstrap.py、zensical.toml、overrides/、docs/assets/（mathjax.js 訂閱 document$；MathJax 3.2.2）。頁面的敘述都成立。
- 外部來源：
  - PyPI 0.0.67 的 Requires-Python 是 >=3.10，wheel 為 cp310-abi3。
  - Zensical 數學公式文件確有 document$ 與 instant navigation 的整合說明。
  - git-lfs 在 0043a645 的 FAQ 有 `git lfs migrate import --everything`；GitHub 的 LFS 單檔上限單位是 GB。
- git 歷史裡最大的 blob 是 1.17 MB。
- 五頁的 82 個外部連結都回 200 或 206。

**實際照頁面做的步驟**
- 用 Python 3.12 建 .venv-docs、裝 requirements-docs.txt。
- 依序跑 5 個指令：
  - validate_preparation 0。
  - validate_lessons 只因所有導覽頁都「no review」而失敗。
  - validate_curriculum_evidence 只因紀錄還沒重產而失敗（00-warmup 的 case_sha256 不符）。
  - zensical build --clean --strict 0。
  - validate_site 0（含 numbered lists 檢查）。
- validate_lessons 與 validate_curriculum_evidence 的失敗都屬於審查用的事實與寫作規範清單列的「發布時才完成」狀態。
- zensical serve 預設在 localhost:8000，根路徑會轉到 /learn_to_yolo/。
- 用 Python 3.9 實跑：validate_lessons 與 validate_site 確實停在 `ModuleNotFoundError: No module named 'tomllib'`；用 3.10 語法解析這些腳本都通過。
- tests/test_notebook_bootstrap.py 6 項通過。

**數字**
- 沒有新加入 Mac 上量的數字，也沒有依紀錄而變的數字。
- 頁面的 byte 數、換算與加總都正確。
- 用 SHA-256 為 52425fb6… 的 annotations.tar.gz 重驗 Pet 的盤點：3,686／7,390／3,680／3,669／9／41，以及 Abyssinian_1 的 [107,81,444,328)、[92,66,460,343)、CRC，全部相符。

**敘述與一致性**
- 沒有敘述修訂經過，範圍限制都寫成現況。
- 與 status.md、validation/curriculum.md、architecture.md、publish.md、data.md、README 一致。
- 「像素」改「畫素」與術語表一致；連結文字〈驗證範圍〉與 status.md 標題相同。

**渲染**
- README 的 9 個相對連結都存在；43 本 notebook 都是有效 JSON。
- course-research 建置後只有一個 8 項的 &lt;ol&gt;，8 行「來源：」都在項目內；headless Chromium 截圖顯示編號 1–8。

**檔案位置**
- 紀錄與截圖：同路徑的暫存副本（prep.log、lessons.log、evidence.log、build.log、site.log、links.log、course-research-top.png）

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：通過

讀了 reviews/research-lfs.md：第 2 次寫「上一輪三項建議都已修正」，第 3 次寫「lfs #1–#2…已改」，每項都有交代。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/research-lfs.md | 兩處「建議事項在下方〈定稿修正〉逐項處理」指向不存在的段落；「修正後由另一位 AI 檢查改動」的內容是別批（「指派的 14 頁裡…其餘 12 頁沒有 diff」），本頁在那一批沒有修正；另有「派工單」「暫存副本」。 | 已修正：沒有〈定稿修正〉時不再寫指向句；修正後的檢查只掛在這一批真的有改動的頁面。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次 3 項建議有處理清單；第 2 次 2 項，由第 3 次確認（「lfs #1–#2…已改」）；不再指向不存在的〈定稿修正〉，也不再掛別批的檢查。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/research-lfs.md 第 26、51–52、150 行 | 〈上述處理〉第 1 項點名「暫存副本」殘句並寫「內部用語換成白話」，但仍有「4. 建置：在 `暫存副本` 跑…」「（記錄檔暫存副本）」×2，以及「- 暫存副本- 紀錄與截圖：同路徑的暫存副本（prep.log…」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 26、51–52 行已改；第 150 行只改了一個字元，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/research-lfs.md 第 150 行 | 點名的「- 暫存副本- 紀錄與截圖：同路徑的暫存副本（prep.log…」只少了一個「- 」，變成「- 暫存副本紀錄與截圖：同路徑的暫存副本（prep.log、…）」，仍是殘句。處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

第 4 輪第 1 項屬實：第 150 行殘句仍在。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 3 輪第 1 項處理欄（第 170 行） | 「5 處中 4 處已改寫或刪除」把引用的第 2 輪處理說法「內部用語換成白話」也算成已改寫，但它仍在第 162 行。實際改寫的紀錄文字是 3 處。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[reference當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/reference.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/reference.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/reference.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本附頁由獨立銜接與技術審閱者核對當前來源；頁內既有審閱歷史的接觸另記，未冒稱完全無歷史提示。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。
