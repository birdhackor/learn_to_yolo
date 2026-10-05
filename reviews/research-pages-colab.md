# 審查紀錄：Pages 與 Colab

審查範圍：`docs/research/pages-colab.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要等級的問題，另有 4 項建議建議修。

**先前審查意見與寫法**
- 7 條先前審查意見都已處理：第 3、31、33、35、54 行的遷移敘述已刪除，第 58 行改成照 pages.yml 逐步描述，第 68 行不再寫死 tag。
- 全頁沒有製作或修訂經過、計畫式語氣（待補、之後會），也沒有提到撰寫平台。

**逐條事實核對**（對照程式與官方文件）
- 程式與設定：`.github/workflows/pages.yml`（觸發方式、concurrency、checkout `lfs: false`、Python 3.12 與 pip 快取、五個步驟的名稱與指令、`upload-pages-artifact@v3`、`deploy-pages@v4`、`needs`、逾時、權限）、validate_preparation／validate_lessons／review_coverage／validate_curriculum_evidence／evidence_records／validate_site、build_lesson_notebooks（`--ref` 寫入的位置與環境格行為）、record_evidence、verify_curriculum、`zensical.toml`、requirements-docs.txt、overrides、mathjax.js 與 MathJax 3.2.2、`.gitignore`、section-map.json。
- 指令實測：`zensical build`／`serve --help` 與頁面相符；serve 的根網址會 302 轉到 `/learn_to_yolo/`，所以「預設網址是 `localhost:8000`」沒問題。
- 測試：`tests/test_notebook_bootstrap.py` 6 passed；42 本 notebook 環境格的 REF 都與 `source_ref` 一致。
- GitHub 設定：`gh api` 確認 Pages 來源是 workflow、`github-pages` environment 存在、repository 是 public。
- 官方文件：Zensical 搜尋文件（`lang`／`pipeline` 不支援、搜尋介面沒有在地化）、GitHub「Git LFS cannot be used with GitHub Pages sites」、Colab GitHub 示範 notebook 的原文都與頁面相符；頁面所有外部連結都回應 200。

**範圍與刪除內容**
- 範圍說法沒有超出紀錄：v0.2／v0.3 的發布驗證紀錄都寫明不是 Colab 託管 runtime 的測試；publish.md 第 6 節的發布驗證也只在 Colab 開一節，所以「沒有逐節在 Colab 執行過」在發布後仍然成立。GPU 與審查範圍也沒有誇大。
- 刪掉的兩句都沒有造成遺漏。「artifact 不含 credentials」沒有任何檢查守著，刪除是對的。「公開 notebook 原始檔也已核對」放在 v0.4.0 是說過頭，而這項操作寫在 publish.md 第 6 節第 10 步，本頁有連過去。原本的範圍限制（Google 登入、Colab GPU、GPU 完整訓練）與手動檢查步驟（中英文搜尋、直接開頁／換頁／深淺色）都還在。

**建置**
- 在暫存副本裡，`zensical build --clean --strict` 與 `validate_site.py` 都 exit 0（重跑一次也一樣）。產生的頁面有 6 個 h2，第 1–5 步正確轉成編號清單。
- 這次查核的頁面的 diff 只有 `docs/research/pages-colab.md`。共用工作樹裡其他被改的檔案屬於別的部分。

**4 項 should**
1. 第 3 步把「日期／PyTorch／CPU 要寫在執行紀錄區塊裡」寫得比檢查嚴；程式只查整頁。
2. 第 2 步的「剛好 42 節」與檔案實際的 43 筆對不上；「文字與圖」實際只涵蓋 SVG。
3. 「改到 Colab 會執行的程式要發新 tag」的列舉漏了環境格本身。
4. 「環境格」先用、後說明。

**附帶（不屬於本頁）**
- 審查用的事實與寫作規範清單寫「每節一份」審查紀錄，但 `review_coverage.py` 要求導覽裡每一頁都要有；本頁照程式寫，是對的。
- 工作樹目前沒有 `reviews/coverage.json`，發布前不補上的話，CI 第 2 步會失敗。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/research/pages-colab.md 第 69 行（Pages 發布，第 3 步） | 「頁面的執行紀錄區塊要寫著紀錄的日期、PyTorch 版本與 CPU 型號」寫得比程式實際檢查的範圍大。validate_curriculum_evidence.py 第 74–77 行只要求頁面恰有一個 `<!-- curriculum-evidence:start -->` 和一個 end 標記；日期（`executed_at_utc` 的前 10 字）、`torch` 字串和 `machine.cpu` 出現在整頁任何位置都算通過，沒有限定要在區塊裡。這次查核指示要求照 pages.yml 實際執行的內容描述檢查，這句把守衛的範圍寫大了。 |
| 2 | 建議 | docs/research/pages-colab.md 第 67–68 行（Pages 發布，第 1、2 步） | (a) 第 2 步說「`section-map.json` 要剛好有 42 節」，但這個檔案實際有 43 筆。多出的一筆是 kind 為 `preparation` 的 `preparation-environment`（對應 docs/index.md 與 00_environment_check.ipynb，source_ref 是 main）。validate_lessons.py 第 70–71 行只數 kind 為 `lesson` 的項目。第 1 步的「每個就緒的小節」又把這一筆算進去，兩步的範圍不一樣，維護者對照檔案會數到 43。(b) 「紀錄涵蓋該頁目前的文字與圖」也寫得太寬：review_coverage.py 第 55 行的 digest 只收頁面上用 Markdown 連結顯示的 `.svg`，docstring 和 publish.md 也都寫 SVG。目前沒有頁面顯示其他格式的圖，所以現在不會出錯。 |
| 3 | 建議 | docs/research/pages-colab.md 第 87 行（Colab 與教材版本，第三段） | 括號裡列出的「Colab 會執行的程式（`lesson_cases/`、它們 import 的模組，以及 notebook 可選實驗執行的程式）」漏了環境格本身。環境格由 `scripts/build_lesson_notebooks.py` 的 `bootstrap()` 產生，固定了 torch 2.9.1、numpy、Pillow、matplotlib 的版本，第 20 章另外固定 onnx 與 onnxruntime。照字面讀，只改環境格時不屬於列出的任何一類，維護者會以為不必發新 tag。而且沒有檢查會比對 notebook 的環境格與 `bootstrap()`：validate_lessons／validate_site 只核對 Colab 連結與 metadata 的 `source_ref`。結果是 Colab 讀者拿不到這次修改，CI 也不會擋下。 |
| 4 | 建議 | docs/research/pages-colab.md 第 83 行與第 91 行 | 「環境格」在〈Colab 與教材版本〉第一段就出現兩次（「notebook 的環境格」「環境格要 clone 的 tag」），但「環境格（每本 notebook 的第二格）」這個說明要到第五段才出現，讀者先讀到的是沒有說明的詞。 |

建議事項的處理（處理後由下一次複查檢查）：

- [should] 第 3 步執行紀錄區塊：已採納。改寫成「課程頁上執行紀錄區塊的開始標記與結束標記都要恰好一個，紀錄的日期、PyTorch 版本與 CPU 型號也要出現在頁面上（比對的是整頁文字，不限區塊內）」，與 scripts/validate_curriculum_evidence.py 第 92–95 行一致（檢查內容沒變，只是另一個 agent 改檔後行號從 74–77 移到這裡）。工作樹裡就有實例：07-training.md 的 CPU 型號只出現在第 192 行正文，不在區塊裡。同一句裡的「並確認 Modal 上的 app 已經停止」順手改成「並記載 Modal 上的 app 已確認停止」，因為這一步只讀紀錄裡的 modal_stop_confirmation.verified，不會去問 Modal。
- [should] 第 1、2 步的 42 與 43 筆：已採納。第 1 步寫明 section-map.json 除了 42 節課程，還有一筆 kind 為 preparation 的環境檢查 notebook（〈首頁〉連到的 notebooks/00_environment_check.ipynb）；每一筆的 ID 都不能重複、引用的資料集都要在 manifest 裡，就緒的還要有閱讀頁、notebook 與 source_ref（validate_preparation.py 第 30–40 行）。第 2 步改成「kind 為 lesson 的項目要剛好 42 個，而且都已就緒」（validate_lessons.py 第 70–71、75 行）。另查過 validate_site.py 第 223 行也只數 lesson，所以第 5 步和搜尋段落寫的「42 節」本來就對。
- [should] 「文字與圖」寫得太寬：已採納。改成「文字與頁面上的 SVG 圖」（review_coverage.py 第 55 行只收 Markdown 引用的 .svg）。也查過 docs/assets/diagrams 裡有 4 張 PNG，但沒有任何頁面引用，所以現在每一頁顯示的圖都是 SVG。
- [should] 發新 tag 的清單漏了環境格：已採納。這段改成先講原因：各節 notebook 和它執行的程式都取自發布 tag，所以改到其中任何一部分都要發新 tag。再用「例如」列出：環境格與它固定的套件版本（由 build_lesson_notebooks.py 的 bootstrap() 產生）、lesson_cases/、它們 import 的模組、可選實驗執行的程式。測試那句也寫明 tests/test_notebook_bootstrap.py 執行的是 bootstrap() 產生的環境格，不是已提交的 notebook（測試 import 這個函式後用 exec 執行它的輸出）。
- [should] 「環境格」第一次出現時沒有說明：已採納。說明移到〈Colab 與教材版本〉第一段第一次出現的地方，寫成「（各節 notebook 的第二格）」，後面那段直接用「環境格」。用「各節」不用「每本」，是因為環境檢查 notebook 沒有環境格；同一個理由，「最後一格是完整實驗程式」那句也改成「各節 notebook」。
- [額外，非檢查意見] 頁尾兩個連結原本寫「發布步驟」「驗證範圍」，照查核指示的導覽名稱規則改成〈發布與帳號設定〉與〈驗證範圍與後續實驗〉（名稱取自 zensical.toml 第 78、22 行）。在暫存副本跑 zensical build --clean --strict 與 validate_site.py 都通過（exit 0），建好的頁面裡兩個連結都正確渲染。

### 第 2 次複查：通過

結論：通過。沒有必要等級的問題，另有 4 個建議等級的建議。

【我查了什麼】
- 對照了審查用的事實與寫作規範清單、git diff，以及這一頁實際描述的程式：pages.yml、zensical.toml、validate_preparation／validate_lessons／validate_curriculum_evidence／validate_site、review_coverage、evidence_records、miniyolo/provenance、build_lesson_notebooks 的 bootstrap()、tests/test_notebook_bootstrap.py、verify_curriculum、record_evidence。
- 逐句核對：第 1–5 步每一項檢查內容都與程式一致。第 3 步採預設 `--scope all`，所以 GPU 紀錄也會檢查，而且不需要 PyTorch。section-map.json 有 43 筆（42 筆 lesson 加 1 筆 preparation，後者的 source_ref 是 main）。審查涵蓋的範圍（頁面文字、Markdown 引用的 SVG、課程程式與 import 的模組；頁尾執行紀錄區塊與 Colab tag 不算）與 review_coverage.py 一致；42 頁的執行紀錄區塊都在頁尾，結束標記之後沒有其他文字。環境格的每個分支都有對應的測試。
- 外部事實也查過：Zensical 0.0.67 的 Requires-Python 是 >=3.10；`build --strict` 遇到警告確實中止（我在暫存副本放一個壞連結實測，exit 1）；serve 預設 localhost:8000；搜尋相關敘述與官方文件一致；GitHub 文件明寫「Git LFS cannot be used with GitHub Pages sites」；這個 repo 的 Pages 設定 build_type 是 workflow，也有 github-pages environment。
- 頁面沒有敘述修訂經過，也沒有計畫語氣。導覽名稱〈首頁〉〈發布與帳號設定〉〈驗證範圍與後續實驗〉都和 zensical.toml 相符。上一輪的 4 個建議都已改好（上一輪沒有必要）。

【建置檢查】
在暫存副本執行 `zensical build --clean --strict`和 `python3 scripts/validate_site.py`，兩者都通過；頁尾兩個站內連結也正確渲染成 ../../preparation/publish/ 與 ../../status/。

另外在暫存副本做了三個實驗：strict 模式、symlink 的處理、清除舊檔。結果是 Zensical 會把指向檔案的 symlink 複製成一般檔案，指向目錄或已失效的 symlink 直接略過，`--clean` 也會清掉 site/ 裡的舊檔。所以 validate_site 的 symlink 檢查雖然只抓得到指向檔案的 symlink，但建置產出本來就不會有 symlink，實際上沒有人會因此出錯，我沒有列為問題。同樣沒列的還有：「repository 裡不能有 mkdocs.yml」實際只檢查根目錄，以及「網址都要成為連結」不檢查 code 裡的網址。這兩處也指不出會因此做錯事的人。

【維護者提的疑慮】
已提交的 notebook 環境格沒有任何檢查去比對 bootstrap() 目前的輸出。這頁並沒有宣稱有這道檢查，只寫了發新 tag 時要用 build_lesson_notebooks.py 重建，所以這頁本身沒有錯。要不要補這道檢查，仍由上層決定。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/research/pages-colab.md 第 13 行（版本與原生設定）與第 47 行（建置與預覽的指令） | 第 13 行寫「本機建立 `.venv-docs` 時也用 3.12，與 CI 相同」，但第 47 行的指令是 `python3 -m venv .venv-docs`，建出來的版本取決於當下的 `python3`。這台維護用電腦的 `python3` 目前是 3.14.8（pyenv shim），照頁面指令重建會得到 3.14 的環境，與 CI 的 3.12 不同；頁面沒有說明怎麼確保用的是 3.12。現有的 `.venv-docs` 是 3.12.15，所以這句話描述現況是對的，問題在照著指令做的人。 |
| 2 | 建議 | docs/research/pages-colab.md 第 68 行（Pages 發布，第 2 步） | 「標了 `data-excerpt` 的程式摘錄，每一行都要出現在它標明的原檔裡」寫得比程式實際檢查的範圍大。validate_lessons.py 的 `code_lines()`（第 18–42 行）會先去掉註解、空行與 `...` 省略行並合併空白，只比對剩下的程式行。照字面讀，維護者會以為摘錄裡的註解與原檔不同也會被擋下，但實際上不會。〈網頁與 Colab 規格〉（architecture.md 第 51 行）寫的是「每行程式（不計註解、空行與 `...` 省略行）」，兩頁說法不一致。 |
| 3 | 建議 | docs/research/pages-colab.md 第 69 行（Pages 發布，第 3 步） | 「補充紀錄」在這頁第一次出現就是這一步，但頁面沒說它指哪些紀錄。維護者要對照程式才知道：它是 scripts/evidence_records.py 的 `CPU_RECORDS`，也就是 7.4 的 160 步訓練、8.2 的兩份、三份 40 步學習實驗、兩尺度實驗與影片檔紀錄。 |
| 4 | 建議 | docs/research/pages-colab.md 第 93 行（Colab 與教材版本，最後一段） | 這段寫了單元測試，也寫了 CPU 紀錄不涵蓋 Google 登入與 Colab，卻沒提發布時在 Colab 上實際做的那一項檢查：〈發布與帳號設定〉第 6 節第 11 步要在 Colab 開一節新 tag 的 notebook、執行環境格，結果記在 `artifacts/checks/curriculum-publication.json`。審查用的事實與寫作規範清單寫這件事時是兩句一起講（「有單元測試……在 Colab 上的實際執行另見發布驗證紀錄」），〈驗證範圍與後續實驗〉第 29 行也連到這份紀錄。只讀這頁的維護者會以為 Colab 入口完全沒有人驗。這段現有的句子都沒有錯，只是缺了這一項。 |

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

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/research/pages-colab.md 第 13 行（版本與原生設定）「所以先用 `python3 --version` 確認它是 3.12」 | 這句只叫維護者檢查版本，沒說檢查不通過時怎麼辦。這台維護用電腦的 `python3` 是 3.14.8，照著做的人會停在「不是 3.12，接下來呢？」。README 第 11 行同一件事有寫下一步：「不是的話，換成 Python 3.12 直譯器的路徑」，這頁沒有。 |
| 2 | 建議 | docs/research/pages-colab.md 第 71 行（Pages 發布第 3 步）「補充紀錄（各節以外的補充實驗在 CPU 上的紀錄）」 | 「各節以外」容易被讀成「這些補充實驗不屬於任何一節」。實際上 scripts/evidence_records.py 的 CPU_RECORDS 九份裡，有八份是特定小節的可選實驗（第 1、3、4、10 章的 40 步、7.4 的 160 步、8.2 的兩份、18／19 的影片檔），只有 Fashion-MNIST 那份屬於〈資料規劃〉。〈網頁與 Colab 規格〉（architecture.md 第 48 行）寫的是「各節的補充實驗，以及〈資料規劃〉的 Fashion-MNIST 40 步核對」，兩頁的措辭方向相反。照上下文也能讀成「42 節逐節紀錄以外的紀錄」，所以不算錯，但意思不夠明確。不會有人因此做錯操作，因為同一句已經指名清單在 evidence_records.py。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | python3 不是 3.12 時沒說下一步 | 未改：已經修好：第 13 行已寫明換成 3.12 的直譯器，例如 python3.12。 |
| 2 | 先前查核 | 「各節以外的補充實驗」措辭不清 | 未改：已經修好：第 71 行已是「各節的補充實驗與〈資料規劃〉的 Fashion-MNIST 核對在 CPU 上的紀錄」。 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

.venv-docs 的 zensical-0.0.67 METADATA 為 Requires-Python >=3.10；五個 workflow 都用 3.12。tomllib 由 validate_site.py 直接 import，validate_lessons.py 經 review_coverage.py 間接 import，所以兩支在 ≤3.10 都會出現 ModuleNotFoundError；validate_preparation 與 validate_curriculum_evidence 不需要它，句子正確。「補充紀錄（各節的補充實驗與〈資料規劃〉的 Fashion-MNIST 核對在 CPU 上的紀錄）」與 evidence_records.py 的 CPU_RECORDS 相符；內容檢查只有 custom-data-learning、video-file、fashion-mnist 三份，GPU 檢查 status、gpu_calls_finished 與 modal_stop_confirmation，都和 validate_curriculum_evidence.py 一致。五步順序與 pages.yml 相同。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

讀了 reviews/research-pages-colab.md 三輪查核與〈定稿修正〉。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/research-pages-colab.md | 第 2、3 次的 should 有交代（第 3 次寫「pages-colab #1–#3…已改」，〈定稿修正〉2 列），但第 1 次的 4 項 should（第 69 行檢查範圍、section-map 42 節、第 87 行括號、「環境格」重複）沒有任何處理紀錄（resolve:pages-colab 沒被讀入）。 | 已修正：產生器讀入修正時處理的項目與建議事項的處理（處理後由下一次複查檢查），不再輸出空的清單，每項發現都有對應的處理。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次 4 項建議的處理清單已補上（上一輪必要問題）；第 2 次 4 項，由第 3 次逐項判定；第 3 次 2 項在〈定稿修正〉寫「已經修好」，並由〈後續編輯的檢查〉核對 tomllib 與補充紀錄兩句。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/research-pages-colab.md 第 29、44、56、172 行 | 殘句與內部名稱：〈獨立查核之後的編輯〉開頭的殘留標籤「E：.venv-docs 的 zensical-0.0.67 METADATA…」、「這次派工要求照 pages.yml 實際執行的內容描述檢查」、「照派工的導覽名稱規則」、「在暫存副本 `暫存副本` 裡」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 172 行的「E：」已刪，第 44、56 行已改成「查核指示」，第 29 行已清理。處理說明屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/research-pages-colab.md 第 3 輪第 1 項處理欄 | 「「查核指示」改成「查核指示」」是同語反覆。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

第 3 輪第 1 項屬實。第 4 輪第 1 項不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪第 1 項處理欄（第 196 行） | 點名的同語反覆「「查核指示」改成「查核指示」」在第 3 輪第 1 項處理欄，該格已重寫成「已修正：點名的文字已不在紀錄裡（改寫或刪除）。」。處理欄卻寫「未改：…「查核指示」「查核指示」」；這個字串只出現在正文第 44、56 行（改好後的用語）。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
