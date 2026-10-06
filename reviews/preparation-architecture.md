# 審查紀錄：網頁與 Colab 規格

審查範圍：`docs/preparation/architecture.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

第 1 輪查核 docs/preparation/architecture.md：沒有必須修的問題，判定通過。另有 5 個建議修正（should）。

清單裡唯一一條先前審查意見（第 10 行的目錄樹）已處理：目錄樹改成現況，兩處「目前已有」已刪除，先前審查意見要求加入的項目都有。全頁 grep 不到製作經過、計畫語氣（待補、之後會、目前等）或撰寫平台的字眼。

逐句對照程式碼後確認正確的事實：
- 工作流程：pages.yml 只能手動啟動，依序跑 validate_preparation → validate_lessons → validate_curriculum_evidence → zensical build --clean --strict → validate_site，最後只上傳 site/；四個 workflow 都是 workflow_dispatch。
- 建置環境：Pages 建置不需要 PyTorch，provenance 以檔案路徑載入，torch 只在函式內 import。
- 檢查規則：validate_preparation 擋 docs/ 裡 5 MB 以上的檔、.pt/.pth/.onnx/.zip/.tar/.gz 與 LFS pointer；validate_site 再擋同樣的類型加上 .ipynb。
- notebook：build_lesson_notebooks.py 的 --ref 是必填，產生 42 本四格 notebook，改寫各頁 Colab 連結，並設定 section-map 的 source_ref。另有一本環境檢查 notebook 共 43 本，只有最後一格有輸出，而且都是 stream。
- 環境格：設 GIT_LFS_SKIP_SMUDGE=1、clone 指定 tag、用 describe --exact-match 核對版本；PyTorch 不是 2.9.1 就改裝 CPU 版；改裝前 torch 已載入就在裝完後要求重新啟動。我在暫存副本跑 tests/test_notebook_bootstrap.py，6 項全過。
- section-map：42 節 lesson 加 1 筆指向 main 的環境檢查。42 頁頁首都有「在 Colab 執行本節」按鈕，頁尾都有執行紀錄區塊。
- 實驗程式：lesson_cases 沒有使用 cuda、沒有下載、沒有預訓練權重，每支都固定 seed。03-comparison、16-training、17-capstone 三個對照實驗都是同一份資料、同一組初始化、同樣步數。
- 資料：download_data.py 的 list／fetch、--asset、--output 預設 data/downloads、bytes 與 SHA-256 核對都和頁面一致。唯一的 LFS 檔是 fashion-mnist-v1.tar（四個 gzip 加 MIT LICENSE）；pull 指令與 data.md 一致。

刪掉的「預訓練權重展示要標示」那條規則沒有任何一節適用；「各節不用預訓練權重」這個範圍現在在第 67 行正面寫出，所以沒有遺失重要內容。

渲染：在暫存副本重新同步後，zensical build --clean --strict 結束碼 0（No issues found），validate_site.py 結束碼 0。標題錨點沒變。

這個部分的 diff 只有 architecture.md。工作目錄裡其他修改來自同時進行的其他部分，沒有跡象顯示是這位編輯改的。

5 個建議修正：
1. 第 43 行的 data-excerpt 一句要等 lesson 頁標上 data-excerpt 才成立；目前 0 頁使用，excerpt 檢查有 31 個錯誤。
2. 第 45 行的「模型分支／預期資源」說法比各頁實際寫的整齊，用詞也有歧義。
3. 目錄樹：artifacts/checkpoints 與 exports 其實不存在；漏了各節程式的輸出位置與 requirements-video、GPU 的 requirements 檔；curriculum/ 的描述不完整。
4. --output /content/data 沒提到 run_fashion_cnn.py 要加 --data-root。
5. tests/test_notebook_bootstrap.py 等檔還沒加入 git，發布 commit 必須收進去。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/preparation/architecture.md 第 43 行（出版約定第 2 條）「引用程式的片段用 `data-excerpt` 標出來源檔」 | 這句描述的做法，目前 42 頁 lesson 頁沒有一頁用到：全站 grep 不到 data-excerpt。對 lesson 頁跑 excerpt_problems() 會得到 31 個「copied verbatim … mark it with data-excerpt」錯誤。也就是說，validate_lessons.py 的機制是真的，但頁面還沒照做。要等 lesson 頁那一輪把逐字片段標上 data-excerpt，這句才成立。如果那一輪改用改寫片段的方式消掉錯誤，這句就會描述一個沒有任何頁面使用的檢查，讓維護者誤以為頁面上的程式片段都有機器比對。 |
| 2 | 建議 | 第 45 行（出版約定第 4 條）「並寫明資料來源、使用哪個模型分支與預期資源」 | 「模型分支」沒有定義。同一頁第 57 行的「分支」指環境格程式的 if 分支，publish.md 的「分支」指 git branch，維護者容易讀錯。實際上 42 頁裡只有 07-training 寫出「起始分支」。多數頁（例如 09-anchor-clustering、11-csp）的正文也沒有寫預期資源，只有頁尾執行紀錄記下執行的電腦，所以這句把各頁寫得比實際整齊。 |
| 3 | 建議 | 第 9–38 行目錄樹 | (a) artifacts/checkpoints/ 與 artifacts/exports/ 在 repository 和磁碟上都不存在，只是 .gitattributes 裡的 LFS 規則；放在樹上看起來像兩個存在的空目錄。編輯回報「目錄樹每一項都存在」，這點不正確。(b) 各節程式的輸出寫在 artifacts/01-small-cnn.png、04-*.png 和 artifacts/lesson-08-own-images/、lesson-17/～lesson-20/（git ignore），不在 artifacts/runs/；樹上沒列，而 04-coordinates 頁正文會叫讀者去打開 artifacts/04-coordinates.png。(c) 樹上只列 requirements-docs.txt 與 requirements-model.txt，漏了 requirements-video.txt（18／19 節 notebook 的選用格和 record_evidence.py --run 的版本核對會用到），也漏了 requirements-gpu.in／.lock 與 requirements-modal.txt（GPU 工作流程用）。(d) artifacts/checks/curriculum/ 除了每節紀錄，還有補充實驗紀錄、deployment-gpu.json、*-metrics.json 與 index.json。 |
| 4 | 建議 | 第 69 行「`--output` 可改到 Colab 的 `/content/data` 等位置」 | 改了 --output 之後，檔案存在 /content/data/fashion-mnist/，但 run_fashion_cnn.py 預設還是讀 data/downloads/fashion-mnist/（miniyolo/classification.py 第 30 行），必須加 --data-root /content/data/fashion-mnist。〈資料規劃〉有寫這一步，本頁沒寫，這一段也沒有連到〈資料規劃〉。照本頁操作，run_fashion_cnn.py 會找不到資料。 |
| 5 | 建議 | 第 23 行（tests/）與第 57 行「由 `tests/test_notebook_bootstrap.py` 測試」 | tests/test_notebook_bootstrap.py 目前不在 git 裡。同樣沒追蹤的還有 validate_lessons.py 依賴的 scripts/review_coverage.py、scripts/evidence_records.py，以及 miniyolo/provenance.py 等檔。發布 commit 如果漏收，tag 裡就沒有本頁點名的測試，第 7 行說的 Pages 檢查也會在 validate_lessons／validate_curriculum_evidence 失敗。編輯已在疑慮提過這件事，這裡只是再強調一次。 |

建議事項的處理（處理後由下一次複查檢查）：

- [should 1｜data-excerpt] 駁回整合檢查的部分：審查用的事實與寫作規範清單寫明，發布時逐字摘錄的程式區塊都會標上 data-excerpt，這只是工作樹此刻的暫時狀態。第 2 條約定保留，改寫成 validate_lessons.py 實際檢查的內容：區塊開頭的 `python` 換成 `{ .python data-excerpt="lesson_cases/07-loss.py" }` 這類寫法，限 lesson_cases/、miniyolo/、scripts/；逐行比對時不計註解、空行與 `...`；開頭仍是 `python`、三行以上、每行都能在同一個檔找到的區塊會讓檢查失敗。這些都用 excerpt_problems() 跑假頁面逐項驗過。所以無論各頁最後是標上記號還是改寫片段，這句都成立。
- [should 2｜模型分支／預期資源] 已套用：刪掉沒有定義的「模型分支」與「預期資源」。第 4 條改寫成發布時 42 頁都具備的內容：頁首有 Colab 按鈕；頁尾的「實際執行紀錄」寫明日期、CPU 與執行緒數、PyTorch 版本，並附輸出。格式取自 verify_curriculum.evidence_block；validate_curriculum_evidence 會斷言每頁都出現日期、torch 版本與 CPU。notebook 段的「這些分支」也改成「這幾種情況」，免得和 git 分支混淆。
- [should 3｜目錄樹] 已套用 (a)～(c)，(d) 套用大部分。(a) checkpoints/ 與 exports/ 改寫成「.gitattributes 指定給 LFS 的…路徑，目錄不存在」。(b) 新增 artifacts/*.png 與 artifacts/lesson-*/（部分小節程式的輸出，被 .gitignore 排除）；artifacts/runs/ 改成「補充實驗、訓練、推論與檢查工具的輸出」，已對照各腳本 argparse 的預設值。(c) 補上 requirements-video.txt（OpenCV，18、19 節）、requirements-modal.txt（Actions 只裝 Modal 用戶端）、requirements-gpu.in／.lock（CUDA 版 PyTorch、附雜湊的完整清單；第 20 章另加 TensorRT），都對照過 workflow 與 modal_*.py。(d) curriculum/ 一行補上 index.json、多數補充實驗紀錄與第 20 章 GPU 紀錄，樹下另加一句指向 scripts/evidence_records.py（各紀錄的路徑與綁定程式）。*-metrics.json 不列：只有舊審查報告引用它們，依審查用的事實與寫作規範清單，沒有頁面引用的舊驗證紀錄發布時不在工作樹。
- [should 4｜--data-root] 已套用：補上一句，說明 download_data.py 把檔案寫到 <output>/<資料集 ID>/，而 run_fashion_cnn.py 預設讀 data/downloads/fashion-mnist/，所以用 --output /content/data 下載時要加 --data-root /content/data/fashion-mnist。已對照 download_data.py 與 miniyolo/classification.py 第 30 行，和〈資料規劃〉的說法一致。
- [should 5｜test_notebook_bootstrap.py 沒進 git] 頁面不改：這是發布 commit 要收哪些檔案的問題，不是頁面內容。這個檔在工作樹裡，6 個測試在暫存副本全數通過，本頁的敘述在發布時成立。已轉成疑慮交給整合者，並在頁面補上這組測試涵蓋的範圍：pip 與 git 都換成假的，不實際安裝或 clone。
- [額外修正｜導覽名稱] docs/planning/ 一行照導覽名稱改成「課程大綱、公開課程研究、讀者與學習心得」，原本寫的是「讀者問題與學習者個案」。
- [額外修正｜發布狀態] reviews/ 一行原本是「各節審查紀錄」，改成發布時的實況：網站每一頁都有審查紀錄，coverage.json 記下每份審查涵蓋的內容（依審查用的事實與寫作規範清單與 review_coverage.py）。scripts/ 一行補上「審查涵蓋」。

### 第 2 次複查：通過

結論：通過，沒有必要問題；有 3 條建議。

【前提變了】檢查途中出現新 commit 463d3f5（分支 release/lessons-v0.4.0），本頁已收進去，工作樹是乾淨的，所以 `git diff -- docs/preparation/architecture.md` 現在是空的。這次發布的改動要用 `git diff e25db8d -- docs/preparation/architecture.md` 看，內容和交辦時的 diff 相同。

【(1) 敘述是否屬實：成立】我把頁面每一句和審查用的事實與寫作規範清單、repository 逐項對過：
- 建置流程：pages.yml 只有手動啟動，只裝 requirements-docs.txt，檢查步驟的順序正確，任何一步失敗就不上傳。
- notebook 產生器 build_lesson_notebooks.py：四格結構；環境格 clone 時設 GIT_LFS_SKIP_SMUDGE=1、確認 tag 相符；不是 torch 2.9.1 就改裝 CPU 版，若 torch 已載入就要求重開工作階段；頁面上的 Colab 連結與 section-map.json 都被改成指向該 tag。
- 頁尾執行紀錄：verify_curriculum.evidence_block 寫出日期、CPU、執行緒數與 PyTorch 版本。
- 程式摘錄檢查：validate_lessons.py 的 data-excerpt 規則（三行以上、不計註解／空行／`...`、只限三個目錄）和頁面描述一致。
- 大小與副檔名：validate_preparation.py 的 5 MB 上限與副檔名清單、validate_site.py 再擋一次這些類型加 `.ipynb`，都和頁面寫的一樣。
- 資料下載：download_data.py 的 fetch／--asset／--output 與存放位置 <output>/<資料集 ID>/、run_fashion_cnn.py 的 --data-root，頁面寫法正確。
- 其他：requirements 各檔、modal_*.py 照 lock 檔安裝（第 20 章另加 TensorRT）、.gitignore／.gitattributes、section-map.json（42 節加 1 筆指向 main 的環境檢查）、evidence_records.py 的路徑，都和頁面一致。
- 頁面寫的是發布時的狀態，審查用的事實與寫作規範清單列出的暫時狀態（data-excerpt 標記、紀錄重產、reviews/coverage.json）不算問題。頁面沒有誇大 Colab、GPU 或審查範圍。
- 額外確認：
  - data-excerpt 寫法在 Zensical 下確實渲染成 Python 程式區塊。
  - tests/test_notebook_bootstrap.py 6 項全部通過。
  - 5 個官方參考連結都回 HTTP 200。

【(2) 先前的意見：成立】先前沒有必要。駁回第 1 條的理由符合審查用的事實與寫作規範清單。第 5 條（測試檔沒進 git）已不存在：463d3f5 已經 commit 那些檔案。

【(3) 敘述方式：成立】頁面沒有描述修訂或寫作經過，也沒有「之後會」這類計畫式語氣。

【(4) 清楚與命名：3 條 should】
- 第 69 行的「／PNG」：網站上已沒有 PNG 圖，刪掉這兩個字即可。
- 目錄樹漏列兩類發布時仍在、而且有頁面引用的紀錄：curriculum/fashion-mnist-learning.json（沒有綁定程式），以及 data/ 下的 LFS 驗證紀錄。
- docs/research/ 那一行沒照導覽寫頁面名稱：「版本來源查核」，導覽上是「版本來源查證」。

【(5) 建置與網站檢查：成立】在暫存副本裡，`zensical build --clean --strict` 與 `python3 scripts/validate_site.py` 都 exit 0。紀錄在同一層的潤稿檢查-check/ 底下。
- validate_lessons.py 與 validate_curriculum_evidence.py 在副本裡失敗，原因是審查用的事實與寫作規範清單列出的暫時狀態：摘錄還沒標 data-excerpt、紀錄已過期。這兩項不在本輪必要範圍內。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/preparation/architecture.md 第 69 行（〈純閱讀模式如何保留成果〉第一段）「存成文字、表格與 SVG／PNG 圖」 | docs/ 裡已經沒有任何 PNG 圖。grid-validation-00～03.png 已從 docs/assets/diagrams/ 移除，find docs -name '*.png' 找不到任何檔案。網站上的訓練曲線、框圖與結果圖全是 SVG，例如 grid-learning-curve.svg、08-custom-learning.svg、17-capstone.svg。「／PNG」說的格式在網站上不存在。沒有人會因為這兩個字做錯決定，屬於可以直接刪掉的多餘字。 |
| 2 | 建議 | 第 28–33 行目錄樹（data/ 幾行與 artifacts/checks/curriculum/ 一行），以及第 46 行 | 有兩類紀錄發布時仍會在，而且有頁面引用，樹上卻沒有列。(a) artifacts/checks/curriculum/fashion-mnist-learning.json：〈資料規劃〉第 45 行連到它。它不在 scripts/evidence_records.py 裡，也沒有 dependencies_sha256（綁定程式用的雜湊），validate_curriculum_evidence.py 不會檢查它。第 33 行只列「每節紀錄、index.json，另有多數補充實驗與第 20 章 GPU 紀錄」，第 46 行又說補充紀錄都列在 evidence_records.py。維護者整理 curriculum/ 時，可能把這個不在任何清單上的檔當成殘留刪掉。〈資料規劃〉連的是 github.com 的絕對網址，validate_site.py 不檢查外部連結，所以連結壞了也不會被擋下。(b) data/hosted-lfs-run.json、data/remote-lfs-verification.json、data/local-lfs-verification.json：〈資料規劃〉〈發布與帳號設定〉〈Git LFS〉都有引用。樹上只說 artifacts/checks/ 放「發布驗證與其他查核紀錄」，要找 LFS 驗證紀錄的人會找錯目錄。 |
| 3 | 建議 | 第 19 行目錄樹 docs/research/ 一行「資料、平台與版本來源查核」 | 同一段裡 docs/validation/、docs/preparation/、docs/planning/ 三行都照導覽寫出頁面名稱，這一行沒有。導覽的分區叫「資料與平台查核」，底下的頁面是「基礎資料、偵測資料、Git LFS、Pages 與 Colab、版本來源查證」。這行的「版本來源查核」和導覽上的「版本來源查證」差一個字，照這行的字到導覽裡找，會找不到那一頁。 |

### 第 3 次查核：通過

（這一輪同時查核：`docs/preparation/architecture.md`、`docs/preparation/data.md`；下表只列和本頁有關的發現。）

結論：兩頁都通過，沒有必要或建議問題。檢查用的是暫存副本，兩頁內容與 repo 現況逐位元相同。

1. 程式、指令與路徑的敘述都正確，以下逐項對照過：
   - run_fashion_cnn.py 的 --record 會寫入 executed_at_utc、machine、torch、dependencies_sha256、data_sha256 與 train_fashion_cnn 的結果。綁定的程式實算為 12 個檔：腳本本身加 11 個 repo 模組，所以頁面寫「直接或間接 import」是對的。
   - evidence_records.py 的 CPU_RECORDS 有列 fashion-mnist-learning.json；record_evidence.py 先執行 download_data.py fetch 再重產；validate_curriculum_evidence.py 有檢查這份紀錄。
   - artifacts/checks/curriculum/ 在 git ls-files 裡共 52 檔：42 節、index.json、8 份 CPU 補充紀錄、deployment-gpu.json。grid-learning.json 不在這裡，所以「多數補充紀錄」成立。
   - 5 MiB 與 validate_preparation 的 5*1024*1024 一致；docs/ 裡沒有任何 PNG。
   - verify-release.yml 與 verify_release.py 的 site／bootstrap 兩項檢查，和頁面描述相符。
   - tests/、scripts/、.github/workflows/ 三行的描述都屬實；docs/research/ 那行與 zensical.toml 的導覽名稱一致。
   - 11-fusion 那項影響引用的 data-excerpt 約定（現在在第 53 行），與 validate_lessons.py 的 EXCERPT／PYTHON_BLOCK 規則相符。

2. 實際照頁面步驟跑過以下幾項：
   - download_data.py list 列出的狀態與頁面一致。
   - 對 synthetic-rgb、voc2007、coco2017、oxford-iiit-pet 執行 fetch，都回「Candidate only; see docs/preparation/data.md」，結束碼 2。
   - fetch fashion-mnist 有效：--asset 只下載指定檔、--output 存到 <dir>/fashion-mnist/、第二次執行顯示 Verified cache（沿用已校驗的快取）。
   - 照頁面的 40 步指令跑一次：結果寫到 artifacts/runs/fashion-cnn/result.json，紀錄檔的 SHA-256 前後不變。Mac 上的 loss 與紀錄最多差 2.4e-7，正確率相同。
   - --record 搭配其他設定時，parser 會拒絕執行。
   - package_fashion_mnist.py 重建的 tar，SHA-256 與 manifest 相符。tar 裡的檔案在 fashion-mnist/ 底下，用 -C data/downloads 解開就到 loader 的預設路徑。
   - record_evidence.py 的清單有把 fashion-mnist-learning.json 列為過期。

3. 數字：
   - 我自己下載 Pet annotations.tar.gz（SHA-256 相符）盤點，得到 3,686 份 XML，其中 trainval 3,671、不在 split 清單的 15、test 0、trainval 缺 9，與頁面相符。
   - Penn-Fudan ZIP 的 SHA-256 相符，testzip 沒有壞檔，圖片、mask 與標註 TXT 各 170 份。
   - COCO 實測：官方 download.htm 列的是 http 網址；https://images.cocodataset.org 因憑證不符而失敗；S3 路徑回 200，ETag 與 HTTP 網址相同。伺服器時間是 2026-10-04 UTC。
   - 30,878,645、84.60 MB、約 55 MB 等數值都核對過。
   - 編輯沒有加入任何在 Mac 上量的數字。依紀錄而定的 loss 範圍、正確率與 weights_changed，編輯都已列出。

4. 遺留項目的判斷都正確：
   - architecture 1、2(b)、3 與 data 1、2 都已修好。
   - architecture 2(a) 判為過時是對的；若照原本的要求的修正寫「沒有綁定程式」，現在反而會是錯的。

5. 用語與一致性：
   - 兩頁都沒有敘述修訂經過，與審查用的事實與寫作規範清單、status.md、README.md、publish.md、validation/curriculum.md 都沒有矛盾。
   - data.md 新加的 #detection-downloads 錨點，讓後兩個標題的自動 id 往前移一號（#_4→#_3、#_5→#_4）。全站沒有連結指向這兩個 id。

6. 建置與檢查：zensical build --clean --strict 的結果是 No issues found，validate_site.py 與 validate_preparation.py 結束碼都是 0。README 的相對連結都存在，43 本 notebook 都是合法 JSON。

以下不是頁面問題，交給發布流程注意：
- .github/workflows/verify-release.yml 與 scripts/verify_release.py 目前還沒有被 git 追蹤，發布時一定要 commit。
- data/pages-verification.json 仍被 git 追蹤，但沒有任何頁面引用。依審查用的事實與寫作規範清單，發布前應該刪掉。
- 編輯提的第一項疑慮（detection-data.md 與 manifest 的 COCO 網址互相矛盾）已經不成立：detection-data.md 第 43 行現在寫明 manifest 存的是 S3 HTTPS 網址。
- 兩頁的定稿審查要以現在的文字為準。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

讀了 reviews/preparation-architecture.md 三輪查核並查前段工作流程的標籤。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/preparation-architecture.md | 第 1 次 5 項、第 2 次 3 項 should 的處理都沒列；第 3 次只寫「兩頁都通過，沒有 must 或 should 問題」，看不出每項怎麼處理。紀錄寫「建議事項在下方〈定稿修正〉逐項處理」，但沒有這一節。處理其實在 resolve:architecture／edit:architecture-data 的結果裡，生成器沒讀。 | 已修正：產生器讀入修正時處理的項目與建議事項的處理（處理後由下一次複查檢查），不再輸出空的清單，每項發現都有對應的處理。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次 5 項建議的處理已補上清單（上一輪要求）；第 2 次 3 項，由第 3 次逐項判定（architecture 1、2(b)、3 已修好，2(a) 判為過時）；頁面有指令，紀錄記了實際執行；本頁沒有定稿修正，也沒有掛批次檢查。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/preparation-architecture.md 第 85、138 行 | 殘句與內部名稱：「紀錄在同一層的 polish-check/ 底下：暫存副本、暫存副本。」「編輯提的第一項 concern」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 138 行已改成「疑慮」；但第 85 行仍是殘缺路徑，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/preparation-architecture.md 第 85 行 | 點名的「紀錄在同一層的 polish-check/ 底下：暫存副本、暫存副本。」只把 polish 換成「潤稿檢查」並刪掉後半，變成「紀錄在同一層的潤稿檢查-check/ 底下。」，仍是殘缺路徑。處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 2 | 建議 | reviews/preparation-architecture.md 第 3 輪第 1 項處理欄 | 「「疑慮」改成「疑慮」」是同語反覆。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[reference當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/reference.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/reference.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/reference.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-05：v0.6.0 有界更新審閱

新增TinyViT／TinyDINO／bridge與選讀官方DINOv2架構及發布版本／限制。

本次僅重查以上變更，既有正文的歷史審閱保留；沒有把全頁或全書重新標成首次盲讀。[導讀／實際網站審閱](clear-tutorial/vision-v0.6.0/guide-visual-review.md)、[新增支線第三輪及實際前置](clear-tutorial/vision-v0.6.0/transitions-review.md)、[首讀修後複查](clear-tutorial/vision-v0.6.0/vit-recheck.md)記錄各自範圍。全版52本notebook的本機cell執行另見[實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)；不是Google Colab登入執行。來源與圖／程式指紋更新於[coverage.json](coverage.json)。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次全52節的實際方法、處置、證據與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。本頁為維護說明，沒有當作52節中的首次盲讀目標。作者逐項核對命令、目前機器與保存紀錄、圖／coverage的更新流程；非作者技術報告另核其實際涉及的維護delta。

本頁對應處置為 R012，見[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。[基礎技術報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-foundations.json)與[最後維護delta核回](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)保存各自時點；[第三輪](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)與[最後綁定](clear-tutorial/full-review-2026-10-06/final-bindings.json)區分實際查核範圍。

沒有真人學生驗收，四題亦不能保證不漏報；本次 DINO 漏報、引用缺陷及手機細字限制已明列。[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)保存實際命令結果；此頁最新來源綁定於[coverage.json](coverage.json)，歷史審閱保留。

## 2026-10-06：v0.6.1 有界修正複查

僅將當前教材版本與環境提示從 lessons-v0.6.0 同步到 lessons-v0.6.1；舊驗證與歷史版本的範圍保留。

本輪方法、逐批閱讀原始紀錄、非作者技術核對、另一位讀者銜接與實際 Zensical 修改段落截圖見 [v0.6.1 局部複查](clear-tutorial/release-v0.6.1-2026-10-06/README.md)。本次只重新檢查修改處、必要上下文與版本一致性；其餘正文、程式及圖的既有審閱保留原範圍，不改標為整頁或全書新的首次盲讀驗收。
