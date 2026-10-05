# 審查紀錄：課程大綱

審查範圍：`docs/planning/outline.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

判定：通過，沒有必要問題，只有三項建議等級的建議（見下表）。

1. 痕跡與用語：9 條先前審查意見都已處理（outline 7 條、zensical.toml 2 條）。我逐行掃過全文，沒有修訂、改版或規劃期的用語（本輪、第二版、待測、候選、之後會），也沒有寫到撰寫工具。第 84 行的「這個版本沒有 ignore」講的是第 7 章那個模型，跟 07-targets.md 的寫法一致，不是改版敘事。H1 已改為「從小 CNN 到 MiniYOLO：課程大綱」，導覽列也顯示「課程大綱」。

2. 事實核對：每一句都對照過以下來源，都對得上：
   - 42 節課程頁、lesson_cases 程式、section-map.json（42 節加 1 個準備項）、miniyolo/ 的模組
   - artifacts/checks 的紀錄：grid-learning.json 是 160 步、validation/test seed 為 700/7000；custom-data-160-step.json 的 mAP 接近 0；deployment-gpu.json 與 gpu-smoke 兩份 L4 紀錄
   - 審查用的事實與寫作規範清單、data.md、learning-path.md、feedback.md

   以下逐項確認過：
   - 課程模型沒用 BatchNorm 與 Dropout；第 10 章的 TwoScale 是另外寫的網路。
   - 第 11 章四個部分不組合；第 14 章是參數 800 對 1168，計算量是正文手算的。
   - 第 15.2 節有「和 CNN 比較：每層讀得到哪些位置」。
   - 第 16.3 節只對 Progressive Loss 做實驗。
   - 第 17 章的 seed 是 1100、2200、3300，選定模型只評一次 test，也有停止條件。
   - 第 18 章的相機 adapter 從未被呼叫；INT8 只有說明。
   - 紅框 [8,12,24,28] 出現在 7.1、7.2、7.3、7.5、7.6、9.1、11.3、11.4。
   - 沒有任何程式或課程頁用 Penn-Fudan、VOC、COCO 訓練或評估。
   - Pages 工作流程會用 validate_curriculum_evidence.py 強制 notebook 最後一格與 lesson_cases 程式相同。

   GPU、Colab、審查範圍都沒有說得比紀錄多：頁面沒有宣稱本頁已經審查過。

3. 刪掉的內容：被刪的句子都是規劃期、課程沒做到的內容。真實資料集與 v6/v7/v9 等範圍限制，已經改寫進〈課程刻意不做的事〉，沒有遺失唯一的範圍限制、安全步驟或維護操作。

4. 建置：在暫存副本跑了 `zensical build --clean --strict` 與 `scripts/validate_site.py`，兩者都 exit 0。輸出的頁面表格、粗體與連結都正常。

5. zensical.toml：工作區的 diff 裡還有一批章節編號的導覽名稱，那是其他部分先前改的。證據是 16:32 建立的暫存副本已經有這些編號，同時還是舊的「第二版大綱」。這次查核的頁面只改了第 82 行，與 editor 的報告一致。

以下不屬於這次查核的頁面，供負責人參考：
- docs/planning/feedback.md 與 docs/research/detection-data.md、foundation-data.md 連到本頁時，連結文字仍是「教學大綱」；foundation-data.md 還引用了 /workspace 路徑。
- 導覽名稱「驗證範圍與後續實驗」帶有規劃用語「後續實驗」。
- 目前 42 本 notebook 只有 11 本的最後一格與 lesson_cases 程式相同，發布前要用 `build_lesson_notebooks.py --ref lessons-v0.4.0` 重建。
- 本頁改動後，validate_lessons.py 會透過 review_coverage 把本頁的審查列為過期，需要補 reviews/planning-outline.md。

相關檔案：docs/planning/outline.md、zensical.toml

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/planning/outline.md 第 146 行〈資源規格以實測為準〉（對照第 136、155 行） | 第 146 行寫「各節實驗都在 CPU 上執行」，但同一頁第 136 行寫「TensorRT 在一張雲端 NVIDIA L4 上實測」，第 155 行也寫「GPU 上只在雲端 NVIDIA L4 做過兩項小規模檢查」。三句擺在一起，讀者可能以為前後矛盾：第 20 章不是也在 GPU 上跑過嗎？其實「各節實驗」指的是 lesson_cases 的完整程式，全部在 CPU 上執行；L4 上的兩項是另外加做的檢查。 |
| 2 | 建議 | docs/planning/outline.md 第 3 行（開頭段），以及第 67–69、84–86、105–112 行第一次出現術語的地方 | slot、objectness、assignment、ignore、DFL、STAL、MuSGD、C3k2 等術語，大多沒有解釋，頁面也沒有連到術語表。只補了 grid detector、assert、「起點」與 anchor-free 的說明。本頁讀者是對課程設計好奇的人，其中包括高中程度的初學者，讀到第 5 章的 slot 就可能卡住。 |
| 3 | 建議 | docs/planning/outline.md 第 3 行 | 開頭段說本頁會說明「每章……讀到哪裡算完成」，但頁面只替第 0–8 章列出完成條件，第 17 章列的是交付項目。第 9–16 章（表格）和第 18–20 章都沒有完成條件，這句承諾的內容比頁面實際寫的多。 |

建議事項的處理（處理後由下一次複查檢查）：

- [should] 第 146 行 CPU 與 L4 看似矛盾：已採納。〈資源規格以實測為準〉改成「各節的完整程式都只用 CPU 執行……；用到 GPU 的只有另外在雲端 NVIDIA L4 上做的兩項檢查：GPU／checkpoint 實測，以及第 20 章的 TensorRT 一致性（見下方〈課程刻意不做的事〉）」。把 GPU／checkpoint 實測放在前面，避免讀成兩項都屬於第 20 章。第 20 章段落也改成「先在 CPU 上驗證 ONNX Runtime 與 PyTorch……」「TensorRT 另外在一張雲端 NVIDIA L4 上實測」，讓兩處說法一致。查證：lesson_cases 都沒有用 CUDA；20-deployment.py 用的是 CPUExecutionProvider；miniyolo.train 的 --device 預設 cpu，且沒有任何一節呼叫它；modal_deployment.py 與 modal_gpu_smoke.py 都是 gpu="L4"；各份 CPU 紀錄（包括 grid-learning.json）的 device 都是 cpu。
- [should] 術語沒解釋、也沒連到術語表：已採納。開頭段加上「遇到不熟的術語，可以查[術語快速查](../glossary.md)」。slot 第一次出現時補上術語表的原定義「一個能輸出一組框與分數的位置」；沒用檢查意見建議的「一個框」，因為那樣少了分數，與術語表不一致。另外在第一次出現處補短註：assignment（責任分配）、ignore（某項 loss 完全不算的候選）、objectness（物件分數）。表格裡的 CSP、decoupled head、DFL、Progressive Loss、STAL、MuSGD 也加了短註，其中 CSP、decoupled head、Progressive Loss、STAL、MuSGD 不在術語表裡。用語取自術語表與 11.1、12.2、12.4、16.3 節的原文；Progressive Loss 寫成「逐步移向」一對一分支，因為權重終點是 0.1／0.9，並沒有全部移過去。C3k2 不加註，該列已寫出它的「切分→轉換→串接」機制。
- [should] 開頭承諾「每章讀到哪裡算完成」超出頁面實際內容：已採納。改成「每章要解決什麼問題、做哪些實驗，第 0–8 章讀到哪裡算完成、第 17 章結業任務要交付什麼，以及課程刻意不做的事」。已逐章核對：第 0–8 章都有完成條件，第 17 章是交付項目。
- [依任務規則，非檢查意見] 導覽名稱：開頭的 [閱讀路線] 改為 [完整閱讀路線]，[驗證範圍] 改為 [驗證範圍與後續實驗]，與 zensical.toml 的 nav 一致。頁面上其他連結名稱（全套實驗與審查、GPU／checkpoint 實測、資料規劃、版本來源查證、公開課程研究、讀者與學習心得）原本就是 nav 名稱。
- 測試：在暫存副本裡，zensical build --clean --strict 結束碼 0，scripts/validate_site.py 結束碼 0（最後一次修改之後重跑過）。產出頁面含 glossary、learning-path、status 三個連結，表格也正常轉成 HTML 表格。

### 第 2 次複查：通過

第 1 輪檢查：通過，沒有必要問題，只有 1 個建議。

我逐句對照了審查用的事實與寫作規範清單、各課程頁、lesson_cases 程式與 zensical.toml，結果如下：

(1) 陳述都屬實。
- 42 節計數正確。
- 各節都有前置；版本節都說明起點與簡化。
- 42 支 lesson_cases 都有 assert，也都沒用 CUDA。
- miniyolo 與 lesson_cases 都沒用 BatchNorm 或 Dropout，與第 0 章說法一致。
- 3.3 確實用 load_state_dict 共用初始權重。
- 紅框 [8,12,24,28] 確實出現在 7.1、7.2、7.3、7.5、7.6、9.1、11.3、11.4。
- 第 17 章的 seed（1100、2200、3300）、協議與停止條件都相符。
- 第 18–20 章、STAL、MuSGD、Progressive Loss 的描述都與課程頁相符。
- GPU 只有兩項 L4 檢查（gpu-smoke 與 deployment-gpu），與審查用的事實清單一致。
- 頁面沒有過度宣稱 Colab、GPU 或審查範圍。

(2) 前一輪 3 個建議都已採納並確實解決，這一輪沒有被駁回的意見。

(3) 頁面沒有敘述修訂或製作經過，也沒有計畫語氣。第 11 行的「這次實驗」指的是該節的實驗，不是教材的修訂史。

(4) 開頭有術語表連結，slot、assignment、ignore、objectness 與表格內術語的短註都與術語表一致。所有連結名稱都和 nav 相同。唯一的建議在第 154 行：「分類與物件偵測以外的任務不在範圍內」與同頁第 19 章 tracking 矛盾，建議直接列出不涵蓋的項目。

(5) 用 rsync 把 repo 複製到暫存副本（排除 .git、.venv-*、site）。在副本執行：
- zensical build --clean --strict：結束碼 0，No issues found，日誌在暫存副本。
- python3 scripts/validate_site.py：結束碼 0，連結、錨點、Colab 配對與 Markdown 轉換都通過，日誌在暫存副本。

不屬於本頁、只供參考的觀察：
- docs/lessons/07-training.md 與 08-own-images.md 仍寫 miniyolo.train 的報告預設寫到 artifacts/checks/grid-learning.json。但 miniyolo/train.py 的實際預設是 --output 目錄下的 report.json，與審查用的事實清單一致，所以是那兩頁過時。
- docs/lessons/20-deployment.md 寫「程式沒有檢查解出的框數大於 0」，但 lesson_cases/20-deployment.py 現在已經 assert 每張來源都有框。
- docs/lessons/20-deployment.md 還有「本輪」字樣。
- 本頁的審查紀錄要在發布前補上；這屬於工作樹的暫時狀態，不是頁面本身的問題。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/planning/outline.md 第 154 行〈課程刻意不做的事〉第三點「分類與物件偵測以外的任務（例如 segmentation、keypoint 姿態估計），以及半監督學習，也不在範圍內」 | 「分類與物件偵測以外的任務」把範圍框得太大，連 tracking 也算進去了。但同一頁第 132 行的第 19 章就在教簡易 tracking（多物件追蹤也是另一種任務），第 128 行的第 18 章教影片串流。所以這句照字面讀是錯的，和頁面本身矛盾。讀者看到時會短暫困惑，不過上方第 D 部分就列著第 18、19 章，不會因此誤判範圍或做錯事，所以不列為必要。 |

### 第 3 次查核：通過

（這一輪同時查核：`docs/index.md`、`docs/learning-path.md`、`docs/planning/outline.md`；下表只列和本頁有關的發現。）

結論：沒有必要問題，通過。只剩一個 should：首頁「再散佈」和全站「再散布」用字不一致。

我做了哪些檢查（都在暫存副本裡做，log 在同層的暫存副本）：

1. 建置與驗證：zensical build --clean --strict 回報 No issues found、exit 0；validate_site.py exit 0，連結、錨點、Colab 配對、Markdown 呈現都通過；validate_preparation.py exit 0。README 的相對連結都指得到檔案，43 本 notebook 都是合法 JSON。另外跑了 validate_lessons.py，只因為還沒有定稿審查而失敗，這是審查用的事實與寫作規範清單列出的暫時狀態，不算這次查核的頁面的問題。repo 裡這三頁和編者暫存副本的同名檔逐位元相同；同一時段其他被改動的檔案，各屬於其他部分。

2. 頁面敘述和程式、指令對照（實際跑過的有）：
   - 首頁的格子算法：用 build_targets 驗證，紅框 [8,12,24,28] 分到 (列 1, 欄 1)，中心在 (40,20) 的框分到 (列 1, 欄 2)。
   - 7.4：跑了 lesson_cases/07-training.py，確實做了 3 次參數更新；miniyolo.train 的 --help 確實有 --steps、--samples、--device。
   - 大綱 7.3「整批沒有正格」：grid_loss 在這種情況下 box 與 classification 都是 0。
   - 大綱 8.1：跑了 08-own-images.py，非正方形圖 → CHW → letterbox → 推論 → 框還原回原圖，全部成立。
   - 環境檢查 notebook 確實只印 Python、PyTorch、GPU 與 git 資訊。
   - lesson_cases 裡沒有用到 cuda，也沒有下載；08-own-images 與 18-video 讀的檔案都是程式自己先寫出的。所以「資料由程式自己產生，不必下載資料集」成立。

3. 遺留項目與額外指示：
   - outline 第 154 行的遺留項目已修好，寫法和 feedback.md 第 26 行一致。
   - 受程式改動影響的段落裡確實沒有這三頁的項目；拿程式修改清單的程式修正逐一對照三頁內容，沒有任何一句因此變成錯的。
   - 首頁「還沒確認」已改成現在式的「沒有驗證」。它是 status.md〈沒有驗證的事〉的子集，沒有互相矛盾。
   - 改標題前確認過：docs、README、notebooks、scripts、tests、overrides 都沒有連到 _3 錨點，改名後錨點仍是 _3。
   - 摺疊區的新標題不再讓人以為是在講網站的製作經過。
   - 閱讀路線 42 節的連結文字和導覽、section-map 完全一致，一句話描述和各課頁 H1、內容相符。只把 18–20 章標成選修，和大綱、feedback.md 一致。

4. 數字與時態：這次沒有加入任何 Mac 量到的訓練或計時數字。編者列出的依賴紀錄的值，對應的紀錄都在，內容也相符，例如 custom-data-160-step 的 train mAP50 是 0.0068、validation 是 0，gpu-smoke 的 status 是 passed。三頁沒有修訂經過的敘述，也沒有「仍待」「之後會」這類計畫語氣，和審查用的事實與寫作規範清單、status.md、README、課頁都一致。

大綱裡沒被改到的各章敘述，我也對照了課頁和程式：完成條件、表格各列、[8,12,24,28] 沿用到 9.1／11.3／11.4、模型沒有用 Dropout／BatchNorm、3.3 兩個模型從同一份初始權重開始、第 2 與 17 章寫明可以停在哪裡。全部成立。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：通過

讀了紀錄：第 2 次複查寫「前一輪 3 個建議都已採納並確實解決」，第 3 次寫「outline 第 154 行的遺留項目已修好」，每項都有交代。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/planning-outline.md | 「建議事項在下方〈定稿修正〉逐項處理」指向不存在的段落；第 3 次查核的內容是首頁單元的檢查（「log 在同層的暫存副本」）。 | 已修正：沒有〈定稿修正〉時不再寫指向句；修正後的檢查只掛在這一批真的有改動的頁面。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次 3 項建議的處理已補上清單；第 2 次 1 項，由第 3 次確認（outline 第 154 行已修好）；不再寫指向不存在〈定稿修正〉的句子；第 3 次查核註明是和首頁、閱讀路線同一輪。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/planning-outline.md 第 76、86–87、90、105、119 行 | 〈上述處理〉第 1 項寫「內部用語換成白話」，但它點名的「log 在同層的暫存副本」（第 105 行）仍在；另有「與 facts 一致」（第 76、90 行）、「日誌在暫存副本。」（第 86–87 行）、「拿查核範圍清單的程式修正逐一對照」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 76、90 行已改成「審查用的事實清單」，第 105、119 行已改；但第 86–87 行沒改，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/planning-outline.md 第 86–87 行 | 點名的「日誌在暫存副本。」兩處逐字還在，處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 2 | 建議 | reviews/planning-outline.md 第 3 輪第 1 項處理欄 | 「「審查用的事實清單」改成「審查用的事實清單」」是同語反覆。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

第 4 輪第 1 項屬實：第 86、87 行的「日誌在暫存副本。」仍在。第 4 輪第 2 項不實；第 3 輪第 1 項的計數偏高。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪第 2 項處理欄（第 156 行） | 點名的同語反覆「「審查用的事實清單」改成「審查用的事實清單」」在第 3 輪第 1 項處理欄，該格已經重寫，處理欄卻寫「未改」；這個字串只出現在正文第 76、90 行（改好後的用語）。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
| 2 | 建議 | 第 3 輪第 1 項處理欄（第 147 行） | 「5 處中 3 處已改寫或刪除」把第 2 輪的處理說法「內部用語換成白話」也算成已改寫，但它仍在第 139 行。實際改寫的紀錄文字是 2 處。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
