# 審查紀錄：自己的類別與資料

審查範圍：`docs/lessons/08-own-data.md`、頁面上的圖（`docs/assets/diagrams/08-custom-learning.svg`），以及 `lesson_cases/08-own-data.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題，另有 5 項建議。頁面對現行程式的敘述都正確，數字沒有用到 Mac 的值。

**實跑查核**
- lesson 程式：exit 0，stderr 是空的，10 行 stdout 和頁尾執行紀錄逐字相同。
- 練習 1：把 `'test'` 改成 `'validation'`，照樣印出 `rejected source leakage : source leakage`，和參考答案一致。
- 練習 2：加上 `green_rectangle` 後，只有 `assert prediction.shape == (2,4,4,8)` 會失敗。改成 `(2,4,4,9)` 就通過，head 印出 `(2, 4, 4, 9)`。
- 新加的 assert：讓 video_C 與 video_A 完全相同，或只讓兩張空圖相同，都以 `same image in two splits` 中止。
- 摘錄檢查：摘錄比對工具對主 repo 的頁面印出 `[]`。另外兩段 Python 是讀者自己寫的片段，沒有被判成照抄。prose 裡沒有程式行號。
- 網站建置：`zensical build --clean --strict` 與 `validate_site.py` 都通過。摺疊區、表格與摘錄區塊都正確轉出。

**照頁面指令實跑**
- 摺疊區兩條指令依序跑（160 步，再接帶 `--prior-diagnostic` 的 1600 步），都 exit 0。
  - 只有第二次的 learning.svg 有第 160 步紅色虛線，圖例寫「前 160 步的 loss 與該紀錄逐值相同」。
  - 160 步那張沒有紅線。
- 用自建的 my-data 走完「換成自己的資料」兩行指令，都不設 PYTHONPATH，都 exit 0。
  - 第二行寫出同名的 PNG 與 JSON。
  - 預設門檻 0.25；加 `--score-threshold 0.1` 也能跑。

**逐句對照程式**
- validate_annotations：確實不開圖。
- setdefault 的說明，以及 key 為什麼要放尺寸：都正確。
- `*_map50` 鍵名，與 report.json 的 `map`、`ap50_per_class_name`：都正確。
- training_box_diagnosis 的四項檢查：其中 PNG 塗色範圍只對合成資料做，失敗就 RuntimeError。
- `--prior-diagnostic` 的核對：先用 seed 7、700、7000 重建資料，比對 manifest 與解碼 RGB 的 SHA-256，再逐筆比對 train／validation、確認新 test 沒有舊圖、確認設定相同。任一不符就在訓練前停止。
- record_evidence.py 的兩步指令、Colab「可選」段落的指令：和頁面相符。
- caption_lines 各分支與截斷規則：頁面文字和程式逐字相同。
- 配對規則：match_image 與 evaluate_ap 都是「同類、尚未配走的 GT 中 IoU 最大者，IoU ≥ 0.5 才算 TP」。
- check_original_image：程式內呼叫與另開程序各跑一次，門檻 0.1／0.5，比對 JSON。
- 兩個計時範圍、`format_version` 2：都和程式一致。
- 清單的 4 條先前審查意見與受程式改動影響的段落的 29 項都有處理。頁面沒有修訂經過的敘述。

**數字**
- 頁面數字都和現有紀錄相符（錄製機產生）：1.55017、0.16921、0.00680、0.000354692、0.388889、0.666667、0.020489–32.557129、16.519738、7.517、8.273、15,544、0.0548 px。
- 四張圖的 IoU 0.5598／0.5064／0.2999 與分數 1.00／0.46／1.00 吻合；validation 全體 TP5／FP5／FN4。
- 頁面上找不到任何 Mac 值（0.519、0.18664、1.91 等）。
- 編者回報在 7000 字處截斷，待重錄數值清單只看得到前段，見 should 4。

**要知道的暫時狀態**（依審查用的事實與寫作規範清單不算問題）
- repo 裡的 docs/assets/diagrams/08-custom-learning.svg 還是舊的英文圖（`GT: 1 | decoded: 1`、沒有紅線）。頁面描述的是 record_evidence.py 重產後的圖。
- 用 qlmanage 看過舊圖與 Mac 產生的新版圖：新版圖的元素和頁面描述一致。
- 重產兩份 custom-data 紀錄與這張 SVG 之後，要依新紀錄核對頁面上所有 record-dependent 的敘述。8.273 秒依新範圍會變大。

這次編輯沒有改任何 SVG。本頁以外的工作樹改動屬於其他部分，無法逐一歸屬；也沒有發現編者在 repo 裡留下的檔案。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/08-own-data.md 第 195 行「1600 步實驗的前 160 步和這次用同樣的 seed、資料與設定，loss 逐值相同」 | 兩次實驗的 test seed 不同（160 步用 7000，1600 步用 7001），test 圖也不同。讓前 160 步 loss 逐值相同的，是模型 seed、train 資料與設定。讀者對照第 185 行（seed 7、700、7001）和第 199 行（改用新 test seed 7001），會覺得「同樣的 seed、資料」和前後文對不上。 |
| 2 | 建議 | docs/lessons/08-own-data.md 第 195 行「第 158 步約 1.86，box loss 也升到約 0.02」 | 依現有 custom-data-160-step.json 的 loss_history，第 158 步的 box loss 只有約 0.006。約 0.02 出現在第 157 步（0.0178）與第 159–160 步（0.0195、0.0225）。照現在的句子，讀者會以為第 158 步的 box loss 就是 0.02。這句不是這次改的，但在受程式改動影響的段落第 185 行要求「用重產紀錄重新核對」的範圍內。 |
| 3 | 建議 | docs/lessons/08-own-data.md 第 212 行（摺疊區「自己跑 160 步與接續的 1600 步」）「前面那行沒有 `--prior-diagnostic` 的指令」 | 摺疊區才剛講完「第一行」和「第二行」，讀者容易把「前面那行」讀成摺疊區的第一行（`--steps 160`），以為 Colab「可選」段落跑的是 160 步。另外，摺疊區第一行和本節開頭那行一樣，預設寫到 `artifacts/runs/custom-data-learning/`，會覆蓋讀者先前跑出的 1600 步 checkpoint、report.json 與 learning.svg，頁面沒有提醒。 |
| 4 | 建議 | docs/lessons/08-own-data.md 摺疊區「圖下說明的其他寫法」第一句「這次的四張圖只用到上面幾種」 | 這句只在錄製機的紀錄下成立。我在 Mac 照摺疊區指令實跑，藍圖那格就出現「黃 0.17：FP（類別錯，IoU 0.503）」。編者回報在 7000 字處被截斷，可見部分的待重錄數值清單沒有這句，無法確認它有被列入。 |
| 5 | 建議 | docs/lessons/08-own-data.md 第 252 行「兩份 JSON 除了路徑的寫法也要逐值相同」 | 讀者要到第 289 行才知道 detect_image.py 會另外寫出一份同名 JSON。讀到這裡時，「兩份 JSON」從哪裡來沒有交代，初學讀者會卡一下。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 「同樣的 seed、資料與設定」和兩次 test seed 不同（7000／7001）互相矛盾 | 已修正：改成「同樣的模型 seed、train 資料與訓練設定（test 圖不同，但 test 不參與訓練）」。 |
| 2 | 先前查核 | 「第 158 步約 1.86，box loss 也升到約 0.02」的步數不對 | 已修正：查 custom-data-160-step.json 的 loss_history：第 158 步的 box loss 是 0.0061；約 0.02 出現在第 157 步（0.0178）與第 159–160 步（0.0195、0.0225）。句子改成「box loss 在第 157 步與第 159–160 步也升到約 0.02」，並列入待重錄數值清單。 |
| 3 | 先前查核 | 摺疊區「前面那行」指稱不明，也沒提醒預設輸出資料夾會被覆蓋 | 已修正：改成直接寫出「本節開頭那行 `--fixture --steps 1600 --fixture-test-seed 7001`」。已確認腳本的預設 --output 相同，且用 mkdir(exist_ok=True)，所以補一句：第一行會寫到同一個資料夾，覆蓋先前的 checkpoint.pt、report.json 與 learning.svg。 |
| 4 | 先前查核 | 「這次的四張圖只用到上面幾種」只在錄製機的紀錄下成立 | 未改：這句的對象是紀錄那次的執行，用語符合寫作規則。內容取決於紀錄，所以不改文字，列入待重錄數值清單，重產後再核對。 |
| 5 | 先前查核 | 「兩份 JSON」出現時，沒交代它們從哪裡來 | 已修正：補半句：「`detect_image.py` 每次推論都會另存一份記錄框、類別與分數的 JSON，這兩次的兩份 JSON…」。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/06-decode-nms.md`、`docs/lessons/06-evaluation.md`、`docs/lessons/07-heldout.md`、`docs/lessons/08-own-images.md`、`docs/lessons/08-own-data.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/08-own-data.md 第 212 行（摺疊區〈自己跑 160 步與接續的 1600 步〉）「本節開頭那行 `--fixture --steps 1600 --fixture-test-seed 7001`」 | 這一頁的「本節」都指整個 8.2。例如第 281 行「照本節開頭的格式」，指的就是頁首〈固定一種可以查核的格式〉。可是這行指令在第 170 行，是〈完整實驗：訓練、評估、存檔與重新載入〉這一段的開頭，不在頁首。讀者照字面捲到頁首，會找不到這行。後面寫出的指令參數只對得上第 170 行那一行，所以不會讓讀者做錯事，只是多找一下。這個位置描述是這次修改新帶進來的。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

對照 artifacts/checks/gpu-smoke.json：共 40 步、第 20 步存 midpoint；續訓在另一個 container 讀 HF 下載的 checkpoint；model_max_abs_error、optimizer_max_abs_error、history_max_abs_error 都是 0，最終 loss 相同。gpu_smoke.py 與 run_custom_data_learning.py 都用 miniyolo.checkpoint 的 save/load，所以「確認了這件事」站得住，「逐值相同」也正確。G：〈完整實驗：…〉的第一個指令確實是 `--fixture --steps 1600 --fixture-test-seed 7001`，notebook 的「可選」段落用同一行；紅色虛線只在有 `--prior-diagnostic`（followup）時畫；預設輸出目錄 artifacts/runs/custom-data-learning/ 會覆寫 checkpoint.pt、report.json、learning.svg。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核 5 項建議都在〈定稿修正〉處理（4 項已修正、1 項未改並附理由）；批次檢查留下的 1 項，由〈後續編輯的檢查〉核對；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/08-own-data.md 第 13、50 行 | 殘句：「**實跑查核**（都在暫存副本，紀錄在 ../暫存副本）」「用 qlmanage 看過舊圖與 Mac 產生的新版圖（../暫存副本）」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 13 行與第 50 行點名的殘句都已清理，處理說明屬實。
