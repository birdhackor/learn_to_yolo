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

## 紀錄重產後的檢查

2026-10-05，由另一位 AI 在自己的隔離副本 `/tmp/lessons-v0.4.0-reviews/review-08/` 查核。副本由發布審查快照 `/tmp/lessons-v0.4.0-reviews/base/` 複製，未含 `.git`、`site` 或虛擬環境；執行 Python 使用既有 `/workspace/learn_to_yolo/.venv-model/bin/python`，但 cwd 與 PYTHONPATH 都指向本人的副本。沒有修改 `/workspace/learn_to_yolo/` 的檔案、審查紀錄或涵蓋清單，也沒有執行 Git、使用認證或啟動 GPU。

結論：**通過；沒有必要修正，也沒有另外留下的建議修正。** 逐句讀完 `docs/lessons/08-own-data.md`，並讀完既有 `reviews/08-own-data.md`；正文、程式摘錄、兩個練習、引用的主例與補充 JSON 紀錄、顯示的 SVG、網站轉換與初學讀者的說明順序，均符合目前內容。先前紀錄的舊數字和「圖待重產」屬歷史查核狀態；本次以下列已重產紀錄與圖為準。

### 實際執行與比對

隔離複製使用指定的工具與排除項：

```bash
LD_LIBRARY_PATH=/tmp/lessons-v0.4.0-tools/extracted/usr/lib/x86_64-linux-gnu /tmp/lessons-v0.4.0-tools/extracted/usr/bin/rsync -a --exclude='.git' --exclude='site' --exclude='.venv*' /tmp/lessons-v0.4.0-reviews/base/ /tmp/lessons-v0.4.0-reviews/review-08/
```

以下三條命令均在上述副本根目錄實際執行，全部 exit 0；不是只列出指令或重畫舊資料。環境為 Python 3.12.14、PyTorch 2.9.1+cpu、NumPy 2.3.5、Pillow 12.0.0；訓練腳本固定 CPU 2 threads。

```bash
PYTHONPATH=/tmp/lessons-v0.4.0-reviews/review-08 CUDA_VISIBLE_DEVICES= /workspace/learn_to_yolo/.venv-model/bin/python lesson_cases/08-own-data.py
PYTHONPATH=/tmp/lessons-v0.4.0-reviews/review-08 CUDA_VISIBLE_DEVICES= /workspace/learn_to_yolo/.venv-model/bin/python scripts/run_custom_data_learning.py --fixture --steps 160
PYTHONPATH=/tmp/lessons-v0.4.0-reviews/review-08 CUDA_VISIBLE_DEVICES= /workspace/learn_to_yolo/.venv-model/bin/python scripts/run_custom_data_learning.py --fixture --steps 1600 --fixture-test-seed 7001 --output artifacts/runs/custom-data-learning-1600 --prior-diagnostic artifacts/runs/custom-data-learning/report.json
```

- 主例的七種拒絕訊息、train／validation／test 各 2 筆、letterbox 框 `[10.666666984558105,15.375,32.0,26.125]`、head `(2,4,4,8)` 與一步 loss `1.4407`，和目前主例 JSON／頁尾輸出相符。另核對 notebook 最後一格逐字等於主例程式，保存的 stdout 等於 JSON stdout。
- 練習 1：在 `artifacts/runs/review-exercises/exercise1.py` 把 source leakage 的 `'test'` 改成 `'validation'`，實跑 exit 0，仍印 `rejected source leakage : source leakage`。
- 練習 2：在隔離副本產生 `exercise2_expected_failure.py`，只增列 `'green_rectangle'`，實跑因原 `(2,4,4,8)` assert 失敗（exit 1，這是預期失敗）；`exercise2_fixed.py` 同時改成 `(2,4,4,9)`，實跑 exit 0，head `(2,4,4,9)`，loss `1.7942`。沒有新增綠色標註，所以這只能證明新 head／target 接起來，不能證明已學會綠色；參考答案有說清楚。
- 兩份實跑 report 的 initial／final full-train loss、全部 loss_history、梯度範圍、參數差、三個 split 的 AP／precision／recall／TP／FP／FN、training_box_diagnosis、reload_checks，與對應新紀錄逐值相同。四張 validation 範例的標註、框、類別、分數、配對與嵌入 PNG 指紋也逐值相同。
- `--prior-diagnostic` 的實跑結果確認：重建 seed 7／700／7000 的標註與 RGB 指紋對得上 160 步 report；train／validation PNG 和標註相同，test seed 7001 是新圖；模型、Adam、learning rate 及 0.1／0.5／0.5 門檻沒有變；前 160 步所有 loss 分項逐值相同，first_differing_step 為 None。程式是在這些資料與設定核對通過後才建立模型、開始訓練；test 僅在最後一次評估迴圈使用，沒有用分數選 checkpoint。
- 兩份實跑 `learning.svg` 分別和 `08-custom-160-step.svg`、`08-custom-learning.svg` 逐 byte 相同。160 步圖沒有診斷紅線；1600 步圖有第 160 步紅線與「前 160 步的 loss 與該紀錄逐值相同」。前者沒有顯示在本頁，仍順便核對其來源；本頁顯示／連結的 SVG 只有 `08-custom-learning.svg`。
- 重新載入的 raw／decoded predictions、Adam state、RNG state 均精確相同；raw 最大誤差 0。checkpoint 格式 2、scheduler None 與 Python／NumPy／PyTorch RNG 的說明對得上 `miniyolo/checkpoint.py`。原圖推論的程式內呼叫與獨立 `scripts/detect_image.py` 程序均由實跑確實執行，JSON 的非路徑欄位精確相同，路徑指向同一檔案；letterbox 還原的框用 1e-4、分數用 1e-6 容差核對，均通過。

### 新數字、手算與 SVG

160 步 full-train loss 為 `1.5501668453216553 → 0.16977311670780182`。第 144–153 步有 8/10 步 total loss 落在約 0.01–0.04；第 150、153 步為 0.10549、0.05455，所以正文的「大多」正確。第 158 步 total `1.865554690361023`、classification `1.776458978652954`；box 約 0.02 出現在 157／159／160 步（0.0178213／0.0195062／0.0224943），而 158 步 box 是 0.00607728。160 步正格最小預測高度 `0.0545400679` pixel，正文「約 0.05」正確。全部 24 張 train PNG 的前景與 JSON 框相同，21 個物件／21 正格、非零寬高 target、負格 box 梯度和為 0，四項診斷全部通過。

1600 步 full-train loss 為 `1.5501668453216553 → 0.00011704797361744568`；梯度 L2 範圍 `0.024197157472372055–32.54474639892578`，每步有限且非零；參數變化 L2 `17.216838217570434`；15,544 個參數。正文四捨五入與新紀錄吻合。曲線第一步是 8 張 minibatch 的 loss `1.5648255348205566`，不等於 24 張一起算的初始 loss，正文也已區分。

以獨立 Python 手算（未呼叫 repo 的 `evaluate_ap`、`match_image` 或 `box_iou`）：從實跑 `predictions.json` 與 fixture JSON 取得所有框，按照原圖 W／H 手算整數 letterbox 縮放與 padding，IoU 用交集面積／聯集面積，每類跨圖依分數降冪、與同圖未配對的同類 GT 比較；用 `fractions.Fraction` 計算 precision envelope 與各次 recall 增量。1600 步手算結果：

| split | 紅／藍／黃 AP50 | mAP50 | TP／FP／FN | precision／recall |
|---|---|---|---|---|
| train | 1／1／1 | 1 | 21／0／0 | 1／1 |
| validation | 5/9／0／1/3 | 8/27 = 0.2962962963 | 3／8／6 | 3/11／1/3 |
| test | 1／2/3／2/3 | 7/9 = 0.7777777778 | 7／2／2 | 7/9／7/9 |

validation 各類的 TP／FP 排序是紅 `T,F,T`、藍 `F,F`、黃 `T,F,F,F,F,F`；紅 AP＝(1/3)×1＋(1/3)×(2/3)＝5/9，黃 AP＝1/3，所以 mAP＝(5/9＋0＋1/3)/3＝8/27。test 為紅 `T,T,T`、藍 `T,T`、黃 `T,T,F,F`。160 步也重算：train 藍 `F,F,F,F,F,F,T`，AP＝1/49，三類平均 1/147＝0.0068027211，TP／FP／FN＝1／20／20；validation AP 0、0／10／9；test 藍 `F,T,F`，AP＝1/6，三類平均 1/18，1／8／8。都與紀錄相同。

SVG 範例逐框用原始 float64 算 IoU，得到：

| validation 圖 | 預測分數 | 判定與 IoU |
|---|---|---|
| #1 紅圖、紅預測 | 0.9999966621 | TP，IoU 0.5845538703 |
| #1 紅圖、黃預測 | 0.1581750810 | FP，沒有同類 GT；和紅 GT 最大 IoU 0.2411930556 |
| #2 藍圖、藍預測 | 0.4621165991 | FP，IoU 0.4058014689；藍 GT 為 FN |
| #3 黃圖、黃預測 | 0.9946288466 | FP，IoU 0.4206645747；黃 GT 為 FN |
| #0 空圖 | 無預測 | 無 GT／無預測 |

用 XML 核對：所有 7 個 GT／預測 rect 的 x／y／寬／高都是紀錄框按 224/64 縮放後的值，綠虛線／橘實線的顏色與分組正確；四個 SVG 嵌入 PNG 的 SHA-256 等於紀錄的 letterbox_png_sha256；藍線全部 1600 個點與 loss_history、縱軸 0–2、橫軸 1–1600 一致。caption_lines 的 TP、低 IoU FP、沒有同類 GT、類別錯、GT 已用完、空圖與超過四行的分支均對照正文查核；這次紅圖確實新增黃 FP，所以「紅矩形圖中額外的黃色框」也成立。

以 Playwright＋`/usr/bin/chromium` 實際渲染 1120×840 SVG，並用 view_image 看截圖 `/tmp/lessons-v0.4.0-reviews/review-08-figure.png`；四個 panel、框、文字與紅線可辨認，沒有遮擋或截斷。紅圖最長說明寬 258.34375，小於兩個 panel 文字起點間距 274，下一張圖的 FN 文字沒有重疊。Chrome 的 file:// 導覽被環境政策擋下，改把原 SVG 原文置入 HTML 渲染，未改 SVG；這不影響框或字型排版的查核。

### 程式、來源與閱讀順序

JSON 的 class 順序／索引、有限數字／正整數／半開區間 xyxy、bool 排除、空框 `[0,4]`、先檢查全部紀錄再挑 split、圖片實際尺寸、來源與 path 的拒絕條件、setdefault 的尺寸 key、RGB 完全重複檢查、HWC→CHW、DataLoader／collate、同步 letterbox、head 的 5+C 與 1×1 權重 shape、target 的格中心、同格衝突在訓練前停止，逐句對照主例與 repo 模組，沒有發現誤述。120×80 的 `[20,10,60,30]` 手算結果為 `[10.6666667,15.375,32,26.125]`；半開右下角加 1 的例子與程式 `[10:30,20:60]` 相符。YOLO normalized cxcywh→pixel xyxy 的四個公式正確。

AP 是本書的單 IoU all-points 定義，三類都有 GT；提高候選截斷門檻移除排序末端候選，不會增加 precision envelope 面積；因 NMS 也按分數從高到低，刪除低分尾端不會改高分候選的抑制。正文有分開候選門檻 0.1 與畫框門檻 0.25，也說清楚三類／小樣本／合成圖、test seed 7000 已看過所以換 7001、真實 test 無法任意重生、checkpoint reload 不等於已驗證續訓。沒有把單次的 validation 與 test 差異說成穩定排名。

對初學讀者逐句檢查：JSON、索引、來源、split、batch／target、head／logit、TP／FP／FN、RNG、scheduler 的說明在使用處或前置課中有交代；自主練習指出要改的 assert，自己的 JSON 指令說清 root 與 annotations 路徑、覆蓋輸出及門檻差異；兩個完整實驗與 Colab 主例沒有混用。未發現需要本次發布前處理的閱讀障礙。

查閱的官方來源（固定版本；本頁沒有引用某篇原始論文的主張）：

- CPython **v3.12.14** `Doc/library/stdtypes.rst`：[bool 是 int 子類別、dict.setdefault](https://github.com/python/cpython/blob/v3.12.14/Doc/library/stdtypes.rst)。原始來源分別在第 837、4622–4626 行；與 runtime 的版本相同。
- PyTorch **v2.9.1**：[DataLoader](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/data/dataloader.py) 的 batch_size、shuffle 與 collate_fn 官方說明（148–164 行）；[Adam](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/adam.py) 的 state step／exp_avg／exp_avg_sq（167–183 行）及更新公式。確認正文說的 Adam 移動平均與保存狀態。
- Pillow **12.0.0**：[PIL.Image](https://github.com/python-pillow/Pillow/blob/12.0.0/src/PIL/Image.py) 的 convert 與 verify：convert 回傳轉換副本；verify 檢查圖檔而不解碼，Dataset 讀取時重新開檔，與程式一致。
- Ultralytics YOLOv5 **v7.0**：[utils/dataloaders.py](https://github.com/ultralytics/yolov5/blob/v7.0/utils/dataloaders.py)（676–677、1019–1021 行），normalized xywh、五欄與範圍檢查，確認正文轉換格式的前提。半開區間是本書的定義，沒有宣稱所有標註工具都使用它。

官方 PyTorch／Pillow 文件站此次回 HTTP 403；改查以上與安裝版本相同的官方 tag 原始程式與 docstring，來源可讀且內容吻合。YOLOv5 v7.0 不含先嘗試的 `docs/tutorials/train_custom_data.md`（404），改讀固定 tag 的實際 loader。這些初次取來源的失敗沒有被當成通過證據；成功來源已存在隔離副本的 `artifacts/runs/review-sources/`。

### 網頁與範圍限制

在隔離副本實際執行 `/workspace/learn_to_yolo/.venv-docs/bin/zensical build --clean --strict`（Zensical 0.0.67），exit 0、No issues found；再執行 `/workspace/learn_to_yolo/.venv-docs/bin/python scripts/validate_site.py`，exit 0，Markdown 表格／編號清單／摺疊區／公式／連結及 artifact 邊界檢查全部通過。另以 `scripts/validate_lessons.py` 的原 excerpt_problems 函式檢查本頁，回傳 `[]`；沒有替副本或 root 寫入 review coverage。

本頁的原紀錄訓練時間 `10.877128770982381` 秒、完整流程 `14.00242607301334` 秒，正文 10.877／14.002 秒正確。對照 perf_counter 的開始、結束點與 JSON scope：訓練時間只有迴圈；完整流程還含 fixture／prior 重建、檢查、評估、存載、兩種原圖推論（包括另一 Python 的啟動和 import）、輸出與 SVG；不含本程序啟動／import、安裝或最後 report 寫出。本人重跑 160 步是迴圈 1.2086235810 秒／完整流程 4.2814818970 秒；1600 步是 11.5001596420 秒／15.6183919680 秒。重跑耗時沒有被拿來替換作者紀錄，也沒有用作效能測試。

本次沒有 GPU 執行、真實照片測試、真人學生測試或 Colab 託管 runtime 測試。正文引用的 GPU 續訓主張僅對照既有 `artifacts/checks/gpu-smoke.json`：L4、共 40 步／midpoint 20、另一個 container 的 resume、model／optimizer／history 最大誤差皆為 0、loss_and_rng_trace_matches True；這是查核已有紀錄，不是本次重新驗證 GPU。程式碼中的評估次數、固定設定與資料指紋可查核，無法獨立證明歷史上人的全部選擇過程；紀錄與正文沒有超出這個證據範圍。

### 涵蓋內容 SHA-256

下表正文 hash 使用 `scripts/review_coverage.py::digest` 的規則：Colab tag 正規化為 `<tag>`，排除頁尾自動執行紀錄。另保留未正規化、仍排除頁尾的正文 hash：`5f567e98647adc12807e512227161114bc4713c690a785b3452b2894e82d4038`。圖、程式與 JSON 是完整檔案 SHA-256。

| 檔案／內容 | SHA-256 |
|---|---|
| `docs/lessons/08-own-data.md` | `1c444c9bae61d4f41022a1ce1ea9292066f3878bfd88d9c597c4a5bb6471d245` |
| `docs/assets/diagrams/08-custom-learning.svg` | `83036ad187af44e64bf54200e47b47b414dfab96ef03b4e5db79321d187f09a7` |
| `lesson_cases/08-own-data.py` | `d3be8a09b62a77c542ba48a93e10c36f5a8a2f34e587b2e207c4d7e41c8f300a` |
| `miniyolo/__init__.py` | `785b058b2b011124243b06ee59df8e59dfa75b77f050b897479e5be636763fc9` |
| `miniyolo/custom_data.py` | `7dd9999f03dbc32ce1ec951c58121c8f746d381b0244e4c2a82cca59e7ca7e55` |
| `miniyolo/data.py` | `cccad00e2c4f96eb6567eafc9e12248379c6b715fc1790d75518a253baa6181d` |
| `miniyolo/geometry.py` | `6a6b57d3493888e99dae4a012dab78127b8b543a0e01d8107963a3d6e63d8483` |
| `miniyolo/inference.py` | `995ac8f942d0c1e43d94f2efb3b7adcb91a9feb6b9bab923615209f0c190c313` |
| `miniyolo/losses.py` | `81fa9331c9a2aebe2e9c6c453a5313e64566bb20c3fe77ca45ec9df53424d4e4` |
| `miniyolo/metrics.py` | `53ce982e37cbd96c784f75c7d30faf99d52f79ab83ca7b8114eb21b4327330e0` |
| `miniyolo/models.py` | `49d029ea4ba2650ce8933cf97e3d25dc7aff2ca4e972eec19cdda17b0f4900e6` |
| `miniyolo/targets.py` | `2c8e32f2845b3bf970c77304c5cca0f999083d36a4d5af2f75f14aa89b823f16` |
| `scripts/run_custom_data_learning.py` | `b8b3b442e4e2349173a2eb8198b33f966d1e75d80c8b1edb07d1af4d3b5e99e5` |
| `scripts/detect_image.py` | `69738c487cfad24bea2009384712ca0ba958ac1fd394e2226ae33d7de9682c6a` |
| `scripts/record_evidence.py` | `8fd91178fb6927861ffa1dd1ee20ab80f0438ad151ef1f09da02a2799399c035` |
| `miniyolo/checkpoint.py` | `2144e2afa4d89382a7b3cd253755bb851e35fdc0847ee7179dee02dbbf4ca49b` |
| `miniyolo/train.py` | `aa5567f11be6ca7aee97bd082b9d5964414b12b73453763c915c7ffb409c4276` |
| `miniyolo/provenance.py` | `21769813c36b45f513c9f0d6125194f3baa5ae265278ffd44a796d298d124ed3` |
| `miniyolo/figures.py` | `155222c5bb0d6942bcc231d5c32c6f1a4db662293c8cf909560f1f9b75a3ea15` |
| `artifacts/checks/curriculum/custom-data-160-step.json` | `d1b5582921555b93338f26088ceb2d4e3a84b8a4b04266965df86a4ea37b0f96` |
| `artifacts/checks/curriculum/custom-data-learning.json` | `fb22ce8356fbd747f967c3b82df53bd55de840d826d44fc73abfdba315f8e584` |
| `artifacts/checks/curriculum/08-own-data.json` | `861a54eeaae0573b1ecbef7b5cbcd92157bab333b6fb5d85f6b6ba3a10e3b6ff` |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[grid當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/grid.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/grid.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/grid.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `detector` 實讀本頁 6 個凍結單元；首次使用／前文方法範圍四題位置為 08-own-data/00:scope_overview, 08-own-data/01:first_use, 08-own-data/02:first_use, 08-own-data/03:first_use, 08-own-data/04:first_use，頁末為 08-own-data/05。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/detector.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/detector-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/detector.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R027；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-applications.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-06：v0.6.1 有界修正複查

RNG state 段落補英文全名與中文意義；名稱檢查只讀提供的局部語境，非整頁首次閱讀。

本輪方法、逐批閱讀原始紀錄、非作者技術核對、另一位讀者銜接與實際 Zensical 修改段落截圖見 [v0.6.1 局部複查](clear-tutorial/release-v0.6.1-2026-10-06/README.md)。本次只重新檢查修改處、必要上下文與版本一致性；其餘正文、程式及圖的既有審閱保留原範圍，不改標為整頁或全書新的首次盲讀驗收。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 b 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁相關處置：DEC-028、DEC-029；包含採用、保留或後文撤回的來源與理解收益。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `b_grid` 實讀本頁 13 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/b_grid/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：無額外局部修正。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/08-own-data.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。


## 2026-10-08：章節字母前綴統一

本次僅將H1節號補上所屬B／C／D／E，並同步側邊目錄與閱讀路線。正文（第一行之後）、章號、路徑、圖、程式與實驗設定未變；沒有重新冒稱概念首讀。來源逐頁比對與標題／導航實頁檢查另存於[格式核對](formatting/section-prefixes-2026-10-08/source-check.json)。
