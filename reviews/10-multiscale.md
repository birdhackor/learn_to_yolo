# 審查紀錄：YOLOv3 多尺度

審查範圍：`docs/lessons/10-multiscale.md`、頁面上的圖（`docs/assets/diagrams/10-multiscale-learning.svg`、`docs/assets/diagrams/10-multiscale.svg`），以及 `lesson_cases/10-multiscale.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題，只有一個 should：頁面沒寫明兩個摘錄裡的中文註解是本頁加的。

所有程式都在暫存副本裡執行，沒有在 repo 內執行任何程式。

**1. 程式敘述都成立**
- lesson_cases/10-multiscale.py：exit 0，stderr 是空的，stdout 和 artifacts/checks/curriculum/10-multiscale.json 存的那份逐字相同。
- 兩題自主練習照頁面的題目，用 build_targets 實算：stride 8 落在 (gx=3,gy=2)、格內 xy (.5,.5)、wh .25；stride 16 落在 (gx=1,gy=1)、格內 xy (.75,.25)。和參考答案一致。
- 跨尺度去重：兩個人工 logits 解碼後的框逐位元相同，IoU 1.0，label 只有 0，NMS 後 2→1。
- 照「想自己重跑」那段跑 run_multiscale_learning.py（不加 --record）：
  - 只寫出 artifacts/runs/multiscale-learning/ 下的 report.json 與 learning.svg，網站用的紀錄和圖都沒變。
  - 印出的 JSON 不含 loss_history。
  - notebook 的〈可選：兩尺度 40 步與實際預測〉已經是同樣的兩步和 artifacts/runs 路徑。
  - 環境格有 os.chdir 到 repo，所以相對路徑在 Colab 可用。
  - record_evidence.py 發布時會用 --record 重產網站圖。
- 頁面上其他和程式有關的敘述都對照過 miniyolo 與相關章節的程式，全部成立：
  - shape 表、80 個候選、logits 元素 112／560、fine_head 119 個參數、全部 8702 個參數、感受野 15／31。
  - decode_grid 先用 score 門檻篩選、再做尺度內分類別 NMS。
  - evaluate_ap 的配對 IoU 門檻 0.5；第 9 章 ignore 的 0.2；第 7 章顯示門檻 0.25 的用語；GridDetector 的三項差異；第 11 章 mix 是在 8×8 上的 3×3 卷積。

**2. 先前審查意見與受程式改動影響的段落都處理了**
- 兩個 Colab 連結已經指向 lessons-v0.4.0。
- display 路徑已改成 artifacts/runs/…，並說明了 --record 的用途。
- 圖說照新圖重寫；0.13 那個框補上標籤位置。
- SVG desc 那句已改；git diff 確認這個 SVG 只改了這一句。
- 全頁 grep 過，沒有修訂、審查或製作經過的敘述，也沒有程式行號。

**3. 數字**
- 這次的修改沒有新增任何數字。
- 依紀錄而定的數值都對過 10-multiscale-learning.json，編輯也都列出來了：6.504、3.6594→0.0703、mAP 0.5、AP 0／1、三個框與分數、IoU 0.44／0.86／0.30／0.28、寬差約 3、高差約 5.4 pixel、平方誤差 0.002／0.007。
- 平方誤差的論證成立：小框確定是 fine 格 (1,1) 輸出的，實測平方誤差是 0.0022 和 0.0072。
- 這台 Mac 跑出的數字和紀錄只差在很後面的小數位。

**4. 摘錄**
- 摘錄比對工具對 repo 和暫存副本都印出 []。
- set 比對看不出順序，所以我另外逐行核對：兩段摘錄的順序、縮排和 `...` 省略的範圍都和原檔一致，兩段在 build 後的 HTML 裡都渲染成 Python 程式區塊。

**5. 可讀性**
- 同一個名字 fine／coarse 在兩處意思不同，引言已經說明。
- 新出現的 forward、特徵圖，在第 1 章和術語表都解釋過。
- 唯一的建議就是 problems 裡那一條建議。

**6. 渲染與改動範圍**
- `zensical build --clean --strict` 的 exit 是 0，輸出「No issues found」；validate_site 的 exit 也是 0。
- 用 qlmanage 渲染：10-multiscale.svg 畫面乾淨，viewBox、title、desc 都在，座標換算和格子位置都對得上程式。暫存副本重產的 learning.svg 和新圖說一致（圖例在上方中央、綠虛線是 GT、橙實線是預測框、0.13 的標籤在框左側、圖的底部）。
- repo 裡和本節有關的檔案，只有列出的兩個有改動。
- validate_lessons 失敗，原因是 07-inference、09-anchor-clustering、18-video、20-deployment 這幾頁的摘錄，不是本頁。本頁的 notebook 檢查都通過，審查涵蓋檢查顯示「no review」，這是預期中的（編輯已在疑慮裡寫出）。

**對編輯疑慮的看法**
- I8 那段 MSE 說明：是 HEAD 原有的內容，數字可以從紀錄的框座標算出來，而且正確，不必列為問題，要不要照 I8 縮短交給 owner 決定。
- 第 141 行「留給後續的融合實驗」：11-fusion.md 第 37 行已經明說沒有接上 TwoScale，讀者不會被誤導，不需要改。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/10-multiscale.md 第 67 行（第一個摘錄的引言）與第 104 行（第二個摘錄前一句） | 兩個 data-excerpt 區塊的程式行都和 lesson_cases/10-multiscale.py 逐字相同，但區塊裡的中文註解（例如「# 兩個 head 的輸出，不是 forward 裡的特徵圖」「# 留下的就是小框 [5,5,13,13]」）程式裡沒有。頁面沒告訴讀者這些註解是本頁加的：第 67 行只寫「下面摘出這幾行」，第 104 行連「摘自完整程式」都沒寫。讀者拿去對照 notebook 最後一格時，程式行對得上，註解卻找不到。同一版的其他頁（02-diagnostics、03-identity、04-localization、06-decode-nms、11-augmentation 等）都會寫明「中文註解是本頁加的」，只有本頁沒寫。讀者不會因此做錯事，所以列為建議。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：在暫存副本中工作，沒有在 repo 內執行或寫入任何東西。

閱讀的內容：
- 審查用的事實與寫作規範清單、docs/lessons/10-multiscale.md 全文、docs/glossary.md。
- lesson_cases/10-multiscale.py、scripts/run_multiscale_learning.py。
- miniyolo 的 targets／losses／inference／geometry／models。
- notebooks/10-multiscale.ipynb 與 scripts/build_lesson_notebooks.py 的 optional_experiment。
- scripts/validate_lessons.py：它比對摘錄時會先去掉註解。
- 交叉引用的 01-small-cnn（感受野、卷積公式）、06-decode-nms（class-wise NMS）、07-inference（fixture 填法、顯示門檻 0.25）、09-anchors（尺寸 IoU、ignore 0.2）、11-fusion（用隨機張量、沒有接回 TwoScale），以及 zensical.toml 的導覽標題。

執行與建置：
- 跑 lesson_cases/10-multiscale.py：exit 0，五行輸出和頁尾區塊逐字相同。
- 跑 scripts/run_multiscale_learning.py，加與不加 PYTHONPATH 各一次，都成功，report.json 和 learning.svg 寫在 artifacts/runs/multiscale-learning/。Mac 上的數值只拿來參照，沒有當成正確值。
- 用 zensical build --clean --strict 建站：沒有問題。用 Playwright 快取裡的 headless Chromium 截圖，寬度 1280 與 390 各一次；Playwright 的 Firefox 沒有安裝。
- 截圖前把新畫的 learning.svg 複製進暫存副本的 site，因為 repo 裡那份還是舊版英文圖。
- 用 qlmanage 把 10-multiscale.svg、舊版與新版的 learning.svg 轉成 PNG，並放大格子區檢查斜線怎麼疊。

核對與練習：
- 手算紀錄框的 IoU（0.441、0.863、0.299、0.279），也手算參數數（8702、119）與感受野（1→3→7→15→31）。
- 照頁面指示手算練習 1、2，再用 build_targets 驗證：box [20,12,36,28] 在 grid 8 得到 (gx=3,gy=2)、[.5,.5,.25,.25]；在 grid 4 得到 (1,1)、[.75,.25,.25,.25]。兩題都和參考答案相符。
- 實作〈常見錯誤〉第 3 條，只算 fine loss：deep 與 coarse_head 的 grad 是 None，和頁面說法相符。
- 重跑 40 步並拆開 loss：total＝5×box＋obj＋cls；小框中心 y 偏約 0.9 pixel，平方誤差 0.0126；中心偏移不影響交集。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/10-multiscale.md 第 141 行「本節先保留兩個 head，留給後續的融合實驗；保留的是可以繼續研究的介面」 | 這句寫成計畫（「留給後續」），而且和教材實際內容不符。11.2〈特徵融合〉用 8 與 16 channel 的隨機張量示範，明說沒有接上第 10 章的特徵、沒有訓練，也沒有做品質對照。讀者會以為第 11 章會拿這兩個 head 做融合實驗，讀到 11.2 才發現沒有。 |
| 2 | 建議 | 第 32 行圖說；docs/assets/diagrams/10-multiscale.svg 的格子區 | 圖說寫「淡色格是訓練時的正格、斜線格是負格」，但圖中 fine 格 (1,1) 是淡紅加斜線（斜線屬於包住它的 coarse 格 (0,0)），coarse 格 (2,2) 裡的 fine 格 (5,5) 也是淡藍加斜線。照圖說的規則讀，會看到一塊同時是正格又是負格的區域。「斜線格是負格」也容易讓人以為白格不是負格。「正格以外全是負格」只寫在圖底部的小字和 SVG 的 desc 裡，畫面上的圖說沒寫。 |
| 3 | 建議 | 10-multiscale.svg 上方圖例「stride 16（每格 16 畫素）」「stride 8（每格 8 畫素）」，對照第 9 行「每格對應 16×16=256 個畫素」 | 「每格 16 畫素」可以讀成一格只有 16 個畫素，和正文的 256 個畫素衝突。圖裡寫「畫素」，正文多寫「pixel」，同一頁兩種寫法。 |
| 4 | 建議 | 第 9 行「每格對應 16×16=256 個畫素……要和格內的背景一起被概括成同一組數字」，與第 28 行「coarse 是 31×31」 | 第 9 行讓讀者以為 coarse 特徵只受自己那格的 256 個畫素影響；第 28 行又說它的理論感受野是 31×31。兩句沒有接起來，數學好的讀者會卡在「到底是 16×16 還是 31×31」。15、31 只給了結果，沒有像第 1 章那樣列出逐層的算式。 |
| 5 | 建議 | 第 99 行第一次出現的「最後做一次分類別 NMS」，之後第 124、131、168 行也用到 | 「分類別 NMS」沒有定義，術語表也查不到。6.1 節的寫法是「按預測類別分組，各組分別做 NMS（class-wise NMS）」。「分類別」容易被讀成「分類」加「別」。 |
| 6 | 建議 | 第 104 行「兩份人工 logits 的框值，分別由……兩欄的 target 換算而來」 | 沒說怎麼換算，也沒說 obj 和類別 logits 填了什麼。程式用 torch.logit（sigmoid 的反函數）把 target 換回 logits；正格的 obj 和 class0 填 10、class1 填 −10，其他格的 obj 填 −20。讀者要知道這些，才看得懂為什麼每個尺度只剩 1 個框、兩框的 score 都接近 1。這和 7.5 節的人工 fixture 是同一套填法，但頁面沒有連回去。 |
| 7 | 建議 | 第 67 行，以及第 69–87、106–122 行的兩段程式摘錄 | 摘錄裡的中文註解，lesson_cases/10-multiscale.py（也就是 notebook 最後一格）都沒有；程式裡反而是英文註解，例如 # Disjoint scale assignment…。多數課頁會寫「中文註解是本頁加的」，本頁只寫「下面摘出這幾行（... 表示省略的行）」，讀者打開 notebook 對照時，可能以為自己找錯格。另外在 1280 px 寬的桌機上，第 121 行的尾註會被切成「[5,5,13,1」，要橫向捲動才看得到結尾。 |
| 8 | 建議 | 第 126–131 行〈應核對〉清單，對照頁尾區塊的五行輸出 | 清單順序和輸出順序不同：輸出第 1 行就是 NMS 的 2 -> 1。「表內兩個 target」要對到 `stride16 small target [0.5625, 0.5625, 0.125, 0.125]`，但表裡沒有叫 target 的列，讀者得自己把「格內 xy」和「normalized wh」兩列拼起來。最後一行 `both heads received gradients; one update; scale encode/decode checked` 是寫死在 print 裡的字，真正做核對的是前面的 assert；第 9 章遇到同類訊息有說明，本頁沒有，初學者會以為這行是量出來的結果。 |
| 9 | 建議 | 第 177 行「loss 已經降到 0.07，小框的 IoU 為什麼還不到 0.5？」整段 | 讀者沒辦法把 0.002、0.007、0.016 和總 loss 0.07 連起來，因為頁面沒寫 grid_loss 的組成：total＝5×box＋objectness＋classification，box 是正格 4 個座標平方誤差的平均。少了這個，「在 loss 裡很小」無從驗算。另外，用紀錄的框算，中心 y 偏約 0.9 pixel，平方後約 0.013，其實是框 loss 裡最大的一項。文中「差的主要是寬和高」講的是 IoU 低的原因（本例中心偏移幾乎不影響交集），不是 loss 的組成；兩件事沒分開，讀者自己算會覺得前後矛盾。 |
| 10 | 建議 | 第 20–21 行〈與原版 YOLOv3 的差異〉：「和它重疊最大（IoU 最高）的那一個 anchor」與「與 GT 重疊超過 0.5」 | anchor 只有寬高、沒有位置。第 9 章教的是把中心疊在一起比的「尺寸 IoU」，best anchor 就是尺寸 IoU 最大的那一個。這裡只寫「重疊最大」，緊接著 ignore 又用同一個詞「重疊」，讀者分不出兩處比的是哪一種 IoU，還可能以為 anchor 也有位置。 |
| 11 | 建議 | 第 15 行「本節也只加多尺度預測、不做特徵融合」 | 「特徵融合」第一次出現時沒有解釋，定義放在第 23 行預設收合的摺疊區。沒展開摺疊區的讀者，讀到這裡不知道融合是什麼，也看不懂後半句「避免把效果同時歸因給兩件事」。 |
| 12 | 建議 | 第 145 行〈常見錯誤〉第 1 條，以及第 149–158 行練習與參考答案 | 「target 的格子索引先 gy、再 gx」這條提醒，在本頁找不到實例：頁面上的程式索引全是對稱的（[0,0,0]、[0,1,1]、[0,5,5]、格 (2,2)），讀者從沒看過一次先 gy 再 gx 的寫法。練習 1 的答案 (gx=3,gy=2) 正好不對稱，卻沒有接上程式的寫法。 |
| 13 | 建議 | 第 181 行「想自己重跑」 | 頁面要讀者「以自己那次的報告為準」，卻沒說報告 JSON 的哪個欄位對到正文哪個數字：3.6594 是 initial_loss，0.0703 是 last_pre_update_loss，6.504 是 weight_delta_l2，mAP50 是 training_image_metrics 的 map，各類 AP 在 ap_per_class，框與分數在 prediction。報告還會印出 machine、code、dependencies_sha256 等十幾個 SHA-256，以及正文沒提到的 precision 0.333、recall 0.5，初學者不知道該看哪幾個。 |
| 14 | 建議 | notebook 第三格的〈可選：兩尺度 40 步與實際預測〉（由 scripts/build_lesson_notebooks.py 的 optional_experiment 產生，10-multiscale 分支） | 「### 可選：兩尺度 40 步與實際預測」這個標題就在完整實驗那格的正上方。文字只寫「跑完完整實驗後」，不像 01／03／04 的版本寫「跑完下面的完整實驗後……不取代下面的完整實驗」。在 Colab 的目錄裡，完整實驗那格會被歸到「可選」標題底下，讀者可能以為下一格就是可選的 40 步程式。網頁第 181 行又特地把讀者指到這一段。 |
| 15 | 建議 | 第 34–41 行與第 53–59 行兩張表（手機寬度） | 在 390 px 寬的截圖裡，表格中沒有空格的 inline code 會在數字中間斷行，例如「(0+.5625)×1」換行接「6=9」、「(.5625,.562」換行接「5)」、「[1,3,64,64」換行接「]」。手機讀者可能把 ×16 看成 ×1。 |
| 16 | 建議 | 第 164 行的 10-multiscale-learning.svg（由 scripts/run_multiscale_learning.py 繪製：figsize 8×3.6、框旁標籤 fontsize 8） | 在 390 px 寬的截圖裡，左右兩張圖並排縮小後，「類別 c score s」標籤和軸標題只剩約 5 px 高，看不清楚。偏偏第 166、175 行正要讀者去讀這些標籤，例如 0.13 那個框的標籤位置與分數。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：工作副本：暫存副本（用 rsync 建立，排除 .git、site、.venv*、artifacts/runs、data/curated、data/downloads）。沒有在原 repo 執行或寫入任何東西。

【一手來源】
1) YOLOv3 論文 PDF：https://arxiv.org/pdf/1804.02767v1 ，用 Swift PDFKit 抽出文字後閱讀。
- 2.1：「This should be 1 if the bounding box prior overlaps a ground truth object by more than any other bounding box prior. If the bounding box prior is not the best but does overlap a ground truth object by more than some threshold we ignore the prediction … We use the threshold of .5. Unlike [17] our system only assigns one bounding box prior for each ground truth object.」
- 2.2：「we simply use independent logistic classifiers. During training we use binary cross-entropy loss … (i.e. Woman and Person)」
- 2.3：「YOLOv3 predicts boxes at 3 different scales」「we predict 3 boxes at each scale … N × N × [3 ∗ (4 + 1 + 80)]」「upsample it by 2× … merge it with our upsampled features using concatenation」「9 clusters and 3 scales … divide up the clusters evenly across scales」
- 2.4：「It has 53 convolutional layers so we call it … Darknet-53」，以及表 1。
- 2.5：「We use multi-scale training」

2) 官方 darknet，固定在 https://github.com/pjreddie/darknet/tree/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9 ：
- src/yolo_layer.c
  - 第 100 行：`float tw = log(truth.w*w / biases[2*n]);`
  - 第 141、143 行：xy、obj、class 都經 LOGISTIC。
  - 第 179–181 行：`if (best_iou > l.ignore_thresh) { l.delta[obj_index] = 0; }`
  - 第 203–216 行：`truth_shift.x = truth_shift.y = 0;`，對 `l.total` 個 anchor 算 `box_iou(pred, truth_shift)`，再用 `int_index(l.mask, best_n, l.n)` 判斷 best anchor 在不在本層。
  - 第 190、219 行：框 loss 乘上 scale `(2-truth.w*truth.h)`。
  - 第 316–343 行：get_yolo_detections。
- cfg/yolov3.cfg
  - 第 607–614、693–700、780–787 行：三層的 mask 依序是 6,7,8／3,4,5／0,1,2，`ignore_thresh = .7`。
  - 第 618–633、705–720 行：route、upsample stride=2、`layers = -1, 61`／`-1, 36`。
  - 用腳本數第 0–74 層：52 個 convolutional、23 個 shortcut。
- cfg/darknet53.cfg：共 53 個 [convolutional]，最後一個是 filters=1000 的分類層。
- src/network.c 第 510–566 行：num_detections、fill_network_boxes 逐層收集 YOLO 層的框。
- src/box.c 第 58 行起：do_nms_sort，逐類別處理。
- examples/detector.c 第 597–603 行：test_detector 先呼叫 get_network_boxes，再呼叫一次 `do_nms_sort`。
- cfg/yolov3.cfg 的歷史（用 gh api repos/pjreddie/darknet/commits?path=cfg/yolov3.cfg 查）：
  - e84933bfdd7315736c442a41d9aed163843dda54（2018-03-25）：ignore_thresh = .5
  - 481d9d98abc8ef1225feac45d04a9514935832bf（2018-05-06）：.7
  - f86901f6177dfc6116360a13cc06ab680e0c86b0：.7

【本 repo 檔案】
- 主要檔案：docs/lessons/10-multiscale.md、lesson_cases/10-multiscale.py。
- miniyolo/：targets.py、losses.py、inference.py、geometry.py、metrics.py、models.py（GridDetector）、figures.py。
- scripts/：run_multiscale_learning.py、run_learning_extensions.py、build_lesson_notebooks.py、validate_lessons.py（code_lines／excerpt_problems）。
- notebooks/10-multiscale.ipynb，含〈可選：兩尺度 40 步與實際預測〉；最後一格和程式檔相同。
- 用來交叉比對的課程頁：01-small-cnn.md（感受野規則）、07-inference.md（顯示門檻 0.25）、08-own-data.md（來源切分）、09-anchors.md（anchor×exp(tw)、ignore 的尺寸 IoU >0.2）、11-fusion.md 與 lesson_cases/11-fusion.py（隨機張量，沒有接上 TwoScale）、11-iou-loss.md。
- 為了確認「中文註解是本頁加的」這個慣例，用 grep 查過其他課程頁。
- 只看背景、不比較的檔案：artifacts/checks/curriculum/10-multiscale-learning.json。
- 其他：兩張 SVG、審查用的事實與寫作規範清單。

【執行的程式與指令】
- 摘錄比對 → []
- PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/10-multiscale.py → exit 0，5 行輸出和頁面一致。
- 同樣的環境執行 scripts/run_multiscale_learning.py → exit 0，產生 artifacts/runs/multiscale-learning/{report.json,learning.svg}。
- 自寫探針 probe_heads.py：確認 class 0 框來自 fine 格 (1,1)、0.13 框來自 coarse 格 (row 3, col 2)，並算出各項平方誤差。
- 自寫探針 probe_nms.py：用 miniyolo.geometry.nms 驗證「一次合併 NMS 留 A、C；先尺度內再合併只留 A」。
- .venv-docs/bin/zensical build --clean --strict → exit 0，No issues found。另外檢查本頁 HTML 的表格、摺疊區與摘錄。
- qlmanage -t -s 1200／2400 轉出三張 PNG（新 learning.svg、repo 舊版 learning.svg、10-multiscale.svg）後看圖，並用 PIL 放大右圖。
- python3 腳本比對 notebook 最後一格與程式檔、數 cfg 的層數。

【手算】
- target 表、44÷16、44÷8、練習答案。
- 64→32→16→8→4，感受野 15 與 31。
- 參數 224+1168+2320+4640+119+231=8702；logits 112 與 560；大框蓋到 9 個 fine 格；負格 78 個。
- 用紀錄框算 IoU 0.441／0.863／0.299／0.279；寬高誤差平方 0.00216／0.00715；中心偏移 0.16 與 0.90 px。
- 感受野位置：coarse 格 i 是 [16i−15, 16i+15]，fine 格 i 是 [8i−7, 8i+7]。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/10-multiscale.md 第 124 行（「完整流程是：各尺度 decode（…再做尺度內 NMS）→ 串接 → 再做一次分類別 NMS」），以及第 13 行「與原版的差異集中在下方摺疊區」 | 本節推論比原版多了一道「尺度內 NMS」，來自重用第 7 章的 decode_grid。官方 darknet（f6afaab）的做法是：network.c 的 fill_network_boxes 先把三個 yolo 層的框全部收齊，examples/detector.c 第 603 行只呼叫一次 `do_nms_sort`（逐類別、跨所有尺度），沒有尺度內 NMS。頁面把本節的做法稱為「完整流程」，摺疊區也沒列出這項差異。兩種做法並不等價：先在尺度內做貪婪 NMS 再合併，遇到連鎖重疊時留下的框可能不同。我用 miniyolo.geometry.nms 實測：A=[0,0,10,10] 來自 coarse，score .9；B=[3,0,13,10]、C=[6,0,16,10] 來自 fine，score .8、.7。A–B 與 B–C 的 IoU 都是 0.538，A–C 是 0.25。合併後做一次 NMS 會留下 A、C；先在 fine 內做 NMS（B 刪掉 C）再合併，只剩 A。 |
| 2 | 建議 | docs/lessons/10-multiscale.md 第 177 行（解釋 loss 降到 0.07、小框 IoU 卻不到 0.5 的段落，結尾是「第 11 章的 IoU 類 loss … 就是針對這種落差」） | 這段說寬高誤差在 loss 裡很小，原因是寬高除以全圖 64。這是本節沿用第 7 章 sigmoid×64 寫法造成的，不是原版 YOLOv3 的性質，頁面卻沒有說明原版在這點不同。原版的寬高是相對 anchor 的對數比例：yolo_layer.c 第 100 行寫 `float tw = log(truth.w*w / biases[2*n]);`。官方程式還在第 190、219 行把框的梯度乘上 `(2-truth.w*truth.h)`，讓小框的權重較大（論文沒寫這點）。用頁面引用的紀錄框座標算：寬 5.03 對 8、高 13.41 對 8，換成 ln 比值後平方約 0.22 與 0.27，遠大於本節的 0.002 與 0.007。照現在的寫法，讀者可能以為「loss 低但小框 IoU 低」是 YOLOv3 普遍會有的問題，只能靠 IoU 類 loss 解決。 |
| 3 | 建議 | docs/lessons/10-multiscale.md 第 20 行，摺疊區〈與原版 YOLOv3 的差異〉的「誰負責哪個尺度」 | 「只交給和它重疊最大（IoU 最高）的那一個 anchor」沒有說這個 IoU 怎麼算。anchor 只有寬高、沒有位置。官方程式的做法是：把 GT 和 9 個 anchor 的中心都放到原點，只比寬高（yolo_layer.c 第 203–214 行：`truth_shift.x = truth_shift.y = 0;`，對 `l.total` 個 anchor 算 `box_iou(pred, truth_shift)`），再用 `int_index(l.mask, best_n, l.n)` 判斷 best anchor 是否屬於本層。只寫「重疊最大」，初學者可能以為 anchor 在圖上有固定位置，是拿位置去比重疊。 |
| 4 | 建議 | docs/lessons/10-multiscale.md 第 21 行，摺疊區的「ignore」 | 門檻 0.5 出自論文第 2.1 節（「We use the threshold of .5.」），官方第一次釋出的 cfg/yolov3.cfg（commit e84933b，2018-03-25）也是 .5。但官方 darknet 從 commit 481d9d9（2018-05-06）起，把三個 [yolo] 層都改成 `ignore_thresh = .7`，目前的 master（f6afaab）仍然是 .7。頁面只寫「原版…重疊超過 0.5」，讀者去看現行官方程式會看到 0.7，不知道該信哪一個。 |
| 5 | 建議 | docs/lessons/10-multiscale.md 第 24 行，摺疊區的「主幹網路」 | 「原版的主幹網路是 Darknet-53，有 53 個卷積層」不精確。53 這個數字包含 ImageNet 分類用的最後一個 1×1 卷積：cfg/darknet53.cfg 共有 53 個 [convolutional]，最後一個是 filters=1000 的分類層。YOLOv3 做偵測時去掉了分類部分，cfg/yolov3.cfg 第 0–74 層的主幹是 52 個卷積加 23 個 shortcut。 |
| 6 | 建議 | docs/lessons/10-multiscale.md 第 9 行（「每格對應 16×16=256 個畫素…要和格內的背景一起被概括成同一組數字」） | 這句把格子負責的 16×16 區域，當成這格特徵所概括的範圍。這和第 28 行說的「stride 不等於感受野；coarse 的理論感受野是 31×31」不一致。依本節架構（四層 3×3、stride 2、padding 1），coarse 格 i 的理論感受野是第 16i−15 到 16i+15 個畫素：比 16×16 大，也不和格子邊界對齊。例如負責大物件的 coarse 格 (2,2)，看到的是 17–47，大物件 [32,56] 的右下一段不在範圍內。讀者讀到第 28 行時，會不知道哪個說法才對。 |
| 7 | 建議 | docs/lessons/10-multiscale.md 第 67 行（「下面摘出這幾行」）與第 104 行（第二個摘錄區塊前的引導句） | 兩個 data-excerpt 區塊裡的中文註解，例如 `# 三次 stride 2 卷積 → 細尺度特徵 [1,16,8,8]`、`# decoded_scales：…`，在 lesson_cases/10-multiscale.py 和 notebook 最後一格都找不到。validate_lessons 比對時會略過註解，所以檢查照樣通過。其他課程頁（02-diagnostics、03-identity、05-assignment、06-decode-nms、11-augmentation 等）都寫明「中文註解是本頁加的」，本頁卻沒寫，讀者在 Colab 對照程式時會找不到這些註解。 |
| 8 | 建議 | docs/lessons/10-multiscale.md 第 32 行，格子圖的圖說（「淡色格是訓練時的正格、斜線格是負格」） | 這句讀起來像只有斜線格是負格。實際上，除了兩個淡色正格，其餘 78 格（coarse 15 格加 fine 63 格）全是負格；斜線只標出「有物件中心、卻仍是負格」的那兩格。另外，圖中斜線和淡色疊在一起：coarse 格 (0,0) 的斜線蓋住了 fine 格 (1,1) 的淡紅，fine 格 (5,5) 的斜線落在 coarse 格 (2,2) 的淡藍裡。圖說沒有解釋這是兩種格子各自的格，讀者可能誤以為 fine 格 (1,1) 同時是正格又是負格。 |
| 9 | 建議 | docs/lessons/10-multiscale.md 第 141 行（「本節先保留兩個 head，留給後續的融合實驗」） | 課程裡沒有任何沿用 TwoScale 兩個 head 的融合實驗。第 11 章〈特徵融合〉用的是 8／16 channel 的隨機張量，頁面還明寫「並沒有直接接上第 10 章 16／32 channel 的特徵」，也沒有做品質對照。這句會讓讀者以為後面會在這個模型上做融合實驗；而「留給後續」也是審查用的事實與寫作規範清單寫作規則要避免的計畫式寫法。 |
| 10 | 建議 | docs/assets/diagrams/10-multiscale.svg 上方圖例（「stride 16（每格 16 畫素）」「stride 8（每格 8 畫素）」） | 「每格 16 畫素」可以讀成「每格只有 16 個畫素」，和正文第 9 行「每格對應 16×16=256 個畫素」矛盾。這裡要說的其實是格子的寬度。 |
| 11 | 建議 | docs/lessons/10-multiscale.md 第 13 行（「歷史機制：YOLOv3 … 在三個尺度上預測」） | 論文裡的「multi-scale」指兩件不同的事。第 2.3 節〈Predictions Across Scales〉是本節做的多尺度預測；第 2.5 節另寫「We use multi-scale training」，指的是訓練時隨機改變輸入尺寸（cfg 裡的 random=1）。本節標題叫「YOLOv3 多尺度」，卻沒有指明論文章節，讀者對照論文時容易把兩者混在一起。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 兩段摘錄裡的中文註解沒註明是本頁加的 | 已修正：第一段的引言改成「中文註解是本頁加的」。第二段前的句子改成「完整程式 main 裡的這幾行……（中文註解是本頁加的）」。摘錄比對工具結果仍為 []。 |
| 2 | 技術查核 | 推論比原版多一道尺度內 NMS，差異沒寫出來 | 已修正：摺疊區新增〈推論的 NMS〉一項：原版收齊三個尺度後只做一次逐類別 NMS；本節重用 decode_grid，各尺度內會先各做一次。框連鎖重疊時，兩種做法留下的框可能不同（已用 A／B／C 例子確認）。正文「完整流程」改成「本節程式的完整流程」，並註明原版只做最後那一次。 |
| 3 | 技術查核 | 「loss 低、小框 IoU 低」的原因來自本節的寬高寫法，原版不同 | 已修正：補一句：原版的寬高 target 是 ln(GT 寬 ÷ anchor 寬)，官方程式還把框 loss 的權重乘上 2−w·h（GT 越小權重越大），所以同樣的偏差在原版 loss 裡分量大得多。原本講 IoU 類 loss 的那句保留，「就是」改成「也是」。 |
| 4 | 技術查核 | best anchor 只寫「重疊最大」，沒說是尺寸 IoU | 已修正：改成「9 個 anchor 中尺寸 IoU 最高的那一個（第 9 章：中心疊在一起、只比寬高）」，由那個 anchor 所在的尺度、GT 中心所在的那一格負責。 |
| 5 | 技術查核 | ignore 門檻：論文寫 0.5，官方 cfg 是 .7 | 已修正：寫明論文的門檻是 0.5，官方 Darknet 的 cfg/yolov3.cfg 是 ignore_thresh = .7。也寫明官方程式比的是解碼後的預測框（含位置）和 GT 的一般 IoU。 |
| 6 | 技術查核 | Darknet-53 的 53 層含分類層，YOLOv3 實際用 52 個卷積層 | 已修正：改成「論文說它有 53 個卷積層，這包含 ImageNet 分類用的最後一層；YOLOv3 偵測時去掉分類層，用其餘 52 個」。 |
| 7 | 技術查核 | 第 9 行把格子範圍當成特徵概括的範圍，和感受野 31×31 的說法不一致 | 已修正：改成：每格「負責」16×16=256 個 pixel（物件中心落在這裡就歸這一格），特徵實際受影響的範圍（感受野）更大，理論上是 31×31；小物件在負責範圍裡占 1/4，在感受野裡占得更少。 |
| 8 | 技術查核 | 摘錄裡的中文註解沒註明是本頁加的 | 已修正：和先前查核留下的項目那條同一處修改，兩段摘錄都已註明。 |
| 9 | 技術查核 | 格子圖的圖說沒講清楚負格與斜線 | 已修正：圖說改寫：淡色是兩個正格；斜線標的是有物件中心、卻仍是負格的兩格，其餘白格也都是負格。另說明兩種格子疊在一起：fine 格 (1,1) 雖然被 coarse 格 (0,0) 的斜線蓋到，仍是正格；fine 格 (5,5) 在淡藍的 coarse 格 (2,2) 裡，情形相同。 |
| 10 | 技術查核 | 「留給後續的融合實驗」是計畫式寫法，課程裡也沒有這個實驗 | 已修正：改成描述現況：保留兩個 head 的接法，不代表兩尺度效果比較好；第 11 章的融合模組是設計給 8×8 fine head 用的，該節用隨機張量檢查接法，沒有接回本節模型訓練。內容已對照 11-fusion.md。 |
| 11 | 技術查核 | 圖例「每格 16 畫素」可讀成一格只有 16 個畫素 | 已修正：圖例和 desc 都改成「每格寬 16 畫素」「每格寬 8 畫素」。估算文字寬度仍在 720 的畫布內。 |
| 12 | 技術查核 | 論文裡的「多尺度」有兩種意思，頁面沒有區分 | 已修正：在論文連結後補上：本節對應第 2.3 節 Predictions Across Scales；第 2.5 節的 multi-scale training 是訓練時隨機改變輸入尺寸，本節沒有做。 |
| 13 | 讀者審查 | 第 141 行是計畫式寫法，也和第 11 章的內容不符 | 已修正：和技術查核那條同一處修改。 |
| 14 | 讀者審查 | 圖說的規則讀起來會出現同時是正格又是負格的區域 | 已修正：和技術查核那條同一處圖說改寫，補上兩種格子疊在一起的說明。 |
| 15 | 讀者審查 | 圖例「每格 16 畫素」有歧義；頁面 pixel 和畫素混用 | 已修正：圖例改成「每格寬 N 畫素」。第 9 行的單位統一改用 pixel。SVG 和術語表本來就用「畫素」，那些地方屬於全課程的用字慣例，沒有改。 |
| 16 | 讀者審查 | 第 9 行（16×16）和第 28 行（31×31）沒有接起來，感受野也沒列算式 | 已修正：第 9 行改法同技術查核那條。第 28 行補上第 1 章的算法：每層增加 (3−1)×間距，間距依序是 1、2、4、8，範圍依序是 1→3→7→15，再一層是 15+2×8=31。 |
| 17 | 讀者審查 | 「分類別 NMS」沒有定義 | 已修正：第一次出現時補上「按預測類別分組、各組各做一次 NMS，也就是 6.1 節〈人工框解碼與 NMS〉的 class-wise NMS」，並加上連結。 |
| 18 | 讀者審查 | 沒說人工 logits 怎麼由 target 換算，也沒說 obj 和類別填了什麼 | 已修正：補一句：框值用 torch.logit 從 target 換回；正格的 obj 與 class 0 填 10、class 1 填 −10，其他格的 obj 填 −20，所以每個尺度 decode 後只剩一個框。已對照程式確認。 |
| 19 | 讀者審查 | 摘錄的中文註解沒註明；第二段摘錄的行尾註解在桌機寬度被截斷 | 已修正：兩處都註明了。「留下的就是小框」那行註解移到 assert 的上一行，摘錄比對工具結果仍為 []。 |
| 20 | 讀者審查 | 〈應核對〉清單沒對應到輸出行；最後一行固定字樣可能被當成量測結果 | 已修正：各項標出對應的輸出行號，並說明第 2、3 行的 4 個數依序是格內 x、y 與 normalized w、h。另補一句：第 5 行是寫在 print 裡的固定字樣，所有斷言都通過才會印出。清單順序沒有重排，這屬於寫作偏好。 |
| 21 | 讀者審查 | 第 177 行：loss 組成沒寫，IoU 低的原因和 loss 的組成混在一起 | 已修正：只改了會混淆的那一處：「差的主要是寬和高」改成「讓 IoU 掉下來的主要是寬和高」，把 IoU 和 loss 分開。沒有補 5×box 權重，也沒把中心偏移改成 0.9 pixel→0.013。理由是紀錄裡的框是 40 次更新之後的，0.07 卻是第 40 次更新之前量的，兩者硬湊起來驗算會誤導讀者；而且這樣會多出新的紀錄相依數字。 |
| 22 | 讀者審查 | best anchor 與 ignore 都只寫「重疊」，分不出比的是哪種 IoU | 已修正：best anchor 寫明是尺寸 IoU；ignore 寫明官方程式比的是解碼後的預測框和 GT 的一般 IoU（同技術查核那兩條）。 |
| 23 | 讀者審查 | 「特徵融合」第一次出現時沒有解釋 | 已修正：第 15 行加括號說明「把深層特徵放大後，和淺層特徵合在一起再預測」，並連到第 11 章〈特徵融合〉。「留到第 11 章再單獨加入」改成「在第 11 章單獨介紹」。 |
| 24 | 讀者審查 | 「先 gy 再 gx」在頁面上沒有實例 | 已修正：第 1 題答案補上：程式裡要寫 target['box'][0,2,3]（先 gy=2、再 gx=3），寫成 [0,3,2] 會讀到另一格。已在暫存副本用 build_targets 實跑，正格確實在 [0,2,3]。 |
| 25 | 讀者審查 | 第 181 行「以自己的報告為準」，沒對照報告欄位 | 未改：這是加值說明，不是正確性問題。報告欄位名稱（initial_loss、last_pre_update_loss、weight_delta_l2、map、ap_per_class、prediction）本身看得懂，正文也已說明每個數字的意思。 |
| 26 | 讀者審查 | notebook 的「可選」標題位置會讓完整實驗格被歸到它底下 | 未改：這段由 scripts/build_lesson_notebooks.py 產生，不在本次可編輯的檔案內。頁面第 181 行對這段的描述是正確的。 |
| 27 | 讀者審查 | 兩張表在手機寬度會在數字中間斷行（×16 看成 ×1） | 已修正：兩張表的 code span 裡，逗號與運算子兩側都加了空格，例如 (0 + .5625) × 16 = 9、[1, 3, 64, 64]，讓斷行落在空格上。 |
| 28 | 讀者審查 | learning.svg 在手機寬度標籤太小 | 未改：這張圖是程式繪製的，規定不得手改。要改得修改 scripts/run_multiscale_learning.py，再到紀錄機器上重產，不在本次範圍。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/09-anchors.md`、`docs/lessons/09-anchor-clustering.md`、`docs/lessons/10-multiscale.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/10-multiscale.md 第 9 行「本例理論上是 31×31（見下一節）」 | 這門課的「節」指一個課程頁。例如 09-anchors 的「下一節只改 anchor 尺寸的選法」指的是 09-anchor-clustering，09-anchor-clustering 的「上一節」指的是 09-anchors；第 10 章在導覽裡也只有一頁。照這個用法，「見下一節」會被讀成下一頁 11.1，但 31×31 的算式其實寫在本頁下方的〈同一個小框的兩種 target〉。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

〈同一個小框的兩種 target〉是本頁的 H2，該節說明感受野 1→3→7→15→31。lesson_cases/10-multiscale.py 是四層 3×3、stride 2 卷積，計算正確，第一段的交叉引用指得到。yolov3.cfg 的 ignore_thresh=.7、論文寫的門檻 .5，與摺疊區一致。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

比對紀錄的 28 項發現與〈定稿修正〉24 列，並查定稿修正的處理清單。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/10-multiscale.md〈定稿修正〉 | 4 項發現的處理被漏掉：「圖例「每格 16 畫素」可讀成一格只有 16 個畫素」（技術，已修正）、「圖例「每格 16 畫素」有歧義；頁面 pixel 和畫素混用」（讀者，已修正）、「notebook 的「可選」標題位置會讓完整實驗格被歸到它底下」（未改）、「learning.svg 在手機寬度標籤太小」（未改）。這些處理的 page 是 SVG 或 notebook 路徑，生成器沒歸到本頁。 | 已修正：頁面顯示的圖與該節 notebook 的處理都歸到本頁，〈定稿修正〉列出這 4 項。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：讀者 16 項、技術 11 項與先前查核 1 項，都在〈定稿修正〉28 列處理（上一輪漏掉的 SVG 與 notebook 4 項已補上）；〈後續編輯的檢查〉核對了 31×31 交叉引用與 yolov3.cfg 的 ignore_thresh；技術查核列出 YOLOv3 論文來源。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/10-multiscale.md〈留下的意見〉第 2 列、〈定稿修正〉第 13–14 列、〈上述處理〉摘要 | 〈留下的意見〉第 2 列是 docs/lessons/09-anchors.md Darknet 句的意見，處理寫「這項修正由下方〈後續編輯的檢查〉核對」，但本紀錄的〈後續編輯的檢查〉沒有核對 09-anchors 那句（核對在 reviews/09-anchors.md）。另有內部用語「和 tech 那條同一處修改」，以及〈上述處理〉摘要「並查 lean 工作流程 fix:b4a 的處理清單」。 | 已修正：09-anchors 的意見移回該頁紀錄；「tech 那條」改成「技術查核那條」，工作流程名稱換成白話。 |

### 第 4 輪：上一輪的處理：通過

〈留下的意見〉只剩本頁第 9 行「見下一節」一列，由本紀錄〈後續編輯的檢查〉第 1 輪核對（31×31 交叉引用）；09-anchors 的 Darknet 列已移走。第 208、209、211 行已改成「技術查核那條」，第 2 輪摘要也不再有 lean／修正清單。第 3 輪處理屬實。

## 紀錄重產後的檢查

2026-10-05，另一位 AI 依 `AGENTS.md` 與 `docs/preparation/publish.md` 第 7 步，獨立審查 `docs/lessons/10-multiscale.md` 的全文、兩張 SVG、程式與其 import 的 repo 模組、兩份新版 JSON、練習與答案、notebook 與兩段程式摘錄。原有 `reviews/10-multiscale.md` 僅用來列出待重新查證的來源與項目，沒有沿用其通過結論。**本輪沒有必要問題；有一項可選的手機閱讀改善。**

### 工作副本與實跑範圍

副本在 `/tmp/lessons-v0.4.0-reviews/review-10/`，用發布審查提供的 `rsync -a --exclude=.git --exclude=site --exclude='.venv*'` 從 `/tmp/lessons-v0.4.0-reviews/base/` 建立。模型命令一律以副本為 cwd，`PYTHONPATH=.`，使用 `/workspace/learn_to_yolo/.venv-model/bin/python`（Python 3.12.14、PyTorch 2.9.1+cpu、2 執行緒）；文件命令使用原 repo 現有 `.venv-docs` 的工具。沒有執行 git、GPU 或遠端 workflow，也沒有改 root 的 tracked files 或 coverage。副本中只新增實跑／瀏覽器輸出，未修改課文、程式、notebook 或網站紀錄。

實際執行：

- `PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python lesson_cases/10-multiscale.py`：exit 0；assert 全通過；五行 stdout 與新版 `10-multiscale.json`、頁尾及 notebook 保存的 stdout 完全一致。沒有拿最後一行固定字樣當成量測結果。
- `scripts/run_learning_extensions.py --help` 與 `scripts/run_multiscale_learning.py --help`：一般 learning runner 只接受 01、03、04 三節，沒有本節或 `--steps`；本節使用專用腳本，40 步寫在程式裡。
- `PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python scripts/run_multiscale_learning.py --output artifacts/runs/release-review-10`：exit 0，真的做了 40 次 CPU 更新並重產 SVG，未用 `--record` 覆寫紀錄。新 `report.json` 的所有欄位（含 machine、dependency hashes、40 個 loss、AP、框、score）與發布副本的新 `10-multiscale-learning.json` 完全一致；SVG 逐 byte 相同。
- 讀取 notebook 最後一格原始 source，確認與 lesson case 逐字相同，再以 `__name__='__main__'` 在獨立 Python namespace 執行該格；stdout 再次與保存輸出及 JSON 一致。讀過 bootstrap 與可選 40 步指令，確認 `os.chdir(repository)` 讓後續相對路徑成立；未執行會 clone/install 的 bootstrap，也未實測 Google Colab 託管 runtime。
- `.venv-docs/bin/zensical build --clean --strict`：通過。`python3 scripts/validate_site.py`：60 頁、42 課、所有站內連結／錨點、Colab 配對、Markdown 轉換與 artifact 邊界通過。
- `python3 scripts/validate_lessons.py`：全課程 notebook／摘錄比對先通過，之後在舊 coverage 的過期清單停止；清單包含本節重產的 learning SVG 與其他待重審頁面。這是本輪尚未追加審查／更新 coverage 的預期結果，不能聲稱整支 validator 已通過。本輪沒有自行更新 coverage。

### 程式、手算與學習紀錄

逐句對照了 `lesson_cases/10-multiscale.py`、`scripts/run_multiscale_learning.py`、`scripts/run_learning_extensions.py`，以及 import map 中的 `miniyolo/__init__.py`、`data.py`、`models.py`、`targets.py`、`losses.py`、`inference.py`、`geometry.py`、`metrics.py`、`figures.py`、`provenance.py`。另用 `repo_dependencies(...)` 重新計算兩份紀錄的全部相依檔案 SHA-256，均與 JSON 相符。未將讀過但本節沒有呼叫的模組功能當成實測。

核對結果如下：

- 小框中心 `(9,9)`：coarse `(gx,gy)=(0,0)`、xy `.5625,.5625`；fine `(1,1)`、xy `.125,.125`；兩邊 wh 都是 `.125,.125`，decode 都回到 `[5,5,13,13]`。大框中心 `(44,44)`：coarse `(2,2)`、xy `.75,.75`、wh `.375,.375`，fine 中心所在負格 `(5,5)` 正確。大框覆蓋的 9 個 fine 格都沒有被分配 positive。
- 卷積 shape `64→32→16→8→4`、stride 8/16、感受野 `1→3→7→15→31`、head 的 `[1,8,8,7]`／`[1,4,4,7]`、80 候選與 560 logits 元素均正確；fine head 參數為 `16×7+7=119`，全部參數 8702。
- 額外以 `torch.autograd.grad` 分別對兩項 loss 求梯度：fine loss 到 `early`、`fine_head`；coarse loss 到 `early`、`deep`、`coarse_head`，與課文的連線分工一致。
- 練習以 `[20,12,36,28]` 建 target 實跑：fine 正格索引 `[0,2,3]`，值 `[.5,.5,.25,.25]`；coarse 索引 `[0,1,1]`，值 `[.75,.25,.25,.25]`。答案與先 gy 再 gx 的說明正確。
- 40 步紀錄：initial loss `3.6593661308288574`、最後更新前 `0.07033687829971313`、權重變化 L2 `6.503562899854255`。每步有限 loss／所有參數有限梯度／兩個 head 的非零 weight gradient，都是實跑 assert，而不是僅從已存 JSON 推斷。
- 依紀錄框手算及以 `box_iou` 交叉檢查：小框與 GT IoU `0.4410430693`，大框與 GT `0.8629898026`，低分 class 1 框與 GT `0.2991216887`；兩個 class 1 預測框之間 IoU `0.2788788844`。AP50 為 class 0=0、class 1=1、mAP50=.5，低分 FP 排在 TP 後，不降低 interpolated AP 的解釋正確。
- 小框實際 wh `[5.0285931,13.4116001]`，中心誤差 `[-.1609917,-.8991833]` pixel。wh 正規化平方誤差 `[.00215558,.00714976]` 與課文約 .002/.007 一致；「偏 1 pixel 的 fine xy 平方約 .016」是示例值 `(1/8)^2=.015625`，沒有誤稱為紀錄中實際中心誤差。
- 用另造的 A/B/C 連鎖重疊框實跑 NMS：合併一次留下 A/C；fine 先刪 C 再合併只剩 A，證明折疊區所述兩種流程可能不同。人工 duplicate 框的 2→1 與實際模型預測／AP 有清楚區分，沒有誤稱多尺度提升小物件品質。
- 對照第 11 章頁面與 `lesson_cases/11-fusion.py`：該節確實只用隨機 shallow/deep 張量確認融合接法，沒有接回本節 TwoScale 訓練。

### 重新查閱原始來源

重新透過 HTTPS 取得並閱讀以下來源；下載清單、URL 與每個來源 SHA-256 保存在 `/tmp/lessons-v0.4.0-reviews/review-10-sources/downloads.json`。本輪沒有依賴先前 review 摘錄代替原文。

- [YOLOv3 原論文 v1 PDF](https://arxiv.org/pdf/1804.02767v1)，SHA-256 `37049049b5e06f67c6cd22b72f7b9352914b0eb3a03e99abf54594c1005a8468`；用 `pdftotext -layout` 抽出原文重新閱讀第 2.1–2.5 節。單一最佳 prior、論文 ignore=.5、獨立 logistic 類別與 BCE／Woman+Person、三尺度／每尺度 3 框、9 clusters 平均分組、upsample+concat、Darknet-53 與 multi-scale training 的敘述均對上。
- 官方 pjreddie/darknet 固定 commit [`f6afaabcdf85f77e7aff2ec55c020c0e297c77f9`](https://github.com/pjreddie/darknet/tree/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9)。重新閱讀 [`src/yolo_layer.c`](https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/src/yolo_layer.c#L93)：第 100–106 行 log-width/height target 與 scale，第 164–180 行解碼框／GT 一般 IoU 的 ignore，第 203–228 行中心重疊的尺寸 IoU best anchor 與 mask assignment；`2-truth.w*truth.h` 確實縮放框 delta。
- 同 commit 的 [`cfg/yolov3.cfg`](https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/cfg/yolov3.cfg#L606)：三個 mask 依序 6,7,8／3,4,5／0,1,2，ignore=.7、truth_thresh=1；三個預測尺度以及兩次 upsample stride=2／route concat 與課文一致。程式計數前 75 層得到 52 conv＋23 shortcut；[`cfg/darknet53.cfg`](https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/cfg/darknet53.cfg) 有 53 conv，最後一層 filters=1000，因此 53/52 的補充正確。
- 同 commit 的 [`src/network.c`](https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/src/network.c#L542)、[`src/box.c`](https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/src/box.c#L58)、[`examples/detector.c`](https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/examples/detector.c#L599)：先收齊所有 YOLO 層的框，再呼叫一次逐類別 `do_nms_sort`；第 63–79 行 random resize 也確認 multi-scale training 是變更輸入尺寸，和本節多尺度 head 不同。

### SVG 逐點查核與實際網頁

獨立查核腳本 `/tmp/lessons-v0.4.0-reviews/review-10-svg-audit.py` 直接讀發布 SVG 的 XML 與 JSON，以圖上 tick／plot bounds 反推座標，而非只確認檔案存在：40 個 loss 頂點逐點符合，最大 SVG 座標誤差 `2.70×10^-6`（SVG 小數取整）；2 個 GT、3 個預測框的各頂點與紀錄相符；3 個預測標籤對上 class/兩位小數 score；GT 綠虛線與 prediction 橙實線、圖例對上。

靜態 `10-multiscale.svg` 另逐項核對 `88+6x,150+6y`：14 條內部粗／細格線、6 個物件框／正負格矩形、兩個中心 `(142,204)`／`(352,414)`、所有格號及表內值均符合，斜線疊在 fine 正格／coarse 正格上的圖說與畫法相符。

只啟動自己的副本 site HTTP server（port 8804），用 Playwright 的 Chromium `/usr/bin/chromium`，args `--no-sandbox --disable-gpu --disable-dev-shm-usage` 檢查桌機 1440×1000 和手機 390×844。實際看過兩張 SVG、桌機及手機兩張表、三個展開區塊和答案；瀏覽器核對兩段 Python 摘錄與 stdout code block 文字仍完整。兩張 SVG 的 53／35 個 text 經 transform 後的螢幕 bounds 均在畫布內；無 page error、頁面沒有水平溢出，兩圖載入成功。截圖與文字 bounds 輸出在副本 `artifacts/runs/release-review-10/browser/`。本輪結束已停止自己啟動的 server；未碰其他 server。

| 檔案 | SHA-256 |
| --- | --- |
| docs/lessons/10-multiscale.md | `e00dba88187fff50bc74c7d76ed0ddec22d3233089e38fd949b97eafd246c290` |
| lesson_cases/10-multiscale.py | `add9aefa8efec7180d06ab255807ac4fa6847c490c8447774191f0a6566ed6a6` |
| scripts/run_multiscale_learning.py | `db5480b8b226793092a6c59b4c79197ad0d33e7327a384f5b9a46eb573413e89` |
| notebooks/10-multiscale.ipynb | `8573fc65e52c9970c0157dc1742370e50688f1f30d4b10b3ec8cb26d344b6feb` |
| artifacts/checks/curriculum/10-multiscale.json | `c9ec68945d9d8c6b76f633e3fc169e92bc45c3d4e9f5c86058546dfe39c324f5` |
| artifacts/checks/curriculum/10-multiscale-learning.json（與重跑 report 相同） | `b825d38aa28ad98d0c84fe38f8be79431fae3f13d18f2690e5028da6f122ec09` |
| docs/assets/diagrams/10-multiscale.svg | `b66a8a3d3f6b177bdfb3e865fb8be72b1b4fe8b083e4cdef69d43af26df8ef31` |
| docs/assets/diagrams/10-multiscale-learning.svg（與重產 SVG 相同） | `4c985b8824c35b49b6ca1a88df45d7980e60176906c51dff86feab779aaabd17` |

### 本輪發現與處理

| # | 嚴重度 | 位置 | 發現 | 處理 |
| --- | --- | --- | --- | --- |
| 1 | 可選 | `docs/assets/diagrams/10-multiscale-learning.svg` 在本節 390 px 手機版面 | 實際圖片寬約 343 px，左右兩圖並排縮小，8pt 的「類別 c score s」標籤約只剩 4.8 px，閱讀吃力；正文第 177 行會請讀者辨識低分框標籤。桌機完整可讀，座標與標籤均正確。 | 保留為可選改善，沒有改圖或腳本。可在未來修改繪圖腳本，提供手機可讀的上下排圖／較大標籤，再重產紀錄與重審；這不影響本輪機制與數值正確性。 |

從不熟悉本專案、但懂基本 Python／神經網路的讀者角度逐句閱讀：fine/coarse 命名、格子與感受野、人工尺度責任、互斥類別、尺度內與跨尺度 NMS、AP 與低 loss 的落差，以及固定訓練圖的限制已有足夠說明；未發現需阻擋發布的理解問題。

### 編輯處理

保留手機雙欄學習圖的閱讀建議，供之後調整圖板版型時一併處理。正文已列出三個預測的類別、score 與 IoU，這次維持目前圖版及其逐點核對結果，沒有修改正文或繪圖程式。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[evolution當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/evolution.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/evolution.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/evolution.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_a` 實讀本頁 5 個凍結單元；首次使用／前文方法範圍四題位置為 10-multiscale/00:first_use, 10-multiscale/02:first_use，頁末為 10-multiscale/04。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_a.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_a-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_a.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS05。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 c1 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `c_foundation` 實讀本頁 6 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/c_foundation/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：R017, R023。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/10-multiscale.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
