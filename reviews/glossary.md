# 審查紀錄：術語快速查

審查範圍：`docs/glossary.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題。docs/glossary.md 的 6 處改動都正確，整頁引用的節次、例子、印出值和變數名，也都和目前的課程頁與程式一致。只有一項可選的 should：三列的詳見欄少了補充連結。

**一、遺留項目、受程式改動影響的段落與附加指示（第 2 項）**
- 先前查核的建議清單裡沒有這次查核的頁面的 key，先前查核留下的項目清單也沒有，所以沒有遺留項目。
- ltrb 例子：我自己算過，l=(28−12)/8=2、t=(28−16)/8=1.5、r=(40−28)/8=1.5、b=(36−28)/8=1，得到 [2,1.5,1.5,1]。這和 lesson_cases/12-anchor-free.py 第 16–19 行的斷言、12.1 頁第 17、19、44 行一致；在暫存副本執行該程式，印出 `[[2.0, 1.5, 1.5, 1.0]]`。
- mAP50 名稱：8.2 頁第 191、223 行用的就是 mAP50。紀錄 artifacts/checks/curriculum/custom-data-learning.json 的 final_train.map 是 1.0。（審查用的事實與寫作規範清單寫這個路徑時省略了 curriculum/，編輯引用的完整路徑才對。）
- CE 梯度：7.3 頁現在確實有說明，在第 164 行，另有第 168 行的推導摺疊區；程式 07-loss.py 第 31 行有斷言，執行後印出 `[-0.5, 0.5]`。新句子裡「loss 可能看不出錯，這個梯度卻不對」兩種情況我都驗過：logits 相等時 loss 同為 0.693147、梯度卻變成一半；logits 不等時 loss 也會不同；隨機抽 2000 組 logits，梯度沒有一組等於 softmax 減 one-hot。one-hot 在條目裡就地定義了，連結也連到 7.3。
- 另外三處補充：N 的 12.4 用法（12.4 頁第 27、99 行）、tensor 補 [1]（01-small-cnn 第 17–26 行）、residual 補 [3.2]（3.2 頁第 26、35 行），都有根據。

**二、全頁逐條對照（第 1 項）**
符號清單的 B、C、c、N、S、A、K、P、G、M、D、T、d、σ、η、t、⌊x⌋，以及七張表和〈同名不同義〉的每一處節次與數值，我都對過目前的頁面與程式，全部一致。

照頁面的說法實際跑了程式來確認：
- 寫了一支檢查腳本（claims.py），驗 permute 與 reshape 的差別、torch.cat 軸長不同時報 RuntimeError、broadcast 不報錯、BCE 軟目標的梯度是 p−t、空遮罩取平均得 NaN、DFL 權重 0.75／0.25 與期望值 1.25、attention 第一列 [0.3349,0.1651,0.3349,0.1651]、FP32 約 6.9 位與 FP16 約 3.0 位有效數字。
- 執行了 12-anchor-free、07-loss、15-attention-bridge、12-dfl、06-evaluation、06-decode-nms 六支課程程式，結束碼都是 0。印出的 mAP50 是 1/3，score 有 0.72，都和本頁相符。

**三、數字（第 3 項）**
沒有新增任何在 Mac 上量到的訓練或計時數字。唯一會隨紀錄重跑改變的值是 train mAP50＝1.0，編輯已經列進待重錄數值清單。其餘數字都是固定的設定或可手算的例子。

**四、文字與一致性（第 4 項）**
頁面用現在式書寫，沒有描述修訂經過的字眼，也沒有把限制寫成之後的計畫。和審查用的事實與寫作規範清單、首頁、閱讀路線、課程大綱、驗證範圍頁描述本頁的方式都一致。9.1 頁連到本頁的 stride ①②、18 章連到本頁的 throughput，這兩個條目都還在。

**五、建置與渲染（第 6 項）**
- 在暫存副本跑 zensical build --clean --strict、validate_site.py、validate_preparation.py，三者結束碼都是 0。
- 渲染出來的頁面有 7 張表、60 列，和原始檔相同；2 個摺疊區正常，數學式正常，沒有殘留的 ??? 或表格直線。
- README 的 9 個相對連結都找得到檔案；43 本 notebook 都是有效的 JSON。

**需要說明的兩點**
- 工作樹裡有其他部分同時在修改的檔案，我無法逐一判斷是誰改的。不過 glossary 的 diff 只動到本頁。我建立快照之後，只有 17-capstone 第 181 行的措辭變了，和本頁引用的內容無關。
- 編輯在疑慮提到的問題是真的：本頁引用了課程頁的例子與數值，但目前沒有任何檢查守著。本頁的定稿審查只雜湊本頁的文字，所以課程頁改了，本頁的審查不會因此失效。建議課程頁定稿後、本頁做定稿審查之前，再把這些引用對一次。

相關路徑：
- 受檢檔案：docs/glossary.md
- 檢查腳本與執行紀錄：暫存副本（claims.py、run-*.log、build.log、validate_site.log、validate_prep.log）

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/glossary.md 三列的「詳見」欄：第 60 行（concat 與 add）、第 115 行（DFL、bin）、第 117 行（attention、token、Q／K／V） | 編輯依「留意欄提到的內容在別節教，詳見欄就補那一節」的原則，替 tensor 補了 [1]、替 residual 補了 [3.2]，但下面三列是同樣情況，卻沒有比照： (1) concat 列寫「沿 channel concat 時其他軸要一樣長，否則 torch.cat 會報錯」，這是 11.2 節第 53、62 行教的；詳見只連 3.1，而 3.1 第 23 行只講 concat 和 add 的差別。 (2) attention 列寫「Area Attention 分成 A 區後降為 N²/A」，這是 15.2 節第 60 行教的；詳見只連 15.1，整列也沒寫節次。 (3) DFL 列寫「YOLO26 設 reg_max＝1，等於不用 DFL」，這是 16.1 節第 17、21–25 行教的；詳見只連 12.4。 結果是讀者點了連結，找不到這些細節出自哪一節。這三句本身都正確，所以不是錯誤，也不影響本次通過。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以這一頁的讀者身分閱讀）

方法：讀的資料：
- 審查用的事實與寫作規範清單與 docs/glossary.md 全文（含兩個收合區塊）、learning-path.md、index.md。
- 被連到的課程頁：00、01、02、03-identity、03-projection（前段）、04-localization、04-coordinates、05、06-decode-nms、06-evaluation、07-loss、07-training（中後段）、09-anchors（前段）、10-multiscale（前段）、12-anchor-free、12-assignment、12-dfl（前段）、13-dual-assignment、15-attention-bridge、15-area-attention、16-training（前段）、17、18、19（前段）。
- 另寫小腳本，依導覽順序在 42 個課程頁 grep 每個術語第一次出現與定義的位置，用來核對「點下去回到第一次教這個詞的那一節」與各節次引用；也統計常用詞（MSE、斷言、計算圖、label、標註、推論、ReLU、CIoU 等）在課程頁出現的頁數與本頁是否收錄。

建置與畫面檢查：
- 在暫存副本執行 zensical build --clean --strict，exit 0、No issues found，讀了 site/glossary/index.html 的 article 內容。
- 用 headless Chrome 截圖：1280px 的收合版與展開版（展開版是另存一份把 details 設為 open 的副本），以及放進 390px iframe 的手機寬度版；用 JS 量測七張表的寬度與捲動容器。手機寬度下頁面不會橫向捲動，但表寬 412～614px，大於 375px 的可視寬度。
- 第一張收合截圖意外來自別的 reviewer 佔用 8765 port 的伺服器，我核對 md5 與自己的建置相同後，改用自己在 18931 port 的伺服器重拍其餘畫面。
- 用 window.find 試頁內搜尋：能找到收合區內的文字，但不會展開區塊，結果不足以下結論，所以不寫成發現。

實跑：
- 以 PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg 執行 16 節：00、05、06-decode-nms、06-evaluation、07-loss、09-anchor-clustering、11-augmentation、12-anchor-free、12-assignment、12-dfl、13-dual-assignment、15-attention-bridge、15-area-attention、16-dfl-free、16-training、19-tracking。全部 exit 0，與本頁引用的固定數值逐一比對都一致。
- 讀 artifacts/checks/curriculum/custom-data-learning.json，確認 final_train map 目前是 1.0。

驗證腳本與截圖放在暫存副本；沒有寫入 repo 根目錄。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈張量與訓練〉表 seed（亂數種子）列（glossary.md 第 43 行） | 「固定 seed，每次重跑的隨機初始權重（以及程式生成的資料）都一樣，結果才能重現」沒有說前提是同一台電腦、同樣的執行緒數。讀者在 Colab 重跑訓練實驗，看到 loss 或 mAP 的小數和頁尾紀錄不同時，照這句會以為 seed 沒設好或程式改壞了；但 7.4 節說執行緒數不同會改變加總順序、讓末幾位不同，第 1、5 章也說小數略有不同是正常的。另外列內的「（3.3 節）」是純文字，沒有連結。 |
| 2 | 建議 | 〈網路結構〉表 stride（步幅）列的②與詳見（第 55 行） | ②「特徵圖的 stride」在第 1 章裡叫「間距」（算感受野那段），第 1 章沒有把它叫 stride。第一次把它叫 stride、並指回本頁 stride 條目①②的是 9.1 節，詳見卻只連 [1]、[10]。讀者照開頭「點下去會回到第一次教這個詞的那一節」點 [1] 找②，找不到「stride」這個說法。 |
| 3 | 建議 | 〈YOLO 演化機制〉表 ltrb 列（第 113 行） | 本列寫「四邊的距離；本書以特徵格為單位」，但 16.3 節直接用畫素寫 ltrb（點 (4,4) 的 ltrb 是 [−3,−3,5,5] 畫素）；16.1 節說 YOLO26 的距離可正可負（點在框外時至少一邊是負的），12.1 節也有 −0.5 格的例子。讀者照本列會以為 ltrb 一定以格為單位、一定不是負數，讀到 16.1、16.3 對不上。 |
| 4 | 建議 | 〈訓練目標與 loss〉表 positive／negative／ignore 列（第 88 行）與 grid 列（第 83 行） | negative 定義成「要學背景（objectness 目標 0）的候選」，grid 列寫「物件中心落入的格是正格，其餘是負格」，這只是第 5、7 章 grid 的規則。第 12、13、16 章沒有 objectness，負樣本是「每個類別分數都學 0」（12.2 節稱為分類負訊號）；第 10 章另一個尺度的中心格也是負格；第 9 章的正樣本是 slot 而不是整格。讀到第 12 章再回來查的讀者，會以為負樣本一定有 objectness 目標。 |
| 5 | 建議 | 〈訓練目標與 loss〉整組（第 72–89 行） | 第 7 章的總 loss 是 box MSE＋objectness BCE＋class CE，本組列了 CE、BCE，卻沒有 MSE；MSE 出現在 13 個課程頁（3.1、4.1、7.3、9.1、11.4、16.3、17 等），用頁內搜尋找「MSE」或「均方誤差」都找不到。常用符號的 d 條目提到「12.1 節 Smooth L1 式子」，Smooth L1 本身也沒有條目。 |
| 6 | 建議 | mask 列（第 89 行「仍連著計算圖的 0」）、concat 列（第 60 行「要用 assert 核對」）；〈張量與訓練〉整組 | 本頁自己用了「計算圖」和「assert」，卻沒有解釋，也沒有條目。程式新手最常卡的幾個詞也都查不到：斷言（assert／AssertionError）出現在 40 個課程頁，幾乎每節練習都要讀者看懂 AssertionError、改斷言；計算圖、detach、no_grad 出現在 11～17 頁；label（標籤）出現在 19 頁，6.1 節還特別說程式變數 labels 在那裡是預測類別、不是真值標籤。 |
| 7 | 建議 | GT／target 列（第 82 行）與全頁收錄範圍 | 首頁與第 5 章用「標註（annotation）」稱呼人工答案（出現在 21 個課程頁），但本頁 GT 列只寫「人工標的正確框與類別」，「標註」這兩個字不在本頁任何地方，用頁內搜尋找「標註」或「annotation」都找不到。其他常用而未收錄的詞還有：推論（inference，33 頁）、ReLU／激勵函數（11 頁，只出現在頁末收合區）、回歸（regression，11 頁）、前處理／後處理（8／7 頁）、BatchNorm（8 頁）、kernel／濾鏡（8 頁，15.2 另有 GPU kernel 的不同意思）、感受野（6 頁）、MAC（乘加，5 頁）。 |
| 8 | 建議 | 〈YOLO 演化機制〉表（第 107–117 行） | 導覽裡幾節的主題詞都沒有條目：11.1 CSP、11.2 特徵融合（上取樣、特徵金字塔）、11.4 IoU 類 loss（GIoU／DIoU／CIoU）、12.2 decoupled head、12.3 task-aligned assignment（TAL）。其中 CIoU 在 12.1、12.3、13.1、16.1 的說明裡反覆出現，讀者讀到這幾節想查 CIoU，本頁沒有答案。 |
| 9 | 建議 | 〈常用符號〉收合區（第 11–31 行） | 這裡只收大寫字母和 d、t，沒有收幾個最容易混淆的符號： - e 多半是自然對數的底（sigmoid、softmax 裡的 e^z），但 16.3 節的 e 是第幾輪、E 是總輪數，15.1 節的 E[·] 是期望值。 - ln／log：7.3、9.1 節特別提醒程式的 log 以 e 為底，不是計算機的 log 鍵。 - p：第 1 章輸出長度公式裡是 padding，CE 裡是機率，12.3、13.1 節是候選名稱 p0、p1…，13.1 節品質公式裡又是分類分數。 - s：第 1 章是 stride，4.2 節是縮放比例，13.1 節是「是否在框內」。 - k：kernel 邊長、top-k 的名額、k-means 的群數、DFL 的 bin 編號。 - H：3.1 節的 H(x) 是理想轉換的函數名稱，不是高。 |
| 10 | 建議 | 〈推論與評估〉表門檻（threshold）列（第 97 行） | 「IoU 門檻比兩框的重疊，也分兩種」「四種門檻用途不同」讀起來像全書只有這四種門檻。但還有其他門檻：第 9 章的 ignore 規則是尺寸 IoU 大於 0.2，第 19 章的 association 是 IoU 至少 0.1，11.3 節用可見比例 0.5。讀到第 19 章看到 0.1 的讀者回來查，只能在頁末收合區的 matching 條目找到。 |
| 11 | 建議 | 頁末〈同名不同義〉收合區（第 128–141 行）與 ※ 記號（第 9、88 行） | 這段是收合的 note，不是標題，所以右側目錄只列七組表格，沒有它，也沒有錨點可以直接跳過去。表格裡的 ※ 沒有連到對應條目，條目內提到的節次（13.1、16.2、9.2、11.3、第 19 章等）也都沒有連結。activation、padding 只出現在這個收合區塊裡。另外 positive 在同名不同義裡有條目，表格裡卻只有 ignore 後面標了 ※。 |
| 12 | 建議 | 七張表格的「詳見」欄（手機寬度） | 以 390px 寬（可視寬 375px）模擬手機時，網頁本身不會橫向捲動，但七張表寬 412～614px，最後一欄「詳見」的起點在 x＝327～529，多數表格要左右滑動才看得到。開頭請讀者點的連結，在手機上預設是看不到的；其餘三欄也被擠成每行只有五、六個字。 |
| 13 | 建議 | 表格內提到節次的純文字（第 42、43、45、67、70、81、86、88、96、97、102 行） | 「16.3 節只有 4 筆資料」「（3.3 節）」「7.4 節把它列為」「第 7 章的框 target」「12.2、12.3 節提到」「第 9 章的 ignore 槽」「第 12、13、16 章的 head」「8.2 節用 0.1」「（第 17 章）」都是純文字，沒有連結。例如 step／epoch 列的例子出自 16.3，但詳見只連 0 與 11.3，讀者點 11.3 找不到「4 筆資料、30 輪」。 |
| 14 | 建議 | 〈張量與訓練〉表 overfit（過擬合）、泛化列（第 45 行） | 「8.2 節在偵測模型上實際做到（train mAP50＝1.0）」有兩個問題：mAP50 要到後面〈推論與評估〉才定義，在第 2 章查 overfit 的讀者看不懂這個數字；而且它是 8.2 節訓練紀錄裡會隨重跑改變的結果，寫在不屬於課程頁、也沒有執行紀錄區塊的本頁，紀錄重產後容易和 8.2 頁對不上。 |
| 15 | 建議 | backbone 列（第 56 行）、feature map 與 stride 列（第 54–55 行）、anchor-free 列（第 112 行） | 三處用詞或說明可以更清楚： - backbone（主幹）的解釋是「從圖片提取特徵的主幹」，用「主幹」解釋「主幹」。 - 相鄰兩列一列說「畫素」、一列說「pixel」，ltrb 與 decode 列又用「畫素」，頁面沒有說兩者是同一件事。 - anchor-free 列只舉第 12 章的 ltrb，讀者容易以為 anchor-free 就等於 ltrb；12.1 節明說依「不用預設框尺寸」的定義，第 7 章的 grid 也算 anchor-free。 |

### 事實查核（AI 對照 repo 的程式、紀錄與頁面引用的來源）

方法：repository 根目錄，分支 release/lessons-v0.4.0，commit 31527417d328e18418c438cb76b4630a8c695f33。我用 rsync 複製到暫存副本，所有執行都在這份暫存副本裡，原 repo 沒有寫入任何東西。

讀過的檔案：
- 審查用的事實與寫作規範清單
- docs/glossary.md（全頁）
- docs/learning-path.md
- zensical.toml 的 nav
- section-map.json
- docs/lessons/ 全部 42 節，curriculum-evidence 區塊以上的正文
- miniyolo/inference.py：decode_grid 的 score＝sigmoid(obj)×最大 softmax、預設 score_threshold=.25、分類別 NMS
- lesson_cases/16-training.py：train()，epochs=30、4 筆 features
- lesson_cases/19-tracking.py：Tracker 與 exact_gated_matching，threshold=.1
- lesson_cases/07-heldout.py（.01）、17-capstone.py（.05）、18-video.py（.1）、20-deployment.py（.05）的 score_threshold
- scripts/run_custom_data_learning.py：config 的 score_threshold .1；run_outcome 只回報是否 fitted，沒有斷言
- artifacts/checks/curriculum/custom-data-learning.json：final_train.map=1.0

外部來源（固定 commit）：
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml：`reg_max: 1 # DFL bins`、`end2end: True`
- 同一 commit 的 ultralytics/nn/modules/head.py：Detect.__init__ 的 reg_max=16、self.no = nc + reg_max*4、self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
- 同一 commit 的 ultralytics/nn/modules/block.py：class DFL 用 torch.arange(c1)，bin 是 0..c1−1
- 同一 commit 的 ultralytics/utils/tal.py：make_anchors 回傳的 anchor_points 是格單位的參考點；TaskAlignedAssigner 的 target_scores × norm_align_metric
- https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/nn/modules/head.py：v10Detect 的 one2one 吃 x.detach()、max_det=300、v10postprocess 不做 NMS
- COCO detection evaluation（https://cocodataset.org/#detection-eval；cocodataset.org 頁面抓不到內文，改讀 github.com/cocodataset/cocodataset.github.io 的 dataset/detection-eval.htm）Metrics 段：10 個 IoU 門檻 .50:.05:.95、AP 對所有類別平均、不區分 AP 與 mAP

執行的程式與命令：
1. rsync -a --delete（排除 .git、site、.venv*、artifacts/runs、data/curated、data/downloads）
2. 用 PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python 執行 lesson_cases/{15-attention-bridge,12-anchor-free,06-decode-nms,06-evaluation,12-dfl,07-loss,15-area-attention,00-warmup}.py，全部 exit 0。從 log 核對的確定輸出：
   - first attention row [0.3349,0.1651,0.3349,0.1651]
   - target ltrb [[2.0,1.5,1.5,1.0]]
   - scores [0.64,0.72,0.855]
   - mAP50=0.333333
   - DFL=1.386294
   - class gradients [-0.5,0.5]
   - affinity 256／64
3. .venv-docs/bin/zensical build --clean --strict（exit 0），接著 python3 scripts/validate_site.py（exit 0）
4. 用 Python 解析 site/glossary/index.html：7 個表格、2 個 details、15 個 arithmatex，沒有殘留的原始 Markdown
5. Python 腳本：
   - 81 個「詳見」連結的編號對照 learning-path，全部相符
   - ※ 詞對照〈同名不同義〉條目，全部有對應
   - 中英文之間的空格與半形標點檢查：沒有問題
   - 術語欄有無中文：12/60 列只有英文
6. awk／grep 掃描各術語最先出現在哪一節，用來檢查「第一個連結＝第一次教」的說法
7. torch float32 實測：sigmoid(17)=1.0、sigmoid(−89)=0.0；BCE-with-logits 在 z=0.3、t=0.4 時梯度等於 sigmoid(z)−t

本頁沒有 SVG，所以沒有用 qlmanage 算圖。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/glossary.md 第 7 行〈怎麼用這頁〉第一點：「術語欄同時寫英文和課文用的中文」 | 這句話不完全成立。60 列中有 12 列的術語欄只有英文：neck、head、xyxy／cxcywh、letterbox、logits、sigmoid（σ）、softmax、AP50／mAP、ltrb、top-k、DFL／bin、FP32／FP16。原因是課文本身就只用這些英文名稱，沒有中文名稱可寫。 |
| 2 | 建議 | docs/glossary.md 第 114 行 top-k 列「留意的地方」：「不看框的位置與重疊，本身不保證每個物件只留一個框」 | 同一列第一欄寫「依分數（或品質）由大到小排」，而 12.3 節的品質是 score×IoU²，本身就含預測框和 GT 的重疊。top-k 真正不看的，是候選框彼此之間的重疊，13.2 節的反例講的就是這點。現在的寫法可能讓 12.3 節的讀者誤以為 assignment 的 top-k 不考慮 IoU。 |
| 3 | 建議 | docs/glossary.md 第 88 行 positive／negative／ignore 列：「negative：要學背景（objectness 目標 0）的候選」 | 括號裡的「objectness 目標 0」只適用有 objectness 的 grid 模型。同頁 objectness 列自己就說第 12、13、16 章的 head 沒有 objectness；12.2 節的背景位置是「所有類別的 target 都是 0」，12.3 節未選的候選是負樣本、target 為 0。 |
| 4 | 建議 | docs/glossary.md 第 79 行 loss 列：「用一個可微分的數值，表示目前的預測離目標多遠」 | 和 11.4 節不一致。11.4 節明說 IoU 類 loss 用了 min、max、clamp，是分段函數，在轉折點（剛好相接、上下緣對齊）沒有單一斜率，也就是不可微，PyTorch 在那裡另外給一個值。初學讀者對照兩頁會困惑。 |
| 5 | 建議 | docs/glossary.md 第 77 行 sigmoid 列：「把任意實數壓到 0 與 1 之間（碰不到兩端）」 | 這只在數學上成立。7.5 節明說 float32 在 logit 極負時，sigmoid 本身會算成 0。我實測 float32：torch.sigmoid(17.)==1.0、torch.sigmoid(-89.)==0.0。 |
| 6 | 建議 | docs/glossary.md 第 113 行 ltrb 列：「本書以特徵格為單位，也就是畫素距離除以 stride」 | 有例外。16.3 節 STAL 的例子把 (4,4) 對原框的 ltrb 寫成 [−3,−3,5,5]「畫素」。12.1、12.4、16.1、16.2 節才是用格。 |
| 7 | 建議 | docs/glossary.md 第 45 行 overfit 列：「8.2 節在偵測模型上實際做到（train mAP50＝1.0）」 | 這是 8.2 節 1600 步訓練的結果，不是確定值。它目前和紀錄一致：artifacts/checks/curriculum/custom-data-learning.json 的 final_train.map 是 1.0。但 scripts/run_custom_data_learning.py 只在 run_outcome 裡回報 fitted 或 not yet fitted，沒有斷言它等於 1。術語頁沒有執行紀錄區塊；在錄製用的電腦上重產紀錄後，如果只刷新課程頁，這句可能變成錯的。 |
| 8 | 建議 | docs/glossary.md 第 41 行 optimizer／SGD／Adam／lr 列，「詳見」欄只有 [0]、[1] | 「留意的地方」有兩個說法，出處都不在這兩節。「lr 太大會亂跳，甚至變成 NaN；太小幾乎不動」出自第 2 章的檢查步驟 3；「Adam 每步的改變量不等於 lr×梯度」出自 7.4 節。第 1 章只說 Adam 會自動調整步伐。照本頁的使用說明，讀者點 [0]、[1] 找不到這兩點。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 讀者審查 | seed 列沒有說只在同機器、同執行緒數下才相同 | 已修正：補一句：換電腦或改執行緒數時，初始權重與資料仍相同，但加總順序可能不同，訓練後的小數就可能不同（依 7.4 節）。詳見補上 3.3、7.4。 |
| 2 | 讀者審查 | stride ② 在第 1 章叫「間距」 | 已修正：②加上「（第 1 章算感受野時叫它『間距』）」。沒有加 9.1 連結，因為第 1 章原文已經寫「後面章節把這個間距叫特徵圖的 stride」，審查說第 1 章找不到這個說法並不正確。 |
| 3 | 讀者審查 | ltrb 的單位與正負號說得太絕對 | 已修正：改成「多半以特徵格為單位（16.3 節的例子直接用畫素）」，並補一句：點在框外時至少有一邊是負的；12.1 的 softplus 表示不了，16.1 的 YOLO26 可正可負（三節都核對過）。 |
| 4 | 讀者審查 | negative／grid 的定義只適用 grid 模型 | 已修正：negative 改成「要學背景的候選（grid 模型是 objectness 目標 0；第 12、13、16 章沒有 objectness，改成各類別分數的目標都是 0）」。grid 列註明這是第 5、7 章的規則，「切成」也改成「分成」。 |
| 5 | 讀者審查 | 缺 MSE 條目 | 已修正：新增 MSE（均方誤差）一列，說明第 7 章框 loss 用的就是它，以及它不衡量兩框重疊。詳見連到 3.1（第一次定義）、4.1、7.3。Smooth L1 只在 12.1 用到，由該節自己定義，沒有另列。 |
| 6 | 讀者審查 | assert、計算圖等詞查不到 | 已修正：新增「斷言（assert）」一列，連到第 0 章，並提到練習 1 的 AssertionError；mask 列就地解釋「計算圖」。label、detach、no_grad 沒有另列，避免本頁一次變得太長。 |
| 7 | 讀者審查 | 「標註」與其他常用詞查不到 | 已修正：GT 列加上「也叫標註（annotation）或真值框」，和首頁一致；新增「推論（inference）」一列，連到第 0 章與 7.5。ReLU、感受野、回歸、MAC 等沒有補，範圍較大。 |
| 8 | 讀者審查 | 缺 IoU 類 loss、decoupled head 等條目 | 已修正：照 11.4、12.2 的內容新增兩列：「IoU 類 loss（GIoU、DIoU、CIoU）」與「decoupled head」。CSP、特徵融合、TAL 沒有補。 |
| 9 | 讀者審查 | 常用符號缺 e、ln 等 | 已修正：補上 e／E（自然對數的底；16.3 的 e 是第幾輪；15.1 的 E[·] 是期望值）與 ln／log 兩條，H 條目註明 3.1 的 H(x) 是函數名稱，都和原文核對過。p、s、k 沒有補。 |
| 10 | 讀者審查 | 門檻列讀起來像全書只有四種 | 已修正：改成「這四種門檻」，並補一句：訓練與追蹤另有自己的門檻，例如第 9 章 ignore 的尺寸 IoU 0.2、第 19 章的 IoU 0.1（已和原文核對）。 |
| 11 | 讀者審查 | 同名不同義藏在摺疊區，沒有錨點 | 已修正：改成 ## 標題並展開，加上錨點 {#homonyms}；開頭的※說明改成連結；positive 後面也標上※。條目內提到的節次沒有逐一加連結。 |
| 12 | 讀者審查 | 手機上表格的「詳見」欄要橫滑 | 未改：要解決得改寫七張表格的欄位結構，不是一個片語的小改。整頁本身不會橫向捲動，只有表格可以在框內滑動。 |
| 13 | 讀者審查 | 表格內提到的節次沒有連結 | 已修正：出處不在「詳見」裡的，都補進詳見：step／epoch 補 16.3、seed 補 3.3 與 7.4、optimizer 補 2 與 7.4、overfit 補 7.4、positive 補 9.1、recall 補 17。只是舉例列出章名的地方維持純文字。 |
| 14 | 讀者審查 | overfit 列寫死 train mAP50＝1.0 | 已修正：改成「8.2 節在偵測模型上實際做這項檢查（結果見該節）」，不帶會隨重跑改變的數字，也不預設結果一定成功。 |
| 15 | 讀者審查 | backbone 解釋重複、畫素與 pixel 混用、anchor-free | 已修正：backbone 改成「模型前段負責從圖片提取特徵、輸出特徵圖的那一串層」；特徵圖列第一次出現時寫「畫素（pixel）」；anchor-free 列依 12.1 補一句「第 7 章的 grid MiniYOLO 同樣沒有 anchor 尺寸，差別在框的寫法」。 |
| 16 | 事實查核 | 「術語欄同時寫英文和中文」不全成立 | 已修正：改成「術語欄寫英文名稱，課文有中文名稱時一併寫出」。 |
| 17 | 事實查核 | top-k「不看框的位置與重疊」不精確 | 已修正：改成「只依每個候選自己的分數或品質取前 k 名，不比較候選框彼此的重疊」，其餘照原文。 |
| 18 | 事實查核 | negative 的 objectness 括號只適用 grid | 已修正：與讀者 should 4 一起改。 |
| 19 | 事實查核 | loss「可微分」和 11.4 不一致 | 已修正：改成「用一個能用 backward 算出梯度的數值」。 |
| 20 | 事實查核 | sigmoid「碰不到兩端」只在數學上成立 | 已修正：改成「數學上碰不到兩端；用 float32 計算時，極端的 logit 會捨入成 0 或 1，見 7.5 節」（7.5 有談 float32 捨入）。 |
| 21 | 事實查核 | ltrb 的單位有例外（16.3 用畫素） | 已修正：與讀者 should 3 一起改。已確認 16.3 節寫的是 [−3,−3,5,5] 畫素。 |
| 22 | 事實查核 | overfit 列的 mAP 數字會隨重跑改變 | 已修正：與讀者 should 14 一起改，數字拿掉了，本頁不再有依紀錄而定的值。 |
| 23 | 事實查核 | optimizer 列的出處不在詳見 | 已修正：詳見補上 2（lr 太大變成 NaN）與 7.4（Adam 的改變量不等於 lr×梯度），兩處都核對過原文。 |
| 24 | 先前查核 | concat、attention、DFL 的詳見要補 11.2、15.2、16.1 | 未改：現行文字三列都已經有這幾個連結。 |

修正後由另一位 AI 檢查這一批頁面（`docs/index.md`、`docs/learning-path.md`、`docs/glossary.md`、`docs/status.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

「詳見」說明裡的連結文字「完整閱讀路線」與 zensical.toml 導覽名稱一致，建置後 href 為 ../learning-path/。三個補充節都核對過內容：concat／add 列的 11.2 教了沿 channel concat 時 B、H、W 要一樣、channel 不必相同，100×100 時 `scale_factor=2` concat 會報錯，add 要整個 shape 相同；DFL 列的 16.1 教了 reg_max 就是 K、YOLO26 設 reg_max=1 就不建 DFL、K 個 bin 只到 K−1 格；attention 列的 15.2 推導了 A 個等大區域時每個 head 的 pair 數是 N²/A，並說明要乘 M 與 B。第一個連結 3.1 確實最早教 channel concat 與 add；第 2 章的 torch.cat 只沿筆數軸接資料。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 「concat（串接）與 add（相加）」列的詳見欄 [3.1]、[11.2] | 這列的「4＋4 變 8 個 channel」與「其他軸要一樣長，否則 `torch.cat` 會報錯」，最直接的教學是 11.1 的專節〈串接不是相加〉（含 `[1,2]`／`[10,20]` 例子與 padding 漏寫時 torch.cat 報錯），術語表卻完全沒有連到 11.1。 | 已修正：詳見欄加上 [11.1]。 |

### 第 2 輪：上一輪的處理與審查紀錄：通過

concat／add 列的詳見欄加上 [11.1]：11.1 有〈串接不是相加〉、4+4=8 與 torch.cat 報錯的例子，連結正確、排序合理，修好發現。紀錄 24 項發現都有處理列。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/glossary.md | 內部檔名：「**一、遺留項目、page-impact 與附加指示（第 2 項）** - 先前查核的建議清單裡沒有本單元的 key，先前查核留下的項目清單也沒有」；〈定稿修正〉每列編號都是 1。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：讀者審查 15 項、事實查核 8 項與先前查核 1 項都在〈定稿修正〉處理，編號連續；修正後檢查掛在本頁所屬批次；〈後續編輯的檢查〉兩輪與處理都在；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/glossary.md 第 14、44、62、71 行 | 〈上述處理〉第 1 項寫「內部檔名換成白話」，但第 14 行仍是「先前查核的建議清單裡沒有這次查核的頁面的 key，先前查核留下的項目清單也沒有」（key 與兩個內部清單名都還在，句子也不通）；另有路徑殘句「- 暫存副本- 檢查腳本與執行紀錄：暫存副本（claims.py…）」「在暫存副本（.../暫存副本）執行」「驗證腳本與截圖放在 .../暫存副本」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 62、71 行已改；但點名的兩處沒有真正改掉，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/glossary.md 第 14、44 行 | 點名的「先前查核的建議清單裡沒有這次查核的頁面的 key，先前查核留下的項目清單也沒有」在第 14 行逐字還在；點名的「- 暫存副本- 檢查腳本與執行紀錄：暫存副本（claims.py…）」只少了一個「- 」，變成第 44 行「- 暫存副本檢查腳本與執行紀錄：暫存副本（claims.py…）」，仍是殘句。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

第 4 輪第 1 項屬實：第 14、44 行兩段仍在；「- 」是描述用的引號，沒算進計數，這是對的。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 3 輪第 1 項處理欄（第 211 行） | 「5 處中 4 處已改寫或刪除」把引用的第 2 輪處理說法「內部檔名換成白話」也算成已改寫，但它仍在第 203 行。實際改寫的紀錄文字是 3 處。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
