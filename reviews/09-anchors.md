# 審查紀錄：YOLOv2 anchor 與框參數化

審查範圍：`docs/lessons/09-anchors.md`、頁面上的圖（`docs/assets/diagrams/09-anchors.svg`），以及 `lesson_cases/09-anchors.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要問題，只有一個 should：第 99 行少寫「obj」，建議補回，見下表。所有程式都在暫存副本執行，repo 裡沒有跑任何程式。

1. 關於程式的敘述都成立。
   - lesson_cases/09-anchors.py 只在 baed4ff 改過，SHA-256 是 7a7e8c…，和紀錄的 case_sha256 相同。暫存副本實跑 exit 0，stdout 和 artifacts/checks/curriculum/09-anchors.json 逐字相同。
   - 我照頁面指示，在 `loss.backward()` 下一行、用相同縮排加上那行 print 再跑。最先印出的是 `[-0.016129031777381897, 0.0] [0.016129031777381897, 0.016129031777381897]`，接著才是原本的四行，和頁面寫的「約 [−0.0161, 0]、[0.0161, 0.0161]，最先印出」一致。索引說明也對：y=1、x=1、`:` 取兩槽、4 是 obj。
   - 用 probe 看 SGD 一步後的變化：31 個槽、35 個數值改變。positive 槽變的是 tx、ty、obj、class0、class1；30 個 negative 槽只變 obj；ignore 槽沒變；positive 槽的 tw、th 梯度是 0，也沒變。和頁面上解釋 `one slot update completed` 那段一致。
   - 斷言共 7 個（3＋4），每一項的內容和容許誤差都對得上。
   - 突變測試：把 BCE 改成 reduction='sum'，程式照樣通過，所以斷言確實沒有守 ±0.0161；讓 ignore 也參與 BCE，就出現 AssertionError。因此「斷言只核對 ignore 的 0」是真話。
   - 自主練習的參考答案（尺寸 IoU 0.5／0.125、計數 1／0／31、ln2、ln4）我用程式裡的 size_iou 核對過，都正確。舊敘述也逐條對照了 losses.py（5*box、sigmoid 後的 wh、objectness 取平均）、targets.py（同一格碰撞拋 ValueError）和 inference.py（中心解碼），都成立。
2. 條目都處理了。
   - L3 的 Colab 連結已在 463d3f5 改成 lessons-v0.4.0。
   - L97 那條原本要靠程式補 assert 才能改寫，但定稿程式沒有補。編輯保留了真實的範圍敘述，改成現在式，並加上讀者可以自己跑的檢查。程式沒有修，所以不算殘留的修補說明。
   - 受程式改動影響的段落和程式修改清單都沒有本頁的條目。整頁沒有修訂、審查或製作過程的敘述；L133 的「這次更新」指的是 SGD 那一步，不是教材修訂。
3. 數字：新增的都是確定性、可以手算的值，沒有來自這台 Mac 的計時或訓練數字。依賴紀錄的值編輯都已經列出。
4. 摘錄：摘錄比對工具對 repo 和暫存副本都印 []。簡化版程式區塊有說明它是簡化版；那行 print 不是程式原文。正文裡的「第 N 行」都指輸出的行，不是程式行號。暫存副本的 validate_lessons 一路通過摘錄檢查和 notebook 與程式一致的檢查，只在審查覆蓋那一步失敗。失敗原因是全站都還沒有審查紀錄，屬於發布前的暫時狀態，審查用的事實與寫作規範清單已註明。
5. 可讀性：`.grad`、backward、`.tolist()` 在第 0、2、4 章都教過。加 print 的作法和第 0 章是同一種，順序也講得通。
6. 建置：zensical build --clean --strict exit 0，validate_site.py exit 0。摺疊區裡的程式區塊算圖正確。SVG 有 viewBox、title、desc，xmllint 通過，qlmanage 算圖乾淨，座標和程式一致（紅框 [8,12,24,28]、8×8 疊上後是 [12,16,20,24]、中心 (16,20)）。這次查核的頁面只動了頁面；SVG、程式、notebook、紀錄都沒變。工作樹裡 09-anchor-clustering.md 也有改動，但那是另一個部分依它自己的先前審查意見做的修改。

另外一件不在這次查核的頁面範圍內的事：頁面在 evidence 標記前有 3 個換行，verify_curriculum 算圖時會刪掉其中 2 個，頁面文字的 hash 會跟著變。程式問題清單已有記錄。審查覆蓋要在算圖之後才記錄，否則 validate_lessons 會失敗。

紀錄檔

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/09-anchors.md 第 99 行（摺疊區〈手算：零 logit 時的 objectness 梯度〉）「上面三種槽的梯度都是手算的，完整程式不會印出；它的斷言（assert）只核對其中 ignore 槽的 0。」 | 原句寫的是「三種 obj 梯度中」，改寫後只剩「三種槽的梯度」，少了 obj。如果照字面讀成「這三種槽的所有梯度」，「斷言只核對其中 ignore 槽的 0」就會和下方第 3 項斷言衝突，因為第 3 項也核對了 30 個 negative 槽的 tx、ty、tw、th 梯度為 0。放在上下文裡大多讀得懂，但不如原句精確。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://arxiv.org/abs/1612.08242（全文為 https://arxiv.org/pdf/1612.08242v1）：Abstract；2. Better 的 Convolutional With Anchor Boxes、Dimension Clusters（含 Figure 2、Table 1）、Direct location prediction（b_x、b_y、b_w、b_h 公式）；3. Faster 的 Darknet-19；另全文搜尋 ignore／threshold，確認論文沒有描述 ignore 規則
- https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/src/region_layer.c 第 232–252 行（以預測框對任一 GT 的 best_iou > l.thresh 決定 ignore）、第 266–292 行（best anchor：truth_shift 與 bias_match，只比寬高）
- https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/cfg/yolov2-voc.cfg [region] 段：第 243 行 bias_match=1、第 249 行 rescore=1、第 257 行 thresh = .6

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 第 77 行附近，〈一格兩槽，不等於兩種類別〉「為什麼要 ignore？」段的最後一句：「0.2 是本節為了示範三種狀態設的教學值，不是原版設定；這個簡化的 ignore 規則不能代表原版所有訓練細節。」 | 「簡化的 ignore 規則」加上「不是原版設定」，會讓讀者以為原版規則和本節是同一種：看同格其他 anchor 的尺寸 IoU，只是門檻不同。實際上 YOLO9000 論文沒有描述任何 ignore 規則（全文的 ignore 一詞只出現在 WordTree 段）。官方 Darknet 的 region layer（src/region_layer.c 第 232–252 行）做法不同：每一格的每個槽，先把預測框連同位置解碼出來，再和圖中任一 GT 算一般 IoU；最大值超過 l.thresh，就把 objectness 的 delta 設為 0。cfg/yolov2-voc.cfg 裡的 thresh 是 .6。也就是說，原版比的是預測框，不是 anchor 尺寸；範圍是所有格，不限同一格。讀者若拿本頁去讀原版程式或自己實作 YOLOv2，會去找一條不存在的規則。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 來源對照 | 「簡化的 ignore 規則」讓人以為原版也是同格比尺寸 IoU | 已修正：把「簡化的」改成「本節自訂的，和原版不同」，並補上原版做法：YOLOv2 論文沒有說明 ignore 規則；官方 Darknet 先把每個槽的預測框解碼（含位置），再和每個 GT 算一般 IoU，最大值超過 0.6（yolov2-voc.cfg 的 thresh）就不算 objectness loss。這條規則不限同一格，比的也不是 anchor 尺寸。「不代表原版其他訓練細節」的意思保留。 |
| 2 | 先前查核 | 「三種槽的梯度」少了 obj，和第 3 項斷言看似衝突 | 已修正：改成「上面三種槽的 obj 梯度都是手算的」。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/09-anchors.md`、`docs/lessons/09-anchor-clustering.md`、`docs/lessons/10-multiscale.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/09-anchors.md 第 77 行（「為什麼要 ignore？」段）描述官方 Darknet ignore 規則的句子 | 「把每個槽解碼後的預測框……和圖中每個 GT 算一般的 IoU，最大值超過 0.6……就不算這個槽的 objectness loss」沒有排除負責該 GT 的槽。region_layer.c 會先把超過 thresh 的槽的 obj delta 設成 0，但接著會把負責 GT 的那個槽（best anchor）改寫成 object_scale×(iou − output)（yolov2-voc.cfg 設 rescore=1），所以這個槽照樣算 objectness loss。照字面讀，預測已經很準的 positive 槽也不學 objectness，和官方程式不符。10-multiscale 的同類敘述已經寫成「不是最佳、但……」，兩頁的精確度不一致。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

下載 pjreddie/darknet（master＝f6afaab）的 src/region_layer.c 與 cfg/yolov2-voc.cfg 對照。第 238–251 行對每個槽用解碼後的預測框和所有 GT 算 IoU，最大值大於 thresh 就把 objectness 的 delta 設 0；第 265–306 行再為每個 GT 的負責槽（GT 中心所在格、尺寸 IoU 最佳的 anchor）重設 objectness 的 delta，rescore=1 時目標是 IoU。cfg 的 thresh=.6、rescore=1。「範圍不限同一格、比的不是 anchor 尺寸」正確；YOLOv2 論文確實沒有寫 ignore 規則。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 〈一格兩槽〉前「為什麼要 ignore？」段：「而且不是負責該 GT 的槽，就不算這個槽的 objectness loss」 | 程式的例外條件是「不是任何 GT 的負責槽」：只要這個槽是某個 GT 的負責槽，它的 objectness delta 就會被重設，不論它和哪個 GT 的 IoU 最大。照頁面寫的「該 GT」，一個和 GT A 重疊最多、卻是 GT B 負責槽的槽會被誤判為 ignore。另外，這段官方程式說法在頁面上沒有任何來源連結。 | 已修正：改成「不是任何 GT 的負責槽」，並加上 Darknet 固定 commit（f6afaab）的 region_layer.c 第 237–306 行與 yolov2-voc.cfg 第 257 行的連結；這兩段已對照原始碼核對。 |

### 第 2 輪：上一輪的處理與審查紀錄：通過

下載 pjreddie/darknet 固定 commit f6afaab 的 src/region_layer.c 與 cfg/yolov2-voc.cfg 核對：第 236 行用 get_region_box 解碼含位置的預測框，第 237–252 行對圖中每個 GT 算 box_iou、best_iou > l.thresh 就把 objectness delta 設 0；第 265–306 行在 GT 中心所在格以尺寸 IoU（bias_match）選負責槽並重設 objectness delta（rescore）；cfg 第 257 行 thresh = .6。句子屬實，修好「不是任何 GT 的負責槽」與缺連結的發現；兩個連結回 200，間距合規。全文讀了本頁審查紀錄。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | region_layer.c 連結 #L237-L306 | 標示範圍從第 237 行開始，句子說的「解碼後的預測框」在第 236 行（box pred = get_region_box(...)），不在標示內。 | 已修正：改成 `#L236-L306`。 |
| 2 | 建議 | 「為什麼要 ignore？」段的 Darknet 句 | 一句連接六個條件，主詞從程式換到槽；「負責槽」在本頁沒有定義（本頁用 positive、「負責學這個物件」）。 | 已修正：拆成兩句，並說明負責學某個 GT 的槽相當於本節的 positive。 |
| 3 | 建議 | reviews/09-anchors.md | 內部用語與截斷：「impact.json 和程式修改清單都沒有本頁的條目」「L97 那條原本要靠程式補 assert 才能改寫」「摘錄比對工具對 repo 和暫存副本都印 []」「審查用的事實與寫作規範清單已註明」（路徑清理留下的空格）、結尾「`.tol…」；〈定稿修正〉兩列編號都是 1；〈留下的意見〉沒有處理欄（後續編輯的檢查只間接顯示已改）。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

下載 pjreddie/darknet f6afaab 的 src/region_layer.c 與 cfg/yolov2-voc.cfg，以及 YOLO9000 v1 全文核對「為什麼要 ignore？」段的 Darknet 兩句。第 236 行 get_region_box 解碼含位置的預測框；第 238–251 行逐一和圖中 GT 算 box_iou，best_iou > l.thresh 時把 objectness delta 設 0；第 265–306 行在 GT 中心所在格，以 bias_match 尺寸 IoU 選負責槽，重設 objectness delta（rescore=1 時目標是 IoU）；cfg 第 257 行 thresh = .6；論文全文的 ignore／threshold 只出現在 WordTree 段。連結範圍 #L236-L306、#L257 正確。拆成兩句並說明「相當於本節的 positive」，回應了上一輪兩項建議，讀來通順；GT 有術語表定義；網站建置通過。紀錄：〈定稿修正〉兩列與〈留下的意見〉兩列的編號連續，也都有處理欄。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/09-anchors.md 第 11、22、23、29、31 行 | 〈上述處理〉第 3 項寫「已修正：內部用語換成白話」，但它點名的「L97 那條原本要靠程式補 assert 才能改寫」「摘錄比對工具對 repo 和暫存副本都印 []」仍在；另有「紀錄檔：暫存副本、暫存副本、暫存副本09-anchors-print.py、暫存副本09-anchors.svg.png」「code-issues.md 已有記錄」（不在 repo 的內部檔）、「受程式改動影響的段落和查核範圍清單都沒有本頁的條目」「沒有 must 問題，只有一個 should」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 2 | 建議 | reviews/09-anchors.md〈定稿修正〉〈留下的意見〉第 1 列 | 這列是 docs/lessons/10-multiscale.md 第 9 行「見下一節」的意見，因為意見文字提到 09-anchors 才被掛到本頁。處理寫「這項修正由下方〈後續編輯的檢查〉核對」，但本紀錄的〈後續編輯的檢查〉只核對 Darknet 句，那項的核對在 reviews/10-multiscale.md。 | 已修正：產生器改以位置欄最先提到的頁面歸屬意見，這一列已移回 10-multiscale 的紀錄。 |

### 第 4 輪：上一輪的處理：有必要問題

〈留下的意見〉只剩本頁 Darknet 句一列，10-multiscale「見下一節」那列已移走，沒有別頁的遺留意見；第 3 輪第 2 項屬實。第 3 輪第 1 項中，「查核範圍清單」「程式問題清單」確實換掉了，但其餘點名的殘句沒改，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/09-anchors.md 第 22、25 行，對應第 3 輪第 1 項處理 | 處理寫「點名的路徑殘句與內部用語已清理或換成白話」，但點名的「L97 那條原本要靠程式補 assert 才能改寫」（第 22 行）與「摘錄比對工具對 repo 和暫存副本都印 []」（第 25 行）逐字還在。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 2 | 建議 | reviews/09-anchors.md 第 11、31 行 | 第 31 行只剩孤立的「紀錄檔」；第 11 行仍寫著「只有一個 should」。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

第 3 輪第 2 項屬實：〈留下的意見〉只剩 Darknet 句一列，10-multiscale 那一列已移到該頁紀錄第 229 行。第 4 輪第 1 項屬實：第 22、25 行兩段仍在，舊的處理說法已由產生器重寫。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 3 輪第 1 項處理欄（第 96 行） | 「7 處中 5 處已改寫或刪除」把發現引用的第 2 輪處理說法「已修正：內部用語換成白話」也算成已改寫，但它仍在第 2 輪第 3 項處理欄（第 88 行）。實際改寫的紀錄文字是 4 處：紀錄檔清單、code-issues.md、查核範圍清單、「沒有 must 問題」。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
| 2 | 建議 | 第 4 輪第 2 項處理欄（第 106 行） | 發現點名兩處：第 31 行孤立的「紀錄檔」與第 11 行的「只有一個 should」，兩處都還在。處理欄只寫「點名的 1 處文字仍在紀錄裡：「只有一個 should」」，漏了「紀錄檔」，原因是產生器略過 4 個字以下的引文。「未改」這個判斷本身正確。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
