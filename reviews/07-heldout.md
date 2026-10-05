# 審查紀錄：獨立資料評估

審查範圍：`docs/lessons/07-heldout.md`、頁面上的圖（`docs/assets/diagrams/grid-learning-curve.svg`、`docs/assets/diagrams/grid-learning-predictions.svg`），以及 `lesson_cases/07-heldout.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要，也沒有建議。所有驗證都在暫存副本做，沒有在 repo 內執行任何程式。

1. 程式敘述（必查）
- 課程程式 exit 0，兩行輸出與 artifacts/checks/curriculum/07-heldout.json 的 stdout 逐字相同。
- 練習照頁面做：從建好的 HTML 取出參考答案那格（已帶 4 格縮排），貼在 print('artificial evaluation fixture', metrics) 下一行後執行。exit 0，沒有 AssertionError，輸出和原本逐字相同，與頁面說的一致。
- 自己重算：0.95 變體的 AP 是 0.25；逐圖算 AP 時 A=1、B=0，平均 0.5；seed 700 的 16 張圖共 18 個物件、seed 7000 共 19 個，每張 0～2 個（含空圖）；seed 7 和 901 畫出的圖互不相同。
- 頁面寫的門檻與預設值都和程式一致：score≥0.05、同類 NMS、配對時先排除已配對的 GT、evaluate_ap 遇到 x2<x1 或類別超出範圍會報錯、train.py 在訓練前後各評一次 validation、test 只評一次、引用的第 6 章練習。
- 圖的讀法逐句對照 scripts/render_learning_evidence.py 的 predictions_svg() 與 verdicts()，全部成立：每張圖上方寫「圖片 k：n 個真值、m 個預測」；框旁兩行標籤「#j class c score s」與「TP／FP IoU x.xx」；沒有可配對的同類真值時寫「FP 沒有可配對的同類真值」；沒被配對的 GT 標 FN。

2. 先前審查意見與受程式改動影響的段落
- L3、L78 的 Colab 連結已是 lessons-v0.4.0，與 section-map 和 notebook 的 metadata 一致。
- L85 替頁面辯護的那句已刪。
- L144 的「補充過的」已改寫。07-training.md 在 160 步補充之後還有「本節證明了什麼、還沒證明什麼」一節，所以不寫「頁末」的理由成立。
- 受程式改動影響的段落的 167、169 兩行都已改成新版面的寫法。
- 全頁沒有修訂或作者經過的敘述，也沒有替已修好的缺陷留下的補丁說明。169 行仍保留圖片 0 的 FP／FN 解說：這段講「分數高不代表位置準」，並推出 precision=15/16，是本節教學的主體，不是補丁。

3. 數字
- 表格與條列的數字都和 artifacts/checks/grid-learning.json 相符：0.0018、0.8036、0.7749、0.8824／0.7895、每類 AP 0.857／0.693、15/17、15/19、0.9375=15/16。
- 圖片 0 的數字由目前追蹤的紀錄算出：#0 的 IoU 0.6238、#1 的 IoU 0.4735、score 0.98042（顯示 0.980）。
- 沒有用到這台 Mac 的數字。Mac 重跑會得到 score 0.98100，顯示成 0.981；頁面用的是錄製機（AMD EPYC）的 0.980。
- 隨重錄可能改變的數字，編輯都已列出。

4. 程式摘錄
- 摘錄比對工具對 repo 和暫存副本都印出 []。
- 變異測試：把摘錄改成 score_threshold=.02，檢查抓得到；拿掉 data-excerpt 標記，會被報 copied verbatim。
- validate_lessons.py 的摘錄斷言 42 頁全部通過；它最後停在審查涵蓋檢查（reviews 要等發布時重審，預期如此）。
- 正文沒有引用程式行號。

5. 可讀性：新的讀圖段落順序清楚，術語都已定義，初學讀者不會被誤導。

6. 建置與 SVG
- zensical build --clean --strict 與 validate_site.py 都通過。摘錄區塊渲染成 language-python。
- 頁面只改了 5 行增、5 行刪。執行紀錄區塊的 md5 與 HEAD 相同；SVG、紀錄、notebook、reviews 都沒被動到。
- 用目前追蹤的紀錄呼叫 predictions_svg() 畫出新圖：viewBox 是 900×958，有 title 和 desc。用 qlmanage 看過，標籤互不重疊，內容和頁面描述一致。

工作樹的暫時狀態（不算問題）：追蹤中的 docs/assets/diagrams/grid-learning-predictions.svg 仍是舊版面（900×725、右側寫「pred k」），所以此刻頁面文字和圖對不上。scripts/record_evidence.py 重錄 grid-learning.json 時會先訓練、再跑 render_learning_evidence.py，另有 grid_figures_differ() 比對渲染器輸出與現有圖檔，因此發布時圖會換成頁面描述的新版面。依審查用的事實與寫作規範清單，頁面不應改寫成描述這個暫時狀態。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html — 3.4.1 Average Precision (AP)（錨點 SECTION00044100000000000000，含「prior to 2010 … 0,0.1,…,1」「VOC2010-2012 … all unique recall values」）、4.4 Evaluation（錨點 SECTION00054000000000000000，含 multiple detections 與 difficult 的說明）、2.5 Ground Truth Annotation 與 10.x 的 difficult 說明
- http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar — VOCcode/VOCevaldet.m（ovmax／jmax 迴圈、diff／det 判斷、npos 計算、+1 像素 IoU）、VOCcode/VOCap.m（補 (0,0)／(1,0)、右側最大值包絡、在 recall 變化處累加面積）
- https://cocodataset.org/#detection-eval（頁面原始檔：https://github.com/cocodataset/cocodataset.github.io commit 5e1c4da72464b1c6f068df0c02c91e3000ea62c4, dataset/detection-eval.htm）— 2. Metrics 註 1（10 個 IoU 門檻 .50:.05:.95）、註 2（不區分 AP 與 mAP）、註 6（每張圖跨所有類別最多 100 個偵測）、評估參數 iouThrs／recThrs（R=101）／maxDets
- https://github.com/cocodataset/cocoapi commit 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9, PythonAPI/pycocotools/cocoeval.py — L108-L109（crowd→ignore）、L163-L176（computeIoU 逐圖逐類截斷 maxDets）、L251-L294（evaluateImg：GT 依 ignore 排序、dt[0:maxDet]、跳過已配對非 crowd GT 後取 IoU 最高者、dtIg=gtIg[m]）、L372-L405（npig 排除 ignore GT、右往左取最大值的包絡、searchsorted 讀 101 個 recall 點）、L452-L455（只平均 >-1 的項目）、L506-L507（iouThrs、recThrs 定義）
- https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html — 函式說明（移除與較高分框 IoU > iou_threshold 的框；回傳索引依分數遞減排序）
- https://pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html — 函式說明（idxs 為每個框的類別索引；不同類別之間不做 NMS；回傳索引依分數遞減排序）

這一頁沒有發現與來源不符的說法。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 來源對照 | VOC／COCO 配對順序、difficult 計數、crowd 當 ignore、每類 100 個上限，讀者照頁面連結查不到出處 | 已修正：已下載官方 VOCdevkit_18-May-2011.tar，也取得固定 commit 的 cocoeval.py，逐項核對：VOCevaldet.m 先跑完全部 GT 求 ovmax／jmax，再看 diff、det，npos 只算非 difficult 的物件；evaluateImg 依類別用 dt[0:maxDet] 截斷，配對時用 gtm 跳過已配對且不是 crowd 的 GT；_prepare 把 crowd 設成 ignore。頁面改了兩處：第一處，「原始規則可讀…」改成「規則文字見…；配對順序、difficult 怎麼計數、候選數上限這類細節，以官方評分程式為準」；第二處，摺疊區補一段出處，附 devkit tar 的連結與 cocoeval.py 固定 commit 的 #L235-L296、#L106-L109。行號範圍沒照建議用 L251-L294，因為那一段沒有含到配對的寫入（dtm、gtm）。07-heldout 的「配對順序」條目只補一個短語，指出出處在 06-evaluation。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/06-decode-nms.md`、`docs/lessons/06-evaluation.md`、`docs/lessons/07-heldout.md`、`docs/lessons/08-own-images.md`、`docs/lessons/08-own-data.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：通過

讀了 reviews/07-heldout.md：獨立查核、來源對照、修正後檢查齊全。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/07-heldout.md | 06-evaluation 那項來源修正也在本頁「配對順序」條目補了一個短語，但本頁紀錄沒有記這次改動，只在別批的檢查摘要裡出現「07-heldout：補的短語和 06-evaluation 摺疊區的內容一致」。另有「2. trace 與 impact - L3、L78」。 | 已修正：同一項處理也列進本頁〈定稿修正〉。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：06-evaluation 那項來源修正也連帶改了本頁，已列進本頁〈定稿修正〉（上一輪的建議已處理）；〈來源對照〉列出來源；〈後續編輯的檢查〉齊全；結構檢查通過。沒有發現問題。

### 第 4 輪：上一輪的處理：通過

第 3 輪沒有發現，沒有處理說明需要核對。
