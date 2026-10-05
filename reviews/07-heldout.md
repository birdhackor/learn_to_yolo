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

## 紀錄重產後的檢查

2026-10-05，由另一位 AI 在獨立副本重審 `docs/lessons/07-heldout.md` 全頁、`lesson_cases/07-heldout.py` 及其直接／間接 import 的 repo 模組、`grid-learning-curve.svg` 與 `grid-learning-predictions.svg`。讀過 AGENTS.md、發布流程第 7 節、舊審查、現行逐節紀錄與 160 步紀錄；本次結論是通過，沒有必要問題，有一項可選改善。頁尾自動執行紀錄及 Colab tag 不計入審查涵蓋 hash，但另對照現行 JSON／notebook 檢查。

### 執行、練習與獨立計算

本次使用 Python 3.12.14、PyTorch 2.9.1+cpu、INTEL(R) XEON(R) PLATINUM 8573C、2 個執行緒。下列工作都在副本進行：

```bash
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python lesson_cases/07-heldout.py
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python -m miniyolo.train --steps 160 --samples 32 --device cpu --output artifacts/runs/review-07-heldout-grid --report artifacts/runs/review-07-heldout-grid/report.json
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python review07-checks.py
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python scripts/render_learning_evidence.py --report artifacts/checks/grid-learning.json --output artifacts/runs/review-07-heldout-grid/rendered
/workspace/learn_to_yolo/.venv-docs/bin/zensical build --clean --strict
/workspace/learn_to_yolo/.venv-docs/bin/python scripts/validate_site.py
/workspace/learn_to_yolo/.venv-docs/bin/python review07-browser.py
/workspace/learn_to_yolo/.venv-docs/bin/python review07-sources.py
```

1. 案例 exit 0，兩行 stdout 與現行 `artifacts/checks/curriculum/07-heldout.json` 逐字相同。fixture 得到 AP50=0.5、precision=1/3、recall=0.5；藍類沒有 GT，因此 AP=None。三步模型的 AP、precision、recall 都為 0，與正文的「管線冒煙測試」定位一致。
2. 從 Zensical 實際 HTML 的參考答案 code block 取出保留 4 格縮排的練習，插入 `print('artificial evaluation fixture', metrics)` 下一行，在副本產生的練習檔執行。exit 0、斷言通過，stdout 仍與原案例相同。
3. 另外用普通 Python list 計算半開 xyxy 面積／IoU，跨圖逐類按 score 排序，排除已配對 GT，再以 Fraction 精確計算 PR 包絡面積；沒有呼叫 `miniyolo.metrics` 或 renderer 的 matching 函式。原 fixture 得到 AP=.5、precision=1/3、recall=.5；刪除兩個低分 FP 得到 AP=.5、precision=1、recall=.5；將背景 FP 改為 .95 得到 AP=.25。逐圖 AP 為 A=1、B=0，平均 .5，因此正文說此 fixture 無法展示逐圖平均與全體 AP 的差異正確。
4. 三步 train seed=7 與 held-out seed=901 的全部 4×4 圖片配對都不相同。確認 ShapeDataset 每索引以 seed+1009×index 產生圖片，該批 held-out 不進 optimizer；不是把人工預測冒充模型輸出。
5. 獨立重跑 160 步後的全部 loss_history（160×4 個 loss）與四張 validation_examples 的 GT／預測逐值等於現行 `grid-learning.json`。以重跑 checkpoint 對全部 validation／test 圖取得實際框，另用上述獨立計算重算；結果如下。

| 資料 | 每類 AP50：紅／藍 | mAP50 | GT／預測框 | TP／FP／FN | precision／recall |
|---|---|---|---|---|---|
| validation | 6/7／3/4 | 0.8035714285714286 | 18／16 | 15／1／3 | 15/16=0.9375／15/18=0.8333333333333334 |
| test | 6/7／160/231 | 0.7748917748917749 | 19／17 | 15／2／4 | 15/17=0.8823529411764706／15/19=0.7894736842105263 |

更新前 validation mAP50=0.0017507002801120447；因此正文的 .0018、.8036、.7749、.8824／.7895、每類 test AP .857／.693 都正確。validation 唯一 FP 的 trace 是 image=0、pred_index=1。test 多找到一個物件會差 1/19≈.0526 recall，也支持正文對小樣本波動的提醒。

6. 逐句對照 `miniyolo.train`：固定 seed 7／700／7000、train 32 張、validation／test 各 16 張、64×64、每張 0～2 個物件且包含空圖；只在 train 圖更新權重。訓練前／後各評一次 validation，末端評一次 test，沒有按 validation 自動選設定，也沒有按 test 結果再更新模型。查核另外載入同一 checkpoint 重算，沒有以查核結果改設定。
7. 候選門檻、同類 NMS、matching IoU 的區分成立；三步案例 score=.01、160 步 score=.05，兩者的 NMS／matching 都為 .5。確認無效的 x2<x1、類別超出範圍都被 evaluate_ap 拒絕。摘錄專用檢查對本頁回傳空清單；notebook 最後一格等於案例、已存輸出等於現行 JSON，source_ref=lessons-v0.4.0。

### 實際 SVG 與初學者閱讀

- 依現行紀錄重畫兩張 SVG，與副本追蹤圖逐 byte 相同。另直接解析實際 SVG，而非只看 renderer：曲線的四條 polyline 每條 160 個座標，都對上當前 loss_history；total=5×box+objectness+classification 的加權關係全部成立。
- predictions SVG 的四張 embedded PNG 與 seed=700 重生的圖片逐 pixel 相同；實際綠虛線 GT 與橙實線預測框的 x／y／width／height，皆與 JSON 按 5 倍比例映射的座標相同（SVG 的小數取整容差 .005）。
- 圖片 0 藍框 #0 IoU=0.6238343188704405，TP；紅框 #1 score=0.9804154634475708、IoU=0.4734199231501358，FP；紅色 GT 為 FN。實際 SVG 的文字也依序為 `TP IoU 0.62`、`#1 class 0 score 0.980`、`FP IoU 0.47` 與 `FN`，與新正文完全一致。validation 唯一 FP 與 precision=15/16 的說明成立。
- Zensical strict build 與 validate_site.py 通過；本頁渲染為 3 張表格、3 個摺疊區、2 個 Python 程式塊，練習縮排保留。Chromium 151.0.7922.173 使用 `/usr/bin/chromium` 與 `--no-sandbox --disable-gpu --disable-dev-shm-usage`，實際查看桌面頁、兩張原 SVG 與 390px 手機頁。沒有 JavaScript 錯誤，SVG 所有文字在 viewBox 內，沒有標籤重疊或裁切，手機沒有橫向頁面溢出。只啟動副本的暫時 HTTP server，檢查後已停止。
- fixture、GT、TP／FP／FN、seed、held-out、validation／test、checkpoint、評估協議、regression test 的解釋符合高中程度、數學好但程式新手的閱讀順序。尤其保留「FP 降 precision 不一定降 AP」「score 高不代表 IoU 高」「三步可執行不是學習成功」三個關鍵區分。合成 seed 的獨立性只支持同生成規則的結論；本文沒有外推到照片或主張現代機制優劣。

| # | 程度 | 位置 | 發現與處理 |
|---|---|---|---|
| 1 | 建議 | 補充實驗圖板／手機讀圖 | 390px 手機版會等比縮小整張 900px SVG，預測標籤約 6px，直接讀圖上的兩行標籤較吃力；正文已有完整讀圖解說，數字與圖均正確，因此不是發布必要修正。可選擇讓圖連到原 SVG，提供開啟原尺寸的入口，或另做窄畫面圖板。建議已記下，本次沒有修改。 |

### 官方來源核對

查核重新開啟以下來源，全部 HTTP 200，保留 TLS 驗證；不是沿用舊審查結論或靠記憶判定。

- [VOC2012 官方文件](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html#SECTION00044100000000000000)，3.4.1：右側最大 precision 包絡、2010 以前 11 個 recall 取樣位置、VOC2010–2012 全部唯一 recall 的積分。
- [VOCdevkit_18-May-2011.tar](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar)：`VOCcode/VOCevaldet.m` 先在全部同圖同類 GT 求 ovmax／jmax，再查 diff／det；`VOCcode/VOCap.m` 補 recall 0／1 端點、反向 precision 包絡、在 recall 跳動處積分。這支持本頁說的 VOC 與本書配對順序差異。
- [COCO 官方評估程式固定 commit 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py)：_prepare L106–109 的 crowd ignore、computeIoU L163–176 逐圖逐類截 maxDets、evaluateImg L251–294 跳過已配對非 crowd GT 再找最佳配對、accumulate L372–405 的 101 recall 點包絡取樣、summarize L452–455 的有效項平均、Params L506–508 的 10 IoU／101 recall／100 maxDet。
- [COCO 官方說明固定 commit 5e1c4da72464b1c6f068df0c02c91e3000ea62c4](https://github.com/cocodataset/cocodataset.github.io/blob/5e1c4da72464b1c6f068df0c02c91e3000ea62c4/dataset/detection-eval.htm)：10 個 IoU 門檻與各類別平均、R=101。網頁文字第 6 點寫「跨所有類別最多 100」；本頁依實際固定程式說「每張圖每個類別最多 100」，此處的程式核對成立。它也足以確認將本書單 IoU all-points AP 多跑 10 次，仍不等於 COCO AP@[.50:.95]。
- [PyTorch v2.9.1 Module 原始碼](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/module.py#L2894)：eval 切成 train(False)，只影響特定在訓練／評估模式行為不同的層。
- [PyTorch v2.9.1 grad_mode 原始碼](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/autograd/grad_mode.py#L212)：inference_mode 與 no_grad 同屬推論梯度控制，會另外關閉 view tracking／version counter，產生的 tensor 不可參與 autograd 記錄的運算。支持本頁摘錄中文註解。

本頁明確使用「單 IoU=.5、每類 all-points、只平均有 GT 類別」的簡化指標，且介紹 VOC／COCO 差異；沒有把受控紅／藍矩形任務、人工 fixture 或三步模型當作官方 benchmark 成績。

### 查核版本的 SHA-256

以下頁面 hash 是含自動執行區塊的整份檔案 hash；正式 review coverage 另依工具的排除規則計算。本次只核對 hashes，沒有寫 coverage。

| 檔案 | SHA-256 |
|---|---|
| `miniyolo/__init__.py` | `785b058b2b011124243b06ee59df8e59dfa75b77f050b897479e5be636763fc9` |
| `miniyolo/checkpoint.py` | `2144e2afa4d89382a7b3cd253755bb851e35fdc0847ee7179dee02dbbf4ca49b` |
| `miniyolo/data.py` | `cccad00e2c4f96eb6567eafc9e12248379c6b715fc1790d75518a253baa6181d` |
| `miniyolo/geometry.py` | `6a6b57d3493888e99dae4a012dab78127b8b543a0e01d8107963a3d6e63d8483` |
| `miniyolo/inference.py` | `995ac8f942d0c1e43d94f2efb3b7adcb91a9feb6b9bab923615209f0c190c313` |
| `miniyolo/losses.py` | `81fa9331c9a2aebe2e9c6c453a5313e64566bb20c3fe77ca45ec9df53424d4e4` |
| `miniyolo/metrics.py` | `53ce982e37cbd96c784f75c7d30faf99d52f79ab83ca7b8114eb21b4327330e0` |
| `miniyolo/models.py` | `49d029ea4ba2650ce8933cf97e3d25dc7aff2ca4e972eec19cdda17b0f4900e6` |
| `miniyolo/provenance.py` | `21769813c36b45f513c9f0d6125194f3baa5ae265278ffd44a796d298d124ed3` |
| `miniyolo/targets.py` | `2c8e32f2845b3bf970c77304c5cca0f999083d36a4d5af2f75f14aa89b823f16` |
| `miniyolo/train.py` | `aa5567f11be6ca7aee97bd082b9d5964414b12b73453763c915c7ffb409c4276` |
| `lesson_cases/07-heldout.py` | `ada566e4f1c294156116f7771c2bf31dfc7d23770a7e76a56331a595cf888568` |
| `docs/lessons/07-heldout.md` | `f27210db980f5cba76ef3bdd7047f660fb0bc352b630267429796b5e14ec1be4` |
| `artifacts/checks/curriculum/07-heldout.json` | `38186fc33a15af761c5e4bc5baf00a047cf4009afbb4b46967eb66add6c5c332` |
| `artifacts/checks/grid-learning.json` | `c66887b113691433f4f8787cb55fbbb6f4dacf26d635565dd259b5988417a298` |
| `docs/assets/diagrams/grid-learning-curve.svg` | `818baea73d025d34b16eb67e16947d46b7615f49b6d4618f3c3f0567c120682d` |
| `docs/assets/diagrams/grid-learning-predictions.svg` | `8eb7ba62adfa0df235a3530982ed6c73cdec7ced2ae47fcceabb1eac8b2b0bd4` |
| `scripts/render_learning_evidence.py` | `89d764abed5b7e5e6fd998abbe1f324ed1eee195ffc0fe6a7694f1b799cee952` |

### 實際開啟來源的 SHA-256

| 來源 | SHA-256 |
|---|---|
| `voc2012.html` | `fb5900c42542afda9aca2543f52554af414ceeda9326480202b79ca695cbc90c` |
| `VOCdevkit_18-May-2011.tar` | `6101e33483e1f252821085f4b85634d334c1d44a0a5bc3921cd64320a40bd2cf` |
| `VOCdevkit/VOCcode/VOCap.m` | `03e77db8836fe9a0b680f623de9f0f3047745581fa0193e5f90b09ae5ba82da3` |
| `VOCdevkit/VOCcode/VOCevaldet.m` | `c98716fd7f256af142d4362f3dc1d55f39339befa0c88ecf3aa83f786843bd78` |
| `cocoeval.py` | `e514af401848d5a4cc5d3512bd93d2c317f92272c6d30f218e0f00b7b4110534` |
| `coco-detection-eval.htm` | `2e434ed6c2b5eebeef6cefc225f2e32fa2478a25d276679d7a37e975064ab579` |
| `torch-2.9.1-module.py` | `d076e98c7037d342c82662c13a11dbb2584d080279a140431243475f27056dfd` |
| `torch-2.9.1-grad_mode.py` | `84d64257ccd56bd1af2994d84e34f4f4b04e11a90b8b82edccfe27ce210e7318` |

### 編輯處理

保留手機圖板的閱讀建議。本頁正文已逐項寫出圖上標籤、配對與 IoU 的數字，這次不另改圖板版型；圖與文字的數值檢查通過。沒有因此修改本頁正文。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[grid當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/grid.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/grid.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/grid.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。
