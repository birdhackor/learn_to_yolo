# 審查紀錄：完整圖片推論

審查範圍：`docs/lessons/07-inference.md`、頁面上的圖（`docs/assets/diagrams/07-data.svg`、`docs/assets/diagrams/object-journey.svg`），以及 `lesson_cases/07-inference.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要問題，只有 2 個用詞上的 should（見下表）。所有程式都在暫存副本執行。紀錄和腳本放在同層的暫存副本（exercise.py/.out、probe.py/.out、build.log、validate_site.log、validate_lessons.log、ql/*.png、mut/）。repo 內只照指示跑了摘錄比對工具。

1) 程式敘述：全頁對照 lesson_cases/07-inference.py，以及 miniyolo 的 inference、geometry、targets、models、data，全部成立。
- 執行結果：本節在暫存副本跑 exit 0、stderr 為空，stdout 和 artifacts/checks/curriculum/07-inference.json 逐字相同，case_sha256 也一致。
- 新增敘述逐條核對都成立：
  - one_image 斷言：藍色通道總和 0、紅色通道總和 256。
  - collate 把圖疊起來，標註原樣放進 list。
  - smoke 兩個斷言檢查的內容；.eval() 回傳模型本身。
  - 未訓練模型 16 格全被 0.25 濾掉：實測 score 0.0602–0.0615，obj logit −1.997～−1.982。
  - 第 0 張紅、藍兩框的 score 在 float32 下完全相等（都是 0.9999545812606812），程式按類別分開比對。
  - 同類去重：斷言寫在 print 之前，印出的 2、1 是寫死在 print 裡的數字。
  - 7.4 的預測圖：框來自 _evaluate→decode_grid，門檻是 config.score_threshold=0.05，框是橙色 #fb923c。現行紀錄（舊圖）和新渲染器都是這樣。
- 全頁其餘數字重算都對：
  - clamp 例 [0,0,24,24]。
  - float32 在 24 附近的間距 1.9×10⁻⁶；sigmoid(−18)≈1.52×10⁻⁸，這個框確實被丟掉。
  - sigmoid(−89) 在 float32 是 0（−88.7 還不是 0）。
  - IoU 0.99780；鄰格框 [7.984,12,23.984,28]。
  - logit(0) 引發 ValueError；對空結果取 [0] 引發 IndexError。
  - NMS 只刪 IoU 大於門檻的框；評估以 IoU≥0.5 為命中。
- 練習照頁面步驟做：先依序跑 notebook 程式格（環境格改成 chdir＋sys.path），再開新格貼 score_raw，最後貼參考答案。兩個斷言都通過：門檻 0.70 留 1 框 [16,12,32,28]、score 0.72（誤差 3.1×10⁻⁸），門檻 0.75 是 0 框。nms_iou 改成 0.9 或 1.0 仍是 0 框，和答案一致。

2) 待處理項目：先前審查意見 L3、L111 的 Colab 連結都已是 lessons-v0.4.0，和 section-map、notebook metadata 一致。受程式改動影響的段落沒有本頁和兩張 SVG 的條目。全頁沒有製作或修訂經過的敘述。

3) 數字：新增的數字（256、0.05、0.25）都是固定值，沒有從這台 Mac 帶進會隨機器改變的數字。要等新紀錄重產才能確認的值，editor 已列出。

4) 摘錄：
- 摘錄比對工具對 repo 和暫存副本都印出 []。
- 兩段摘錄和程式第 21–28、29–36 行逐行連續一致，只多了中文註解。
- 兩段我各改壞一行，check 都有抓到。
- 另外兩段 python 是讀者自己輸入的練習程式，不在任何 repo 檔。
- 正文沒有寫程式行號。
- 暫存副本裡的 validate_lessons：摘錄檢查 42 頁全過，只卡在預期中的審查覆蓋檢查。

5) 渲染：
- zensical build --clean --strict 和 validate_site.py 都 exit 0。兩段 data-excerpt 區塊都正常轉成 Python 高亮，包括摺疊區裡那段。
- 07-data.svg、object-journey.svg 用 qlmanage 看過，畫面正常，都有 viewBox、title（第一個子元素）和 desc。
- 頁面對兩張圖的描述都成立：07-data.svg 有淡格線和紅框中心，沒有負責格和解碼結果；object-journey.svg 只畫紅框，淺藍虛線是負責格，「第 2 列、第 2 欄」就是 gy=1、gx=1。
- 07-data.svg 在工作樹中只改了 desc 的一個括號。那是 07-data 的清單項目，不是本頁 editor 改的。

6) 範圍：本頁相關的 notebook、lesson_cases、紀錄和 object-journey.svg 都沒有變動，只有頁面本身改了。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/07-inference.md 第一段摘錄的註解（約第 79 行）：「# 剛建立、未訓練的模型；.eval() 切到推論模式，並回傳模型本身」 | 註解說 `.eval()`「切到推論模式」，下一行緊接著就是 `torch.inference_mode()`，字面上也是「推論模式」。初學者讀到這裡，容易以為兩者是同一件事，或以為 `.eval()` 已經關掉計算圖記錄。要讀到後面「`eval()` 和 `inference_mode()` 用途不同」那段才會被更正。第 0 章對 eval() 的說法是「評估／推論模式」，本行只留「推論模式」，剛好和下一行撞名。 |
| 2 | 建議 | docs/lessons/07-inference.md 約第 132 行：「那裡的 score 門檻是評估用的 0.05，比本節的顯示門檻 0.25 低。」 | 這句的內容正確：config.score_threshold=0.05，_evaluate 把它傳給 decode_grid，渲染器也用它。但術語表、6.2 節的門檻表、8.1 節和第 10 章都把 0.05 叫「候選截斷門檻」（評估用），這裡只寫「評估用的 0.05」，沒有用術語表的名稱。審查用的事實與寫作規範清單要求術語和術語表一致。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://arxiv.org/abs/1506.02640（YOLOv1，以 ar5iv HTML https://ar5iv.labs.arxiv.org/html/1506.02640 讀全文，即 arXiv 最新版 v5）：§2 Unified Detection（中心所在格負責；每格 B 個框與 confidence=Pr(Object)*IOU；沒有物件時 confidence 為 0、有物件時等於 IOU；每格一組條件類別機率；Pascal VOC 的設定 S=7、B=2、7×7×30）、式 (1)（測試時把類別條件機率乘上 confidence）、§2.1（最後一層用線性輸出）、§2.2 Training（xy 以格為基準、wh 以整張圖正規化，都落在 0 到 1；sum-squared error；λcoord=5、λnoobj=.5 的理由；預測寬高的平方根；由當下 IOU 最高的預測器負責）
- https://arxiv.org/abs/2004.10934（YOLOv4，以 ar5iv HTML https://ar5iv.labs.arxiv.org/html/2004.10934 讀全文）：§3.4〈Eliminate grid sensitivity〉，sigmoid 乘上大於 1 的係數（07-targets 提到這件事，但沒有附連結）
- https://pytorch.org/docs/stable/generated/torch.optim.Adam.html（轉址到 https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html）：演算法框（m_t、v_t、偏差修正、θ_t 更新式）與 betas 參數說明「running averages of gradient and its square」，預設值 (0.9, 0.999)
- https://github.com/cocodataset/cocoapi 的 commit 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9，PythonAPI/pycocotools/cocoeval.py：第 506–508 行（iouThrs 是 .50:.05:.95 共 10 個門檻，recThrs 是 101 點）、第 370–376 行（沒有真值的類別直接跳過，precision 留在 -1）、第 396–410 行（先取 precision 包絡線，再在 101 個 recall 位置取值）、第 452–455 行（取平均時排除 -1）
- 本機 PyTorch 2.9.1（.venv-model）實測函式庫行為：cross_entropy 的 target 為 -1 時報 IndexError；類別放在最後一軸時報 RuntimeError（shape 不合）；mse_loss 與 cross_entropy 對空 tensor 取平均得到 NaN；torch.stack 的錯誤訊息開頭；對常數 0 呼叫 backward 會報錯；float32 的 sigmoid(-18)≈1.52e-8，sigmoid(-89)=0；24 附近的 ulp 是 1.9e-6，所以 24±w/2 會捨入成同一個數；inference_mode 產生的 tensor 不能拿去算梯度；squeeze 會刪掉所有長度為 1 的軸

這一頁沒有發現與來源不符的說法。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 摘錄註解寫 `.eval()`「切到推論模式」，和下一行的 `inference_mode()` 撞名 | 已修正：查證屬實。註解改成「.eval() 切到評估（eval）模式，並回傳模型本身」，和第 0 章對 eval 模式的叫法一致。同頁〈常見錯誤〉裡同樣把 eval 模式寫成「推論模式」的那句，也改成「輸出會和 eval 模式下不同」，免得和 inference_mode 混淆。摘錄比對工具結果為 []。 |
| 2 | 先前查核 | 「評估用的 0.05」沒有用術語表的名稱「候選截斷門檻」 | 已修正：查證屬實：miniyolo.train 的 score_threshold 是 0.05，render_learning_evidence 也用這個值濾預測框。改成「那裡用的是評估用的候選截斷門檻 0.05，比本節的顯示門檻 0.25 低，所以那張圖可能畫出 score 不到 0.25 的框」，同時說明這對讀者看圖的影響。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/05-assignment.md`、`docs/lessons/07-data.md`、`docs/lessons/07-targets.md`、`docs/lessons/07-loss.md`、`docs/lessons/07-inference.md`、`docs/lessons/07-training.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核 2 項建議都在〈定稿修正〉處理；〈來源對照〉列出來源；批次檢查掛在本頁；結構檢查通過。沒有發現問題。

### 第 4 輪：上一輪的處理：通過

第 3 輪沒有發現，沒有處理說明需要核對。
