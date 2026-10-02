# 第 10 課陌生讀者審閱

審閱範圍：只讀 `docs/lessons/10-multiscale.md`、`lesson_cases/10-multiscale.py`。本課沒有引用圖檔，`docs/assets/diagrams/` 亦未找到第 10 課或 multiscale 圖，因此沒有另外的圖可核對。忽略 Colab 佔位。

已執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/10-multiscale.py`，退出碼 0。表格的 `.5625 / .125`、兩個 head 的 `(1,8,8,7) / (1,4,4,7)`、`64+16=80`、雙 head 非零梯度及相同小框的 encode/decode 都符合程式；練習的 `(gx=3,gy=2)`、格內 `(.5,.5)`、wh `(.25,.25)` 也正確。小框的 pixel 大小、stride/cell 換算，以及 stride 不等於感受野已說清楚；也明確限定為兩尺度、每格單框的 YOLOv3 機制簡化，沒有宣稱 AP 提升。

以下三項是自足性與演示缺口，沒有找到數值矛盾。

1. **補 7 通道的含義，並限定 80 是篩選前候選數。**
   - 位置：教材第 23、35、39 行；案例第 15–16、50–55、59 行。
   - 只懂 CNN 的讀者會知道 1×1 卷積輸出 shape，卻無法從本課判斷最後的 `7` 如何拆開、為何每個位置只算一個候選；`raw[...,4]` 與 `[10.,10.,-10.]` 亦是未解釋的索引。案例最後列印 80，但人工 logits 經 threshold 後每尺度只留下 1 框，容易混淆候選和最終偵測數。
   - 具體修法：第 23 行後加一段：本例 7 通道為 `[tx,ty,w_norm,h_norm,obj_logit,class0_logit,class1_logit]`（xy/wh 在 decode 時經 sigmoid），即 `4+1+2`；每格 1 框，因此 fine/coarse 可先視為 `[1,64,7]` / `[1,16,7]`，合計 `[1,80,7]`。80 是分數篩選及 NMS 前的位置數，輸出框數可以更少。案例第 51、54 行旁註明 objectness 與兩個類別索引；第 59 行可由 `fine.shape[1]*fine.shape[2] + coarse.shape[1]*coarse.shape[2]` 算數量，使輸出與 tensor 直接對應。

2. **補人工尺寸分配對各 head 的背景與 class label 的實際影響。**
   - 位置：教材第 23、33 行；案例第 28–29、41–42 行。
   - 現在有說「小物件只分配 fine、大物件只分配 coarse」，但「負格仍提供背景監督」沒有說明另一尺度看到的同一個真實物件會怎麼標。本例又剛好把 small 設 class 0、large 設 class 1，讀者容易把兩 head 當成分別負責兩類，或以為每個 GT 都在兩 head 學一次。
   - 具體修法：第 33 行前加具體 target 對照：fine 只在 `(gx=1,gy=1)` 標 objectness=1、class=0；coarse 只在 `(2,2)` 標 objectness=1、class=1。未被分配到該 head 的 GT 不形成正格，其中心格仍是該 head 的 objectness 負例，負格沒有 box/class 監督。接著明說責任依尺寸、與 class ID 無關，兩個 head 都能預測 0/1 兩類；此處是互斥單標籤類別的教學簡化，原版 YOLOv3 使用各類獨立的 logistic 分數。保留「不是原版最佳 anchor assignment」的限定。

3. **把跨尺度拼接與共用 NMS 做成一個可驗證的例子。**
   - 位置：教材第 33、35 行；案例第 48–56 行。
   - 教材正確要求所有尺度先 decode 到 input pixel，再串接 boxes/scores/labels、做一次分類別 NMS；案例却在 for-loop 內各自 decode/斷言後丟棄結果，沒有拼接，也沒有證明兩個尺度對同一小框的重複輸出會被消掉。讀者只能記住規則，不能從本課程式看懂對應操作。
   - 具體修法：保留兩次人工 logits 的完整 `decode_grid(...)[0]` 結果；展示沿候選維度對 `boxes`、`scores`、`labels` 分別 `torch.cat`，並確認同一小框在拼接後有 2 筆、單次分類別 NMS 後剩 1 筆。教材第 33 行附上這幾行與 shape（boxes `[2,4]`、scores/labels `[2]`）。此 toy 只有一個候選/尺度，可先沿用目前 decode helper，再用合併後的 NMS 示範跨尺度去重；不要把 helper 每尺度已做的 NMS 當成這一步的替代。第 35 行的執行預期加上「跨尺度重複框 2→1」。


## 作者修訂與驗證（2026-10-02）

補7通道、80為score/NMS前候選、尺寸責任造成另一head中心為負格，以及兩head類別獨立、softmax與原版YOLOv3 logistic差異。case真的cat兩尺度boxes／scores／labels，分類別NMS驗證2→1；推論不能使用未知GT尺寸先排除head。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/10-multiscale.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。
