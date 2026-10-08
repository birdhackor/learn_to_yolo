# 全套實驗與審查

本頁列出教材的每一份執行紀錄與審查紀錄：52 節各自的實驗、把幾節實驗延長或接上真實檔案與資料的補充實驗、兩項 GPU 檢查、網站導覽裡每一頁的審查，以及發布後的公開驗證。這些結果能支持哪些結論、哪些事沒有驗證，見〈[驗證範圍](../status.md)〉。

## 執行紀錄怎麼產生、怎麼檢查

每一節的完整程式（`lesson_cases/<節>.py`，也就是 notebook 的最後一格）都用 CPU 從頭跑完，程式裡的 assert 全部通過。印出的文字存成 `artifacts/checks/curriculum/<節>.json`；紀錄裡另有執行的日期（UTC）、電腦（作業系統、CPU 型號、使用的執行緒數、Python 與 PyTorch 版本等），以及用 SHA-256（由檔案內容算出的指紋，內容改一點就不同）記下的程式：該節程式，加上它直接或間接 import 的每個 repo 檔案。notebook 最後一格存的輸出，和每節頁尾的〈實際執行紀錄〉，都從這份紀錄填入。

紀錄綁定的檔案都沒變，紀錄就一直有效；其中任何一個改了，紀錄才算過期，要重跑。過期的紀錄由 `scripts/record_evidence.py` 列出並重產（GPU 紀錄經 GitHub Actions 重跑），步驟見〈[發布與帳號設定](../preparation/publish.md)〉。沒有過期的紀錄沿用，保留原本的日期與電腦，所以各節頁尾的日期不一定相同。發布網站的 Pages 工作流程（Publish Learn to YOLO）建置網站之前，先用 `scripts/validate_curriculum_evidence.py` 核對：每份紀錄都要對得上目前的程式，notebook 最後一格的輸出要和紀錄一字不差，頁面上也要找得到紀錄的日期、CPU 型號與 PyTorch 版本（比對整頁文字）；有一項不符，網站就不會發布。頁尾區塊裡的輸出由 `scripts/verify_curriculum.py` 從紀錄寫入，這項檢查不逐字比對它，也不檢查網站上的任何實驗圖。這項檢查只比對、不重跑程式，確認的是紀錄對得上目前的程式、notebook 存的就是紀錄的輸出；頁面講得對不對，由下方的審查負責。

第 9–16 章（各版 YOLO 的機制）的通過條件，是數值、shape、監督訊號與梯度符合該節寫出的定義，不是把每個版本的完整模型重訓一次。沒有在相同資料與相同訓練預算（例如步數）下做完整對照的地方，教材不寫 AP 或速度提升的結論。

第 21–23 章是選讀支線。tiny ViT 用 60 步訓練紅／藍矩形分類；tiny DINO 用 160 步無標籤訓練，再凍結特徵做近鄰、linear probe 與定位。兩者都核對中途 checkpoint 恢復後與連續訓練的狀態。原始 DINO 的完整 multi-crop、DINOv2 與 DINOv3 預訓練都沒有重現。23.1 主例只算手工 Gram 矩陣；另有選讀的官方 DINOv2 CPU 特徵實驗，約 84.2 MiB 權重不進 Git。

## 逐節執行清單

「實驗範圍」欄只粗分各節實驗的重點：「固定數值／幾何／張量機制」的節主要核對數值、shape、監督與梯度，其中不少節也會為了示範機制更新參數（例如 12.4、16.1）；「短步更新／管線」的節，重點是讓參數真的更新，或把資料、模型、解碼與評估串成一條跑通，步數從第 0 章的 1 步到第 17、18 章的 160 步不等。是否更新、更新幾步，以各節頁面為準。

|節次／教材|實驗範圍|執行紀錄|審查紀錄|
|---|---|---|---|
|0 [一次學習的超短暖身](../lessons/00-warmup.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/00-warmup.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/00-warmup.md)|
|1 [VGG 風格小 CNN](../lessons/01-small-cnn.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/01-small-cnn.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/01-small-cnn.md)|
|2 [訓練診斷](../lessons/02-diagnostics.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/02-diagnostics.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/02-diagnostics.md)|
|3.1 [ResNet identity shortcut](../lessons/03-identity.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-identity.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/03-identity.md)|
|3.2 [ResNet projection shortcut](../lessons/03-projection.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-projection.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/03-projection.md)|
|3.3 [Plain／residual 對照](../lessons/03-comparison.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-comparison.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/03-comparison.md)|
|4.1 [單物件分類與定位](../lessons/04-localization.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-localization.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/04-localization.md)|
|4.2 [座標轉換與還原](../lessons/04-coordinates.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-coordinates.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/04-coordinates.md)|
|5 [多物件輸出與責任分配](../lessons/05-assignment.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/05-assignment.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/05-assignment.md)|
|6.1 [人工框解碼與 NMS](../lessons/06-decode-nms.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/06-decode-nms.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/06-decode-nms.md)|
|6.2 [人工框評估與 AP50](../lessons/06-evaluation.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/06-evaluation.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/06-evaluation.md)|
|7.1 [Grid MiniYOLO 資料](../lessons/07-data.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-data.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/07-data.md)|
|7.2 [Grid MiniYOLO targets](../lessons/07-targets.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-targets.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/07-targets.md)|
|7.3 [Grid MiniYOLO loss](../lessons/07-loss.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-loss.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/07-loss.md)|
|7.4 [三步訓練與診斷](../lessons/07-training.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-training.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/07-training.md)|
|7.5 [完整圖片推論](../lessons/07-inference.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-inference.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/07-inference.md)|
|7.6 [獨立資料評估](../lessons/07-heldout.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-heldout.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/07-heldout.md)|
|8.1 [自己的圖片推論](../lessons/08-own-images.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/08-own-images.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/08-own-images.md)|
|8.2 [自己的類別與資料](../lessons/08-own-data.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/08-own-data.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/08-own-data.md)|
|9.1 [YOLOv2 anchor 與框參數化](../lessons/09-anchors.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/09-anchors.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/09-anchors.md)|
|9.2 [尺寸聚類](../lessons/09-anchor-clustering.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/09-anchor-clustering.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/09-anchor-clustering.md)|
|10 [YOLOv3 多尺度](../lessons/10-multiscale.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/10-multiscale.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/10-multiscale.md)|
|11.1 [CSP](../lessons/11-csp.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-csp.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/11-csp.md)|
|11.2 [特徵融合](../lessons/11-fusion.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-fusion.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/11-fusion.md)|
|11.3 [圖與框同步增強](../lessons/11-augmentation.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-augmentation.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/11-augmentation.md)|
|11.4 [IoU 類 loss](../lessons/11-iou-loss.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-iou-loss.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/11-iou-loss.md)|
|12.1 [Anchor-free](../lessons/12-anchor-free.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-anchor-free.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/12-anchor-free.md)|
|12.2 [Decoupled head](../lessons/12-decoupled-head.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-decoupled-head.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/12-decoupled-head.md)|
|12.3 [Sample assignment](../lessons/12-assignment.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-assignment.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/12-assignment.md)|
|12.4 [DFL](../lessons/12-dfl.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-dfl.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/12-dfl.md)|
|13.1 [YOLOv10 dual assignment](../lessons/13-dual-assignment.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/13-dual-assignment.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/13-dual-assignment.md)|
|13.2 [NMS-free 推論](../lessons/13-nms-free.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/13-nms-free.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/13-nms-free.md)|
|14 [YOLO11 特徵模組](../lessons/14-feature-module.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/14-feature-module.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/14-feature-module.md)|
|15.1 [Feature map 到 attention](../lessons/15-attention-bridge.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/15-attention-bridge.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/15-attention-bridge.md)|
|15.2 [YOLOv12 Area Attention](../lessons/15-area-attention.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/15-area-attention.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/15-area-attention.md)|
|16.1 [YOLO26 DFL-free](../lessons/16-dfl-free.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-dfl-free.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/16-dfl-free.md)|
|16.2 [YOLO26 推論 head](../lessons/16-inference-head.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-inference-head.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/16-inference-head.md)|
|16.3 [YOLO26 訓練補強](../lessons/16-training.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-training.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/16-training.md)|
|17 [靜態偵測結業任務](../lessons/17-capstone.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/17-capstone.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/17-capstone.md)|
|18 [影片串流](../lessons/18-video.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/18-video.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/18-video.md)|
|19 [簡易 tracking](../lessons/19-tracking.md)|固定數值／幾何／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/19-tracking.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/19-tracking.md)|
|20 [ONNX／TensorRT](../lessons/20-deployment.md)|短步更新／管線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/20-deployment.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/20-deployment.md)|

|節次／教材|實驗範圍|執行紀錄|審查紀錄|
|---|---|---|---|
|[21.1 圖片切成 patch](../lessons/21-patches.md)|固定數值／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/21-patches.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/21-patches.md)|
|[21.2 Patch 如何交換資訊](../lessons/21-attention.md)|固定數值／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/21-attention.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/21-attention.md)|
|[21.3 組成 tiny ViT](../lessons/21-transformer.md)|固定數值／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/21-transformer.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/21-transformer.md)|
|[21.4 訓練、評估與恢復 ViT](../lessons/21-training.md)|完整小模型訓練／評估／checkpoint|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/21-training.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/21-training.md)|
|[22.1 沒有標籤的兩種視圖](../lessons/22-views.md)|固定數值／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-views.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/22-views.md)|
|[22.2 一致但沒有資訊：collapse](../lessons/22-collapse.md)|固定數值／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-collapse.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/22-collapse.md)|
|[22.3 從零實作 DINO 核心](../lessons/22-distillation.md)|完整小模型訓練／評估／checkpoint|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-distillation.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/22-distillation.md)|
|[22.4 特徵有沒有用：近鄰與 linear probe](../lessons/22-features.md)|凍結特徵／下游訓練與評估|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-features.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/22-features.md)|
|[23.1 DINO 版本與官方預訓練特徵](../lessons/23-dino-versions.md)|固定數值／張量機制|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/23-dino-versions.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/23-dino-versions.md)|
|[23.2 凍結 patch 特徵接回定位](../lessons/23-detection-bridge.md)|凍結特徵／下游訓練與評估|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/23-detection-bridge.json)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/23-detection-bridge.md)|

## 補充實驗清單

下面這些實驗把第 1、3、4、7、8、10 章的實驗延長、讓第 18、19 章讀寫真正的影片檔，並用 Fashion-MNIST 核對分類的訓練流程，結果在對應的頁面解說。每份紀錄也都記下執行的電腦，並用 SHA-256 綁定所用的腳本與它 import 的模組：完整的檔案清單記在各紀錄的 `dependencies_sha256`，`scripts/evidence_records.py` 列出每份紀錄從哪些入口檔開始追查 import。

|實驗|紀錄|解說|
|---|---|---|
|第 1 章的小 CNN：同樣 8 張固定的色塊圖，Adam 更新 40 次，只在這 8 張訓練圖上評分|[`01-small-cnn-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/01-small-cnn-learning.json)|[1](../lessons/01-small-cnn.md)|
|第 3 章：plain 與 residual 兩個網路從相同的初始權重出發，用 8 張圖各做 40 次 SGD 更新，再用 4 張位置平移過的圖評估|[`03-comparison-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-comparison-learning.json)|[3.3](../lessons/03-comparison.md)|
|第 4 章的單物件模型：2 張固定的圖，Adam 更新 40 次，只在這 2 張訓練圖上評分|[`04-localization-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-localization-learning.json)|[4.1](../lessons/04-localization.md)|
|第 7 章的訓練指令 `python -m miniyolo.train`：32 張圖從頭訓練 160 步，另用新 seed 畫的圖做 validation 與 test|[`grid-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/grid-learning.json)|[7.4](../lessons/07-training.md)、[7.6](../lessons/07-heldout.md)|
|第 8 章的自有格式資料：訓練 160 步與 1600 步，checkpoint 存檔後重讀|[`custom-data-160-step.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/custom-data-160-step.json)、[`custom-data-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/custom-data-learning.json)|[8.2](../lessons/08-own-data.md)|
|第 10 章的兩尺度模型：1 張圖，更新 40 次，兩個 head 都解碼並畫出預測，只在這張訓練圖上評分|[`10-multiscale-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/10-multiscale-learning.json)|[10](../lessons/10-multiscale.md)|
|第 18、19 章：把 12 幀畫面寫成無損的 FFV1 AVI 檔再讀回，比對畫素、模型預測與疊圖，並把預測接上 tracker|[`video-file.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/video-file.json)|[18](../lessons/18-video.md)、[19](../lessons/19-tracking.md)|
|Fashion-MNIST：真實分類資料的讀取檢查與 40 步分類管線核對|[`fashion-mnist-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/fashion-mnist-learning.json)|[資料規劃](../preparation/data.md#fashion-mnist)|

## GPU 檢查

GPU 上的檢查只有兩項，都由手動啟動的 GitHub Actions（GitHub 提供的自動執行程式服務）工作流程在 Modal（租用雲端 GPU 的服務）的一張 NVIDIA L4 上執行。紀錄綁定 `miniyolo/` 裡的實驗程式（`deployment_gpu.py`、`gpu_smoke.py`）與它們 import 的模組；啟動 Modal 工作的 `scripts/modal_deployment.py`、`scripts/modal_gpu_smoke.py` 不在其中，雖然它們也定義了一部分在 Modal 上執行的步驟（見〈[GPU／checkpoint 實測](gpu-smoke.md)〉）：

- 第 20 章：GridDetector 訓練 40 步後匯出 ONNX，建出 TensorRT 的 FP32 engine 與允許 FP16 的 engine，核對 batch 1–4 的原始輸出、解碼後的框與分數都在容許誤差內和 PyTorch 相符，類別與候選順序完全相同（[`deployment-gpu.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/deployment-gpu.json)，解說在[第 20 章的 L4 結果](../lessons/20-deployment.md#l4-results)）。沒有檢查每一層實際用的精度，所以不能確定每一層都用 FP16。
- 存檔續訓：GridDetector 在 GPU 上訓練、中途存 checkpoint，換一個新的 container（獨立的執行環境）讀回後接著訓練，和不中斷的訓練相比，最大差異為 0（[`gpu-smoke.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/gpu-smoke.json)，解說在〈[GPU／checkpoint 實測](gpu-smoke.md)〉）。

這兩項確認的是 GPU 上的訓練、存檔、續訓與部署管線，不是各機制的品質或速度比較。

## 審查

### 2026-10-06：依新版四題重審全部 52 節

本輪凍結 `64a25d4`，六位新讀者按各自實際補讀的前文逐段閱讀，保存 995 個閱讀單元、212 次四題檢查（包括 52 次頁末），再開放下一段。概念介紹、名稱、缺圖與訓練證據問題分別處理；改寫後另作非作者技術核對，以及全路線銜接和實際桌機／手機檢查。原始筆記、修正決策、技術來源與視覺範圍保存在[本輪紀錄](https://github.com/birdhackor/learn_to_yolo/tree/main/reviews/clear-tutorial/full-review-2026-10-06)。

四題仍有漏抓：22.1 的 DINO 命名方法介紹被首次讀者當成「後面會教」，未列問題；另有部分答案的逐項引文不足、把推論標成正文明說。獨立紀錄查核保留這些缺陷，教材已補當下必要的介紹，原始答案沒有回寫。故不能把本輪統稱為嚴格盲讀規則全部合格，也不能由回答了四題保證沒有漏項。

手機的第 4、20 章仍有圖內細字偏小，必要分工與數字可由相鄰正文讀出；本輪沒有把所有圖字概括判成手機可讀。第 4 章過長的必要公式已拆行，新的第 17–19 章結果圖用直式呈現。AI 審閱與測試仍不等同真人學生學習效果。

既有主線上一輪依 repo 的 `clear-tutorial` skill 重審，審查者都是 AI；下列是 `16f6910` 那一輪的歷史範圍，新支線另記於下方。先前的完整頁面審查仍保留，不改標成逐段盲讀。

1. **先記首次閱讀的理解。** 凍結 `16f6910` 的 59 個導覽頁，由六位獨立讀者按段落讀。每開放下一段前，先保存當下的理解、原文卡點、猜測與缺圖；沒有先交全文再要求扮演初學者。各組按指定路線實際補讀前提，並非每組都已讀過整本書。
2. **修改後查技術與證據。** 由未撰寫該頁的人對照程式、執行紀錄與圖；疑點回查論文、固定版本的官方程式或文件。必要實驗用 CPU 重跑；這次文字與圖解修訂沒有新增 GPU 訓練。
3. **再看銜接與修正。** 另一位讀者檢查受影響的前文、本節和下一節，並在實際 Zensical 頁面看桌面、手機的圖與公式。修改處有對應複查，不以一份全文摘要代替。

共用檔案系統沒有技術隔離，首次閱讀依揭露規則執行。本輪也記下協調者的問題：部分前提頁漏列、08後段太早收到作者任務提示；這些紀錄不能全部算嚴格盲讀證據，原始問題保留，修正後另作複查。這些限制和每項處理見[本輪原始閱讀與修正紀錄](https://github.com/birdhackor/learn_to_yolo/tree/main/reviews/clear-tutorial/16f6910)。

AI 審查能幫忙找卡點，不等於真人學生已看懂；本專案沒有做真人學生的學習效果測試。

審查紀錄存在 `reviews/`。`reviews/coverage.json` 用 SHA-256 記下每份審查看的是哪個版本：頁面文字、頁面上的 SVG、PNG 等圖，以及課程頁的程式與它 import 的模組；頁尾自動產生的執行紀錄區塊與 Colab 連結裡的 tag 不算在內。Pages 工作流程建置網站之前，`scripts/validate_lessons.py` 會核對這些 SHA-256；有任何一頁在審查之後又改過，網站就不會發布。這項檢查會核對正文的內容指紋，卻不會自動核算正文數字與重產紀錄是否一致：紀錄重產後，由維護者在 `docs/` 與 `README.md` 搜尋紀錄的檔名與舊數字，對照新紀錄逐一核對，改了正文就重審那一頁（見〈[發布與帳號設定](../preparation/publish.md)〉）。課程頁的審查紀錄連在上方〈逐節執行清單〉的表格裡，其他頁如下：

|頁面|審查紀錄|
|---|---|
|[首頁](../index.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/index.md)|
|[完整閱讀路線](../learning-path.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/learning-path.md)|
|[術語快速查](../glossary.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/glossary.md)|
|[驗證範圍](../status.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/status.md)|
|[GPU／checkpoint 實測](gpu-smoke.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/validation-gpu-smoke.md)|
|[全套實驗與審查](curriculum.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/validation-curriculum.md)|
|[資料規劃](../preparation/data.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/preparation-data.md)|
|[發布與帳號設定](../preparation/publish.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/preparation-publish.md)|
|[網頁與 Colab 規格](../preparation/architecture.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/preparation-architecture.md)|
|[課程大綱](../planning/outline.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/planning-outline.md)|
|[公開課程研究](../planning/course-research.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/planning-course-research.md)|
|[讀者與學習心得](../planning/feedback.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/planning-feedback.md)|
|[基礎資料](../research/foundation-data.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/research-foundation-data.md)|
|[偵測資料](../research/detection-data.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/research-detection-data.md)|
|[Git LFS](../research/lfs.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/research-lfs.md)|
|[Pages 與 Colab](../research/pages-colab.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/research-pages-colab.md)|
|[版本來源查證](../research/version-sources.md)|[審查](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/research-version-sources.md)|

## ViT／DINO 支線的審閱與選讀紀錄

新支線依 clear-tutorial 分成逐段首次閱讀、技術與證據核對、前後銜接三輪；原始記錄與修正複查保存在 [本次審閱目錄](https://github.com/birdhackor/learn_to_yolo/tree/main/reviews/clear-tutorial/vision-v0.6.0)。先前 `16f6910` 的審閱是舊版紀錄，不充當新增十節的審閱。AI 審閱仍不等於真人學生的理解測試。

[官方 DINOv2 選讀紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/dinov2-pretrained.json)保存固定官方程式、權重 SHA-256、前處理、輸出 shape、近鄰與計時範圍。這是兩張受控圖的凍結特徵提取，不是自然影像語意品質評測，也沒有 DINOv3 訓練。

## 發布後的公開驗證

2026-10-08 的網頁文字與圖解修訂，公開驗證另存於[本輪審閱紀錄](https://github.com/birdhackor/learn_to_yolo/tree/main/reviews/clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair)。這次比對當次嚴格建置與公開網站的正文、圖檔，並核對 52 個 Colab notebook 的程式碼儲存格。實驗程式沒有改動，入口仍使用 `lessons-v0.6.1`；下文的 tag 發布驗證保留原本日期與範圍。

網站發布後，手動啟動的 GitHub Actions 工作流程 **Verify published lessons**（`.github/workflows/verify-release.yml`）在 GitHub 提供的 Linux runner 上，從公開的 tag 重新檢查一次（本版的 tag 是 `lessons-v0.6.1`）。下文說明本版的驗證流程；連結紀錄實際驗證的版本以各自的 `source_ref` 為準。`lessons-v0.5.0` 的紀錄只涵蓋原有 42 節，不驗證新增的十節。工作流程把結果上傳成 Actions 的 artifact，維護者再用 `scripts/verify_release.py save --run <run 編號>` 存成兩份紀錄：

- [`curriculum-publication.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-publication.json)：在 runner 上嚴格建置（`zensical build --strict`，有警告就算失敗）這個 tag 的網站，再和公開網站比對：導覽裡每一頁的正文，以及 `assets/diagrams/` 的每張圖，都要完全相同。也確認每節的 Colab 連結開的是這個 tag 的 notebook、環境格固定在這個 tag、最後一格就是該節程式，以及先前發布的 tag 仍指向原本的 commit。
- [`curriculum-release-bootstrap.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-release-bootstrap.json)：除了第 20 章，51 本 notebook 的環境格逐字相同，所以用第 0 章的 notebook 測四種情況：沒有 PyTorch、裝著其他版本、已經是 2.9.1，以及其他版本已經載入（環境格改裝 2.9.1 後要求重新啟動，用新的程序重跑兩格後通過）。每種情況都用全新的資料夾與虛擬環境，照 notebook 的順序執行環境格與最後一格，最後一格的輸出要和 notebook 存的輸出相同。第 20 章的環境格另外安裝 ONNX 套件，所以另在沒有 PyTorch 的環境執行一次它的兩格（輸出含計時，不要求相同）。接著在另一份全新的 clone 裡，依序執行 README 的 bash 區塊裡的每個指令（不會自己結束的 `zensical serve` 除外），其中包括 pytest，以及把 52 節的實驗格各跑一次、和紀錄比對輸出的 `scripts/check_lesson_runtime.py`（輸出和紀錄不同不算失敗，哪些節逐字相同記在紀錄的 `lesson_runtime`）。

依發布步驟，兩份紀錄最外層的 `passed` 都是 `true`，才 commit 到 main。這兩份紀錄在 tag 建立之後才產生，所以 tag 裡的 `curriculum-publication.json` 是上一版的驗證。紀錄裡逐項記著每一頁、每種情況與每個指令的結果。runner 是 GitHub 的 Linux 機器，不是 Google Colab 的託管 runtime；這兩項驗證沒有登入 Google，也沒有用到 GPU。

## 沒有驗證的事

偵測實驗都只用合成圖或小型自製資料。沒有真人學生的學習成效研究，沒有在真實照片資料集上長時間訓練，也沒有 INT8 量化或實體攝影機的實測。notebook 沒有在 Google Colab 的託管 runtime 上執行過，Colab 可能分配的 GPU 也沒有測。完整的清單見〈[驗證範圍](../status.md)〉的〈沒有驗證的事〉。
