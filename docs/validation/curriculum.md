# 全套實驗與審查

本頁列出教材的每一份執行紀錄與審查紀錄：42 節各自的實驗、用同一套程式多跑一些步數的補充實驗、兩項 GPU 檢查、網站導覽裡每一頁的審查，以及發布後的公開驗證。這些結果能支持哪些結論、哪些事沒有驗證，見〈[驗證範圍](../status.md)〉。

## 執行紀錄怎麼產生、怎麼檢查

每一節的完整程式（`lesson_cases/<節>.py`，也就是 notebook 的最後一格）都在同一台電腦上用 CPU 從頭跑完，程式裡的 assert 全部通過。印出的文字存成 `artifacts/checks/curriculum/<節>.json`；紀錄裡另有執行的日期（UTC）、電腦（作業系統、CPU 型號、使用的執行緒數、Python 與 PyTorch 版本），以及用 SHA-256 記下的程式：該節程式，加上它直接或間接 import 的每個 repo 檔案。notebook 最後一格存的輸出，和每節頁尾的〈實際執行紀錄〉，都從這份紀錄填入。

紀錄綁定的檔案都沒變，紀錄就一直有效；其中任何一個改了，紀錄才算過期，要重跑。Pages 工作流程建置網站之前，先用 `scripts/validate_curriculum_evidence.py` 核對：每份紀錄都要對得上目前的程式，notebook 最後一格的輸出要和紀錄一字不差，頁尾的執行紀錄區塊也要和紀錄一致；有一項不符，網站就不會發布。這項檢查確認頁面、notebook 與紀錄出自同一份程式；頁面講得對不對，由下方的審查負責。

機制章的通過條件，是數值、shape、監督訊號與梯度符合該節寫出的定義，不是把每個版本的完整模型重訓一次。沒有在相同資料與預算下做完整對照的地方，教材不寫 AP 或速度提升的結論。

## 逐節執行清單

「實驗範圍」欄只粗分各節實驗的重點：「固定數值／幾何／張量機制」的節主要核對數值、shape、監督與梯度，其中不少節也會為了示範機制更新參數（例如 12.4、16.1）；是否更新、更新幾步，以各節頁面為準。

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

## 補充實驗清單

各節程式只跑幾步、幾秒鐘。下面這些實驗用同一套程式多跑一些步數，或接上真實的檔案與資料，結果在對應的頁面解說。每份紀錄同樣記下執行的電腦，並用 SHA-256 綁定所用的腳本與它 import 的模組；哪份紀錄綁定哪些檔案，列在 `scripts/evidence_records.py`。

|實驗|紀錄|解說|
|---|---|---|
|第 1 章的小 CNN：同樣 8 張固定的色塊圖，Adam 更新 40 次|[`01-small-cnn-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/01-small-cnn-learning.json)|[1](../lessons/01-small-cnn.md)|
|第 3 章：plain 與 residual 兩個網路從相同的初始權重出發，用 8 張圖各做 40 次 SGD 更新，再用 4 張位置平移過的圖評估|[`03-comparison-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-comparison-learning.json)|[3.3](../lessons/03-comparison.md)|
|第 4 章的單物件模型：2 張固定的圖，Adam 更新 40 次，只在這 2 張訓練圖上評分|[`04-localization-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-localization-learning.json)|[4.1](../lessons/04-localization.md)|
|第 7 章的訓練 CLI：32 張圖從頭訓練 160 步，另用新 seed 畫的圖做 validation 與 test|[`grid-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/grid-learning.json)|[7.4](../lessons/07-training.md)、[7.6](../lessons/07-heldout.md)|
|第 8 章的自有格式資料：訓練 160 步與 1600 步，checkpoint 存檔後重讀|[`custom-data-160-step.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/custom-data-160-step.json)、[`custom-data-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/custom-data-learning.json)|[8.2](../lessons/08-own-data.md)|
|第 10 章的兩尺度模型：1 張圖，更新 40 次，兩個 head 都解碼並畫出預測|[`10-multiscale-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/10-multiscale-learning.json)|[10](../lessons/10-multiscale.md)|
|第 18、19 章：把 12 幀畫面寫成無損的 FFV1 AVI 檔再讀回，比對畫素、模型預測與疊圖，並把預測接上 tracker|[`video-file.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/video-file.json)|[18](../lessons/18-video.md)、[19](../lessons/19-tracking.md)|
|Fashion-MNIST：真實分類資料的讀取檢查與 40 步分類管線核對|[`fashion-mnist-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/fashion-mnist-learning.json)|[資料規劃](../preparation/data.md)|

## GPU 檢查

GPU 上的檢查只有兩項，都由手動啟動的 GitHub Actions 工作流程在 Modal 的一張 NVIDIA L4 上執行，紀錄同樣綁定所執行的程式：

- 第 20 章：GridDetector 訓練 40 步後匯出 ONNX，建出 TensorRT 的 FP32 engine 與允許 FP16 的 engine，核對 batch 1–4 的原始輸出、解碼後的框、類別與分數都和 PyTorch 一致（[`deployment-gpu.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/deployment-gpu.json)，解說在[第 20 章](../lessons/20-deployment.md)）。沒有檢查每一層實際用的精度，所以不說全部的層都用 FP16。
- 存檔續訓：GridDetector 在 GPU 上訓練、中途存 checkpoint，換一個新的 container 讀回後接著訓練，結果和不中斷的訓練一致（[`gpu-smoke.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/gpu-smoke.json)，解說在〈[GPU／checkpoint 實測](gpu-smoke.md)〉）。

這兩項確認的是 GPU 上的訓練、存檔、續訓與部署管線，不是各機制的品質或速度比較。

## 審查

審查者都是 AI，沒有真人學生測試。每一節課程頁有兩份意見：一位扮演初學讀者（高中程度、數學好、程式新手）的 AI，檢查看不看得懂、圖文與數字是否一致；一位 AI 技術查核，對照原始論文、固定 commit 的官方程式與計算。網站導覽裡的其他 17 頁，各有一位以該頁讀者身分閱讀的 AI（前導頁是初學讀者，維護頁是維護者），以及一位對照 repo 的程式、紀錄與頁面引用來源的 AI 事實查核。每份審查紀錄都列出全部的發現、每個發現怎麼處理，以及修改後的複查。

審查紀錄存在 `reviews/`。`reviews/coverage.json` 用 SHA-256 記下每份審查看的是哪個版本：頁面文字、頁面上的 SVG 圖，以及課程頁的程式與它 import 的模組；頁尾自動產生的執行紀錄區塊與 Colab 連結裡的 tag 不算在內。Pages 工作流程建置網站之前，`scripts/validate_lessons.py` 會核對這些 SHA-256；有任何一頁在審查之後又改過，網站就不會發布。課程頁的審查紀錄連在上方〈逐節執行清單〉的表格裡，其他頁如下：

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

## 發布後的公開驗證

網站發布後，手動啟動的 GitHub Actions 工作流程 **Verify published lessons**（`.github/workflows/verify-release.yml`）在 GitHub 提供的 Linux runner 上，從公開的 tag 重新檢查一次（本版的 tag 是 `lessons-v0.4.0`），結果存成兩份紀錄：

- [`curriculum-publication.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-publication.json)：在 runner 上嚴格建置這個 tag 的網站，逐頁比對公開網站上每一頁的正文與每一張圖；確認每節的 Colab 連結開的是這個 tag 的 notebook、環境格固定在這個 tag、最後一格就是該節程式；也確認先前發布的 tag 仍指向原本的 commit。
- [`curriculum-release-bootstrap.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-release-bootstrap.json)：每種情況都用全新的資料夾與虛擬環境，照 notebook 的順序執行環境格與最後一格：沒有 PyTorch、裝著其他版本、已經是 2.9.1，以及其他版本已經載入（環境格改裝 2.9.1 後要求重新啟動，重新執行後通過）；第 20 章的環境格另外安裝 ONNX 套件。接著在另一個全新的 clone 依序執行 README 的每個指令。

runner 是 GitHub 的 Linux 機器，不是 Google Colab 的託管 runtime；這兩項驗證沒有登入 Google，也沒有用到 GPU。

## 沒有驗證的事

沒有真人學生的學習成效研究，沒有在真實照片資料集上長時間訓練，也沒有 INT8 量化或實體攝影機的實測。notebook 沒有在 Google Colab 的託管 runtime 上逐節執行。
