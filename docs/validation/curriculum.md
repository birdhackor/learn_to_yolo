# 全套教材實驗與兩輪審查

本輪先依閱讀順序執行42節，各節通過後立即保存stdout、同步notebook並修訂該頁。接著補必要短訓練與部署，再進行兩輪獨立審查，審查者都是AI：扮演初學讀者的AI檢查理解、圖文與數字；另一批AI技術審查者核對技術與原始來源。審查修訂後重新執行受影響範例。

機制章的通過條件是數值、shape、監督與梯度符合所列定義，不要求每一版本重訓完整模型。沒有相同資料與預算的完整對照，就不寫AP或速度提升結論。

## 逐節執行清單

「實驗範圍」欄只粗分各節實驗的重點，不是「有沒有更新參數」的清單：標為「固定數值／幾何／張量機制」的節主要核對數值、shape、監督與梯度，其中不少節也會為了示範機制更新參數（例如12.4、16.1）；是否更新、更新幾步，以該頁為準。

|節次／教材|實驗範圍|結果紀錄|
|---|---|---|
|0 [一次學習的超短暖身](../lessons/00-warmup.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/00-warmup.json)|
|1 [VGG 風格小 CNN](../lessons/01-small-cnn.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/01-small-cnn.json)|
|2 [訓練診斷](../lessons/02-diagnostics.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/02-diagnostics.json)|
|3.1 [ResNet identity shortcut](../lessons/03-identity.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-identity.json)|
|3.2 [ResNet projection shortcut](../lessons/03-projection.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-projection.json)|
|3.3 [Plain／residual 對照](../lessons/03-comparison.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-comparison.json)|
|4.1 [單物件分類與定位](../lessons/04-localization.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-localization.json)|
|4.2 [座標轉換與還原](../lessons/04-coordinates.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-coordinates.json)|
|5 [多物件輸出與責任分配](../lessons/05-assignment.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/05-assignment.json)|
|6.1 [人工框解碼與 NMS](../lessons/06-decode-nms.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/06-decode-nms.json)|
|6.2 [人工框評估與 AP50](../lessons/06-evaluation.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/06-evaluation.json)|
|7.1 [Grid MiniYOLO 資料](../lessons/07-data.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-data.json)|
|7.2 [Grid MiniYOLO targets](../lessons/07-targets.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-targets.json)|
|7.3 [Grid MiniYOLO loss](../lessons/07-loss.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-loss.json)|
|7.4 [三步訓練與診斷](../lessons/07-training.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-training.json)|
|7.5 [完整圖片推論](../lessons/07-inference.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-inference.json)|
|7.6 [獨立資料評估](../lessons/07-heldout.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-heldout.json)|
|8.1 [自己的圖片推論](../lessons/08-own-images.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/08-own-images.json)|
|8.2 [自己的類別與資料](../lessons/08-own-data.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/08-own-data.json)|
|9.1 [YOLOv2 anchor 與框參數化](../lessons/09-anchors.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/09-anchors.json)|
|9.2 [尺寸聚類](../lessons/09-anchor-clustering.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/09-anchor-clustering.json)|
|10 [YOLOv3 多尺度](../lessons/10-multiscale.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/10-multiscale.json)|
|11.1 [CSP](../lessons/11-csp.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-csp.json)|
|11.2 [特徵融合](../lessons/11-fusion.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-fusion.json)|
|11.3 [圖與框同步增強](../lessons/11-augmentation.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-augmentation.json)|
|11.4 [IoU 類 loss](../lessons/11-iou-loss.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-iou-loss.json)|
|12.1 [Anchor-free](../lessons/12-anchor-free.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-anchor-free.json)|
|12.2 [Decoupled head](../lessons/12-decoupled-head.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-decoupled-head.json)|
|12.3 [Sample assignment](../lessons/12-assignment.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-assignment.json)|
|12.4 [DFL](../lessons/12-dfl.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-dfl.json)|
|13.1 [YOLOv10 dual assignment](../lessons/13-dual-assignment.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/13-dual-assignment.json)|
|13.2 [NMS-free 推論](../lessons/13-nms-free.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/13-nms-free.json)|
|14 [YOLO11 特徵模組](../lessons/14-feature-module.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/14-feature-module.json)|
|15.1 [Feature map 到 attention](../lessons/15-attention-bridge.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/15-attention-bridge.json)|
|15.2 [YOLOv12 Area Attention](../lessons/15-area-attention.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/15-area-attention.json)|
|16.1 [YOLO26 DFL-free](../lessons/16-dfl-free.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-dfl-free.json)|
|16.2 [YOLO26 推論 head](../lessons/16-inference-head.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-inference-head.json)|
|16.3 [YOLO26 訓練補強](../lessons/16-training.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-training.json)|
|17 [靜態偵測結業任務](../lessons/17-capstone.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/17-capstone.json)|
|18 [影片串流](../lessons/18-video.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/18-video.json)|
|19 [簡易 tracking](../lessons/19-tracking.md)|固定數值／幾何／張量機制|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/19-tracking.json)|
|20 [ONNX／TensorRT](../lessons/20-deployment.md)|短步更新／管線；具體步數以該頁為準|[通過](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/20-deployment.json)|

## 補充實驗清單

1. 第1章：原模型固定8張色塊，40步Adam；確認有限非零梯度、權重改變、loss與訓練預測。
2. 第3章：相同初始權重與40步SGD的plain／residual對照；記錄梯度、loss與4張位移validation，允許兩者沒有排名差異。
3. 第4章：原單物件模型、2張資料、40步Adam；確認框IoU、分類與loss，明示只評訓練圖。
4. 第7章：原訓練CLI重跑160步合成資料，重新產生曲線、獨立validation框與test報告。
5. 第10章：同一個TwoScale模型40步訓練1張圖，實際decode兩head並畫預測；不宣稱held-out小物件提升。
6. 資料準備：Fashion-MNIST的SHA-256／IDX讀取與40步分類驗證，保留官方test邊界。
7. 第20章：手動Actions呼叫單張L4，原GridDetector40步後實際建FP32／允許FP16混合精度的TensorRT engine；核對B1–4的raw、非空decoded框、類別與分數。未檢查逐層精度，不宣稱全部層使用FP16。engine只放專用Volume，不進普通Git。
8. 第8章：補JSON／PNG的三類完整訓練入口。160步定位不足的紀錄保留；依train的座標／mask診斷，事前固定1600步，其餘設定與train／validation PNG不變。train AP50=1.0，validation=.388889，新的test seed7001只評一次、AP50=.666667；checkpoint重讀及原圖推論通過。
9. 第18／19章：真正寫出並重讀12幀FFV1 AVI；RGB、預測與疊圖相同，capture在檔尾、提前close及失敗時均釋放。真實class0預測接上tracker，保留漏檢造成新ID的結果；不宣稱新的追蹤品質指標。

## 已有GPU與checkpoint證據

[L4／Volume／私有HF／精確續訓](gpu-smoke.md)保留獨立實測紀錄。它驗證訓練與保存恢復管線，不替代各現代機制的品質比較。

## 審查交付

本頁列出的審查（含下方〈對照大綱的最後補齊〉的四份）都在2026-10-02發布`lessons-v0.3.0`之前完成，看的是當時的頁面文字與圖；之後各節頁面、首頁等為了更好讀而改寫過，改寫時新增的說明、例題、練習答案，以及重畫或新增的圖，不在這些審查的範圍內。

兩輪均已完成：六個AI模擬讀者與另外六個AI技術審查者（都是另外派出、各自獨立閱讀的AI助手），各輪覆蓋全部42節。初讀發現完整保留；必要修改由提出問題的審查者獨立複查，或在初審未發現必要技術修正。沒有把作者自己改完當成審查通過。

|組別／範圍|初讀、修訂與複查|論文／原始碼核實與複查|
|---|---|---|
|A：暖身、CNN、ResNet、單物件與座標（8節）|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-reader-a.md)|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-accuracy-a.md)|
|B：grid、解碼、AP、完整訓練與評估（9節）|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-reader-b.md)|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-accuracy-b.md)|
|C：自有資料、anchor、多尺度、CSP與融合（7節）|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-reader-c.md)|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-accuracy-c.md)|
|D：增強、IoU、anchor-free、DFL與dual（8節）|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-reader-d.md)|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-accuracy-d.md)|
|E：YOLO11、attention與YOLO26（6節）|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-reader-e.md)|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-accuracy-e.md)|
|F：結業、影片、tracking、部署及共同頁（4節）|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-reader-f.md)|[已完成](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/curriculum-accuracy-f.md)|

[逐節覆蓋與報告校驗](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-review-coverage.json)保存12份報告的SHA-256。技術查核實際讀原論文相關章節、公式與固定官方code；每份報告列出短引文和出處。修正包括練習與assert同步、空框逆變換證據、梯度總量有限性、AP與官方VOC的配對差異，以及YOLO26論文／程式碼的預設模式與STAL門檻差異。

`python scripts/validate_curriculum_evidence.py`逐節核對目前case的SHA-256、notebook code與stdout、JSON/index及頁面證據；Pages建置也執行此檢查。它驗證版本一致，科學結論仍依獨立查核。

教材固定為新的`lessons-v0.3.0`，已發布的`lessons-v0.1.0`與`lessons-v0.2.0`保留不動。前輪從公開v0.2空checkout，實際執行00／20兩種依賴組合的notebook初始化格及CPU案例，兩者通過；[v0.2公開tag初始化驗證](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-release-bootstrap.json)保留完整輸出。這是本地直譯器上的公開clone驗證，沒有登入Google或聲稱測過Colab分配的GPU。

v0.3發布後再做公開空checkout，00／20兩種初始化格和CPU案例再次通過，全部42節source與已審查工作區相同；[v0.3初始化紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-release-bootstrap-v0.3.0.json)保存輸出。另從該tag、未設定PYTHONPATH的子程序實際執行新的1600步資料入口與影片檔案入口，數值和已審查結果一致；[公開tag補充入口驗證](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-public-closure-bootstrap.json)保留結果。這仍是本地CPU驗證，未登入Google Colab。

## 對照大綱的最後補齊

另由獨立審查對照大綱，發現第8章雖有一步更新，尚缺完成條件中的overfit與獨立圖片推論；已補上述〈補充實驗清單〉第8項。第18／19章的實際檔案及接線也一併補驗證。新內容再經兩份從初次閱讀者角度進行的審查與另外兩份技術審查，保留初讀與修後複查：

|範圍|讀者審查|技術核查|
|---|---|---|
|自有JSON資料、訓練與checkpoint|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/closure-reader-custom-data.md)|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/closure-accuracy-custom-data.md)|
|影片檔案與tracking接線|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/closure-reader-application.md)|[紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/closure-accuracy-application.md)|

[收尾證據與費用範圍](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-closure.json)保存四份新review的SHA、已關閉結果與原GPU停止紀錄；必要問題均經提出者獨立複查。新版來源／輸出檢查也驗證兩條補充入口及四份review沒有版本不一致。

收尾沿用已有L4／Volume／HF／TensorRT證據，沒有新增Modal GPU工作或修改帳號費用上限。使用者本輪費用上限10美元；新增訓練全在本機CPU。網站發布使用標準Linux Actions，build最多10分鐘、deploy最多5分鐘；即使以[官方每分鐘0.006美元](https://docs.github.com/en/billing/reference/actions-runner-pricing)估算，每次runner計算上限約0.09美元。這是保守計算範圍，不是帳單；不包含先前已完成的GPU工作或整個帳號其他工作。

真人學生學習成效、完整真實場景長訓練、INT8與攝影機實測仍屬後續工作。
