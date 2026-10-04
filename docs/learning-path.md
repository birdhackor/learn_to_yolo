# 閱讀路線：先追一個問題，再加下一個機制

全書分為第 0–20 章、共 42 節，從第一節（第 0 章的暖身）依序讀即可。你會走過小 CNN（卷積神經網路）、ResNet（殘差網路）、單物件定位，以及可訓練的格子偵測器（grid detector）：它把圖切成格子，每格各自預測框，也就是第 7 章的 MiniYOLO。之後再用小實驗理解 YOLO 的不同設計。每節網頁都附一個獨立的 Colab notebook；只想先讀網頁時，不必啟動 Colab 的執行環境（runtime，Colab 替你開的雲端機器）。第一次遇到或想複習術語時，都可以查[術語快速查](glossary.md)。

**開始前需要會什麼？**

- **必要：**會基本 Python（變數、if／for、函式、list）。
- **有了比較輕鬆：**照著範例跑過一次 PyTorch（Python 的深度學習函式庫），聽過神經網路與卷積的概念。第 0 章從一個參數的手算講起，第 1 章說明卷積。
- **數學：**高中程度的函數、對數、向量與矩陣乘法即可。導數（變化率）與連鎖律，第 0 章會用數字例子從頭說明。數學不需要熟背；公式會連到數字與程式。

??? note "形狀記號"

    後面常用 `[B,C,H,W]` 描述一批圖片的形狀。例如一次送 8 張 64×64 的彩色圖，形狀寫成 `[8,3,64,64]`：B=8 是這一批（batch）的圖片張數；C=3 是每個位置的數值個數（channel，通道），RGB 彩色圖有紅、綠、藍 3 個；H、W 分別是高度、寬度。C 有時代表類別（class）數，各節會寫明是哪一種。

**先取得一個成果：**第 0–7 章完成單張圖片的物件偵測流程；第 8 章接自己的圖片與資料。**完整演化：**再讀第 9–16 章。**整合與應用：**第 17 章是結業任務，前置在第 6–7 章，不必先讀第 9–16 章；第 18–20 章是選修，可按需求挑讀。

第 9–16 章（版本節）不會把每一代 YOLO 的改動依序全部疊到同一個模型上。每節都寫明自己的起點（從哪個簡化設定出發），以及只改了哪一項機制，方便看出這一項改動本身的作用。

## A. 先讓 CNN 學得動（第 0–3 章）

VGG 是牛津大學 Visual Geometry Group 提出的經典 CNN，以堆疊 3×3 小卷積為主。第 3 章的三節都在講 ResNet 的捷徑（shortcut）：identity 版把輸入原樣加到幾層卷積算出的結果上；projection 版在形狀不同時，先把輸入轉成相同形狀再相加；plain／residual 對照則比較沒有捷徑和有捷徑的網路。

- 0 [一次學習的超短暖身](lessons/00-warmup.md)：模型學一次（更新一次參數）時發生什麼事
- 1 [VGG 風格小 CNN](lessons/01-small-cnn.md)：讓局部圖樣重複使用
- 2 [訓練診斷](lessons/02-diagnostics.md)：loss 不降時，先查哪裡、再查哪裡
- 3.1 [ResNet identity shortcut](lessons/03-identity.md)：先確定真的能原樣通過
- 3.2 [ResNet projection shortcut](lessons/03-projection.md)：對齊形狀也在學轉換
- 3.3 [Plain／residual 對照](lessons/03-comparison.md)：先控制比較條件

## B. 從分類走到可評估的偵測（第 4–8 章）

第 6 章還不用模型：本來該由模型產生的數字、框與分數，都改由人手指定，所以叫「人工框」。這一章用它們練習解碼（把模型輸出的數字換回圖上的框與分數）、NMS（Non-Maximum Suppression，非極大值抑制：刪掉和高分框重疊太多的同類框）和計分。AP50 是一種評估分數：預測框和答案框的 IoU（Intersection over Union，重疊面積÷聯集面積）至少 0.5 才算找對，再據此算出平均精確率（AP，Average Precision）。

- 4.1 [單物件分類與定位](lessons/04-localization.md)：類別之外，還要回答在哪裡
- 4.2 [座標轉換與還原](lessons/04-coordinates.md)：框跟圖片一起移動
- 5 [多物件輸出與責任分配](lessons/05-assignment.md)：哪個預測負責哪個物件
- 6.1 [人工框解碼與 NMS](lessons/06-decode-nms.md)：少一個框不一定更好
- 6.2 [人工框評估與 AP50](lessons/06-evaluation.md)：把預測逐筆算成證據
- 7.1 [Grid MiniYOLO 資料](lessons/07-data.md)：先讓資料可以被檢查
- 7.2 [Grid MiniYOLO targets](lessons/07-targets.md)：把框變成訓練目標
- 7.3 [Grid MiniYOLO loss](lessons/07-loss.md)：loss 必須能手算
- 7.4 [三步訓練與診斷](lessons/07-training.md)：先只更新 3 次參數，確認程式能跑；同頁再看 160 步（160 次參數更新）的學習實驗
- 7.5 [完整圖片推論](lessons/07-inference.md)：把輸出接回圖片
- 7.6 [獨立資料評估](lessons/07-heldout.md)：用沒參與訓練的圖片評估模型
- 8.1 [自己的圖片推論](lessons/08-own-images.md)：先把座標換算和類別順序弄對
- 8.2 [自己的類別與資料](lessons/08-own-data.md)：類別、標註與來源切分（同一個來源的圖片，只能整批分到訓練、驗證或測試其中一組）

## C. YOLO 的演化機制（第 9–16 章）

- 9.1 [YOLOv2 anchor 與框參數化](lessons/09-anchors.md)：anchor 是尺寸起點
- 9.2 [尺寸聚類](lessons/09-anchor-clustering.md)：先驗（anchor 的預設尺寸）由哪一份資料決定
- 10 [YOLOv3 多尺度](lessons/10-multiscale.md)：同一個 pixel 框看兩種尺度
- 11.1 [CSP](lessons/11-csp.md)（Cross Stage Partial）：分一部分通道走較短的路
- 11.2 [特徵融合](lessons/11-fusion.md)：把深層資訊送回細網格
- 11.3 [圖與框同步增強](lessons/11-augmentation.md)：畫素怎麼變，框就怎麼變
- 11.4 [IoU 類 loss](lessons/11-iou-loss.md)：沒有重疊時還能往哪裡移
- 12.1 [Anchor-free](lessons/12-anchor-free.md)：從候選點（可以各自輸出一個框的位置）量出四條邊
- 12.2 [Decoupled head](lessons/12-decoupled-head.md)（head：模型最後把特徵轉成預測的部分）：分類和定位在哪裡分工
- 12.3 [Sample assignment](lessons/12-assignment.md)：哪個候選值得被教
- 12.4 [DFL](lessons/12-dfl.md)（Distribution Focal Loss）：把一條邊距離學成分佈
- 13.1 [YOLOv10 dual assignment](lessons/13-dual-assignment.md)：訓練時多教，推論時少重複
- 13.2 [NMS-free 推論](lessons/13-nms-free.md)：拿掉 NMS 前，重複候選學會了什麼
- 14 [YOLO11 特徵模組](lessons/14-feature-module.md)：拆路徑、保留中間成果、再融合
- 15.1 [Feature map 到 attention](lessons/15-attention-bridge.md)：四個位置怎麼互相讀取
- 15.2 [YOLOv12 Area Attention](lessons/15-area-attention.md)：互動範圍是一筆預算
- 16.1 [YOLO26 DFL-free](lessons/16-dfl-free.md)：移除 bins（DFL 替一條邊的距離準備的 0、1、2、… 格這些整數刻度），仍要把框學好
- 16.2 [YOLO26 推論 head](lessons/16-inference-head.md)：把訓練用的分支從部署模型（實際交付使用時的模型）真正拿掉
- 16.3 [YOLO26 訓練補強](lessons/16-training.md)：Progressive Loss、STAL 與 MuSGD

## D. 整合與應用（第 17–20 章）

- 17 [靜態偵測結業任務](lessons/17-capstone.md)（靜態＝單張圖片，不是影片）：用一次有理由的改動交付結果
- 18 [影片串流](lessons/18-video.md)：處理每一幀，並分清 FPS（frames per second，每秒幀數）與延遲
- 19 [簡易 tracking](lessons/19-tracking.md)（追蹤：在影片裡跨畫面維持同一個物件的編號）：框很準，ID 仍可能換人
- 20 [ONNX／TensorRT](lessons/20-deployment.md)（ONNX：Open Neural Network Exchange，一種通用的模型格式；TensorRT：NVIDIA 的推論加速工具）：匯出後先證明同一個輸入得到同一個結果

## 遇到卡住的地方

以下建議在讀到第 7 章之後最有用：

1. 挑同一個物件，對照三樣東西：標註的框、轉成模型要學的數值（target）、從模型輸出換回來的框（解碼結果）。看座標與 tensor 的形狀有沒有對上。
2. 不要只看一個總 loss。總 loss 通常由幾項誤差組成（第 7 章是框、有沒有物件、類別三項），要分開看。
3. 算式看不懂時，先把正在讀的那一節例子裡的具體數字代進去算一遍，觀察形狀（shape）與單位。
4. 實驗失敗時，先保存完整的錯誤訊息，再照[訓練診斷](lessons/02-diagnostics.md)的順序排查。

哪些結果有實測、哪些事沒有驗證，見[驗證範圍](status.md)。
