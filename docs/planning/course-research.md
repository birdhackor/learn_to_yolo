YOLO 教學專案：公開教材與課程研究

查核日期：2026-10-02（Asia/Taipei）。本輪由六個 subagent 分工分析；主代理在環境恢復後取得官方網頁、官方 repository 原文、作業 notebook 與 PDF，轉交對應內容供各分支核對。以下區分已讀來源的事實與我們的設計建議。

研究目標：中文為主、PyTorch、小型簡化模型、從 VGG 風格 CNN 到 ResNet、分類轉偵測，再研究 YOLO 各分支的機制、動機、收益與代價。此文件是課綱研究，不是正式教材或實作。

已查核資料中，D2L 最適合參考小型模型與章末實驗；DeepLearning.AI 最貼近 CNN → ResNet → 偵測 → YOLO 的順序；CS231n 提供訓練診斷；李宏毅 HW3 提供 baseline 與 residual 練習；Harvard CS50 AI 提供實驗紀錄格式；MIT 與 fast.ai 提供短實作與回饋節奏。已讀材料沒有提供完整涵蓋早期到现代 YOLO、讓同一個簡化模型連續演化的課綱。

1. **D2L：最適合借用章節與實驗骨架。**

   已讀 VGG、ResNet、anchor、SSD 的章節原文、PyTorch 程式、Summary 與 Exercises。VGG 章研究參數、FLOPs、記憶體與 FC 成本，並質疑將 Fashion-MNIST 的 28×28 輸入放大到 224×224；ResNet 從 identity mapping 與函數族說明 residual connection；anchor 章以畫框、改 sizes／ratios、IoU 與 NMS 說明機制；SSD 章先依賴框、anchor、多尺度與資料集知識，再組合 TinySSD 的訓練及預測。

   值得吸收的是「局部機制可視化 → 改一項設計 → 比較結果與成本」。例如改 anchor 比例看覆蓋，討論 NMS 是否誤刪有效框，或比較回歸 loss。SSD 的 target、head 與 loss 仍屬 SSD 規格，不能直接當成 YOLO 的歷史設計。

   來源：[VGG](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/chapter_convolutional-modern/vgg.md)、[ResNet](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/chapter_convolutional-modern/resnet.md)、[Anchor](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/chapter_computer-vision/anchor.md)、[SSD](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/chapter_computer-vision/ssd.md)。

2. **DeepLearning.AI：最接近目前主線的課程先例。**

   官方公開頁確認四個模組：Week 1 CNN 基礎；Week 2 經典架構、ResNets、1×1 convolution，包含 Residual Networks 作業；Week 3 物件偵測，逐項介紹定位、sliding windows、IoU、NMS、anchor、YOLO，包含 Car detection with YOLO 作業；Week 4 人臉辨識與 style transfer。

   可以吸收其順序，把幾何與後處理先拆開，再接成系統。這次只核對公開作業名稱，未讀付費／登入後的完整作業內容，因此不能稱該作業要求從零訓練 YOLO，也不能用它替代現代 YOLO 機制研究。

   來源：[官方 CNN 課程頁](https://www.coursera.org/learn/convolutional-neural-networks)。

3. **Stanford CS231n：訓練诊斷與課堂／作業範圍的分工。**

   Spring 2025 課表確認 L3 optimization、L4 NN／backprop、L5 CNN、L6 AlexNet／VGG／ResNet、L9 detection／segmentation。L7、L8 為 RNN、attention；我們不必按原課表照搬這些支線。L9 列有 single-stage、two-stage、YOLO、DETR，但同堂也涵蓋多種 segmentation 與模型理解。

   A2 包含 BatchNorm、Dropout、卷積層實作、PyTorch CIFAR-10 自設 CNN、RNN captioning；A3 包含 Transformer captioning、自監督分類、DDPM、CLIP／DINO。這兩份作業不是 YOLO 偵測器實作。

   訓練筆記有 gradient check、sanity check、loss、train／validation accuracy、權重與更新比例、activation／gradient 分布。它建議先記住極少資料，例如 20 筆，並提醒能記住小資料仍可能有 bug，不能證明泛化。

   我們應借用小型訓練診斷單元；偵測部分則比概論課拆得更細，加入單物件定位與多物件配對的橋梁。

   來源：[2025 課表](https://cs231n.stanford.edu/2025/schedule.html)、[A2](https://cs231n.github.io/assignments2025/assignment2/)、[A3](https://cs231n.github.io/assignments2025/assignment3/)、[訓練筆記](https://cs231n.github.io/neural-networks-3/)、[CNN 筆記](https://cs231n.github.io/convolutional-networks/)。

4. **台大李宏毅：問題動機、分級練習與可見的成本。**

   2021 CNN 講義從「影像需要 fully connected 嗎？」出發，依局部圖樣與位置重複性引出 receptive field、parameter sharing、convolution、pooling。講義也說明 CNN 不自然具有縮放／旋轉不變性，以 AlphaGo 不採 pooling 提醒設計要看任務。

   2021 HW3 為 Food-11 影像分類，提供 PyTorch Conv–BN–ReLU–Pool baseline；Easy 跑範例，Medium 改架構或 augmentation，Hard 使用未標註資料。2022 HW3 目標包含 residual；報告要求展示多種 augmentation，以及在提供的模型中只修改 forward 接上 residual connections。不同 baseline 有 GPU time 估計，從約 15–20 分鐘到數小時。這些是原作業設備及設定的估計，不是我們專案的時間承諾。

   可吸收「先局部接線，再比較完整訓練」的 ResNet 練習，以及快速機制驗證／較完整效果比較兩種實驗規模。兩年度任務皆為分類，沒有 bounding-box YOLO 作業。教材與資料的使用條件需個別遵守；這裡借用教學設計，不重新散布課程資料。

   來源：[2021 課表](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)、[CNN 講義](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/cnn_v4.pdf)、[2021 HW3 官方連結 notebook](https://github.com/ga642381/ML2021-Spring/blob/main/HW03/HW03.ipynb)、[2022 課表](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php)、[2022 HW3 PDF](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2022-course-data/Machine%20Learning%20HW3%20-%20Image%20Classification.pdf)。

5. **Harvard：把實驗過程也變成成果。**

   CS50 AI Neural Networks 從 perceptron、multilayer、backprop 到 TensorFlow、CNN／pooling。Traffic 作業用 TensorFlow 分類 GTSRB 的 43 類交通標誌，提供流程、讓學生實作 load_data 與 get_model，也提供 3 類小資料練習。作業允許調整 conv／pool 層數、filter 數量與尺寸、hidden layers、dropout；README 要求記錄試了什麼、有效／無效之處與觀察。

   我們可以把這種紀錄固定成每章四個問題：「改什麼、為什麼、結果、代價」。Traffic 是分類作業，不是框偵測。原 `/2024/` 網址已轉向現行頁面，因此引用現行連結，不將內容硬標成固定年度。

   CS109B 2018 的官方課表也已讀取：先講 NN、regularization、optimization，Week 10 CNN／lab，另有 ConvNet 範例 section 列 VGG、ResNet、Inception。其整體包含大量統計、RNN、生成模型，不適合照搬成 YOLO 主線；未查核其完整作業。

   來源：[Neural Networks](https://cs50.harvard.edu/ai/notes/5/)、[Traffic](https://cs50.harvard.edu/ai/projects/5/traffic/)、[CS109B 2018 課表](https://harvard-iacs.github.io/2018-CS109B/pages/schedule.html)。

6. **MIT：固定資料逐步换模型，數學按需補。**

   6.S191 2026 官方頁列導數、矩陣乘法為先備，其餘隨課說明；有 Deep Computer Vision 與 Lab 2。官方目前提供 PyTorch、TensorFlow 兩套 notebook。PyTorch Lab 2 Part 1 在同一份 MNIST 上先訓練 FC，再訓練小 CNN，讓學生完成 layers、forward、evaluation 等 TODO。

   Lab 2 Part 2 雖名為 facial detection，已讀程式實際是 n_outputs=1 的 face／non-face classifier，再研究 DB-VAE debiasing，不能稱為 bounding-box detector。

   MIT Press《Foundations of Computer Vision》CNN 章以形狀與數字例子介紹 tensor，使用水平／垂直線的 toy classification 與刻意構造的域外失敗，接著說明 global pooling、spatial outputs 與 residual connections。可以借用其「空間資訊去哪了？」作為分類轉偵測的銜接。

   來源：[6.S191](https://introtodeeplearning.com/)、[官方 Lab 2](https://github.com/MITDeepLearning/introtodeeplearning/tree/master/lab2)、[Visionbook CNN 章](https://visionbook.mit.edu/convolutional_neural_nets.html)。

7. **fast.ai 與 NYU：回饋節奏與數學連結。**

   fast.ai 官方頁確認 2022 Part 1 九堂課，以及第二堂結束前以自蒐資料建立、部署模型的安排。值得吸收早期可見成果、理論置於具體問題中的節奏；本專案仍以透明的簡化模型實作為核心。

   NYU 2021 課表早期有 backprop、parameter sharing、自然訊號與 convolution，後续則大量延伸 EBM、graphs、control、optimization。可用來參考模型／數學／程式的連結；目前證據不支持它是 YOLO 作業課。

   來源：[fast.ai](https://course.fast.ai/)、[NYU 2021](https://atcold.github.io/NYU-DLSP21/)。

8. **教學 repository：区分手刻訓練與手刻推論。**

   AladdinPersson 的 YOLOv1 model／loss 與 YOLOv3 train 已讀取。v1 明示教學改動：加入 BatchNorm、FC 4096 縮為 496；v3 訓練累加三尺度 loss，並接上 bbox 解碼與 mAP 評估。這是自行寫模型與訓練的參考，不是現成 YOLO API 包裝。已讀檔案不足以證明有系統消融或同一模型的連續演化教材。

   Ayoosh README 明確說目前僅有 detection module，要求下載既有 yolov3.weights，附五部分手刻教學與 image／video／camera 推論；其 PyTorch 0.4 設定已老舊。適合參考解碼與張量說明，不作從零訓練範本。

   來源：[Aladdin v1 model](https://github.com/AladdinPersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/model.py)、[v1 loss](https://github.com/AladdinPersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/loss.py)、[v3 train](https://github.com/AladdinPersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/train.py)、[Ayoosh](https://github.com/ayooshkathuria/pytorch-yolo-v3)。

對我們大綱的具體修訂建議：

- 前半主線：NN／必要數學短暖身 → 小型 VGG 風格 CNN → 訓練診斷 → 小型 ResNet → 單物件分類＋定位 → 多物件數量、背景、配對、重複框 → grid MiniYOLO。
- CNN 章由影像局部性與重複性提出動機；分類聚合空間資訊的方式，作為後續定位問題的伏筆。
- ResNet 先做 forward 接線，再在相近條件下比較 plain／residual，分開「會實作」與「理解效果」。
- 偵測前先以人工框練習座標、IoU、assignment 與 NMS，減少首次訓練時混在一起的錯誤。
- 每章保留前版模型、同一套資料切分與固定困難場景：小物件、同格多物件、極端長寬比、重疊物件、背景誤報。
- 每章一個主要改動、一項對照實驗、一份簡短紀錄；量測精度、收斂、時間、記憶體，也记录新增設定與程式複雜度。
- 實驗分為管線驗證、控制場景、真實資料評估。能記住小樣本或在玩具資料成功，不能取代泛化與座標／評估檢查。
- 每個公式附符號、tensor shape、小數值例子與程式對應；完整推導可跳讀。
- 每章標示「歷史機制／教學簡化／實驗改動」，避免把簡化 MiniYOLO 冒充官方模型。
- 用可跑 notebook 看圖與探索，用少量清楚的 Python 模組保存模型及訓練流程。版本演化仍以各分支論文與官方實作另行核實。

查核限制：Michigan WI2022 的主頁、index、schedule、assignment5 本輪皆回 HTTP 503，保留為待查來源，不引用其作業細節。本輪核對文字、程式與課表，未執行參考專案、評測其模型，亦未逐一觀看完整課程影片。

原始抓取紀錄及文字摘錄保存在 `/tmp/yolo-curriculum-research/`；未修改 `learn_to_yolo` 儲存庫的程式、教材或設定。
