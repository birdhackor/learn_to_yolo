# E.23.2 保留 patch 的位置，讓新 head 學顏色與框

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/23-detection-bridge.ipynb){ .md-button }

[22.4 節](22-features.md)只讓讀取器回答整圖顏色。現在增加一個要求：「紅或藍之外，矩形在哪裡？」**能否保留自監督學到的讀圖主體，讓一個新 head 學位置答案？** 這次使用每個 patch 的特徵，保留它們的空間排列。

## 同一張圖，多了四條邊的答案

材料仍是 32×32 RGB，每張只有一個紅色或藍色矩形。train／validation／test 為 128／64／64 張，seed 為 101／202／303。下面是四張實際 test 圖，以及這次模型得到的框。

![四張實際test圖上的真值與預測框，包含類別正確但框未達IoU0.5的失敗](../assets/diagrams/23-detection-predictions.svg)

先看左上 test 0：圖片內容是藍矩形；綠實線標真值，橘虛線是預測。綠橘用來區分答案與預測，不是新的物件類別。模型把顏色答對了，框卻偏離矩形，兩種工作需要分開計分。

框用 **xyxy** 表示左、上、右、下四條邊。原點在左上，x 向右、y 向下，test 0 的真值 `[9,4,25,12]` 以像素為單位，寬 16、高 8。這接續[4.1 的定位](04-localization.md)與[6.2 的 IoU](06-evaluation.md)；本節仍會把使用的座標與配對方式說清楚。

先自監督、再學框：本頁程式自行從隨機權重做 160 步小型 DINO，只讀 train images。然後凍結 teacher backbone，才用 train labels 與 boxes 訓練新 head。沒有下載 DINOv2／DINOv3，也不依賴上一頁的 checkpoint。validation 只報結果、不挑設定；test 不進參數更新或標準化統計。

## 用位置順序，還原一張 patch grid

TinyViT 的 `forward_features` 同時保留 CLS `[B,32]` 與 patches `[B,16,32]`。CLS 適合讀整圖；現在想利用每個區域與原圖格子的對應，所以拿後面的 16 個 patch 向量，排回 4×4。

順序仍是由上到下、每列由左到右。先交換 patch 數與特徵數兩軸，得到 `[B,32,16]`，再把 16 拆成 4×4，得到 `[B,32,4,4]`。這張 **patch grid（小塊特徵網格）**與 CNN feature map 一樣，以每個空間格存放一組通道特徵；這一步只改排列。

```python
patches = extract_features(backbone, images, kind="patches")  # [B,16,32]
feature_map = patch_feature_map(patches)                     # [B,32,4,4]
```

第 `(row,column)` 格仍是原圖相應 8×8 patch 的槽位。它經過 attention 後也含其他區域資訊，不能當成只讀這 8×8 像素的原始局部特徵。但「這個向量留在哪個位置」仍提供空間對應，讓新的讀取器利用排列學框，並不是已經得到框答案。

![凍結ViT的16個patch還原4×4grid，再用有標籤的單物件head輸出顏色和一個框](../assets/diagrams/23-detection-bridge.svg)

上半部只做凍結 backbone 的前向，下半部的新 head 才更新。每張圖最後只有一個預測框；網格 16 個位置不是 16 個框槽。這裡沒有候選責任分配或 NMS，先研究「特徵如何交給一個定位任務」。

## Head 讀整個網格，分兩路回答

與前頁線性探測一樣，先只用 train 特徵估統計，但這次對**每個網格位置、每個通道**分別估平均與標準差，最小標準差取 0.05。val／test 都沿用同一份 train 統計。

把每張圖的 `32×4×4=512` 個值按固定順序攤平，經 `Linear(512,64)` 與 GELU，得到共用的 64 維表示。雖然攤平，每個槽位仍有固定順序，head 可以對不同位置使用不同權重；它沒有把網格先平均成無位置的一個值。

共用表示分兩路：`Linear(64,2)` 輸出紅、藍 logits；`Linear(64,4)` 給框的四個 raw 值。框路先 sigmoid，前兩個作中心 `cx,cy`，後兩個再轉成寬高：`size=0.05+0.7×sigmoid(raw)`。

這些數字是**原圖邊長的比例**，不是 grid 格數；例如 `w=0.5` 對應 32×0.5=16 像素。中心與寬高換成四條邊：

\[
\begin{aligned}
x_1&=c_x-w/2,& y_1&=c_y-h/2,\\
x_2&=c_x+w/2,& y_2&=c_y+h/2.
\end{aligned}
\]

邊界再截到 0～1，乘 32 才變回圖中的像素框。截邊可能改變最終中心與尺寸；截邊前的預測寬高限制在 0.05～0.75，是配合本例 8～16 像素矩形的簡化設定，不能原樣表示所有大框。

這個 head 看整個網格，一次回歸全圖唯一物件，不採「某 patch 負責某框」的解碼規則。

## 用相同座標空間，計算顏色與框的誤差

真值也除以 32。test 0 的 `[9,4,25,12]` 因此變成 `[0.28125,0.125,0.78125,0.375]`，與模型的 normalized xyxy 位於同一空間。

顏色用交叉熵，框選用 **Smooth L1（平滑 L1 loss）**，示範另一個常見的框回歸 loss；這次先固定一組可執行的設定，重點是檢查新 head 能否讀凍結特徵學框。

Smooth L1 比較四條邊，再取 batch 與座標平均。對單一誤差 e，預設切換門檻 β=1 時，`|e|<1` 算 `0.5e²`，否則算 `|e|−0.5`。這裡兩邊座標皆為 0～1，誤差最多為 1，實際在平方區或兩段接點，等於這四條邊的 MSE 的一半；一般的大誤差線性區並不是這個例子主要用到的部分。

總誤差為

\[
L=L_{\mathrm{color}}+10L_{\mathrm{box}}.
\]

10 是本例固定的權重，用來調整兩項在總 loss 的份量，不是 DINO 或 YOLO 的預設。以 Adam、學習率 0.003、batch 32 訓練 head 200 步，head 初始化 seed 為 901。

backbone 的特徵先算好且不帶梯度。程式檢查 head 梯度有限、整體非零，權重確有變化，同時確認 backbone 沒有梯度、權重前後完全相同。這證明更新的是新的顏色與框讀取器。

## 按同一圖片配對，看框是否真有對上

每圖固定一個真值、一個預測，直接按圖片配對。顏色取 `argmax`；框以 **IoU（Intersection over Union，交集面積÷聯集面積）**衡量重疊。最後數有多少張同時「顏色答對且 IoU≥0.5」。

這個張數比例沒有按置信分數排序，也沒有 precision／recall 曲線，不是 AP50 或 mAP。若未讀多物件評估，先把它當成每張圖的一個成功條件即可。

保存的 CPU head 訓練 loss 從 0.8861 降到 0.000967。接著固定權重評三批完整圖片，結果如下。

手機上可左右滑動表格，查看完整欄位。

|切分|顏色答對|平均配對 IoU|顏色對且 IoU≥0.5|
|---|---|---|---|
|train|128／128|0.8594|128／128|
|validation|64／64|0.6160|49／64|
|test|64／64|0.6057|54／64|

回到頁首同一個 test 0：預測藍色，框約 `[13.05,0,23.61,13.82]` 像素，真值為 `[9,4,25,12]`。框右移且包含物件上方背景，IoU=0.4459，沒有達到 0.5。test 1 的 0.3522 也失敗；test 2、3 的 0.5611、0.6326 則過門檻。圖與數字配在同一張圖上，顏色全對沒有掩蓋定位失敗。

這支持凍結的小型自監督 patch 特徵可接有標籤的單物件 head，並在同規則新圖得到表中結果。沒有 random backbone 的框 head 對照，也沒有重訓 YOLO，不能判成 DINO 比 random 或 CNN 更適合偵測。自然照片、多物件及背景誤報都沒有測試。

## 保存推論需要的整條路

執行 `PYTHONPATH=. python lesson_cases/23-detection-bridge.py` 或頁首 notebook，會自行完成短 SSL、head 訓練與評分。產物不只保存 head，還包括 backbone、模型設定、train 特徵平均與標準差及類別名稱；新圖要走相同特徵提取和標準化，再由相同順序的分類與框路輸出。

完整設定與配對框可查[實際執行紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/23-detection-bridge.json)。如果只保存 head，卻換掉特徵或用新圖重估統計，就不是這次訓練好的同一條推論路徑。

## 第二個矩形出現時，缺的是什麼

同一張圖再放第二個矩形，這個 head 能同時輸出兩個框嗎？加 NMS 能解決嗎？

??? note "參考答案"

    不能。框路固定只給四個數，只有一個框槽。NMS 只能刪已有候選，不能製造第二個框。要做多物件，須改輸出結構、責任分配及資料與評分方式；可回到第 7 章的 grid head，思考如何接這份 patch grid。

本支線把工作分成三段：ViT 提供整圖與逐位置表示，DINO 在沒有類別標籤時訓練它，最後由適合任務的有標籤 head 讀出答案。這個框 head 是本書的橋接設計，並不會因為 backbone 用 DINO 訓練，就變成官方同名偵測器。

??? note "兩個 DINO 名稱與本例來源"

    [2021 自監督 DINO](https://arxiv.org/abs/2104.14294) §3.1 說明取 backbone 作下游特徵，§3.2 交代評測。本書的單物件 head 沒有重現其偵測或分割實驗，也不是官方 DINOv2／DINOv3 的效能測試。

    [2022 DINO detector](https://arxiv.org/abs/2203.03605) 是 *DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection*，官方專案為 [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO/tree/d84a491d41898b3befd8294d1cf2614661fc0953)。它的偵測機制與本章 teacher、center、投影槽不是同一回事。

[回到凍結特徵評測](22-features.md) · [回到版本與預訓練權重](23-dino-versions.md) · [回到多物件 grid 責任分配](07-targets.md)

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/23-detection-bridge.json)

??? example "展開本次實際輸出"

    ```text
    {
      "ssl_steps": 160,
      "head_steps": 200,
      "device": "cpu",
      "split": {
        "train": {
          "samples": 128,
          "seed": 101
        },
        "val": {
          "samples": 64,
          "seed": 202
        },
        "test": {
          "samples": 64,
          "seed": 303
        }
      },
      "ssl_first_loss": 1.9427120685577393,
      "ssl_last_loss": 1.8742694854736328,
      "feature_shapes": {
        "patches": [
          128,
          16,
          32
        ],
        "map": [
          128,
          32,
          4,
          4
        ]
      },
      "head": "Flatten512 -> Linear64/GELU -> class Linear2 + box Linear4; sigmoid center, size=.05+.7*sigmoid; clamp decoded xyxy to [0,1]",
      "head_learning_rate": 0.003,
      "head_batch_size": 32,
      "head_seed": 901,
      "loss": "cross_entropy(class) + 10*smooth_l1(normalized_xyxy)",
      "first_head_step": {
        "step": 1,
        "loss": 0.8860688209533691,
        "class_loss": 0.6687704920768738,
        "box_loss": 0.021729834377765656
      },
      "last_head_step": {
        "step": 200,
        "loss": 0.0009669912979006767,
        "class_loss": 1.56007481564302e-05,
        "box_loss": 9.513905388303101e-05
      },
      "backbone_unchanged": true,
      "backbone_has_no_grad": true,
      "head_weights_changed": true,
      "metrics": {
        "train": {
          "count": 128,
          "class_correct": 128,
          "class_accuracy": 1.0,
          "mean_iou": 0.8593783974647522,
          "iou_ge_0_5_count": 128,
          "class_correct_and_iou_ge_0_5_count": 128
        },
        "val": {
          "count": 64,
          "class_correct": 64,
          "class_accuracy": 1.0,
          "mean_iou": 0.616032600402832,
          "iou_ge_0_5_count": 49,
          "class_correct_and_iou_ge_0_5_count": 49
        },
        "test": {
          "count": 64,
          "class_correct": 64,
          "class_accuracy": 1.0,
          "mean_iou": 0.605698823928833,
          "iou_ge_0_5_count": 54,
          "class_correct_and_iou_ge_0_5_count": 54
        }
      },
      "examples": [
        {
          "test_index": 0,
          "source_seed": 303,
          "truth_xyxy_pixel": [
            9,
            4,
            25,
            12
          ],
          "predicted_xyxy_pixel": [
            13.051433563232422,
            0.0,
            23.608627319335938,
            13.815743446350098
          ],
          "truth_class": "blue",
          "predicted_class": "blue",
          "iou": 0.4459264874458313
        },
        {
          "test_index": 1,
          "source_seed": 303,
          "truth_xyxy_pixel": [
            10,
            17,
            18,
            27
          ],
          "predicted_xyxy_pixel": [
            13.031064987182617,
            15.76512336730957,
            22.852384567260742,
            26.95350456237793
          ],
          "truth_class": "blue",
          "predicted_class": "blue",
          "iou": 0.35220107436180115
        },
        {
          "test_index": 2,
          "source_seed": 303,
          "truth_xyxy_pixel": [
            17,
            4,
            25,
            17
          ],
          "predicted_xyxy_pixel": [
            17.466459274291992,
            2.6046667098999023,
            28.128145217895508,
            15.542059898376465
          ],
          "truth_class": "red",
          "predicted_class": "red",
          "iou": 0.5610500574111938
        },
        {
          "test_index": 3,
          "source_seed": 303,
          "truth_xyxy_pixel": [
            13,
            16,
            28,
            31
          ],
          "predicted_xyxy_pixel": [
            12.647469520568848,
            19.411531448364258,
            26.247325897216797,
            32.0
          ],
          "truth_class": "red",
          "predicted_class": "red",
          "iou": 0.632573664188385
        }
      ],
      "elapsed_seconds": 3.5390920539939543,
      "evaluation": "one prediction paired with one truth per image; class argmax, paired IoU, joint class-correct and IoU>=.5; no AP/mAP",
      "limitation": "tiny frozen self-supervised ViT plus supervised single-object head; not DINO detector, not official DINOv2 backbone, no architecture ranking or natural-image transfer claim"
    }
    actual boxes: artifacts/runs/dino/23-detection-bridge.svg
    ```

<!-- curriculum-evidence:end -->
