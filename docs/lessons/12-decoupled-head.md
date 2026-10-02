# 12.2 Decoupled head：分類和定位在哪裡分工

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/12-decoupled-head.ipynb) · 原始碼：`lesson_cases/12-decoupled-head.py`

前置是 Conv2d、反向傳播與多項 loss。backbone 是影像轉特徵的主幹，head 是把特徵轉成預測的輸出模組。現在同一特徵要回答兩個問題：「這裡是哪一類？」和「框的邊在哪裡？」分類常需要辨識形狀與語意，定位要保留精細位置。把兩者全塞進同一串卷積，可能限制各自能學的轉換。本節把 head 分成兩串卷積，觀察梯度究竟如何走。

歷史機制是 YOLOv8 類設計的分類、框回歸分支分開；本次起點是共享 CNN 特徵，本章簡化成單尺度、連續四邊距離、兩個 sigmoid 類別 logits。實驗後可以保留分支結構，但觀察只支援梯度隔離和成本描述，不能宣稱真實資料精度提高。

## 先把各軸說清楚

兩張影像輸入 `[B,3,H,W]=[2,3,4,4]`。共享卷積產生 `[2,8,4,4]`，之後每個分支各有一層 `3×3,8→8` 與一層 `1×1` 投影。

| 路徑 | 最後 shape | 每個位置表示什麼 |
| --- | --- | --- |
| box branch | `[2,4,4,4]` | 四個 channel 為 `ltrb` 距離 logits |
| class branch | `[2,2,4,4]` | 兩個 channel 各是類別 logit |
| shared backbone | `[2,8,4,4]` | 兩個任務共同讀取的特徵 |

box shape `[2,4,4,4]` 的 axis1 是四個channel，axis2、axis3纔是高度、寬度；B=2代表兩張圖。ltrb 是點到框左、上、右、下四邊的距離。程式的 `boxes` 變數其實是 raw logits，可正可負，經 `F.softplus(boxes)` 才變為正距離，再與1.5格的target比較，並非已解碼的xyxy框。在這個演示中所有位置都是正樣本，class0 target 為 1、class1 為 0；刻意省去 assignment（候選與真值的責任分配），讓我們只追蹤 head 的梯度。完整偵測器只在選定正樣本上計算框 loss，背景仍供應分類負訊號。

## 三次 backward 解開「分開」的意思

第一次只反傳 box loss：box branch 參數有梯度，class branch 的 `.grad` 應是 `None`。第二次清空梯度，只反傳 classification loss：結果反過來。兩次都會到共享 backbone，因為兩條路都使用同一份特徵。因此 decoupled 不是兩個完全獨立網路，也不保證共享特徵沒有任務衝突。

```python
box_loss.backward(retain_graph=True)
assert model.class_branch[0].weight.grad is None
box_backbone_grad = model.backbone.weight.grad.clone()
model.zero_grad()
cls_loss.backward(retain_graph=True)
class_backbone_grad = model.backbone.weight.grad.clone()
```

第三次反傳兩者相加。鏈式法則告訴我們共享參數 `θ` 的梯度是 `∂Lbox/∂θ + ∂Lcls/∂θ`；程式逐元素 assert 驗證。接著 optimizer.step，確認 backbone 權重改變。沒有 step，便只知道圖可反傳，還沒驗證訓練會更新。

以向量內積看兩份梯度方向：cosine 接近 1 表示這批資料上較一致，接近 −1 表示較相反，接近 0 表示大致正交。程式會印出一次實測值；它是固定 seed 的人工 target 診斷，不能代表整個任務的平均情況。也不能看到負 cosine 就直接宣稱 decoupling 無效，因為分支仍提供各自的特徵轉換。

## 可核對的成本

本例含 bias 的參數共 1,446：backbone 為 `8×3×3×3+8=224`，兩個 `8→8` 的 3×3 卷積各 584，box 投影 36，class 投影 18。若使用一個 `8→6` 的直接 1×1 head，只有 54 個 head 參數；那是容量也不同的基線，不能把效果差異都歸因於分工。公平比較應選相近寬度、層數或成本，並說明對齊哪一項。

執行 `PYTHONPATH=. python lesson_cases/12-decoupled-head.py`，必須印出 box `(2,4,4,4)`、class `(2,2,4,4)`，單任務不流入另一分支、總梯度等於分項之和及參數數量。梯度 cosine 可作檢查訊號，沒有通用的理想數字。本例只需 CPU。

## 收益、代價與下一項判斷

分支讓定位與分類使用不同的後段濾波器，也能分開設計寬度及 loss；代價是額外卷積、activation 記憶體和需要調整的容量。當 backbone 已經很小，增加兩條寬分支可能比共享 head 更耗時；當資料很少，更多參數也可能過擬合。速度必須實測整條推論路徑，不能只看分支數。

常見錯誤是忘記在兩次 backward 之間清梯度，導致第二次看到的是累加；把 `.grad=None` 與零 tensor 混為一談；或對其中一份 feature 做 `detach()`，意外切斷它對 backbone 的訓練。本節兩個任務都需要訓練共享特徵，不在分支前做detach。

自主練習：只將分類 loss 乘 2，再驗證總 backbone gradient。答案為 `gbox+2*gcls`，不是「只有 class branch 梯度變兩倍」；共享 backbone 也會接收較強分類訊號。接著若想對照 coupled head，先寫下對齊參數還是對齊延遲，再用相同資料與訓練步數比較。

來源查覈：2026-10-02。[Ultralytics Detect 中 cv2／cv3 的分支定義](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)。本章的兩層小分支、全正樣本與 Smooth L1 是教學簡化。

練習的兩處要一起改：總反傳寫`(box_loss+2*cls_loss).backward()`，預期backbone梯度寫`box_backbone_grad+2*class_backbone_grad`；輸出說明也同步。不先把分項gcls乘2，以免再乘一次。coupled head是共享末端卷積，再一次輸出box與class；decoupled則在末端分路。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-decoupled-head.json)

??? example "展開本次實際輸出"

    ```text
    box / class shapes: (2, 4, 4, 4) (2, 2, 4, 4)
    classification-only backward leaves box branch grad=None
    shared-backbone gradient cosine: -0.0073
    total backbone gradient = box gradient + class gradient: verified
    parameters: 1446
    ```

<!-- curriculum-evidence:end -->
