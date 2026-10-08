# E.22.1 收起紅藍答案，同一張圖還能提供什麼學習訊號？

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/22-views.ipynb){ .md-button }

[前一節](21-training.md)靠紅／藍答案教 ViT。現在假設只有圖片，沒有人工寫好的類別答案，但仍想先學一套能交給後續任務的影像特徵。**圖片本身，有沒有我們已經知道、可以拿來教模型的關係？**

先從最容易取得的關係開始：把同一張圖做成兩種看法，我們不用知道它的類別，也知道這兩份材料來自同一原圖。

## 同一張圖，做出不同看法

材料沿用 32×32 RGB 的紅／藍矩形規則。本節另外生成一張紅矩形來展示資料操作，不是直接取前章 128 張 train 的圖 0。

從原圖各抽一次較大的正方形區域，縮放回 32×32，再各自決定是否左右翻轉、調整亮度。得到的兩張衍生圖片叫 **views（視圖）**：它們不是完全相同的像素，卻保留一些共同內容。

![同一張32×32矩形生成兩個global view，並顯示原圖上的裁切區域](../assets/diagrams/22-views.svg)

看原圖上的兩個裁切範圍，再看它們各自變成的 view。A、B 來自同一張圖，不是找兩張恰好都是紅色的圖片配對。圖下方還有一個只看左上背景的小裁切：它提醒我們，操作也可能把要保留的物件裁掉。

圖中的 crop 座標是**原圖像素** `[x1,y1,x2,y2]`，依序表示左、上、右、下邊界。原點在左上，x 向右、y 向下，右與下邊界不包含在裁切中。縮放、翻轉後，view 內的位置已改變，原圖框不能直接當成 view 的框答案。這裡只用 crop 記錄操作，不拿框來訓練。

這張實際示例以 seed 101 生成 1 張圖、取該單張資料的索引 0，view seed 為 22。A、B 的 crop 是 `[1,1,29,29]`、`[2,1,29,28]`。下方反例是 `[0,0,8,8]`，不加入後面的兩個大 view 訓練。

## 希望特徵保留什麼，決定怎麼改圖片

本例抽原圖面積比例約 0.65 到 1 的正方形 crop，稱為 **global view（全局視圖）**：它看到較大範圍。程式先抽面積比例、求邊長並四捨五入，再抽位置；小圖取整後，實際面積比例會略有不同。

用大 crop 是為了讓兩份看法較可能保有矩形內容；它沒有讀 box，也沒有保證物件必定留下。矩形靠角落時，仍可能只剩一部分或完全裁掉。若兩份 view 都只剩背景，要求它們一致，就不能教模型保留這次已被裁去的矩形。前面的反例正是這個限制，不能把任意裁切都當成不改內容。

亮度變化以同一倍數調整 RGB，沒有交換紅藍通道、改色相或轉灰階。因為後續還要檢查紅藍分類，這裡希望減少對亮度與翻轉的依賴，卻保留顏色差異。若把顏色線索抹掉再要求一致，就會與這個用途衝突。

因此，沒有類別標籤的 loss 仍需要人選擇合適的資料操作；「沒傳 labels」不代表對用途沒有任何判斷。

## 把已知的同圖關係，變成學習要求

沿用前章的 ViT 讀圖架構：patch 投影、CLS 與位置向量、兩個 Transformer blocks，得到每張圖的 32 維 CLS。產生這些特徵的主體稱為 **backbone（骨幹網路）**；後面的頭再依任務把特徵變成答案。

讓同圖兩個 view 各走一次 backbone，要求兩份表示相近，叫 **view consistency（跨視圖一致性）**。位置、亮度或裁切可以不同，仍要求模型保留兩份看法共同的內容，讓後續讀取器不必完全依賴某一次像素排列。

這個要求只知道哪兩個 view 來自同一圖；它沒有直接告訴模型「兩張不同原圖都屬於紅色」，也沒有提供其他類別答案。以資料自身可取得的關係當目標，叫 **self-supervised learning（自監督學習）**。前章的 supervised learning（監督學習）用人工類別答案，現在換的是學習目標來源。

本章選用 **DINO（self-distillation with no labels，無標籤的自蒸餾）**來實現這件事。「蒸餾」是用一個模型的輸出教另一個模型；「自」表示教學目標也由訓練中的模型逐步形成。teacher（教師）提供目標，student（學生）學著回答，teacher 的權重再緩慢追蹤 student。它不是一份早已知道紅藍答案的教師模型。

DINO 會把 CLS 經投影頭變成訓練用分佈，再比較不同 view；[22.3 節](22-distillation.md)負責拆開這個更新。訓練後可保留 backbone，讓分類或定位等任務使用特徵，這些下游任務仍可能需要自己的標籤。

我們沿用的是 ViT 架構，後面的自監督實驗會重新從隨機權重出發，沒有載入 21.4 已看過紅藍標籤的模型。這樣才可以把「無標籤學特徵」與前章有標籤的訓練分開。

## 程式先只準備輸入

本節不更新模型，只生成 views。下面兩次呼叫都只接收圖片，沒有 labels 或 boxes：

```python
import torch
from miniyolo.self_distillation import random_view

generator = torch.Generator().manual_seed(1007)
view1, crop1 = random_view(images, generator)
view2, crop2 = random_view(images, generator)
```

`images` 是 `[B,3,32,32]`、值介於 0 和 1，B 為原圖張數。兩批 views 也是這個 shape；兩份 crops 各為 `[B,4]`，記錄原圖上的像素邊界。`generator` 保存亂數序列的位置，同一種子與同樣呼叫順序可重做同一批操作。

執行 `PYTHONPATH=. python lesson_cases/22-views.py` 或頁首 notebook，核對 crop、view shape，以及恢復 RNG 狀態後能否重做 views。輸出回答的是「資料怎麼做出來」，沒有證明特徵已經學好。

現在的學習想法還有一個漏洞：若每張圖都輸出同一向量，任何兩個 view 不也一致嗎？[下一節](22-collapse.md)先把這個失敗算出來，再決定訓練還要處理什麼。

## 改一種增強，預測它會教什麼

若有一半機率交換紅、藍通道，再要求同圖兩個 view 表示一致，對後來的紅藍分類可能有什麼影響？

??? note "參考答案"

    同一張紅矩形可能變成一紅一藍兩份 view。一致性要求會鼓勵忽略紅藍差異，與顏色分類的用途衝突。這不表示所有任務都不能用色彩增強，而是要對照希望保留的內容。

??? note "原始 DINO 為什麼還有局部 view"

    [DINO 論文](https://arxiv.org/abs/2104.14294) §3.1 的 multi-crop 包括兩個 global views 與多個較小的 local views。teacher 只看 global，student 也看 local，讓局部看法朝較完整的看法學習。固定官方 [DataAugmentationDINO](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py) 可查其實際增強流程。

    本書只做兩個 global views，沒有 multi-crop、自然影像或原版大量訓練。DINO 指 2021 年的自監督方法，與同名的 DETR 路線 [DINO detector](https://arxiv.org/abs/2203.03605) 是不同模型。

[上一節：21.4 有標籤訓練](21-training.md) · [下一節：22.2 一致也可能沒有資訊](22-collapse.md)

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-views.json)

??? example "展開本次實際輸出"

    ```text
    {
      "source_seed": 101,
      "source_sample_index": 0,
      "source_samples": 1,
      "view_generator_seed": 22,
      "same_rng_reproduces_views": true,
      "input_shape": [
        1,
        3,
        32,
        32
      ],
      "source_bbox_xyxy_pixel": [
        14,
        10,
        22,
        22
      ],
      "global_a_crop_xyxy_pixel": [
        1,
        1,
        29,
        29
      ],
      "global_b_crop_xyxy_pixel": [
        2,
        1,
        29,
        28
      ],
      "local_crop_xyxy_pixel": [
        4,
        11,
        19,
        26
      ],
      "empty_crop_xyxy_pixel": [
        0,
        0,
        8,
        8
      ],
      "empty_crop_object_intersection_pixel2": 0,
      "view_shape": [
        1,
        3,
        32,
        32
      ],
      "ssl_inputs": "images only; labels/bboxes excluded",
      "role": "official DINO: teacher gets global; student gets global+local; later tiny trainer uses only 2 globals",
      "limitation": "same source does not guarantee a crop contains the object; views are not labeled detection examples"
    }
    actual views: artifacts/runs/dino/22-views.svg
    ```

<!-- curriculum-evidence:end -->
