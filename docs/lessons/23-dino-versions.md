# E.23.1 DINO 版本：整圖相近之外，局部關係怎麼保留？

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/23-dino-versions.ipynb){ .md-button }

[上一節](22-features.md)用一個 CLS 向量讀整圖顏色。若改問「第一張圖這一塊，在第二張圖哪裡最接近」，一個整圖向量就不能指出位置；我們需要每個 patch 的表示。**同一套影像特徵，怎麼分別服務整體比較與局部對應？**

先看一份已保存的官方模型結果，再理解後續 DINO 版本為什麼也關心局部特徵。讀網頁不必下載權重；頁首 notebook 的預設程式只做後面的手工關係算例。

## 同樣的物件換位置，問整體與局部兩個問題

下面兩張 224×224 RGB 圖都由程式畫出紅方塊、綠圓形、藍三角形，只把它們移到不同位置。這些材料方便確認位置對應，沒有用自然照片來評測模型品質。

![官方 DINOv2 的受控輸入與 patch 近鄰](../assets/diagrams/23-dinov2-patch-neighbor.svg)

這份保存結果來自官方 **DINOv2 ViT-S/14，沒有 register 的版本**。它是已訓練好的模型，與我們從隨機權重訓練的 TinyViT 不同；本專案只固定權重做前向提取特徵。S 表示此系列的小型模型，14 是 patch 邊長。224×224 因此排成 16×16，共 256 塊，每塊輸出 384 維特徵。

先看整體：兩張圖的 CLS cosine 約 0.9850，表示這兩份整圖向量方向很接近。這不指出哪個物件在哪裡，也不能推成所有物件重排都不影響 CLS。

再看圖中 query：第一張圖 `(row=4,col=4)` 的 patch，列、欄從 0 起，落在紅方塊內。用它的特徵與第二張圖全部 256 塊做 cosine 比較，由大到小排序。top-1 在 `(11,11)`，相似度約 0.8331，落在第二張紅方塊；top-2 在 `(4,4)`，約 0.7685，卻落在綠圓形。

每個 patch 都有向量，所以可以把相近程度放回第二張圖的網格，詢問局部對應。但最高的一塊對上，還不是可靠的物件辨識證據；top-2 已顯示錯誤候選，而且這組材料把顏色與形狀綁在一起，無法單獨歸因於其中一項。cosine 也不是類別機率或 IoU。

register 是額外加入、不直接對應某塊圖片的 token。本例選用沒有 register 的模型。官方 `forward_features` 的輸出與前章相似，但字典名稱不同：

|特徵與官方 key|本例 shape|讀取的粒度|
|---|---|---|
|CLS：`x_norm_clstoken`|`[2,384]`|每張圖一個整體向量|
|patch：`x_norm_patchtokens`|`[2,256,384]`|每張圖每塊一個向量，可排回 16×16|
|register：`x_norm_regtokens`|`[2,0,384]`|本模型沒有 register，數量為 0|

這些 `x_norm_*` 是官方最後 LN 的輸出；算 cosine 時還另將向量除以自身長度。patch 的位置槽仍對應原圖格子，但內容已經過 attention，並不是只含那塊的原始像素。

## 從同圖一致，到也教局部位置

第 22 章用原始 **DINO（self-distillation with no labels，無標籤自蒸餾）**的簡化核心，讓整圖 CLS 的訓練輸出跨 view 對齊。逐 patch 特徵也會隨 backbone 更新，但那個 loss 沒有為每個 patch 另設目標。

DINOv2、DINOv3 繼續處理「學一套可交給下游的影像表示」。下面不是只改一行的對照：資料、規模、訓練時間與其他方法都改了，論文成績不能全部歸因於表中的單一設計。本書沒有重訓官方 DINOv2 或 DINOv3。

|方法|如何安排整圖與局部的學習|回應的需要與代價|
|---|---|---|
|原始 DINO|整圖 CLS 跨 view 對齊；teacher 讀 global，student 也讀 local views|從無標籤圖片學表示；多 view 增加前向計算，仍需處理塌縮|
|DINOv2|保留整圖任務，另遮住 student 的部分 patches，讓它預測 teacher 在那些位置的輸出|讓局部表示也接受位置層級的學習要求；搭配更大資料與額外訓練項目|
|DINOv3|長訓練後段加入 Gram anchoring，約束 patch 之間的相似關係|回應論文觀察到的局部特徵品質退化；增加參考模型與關係計算|

DINOv2 的 patch 任務中，teacher 看未遮住的圖，student 從剩下的內容預測被遮位置的分佈。這與只要求 CLS 一致不同：局部位置也有對應的目標。DINOv3 關心的「patch 關係」又是另一個約束，先用小例子看它保留什麼。

## 三塊之間的關係，可以排成一張表

假設三個 patch 已有兩維特徵，並各自除以長度，使其長度為 1。這些是方便手算的指定向量，不是官方模型的物件答案，兩維也不是紅／藍類別。

|patch|特徵向量|方向關係|
|---|---|---|
|A|`[1,0]`|沿第一個方向|
|B|`[0,1]`|沿第二個方向|
|C|`[0.7071,0.7071]`|位於兩方向之間|

兩兩取內積，便得到相似關係。A 與 B 的內積為 0，A 與 C 約 0.7071；因為各向量長度為 1，這裡就是 cosine similarity。

![三個手工 patch 向量的 Gram 關係表](../assets/diagrams/23-gram-relations.svg)

列、欄都是 A、B、C，每格表示該對 patch 的相似程度。將向量排成矩陣 F，**Gram 矩陣**就是 `F @ F.T`。它記錄的是特徵之間的關係；與 attention 不同，沒有 softmax、不要求列和為 1，也沒有再乘 V 來混合內容。

## 保留關係，與逐維抄數字有何不同

現在將三個向量一起旋轉 90 度。每個向量的數字改變，但相互夾角沒變，Gram 表也不變。反過來，若全部換成 `[1,0]`，所有內積都變成 1，原先 A、B 不同、C 介於兩者間的關係就消失。

|改法|與原特徵的平均平方差|與原 Gram 表的平均平方差|
|---|---:|---:|
|三個向量一起旋轉|1.0000|0.0000|
|三個向量都換成 `[1,0]`|本例不以此數字比較|約 0.2603|

``` { .python data-excerpt="lesson_cases/23-dino-versions.py" }
anchor_gram = anchor @ anchor.T
rotated_gram = rotated @ rotated.T
constant_gram = constant @ constant.T
rotated_gram_mse = (rotated_gram - anchor_gram).square().mean()
constant_gram_mse = (constant_gram - anchor_gram).square().mean()
```

`anchor` 是指定的參考特徵，`rotated` 是共同旋轉後的特徵，`constant` 是常數回答。這個比較讓保留對象可見：允許改變特徵座標，只要相同視圖、相應 patch 之間的關係仍吻合。

**Gram anchoring（Gram 關係錨定）**用參考模型的 patch 關係約束 student。當參考表還有局部差異，而 student 把 patches 混成相同表示時，兩張關係表會出現差距，提供保留差異的要求。它不要求每個特徵值照抄，但也不是由這個手算就證明保住了所有語意。

頁首 notebook 可以直接執行這個三向量例子，不下載任何模型、沒有 optimizer 更新。它教的是關係約束，沒有實作 DINOv3 的完整 loss、權重或排程。

??? note "DINOv3 的參考模型怎麼來"

    [DINOv3 第 4.2／4.3 節與附錄 C](https://arxiv.org/abs/2508.10104v1) 用較早期 EMA teacher 的 snapshot 作 Gram teacher；refinement 在主訓練後段開始，並定期用當前 EMA teacher 刷新。較高解析度 Gram teacher 的特徵還會降採樣到 student 網格。

    它不是永久抄最初模型的向量。本頁指定三個特徵只說明「關係可以不同於逐維數字」，沒有重現長訓練退化、refresh 或高解析度實驗。

## 要自己取官方特徵，再下載這一份權重

看保存結果與做手工算例都不需要下載。若要重做前面的官方 DINOv2 操作，先跑本頁 notebook 的環境格，再另開一格：

```python
!python scripts/run_dino_pretrained.py --report artifacts/runs/dinov2-pretrained/report.json
```

這個選讀腳本載入的非 register ViT-S/14 有 22,056,576 個參數，權重約 84.2 MiB。第一次需要網路下載，不需要 token 或 GPU。腳本由固定官方 commit 取得程式，核對程式壓縮檔與權重 SHA-256 後載入。

RGB 值先轉成 `[0,1]`，resize 到 224×224，再依 ImageNet mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]` 逐通道正規化。本例圖片原本就為此尺寸，沒有裁切、空間尺寸不變。

權重、官方程式、輸入圖、NPZ 特徵與報告寫到不追蹤的 `artifacts/runs/`，不放進普通 Git 或 Pages。[保存的 CPU 實跑紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/dinov2-pretrained.json) 可核對來源、SHA、機器與計時範圍；它是特徵提取，不是自然影像品質排名，也沒有 DINOv3 實跑。

## 改一個關係，先預測

1. 將手工 C 改成 `[1,0]`，Gram 表哪些列、欄會改？A 與 C 還只有約 0.7071 的相似嗎？
2. 同時交換 F 的兩個特徵通道，Gram 表會變嗎？若只交換 patch A、B 的順序，列與欄如何移動？分清改特徵座標與改位置順序。
3. 官方例子已有 top-2 找錯物件。要主張可靠局部對應，還需要哪些圖片、變化與判準？一組最高相似度為何不足？

修改手工例子時，既有 assert 核對的是原題，要依新題更新期望值再比較。

??? note "讀官方資料時會遇到的其他名詞"

    **iBOT** 是 Image BERT Pre-Training with Online Tokenizer（用線上 tokenizer 做影像 BERT 預訓練）。它的線上 tokenizer 是訓練中的 teacher，替圖片位置提供目標；DINOv2 使用其遮住 patch 的任務，並為整圖與 patch 任務使用分開的 heads。

    **Sinkhorn–Knopp** 對一批 teacher 輸出共同平衡，DINOv2 用它取代原始 DINO 的 teacher softmax-centering。**KoLeo** 名稱來自 Kozachenko–Leonenko 熵估計，是鼓勵圖級特徵分散的正則項。本書都沒有實作。

    **Register tokens** 是額外、沒有直接對應圖片 patch 的 tokens，來自後續〈Vision Transformers Need Registers〉。官方 DINOv2 同時有 register 與非 register 模型，不能把所有 DINOv2 都寫成有 register。

    本頁 DINO 系列都指自監督表示學習；[DINO detector](https://arxiv.org/abs/2203.03605) 是另一個 DETR 路線的偵測方法。

原始來源：

- [DINO，2104.14294v2](https://arxiv.org/abs/2104.14294v2)：§3.1、Algorithm 1 的 multi-crop、teacher／student 與 centering／sharpening。
- [DINOv2，2304.07193v2](https://arxiv.org/abs/2304.07193v2)：第 4 節訓練方法；選讀腳本固定到[官方 commit](https://github.com/facebookresearch/dinov2/tree/e1277af2ba9496fbadf7aec6eba56e8d882d1e35)。
- [Vision Transformers Need Registers，2309.16588](https://arxiv.org/abs/2309.16588)：register 的模型修改與實驗。
- [DINOv3，2508.10104v1](https://arxiv.org/abs/2508.10104v1)：§4.2／4.3 與附錄 C 的 Gram teacher、refinement、refresh。

整圖與 patch 特徵已經分清；最近的 patch 仍不是物件框。[下一節](23-detection-bridge.md)回到自己的 TinyViT，保留 patch 網格，讓有標籤的 head 學出一個框。

[下一節：23.2 凍結特徵接回定位](23-detection-bridge.md)

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/23-dino-versions.json)

??? example "展開本次實際輸出"

    ```text
    {
      "mode": "hand-specified normalized patch relations; no optimizer, no downloads",
      "anchor_features": [
        [
          1.0,
          0.0
        ],
        [
          0.0,
          1.0
        ],
        [
          0.7071067690849304,
          0.7071067690849304
        ]
      ],
      "anchor_gram": [
        [
          1.0,
          0.0,
          0.7071067690849304
        ],
        [
          0.0,
          1.0,
          0.7071067690849304
        ],
        [
          0.7071067690849304,
          0.7071067690849304,
          0.9999999403953552
        ]
      ],
      "rotated_feature_mse": 1.0,
      "rotated_gram_mse": 0.0,
      "constant_gram": [
        [
          1.0,
          1.0,
          1.0
        ],
        [
          1.0,
          1.0,
          1.0
        ],
        [
          1.0,
          1.0,
          1.0
        ]
      ],
      "constant_gram_mse": 0.26034948229789734,
      "scope": "Mean squared Gram difference illustrates relation preservation; not the complete DINOv3 loss or schedule."
    }
    ```

<!-- curriculum-evidence:end -->
