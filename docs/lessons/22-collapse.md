# 22.2 一直回答相同向量，也算兩個 view 一致嗎？

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/22-collapse.ipynb){ .md-button }

上一節把同一張紅色矩形變成兩個 view，要求模型對它們給出相近的表示。現在換到第二張藍色矩形：我們仍希望它的兩個 view 相近，卻不希望所有圖片都變成同一份答案。本節的問題是：**只要求同圖一致，為什麼還不足以得到可用的特徵？**

## 先看四張 view，和兩種回答

材料仍是 32×32 RGB 圖：一張紅色矩形 A、一張藍色矩形 B，各產生 view 1 和 view 2。這裡暫時用只有兩個數的向量手算，不跑 ViT，也不把這兩個數稱為「紅、藍機率」。例如 A 的兩個 view 都得到 `[1,0]`，B 的兩個 view 都得到 `[0,1]`，同圖一致，兩張圖也還有差異。

另一種模型不管看到什麼，都回答 `[1,0]`。A 的兩個 view 一樣，B 的兩個 view 也一樣；可是看到 A 或 B，答案完全沒有差別。這叫 **collapse（塌縮）**：表示把不同輸入壓成同一個結果，丟掉了後續任務可能需要的差異。

![同圖一致的兩種手工回答：保留圖片差異，以及所有圖都塌縮成同一向量](../assets/diagrams/22-collapse.svg)

圖中的箭頭只是「圖片送入模型、得到向量」。上半部保留了 A、B 的差異；下半部四個輸出全部相同。本圖是手工算例，並非訓練後的特徵散點圖。

## 一個零 loss，不能替兩個模型分高下

為了看清漏洞，先用最直接的平方差當一致性 loss：兩個 view 的向量逐項相減、平方，再相加。A 的兩個輸出都為 `[1,0]`，所以差是 `[0,0]`，loss 為 0；B 也是一樣。

| 手工模型 | A 的 view 1／2 | B 的 view 1／2 | 兩張圖的同圖一致性 loss 加總 |
| --- | --- | --- | --- |
| 保留差異 | `[1,0]`／`[1,0]` | `[0,1]`／`[0,1]` | 0 |
| 常數回答 | `[1,0]`／`[1,0]` | `[1,0]`／`[1,0]` | 0 |

兩者都拿到最小值 0。一致性這個條件只問「A 的兩個 view 是否一樣」，沒問「A 和 B 是否仍可區分」。模型若能用一個常數滿足目標，就沒有從這個 loss 得到保留圖片差異的理由。這是目標本身的漏洞，不是因為某次訓練步數不夠。

這個平方差只用來展示漏洞。[下一節](22-distillation.md)的 DINO 實際比較的是 K 維分佈，以交叉熵訓練。交叉熵衡量回答對目標分佈的吻合程度，仍不能自動解決所有圖都給同一分佈的問題：兩份模型若對所有圖都偏向同一個輸出槽，兩者也可能很接近。分佈「一致」和特徵「有區別」仍是兩件事。

## 下游任務會看見這個漏洞

想像現在才拿出 A、B 的紅藍答案，訓練一個讀取特徵的分類器。如果每張圖都送進同一向量 `[1,0]`，分類器收到完全一樣的輸入，便無法按圖片內容分辨紅藍。它最多照兩類的數量或自己的偏好猜。這裡的紅藍各一張，常數預測只能答對 1／2。

因此，檢查塌縮時要同時觀察不同圖片的特徵差異。譬如固定一批圖片，分別取出每張圖的向量，在每個維度計算它們的標準差（值散開的程度）。所有圖都等於同一向量時，每個維度的標準差都是 0。同圖兩個 view 很像、不同圖卻還有變化，比只看第一個條件更有資訊。

標準差大也不是「學會紅藍」的保證。模型可以保留背景噪聲，卻丟掉矩形顏色。這是[22.4 節](22-features.md)還要把 backbone 凍結，拿獨立圖片做最近鄰和線性分類評測的原因。**一致性 loss、特徵變化、下游表現各自回答不同問題。**

## 執行這個失敗例

前面的二維平方差，是先看清漏洞的手算。完整程式再把同一個問題接到下一節要用的分佈：手工指定 8 張圖全部有相同的 32 維特徵，投影輸出長度 K=16，兩個 view 的輸出也都相同。所有 logits 為 0 時，softmax 平均分到 16 個槽，跨 view 交叉熵約 2.773；若固定第 0 槽為 10、其他槽為 −10，分佈集中在同一個槽，交叉熵只有約 `6.18×10⁻⁷`。

這兩種手工分佈的特徵標準差都為 0，且一律預測紅色時，紅藍平衡的 8 個答案只答對 4／8。更低的交叉熵沒有換來能區分圖片的特徵。這裡的交叉熵計算會在下一節解釋；此刻先讀「回答相同，分佈可以很尖，loss 卻很小」的反例。

在本機執行 `PYTHONPATH=. python lesson_cases/22-collapse.py`，或使用頁首 notebook。程式只有手工輸出和數值計算，沒有更新模型參數，也不是「關掉 center 後訓練一定塌縮」的對照。正常結束支持的是這個反例成立，不能宣稱「我們已訓練出防塌縮模型」。

下一節保留「同圖不同 view 相互學習」，並加入一個緩慢更新的 teacher、輸出中心與不同溫度。要追蹤的是它們各自改哪一個量，以及這些措施如何一起處理塌縮。

## 改一件事，先預測

把前面手算的常數回答從 `[1,0]` 換成 `[7,-3]`，同圖平方差會改變嗎？不同圖間的特徵標準差呢？先回答，再用 Python 建立四個相同向量核對。

??? note "參考答案"

    兩者都仍是 0：每對輸出逐項相減仍是零；不同圖在每一維也仍取相同值。向量很長、數字很大，不代表它保留了圖片差異。

??? note "原始 DINO 對塌縮的討論"

    [DINO 論文](https://arxiv.org/abs/2104.14294) §3.1 說明無標籤的 student／teacher 交叉熵目標；§5.3〈Avoiding collapse〉分析 centering（中心化）與 sharpening（讓 teacher 分佈更尖銳）的互補關係。本文的二維常數向量和平方差表格是為了解釋問題而加入的手工例子，不是該論文的實驗結果。

[上一節：22.1 同圖的兩個 view](22-views.md) · [下一節：22.3 teacher 如何提供目標](22-distillation.md)

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-05 在 INTEL(R) XEON(R) PLATINUM 8573C（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-collapse.json)

??? example "展開本次實際輸出"

    ```text
    {
      "constant_uniform": {
        "cross_view_ce": 2.7725887298583984,
        "feature_std": 0.0,
        "normalized_feature_std": 0.0,
        "mean_pair_cosine": 0.9999999403953552,
        "mean_output_entropy": 2.7725887298583984,
        "marginal_output_entropy": 2.7725887298583984
      },
      "constant_peaked": {
        "cross_view_ce": 6.183461209730012e-07,
        "feature_std": 0.0,
        "normalized_feature_std": 0.0,
        "mean_pair_cosine": 0.9999999403953552,
        "mean_output_entropy": 6.183461209730012e-07,
        "marginal_output_entropy": 6.183461209730012e-07
      },
      "constant_feature_balanced_color_accuracy": 0.5,
      "constant_feature_balanced_color_correct": 4,
      "constant_feature_balanced_color_count": 8,
      "construction": "hand-written outputs, no learned model and no optimizer",
      "interpretation": "low cross-view CE alone does not establish useful features; entropy/std/cosine describe different failures",
      "centering_limit": "not an ablation: this does not show that disabling centering immediately collapses every training run"
    }
    ```

<!-- curriculum-evidence:end -->
