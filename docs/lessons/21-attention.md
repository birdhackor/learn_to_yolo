# 21.2 ViT attention：讓小塊交換內容

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/21-attention.ipynb){ .md-button }

[上一節](21-patches.md)把一張 32×32 RGB 圖切成 16 個 patch，投影成 32 維向量，加上 CLS 和位置向量。紅矩形的答案仍是 0、藍矩形仍是 1。**現在，每個小塊怎麼取得其他小塊的內容？** 本節把第 15.1 節的 Q/K/V 讀取法用到這 17 個 tokens。

## 一塊怎麼讀其他塊的內容

材料仍是 seed 101 的訓練圖 3，正確答案為紅色 0。紅矩形跨過幾個 patch：patch 1 只有背景，本身沒有矩形的紅藍訊號；patch 6 已經含紅色。互讀可以讓各塊取得其他塊的內容，CLS 也能藉此彙整整圖資訊。這個顏色任務可以用第 1 章的 CNN 完成；本節研究的是 attention 怎麼交換內容。

下面仍選 patch 6 當接收者，展示它如何混合各來源的特徵。圖中只畫 patch 1、6、7、10 四個來源，讓箭頭能看清楚；真正的模型同時讀全部 16 塊和 CLS。

![手工讀取示意：四個實際patch的內容流向接收patch6；箭頭表示來源value按讀取比例送到接收者，非物件移動](../assets/diagrams/21-attention-exchange.svg)

箭頭從**提供內容的來源**指向**接收者**。它不表示把原圖搬動或把紅色塗到別塊；改變的是 patch 6 的特徵向量。每個來源提供一份 V，接收者按權重相加，得到包含其他位置資訊的新向量。CLS 也以同樣方式參與讀取，所以它能取得整圖各塊的內容。

## 四個 heads，各分配自己的讀取比例

第 15.1 節已經算過：Q 是接收者的 query、K 是來源的 key，兩者比對後經 softmax 得到權重，真正混合的內容來自 V。這裡仍是 **self-attention**：Q、K、V 都由同一串輸入 tokens 產生。「self」指來源和接收者來自同一串，不是只讀自己。

本模型用 **4 個 attention heads**。Q、K、V 先各有 32 個特徵，再各分成 4 組，每組 8 個特徵。每組各算一張權重表：同一個接收者在不同 head 可以給來源不同的比例。這裡的 head 是注意力頭；最後給紅藍分數的 classification head 是另一個部件。

設 B 是圖片數，T=17 是含 CLS 的 token 數，d=8 是每個 head 的特徵數。Q、K、V 的 shape 都是 `[B,4,17,8]`，權重表是 `[B,4,17,17]`。後兩軸依序是接收者 query、來源 key；每個接收者那一列沿來源軸加起來等於 1。這次仍用第 15.1 節的計算：

\[
A=\operatorname{softmax}_{\text{來源}}\!\left(\frac{QK^{\mathsf T}}{\sqrt{8}}\right),\qquad O=AV.
\]

A 是每個 head 的讀取比例，不是可學參數；產生 Q/K/V 的投影才是可學參數。O 是各 head 混合後的特徵，shape `[B,4,17,8]`。把四組結果接回 32 維，再做一次可學的 32→32 投影，attention 的輸出便回到 `[B,17,32]`。

``` { .python data-excerpt="miniyolo/vision_transformer.py" }
    def forward(self, tokens, return_attention=False):
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim or tokens.shape[1] < 1:
            raise ValueError("tokens 必須是非空的 [B,N,embed_dim]")
        batch, count, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, count, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(dim=-1)
        mixed = (self.attention_dropout(weights) @ v).transpose(1, 2).reshape(batch, count, self.embed_dim)
        output = self.output_dropout(self.projection(mixed))
        return (output, weights) if return_attention else output
```

程式的 N（`count`）就是正文的 T。`reshape` 先把一次 QKV 投影的 96 個特徵拆成三份、各四組；`permute` 把 QKV 軸移到前面，`unbind(0)` 才依序取得 Q、K、V。最後 `transpose(1,2)` 把 head 軸放回每個 token 裡，再接成 32 維。

`return_attention=True` 另外回傳 softmax 後、dropout 前的權重，好核對每列和為 1。訓練時若啟用 dropout，實際使用的權重會隨機刪除部分值並縮放，該次使用的列和不必恰好是 1。本節的機制檢查使用 `eval()` 關閉 dropout。

## 交換兩塊，位置線索改變什麼

現在做一個機制檢查，不訓練模型。保持 CLS 在序列第 0 槽，把兩個 patch 的**內容向量**交換；原圖分類答案不變，仍要辨認紅或藍。這次比較完整、未訓練 TinyViT 的整圖特徵：tokens 經過兩個 Transformer blocks 和最後 LayerNorm 後，取出的 CLS 向量。下一節才拆開這條完整路徑；此刻只觀察這個 32 維整圖特徵有沒有改變，不計算分類 head 的紅藍分數或正確率。

先知道這條完整路徑裡各部件的分工就夠了：Transformer block 用 attention 讓位置互相讀取，再用 **MLP（Multi-Layer Perceptron，多層感知器）**這個小型全連線網路修改每個 token 內的特徵。**LayerNorm（Layer Normalization，層正規化）**整理每個 token 自己的特徵尺度。MLP 與 LayerNorm 都對各 token 使用同一套規則，不在這一步混合不同位置；residual 則把同一位置的修正加回原值，和第 3.1 節相同。因此，交換 patch 順序後，這些步驟的輸出也跟著同樣交換；它們不會讓固定在第 0 槽的 CLS 多出位置線索。具體順序與公式留到下一節。

不加位置向量時，交換只是把同一批內容換順序。每個 token 用相同的 QKV 投影，attention 又讀全部來源，因此來源順序換了，對 CLS 的加權總和不變；patch 輸出的順序則跟著交換。後續逐 token 的運算和 residual 也保留這個性質。

加入位置向量後，若**槽位的位置向量留在原位**，交換兩塊內容會形成新的「內容＋位置」配對，CLS 輸出就可能改變。若把已加完位置的整個向量一起搬走，內容和位置配對沒有改，仍只是同一串向量重排；不能用那種操作測出位置的作用。

![手工交換示意：不加位置時交換內容只是重排；加入位置時位置留在原槽，內容與位置的配對改變](../assets/diagrams/21-position-swap.svg)

完整程式以同一張 train 圖 3，交換 patch 1 與 patch 16，保持 CLS 在第 0 槽；關閉 dropout 後，實際結果如下。最大絕對差是對比較的全部特徵逐一相減、取絕對值，再取最大值：

|比較|最大絕對差|
|---|---:|
|不加位置，交換前後的整圖特徵（CLS）|0.0000003576|
|不加位置，patch 輸出與應有的交換順序|0.0000004768|
|位置留在原槽，只交換內容，交換前後的整圖特徵（CLS）|0.001603425|

前兩項小於程式採用的容許誤差 0.000002，視為 float32 捨入造成的差異；第三項則明顯超過它。這能展示位置向量如何改變運算的輸入，**不能據此說位置向量提升了紅藍分類正確率**：顏色答案本來就不依賴位置，也沒有在這裡做有／無位置的訓練對照。

完整程式前半也會重做第 15.1 節的四個二維手工 tokens `[1,0]、[0,1]、[1,1]、[0,0]`，並對這個小 attention 模組做一次 SGD 更新，確認 QKV 收到梯度。這與上表的未訓練 TinyViT 交換檢查是兩個部分；手工目標不是紅藍答案，不能把一次更新當成分類學習。

attention 圖同樣有範圍：高權重表示這層這個 head 從某來源讀得多；後面還有投影、residual、MLP 和其他層，所以它不是分類答案的完整解釋。這個實驗也沒有輸出框，不能把箭頭或權重當成物件偵測成功。

## 小變化：patch 變小，成對讀取多多少

沿用上一節的 P=4 小變化。32×32 原圖會有 64 個 patches，加 CLS 後 T=65。若每個接收者都和每個來源比較，單圖、單 head 的表有多少個值？和 P=8 相比幾倍？

??? note "參考答案"

    P=8 有 T=17，表內 `17×17=289` 個值；P=4 有 T=65，表內 `65×65=4225` 個值，約 14.62 倍。四個 heads 分別有 1156 和 16900 個權重值。不能只數 16² 與 64²，因為 CLS 也參與讀取。

    這只比較 attention 的成對項目數，假設 B、head 數與每個 head 的特徵數相同。投影、MLP、資料搬移等成本沒有包含，也不能直接說執行秒數變成 14.62 倍。

再想一個小變化：若一個 head 的某接收者 Q 全為 0，與 17 份 K 的分數都是 0，softmax 會如何讀取？

??? note "參考答案"

    17 個來源各占 1/17，輸出是 17 份 V 的平均，包括 CLS 的 V。這只描述該 head 的加權內容；四個 heads 接回之後還有輸出投影，不會直接等於原圖 RGB 的平均。

機制來源：[ViT 原論文 Appendix A](https://arxiv.org/html/2010.11929#A1) 定義多 head self-attention；[§3.1](https://arxiv.org/html/2010.11929#S3.SS1) 說明位置 embedding；Appendix D.4 提供原論文的位置設定對照。本文的交換檢查是課堂機制實驗，沒有重做論文的 ImageNet 對照。

[上一節：21.1 圖片切塊](21-patches.md) · [下一節：21.3 完整 Transformer block](21-transformer.md) · [在 Colab 重做交換](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/21-attention.ipynb)

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-06 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/21-attention.json)

??? example "展開本次實際輸出"

    ```text
    {"event": "attention", "manual_tokens": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], "first_query_scores_before_scale": [1.0, 0.0, 1.0, 0.0], "scale": 1.4142135623730951, "first_attention_row": [0.3348807692527771, 0.1651192307472229, 0.3348807692527771, 0.1651192307472229], "first_weighted_value": [0.6697615385055542, 0.5], "manual_weight_shape": [1, 1, 4, 4], "manual_loss": 0.1594690978527069, "q_k_v_gradient_abs_sums": [0.07701923698186874, 0.07701923698186874, 0.4417698085308075], "qkv_weights_changed": true, "vit_tokens_shape": [1, 17, 32], "vit_heads": 4, "vit_head_dim": 8, "vit_attention_shape": [1, 4, 17, 17], "vit_attention_scores": 1156, "vit_output_shape": [1, 17, 32], "vit_row_sum_max_error": 1.7881393432617188e-07, "swap_patch_sequence_indices": [1, 16], "without_position_cls_max_change": 3.5762786865234375e-07, "without_position_patch_equivariance_max_error": 4.76837158203125e-07, "with_fixed_position_cls_max_change": 0.0016034245491027832, "limitation": "One SGD update validates the mechanism; random ViT attention weights are not explanations of a trained decision."}
    ```

<!-- curriculum-evidence:end -->
