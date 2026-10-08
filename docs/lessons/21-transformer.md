# E.21.3 ViT Transformer：交換內容之後，怎麼得到整圖答案

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/21-transformer.ipynb){ .md-button }

[上一節](21-attention.md)讓每個 token 讀到其他區域的內容。現在還要把讀到的資訊用進表示，最後回答「這張矩形是紅或藍」。**怎麼從一次 attention 接成完整的分類路徑？**

## 把交換與整理分成兩次修正

輸入是一串中間表示：16 個 patch 加 CLS，每個有 32 個特徵。這時還沒有紅藍分數。我們把一輪處理組成 **Transformer block（Transformer 區塊）**，輸出仍保留相同的 token 數與特徵寬度，才能繼續接下一輪。

這一輪有兩種工作。attention 讓位置互相讀取；接著，一個小型全連線網路組合每個 token 內已有的特徵。兩段都用[3.1 節](03-identity.md)的 `原輸入＋修正`：已有表示走捷徑，另一條路提供改變。

![手工資料流：先LayerNorm再attention並加回輸入；再LayerNorm和MLP並加回第一段結果；兩條shortcut保留未正規化的tokens](../assets/diagrams/21-preln-block.svg)

沿圖看兩個加號。右邊的捷徑各保留當下輸入；左邊的修正路徑先整理數值，再做 attention 或特徵組合。第二個加號接到的是第一個加號**已更新**的結果，不是又回到最初輸入。

## LayerNorm 整理的是誰的數值

圖上的 **LayerNorm（Layer Normalization，層正規化，LN）**，對每個 token 自己的 32 個特徵算平均與變異數；這裡的變異數是各值與平均之差的平方，再取平均。先減平均，再除以「變異數加一個很小正數」的平方根，最後套用每個特徵可學的倍率與偏移。小正數避免除以 0。

用兩維作手工示意：`[2,6]` 的平均為 4、變異數為 4。忽略小正數，倍率取 1、偏移取 0，輸出約 `[-1,1]`。原本的偏移與尺度被整理後，修正路徑讀到的是較一致的數值尺度。實際是對 32 維做這件事，shape 不變。

LN 不在不同圖片間取平均，也不把 17 個 tokens 合成一個。每個 token 各自處理；不同位置的內容交換仍由 attention 負責。圖中捷徑保留未經 LN 的原值，所以 LN 是修正路徑的準備，沒有取代原表示。

這種「先 LN，再 attention 或 MLP」的順序稱為 **Pre-LN**。若看別的架構圖，先運算、相加後才 LN 是另一種順序，不能對調著解讀。

## MLP 用已取得的內容，組合每個 token 的特徵

**MLP（Multi-Layer Perceptron，多層感知器）**在此是兩個線性層，中間加非線性：32 維先擴到 64 維，經 GELU，再回到 32 維。**GELU（Gaussian Error Linear Unit，高斯誤差線性單元）**與前文 ReLU 一樣，讓這串運算能表達非線性關係；它使用平滑的數值變化。本節追資料流不需要手算 GELU 公式。

attention 已經把其他位置的內容送進 token；MLP 在每個位置內，重新組合這些特徵。同一套 MLP 分別套到所有 tokens，沒有在此新增跨位置讀取。中間 64 維提供額外的特徵組合空間，最後回到 32 維，才能逐值加回捷徑。

令 X 是 block 輸入，U 是第一個加號後的表示，Y 是輸出，三者都是 `[B,17,32]`，B 是圖片數。上一節的四 head attention 簡寫為 **MSA（Multi-head Self-Attention，多頭自注意力）**：

\[
\begin{aligned}
U &= X+\operatorname{MSA}(\operatorname{LN}_1(X)),\\
Y &= U+\operatorname{MLP}(\operatorname{LN}_2(U)).
\end{aligned}
\]

兩份 LN 各有自己的倍率與偏移。兩次相加後都沒有再接 ReLU；如果某條修正路徑輸出 0，該次加法就保留原值。

``` { .python data-excerpt="miniyolo/vision_transformer.py" }
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, embed_dim), nn.Dropout(dropout))

    def forward(self, tokens):
        tokens = tokens + self.attention(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))
```

`embed_dim=32`、`hidden=64`；第二行用的是第一行更新後的 `tokens`。dropout 訓練時隨機刪值、縮放保留值，`eval()` 時關閉，不改 shape。

## 兩輪處理之後，由 CLS 回答顏色

本例串接兩個 blocks，參數各自獨立。第二個 block 讀到的是第一輪已修正的表示，因此它會重新計算 Q／K／V 與讀取比例，再更新表示。

![TinyViT完整分類流程：RGB圖→patch投影→CLS與位置→兩個PreLNblocks→最終LN→取CLS→線性分類，輸出紅藍兩分數](../assets/diagrams/21-classification-flow.svg)

兩個 blocks 後，再對各 token 做一次 LN。取第 0 個 CLS 作整圖特徵，交給 `Linear(32,2)`，產生順序為紅、藍的兩個 logits。取較大分數的索引，就是預測類別 0／1。分類頭讀的是已彙整圖片的 CLS，不是 patch 編號。

這個小模型叫 **TinyViT**：原圖邊長 32、patch 邊長 8、表示寬度 32、4 個 attention heads、兩個 blocks、MLP 中間寬度 64，共 23,970 個可學參數。它走完一條完整的 ViT 分類路徑；能力是否足夠，仍須訓練與評估。

`forward_features` 在分類前保留三種輸出，方便之後選用整體或區域表示：

|名稱|shape|從哪裡取得|
|---|---|---|
|`tokens`|`[B,17,32]`|最後 LN 後的全部序列|
|`cls`|`[B,32]`|序列第 0 個 token|
|`patches`|`[B,16,32]`|後 16 個 tokens，仍按原圖格子順序|

``` { .python data-excerpt="miniyolo/vision_transformer.py" }
    def forward_features(self, images, use_position=True):
        tokens = self.encode_tokens(self.embed_tokens(images, use_position=use_position))
        return {"cls": tokens[:, 0], "patches": tokens[:, 1:], "tokens": tokens}

    def forward(self, images):
        return self.head(self.forward_features(images)["cls"])
```

`encode_tokens` 執行兩個 blocks 與最後 LN。完整程式也把第一個 block 的兩次相加手動拆開，結果與 `block(embedded)` 完全一致，`manual_residual_matches_block=true`；並得到 CLS `[1,32]`、patches `[1,16,32]`、logits `[1,2]`。這是公式與實作的一致性檢查，沒有 optimizer 更新。

## 完整模型核對位置的作用 { #position-check }

現在可以回到上一節的換位問題，知道比較的是哪個輸出：**未訓練 TinyViT 經兩個 blocks、最後 LN 後的 CLS**，不是分類分數。

不加位置時，attention 讀取相同一批內容；LN、MLP 對各 token 使用相同規則，捷徑也只把同位置的修正加回。因此交換 patch 順序後，patch 輸出跟著交換，留在第 0 槽的 CLS 不變。加位置後，若槽位的位置向量固定，只換內容，配對改變，CLS 就可能改變。

[21.2 完整程式](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/21-attention.ipynb)用同一張 train 圖 3，交換 patch 1 與 16，關閉 dropout，保存了以下結果。最大絕對差是比較的特徵逐項相減、取絕對值後的最大值。

|比較|最大絕對差|
|---|---:|
|不加位置，交換前後的 CLS|0.0000003576|
|不加位置，patch 輸出與應有的交換順序|0.0000004768|
|位置留在原槽，只交換內容，交換前後的 CLS|0.001603425|

前兩項小於該程式的容許誤差 0.000002；第三項超過它。結果符合「位置與內容配對會改變完整模型的輸入與特徵」。沒有做有／無位置的訓練對照，而且紅藍答案本來就不依賴位置，所以差值不能證明位置向量提高分類正確率。

本節程式另外把全部 patches 逆序排列，仍保留 CLS 第 0 槽，這是另一個換位檢查。它得到不加位置的重排誤差約 `3.5763×10⁻⁷`，固定位置時 CLS 最大變化約 0.030437；數值不能當成上表「只交換兩塊」的同一次測量。兩者都只是未訓練模型的機制核對。

## 看懂分工後，改一個部件

把 MLP 中間寬度從 64 改為 128，最後仍輸出 32 維。捷徑還能相加嗎？它會直接讀到更多 patch 嗎？

??? note "參考答案"

    可以相加，最終 shape 仍是 `[B,17,32]`。中間加寬增加參數與運算，但 MLP 仍逐 token 處理；跨位置讀取由 attention 負責，不能僅憑加寬保證分類更準。

若兩條修正路徑最終都輸出 0，block 輸出是什麼？再接圖中的最後 LN，還保證等於原始 X 嗎？

??? note "參考答案"

    block 的兩次加法都加 0，所以 Y=X。block 外的最後 LN 會整理數值，沒有保證 `LN(X)=X`。查 block 的 identity 性質，要分清 block 與外面的最終 LN。

模型路徑已接好。[下一節](21-training.md)才讓全部部件真的更新，測試新圖片，並檢查中途存檔能否接回同一次訓練。

??? note "與原始 ViT 的對應"

    [ViT §3.1 式 (2)～(4)](https://arxiv.org/html/2010.11929#S3.SS1) 定義 Pre-LN、兩條 residual、MLP 的 GELU 與最終 CLS 的 LN。[§4.1 Table 1](https://arxiv.org/html/2010.11929#S4.SS1) 的 ViT-Base 深度／寬度／MLP／heads 為 12／768／3072／12，本例是 2／32／64／4。分類頭只用單一線性層，沒有論文預訓練 head 的隱藏層，也沒有大規模預訓練。

[上一節：21.2 交換內容](21-attention.md) · [下一節：21.4 訓練與還原](21-training.md) · [在 Colab 核對模型](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/21-transformer.ipynb)

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/21-transformer.json)

??? example "展開本次實際輸出"

    ```text
    {"event": "transformer", "block_count": 2, "input_tokens_shape": [1, 17, 32], "pre_ln": true, "first_attention_update_norm": 0.18654008209705353, "first_mlp_update_norm": 0.2148894965648651, "manual_residual_matches_block": true, "cls_feature_shape": [1, 32], "patch_feature_shape": [1, 16, 32], "logit_shape": [1, 2], "permutation": [0, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "without_position_equivariance_max_error": 3.5762786865234375e-07, "with_fixed_position_cls_max_change": 0.03043721616268158, "parameters": 23970, "limitation": "Untrained mechanism check; changed CLS values do not show successful color classification."}
    ```

<!-- curriculum-evidence:end -->
