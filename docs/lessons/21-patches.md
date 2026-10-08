# E.21.1 ViT patches：先把圖片變成可讀取的小塊

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/21-patches.ipynb){ .md-button }

[15.1 節](15-attention-bridge.md)讓特徵圖上的位置互相讀取：每個位置先有一個向量，attention 再決定從誰讀多少。現在從原圖出發，還沒有 CNN 提供的特徵圖。**要讓 attention 讀圖片，先交給它哪些向量？**

**ViT（Vision Transformer，視覺 Transformer）**選擇先把圖片切成固定大小的小塊，每塊轉成一個向量，再讓這些向量交換資訊。先看它如何準備讀圖的材料。

## 看同樣的顏色題，改變讀圖方式

答案沿用[小 CNN](01-small-cnn.md)的規則：紅矩形是類別 0，藍矩形是類別 1。下面是實際訓練資料的前四張圖。

![seed101訓練資料的第0至3張實際圖片；答案依序為1、1、1、0，每張都是32乘32RGB](../assets/diagrams/21-materials.svg)

圖號 0～3 用來找到某一張圖片；答案 0／1 才表示紅／藍。兩類都是矩形，寬高、位置、亮度及暗色背景雜訊的抽樣不依賴類別。因此要回答的是顏色，位置和形狀種類都不是類別答案。

每張原圖是 32×32 RGB，三個顏色通道的值介於 0 和 1。一張圖在 PyTorch 裡寫成 `[1,3,32,32]`：一張、三通道、高、寬。這裡仍可用 CNN 分類；我們換的是準備特徵與交換資訊的方式，並沒有換答案規則。

## 每個 token 先負責一塊圖

**Patch** 就是切出來的小塊。把原圖用邊長 8 pixel 的視窗切開，不重疊、不補邊，每列有 4 塊，共 16 塊。

![實際訓練圖3切成4乘4塊；1至16是patch編號，每塊8乘8pixel，依列排列](../assets/diagrams/21-patch-grid.svg)

沿圖中編號讀：先從左到右排完第一列，再往下排下一列，直到 patch 16。這與 15.1 把特徵圖攤平的 row-major 順序相同。4×4 是小塊的排列格數；32×32 才是原圖的 pixel 尺寸。

切圖線沒有在找矩形邊界。紅矩形跨過好幾塊，一塊可以混有矩形和背景，也可以全是背景。這種安排先把整張圖分成固定的讀取單位，物件的判斷留給後面的可學運算。

一塊包含 R、G、B 各 8×8 個值，攤平後是 `3×8×8=192` 個數。塊內先排 R 的 64 個 pixel，再排 G、B；每個通道內也按列排列。整圖因此得到 `[1,16,192]`：一張圖、16 塊、每塊 192 個像素值。這一步只重排資料，沒有可學參數。

## 共用一個投影，把像素變成特徵

我們接著讓每塊用同一個可學轉換，將 192 個像素值組合成 32 個特徵。這叫 **patch embedding（小塊的向量表示）**；一塊的表示就是一個 patch token。

共用轉換讓各塊都用同一種讀法：它在每個區域都計算相同的一組特徵，不為左上和右下另建不同的像素讀取器。32 是本例選定的表示寬度，不是圖片邊長，也沒有指定其中哪一維必須是紅色或座標。訓練會調整這些組合。在這一步，一塊仍只讀自己的像素，相鄰塊還沒有交換內容。

實作用 `Conv2d(3,32,kernel_size=8,stride=8)`。每次視窗剛好蓋住一塊、跳到下一塊；每個輸出特徵都是「192 個值各乘權重、相加、加 bias」。因此它等同對每塊套用同一個 `Linear(192,32)`。輸出先是 `[1,32,4,4]`，再把每個位置的 32 個值排成一列，得到 `[1,16,32]`。

``` { .python data-excerpt="miniyolo/vision_transformer.py" }
    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"images 必須是 [B,3,{self.image_size},{self.image_size}]")
        return self.projection(images).flatten(2).transpose(1, 2)
```

這裡 `self.projection` 就是上述卷積。`flatten(2)` 把 4×4 併成 16 個位置；`transpose(1,2)` 交換位置與特徵兩軸。與 15.1 的整理方法相同，但向量來源換成原圖的 patch 投影。

## 分散的內容，要在哪裡彙整成整圖答案

現在有 16 份區域表示，最後卻只需要一個紅／藍答案。我們在序列前面加一個 **CLS token**；CLS 來自 classification（分類），它是專門放整圖表示的位置。

CLS 起初是一個可學的 32 維向量。它不是切自某塊圖，也不是正確類別標籤；各張圖先取得相同初值。接上 attention 後，它才會讀到各張圖的內容，形成不同的整圖表示，供分類頭使用。

序列成為 `[CLS, patch 1, …, patch 16]`，共 17 個 tokens，shape 是 `[1,17,32]`。程式索引 0 是 CLS，索引 1～16 對應圖中的 patch 1～16。

還有一件事：同一組 patch 內容，可以排成不同圖片。若只交出內容向量，attention 的共用讀法沒有額外線索知道哪塊在左上、哪塊在右下。**Position embedding（位置向量）**為每個序列槽位準備一個可學向量，加到放在那裡的內容上，讓「內容」同時帶著「所在槽位」的線索。

![手工流程示意：patch內容加上所屬槽位的位置向量，CLS同樣有位置向量，形成17個32維tokens](../assets/diagrams/21-token-position.svg)

圖中每個加號都是逐特徵相加。用兩維手工例子看：內容 `[2,5]` 加位置 `[0.1,-0.2]` 得 `[2.1,4.8]`。實際是 32 維加 32 維，結果仍為 32 維。這個例子只教加法，不是模型的實測特徵。

17 個槽位各有一份位置向量，包括 CLS。它們不是直接寫入 `(x,y)` 像素座標；槽位與原圖格子的對應來自剛才固定的排列順序。位置線索讓模型能依排列處理內容，但本例的顏色答案本來就不依賴位置，不能因此宣稱它會提高分類正確率。

??? note "對照 CLS 與位置的實作"

    ``` { .python data-excerpt="miniyolo/vision_transformer.py" }
        def embed_tokens(self, images, use_position=True):
            patches = self.patch_embed(images)
            tokens = torch.cat((self.cls_token.expand(images.shape[0], -1, -1), patches), dim=1)
            if use_position:
                tokens = tokens + self.pos_embed
            return self.embedding_dropout(tokens)
    ```

    `expand` 讓 B 張圖各取得相同的 CLS 初值，`cat(...,dim=1)` 把它加在序列前面。`pos_embed` 是 `[1,17,32]`，相加時各張圖用同一套位置向量。

    `embedding_dropout` 在訓練時隨機將部分特徵設為 0，再縮放保留值；`eval()` 時關閉。下面的切塊核對採預設 dropout=0，沒有這項隨機變化。

## 用一張刻意設計的圖，核對排列有沒有錯

真正的 32 維投影是可學的，尚未訓練時不容易直接看出每一維的意思。完整程式另造一張測試圖：第 k 塊內部全是 RGB `[k/16,0.25,0.75]`，k 從 0 到 15。再手動設定 **192→3 的投影**，讓輸出恰好是該塊的 R、G、B 平均。

|閱讀用 patch 號|程式 k|塊內每個 pixel 的 RGB|
|---|---:|---|
|1|0|`[0,0.25,0.75]`|
|2|1|`[0.0625,0.25,0.75]`|
|16|15|`[0.9375,0.25,0.75]`|

這張圖的 R 值逐塊增加，G、B 不變，所以輸出若依序為 `0、0.0625、…、0.9375`，就能直接核對 patch 是否排對。它不是前面的矩形訓練圖；3 維也是方便查順序的手工投影，不是 TinyViT 的表示寬度。

保存的實際輸出得到 `projection_shape=[1,16,3]`，RGB 平均符合設定；TinyViT 另核對 `[1,16,192] → [1,16,32] → [1,17,32]`。給各塊相同內容時，不加位置的 patch 表示相同，加位置後則不同。這些結果回答「輸入整理是否符合約定」，沒有 optimizer 更新，也還沒證明紅藍分類成功。

## 換小塊之前，先算會增加什麼

原圖仍是 32×32，表示寬度仍是 32，只把 patch 邊長從 8 改成 4：共有幾塊？每塊原始向量有幾個數？加 CLS 後有幾個 tokens？

??? note "參考答案"

    每列 8 塊，共 64 塊；每塊有 `3×4×4=48` 個數；加 CLS 後有 65 個 tokens。每個 token 覆蓋較小區域，但要處理的 tokens 更多。改 patch 大小須重新建立模型；位置向量也要從 17 槽變成 65 槽，不能直接沿用原 checkpoint。

現在已經把圖準備成一串帶位置線索的向量。[下一節](21-attention.md)讓它們互相讀取，並檢查增加 tokens 如何增加成對計算。

??? note "支線入口與架構來源"

    這條選讀支線以[第 1 章](01-small-cnn.md)的分類、[3.1 節](03-identity.md)的原樣捷徑及[15.1 節](15-attention-bridge.md)的 Q／K／V 為起點；不要求先讀完全部 YOLO 版本。

    本節採用 [ViT 原論文 §3.1 式 (1)](https://arxiv.org/html/2010.11929#S3.SS1) 的 patch 投影、可學 CLS 與位置向量。圖、向量寬度與模型深度縮小到可在 CPU 看清流程，沒有載入原論文預訓練模型。

[下一節：21.2 patch 交換內容](21-attention.md) · [在 Colab 核對切塊](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/21-patches.ipynb)

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/21-patches.json)

??? example "展開本次實際輸出"

    ```text
    {"event": "patches", "image_shape": [1, 3, 32, 32], "patch_size": 8, "raw_patch_shape": [1, 16, 192], "raw_vector_order": "channel,row,column", "patch_order": [[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 1, 0], [5, 1, 1], [6, 1, 2], [7, 1, 3], [8, 2, 0], [9, 2, 1], [10, 2, 2], [11, 2, 3], [12, 3, 0], [13, 3, 1], [14, 3, 2], [15, 3, 3]], "patch_red_means": [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375], "projection_shape": [1, 16, 3], "projection_demo": "RGB means; hand-set weights", "vit_patch_embedding_shape": [1, 16, 32], "tokens_with_cls_shape": [1, 17, 32], "cls_index": 0, "first_patch_sequence_index": 1, "learned_position_shape": [1, 17, 32], "same_content_equal_without_position": true, "same_content_equal_with_position": false, "full_attention_scores_per_head": {"patch8_with_cls": 289, "patch4_with_cls": 4225}, "score_count_ratio_with_cls": 14.619377162629759, "patch_only_score_count_ratio": 16.0, "limitation": "Hand-set projection explains token construction; no optimizer update or learned classification here."}
    ```

<!-- curriculum-evidence:end -->
