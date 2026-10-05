# ViT 作者交付與凍結紀錄

本檔是作者來源與工作範圍紀錄，不是首次閱讀、技術或銜接通過結論。root 協調者另行保存獨立審閱。

基底 code commit：`823232bf9eba4074b4db4c99b6b076feaa1a9992`。新增實作未提交時，以來源 SHA-256 另行標識。

## 已實際讀取的先備與規範

- `AGENTS.md`、`.agents/skills/clear-tutorial/SKILL.md` 與 `references/review-protocol.md`。
- `docs/lessons/01-small-cnn.md`：RGB/NCHW、logits、cross_entropy、optimizer。
- `docs/lessons/03-identity.md`：逐值 residual、shape、identity 與梯度意義。
- `docs/lessons/15-attention-bridge.md`：排列、Q/K/V、softmax、加權讀取、成本與限制。

## 來源、材料與執行證據

ViT 原論文 HTML 實際讀取 §3.1 式 (1)～(4) 及 §4.1 Table 1；URL：https://arxiv.org/html/2010.11929 。
矩形圖片由 `ColorRectangles(samples=128,image_size=32,seed=101)` 的原始像素產生，再轉成8位元PNG嵌入SVG；圖號0～3標籤為1/1/1/0。圖3與patch圖同源。
21-attention及21-transformer機制案例已改為同一張128train圖3並實測。
訓練正文與60點loss圖採 `artifacts/runs/vit-training/result.json` 與 `artifacts/runs/vit-cases/21-training.stdout`，不捏造分數。
actual training：AdamW .003／.01，60步，batch32，dropout .1；固定train loss .710058987→.007154781，val/test各64/64；model/optimizer續訓差0，loss/RNG相同。
shape與parameter facts：圖32／P8／D32／depth2／heads4，16patch+CLS17，23970參數，CLS `[B,32]`、patches `[B,16,32]`、tokens `[B,17,32]`。

## 有意義的閱讀單位

每頁開場（主問題／導覽）及下列H2各為一個完整閱讀單位；程式、相應圖與圖說同其所在單位，不能先透露後面單位。

### 21-attention.md

- 矩形跨格，一塊需要讀其他塊
- 四個 heads，各分配自己的讀取比例
- 交換兩塊，位置線索改變什麼
- 小變化：patch 變小，成對讀取多多少

### 21-patches.md

- 同一種矩形，先只回答顏色
- 8×8 的小塊，變成 192 個數
- 同一個投影，把每塊變成 32 個特徵
- 添一個 CLS，再加每個槽位的位置向量
- 用手工材料核對順序
- 小變化：把邊長從 8 改成 4

### 21-training.md

- 先分開三批材料
- 60 次更新，哪些部件一起學
- Loss 曲線與新圖，分開看
- 第 30 步存檔，需要保留什麼
- 小變化：只還原模型，保留新的 optimizer

### 21-transformer.md

- 一個 block 有兩次修正
- LayerNorm：每個 token 自己整理數值
- MLP：在每個位置組合特徵
- 兩個 blocks，再取 CLS 分類
- 小變化：MLP 中間寬度改成 128

## 作者實際查證與未驗證範圍

已逐一核對每個 `data-excerpt` 是來源檔的原文substring，四頁全部吻合。九份SVG已解析XML、title/viewBox、width480與最小font22；Chromium對全部九圖檢查文字bounds，沒有超出畫布。
已查看patch交換圖、Pre-LN圖與loss圖的實際Chromium渲染，並修正環境字型缺少圈號造成的方框。這是作者視覺自查，不能稱獨立驗收。
完整Zensical桌面與手機頁面檢查交由root另行完成：作者尚未完成頁面級viewport驗證。
作者沒有進行逐段盲讀，也沒有自行標記通過。

## 凍結內容 SHA-256

|檔案|SHA-256|
|---|---|
|`docs/lessons/21-attention.md`|`45604d87d0f718c5502574cd22b2227dc663ff2e7e2c192841bff01eaebb735f`|
|`docs/lessons/21-patches.md`|`e16ab460deee5468d043005ebc2c3fc6fbae9079d6f705151933f5e8d54da0ae`|
|`docs/lessons/21-training.md`|`0859c5db9239431c9ebe08c7fa55b6f6df4e43793aed7a63142e43b9a6d97369`|
|`docs/lessons/21-transformer.md`|`d8fa05e42d1850b1796b0e918097c2bce2e54f69cecaa2e45f7c8e292a2e8405`|
|`docs/assets/diagrams/21-attention-exchange.svg`|`b0e967a5ff76670cee2caddb5acbc00666298ecd434cc6a0cae79e1f520639fb`|
|`docs/assets/diagrams/21-checkpoint-resume.svg`|`4e303f6f1f86907193c9196740f4940379d6eab1f3bf1dad56a70d5dbbca0593`|
|`docs/assets/diagrams/21-classification-flow.svg`|`3e10ea30f2e11c98dd47f6dd2a67b901b3131e185fa7fa8e81fe749f51c9b900`|
|`docs/assets/diagrams/21-materials.svg`|`f6326e65d13df7a41a46377cd7b8bb468e5a8c37b318e33cc33a797c81cf1391`|
|`docs/assets/diagrams/21-patch-grid.svg`|`5a5ba8c89227d49f38627ff7bb434be054c2313cbf6d8d1b3cb879c2833b3ede`|
|`docs/assets/diagrams/21-position-swap.svg`|`76250708df2071b9985faa950f7da42fe3b29e4fc276b9562579344f5ae71987`|
|`docs/assets/diagrams/21-preln-block.svg`|`94341723952941101a6145c1f8421c068e9d469a99165c71f346c1e62b571075`|
|`docs/assets/diagrams/21-token-position.svg`|`0588efde6f7cad857cd2a849933b8ed80f3b3f84c230d922787b1897c9038135`|
|`docs/assets/diagrams/21-training-loss.svg`|`df47bd7953e50fdad41d8c35e74647509c3202df8a861cf5b69b107f1054dd8a`|
|`miniyolo/vision_data.py`|`92b1cc3885a0b897eba2a24ac1cd604c55b5ea903a1886904e64b9364c67dde1`|
|`miniyolo/vision_transformer.py`|`212c8fcd59df0abd2ff9d0e545dbee051bbbe306089f5e0973d47783090fa394`|
|`lesson_cases/21-patches.py`|`f177f249914e9fb8700459235531b5db3cdf8e252562bdb0c4c795a2aead5983`|
|`lesson_cases/21-attention.py`|`4f3f64daff62870aa15000fb64308b89e0035194cced3ee0fa44e448fb614c25`|
|`lesson_cases/21-transformer.py`|`ca67a8e8bb8aedf4f362e4c0a323bd0ee26ee20ee53a17163539a87160c6fce4`|
|`lesson_cases/21-training.py`|`fbf17b83a9daaf98c1b3e29cc54852369ea0379b95cd42bd9da3e3eec0ce4e16`|
|`artifacts/runs/vit-cases/21-attention.stdout`|`f3ad4d996a763feda9d216c5ad2d760a4017537a57bdaa3d3efe54c5f1eb81e3`|
|`artifacts/runs/vit-cases/21-transformer.stdout`|`093afd9039dbd562fe72940d96393206ad348345a6fb2979b002b2e3a995a50b`|
|`artifacts/runs/vit-cases/21-training.stdout`|`d3811e9ab57ff935bf60f1a0facc16040afdf568b9b7dbc2fe8ba5085459856e`|
