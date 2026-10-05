# ViT 支線：獨立分段首次閱讀紀錄

## 範圍與方法

讀者背景：基本 Python／PyTorch、NN／CNN；大學數學學過但不熟，不知道本專案。依 clear-tutorial 與 review-protocol，先順序讀指定前置，再讀目標四頁。唯一正文入口為 `gate.py next vit`；每個 unit 都先以自己的話記錄理解、元素、下一步依據、原文卡點、必要圖與小變化，成功 `record` 才取得下一 unit。

完成 46/46 units：01-small-cnn 9、03-identity 7、15-attention-bridge 6，以及 21-patches 7、21-attention 5、21-transformer 6、21-training 6。原始逐段紀錄保存在 [first-read/vit.jsonl](first-read/vit.jsonl)，含當時可見來源、unit／figure SHA256、行號和 reader_notes。沒有一次讀 packets、沒有讀實作、外部題解、作者筆記或其他審查稿。共用 FS 不是技術隔離；此處的獨立首次閱讀來自實際遵循揭露規則，不宣稱無法存取全文。

紀錄時 Git HEAD 為 `823232bf9eba4074b4db4c99b6b076feaa1a9992`。本報告只反映 gate 凍結正文；不以 HEAD 代替下表的正文內容指紋，也不反映後續修改。未修改教材、未 commit、未執行訓練或技術查證。

|首次閱讀來源|凍結 source SHA256|
|---|---|
|01-small-cnn.md|`434c49f8ecd00ae79cd7ee76db4ae33935957d3b936a223058958fe717a5a82c`|
|03-identity.md|`5af0147567fe9bfefb6424962c8a2dde6001f1f75ff09a41403112a146cf4555`|
|15-attention-bridge.md|`eba51744673bf59c6c1e9cc913d3bb2d83faa73ed7463465b4f0728682f4cf8e`|
|21-patches.md|`e16ab460deee5468d043005ebc2c3fc6fbae9079d6f705151933f5e8d54da0ae`|
|21-attention.md|`45604d87d0f718c5502574cd22b2227dc663ff2e7e2c192841bff01eaebb735f`|
|21-transformer.md|`d8fa05e42d1850b1796b0e918097c2bce2e54f69cecaa2e45f7c8e292a2e8405`|
|21-training.md|`0859c5db9239431c9ebe08c7fa55b6f6df4e43793aed7a63142e43b9a6d97369`|

## 全節首讀復述

### 實際讀過的前置

小 CNN 先用八張固定大小紅／藍矩形，說清圖號與類別不同、NCHW 軸及 logits／argmax／交叉熵。卷積每個局部視窗乘同一套權重，所以權重數不隨掃過的位置增加；濾鏡數決定特徵 channel，而非輸出仍是 RGB。兩個 block 的卷積、ReLU 與 pooling 使高寬縮小、channel 增加；GAP 把每個 channel 平均，省參數但收起值所在位置。三步 SGD 的權重確有改變，卻全部猜藍；重新開始的四十步 Adam 才把八圖全答對。後者仍只支持這批訓練圖，不能當新圖成績或單獨歸因更多步數。

identity 的主分支算修正 F，shortcut 原樣傳 x，逐值相加 y=x+F，不是串接。教學版相加後沒有 ReLU，F=0 時負數也保留；有相加後 ReLU 就不能對任意正負 x 宣稱整個 block 是 identity。兩路完整 shape 相同才符合設計，broadcast 可放過錯誤。shortcut 的直接係數 1 不代表總梯度永遠 1，主路可抵消；人工零分支用 sum(y) 只算梯度，不是訓練。隨機分支與全零目標的兩步更新是另一個機制檢查，沒有比較分類準確率。

attention 前置把兩 channel 的 2×2 特徵圖按位置排成四個二維 tokens。Q 問要找什麼、K 供比對、V 是實際內容；identity 初投影只是方便手算，不表示三塊永久共用參數。縮放內積後沿來源軸 softmax，每個接收者分配一份讀取比例，再 @V 得加權內容。這些比例隨輸入重算，訓練的是投影。手算左上權重 [.3349,.1651,.3349,.1651]、輸出 [.6698,.5]，再用特定重建目標驗證一次 SGD。N² 配對成本與沒有直接位置線索的限制都有交代；不能把 attention 比例當最後預測的因果貢獻。

### 21.1 Patches

任務仍是紅 0、藍 1，但資料換成尺寸、位置、亮度及暗雜訊都有變化的 32×32 RGB 矩形；這些變化不依類別。P=8 pixel、不重疊無 padding，切成 4×4 共 16 塊。每塊先排 R64、G64、B64，raw shape [1,16,192]，這只是資料重排。

共用 Conv2d(3,32,8,stride=8) 等同每塊 Linear(192,32)，得到 [1,16,32] 的 patch embedding。D=32 是特徵数，不是 pixel 邊長。加一個可學 CLS 作整圖彙整槽，再逐值加 17 槽的位置向量，成 [1,17,32]；CLS 不是標籤，位置也不是直接 pixel 座標。手工 192→3 平均投影另用固定 RGB 圖核對順序，不是 TinyViT 的 32 維投影，不含 optimizer 更新。

我的小變化：P=16 時 4 塊、每塊 768 值、加 CLS 後 5 tokens；D 若仍 32，不能把 raw 寬度和 embedding 寬度混用。

### 21.2 Attention

17 tokens 含 CLS 都可作來源與接收者。D=32 的 Q/K/V 各分成 4 heads，每 head 8 特徵；各算 [B,4,17,17] 的來源比例，再混 V、接回 32 維並做輸出投影。self 表示同一串來源，不是只讀自己；attention head 不同於分類 head。回傳權重是 dropout 前，訓練真正使用的列和不必 1，機制檢查 eval 關掉 dropout。

無位置時，只交換 patch 內容、CLS 槽固定，是同一批內容重排，CLS 加權總和不變；有位置且位置留槽，內容與位置配對改變，CLS 可能變。兩個無位置最大差約 2.384e−7，低於 2e−6 容差；有位置 .001603365 超過容差。這支持機制變化，不是有位置分類更準。P=4 的 T=65，單 head 4225 項，是 T=17 的 289 項約 14.62 倍；不能省掉 CLS，也不能直接乘成秒數。

我的小變化：只搬已加位置的整個 token，仍是同一批向量重排，無法測出位置配對作用。D 固定而 heads 變 8，d 變 4，不是 D 或 token 數加倍。

### 21.3 Transformer

一個 Pre-LN block 做兩次 residual：U=X+MSA(LN1(X))，Y=U+MLP(LN2(U))。LN 逐 token 整理 32 特徵尺度，各特徵有倍率與偏移；shortcut 保留未經 LN 的當下表示。MLP 共用 32→64→32、GELU，組合 token 內特徵，不自行讀其他位置。兩份 LN 不共用參數，相加後沒有 ReLU。

TinyViT 串兩個參數獨立 blocks，再做最終逐 token LN；tokens [B,17,32]、cls [B,32]、patches [B,16,32]。分類只讀最終第 0 CLS，Linear(32,2) 給紅、藍 logits。人工拆 block 與實作相同、shape 与 23,970 參數數目只核路徑，本頁沒有 optimizer 更新或分類能力評測。

我的小變化：兩修正為 0 時 block 輸出 X，但外部最終 LN 不保證仍是 X；例如手工 [2,6] 可變約 [-1,1]。hidden 加寬仍回 32，沒有增加跨 patch 讀取。

### 21.4 Training

train128/seed101 進更新、val64/202 觀察、test64/303 在設定固定後評估；本次固定 60 步，不按 val 挑 checkpoint。每步有放回抽 32 張，CPU float32、seed7、dropout .1、AdamW lr .003/decay .01，整模型所有部件都交 optimizer。反傳後先查 loss/梯度 finite、全體 norm 正，再 step。

batch 曲線每點是該次更新前、不同抽樣與 dropout；另 eval 固定 train 更新前後，再 eval val/test。train 從 64/128 到 128/128，val/test 各 64/64；配合六組參數確實改變，支持本次執行答對這些同規則矩形，不是自然照片 100%、ViT 優於 CNN，或 attention／位置唯一造成成功。

第 30 次更新後 checkpoint 保留模型、AdamW 梯度歷史/步數、RNG 當下位置、模型與類別設定。新模型建構耗亂數，需最後還原 RNG。上路連續 60 步、下路載入後接 31～60：模型/optimizer 最大差 0、續接 loss 每點及末 RNG 完全相同、步數皆 60，支持本次 CPU 設定能接同次訓練，不是跨硬體或版本逐位元保證。

我的小變化：只還原模型、即使新 AdamW lr 改回 .003，仍缺三十步歷史/步數，不能保證原第 31 步一致。推論完成、檔案存在與完整續訓一致是不同判準。

## 當場卡點與後文狀態

沒有記到 blocking；有 3 處 burden、2 處 optional。下列保存初讀當時問題，後文釐清不倒改原始紀錄。

1. **01-small-cnn/08，optional。** 原文「對照上面的表：`models.cnn` 底下的 `initial_loss`、`last_pre_update_loss` 是表中箭頭兩邊的 loss，`train_accuracy`、`validation_accuracy` 是後兩欄」。已讀表為兩欄四列，沒有箭頭；能靠欄位名猜，卻找不到所說位置。需要用實際列名對應。首次閱讀中沒有後文再釐清。
2. **03-identity/01，burden。** 原文「主分支的 18 個權重全部設成 0，所以 F(x) 全為 0」。先前 CNN 含 bias，此刻未說無 bias，需猜為何清零權重即可 F=0、為何不是 20 參數。直到 **03-identity/03** 的 `bias=False` 才明確釐清。應在前面使用結論前先說本節不用 bias。
3. **21-patches/04，optional。** 原文「前面切塊核對的實驗採預設 dropout=0」。此時前文尚無具體核對實驗，需猜前指；直到 **21-patches/05** 才見手工測試圖及測試投影。dropout=0 意義本身清楚，問題是導覽位置。
4. **21-attention/01，burden。** 原文「## 矩形跨格，一塊需要讀其他塊」。仍只答顏色，patch6 自己已有紅色；能說明如何互讀，卻無法由當前內容說明為何此任務需要互讀。後續訓練沒有架構對照，也明說不能歸因 attention；必要性未被解決。應區分機制展示與此顏色任務的必要條件，或說明選用背景接收者的具體資訊缺口。
5. **21-attention/03，burden。** 原文「觀察 CLS 輸出有沒有改變，而不是評分準確率。」表又稱未訓練 TinyViT 檢查，卻尚未教完整路線，無法定位取單 attention 後第0token或完整 encoder 後表示。直到 **21-transformer/04** 才定義兩 blocks／最終 LN 後 `cls`，可幫助推斷，但原表未直接連接該測量點。應在表旁明說輸出位置與 32 維表示，避免猜。

## 圖像與未驗證範圍

實際檢視 17 張當前 packet 允許的桌面 raw SVG PNG 預覽；最初 CNN 材料圖也讀了 SVG 原始碼，之後預覽就緒才實際看圖。材料與類別、掃窗、CNN 資料流、identity 旁路、attention 来源、patch格線、CLS／位置、內容交換、Pre-LN雙旁路、完整分類、loss軸及續訓流程均可讀，沒有記到必要圖缺漏。全部 figure 原始 SHA256 保存在 raw ledger 的 disclosed_source，不以預覽檔存在代替觀看。

**沒有看 Zensical 實際頁面、手機、頁面切換或實際數學排版。** raw SVG 桌面可讀不代表網站正文嵌入後可讀。沒有執行訓練、練習、checkpoint 還原，沒有查證數字／架構／原論文；文中實測是教材的報告，而非本閱讀者重跑。

四個目標頁可以完整復述，但上述首次閱讀卡點仍需處理；不以完成 46/46 或正確復述消除當場問題。這是獨立 AI 首讀輔助證據，沒有測試真人學生的學習成效。後續修後複查另存，不改此原報告或 raw 紀錄。
