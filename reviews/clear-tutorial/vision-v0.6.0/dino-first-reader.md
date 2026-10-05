# DINO 支線獨立首次閱讀紀錄

讀者背景：有基本 Python／PyTorch、NN／CNN 理解，大學數學學過但不熟，開始時不知道本專案。依 clear-tutorial 的逐段揭露規則，以 gate 為唯一正文入口；每個 unit 先用自己的話記錄，再成功 record，才 next。共用檔案系統可存取其他檔案，這是行為上的閱讀限制，沒有技術隔離。

本輪完成 **93／93 units**，最後 gate 返回 `READING COMPLETE`。其中 54 個是實際前置閱讀，39 個是 22.1～23.2 的六個 target 頁面。沒有提前閱讀全頁、packets.json／frozen-packets.json、未揭露後文、教材實作、作者筆記、其他讀者／技術審查結果或外部解釋。只讀允許的 clear-tutorial SKILL.md 和 review-protocol.md。沒有修改教材、訓練模型或 commit。

記錄時的工作樹 HEAD：`823232bf9eba4074b4db4c99b6b076feaa1a9992`。每個當前 packet 提供 source／unit／figure SHA-256；正文以那些凍結 packet 為準，HEAD 不代替來源指紋。即時 note 每次寫到 ignored 的 `artifacts/runs/vision-firstread/reader-note-dino.json` 並交 gate record；另外保存自己的 93 份理解快照到 `artifacts/runs/vision-firstread/dino-notes-snapshot.json`。沒有用此報告回改先前 note。

## 實際閱讀範圍

|順序範圍|頁面|units|
|---|---|---:|
|前置|01-small-cnn、03-identity、15-attention-bridge|9＋7＋6|
|前置|21-patches、21-attention、21-transformer、21-training|7＋5＋6＋6|
|target|22-views、22-collapse、22-distillation、22-features|6＋6＋7＋7|
|target|23-dino-versions|7|
|插入前置|04-localization|8|
|target|23-detection-bridge|6|

沒有閱讀其他 YOLO 主線頁面。外部論文及實作連結只讀到正文中的介紹，未打開核實；未執行教材程式，也未下載官方 DINOv2 權重。本輪屬於 AI 首次閱讀檢查，不是技術證據審查，也不是實際學生理解測試。

當前 packet 附上的桌機 raw SVG PNG 都已查看；重複素材以相同指紋確認為先前看過的圖。它們不是 Zensical 頁面。**Zensical 桌機、手機、數學排版、折疊框與頁面切換全部未驗證。**

## 即時卡點與後文釐清

共記錄 3 項 burden、3 項 optional，沒有記錄 blocking 概念斷點。能復述主線不會消除以下首次閱讀問題，也不代表所有視覺呈現通過。

|位置／嚴重度|當時原文與問題|後文／目前狀態|需要的調整|
|---|---|---|---|
|22-views/01，burden|「A、B 的原圖 crop 分別為 `[1,1,29,29]`、`[2,1,29,28]` 像素。」四數首次出現未說軸、順序與終點；看圖能懂 view，但要把數字對上邊界須猜 xyxy 或 xywh。|**22-views/04 才釐清**是原圖 xyxy、x 右 y 下、右下不含。原 note 保留。|把座標約定前移到首次 crop 數字。|
|22-distillation/01，burden|「再按 ①、②、③、④追蹤更新次序。」raw 桌機 PNG 中②③④呈方框，不能靠編號追圖；操作文字仍可辨識。|未關閉。只證 raw 預覽問題，沒有推定 Zensical 也缺字。|普通 1／2／3／4 或可靠字體，並在真頁驗證。|
|22-features/05，burden|「圖上半部是同一設定下的 160 步跨 view loss。」raw PNG 約130～145步低谷掉到可見下軸外，曲線消失一段；無截斷說明，不能讀完整變化幅度。|未關閉。未看實作或 JSON 追算最低值。|縱軸包含全部曲線，或標明截斷及最低點。|
|01-small-cnn/08，optional|「`initial_loss`、`last_pre_update_loss` 是表中箭頭兩邊的 loss」。當前表沒有箭頭，能靠欄位名推回，但多了無效查找。|未關閉。|改指表中前兩項 loss，或確實畫箭頭。|
|22-views/01，optional|「以 seed 101 生成 1 張圖，取索引 0」。21章 seed101 圖0是藍；本頁同 seed／索引0卻為紅。讀者尚不知是不是同一圖片或重新生成。|未關閉；沒有查生成器替正文補原因。|說明本頁另生成1張，與21章128張批次圖0的關係。|
|22-features/02，optional|「SSL 找到 train 59，random 找到 train 64」。首次 random 還不清楚是隨機參數特徵或隨機選參考圖。|**22-features/04 才釐清**是同次初始化、未自監督更新的 backbone。|首次提及即加「未訓練隨機初始化 backbone」。|

22-collapse/04 先展示分佈交叉熵數值，但明說此刻只讀「尖的常數分佈也可有很低 loss」，公式下一節才教；我當時沒有借外部公式計算。22-distillation/03 揭露軟目標交叉熵後，才把該反例連到公式。這是有明確閱讀邊界的預告，沒有記為阻礙。

## 用自己的話完整復述

前置閱讀讓我能追完整路徑：CNN 同一局部 filter 跨位置共用，RGB／NCHW／logits／類別答案對上；GAP 平均本身收起位置，特徵之前仍可能帶間接位置線索。identity residual 是原輸入加修正，shape 要相同，shortcut 係數1的直接梯度路徑不代表總梯度必為1，也不保證分類提升。Attention 把每位置排成 token，以 Q/K 比對、softmax 每接收者的來源分佈、讀 V 加權和；比例不是參數或因果貢獻。ViT 從原圖切 patch、共享投影，加入可學 CLS 與槽位位置，再經兩個 Pre-LN attention／MLP residual block、最後 LN，取 CLS 得紅藍分數。前置訓練和 checkpoint 例子分清真正 optimizer 更新、固定資料評分，以及恢復權重／optimizer／RNG 才能接同一次 CPU 訓練。定位前置則教 xyxy 半開 pixel 邊界、cxcywh 正規化、框 MSE 與 IoU 不同；兩張訓練圖的擬合不證泛化或展平比 GAP 更好。

**22.1：**不給訓練顏色標籤，從同張圖裁兩個大平方、resize、可能翻轉與亮度變化，形成兩個 global view。配對依同原圖，不是任意同色圖。ViT CLS 希望保共同內容，減少對當次增強依賴，這是自監督目標來源。大 crop 也可能裁走矩形；只讀 image 不用 box 保證包含物體。顏色任務下，交換紅藍或灰階會抹掉所需線索，view 設計仍有人的用途判斷。此頁只生成資料，未更新模型。

**22.2：**同圖兩 view 一致，並沒有要求跨圖還可區分。A兩view向量相同、B兩view另有不同向量，和所有圖都回答同常數，都能拿平方差0。這是目標漏洞，不是步數少。換尖常數分佈也可讓跨view交叉熵很低。跨圖每維標準差0能辨完整常數塌縮，但標準差非零也可能只保留背景噪聲，所以一致性 loss、特徵變化、下游用途要分開看。此頁是手工反例，沒有關 center 的訓練對照。

**22.3：**student、teacher 同結構同初權重，teacher 沒有事先知道顏色。CLS32經投影頭得K16無命名槽 logits，舊顏色頭凍住不用。Teacher 用舊 center 減 raw logits，再除較低溫度 softmax 得固定軟目標；student 不減 teacher center，使用自己的溫度。t-view1教s-view2，反方向也算，兩方向及batch平均交叉熵。Stop-gradient與optimizer範圍使只有student backbone／投影頭走梯度。Student step之後，teacher參數用新student參數EMA；center另用本步已算好的raw teacher輸出均值EMA，給下一步，不能回改本步目標。Center抑制長期單槽偏高、銳化避免全平均，兩者配合仍不保證所有資料有效。160步是真小模型CPU更新；80步存檔續至160的全狀態／loss／下一view一致，證續訓而非特徵一定有用。

**22.4：**獨立重做160步無標籤SSL，凍 teacher backbone，只取完整原圖CLS32，不取K16投影分佈。1-NN用train特徵＋此時才附上的train顏色答案，以cosine找新圖最近參考，test不能進庫。Linear probe先用train各維統計標準化三批，僅新Linear32→2用train標籤120步，backbone不動。公平baseline是同次初始化未SSL更新的backbone；兩個讀取器設定／資料／probe起點與预算相同。random和SSL在val/test兩種讀法皆64/64，因此能說兩套特徵在此顏色用途可讀，**不能說SSL提升能力**。SSL std較高、跨圖cosine較低不改這個結論，loss曲線目標隨teacher／center變而非固定答案。

**23.1：**2021 DINO、v2、v3是自監督特徵路線，2022同名DETR detector另一路。原CLS跨視角對齊，v2加入被遮patch任務及更大訓練配方，v3長訓後用參考patch的Gram關係約束；其他條件也變，不能歸功單項。手工unit patch向量兩兩內積形成Gram，沒有softmax或V混合；一起旋轉保夾角／Gram，不必逐值抄特徵，全同向量則毀關係。這不是v3訓練或完整loss。選讀官方v2 S/14固定權重，只對兩張224受控圖抽CLS384及256個patch384；query紅方塊最近是紅但第二近是綠，形狀與顏色綁定，不能稱可靠物件辨認。CLS cosine高只描述這兩圖，近鄰不是框。Register不是所有v2必有；本官方例無register，也沒跑v3。

**23.2：**回自己的tiny SSL、獨立160步後凍teacher。16個patch32按原格序換軸還成32×4×4，槽位保位置但經attention已混其他塊。Train逐位置／channel統計標準化後，512→64共用層分顏色2logits及框4raw；中心sigmoid，寬高限制.05～.75原圖比例，再轉normalized xyxy、截邊。只新head用train類別／框訓練200步，CE＋10 Smooth L1，backbone前後完全相同。每图一GT一pred直接配，顏色argmax、IoU以及兩者同時過IoU.5門檻；test64顏色全對、平均IoU.6057、54/64同時過，四張圖也展示定位失敗。這個比例沒有排序PR計算，不能叫AP/mAP；沒有random定位head或CNN/YOLO對照，不能說SSL較適合偵測。固定一框槽放第二物体也不會多輸出框，NMS只能刪候選不能生框。

## 理解變式與判讀界線

每個 unit 的 variation 都只使用當時已教概念，未教便記尚不知道。主要例子的實際預測包括：P8改P4得到64patch／65token，attention配對含CLS由289到4225約14.62倍；零Q讓該head均分來源；交換內容而位置留槽和搬整個加完位置向量不同；全部常數向量即使放大仍std0；teacher momentum=1不再追student；共同旋轉保Gram但單獨轉一向量會改關係；中心靠邊截框後原中心／寬高會改；增加第二矩形超出單框輸出能力。這些預測已保存在即時note，不拿後文答案改原預測。

閱讀程式前後都記錄任務及範圍：資料重排／view生成不等學習，手工collapse與Gram不等實訓；backward不等step；DINO student／head訓練、teacher／center EMA、凍結評测各改不同量。程式輸出正常或shape合法只支持對應流程；有限非零梯度及參數變化支持更新，獨立固定資料判准才支持當頁用途。沒有任何程式執行由本讀者重新驗證。

本輪主線可復述，仍保留上述原始卡點。兩處後文定義應前移，兩個 raw 圖問題尚待處理，真實網站呈現尚未驗證；不以完整復述把這些項目改寫成「首次閱讀與視覺全部通過」。
