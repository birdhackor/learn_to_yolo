# ViT／DINO 十頁第三輪：前後銜接獨立審閱

日期：2026-10-05。審閱者：`vision_transitions`，不同於作者、首次閱讀者及技術審查者。背景：基本 Python／PyTorch、NN／CNN，大學數學學過但不假定熟練。

**結論：此處列出的 21.1–23.2 十頁之文字、概念、材料與數據口徑的前後銜接 scope 通過；未發現必改項。** 這是第三輪全文非盲讀，不是第一輪分段盲讀，也不是全書驗收或真人學生理解測試。

審閱開始前已讀實際第 1 章、3.1、15.1 與閱讀路線，理解起點保存於 [transitions-prerequisites.md](transitions-prerequisites.md)。收到明確開放後，依 21.1→21.2→21.3→21.4→22.1→22.2→22.3→22.4→23.1 的順序閱讀；23.2 前完整補讀實際 `04-localization.md`、`06-evaluation.md`，再讀 23.2。在完成逐頁判讀與原稿結論寫入前，未開作者、首次閱讀者或技術審查報告作為答案，也未讀目標實作、notebook 或獨立 JSON。以下數字只核對正文／頁內紀錄的口徑，不重做技術證據驗證。

**收尾的意外接觸紀錄：**原稿已寫入後，檢查繁簡字的 `rg` 呼叫漏限定到本報告，意外在全 repo 搜尋，截斷輸出中出現少量 notebook 片段與舊第16章審閱片段。沒有拿這些片段補寫已完成的概念判斷，也沒有重標為技術審查；但因此不能宣稱本回合從未接觸 repo 其他內容。後續搜尋已限定自己的報告。這一輪始終標為非盲讀，沒有第一輪盲讀通過的主張。

工作樹 HEAD：`823232bf9eba4074b4db4c99b6b076feaa1a9992`。各頁閱讀開始前保存 source SHA-256；讀完複核，十頁及兩份補讀前文均無 bytes 變動。最初四份前置也仍與前置紀錄的指紋相同。工作樹未提交內容以各頁指紋為準。

## 實際補讀的定位／評估前文

4.1 的當地定義足以支援 23.2：原點左上、x 向右／y 向下、半開 xyxy、pixel 與正規化比例、中心／寬高換回邊界、sigmoid 並不保證框在圖內、需另定截邊規則。4.1 行 46 用固定格子對線性權重的數字例子，解釋展平保留位置與 GAP 平均的差別；行 115 手算交集／聯集，行 117 明說 MSE 與 IoU 是不同量。兩張訓練圖、3 步／40 步的分數未被當成新支線的獨立測試或公平對照。

6.2 行 17 複習半開框與 IoU；行 19 說明逐圖片／類別配對，行 44–51 區分 precision、recall 的分母，行 59–103 說明 score 排序、PR 包絡與 AP50。這讓我能辨識 23.2 的單圖直接配對比例缺少 AP 的排序／PR 計算，不會只因門檻同為 0.5 就叫 AP50。本輪沒有依賴未讀的多物件 assignment、NMS 算法或 letterbox；那些不是完成當前單物件例子所需。

## 21.1：feature token 換成原圖 patch

- **實際前文→當節：**15.1 的 token 是 feature map 上一個位置的 channel 向量；本頁行 5 提出「如果直接從原圖開始，一個 token 要裝什麼？」才改到原圖 patch。行 11：「沿用第 1 章的答案規則……矩形的寬、高、位置和亮度會變，背景也有少量暗色雜訊。」保留紅 0／藍 1、32×32 RGB，明說不再是固定 12×12 的八張圖。
- **理解與下一節入口：**raw patch `[1,16,192]` 是按 channel／row／column 整理的像素；可學投影才給 32 維 embedding。CLS 不是 patch 或 label；位置向量逐值相加、不串接，槽位由排列順序對上原圖。行 71 明說均色手工圖「不是前面的矩形訓練資料」；行 79 分開 192→3 人工 RGB 平均投影與真正 192→32 embedding。下一節要交換的是已構造的 17 個 token 特徵，不是搬像素。
- **讀者小變化：**若保持 P=8，只把 D=32 改成 16，raw 向量仍 192、patch 仍 16、加 CLS 仍 17；改的是每個 embedding／CLS／位置向量的長度。這由 P 與 D 的當地定義可直接預測，不能把 D 當圖寬。
- **卡點／需改：**無必要說明斷點；不需改。必要圖面、實際 patch 邊界可讀性未由本審閱者視覺驗證。

## 21.2：單 head 的 Q/K/V 接上四 heads

- **前→當：**行 5 清楚沿用 16 patches＋CLS 與 32 維向量。行 21：「Q、K、V 先各有 32 個特徵，再各分成 4 組，每組 8 個特徵。」由 15.1 的 d 定義可接上此處 √8；分類 head 與 attention head 在當地分開。序列 T／程式 N 同義，`[B,4,17,17]` 後兩軸仍是接收／來源。
- **當→後：**行 50 明說交換檢查用完整、未訓練模型的最終 CLS，且「下一節才拆開這條完整路徑；此刻只觀察……有沒有改變」。因此本輪把未教的 LN／完整 blocks 當明示的黑盒，不猜它們的公式來算答案；21.3 隨即實際拆開。15.1 的四個手工 tokens／一次 SGD 與未訓練 TinyViT 交換分別標明，沒有偷換成分類訓練。
- **口徑：**softmax 的列和為 1，回傳的是 dropout 前權重；`eval()` 機制檢查關閉 dropout，並說明啟用後實際列和可變。交換未加位置內容只是重排；固定槽位位置向量再交換內容才改配對。最大特徵差不等於分類正確率；單 head 289→4225 的約 14.62 倍含 CLS，沒有當成執行秒數倍數。
- **小變化：**若固定一個 head 的接收 Q 為零，其他 K 如何不同都得零分，各來源占 1/17；不能因此說四個 heads 的最終投影或原 RGB 也直接等於平均。所需概念已由 15.1／當頁提供。
- **卡點／需改：**對「未訓練整圖特徵」與「訓練後答案」的邊界當地清楚；無必改。圖箭頭與換位圖面未驗證。

## 21.3：attention、MLP 與兩次 residual 的先後

- **前→當：**3.1 已教逐值 `x+F(x)`、shape 與相加後 ReLU 的範圍；本頁行 11 說做兩次這種修正，行 19–29 在正式公式／程式前定義 LN、Pre-LN、MLP／GELU。LN 每 token 處理特徵軸，MLP 共用權重但不跨 token，故不能把 MLP 加寬當更多位置互讀。
- **當→後：**行 54：「第二行相加用的是第一行已更新的 tokens，對應公式中的 U。」兩條 shortcut 留原值，LN 在修正路上，最終 LN 位於 blocks 外；`cls`、`patches`、`tokens` 的資料表清楚支援後續分類／定位。行 83 說此頁沒有 optimizer 更新，下一節才更新全模型。
- **小變化：**若第一段 attention 修正為零，U=X；第二段仍應計算 `MLP(LN2(X))`，Y 不必等於 X。若兩段修正都零才 Y=X，但 blocks 外的 LN 仍不保證 identity。由當地公式能預測，不把 3.1 的零分支結論套到整個 TinyViT。
- **卡點／需改：**無必要跳躍；不需改。實際 Pre-LN／shortcut 圖和公式排版未驗證。

## 21.4：機制檢查換成完整訓練與接續

- **前→當：**承接相同 TinyViT，重新交代 128／64／64 圖、seed 101／202／303，train 可更新、val/test 不可。行 29 明說 optimizer／設定和第 1 章不同，不能以分數比較 CNN／ViT。dropout 的訓練／eval 差別接上 21.1–21.3。
- **口徑與當→後：**行 51：「每次記錄的 loss 都在該次更新之前，並且有當次抽樣和 dropout 的影響。」曲線兩端 0.730835／0.008318 是不同抽樣 batch；表中的固定 train 0.710059／0.007155 與 val/test 用 eval()，不是同一種量測。分類分母、argmax 與沒有框／IoU 已說。checkpoint 定義先於 RNG state、optimizer 移動平均與續訓比較；恢復 seed 不等於恢復亂數位置。結尾說保留「讀圖路徑」、改學習訊號，22.1 先處理 view、22.3 才說自監督更新。
- **小變化：**即使模型與 optimizer 恢復了，若 RNG 回到 seed 7 的開頭，下個 batch／dropout 可不同，所以第 31 點 loss 不保證與上路一致；仍可用載入權重作 eval 推論。這由當頁 checkpoint 材料直接推出。
- **卡點／需改：**無必改。未實際讀 checkpoint、重跑接續或查看 loss 圖面。

## 22.1：從顏色答案換成同來源兩個 view

- **前→當：**行 5 說訓練時收起顏色答案、只留圖；行 9 明說示範紅圖另生成，「不是取前章 128 張資料裡的索引 0」，所以 seed 101／index 0 不會被錯認成前章的藍圖。crop 的原圖 pixel、xyxy、半開邊界先定義，view 縮放／翻轉後不能沿用原框。
- **當→後：**同原圖兩次隨機變化形成 view，不是任意兩張同色圖。亮度不改色相，crop 可能裁走物件，沒有 box 保證。行 33：「原圖 A 和另一張原圖 B 是否也應該相近，不能從這個條件推出。」這正好接 22.2 的 collapse。此頁只做資料操作、沒有更新 ViT；投影頭／DINO 分佈只作下一步問題預告，不借未讀細節推理。
- **小變化：**若將示範 `[0,0,8,8]` 空 crop 放進一致性訓練，模型可能被要求把只有背景的 view 與含矩形的 view 拉近。共同來源並不保證共同物件；不能以「無 label」判增強適合顏色用途。
- **卡點／需改：**無必改。view 圖實際是否顯示物件、框及文字未驗證。

## 22.2：一致性與可區分性分開

- **前→當：**先增加一張藍圖 B，明示兩維手算、不跑 ViT且不是紅藍機率。平方差例子完全用已知逐值相減／平方；兩模型同圖 loss 都零，常數模型卻無法分 A/B。新增八張 32 維／K=16 手工分佈時，行 40–44 明說是另一個數值反例、沒有參數更新；交叉熵計算留給下一節，沒有把這頁當防塌縮訓練。
- **當→後／口徑：**行 36：「一致性 loss、特徵變化、下游表現各自回答不同問題。」std 非零可能只是噪聲，不能推紅藍能力；K 個輸出槽與 32 維 backbone、兩個人工特徵數沒有混同。teacher／center／溫度是下一頁要追的量，不把未知算法當當前反例的原因。
- **小變化：**若把八個答案改成七紅一藍，常數猜紅可答對 7/8，但同圖平方差與跨圖 std 仍零。較高總正確率不能自行證明常數表示保留了圖片差別；理由由本頁的輸入同一與類別分母可推出。
- **卡點／需改：**無阻礙主要概念的卡點；不需改。未驗證 collapse 圖呈現／數值執行。

## 22.3：teacher／student、center 與權重更新分開

- **前→當：**兩份模型同結構同起點，teacher 不是已知紅藍答案的外來模型；行 77 不下載權重、不要求前頁 checkpoint。原分類頭凍結不使用，新的 projection head 將 CLS 32 維轉 K=16 槽，且槽沒有紅／藍名稱。K=3／τ=0.1／EMA=0.9 的手算與實驗 16／0.08／0.15／0.95／0.9 分開。
- **當→後：**舊 center 減 raw logits，再除 teacher 溫度／softmax；student 不減此 center。軟目標交叉熵定義在更新前，交叉配對兩個 view、停止 teacher 梯度、只 optimizer student，然後更新 center／teacher 給下一步。文字順序與程式先 teacher EMA 後 center 的差別，行 88 明說 center 仍使用本步前向已算的 raw outputs，因此無需猜是否重算。RNG／optimizer 接續要求延續 21.4，另外加 student／teacher／center／schedule 的完整狀態。loss 約 1.943→1.874 只證明此路徑與更新，下一頁才評特徵用途。
- **小變化：**把 center 手算的 `m_c` 改成 0，下一步 center 立即等於本 batch raw-logit 平均 0.3；本步已產生的目標仍用 0.1。不能用新 center 回頭改當前 target，也不能把此改動當 teacher 權重已複製 student。
- **卡點／需改：**無必改。此處公式把第 1 章的 `ln` 寫成 `log`；本輪依前文同名交叉熵的自然對數理解。若要額外強化紙筆核算，可註明 `log=ln`；屬選讀改善，未阻礙此處更新量／先後／軟目標的理解。未重跑 EMA／續訓或核實圖箭頭。

## 22.4：凍結表示之後，標籤才進讀取器

- **前→當：**行 11 明說從隨機參數重做 160 步，不依賴上一節的執行或舊 checkpoint。取 teacher backbone 的原圖 CLS `[32]`，不取 K=16 投影分佈、不抽新 crop。凍結／eval 特徵不更新，1-NN train labels 借答案、test labels 計分；linear probe 則是另有標籤的新頭訓練，兩段清楚分開。
- **口徑／當→後：**cosine 先消長度，與 attention 原始內積區別已明說。線性頭標準化只用 train 統計，和逐 token LN 是不同操作。random 保留同一次初始化、SSL 為訓練後 teacher；兩個新頭同 seed 900／optimizer／120 步。64/64 分母明確，沒有只呈 SSL。std／cosine 是 64 張 test 的維度／兩兩統計，與22.3的train diagnostics不混同。此頁說明圖級 CLS 用途後，23.1 再問局部 patch。
- **小變化：**將一套特徵每個向量乘同一正數，1-NN 的 cosine 排序不變，因為先除向量長度；卻不能把這推到未正規化 attention 內積或跨圖 raw std。這個區別來自本頁及15.1已讀內容。
- **卡點／需改：**無必改。下游得分與凍結確認未由我執行，近鄰圖面未驗證。

## 23.1：tiny 範例、Gram 手算與官方特徵分開

- **前→當：**開場承接凍結特徵評分，行 9 立即分開 notebook 的三向量主例與後段官方選讀；不把「換官方」誤當必下載。DINO 原始、v2、v3 與同名 detector 的材料來源分清楚，表明不是控制變因效果比較。
- **當→後／同詞：**Gram 先給單位向量、算內積、定義矩陣，再說無 softmax／不用 V，避免和 attention affinity 同名混讀。旋轉維持關係與constant改掉關係只驗手算，不說有DINOv3訓練。官方v2明說另一個架構與已訓練權重：224圖／P14／256 patches／384維、無register，與tiny32／P8／16／32不同。兩張紅方塊綠圓藍三角是新材料；top-1／top-2與形色綁定、CLS cosine非機率／IoU都交代。行104：「下一節會回到自己的 tiny DINO」，所以23.2的32圖不是誤把官方224圖硬還原。
- **小變化：**若只把手工 C 改為 `[-0.7071,-0.7071]`，A/B 間仍0，C對A/B兩對由正變負，C與自己仍約1；改動落在C那列／欄，不是softmax各列分母一起重算。只用本頁定義即可回答。
- **卡點／需改：**無必改。未下載官方模型、重做預處理或查外部論文，Gram／官方近鄰圖面未驗證。

## 23.2：patch 序列接回單物件框

- **實際前→當：**已補讀4.1／6.2，且21.3實際保留有順序的patch輸出；開場說回到小型ViT，行15再明說從隨機做160步、凍結teacher、不下載v2/v3。行9「材料不換」指回到22.4的32×32單矩形規則，並立即列128／64／64和101／202／303；不是延續23.1官方的兩張224圖。
- **當→輸出：**`[B,16,32]`先transpose成`[B,32,16]`再reshape成4×4，和15.1逆向還原同原則；attention混入遠方內容但槽位仍對應原patch，避免把grid特徵當只看8×8的框答案。展平512→64→分類／框分兩路，正規化中心與受限寬高、xyxy截邊／乘32都在loss之前交代。和4.1的模型、MSE normalized cxcywh不同，此處明確採用Smooth L1 normalized xyxy與權重10，不以兩節數字排名。
- **評分與後續邊界：**每圖一GT一pred直接配對，分類答對與IoU≥0.5是並行判準，平均IoU是另一項；行54：「沒有對置信分數排序，也沒有 PR 曲線，所以不是 AP50 或 mAP。」test54/64、平均0.6057、test0類對框0.4459未達標不互相冒充。單槽不需assignment／NMS，回主線grid是下一任務，不暗稱已完成多物件。
- **小變化：**把聯合成功門檻提高到IoU≥0.6，四個列出的例子只剩test3的0.6326達標，test2的0.5611轉為失敗，紅藍分類不因此改變。全部64張的聯合數只可確定不增加，不能僅據四圖算出新的整批分子，更不能稱新AP60。
- **卡點／需改：**無必改。未執行head訓練、讀完整64框或視覺檢查預測圖。

## 範圍限制與指紋

本輪只對已列十頁和實際相關前文的文字／資料表示／例子轉換／術語／數據解讀銜接作結論。每頁都有能由已讀內容預測的變化，不用作者意圖或程式補必要說明。沒有執行訓練、驗證模型與JSON技術一致性、外部原始來源查證或桌面／手機瀏覽器檢查；頁面裡的圖說與需要圖的問題已辨認，實際圖的可見性／可讀性仍不是本審閱者完成的證據。沒有真人學生參與，也不宣稱全書的52節皆已驗收。

### 本輪實際讀過的來源

| source | bytes | SHA-256 |
| --- | ---: | --- |
| docs/lessons/21-patches.md | 10463 | f726d6ef624a7954f8b881b3ca77019a64a2c35085a6d24ce1775b939ffa8baa |
| docs/lessons/21-attention.md | 10440 | 8390a0b5e807f97d9bc9efc1353a9ea64ec678dacd1022f96bd416e470b1edc3 |
| docs/lessons/21-transformer.md | 9534 | ddd746d8b773ee45165fea6306a8d56f3c12e26b7d1534c10487bc8adf1cbb47 |
| docs/lessons/21-training.md | 15183 | a540dfe8ed01695d87c33314c6c2b6c92ff8086362daa05486ac8987ca0f440f |
| docs/lessons/22-views.md | 9085 | 3c78cab45368317420b3287b6bac61ac3096cca2c83ace8ca4428675004ca1fc |
| docs/lessons/22-collapse.md | 7798 | b5e3dac06b3fc34668c454b2b91295b4cfd56d0c30dd6ebeddc5b2b6651a8f29 |
| docs/lessons/22-distillation.md | 13573 | c188d3699bcbc90c91a1d215cfe53e24bd92c61bfe685964625dff21377e5ba4 |
| docs/lessons/22-features.md | 12747 | 9931f087895acdb5862d7df7b5959dec88041eb7fcbc839ef9e0104734628ed2 |
| docs/lessons/23-dino-versions.md | 12453 | d1e093fd14b9268c8d47f988956bcfdb309a22beee8679bcdd90f31b6688108e |
| docs/lessons/23-detection-bridge.md | 13490 | 16e7dcbe497493fcfd2c2e5881194f2595ecfb2d11959ad01949334fc0f39236 |
| docs/lessons/04-localization.md | 24974 | 7857e6e2c9c8f420d8b0cb89cbb25572754033ae19094c6a3a0b70c2899a96f3 |
| docs/lessons/06-evaluation.md | 22105 | 8e1de6c80cd2027951169ea056feac1b02084019ff87fba6316e756c7f38206f |

### 正文引用圖指紋

下列為完成時保存的內容指紋；僅 hash，不代表實際圖面／頁面視覺驗證。

| source | SHA-256 |
| --- | --- |
| docs/assets/diagrams/21-materials.svg | f6326e65d13df7a41a46377cd7b8bb468e5a8c37b318e33cc33a797c81cf1391 |
| docs/assets/diagrams/21-patch-grid.svg | 5a5ba8c89227d49f38627ff7bb434be054c2313cbf6d8d1b3cb879c2833b3ede |
| docs/assets/diagrams/21-token-position.svg | 0588efde6f7cad857cd2a849933b8ed80f3b3f84c230d922787b1897c9038135 |
| docs/assets/diagrams/21-attention-exchange.svg | b0e967a5ff76670cee2caddb5acbc00666298ecd434cc6a0cae79e1f520639fb |
| docs/assets/diagrams/21-position-swap.svg | 76250708df2071b9985faa950f7da42fe3b29e4fc276b9562579344f5ae71987 |
| docs/assets/diagrams/21-preln-block.svg | 94341723952941101a6145c1f8421c068e9d469a99165c71f346c1e62b571075 |
| docs/assets/diagrams/21-classification-flow.svg | 3e10ea30f2e11c98dd47f6dd2a67b901b3131e185fa7fa8e81fe749f51c9b900 |
| docs/assets/diagrams/21-training-loss.svg | df47bd7953e50fdad41d8c35e74647509c3202df8a861cf5b69b107f1054dd8a |
| docs/assets/diagrams/21-checkpoint-resume.svg | 4e303f6f1f86907193c9196740f4940379d6eab1f3bf1dad56a70d5dbbca0593 |
| docs/assets/diagrams/22-views.svg | c6ac553db3b63d61ac75d38b7cb98640b671434e71e1781c108815b3a24a8d31 |
| docs/assets/diagrams/22-collapse.svg | 75355d38843db229681ee0358fb2b552a5b20773844462e426ffd8e611b331d8 |
| docs/assets/diagrams/22-distillation.svg | e216231e715cd2a9684c3bbdc49410e3a671ac18d315d4135c031177161bc1f2 |
| docs/assets/diagrams/22-neighbors.svg | 951440b4c4ceeccea25d66f9bd17c4cec44ef7ba7fc16497e1d390a9b838f73a |
| docs/assets/diagrams/22-features.svg | 73b81c9c9853e845688921e2bfb8c532c704c4a0d84ddf021fc278b3590b547b |
| docs/assets/diagrams/23-gram-relations.svg | 61451b840d9c6c3a917b88b73c4c4b863fd031e52b84182ab3abdbd557d2ea4e |
| docs/assets/diagrams/23-dinov2-patch-neighbor.svg | 26aeadd087c5970f2d5c25cdbc1a276e6be4dca757f272802f37f3529d0ff048 |
| docs/assets/diagrams/23-detection-predictions.svg | a52635e89a4ef0975e796fb596c25ea0aebf5e43612042946ab812feab7baaa3 |
| docs/assets/diagrams/23-detection-bridge.svg | acf675cb2992480924913bb7ac24d240182ce282946ab5ea9c3918c2a8731dca |
| docs/assets/diagrams/04-localization.svg | a4d35a3b11a2cae489f45a232a0343c73bfbb09aabbd1be4acacca9f567b700e |
| docs/assets/diagrams/04-localization-learning.svg | dc8c7a1d69ef0a22a6e8a399bed8349ddd5eacd7cef1f57a45c00cd2de388baf |
| docs/assets/diagrams/06-evaluation.svg | 8289113b070ec540923910c95227d2592d09545c0cb2e6ce215091b8ce8674fc |
