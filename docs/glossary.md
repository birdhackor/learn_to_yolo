# 術語快速查：卡住時回來看一眼

這是讀課文時的查表，不需先背。每節的數值例子與程式纔是主要學習路線。

| 術語 | 先用一句話記住 | 留意的地方 |
| --- | --- | --- |
| tensor／shape | 帶有多個軸的數值容器／每軸長度 | `[B,C,H,W]` 的軸順序不能只靠 reshape 改成 HWC |
| batch | 同一次 forward 一起處理的圖片 | 每張圖的物件數仍可不同 |
| channel | 同一位置的特徵維度 | RGB有3個；中間特徵不一定是顏色 |
| logits | 尚未轉為機率的分數 | cross entropy／BCEWithLogits 要喫 logits |
| sigmoid | 將一個數轉到0到1 | 可表達獨立事件；不是多類互斥選擇 |
| softmax | 把一組分數轉成合計1的比例 | 要沿正確的類別／bin軸做 |
| loss | 用一個可微分數值表示目前錯多少 | 不是 accuracy，也不直接等於 AP |
| gradient／梯度 | 參數微小改變時，loss的變化率 | backward算它，optimizer.step才更新參數 |
| backbone | 從圖片提取特徵的主幹 | 不自行決定最終框的表示 |
| neck | 在主幹與預測間整理或融合特徵 | 小模型可省略獨立neck |
| head | 將特徵轉成任務輸出 | 最後每個channel的定義必須對得上target |
| stride | 相鄰候選位置在原圖隔多遠 | pixel距離除stride纔是feature-map格數 |
| bbox | 物件的矩形框 | 本書xyxy用左上、右下畫素座標，面積不加1 |
| xyxy／cxcywh | 左上右下／中心加寬高 | pixel和normalized兩種單位要分清楚 |
| letterbox | 等比resize後補邊 | 儲存實際縮放、補邊與原尺寸才能還原 |
| GT／target | 原始真值標註／送給loss的監督表示 | target由GT轉換而來，格式不一定相同 |
| candidate | 模型可輸出一個預測的位置或slot | 候選數不是實際物件數 |
| objectness | 該候選有物件的訊號 | 本書grid是0/1監督；不同版本的分數設計不同 |
| anchor | 預設的框尺寸／比例模板 | slot和class是獨立軸，anchor不綁某一類 |
| anchor-free | 不使用預設框尺寸模板 | 仍需候選位置、assignment與框參數化 |
| assignment | 訓練時決定誰負責哪個GT | 正樣本學框；負樣本通常提供背景訊號 |
| positive／negative／ignore | 要學物件／要學背景／暫不計某項loss | 各版的規則不同；ignore不是空圖片 |
| mask | 挑出要計算某項loss的位置 | 空正樣本也必須處理合法，不產生NaN |
| decode | 把head表示轉回畫素框與分數 | 不是模型再次看圖，也不是評估 |
| IoU | 交集面積除聯集面積 | 範圍0到1；不重疊時為0 |
| NMS | 高分框保留，抑制同類高重疊候選 | 比的是prediction與prediction，可能誤刪靠很近的真物件 |
| matching | 評估時把prediction與GT配對 | 同圖同類；GT最多成功配對一次 |
| TP／FP／FN | 正確偵測／錯誤預測／漏掉的真值 | 高分錯框仍是FP |
| precision／recall | TP/(TP+FP)／TP/(TP+FN) | 提高顯示門檻可能讓recall下降 |
| PR／AP | 排序後的precision-recall曲線／其插值面積 | AP不是單點precision×recall |
| AP50／mAP | IoU門檻.5的AP／依約定跨類別平均AP | 本書用有GT類別的平均；不是COCO多IoU門檻AP |
| DFL | 對連續邊距離的相鄰bins做加權CE監督 | 距離bin分類與物件類別分類是兩個任務 |
| attention | 依相似度，把其他位置的value加權混合 | Q/K/V、token數與互動範圍決定成本 |
| overfit／泛化 | 擬合所見資料／對未參與更新的資料也能工作 | 小資料overfit透過，仍不保證泛化 |
| validation／test | 用來選設定／最後評估已定設定 | 反覆拿test調參，它就不再是獨立test |
| latency／throughput | 完成單次工作的時間／單位時間處理量 | batch加速不一定降低單張延遲 |
| tracking | 跨frame維持同一物件的ID | 偵測框正確不保證ID不交換 |
