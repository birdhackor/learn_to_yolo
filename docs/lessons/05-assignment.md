# 多物件輸出與責任分配：哪個預測負責哪個物件

單物件定位每圖輸出一個框，但真實圖片可能零、一或很多物件。只把head改成「輸出更多框」仍少一件事：訓練時要告訴每個候選位置該學誰，以及哪些位置學背景。本節用固定兩物件圖，將annotation一路連到target與loss mask。前置只需懂xyxy、中心寬高與分類loss。

歷史概念參考 [YOLOv1原始論文](https://arxiv.org/abs/1506.02640) 的grid與中心責任。本課MiniYOLO採每格**一個框slot**、獨立objectness、softmax類別；省略原版每格多框與其confidence/loss設計，所以不是YOLOv1重現，也不是完整多物件容量方案。

[在 Colab 執行](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/05-assignment.ipynb)，或 `PYTHONPATH=. python lesson_cases/05-assignment.py`。CPU案例生成target、驗證同格碰撞，再在人工feature map上更新head兩步；沒有從圖片訓練detector效果。

## 固定輸出，如何接變動數量標註

Annotation每張是boxes[N,4] pixels xyxy及labels[N]，N可不同，也可0。模型head固定輸出[B,S,S,5+C]，B是圖片數、S=4是每軸格數、C=2是類別數，所以本例[B,4,4,7]。

最後7軸依序為tx、ty、tw、th、objectness logit、class0 logit、class1 logit。xy經sigmoid表示格內中心偏移，wh經sigmoid表示**整張圖**正規化寬高；obj也經sigmoid，類別經softmax。這些是本課模型的定義，不能照搬成所有YOLO版本的解釋。

Backbone提取特徵，neck整理或融合特徵，head輸出候選答案。本節只驗證head與監督。Sliding window逐區切圖檢查，two-stage先提proposal再分類／修框，single-stage直接輸出一組候選；這些是管線差異，不表示single-stage不需要assignment。

## 同一個物件走過四步

![兩個物件的grid中心責任、target及正負格](../assets/diagrams/05-assignment.svg)

圖片64×64、4×4格，每格16pixels。紅框[4,4,20,20]中心(12,12)、寬高(16,16)，藍框[36,36,52,52]中心(44,44)、寬高也(16,16)。

1. 將中心除16，紅得到(0.75,0.75)、藍得到(2.75,2.75)。
2. 取下整數決定column與row，紅由row0／col0負責、藍由row2／col2負責。
3. 減掉整數格索引，兩者格內中心偏移都是(0.75,0.75)。
4. 寬高除整圖64，都是(0.25,0.25)，不是除格尺寸16。

因此兩個正格的box target都是[0.75,0.75,0.25,0.25]，class target分別0／1、objectness target為1。紅框跨過格邊界也沒關係，責任只看中心，不看整個框是否完全放在格裡。

## 正、負、ignore與mask

本模型正樣本是有中心指派的2格；其餘14格是負樣本，objectness target0。**本版沒有ignore格**。Ignore是其他assignment設計可能使用的「這個候選暫不給某項loss」狀態，不能把它與背景0混在一起。

| 欄位 | Shape | 哪些位置進loss |
| --- | --- | --- |
| box | [B,4,4,4] | 只有正格 |
| objectness | [B,4,4] | 正格與負格都計 |
| class_ids | [B,4,4] long | 只有正格 |
| positive | [B,4,4] bool | 決定框與類別mask |

負格的box填0只是容器預設值，不是在教模型「背景框應四個0」。負格class_ids填−1也不表示第−1類；class loss先用positive取出有意義的位置，才能交給cross entropy。

```python
box_loss = F.mse_loss(pred[..., :4].sigmoid()[positive], box_target[positive])
obj_loss = F.binary_cross_entropy_with_logits(pred[..., 4], objectness)
cls_loss = F.cross_entropy(pred[..., 5:][positive], class_ids[positive])
```

F是PyTorch functional簡寫。案例用retain_grad檢查**head輸出位置**：負格的框與類別梯度為0，obj梯度非0。共享的head權重仍會由其他位置更新，不能從某個負格mask=0推論整個卷積不學。這段short loss假設batch至少有一個正樣本；全空batch的有效零loss與reduction會在Grid loss節另拆解。

## 容量限制必須讓人看得見

把兩個同類紅框中心放在(12,12)與(6,6)，兩者都進row0／col0，但head只給一個slot。即使類別相同，也不能用一個框表示兩個獨立物件；案例raise ValueError，避免默默覆蓋其中一個target。

增加slot、更多候選位置、不同尺度或更複雜assignment都可能改善容量，代價是額外候選、配對規則與loss設計。Anchor slot與class軸是獨立的：某個slot可預測任何類別，不是「紅專用anchor、藍專用anchor」。

Assignment在**訓練前生成target時**決定誰負責誰；NMS則在**推論解碼後**處理同一物件的多個重複預測。兩個真實物件碰到同一個slot是輸出容量不夠，與多個slot重複預測一個物件是不同問題。NMS只能刪候選，不能補回訓練target因容量不足而遺失的第二個物件，所以不能用NMS解決本節的同格碰撞。

常見錯誤是row／col和xy軸顛倒、把寬高除格尺寸、讓負格也學框與類別，或覆寫同格target卻未報錯。先回到固定兩框的中心數值核對，再擴大資料。

## 核對、練習與答案

案例batch含一張兩物件圖、一張空圖，應印box [2,4,4,4]、obj [2,4,4]、正2／負30／ignore0。第一張正格為[[0,0],[2,2]]，第二張全負；同格同類案例應明確報碰撞。兩步head更新只核對mask與gradient。實跑box loss約0.0871→0.0861、obj0.7031→0.6839、class0.5235→0.4744，這些是人工features的結果。

練習把藍框向左移16pixels，成[20,36,36,52]。答案中心(28,44)，由row2／col1負責，target仍[0.75,0.75,0.25,0.25]。若把S由4改8，候選由16格成64格、同圖負格62；寬高target仍0.25，但中心格索引與偏移需重算。密網格的收益是更細的候選位置，代價是更多背景候選與計算。
