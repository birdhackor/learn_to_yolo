# 13.2 NMS-free：拿掉 NMS 前，重複候選學會了什麼

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/13-nms-free.ipynb) · 原始碼：`lesson_cases/13-nms-free.py`

讀過 NMS 與一對多／一對一監督即可開始。假設一張圖有兩個物件，偵測器卻輸出三個高分框，其中兩個是同一物件。把 NMS 函式刪除，框數變多；這並沒有改善模型。本節固定框位置，只改「哪個候選被教成正樣本」，親眼看分數的變化，再比較後處理。

歷史機制是YOLOv10訓練出推論用的一對一head（把特徵轉成預測的輸出模組），使用分數選擇而省略NMS的兩框IoU排除。實際YOLOv10訓練有共享特徵與一對多／一對一雙分支assignment；本次直接指定哪個候選是正target，沒有重現那些選擇步驟。起點是四個人工候選，簡化為四個可學二元logits，不學框、不接CNN，隔離監督的作用。兩條訓練採相同步數、learning rate和初值。這是原因驗證，不能稱作完整YOLOv10或已達成真實任務精度。

## 明確知道哪些框是重複

GT A 在 `[0,0,10,10]`，B 在 `[20,0,30,10]`。候選 p0 恰好等於 A，p1 為 `[1,0,11,10]`，p2 等於 B，p3 為無物件背景 `[40,0,50,10]`。四個框 shape `[P,4]=[4,4]`，四個 logits shape `[4]`。本例只有一類，分數為 `sigmoid(logit)`。

p0、p1 相交面積90，各自面積100，IoU為 `90/(100+100−90)=90/110≈.8182`。NMS閾值 .5 時它們互相競爭。p0 和p2不重疊，應各保留一個；p3應以低分排除。

| 分支 | p0 | p1 | p2 | p3 | 訓練訊息 |
| --- | --- | --- | --- | --- | --- |
| 一對多 target | 1 | 1 | 1 | 0 | A 的兩個候選都被獎勵 |
| 一對一 target | 1 | 0 | 1 | 0 | A 只獎勵 p0，其餘學低分 |

兩套 logits 都從0開始，分數 .5。每步 mean BCE的梯度為 `(sigmoid−target)/4`，用相同SGD更新100步。結果正target接近 .96、負target接近 .04；不是手動把one-head分數填低，而是loss真正教出來。

```python
loss = F.binary_cross_entropy_with_logits(logits, target)
loss.backward()
optimizer.step()
scores = logits.sigmoid()
```

## 三條推論路徑比同一組答案

score threshold設 .5。一對多不做NMS留下p0,p1,p2，共3框；一對多加NMS留下兩框，符合兩物件；一對一不做NMS留下p0,p2，也是2框。分數相同時NMS可能保留p0或p1，本例檢查框數和物件覆蓋，不依賴未保證的tie排序。

NMS-free仍然可以有score filtering與top-k。NMS特有的步驟是按兩框幾何重疊刪除候選，top-k只是依分數保留固定數量。再用人工scores `[.92,.90,.80,.05]`取top-2，得到p0,p1，兩個都屬A，反而漏B。這個反例說明top-k本身不具「每物件只留一個」能力；它需要適合的監督與足夠好的一對一head。

也可以先手算這組人工框的評估：IoU matching門檻設.5，按.92、.90、.80排序，p0第一次匹配A是真陽性TP；p1重複匹配已用過的A，算假陽性FP；p2匹配B是TP。兩個GT都找到，所以漏檢FN=0，precision=`TP/(TP+FP)=2/3`，recall=`TP/(TP+FN)=1`。逐點precision為1、1/2、2/3，recall為1/2、1/2、1；all-points interpolated AP面積為`.5×1+.5×(2/3)=5/6`。NMS後或人工一對一結果只留兩個正確框，此例AP=1。這是可手核對的人工評分，不是模型在獨立圖片上的效果；top-2若留下p0,p1，recall反而只有.5。

執行 `PYTHONPATH=. python lesson_cases/13-nms-free.py`，核對 many/no-NMS為3框、many/NMS為2框、one/no-NMS為`[0,2]`、duplicate IoU約.8182，以及top-2漏掉p2。每條分數訓練都實際backward和step。本例僅CPU，框完全已知，沒有資料或模型權重下載。

## 省了哪個成本，又增加哪個風險

省去NMS可以減少逐對IoU、排序後的依次抑制及部署工具對此運算的支援需求，尤其當候選多時；實際速度仍受候選數、硬體與實作影響。代價是訓練需要學好唯一負責人；一對一正訊號較稀疏，排名不穩或兩物件非常靠近時，可能出現重複、漏檢或錯誤winner。不能只觀察「輸出框很少」，那也可能是所有分數都太低。

真正模型的完成條件應是：held-out AP、重複FP與漏檢都可接受，同時固定解析度、輸出上限、score threshold與裝置，量測端到端時間。本節沒有做這個效果試驗，結論限於「改target確實會改變重複框的高分傾向」。

常見錯誤是每個GT有一個positive便認為每張圖只剩一個框、在train用one-head但eval誤取many-head、把top-k叫成NMS、或看見重疊真值便希望系統刪除其中一個。NMS-free的目標是同一物件不重複，不是所有互相重疊的物件只留下一個。

自主練習：將one-target改成`[0,1,1,0]`。答案是一對一仍有兩個高分候選，但A由p1負責。這提醒我們唯一性不要求與原始GT完全同位置，仍須計算定位IoU。再將score threshold升到.99：這次兩分支可能全被過濾，框數雖為零，recall也為零；threshold不是架構成功的證據。

來源查覈：2026-10-02。[YOLOv10 論文](https://arxiv.org/abs/2405.14458)、[官方 v10Detect 推論與 postprocess](https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/nn/modules/head.py)。
