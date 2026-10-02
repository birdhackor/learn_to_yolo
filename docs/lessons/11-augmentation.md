# 增強：畫素怎麼變，框就怎麼變

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/11-augmentation.ipynb){ .md-button }

前置：[資料契約](07-data.md)、[自己的資料](08-own-data.md)。這次只改訓練資料變換。增強能產生位置和外觀變化，但若框沒跟著圖片走，會把正確標註變成錯誤監督。先使用可逆、可手算的flip，再使用會裁掉物件的crop。

歷史來源：[YOLOv4](https://arxiv.org/abs/2004.10934) 討論bag of freebies，包括Mosaic等訓練策略；[YOLOv5 v6.0 augmentations.py](https://github.com/ultralytics/yolov5/blob/v6.0/utils/augmentations.py)提供工程實現。這些包含多種策略，不等於本節兩個操作。本章簡化為人工固定horizontal flip與crop，沒有Mosaic四圖拼接、mixup或隨機透視。起始是64×64 grid資料，先驗證同步變換，是否在訓練保留由固定budget對照決定。

## Flip 的邊界要用半開區間

紅框 `[8,12,24,28]` 在寬64圖片上水平翻轉。原畫素 x=8…23變成x=55…40，因此新框是 `[40,12,56,28]`。公式是 `new_x1=W−old_x2`、`new_x2=W−old_x1`；y不變。框坐標代表畫素區間的邊界，不是單個畫素索引，所以不用W−1。畫素本身的索引翻轉則是W−1−x，兩者切勿混淆。

```python
flipped_image = image.flip(-1)  # CHW 的最後一軸是寬
new_boxes[:,0] = W - boxes[:,2]
new_boxes[:,2] = W - boxes[:,0]
```

labels不變，紅矩形仍是class0。連做兩次flip應逐值恢復原圖和框，是很強的可逆檢查。注意若類別依賴方向，例如左轉箭頭，flip可能改變語義；這時應重新定義標籤變換或禁用該增強。

## Crop 會改座標，也會改可見面積

從原圖裁出left=16、top=8、width=32、height=32的區域。原框先減掉offset，得到 `[-8,4,8,20]`，再裁切到新圖範圍，成 `[0,4,8,20]`。剩下寬8、高16，visible area=128；原面積256，可見比例=.5。

案例規定「正面積且visible比例>=.5才保留」，所以留下；若閾值改.6就移除框。使用同一個keep mask過濾labels，不能只有boxes變少。閾值是訓練標註策略的選擇，可能讓被裁剩的小物體成為未標註前景；需觀察此策略是否適合任務，不能把刪除比例高當成增強越強越好。

```python
shifted = boxes - torch.tensor([left,top,left,top])
clipped = clip_to_crop(shifted)
keep = (visible_area > 0) & (visible_area / original_area >= .5)
new_boxes, new_labels = clipped[keep], labels[keep]
```

裁剪後輸入是 `[3,32,32]`，框也是32×32座標；若模型仍要求64輸入，接著要同時resize/letterbox圖片與框。再使用新的框中心建立targets，不能沿用增強前的cell。flip後的紅中心48,20在4×4 grid變到x3,y1；監督位置確實應改變。

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/11-augmentation.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/11-augmentation.py`。應看到flip框`[40,12,56,28]`、雙flip還原、crop框`[0,4,8,20]`、可見面積128/256=.5，以及.6移除結果。程式把變換後彩色pixel與新框區域核對，實際用keep篩labels：.5時留下值為`[0]`的long tensor，.6時boxes變shape`[0,4]`、labels變shape`[0]`的空long tensor。空圖是沒有目標的圖片，案例使用全零像素配空boxes及空long labels，flip和crop後三者都仍正確；不能以帶紅物件卻空標註的圖代替空圖。這是資料幾何實驗，不需backward，也沒有AP結論。

## 收益與代價

同步增強提供新的位置／尺度／外觀，可能減少對固定背景和位置的依賴；代價是資料處理、標註規則與更難最佳化的樣本。validation/test固定前處理，通常不使用隨機訓練增強，否則每次指標都在變。固定seed利於重現，不等於每個epoch必須用完全相同增強；正式實驗應記錄隨機策略與預算。

常見錯誤：flip channel或height軸；用W−1變換框邊界；先修改x1再用它算x2；crop後留下零面積框；漏掉labels的keep mask。自主練習：64寬圖上的整張框`[0,0,64,64]` flip後是多少？答案：不變。紅框只crop到一半時是否一定刪除？答案：取決於明寫的可見比例策略，這裡.5保留，.6刪除。
