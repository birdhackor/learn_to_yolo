# 單物件分類與定位：類別之外，還要回答在哪裡

分類器回答「這張圖是紅或藍」，定位器還要指出矩形的位置。一張圖只有一個物件時，可以從相同backbone分出兩個head：一個分類，另一個輸出四個框數字。前置只需懂卷積與loss；本頁定義框、正規化和IoU，不假定已讀過偵測章。

本節是MiniYOLO之前的簡化單物件模型，不是某個完整YOLO版本。Backbone是抽特徵的主幹，head是把特徵轉答案的末端；本例沒有背景圖或變動物件數，這些限制下一階段才處理。

[在 Colab 執行](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/04-localization.ipynb)，或 `PYTHONPATH=. python lesson_cases/04-localization.py`。CPU、兩張32×32人工圖形、3步SGD；展示梯度和框疊圖，不宣稱已學會定位。

## 分類摘要為何未必適合位置

Global average pooling把每個channel的所有位置取平均。例如4×4特徵圖只有一個值1，其餘0；1在左側或右側，平均都是1/16=0.0625。案例先人工建立兩張feature map並assert平均相同，證明這個摘要沒有保留排列。

卷積的邊界、padding或其他特徵可能仍含位置訊息，不能由此斷言所有GAP模型都完全不能定位。更直接的教學方式是讓框head讀取保留空間的feature map。代價是對固定尺寸的依賴與更多參數。

本例backbone的 `[B,3,32,32]` 經4channel卷積及pooling成 `[B,4,16,16]`。分類head對空間平均得到 `[B,4]`，再輸出 `[B,2]` logits（尚未轉成機率的原始類別分數）；框head將4×16×16展平為1024個值，輸出 `[B,4]`。B是batch圖片數。Flatten保留固定位置對應，不等於空間平均。

## 四個框數字究竟是哪四個

圖左上角為原點，x向右、y向下。`xyxy` 是左上與右下：\((x_1,y_1,x_2,y_2)\)。本課框採半開區間，寬是 \(x_2-x_1\)，不加1。`cxcywh` 是中心、寬高：

\[
c_x=(x_1+x_2)/2,\quad c_y=(y_1+y_2)/2,\quad
w=x_2-x_1,\quad h=y_2-y_1.
\]

真值框 `[4,6,16,18]` pixels轉成 `[10,12,12,12]`。在32×32圖，把x與w除32、y與h除32，得到 `[0.3125,0.375,0.375,0.375]`；這四個值是無單位的正規化目標。非正方形圖片應分別除寬W和高H，不能一律除同一個數。

框head的四個logits經sigmoid限制到0到1，表示正規化cx、cy、w、h；這是四個座標比例，不是四種類別的機率。還原時左右界是 \(c_x\pm w/2\)，上下界是 \(c_y\pm h/2\)，再乘圖尺寸。寬高為正不保證框落在圖內，因為中心靠邊時框仍可越界；裁切框應是明確的前處理或推論規則。

## 分類loss與定位loss分開看

```python
optimizer.zero_grad(set_to_none=True)
logits, boxes = model(images)
class_loss = F.cross_entropy(logits, labels)
box_loss = F.mse_loss(boxes, normalized_cxcywh)
loss = class_loss + 5 * box_loss
loss.backward()
optimizer.step()
```

F是 `torch.nn.functional` 的簡寫；完整案例直接使用全名。MSE是每個座標誤差平方，再對batch及四個座標取平均。人工預測xyxy `[6,8,18,20]` pixels，先轉cxcywh `[12,14,12,12]` pixels，再除32成 `[0.375,0.4375,0.375,0.375]`；真值正規化是 `[0.3125,0.375,0.375,0.375]`。兩項中心誤差各0.0625，寬高誤差0，所以box MSE為 \(2(0.0625)^2/4=0.001953125\)。不能將xyxy直接與cxcywh目標相減。

分類loss和框loss量尺不同，直接相加可能讓某項作用太弱。本節權重5只是固定示範設定，沒有證據說它最合適。程式同時列出classification、box和total，別把total降低當作兩項都改善。

## IoU回答重疊程度

![人工真值與偏移框的IoU手算](../assets/diagrams/04-localization.svg)

GT是真值標註，pred是預測，本圖的pred專門人工設定。Intersection over Union（IoU）是交集面積除聯集面積，0表示無重疊，1表示框完全一致。上圖兩框各144px²，交集100、聯集188，IoU約0.532。它評估整個框的幾何重疊，與四個座標的MSE是不同的數字；相同pixels偏移對小框通常更嚴重。

## 核對、收益與代價

程式應輸出人工兩特徵圖平均0.0625、首張target `[0.3125,0.375,0.375,0.375]`、class shape `[2,2]`、box shape `[2,4]`。它確認框head梯度非零與參數更新，並輸出 `artifacts/04-localization.png` 的GT／pred疊圖；pred來自只更新3步的模型，品質無保證。網頁上圖則是**人工固定框**，專門提供可手算的IoU。

本次三步總loss約0.8099→0.7576，分類0.7173→0.7123，框0.0185→0.0091。這只是兩張訓練圖片的結果，沒有獨立定位評估。收益是把分類與位置接在一個forward；代價是本例flatten框head有4100參數，若同樣4channel直接平均再linear只需20，但失去直接位置對應。輸入尺寸變動也會使flatten head不匹配，不能只改圖片尺寸不改模型。

## 常見錯誤、自主練習與答案

常見錯誤：xyxy與cxcywh混用、pixels與正規化混用、把框輸出順序當成y先x後，或將標籤框重畫得好看便當作模型預測。手算練習將人工預測改成真值向右移6px、y不變，也就是xyxy `[10,6,22,18]`；不用改訓練模型的輸出。答案交集 \(6\times12=72\)，聯集216，IoU=1/3；正規化cx誤差6/32，MSE為 \((6/32)^2/4=0.0087890625\)。同時核對圖與數字，別只看loss。

## 延長到40步：這次真正學到了什麼

這是與上方三步管線檢查分開的補充實驗，沿用同一個模型與資料。seed7、CPU、40次更新；第1／4章改用Adam lr=.01，第3章仍用SGD lr=.1，所以不能把第1／4章的差異單獨歸因於步數。

|模型|首步→最後更新前loss|訓練分類accuracy|獨立validation分類accuracy|
|---|---|---|
|localizer|0.809883 → 0.430213|1.00|未評估|

![本次固定資料40步的實際loss](../assets/diagrams/04-localization-learning.svg)

所有更新的梯度有限且L2長度非零，權重確實改變。曲線只評固定訓練批次；不能據此宣稱真實圖片或深層架構的泛化效果。

兩張訓練圖的框IoU為`[0.6945, 0.7752]`。這次直接用模型框對GT計算，與上面的人工IoU手算圖是不同證據。

可在本節Colab完成環境格後另開code cell：`!python scripts/run_learning_extensions.py --section 04-localization`。原始完整紀錄：[40步結果](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-localization-learning.json)。

曲線橫軸是訓練步序，loss 在該次更新前量測：第1點尚未更新，第40點是第40次更新前的值；更新後的預測另行評估。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-localization.json)

??? example "展開本次實際輸出"

    ```text
    two different positions -> same global mean=0.0625
    step=0, classification=0.7173, box=0.0185, total=0.8099
    step=1, classification=0.7146, box=0.0127, total=0.7779
    step=2, classification=0.7123, box=0.0091, total=0.7576
    first_target_cxcywh=[0.3125, 0.375, 0.375, 0.375], class_shape=(2, 2), box_shape=(2, 4)
    predicted pixel xyxy=[[7.4885783195495605, 7.199568748474121, 20.595396041870117, 19.995590209960938], [9.941701889038086, 8.684192657470703, 23.892370223999023, 22.775699615478516]]
    overlay=artifacts/04-localization.png; no held-out detection claim
    ```

<!-- curriculum-evidence:end -->
