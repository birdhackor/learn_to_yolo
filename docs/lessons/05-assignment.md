# B.5 多物件輸出與責任分配：哪個預測負責哪個物件

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/05-assignment.ipynb){ .md-button }

單物件定位器每張圖交出一個類別與一個框。現在同一張圖有紅、藍兩個物件，另一張卻只有背景：答案的數量變了，模型不能仍然每張交一個框。即使多輸出幾個框，訓練時也會立刻遇到問題：**第一個輸出該學紅框，還是藍框？沒有分到物件的輸出又該學什麼？**

我們先用一個固定安排解決這件事：把 64×64 輸入切成 4×4 格，每格提供一個候選預測位置，能輸出一個框，稱為 **slot（槽）**。一張圖始終有 16 個 slot，標註則可以是 0 個、2 個或更多。訓練前，將每個真值分給一個 slot，這項工作叫 **assignment（責任分配）**。下面沿著兩個物件，將標註變成能和每格輸出比較的 target。

## 先決定，哪一格負責哪個物件

本節採用「物件中心在哪一格，就由哪一格負責」。列號 row 往下、欄號 col 往右，均從 0 起算；每格 64/4=16 px。紅框 `[4,4,20,20]` 的中心是 (12,12)，藍框 `[36,36,52,52]` 的中心是 (44,44)。

![紅藍兩框以中心選出責任格；框本身仍可跨格](../assets/diagrams/05-assignment.svg)

黑點是物件中心，淡紅、淡藍格是各自負責的格：紅由 row0／col0，藍由 row2／col2 負責。兩框都跨過四格，但**只看中心**，不要求整個框塞在責任格內。這讓每個物件得到唯一的一份 target；中心正好在格線上時，取整規則會交給右邊或下邊的格。

除了決定格號，還要告訴那格框的中心和大小。格號已記住大致位置，因此中心只需表示「從這格左上角往右、往下走多少」；寬高則仍相對整張圖。四步如下：

1. 中心除以 16：紅得到 (0.75,0.75)，藍得到 (2.75,2.75)。
2. 捨去小數（floor）：x 的整數部分是 col，y 的整數部分是 row。
3. 減掉整數格號，得到格內偏移。兩框都是 (0.75,0.75)，以格寬 16 為 1。
4. 寬高都是 16 px，除以**整圖 64**，得到 (0.25,0.25)。

因此兩格的 box target 都是 `[0.75,0.75,0.25,0.25]`，順序仍是 x、y、w、h。相同 target 不表示同一個位置：存放的格不同。中心還原時，紅的 x=(0＋0.75)×16=12，藍的 x=(2＋0.75)×16=44；y 同理。格號與偏移合在一起，才是完整中心。

為什麼中心用格寬，寬高卻用整圖？本節框的四個輸出都經 sigmoid，只能介於 0～1。中心既在責任格內，格內偏移就在這個範圍；框卻可以比格子大，例如寬 40 px，除以 16 得 2.5，輸出不了。除以整圖 64 得 0.625，就能表示它。

將格數寫成 S，格寬 \(g=64/S\)，一般式便是：

\[
\text{col}=\left\lfloor c_x/g\right\rfloor,\qquad
\text{row}=\left\lfloor c_y/g\right\rfloor,
\]

\[
\text{target}=\left[c_x/g-\text{col},\ c_y/g-\text{row},\ w/64,\ h/64\right].
\]

這裡 cx、cy、w、h 都是 px。寫進 tensor 時用 `[圖片,row,col]`，也就是先 y 後 x；裡面的框向量仍先 x 後 y。`build` 的核心和上述四步一致：

``` { .python data-excerpt="lesson_cases/05-assignment.py" }
# 步驟 1：中心 ÷ 64 × 4，等於除以格寬 16
grid_center = center / image_size * grid
# 步驟 2：floor 捨去小數，再轉成兩個 Python 整數；
#        第一個數來自 x，所以是 col；第二個來自 y，所以是 row
col, row = grid_center.floor().long().tolist()
...
# 步驟 3、4：把 [x 偏移, y 偏移, w/64, h/64] 存進第 b 張圖的 [row, col] 那一格
boxes[b, row, col] = torch.cat((grid_center - grid_center.floor(), size / image_size))
```

`grid` 是 S、`image_size` 是 64、`center` 是 (x,y)、`size` 是 (w,h)。`...` 略去寫入前的同格碰撞檢查，後面會看它為何不能省。處理每個框前，`build` 也確認 x2>x1、y2>y1、四個邊界在 0～64、類別必須是 0 或 1（本例兩類），避免把壞標註分進格子。

## 分到物件的格，和其餘格，要學不同答案

這張圖有 2 格分到物件，稱為**正格（正樣本）**；其餘 14 格是**負格（負樣本）**。這裡一個樣本是一格，不是一張圖。第二張空圖仍有 16 格，全部是負格，所以 B=2 的這批資料共有正 2、負 30。

如果每格只輸出框與紅／藍類別，空圖也會被迫選一種類別。於是另加 **objectness**，專門回答「這格是否分到物件」：正格 target=1，負格 target=0。它和物件類別分開，背景不是紅／藍 softmax 的第三類。

固定容器現在有四部分：

| 欄位 | Shape（本例 B=2、S=4） | 正格 | 負格 |
| --- | --- | --- | --- |
| box target | `[2,4,4,4]` | 框的四個比例 | 填 0，只占位 |
| objectness | `[2,4,4]` | 1 | 0 |
| class_ids | `[2,4,4]`，整數 | 紅 0／藍 1 | −1，只占位 |
| positive | `[2,4,4]`，布林 | True | False |

`build([two,empty])` 的 list 每張圖一個 dict，含 `[N,4]` 的 xyxy px `boxes` 與 `[N]` 的 `labels`；N 可不同，也可為 0。它依序回傳上表四個 tensor，主程式依序叫它們 `box_targets`、`objectness`、`class_ids`、`positive`。物件數改變的是哪些格被填入，容器 shape 始終相同。

`positive` 是選取哪些位置的 True／False 表，稱為 **mask（遮罩）**。框和類別只在正格有答案，所以只取 `t[positive]` 進 loss。負格的框 0 不是「背景應預測四個 0」，類別 −1 也不是新類別；不遮掉 −1，`cross_entropy` 會報類別越界。

另外有些設計使用 **ignore**，意思是某候選暫不計入某項 loss。負格則確實計 objectness loss，要把輸出推向 0；兩者不能混用。**本節沒有 ignore**，每格都有 objectness 監督。

| 狀態 | Objectness loss | 框與類別 loss |
| --- | --- | --- |
| 正格 | target=1 | 都計入 |
| 負格 | target=0 | 都不計 |
| ignore（本節沒有） | 略過，不推向 0 或 1 | 依各設計規定 |

??? note "選讀：第 9 章的 anchor 為什麼會有 ignore"

    例如第 9 章會教 anchor（預設的框尺寸）：每格有多個 slot，各配一種 anchor。尺寸和物件最接近的 slot 當正樣本。尺寸接不接近用尺寸 IoU 衡量，它只比大小、不看位置。同一格的其他 slot 若和物件的尺寸 IoU 大於 0.2（第 9 章的規則），表示尺寸也還算接近，硬罰成背景不合理，就設成 ignore。ignore 不算 objectness loss，也沒有框、類別 loss。本節每格只有一個 slot，沒有這種狀態。


## 模型怎麼一次交出 16 格的答案

現在 target 已就位，head 要提供相同位置的輸出。本例每格交出 **7 個原始數字**：tx、ty、tw、th、objectness logit、class0 logit、class1 logit。logit 是未轉成比例的實數；前四項經 sigmoid 成為剛才的框表示，objectness 經 sigmoid 成為這格的有物件分數，兩類 logits 經 softmax 給紅／藍比例。

所以 head 輸出為 `[B,S,S,5+C]`，C 是類別數，本例 C=2，輸出為 `[2,4,4,7]`。`[..., :4]` 取框、`[...,4]` 取 objectness、`[...,5:]` 取類別，`...` 保留前面的軸。這是本節 MiniYOLO 的約定，不是所有 YOLO 版本的共用格式。

head 是 `nn.Conv2d(4,7,1)`：A.3.2 的 1×1 卷積在每格用同一組權重，把 4 個特徵變成 7 個數。這次 `features` 為 `torch.randn` 產生的 `[2,4,4,4]`，代替 backbone 的輸出，以便單獨核對 head。卷積先給 `[2,7,4,4]`，再用 `permute(0,2,3,1)` 把類別與框欄位移到最後一軸，得到 `prediction`。

三項 loss 接法如下：

``` { .python data-excerpt="lesson_cases/05-assignment.py" }
# prediction: [2,4,4,7]；positive: [2,4,4]，其中 2 格是 True
# prediction[..., :4] → [2,4,4,4]；再用 [positive] 取出正格 → [2,4]（2 個正格 × 4 個框值）
box_loss = nn.functional.mse_loss(prediction[..., :4].sigmoid()[positive], box_targets[positive])
# prediction[..., 4] 和 objectness 都是 [2,4,4]：32 格全算
object_loss = nn.functional.binary_cross_entropy_with_logits(prediction[..., 4], objectness)
# prediction[..., 5:][positive] → [2,2]（2 個正格 × 2 個類別 logit）；class_ids[positive] → [2]，值是 [0,1]
class_loss = nn.functional.cross_entropy(prediction[..., 5:][positive], class_ids[positive])
loss = box_loss + object_loss + class_loss
```

框的 MSE 要自己先 sigmoid，才能和 target 比較。有沒有物件是每格各自的二元問題，用 **BCE（Binary Cross Entropy，二元交叉熵）**；`binary_cross_entropy_with_logits` 內部已包含 sigmoid，因此傳原始 objectness logit。類別沿用 `cross_entropy`，直接收原始類別 logits，內部以穩定的 log-softmax 等計算評分。不要在這兩項 loss 之前再轉一次機率。

本節把三項等權相加，先看監督是否接對，不用這兩步示範比較最佳權重。取正格後，框為 `[2,4]`、類別 logits 為 `[2,2]`、類別 target 為 `[2]`，值是 `[0,1]`；objectness 則 32 格全算。

mask 的效果可由梯度看見：負格的框與類別不影響 loss，這些**輸出位置**的梯度是 0；objectness 仍會收到學背景的梯度。head 權重由各格共用，正格與負格的監督都會傳回同一組權重，因此權重仍更新，負格的框值也可能跟著變。某些輸出梯度為 0，不代表整個卷積不學。

這段簡化 loss 要求整批至少一個正格。若整批皆空圖，框和類別各取出 0 個數，再取平均會得到 NaN，總 loss 也壞掉。空圖本身合法；實作必須另處理「這項沒有答案可平均」，例如[完整 Grid loss](07-loss.md)在這時回傳仍連著計算圖的 0，objectness 仍正常計算。

??? note "選讀：如何核對 mask 的梯度與 shape"

    程式呼叫 `prediction.retain_grad()`，才可在 backward 後檢查這個中間輸出的 `.grad`；參數梯度則預設會保留。每步確認負格框、類別梯度為 0，負格 objectness 梯度總量非零。BCE 對每格 logit z 的梯度為 (sigmoid(z)−target)/32，負格 target=0，會將有物件分數往下推。

    `t[positive]` 按圖片順序，再逐列由左至右取正格；`positive[0].nonzero()` 同樣按 row、col 列出第一張的正格。若要觀察 −1 類別的錯誤，先將類別 logits 展平成 `[32,2]`、target 成 `[32]`。只刪掉 loss 行上的 mask，類別仍在最後軸，會先因 `cross_entropy` 期待類別在第二軸而報 shape 錯誤。

## 一格一個框，有什麼放不下

紅框中心 (12,12) 與另一個紅框 `[2,2,10,10]` 的中心 (6,6)，都落在 row0／col0。這格只有一個 slot，**同類別也不能把兩個獨立物件寫成一個框**。`build` 若發現這格已是正格，就 `raise ValueError`，而不是讓後來的 target 默默覆蓋第一個。

這裡的容量是「輸出最多能表示幾個物件」：全圖最多 16 個，還要求它們的中心分布在不同格。一張圖只有兩個，也可能因同格碰撞放不下。slot 並非類別專用，每個 slot 都有完整的紅／藍分數。

把格切細、每格增加 slot，或在多種解析度提供候選，能讓不同物件有更多可用位置；代價是候選更多、分配和 loss 更複雜。讓多個候選學同一物件則提供更多監督，但也更容易輸出重複框。本節先採一物件一格，保留這個限制；格子與中心規則參考 [YOLOv1 原論文](https://arxiv.org/abs/1506.02640)，不是完整重現。

訓練前的 assignment 使用標註來選責任格；推論時的 **NMS（Non-Maximum Suppression，非極大值抑制）**則刪除重複預測，下一節會做。刪候選不能補回被覆蓋的訓練物件，所以 NMS 無法解決本節的同格碰撞。

## 用不對稱的框，才抓得到寫反

在 Colab 或本機跑 `PYTHONPATH=. python lesson_cases/05-assignment.py`，CPU 程式先生成 target，再核對碰撞與 mask，最後讓人工特徵上的 head 更新兩步。**沒有用圖片訓練偵測器**，這不是偵測效果實驗。

主例中心 x=y、寬=高，寫反 row／col、偏移 x／y 或寬高，答案碰巧都不變。因此完整程式另外檢查兩個不對稱框：

| xyxy（px）／類別 | 中心、寬高 | 責任格 (row,col) | Box target |
| --- | --- | --- | --- |
| `[2,4,22,16]`／0 | (12,10)、(20,12) | (0,0) | `[0.75,0.625,0.3125,0.1875]` |
| `[36,12,52,28]`／1 | (44,20)、(16,16) | (1,2) | `[0.75,0.25,0.25,0.25]` |

第一框分開 x／y 偏移與寬高，第二框則分開 row／col。這些變化核對同一個轉換規則，不是增加訓練圖片。

??? note "完整程式的不對稱框核對：兩個框各抓哪一種寫反"

    完整程式把兩個框寫成同一張圖的標註 `odd`，呼叫 `build([odd])`（沒寫 `grid`，所以 S=4）：

    - 第一框 [2,4,22,16]，類別 0：中心 (12,10)、寬高 (20,12)。12/16＝0.75、10/16＝0.625，捨去小數後由 row0／col0 負責；寬高除以 64 是 0.3125、0.1875，所以 target 是 [0.75,0.625,0.3125,0.1875]。x、y 偏移寫反會得到 [0.625,0.75,…]；寬、高寫反，後兩個數會對調。但它在 row0／col0，row、col 寫反時格子不變。
    - 第二框 [36,12,52,28]，類別 1：中心 (44,20)、寬高 (16,16)。44/16＝2.75、20/16＝1.25，捨去小數後由 row1／col2 負責；格內偏移 (0.75,0.25)，寬高都是 16/64＝0.25，所以 target 是 [0.75,0.25,0.25,0.25]。算格子時把 row、col 寫反，它會落到 row2／col1；x、y 偏移寫反，target 會變成 [0.25,0.75,0.25,0.25]。它寬＝高，寬、高寫反只能靠第一框抓。

    程式用遮罩 `odd_positive` 取出這兩個正格的 target 和類別，順序和正格清單 `odd_cells` 相同（由上而下逐列，所以 [0,0] 在前），再斷言 `odd_cells == [[0, 0], [1, 2]]`、兩組 target 依序如上、類別依序是 0、1。上面三種寫反，各至少會讓其中一個斷言失敗。

    這些數都是分母為 2 的冪次的分數（例如 0.625＝5/8、0.1875＝3/16），float32 能精確存下，計算過程也沒有捨入誤差，所以斷言直接用 `==` 比較。大多數小數不是這樣，例如 0.8 在 float32 裡存不精確，所以第 4 章〈[座標轉換與還原](04-coordinates.md)〉核對往返時用的是容許極小誤差的 `torch.allclose`。執行時印出的 `asymmetric boxes: …` 那一行，就是這兩個正格的位置、target 與類別。第二框刻意不用下面練習的藍框，所以這一行不會先透露練習的答案。


輸出中的主例正格應是 `[[0,0],[2,2]]`，第一格 target 是 `[0.75,0.75,0.25,0.25]`；第一張負 14、整批正 2 負 30 ignore 0。同格碰撞應報 image=0、row=0、col=0。兩步更新前的 box loss 約 0.0871→0.0861、obj 0.7031→0.6839、class 0.5235→0.4744，只描述這組人工特徵的更新。

??? note "選讀：輸出欄位的計數方式"

    `box_shape` 是 `[2,4,4,4]`，其餘兩項 target shape 為 `[2,4,4]`。`first_target` 讀第一個正格，`first_image_negative` 只數第一張的負格，`negatives` 數整批。此例以 objectness=0 判負格，`ignores` 數非正、也非負的格，結果為 0；後面的 anchor 設計會另用 mask 區分負格與 ignore，不能只看 target 數值。程式最後一行表示本次斷言都通過，包括梯度與權重更新檢查。

    常見錯誤也對應這些檢查：寬高誤除 16，主例 target 變 `[0.75,0.75,1,1]`；負格也進框／類別 loss，梯度檢查失敗；不查碰撞而覆蓋 target，測試收不到預期的 ValueError，會報 AssertionError。

## 停一下：位置換了，哪些 target 跟著變

自主練習：把藍框向左移 16 pixels，變成 [20,36,36,52]，紅框不動。先自己算，再展開參考答案：

1. 藍框的新中心是多少？
2. 它由哪個 row／col 負責？
3. 它的 box target 是什麼？
4. 若把 S 由 4 改成 8，候選由 16 格變成 64 格，每格 8 pixels。紅、藍兩框各由哪一格負責？格內偏移和寬高 target 是多少？第一張圖有幾個負格？

想用程式核對時，不要直接改 `main()` 裡的資料：`main()` 的斷言寫死了原題答案，資料一改就會失敗。請先執行本節的完整程式，讓 `build` 有定義，再另開一個 notebook 程式格（code cell），只呼叫 `build()`。參考答案裡附了可以直接貼上的程式。

??? note "參考答案"

    1. 中心是 ((20+36)/2, (36+52)/2)＝(28,44)。
    2. 28/16＝1.75、44/16＝2.75，捨去小數得 col1、row2，所以由 row2／col1 負責。若算成 row1／col2，就是把 row、col 寫反了。
    3. 偏移是 (0.75,0.75)，寬高仍是 16/64＝0.25，所以 target 仍是 [0.75,0.75,0.25,0.25]。
    4. S=8 時每格 64/8＝8 pixels。紅中心 (12,12)/8＝(1.5,1.5)，由 row1／col1 負責；藍中心 (28,44)/8＝(3.5,5.5)，x 得 col3、y 得 row5，所以由 row5／col3 負責，索引寫成 [5,3]。兩者的格內偏移都是 (0.5,0.5)，寬高 target 仍是 0.25。第一張圖的負格是 64−2＝62。

    下面的程式用斷言核對第 2～4 題的答案：

    ```python
    moved = {'boxes': torch.tensor([[4.,4.,20.,20.],[20.,36.,36.,52.]]),
             'labels': torch.tensor([0,1])}
    boxes, obj, cls, positive = build([moved])
    # nonzero() 列出所有 True 位置的索引，每組是 [row, col]
    assert positive[0].nonzero().tolist() == [[0,0],[2,1]]
    assert boxes[0, 2, 1].tolist() == [0.75, 0.75, 0.25, 0.25]  # 藍框那一格的 target
    dense_boxes, dense_obj, dense_cls, dense_positive = build([moved], grid=8)  # grid 就是 S
    assert dense_positive[0].nonzero().tolist() == [[1,1],[5,3]]
    assert dense_boxes[0, 1, 1].tolist() == [0.5, 0.5, 0.25, 0.25]  # 紅框
    assert dense_boxes[0, 5, 3].tolist() == [0.5, 0.5, 0.25, 0.25]  # 藍框
    # ~ 用在 bool tensor 上是逐格 not，把 True、False 互換（和 Python 的 ~True 得 -2 不同），
    # 所以 (~dense_positive[0]).sum() 是第一張圖的負格數
    assert (~dense_positive[0]).sum() == 62
    ```

??? note "操作選讀：讓完整 main() 的主例與人工 head 也改成 S=8"

    上面的主要練習只呼叫 `build`，用 target 就能核對責任格、格內偏移與寬高。如果還想改完整實驗，請另外複製一份程式；這裡保留主例的紅、藍兩框（藍框不左移），並一起改 `main()` 中以下幾處：

    |位置|改成的值|
    |---|---|
    |主例生成 target|`build([two, empty], grid=8)`|
    |人工特徵圖 `features`|`torch.randn(2, 4, 8, 8)`|
    |`prediction.shape` 斷言|`(2, 8, 8, 7)`|
    |整批負格數的斷言|`negative.sum() == 126`；正格仍是 2、ignore 仍是 0|
    |主例正格的斷言|`positive[0, 1, 1]` 與 `positive[0, 5, 5]`；空圖仍沒有正格|
    |紅框 target 的斷言|查 `box_targets[0, 1, 1]`，答案改成 `[0.5,0.5,0.25,0.25]`|

    只在主例那次呼叫寫 `grid=8`，**不要改 `build` 的預設 `grid=4`**。不對稱框 `odd` 和同格碰撞測試 `collision` 都是照 S=4 設計，保留原本的呼叫和斷言，才能繼續核對它們。

    改完後，主例印出的 `positive_cells` 是 `[[1,1],[5,5]]`、`first_target` 是 `[0.5,0.5,0.25,0.25]`、第一張負格 62、整批負格 126；這些值都從 target 讀出，不必改 print。不對稱框和碰撞那兩行不變。人工 head 的 loss 數字會改變，頁尾的 S=4 紀錄不能拿來當 S=8 答案。

    若誤改 `build` 的預設為 8，不對稱框會落在 `[[1,1],[2,5]]`，原本 `odd_cells == [[0,0],[1,2]]` 的斷言先失敗。即使同步改掉這組斷言，碰撞測試的兩框也已不在同一格，程式會因沒有收到預期的 ValueError 而報 AssertionError；那是格子變細後不再碰撞，不能刪掉測試來掩蓋差異。

密網格的收益是更細的候選位置；代價是更多要學背景的格，計算也更多。


??? note "本節的 MiniYOLO 和 YOLOv1 原版差在哪"

    YOLOv1 把圖切成格子，物件中心落在哪一格，就由那一格負責。原版每格預測多個框（論文的設定是 2 個），每個框各帶一個信心分數（confidence），loss 的設計也和本節不同。

    本節的 MiniYOLO 每格只輸出**一個框**。「這格有沒有物件」用一個單獨的 objectness 分數表示，不是把背景當成 softmax 裡的另一個類別。這一點原版也一樣：原版每個框的 confidence 就是這種獨立的分數，也沒有背景類別。差別在要學的目標：原版負責物件的那個框，confidence 要學預測框與真值框的 IoU（第 4 章），不是固定的 1；本節的 objectness 只學 1（正格）或 0（負格）。兩個物件類別本節用 softmax；原版每格一組類別機率，不經過 softmax：最後一層是線性輸出，類別機率和其他項一樣直接用平方誤差訓練。原版的每格多框、以 IoU 為目標的 confidence 與 loss 設計，本節都省略，所以不是 YOLOv1 的重現。每格只能放一個物件，所以也不保證多個物件都放得下（見前面〈一格一個框，有什麼放不下〉）。

??? note "延伸：偵測器的幾種做法"

    Backbone 抽取特徵，head 給出答案；有些架構在兩者之間加入 neck，整理或融合特徵。本節用人工特徵，只核對 head，沒有展示整套特徵抽取流程。

    - 滑動視窗（sliding window）：固定大小的窗在圖上逐位置移動，每個位置各跑一次分類器。
    - 兩階段（two-stage）：先找出可能有物件的候選區域（proposal），再對每一塊分類並修正框。
    - 單階段（single-stage，YOLO 屬於這類）：一次 forward 直接輸出所有候選框與分數。

    這些是整體流程的差異。不論哪一種做法，訓練時都要決定哪個輸出負責哪個物件，這就是 assignment；YOLO 這類單階段做法也不例外。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/05-assignment.json)

??? example "展開本次實際輸出"

    ```text
    box_shape=(2, 4, 4, 4), objectness_shape=(2, 4, 4), class_shape=(2, 4, 4)
    positive_cells(row,col)=[[0, 0], [2, 2]], first_target=[0.75, 0.75, 0.25, 0.25], first_image_negative=14
    batch positives=2, negatives=30, ignores=0
    asymmetric boxes: positive_cells(row,col)=[[0, 0], [1, 2]], targets=[[0.75, 0.625, 0.3125, 0.1875], [0.75, 0.25, 0.25, 0.25]], classes=[0, 1]
    capacity limit correctly raised: same-cell collision at image=0, row=0, col=0
    step=0, box=0.0871, obj=0.7031, cls=0.5235
    step=1, box=0.0861, obj=0.6839, cls=0.4744
    negative positions supervise objectness only; artificial-feature head update verified
    ```

<!-- curriculum-evidence:end -->
