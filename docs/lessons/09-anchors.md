# C.9.1 YOLOv2 機制：anchor 是尺寸起點

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/09-anchors.ipynb){ .md-button }

[第 7 章的 targets](07-targets.md) 已經能把紅框 `[8,12,24,28]` 交給中心所在的格子，讓這格學中心位置、寬高與類別。現在保留這個任務，改問寬高的起點：若訓練資料反覆出現幾種尺寸，能否先給模型這些參考尺寸，再讓它學每個物件相對於參考值的差異？

這些固定的參考寬高叫 **anchor（錨框）**，也叫尺寸先驗（prior）。它不是已找到的物件，不帶類別，也不靠梯度更新。第 7 章直接學寬、高各佔全圖多少；這裡改學「相對於 anchor，要放大或縮小幾倍」。回歸（regression）在這裡就是預測連續數值。本節還把每格的一個槽（slot，可各自輸出一個框的位置）增加成兩個，每槽使用自己的 anchor。

## 同一個紅框，換成 anchor 寫法是哪幾個數字

仍用 64×64 pixel 圖、4×4 格，每格寬 16 pixel。兩個 anchor 是 `[16,16]` 與 `[8,8]`（寬、高，pixel）。紅框中心是 `(16,20)`、寬高是 `(16,16)`；中心仍由格 `(gx=1,gy=1)` 負責。把中心除以 16，再扣掉整數格座標，得到格內比例 `(ox,oy)=(0,0.25)`。

先看寬高如何改寫。假如 anchor 寬是 16，紅框寬也是 16，所需倍率為 1；若 anchor 寬是 8，所需倍率為 2。模型不用直接輸出這個倍率，而是輸出它的自然對數：

\[
t_w=\ln(w/a_w),\qquad t_h=\ln(h/a_h).
\]

`w,h` 是物件寬高，`a_w,a_h` 是這個槽的 anchor 寬高；兩者同為 pixel，所以比值沒有單位。ln 是以 e≈2.718 為底的自然對數，程式裡的 `log` 也是 ln。解碼時用它的反函數 exp 還原：

\[
w=a_w e^{t_w},\qquad h=a_h e^{t_h}.
\]

這樣模型可輸出任何實數，解出的寬高仍為正。`tw=0` 時倍率是 1、框寬等於 anchor；`tw=ln2≈0.693` 時寬放大 2 倍；`tw=−ln2` 時縮成一半。anchor 16 始終是 16，改變的是預測框的寬。把 sigmoid 再套在 tw 上會破壞這個用途：exp(sigmoid(tw)) 只在 1 到 e 之間，連縮小成一半也表示不了。

中心的解碼沿用第 7 章：

\[
c_x=(g_x+\sigma(t_x))\times16,\qquad c_y=(g_y+\sigma(t_y))\times16.
\]

σ 是 sigmoid；gx、gy 是從 0 起算的欄、列編號。16 是特徵位置在輸入圖上的間距，也就是本頁的 stride，與第 1 章的感受野「間距」相同；它不是卷積視窗每次移動幾格的那個 stride。

**進 loss 的中心答案是比例，不是 raw tx、ty。** 本例比較 sigmoid(tx,ty) 和 `(0,0.25)`，寬高則直接比較 raw tw、th 和 ln 比值。例如誤把 ox=0 填成 tx=0，sigmoid(0)=0.5，中心就變成 `(1+0.5)×16=24`，不是 16。

## 先選尺寸起點，再決定誰負責

同一格的兩個槽，哪個較適合學這個 16×16 紅框？anchor 沒有位置，選尺寸時把兩框中心疊在一起，只算**尺寸 IoU**。交集寬、高取各自較小者，交集面積除以聯集面積：

\[
I=\min(w_1,w_2)\min(h_1,h_2),\qquad \mathrm{size\ IoU}=I/(w_1h_1+w_2h_2-I).
\]

16×16 對 16×16 是 1；對 8×8 是 `64/(256+64−64)=0.25`。最大者叫 best anchor，所以本例選第 0 個、16×16。這只是用 GT 在訓練時挑負責槽；推論沒有 GT 可用，兩個槽都會輸出候選。

![格 (1,1) 的兩個 anchor 疊在紅框中心：16×16 的尺寸 IoU 1，8×8 的尺寸 IoU 0.25](../assets/diagrams/09-anchors.svg)

紫色虛線框表示兩個尺寸起點，兩者都疊在紅框中心 `(16,20)`。較大的和紅框重合，選它後 `tw=ln(16/16)=0`、`th=0`。同一個物件若用 8×8 起點，兩項都要 ln2≈0.693147。這讓尺寸先驗的用途可見：接近物件的 anchor 已提供大部分尺寸，模型只學剩下的修正；這不代表實際訓練一定更準。

因此負責槽的四項訓練答案是 **`[0,0.25,0,0]`**：前兩項是格內中心比例，後兩項是寬高的 ln 修正量。

| 答案 | 程式中的來源 | 拿來比較的預測 |
| --- | --- | --- |
| `(ox,oy)=(0,0.25)` | `offsets` | `raw[pos][:,:2].sigmoid()` |
| `(tw,th)=(0,0)` | `torch.log(wh/anchors[best])` | `raw[pos][:,2:4]` |

??? note "反解中心 raw 值，只用來核對 decode"

    完整程式另建 `encoded`，驗證 encode→decode 能回到原框。寬高仍是上表的 ln 比例；中心則用 sigmoid 的反函數 `logit(p)=ln(p/(1−p))`，把比例換回 raw 值。**`encoded` 的前兩項不是中心 loss 的 target；訓練仍比較 `offsets`。**

    ox=0 時 logit 趨向負無限大，不能得到有限值，因此程式先用 clamp 把比例夾在 `[0.0001,0.9999]`。得到 tx=ln(0.0001/0.9999)≈−9.21024、ty=ln(0.25/0.75)≈−1.09861。

    再 decode：cx=(1+0.0001)×16=16.0016，還原框為 `[8.0016,12,24.0016,28]`，誤差約 0.0016 pixel。這個 clamp 近似只用在反解核對；訓練的中心比例答案仍是精確的 `(0,0.25)`。

??? note "如果 anchor 改用「格」當單位"

    本節的 anchor 和解碼出的寬高都以 pixel 為單位，所以「寬 ÷ anchor 寬」這個比值沒有單位。若改用 feature-map 單位，也就是以「格」為單位，1 格 = stride = 16 pixel：16 pixel 的 anchor 寫成 1，8 pixel 寫成 0.5，decode 時再乘 stride 換回 pixel。這時 anchor 必須一併除以 stride=16；不能用 pixel 的 anchor 再多乘一次 stride。例如 decode 寫成 w = anchor × exp(tw) × 16，anchor 卻誤填 pixel 的 16，tw=0 時寬會變成 16×1×16 = 256 pixel，比整張 64 pixel 的圖還大。

## 一格兩槽，不等於兩種類別

每個槽都輸出 `[tx,ty,tw,th,obj,class0,class1]` 七個數；最後兩項是自己的類別 logits，所以兩個槽都可預測紅類或藍類。輸出 shape 是 `[B,4,4,A,5+C]`，軸依序是圖片、格列 y、格欄 x、槽、輸出數。本例 A=2、C=2，shape 為 `[B,4,4,2,7]`。增加類別只增加最後一軸；C=3 時變成 8 個數，候選仍是 32 個。

增加槽後，沒被選為 best 的槽是否全要學背景？圖裡 8×8 槽雖不是最佳，仍在同一責任格，尺寸也有些接近。本節選擇暫不壓低它的 objectness，稱為 **ignore**。這個教學規則是：同格、非 best，而且尺寸 IoU **大於 0.2**。其他格即使有物件畫素，仍是 negative；negative 表示沒分到負責物件，不表示感受野裡完全沒有物件。

| 狀態 | 哪些槽 | 數量 | objectness | 框與類別 |
| --- | --- | --- | --- | --- |
| positive | 格 (1,1) 的 best 槽 0 | 1 | 目標 1 | 學紅框與 class 0 |
| ignore | 同格的槽 1，尺寸 IoU 0.25>0.2 | 1 | 不算 loss | 不算 loss |
| negative | 其他 15 格各兩槽 | 30 | 目標 0 | 不算 loss |

objectness BCE 只平均非 ignore 的 **31 個槽**。raw 全 0 時 sigmoid(obj)=0.5；每個 obj logit 的梯度為 `(0.5−target)/31`，所以 positive 約 −0.0161、negative 約 +0.0161、ignore 是 0。第 7 章單圖是 16 格，分母不同，不能沿用那裡的 ±0.03125。

??? note "原版 YOLOv2 的 ignore 規則"

    [YOLOv2 論文](https://arxiv.org/abs/1612.08242)沒有說明 ignore。官方 Darknet 的 [region_layer.c](https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/src/region_layer.c#L236-L306) 比的是每個槽解碼後的預測框（含位置）與 GT 的一般 IoU；最大值超過 0.6（[yolov2-voc.cfg](https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/cfg/yolov2-voc.cfg#L257) 的 thresh）便不計 objectness loss，範圍不限同格，也不是 anchor 尺寸。負責某個 GT 的槽是例外，仍算 objectness。本節的規則不代表原版其他訓練細節。

??? note "手算：零 logit 時的 objectness 梯度"

    本節程式的 raw 初值全為 0，所以每個 obj logit 都是 0，sigmoid(0)=0.5。BCE 對 obj logit 的梯度是 (sigmoid(logit) − 目標) ÷ 參與平均的槽數，本例分母是 31：

    - positive 槽：(0.5 − 1)/31 ≈ −0.0161
    - negative 槽：(0.5 − 0)/31 ≈ +0.0161
    - ignore 槽：0，因為它不在平均裡

    第 7 章單張圖的分母是 16 格，所以是 ±0.03125。

    上面三種槽的 obj 梯度都是手算的，完整程式不會印出；它的斷言（assert）只核對其中 ignore 槽的 0。想親眼看到 ±0.0161，可以在完整程式 `loss.backward()` 的下一行加上這一行，縮排和 `loss.backward()` 對齊：

    ```python
    print(raw.grad[0,1,1,:,4].tolist(), raw.grad[0,0,0,:,4].tolist())
    ```

    `raw.grad` 是 backward 存在 raw 上的梯度，形狀和 raw 一樣是 [1,4,4,2,7]。`[0,1,1,:,4]` 依序取第 0 張圖、格列 y=1、格欄 x=1、兩個槽全取（`:`），最後的 4 是 7 個數裡的編號（從 0 起編：0 到 3 是 tx、ty、tw、th，4 是 obj）。所以它是格 (1,1) 兩個槽的 obj 梯度，約為 [−0.0161, 0]：槽 0 是 positive，槽 1 是 ignore。`[0,0,0,:,4]` 是格 (0,0) 兩個 negative 槽的 obj 梯度，約為 [0.0161, 0.0161]。加上的這一行會最先印出，排在完整程式本身的四行輸出之前。

下面摘錄完整 loss。`pos`、`ignore` 是 `[1,4,4,2]` 的布林 mask；`wh=[16,16]`，`offsets=[0,0.25]`，`[None]` 加一軸成 `[1,2]`，對齊唯一正槽。

``` { .python data-excerpt="lesson_cases/09-anchors.py" }
ious = size_iou(wh[None],anchors)[0]   # [1,.25]
best = int(ious.argmax())             # 0：16×16 anchor
...                                  # 省略：建立 raw、pos、ignore、obj_target
valid = ~ignore                      # 只取非 ignore 的 31 個槽
...
regression = F.mse_loss(raw[pos][:,:2].sigmoid(), offsets[None])
regression = regression + F.mse_loss(raw[pos][:,2:4],torch.log(wh/anchors[best])[None])
objectness = F.binary_cross_entropy_with_logits(raw[...,4][valid],obj_target[valid])
classification = F.cross_entropy(raw[pos][:,5:],torch.tensor([0]))
loss = regression+objectness+classification
```

中心、寬高兩項 MSE 加成 `regression`；`classification` 只讀正槽的類別 logits，答案為 class 0。這裡四項直接相加，沒有沿用第 7 章框 loss 乘 5 的權重，所以也不能把兩節 loss 當成效果對照。

## 這次單步更新能核對什麼

執行 `PYTHONPATH=. python lesson_cases/09-anchors.py`，或用頁首 Colab。程式沒有圖片與 CNN，直接把全零的 `raw [1,4,4,2,7]` 設為 `nn.Parameter`，由 SGD 更新這 224 個數字；兩個 anchor 固定不變。

一次更新會改 positive 的中心、obj、類別，以及 negative 的 obj。positive 的 tw、th 已等於 target 0，這一步不變；ignore 全部不變。斷言核對尺寸 IoU、反解中心值與 sigmoid 比例、encode→decode 誤差小於 0.02 pixel、ignore 的七項梯度為 0、非 positive 的框梯度為 0，以及更新前後 raw 不同。這些檢查回答的是「答案表示與責任遮罩接對了嗎」。

輸出四行依序是尺寸 IoU 與 best/log wh、格內比例與 encoded logits、`1 1 30` 三種槽數、decode 框約 `[8.0016,12,24.0016,28]`。最後的 `one slot update completed` 是固定字樣，實際更新項目如上，不是量出只更新一槽。沒有看圖的模型，這次也沒有 AP 可報。

## 多一個尺寸起點，仍有哪些限制

anchor 也可能讓同格兩物件分到不同槽。例如一個 16×16、一個 8×8，各選自己的 best；第 7 章一格一槽則會衝突、拋出 ValueError。但兩個都 16×16 時仍選槽 0，衝突沒有消失。本節程式只處理一物件，這段是責任容量的概念例子。

代價是要決定 anchor 尺寸、數量、assignment 與 ignore。候選多了，解碼、score 篩選與 NMS 的工作也增加。推論時全部 32 槽先用各自 anchor 解碼，再走第 7 章的分數門檻與 NMS；本節程式尚未做推論。

exp 還會快速放大輸出：tw=5 時 e⁵≈148，16 pixel 起點會解成約 2375 pixel。要記錄 tw、th 範圍，檢查 loss 和框是否有限。若要知道 anchor 對偵測有沒有幫助，需固定資料、loss 權重和訓練設定做新舊對照；本例只有機制證據。

自主練習（手算即可，不必改程式；先自己算，再展開答案）：

1. 一個寬 32、高 16 的物件，配 16×16 anchor 時，target 的 tw、th 是多少？若改配 8×8 anchor 呢？
2. 假設它是圖中唯一的物件。它對兩個 anchor 的尺寸 IoU 各是多少？哪個槽是 positive？有沒有 ignore？positive／ignore／negative 各幾個？

??? note "參考答案"

    **第 1 題**：配 16×16 時，tw = ln(32/16) = ln2、th = ln(16/16) = 0，也就是 `[ln2,0]`。若換 8×8 anchor，則 tw = ln(32/8) = ln4、th = ln(16/8) = ln2，也就是 `[ln4,ln2]`。

    **第 2 題**：對 16×16，交集 = min(32,16) × min(16,16) = 256，尺寸 IoU = 256/(512+256−256) = 0.5。對 8×8，交集 = 8×8 = 64，尺寸 IoU = 64/(512+64−64) = 0.125。所以 best=0，槽 0（16×16）是 positive。槽 1 的 0.125 沒有大於 0.2，不設 ignore，是 negative。計數是 1／0／31：同格的槽 1 加上其他 15 格 × 每格 2 槽，共 31 個 negative。

[YOLO9000 論文](https://arxiv.org/abs/1612.08242)中的 YOLOv2 使用 anchor 尺寸修正與格內中心限制，並用尺寸聚類找先驗。本頁只保留這些表示與責任關係，沒有 Darknet-19 主幹或完整 YOLOv2 訓練。

手填尺寸只示範了起點的角色。接下來用訓練框的尺寸選起點，才知道這份資料常出現什麼寬高。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/09-anchors.json)

??? example "展開本次實際輸出"

    ```text
    size IoU [1.0, 0.25] best anchor 0 log wh [0.0, 0.0]
    ratio offsets [0.0, 0.25] encoded logits [-9.210240364074707, -1.0986123085021973]
    positive/ignore/negative 1 1 30
    decoded [8.00160026550293, 12.0, 24.00160026550293, 28.0] one slot update completed
    ```

<!-- curriculum-evidence:end -->
