# 12.3 Sample assignment：哪個候選值得被教

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/12-assignment.ipynb){ .md-button }

上一頁的 head 已從同一特徵分別輸出四距離與類別分數，但為了追梯度，把每個位置都當成正樣本。實際圖片可能只有兩個物件，候選卻很多：我們要挑出合適候選學各 GT，其他學背景，不能讓每個位置都報物件。這一步就是 **sample assignment（樣本分配）**；sample 指訓練的正、負樣本。

第 5、7 章按 GT 中心所在格分責任，規則只看標註，訓練前可算好。現在利用模型**當下**的分類分數與預測框，選既能認出物件、又能對準框的候選；每步輸出變了，責任也可能變，稱動態 assignment。

真值 GT（ground truth）含人工框與類別；候選則是某特徵位置的一組預測。點的位置決定是否能表示 GT，預測框與分類 score 決定當前品質，這兩種幾何角色先分清。

## 兩個真值、四個候選

GT A=`[0,0,20,16]`、class 0；B=`[12,0,32,16]`、class 1。候選 p0～p3 是從 0 起算的固定 ID，參考點為 `(8,8),(16,8),(24,8),(40,8)`。點嚴格在框內纔有資格：`x1<x<x2、y1<y<y2`，等號不算。p0 只在 A、p1 在兩框、p2 只在 B、p3 在兩框外；點(20,8)若在 A 右界，也不算 A 內。

![同一條 x 數線上的 GT A／B、四參考點與各候選預測框](../assets/diagrams/12-assignment.svg)

黑點向上對綠框查資格，向下藍條則是各點對應的預測框，用來算 IoU。p2 的藍框與 A 有重疊，但它的黑點 x=24 不在 A 內；框重疊不能代替點資格。

人工指定預測框依次 `[0,0,10,16]、[4,0,24,16]、[14,0,32,16]、[40,0,50,10]`。這些只用於算框與 GT 的 IoU，不保證都能由某個 head、同一參考點解出。分類 score 也是手填，不是訓練結果。points shape `[P,2]=[4,2]`，GT `[G,4]=[2,4]`，下面的表 `[G,P]=[2,4]`。

每列取**該 GT 類別**的 score。p1 的 class 0 爲 0.7、class 1 為 0.8，A 列用 0.7、B 列用 0.8；各自 sigmoid，不要求相加為 1。若兩 GT 同類，score 兩列相同，IoU 列仍可能不同。

手機上可左右滑動表格，查看完整欄位。

| 真值 | p0 score／IoU | p1 score／IoU | p2 score／IoU | p3 score／IoU |
| --- | --- | --- | --- | --- |
| A | 0.9／0.5 | 0.7／0.6667 | 0.1／0.1875 | 0.8／0 |
| B | 0.1／0 | 0.8／0.4286 | 0.7／0.9 | 0.9／0 |

p1 對 A 交集 256、兩面積共 640，IoU=`256/(640−256)=2/3`；對 B 交集 192，IoU=`192/(640−192)=3/7`。同一欄兩個 IoU 都從同一預測框算，不能為配想要的答案各填一個任意重疊值。

若讓每個點去學最近的 GT，p3 距 A 中心 30、距 B 中心 18，會挑 B，卻仍在 B 框外；要表示 B 就需右距離 32−40=−8，這裡的正距離表示不接受。只看分類高分也不夠：它對 B 的分數 0.9 最高，預測框卻完全沒碰 B。選它就讓高信心與錯框綁在一起。

## 用資格、品質與名額逐步縮小選擇

本例採 **task-aligned（任務對齊）**的結構，讓分類與定位共同決定該教誰。它來自 [TOOD（Task-aligned One-stage Object Detection，任務對齊單階段偵測）](https://arxiv.org/abs/2108.07755) 的 Task Alignment Learning（TAL，任務對齊學習）；assignment 是其中選正樣本的一部分。

先排除參考點不在 GT 內者，再給每一對 GT／候選算品質 `metric=score×IoU²`。一般式為 score^α×IoU^β，本例α1、β2 方便手算。分類高而框差，或框準但分錯類，都會使乘積變低；這不是隻取兩項中的較大值。

A 未遮罩品質約 `[0.225,0.3111,0.0035,0]`，乘資格 `[1,1,0,0]` 得 `[0.225,0.3111,0,0]`；B 遮罩後 `[0,0.1469,0.567,0]`。p2 對 A 雖 IoU0.1875，仍被點資格歸零。資格成立且品質>0 才進下一步。

每 GT 按品質留 **top-k（前 k 名）**，本例 k=2：A 選 p0、p1，B 選 p1、p2。限制名額是因為大框內可有大量點，不必把框很歪的候選都教成報物件；未選者在本例學背景，沒有 ignore。

這個 k=2 例每 GT 剛好只有兩個合格者，排序不改變入選集合。要看品質排名的用途，暫改 k=1：A 選 p1，因 0.3111>0.225，B 選 p2，因 0.567>0.1469，owner 為 `[-1,0,1,-1]`。若同樣資格、只按 score 排序，A 選 p0、B 選 p1，owner 為 `[0,1,-1,-1]`，B 反而舍棄 IoU0.9 的 p2，用 IoU0.4286 的 p1。這個對照才呈現把框品質放進排名的改變。

## Top-k 與衝突是兩步

回到 k=2，p1 被 A 與 B 都選中。它只有一組四距離，不能同時學兩個不同物件，所以解衝突時比較它對 GT 的 IoU，交給 2/3 較大的 A，而非 3/7 的 B。

最後 **owner=`[0,0,1,−1]`**：p0、p1 學 A，p2 學 B，p3 背景。owner 存的是 GT 清單編號 A0、B1，**不是類別**；本例剛好相同，取類別仍要再查 GT 的 label。top-2 只限制衝突前的名額，解完後不保證每 GT 仍有兩點，也不保證每 GT 有正樣本。

``` { .python data-excerpt="lesson_cases/12-assignment.py" }
# points [P,2]、gt [G,4]；scores 與 overlaps（IoU 表）都是 [G,P]；k=2
# 在這之前，owner 已建成 4 個 -1（每個候選先當背景）

# 第 1 步，資格表 inside [G,P]：參考點嚴格在框內才是 True
inside = ((points[None] > gt[:, None, :2]) & (points[None] < gt[:, None, 2:])).all(-1)
# 第 2 步，品質 [G,P]＝score×IoU²（α=1、β=2 這一組是方便手算的教學值，不是原始設定）
# 原始設定：TOOD 論文用 α=1、β=6；YOLOv8 訓練時用 α=0.5、β=6
metric = scores * overlaps.square()  # teaching pair alpha=1, beta=2; TOOD uses 1, 6 and YOLOv8 0.5, 6
selected = torch.zeros_like(inside)  # [G,P]，全 False：GT g 有沒有選中候選 p
for g in range(len(gt)):  # 第 3 步：每個 GT 各挑前 k 名
    eligible = torch.where(inside[g] & (metric[g] > 0))[0]  # 合格候選的編號，A 得 [0,1]
    n = min(k, len(eligible))  # 合格的不到 k 個就全收
    if n:  # n=0（沒有合格候選）就跳過
        selected[g, eligible[metric[g, eligible].topk(n).indices]] = True
for p in range(len(points)):  # 第 4 步：逐個候選解衝突
    candidates = torch.where(selected[:, p])[0]  # 選中 p 的 GT 編號，p1 得 [0,1]
    if len(candidates):  # 沒有 GT 選它，owner[p] 保持 -1
        owner[p] = candidates[overlaps[candidates, p].argmax()]  # 交給 IoU 最大的 GT
```

這是完整 assign 的核心。`selected [G,P]`記每 GT 選了誰，owner 記每候選最後歸誰，角色不同。官方 Ultralytics 也會把此例 p1 交 A，但重疊用 CIoU、比較 GT 範圍也不同；下一章的 dual assignment 小例則改比品質解衝突，不能混成一套規則。

??? note "逐行走查：用 A（g=0）和 p1 帶一次"

    先認兩個函式：`torch.where(條件)[0]` 回傳條件為 True 的位置編號；`.topk(n)` 取最大的 n 個值，`.indices` 是這些值所在的位置。名稱對照：`scores`＝分類 score 表、`overlaps`＝IoU 表、`inside`＝資格表、`metric`＝品質、`selected`＝哪個 GT 選中哪個候選的 True／False 表。下面 T＝True、F＝False。

    1. A（g=0）：`inside[0]`＝[T,T,F,F]，遮罩前的品質 `metric[0]`≈[0.225, 0.3111, 0.0035, 0]。兩個條件都成立的只有 p0、p1，所以 `eligible`＝[0,1]。
    2. `n`＝min(2, 2)＝2。
    3. `metric[0, eligible]`＝[0.225, 0.3111]。`.topk(2).indices`＝[1,0]：這是在 `eligible` 這個小清單裡的位置，較大的 0.3111 在位置 1，較小的 0.225 在位置 0。
    4. `eligible[[1,0]]`＝[1,0]，把清單裡的位置換回候選編號，於是 `selected[0,1]`、`selected[0,0]` 設成 True。
    5. B（g=1）同理：`eligible`＝[1,2]，品質 [0.1469, 0.567]，`.topk(2).indices`＝[1,0]，換回候選編號是 [2,1]。第一個迴圈跑完，`selected`＝[[T,T,F,F],[F,T,T,F]]。
    6. p1（p=1）：`selected[:,1]`＝[T,T]，所以 `candidates`＝[0,1]，A、B 都選了 p1。`overlaps[[0,1],1]`＝[0.6667, 0.4286]，`argmax()`＝0，這也是在 `candidates` 裡的位置，所以 `owner[1]`＝`candidates[0]`＝0，p1 歸 A。
    7. p3（p=3）：沒有 GT 選它，`candidates` 是空的，`owner[3]` 保持 −1。

選或不選是離散跳躍，像[第 1 章](01-small-cnn.md)的 argmax，不提供可用的連續梯度。完整 assign 有 `@torch.no_grad()`；真實訓練把當步 score 與 decoded box detach 後交它定 target，再用**未 detach 的原輸出**算 loss 反傳。不是把 head 輸出全 detach，後者會切掉學習。

## 把 owner 接回 loss

owner≥0 為 foreground（前景、正樣本），−1 為背景。若接回上頁兩類 head，先用硬 target 示意，應按所屬 GT 的類別寫 1，背景全 0：

| 候選 | owner：GT 編號 | 兩類 target：class 0、class 1 | 本節前景 target |
| --- | --- | --- | --- |
| p0 | 0：A | `[1,0]` | 1 |
| p1 | 0：A | `[1,0]` | 1 |
| p2 | 1：B | `[0,1]` | 1 |
| p3 | −1：背景 | `[0,0]` | 0 |

本程式只實作最後一欄，另建四個全 0 可更新 logits，不訓練上方手填 score 表，也沒有接上一頁圖片模型。前景 target `[1,1,1,0]` 像第 7 章 obj，用以單獨觀察責任帶出的方向。正式 YOLOv8 沒有 obj，正樣本類別 target 按品質縮成 0～1 小數，且框 loss 只讀 owner 指定的 GT；背景不算框 loss。

mean BCE 對每個 logit 的梯度爲 `(sigmoid(z)−t)/4`，4 是候選平均的分母。z 全 0，所以正樣本爲−0.125、背景+0.125；lr 1 的一次 SGD 使 logits 變 `[0.125,0.125,0.125,−0.125]`，正升、負降，這後一組沒有印出。

沒有 GT 時 G=0，owner 應全部−1，仍可教背景；沒有正樣本，框 loss 不可對空集合取 mean，以免 NaN。訓練 assignment 可讓 A 有兩正樣本，評估配對卻每 GT 只配一次，重複算 FP；不能用訓練多配規則算 AP。

執行 `PYTHONPATH=. python lesson_cases/12-assignment.py` 或頁首 Colab，核對品質表、owner、前景 target、BCE 梯度、空圖 owner 與練習改框結果。品質四捨五入三位印成 0.311、0.147，沒有品質逐值 assert；四個斷言核對 owner、梯度、空 owner 與改框 owner。這個單步只核對分配與監督方向，沒有 AP。

??? note "本例和官方實作差在哪"

    以本節最後「來源」連結的 Ultralytics 原始碼為準，主要差異如下：

    - **指數與名額**：品質一樣是 score^α × IoU^β，但 `loss.py` 設 α=0.5、β=6，每個 GT 取前 10 名；本例為了手算，用 α=1、β=2、前 2 名。
    - **重疊度**：官方算重疊度用的不是普通 IoU，而是第 11 章〈[IoU 類 loss](11-iou-loss.md)〉的 CIoU，並把負值截成 0；品質和解衝突都用它。本例為了手算，用普通 IoU。
    - **解衝突時比哪些 GT**：本例只在選中這個候選的 GT 之間比 IoU；官方則在所有 GT 的重疊度表中取最大值；不符合框內資格的位置先填 0，並非只在 top-k 選中這個候選的 GT 之間比。因此候選可能交給原本沒選它的 GT；全零同分時的主人依固定實作的 `max` 結果。本例只有兩個 GT，兩種比法結果相同。
    - **正樣本的 target**：官方不是 1，而是依品質縮放、介於 0～1 的小數；本例簡化成 1。
    - **跨尺度**：官方把幾種 stride（通常是 8、16、32）特徵圖上的候選點放在一起排名、挑前 k 名；本例只有一個尺度的四個點。
    - **整批計算**：官方把一個 batch 的圖片一起算。每張圖的 GT 數不同，就補齊到相同長度，再用遮罩標出哪些是真的 GT。本例只有一張圖，用 Python 迴圈逐一處理。
    - **小物件的候選**：官方選候選時，會把邊長小於 16 的 GT 框以中心暫時放大，讓小物件也有候選點（[16.3 節](16-training.md)的 STAL（Small-Target-Aware Label Assignment，照顧小物件的候選分配）會介紹，現在可以先跳過）；本節沒有套用。

每批都要算 GT×候選的重疊、遮罩、排序與衝突；模型初期預測差，排名也可能不穩。β更大會放大 IoU 差異：0.5 與 2/3 平方差約 1.8 倍，六次方約 5.6 倍，所以不是越大越好。正式使用要記規則和版本，也要按每 GT 查正樣本數：總數沒變，個別物件仍可能失去候選。

容易出錯的是漏框內資格、取錯 GT 類別 score、未解衝突就讓一候選有兩 target，或把所有未選者當 ignore，丟掉背景負訊號。

自主練習（先自己算，再展開答案）：

1. 把 p1 的預測框改成 GT B 的 `[12,0,32,16]`，重新計算整欄 IoU，不只改一個 IoU 數字。提示：新框和 A 的交集寬是 20−12=8。A、B 各選誰？owner 變成什麼？這題手算即可；完整程式結尾已經做了這個修改，算完再對照輸出的最後一行。
2. 改一個分數：在 Colab 裡改「本節可修改的完整實驗」下面那一格程式；本機則改 `lesson_cases/12-assignment.py`，兩者是同一份程式。把 `main()` 裡 `scores` 第一列的第二個數，也就是 A 對 p1 的 `.7`，改成 `.1`。先預測：A 的品質、top-2 的選擇和 owner 會不會變？哪些斷言（assert）要跟著改？再執行核對。

??? note "參考答案"

    **第 1 題**：新框和 A 相交 8×16=128，兩框面積各 320，IoU=128/(640−128)=0.25；新框和 B 完全相同，IoU=1。品質分別為 0.7×0.25²=0.04375 與 0.8×1²=0.8。A、B 仍都選 p1，但解衝突時比 IoU（0.25<1），p1 改屬 B，owner 變成 `[0,1,1,-1]`。A 的正樣本數由 2 變 1。完整程式結尾已有這個修改的斷言：`assert moved_owner.tolist() == [0, 1, 1, -1]`。

    練習前後全批都是 3 個正樣本，只看總數會以為沒事，其實 A 少一個、B 多一個。物件彼此重疊很多的圖（擁擠場景）常出現這種搶候選的情況，所以要分別數每個 GT 有幾個正樣本。

    **第 2 題**：A 遮罩後的品質變成 [0.225, 0.0444, 0, 0]（0.1×(2/3)²≈0.0444；輸出第一行只印到小數第 3 位，是 0.044）。A 的合格候選仍只有 p0、p1，兩個都入選；B 仍選 p1、p2。衝突比的是 IoU（2/3>3/7），p1 仍歸 A，owner 不變，還是 `[0,0,1,-1]`。所有斷言都不必改，照樣通過。可見本例只改分數、而且改完仍大於 0 時，看不出變化：每個 GT 只有兩個合格候選，衝突又只看 IoU。若把某個合格候選的分數改成 0，它的品質也變成 0，就不再是那個 GT 的合格候選：例如把 A 對 p1 的 `.7` 改成 0，A 只剩 p0 合格，p1 改歸 B，owner 變成 `[0,1,1,-1]`，`main()` 的第一個斷言就不通過。

    延伸：分數維持 `.1`，再把 `assign` 裡解衝突的那一行改成比品質，也就是把 `owner[p] = candidates[overlaps[candidates, p].argmax()]` 的 `overlaps` 換成 `metric`。這時 A 對 p1 的品質 0.0444 小於 B 的 0.1469，p1 歸 B，owner 變成 `[0,1,1,-1]`。`main()` 裡的 `assert owner.tolist() == [0, 0, 1, -1]` 要改成 `[0, 1, 1, -1]`，其他斷言不變。兩種衝突規則在這裡給出不同答案。

參考來源：[TaskAlignedAssigner：品質、top-k、衝突與依品質縮放的 target](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py)、[loss.py：訓練時把 detach 後的預測交給 assigner，並設定 α、β 與 top-k](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-assignment.json)

??? example "展開本次實際輸出"

    ```text
    inside-masked score * IoU^2: [[0.225, 0.311, 0.0, 0.0], [0.0, 0.147, 0.567, 0.0]]
    candidate owner (-1 background): [0, 0, 1, -1]
    foreground target: [1.0, 1.0, 1.0, 0.0]
    gradient: [-0.125, -0.125, -0.125, 0.125]
    empty image owner: [-1, -1, -1, -1]
    exercise: move predicted box p1 to GT B; owner: [0, 1, 1, -1]
    ```

<!-- curriculum-evidence:end -->
