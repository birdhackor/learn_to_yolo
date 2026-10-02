# 12.3 Sample assignment：哪個候選值得被教

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/12-assignment.ipynb) · 原始碼：`lesson_cases/12-assignment.py`

前置是候選點、IoU 和 classification logits。密集偵測器對一張影像輸出許多候選，但標註可能只有兩個物件。每個候選都學最近的真值嗎？如果它位於框外，或分類分數很高但位置很差，這個選擇可能給出矛盾訊號。本節依序看資格、品質、數量限制和衝突，不把 assignment 當成一個黑盒。

歷史機制是 YOLOv8 使用與分類、定位品質相關的 task-aligned assignment。本次以 `score×IoU²`、框內資格、每個 GT top-2 和最高 IoU 解衝突，示範 TAL 的部分結構。官方實作的指數、quality target 正規化、跨尺度與 batch 細節更多，本例不稱完整 TAL。GT指ground truth，也就是人工真值標註；候選是某個特徵位置輸出的預測，assignment是它與GT的責任分配。這節不改變推論NMS。

## 兩個真值、四個候選

真值A是`[0,0,20,16]`、類別0；B是`[12,0,32,16]`、類別1，兩框有重疊。候選點依序為 `(8,8),(16,8),(24,8),(40,8)`。第一點只在 A 內，第二點同時在 A、B 內，第三點只在 B 內，第四點在兩框之外。shape 分別為 points `[P,2]=[4,2]`、GT `[G,4]=[2,4]`、品質表 `[G,P]=[2,4]`。P是候選數、G是真值物件數。分類score表每列取該GT類別的機率，通常由`[P,C]`分類輸出按GT class索引而來，本例直接給已索引好的`[G,P]`。它是獨立sigmoid類別分數，不要求同一候選各類機率相加為1。

本例固定各候選已解碼的預測框，再由程式真正計算它與每個GT的IoU，不是點與框的IoU。四個預測框是 `[0,0,10,16]`、`[4,0,24,16]`、`[14,0,32,16]`、`[40,0,50,10]`。同一預測框對A、B的IoU必須來自同一幾何位置，不能任意填入兩個彼此不可能同時成立的數字。分類score與算得IoU如下：

| 真值 | p0 的 score／IoU | p1 | p2 | p3 |
| --- | --- | --- | --- | --- |
| A | 0.9／0.5 | 0.7／.6667 | 0.1／.1875 | 0.8／0 |
| B | 0.1／0 | 0.8／.4286 | 0.7／0.9 | 0.9／0 |

資格mask要求x、y嚴格位於框內，邊界點也排除，且品質metric必須>0才進入top-k；本例沒有ignore候選。A 的合格品質是 `[.225,.3111,0,0]`；B 是 `[0,.1469,.567,0]`。以p1對A為例，相交16×16=256、兩框面積各320，IoU=`256/(640−256)=2/3`；對B相交12×16=192，IoU=`192/(640−192)=3/7`。p0分類分數較高，但A的p1位置較好，乘上IoU²後排名反轉。這讓分類與定位一起參與分工，避免只追求一項高分。

## Top-k 與衝突是兩步

A 選 p0、p1，B 選 p1、p2。p1 被兩個 GT 選中，但單一候選的四邊距離與類別 target 不能同時代表兩個不同物件。因此本例比較其IoU：對A為2/3、對B為3/7，交給A。最後owner為`[0,0,1,-1]`，−1代表背景。top-2是每GT最多先選兩個，不保證解衝突後仍有兩個，也不是每個GT都必然有正樣本。

```python
metric = scores * overlaps.square()
eligible = torch.where(inside[g] & (metric[g] > 0))[0]
selected[g, eligible[metric[g, eligible].topk(n).indices]] = True
owner[p] = candidates[overlaps[candidates, p].argmax()]
```

`assign` 用 `torch.no_grad()`。目前的 prediction 參與決定 target，卻不透過離散 top-k 反傳；隨後 loss 對仍保留計算圖的 logits 反傳。不要把「target 計算 detach」誤寫成「模型輸出全 detach」，後者會使 head 學不到。

## 把 owner 接回 loss

供assignment的固定類別score並不是下面要更新的logits；本例另建立四個前景logits，隔離展示owner如何變成loss target。完整模型會對同一head的未detach輸出計算loss。為清楚核對，本次把正樣本target簡化成 `[1,1,1,0]` 的二元前景，不冒充官方 TAL quality class targets。四個 logits 都是 0，sigmoid 為 .5；mean BCE 的 logit gradient 是 `(p−target)/4`，所以印出 `[-.125,-.125,-.125,+.125]`。一次 SGD 後正位置 logit 上升，背景下降。完整 detector 的正樣本還要取 owner 對應的 GT 框和類別；背景不計框回歸 loss。

空影像也要合法。程式給 `G=0`，回傳四個 −1；所有候選可提供背景訊號，不要因為沒有 GT 就略過整張影像，更不要對空正樣本直接取 mean 得到 NaN。這節的 foreground 小例子沒有評估 AP，只有責任規則的可核對答案。

執行 `PYTHONPATH=. python lesson_cases/12-assignment.py`，必須得到上述品質表、owner、BCE 梯度與空圖結果。修改一個分數後，先預測 owner 是否變，才跑程式。無須下載資料。

## 收益、代價與常見錯誤

動態品質讓責任隨模型進步調整，比固定中心規則更能利用好候選；代價是每批要計算 GT×候選的 IoU、排名、mask 和衝突。初期預測都差時，排名也可能不穩。指數提高會更強調好框，但讓品質較差的候選指標更接近零；它不是免費增益。

常見錯誤包括先 top-k 再忘記框內資格、將 GT 類別 score 索引成另一類、讓一個候選擁有兩份 target、把未選候選都叫 ignore，以及把訓練 assignment 和評估 TP matching 混用。官方設定與版本會變；本節只說明固定的教學規則。

自主練習：把p1的預測框改成GT B的`[12,0,32,16]`，重新計算整欄IoU，不只改一個IoU數字。答案為p1對A的IoU=.25、對B=1，品質分別為`.7×.25²=.04375`與`.8×1²=.8`；A、B仍都選p1，但解衝突後p1屬B，owner成`[0,1,1,-1]`。A正樣本數由2變1。案例結尾已有這個干預的assert，若擁擠資料常發生此情況，要觀察每GT正樣本覆蓋，而不只看全批正樣本總數。

來源查覈：2026-10-02。[TaskAlignedAssigner 的品質、top-k、衝突與 target 正規化](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py)、[訓練時 detached prediction 用於 assignment](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)。這份現行 source 還含 YOLO26 小物件候選擴張；本節未套用該規則。

本例p2的預測框雖與A重疊，參考點在A之外，因此A的p2品質被mask成0；候選點資格與預測框IoU是兩項不同幾何檢查。本節衝突用最高IoU，下一節品質表的小實驗改用最高quality，不能當成沿用同一個assigner。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-assignment.json)

??? example "展開本次實際輸出"

    ```text
    inside-masked score * IoU^2: [[0.22499999403953552, 0.3109999895095825, 0.0, 0.0], [0.0, 0.1469999998807907, 0.5669999718666077, 0.0]]
    candidate owner (-1 background): [0, 0, 1, -1]
    foreground target: [1.0, 1.0, 1.0, 0.0]
    gradient: [-0.125, -0.125, -0.125, 0.125]
    empty image owner: [-1, -1, -1, -1]
    exercise: move predicted box p1 to GT B; owner: [0, 1, 1, -1]
    ```

<!-- curriculum-evidence:end -->
