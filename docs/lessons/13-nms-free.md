# 13.2 NMS-free：拿掉 NMS 前，重複候選學會了什麼

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/13-nms-free.ipynb){ .md-button }

上一節把少報重複的要求交給一對一監督。這份要求究竟改變了什麼？假設 A 附近有兩個框得不錯的候選 p0、p1，B 則有 p2。若三者都拿高分，直接刪掉 NMS，輸出仍會是 A 兩框、B 一框。我們需要讓 A 的一個候選留高分、另一個降下去，同時保住 B。

先把框的位置固定，就能只研究這一步：哪個候選被當正樣本，會怎麼改變它的分數。再把同一組分數交給三條推論路徑，比較「訓練先壓低重複」與「最後用 NMS 刪重複」各做了哪件事。

## A 的兩個候選，只有一個需要報出

本例只有一類。GT A 是 `[0,0,10,10]`，GT B 是 `[20,0,30,10]`，座標是畫素 xyxy（左、上、右、下）。四個固定候選如下：

| 候選 | 框 | 與物件的關係 |
| --- | --- | --- |
| p0 | `[0,0,10,10]` | 恰好等於 A |
| p1 | `[1,0,11,10]` | A 右移 1 畫素，仍框住 A 大部分 |
| p2 | `[20,0,30,10]` | 恰好等於 B |
| p3 | `[40,0,50,10]` | 背景 |

p0、p1 的交集寬 9、高 10，面積 90；兩框面積各 100，所以 IoU=`90/(100+100−90)=90/110≈0.8182`。NMS 的 IoU 門檻若是 0.5，就會刪去兩者中較後被處理的那一個。p2 與它們不重疊，應該保留；p3 則應因低分被濾掉。所需答案是 A、B 各一框，不能只要求框數變少。

沿用 13.1 的監督差別，但這次直接人工指定由 p0 負責 A，省去 assignment 的挑選步驟：

| target | p0 | p1 | p2 | p3 |
| --- | --- | --- | --- | --- |
| 一對多 | 1 | 1 | 1 | 0 |
| 一對一 | 1 | 0 | 1 | 0 |

兩份 target 只差 p1。一對多教 p0、p1 都報 A；一對一教 p1 即使也框住 A，仍要低分。p2 兩邊都受 B 監督，p3 兩邊都是背景。這不是官方規則保證會挑 p0，而是讓比較只增加「p1 要高還是低」這一個變化。

![四個候選框的 x 範圍與兩套 target 的訓練後分數](../assets/diagrams/13-nms-free.svg)

圖左依比例畫出 GT 和候選的 x 範圍，所有框的 y 都是 0～10。圖右列兩套 target 與各自訓練 100 步後的分數；黃底 p1 是唯一改變的要求。這些分數來自頁末保存的執行紀錄，下面說明它們如何學出來。

## 不改框，只讓四個分數學 target

每個候選有一個可學 logit，sigmoid 後是它的分數。四個 logit 本身就是四個參數，shape `[4]`；框的 shape 是 `[P,4]=[4,4]`，但框不更新。兩份 target 各自從全 0 的 logits 開始，所以起點分數同為 sigmoid(0)=0.5。

這裡沒有 CNN，也沒有共用特徵的雙 head。兩次獨立實驗使用相同的初值、SGD 學習率 1 和 100 次更新，只換 target。程式把它們叫 `many-head`、`one-head`，下文稱一對多分數、一對一分數，指的都是這四個可學數字。

BCE 對單一 logit 的梯度是「sigmoid 分數−target」；四項取平均，還要除以 4。第一步，target=1 的 logit 梯度為 `(0.5−1)/4=−0.125`，SGD 把它從 0 推到 0.125；target=0 的則推到 −0.125。於是 p1 在兩次實驗從同一個起點往相反方向走，其他三個保持相同。完整程式的 `fit_scores` 如下，只添加中文註解：

``` { .python data-excerpt="lesson_cases/13-nms-free.py" }
def fit_scores(target):
    logits = nn.Parameter(torch.zeros(4))  # 四個 logit 本身就是要學的參數，起點都是 0
    optimizer = torch.optim.SGD([logits], lr=1.)  # 學習率 lr=1
    for _ in range(100):  # 更新 100 步
        optimizer.zero_grad()  # 先清掉上一步的梯度（backward 會累加梯度）
        loss = F.binary_cross_entropy_with_logits(logits, target)  # 預設取四項平均（mean）
        loss.backward()  # 每個 logit 的梯度是 (sigmoid(logit) - target) / 4
        optimizer.step()  # logit = logit - lr * 梯度
    return logits.sigmoid().detach()  # 轉成分數；detach() 讓回傳的分數脫離計算圖
```

100 步後，target=1 的分數約 0.96，target=0 的約 0.04，logit 約 ±3.14。p1 的低分是 loss 經 backward、step 教出的，沒有手填。sigmoid 會逼近 1 或 0，不會在有限 logit 時恰好等於它們。

這個簡化保留了「target 改變梯度方向，梯度改變排序分數」的關係。四個 logit 彼此獨立，容易分開學；真實相鄰候選從同一網路和相近特徵產生分數，未必能如此乾淨地一高一低。這也是 13.1 保留一對多特徵監督的用途；本例沒有測這項協作。

## 同一組框，三條路如何保住 A 和 B？

先以 score>0.5 篩掉低分，再比較以下三條路徑。score 門檻比的是候選自己的分數；NMS 的 IoU 門檻則比兩個預測框的重疊。兩者雖都設 0.5，作用不同。

| 路徑 | 留下候選 | 解讀 |
| --- | --- | --- |
| 一對多，不做 NMS | p0、p1、p2 | A 重複，B 保住 |
| 一對多，加 NMS | p0、p2 | 最後依重疊刪 p1，A、B 各一框 |
| 一對一，不做 NMS | p0、p2 | p1 已學低分，由 score 篩掉，A、B 各一框 |

一對多的 p0、p1、p2 初值與 target 都相同，最後分數也完全相同。本例 NMS 用 `scores.argsort(descending=True, stable=True)`，同分維持輸入順序 p0、p1、p2。第一輪保留 p0，刪掉和它 IoU=0.8182 的 p1；第二輪保留 p2。stable 指穩定排序：它決定同分者先後，不是衡量框品質。

NMS 不讀 GT，所以不知道 p0 比 p1 更接近 A。若先送 p1，它會留 p1；p1 對 A 的 IoU 仍高於 0.5，也能算找到 A。程式因此同時檢查框數與物件覆蓋，而不把「一定留 p0」當成 NMS 的責任。以下是完整程式的三條路徑：

``` { .python data-excerpt="lesson_cases/13-nms-free.py" }
boxes = torch.tensor([[0., 0., 10., 10.], [1., 0., 11., 10.], [20., 0., 30., 10.], [40., 0., 50., 10.]])  # p0～p3
many = fit_scores(torch.tensor([1., 1., 1., 0.]))  # 一對多分數
one = fit_scores(torch.tensor([1., 0., 1., 0.]))  # 一對一分數
many_ids = torch.where(many > .5)[0].tolist()  # 一對多不做 NMS：分數大於 score 門檻的候選編號
one_ids = torch.where(one > .5)[0].tolist()  # 一對一不做 NMS
kept = nms(boxes[many_ids], many[many_ids])  # 一對多加 NMS；threshold 預設 .5，是 NMS 的 IoU 門檻
many_after_nms = [many_ids[i] for i in kept]  # 把 kept 的位置換回原本的候選編號
assert len(many_ids) == 3 and len(many_after_nms) == 2  # 框數
assert one_ids == [0, 2]  # 一對一不做 NMS 留下 p0、p2
assert 2 in many_after_nms and any(i in many_after_nms for i in (0, 1))  # 覆蓋斷言
```

`kept` 是送進 NMS 那一組的局部位置，所以再用 `many_ids[i]` 翻回原候選編號。最後的覆蓋斷言要求 B 有 p2，A 留 p0 或 p1 都可；只檢查框數會漏掉「兩框都屬 A、B 消失」的錯。

## top-k 為什麼不能替你辨認重複？

NMS-free 的意思是推論不跑 NMS，仍可保留 score 門檻和 top-k。YOLOv10 推論用一對一 head，先挑分數最高的 k 個（框, 類別）組合，預設最多 300，再以 score 門檻篩選；多類時，同一框可能以不同類別占多筆。本例每框只有一分，所以挑組合等於挑框。

top-k 只比較分數，k 是輸出上限，不是圖中物件數。沿用四個框，另設一組人工分數 `[0.92,0.90,0.80,0.05]`：這代表 p1 也拿高分、一對一沒學好的情況，並非前面訓練的結果。即使 k=2 剛好等於 GT 數，top-2 也會挑 p0、p1，兩框都報 A，漏掉 B。它沒有 NMS 那種「看兩框在哪裡」的資訊；少報重複必須先由分數完成。

若拿剛才訓練的一對多分數做 top-2，三個高分者完全平手；`topk` 沒有 stable 選項，不能保證是哪兩個。這也不能當成去重規則。

重複還會增加 FP：評估時一個 GT 只能被配對一次。人工分數排序 p0→p1→p2，p0 找到 A，p1 因 A 已被配走算 FP，p2 找到 B。配對 IoU 門檻 0.5 比的是預測與 GT，和上面的 NMS 門檻又不同。其 AP50 可手算為 5/6；只留 p0、p2 時為 1，細算如下。

??? note "手算評估：重複框讓 AP 從 1 降到 5/6"

    這裡評的是不做 NMS 時留下 p0、p1、p2 三個候選的輸出（和一對多不做 NMS 留下的候選相同），分數改用上面的人工分數排序，三個分數各不相同。人工分數中 p3 的 0.05 低於 score 門檻 0.5，先被刪掉，不進評估。

    配對 IoU 門檻設 0.5，依分數由高到低逐筆判定，規則同[第 6 章](06-evaluation.md)（P 是 precision，R 是 recall）：

    | 排名 | 候選 | score | 判定 | P | R |
    | --- | --- | --- | --- | --- | --- |
    | 1 | p0 | 0.92 | TP：配對 A（IoU=1） | 1 | 1/2 |
    | 2 | p1 | 0.90 | FP：A 已被 p0 配對，p1 和 B 不重疊 | 1/2 | 1/2 |
    | 3 | p2 | 0.80 | TP：配對 B（IoU=1） | 2/3 | 1 |

    兩個 GT 都找到，所以漏檢 FN=0；最後 precision=`TP/(TP+FP)=2/3`，recall=`TP/(TP+FN)=1`。

    算 AP 前先取包絡：在每個 recall 位置，取「往右看拿得到的最高 precision」。recall 0 到 1/2 這段，往右看最高的是排名 1 的 precision 1，所以高度是 1；排名 2 的 FP 沒有增加 recall，不占寬度。recall 1/2 到 1 這段，高度是排名 3 的 2/3。all-points interpolated AP 的面積為 `.5×1+.5×(2/3)=5/6`。

    NMS 後（p1 被刪掉），或只留 p0、p2 的結果，都只剩兩個正確框，此例 AP=1。top-2 若留下 p0、p1，recall 反而只有 0.5。這是可以手算核對的人工評分，不是模型在獨立圖片上的效果。

    若像[第 7 章評估 held-out 圖片](07-heldout.md)時那樣，把候選截斷門檻設得很低（例如 0.01），p3 也會進入評估：它排第 4、是 FP，最終 precision 變成 2/4；但 recall 在排名 3 已經是 1，所以 AP 仍是 5/6。


## 省去逐框刪除，訓練便多了一項責任

NMS 要依高分順序逐輪保留、比較 IoU、刪候選；後面的決定依賴前面保留了誰。把它省掉，既省這些重疊比較和順序處理，也減少部署工具必須支援的運算。top-k 仍要挑高分者，並不是所有排序成本都消失；實際省多少時間取決於候選數、硬體和實作。

相應的責任是：一對一分數要在相近候選中選對負責者。若兩者都高分就重複，都低分就漏檢，或較差的框勝出也會傷害定位。真正不同的兩個物件本來可能互相重疊，不能為了少框把其中一個刪掉。要檢查的是同一物件不重複，同時不同物件仍能找到。

本例支持「改監督能壓低重複候選分數」這個機制，沒有測真實 YOLOv10 的品質或速度。正式評測要在 held-out 圖片檢查 AP、重複 FP 與漏檢；計時則固定解析度、score 門檻、輸出上限和裝置，把前處理、模型、解碼與篩選一起計入端到端時間。

執行 `PYTHONPATH=. python lesson_cases/13-nms-free.py`，或在 Colab 跑完整程式。前兩行各四分數對應 p0～p3；三條路徑依序印 `[0,1,2]`、`[0,2]`、`[0,2]`，重複 IoU 為 0.8182，人工高重複分數的 top-2 為 `[0,1]`。兩組分數都經過實際參數更新，框位置則一直固定。

## 自主練習

自主練習：在完整程式（Colab 裡「本節可修改的完整實驗」下方那格，或本機的 `lesson_cases/13-nms-free.py`）動手改，先預測結果，再執行核對。上面摘錄的三行斷言寫死了原題的答案；改了程式，答案不同的斷言就要跟著改成你預測的結果，否則程式會在那一行報錯停下。要改哪幾行、改成什麼，見參考答案。

1. 把一對一 target 那行 `one = fit_scores(torch.tensor([1., 0., 1., 0.]))` 改成 `[0., 1., 1., 0.]`，也就是改由 p1 負責 A。一對一不做 NMS 會留下哪幾個候選？評估時，A 留下的那個框還算 TP 嗎？
2. 回到原本的 target，只把兩處 score 門檻（`many > .5`、`one > .5`）提高到 0.99。三條推論路徑各剩幾框？recall 是多少？

??? note "參考答案"

    **第 1 題**：新的 target 讓 p1、p2 學高分，p0、p3 學低分。一對一仍有兩個高分候選，不做 NMS 留下 `[1, 2]`。斷言 `assert one_ids == [0, 2]` 要改成 `assert one_ids == [1, 2]`；其他斷言不用改。

    A 唯一的輸出是 p1，它和 A 的 IoU 是 0.8182，評估時仍要跟配對 IoU 門檻 0.5 比；有達到，所以還是 TP。本節人工指定的 target 讓每個物件由一個候選負責，不保證那個框和 GT 完全重合；真實模型也不保證剛好留下這個數量。若唯一留下的框和 GT 的 IoU 低於 0.5，評估時會同時多一個 FP、少找到一個物件（FN）。

    **第 2 題**：這次 100 步訓練出的兩組分數，最高只有約 0.96，沒有任何候選大於 0.99，所以三條推論路徑都是 0 框。沒有重複框，但兩個物件也都沒找到，recall=0。框少不代表成功；靠調高門檻讓框變少，不是架構成功的證據。

    程式要改的地方：

    - 只改兩處 score 比較：`many_ids = torch.where(many > .5)[0].tolist()` 與 `one_ids = torch.where(one > .5)[0].tolist()` 的 `.5` 改成 `.99`。**不要改 `nms` 的 `threshold=.5`**，那是 NMS 的 IoU 門檻。
    - 框數斷言改成 `assert len(many_ids) == 0 and len(many_after_nms) == 0`。
    - `assert one_ids == [0, 2]` 改成 `assert one_ids == []`。
    - 覆蓋斷言 `assert 2 in many_after_nms and any(i in many_after_nms for i in (0, 1))` 改成 `assert many_after_nms == []`。score 篩選後已經沒有框，不能再要求保留 p2 或 p0／p1。

參考來源：[YOLOv10 論文](https://arxiv.org/abs/2405.14458)、[官方 v10Detect 推論](https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/nn/modules/head.py)（`v10Detect` 是官方程式中 YOLOv10 偵測 head 的類別名稱；這裡的 `max_det = 300` 只在匯出模型時使用）、[postprocess 的 top-k（`v10postprocess`）](https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/ops.py)（postprocess 指模型輸出後挑框的後處理）、[預測時只取一對一 head 的輸出，top-k 後再用 score 門檻篩選](https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/models/yolov10/predict.py)（k 取自設定 `max_det`）、[預設設定（`max_det: 300`）](https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/cfg/default.yaml)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/13-nms-free.json)

??? example "展開本次實際輸出"

    ```text
    many-head scores: [0.959, 0.959, 0.959, 0.041]
    one-head scores: [0.959, 0.041, 0.959, 0.041]
    many/no NMS: [0, 1, 2] many/NMS: [0, 2] one/no NMS: [0, 2]
    top-2 on duplicate-heavy scores: [0, 1] (misses object at candidate 2)
    duplicate IoU=0.8182
    ```

<!-- curriculum-evidence:end -->
