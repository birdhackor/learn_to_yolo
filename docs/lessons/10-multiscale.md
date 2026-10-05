# 10 YOLOv3 機制：同一個 pixel 框看兩種尺度

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.0/notebooks/10-multiscale.ipynb){ .md-button }

只用 4×4 的格子預測時，8×8 pixel 的小物件容易被背景淹沒。本節試著多加一個較細、8×8 格的預測 head。讀完你能算出同一個框在兩種格子下的 target，說出兩個 head 各學什麼，也知道推論時怎麼合併兩個 head 的框、去掉重複。

前置：第 7 章的 [grid targets](07-targets.md)。第 9 章的 [anchor](09-anchors.md) 用來對照原版，這個主例本身不用 anchor。

64×64 圖若只用 4×4 格，相鄰位置相隔 16 pixel，每格負責 16×16 的中心範圍。8×8 pixel 的物件只占這範圍的 1/4，要和周圍背景一起概括成一組特徵，細節容易被稀釋。這次加上 8×8 格、相鄰位置只隔 8 pixel 的 head，先檢查接法與成本，再看一個短訓練結果。

模型同時輸出 fine（8×8 細格）和 coarse（4×4 粗格）。本例人工把小物件交給 fine、大物件交給 coarse；推論則把兩支的框換回 pixel 座標、合併後去重。這是 YOLOv3 多尺度預測的簡化示範，不重現原版；完整差異放在主例之後。

本節改用小網路 `TwoScale`，channel 數、取得 4×4 的方式與初始化都不同於第 7 章的 GridDetector，不能當成「原模型只加一個 head」的品質對照。這次只做多尺度預測；深淺特徵融合留在第 11 章〈[特徵融合](11-fusion.md)〉。

## 同一個小框的兩種 target

材料是一張 64×64 黑底圖：紅色小框 `[5,5,13,13]`（class 0），中心 `(9,9)`、寬高 `(8,8)`；藍色大框 `[32,32,56,56]`（class 1），中心 `(44,44)`、寬高 `(24,24)`，單位都是 pixel。訓練只分配小框給 fine、大框給 coarse；兩個 head 都能預測兩類，分配和類別編號無關。

下面分開看兩支的責任格。綠格是正格，斜線強調「有另一個物件的中心，卻沒分配給這支」的負格；白格也全是負格。格座標寫 `(gx,gy)`，實際 tensor 索引仍先列 y、再欄 x。

![示意：coarse 4×4 格只以大框中心所在的 (2,2) 為正格，小框中心所在的 (0,0) 是負格](../assets/diagrams/10-multiscale-coarse.svg)

![示意：fine 8×8 格只以小框中心所在的 (1,1) 為正格，大框中心所在的 (5,5) 是負格](../assets/diagrams/10-multiscale-fine.svg)

同一個小框中心 `(9,9)`，除以 stride（相鄰位置的 pixel 間距）後，整數部分是格座標，小數部分是格內 xy：

| 項目 | 4×4／stride16 | 8×8／stride8 |
| --- | --- | --- |
| center ÷ stride | `(.5625, .5625)` | `(1.125, 1.125)` |
| cell `(gx,gy)` | `(0,0)` | `(1,1)` |
| 格內 xy | `(.5625, .5625)` | `(.125, .125)` |
| normalized wh（÷64） | `(.125, .125)` | `(.125, .125)` |
| decode cx | `(0 + .5625) × 16 = 9` | `(1 + .125) × 8 = 9` |
| decode w | `.125 × 64 = 8` | `.125 × 64 = 8` |

這張表只比較同一個框在兩種格子下怎麼編碼、能否還原。實際訓練時，本例小框只交給 8×8 那一支；4×4 那欄不進訓練 loss，完整程式只用它核對表中數值，以及後面「兩尺度都預測同一小框」的人工 decode 檢查。

注意 wh 仍除以全圖 64，不跟 grid 變。本節不用 anchor，wh 沿用第 7 章的寫法：sigmoid 後乘全圖 64，兩個尺度的 target 都是 0.125，都還原為 8 pixel。兩種 target 都應還原到同一個框，不能讓小框因尺度不同被放大兩倍。

??? note "stride、感受野與 anchor 單位"

    stride 不是感受野。`early` 是三層 3×3、stride 2 卷積；每層的範圍增加 `(3−1)×上一層間距`，間距依序為 1、2、4，所以 fine 理論感受野為 1→3→7→15。`deep` 再增加 2×8，得到 coarse 的 31×31。它們都比各自負責的格範圍大。

    本節用 sigmoid×64 表示寬高，不用 anchor。若改成第 9 章的 anchor×exp(tw)，必須寫明 anchor 單位：8 pixel 的 anchor 在 fine 是 1 格、在 coarse 是 0.5 格。把 fine 的「1 格」誤乘 coarse 的格寬 16，就會把 8 pixel 的框解成 16 pixel。

## 兩個 head 的實際連線

一張圖先經過 `early` 得到 8×8 特徵；一支直接接 `fine_head`，另一支經 `deep` 降到 4×4 再接 `coarse_head`。圖與下方教學寫法用 `_features` 表示特徵、`_pred` 表示七項預測，避免把兩者都叫 fine／coarse。

![示意：TwoScale 從同一張 fine 特徵分出兩條路，得到 fine_pred 與 coarse_pred，再各算 loss](../assets/diagrams/10-multiscale-branches.svg)

特徵 shape 是 `[圖片,channel,列,欄]`。1×1 head 把每個位置的 channel 組合成 7 項，再用 permute 移到最後一軸；prediction shape 是 `[圖片,列,欄,7]`。七項依序為 `[tx,ty,tw,th,obj_logit,class0_logit,class1_logit]`。fine 有 64 個候選，coarse 有 16 個，共 80 個；這是 score 篩選與 NMS **之前**的數量。

下面把同一次 forward 的運算攤開，用 `_features`／`_pred` 寫清楚資料角色；層與 loss 不變，原始命名放在摺疊區供對照。

```python
fine_features = model.early(image)                    # [1,16,8,8]
coarse_features = model.deep(fine_features)            # [1,32,4,4]
fine_pred = model.fine_head(fine_features).permute(0,2,3,1)
coarse_pred = model.coarse_head(coarse_features).permute(0,2,3,1)
# fine_small：小框的 8×8 target；coarse_large：大框的 4×4 target
loss = grid_loss(fine_pred,fine_small)['total'] + grid_loss(coarse_pred,coarse_large)['total']
```

??? note "完整程式的原始命名"

    完整 `forward` 的 `fine`／`coarse` 是特徵；`main` 收到的 `fine`／`coarse` 是兩個 head 的 prediction。上面的 `_features`／`_pred` 只區分這兩種角色，沒有更動執行程式。

    ``` { .python data-excerpt="lesson_cases/10-multiscale.py" }
    def forward(self,x):
        fine = self.early(x)
        coarse = self.deep(fine)
        return (self.fine_head(fine).permute(0,2,3,1),
                self.coarse_head(coarse).permute(0,2,3,1))
    ...
    fine,coarse = model(image)
    ...
    loss = grid_loss(fine,fine_small)['total'] + grid_loss(coarse,coarse_large)['total']
    ```

反向傳播時，`early` 同時收到兩項 loss 的梯度；`deep`、`coarse_head` 只收到 coarse 那一項，`fine_head` 只收到 fine 那一項。

coarse 正格的位置可驗算：44÷16=2.75，所以是 `(2,2)`、格內 xy `(0.75,0.75)`、wh `(0.375,0.375)`。同一大框中心在 fine 是 `(5,5)`，但那裡仍學背景；小框中心在 coarse 的 `(0,0)` 也一樣。沒分配給這支的物件，不學框或類別，若報出物件，還會被 objectness loss 懲罰。這是本例沒有 ignore 的代價，不能當成原版 YOLOv3 的監督規則（差異見頁末摺疊區）。

**推論時**沒有 GT 尺寸可查，不能先按真實物件大小刪掉另一個 head 的輸出，而是讓各 head 的分數與共同的 NMS 來處理候選。步驟是：先將所有尺度 decode 到同一套輸入 pixel 座標，再串接 boxes／scores／labels，最後做一次分類別 NMS（按預測類別分組、各組各做一次 NMS，也就是 6.1 節〈[人工框解碼與 NMS](06-decode-nms.md)〉的 class-wise NMS）。這裡有兩件事不能做：

- 不能直接串接不同尺度的格子索引（cell 索引）。fine 的格 (1,1) 代表 x 在 8–16 pixel，coarse 的格 (1,1) 代表 x 在 16–32 pixel；同樣叫 (1,1)，位置卻不同。所以每個尺度要先用自己的 stride decode 成 pixel 框，再串接。
- 不能只做各尺度內部的 NMS，否則會漏掉跨尺度的重複框。

完整程式另用人工 logits，讓兩個尺度都預測小框 `[5,5,13,13]`。兩份框值由前面的 target 表反解而來，不經模型。它們同類、座標相同，IoU=1；各尺度各有一框，合併後 NMS 才把數量 **2→1**。這說明尺度內 NMS 代替不了跨尺度去重。

本節的 `decode_grid` 會先做尺度內 NMS，合併後再做一次分類別 NMS；原版只在合併後做一次。兩者的差異見頁末補充。

??? note "人工 logits 與合併程式細節"

    框值由 target 經 `torch.logit` 反解；正格的 obj／class 0 填 10、class 1 填 −10，其餘 obj 填 −20。每個尺度 decode 後只剩同一小框。下面保留原程式的名稱；NMS 回傳子集位置，再用 `indices[...]` 換回合併後的列號。

    ``` { .python data-excerpt="lesson_cases/10-multiscale.py" }
    # decoded_scales：兩個尺度（先 coarse、後 fine）各自用 decode_grid 解碼的結果，
    # 每個都含 boxes、scores、labels；本例每個尺度只有 1 個框
    boxes = torch.cat([p['boxes'] for p in decoded_scales],dim=0)  # [2,4]
    scores = torch.cat([p['scores'] for p in decoded_scales],dim=0) # [2]
    labels = torch.cat([p['labels'] for p in decoded_scales],dim=0) # [2]
    assert boxes.shape == (2,4) and scores.shape == labels.shape == (2,)  # 核對上面三個 shape
    selected = []
    for label in labels.unique():                # unique()：列出出現過的類別，本例只有 0
        # where(條件) 對每個軸各回傳一份位置；labels 只有一軸，[0] 取出這一類候選的列號
        indices = torch.where(labels==label)[0]
        # nms 回傳的是在子集 boxes[indices] 裡的位置，用 indices[...] 換回原本的列號
        selected.append(indices[nms(boxes[indices],scores[indices],iou_threshold=.5)])
    selected = torch.cat(selected)               # 各類別留下的列號接成一串
    assert len(selected) == 1                    # 兩個重複框只留下一個
    # 留下的就是小框 [5,5,13,13]
    assert torch.allclose(boxes[selected],small['boxes'],atol=1e-4)
    ```

這個單步程式確認兩個 head 有梯度並更新，兩種 target 能還原同一框、跨尺度重複框能去掉。它是機制驗證，沒有小物件 AP 提升證據。

??? example "執行單步程式與核對輸出"

    執行[本節 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.0/notebooks/10-multiscale.ipynb)，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/10-multiscale.py`。應核對：

    - 表內兩個 target：輸出第 2、3 行的 4 個數，依序是表中的格內 x、y 與 normalized w、h。
    - 兩個 head 的輸出 shape，以及候選數 `64+16=80`（輸出第 4 行）。
    - 兩個 head 都有非零梯度，並做了一次更新。
    - 人工 logits 分別經兩種尺度 decode，都還原成同一個小框；再實際用 `torch.cat` 串接 boxes／scores／labels 並做分類別 NMS，跨尺度重複框 2→1（輸出第 1 行）。

    輸出第 5 行是 print 的固定字樣，不是量出的結果；前面的斷言全部通過才會印到這裡。


## 收益、代價與適用條件

較細特徵提供更多空間位置：8 pixel 寬的小框在 stride 8 下約占一格，而不是 stride 16 下的半格，可能更容易保留細節，也減少某些同格碰撞（兩個物件中心落進同一格）。代價是候選數從 16 增加到 80，logits 元素從 112（4×4×7）增加到 560（8×8×7+4×4×7），額外的 head 與變多的候選也增加計算及記憶體。不過本例的 8×8 feature 本來就是 `deep` 的輸入，只留 coarse head 也要算；網路裡新增的只有 1×1 的 `fine_head`（119 個參數）。若像第 11 章〈[特徵融合](11-fusion.md)〉那樣在 8×8 上再加卷積，計算及記憶體才會明顯增加。細尺度沒有憑空增加原圖的畫素：若原圖小物件已被 resize 到 2 pixel，增加 head 仍不能恢復消失的資訊。

多尺度適合同一份資料裡同時有很小和很大物件的情況；如果物件大小都差不多，只選一個合適的尺度可能就夠。例如物件都不小、原本的 4×4 已經夠用時，多出的 64 個候選就只是成本。

正式比較單尺度與兩尺度時，資料的來源切分（第 8 章）、訓練步數與評估門檻都要固定相同，並分別回報小物件與大物件的品質（例如 AP）、候選數，以及端到端時間（從輸入圖片到輸出最終框的總時間，含 decode 與 NMS）。如果加 head 的同時也延長訓練或改了輸入尺寸，就分不出改善來自多尺度還是其他改變。本節保留兩個 head 的接法，不代表兩尺度的效果比較好。第 11 章〈[特徵融合](11-fusion.md)〉的融合模組，是設計給這種 8×8 fine head 用的；該節用隨機張量檢查接法，沒有接回本節模型訓練。

常見錯誤：

- xy 索引對調。target 的格子索引順序是先 gy（列）、再 gx（欄）。
- 把 fine 的 wh 除以 8。wh 一律除以全圖 64，不跟格子變。
- 某個 head 沒有 loss，卻期待它學會。例如只算 fine 的 loss，`deep` 與 `coarse_head` 就收不到梯度。

自主練習（手算即可，不必改程式；先自己算，再展開答案）：

1. 中心 `(28,20)` 在 stride 8 下落在哪一格？格內 xy 是多少？若這個框寬高是 16×16，兩個尺度的 wh target 各是多少？
2. 同一個中心在 stride 16 下落在哪一格？格內 xy 是多少？

??? note "參考答案"

    **第 1 題**：28÷8=3.5、20÷8=2.5，所以落在 `(gx=3,gy=2)`，格內 xy `(.5,.5)`。在程式裡讀這格的 target，索引要寫 `[0,2,3]`（第 0 張圖、先 gy=2、再 gx=3），例如 `target['box'][0,2,3]`；寫成 `[0,3,2]` 會讀到另一格。wh 為 16×16 時，兩個尺度都使用 `.25,.25`（16÷64=0.25，不跟格子變）。

    **第 2 題**：28÷16=1.75、20÷16=1.25，所以落在 `(gx=1,gy=1)`，格內 xy `(.75,.25)`；wh 仍是 `.25,.25`。

## 從單步接線到 40 步實際預測

現在問兩個 head 能否學會這張圖。補充腳本沿用同一個 `TwoScale`、同一張紅小框與藍大框圖，seed 7、Adam、learning rate 0.01，訓練 40 步。兩個 head 每步都有有限、非零梯度，全部 8702 個參數的權重變化 L2 長度為 6.504，證明權重有改變。fine＋coarse loss 從第 1 次更新前的 3.6594，降到第 40 次更新前的 0.0703。

下圖是 **40 次更新全部完成後**，模型在同一張訓練圖上的真實預測。框是 logits 經各尺度 decode、合併、分類別 NMS 得到，沒有手填答案。為了算 AP，score 門檻用 0.05，低於畫框常用的 0.25。

![實測：40 次更新後的三個預測框，標出框 ID、TP／FP 與同類 GT IoU，紅色小框 GT 漏檢](../assets/diagrams/10-multiscale-observed-predictions.svg)

綠色虛線是 GT，橙色實線是預測。**#0～#2 依保存報告的 `prediction` 列號編號，不是分數排名**；IoU 比的是預測與同類 GT，配對仍依 score 由高到低、每個 GT 只用一次。

| 框 ID | class／score | 同類 GT IoU | IoU≥0.5 配對結果 |
| --- | --- | --- | --- |
| #0 | 0／0.763 | 0.44 | FP；紅色小框為 FN |
| #1 | 1／0.836 | 0.86 | TP，配到藍色大框 |
| #2 | 1／0.131 | 0.30 | FP；藍色 GT 已由 #1 配對 |

class 0 的 AP50=0，class 1 的 AP50=1，因此 mAP50=(0+1)÷2=0.5000。#2 與 #1 的 IoU 約 0.28，小於 NMS 的 0.5 門檻，所以留下；但它排在 TP 後面，沒有拉低這次 class 1 的 AP。本例每類剛好只有一個物件；一般的小物件 AP 應按面積分組，不能和 class 0 的 AP 畫等號。

loss 已降到 0.07，小框仍未達 IoU 0.5：#0 約 `[6.3,1.4,11.4,14.8]`，中心只偏不到 1 pixel，卻比 8×8 GT 窄約 3、高約 5.4 pixel。寬高 loss 用除以 64 的比例，這些誤差在 MSE 中很小，對小框 IoU 卻影響很大。**loss 下降不保證小框定位達標。**

這只是固定訓練圖的檢查，沒有獨立資料的小物件 AP 提升證據，也沒有單尺度品質對照。要只比較多 head 的效果，應固定 `TwoScale` 的 backbone、初始化、資料與訓練預算，另做只留 coarse head 的對照。

??? note "查看逐步 loss 與寬高誤差的細算"

    ![同一次 40 步實測的原始生成圖：逐步 loss 與預測](../assets/diagrams/10-multiscale-learning.svg)

    曲線每點都在該次更新前量，縱軸是 fine＋coarse loss；右圖則在 40 次更新全部完成後量。上方靜態預測圖用同一次保存的報告重畫，另加配對標示；框座標與分數沒有更改。

    小框寬高差約 3、5.4 pixel，除以 64 再平方約 0.002、0.007；fine 中心偏 1 pixel 則是格內比例差 1/8=0.125，平方約 0.016。因此本例 loss 對寬高偏差較寬容。原版 YOLOv3 用相對 anchor 的 ln 比例，官方框 loss 還乘 `2−w·h`（GT 寬高除以全圖的比例；框越小權重越大），不是本例這套寬高 loss。〈[IoU 類 loss](11-iou-loss.md)〉則直接用 IoU 度量定位誤差。

??? example "重跑 40 步與報告來源"

    在 repository 根目錄執行 `python scripts/run_multiscale_learning.py`；Colab 則先執行環境格，再另開 code cell 執行 `!python scripts/run_multiscale_learning.py`。輸出在 `artifacts/runs/multiscale-learning/`：`report.json` 保存逐步 loss、指標、框與分數，`learning.svg` 是本次生成圖。

    ```python
    from IPython.display import SVG, display
    display(SVG(filename="artifacts/runs/multiscale-learning/learning.svg"))
    ```

    notebook 的〈可選：兩尺度 40 步與實際預測〉也有這兩步。不同機器的末位可能不同，以自己報告為準；自行重跑不必加 `--record`。這個選項才會改寫網站的 JSON 與原生成圖，不會重畫本頁新增的靜態配對圖。

    本頁使用的[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/10-multiscale-learning.json)保存 loss、權重變化、AP、`prediction` 框與分數。IoU 由同份框座標算出；GT 來自同次腳本固定的兩個框。

??? note "與原版 YOLOv3 的差異"

    [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) 的第 2.3 節是三尺度預測；第 2.5 節 multi-scale training 則是隨機改變輸入尺寸，本例沒有做。

    - **尺度與 anchor**：原版在三種格子大小上預測（stride 32、16、8），每種各配 3 個 anchor（共 9 個），每格預測 3 個框。本節只有 stride 16、8 兩種，每格 1 個框，不用 anchor。
    - **誰負責哪個尺度**：原版用尺寸聚類（k-means，見第 9 章〈[尺寸聚類](09-anchor-clustering.md)〉）得到 9 個 anchor，依大小平均分給 3 個尺度，最小的 3 個在最細的尺度。每個 GT（真值框）只交給 9 個 anchor 中尺寸 IoU 最高的那一個（尺寸 IoU 見第 9 章：把中心疊在一起、只比寬高），由那個 anchor 所在的尺度、GT 中心所在的那一格負責。所以物件由哪個尺度負責，是經由 anchor 間接決定的，不是直接設一個尺寸門檻。本節則是人工把兩個物件各自指定給一個 head。
    - **ignore**：原版對「不是最佳、但與某個 GT 重疊超過門檻」的預測設 ignore，不計 objectness loss；官方程式比的是解碼後的預測框（含位置）和 GT 的一般 IoU。論文寫的門檻是 0.5，官方 Darknet 的 cfg/yolov3.cfg 設為 ignore_thresh = .7。本節沒有 ignore，正格以外全是負格，後果見前文〈兩個 head 的實際連線〉。第 9 章的 ignore 是本書自訂的教學規則（同格另一槽的尺寸 IoU 大於 0.2），和原版不同。
    - **類別分數**：本節兩類用 softmax，兩類機率加起來是 1，一個框只能屬於一類（單標籤）。原版對每個類別各做 sigmoid（也叫 logistic 函數），用二元交叉熵（BCE）訓練，各類分數互相獨立，所以同一個框可以同時是 Woman（女人）和 Person（人），這是論文舉的例子。兩種 head 輸出的意義不同，不能直接互換。
    - **特徵融合**：原版較細的尺度，會把深層、格子較少的特徵放大 2 倍（upsample），再和淺層、格子較多的特徵沿 channel 接起來（concat），然後才預測。這叫特徵融合，本節不做，第 11 章才做。
    - **主幹網路**：原版的主幹網路（backbone，負責抽取特徵的前段網路）是 Darknet-53（論文說它有 53 個卷積層，這包含 ImageNet 分類用的最後一層；YOLOv3 偵測時去掉分類層，用其餘 52 個）。本節不用，改用自己寫的小網路 TwoScale。
    - **推論的 NMS**：原版官方程式把三個尺度的框收齊後，只做一次逐類別的 NMS。本節重用第 7 章的 `decode_grid`，各尺度內會先各做一次 NMS，合併後再做一次。框連鎖重疊時（A 與 B、B 與 C 的 IoU 都超過門檻，A 與 C 沒有），兩種做法留下的框可能不同。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-05 在 INTEL(R) XEON(R) PLATINUM 8573C（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/10-multiscale.json)

??? example "展開本次實際輸出"

    ```text
    cross-scale duplicate boxes before/after class-wise NMS 2 -> 1
    stride16 small target [0.5625, 0.5625, 0.125, 0.125]
    stride8 small target [0.125, 0.125, 0.125, 0.125]
    head shapes (1, 8, 8, 7) (1, 4, 4, 7) candidates before score/NMS 80
    both heads received gradients; one update; scale encode/decode checked
    ```

<!-- curriculum-evidence:end -->
