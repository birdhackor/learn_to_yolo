# 12.4 DFL：把一條邊距離學成分佈

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/12-dfl.ipynb) · 原始碼：`lesson_cases/12-dfl.py`

前置是 softmax、cross entropy、四邊距離和反向傳播。連續回歸直接輸出一個距離，能否改成「它比較接近 1 格，但也有一部分落在 2 格」？Distribution Focal Loss，簡稱 DFL，為非整數距離提供相鄰兩個 bin 的監督，再用分佈期待值還原距離。本節會算 loss、梯度和解碼，避免只把多個 channel 叫成「分佈」。

歷史機制來自 Generalized Focal Loss，YOLOv8 等模型使用以分佈表示的四邊回歸。起始分支是 anchor-free distances；本次只用一條邊、四個 bin `0,1,2,3`。實際 detector 四邊各有一份分佈，還要框 IoU loss；本例不包含分類或完整訓練。實驗後是否保留 DFL，應由定位品質和部署成本決定。

## 1.25 格不是第 1 類

stride 為 8 畫素／格，真值左邊距離為 10 畫素，換成 `y=1.25` 格。向下取整為 `l=1`，右鄰 bin 為 `r=2`。兩個 target 權重為 `r−y=.75` 與 `y−l=.25`，其加權座標 `.75×1+.25×2=1.25`。

logits shape 為 `[N,K]=[1,4]`，softmax 後得到 `p0,p1,p2,p3`，四個值相加為 1。loss 是 `−.75 log(p1)−.25 log(p2)`；不是先把 1.25 四捨五入成 1，也不是隻對期待值做平方誤差。四邊完整版 raw head 是 `[B,P,4K]`，reshape 成 `[B,P,4,K]`，softmax 要沿最後的 bin 軸做。

```python
left = target.floor().long()
right = left + 1
wr = target - left
loss = ((1-wr) * cross_entropy(logits, left, reduction='none')
        + wr * cross_entropy(logits, right, reduction='none')).mean()
distance = (logits.softmax(-1) * torch.arange(K)).sum(-1)
```

初始 logits 全 0，機率各 .25，期待距離是 1.5 格，DFL 是 `log(4)≈1.386294`。加權 cross entropy 的 logit gradient 為「預測機率−target 分佈」，得到 `[.25,−.50,0,.25]`。bin2 的梯度此刻是 0，因為它目前 .25 恰好符合 target；這不代表 bin2 永遠不更新，softmax 的機率會隨其他 logits 變動。

把 target 寫成四個槽就是 `[0,.75,.25,0]`；初始預測 `[.25,.25,.25,.25]` 逐槽減掉它，便得到上面的梯度。

## Loss 和期待值各負責什麼

程式反傳初始 loss，逐值 assert 上述梯度，再做 400 次 SGD。最後 bin1 接近 .75、bin2 接近 .25，其他 bins 接近零，期待值在 1.25 格附近。乘 stride 後接近 10 畫素。softmax 有限 logits 不能讓其他 bins 完全等於零，因此沒有要求期待值逐位等於真值。

兩份分佈 `[0,.75,.25,0]` 與 `[.375,0,.625,0]` 的期待值都為 1.25。只看期待距離無法分辨它們；DFL 偏好相鄰 bins 的 target，提供額外監督。在完整 detector，IoU loss 透過期待距離、解碼框來衡量整框幾何，DFL 則約束各邊的分佈。不要把分佈寬度直接宣稱成校準好的不確定性：這需要另外驗證。

執行 `PYTHONPATH=. python lesson_cases/12-dfl.py`，檢查 uniform expectation `1.50`、DFL `1.386294`、梯度 `[.25,−.5,0,.25]`，以及學得期待值距離 1.25 小於 .02 格。程式真的做 backward 與 step，只需 CPU。

## 距離範圍是明確成本

四個 bins 的期待距離在 `[0,3]` 格；本例 DFL target 需嚴格小於 3，纔能有右鄰 bin。官方實作對邊界採 clamp，以免右索引越界。本例用 assertion 讓讀者直接看見範圍契約。K=16、stride8 時期待距離最多 120 畫素；多尺度與候選選擇會影響是否覆蓋大框，不能用一個小尺度代表整個模型。

對兩類分類、100 個候選，直接四邊輸出需要 `100×(4+2)=600` 個數；K=16 的 DFL head 需要 `100×(64+2)=6600` 個數。這是 raw 輸出大小，還未算中間卷積。分佈表示有更細緻的 supervision，也增加 logits、softmax、期待值操作與匯出路徑。這個小例子沒有比較真實 AP，不能斷言額外成本一定值得。

常見錯誤是把 softmax 放在候選軸、忘記四邊各自歸一化、把DFL誤當物件類別的分類loss、對 bins 再乘兩次 stride，或把 reg_max 當成「最大 bin 數值」而漏掉 bins 從 0 起。索引和值的差一要在實作旁標清楚。DFL確實是距離bin的加權cross entropy，監督的是距離刻度，不是紅矩形／藍矩形。兩份CE要先reduction=none，逐樣本乘相鄰bin權重後才mean；先將CE平均會把不同target權重混在一起。

自主練習：target 改成 2.6 格、K仍為4。答案是 bin2 權重 .4、bin3 權重 .6；初始梯度 `[.25,.25,−.15,−.35]`。若 target 改成 3.2，應擴增 K 或重新設計尺度，不能讓程式讀不存在的 bin4。再說明為何「期待值誤差很小」不足以驗證 DFL 正確：兩份不同分佈可以有同樣期待值。

來源查覈：2026-10-02。[Generalized Focal Loss 原論文](https://arxiv.org/abs/2006.04388)、[Ultralytics DFLoss 的相鄰 bin 加權](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)、[DFL 期待值模組](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py)。

bin是距離刻度，本例0／1／2／3分別代表0／1／2／3格。target=2.6的練習需同步三處：wanted_grad改`[[.25,.25,-.15,-.35]]`；期待值斷言改與2.6比較、容差.02；概率改檢查bin2與bin3接近.4／.6（各誤差<.02）。最後展示「兩分佈同期待值1.25」的a/b是獨立人工例，與本題2.6無關，可保留原值，不把它改成訓練target。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-dfl.json)

??? example "展開本次實際輸出"

    ```text
    uniform expectation=1.50; DFL=1.386294
    initial logit gradient: [[0.25, -0.5, 0.0, 0.25]]
    learned bin probabilities: [[0.0024999999441206455, 0.7475000023841858, 0.24740000069141388, 0.0024999999441206455]]
    learned expectation=1.2500 cells = 9.9999 pixels at stride 8
    different distributions can share expectation=1.25
    ```

<!-- curriculum-evidence:end -->
