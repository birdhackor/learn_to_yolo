# 12.1 Anchor-free：從候選點量出四條邊

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/12-anchor-free.ipynb) · 實驗原始碼：`lesson_cases/12-anchor-free.py`

讀過框座標、stride 和第 7 章 grid detector 即可開始。問題是：一個偵測位置要先選「寬 20、高 10」的 anchor，再學偏移嗎？如果資料的長寬比改變，這組尺寸先驗是否仍合適？本節換掉框的表示方式，保留候選位置和需要有人負責物件的規則。

歷史機制是 YOLOv8 相關設計以特徵格點和四邊距離回歸框，不需預先給 anchor 尺寸。起始分支是直接框回歸的 MiniYOLO；本節簡化成單一候選點、四個正距離與 Smooth L1；實驗只驗證 encode、decode 與距離可學性，不是重現整套 YOLOv8。下一節可以保留此表示，DFL 則在另一節獨立加入。

## 一個物件的數字旅程

輸入影像是 `64×64` 畫素，特徵圖 `8×8`，stride 為 8 畫素／格。真實密集 head 的候選通常由各特徵 cell 中心產生：`((column+.5)×stride,(row+.5)×stride)`，格點 `(2,2)` 的中心是 `(20,20)`。本例刻意用人工參考點 `(24,24)` 來方便手算；它不是這份 stride8 格網的 cell 中心，這個實驗只研究距離表示。點位置不等於預測框中心，框可以向四邊不對稱延伸。

真值框 `xyxy=[12,16,40,36]`，從點到左、上、右、下的畫素距離為 `[12,8,16,12]`。除以 stride 得到特徵格單位 `ltrb=[1.5,1,2,1.5]`。距離不必是整數，格子是取樣位置，不是框邊的量化網格。

| 欄位 | shape | 意義與單位 |
| --- | --- | --- |
| candidate points | `[P,2]` | 原影象素 `x,y`，例中 `P=1` |
| distance output | `[B,P,4]` | `left,top,right,bottom`，特徵格單位 |
| decoded boxes | `[B,P,4]` | 原影象素 `x1,y1,x2,y2` |
| class logits | `[B,P,C]` | 每候選、每類別的分類輸出，本次不加入 |

其中 B 是圖片數、P 是候選點數、C 是類別數。表格列出完整 head 契約；本例只有一張圖，程式省去 batch 軸，距離與框實際 shape 都是 `[1,4]`，這個 1 是 P，不是四邊中的一項。

解碼先把距離乘回 stride，再由點減去左上距離、加上右下距離：`x1=24−1.5×8=12`、`y1=24−1×8=16`、`x2=24+2×8=40`、`y2=24+1.5×8=36`。框與點都要在同一座標系；若點已是畫素，不能再乘一次 stride。

## 為何仍要 assignment

移動候選點到 `(48,24)`，它在真值右側。右邊距離會變成 `40−48=−8`，與「正距離」契約衝突。不是每個點都該學同一個框；要挑出適合負責的點，其他點學背景。有些設計限制點在框內，有些加入中心區或動態品質條件。去掉尺寸 anchor 不會去掉這個選擇問題。

早期 grid MiniYOLO 同樣沒有 anchor 尺寸，但其輸出為 cell 內中心 offset 加全圖正規化寬高，而且第一版每格只負責一個中心。這裡改成點到四邊距離，實際 YOLOv8 還使用多尺度候選、獨立分類分支和動態 assignment。兩者共享一項特徵，不能因此當成同一個 detector。

## 真的更新四個距離

程式直接把四個raw數值做成nn.Parameter，沒有圖片或CNN。softplus(x)=ln(1+exp(x))把任意實數轉成正距離，softplus(0)≈.693；Smooth L1對小誤差用平方、大誤差用線性，這裡比較的是格單位的四邊距離。程式讓四個可學 logits 從 0 開始，使用 `softplus` 保持正距離。這是本章的連續回歸選擇；YOLOv8 的距離分布輸出會在 DFL 節介紹。

```python
distance = torch.cat((point - gt[:, :2], gt[:, 2:] - point), -1) / 8
prediction = F.softplus(raw)
loss = F.smooth_l1_loss(prediction, distance)
loss.backward()
optimizer.step()
```

`softplus(0)≈0.693`，初始預測四邊相同，真值卻不對稱，因此 loss 有非零梯度。80 次 CPU 更新後，檢查 loss 小於初值的百分之一，再解碼到畫素；這比「程式印出 `[1,4]`」多驗證了一條從真值到參數再到框的路。

執行 `PYTHONPATH=. python lesson_cases/12-anchor-free.py`。必須看到 target `[1.5,1.0,2.0,1.5]`、解碼真值 `[12,16,40,36]`、loss 下降，以及框外點需要負距離的訊息。最後學得的框容許小誤差，assertion 的重點是 loss 下降與真值往返完全一致。本案例無 GPU、資料下載或訓練權重。

## 收益、代價與容易搞錯的地方

省掉 anchor 尺寸清單後，不需因長寬比分佈變化重新聚類先驗；代價轉移到候選密度、距離範圍與正樣本選擇。stride 太大時，小物件可能沒有合適點。密集候選仍可對同一物件輸出多框，推論的非極大值抑制（NMS）才處理這類重複。anchor 尺寸屬於框表示與候選設計，NMS 屬於推論篩選階段，因此 anchor-free 並不保證 NMS-free。這個小實驗沒有測 AP，也不能宣稱換表示一定更準。

常見錯誤是把 `ltrb` 當成 `xyxy`、把特徵格單位當畫素、交換左右順序，或忘記框外候選不可直接使用正距離 target。另一種錯誤是把 point 稱為「沒有 anchor」後又忽略 source 中 `anchor_points` 的名稱；那裡的 anchor 指參考點，不一定指預設尺寸框。

自主練習：保持框不變，把點改成 `(20,20)`，stride 改為 4。先算距離，再修改程式。答案為畫素 `[8,4,20,16]`、格單位 `[2,1,5,4]`，解碼仍是同一框。請同步修改 `main()` 的 point、target 距離除數、預期距離 assertion，以及每個 `decode(...,8)` 的 stride，或先集中成一個 stride 變數；大距離可能需要更多步才能透過原本嚴格的下降門檻。若只改 stride 卻沿用舊 distance，框就會縮小；這是單位錯誤，不是模型能力問題。

來源查覈：2026-10-02。參見 [Ultralytics Detect 的距離解碼與分支](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)、[bbox2dist／dist2bbox 與候選點](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py)。本章的 softplus 小模型是教學選擇。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-anchor-free.json)

??? example "展開本次實際輸出"

    ```text
    target ltrb in feature cells: [[1.5, 1.0, 2.0, 1.5]]
    decoded target pixels: [[12.0, 16.0, 40.0, 36.0]]
    distance loss 0.376236 -> 0.000012
    learned box pixels: [[12.029999732971191, 16.059999465942383, 39.9900016784668, 35.970001220703125]]
    outside point requires a negative distance: assignment is still necessary
    ```

<!-- curriculum-evidence:end -->
