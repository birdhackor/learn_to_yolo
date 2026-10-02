# 尺寸聚類：先驗由哪一份資料決定

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/09-anchor-clustering.ipynb){ .md-button }

前置：[anchor 參數化](09-anchors.md)。這次只改 anchor 尺寸的選法，保留候選數 A=2、center decode與模型分支。問題是手填16×16，若資料多為8×8小方形和32×16寬矩形，是否浪費回歸力氣？

歷史來源：[YOLO9000](https://arxiv.org/abs/1612.08242) 的 dimension clusters，使用距離 `1−IoU`，避免普通歐氏距離偏重大框。本章用6個已知寬高、固定初始中心與mean更新，展示一個k-means風格程式；不是重現原論文資料、k值、隨機重啟或效果。本次只觀察形狀覆蓋，不測偵測AP；實驗後是否保留需靠held-out模型對照。

## IoU 在這裡比較什麼

輸入是訓練框經同一前處理後的 `wh[N,2]`，單位pixel，這次沒有位置資訊。把兩個框中心重合，交集寬高為各維的min；IoU=`交集/(面積1+面積2−交集)`。8×8與16×16得到64/256=.25，所以距離=.75；8×8與9×8得到64/72=.888889，距離=.111111。

同形狀同倍縮放後，IoU不變。8×8對9×8、80×80對90×80都是.888889；普通距離卻分別是1與10。IoU距離更關注相對尺寸差異，但它只描述形狀是否接近，不含物件顏色、類別與定位能力。

## 手做一次分羣

6個train尺寸為 `[8,8]、[9,8]、[8,9]、[32,16]、[30,16]、[32,18]`。固定初始anchor是第一個與最後一個，避免每次執行看到不同答案。

這裡assignment只決定每個寬高樣本的羣編號，產出的anchor尺寸再交給detector；訓練時GT（標註真值）與候選的責任配對是另一個步驟，羣編號不是正／負／ignore標籤。

1. 對每個尺寸計算到兩個anchor的1−IoU，選距離小者。
2. 前三個歸羣0、後三個歸羣1。
3. 更新羣0平均寬高為`[25/3,25/3]=[8.3333,8.3333]`；羣1為`[94/3,50/3]=[31.3333,16.6667]`。
4. 再分配與更新，直到中心不再變或到10輪。

本例以逐維mean更新作為容易手算的啟發式。因為距離是1−IoU，mean不保證最小化固定分羣內的距離總和，也不保證每輪目標下降。逐維median可作對照，同樣不能直接宣稱是此目標的一般最佳解。真實使用應比較初始值／多次重啟、空羣處理及k選擇；本例空羣保留舊中心，不憑空生成一個先驗。

```python
assignment = (1 - size_iou(sizes, anchors)).argmin(dim=1)
for k in range(len(anchors)):
    selected = sizes[assignment == k]
    if len(selected):
        updated[k] = selected.mean(0)
```

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/09-anchor-clustering.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/09-anchor-clustering.py`。groups應為`[0,0,0,1,1,1]`、anchor為上述分數，train mean best size IoU應比兩個16×16提高。案例還把資料和anchor同時乘2，驗證IoU不變；最後用`[4,40]、[40,4]`模擬新來源，輸出覆蓋變差的數字。這些是尺寸統計，不是AP提升。案例另外以逐維median得到`[8,8]`與`[32,16]`，這六筆的mean best size IoU是.9340，高於mean更新的.9119；仍只表示這六個尺寸更接近先驗。

## 只用 train 聚類的理由

若用validation或test框一起選先驗，配置已偷看評估分佈。先固定來源切分，再把train框按實際resize／letterbox轉到輸入pixel，才收集wh。每個anchor不綁類別；若不同類別的尺寸分佈差很大，可觀察各類覆蓋，但仍需所有候選保有完整class軸。

收益是先驗更貼近已知訓練形狀，較少需要大幅exp修正；代價是資料統計與配置管理，若把羣數k當作每個位置的anchor數A，增加k也會增加偵測候選，並影響後續GT與候選配對的計算量。對新的攝影距離、極端長寬比、不同輸入尺寸，先驗可能不合。訓練尺寸覆蓋高也不保證小物件特徵足夠；第10章將改解析度。

自主對照：median在這六筆的覆蓋較高，是否足以選它或宣稱AP改善？答案：都不足；需在固定獨立資料上訓練並評估模型，不能讓train尺寸統計替代泛化。

常見錯誤：拿xyxy四個座標直接聚類；收集原圖pixel卻在64輸入使用；把held-out標註當免費資訊；把mean best IoU稱為mAP。自主練習：輸入尺寸64改128且保持同樣內容比例，anchor怎麼改？答案：wh一併乘2，尺寸IoU保持一致。只改影像而不改anchor，尺寸回歸起點便不同了。

本節刻意用兩個相同16×16尺寸作弱基線，其mean best size IoU=.3817，**不是上一節配置的結果**。上一節的16×16與8×8在相同六筆尺寸上是.7093。mean best size IoU是每筆wh先找IoU最大的anchor，再把六個最大值平均；逐維median則是每個維度各自排序取中間值，例如寬[8,9,8]排序[8,8,9]取8。這些是尺寸覆蓋，不是AP。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/09-anchor-clustering.json)

??? example "展開本次實際輸出"

    ```text
    groups [0, 0, 0, 1, 1, 1] anchors pixel [[8.333333015441895, 8.333333015441895], [31.33333396911621, 16.66666603088379]]
    train mean best size IoU 0.3817 -> 0.9119
    median anchors [[8.0, 8.0], [32.0, 16.0]] train coverage 0.934
    new-source shape coverage 0.1975 not detector AP
    ```

<!-- curriculum-evidence:end -->
