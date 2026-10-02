# 16.1 YOLO26 DFL-free：移除 bins，仍要把框學好

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/16-dfl-free.ipynb) · 原始碼：`lesson_cases/16-dfl-free.py`

前置是[四邊距離](12-anchor-free.md)和[DFL期待值](12-dfl.md)。ltrb是點到左／上／右／下邊界的距離，本例一格8pixel；xyxy是左上x,y與右下x,y。DFL每條邊輸出K個logits，再softmax、加權成距離；這條路能提供分佈監督，也增加head輸出與匯出操作。若部署需求重視簡潔，可以直接預測四個距離嗎？本節比較表示與範圍，避免把「移除DFL」誤認成「沒有定位loss」。

歷史機制已核對YOLO26官方配置的`reg_max:1`，以及Detect在`reg_max>1`時才建立DFL，否則使用identity。本次由第12章的距離分佈回歸，改為四個直接回歸數值；小實驗使用Smooth L1教學目標，而非完整官方IoU與正規化L1組合。實驗後可將直接回歸保留為部署候選，但尚未證明與DFL精度相同。

![候選點與四條距離的解碼](../assets/diagrams/16-distance-decode.svg)

## 這裡的1不是隻剩一種距離

K=16的DFL四邊raw輸出為64個logits，分成`[4,16]`，bin座標為0到15。期待值只能落在`[0,15]`格。`reg_max=1`的官方分支跳過DFL，不是對單一bin做softmax得到永遠0；它把四個數直接交給距離解碼。

本例真值為`ltrb=[1.25,2.5,18,3]`格，stride8時對應`[10,20,144,24]`畫素。第三條邊18格超過16bin的最大期待距離15格。連續回歸沒有這個有限bin上界，程式能學到18。這只示範表示範圍，不代表所有大物件都要求一個stride8候選：多尺度DFL模型也能使用更大的stride。

兩種表示最後都走四邊距離解碼。候選點像素座標`(80,80)`、stride8時：`x1=80−1.25×8=70`、`y1=80−2.5×8=60`、`x2=80+18×8=224`、`y2=80+3×8=104`，得到xyxy`[70,60,224,104]`。本例只研究座標，沒有對應的輸入圖片。程式也把學得距離解碼，逐值assert這個框。

對1.25格，DFL監督bin1與bin2，權重.75／.25；期待值是按機率加權平均，`1×.75+2×.25=1.25`。直接回歸則讓一個數值接近1.25。案例另印出未訓練的16bin logits全0，因此每bin機率`1/16=.0625`、期待值7.5格；它不是已訓練的DFL對照，不可用這個7.5和direct結果比較精度。

```python
distance = nn.Parameter(torch.zeros(1, 4))
loss = F.smooth_l1_loss(distance, target)
loss.backward()
optimizer.step()
```

直接數值不一定使用sigmoid或softplus。現行官方`reg_max=1`距離分支沒有DFL期待值的非負、有限範圍約束，可表達帶符號距離；框仍須透過loss學成合理幾何。第12.1節為教學方便使用softplus，是另一種設計選擇，不能說YOLO26也採用它。當小物件候選資格擴張到GT外部時，這個區別尤其值得留意。

## 真正測一次學習與輸出大小

本例把四個自由參數從0擬合到一組target，沒有feature輸入，也沒有訓練CNN head；它只能證明表示可涵蓋18格，不能判斷模型定位精度。PyTorch預設Smooth L1的beta=1：誤差絕對值≥1時單項loss為`|error|−.5`，小於1時為`.5×error²`。初始四項是`[.75,2,17.5,2.5]`，平均為5.6875；四項都在大誤差區，每個distance的梯度為−.25。lr=.5的第一步各增加.125，程式印出並assert。

總共更新300次後loss接近0，四邊接近`[1.25,2.5,18,3]`。大誤差區梯度幅度有限，進入小誤差區後隨誤差縮小，所以18格需要較多步才靠近目標；這是自由參數的優化觀察，不是版本的收斂保證。

再固定B=1、P=100候選、C=2類別，並假設只比較raw框與類別輸出：

| 表示 | 每候選數值 | 全部數值 | float32 bytes |
| --- | --- | --- | --- |
| DFL，K=16 | `4×16+2=66` | 6,600 | 26,400 |
| 直接四邊 | `4+2=6` | 600 | 2,400 |

這是raw tensor大小，不是完整模型記憶體、參數或延遲。若最後兩路head的hidden寬度不同，卷積成本也會變；本節沒有假裝只改輸出就測得整版效能提升。

執行`PYTHONPATH=. python lesson_cases/16-dfl-free.py`，檢查direct學得18、finite-bin上界15、6600／600與26400／2400，以及loss低於`1e-6`。純CPU，沒有模型下載。本例不拿超出bin範圍的target去執行DFL cross entropy，因為那樣會違反索引契約，並非公平的loss比較。

## 官方訓練仍保留什麼

查覈版本的`BboxLoss`在DFL-free路徑仍計算CIoU，另外把預測和target距離乘stride、依影像寬高正規化，再計算L1。原始程式中的變數仍叫`loss_dfl`，但此分支的實際內容是L1；閱讀程式要看運算，不只看名稱。權重、正樣本品質與assignment還會影響訓練，不能刪掉所有box loss後期待框自己變準。

收益是少了bins logits、softmax、期待值與距離上界，推論圖較容易處理；代價是失去相鄰bin分佈監督，要確認直接回歸的尺度、梯度與定位表現。少了一個運算並不自動意味各裝置都更快，應交給第20章的實際匯出與延遲比對。

常見錯誤是對K=1仍做softmax、把reg_max當四邊最大值、把DFL-free等同NMS-free，以及只減少輸出channels卻仍用舊DFL loss reshape。移除DFL是框表示變更；是否省NMS是監督與推論路徑變更，兩項要分開驗證。

自主練習先做紙筆範圍判斷：若右邊改成8格、K仍16，兩種表示都可涵蓋它，不能再以「DFL放不下」支持直接回歸。若修改程式target，必須同步修改固定初始loss、最大距離超界與解碼答案assert，不沿用18格的檢查。再用已加進案例的帶符號例子`[-1,2,3,4]`，同一點`(80,80)`、stride8解碼成`[88,64,104,112]`。候選點在框左側，卻仍能成合法框；程式同時確認x1<x2、y1<y2。這不等於所有任意負距離都合法，需逐框檢查幾何。

來源查覈：2026-10-02。[YOLO26官方配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml)、[Detect的DFL/identity分支](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)、[DFL-free BboxLoss](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)。

distance低於target時增加它會降低loss，因此梯度為負；四邊mean使每項係數除4，SGD再以舊值−lr×梯度更新。right改8的練習同步四處：target[0,2]改8；initial_terms預期第3項改7.5；原範圍assert改`target.max()<=k-1`（code使用小寫k）；decoded預期x2改144。mean會自動變3.1875，不用改計算式。新的target在DFL範圍內，但原題18格不可。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-dfl-free.json)

??? example "展開本次實際輸出"

    ```text
    first Smooth L1 gradient: [[-0.25, -0.25, -0.25, -0.25]]
    first SGD distance: [[0.125, 0.125, 0.125, 0.125]]
    direct learned distances: [[1.25, 2.5, 18.0, 3.0]]
    Smooth L1 5.687500 -> 0.000000
    decoded direct box pixels: [[70.0, 60.0, 224.0, 104.0]]
    signed-distance example box pixels: [[88.0, 64.0, 104.0, 112.0]]
    untrained DFL probability per bin: [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
    untrained DFL expected distances: [[7.5, 7.5, 7.5, 7.5]]
    DFL target for 1.25: bins 1/2 weights .75/.25; direct target is the value 1.25
    finite-bin expectation range: [0, 15] ; direct target reaches 18.0
    raw head values DFL / direct: 6600 600
    float32 raw head bytes: 26400 2400
    DFL-free still requires box supervision; this experiment uses Smooth L1
    ```

<!-- curriculum-evidence:end -->
