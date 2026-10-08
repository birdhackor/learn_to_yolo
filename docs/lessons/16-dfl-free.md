# 16.1 YOLO26 DFL-free：移除 bins，仍要把框學好

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/16-dfl-free.ipynb){ .md-button }

不論用多少條特徵路徑，偵測 head 最後仍要把候選變成框。[12.4 的 DFL](12-dfl.md) 讓每條邊輸出一組距離刻度的分數，用相鄰刻度監督，再取期望距離；這也讓最後輸出與解碼多了工作。若部署希望減少這些輸出和運算，可以改回每條邊一個數嗎？又要用什麼訊號把框教好？

Ultralytics 的 YOLO26 採用 DFL-free：直接輸出距離，跳過 DFL 的 bins、softmax 和期望值。先沿用同一個候選點與四邊解碼，比較改的是哪一份表示，再看小實驗和官方 loss 各負責什麼。

## 框不變，距離如何表示可以改

候選點是 (84,84) 畫素，stride=8；它是從 0 起算第 10 欄、第 10 列的格心，`((10+.5)×8,(10+.5)×8)`。真值四邊距離 ltrb=`[1.25,2.5,18,3]` 格，一格 8 畫素，換成 `[10,20,144,24]` 畫素。

沿用 [12.1](12-anchor-free.md) 的公式，左、上從點減去距離，右、下則加上：

\[
[x_1,y_1,x_2,y_2]
=[84-1.25×8,\ 84-2.5×8,\ 84+18×8,\ 84+3×8]
=[74,64,228,108].
\]

![候選點與四條距離的解碼](../assets/diagrams/16-distance-decode.svg)

紅點是候選點，四個雙箭頭是 l、t、r、b。藍框右邊 x=228；橘色虛線 x=204 是本例 K=16 的 DFL 向右可表示的最遠位置。兩種表示仍要給這套解碼四個距離，差別發生在距離怎麼得到。

DFL 每邊輸出 K=16 個 logits，對應 bin0～15，即 0～15 格。softmax 得到機率 p₀～p₁₅，再取 `0×p₀+1×p₁+…+15×p₁₅`。這是 0～15 的加權平均，所以一定落在 `[0,15]`。左邊 1.25 可以由 bin1、bin2 的 0.75、0.25 還原：`1×.75+2×.25=1.25`；右邊 18 則不在這組刻度內。

直接回歸每邊只輸出一個數，把它本身交給解碼。它保留了「四邊距離表示同一個框」，移除了由有限 bins 帶來的範圍上限與分佈監督。相應地，距離的誤差要由其他框 loss 教，不再由相鄰 bins 的 target 教。

這個 18 格例子是表示範圍對照，不能說 DFL 畫不出右側 144 畫素：15×stride 在 stride 8／16／32 是 120／240／480 畫素，同一個 144 在 stride 32 只有 4.5 格，DFL 放得下。官方也有這三種尺度；這裡故意只看 stride 8，讓上限發揮作用。超範圍的 DFL target 若直接讀 bin18、19 會越界；官方先截到 14.99，見 12.4，所以本例不拿 18 格的 DFL loss 當公平品質對照。

## reg_max=1 是繞過 DFL，不是在 bin0 上做 softmax

官方參數 `reg_max` 是每邊 bin 數，即前面的 K，名字不是「最大距離」。設定檔寫 `reg_max: 1` 時，`Detect` 不建立 DFL，改放 `nn.Identity`，讓四個原始值直接通過。若誤把 K=1 當成只用一個 bin：唯一機率永遠為 1，唯一 bin0 代表 0，四條距離就永遠全 0，框縮成一點。這不是官方做法。

同樣是直接回歸，12.1 用 softplus 保證正距離；YOLO26 不加 softplus，因此可輸出帶符號距離。負值表示框邊在點的另一側。例如 l=−1 格時，x1=`84−(−1)×8=92`，點的 x=84 在框左邊之外。

這在候選點可被選到原 GT 外時用得到：YOLO26 選候選會暫時擴張很小的框，框外點也可能有資格，回歸答案卻仍是原框。至少一邊的 target 便可能為負；只有正值的 softplus 或有限非負 bins 表示不了這種關係。[16.3](16-training.md)會用格心圖展開候選資格規則；此處需要的關係是「點在框外，仍要能回到原框」。

可為負不等於任何四個數都能形成框。框寬是 `(l+r)×stride`、框高是 `(t+b)×stride`，所以合法框需要 l+r>0、t+b>0。失去非負和 bin 上限後，距離與框的合理性要靠學習和使用時檢查維持；頁末練習用正負距離各解一次。

## 四個自由參數，真的能走到這組距離嗎？

把問題縮到四個數的優化：沒有圖片、CNN 或特徵，每個距離自己是一個參數，從 0 起步，不加 softplus。這和學習由圖片找框不同；它讓我們看清直接表示接 loss、反傳與解碼是否能完成。小實驗沿用已教的 Smooth L1，官方 loss 另在下面說明。

``` { .python data-excerpt="lesson_cases/16-dfl-free.py" }
target = torch.tensor([[1.25, 2.5, 18., 3.]])  # 四邊真值 [l,t,r,b]，單位：格
distance = nn.Parameter(torch.zeros(1, 4))     # 1 個候選點 × 4 條邊，從 0 開始；要學的就是這四個數
optimizer = torch.optim.SGD([distance], lr=.5)
...
for step in range(300):                        # 更新 300 次
    optimizer.zero_grad()                      # 清掉上一步的梯度，否則會累加
    loss = F.smooth_l1_loss(distance, target)  # 四邊各算 Smooth L1，再取平均
    loss.backward()                            # 算出 loss 對四個數的梯度
    ...
    optimizer.step()                           # 新值 = 舊值 - lr × 梯度
    ...
```

Smooth L1 預設 beta=1。每邊誤差 e=distance−target，|e|≥1 時 loss=`|e|−0.5`，|e|<1 時是 `0.5e²`，最後四邊平均。初始誤差 `[−1.25,−2.5,−18,−3]`，各項 loss `[0.75,2,17.5,2.5]`，平均 5.6875。

四邊都在線性段，單項對 distance 的梯度是 −1；平均後各為 −0.25。SGD 學習率 0.5，第一步各從 0 變成 `0−.5×(−.25)=.125`。程式用斷言核對這個梯度與更新，而不只看最後 loss 下降。

300 次更新後，距離接近 `[1.25,2.5,18,3]`，loss 低於 1e−6；同一套解碼得到 `[74,64,228,108]`，也逐值核對。右邊的 18 需要較多步：線性段每步只前進 0.125 格，136 步才把誤差從 18 降到 1，第 137 步後進平方段；左邊 1.25 只需 3 步。進平方段後梯度是 e/4，每次誤差乘 7/8。這解釋本例的收斂，不是完整 YOLO26 的訓練速度。

程式另用一組未訓練的 DFL 核對期望公式：16 個 logits 全 0，每 bin 機率 1/16，距離為 0～15 的平均 7.5。它沒有被訓練，不能拿 7.5 和直接回歸學到的值比較準確度。

### 輸出減了哪些數值？

固定 B=1、P=100 候選、C=2 類別，只比較最後 raw（未轉換）框與類別輸出：

| 表示 | 每候選數值 | 全部數值 | float32 bytes，每數 4 bytes |
| --- | --- | --- | --- |
| DFL，K=16 | `4×16+2=66` | 6,600 | 26,400 |
| 直接回歸 | `4+2=6` | 600 | 2,400 |

這是移除每邊 16 logits 的直接收益。softmax、乘刻度、加總的步驟也被拿掉，匯出工具與裝置少處理幾種運算。但表格不是全模型記憶體或參數數；head 中間層寬度若也不同，卷積成本會另變，實際延遲要在目標裝置量測。

執行 `PYTHONPATH=. python lesson_cases/16-dfl-free.py`，對照首步 −0.25／0.125、學到的距離與框、6600／600 數值和 26400／2400 bytes。`Smooth L1 5.687500 -> 0.000000` 只印六位小數，不代表數學上精確等於 0；小於 1e−6 由斷言核對。CPU 即可執行。

## 拿掉 DFL，官方仍如何教框？

官方 `BboxLoss` 在 reg_max>1 時計算 CIoU 加 DFL；reg_max=1 時保留 [CIoU](11-iou-loss.md)，把 DFL 換成正規化 L1。CIoU 比較解碼框的重疊、中心與寬高比；L1 直接比較四邊距離，兩者仍提供定位訊號。DFL-free 不是把框 loss 都刪掉。

正規化 L1 先把格距離乘 stride 變成畫素，再讓 l、r 除圖寬，t、b 除圖高，四邊取絕對誤差平均。右邊 18 格×8=144 畫素，若圖是 640×640，正規化 target 就是 144/640=0.225。這讓距離誤差以整圖尺寸為尺度，而非直接混合不同 stride 的格數。

合併全部正樣本時，先乘各正樣本的品質類別 target（[12.3](12-assignment.md) 的 0～1 軟值），加總後除以這些 target 的總和；CIoU 也用相同加權。最後 L1 再乘超參數 `dfl`，預設 1.5。程式中的變數仍叫 `loss_dfl`、超參數仍叫 `dfl`，但此分支真的算 L1，輸出名稱是 `l1_loss`。名稱保留相同位置的設定接口，不能靠名稱認定還在使用 bins。

失去相鄰 bin 監督後，直接回歸的尺度、梯度與定位品質要重新評估。這個四參數實驗沒有官方 CIoU+L1，也沒有訓練 DFL 對照或獨立圖片，所以不能回答「同樣準嗎」或「真實模型快多少」。它回答的是表示能否涵蓋距離、更新能否接到答案，以及 raw 輸出省了多少。

DFL-free 和 NMS-free 也控制不同工作：`reg_max` 改框距離表示，`end2end: True` 建立一對一推論分支；是否省 NMS 仍要靠監督、分數和推論路徑。改成每候選四個距離後，loss 和解碼也必須同步，不能仍把 raw 重排成每邊 16 bins。元素數剛好能整除時甚至可能不報錯，只是把不同候選默默混成一組。

## 自主練習

先自己算，再展開參考答案。第 2 題要改完整程式（Colab 裡那份，或本機的 `lesson_cases/16-dfl-free.py`）。

1. 紙筆題：右邊改成 8 格，K 仍是 16。DFL 放得下嗎？直接回歸呢？這時還能用「DFL 放不下」來支持直接回歸嗎？
2. 把完整程式 `main()` 裡 `target` 的右邊從 18 改成 8。還有哪幾個斷言要跟著改、改成什麼？沒跟著改的斷言會在那一行報錯停下。初始的平均 loss 變成多少？
3. 程式裡另有一個帶符號例子 `[-1,2,3,4]`。用同一點 (84,84)、stride 8 解碼，會得到什麼框？候選點在框的哪一側？再試 `[-3,2,2,4]`，解出的框合法嗎？一般來說，四個距離要滿足什麼條件，框才合法？

??? note "參考答案"

    **第 1 題**：8 格在 0～15 之內，DFL 和直接回歸都放得下。所以這時不能再以「DFL 放不下」支持直接回歸。

    **第 2 題**：連 `target` 在內，要同步改四處，不能沿用 18 格的檢查：

    - `target = torch.tensor([[1.25, 2.5, 18., 3.]])` 的 `18.` 改成 `8.`（也就是 `target[0,2]`）；
    - `initial_terms` 斷言的預期值 `[[.75, 2., 17.5, 2.5]]`，第 3 項 `17.5` 改成 `7.5`；
    - 範圍斷言 `assert expected.max() <= k - 1 and target.max() > k - 1` 的後半，改成 `target.max() <= k - 1`（程式裡的 k 是小寫）；
    - `decoded` 斷言的預期值 `[[74., 64., 228., 108.]]`，x2 的 `228.` 改成 `148.`（84＋8×8＝148）。

    初始平均 loss 會自動變成 (0.75＋2＋7.5＋2.5)/4＝3.1875，計算式不用改。其他斷言不用改：四個誤差仍都小於 −1，第一步梯度仍是 −0.25；300 步後 loss 也仍低於 `1e-6`。新的 target 在 DFL 範圍內，原題的 18 格則不在。

    **第 3 題**：`[-1,2,3,4]` 解碼成 x1＝84−(−1)×8＝92、y1＝84−2×8＝68、x2＝84＋3×8＝108、y2＝84＋4×8＝116，也就是 `[92,68,108,116]`。x1＝92 比候選點的 x＝84 還大，所以候選點在框的左側（框外），卻仍能構成合法框；程式也用斷言確認 `x1<x2`、`y1<y2`。

    但不是任意負距離都合法。框寬＝x2−x1＝(l＋r)×stride，框高＝(t＋b)×stride，所以只要 `l+r>0` 且 `t+b>0`，框就合法（`x1<x2`、`y1<y2`）。反例 `[-3,2,2,4]`：x1＝84＋24＝108、x2＝84＋16＝100，`x1>x2`，不是合法框。

參考來源：[YOLO26 官方配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml)、[Detect 的 DFL/identity 分支](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)、[DFL-free BboxLoss](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)、[超參數 dfl 的預設值](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/default.yaml)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-dfl-free.json)

??? example "展開本次實際輸出"

    ```text
    first Smooth L1 gradient: [[-0.25, -0.25, -0.25, -0.25]]
    first SGD distance: [[0.125, 0.125, 0.125, 0.125]]
    direct learned distances: [[1.25, 2.5, 18.0, 3.0]]
    Smooth L1 5.687500 -> 0.000000
    decoded direct box pixels: [[74.0, 64.0, 228.0, 108.0]]
    signed-distance example box pixels: [[92.0, 68.0, 108.0, 116.0]]
    untrained DFL probability per bin: [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
    untrained DFL expected distances: [[7.5, 7.5, 7.5, 7.5]]
    DFL target for 1.25: bins 1/2 weights .75/.25; direct target is the value 1.25
    finite-bin expectation range: [0, 15] ; direct target reaches 18.0
    raw head values DFL / direct: 6600 600
    float32 raw head bytes: 26400 2400
    DFL-free still requires box supervision; this experiment uses Smooth L1
    ```

<!-- curriculum-evidence:end -->
