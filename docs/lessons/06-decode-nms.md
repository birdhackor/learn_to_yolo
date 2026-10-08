# B.6.1 人工框解碼與 NMS：少一個框不一定更好

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/06-decode-nms.ipynb){ .md-button }

上一節替每格安排了訓練答案。使用時，每格仍會交出 7 個數，卻沒有標註告訴我們哪格是正格：要從全部候選中算出位置、讀取分數，再決定畫出哪些框。同一物件也可能被相鄰格各畫一次，需要清除重複。

這次保留 64×64、4×4 格與兩類的表示，人工設定輸出，讓每一步有可以手算的答案。先追蹤 row1／col2 的一個候選，最後把它和另兩個框一起處理，看看去重究竟能解決什麼、又會留下什麼。

## 格號與四個比例，如何變成圖上的框

每張圖的原始輸出為 `[4,4,7]`，最後一軸依序是 tx、ty、tw、th、objectness logit、兩類 logits。框的原始數字經 sigmoid 變成比例；前兩項是格內中心偏移，後兩項是整圖寬高比例。將它們換回能畫出的 px 框，稱為 **decode（解碼）**。

假設 row1／col2（第 2 列、第 3 欄）的四個 sigmoid 值為 `[0.25,0.75,0.25,0.125]`。每格 16 px，所以中心為

\[
c_x=(2+0.25)\times16=36,\qquad c_y=(1+0.75)\times16=28.
\]

寬高則是 0.25×64=16、0.125×64=8 px。中心左右各 8、上下各 4，得到 xyxy **`[28,24,44,32]`**。

這正是上一節 target 換算的反方向。以 S 表示每軸格數、σ 表示 sigmoid，一般式為

\[
c_x=(\text{col}+\sigma(t_x))\frac{64}{S},\qquad
c_y=(\text{row}+\sigma(t_y))\frac{64}{S},
\]

\[
w=\sigma(t_w)\times64,\qquad h=\sigma(t_h)\times64.
\]

「格號＋偏移」以格為單位，乘格尺寸；寬高以整圖為單位，乘圖尺寸。用反時，寬誤乘 16 會只剩 4 px，縮成原本的 1/S；中心誤乘 64 則會得到 cx=144，跑出畫布。

## 每格都有框，哪些值得留下

只算位置，每個格都會冒出一個框，包括背景格。上一節另教 objectness 判斷有沒有物件，類別分數則判斷是哪類；現在把兩者合成一個 **score（排序分數）**，用來篩選候選：

\[
\text{score}=\sigma(\text{obj})\times
\max_c\operatorname{softmax}(\text{class})_c.
\]

class 是兩類 logits，c 遍歷各類別；softmax 給出加總為 1 的類別比例。最大的比例決定預測類別，再乘 objectness 的 sigmoid 值。本例 row1／col2 的 objectness=0.9、類別比例 `[0.8,0.2]`，所以預測類別 0，score=0.9×0.8=**0.72**。0.9 是 sigmoid 後的值，不是原始 logit。這是本節模型的分數定義，不是所有偵測器的通用公式。

設定 score 門檻 0.25，保留 score≥0.25 的格。本例除了三個人工候選，其餘 13 格的 obj logit 皆為 −12，sigmoid 約 0.000006；兩類 logits 都是 0，各占 0.5，所以 score 約 0.000003，被篩掉。16 格最後留下三個候選。

score 是模型自己提供的排序訊號，**沒有拿 GT 檢查**。高分不保證框對，也不能直接讀成「正確機率」。precision（精確率）則要將交出的框與真值配對後，計算對框占幾成，下一節會做。本例分數都是人工指定，沒有校準成可靠的正確機率。

??? note "什麼是校準（calibration）"

    校準是在大量有真值的資料上做統計：把 score 約 0.8 的框都收集起來，看是否真的大約八成框對了物件；各種分數都對得上，才算校準良好。沒做過這種檢查之前，score 只能用來比較哪個候選比較可能對，不能讀成「有 85.5% 機率正確」。本節的分數是人工指定的，沒有做過這種檢查。


??? note "人工 logits 是怎麼倒推出來的"

    想讓 σ(t)=p，就從 \(\frac{1}{1+e^{-t}}=p\) 解出 \(e^{-t}=\frac{1-p}{p}\)，所以

    \[
    t=\ln\frac{p}{1-p}.
    \]

    例如要 \(\sigma(t_x)=0.25\)，就取 \(t_x=\ln\frac{0.25}{0.75}=\ln\frac{1}{3}\approx-1.0986\)；要 \(\sigma(\text{obj})=0.9\)，就取 \(\text{obj}=\ln 9\approx2.197\)。類別 logits 取 (ln 0.8, ln 0.2)，softmax 之後正好回到 [0.8,0.2]：因為 \(e^{\ln p}=p\)，而 0.8+0.2=1。完整程式的 `fill` 函式就是這樣做：`torch.logit(p)` 算出 ln(p/(1−p))，類別比例則直接取 `.log()`。


## 三個候選裡，有一個重複，也有一個誤報

三個框都預測類別 0。先依圖上的索引辨認它們；索引來自格子順序（先 row 後 col），**不是分數排名**：

| 候選索引／顏色 | 來源格 (row,col) | 解碼 xyxy（px） | Score |
| --- | --- | --- | --- |
| 0／黃 | (1,1) | `[23.2,24,39.2,32]` | 0.64 |
| 1／藍 | (1,2) | `[28,24,44,32]` | 0.72 |
| 2／紅 | (3,0) | `[4,52,12,60]` | 0.855 |

![同一張人工圖，分數篩選後的三個候選與 NMS 後的兩個候選](../assets/diagrams/06-decode-nms.svg)

上下是同一張 64×64 圖。綠填色為唯一的 GT `[28,24,44,32]`，與藍框完全重合；黃虛線只偏了一點，也是這個物件的候選；左下紅框旁沒有真值，是背景誤報。圖把真值畫出來，讓我們判讀結果，**後面的去重演算法不會讀 GT**。

??? note "黃框和紅框是怎麼解碼出來的"

    三個候選都用同一套解碼公式（每格 16 pixels、整圖 64）：

    | 索引 | 格 (row,col) | σ(tx), σ(ty), σ(tw), σ(th) | σ(obj) | 類別比例 | 解碼框 | score |
    | --- | --- | --- | --- | --- | --- | --- |
    | 0 黃 | (1,1) | 0.95, 0.75, 0.25, 0.125 | 0.8 | [0.8,0.2] | [23.2,24,39.2,32] | 0.8×0.8=0.64 |
    | 1 藍 | (1,2) | 0.25, 0.75, 0.25, 0.125 | 0.9 | [0.8,0.2] | [28,24,44,32] | 0.9×0.8=0.72 |
    | 2 紅 | (3,0) | 0.5, 0.5, 0.125, 0.125 | 0.95 | [0.9,0.1] | [4,52,12,60] | 0.95×0.9=0.855 |

    例如黃框中心是 ((1+0.95)×16, (1+0.75)×16)=(31.2,28)，寬高和藍框一樣是 16×8，所以是 [23.2,24,39.2,32]。紅框中心是 ((0+0.5)×16, (3+0.5)×16)=(8,56)，寬高都是 0.125×64=8，所以是 [4,52,12,60]。


**NMS（Non-Maximum Suppression，非極大值抑制）**把高度重疊的候選當成可能的重複：每輪保留剩下候選中的最高分框，再刪掉與它 IoU **大於**門檻的其他框，直到候選耗盡。本節按預測類別分組，稱為 **class-wise NMS**；不同類別不互刪，這三框則在同一組處理。

門檻設 0.5，逐輪追蹤圖上的索引：

1. 分數最高的是紅框 2（0.855），先保留。它與藍、黃的 IoU 都為 0，兩框仍留下等待。
2. 剩下藍框 1（0.72）、黃框 0（0.64），保留藍。兩者 IoU≈0.5385>0.5，所以刪掉黃。
3. 沒有候選了，結束。回傳保留索引 **`[2,1]`**，按 score 由高到低排列。

第 2 輪的 IoU 可沿用單物件定位的算法：交集寬是 min(44,39.2)−max(28,23.2)=11.2，高是 8，面積 89.6；兩框各 128，聯集 128＋128−89.6=166.4：

\[
\operatorname{IoU}=89.6/166.4=7/13\approx0.5385.
\]

藍黃的重疊被清除了，紅色誤報卻仍在。NMS 比的是**候選與候選**，分數高、又不和別框重疊的錯框會被保留；若高分錯框和藍框重疊超過門檻，反而可能先保留錯框、再刪掉對框。它不修正位置，也不判斷框對不對。

訓練雖只讓中心格的 objectness 學 1，相鄰格看到相近影像，實際預測仍可能有不低的分數，這就是重複候選會出現的原因。本例則直接人工設出相鄰的藍黃兩框，用來看清去重動作。

## 調高門檻，少畫一個框，就變好了嗎

把 **score 門檻**從 0.25 提高到 0.75，藍框 0.72 和黃框 0.64 都在解碼篩選時消失，根本進不了 NMS；只剩最高分的背景紅框 0.855。結果框更少，但唯一的真物件也不再被找到，誤報依然存在。這個人工反例說明，畫面清爽不能代替真值檢查，也不是建議所有圖片都降低門檻。

**NMS 的 IoU 門檻**改的則是另一件事：

- 門檻低，例如 0.3，較容易刪掉重複，也較容易誤刪彼此貼近的同類物件。
- 門檻高，例如 0.7，兩個真物件較容易都留下，重複候選也更容易殘留。

若兩個同類物件的正確框本來就高度重疊，NMS 不知道是兩個物件，會把低分框當成重複。門檻的取捨應依資料與需求判斷，不能以保留數量當唯一答案。

候選越多，也越花計算。這個實作每保留一框就跑一輪，和剩下框算 IoU；n 框最壞為 n(n−1)/2 次，100 框最壞 4950 次。提高 IoU 門檻通常刪得少、留下多，要跑的輪數也更多。

## 解碼後的框，仍在輸入畫布上

本例直接使用 64×64 人工座標，不做原圖還原。若實際輸入來自 letterbox，解碼框仍相對含補邊的輸入畫布；要畫回原圖，依[座標轉換](04-coordinates.md)的順序，使用**每張圖自己的 metadata**：先扣補邊，再除實際縮放比例。

??? note "假設的還原小例子（本例沒有做這一步）"

    假設這張 64×64 來自第 4 章〈座標轉換與還原〉那張寬 80、高 40 的原圖：scale 0.8、上方 padding 16、左右沒有 padding。藍框 [28,24,44,32] 回到原圖是

    \[
    \left[\frac{28-0}{0.8},\ \frac{24-16}{0.8},\ \frac{44-0}{0.8},\ \frac{32-16}{0.8}\right]=[35,10,55,20].
    \]

    這只是假設的情境，用來複習「先扣 padding、再除 scale」；本節的程式沒有做還原。


## 對照程式：一次算完所有格

在 Colab 或本機跑 `PYTHONPATH=. python lesson_cases/06-decode-nms.py`，CPU 程式只用人工 logits 做幾何與排序，不做訓練。`decode` 的前半段把剛才的公式套到 16 格：

``` { .python data-excerpt="lesson_cases/06-decode-nms.py" }
def decode(logits, score_threshold=0.25, image_size=64):
    # logits 的 shape 是 [4,4,7]；shape[0] 是第一軸的長度，也就是每軸格數 4
    grid = logits.shape[0]
    # torch.arange(grid) 是 [0,1,2,3]；row、col 是兩張 4×4 的表，分別記下每一格的列號與欄號（見下方說明）
    row, col = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    # torch.stack((col, row), -1) 是每一格的 (x 方向格號, y 方向格號)
    # logits[..., :2]：「...」表示前面的軸（4×4 格）全取，「:2」取最後一軸前兩個值 (tx,ty)
    # image_size 用預設值 64，所以「/ grid * image_size」就是 ÷4×64，也就是 ×16
    center = (torch.stack((col, row), -1) + logits[..., :2].sigmoid()) / grid * image_size
    # [..., 2:4] 是 (tw,th)，乘整圖尺寸
    size = logits[..., 2:4].sigmoid() * image_size
    # center - size / 2 是 (x1,y1)，center + size / 2 是 (x2,y2)
    # torch.cat 沿最後一軸（-1）把兩者接成 [x1,y1,x2,y2]
    boxes = torch.cat((center - size / 2, center + size / 2), -1)
    ...
```

`row`、`col` 是每格的列號與欄號表。`stack((col,row),-1)` 讓位置 `[row,col]` 存著 (x 格號,y 格號)，再加上該格偏移；例如位置 `[1,2]` 的格號是 (2,1)。`size` 算寬高，`cat` 將左上、右下接成 xyxy。`...` 略去的後半段算 score、用門檻選格，再回傳 boxes、scores、labels；這裡 `labels` 是預測類別，不是 GT。

??? note "2×2 的小例子：meshgrid 產生了什麼"

    把 4 換成 2，表比較小，容易看清楚：

    ```python
    row, col = torch.meshgrid(torch.arange(2), torch.arange(2), indexing="ij")
    # row = [[0, 0],    col = [[0, 1],
    #        [1, 1]]           [0, 1]]
    torch.stack((col, row), -1)
    # 結果：
    # [[[0, 0], [1, 0]],
    #  [[0, 1], [1, 1]]]
    ```

    `indexing="ij"` 表示第一張表沿第一個軸（列）變化：`row[i,j]=i`、`col[i,j]=j`。疊起來之後，位置 `[1,0]` 的值是 [0,1]：列號 1、欄號 0 的格子，x 方向格號 0、y 方向格號 1。若改成 `indexing="xy"` 卻仍寫 `row, col =`，兩張表會對調，格號就錯了。


程式應印出表格的三個框、score `[0.64,0.72,0.855]`、藍黃 IoU=0.5385、NMS 索引 `[2,1]`。另測完全相同但類別不同的兩框，兩個都保留；空輸入回傳空索引。這些核對的是已知答案與邊界情況，不能據此宣稱已訓練的模型效果。

??? note "選讀：三種配對工作與四種門檻"

    訓練 assignment 用 GT 決定輸出應學誰；本節 NMS 只比候選彼此來去重；評估 matching（配對）則比預測與 GT，判斷對錯。三者的資料與目的不同。

    | 門檻 | 比較什麼 | 本節／下節的角色 |
    | --- | --- | --- |
    | 顯示 score | 每候選的 score | 本節 0.25 決定進入 NMS 的候選，反例改 0.75 |
    | NMS IoU | 候選與候選 | 本節 0.5 控制去重 |
    | 評估候選截斷 | 每候選的 score | 交給評估前先刪框，下一節示範 |
    | 評估配對 IoU | 預測與 GT | 判斷定位是否夠準，下一節使用 |

    本節真正使用前兩項，後兩項各有自己的設定。評估候選被截掉後，評估器完全看不到它。要比較 AP（Average Precision，平均精確率，下一節手算），必須固定評估規則，不能只從顯示框數推斷它會變好。

    本節自己實作 NMS，不需 torchvision。官方 [torchvision.ops.nms](https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html) 不分類別；按類別分組的 `batched_nms` 以 `idxs` 指定類別。兩者都刪 IoU 大於門檻者，回傳索引按分數遞減。

### 自主練習

完整程式（Colab 裡的那份）已在主案例之後放好兩段獨立的呼叫：主案例的門檻與斷言（assert）都不動，兩題互不影響。你只要先自己算出答案，再看輸出或展開參考答案核對：

1. 主案例的三個候選不變，只把 NMS 的 IoU 門檻從 0.5 改成 0.6。`class_nms` 會回傳哪些索引？順序為何？
2. 從同一組 logits 重新 decode，只把 score 門檻從 0.25 改成 0.70，不做 NMS。會剩下哪些 score？照什麼順序排？

兩題都不必修改函式預設值；執行後，輸出裡 `exercise NMS .6 …` 與 `exercise score .70 …` 這兩行就是結果。

??? note "參考答案"

    以下摘自完整程式的 `main()`（省略 `print`，`...` 表示省略的行；中文註解是本頁加的）：

    ``` { .python data-excerpt="lesson_cases/06-decode-nms.py" }
    # 練習 1：只把 NMS 的 IoU 門檻改成 0.6
    keep_iou06 = class_nms(boxes, scores, labels, threshold=.6)
    assert keep_iou06.tolist() == [2, 1, 0]
    ...
    # 練習 2：從同一組 logits 重新 decode，只把 score 門檻改成 0.70
    boxes70, scores70, labels70 = decode(logits, score_threshold=.70)
    assert len(boxes70) == 2 and labels70.tolist() == [0, 0]  # 只剩 2 個框，預測類別都是 0
    # allclose：逐項要求 |a−b| ≤ atol + rtol×|b|（a 是 `scores70`，b 是參考值 `[.72, .855]`），全部成立就算相等（浮點數有微小誤差）
    # atol 是固定的絕對容許差；rtol 是相對容許差（要乘上 |b|），沒寫時預設 1e-5，所以這裡實際容許約 8.2e-6～9.6e-6，不只 1e-6
    assert torch.allclose(scores70, torch.tensor([.72, .855]), atol=1e-6)
    ```

    練習 1：仍用 score 門檻 0.25 篩出的三個候選。第 1 輪保留紅框（索引 2），它和另外兩框都不重疊。第 2 輪保留藍框（索引 1）；藍黃的 IoU 約 0.5385，沒有超過新門檻 0.6，所以黃框不刪。第 3 輪保留黃框（索引 0）。照 score 由高到低排，結果是 [2,1,0]；這三個索引都指向原本的三個候選。

    練習 2：decode 篩選後的候選照格子位置排，不照分數排。門檻 0.70 會丟掉黃框 0.64，留下藍框 (row1,col2) 的 0.72 和紅框 (row3,col0) 的 0.855，所以 `scores70` 是 [0.72,0.855]。`boxes70` 只有兩個框：`boxes70[0]` 是藍框（原索引 1），`boxes70[1]` 是紅框（原索引 2），所以原本的索引在這裡對不上。第二個實驗沒有沿用練習 1 的 NMS 門檻 0.6；它只重新 decode，沒有做 NMS。

    哪一種設定比較好，要看和真值配對的結果與實際需求，不能只看畫面上的框數。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/06-decode-nms.json)

??? example "展開本次實際輸出"

    ```text
    candidate_boxes=[[23.2, 24.0, 39.2, 32.0], [28.0, 24.0, 44.0, 32.0], [4.0, 52.0, 12.0, 60.0]]
    scores=[0.64, 0.72, 0.855], duplicate_IoU=0.5385
    NMS keep_indices=[2, 1], kept_scores=[0.855, 0.72]
    threshold .75 keeps only artificial false positive, score=0.855
    exercise NMS .6 keeps original candidate indices=[2, 1, 0]
    exercise score .70 keeps scores=[0.72, 0.855]
    identical boxes of different classes both survive class-wise NMS; empty input passed
    All logits are hand-constructed, not a trained detector.
    ```

<!-- curriculum-evidence:end -->
