# 13.1 YOLOv10 dual assignment：訓練時多教，推論時少重複

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.0/notebooks/13-dual-assignment.ipynb){ .md-button }

第 6 章的 [NMS](06-decode-nms.md) 在推論最後保留高分框，刪掉重疊太多的同類框。本節要回答：能不能在訓練時就教模型少報重複框，推論時省掉 NMS？讀完後，你能說出 YOLOv10 為什麼訓練兩個 head、推論只留一個，並用小程式確認兩個 loss 的梯度走到哪裡。

[12.3](12-assignment.md) 讓一個 GT（人工標註的真值物件）教好幾個候選，叫一對多（one-to-many）。正樣本多，學習訊號豐富，但推論時好幾個候選可能都對同一物件給高分。一對一（one-to-one）的目標是讓每個物件只由一個候選負責，減少重複報出物件的機會。YOLOv10 的做法是先讓每個 GT 只挑一個首選，再處理搶同一候選的衝突；這個初選規則不等於最終每個 GT 都恰好得到一個正樣本。代價是正樣本較少。

YOLOv10 兩種都用，這叫 dual assignment（雙重分配）：訓練時，一對多 head 提供較多正樣本來訓練共享特徵，一對一 head 學會少報重複；推論時只用一對一 head。兩個 head 不是 [12.2](12-decoupled-head.md) 的分類、框兩條分支：官方的每個 head 裡面，都還各有分類與框分支。

![雙 head 的訓練與推論：一對多 loss 回到共享特徵；一對一輸入先 detach，loss 只訓練自己的 head。官方推論只留一對一 head；圖中另標本節 toy 的分配差異](../assets/diagrams/13-dual-head-flow.svg)

圖中實線往下是特徵與輸出，虛線往上是 loss 的梯度。右路的 detach 保留特徵數值，剪斷回到共享特徵的梯度；一對一 head 自己仍會學。下半是官方推論路徑：只保留一對一 head，沒有 NMS。本節程式只示範上半的分類 target 與梯度，沒有做完整偵測或速度評測。

## 官方的一對一：每個 GT 取 top-1，再處理衝突

先看一張已算好的品質表。shape `[G,P]=[2,3]`：兩列是 GT A、B，三欄是候選 p0、p1、p2，數字越大越適合負責該 GT。表頭 p0 等是候選名稱，不是分類機率。這張表與下方程式是人工 toy（小型教學例）；衝突暫時比表中品質，官方則比候選與 GT 的重疊品質（負值截成 0 的 CIoU，見 [12.3](12-assignment.md) 的選讀說明）。

| GT／候選 | p0 | p1 | p2 |
| --- | --- | --- | --- |
| A | 0.90 | 0.85 | 0.10 |
| B | 0.88 | 0.10 | 0.20 |

先用品質表看懂官方的 **top-1 選法**：A 的第一名是 p0，B 的第一名也是 p0。兩個 GT 搶同一個候選，就要解衝突；不是把 p0 同時交給兩人，也不會再替落選者找第二名。

若照本表的教學衝突規則，0.90 高於 0.88，p0 歸 A，B 這一步沒有一對一正樣本。owner 是 `[0,-1,-1]`：三個位置依序寫 p0、p1、p2 的主人，0＝A、1＝B、−1＝背景。**這個 owner 只示範 top-1 加品質解衝突，不是官方程式的數值輸出**；官方還需要框與 CIoU 才能判主人。本表只有兩個 GT，用這個示意規則時，每個 GT 最多一個正候選，衝突後可能沒有。官方實作的衝突規則還有一個差異：它會在所有 GT 中找重疊最大的主人，可能把候選交給原本沒選它的第三個 GT。因此 top-1 限制的是**衝突前的初選**，不能保證最後每個 GT 至多一個正樣本。每個候選最後只屬於一個 GT，則仍成立。

對照一對多：本表每個 GT 取 top-2，A 選 p0、p1，B 選 p0、p2；p0 衝突後仍歸 A，owner 是 `[0,0,1]`。A 有兩個正候選，B 有一個；每個候選最後仍只擁有一個 GT。官方的一對多取 top-10，本表縮為 top-2 方便手算。

### 兩邊的品質排名為什麼要一致？

論文稱這套做法為 consistent dual assignments（一致的雙重分配）。兩個 head 都用這個品質公式：

\[
m=s\cdot p^{\alpha}\cdot\text{IoU}^{\beta}
\]

s 是候選點是否在 GT 框內（在框內為 1，否則為 0）；p 是模型對該 GT 類別的分數；IoU 是預測框與 GT 的重疊程度。12.3 的 `score×IoU²` 對應 α=1、β=2；官方用 α=0.5、β=6，程式中的重疊值實際是負值截成 0 的 CIoU。

在兩個 head 算出相同 p 與 IoU 的假設下，兩邊採相同公式，一對一選中的首選，就是該 GT 在一對多的第一名。這是對每個 GT 的排名保證，沒有保證衝突後每個 GT 都有主人。實際訓練時，兩個 head 各用自己的預測，品質表與第一名也可能不同。

??? note "選讀：論文的排名假設與官方 target"

    論文讓一對一的 α、β 都是一對多的同一個正倍數 r，並假設兩個 head 的初始權重相同，對每對候選與 GT 得到相同的 p、IoU。此時一對一的 m 是一對多的 m 的 r 次方，排名不變；預設 r=1，公式完全相同。論文實驗顯示，相比參數不成比例時，一對一選中的候選更常落在一對多的前幾名，並非保證訓練全程排名相同。

    官方的正樣本分類 target 是 0～1 的軟值：m×（該 GT 正樣本中最大的 CIoU）÷（該 GT 正樣本中最大的 m）；其他欄與背景為 0。若某個 GT 最後只有一個正樣本，這個 target 是 `m×CIoU/(m+eps)`；eps 是避免除零的微小值，只有 m 遠大於 eps 時，才約等於該候選與 GT 的 CIoU。若衝突後它得到多個正樣本，仍要照上述公式逐個算（[官方 tal.py](https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/tal.py#L82-L86)）。下方 toy 只用 0／1 target。

??? note "選讀：官方 top-1 為什麼仍可能留下兩個正樣本？"

    用三個 GT A、B、C 和兩個候選看衝突：A、B 都初選 p0，C 初選 p1；但 p0 與 C 的重疊比與 A、B 更大。官方解衝突時在所有 GT 中比較重疊，所以把 p0 交給 C。p1 原本就歸 C，最後 C 有兩個正樣本，A、B 沒有。名稱「一對一」描述的是設計目標與 top-1 初選，不是這份實作最終分配的嚴格數量保證。

    本次用[固定版本的官方 `select_highest_overlaps`](https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/tal.py) 在 CPU 實際核對了這個反例；[輸入、結果與重現程式](https://github.com/birdhackor/learn_to_yolo/tree/main/reviews/clear-tutorial/16f6910/technical/probes)保存 owner=`[2,2]`，A、B、C 的正樣本數為 `[0,0,2]`。這個結果只針對所連結的 YOLOv10 實作；第 16 章的 YOLO26 在衝突後還會再篩一次，規則不同。

## 程式 toy：換一種配對，觀察兩個 head 的梯度

現在換到本節可執行程式。它保留「共享特徵、兩個 head、一對一輸入 detach」的路徑，兩個 head 只做分類；**一對一改用全域最優，不是官方 top-1**。目的在小表上看清 target 與梯度如何接起來，不用這個 owner 代表官方 YOLOv10。

toy 要求每個 GT 恰好配一個不同候選，且總品質最大（候選數必須足夠）。本表最佳配法是 A→p1、B→p0，總和 0.85+0.88=1.73，owner 是 `[1,0,-1]`。這和剛才 top-1 示意的 `[0,-1,-1]` 不同，也讓 p0 在一對多學 A、一對一卻學 B。這是訓練的 assignment，與 [6.2](06-evaluation.md) 評估時判定 TP 的 matching 分開。

輸入 `[3,4]` 是 `torch.randn` 產生的三列候選特徵，每列四個數；線性 backbone 輸出仍是 `[3,4]`。兩個分類 head 各輸出 `[3,2]`，每個候選有兩個類別 logits。本例 A 是 class 0、B 是 class 1，GT 編號剛好能當類別欄號；一般資料要先由 owner 找 GT，再取類別。

每欄各自做 sigmoid 與 BCE：類別欄 target=1 鼓勵它報該類，背景兩欄都是 0。owner 轉成 target 後：

| 候選 | 一對多 toy target | 全域一對一 toy target |
| --- | --- | --- |
| p0 | `[1,0]`：A | `[0,1]`：B |
| p1 | `[1,0]`：A | `[1,0]`：A |
| p2 | `[0,1]`：B | `[0,0]`：背景 |

下面依完整程式簡化，省略 `targets()` 定義、拆開 logits 兩行，另加 `loss` 變數與中文註解：

```python
features = backbone(inputs)                 # [3,4] → [3,4]
many_logits = many_head(features)           # [3,2]
one_logits = one_head(features.detach())    # [3,2]；只剪斷回到 backbone 的梯度
many_loss = F.binary_cross_entropy_with_logits(many_logits, targets(many_owner))
one_loss = F.binary_cross_entropy_with_logits(one_logits, targets(one_owner))
loss = many_loss + one_loss
```

官方的 detach 也在一對一 head 的輸入，圖中的梯度路徑與 toy 相同。一對一 loss 訓練自己的 head，但不回到共享 backbone／neck；共享特徵由一對多的豐富監督訓練。這和 12.2 的框、分類兩分支都訓練共享特徵不同。論文正文沒有提 detach，這是官方 `head.py` 的實作。

完整程式先只反傳 one loss，確認 backbone 的 `.grad is None`、one head 有非零梯度；再清掉梯度，反傳總 loss，確認 backbone 與兩個 head 有梯度。最後 `optimizer.step()`，檢查兩個 head 的權重都改變。detach 若移到 head 輸出後，連 head 自己都學不到；拿掉 detach，one loss 也會改 backbone。

在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/13-dual-assignment.py`。它仍印出 toy 一對多 `[0,0,1]`、**全域**一對一 `[1,0,-1]`、品質 1.73 對貪心 1.10，以及梯度路徑確認。最後兩個 BCE loss 是更新權重前的值。這個 CPU 小實驗沒有量測無 NMS 的 AP 或推論速度。

??? note "選讀：toy 如何求全域最優，為什麼貪心不一定對？"

    為什麼不把 A 最好的 p0 留給 A？先看貪心（greedy）的做法：每一步先拿眼前最大的數，拿了就不再改。全表最大的是 0.90（A→p0），先定下來；B 只剩 p1（0.10）和 p2（0.20），最多 0.20，總和 1.10。全域最優反而把 p0 讓給 B：A 從 p0 改用 p1 只少 0.05，B 從 p0 改用 p2 卻少 0.68，所以 A→p1、B→p0 的 1.73 比較大。這是可驗證的反例：每一步拿當下最大的值，不保證整體最好。

    完整程式的 `greedy_one_to_one` 函式照這個規則求貪心的配法：每一步只在還沒配到候選的 GT、還沒被拿走的候選之間找最大的一格，定下就不再改，直到每個 GT 都配到候選。斷言確認本表的貪心是 A→p0、B→p2、總和 1.10，也確認全域最優嚴格大於貪心。注意貪心比的是全表剩下的最大數，不是讓 A 先挑：另一個斷言用一張把 B 的 p0 改成 0.95 的小表檢查，這時最大的 0.95 屬於 B，所以 B 先拿 p0，A 再從剩下的 p1、p2 拿較大的 p1。

    完整程式的 `exact_one_to_one` 函式用列舉求全域最優，核心是這兩行（寫法略有簡化）：

    ```python
    best = max(permutations(range(P), G),
               key=lambda cols: sum(quality[g, c] for g, c in enumerate(cols)))
    ```

    以本例 P=3、G=2 逐段看：

    - `permutations(range(3), 2)`（Python 標準函式庫 `itertools` 的函式）依序列出從 3 個候選挑 2 個不同候選的所有排法：(0,1)、(0,2)、(1,0)、(1,2)、(2,0)、(2,1)。每組的第一個數是 A 拿到的候選欄，第二個數是 B 的；例如 (1,0) 表示 A→p1、B→p0。
    - `lambda cols: ...` 是寫在一行裡的臨時函式。輸入一種配法 `cols`，`enumerate(cols)` 依序給出（GT 編號 g, 候選欄 c），把 `quality[g, c]` 加起來，就是這種配法的總品質。例如 `cols=(1,0)` 時依序給出 (0,1)、(1,0)，總品質是 `quality[0,1]+quality[1,0]`，也就是 0.85+0.88=1.73。
    - `max(..., key=...)` 依 key 算出的總品質比大小，回傳總品質最大的那一組。

    | 配法 `cols` | A 拿 | B 拿 | 總品質 |
    | --- | --- | --- | --- |
    | (0,1) | p0 | p1 | 0.90+0.10=1.00 |
    | (0,2) | p0 | p2 | 0.90+0.20=1.10（就是貪心的結果） |
    | (1,0) | p1 | p0 | 0.85+0.88=1.73（最大） |
    | (1,2) | p1 | p2 | 0.85+0.20=1.05 |
    | (2,0) | p2 | p0 | 0.10+0.88=0.98 |
    | (2,1) | p2 | p1 | 0.10+0.10=0.20 |

    所以 `best=(1,0)`。注意方向：`best` 寫的是「每個 GT 拿哪個候選」，A 拿候選 1、B 拿候選 0。owner 反過來寫「每個候選歸哪個 GT」：p0 歸 B（1）、p1 歸 A（0）、p2 沒人拿（−1），所以是 `[1,0,-1]`；完整程式用 `owner[c] = i` 做這個翻譯（完整程式把 GT 編號寫成 i，就是上面簡化版的 g）。這題的 `best` 和 owner 前兩格剛好都是 1、0，看起來很像，方向卻相反；自己改表時，不要把 `best` 後面補一個 −1 就當成 owner。

    這種搜尋要試 `P!/(P−G)!` 種配法。本例 A 先有 3 個候選可選，B 剩 2 個，共 3×2=6 種，就是高中學的排列數。候選與 GT 變多時，這個數字增加得非常快。例如 640×640 的輸入用 stride 8、16、32 三層特徵圖，各有 80×80、40×40、20×20 格，每格一個候選，共 8400 個候選；圖上有 10 個 GT 時，要試 8400×8399×…×8391≈1.7×10³⁹ 種，不可能一一列舉。

    「每個 GT 配一個不同的候選，讓總品質最大」這類問題叫指派問題。匈牙利演算法（Hungarian algorithm）是指派問題的經典解法：不必列出全部組合，就能有效率地求出最佳解；DETR（DEtection TRansformer，偵測 Transformer；另一類也不用 NMS 的偵測器）訓練時就用它做一對一配對。YOLOv10 沒有用它：論文說，一對一改成每個 GT 只取第一名（top-1，沿用 12.3 那種 task-aligned 的品質排名），效果和匈牙利配對相當，額外的訓練時間更少。


## 推論只留哪一個？還有哪些限制？

官方推論丟掉一對多 head，從一對一 head 取全圖分數最高的 k 個（框, 類別）組合，預設最多 300，再以各組合的 score 過濾；不兩兩比 IoU 刪框，因此叫 NMS-free。這個推論 top-k 與訓練「每個 GT 取 top-1／top-10」用途不同。多類時，同一框仍可能以不同類別出現，top-k 自己也不會辨認重複框（[13.2](13-nms-free.md) 有反例）。

收益是共享特徵在訓練時得到較多正樣本，部署只留一對一 head，省去 NMS；論文指出 NMS 會拖慢推論、受 IoU 門檻影響，也妨礙端到端部署。代價是訓練多一套 head、assignment 與 loss。

常見錯誤：

- **只有一對一 head，卻叫 dual assignment。** 缺少一對多 head 提供的監督。
- **兩套 target 共用同一組 head 權重。** toy 的 p0 一邊學 A、一邊學 B；p2 一邊學 B、一邊學背景。官方即使用 top-1，兩邊選中的正候選數也不同，仍常有一邊正、一邊背景的要求，因此用兩組 head。
- **一對一是整張圖只准一個物件。** toy 在候選足夠時要求每個 GT 各配一個；官方 top-1 則限制每個 GT 的初選，衝突可能改變最後的數量。不能由「圖上有兩物件」推得必有兩個正樣本，更不能保證推論剛好輸出兩框。
- **公式一致就沒有 target 衝突。** 排名的一致不等於每個候選主人一致；兩 head 預測也可能不同。

??? note "選讀：同一張品質表，候選主人仍可能不同"

    把表改成 A＝0.95、0.90、0.10，B＝0.10、0.88、0.20，用本頁比品質的示意規則。一對多 top-2 時 A 選 p0、p1，B 選 p1、p2；p1 衝突後因 0.90>0.88 歸 A。一對一 top-1 時 A 取 p0、B 取 p1，p1 歸 B。每個 GT 的首選都一致，同一候選仍可在兩 head 收到不同 target。這不是官方 CIoU 衝突的實跑結果。

是否真的少報重複，還要在沒參與訓練的圖片上看重複框、漏檢與 AP。下一節用位置已知的人工框，檢查只教一個正候選是否能壓低其他候選分數。

## 選讀練習：改 toy 的品質表

在 Colab 裡改「本節可修改的完整實驗」下面那一格程式；本機則改 `lesson_cases/13-dual-assignment.py`，兩者是同一份程式。

1. 把 `main()` 裡 `quality` 的第二列 `[.88, .10, .20]` 改成 `[.88, .10, .95]`，也就是 B 對 p2 的品質改為 0.95。先手算：一對一的全域最優配法和總品質是多少？貪心呢？一對一、一對多的 owner 各變成什麼？完整程式的斷言寫死了原表的答案，直接執行會在斷言停下；想想哪幾個斷言要改、改成什麼。
2. 再把 GT 改成 4 個（`quality` 改成 4 列，每列 3 個數），候選仍是 3 個。執行時會發生什麼事？應該怎麼處理？

??? note "參考答案"

    **第 1 題**：六種配法的總品質依序是 1.00、1.85、1.73、1.80、0.98、0.20，最大的是 (0,2)：A→p0、B→p2，總和 0.90+0.95=1.85。`best=(0,2)` 翻成「每個候選歸誰」，一對一 owner 是 `[0,-1,1]`，不是 `[0,2,-1]`。一對多 owner 仍是 `[0,0,1]`：A 仍選 p0、p1，B 改選 p2、p0，p0 仍因 0.90 高於 0.88 歸 A。

    新表最大的數是 0.95，所以貪心先給 B→p2，再給 A→p0，總和同樣是 1.85。在原表不是最好的貪心，在這張表變成最好；一個成功的例子，不能證明貪心總是對的。

    完整程式裡要改三個斷言，依出現的順序是：

    - `assert one_owner.tolist() == [1, 0, -1]` 改成 `assert one_owner.tolist() == [0, -1, 1]`。
    - `assert greedy_cols == (0, 2) and abs(greedy_value - 1.10) < 1e-6` 改成 `assert greedy_cols == (0, 2) and abs(greedy_value - 1.85) < 1e-6`。`greedy_cols` 和 `best` 一樣寫「每個 GT 拿哪個候選」；新表的貪心仍是 A→p0、B→p2，只有總和變成 0.90+0.95。
    - `assert optimum > greedy_value + 1e-7` 檢查「全域最優嚴格大於貪心」。新表兩者相等，這行會失敗；改成 `assert abs(optimum - greedy_value) < 1e-6`，意思是兩者相等。

    其他斷言不用改。一對多 owner 仍是 `[0,0,1]`，所以 `assert many_owner.tolist() == [0, 0, 1]` 照樣成立；另外兩個把小表直接寫在括號裡的斷言（`one_to_many_owner(torch.tensor(...))` 與 `greedy_one_to_one(torch.tensor(...))` 那兩行）不讀 `quality`，也不受影響。

    三個都改好後執行，第三行印出 `global quality=1.85; greedy quality=1.85`；一對一的 target 變了，最後一行的 one loss 也會跟著變。程式裡的 0.90、0.95 以 float32 儲存，存進去時有極小的捨入，所以實際算出 1.8499999642…，印出時才四捨五入成 1.85。這裡 `optimum` 和 `greedy_value` 是同樣兩個數相加，結果完全相同；用 `abs` 比較只是保險，極小的捨入誤差不能算成「全域比貪心好」。

    **第 2 題**：3 個候選排不出 4 個不同的位置，`permutations(range(3), 4)` 什麼都不產生，`max()` 會報 `ValueError`（Python 3.12 的訊息是 `max() iterable argument is empty`）。程式一呼叫 `exact_one_to_one` 就停下，後面的一對多、貪心與訓練都不會執行。這個報錯是對的：一對一不可能讓 4 個物件各拿一個候選，至少有一個 GT 沒有一對一正樣本，也就是沒配到候選的 GT（unmatched GT）。不要為了讓錯誤消失，改成只配其中 3 個 GT 卻不記錄丟了誰，那會讓一個物件被默默忽略。建議先明確允許 GT 沒配到，並把它們列出來；或增加候選數。這是本節建議的做法，不是 YOLOv10 的規則。

    要在程式裡允許 GT 沒配到，不能只改 `exact_one_to_one`。`greedy_one_to_one` 和它一樣假設每個 GT 都拿得到不同的候選：4 個 GT、3 個候選時，它配完 3 個 GT 就沒有可選的格子，`max()` 會報同一個 `ValueError`。`targets()` 則直接把 GT 編號當類別欄號，這只在本例 A、B 剛好是類別 0、1 時成立：編號 2、3 的 GT 只要分到候選，就會超出只有 2 欄的 target，程式報 `IndexError`；要改成先由 owner 找到 GT，再取那個 GT 的類別。寫死原表答案的斷言，也要照新表的答案改。

參考來源：[YOLOv10 論文](https://arxiv.org/abs/2405.14458)、[官方 repository，固定 commit](https://github.com/THU-MIG/yolov10/tree/453c6e38a51e9d1d5a2aa5fb7f1014a711913397)、[雙 head 與 detach 實作](https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/nn/modules/head.py)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-06 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/13-dual-assignment.json)

??? example "展開本次實際輸出"

    ```text
    one-to-many owner: [0, 0, 1]
    global one-to-one owner: [1, 0, -1]
    global quality=1.73; greedy quality=1.10
    detached one-to-one branch trains its head; only many branch trains backbone
    loss many=0.6855, one=0.7065
    ```

<!-- curriculum-evidence:end -->
