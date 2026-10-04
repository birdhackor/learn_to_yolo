# 3.2 ResNet projection shortcut：對齊形狀也在學轉換

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/03-projection.ipynb){ .md-button }

ResNet 把 block 分成幾個 stage（階段）。同一個 stage 裡，各 block 輸出的特徵圖（卷積層輸出的 `[C,H,W]` 數值）H、W 都相同；進入下一個 stage 時，通常由新 stage 的第一個 block 縮小 H、W 並增加 channel。例如本節主分支輸出 `[B,6,2,2]`，輸入卻是 `[B,3,4,4]`，兩者直接相加不合法。Projection shortcut（投影捷徑）先把輸入轉成和主分支相同的 shape，才加入主分支。通常只有這種 shape 改變的 block 用 projection，其餘 block 仍用上一節的 identity shortcut。

這裡的 projection（投影）沿用 ResNet 論文的叫法（論文寫成 linear projection \(W_s\)），意思只是「乘上一個可學的矩陣，做線性轉換來對齊維度」。它不是數學課把向量投到某個方向的正射影；本例甚至把 3 個 channel 變成 6 個。

前置只需知道 NCHW 與卷積的 stride（卷積窗每次滑動的格數）；本節也會重新說明兩種 shortcut 的差異。讀完本節，你能自己代公式確認兩支的 shape 能否對齊，也能手算 1×1 卷積的輸出。

歷史機制來自 [ResNet 原始論文](https://arxiv.org/abs/1512.03385)：用可學的 projection 處理尺寸或 channel 不同的 shortcut。本節省略原版每個卷積後的 BatchNorm（一種把每個 channel 的數值重新標準化的層），也省略相加之後的 activation（激勵函數，例如 ReLU）；兩層卷積中間的 ReLU 保留。本節只保留一個位在 stage 交界的小型 block，它不是要重現完整 ResNet 的效果。

可以用頁首的按鈕在 Colab 執行，或在本機執行 `PYTHONPATH=. python lesson_cases/03-projection.py`。CPU 實驗分兩部分：第一部分「手設權重驗算」用手動設定的權重，驗算一個 channel 混合後的數值；第二部分「隨機權重更新檢查」另建一個隨機初始化的 block，做 2 步更新。

## 先把兩支畫出來

![Projection 兩支的 shape，以及 F 的 486 個、P 的 18 個可學權重](../assets/diagrams/03-projection.svg)

B 是一次輸入的圖片筆數。主分支 F 與 shortcut P 的設定如下：

| 路徑 | 設定 | 輸入 | 輸出 |
| --- | --- | --- | --- |
| F 第一層 | 3×3 卷積，padding=1、stride=2，3→6 channel | `[B,3,4,4]` | `[B,6,2,2]` |
| F 兩層之間 | ReLU（負值變 0） | `[B,6,2,2]` | `[B,6,2,2]` |
| F 第二層 | 3×3 卷積，padding=1、stride=1，6→6 channel | `[B,6,2,2]` | `[B,6,2,2]` |
| P | 1×1 卷積，padding=0、stride=2，3→6 channel | `[B,3,4,4]` | `[B,6,2,2]` |
| 相加 | 逐個位置、逐個 channel 相加 | 兩個 `[B,6,2,2]` | `[B,6,2,2]` |

邊長用 [VGG 風格小 CNN](01-small-cnn.md) 那一節的公式 \(\lfloor(H+2p-k)/s\rfloor+1\) 計算（H 是輸入邊長，p 是 padding，k 是 kernel 邊長，s 是 stride）。F 第一層是 \(\lfloor(4+2\cdot1-3)/2\rfloor+1=\lfloor1.5\rfloor+1=2\)，P 是 \(\lfloor(4+0-1)/2\rfloor+1=2\)。兩支輸出都是 `[B,6,2,2]`，相加得到：

\[
y=P(x)+F(x).
\]

為什麼 P 用 1×1 卷積？原論文在 channel 變多時考慮過兩種 shortcut：(A) 仍用 identity，以 stride 2 取樣後，多出的 channel 補 0，不增加參數；(B) 用 1×1 卷積做 projection。本節實作 (B)。1×1 是能改變 channel 數的最小卷積，只要 \(6\times3=18\) 個參數（3×3 要 \(6\times3\times9=162\) 個）；看鄰近範圍的工作留給主分支 F。

上一節的 identity 用在同 shape 的情況：\(P(x)=x\)，值直接通過，沒有可學權重。上一節列出 identity shortcut 的兩點收益：主分支權重為 0 時，整段就等於輸入；梯度能沿 shortcut 直接傳回前面的層，不經過卷積權重，本節把這條路叫直接梯度路徑。換成本節要學習的 P 後，就算 \(F=0\)，\(y=P(x)\) 也是 x 經過 channel 加權混合、再每隔一格取樣的新數值，不是 x 的原值。所以第一點（整段等於輸入）在這裡不成立。

??? note "換成 P 之後，上一節的直接梯度路徑還在嗎？"

    梯度仍能沿 P 回傳，但不再原樣傳遞。先假設 \(F=0\)。若 P 把某個輸入值乘上權重 3、加進某個輸出值，那個輸入值增加 0.001，這個輸出值就增加 0.003；上一節的 identity 則是剛好增加 0.001。所以梯度沿 P 回傳時會乘上 P 的權重，不再固定是 1。

    不過這條路只經過一層線性的 1×1 卷積，沒有 ReLU，仍比主分支 F（兩層卷積中間夾一個 ReLU）短。另外，後面會看到 P 只讀 16 個位置中的 4 個；P 沒讀到的 12 個位置，梯度只能經 F 回傳。

## 1×1 卷積看不到鄰居，卻能混合 channel

在某個位置，三個輸入 channel 的值是 1、10、100。如果某個輸出 channel 的權重是 1、2、3，輸出就是每個權重乘上對應的輸入，再加起來：

\[
1\cdot1+2\cdot10+3\cdot100=321.
\]

這就是權重向量 (1,2,3) 和該位置 channel 向量 (1,10,100) 的內積（[暖身那一節](00-warmup.md)講過）。選 1、10、100 的好處是：321 的百、十、個位剛好露出 3、2、1 三個權重。

321 與任何單一輸入值（1、10、100）都不同。1×1 的「1」指空間只看一個位置，不是只讀一個 channel。每個輸出 channel 都有一套跨輸入 channel 的權重。本例無 bias，因此參數是 \(6\times3=18\) 個。

stride 2 會讀取間隔 2 的位置。4×4 輸入時，P 只讀列 0、2 與欄 0、2 交叉的 4 個位置（索引從 0 算起），輸出 2×2。16 個位置中有 12 個完全沒進入 shortcut，它們的資訊只能經由 F 傳下去。這也是 P 和 2×2、stride 2 平均池化（average pooling）的差別：平均池化會用到每一格。F 第一層 3×3 窗的中心剛好也是這 4 個位置，而且這些窗合起來涵蓋輸入的每一格。因此兩支即使 shape 一致，數值來源也不同。

## Shape 與程式對照

下面摘自完整程式：第一行在 `__init__`，建立模型時執行一次；其餘三行在 `forward`，每次輸入資料都會執行。

```python
# 建立模型時：輸入 3 channel、輸出 6 channel、kernel 1×1、stride 2、無 bias
# （padding 沒寫，預設是 0）
self.projection = nn.Conv2d(3, 6, 1, stride=2, bias=False)

# 以下在 forward 裡，每次輸入都執行
# self.branch 就是 F：3×3 stride 2 → ReLU → 3×3 stride 1
main, shortcut = self.branch(x), self.projection(x)
assert main.shape == shortcut.shape
y = main + shortcut  # 完整程式寫成 return main + shortcut
```

projection 的權重 shape 是 `[6,3,1,1]`。卷積權重的四個軸依序是 [輸出 channel, 輸入 channel, kernel 高, kernel 寬]，不是 NCHW；和暖身那一節 Linear 權重 `[K,D]` 一樣，輸出在前。拿掉兩個長度 1 的軸，權重就是一個 6×3 矩陣 W。在 P 讀到的每個位置，把 3 個 channel 的值排成向量 v，算 Wv 就得到對應輸出格的 6 個 channel。

**第一部分：手設權重驗算。** 輸入是 `[1,3,4,4]`、輸出 `[1,6,2,2]`。輸入的 16 個位置，三個 channel 都是 (1,10,100)。F 的權重全設 0，所以 F(x) 全為 0。P 的權重矩陣 W 只有第一列（第一個輸出 channel，程式裡的索引 0）設成 (1,2,3)，其餘 5 列（其餘 5 個輸出 channel）全是 0。因此 \(y=P(x)\)：第一個輸出 channel 的 2×2 每格都是 321，其餘 5 個 channel 全是 0。完整程式裡對應的幾行如下：

```python
with torch.no_grad():  # 手動改權重時不記錄梯度（不包會報錯）
    # 這之前（同一個 with 區塊裡），完整程式已把 F 和 P 的權重全部設成 0
    # 只設 P 第一個輸出 channel（索引 0）的權重：(1,2,3)
    block.projection.weight[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
# 第一部分的輸入 [1,3,4,4]：16 個位置都是 (1,10,100)
# （view 先排成 [1,3,1,1]，expand 再擴展成高 4、寬 4）
image = torch.tensor([1.0, 10.0, 100.0]).view(1, 3, 1, 1).expand(1, 3, 4, 4)
# output 是 block(image) 的輸出，shape 必須是 [1,6,2,2]
assert output.shape == (1, 6, 2, 2)
# torch.full((2, 2), 321.0) 建立每格都是 321 的 2×2，用來和第一個輸出 channel 比對
assert torch.equal(output[0, 0], torch.full((2, 2), 321.0))
```

這個已知答案用來核對，而不是用隨機數「大概看起來差不多」。321 的每一位都對應一個權重，channel 順序弄反會得到 \(1\cdot100+2\cdot10+3\cdot1=123\)，所以這個值能確認權重乘在正確的 channel 上。stride 只能從輸出是 2×2 來核對（stride 1 會得到 4×4）。因為每個位置的輸入都相同，這個測試看不出 P 讀的是哪 4 個位置；要驗證取樣位置，得讓每個位置的值都不同。

**第二部分：隨機權重更新檢查。** 另建一個隨機初始化的 block，輸入是 `[2,3,4,4]`，目標是全 0 的 `[2,6,2,2]`。用 MSE 算 loss（每個輸出與 0 的差平方，再平均），再反傳求梯度。程式確認 P 的梯度不是 0，而且兩步後 P 的權重改變。只有第二部分的隨機 block 在檢查訓練能否更新 P；第一部分把 F 權重全設 0 只是為了方便手算，不建議訓練時這樣初始化（上一節說過：F 的兩層權重全為 0 時，兩層的權重梯度也都是 0，學不動）。

## 收益與代價，要分開算

收益是主分支與 shortcut 在 stage 交界也能相加，而且 P 能學習 channel 轉換。參數方面，F 第一層有 \(6\times3\times3\times3=162\) 個，第二層有 \(6\times6\times3\times3=324\) 個，合計 486；加上 P 的 \(6\times3=18\) 個，總共 **504**。

乘加次數（MAC，一個乘積累加到答案）以每張圖計算。P 是 \(2\times2\times6\times3=72\) 次：2×2 個位置、6 個輸出 channel、每個輸出讀 3 個值。主分支兩層是 \(2\times2\times6\times(3\times9)=648\) 與 \(2\times2\times6\times(6\times9)=1296\)，合計 1944；加上 P 的 72 次，只比主分支多約 3.7%。兩支相加是每張圖 \(6\times2\times2=24\) 次逐值加法，不算在 MAC 裡。P 的 18 個參數與 72 次乘加，就是 (B) 比不增加參數的 (A) 多付的代價。這些數字不包含記憶體存取與 ReLU 的計算，不能直接當作實際延遲。

空間縮小有助降低後續計算，代價是小物件細節可能消失。增加 channel 可以表示更多種特徵（表示容量較大），代價是更多參數與計算。

??? note "為什麼常常是 H、W 減半，同時 channel 加倍？"

    這是 ResNet 論文的設計規則：特徵圖的 H、W 各減半時，channel 加倍，讓每層卷積的計算量大致不變。一層卷積的乘加數是「輸出位置數 × 輸出 channel 數 × 每個輸出讀取的值數」，也就是 \(H\times W\times C_{out}\times C_{in}\times k^2\)；這裡 H、W 是輸出的高、寬，\(C_{in}\)、\(C_{out}\) 是輸入、輸出 channel 數，k 是 kernel 邊長。H、W 各減半，\(H\times W\) 變成四分之一；\(C_{in}\)、\(C_{out}\) 各加倍，\(C_{out}\times C_{in}\) 變成 4 倍，兩者剛好抵消。

    舉例：假設上一個 stage 有一層 4×4、3→3 的 3×3 卷積（這一層是假設的例子，本節程式裡沒有），乘加是 \(4\times4\times3\times(3\times9)=1296\) 次。本節 F 第二層是 2×2、6→6，乘加是 \(2\times2\times6\times(6\times9)=1296\) 次，一樣多。F 第一層位在兩個 stage 交界，輸入仍是 3 個 channel，所以不拿它比。

把兩支 shape 改到能相加，只是 forward 能合法執行的條件，還沒有證明這個設計比 plain network（沒有 shortcut、只把卷積一層層串起來的網路）更準。下一節[「Plain／residual 對照」](03-comparison.md)會把 plain network 和有 shortcut 的網路放在一起比較；那裡用的是同 shape 的 identity shortcut，不是本節的 projection。

## 核對與常見錯誤

程式應印出四個 shape，第一個輸出 channel 印成 `[[321.0, 321.0], [321.0, 321.0]]`，並通過「P 的權重有更新」與「block 共 504 個參數」的 assert 檢查。到這裡，本節的檢查就完成了。

常見錯誤有三種。第一，不要以為 1×1 卷積只是讓值原樣通過、沒有計算：321 的例子就是它在每個位置做加權和。第二，不要因為主分支下取樣（縮小 H、W）後 shape 對不上，就乾脆拿掉 shortcut：那樣這個 block 就少了那條較短的路徑，變回沒有 shortcut 的 plain block。第三，不要用 reshape 硬湊出同 shape：reshape 不改變元素總數，本例每張圖 3×4×4=48 個值，無法 reshape 成 6×2×2=24 個（會報錯）；就算硬湊成 `[12,2,2]` 再挑 6 個 channel，新的每個「channel」其實是原本某一列的 4 個值摺成 2×2，位置與 channel 的意義都亂了。

兩支能否對齊要代公式。本節 F 第一層（k=3、p=1）與 P（k=1、p=0）的 \(H+2p-k\) 都是 \(H-1\)，stride 都是 2，輸出邊長都是 \(\lfloor(H-1)/2\rfloor+1\)，所以任何 H（包括練習的 5）都對齊。換設定就要重算：若 F 第一層改成沒有 padding，F 的輸出邊長對任何 H 都比 P 少 1（4×4 時 F 輸出 1×1、P 輸出 2×2）；若 shortcut 改用 2×2、stride 2 的 pooling，邊長在偶數 H 時對齊、奇數 H 時差 1（H=5 時 pooling 輸出 2×2、F 輸出 3×3）。不能只看 stride 相同。

## 自主練習與答案

在完整程式（Colab 或 `lesson_cases/03-projection.py`）裡，只把第一部分的輸入從 4×4 改成 5×5：把建立 `image` 那行的 `expand(1, 3, 4, 4)` 改成 `expand(1, 3, 5, 5)`。卷積設定不變，第二部分仍用 4×4。先自己回答：

1. F 和 P 的輸出各是幾×幾？兩支還能相加嗎？
2. 第一個輸出 channel 每格是多少？第一部分的兩個 assert（檢查 `output.shape` 的那行、用 `torch.full` 的那行）要改成什麼？
3. 接著再把 `self.projection` 那行的 `stride=2` 改成 `stride=1`，執行時會發生什麼事？該怎麼修？

??? note "參考答案"

    1. F 第一層輸出 \(\lfloor(5+2-3)/2\rfloor+1=3\)；F 第二層 stride 1、padding 1，輸出 \(\lfloor(3+2-3)/1\rfloor+1=3\)，維持 3×3。P 輸出 \(\lfloor(5-1)/2\rfloor+1=3\)。兩支都是 3×3，可以相加。
    2. 每個位置的輸入仍是 (1,10,100)、權重仍是 (1,2,3)，所以第一個輸出 channel 每格仍是 321，只是變成 3×3。把 `assert output.shape == (1, 6, 2, 2)` 改成 `(1, 6, 3, 3)`，`torch.full((2, 2), 321.0)` 改成 `torch.full((3, 3), 321.0)`。第二部分不改，因為它的目標固定是 `[2,6,2,2]`；輸入改大的話，目標也得跟著改，否則 shape 對不上。
    3. P 輸出變成 \(\lfloor(5-1)/1\rfloor+1=5\)，也就是 5×5，F 仍是 3×3。forward 裡的 `assert main.shape == shortcut.shape` 會先失敗：它在 `output = block(image)` 那一步就執行，早於 `output.shape` 的 assert。下一步是修正設計，不是移除 assert。一個通用的修法是讓 F 第一層與 P 的 stride 相同、\(2p-k\) 也相同（F 第二層不改變邊長），這樣不論 H 是多少，兩支的輸出邊長都一樣。本節的 block 位在 stage 交界，本來就要縮小 H、W，所以實際的修法是把 P 改回 `stride=2`。若改成 F 第一層也用 stride 1，兩支雖然對齊，輸出卻變成 5×5，第 2 題改好的兩個 assert 又得改成 5×5 的版本。第二部分的輸出也會變成 `[2,6,4,4]`，和目標 `[2,6,2,2]` 對不上，`mse_loss` 會報錯，目標也得跟著改。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式已於 2026-10-02 用 PyTorch 2.9.1+cpu 在 CPU 上執行過，程式裡的 assert 檢查全部通過。下面是那次印出的原始輸出；每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-projection.json)

??? example "展開本次實際輸出"

    ```text
    input=(1, 3, 4, 4), main=(1, 6, 2, 2), projection=(1, 6, 2, 2), output=(1, 6, 2, 2)
    first output channel=[[321.0, 321.0], [321.0, 321.0]]; 1*1 + 2*10 + 3*100 = 321
    step=0, loss=0.3553
    step=1, loss=0.3350
    projection changed; block parameters=504; no classification-quality claim
    ```

<!-- curriculum-evidence:end -->
