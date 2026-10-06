# 12.4 DFL：把一條邊距離學成分佈

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/12-dfl.ipynb){ .md-button }

一條邊的距離，能不能不只輸出一個數，而是說成「比較接近 1 格，但也有一部分落在 2 格」？本節就把這個距離學成一組機率。讀完你能手算這種寫法的 loss、梯度，以及怎麼把機率還原成距離，也能說出它多付了哪些代價。

前置：softmax 與 cross entropy（第 7 章〈[Grid MiniYOLO loss](07-loss.md)〉）、[12.1 節](12-anchor-free.md)的四邊距離，以及反向傳播。本節距離的單位「格」是特徵格，1 格＝stride 畫素。

為什麼要改？2020 年的 Generalized Focal Loss（GFL）論文認為，直接回歸等於假設每條邊只有一個確定的位置；但邊被遮住或模糊時，它可能在 1 格附近，也可能在 2 格附近。論文因此讓每條邊輸出一組機率，再用 Distribution Focal Loss（簡稱 DFL）把機率集中到真值兩側最近的刻度。這是論文的主張；本節只驗證計算，不證明準確度會提高。名字裡雖然有 Focal，但讀本節不必先學 focal loss（另一種分類用的 loss，本書沒有用到）。

??? note "名字裡的 Focal 是什麼意思"

    Focal 是「聚焦」：原論文說，DFL 讓網路很快把機率集中在真值兩側最近的兩個刻度。DFL 的公式沒有 focal loss 常見的額外調整項，就是加權的 cross entropy。論文把 focal loss、同一篇提出的 QFL（Quality Focal Loss，用在分類分支）和 DFL 都看成 Generalized Focal Loss（focal loss 的推廣）的特例；DFL 是不帶調整項的那一種，因為框只在正樣本上學，沒有類別不平衡的問題，所以只留下 cross entropy 的部分。

這些刻度叫 bin。bin 可想成尺上的整數刻度：本例把距離分成 0、1、2、3 格四個 bin（bin0 代表 0 格、bin1 代表 1 格，依此類推）。模型對每個 bin 輸出一個 logit，softmax 後就是距離落在該刻度的機率。DFL 等於把「距離幾格」當成 4 選 1 的分類題，只是正確答案拆給相鄰兩格：距離不是整數時，由兩側相鄰的兩個 bin 分擔監督。

最後用期望值把機率還原成一個距離。這就是高中機率的期望值：距離＝\(0\cdot p_0+1\cdot p_1+2\cdot p_2+3\cdot p_3\)，\(p_0\)～\(p_3\) 是四個 bin 的機率。多輸出幾個 channel 不會自動變成分佈：要 softmax 成機率、用 DFL 監督、再用期望值解碼。本節三步都算一遍。

YOLOv8 等模型的四邊回歸也使用這種分佈表示。本節的起點是 12.1 的 anchor-free 四邊距離：原本每條邊用 softplus 直接輸出一個距離，這節只拿其中一條邊，改成 4 個 bin 的分佈來示範。實際的偵測器（detector）四條邊各有一份分佈，還要搭配整個框的 IoU loss（第 11 章〈[IoU 類 loss](11-iou-loss.md)〉）一起訓練。本例不包含分類，也沒有圖片或完整的偵測訓練。實驗後是否保留 DFL，應由定位品質和部署成本決定。

## 1.25 格不是第 1 類

stride 為 8 畫素／格。真值的左邊距離（定義同 12.1：候選點到框左邊的距離；這裡另舉一個例子，不沿用 12.1 的數字）是 10 畫素，換成格是 \(d=10/8=1.25\) 格。d 夾在兩個相鄰的 bin 之間：較小的是 \(i=\lfloor d\rfloor=1\)，較大的是 \(i+1=2\)。\(\lfloor d\rfloor\) 是向下取整，也就是不大於 d 的最大整數。

1.25 離 1 差 0.25、離 2 差 0.75，越近的 bin 分到越多：bin1 的權重是 \(2-1.25=0.75\)，bin2 的權重是 \(1.25-1=0.25\)。寫成四個 bin 的 target 分佈，就是 \(t=[0,\ 0.75,\ 0.25,\ 0]\)。它的加權平均 \(0.75\times1+0.25\times2=1.25\)，剛好等於真值。一般式是 bin i 拿 \((i+1)-d\)，bin i+1 拿 \(d-i\)。這叫線性內插：只用相鄰兩格時，這是唯一讓兩個權重加總為 1、加權平均又恰好等於 d 的分法。

logits 的 shape 是 `[N,K]=[1,4]`：N 是要監督的距離（邊）數，本例 1 條；K 是每條邊的 bin 數，本例 4。softmax 沿 bin 軸（最後一軸）做，得到 \(p_0,p_1,p_2,p_3\)，四個值相加為 1。

loss 是 \(-0.75\ln p_1-0.25\ln p_2\)。本節的 log 都是自然對數 ln。一般式是

\[
\mathrm{DFL}=-\sum_{k=0}^{K-1}t_k\ln p_k
\]

一般的分類題只有正確類別的 \(t_k=1\)、其餘都是 0，式子只剩一項；這裡正確答案拆給兩個 bin，所以有兩項。DFL 不是先把 1.25 四捨五入成 1，也不是只對期望值做平方誤差（只看期望值哪裡不夠，見下方〈Loss 和期望值各負責什麼〉）。

解碼時，預測距離是期望值 \(\hat d=\sum_{k=0}^{K-1}k\,p_k\) 格（帽子表示模型的預測，和第 0 章的 \(\hat y\) 一樣），換回畫素是 \(\hat d\times\text{stride}\)。

程式如下。它是完整程式 `dfl`、`expected_distance` 兩個函式的簡化版：完整程式把 `wr` 叫 `weight_right`，K 寫成 `logits.shape[-1]`，還多一行範圍檢查（見〈DFL 的代價〉）。

```python
# 已有 logits（shape [N,K]=[1,4]）與 target（[1.25]，格）；本例 K=4，是每條邊的 bin 數
# left、right 指 bin 軸（數線）上較小／較大的相鄰 bin，和框的左、右邊無關
left = target.floor().long()  # 向下取整，轉成整數索引：1
right = left + 1              # 2
wr = target - left            # right 的權重 d−i＝0.25；left 的權重是 1−wr＝0.75
# reduction='none'：先不平均，回傳每條邊各自的 CE（shape [N]），逐條乘權重後才 mean
loss = ((1-wr) * F.cross_entropy(logits, left, reduction='none')
        + wr * F.cross_entropy(logits, right, reduction='none')).mean()
# torch.arange(K)＝[0,1,2,3]，是各 bin 代表的格數；乘上機率再沿 bin 軸加總，就是期望值
distance = (logits.softmax(-1) * torch.arange(K)).sum(-1)
```

`F.cross_entropy(logits, c)` 就是第 7 章的 \(-\ln p_c\)，所以兩項合起來正好是 \(-0.75\ln p_1-0.25\ln p_2\)。ltrb 四條邊各有自己的一對 `left`、`right` bin。

為什麼不直接取機率最大的 bin（argmax）？argmax 只能得到整數格（1 格＝8 畫素），表示不了 1.25。它也像第 2 章〈[訓練診斷](02-diagnostics.md)〉說的階梯函數：logits 稍微改變，結果通常不變，梯度幫不上忙。期望值能落在刻度之間，又可以微分；在完整 detector 裡，IoU loss 的梯度才能經由期望距離傳回這些 logits。

初始 logits 全 0，四個機率各 0.25，期望距離是 \(0.25\times(0+1+2+3)=1.5\) 格。DFL 是 \(-(0.75+0.25)\ln0.25=\ln4\approx1.386294\)。用計算機的 log 鍵（通常以 10 為底）算 4，會得到 0.602，不是這個值。

加權 cross entropy 對 logits 的梯度是「預測機率−target 分佈」：初始預測 `[0.25, 0.25, 0.25, 0.25]` 逐個 bin 減掉 \(t\)＝`[0, 0.75, 0.25, 0]`，得到 `[0.25, −0.5, 0, 0.25]`。推導與成立條件見下方摺疊區。不會微分也能驗算：把 bin1 的 logit 從 0 改成 0.001，loss 從 1.386294 變成約 1.385794，少了約 0.0005；除以 0.001，斜率約 −0.5，正是 bin1 的梯度。

梯度的正負號指出方向。SGD 往梯度的反方向走：bin1 的梯度是負的，它的 logit 會被拉高。bin0、bin3 不在 loss 式子裡，梯度卻是正的，會被壓低。原因是 softmax 的分母 \(\sum_k e^{z_k}\)（\(z_k\) 是 bin k 的 logit）把所有 logits 綁在一起：bin0 或 bin3 的 logit 變大，\(p_1\)、\(p_2\) 就變小，loss 跟著變大。bin2 的梯度此刻是 0，因為它目前的 0.25 恰好等於 target；這不代表 bin2 永遠不更新，其他 logits 一變，softmax 的機率就跟著變。

??? note "推導：為什麼梯度是「預測機率−target 分佈」"

    softmax 是 \(p_k=e^{z_k}/\sum_m e^{z_m}\)（分母和正文的 \(\sum_k e^{z_k}\) 一樣，只是加總代號改用 m，免得和 \(p_k\) 的 k 混淆）。兩邊取 ln：

    \[
    \ln p_k=z_k-\ln\sum_m e^{z_m}
    \]

    代入 DFL。target 權重加總為 1（\(\sum_k t_k=0.75+0.25=1\)），所以第二項的係數剛好是 1：

    \[
    L=-\sum_k t_k\ln p_k=-\sum_k t_k z_k+\Bigl(\sum_k t_k\Bigr)\ln\sum_m e^{z_m}=-\sum_k t_k z_k+\ln\sum_m e^{z_m}
    \]

    對 \(z_j\) 偏微分。第一項只有 \(k=j\) 那一項含 \(z_j\)，得 \(-t_j\)。第二項用連鎖律：\(\ln u\) 的導數是 \(u'/u\)；這裡 \(u=\sum_m e^{z_m}\)，對 \(z_j\) 的導數是 \(e^{z_j}\)，所以得到 \(e^{z_j}/\sum_m e^{z_m}=p_j\)。合起來：

    \[
    \frac{\partial L}{\partial z_j}=p_j-t_j
    \]

    成立條件有兩個：target 權重加總為 1；本例只有一條邊（N=1）。N 條邊一起算時，本節程式最後的 `.mean()` 會除以 N，每個 logit 的梯度也要再除以 N，和 12.3 節 mean BCE 的梯度要除以 4 同理。

## Loss 和期望值各負責什麼

完整程式（Colab 裡那份）沒有圖片，也沒有 CNN：它把 4 個 logits 直接設成 `nn.Parameter`（預設會算梯度的 tensor，12.1 節用過），從全 0 開始。程式先反傳初始 loss，用斷言（assert）逐值核對上面的梯度，再用 SGD、學習率 0.5 更新 400 次。

最後 bin1 接近 0.75、bin2 接近 0.25，其他 bin 接近 0，期望值在 1.25 格附近；乘 stride 後接近 10 畫素。logits 是有限的數，softmax 永遠不會讓 bin0、bin3 的機率真正變成 0；它們可能把期望值稍微拉偏，所以只要求接近 1.25，不要求完全相等。

只看期望值夠不夠？看兩份分佈 a＝`[0, 0.75, 0.25, 0]` 與 b＝`[0.375, 0, 0.625, 0]`。它們的期望值都是 1.25（b 是 \(0.375\times0+0.625\times2=1.25\)），只看期望距離分不出來。把它們當成預測，代入 target 為 1.25 的 DFL 就分得出來：

- a：\(-0.75\ln0.75-0.25\ln0.25\approx0.562\)。target 為 1.25 時，DFL 最低就是這個值（約 0.562），不是 0。理由是 cross entropy 在預測分佈等於 target 分佈時最小，最小值是 \(-\sum_k t_k\ln t_k\)，而 a 正好就是 target 分佈。式中 bin0、bin3 的權重 \(t_k\) 是 0，這兩項算 0、不計入，所以只剩 bin1、bin2 兩項，也就是開頭算出的約 0.562。用數字對照：換成和 target 差一點的 `[0, 0.8, 0.2, 0]`，DFL 是 \(-0.75\ln0.8-0.25\ln0.2\approx0.570\)，比 0.562 大。
- b：\(p_1=0\)，\(-0.75\ln0\) 是無限大。

只看期望值，兩者一樣好；DFL 卻強烈排斥 b，把機率推到真值兩側。分工可以用兩句話說：DFL 在訓練時決定每條邊的機率形狀；期望值把分佈變回一個距離，用來畫框和算 IoU loss。

放回完整 detector：改用 DFL 後，12.2 節的框分支每個候選要輸出 4×K 個數（原本是 4 個），shape 是 `[B,P,4×K]`；B、P 是 12.1 的圖片數與候選數。先 reshape 成 `[B,P,4,K]`，四條邊各 K 個，softmax 要沿最後的 bin 軸做。完整模型只對正樣本算 DFL（12.3 節：背景不計框回歸 loss）；把正樣本的四條邊攤平後，就是本節的 `[N,K]`，N＝正樣本數×4。IoU loss 則透過期望距離與解碼後的框，衡量整個框的幾何；DFL 約束各邊的分佈。

開頭提過，原論文認為分佈能表達邊界的模糊。分佈攤得很開，看起來像模型對這條邊沒把握；但攤得多開，是不是真的對應誤差多大（這叫校準），要另外用資料檢查：攤得越開的邊，實際誤差是否真的越大。沒驗證前，不能把它當成可信的信心指標。

執行 `PYTHONPATH=. python lesson_cases/12-dfl.py`，檢查四個機率相等時的期望值（輸出中的 uniform expectation）為 1.50、DFL 為 1.386294、梯度為 `[0.25, −0.5, 0, 0.25]`，以及學到的期望值與 1.25 相差不到 0.02 格。程式真的做 backward 與 step，只需 CPU。

## DFL 的代價：距離有上限、輸出變多

四個 bin 的期望距離只能落在 `[0,3]` 格。本例的 DFL target 還要更嚴，必須嚴格小於 3，才能有較大的相鄰 bin。target 等於 3 時，較大的相鄰 bin 是 bin4，權重是 3−3＝0。權重是 0 也沒用：`F.cross_entropy` 照樣要讀 bin4 這個索引，而 bin 只有 0～3，於是報錯（IndexError）。所以完整程式在算 loss 之前，先用一行斷言檢查範圍，超出就報錯停下，讓你直接看到這個限制：

``` { .python data-excerpt="lesson_cases/12-dfl.py" }
assert ((target >= 0) & (target < logits.shape[-1] - 1)).all()  # 0 ≤ target < K−1，本例即 < 3
```

Ultralytics 的官方實作不報錯，而是用 clamp（把超出範圍的值夾回範圍內）：target 被夾到 K−1−0.01，本例 K=4 時是 2.99。clamp 只防程式讀到不存在的 bin。照官方做法，3.2 格會被當成 2.99 格來學，這條邊被截短，框會學得比實際小，而且不會有錯誤訊息。所以 target 超出範圍時，要擴增 K 或改尺度（見自主練習），不能只靠 clamp。

Ultralytics 的偵測 head 預設每條邊 K=16 個 bin，它的程式把每邊的 bin 數 K 叫 reg_max：16 代表 bin 0～15，最大 15 格（clamp 上限是 14.99）。stride 8 時，期望距離最多 \(15\times8=120\) 畫素；同樣 K=16，stride 16 最多 240 畫素，stride 32 最多 480 畫素。大框可以交給大 stride 的候選負責（第 10 章〈[YOLOv3 多尺度](10-multiscale.md)〉），所以多尺度與候選選擇會影響是否涵蓋大框，不能用一個小尺度代表整個模型。

輸出也變多。對兩類分類、100 個候選，直接四邊輸出需要 `100×(4+2)=600` 個數；K=16 的 DFL head 每個候選有 4×16＝64 個距離 logits，需要 `100×(64+2)=6600` 個數。這是 raw 輸出（head 最後的原始輸出）大小，還未算中間卷積。分佈表示有更細緻的監督，也多了 softmax 與期望值這些運算；之後要把模型轉成能在其他裝置上執行的格式（匯出，第 20 章）時，這些運算也都要被支援。這個小例子沒有比較真實 AP，不能斷言額外成本一定值得。

## 常見錯誤

- **softmax 沿錯軸**：softmax 要沿長度 K 的 bin 軸（最後一軸）做，不是沿候選軸 P。沿候選軸做，會變成不同候選互相分機率。
- **四條邊一起做 softmax**：先 reshape 成 `[4,K]`，四條邊各自做 softmax，讓每邊 K 個機率加總為 1；不要把 4×K 個 logits 一起做。
- **多乘一次 stride**：期望值（格）乘一次 stride 就是畫素；已經是畫素的距離，別再乘一次。
- **把 DFL 當成物件類別的分類 loss**：DFL 確實是加權 cross entropy，但分類的對象是距離 bin，監督的是距離刻度，不是紅矩形／藍矩形這種物件類別。
- **把 Ultralytics 的 reg_max 當成最大 bin 值**：Ultralytics 的 reg_max 是 bin 數 K，bin 從 0 起算，最大值是 K−1。GFL 論文用的 mmdetection（另一套開源的物件偵測程式庫）實作卻把 reg_max 定為最大值（16 代表 0～16 共 17 個 bin）。看程式時要先確認是哪一種，並在實作旁標清楚：bin 數 K 與最大值 K−1 差一（off-by-one，寫程式常見的差 1 錯誤）。
- **先把 CE 平均再乘權重**：兩份 CE 要先用 `reduction='none'` 保留每條邊各自的值，逐條乘上自己的相鄰 bin 權重之後才取 mean。先平均會把不同邊的 target 權重混在一起，例子見下方摺疊區。

??? note "先平均會出什麼錯：兩條邊的例子"

    本節只有一條邊（N=1），先平均或後平均結果一樣，看不出差別。換成兩條邊：

    - 邊 A：target 1.25，`left`＝bin1、`right`＝bin2，權重 0.75、0.25；softmax 後的預測機率是 `[0.1, 0.6, 0.2, 0.1]`。
    - 邊 B：target 2.6，`left`＝bin2、`right`＝bin3，權重 0.4、0.6；預測機率是 `[0.1, 0.1, 0.3, 0.5]`。

    正確做法（`reduction='none'`）讓每條邊用自己的 CE 乘自己的權重：

    - 邊 A：\(0.75\times(-\ln0.6)+0.25\times(-\ln0.2)\approx0.785\)
    - 邊 B：\(0.4\times(-\ln0.3)+0.6\times(-\ln0.5)\approx0.897\)

    兩條平均約 0.8415。

    錯誤做法是拿掉 `reduction='none'`。`F.cross_entropy` 預設會取平均，兩條邊的 `left` CE 先被平均成一個數 \((-\ln0.6-\ln0.3)/2\approx0.857\)，`right` CE 也被平均成 \((-\ln0.2-\ln0.5)/2\approx1.151\)。再乘上兩條邊各自的權重、取平均，得到約 0.9823。A 的權重被乘到混了 B 的平均 CE 上，結果就錯了。

## 自主練習

1. 把 target 改成 2.6 格，K 仍為 4。先手算：哪兩個 bin 有權重、各是多少？初始梯度是多少？再到完整程式（`lesson_cases/12-dfl.py`，也就是 Colab 裡那份）的 `main()` 修改 `target`，執行核對。跟著要改三處：梯度斷言用的 `wanted_grad`、期望值和 1.25 比較的斷言，以及檢查 bin 機率的斷言（`probs[0, 1] > .73 and probs[0, 2] > .24` 那一行）。新的寫法先自己想。
2. 若 target 改成 3.2 格，程式會怎樣？該怎麼處理？
3. 為什麼「學到的期望值誤差很小」不足以驗證 DFL 實作正確？

??? note "參考答案"

    **第 1 題**：\(\lfloor2.6\rfloor=2\)，相鄰的是 bin2 與 bin3。bin2 權重 \(3-2.6=0.4\)，bin3 權重 \(2.6-2=0.6\)，target 分佈是 `[0, 0, 0.4, 0.6]`。初始梯度＝`[0.25, 0.25, 0.25, 0.25]`−`[0, 0, 0.4, 0.6]`＝`[0.25, 0.25, −0.15, −0.35]`。

    完整程式 `main()` 要改的四行如下：

    ```python
    target = torch.tensor([2.6])                          # 原本是 [1.25]
    wanted_grad = torch.tensor([[.25, .25, -.15, -.35]])  # 原本是 [[.25, -.50, .0, .25]]
    assert abs(distance.item() - 2.6) < .02               # 原本和 1.25 比，容差仍是 .02
    assert abs(probs[0, 2] - .4) < .02 and abs(probs[0, 3] - .6) < .02  # 取代 probs[0, 1] > .73 那一行
    ```

    改完執行，斷言全部通過。印出的初始梯度裡，後兩個值是 −0.15000009…、−0.34999990…，不是剛好 −0.15、−0.35。這不是算錯：`torch.tensor([2.6])` 預設存成 float32（32 位元浮點數），而 float32 存不下剛好的 2.6，只能存最接近的 2.5999999…（第 4 章〈[座標轉換與還原](04-coordinates.md)〉講過 float32 存不準大多數小數），bin2、bin3 的權重因此各差了不到一千萬分之一。梯度斷言用 `torch.allclose` 逐項比較，每一項的差距都在它預設的容差內就算相等，所以照樣通過。學到的機率那一行沒有這種尾巴，因為程式先用 `.double()` 換成位數更多的 float64（64 位元浮點數），再四捨五入到小數 4 位才印。

    學到的機率接近 target 分佈 `[0, 0, 0.4, 0.6]`，bin0、bin1 只剩很少的機率。期望值略小於 2.6：殘留的少量機率都在 bin0、bin1，把平均往下拉，但仍在 0.02 的容差內。程式最後示範「兩份分佈同期望值 1.25」的 a、b，是獨立的人工例子，和本題的 2.6 無關，保留原值即可，不要把它改成訓練 target。

    **第 2 題**：3.2 格超過 bin3，完整程式的範圍斷言會直接報錯停下。不能讓程式讀不存在的 bin4，也不該靠 clamp 把它截成 2.99 格。應擴增 K，或重新設計尺度：3.2 格在 stride 8 是 25.6 畫素；改由 stride 16 的候選負責，就只有 1.6 格，K=4 也放得下。

    **第 3 題**：兩份不同的分佈可以有同樣的期望值，例如上面的 a、b 都是 1.25。期望值對了，分佈形狀仍可能錯，所以還要檢查各 bin 的機率。

參考來源：[Generalized Focal Loss 原論文](https://arxiv.org/abs/2006.04388)、[Ultralytics DFLoss 的相鄰 bin 加權](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)、[DFL 期望值模組](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py)、[mmdetection GFL head 的 reg_max 定義](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/gfl_head.py)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-05 在 INTEL(R) XEON(R) PLATINUM 8573C（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-dfl.json)

??? example "展開本次實際輸出"

    ```text
    uniform expectation=1.50; DFL=1.386294
    initial logit gradient: [[0.25, -0.5, 0.0, 0.25]]
    learned bin probabilities: [[0.0025, 0.7475, 0.2474, 0.0025]]
    learned expectation=1.2500 cells = 9.9999 pixels at stride 8
    different distributions can share expectation=1.25
    ```

<!-- curriculum-evidence:end -->
