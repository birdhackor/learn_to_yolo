# C.12.4 DFL：把一條邊距離學成分佈

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/12-dfl.ipynb){ .md-button }

目前每個候選的四距離各只輸出一個數。邊界模糊或被遮住時，仍要輸出框，但我們也可能希望在邊附近保留多個位置的相對權重，而非在中間立刻壓成單一距離。2020 年的 [Generalized Focal Loss（GFL，廣義聚焦損失）](https://arxiv.org/abs/2006.04388)因此提出每邊一份分佈；使用時再取一個距離畫框。

本頁沿用 anchor-free 的點到邊距離，只拿其中一邊，把一個連續數改成多個刻度的機率。**DFL（Distribution Focal Loss，分佈聚焦損失）**是教這份分佈的 loss，不是物件類別 loss。

原論文把分佈與邊界模糊連起來；本例沒有圖片與模糊邊界，只驗證表示、監督與解碼，不證明準確度改善，也不把分佈寬度直接當成可信的不確定程度。

## 1.25 格不是第 1 類

另取一條左邊距離 10pixel，stride 8，所以真值 d=10/8=**1.25 格**；這不是上一頁某個 GT 的數字，只保留同一距離意義。把尺設為 0、1、2、3 格，這四個刻度叫 **bin**。模型為各 bin 各輸出 logit，再沿 bin 軸 softmax 成機率 p0～p3，總和 1。

1.25 介於 bin 1、bin 2。若直接四捨五入成 1，就丟掉 0.25 格；因此用兩相鄰刻度共同表達它：bin 1 權重 2−1.25=0.75，bin 2 權重 1.25−1=0.25。兩權重總和 1，加權距離 `0.75×1+0.25×2=1.25`。

| bin 代表距離（格） | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| target t | 0 | 0.75 | 0.25 | 0 |
| 全零 logits 的初始 p | 0.25 | 0.25 | 0.25 | 0.25 |

一般取 i=floor(d)，向下取整；bin i 拿 `(i+1)−d`，bin i+1 拿 `d−i`。只用這兩 bin 時，這是讓權重和 1、加權距離恰好 d 的線性內插，不是把小數當物件類別編號。

訓練用加權 cross entropy：

\[
\begin{aligned}
\mathrm{DFL}&=-\sum_{k=0}^{K-1}t_k\ln p_k\\
&=-0.75\ln p_1-0.25\ln p_2.
\end{aligned}
\]

K 是 bin 數，本例 4。普通分類只有一類 target 1，DFL 將答案分給兩刻度；log 均為自然對數 ln。logits shape `[N,K]=[1,4]`，N 是監督的邊數，目前 1。

解碼不是取 argmax，而是期望距離，單位仍是格：

\[
\begin{aligned}
\hat d&=\sum_{k=0}^{K-1}k p_k\\
&=0p_0+1p_1+2p_2+3p_3.
\end{aligned}
\]

乘一次 stride 即可換 pixel。argmax 只有整數格，不能回到 1.25，且名次沒變時輸出不變、無可用梯度；期望值能落在刻度間並可微，完整偵測器的 IoUloss 纔可經它傳回 logits。

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

這是完整 `dfl`、`expected_distance` 的簡寫，wr 對應 `weight_right`，K 來自最後一軸長度，完整版本還檢查範圍。這段的 left、right 指數線上相鄰 bin，不是框左、右邊。CE 必須先 `reduction='none'`保留**逐邊**loss，乘各自權重後才 mean；不是先把不同邊混成平均 CE。

??? note "名字裡的 Focal 是什麼意思"

    Focal 是「聚焦」：原論文說，DFL 讓網路很快把機率集中在真值兩側最近的兩個刻度。DFL 的公式沒有 focal loss 常見的額外調整項，就是加權的 cross entropy。論文把 focal loss、同一篇提出的 QFL（Quality Focal Loss，用在分類分支）和 DFL 都看成 Generalized Focal Loss（focal loss 的推廣）的特例；DFL 是不帶調整項的那一種，因為框只在正樣本上學，沒有類別不平衡的問題，所以只留下 cross entropy 的部分。

## 全零 logits 如何朝兩個刻度靠近

初始 p 各 0.25，期望為 `(0+1+2+3)/4=1.5格`；DFL 為 `−(0.75+0.25)ln0.25=ln4≈1.386294`。加權 CE 對每 logit 的梯度爲 **p−t**，這裏是 `[0.25,−0.5,0,0.25]`。條件是 target 權重和 1、N=1；N 條邊取 mean，每個梯度還要除 N。

SGD 減梯度，bin 1 logit 上升，bin 0 與 bin 3 下降。它們的 target 雖 0，仍影響 softmax 分母；若其 logit 上升，p1、p2 就變小、loss 變大，所以梯度不為 0。bin 2 此刻 p=target=0.25，梯度恰 0，但其他 logit 更新後 p2 會變，並非永不更新。

不會微分也可用小改變核對 bin 1：其 logit 從 0 到 0.001，loss 約從 1.386294 降到 1.385794，斜率約−0.5。完整程式先對初始 loss 反傳，用 assert 逐值核對四梯度，再用 lr 0.5 的 SGD 更新 400 次。

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

執行 `PYTHONPATH=. python lesson_cases/12-dfl.py` 或頁首 Colab。它沒有 CNN，直接訓練 4 個 `nn.Parameter` logits。保存結果的機率為 `[0.0025,0.7475,0.2474,0.0025]`，期望約 1.2500 格，即 stride 8 下 9.9999pixel；斷言檢查與 1.25 差小於 0.02 格及主要 bin 機率。有限 logits 的 softmax 不會讓另外兩 bin 真正為 0，所以只要求接近答案。

## Loss 和期望值各負責什麼

期望值正確就足夠嗎？比較兩份人工分佈：a=`[0,0.75,0.25,0]`，b=`[0.375,0,0.625,0]`。a 是 `0.75×1+0.25×2=1.25`，b 是 `0.375×0+0.625×2=1.25`；若只看解碼距離，兩者相同。

DFL 卻分得出來。a 的 loss 爲 `−0.75ln0.75−0.25ln0.25≈0.562`；b 的 p1=0，真值卻要求 bin 1 有權重，loss 爲無限大。a 已經是 target 分佈，cross entropy 的最小值就是 target 熵約 0.562，**不是 0**；換成 `[0,0.8,0.2,0]`，loss 約 0.570，反而較高。權重 0 的兩項不計，所以沒有 `0×ln0` 的需求。

DFL 在訓練時把機率集中到 GT 兩側，約束每邊的分佈形狀；期望值把它壓回一個可解碼距離；整框 IoUloss 再檢查四邊合起來的幾何。只輸出多 channel 沒有完成這三項工作，只有期望誤差小也不足以驗證 DFL。

放回 [解耦 head](12-decoupled-head.md)，框分支每候選從 4 個數變 4×K 個 logits，shape `[B,P,4×K]`，reshape 為 `[B,P,4,K]`，四邊**各自**沿 K 軸 softmax。只對 [assignment](12-assignment.md) 選出的正樣本算 DFL；攤平為本例 `[N,K]`時，N=正樣本數×4。背景沒有框監督，分類與品質 target 是另一分支的工作。

分佈較寬可能看起來像沒把握，但還需檢查較寬者是否實際誤差較大，這叫校準。本例只給固定相鄰 bin 監督，沒有這份校準證據。

??? note "先平均會出什麼錯：兩條邊的例子"

    本節只有一條邊（N=1），先平均或後平均結果一樣，看不出差別。換成兩條邊：

    - 邊 A：target 1.25，`left`＝bin 1、`right`＝bin 2，權重 0.75、0.25；softmax 後的預測機率是 `[0.1, 0.6, 0.2, 0.1]`。
    - 邊 B：target 2.6，`left`＝bin 2、`right`＝bin 3，權重 0.4、0.6；預測機率是 `[0.1, 0.1, 0.3, 0.5]`。

    正確做法（`reduction='none'`）讓每條邊用自己的 CE 乘自己的權重：

    - 邊 A：\(0.75\times(-\ln0.6)+0.25\times(-\ln0.2)\approx0.785\)
    - 邊 B：\(0.4\times(-\ln0.3)+0.6\times(-\ln0.5)\approx0.897\)

    兩條平均約 0.8415。

    錯誤做法是拿掉 `reduction='none'`。`F.cross_entropy` 預設會取平均，兩條邊的 `left` CE 先被平均成一個數 \((-\ln0.6-\ln0.3)/2\approx0.857\)，`right` CE 也被平均成 \((-\ln0.2-\ln0.5)/2\approx1.151\)。再乘上兩條邊各自的權重、取平均，得到約 0.9823。A 的權重被乘到混了 B 的平均 CE 上，結果就錯了。

## DFL 的代價：距離有上限、輸出變多

K=4 的期望只能在 0～3 格。本程式的 target 還必須**0≤d<3**：d=3 時會讀取 right=bin 4，即使它的權重是 0，CE 仍先檢查索引而報 IndexError。loss 前因此有範圍 assert：

``` { .python data-excerpt="lesson_cases/12-dfl.py" }
assert ((target >= 0) & (target < logits.shape[-1] - 1)).all()  # 0 ≤ target < K−1，本例即 < 3
```

Ultralytics 用 clamp 把 target 夾到 `K−1−0.01`，K4 是 2.99。這隻保護索引，不會增加可表示距離：3.2 會被當 2.99 學，截短那條邊、沒有錯誤訊息。若常超範圍，要增加 K 或改候選尺度，不是隻靠 clamp。

官方默認 K=16，每邊 bin 0～15，期望最多 15 格，target clamp 上限 14.99。stride 8 最多 120pixel，stride 16 為 240、stride 32 為 480；大框可用較大 stride 候選，多尺度與 assignment 都會影響覆蓋，不能從一個小尺度判整模型。

對 100 候選、兩類，直接四距離 raw 輸出 600 個數；K16 則 4×16=64 個距離 logits，raw 輸出共 `100×(64+2)=6600`，還未算中間卷積。也新增 softmax 與期望運算，匯出部署須支援它們。這份成本是否值得，需要定位品質與部署實測；本例沒有 AP 對照。

Ultralytics 的 `reg_max` 指**bin 數 K**，16 代表 0～15；GFL 所用 mmdetection 則指最大 bin 值，16 代表 0～16 共 17bin。閱讀程式要先核這個差 1 約定，不能只看名稱。還要注意 softmax 沿 bin、不沿候選，也不要四邊共用一個 softmax，或已換 pixel 後再乘 stride。

## 自主練習

1. 把 target 改成 2.6 格，K 仍為 4。先手算：哪兩個 bin 有權重、各是多少？初始梯度是多少？再到完整程式（`lesson_cases/12-dfl.py`，也就是 Colab 裡那份）的 `main()` 修改 `target`，執行核對。跟著要改三處：梯度斷言用的 `wanted_grad`、期望值和 1.25 比較的斷言，以及檢查 bin 機率的斷言（`probs[0, 1] > .73 and probs[0, 2] > .24` 那一行）。新的寫法先自己想。
2. 若 target 改成 3.2 格，程式會怎樣？該怎麼處理？
3. 為什麼「學到的期望值誤差很小」不足以驗證 DFL 實作正確？

??? note "參考答案"

    **第 1 題**：\(\lfloor2.6\rfloor=2\)，相鄰的是 bin 2 與 bin 3。bin 2 權重 \(3-2.6=0.4\)，bin 3 權重 \(2.6-2=0.6\)，target 分佈是 `[0, 0, 0.4, 0.6]`。初始梯度＝`[0.25, 0.25, 0.25, 0.25]`−`[0, 0, 0.4, 0.6]`＝`[0.25, 0.25, −0.15, −0.35]`。

    完整程式 `main()` 要改的四行如下：

    ```python
    target = torch.tensor([2.6])                          # 原本是 [1.25]
    wanted_grad = torch.tensor([[.25, .25, -.15, -.35]])  # 原本是 [[.25, -.50, .0, .25]]
    assert abs(distance.item() - 2.6) < .02               # 原本和 1.25 比，容差仍是 .02
    assert abs(probs[0, 2] - .4) < .02 and abs(probs[0, 3] - .6) < .02  # 取代 probs[0, 1] > .73 那一行
    ```

    改完執行，斷言全部通過。印出的初始梯度裡，後兩個值是 −0.15000009…、−0.34999990…，不是剛好 −0.15、−0.35。這不是算錯：`torch.tensor([2.6])` 預設存成 float32（32 位元浮點數），而 float32 存不下剛好的 2.6，只能存最接近的 2.5999999…（第 4 章〈[座標轉換與還原](04-coordinates.md)〉講過 float32 存不準大多數小數），bin 2、bin 3 的權重因此各差了不到一千萬分之一。梯度斷言用 `torch.allclose` 逐項比較，每一項的差距都在它預設的容差內就算相等，所以照樣通過。學到的機率那一行沒有這種尾巴，因為程式先用 `.double()` 換成位數更多的 float64（64 位元浮點數），再四捨五入到小數 4 位才印。

    學到的機率接近 target 分佈 `[0, 0, 0.4, 0.6]`，bin 0、bin 1 只剩很少的機率。期望值略小於 2.6：殘留的少量機率都在 bin 0、bin 1，把平均往下拉，但仍在 0.02 的容差內。程式最後示範「兩份分佈同期望值 1.25」的 a、b，是獨立的人工例子，和本題的 2.6 無關，保留原值即可，不要把它改成訓練 target。

    **第 2 題**：3.2 格超過 bin 3，完整程式的範圍斷言會直接報錯停下。不能讓程式讀不存在的 bin 4，也不該靠 clamp 把它截成 2.99 格。應擴增 K，或重新設計尺度：3.2 格在 stride 8 是 25.6 畫素；改由 stride 16 的候選負責，就只有 1.6 格，K=4 也放得下。

    **第 3 題**：兩份不同的分佈可以有同樣的期望值，例如上面的 a、b 都是 1.25。期望值對了，分佈形狀仍可能錯，所以還要檢查各 bin 的機率。

參考來源：[Generalized Focal Loss 原論文](https://arxiv.org/abs/2006.04388)、[Ultralytics DFLoss 的相鄰 bin 加權](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)、[DFL 期望值模組](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py)、[mmdetection GFL head 的 reg_max 定義](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/gfl_head.py)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-dfl.json)

??? example "展開本次實際輸出"

    ```text
    uniform expectation=1.50; DFL=1.386294
    initial logit gradient: [[0.25, -0.5, 0.0, 0.25]]
    learned bin probabilities: [[0.0025, 0.7475, 0.2474, 0.0025]]
    learned expectation=1.2500 cells = 9.9999 pixels at stride 8
    different distributions can share expectation=1.25
    ```

<!-- curriculum-evidence:end -->
