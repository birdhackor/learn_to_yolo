# 16.3 YOLO26 訓練補強：Progressive Loss、STAL 與 MuSGD

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/16-training.ipynb){ .md-button }

部署只用 one，不代表整段訓練應始終把相同比重交給 many 和 one。many 要教共用特徵，one 要學推論時的候選分數；兩者在不同訓練階段的需求可能不同。另一個問題更早發生：小物件裡若沒有任何候選點，它連正樣本訊號都拿不到。

本節沿這兩項工作拆解 YOLO26 的訓練設計。先讓兩個簡單 head 學同一條直線，觀察 loss 比重如何縮放更新、如何公平比較；再把小框放到格心圖上，觀察「有資格」和「被選中」的差別。最後才看優化器對更新方向做的另一種調整，避免把不同改動的結果混在一起。

## 在訓練的哪一段，多教哪個分支？

many、one 沿用前兩章的一對多、一對一 head。many 的較多正候選能提供較密集的特徵監督；one 的監督把責任集中到少數候選，服務本文選的 NMS-free 推論路徑（官方 API 要設 `nms=False`，見 [16.2](16-inference-head.md)）。YOLO26 的 Progressive Loss（漸進 loss）前期給 many 較大比重，後期往 one 移動。

這不是把 one loss 接回 backbone。官方 one 輸入仍 detach：many loss 教 backbone 和 many head，one loss 只教 one head。因此排程同時改兩種不同更新的尺度；後期強化 one，也會減少 many 對特徵的更新。轉移速度與終點是一項取捨，不能先假定所有資料都受益。

官方計算雙分支 loss 的 class 叫 `E2ELoss`；E2E 是 end-to-end（端到端），在這裡指 one 分支可直接給框，不靠 NMS 去重。固定版本預設 many 權重從 0.8 降到 0.1，one 從 0.2 升到 0.9。記 a、b 是兩份 loss 的倍數，而不是直線參數：

\[
L=aL_{\text{many}}+bL_{\text{one}},\qquad b=1-a.
\]

one head 參數 θ 不在 many 的計算裡，所以它的梯度是 `b×∂L_one/∂θ`。如果底下梯度相同，b=0.9 時更新幅度是 b=0.2 的 4.5 倍；實際訓練 θ 和預測一直變，梯度本身也在變，不能因此推論每個後期步都走更遠。

### 用 30 次更新把排程寫清楚

epoch 是訓練資料用過一輪。本例只有四筆，每輪一次全部使用、做一次 SGD，所以 E=30 輪就是 30 次更新。e 從 0 到 29：

\[
a(e)=\max\!\left(1-\frac e{E-1},0\right)(0.8-0.1)+0.1,
\qquad b(e)=1-a(e).
\]

分母 E−1 讓末輪 e=29 恰好到終點；max 在超過末輪時讓 a 停在 0.1，本例用不到超出情況。a 每輪降 0.7/29：首輪 0.8/0.2，中間 e=15 約 0.438/0.562，末輪 0.1/0.9。官方是在每個 epoch 結束才更新這組權重，真實一輪的多個 step 用同一組 a、b，不是每 step 都改。

## 先用一條已知答案，分辨 loss 權重的作用

現在把兩個 head 各縮成 `Linear(1,1)`，預測 wx+bias。四筆 features 是 −1、0、1、2，shape `[4,1]`；答案是 2x+1，即 −1、1、3、5。兩個 head 都學這份答案，各用 MSE，沒有共享 backbone，也沒有不同 assignment。這保留「總 loss 乘 a、b，梯度跟著縮放」；它沒有重現前期 many 教特徵、後期 one 使用特徵的協作。

每次用 seed=7 重設兩個 head 的初值，固定資料、SGD 學習率 0.05 和 30 次更新。先看程式如何在每輪接入權重；`...` 省略的是首步紀錄：

``` { .python data-excerpt="lesson_cases/16-training.py" }
def train(progressive, epochs=30, final_many=.1, fixed_weights=(.8, .2)):
    torch.manual_seed(7)  # 每次呼叫都重設 seed，所以三次的初始參數相同
    features = torch.tensor([[-1.], [0.], [1.], [2.]])
    target = 2 * features + 1
    many, one = nn.Linear(1, 1), nn.Linear(1, 1)
    optimizer = torch.optim.SGD(list(many.parameters()) + list(one.parameters()), lr=.05)
    history = []
    first_step = {}
    for epoch in range(epochs):  # epochs 就是 E=30
        # progressive=True 用上式的 a(e)、b(e)；False 則每輪都用同一組 fixed_weights
        a, b = weights(epoch, epochs, final_many) if progressive else fixed_weights
        optimizer.zero_grad()  # 清除上一輪累積的梯度
        lm = F.mse_loss(many(features), target)  # 上式的 L_many
        one_prediction = one(features)
        lo = F.mse_loss(one_prediction, target)  # 上式的 L_one
        # 下面三處 ... 省略的，都是只在第 0 輪執行、把首步數值記進 first_step 的程式
        ...
        loss = a * lm + b * lo
        loss.backward()
        assert many.weight.grad is not None and one.weight.grad is not None
        ...
        optimizer.step()
        ...
        history.append((a, b, float(lo.detach())))  # 每輪記下 a、b 和這輪更新前 one 的 MSE
    return F.mse_loss(one(features), target).item(), history, first_step
```

這個最基本 SGD 沒有動量或其他機制；one 每次更新等於用 `0.05×b` 的有效學習率。固定 b=0.2 時是 0.01；排程從 0.01 升到 0.045。a 完全不進入 one 的更新，不能從 one 的結果宣稱「早期多給 many」有幫助。真實官方結構加上 shared features、detach 與優化器後，不等於只調同一個學習率。

### 第一個 step，可以核對到哪裡？

排程首輪 b=0.2，one 的數值如下，均四捨五入到六位：

| one 分支計算 | 數值 |
| --- | --- |
| 初始 w／bias | 0.318423／0.313781 |
| 四筆 forward `wx+bias` | −0.004643、0.313781、0.632204、0.950627 |
| MSE | `mean((prediction−[-1,1,3,5])²)=5.866377` |
| 已乘 b 的 dw／dbias | −1.146190／−0.610803 |
| SGD 更新後 w／bias | 0.375733／0.344321 |

MSE 平方微分有 2，四筆平均除以 4，再乘 b=0.2。w 的梯度還乘 x，bias 的不乘：`dw=.2×(2/4)×sum((prediction−target)×x)`，`dbias=.2×(2/4)×sum(prediction−target)`。更新 `w_new=.318423−.05×(−1.146190)=.375733`。用表中捨入值手算，末位可能差 1；程式斷言用未捨入的數核對。

固定 0.8/0.2 的第一步與表完全相同；後面的等總量組 b=0.55，初始預測、MSE 相同，加權梯度卻是表中 2.75 倍，更新不同。紀錄的 `one_gain` 是 b，從 1−0.8 算出，在浮點中印 0.19999999999999996，只是儲存誤差，不是另一種權重。

## 排程較低的 MSE，是來自順序還是給得更多？

如果只拿排程和一直 0.8/0.2 比，兩個因素同時變了：b 每輪的大小，以及 30 輪給 one 的總量。排程 b 是從 0.2 到 0.9 的等差數列，總和 `30×(.2+.9)/2=16.5`；固定 0.2 只有 6。因為 b 直接縮放 one 的更新，這個總量差足以影響比較。

因此再加一組等總量對照：把排程 b 的平均 16.5/30=0.55 固定在每輪，many/one 即 0.45/0.55。它與排程有同樣初值、資料、步數、學習率和 b 總和，只剩每輪 b 的數值不同。完整程式由排程算出平均，不手填：

``` { .python data-excerpt="lesson_cases/16-training.py" }
def total_one_weight(history):
    """Sum of the one-head loss weight b over all epochs of one run."""
    return sum(b for _, b, _ in history)  # history 每輪一筆 (a, b, one 的 MSE)，這裡只加 b


def main():
    ...  # 省略：設定 seed 與執行緒數
    final_many = .1  # 自主練習只改這個值
    fixed, fixed_history, _ = train(False)  # 固定 0.8/0.2（fixed_weights 的預設值）
    progressive, history, first_step = train(True, final_many=final_many)  # 上式的排程
    # 排程的首末端點：第 0 輪 a=0.8，最後一輪 a=final_many
    assert abs(history[0][0] - .8) < 1e-7 and abs(history[-1][0] - final_many) < 1e-7
    matched_one = total_one_weight(history) / len(history)  # 排程 30 輪 b 的平均
    matched_weights = (1 - matched_one, matched_one)  # 本例是 (0.45, 0.55)
    matched, matched_history, _ = train(False, fixed_weights=matched_weights)  # 等總量對照組
    assert abs(total_one_weight(matched_history) - total_one_weight(history)) < 1e-9
    ...  # 省略：核對首步表的斷言、印出結果與 STAL 例子
```

三次訓練都在 30 次更新後重新 forward，計算同四筆資料的 one MSE；不是拿末次更新前的 loss：

| 權重設定 | 30 輪 b 總和 | 更新後 one MSE |
| --- | --- | --- |
| 固定 many/one 0.8/0.2 | 6.000 | 0.663073 |
| progressive | 16.500 | 0.016285 |
| 等總量固定 0.45/0.55 | 16.500 | 0.016882 |

大差距在等總量組也出現了，說明本例排程對固定 0.2 的大部分收益來自 one 得到更多加權更新。排程只比等總量組略低；這來自 b 每輪大小不同，不能歸因於早期 many 的幫助，因為兩個 head 互不相連。

本例的線性 MSE 還有一個特殊性：把同一組 30 個 b 改順序，one 結果不變。選讀用參數誤差的更新式推導這一點，也解釋有大有小的 b 為何比全部 0.55 略低；程式沒有跑倒序對照。這不是實際雙 head 網路的普遍性質。本例支持 loss 權重會改變分支的學習速度，沒有支持所有模型都應採此排程。

執行 `PYTHONPATH=. python lesson_cases/16-training.py`。前兩行對照首末權重；中間三行對照表中的 b 總和、one MSE；首步紀錄對照前一小節的 forward、梯度和更新。STAL 的輸出則是下面另一個資格例子，沒有混入三次線性訓練。

??? note "為什麼 b 每輪大小不同，one MSE 會略低一點"

    記 \(d=(w-2,\ \text{bias}-1)\)，也就是 one head 的參數離正確答案 w=2、bias=1 還差多少。這時 prediction−target 等於 \((w-2)x+(\text{bias}-1)\)，照上面算 dw、dbias 的方法（四筆 x 的和是 2、平方和是 6），還沒乘 b 的梯度是 \(Hd\)，其中

    \[
    H=\begin{bmatrix}3&1\\1&2\end{bmatrix}.
    \]

    所以每一輪的更新是 \(d\leftarrow d-0.05\,b\,Hd\)。

    H 有兩個特別的方向：\(u_1=(1,\ 0.618)\) 乘上 H，只會變成 3.618 倍；\(u_2=(1,\ -1.618)\) 乘上 H，只會變成 1.382 倍（都是近似值）。線性代數把這種方向叫特徵向量（eigenvector），倍數 λ 叫特徵值（eigenvalue）；這裡的特徵值和特徵圖上的特徵值是兩回事。把 d 寫成 \(c_1u_1+c_2u_2\)，每一輪的更新就只是把 \(c_1\) 乘上 \(1-0.05\,b\times3.618\)、把 \(c_2\) 乘上 \(1-0.05\,b\times1.382\)。三次訓練中最大的 \(0.05\,b\,\lambda\) 是 \(0.05\times0.9\times3.618\approx0.16\)，所以這些倍數都介於 0 和 1 之間。30 輪後，每個分量都乘上

    \[
    (1-0.05\,b_0\lambda)(1-0.05\,b_1\lambda)\cdots(1-0.05\,b_{29}\lambda),
    \]

    其中 \(b_e\) 是第 e 輪的 b，λ 是 3.618 或 1.382。由這個乘積可以看出兩件事：

    1. 乘法可以交換順序，30 個 b 不管怎麼排，乘積都一樣，所以 b 先小後大本身沒有好處。本節程式沒有跑把 b 倒過來排的對照，這一點是由推導得出的。
    2. b 的總和固定是 16.5 時，這 30 個倍數的總和也固定，是 \(30-0.05\lambda\times16.5\)。算幾不等式（正數的算術平均不小於幾何平均）說：總和固定的正數，全部相等時乘積最大。等總量對照組每輪的 b 都是 0.55，兩個分量都縮得最少；progressive 的 b 有大有小，縮得多一點。

    最後把 MSE 也寫成 \(c_1\)、\(c_2\)。用四筆 x 的和 2、平方和 6 展開平方可以驗證，one 的 MSE 是 \(\tfrac12\,d\cdot(Hd)\)（\(d\) 與 \(Hd\) 內積的一半）。代入 \(d=c_1u_1+c_2u_2\)、\(Hd=3.618\,c_1u_1+1.382\,c_2u_2\) 展開，含 \(c_1c_2\) 的項都乘著 \(u_1\cdot u_2=1\times1+0.618\times(-1.618)\approx0\)（兩個方向互相垂直），所以消失；one 的 MSE 等於 \(c_1^2\)、\(c_2^2\) 各乘一個固定的正數再相加。三次訓練開始時的 \(c_1\)、\(c_2\) 相同，progressive 的兩個分量都縮得比等總量對照組多，所以 progressive 的 one MSE 略低於等總量對照組。


## 小框沒有格心，如何先讓它有候選可選？

stride 8 的候選格心在 x、y 各為 4、12、20、…。真值 `[7,7,9,9]` 是中心 (8,8) 的 2×2 畫素框：附近四格心 (4,4)、(12,4)、(4,12)、(12,12) 全都在框外。依 [12.3](12-assignment.md) 的資格規則，點要嚴格落在 GT 內才能參加排名，這一層就沒有任何候選能當正樣本。

YOLO26 的 STAL＝Small-Target-Aware Label Assignment（照顧小物件的候選分配）只在資格檢查時擴張小框；標註本身不改。

![原真值、候選資格框與四個 stride 8 格心](../assets/diagrams/16-stal-candidates.svg)

圖的影像座標 x 向右、y 向下。小框仍是 `[7,7,9,9]`；以相同中心暫把資格框邊長擴成 16，得到 `[0,0,16,16]`，四個藍點都入池。程式斷言核對 0→4 個資格點，以及原 GT 沒變。這裡只看 stride 8；官方合併 stride 8、16、32 時，stride 16 的 (8,8) 本來就在原框內，不能說這個 GT 在所有尺度都無候選。

固定官方版本在 `TaskAlignedAssigner.select_candidates_in_gts` 裡，把寬或高小於 16 畫素的那一邊暫改成 16，中心不動。寬高分別判斷，所以 2×40 變 16×40，不是 16×16；16 來自 stride `[8,16,32]` 第二項。[論文](https://arxiv.org/abs/2606.03748) 公式 (5) 的門檻寫小於 8，程式寫小於 16；本節以固定程式為準，邊長 2 同時滿足兩者，所以例子數值不受差異影響。

資格框只是打開候選入口，接著仍要品質排序、top-k、衝突處理。固定版本的 many 每 GT 取 top-10；one 先取 top-7、解衝突，再每 GT 留最高品質的至多 1 個（`topk2=1`）。本例沒做這些步驟，說不出四點中誰成為正樣本；入池四點不等於四個正樣本，也不保證最後一定有一個。

### 候選在原框外，回歸什麼答案？

若 (4,4) 被選中，它要回歸原 `[7,7,9,9]`，不是資格框。四邊畫素距離為 `[4−7,4−7,9−4,9−4]=[−3,−3,5,5]`；l、t 為負，正好需要 [16.1](16-dfl-free.md) 的帶符號直接回歸。四個格心各有兩邊為負。

這接起了兩項設計：STAL 讓小物件比較容易有候選參與，DFL-free 讓框外候選也能表示到原邊界的答案。擴張會引入更多框外點，仍需品質排名與衝突篩選避免錯誤監督；若把回歸真值一起換大，模型反而會被教畫大框。資格增加只回答能不能參加，不是小物件 AP 的效果證明。

## 另一種調整：不只縮放梯度，也整理矩陣更新方向

前面的 a、b 是把 loss 梯度乘一個倍數，方向不因此改變。YOLO26 還提出 MuSGD（Muon 與 SGD 混合的優化器），對矩陣參數做另一種處理：先整理更新矩陣各方向的伸縮，再和 SGD 更新組合。這是更新規則的變化，與候選資格或兩份 loss 的配比不同。

動量（momentum）把前幾步梯度加權記下，讓當下更新也參考先前方向。Muon 先取動量形成的更新矩陣，再近似正交化，使不同方向的伸縮倍率靠近 1；MuSGD 將這份結果和另一份 SGD 動量更新按比例相加。矩陣線性層的 weight 是二維；卷積 weight `[out,in,kh,kw]` 則攤成 `[out,in×kh×kw]`。一維 bias 不走 Muon，只用 SGD。

整理方向是為了避免更新矩陣的少數強方向獨占步子。例如 `[[3,0],[0,.1]]` 的兩軸更新倍率差很多；理想正交化會變成單位矩陣，兩個倍率都是 1。官方用 Newton–Schulz 矩陣迭代近似這項運算，不是每次精確得到單位矩陣。具體名字與既有數值核對放在下方選讀。

??? note "Muon 怎麼整理更新量"

    Muon 的名稱來自 MomentUm Orthogonalized by Newton–Schulz，意思是「用 Newton–Schulz 正交化的動量」。

    它先把梯度做動量平均，得到這一步的更新矩陣。再把這個矩陣「正交化」：保留各個方向，把各方向的伸縮倍率（線性代數叫奇異值）都拉到接近 1，避免少數方向獨占這一步。例如更新矩陣是 \(\begin{bmatrix}3&0\\0&0.1\end{bmatrix}\) 時，第一個方向的倍率是 3，第二個方向只有 0.1，這一步幾乎只動第一個方向。理想的正交化結果是單位矩陣，兩個倍率都是 1。

    Newton–Schulz 只用矩陣乘法反覆計算幾次，來近似正交化。官方固定迭代 5 次，結果只是大致接近正交：上面的例子在既有官方程式核對中得到約 \(\begin{bmatrix}0.69&0\\0&1.13\end{bmatrix}\)（官方程式用 bfloat16 計算，這是只用 16 位元存一個數、精度較低的格式；改用一般的 float32 照樣算 5 次，第一個倍率約是 0.70），兩個倍率都拉到 1 附近，但不是精確的 1。

    最後，MuSGD 再加上另一份動量的一般 SGD 更新，兩種更新各乘一個比例後相加。


額外矩陣運算、第二份動量與參數分組會增加運算、記憶體和設定成本；混合比例、weight decay（每步把參數往 0 拉）也會影響結果。weight decay 和前面 many loss 比重遞減是兩件事。

[YOLO26 論文](https://arxiv.org/abs/2606.03748) §3.3.1、§4.3.2 報告從零訓練 COCO 時，MuSGD 500 epoch 的 mAP50–95 為 47.4，SGD 600 epoch 為 47.0（百分制）。這是作者在不同輪數的設定下報告的結果，本節沒有執行 MuSGD；三個線性 MSE 不支持優化器品質結論。自己的比較應固定資料、初始化、步數、增強與 loss 排程，再同時記錄收斂、更新尺度及成本。

??? note "選讀：預設排程與發布模型的訓練配方"

    本頁官方機制來自查核日 2026-10-02 固定的原始碼。官方訓練指南說發布模型先在 Objects365（365 類的大型偵測資料集）預訓練，再以 COCO 微調。資料、增強、超參數與內部設定共同構成訓練配方；這裡的 0.8→0.1 是程式預設，不是發布 checkpoint 的全部實際設定。

    部分內部超參數如 many loss 權重，只能在沒有合併到主線的實驗 Git branch 上設定，一般安裝套件不接受它們。這種程式版本分支，和 many／one 計算分支不同。本節沒有自行拼出 MuSGD 簡化版，也沒有重現發布模型的整套配方；預訓練模型與從零開始的 MiniYOLO 的 AP 差異不能只歸給優化器。

## 自主練習

自主練習：在完整程式（Colab 裡那份，或本機的 `lesson_cases/16-training.py`）只改 `main()` 裡的 `final_many = .1`，改成 `.3`，其他不動。先回答下面四題，再執行核對：

1. 第 29 輪的 many/one loss 權重是多少？
2. 第 0 輪的首步數值（上面那張表）會變嗎？為什麼？
3. 程式印出的 progressive `sum of b` 會變成多少？等總量對照組的 many/one 權重會變成多少？progressive 的 one MSE 會變大還是變小？
4. STAL 讓四個格心進入候選池後，YOLO26 的 one-to-one 分支最後每個 GT 最多有幾個正樣本？

??? note "參考答案"

    1. 0.3/0.7。e=29 時 \(a=0\times(0.8-0.3)+0.3=0.3\)，b=1−0.3=0.7。
    2. 不變。\(a(0)=1\times(0.8-0.3)+0.3=0.8\)，和終點無關，b 仍是 0.2；資料、seed 和初始參數也都沒變，所以首步數值和改之前（`final_many = .1`）完全相同。排程、核對終點的 assert 和印出的結果都用同一個 `final_many` 變數，不必改任何 assert。等總量對照組的權重 `matched_weights` 也由排程的 b 算出，會自動跟著變，程式照樣 assert 它和排程的 b 總和相等。
    3. progressive 印出 `sum of b=13.500`（改之前是 16.500）：b 改成從 0.2 線性增加到 0.7，30 輪的和是 30×(0.2+0.7)/2=13.5，第 1 輪起每一輪都比改之前小。等總量對照組自動變成 `matched fixed many/one (0.55, 0.45)`，`sum of b` 也是 13.500；固定 0.8/0.2 那一行完全不變。progressive 的 one MSE 變大，因為 one 的有效學習率從第 1 輪起每輪都比改之前小。等總量對照組的 one MSE 仍和 progressive 很接近、略高一點：b 總和相同時，b 每輪大小不同的 progressive 會略低（推導見正文的摺疊區）。多留給 many 的 loss 權重，在本例對 one 沒有幫助，因為兩個 head 互不相連。在真實模型（例如官方 YOLO26，backbone 只由 many 分支訓練）上值不值得這樣分配，要另做配對實驗才知道。
    4. 最多 1 個。4 只是有資格進池的點數；查核版本的 one-to-one 分支，最後每個 GT 最多只留品質最高的 1 個（topk2=1）。本例沒有做品質排序，說不出會是哪一個。

參考來源：[E2ELoss 排程](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)、[每個 epoch 結束才更新排程（trainer 呼叫 criterion.update）](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/engine/trainer.py)、[one 分支讀 detach 後的特徵（Detect.forward）](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)、[小物件資格與 top-1 篩選（topk2）](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py)、[MuSGD 公開實作](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/optim/muon.py)、[官方訓練配方（recipe）與重現範圍](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-training.json)

??? example "展開本次實際輸出"

    ```text
    many/one weight first: (0.8, 0.2)
    many/one weight last: (0.1, 0.9)
    fixed many/one (0.8, 0.2): sum of b=6.000, one-head MSE=0.663073
    progressive: sum of b=16.500, one-head MSE=0.016285
    matched fixed many/one (0.45, 0.55): sum of b=16.500, one-head MSE=0.016882
    one-head first forward/backward/step: {"weight": 0.31842339038848877, "bias": 0.3137805461883545, "prediction": [-0.004642844200134277, 0.3137805461883545, 0.6322039365768433, 0.950627326965332], "one_mse": 5.866377353668213, "one_gain": 0.19999999999999996, "weighted_dw": -1.1461899280548096, "weighted_dbias": -0.6108031272888184, "updated_weight": 0.3757328987121582, "updated_bias": 0.34432071447372437}
    small-object eligible points original / expanded: 0 4
    GT regression box remains: [7.0, 7.0, 9.0, 9.0]
    not a full YOLO26/STAL/MuSGD training reproduction
    ```

<!-- curriculum-evidence:end -->
