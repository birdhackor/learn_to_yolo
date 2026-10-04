# 16.3 YOLO26 訓練補強：Progressive Loss、STAL 與 MuSGD

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/16-training.ipynb){ .md-button }

YOLO26 官方介紹了三個訓練技巧：Progressive Loss、STAL 和 MuSGD。本節逐一說明三者在做什麼，只對第一個做實驗。讀完本節，你能說出兩個分支的 loss 權重怎麼隨訓練移動，能手算一步加了 loss 權重的 SGD 更新，也能說明小物件為什麼要放寬候選資格。

前置知識是 [13.1](13-dual-assignment.md) 的雙 head：訓練時有一對多（one-to-many，下面簡稱 many）和一對一（one-to-one，簡稱 one）兩個 head，本節也把它們叫做兩個分支。另外要知道 loss 權重（總 loss 裡每一項 loss 乘上的倍數）和 SGD。

NMS-free 推論（論文的預設；官方 API 要設 `nms=False`，見 [16.2](16-inference-head.md)）只用 one-to-one head，但這不表示訓練時兩分支的 loss 權重要從頭到尾固定不變。小物件也可能因為候選點排得太疏，一個候選都分不到，得不到正樣本的訓練訊號。三個技巧各管一件事：

- **Progressive Loss**（漸進調整兩分支的 loss 權重）：訓練中把總 loss 的比重，從 many 分支慢慢移到推論用的 one 分支。這是本節的主實驗。
- **STAL**（Small-Target-Aware Label Assignment，照顧小物件的候選分配）：小框裡可能一個候選點都沒有，所以選候選時暫時把小框放大。本節只算一個資格例子。
- **MuSGD**（Muon 與 SGD 混合的優化器）：改變矩陣參數的更新方式。官方說它讓訓練更穩定，本節沒有驗證，只做說明。

不要看到三個名稱就一次全加入，否則不知道結果由誰造成。本節的實驗只是兩個線性 head 的小例子，不是完整的 YOLO26 訓練；做了什麼、沒做什麼，整理在下面的摺疊區。

??? note "本節做了什麼、沒做什麼"

    - 文中的官方機制，以查核日（2026-10-02）固定版本的官方原始碼為準。
    - 實驗的起點是兩個線性 head，主要改動只有 loss 權重的排程。
    - 官方發布的 checkpoint（訓練好的模型檔）還用了預訓練、資料增強、超參數與內部設定；一段小實驗不能宣稱完整重現它。
    - 小實驗的 MSE 差距只展示 loss 權重的作用，不能當成 YOLO26 的 AP 增益。
    - 沒有執行 MuSGD，只做文字說明。
    - STAL 只檢查一個例子的資格遮罩（mask：標出哪些候選點有資格），沒有用 STAL 訓練偵測器，所以沒有小物件 AP 的結論。
    - MuSGD 和 STAL 都沒有混入 loss 權重對照的結果。

## Progressive Loss：總 loss 往推論分支移動

13.1 說過：one-to-many 讓每個真值物件可以有多個正候選；one-to-one 讓每個真值物件最多只選一個正候選。訓練早期，many 分支提供較密集的學習訊號；後期加重 NMS-free 推論模式所用的 one 分支。這就是這項排程的動機。代價是後期分給 many 的 loss 權重變少，而且要自己選轉移的速度與終點；不能假設同一種排程適合所有資料。

官方的排程寫在 `E2ELoss` 裡。`E2ELoss` 是官方程式中同時計算 many、one 兩分支 loss 的 Python class；E2E 是 end-to-end（端到端），指 one-to-one 分支可以直接輸出最終的框，不需要 NMS。查核版本的 `E2ELoss` 一開始給 many 的 loss 權重 0.8、one 0.2；訓練中 many 線性遞減到 0.1，one 升到 0.9。

寫成公式前，先定義符號。epoch（一輪）是把整份訓練資料用過一次（[11.3](11-augmentation.md) 說過）。本例只有 4 筆資料，一次就算完，所以一輪就是一次 SGD 更新，共 30 輪。e 是第幾輪（0～29），E=30 是總輪數。a、b 分別是 many、one 的 loss 權重，不是直線的斜率與截距：

\[
a(e)=\max\!\left(1-\frac{e}{E-1},\ 0\right)\times(0.8-0.1)+0.1,\qquad b(e)=1-a(e).
\]

分母用 E−1，是為了讓最後一輪 e=29 時，a 剛好降到 0.1。max(…,0) 只在 e 超過 E−1 時起作用，讓 a 停在 0.1；本例 e 最大是 29，用不到。每過一輪，a 減少 0.7/29≈0.024：第 0 輪是 0.8/0.2，第 15 輪約 0.438/0.562，第 29 輪是 0.1/0.9。官方也是在每個 epoch 結束時才調一次 a、b；真實訓練的一個 epoch 有很多步，這些步都用同一組 a、b。

loss 權重改變的，是兩個分支在 loss 和梯度裡各占多少。總 loss 是

\[
L=a\,L_{\text{many}}+b\,L_{\text{one}}.
\]

記 θ 為 one head 的參數（本例是 w 與 bias）。\(L_{\text{many}}\) 只由 many head 算出，不含 θ，所以 \(\partial L_{\text{many}}/\partial\theta=0\)：

\[
\frac{\partial L}{\partial\theta}
=a\,\frac{\partial L_{\text{many}}}{\partial\theta}+b\,\frac{\partial L_{\text{one}}}{\partial\theta}
=b\,\frac{\partial L_{\text{one}}}{\partial\theta}.
\]

第 0 輪 b=0.2，第 29 輪 b=0.9。若兩個時刻的 \(\partial L_{\text{one}}/\partial\theta\) 一樣大，後者每步走 4.5 倍遠。但 one head 越接近答案，\(\partial L_{\text{one}}/\partial\theta\) 本身會變小，所以後期每一步實際走多遠，不一定比前期大。

本例的兩個 head 都是 `Linear(1,1)`：每筆輸入 1 個數，預測 wx+bias。輸入 features 的 shape 是 `[4,1]`（4 筆、每筆 1 個數），數值為 −1、0、1、2；target 是 2x+1，也就是 −1、1、3、5。兩個 head 的輸出也是 `[4,1]`，各用 MSE（均方誤差：誤差平方的平均）算 loss。實驗跑兩次：一次固定 0.8/0.2 當對照，一次用上面的排程。兩次開始前都把亂數種子（seed）設成 7，所以初始參數相同；資料、SGD 更新次數（30 次）和學習率 0.05 也都相同。兩個線性 head 互不相連，學同一個 target；沒有不同的 assignment，也沒有共享的 backbone。

本例的 SGD 是最基本的版本：每步只做「新參數＝舊參數−學習率×梯度」，沒有加其他機制（例如 MuSGD 段會說明的動量）。所以每一步，one head 的參數減去 \(0.05\times b\times\partial L_{\text{one}}/\partial\theta\)，效果等於 one head 用 0.05×b 當學習率。這個有效學習率從 0.01 增加到 0.045；固定組則一直是 0.01。

```python
optimizer.zero_grad()  # 清除上一輪累積的梯度
many_prediction, one_prediction = many(features), one(features)  # 兩個 nn.Linear(1,1)
many_mse = F.mse_loss(many_prediction, target)
one_mse = F.mse_loss(one_prediction, target)
many_weight, one_weight = weights(epoch, epochs, final_many=.1)  # 回傳上式的 a(e)、b(e)；epochs 就是 E=30
loss = many_weight * many_mse + one_weight * one_mse
loss.backward()
optimizer.step()
```

這段是摘錄，變數名稱寫得比較長。完整程式（Colab 裡那份）的 `many_mse`、`one_mse` 叫 `lm`、`lo`，`many_weight`、`one_weight` 叫 `a`、`b`。完整程式用同一個 `train` 函式跑兩次：`progressive=True` 用上式的排程，`progressive=False` 固定 0.8/0.2 當對照。

seed 7 的第一步可以逐值核對：

| one 分支計算 | 數值 |
| --- | --- |
| 初始 w／bias | 0.318423／0.313781 |
| 四筆 forward `wx+bias` | −0.004643、0.313781、0.632204、0.950627 |
| MSE | `mean((prediction−[-1,1,3,5])²)=5.866377` |
| 已乘 b=0.2 的 dw／dbias | −1.146190／−0.610803 |
| SGD 更新後 w／bias | 0.375733／0.344321 |

表中數字四捨五入到小數第 6 位。用表列數字手算，最後一位可能差 1，例如第一筆會得到 −0.004642。

表中的梯度怎麼來？MSE 來自平方，所以導數有 2；四筆取平均，所以除以 4；最後再乘 one 的 loss 權重 b=0.2。預測 wx+bias 對 w 的變化率是 x，所以 dw 要乘 x；對 bias 的變化率是 1，所以 dbias 不用乘：`dw=.2×(2/4)×sum((prediction−target)×x)=−1.146190`，`dbias=.2×(2/4)×sum(prediction−target)=−.610803`。更新時減去學習率乘梯度，例如 `w_new=.318423−.05×(−1.146190)=.375733`。

完整程式會印出這份首步紀錄，並用斷言（assert）核對。最後報告的 MSE 則是 30 次更新都做完後重新 forward 算出來的，不是沿用最後一次更新前的 loss。

執行 `PYTHONPATH=. python lesson_cases/16-training.py`，程式會核對第 0 輪與第 29 輪的 many/one loss 權重（0.8/0.2 與 0.1/0.9），以及首步表中的參數、MSE 與梯度，並印出 one head 的 MSE。30 次更新後，固定 0.8/0.2 的 one MSE 是 0.663，progressive 是 0.016。這個固定小例中 progressive 較低，程式用 assert 檢查這個觀察；換資料、步數或學習率，不保證 progressive 仍然勝出。

差距從哪裡來？本例兩個 head 互不相連，a 完全不影響 one head。差距只因為 one 的有效學習率較大：30 輪的 b 加起來，progressive 是 16.5，固定組只有 6。若一開始就固定 0.1/0.9，one MSE 還會更低。所以本例看不到「前期多給 many」的好處。官方 YOLO26 的 one 分支讀的是 detach 後的特徵（梯度傳不回 backbone），backbone 只由 many 分支訓練；這種結構下排程有什麼作用，本例沒有測。本例支持的結論是：改變某分支的 loss 權重，會改變該分支學得多快。它不支持「所有場景都該使用這個排程」。

## STAL（Small-Target-Aware Label Assignment）：擴張候選資格，不放大真值框

先看問題。stride 8 的特徵圖每 8 畫素才有一個格心（每一格的中心點，也就是候選點），x、y 座標都是 4、12、20、…。2×2 畫素的小框可能一個格心都框不到。照 [12.3](12-assignment.md) 的資格規則，候選點要嚴格落在 GT（ground truth，人工標註的真值框）內才有資格；框裡沒有格心，這一層就沒有任何候選能學這個物件。STAL 的做法是：只在選候選這一步，把小框暫時放大。下圖是本節的例子。

![原真值、候選資格框與四個 stride 8 格心](../assets/diagrams/16-stal-candidates.svg)

圖用影像座標：x 向右、y 向下增加，所以 (4,12) 在 (4,4) 下方。

數字例子：真值框 `[7,7,9,9]` 只有 2×2 畫素，中心 `(8,8)`。圖中四個藍色格心 `(4,4)`、`(12,4)`、`(4,12)`、`(12,12)` 都不在真值框內。本例只看 stride 8 這一層；官方三層（stride 8、16、32）一起選候選時，stride 16 的格心 (8,8) 本來就在原框內。把候選資格框（只用來判斷資格的框）的邊長擴到 16，變成 `[0,0,16,16]`，四個格心都有了資格。程式以 assert 檢查資格點數由 0 變 4，同時確認原 GT 仍是 `[7,7,9,9]`。回歸真值不能被換成 16×16，否則會教模型畫大框。

這四個格心都在原框外。其中某一點若被選為正樣本，它要回歸的仍是原框，距離會有負值。以 (4,4) 為例，它對原框的四邊距離 ltrb（點到框的左、上、右、下邊）是 [4−7, 4−7, 9−4, 9−4]=[−3,−3,5,5] 畫素，左、上是負距離；四個點各有兩邊為負。[16.1](16-dfl-free.md) 說過，YOLO26 的 reg_max=1（每條邊直接輸出一個距離，不經過 DFL）可以表達帶符號的距離，這裡正好用得上。

官方的規則在 `TaskAlignedAssigner.select_candidates_in_gts` 裡，也就是 12.3 提過的分配器負責選候選的函式。查核版本只在選候選這一步，把 GT 寬或高小於 16 畫素的那一邊暫時改成 16，以中心為基準向外擴張，中心不動。寬、高各自判斷，例如 2×40 的框會變成 16×40，不是 16×16。16 取自三層特徵 stride `[8,16,32]` 的第二項。算 loss 時仍用原框。

四個格心只是進入候選池（通過資格檢查、可以參加後面挑選的候選）。之後還有品質排序（ranking：依預測品質由高到低排）、top-k 和衝突處理（見 12.3），本例都沒有執行；不是所有進入擴張框的點都會成為正樣本。查核版本中，one-to-many 分支每個 GT 取品質前 10 名；one-to-one 分支先取前 7 名並解決衝突，最後每個 GT 最多只留品質最高的 1 個（程式參數 topk2=1）。

[YOLO26 原論文](https://arxiv.org/abs/2606.03748)的 STAL 公式 (5) 寫的是邊長小於 8 才替換成 16；本節固定的官方程式碼用的是小於 16 的門檻。兩者範圍不同，這裡以固定程式碼為準。本例邊長 2 同時滿足兩者，0→4 的數字不受這個差異影響。

收益是小物件比較容易有候選進入候選池。代價是：框外的點也會參與（像上面四個格心；它們若成為正樣本，回歸距離有負值）；品質排序和衝突處理變得更重要；也可能增加錯誤的正樣本訊號。只有資格增加，不代表四個點全被分配，也不保證每個小物件最後都有正樣本。

## 延伸查證（可跳讀）：MuSGD

主例不需要實作這個優化器。MuSGD 除了沿梯度更新，也會整理矩陣參數的更新方向；這裡只說明它和 SGD 的差別。

動量（momentum）是把前幾步的梯度加權記下來，讓更新不只取決於當下這一次的梯度。Muon 是另一種只處理矩陣參數的優化器。MuSGD 的 Muon 部分先整理矩陣參數這一步的更新方向與大小（做法見下方摺疊區），再和 SGD 的動量更新組合，目的在於改變矩陣參數的更新幾何：每一步往哪些方向走、各走多遠。

哪些參數算矩陣？線性層的 `weight` 本來就是 2 維矩陣。卷積層的 `weight` shape 是 [輸出 channel, 輸入 channel, kh, kw]（kh、kw 是 kernel 的高與寬），可以攤平成 [輸出 channel, 其餘三軸相乘] 的矩陣。bias 這類一維參數不走 Muon，只用 SGD 更新。官方用矩陣近似運算（Newton–Schulz）實作 Muon，並替 SGD 那部分另外存一份動量。還有三項設定也會影響結果：

- 參數分組：哪些參數走 Muon、哪些只用 SGD。
- 混合比例：Muon 與 SGD 兩種更新各乘的比例。
- 權重衰減（weight decay）：每步把參數往 0 拉一點，和 loss 權重的遞減無關。

??? note "Muon 怎麼整理更新量"

    Muon 的名稱來自 MomentUm Orthogonalized by Newton–Schulz，意思是「用 Newton–Schulz 正交化的動量」。

    它先把梯度做動量平均，得到這一步的更新矩陣。再把這個矩陣「正交化」：保留各個方向，把各方向的伸縮倍率（線性代數叫奇異值）都拉到接近 1，避免少數方向獨占這一步。例如更新矩陣是 \(\begin{bmatrix}3&0\\0&0.1\end{bmatrix}\) 時，第一個方向的倍率是 3，第二個方向只有 0.1，這一步幾乎只動第一個方向。理想的正交化結果是單位矩陣，兩個倍率都是 1。

    Newton–Schulz 只用矩陣乘法反覆計算幾次，來近似正交化。官方固定迭代 5 次，結果只是大致接近正交：上面的例子用官方程式實際得到約 \(\begin{bmatrix}0.69&0\\0&1.13\end{bmatrix}\)（官方程式用 bfloat16 計算，這是只用 16 位元存一個數、精度較低的格式；改用一般的 float32 照樣算 5 次，第一個倍率約是 0.70），兩個倍率都拉到 1 附近，但不是精確的 1。

    最後，MuSGD 再加上另一份動量的一般 SGD 更新，兩種更新各乘一個比例後相加。

額外的矩陣計算、多存的一份動量、分組與混合設定，會增加運算、記憶體和調參成本。是否換得更好的收斂，要另做配對實驗。本節沒有執行 MuSGD，不能從兩個線性 head 的 MSE 宣稱這個優化器較好。

這些細節在官方程式碼裡都讀得到，但本節不自行拼出未驗證的 MuSGD 簡化版，也不聲稱重現官方發布模型的整套訓練配方（recipe：資料、預訓練、增強、超參數與步數等全部設定）。官方訓練指南說，發布的模型先在 Objects365（有 365 類物件的大型偵測資料集）預訓練，再用 COCO 微調（以預訓練好的參數為起點，在新資料上繼續訓練）。部分內部超參數（例如 many 分支的 loss 權重）只能在實驗用的程式碼分支上設定。這種分支是 Git branch（同一份程式碼另外開出的平行版本），放在 Ultralytics 的 GitHub 上、沒有併入主線（main），和 many／one 分支無關；一般安裝的套件不接受這些設定。所以前面的 0.8→0.1 是固定版本程式的預設，不代表發布模型的實際設定。

正式比較優化器時，要固定資料、初始化、步數、增強和排程，再記錄收斂、參數更新尺度與成本。拿預訓練過的模型和從零開始訓練的 MiniYOLO 比最終 AP，差距不能歸因於優化器。

## 常見錯誤與自主練習

常見錯誤：

- **以為真實模型改 a、b，只等於調一個學習率**：本例兩個 head 互相獨立，才剛好等於調 one 的學習率。查核版本中，one 分支讀 detach 後的特徵，所以 a 縮放的是 backbone 與 many head 的梯度，b 只縮放 one head 的梯度；再加上動量、MuSGD 和權重衰減，效果不等於只調一個學習率。
- **三項同時改，卻只報總 loss**：分不出是誰造成差異。
- **把 STAL 的資格框當成標註框**：16×16 的資格框只用來選候選，回歸真值仍是原框。
- **沒用 MuSGD 做配對實驗，就宣稱它比較好**：本節沒有執行 MuSGD，兩個線性 head 的 MSE 說明不了優化器的好壞。

自主練習：在完整程式（Colab 裡那份，或本機的 `lesson_cases/16-training.py`）只改 `main()` 裡的 `final_many = .1`，改成 `.3`，其他不動。先回答下面四題，再執行核對：

1. 第 29 輪的 many/one loss 權重是多少？
2. 第 0 輪的首步數值（上面那張表）會變嗎？為什麼？
3. progressive 的 one MSE 會變大還是變小？`assert progressive < fixed` 還會通過嗎？
4. STAL 讓四個格心進入候選池後，YOLO26 的 one-to-one 分支最後每個 GT 最多有幾個正樣本？

??? note "參考答案"

    1. 0.3/0.7。e=29 時 \(a=0\times(0.8-0.3)+0.3=0.3\)，b=1−0.3=0.7。
    2. 不變。\(a(0)=1\times(0.8-0.3)+0.3=0.8\)，和終點無關，b 仍是 0.2；資料、seed 和初始參數也都沒變，所以首步數值和改之前（`final_many = .1`）完全相同。排程、核對終點的 assert 和印出的結果都用同一個 `final_many` 變數，不必改任何 assert。
    3. 變大，但仍低於固定組。改程式後重跑，會印出 progressive 的 one MSE 約 0.04（原本是 0.016），仍低於固定 0.8/0.2 的 0.663，所以 assert 仍會通過。原因是 one 的有效學習率變小：第 1 輪起，每一輪的 b 都比原本小，30 輪的 b 加起來從 16.5 降到 13.5。多留給 many 的 loss 權重，在本例對 one 沒有幫助，因為兩個 head 互不相連。在真實模型（例如官方 YOLO26，backbone 只由 many 分支訓練）上值不值得這樣分配，要另做配對實驗才知道。
    4. 最多 1 個。4 只是有資格進池的點數；查核版本的 one-to-one 分支，最後每個 GT 最多只留品質最高的 1 個（topk2=1）。本例沒有做品質排序，說不出會是哪一個。

來源查核：2026-10-02。[E2ELoss 排程](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)、[小物件資格與 top-1 篩選（topk2）](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py)、[MuSGD 公開實作](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/optim/muon.py)、[官方訓練配方（recipe）與重現範圍](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式已於 2026-10-02 用 PyTorch 2.9.1+cpu 在 CPU 上執行過，程式裡的 assert 檢查全部通過。下面是那次印出的原始輸出；每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-training.json)

??? example "展開本次實際輸出"

    ```text
    many/one weight first: (0.8, 0.2)
    many/one weight last: (0.1, 0.9)
    one-head MSE fixed=0.663073, progressive=0.016285
    one-head first forward/backward/step: {"weight": 0.31842339038848877, "bias": 0.3137805461883545, "prediction": [-0.004642844200134277, 0.3137805461883545, 0.6322039365768433, 0.950627326965332], "one_mse": 5.866377353668213, "one_gain": 0.19999999999999996, "weighted_dw": -1.1461899280548096, "weighted_dbias": -0.6108031272888184, "updated_weight": 0.3757328987121582, "updated_bias": 0.34432071447372437}
    small-object eligible points original / expanded: 0 4
    GT regression box remains: [7.0, 7.0, 9.0, 9.0]
    not a full YOLO26/STAL/MuSGD training reproduction
    ```

<!-- curriculum-evidence:end -->
