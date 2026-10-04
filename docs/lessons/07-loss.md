# 7.3 Grid MiniYOLO：loss 必須能手算

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/07-loss.ipynb){ .md-button }

上一節把標註框換成每一格的訓練目標（target）；本節把 target 和模型每格的輸出接起來，算出 loss。讀完你能用紙筆算出三項 loss 與 objectness（這格有沒有物件）的梯度，也知道沒有物件的圖要怎麼處理。

前置：[上一節的 targets](07-targets.md)（每格的 box、objectness、class_ids 與 positive 怎麼來）。本節在訓練前用人工設定的 logits 驗證三件事：數值是否吻合、哪些位置收到梯度、空圖的 loss 是否保持有限（box 與 class 為 0）。只看到 total loss 是有限數字（不是無限大，也不是 NaN；NaN 是 0÷0 這類算不出的值），不足以證明 mask 正確。

歷史來源：[YOLOv1 原論文](https://arxiv.org/abs/1506.02640) 用一個 loss 同時學定位與分類。本章的 loss 和原版一樣分成幾項，但改用三種不同的算法，而且都取平均：正格座標的 MSE（均方誤差）、全格 objectness 的 BCE（二元交叉熵）、正格分類的交叉熵（cross entropy，CE）。原版則是每一項都用平方誤差再加總，並乘上兩個權重：座標項乘 λcoord=5；預測框不含物件時，它的 confidence（類似本章的 objectness）項乘 λnoobj=0.5。下面的公式就是本 repo 的 `grid_loss`，不能當成完整的 YOLOv1 loss。

??? note "原版 YOLOv1 loss 的其他設計"

    下面三件事，本章都沒有做：

    - 平方根寬高：原版預測寬、高的平方根，再算誤差，讓小框的同樣誤差在 loss 裡更顯眼。
    - IoU 當 confidence 目標：原版負責物件的那個框，confidence 的目標是預測框與真值框的 IoU（交集面積除以聯集面積），不是固定的 1。
    - 多框責任規則：原版每格預測多個框（PASCAL VOC 資料集的設定是 2 個），訓練時只讓和真值框當下 IoU 最高的那個框負責。

    λnoobj 的用意：大多數格子沒有物件，它們的 confidence 都被往 0 推，常常蓋過有物件格子的梯度；原版用 λnoobj=0.5 減弱這一項。本章沒有這個權重，後面〈收益、代價與常見錯誤〉會用本例的數字看這個現象。

    本章框項乘的 5，數值與原版 λcoord 相同。不過本章各項先取平均，原版是加總，同一個 5 在兩邊占的份量不能直接對照。

## 一個紅框，每格七個零 logits

先說明本節的符號：

- B 是 batch 的圖片數、S 是每邊格數、C 是類別數；本例 B=1、S=4、C=2。
- prediction 是模型的輸出，shape `[B,S,S,5+C]`。每格 5+C=7 個數，依序是 `tx,ty,tw,th,obj,class0,class1`。前四個 tx～th 是模型對框的輸出，不是 target；下面公式裡的 t 才代表 target。本節還沒有模型，用一個全部填 0 的 tensor 代替。
- logit：還沒經過 sigmoid 或 softmax 的原始實數；prediction 的 7 個數都是 logits。objectness 與類別的 logits 轉換後是機率；框的四個 logits 經 sigmoid 後，是 0 到 1 之間的座標比例。
- sigmoid 把單一個數轉成 0 到 1 之間；softmax 把幾個類別的數轉成總和為 1 的機率。公式在下面。
- mask（遮罩）是由 True／False 組成的 tensor，True 的位置才學框與類別。
- pos 就是 `target['positive']`：shape `[B,S,S]` 的 True／False 表，正格是 True。加總時 True 算 1、False 算 0，所以 `pos.sum()` 就是整批的正格數 Npositive，本例是 1。程式裡的 `pos.any()` 則是問「有沒有任何一格是 True」。

| 欄位 | 原 shape | 正格 mask 後 |
| --- | --- | --- |
| prediction | `[B,S,S,5+C]` | 框 `[Npositive,4]`、類別 `[Npositive,C]` |
| positive | `[B,S,S]` bool | 它就是 mask；本例只有位置 (b,gy,gx)=(0,1,1) 為 True，所以 Npositive=1 |
| box | `[B,S,S,4]` float | `[Npositive,4]` |
| objectness | `[B,S,S]` float | 不取 mask，全格計算 |
| class_ids | `[B,S,S]` long | `[Npositive]` |

本例實際的 shape：prediction `[1,4,4,7]` 用 mask 取出後，框是 `[1,4]`、類別是 `[1,2]`；class_ids 取出後是 `[1]`。

### 三項 loss 的公式

loss 分三項，各教一件事：box 教正格的框在哪裡，objectness 教每一格有沒有物件，classification 教正格是哪一類。先寫出會用到的函式，z 代表一個 logit：

- sigmoid：\(\sigma(z)=\dfrac{1}{1+e^{-z}}\)。例如 \(\sigma(0)=\dfrac{1}{1+1}=0.5\)。
- softmax：C 個類別 logits 記作 \(z_0,\dots,z_{C-1}\)，下標就是類別編號，和第 1 章一樣從 0 起算；本例 \(z_0\)、\(z_1\) 依序是 `class0`、`class1`。類別 k 的機率是 \(\dfrac{e^{z_k}}{\sum_j e^{z_j}}\)，C 個機率加起來是 1。兩類都是 0 時，各是 \(\dfrac{e^0}{e^0+e^0}=0.5\)。
- 交叉熵（CE）：單格的 CE 是 \(-\ln p\)，p 是 softmax 後「正確類別」的機率。\(-\ln p\) 在 p=1 時是 0，p 越小就越大：給正確答案的機率越低，罰得越重。
- 二元交叉熵（BCE）：單格的 BCE 記作 \(\mathrm{BCE}(z,t)=-\bigl[t\ln p+(1-t)\ln(1-p)\bigr]\)。z 是這一格的 objectness logit，\(p=\sigma(z)\)，t 是目標 0 或 1。t=1 時只剩 \(-\ln p\)，t=0 時只剩 \(-\ln(1-p)\)。

ln 是以 e≈2.71828 為底的自然對數；PyTorch 的 log 與 Python 的 `math.log` 也都是它。計算機的 log 鍵通常以 10 為底，用它算 −log 0.5 會得到 0.30103，不是 0.693147。

三項 loss 與總和寫成一般式：

\[
L_{\text{box}}=\frac{1}{4N_{\text{pos}}}\sum_{i\in P}\sum_{k=1}^{4}\bigl(\sigma(z_{i,k})-t_{i,k}\bigr)^2
\]

\[
L_{\text{obj}}=\frac{1}{B\cdot S\cdot S}\sum_{i=1}^{B\cdot S\cdot S}\mathrm{BCE}\bigl(z^{\text{obj}}_i,\,o_i\bigr)
\]

\[
L_{\text{cls}}=\frac{1}{N_{\text{pos}}}\sum_{i\in P}\bigl(-\ln p_{i,c_i}\bigr)
\]

\[
L=5L_{\text{box}}+L_{\text{obj}}+L_{\text{cls}}
\]

其中：

- P 是正格的集合，\(N_{\text{pos}}\) 是正格數，就是上面的 Npositive。
- 整批共 \(B\cdot S\cdot S\) 格，編號 \(i=1,\dots,B\cdot S\cdot S\)。
- 正格 i 的四個框 logits 是 \(z_{i,1},\dots,z_{i,4}\)，框 target 是 \(t_{i,1},\dots,t_{i,4}\)。
- 第 i 格的 objectness logit 是 \(z^{\text{obj}}_i\)，目標是 \(o_i\)（0 或 1）。
- 正格 i 的正確類別編號是 \(c_i\)（0 到 C−1）；\(p_{i,c_i}\) 是這一類在 softmax 後的機率。

沒有正格（\(N_{\text{pos}}=0\)）時，程式直接令 \(L_{\text{box}}\) 與 \(L_{\text{cls}}\) 為 0，原因見後面〈空圖〉。本例代入 B=1、S=4、\(N_{\text{pos}}=1\)。

### 代入本例：每一項都能手算

沿用上一節的紅框 `[8,12,24,28]`，類別 0（紅色）。原框順序是 pixel xyxy，中心 `(16,20)`、寬高 `(16,16)`。每格 `64/4=16` pixel，中心落在 (gy=1,gx=1) 這一格，它就是唯一的正格。格內偏移 x=16/16−1=0（減掉格索引 gx=1），y=20/16−1=0.25（減掉格索引 gy=1）；wh 都是 16/64=0.25。所以這格的框 target 是 `[0,.25,.25,.25]`，class=0。四個 target 都沒有單位：前兩項相對格子，後兩項相對全圖。

再把整個 `[1,4,4,7]` prediction 填 0。代入上面的定義：sigmoid(0)=0.5；兩個類別 logit 都是 0，softmax 後各是 0.5。

1. 框 loss 只取正格，平均它四個座標的平方誤差。四個預測經 sigmoid 都是 0.5，target 是 [0, 0.25, 0.25, 0.25]，誤差（預測減 target）依序是 0.5、0.25、0.25、0.25。平方和是 0.25+3×0.0625=0.4375，除以 4 得 0.109375，也就是 `(.5²+.25²+.25²+.25²)/4=.109375`。座標已正規化（normalized：換算成沒有單位的比例），算的不是 pixel 的平方。
2. objectness loss 用全部 16 格：1 正、15 負。p=σ(0)=0.5，正格取 \(-\ln 0.5\)，負格取 \(-\ln(1-0.5)\)，兩者相等，都是 \(\ln 2\approx0.693147\)；16 格平均因而仍是 0.693147。
3. 類別 loss 只取正格：正確類別（類別 0）的機率是 0.5，所以 CE 是 \(-\ln 0.5=\ln 2\approx0.693147\)。負格不計類別 loss。
4. 本章固定 `total=5×box+objectness+classification`，所以 total=5×0.109375+0.693147+0.693147=1.933169。為什麼框要乘權重？三項的量尺不同，直接相加可能讓某一項作用太弱：本例 box 只有 0.109，另兩項約 0.693（第 4 章也遇過同樣的問題；那裡也提醒過，數值的占比不等於梯度的占比）。5 是本章固定的示範設定，不是搜尋出來的最佳值；loss 本身也不會替你找出最好的權重。

三項都沒有單位，但取平均的分母不同：box 對 `Npositive×4` 個數取平均（每個正格 4 個座標），class 對 `Npositive` 個、objectness 對 `B×S×S` 個；本例依序是 4、1、16。

程式裡的 `reduction` 參數決定怎麼把逐項的 loss 合成一個數：`reduction='mean'` 是相加後除以個數，`reduction='sum'` 只相加。若改成 sum，batch 的圖片數或物件數一改變，共享 CNN 權重收到的總梯度就會跟著改變。「共享」是說所有格子的 logits 都由同一組 CNN 權重算出。注意這裡說的是權重收到的總梯度，不是每個 logit 的梯度；兩者的差別見〈自主練習〉的練習 1。

若把框改用 pixel 座標，loss 的大小會差很多。同樣是 16 pixel 的誤差：用 pixel 算，平方是 256；wh 以全圖 64 pixel 正規化，是 (16/64)²=0.0625，差 4096 倍；xy 以格寬 16 pixel 正規化，是 (16/16)²=1，差 256 倍。改用 pixel 後，box 項通常會壓過另兩項，權重 5 必須重新決定。

### 對應的程式

下面是本 repo `grid_loss`（在 `miniyolo/losses.py`）的簡化版，省略了檢查輸入的部分。完整程式直接呼叫 `grid_loss(prediction, target)`。

```python
import torch.nn.functional as F
# pred：上面那個全部填 0 的 prediction，shape [1,4,4,7]
# target：build_targets([scene], 4, 64, 2) 回傳的 dict，含 box、objectness、class_ids、positive
pos = target['positive']
if pos.any():  # 至少有一個正格嗎？
    box = F.mse_loss(pred[..., :4].sigmoid()[pos],
                     target['box'][pos], reduction='mean')
    cls = F.cross_entropy(pred[..., 5:][pos],
                          target['class_ids'][pos], reduction='mean')
else:  # 沒有正格（空圖），見下面〈空圖〉
    box = (pred[..., :4] * 0).sum()
    cls = (pred[..., 5:] * 0).sum()
obj = F.binary_cross_entropy_with_logits(pred[..., 4],
                                         target['objectness'], reduction='mean')
total = 5 * box + obj + cls
```

`build_targets([scene], 4, 64, 2)` 是上一節的函式，後三個參數依序是每邊格數 S、圖片邊長（pixel）、類別數 C。程式裡的索引這樣讀：

- `pred[..., :4]`：`...` 表示前面的軸（b、gy、gx）全部照拿，`:4` 只在最後一軸取通道 0–3（框），本例 shape 是 `[1,4,4,4]`。同理，`pred[..., 4]` 取通道 4（objectness），`pred[..., 5:]` 取通道 5 以後（類別）。
- `[pos]`：pos 是 `[1,4,4]` 的 True／False 表，只有 (0,1,1) 為 True。拿它當索引叫布林索引（Boolean indexing）：前三軸會併成一軸，只留下 True 的那幾格。所以 `pred[..., :4].sigmoid()[pos]` 是 `[1,4]`，一般是 `[Npositive,4]`。結果仍是 tensor，不是 Python 的 list。
- `target['box'][pos]` 用同一個 pos，也是 `[1,4]`，兩邊的第 i 列對應同一格。

為什麼框要自己先 sigmoid，另兩項卻直接給 logits？`mse_loss` 只做相減、平方、平均，不會替你轉換，所以框要先 sigmoid 成 0 到 1 之間，才能和 target 比。`binary_cross_entropy_with_logits` 名字裡的 with logits，表示它接收 logits、內部自己做 sigmoid；`cross_entropy` 內部會自己做 softmax（第 1 章提過）。所以這兩項都直接給 logits；先轉一次再傳進去，就等於轉了兩次。

## 梯度讓數字成為方向

完整程式建立 prediction 時寫的是 `torch.zeros(1, 4, 4, 7, requires_grad=True)`。prediction 不是模型參數，設了 `requires_grad=True`，PyTorch 才會替它記錄梯度（第 3 章對輸入 x 做過同樣的事）。對 total 呼叫 `backward()` 之後，`prediction.grad` 的每個數，就是 total 對那個 logit 的偏導數：那個 logit 稍微變大時，total 怎麼變。

BCE 對 objectness logit 的梯度是 `(sigmoid(logit)-target)/(B×S×S)`，本例分母才是 16（1×4×4）。因此正格為 `-.03125`，背景為 `+.03125`。total 裡的 5 只乘在 box 上，objectness 的係數是 1，所以這裡不必乘 5。

不會微分也能驗算。把正格的 objectness logit z 從 0 改成 0.001，這一格的 BCE 由 0.693147 變成 0.692647，少了 0.0005；除以 0.001，斜率約 −0.5，正好是 sigmoid(0)−1=0.5−1。objectness 是 16 格的平均，所以 total 的斜率還要除以 16：−0.5/16=−0.03125。背景格同理：BCE 由 0.693147 變成 0.693647，斜率約 +0.5=sigmoid(0)−0，除以 16 得 +0.03125。

??? note "推導：為什麼分子是 sigmoid 減 target"

    只看一格。令 \(p=\sigma(z)\)，單格 BCE 是 \(\ell(z)=-\bigl[t\ln p+(1-t)\ln(1-p)\bigr]\)。

    第一步，求 sigmoid 的導數。把 \(\sigma(z)=(1+e^{-z})^{-1}\) 用連鎖律（chain rule）微分：

    \[
    \sigma'(z)=-(1+e^{-z})^{-2}\cdot(-e^{-z})=\frac{e^{-z}}{(1+e^{-z})^2}
    \]

    把它拆成兩個分數相乘：

    \[
    \frac{e^{-z}}{(1+e^{-z})^2}=\frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}}=\sigma(z)\bigl(1-\sigma(z)\bigr)
    \]

    最後一步用到 \(1-\sigma(z)=\dfrac{(1+e^{-z})-1}{1+e^{-z}}=\dfrac{e^{-z}}{1+e^{-z}}\)。所以 \(p'=p(1-p)\)。

    第二步，對 \(\ell\) 微分。若 u 是 z 的函數，\(\ln u\) 對 z 的導數是 \(u'/u\)（也是連鎖律）。所以 \(\ln p\) 的導數是 \(\dfrac{p(1-p)}{p}=1-p\)，\(\ln(1-p)\) 的導數是 \(\dfrac{-p(1-p)}{1-p}=-p\)。代回去：

    \[
    \frac{d\ell}{dz}=-\bigl[t(1-p)-(1-t)p\bigr]=-\bigl[t-tp-p+tp\bigr]=p-t
    \]

    也就是 sigmoid(z) 減 target。objectness loss 是 B×S×S 格的平均；每個 objectness logit 只出現在自己那一格的 BCE 裡，box 與 class 也不含它，所以 total 對它的梯度是 \((p-t)/(B\cdot S\cdot S)\)。本例 p=0.5：正格 (0.5−1)/16=−0.03125，背景 (0.5−0)/16=+0.03125。

梯度的正負號指出方向。假如把這個 logit 本身當成變數，走一步梯度下降 \(z_{\text{new}}=z-\eta g\)（η 是學習率，g 是梯度）：正格的梯度是負的，z 會變大；背景的梯度是正的，z 會變小。本節只從梯度的正負看方向，沒有真的做這一步更新。

真正訓練時，optimizer 改的是 CNN 權重，logit 跟著權重間接改變。權重收到的梯度，是把每個 logit 的梯度各乘上「這個 logit 對權重的變化率」，再全部加起來。所以「每個 logit 的梯度」和「權重收到的總梯度」是兩回事，〈自主練習〉的練習 1 會用數字比較。

負格的四個框梯度與兩個類別梯度都必須是 0。若負格的框梯度不為 0，表示背景正在學某個無意義的框。

## 空圖：沒有正格時

問題：空圖沒有任何框，pos 的 16 格全是 False。這時 `pred[..., :4].sigmoid()[pos]` 的 shape 是 `[0,4]`，一個數也沒有。對 0 個數取平均，等於 0 除以 0，PyTorch 會得到 NaN（Not a Number）。

後果：box 和 class 都變成 NaN。NaN 和任何數運算，結果仍是 NaN，所以 total 也是 NaN。本 repo 的訓練程式（`miniyolo/train.py`）遇到不是有限數字的 loss，會直接報錯停下。也不能靠刪掉空圖解決：空圖提供「這裡沒有物件」的背景監督，16 格都要學 objectness=0（見〈[Grid MiniYOLO 資料](07-data.md)〉）。

解法：程式先用 `pos.any()` 問有沒有正格；沒有就走 else，令 box 與 class 為「先乘 0、再加總」的結果。scalar 是 shape `[]` 的單值 tensor；這樣得到的就是一個值為 0 的 scalar，它對 prediction 的梯度也是 0。objectness 照常對 16 格取平均，仍是 0.693147，所以空圖的 total 是 0.693147。這個分支讓全背景 batch 的 loss 保持有限。

為什麼要乘 0 再加總，不直接寫常數 0？這個 0 是由 prediction 算出來的，仍連在計算圖上（計算圖：PyTorch 記下「誰由誰算出」的紀錄，backward 沿著它往回算梯度），所以就算單獨對它 backward 也不會報錯。常數 0 和 prediction 沒有關係，單獨對它 backward 會報錯。如果只對 total backward，兩種寫法得到的梯度相同。

## 執行與核對

在 [Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/07-loss.ipynb) 執行，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-loss.py`，然後核對：

- 四項 loss：`box .109375 / objectness .693147 / classification .693147 / total 1.933169`。
- 正格與背景的 objectness 梯度：`-.03125/.03125`。
- 空圖：通過 backward 的檢查，loss 有限，box 與 class 剛好是 0。輸出最後一行 `empty image: finite backward, box/class=0`，表示這部分的斷言（assert）都通過了。

只看「backward 成功、梯度有限」，分辨不出有沒有空圖分支：沒有分支時，梯度其實仍是有限的，壞掉的是 loss 值。真正分得出來的是 box 與 class 等於 0、total 約 0.693147。

這個實驗只用人工設定的數字檢查 loss 的計算，沒有訓練模型，所以沒有 AP（第 6 章的偵測評估分數）結果。

## 收益、代價與常見錯誤

收益是每一項監督和梯度方向都能被驗證。代價有兩個。第一，MSE 只比四個數各差多少，沒有直接衡量預測框和真值框重疊得好不好。第二，物件很少的圖裡，背景格遠多於正格。本例 16 格只有 1 格有物件：背景的 objectness 梯度加起來是 15×0.03125=0.46875，正格只有 0.03125，模型可能先學會到處說沒有物件。不要在尚未確認資料與 mask 時先調權重。第 11 章才單獨研究定位 loss，看 IoU 類 loss（用兩框重疊程度算的 loss）改了什麼。

常見錯誤分兩類。第一類不會報錯，會得到看似正常的有限 loss，只能靠手算與梯度檢查抓出來：

- 先 sigmoid 再傳給 `BCEWithLogitsLoss`。它是程式裡 `F.binary_cross_entropy_with_logits` 的 Python class 版本（放在 `torch.nn` 裡，先建立物件再呼叫），算的是同一件事，內部同樣會做 sigmoid，所以等於做了兩次。本例 objectness 會從 0.693147 變成 0.942827。
- 先 softmax 再傳給 CE，等於做了兩次 softmax。本例兩個類別 logits 都是 0，第一次 softmax 後兩類仍相等（各 0.5），所以錯誤寫法算出的類別 loss 仍是 0.693147，和正確值一樣。因此在本例，只看 loss 抓不到這個錯；要看正格的類別梯度，錯誤寫法只有正確值的一半。
- 把負格也送進 CE。本章負格的 class_ids 是預設值 0，正好是紅色的編號，所以不會報錯，卻會把背景格（本例 15 格）教成紅色。類別 loss 一定要先用 pos 挑出正格。

第二類會報錯，或得到 NaN：

- 若像第 5 章那樣把負格的 class_ids 填 −1，再送進 CE，會報錯。
- 沒有正格時照樣取平均，會得到 NaN（見上面〈空圖〉）。

## 自主練習

三題都先用紙筆算，再展開答案。練習 3 的答案附了核對用的程式。

**練習 1：** 把 batch 複製成兩張完全一樣的圖（B=2），三項 mean loss 變多少？objectness 每個 logit 的梯度變多少？如果兩張圖的 logits 由同一組 CNN 權重算出，權重收到的總梯度又會怎樣？

??? note "參考答案"

    loss：不變。兩張圖的每一項都和原來一樣，分子與分母一起變兩倍，平均還是原值；若使用 sum 才會翻倍。

    每個 logit 的梯度：減半。objectness 的分母從 16 變 32，正格從 −0.03125 變成 −0.015625，背景從 +0.03125 變成 +0.015625。box 與 class 的分母也加倍，梯度同樣減半。

    共享權重的總梯度：不變。權重會收到兩張圖各一份減半的梯度，加起來和只有一張圖時相同。所以不要從「每個 logit 的梯度減半」推論共享模型的總梯度也減半。

    改用 sum 時，objectness 每格的梯度固定是 ±0.5（就是 sigmoid(z) 減 target，不再除以格數）；兩張圖加起來，權重收到的總梯度變兩倍。

    | | mean | sum |
    | --- | --- | --- |
    | objectness 每個 logit 的梯度 | 減半：±0.03125 → ±0.015625 | 不變：固定 ±0.5 |
    | 共享權重收到的總梯度 | 不變 | 變兩倍 |

**練習 2：** 把正格 (0,1,1) 的兩個類別 logits（通道 5、6）都從 0 改成 10，CE 會變嗎？

??? note "參考答案"

    不變。softmax 對共同平移不變：每個類別 logit 都加上同一個數，機率不會變。設兩個類別 logit 是 a、b，分子分母同乘 \(e^{10}\)，約掉後不變：

    \[
    \frac{e^{a+10}}{e^{a+10}+e^{b+10}}=\frac{e^{10}e^{a}}{e^{10}\bigl(e^{a}+e^{b}\bigr)}=\frac{e^{a}}{e^{a}+e^{b}}
    \]

    本例 a=b=0，加 10 之後兩類仍各是 0.5，CE 仍是 \(\ln 2\approx0.693147\)。

**練習 3：** batch 改成〔紅框圖, 空圖〕（B=2），四項 loss 與 objectness 梯度各是多少？

??? note "參考答案"

    B×S×S=32。box 和 class 只平均正格，正格仍只有紅框那 1 格，所以數值不變：box 0.109375、classification 0.693147；正格的框梯度與類別梯度也不變。objectness 改對 32 格平均，但每格仍是 ln 2，平均還是 0.693147，所以 total 仍是 1.933169。

    改變的是 objectness 梯度：分母從 16 變成 32，正格是 −1/64=−0.015625，31 個負格（紅框圖 15 格加空圖 16 格）各是 +1/64=+0.015625。

    完整程式第 27–28 行用兩個斷言，檢查原例 1 個正格、15 個負格的 objectness 梯度是 ±1/32：

    ```python
    assert torch.allclose(prediction.grad[..., 4][pos], torch.full((1,), -1/32))
    assert torch.allclose(prediction.grad[..., 4][~pos], torch.full((15,), 1/32))
    ```

    這一題要改成 `torch.full((1,), -1/64)` 與 `torch.full((31,), 1/64)`。想用程式核對時，不要改原本的 `main()`：在 notebook 最後一格下面新增一個程式碼儲存格，貼上下面的程式另外測試（在自己的電腦上，也可以存成 .py 檔，在 repo 根目錄用 `PYTHONPATH=. python` 執行）。

    ```python
    import torch
    from miniyolo.targets import build_targets
    from miniyolo.losses import grid_loss

    scene = {"boxes": torch.tensor([[8., 12., 24., 28.]]), "labels": torch.tensor([0])}
    empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    target = build_targets([scene, empty], 4, 64, 2)  # 兩張圖：紅框圖、空圖
    prediction = torch.zeros(2, 4, 4, 7, requires_grad=True)
    parts = grid_loss(prediction, target)
    parts['total'].backward()
    pos = target['positive']
    print({k: round(v.item(), 6) for k, v in parts.items()})
    assert torch.allclose(prediction.grad[..., 4][pos], torch.full((1,), -1/64))
    assert torch.allclose(prediction.grad[..., 4][~pos], torch.full((31,), 1/64))
    ```

    把程式裡的 `[scene, empty]` 改成 `[scene, scene]`，也能核對練習 1；這時兩個斷言要改成 `torch.full((2,), -1/64)` 與 `torch.full((30,), 1/64)`。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式已於 2026-10-02 用 PyTorch 2.9.1+cpu 在 CPU 上執行過，程式裡的 assert 檢查全部通過。下面是那次印出的原始輸出；每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-loss.json)

??? example "展開本次實際輸出"

    ```text
    {'total': 1.933169, 'box': 0.109375, 'objectness': 0.693147, 'classification': 0.693147}
    positive/background objectness gradients -0.03125 0.03125
    empty image: finite backward, box/class=0
    ```

<!-- curriculum-evidence:end -->
