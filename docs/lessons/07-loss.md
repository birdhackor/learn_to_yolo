# B.7.3 Grid MiniYOLO：loss 必須能手算

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/07-loss.ipynb){ .md-button }

上一節已經決定每格的答案。現在要確認：當預測不符合答案，loss 會罰哪些數、又把它們往哪個方向推？只看到一個有限的 total，還不知道背景是否被錯教成紅色，也不知道框的 mask 是否漏用。

先讓每格七個 logits 都為 0，算一個紅框的 loss，再核對反傳梯度。這次直接建立輸出 tensor，不建立模型、不更新參數；讀完能手算三項 loss、objectness 與類別梯度，並解釋整批都是空圖時如何保持 loss 有限。

## 全零輸出，怎樣算出紅框的誤差

本例只用一張圖（B=1），每邊四格（S=4）、紅藍兩類（C=2）。紅框 `[8,12,24,28]` 仍由 `(b,gy,gx)=(0,1,1)` 負責，box target 是 `[0,.25,.25,.25]`，class id=0。16 格有 1 正、15 負。

`prediction` 為 `[1,4,4,7]`，每格依序是 `tx,ty,tw,th,obj,class0,class1`。它們都是原始實數 logits。框四項先經 sigmoid，變成格內 xy、全圖 wh 的比例；obj 經 sigmoid 表達物件分數，兩個類別經 softmax 成為互斥的類別機率。

sigmoid 定義為 \(\sigma(z)=1/(1+e^{-z})\)，所以零輸出變成 0.5。兩類 softmax 則為 \(p_k=e^{z_k}/\sum_j e^{z_j}\)，兩個零 logits 各得到 0.5。這已足夠算第一項：

\[
L_{\text{box}}=\frac{(.5-0)^2+(.5-.25)^2+(.5-.25)^2+(.5-.25)^2}{4}=.109375
\]

這是均方誤差（MSE）：只取正格，把四個座標誤差平方後平均。數字是無單位的比例，並非 pixel 的平方；負格沒有正確框，所以不拿它的占位值來算。

## 「有沒有物件」與「是哪一類」分開教

objectness 是每格各自的二選一答案，因此使用二元交叉熵（binary cross entropy，BCE）。一格的目標 t 是 0 或 1，預測 \(p=\sigma(z)\)：

\[
\mathrm{BCE}(z,t)=-[t\ln p+(1-t)\ln(1-p)]
\]

正格 t=1，罰的是 \(-\ln p\)；負格 t=0，罰的是 \(-\ln(1-p)\)。給正確答案的機率越小，罰得越多。零 logit 的 p=0.5，兩種答案都得到 \(\ln2\approx.693147\)。所以 1 正、15 負全部平均，objectness loss 仍為 .693147。

類別則只在正格回答「紅或藍」。交叉熵（cross entropy，CE）取正確類別的 softmax 機率 p，算 \(-\ln p\)。紅類 p=0.5，這一個正格的 classification loss 也是 .693147。背景不算第三類，也不讀負格 class_ids 的預設 0。

這裡 ln 是以 e≈2.71828 為底的自然對數，PyTorch 的 log、Python 的 `math.log` 都用它。計算機若用以 10 為底的 log，−log(0.5) 得 .30103，無法核對本節的 .693147。

本 repo 固定用 `total=5×box+objectness+classification`，代入為：

\[
L=5\times.109375+.693147+.693147=1.933169
\]

三項雖然都無單位，量尺與平均方式不同。框乘 5，是調整它在總目標中的份量；本例框約 .109，另兩項約 .693。這個 5 是固定示範設定，沒有搜尋最佳值，而且 loss 數值占比不等於梯度占比。

## 平均的分母到底是什麼

`pos=target['positive']` 是 `[B,S,S]` 的 True／False mask，`pos.sum()` 是正格數 \(N_{\text{pos}}\)（程式解說也寫作 Npositive），`pos.any()` 問有沒有任何正格。本例框被 mask 選成 `[1,4]`、類別為 `[1,2]`，類別答案是 `[1]`；objectness 仍為 `[1,4,4]`，全格保留。

手機上可左右滑動表格，查看完整欄位。

| loss | 哪些項目進平均 | 分母 | 本例 |
| --- | --- | --- | --- |
| box | 每個正格的四個框座標 | \(4N_{\text{pos}}\) | 4 |
| objectness | 每張圖的全部格子 | \(B S S\) | 16 |
| classification | 每個正格的一個 CE | \(N_{\text{pos}}\) | 1 |

這三個分母不能互換。複製圖片，或加入空圖，可能改變不同項目的分母；頁尾練習會用這個差別預測梯度。

`reduction='mean'` 是加總再除以個數，`'sum'` 則只加總。sum 會讓圖片／物件數的改變直接改變共享 CNN 權重收到的總梯度；mean 對完全相同的批次複製可抵消這個變化。每個輸出 logit 的梯度與共享權重的總梯度仍不同，下面會分開說明。

座標的尺度也不能任意換掉。同樣差 16 pixel，以 pixel 算平方是 256；wh 除全圖 64 後是 `(16/64)²=.0625`，小 4096 倍；xy 除格寬 16 後是 1，小 256 倍。若換回 pixel 的 MSE，框項的權重需要重新決定。

??? note "一般式：把同一規則寫給任意 batch"

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

    沒有正格（\(N_{\text{pos}}=0\)）時，程式直接令 \(L_{\text{box}}\) 與 \(L_{\text{cls}}\) 為 0，原因見正文的全空圖分支。本例代入 B=1、S=4、\(N_{\text{pos}}=1\)。

## 把這個分工寫成程式

以下為 `miniyolo/losses.py` 的 `grid_loss` 簡化版，省略輸入檢查。完整 lesson case 直接呼叫 `grid_loss(prediction,target)`：

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
else:  # 沒有正格（空圖），見下方全空圖分支
    box = (pred[..., :4] * 0).sum()
    cls = (pred[..., 5:] * 0).sum()
obj = F.binary_cross_entropy_with_logits(pred[..., 4],
                                         target['objectness'], reduction='mean')
total = 5 * box + obj + cls
```



`pred[..., :4]` 保留 `(b,gy,gx)` 三軸，只取框四項；同一個 `[pos]` 把預測與答案都選成 `[Npositive,4]`，第 i 列對應同一格。類別同樣只選正格，objectness 則不選。

MSE 只相減、平方、平均，因此框要自己先 sigmoid。BCE 的函式名含 `with_logits`，表示直接接 logits，內部完成對應的 sigmoid 計算；`cross_entropy` 也直接接類別 logits，內部用數值穩定的 log-softmax 計算 CE。這兩項不用先手動轉成機率，否則會多轉一次。

## 一個 loss，如何檢查推動方向

完整程式建立 `torch.zeros(1,4,4,7,requires_grad=True)`，讓 PyTorch 記錄這個人工輸出的梯度。對 total 呼叫 `backward()` 後，`prediction.grad` 表示：某個 logit 稍微增大，total 會如何改變。

BCE 的 objectness 梯度為 `(sigmoid(z)−target)/(B×S×S)`。本例分母 16，所以正格 `(0.5−1)/16=−.03125`，負格 `(0.5−0)/16=+.03125`。total 的 5 只乘框，這裡不乘 5。

可以不微分，改用小變化核對。正格 z 從 0 增至 .001，單格 BCE 約由 .693147 降到 .692647，變化除以 .001 得斜率約 −.5；再除 16 得 −.03125。負格 BCE 約升至 .693647，斜率則為 +.5/16=+.03125。

??? note "推導：為什麼分子是 sigmoid 減 target"

    只看一格。令 \(p=\sigma(z)\)，單格 BCE 是 \(\ell(z)=-\bigl[t\ln p+(1-t)\ln(1-p)\bigr]\)。

    本段另外用兩條微分規則。若 u 是 z 的函數，記 \(u'=du/dz\)：\(e^u\) 對 z 的導數是 \(e^u u'\)；\(1/u\) 的導數是 \(-u'/u^2\)（u 不為 0）。

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


類別梯度為 `(softmax−one-hot)/Npositive`。one-hot 把正確類別記為 1、其他類別記為 0；紅類答案是 `[1,0]`，所以梯度為 `[.5−1,.5−0]/1=[−.5,+.5]`。classification 的係數也是 1，分母是正格數 1，而非 16。

class1 不是答案，仍收到正梯度：softmax 分母連結兩類，class1 增大會降低紅類機率，使 CE 增大。單獨將 class0 改成 .001，CE 約降到 .692647；單獨將 class1 改成 .001，則約升到 .693647，斜率正好為 −.5、+.5。

??? note "推導：為什麼類別梯度是 softmax 減 one-hot"

    只看一個正格，正確類別是 c，類別 logits 是 \(z_0,\dots,z_{C-1}\)。softmax 是 \(p_k=e^{z_k}/\sum_j e^{z_j}\)，所以這一格的 CE 可以拆成兩項：

    \[
    \ell=-\ln p_c=-\ln e^{z_c}+\ln\sum_j e^{z_j}=-z_c+\ln\sum_j e^{z_j}
    \]

    對其中一個 \(z_k\) 偏微分（只讓 \(z_k\) 變，其他 logits 不動）：

    - 第一項 \(-z_c\)：k=c 時導數是 −1，其他類別是 0。記 \(y_k\) 為 one-hot 的第 k 個數（k=c 時是 1，否則是 0），這一項的導數就是 \(-y_k\)。
    - 第二項：令 \(u=\sum_j e^{z_j}\)，它對 \(z_k\) 的導數是 \(e^{z_k}\)。用上一個推導的規則（\(\ln u\) 的導數是 \(u'/u\)），得到 \(e^{z_k}/u=p_k\)。

    合起來：

    \[
    \frac{\partial\ell}{\partial z_k}=p_k-y_k
    \]

    class loss 是 \(N_{\text{pos}}\) 個正格 CE 的平均；每個正格的類別 logits 只出現在自己那一格的 CE 裡，box 與 objectness 也不含它們，所以 total 對它們的梯度是 \((p_k-y_k)/N_{\text{pos}}\)。本例 \(p=[0.5,0.5]\)、\(y=[1,0]\)、\(N_{\text{pos}}=1\)：class0 是 −0.5，class1 是 +0.5。


若直接更新 logit，梯度下降 `z_new=z−ηg` 會把負梯度的 z 調大、正梯度的 z 調小。因此正格 objectness 與紅類 logit 應上升，負格 objectness 與正格藍類 logit 應下降。本節只檢查方向，不做這個更新。

真正訓練會更新 CNN 權重。每個 logit 的梯度還要乘上它對權重的變化率，再把所有帶方向的貢獻加總。不能把所有 logit 梯度直接當作某個權重梯度。本例負格 objectness 的梯度量合計 `15×.03125=.46875`，比單一正格 .03125 多，說明背景監督很多，模型可能先學到處說沒有物件；權重如何變，還要看特徵與共享關係。

框與類別的負格梯度則必須全為 0，因為 mask 沒把它們送進 loss。若不為 0，背景正被教某個無意義框或類別。這也是有限 total 之外必須檢查的事情。

## 整批都是空圖，還有什麼可學

有空圖但批次中仍有正格時，框／類別平均照常進行。若整批沒有正格，mask 後是 `[0,4]` 與 `[0,C]`，對零個數取平均會得到 NaN（Not a Number，無法表示有效數值），total 也跟著壞掉。`miniyolo/train.py` 會拒絕非有限 loss。

不能刪空圖來迴避，因為它仍有全部格子的 objectness=0 答案。程式用 `pos.any()` 分支：無正格時，框／類別改為 `(pred*0).sum()`，得到值為 0 的單值 tensor（scalar，shape `[]`），梯度也為 0。objectness 仍對 16 格平均，零 logits 時 total=.693147。

乘 0 再加總，讓零值仍連在計算圖上；計算圖是 PyTorch 保存誰由誰算出的關係，backward 沿它回傳梯度。這樣即使單獨對框零值 backward 也能執行，常數 0 則沒有這個關係。若只對 total backward，兩者給相同梯度。

## 執行，分別核對值、方向與空圖

在 Colab 執行頁首 notebook，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-loss.py`。四行結果依序核對：box .109375、objectness .693147、classification .693147、total 1.933169；正／負格 objectness 梯度 −.03125／+.03125；正格類別梯度 `[−.5,.5]`；最後 `empty image: finite backward, box/class=0`。

空圖不能只查 backward 成功、梯度有限：少了分支時，空平均的梯度仍可能有限，壞的是 loss 值。程式還要求 box／class 恰為 0、total 約 .693147。這次沒有訓練，也沒有 AP 效果結果。

MSE 方便逐座標核對，代價是沒有直接衡量框的重疊程度；背景格很多也讓「降低 loss」可能先由背景完成。先核對資料、mask 與梯度，再考慮 loss 權重。第 11 章會另研究 IoU 類定位 loss。

## 有限 loss 也可能算錯

常見錯誤分兩類。第一類不會報錯，得到的是看似正常的有限 loss，要拿手算的 loss 與梯度來核對才會發現。本節完整程式的斷言做的就是這種核對，下面三種錯，每一種都會讓其中一個斷言失敗：

- 先 sigmoid 再傳給 `binary_cross_entropy_with_logits`。它內部會再做一次 sigmoid，等於做了兩次。`torch.nn` 裡的 `BCEWithLogitsLoss` 是它的 Python class 版本（先建立物件再呼叫），算的是同一件事，先 sigmoid 再傳給它也是同樣的錯。本例 objectness 會從 0.693147 變成 0.942827，檢查 objectness 等於 ln 2 的斷言會失敗。
- 先 softmax 再傳給 CE，等於做了兩次 softmax。本例兩個類別 logits 都是 0，第一次 softmax 後兩類仍相等（各 0.5），所以錯誤寫法算出的類別 loss 仍是 0.693147，和正確值一樣，只看 loss 抓不到這個錯。差別在梯度：本例只有 1 個正格，正確的類別梯度是 `[-0.5, 0.5]`，錯誤寫法只有一半的 `[-0.25, 0.25]`，所以檢查正格類別梯度的斷言會失敗。只剩一半，是因為第一次 softmax 縮小了差距：class0 的 logit 改成 0.001 時，兩個 logits 相差 0.001，第一次 softmax 後的兩個數卻只相差約 0.0005，CE 的變化也就只剩一半。
- 把負格也送進 CE。本章負格的 class_ids 是預設值 0，正好是紅色的編號，所以不會報錯，卻會把背景格（本例 15 格）教成紅色。本例每格的類別 logits 都是 0，每格的 CE 都是 ln 2，所以類別 loss 仍是 0.693147；但負格收到了類別梯度，檢查負格類別梯度為 0 的斷言會失敗。類別 loss 一定要先用 pos 挑出正格。

第二類會報錯，或得到 NaN：

- 若像第 5 章那樣把負格的 class_ids 填 −1，再送進 CE，會報錯。
- 沒有正格時照樣取平均，會得到 NaN（見上面的全空圖分支），完整程式檢查空圖 box 與 class 都是 0 的斷言會失敗。


## 這套 loss 與 YOLOv1 的關係

[YOLOv1 原論文](https://arxiv.org/abs/1506.02640) 也讓定位與分類一起學，但原版各項使用平方誤差再加總，座標乘 λcoord=5，無物件 confidence 項乘 λnoobj=.5。本章則是平均後的 MSE、BCE、CE；同一個 5 不能直接比較份量，也不能把本頁當成完整原版 loss。

??? note "原版 YOLOv1 loss 的其他設計"

    下面三件事，本章都沒有做：

    - 平方根寬高：原版預測寬、高的平方根，再算誤差，讓小框的同樣誤差在 loss 裡更顯眼。
    - IoU 當 confidence 目標：原版負責物件的那個框，confidence 的目標是預測框與真值框的 IoU（交集面積除以聯集面積），不是固定的 1。
    - 多框責任規則：原版每格預測多個框（PASCAL VOC 資料集的設定是 2 個），訓練時只讓和真值框當下 IoU 最高的那個框負責。

    λnoobj 的用意：大多數格子沒有物件，它們的 confidence 都被往 0 推，常常蓋過有物件格子的梯度；原版用 λnoobj=0.5 減弱這一項。本章沒有這個權重，本頁的背景梯度合計會用數字呈現這個現象。

    本章框項乘的 5，數值與原版 λcoord 相同。不過本章各項先取平均，原版是加總，同一個 5 在兩邊占的份量不能直接對照。

## 自主練習

三題都先用紙筆算，再展開答案。練習 3 的答案附了核對用的程式。

**練習 1：** 把 batch 複製成兩張完全一樣的圖（B=2），三項 mean loss 變多少？objectness 每個 logit 的梯度變多少？如果兩張圖的 logits 由同一組 CNN 權重算出，權重收到的總梯度又會怎樣？

??? note "參考答案"

    loss：不變。兩張圖的每一項都和原來一樣，分子與分母一起變兩倍，平均還是原值；若使用 sum 才會翻倍。

    每個 logit 的梯度：減半。objectness 的分母從 16 變 32，正格從 −0.03125 變成 −0.015625，背景從 +0.03125 變成 +0.015625。box 與 class 的分母也加倍（正格從 1 個變 2 個），梯度同樣減半：每個正格的類別梯度從 [−0.5, 0.5] 變成 [−0.25, 0.25]。這和〈收益、代價與常見錯誤〉裡先 softmax 的錯誤值數字相同，只是碰巧：這裡是正確寫法、分母變成 2；那裡是只有 1 個正格時多做了一次 softmax。

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

    本例 a=b=0，加 10 之後兩類仍各是 0.5，CE 仍是 \(\ln 2\approx0.693147\)。機率沒變，所以類別梯度（softmax 減 one-hot）也仍是 [−0.5, 0.5]。

**練習 3：** batch 改成〔紅框圖, 空圖〕（B=2），四項 loss 與 objectness 梯度各是多少？正格的類別梯度會變嗎？

??? note "參考答案"

    B×S×S=32。box 和 class 只平均正格，正格仍只有紅框那 1 格，所以數值不變：box 0.109375、classification 0.693147。正格的框梯度與類別梯度也不變；類別梯度仍是 [−0.5, 0.5]，因為分母是正格數，仍是 1。objectness 改對 32 格平均，但每格仍是 ln 2，平均還是 0.693147，所以 total 仍是 1.933169。

    改變的是 objectness 梯度：分母從 16 變成 32，正格是 −1/64=−0.015625，31 個負格（紅框圖 15 格加空圖 16 格）各是 +1/64=+0.015625。

    完整程式用下面兩個斷言，檢查原例 1 個正格、15 個負格的 objectness 梯度是 ±1/32：

    ``` { .python data-excerpt="lesson_cases/07-loss.py" }
    assert torch.allclose(prediction.grad[..., 4][pos], torch.full((1,), -1/32))
    assert torch.allclose(prediction.grad[..., 4][~pos], torch.full((15,), 1/32))
    ```

    這一題要改成 `torch.full((1,), -1/64)` 與 `torch.full((31,), 1/64)`。想用程式核對時，不要改原本的 `main()`：在 notebook 最後一格下面新增一個程式碼儲存格，貼上下面的程式另外測試（在自己的電腦上，也可以存成 .py 檔，在 repo 根目錄用 `PYTHONPATH=. python` 執行）。最後一個斷言檢查紅框圖正格 (0,1,1) 的類別梯度。

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
    assert torch.allclose(prediction.grad[0, 1, 1, 5:], torch.tensor([-.5, .5]))
    ```

    把程式裡的 `[scene, empty]` 改成 `[scene, scene]`，也能核對練習 1；這時前兩個斷言要改成 `torch.full((2,), -1/64)` 與 `torch.full((30,), 1/64)`，類別梯度的斷言要改成 `torch.tensor([-.25, .25])`。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-loss.json)

??? example "展開本次實際輸出"

    ```text
    {'total': 1.933169, 'box': 0.109375, 'objectness': 0.693147, 'classification': 0.693147}
    positive/background objectness gradients -0.03125 0.03125
    positive-cell class gradients [-0.5, 0.5]
    empty image: finite backward, box/class=0
    ```

<!-- curriculum-evidence:end -->
