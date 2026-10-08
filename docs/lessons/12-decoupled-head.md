# C.12.2 Decoupled head：分類和定位在哪裡分工

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/12-decoupled-head.ipynb){ .md-button }

上一頁已能由候選點的四邊距離還原框。放回圖片偵測，這個點還要回答「是哪一類」。兩個答案來自同一份 CNN 特徵，但希望後段能各自整理需要的訊息：物件往右移 1pixel，類別仍是紅矩形，定位的左距離卻少 1/stride、右距離多 1/stride。分類希望對小位移保持答案，定位則要追蹤邊的變化。

**Decoupled head（解耦 head）**讓共用 backbone 之後分成定位與分類兩串卷積，各有自己的後段權重。它不要求原本 coupled（耦合）head 不能完成兩任務；分支改的是兩種工作在哪裡共用、在哪裡各自轉換。

第 7 章的一層 1×1 同時輸出框、obj 和類別，是 coupled 的例子。這次沿用 [四正距離、softplus、Smooth L1](12-anchor-free.md)，只做一尺度、兩類、沒有 objectness；YOLOX 在 2021 年已採用 [decoupled head](https://arxiv.org/abs/2107.08430)，YOLOv8 也採用，但本例不重現完整結構或品質比較。

## 同一份特徵，兩份不同的答案

輸入 `x [2,3,4,4]` 是 `torch.rand` 的隨機數，沒有物件；target 也手工固定，只用來追梯度。B=2 張、H=W=4，在每個位置輸出四距離與兩類分數。兩路讀取同一個 f：

```text
x [2,3,4,4] → backbone：3×3(3→8) → ReLU → f [2,8,4,4]
f → box_branch：3×3(8→8) → ReLU → 1×1(8→4) → boxes [2,4,4,4]
f → class_branch：3×3(8→8) → ReLU → 1×1(8→2) → logits [2,2,4,4]
```

`boxes` 的第 1 軸是四個 ltrb 原始數，尚未經 softplus，**不是已解碼 xyxy**。`F.softplus(boxes)`纔是正距離，與四邊都 1.5 格的 `distances` 比 Smooth L1。`logits`是每點兩類的原始分數，和 `classes` 用 BCE 比較。

4×4 有 P=16 點，把 H、W 攤平、再將 channel 移到末尾，框輸出就是上一頁的 `[B,P,4]`，分類是 `[B,P,C]`。數字同為 4 的軸仍分工不同，不可把 channel 當空間邊長。

這個梯度示範把**所有位置**都當正樣本，class 0 target=1、class 1=0，刻意不做 assignment。實際 YOLOv8 只在選出的正樣本算框 loss；背景也算分類 loss，全部類別 target=0。它沒有第 7 章的 objectness，是否有物件改由類別分數表達。

所以兩類各用 sigmoid，不用總和必為 1 的 softmax：sigmoid 允許兩類一起接近 0，代表背景。正式正樣本 target 還依預測品質縮成 0～1 小數，本頁先用硬答案 1／0，下一頁再把選點接到 target。不能把本例的全位置正樣本當成完整偵測監督。

## 三次 backward 解開「分開」的意思

先做**一次 forward**算 box_loss 與 cls_loss，再對同一次結果分別反傳。框 loss 只用 boxes，沿 box_branch 回到 f、backbone；它沒有用 class_branch，所以那支參數的 `.grad` 應為 None。清梯度後單獨反傳分類 loss，情況反過來。

``` { .python data-excerpt="lesson_cases/12-decoupled-head.py" }
boxes, logits = model(x)  # forward：同一份特徵 f 交給兩條分支
box_loss = F.smooth_l1_loss(F.softplus(boxes), distances)  # 只用到 boxes
cls_loss = F.binary_cross_entropy_with_logits(logits, classes)  # 只用到 logits
model.zero_grad()  # 所有參數的 .grad 設成 None
box_loss.backward(retain_graph=True)  # 第一次：只反傳框 loss
assert model.box_branch[0].weight.grad is not None
assert model.class_branch[0].weight.grad is None
# clone() 複製一份獨立的數字，之後 .grad 變了也不影響它
box_backbone_grad = model.backbone.weight.grad.clone()
model.zero_grad()
cls_loss.backward(retain_graph=True)  # 第二次：只反傳分類 loss
assert model.box_branch[0].weight.grad is None
class_backbone_grad = model.backbone.weight.grad.clone()
```

兩分支是 nn.Sequential，`[0]`取第一個 3×3；程式對照它們的 weight.grad。每次 `model.zero_grad()` 預設設回 None，纔可區分這次有沒有走進該參數。前兩次加 `retain_graph=True`，保留共用 backbone 那段的反傳暫存；否則第一次反傳釋放後，第二次會報 `Trying to backward through the graph a second time`。

兩種 loss 都到了 backbone，因此解耦沒有讓兩網完全獨立。記共用權重為θ，單獨反傳的梯度為 `g_box=∂L_box/∂θ`、`g_cls=∂L_cls/∂θ`；clone 保留各自數字，不受後面的.grad 改變。總 loss 是兩者相加，第三次清梯度再反傳：

``` { .python data-excerpt="lesson_cases/12-decoupled-head.py" }
model.zero_grad()
(box_loss + cls_loss).backward()  # 最後一次反傳，不必 retain_graph
assert torch.allclose(model.backbone.weight.grad, box_backbone_grad + class_backbone_grad, atol=1e-7)
```

和的導數是導數的和，兩路在共用 f 會合，因此 backbone 梯度爲 `g_box+g_cls`。torch.allclose 逐值核對，容許 1e-7 浮點差。若第三次前忘清梯度，就混入第二次舊梯度，這項檢查失敗。最後 optimizer.step 更新，並核對 backbone 權重改變；三次 backward 本身不等於學習更新。

## 用 cosine 看兩份梯度的方向

共用 backbone 仍可能同時面對兩任務相反的更新要求。把兩份 weight 梯度 `[8,3,3,3]` 各攤平成 216 維向量（不含 bias），用**cosine similarity（餘弦相似度）**看方向：

\[
\cos\varphi=\frac{g_{\mathrm{box}}\cdot g_{\mathrm{cls}}}{\lVert g_{\mathrm{box}}\rVert\lVert g_{\mathrm{cls}}\rVert}.
\]

這是向量夾角公式，維數雖多，仍用內積除長度乘積。接近 1 是同方向，−1 是相反，0 是大致正交。手算 `(3,4)` 與 `(4,−3)` 內積 0、cosine0；與 `(−3,−4)` 是−1；與 `(6,8)` 是 1。

``` { .python data-excerpt="lesson_cases/12-decoupled-head.py" }
# flatten() 把 [8,3,3,3] 攤平成 216 個數；dim=0 表示沿攤平後的這一軸計算
cosine = F.cosine_similarity(box_backbone_grad.flatten(), class_backbone_grad.flatten(), dim=0)
```

本次 **−0.0073** 接近 0，夾角約 90.4°，應讀成大致正交，不是明顯相反。明顯負值才表示某任務的小更新會使另一 loss 上升，兩梯度相加抵銷部分，是任務衝突。這仍是共用 backbone 的診斷，不能用一個負號宣佈解耦無效；各分支後段依然有自己的轉換。固定 seed、人工 target 的一次值也不能當整任務平均。

??? note "為什麼內積的正負代表「一致」或「衝突」？"

    用一階近似來看。步伐很小時，loss 的變化約等於「梯度和移動量的內積」，就像用切線估計曲線。照分類梯度走一小步，也就是把 \(\theta\) 改成 \(\theta-\eta\,g_{\text{cls}}\)（\(\eta\) 是學習率），框 loss 大約變成

    \[
    L_{\text{box}}(\theta-\eta\,g_{\text{cls}})\approx L_{\text{box}}(\theta)-\eta\,g_{\text{box}}\cdot g_{\text{cls}}
    \]

    內積為正時，框 loss 也跟著下降，兩個任務互相幫忙；內積為負時，這一步會讓框 loss 變大，這就是衝突；內積接近 0 時，這一步對框 loss 幾乎沒有影響。cosine 是內積除以兩個長度，正負號和內積相同，又不受梯度大小影響，所以適合拿來比較方向。

## 可核對的成本

分開輸出層本身沒有新增分工：8→6 的 1×1，六個輸出各有自己一列權重，等價於並排 8→4 與 8→2。真正的分工是前面**各有自己的 3×3**，因此兩 loss 不再共同更新同一串後段卷積。

| 部分 | 帶 bias 參數 |
| --- | --- |
| backbone，3×3 的 3→8 | `8×3×3×3+8=224` |
| 每支 3×3 的 8→8 | `8×8×3×3+8=584` |
| 框輸出 1×1 的 8→4 | `4×8+4=36` |
| 類別輸出 1×1 的 8→2 | `2×8+2=18` |

合計 `224+584×2+36+18=1446`，分支共 1222；一層 coupled 8→6 的 1×1 只需 54。這個基線同時少層、少參數，若有效果差異不能全歸因解耦。

??? note "保留一層 3×3 的 coupled head 有多少參數？"

    讓 coupled head 也先經過一層 3×3：f → 3×3 卷積（8→8）→ ReLU → 1×1 卷積（8→6），前 4 個輸出 channel 給框、後 2 個給類別。參數是 `584+54=638`。它和每條分支一樣是一層 3×3 加一層 1×1、中間 8 個 channel，對齊的是層數與 channel 數；參數量仍只有兩條分支（1,222）的一半左右。

設計需選兩支的層數、通道數；分支增加捲積與反傳暫存 activation（中間張量），小 backbone 上也可能成為主要成本，資料少時更多參數可能過擬合。公平比較先決定對齊通道、層數、參數或延遲，再固定資料與訓練預算；速度要量整條推論路徑。

執行 `PYTHONPATH=. python lesson_cases/12-decoupled-head.py` 或頁首 Colab，核對兩輸出 shape、分類單獨反傳時框支 grad=None、backbone 總梯度之和、cosine 與 1446 參數。`classification-only…` 和 `…verified`是通過對應斷言後的固定文字。這只證明梯度分工與成本，沒有 AP。

## 反傳診斷的三個陷阱

忘清梯度會累加；`.grad=None` 與全零 tensor 也不同。第二次 zero_grad 若用 `set_to_none=False`，第一次已有的框支梯度會變全零，分類沒走入它仍保持零，但 `is None` 檢查不通過。改第一次則沒有差，因那時本來都是 None。

也不能在分支前誤用 detach。例如 `class_branch(f.detach())` 仍讓分類支更新、loss 下降，卻切斷到 backbone 的路；第二次反傳 backbone.grad 為 None，取 `.clone()` 就報 AttributeError。本例兩任務都要訓練共用特徵，所以 f 不 detach。

自主練習：

1. 只把分類 loss 乘 2：第三次改成反傳 \(L_{\text{box}}+2L_{\text{cls}}\)。先預測 backbone 的總梯度，以及框分支、類別分支的梯度各會怎麼變。再改完整程式驗證 backbone 的總梯度；兩條分支的梯度怎麼變，程式不會檢查，請對照參考答案。完整程式就是 Colab 裡「本節可修改的完整實驗」下面那一格程式，內容和 `lesson_cases/12-decoupled-head.py` 相同。要一起改三處：

    - `(box_loss + cls_loss).backward()` 改成 `(box_loss + 2 * cls_loss).backward()`。
    - 下一行斷言的預期值 `box_backbone_grad + class_backbone_grad`，改成 `box_backbone_grad + 2 * class_backbone_grad`。
    - 印出 `total backbone gradient = box gradient + class gradient: verified` 的那行 print，把其中的 `box gradient + class gradient` 改成 `box gradient + 2 * class gradient`。

    第二次單獨反傳的 `cls_loss.backward(retain_graph=True)` 保持不變，理由見參考答案。

2. 開放題：若想對照 coupled head，先寫下要對齊參數量還是延遲（latency：跑一次推論花的時間），再用相同資料與訓練步數比較。本例的輸入是隨機數，只能比參數量與梯度路徑。

??? note "參考答案"

    **第 1 題**：backbone 的總梯度變成 \(g_{\text{box}}+2g_{\text{cls}}\)，改完的斷言會通過。類別分支的梯度也變成 2 倍，框分支的梯度不變。所以答案不是「只有類別分支的梯度變兩倍」：共用的 backbone 也收到 2 倍的分類訊號。

    第二次單獨反傳要保持原本的 `cls_loss`，`class_backbone_grad` 才代表 \(g_{\text{cls}}\) 本身。若那行也改成 `2 * cls_loss`，`class_backbone_grad` 就已經是 \(2g_{\text{cls}}\)；預期值再乘 2，會變成 \(g_{\text{box}}+4g_{\text{cls}}\)，和實際的 \(g_{\text{box}}+2g_{\text{cls}}\) 對不上，斷言就會失敗。

    **第 2 題**：沒有標準答案。舉一個梯度路徑的差別：保留一層 3×3 的 coupled 版本（f → 3×3 → ReLU → 1×1 輸出 6 個 channel，共 638 個參數）裡，那層 3×3 在只反傳框 loss、只反傳分類 loss 時都有梯度，和 backbone 一樣由兩個任務共用；本節的兩層 3×3 則各只收到一種 loss 的梯度。至於最後那層 1×1，只反傳框 loss 時，負責類別的 2 列梯度是全 0 而不是 None：整個 1×1 權重是同一個參數，只要有一部分用到，`.grad` 就是完整的 tensor。就梯度路徑來說，全 0 和 None 都表示這個 loss 沒有梯度流到這些權重，只是 PyTorch 的記法不同。

參考來源：[Ultralytics Detect 中 cv2／cv3 的分支定義](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)。cv2、cv3 是 Ultralytics `Detect` 類別裡兩條分支的成員名：cv2 是框分支，cv3 是類別分支，和 OpenCV 的 `cv2` 套件無關。YOLOv8 的每條分支是兩層 3×3 卷積再接一層 1×1 卷積。連結檔案裡的類別分支 cv3 有兩種寫法：YOLOv8 用的是 `self.legacy` 為真的那一種（兩個 `Conv(…, 3)` 再接 `nn.Conv2d(…, 1)`），另一種是較新版本的類別分支。本節的兩層小分支（一層 3×3 加一層 1×1）、全正樣本與 Smooth L1 是教學簡化。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-decoupled-head.json)

??? example "展開本次實際輸出"

    ```text
    box / class shapes: (2, 4, 4, 4) (2, 2, 4, 4)
    classification-only backward leaves box branch grad=None
    shared-backbone gradient cosine: -0.0073
    total backbone gradient = box gradient + class gradient: verified
    parameters: 1446
    ```

<!-- curriculum-evidence:end -->
