# 12.2 Decoupled head：分類和定位在哪裡分工

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/12-decoupled-head.ipynb){ .md-button }

偵測模型要從同一份特徵回答兩個問題：「這裡是哪一類？」和「框的邊在哪裡？」本節把 head 分成框、類別兩條分支，再用三次 backward 追蹤梯度走哪條路。讀完你能說出 decoupled head 哪裡分開、哪裡仍然共用，能驗算兩個 loss 的梯度在共用的部分怎麼相加，也能算出分支多花了多少參數。

前置：Conv2d、反向傳播、幾個 loss 相加成的總 loss（第 7 章的 `total`），以及 [12.1 Anchor-free](12-anchor-free.md)（候選點到框四邊的距離 ltrb、softplus、Smooth L1）。backbone 是把影像轉成特徵的主幹，head 是把特徵轉成預測的輸出模組。

本節說的「分開」，是和 coupled head 對照：

- **coupled head（耦合 head）**：框和類別經過同一串卷積，最後由同一層 1×1 卷積同時輸出。第 7 章的 head 就是這種：一層 1×1 在每一格一次輸出 7 個數（4 個框值＋objectness＋2 個類別）。
- **decoupled head（解耦 head）**：backbone 之後分成框、類別兩串卷積，各自輸出自己的預測。本節的框分支在每個位置輸出 4 個數，類別分支輸出 2 個數。

為什麼要分開？分類常要認出形狀，辨識這是什麼東西（語意）；定位要保留精細的位置。下面是直覺說明，本節的實驗不驗證準確度。例如第 7 章的紅色矩形往右移 1 畫素，它仍是紅色矩形，所以分類希望特徵對這種小位移不敏感。但若用 12.1 的四邊距離表示框，同一個候選點到框左邊的距離 l 要少 1/stride 格，到右邊的距離 r 要多 1/stride 格（12.1 設定 stride 為 8 畫素／格，這時是 1/8 格），所以定位需要特徵跟著變。若框和類別由同一串卷積一起輸出，權重得同時滿足這兩種要求，兩者可能互相拉扯，限制各自能學的轉換。

2021 年的 [YOLOX](https://arxiv.org/abs/2107.08430) 已在 YOLO 系列採用 decoupled head；YOLOv8 這類偵測器也把 head 分成分類分支與框回歸分支。本節對照的是 Ultralytics YOLOv8 的寫法。我們從共用的 CNN 特徵出發，並做三項簡化：

- 只用一個尺度。
- 框直接輸出四個連續的距離（不是之後 DFL 節的分布）。
- 只有兩個類別，各用一個 sigmoid 分數（YOLOv8 也是每個類別各一個 sigmoid 分數、沒有 objectness；類別數則依資料而定，例如常用的 COCO 資料集有 80 類）。

後面幾節講到的官方 YOLO 版本，head 都保留這種分支結構（例如 [12.4 節](12-dfl.md)的 DFL 就接在框分支上）。但這次的觀察只能支持兩件事：一是各 loss 的梯度只流進自己的分支與共用的 backbone，不進另一條分支（梯度隔離）；二是分支多了多少參數。不能據此宣稱真實資料上的準確度提高。

## 先把各軸說清楚

輸入是兩張 4×4 的「影像」，shape `[B,3,H,W]=[2,3,4,4]`。它們其實是 `torch.rand` 產生的 0 到 1 之間的隨機數，裡面沒有真的物件；target 也是人工固定的值。這些資料只用來追蹤梯度走哪條路。資料流如下，括號裡是輸入→輸出的 channel 數：

```text
x [2,3,4,4] → backbone：3×3 卷積(3→8) → ReLU → f [2,8,4,4]
f → 框分支：3×3 卷積(8→8) → ReLU → 1×1 卷積(8→4) → boxes [2,4,4,4]
f → 類別分支：3×3 卷積(8→8) → ReLU → 1×1 卷積(8→2) → logits [2,2,4,4]
```

完整程式的 `forward` 先算 `f = F.relu(self.backbone(x))`，再把同一個 `f` 交給兩條分支。每條分支最後的 1×1 卷積是輸出層：在每個位置把 8 個 channel 組合成 4 個或 2 個數。

| 路徑 | 最後 shape | 每個位置表示什麼 |
| --- | --- | --- |
| 共用 backbone（`backbone`） | `[2,8,4,4]` | 兩個任務共同讀取的特徵 |
| 框分支（`box_branch`） | `[2,4,4,4]` | 四個 channel 是 `ltrb` 距離的原始輸出 |
| 類別分支（`class_branch`） | `[2,2,4,4]` | 兩個 channel 各是一個類別的 logit |

框分支輸出 `[2,4,4,4]` 的第 0 軸是 B=2（兩張圖），第 1 軸是四個 channel，第 2、3 軸才是高度、寬度。4×4＝16 個位置各是一個候選點，套用 12.1 節的記號就是 P＝16；把高、寬兩軸攤平成 P，再把 channel 移到最後，`[2,4,4,4]` 就是 12.1 節表格的 `[B,P,4]`，`[2,2,4,4]` 就是 `[B,P,C]`。ltrb 是候選點到框左、上、右、下四邊的距離（上一節 anchor-free 的表示）。程式的 `boxes` 變數就是這種原始輸出：還沒經過 softplus 的原始數，可正可負，不是機率（術語表的 logits 也包含這種框的原始輸出）。本頁單說 logits，都指程式裡同名的變數 `logits`，也就是類別分支的輸出。框的原始輸出經 `F.softplus(boxes)` 才變成正距離，再和 1.5 格的 target（程式的 `distances`）比較；`boxes` 不是已解碼的 xyxy 框。

這個 head 沒有 objectness（第 7 章表示「這格有沒有物件」的那個輸出），兩個類別各有一個 sigmoid 分數，用 BCE 學。在這個示範中，所有位置都當成正樣本：類別 target（程式的 `classes`）的 class0 是 1、class1 是 0。這裡刻意省去 assignment（候選與真值的責任分配），讓我們只追蹤 head 的梯度。完整的 YOLOv8 這類偵測器只在選定的正樣本上計算框 loss；背景位置仍要算分類 loss，而且所有類別的 target 都是 0：這就是「分類負訊號」。

??? note "沒有 objectness，背景怎麼學？為什麼不用 softmax？"

    第 7 章的設計是：objectness 在每一格學「有沒有物件」，類別只在正格用 softmax 與交叉熵學「紅還是藍」，背景不算分類 loss。YOLOv8 這類設計沒有 objectness，每個類別各有一個 sigmoid 分數；背景位置所有類別的 target 都是 0，「這裡有沒有物件」改由類別分數一起表達。

    不用 softmax，是因為 softmax 讓各類機率的總和固定為 1，表達不了「哪一類都不是」。sigmoid 讓每類各自落在 0 到 1 之間、彼此獨立，所以每類各用一個 BCE。第 10 章提過，YOLOv3 原版也對每個類別各做 sigmoid。

    YOLOv8 正樣本的類別 target 也不是固定的 1，而是依這個預測的品質給的 0 到 1 之間的分數；本節示範簡化成固定的 1 和 0。

## 三次 backward 解開「分開」的意思

程式先做一次 forward，算出兩個 loss：框 loss（程式的 `box_loss`）用 Smooth L1，只用到框分支的輸出 `boxes`；分類 loss（`cls_loss`）用 BCE，只用到類別分支的輸出 `logits`。接著對同一次 forward 反傳三次：

1. 第一次只反傳框 loss。框 loss 只經過 backbone 與框分支，反傳走不到類別分支：框分支的參數有梯度，類別分支的 `.grad` 應是 `None`。
2. 第二次先清空梯度，再只反傳分類 loss：結果反過來，框分支的 `.grad` 是 `None`。
3. 第三次反傳兩者的和（見下方）。

前兩次都會到達共用的 backbone，因為兩條路都使用同一份特徵 `f`。因此 decoupled head 不是兩個完全獨立的網路，也不保證共用特徵沒有任務衝突。任務衝突是指兩個 loss 想把共用 backbone 的權重往相反方向推，相加時互相抵銷一部分。後面會用兩份 backbone 梯度夾角的 cosine 檢查有沒有這種情況。

完整程式裡的對應片段如下，前三行是 forward 和兩個 loss：

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

`box_branch`、`class_branch` 都是 `nn.Sequential`（第 3 章：把幾層依序串起來），所以 `[0]` 取出的是第一層 3×3 卷積，`.weight.grad` 是它權重的梯度。

`model.zero_grad()` 和前面章節的 `optimizer.zero_grad(set_to_none=True)` 一樣，預設把 `.grad` 設回 None，而不是填 0。第二次只反傳分類 loss，沒走進框分支，框分支的 `.grad` 就維持 None，所以 `is None` 的斷言（assert）成立。

`retain_graph=True` 是什麼？PyTorch 預設在 backward 做完後，就釋放計算圖上為反傳暫存的中間值，以節省記憶體。本節要對同一次 forward 反傳三次，所以前兩次加 `retain_graph=True` 把這些中間值留著；第三次是最後一次，不必再留。拿掉第一次的 `retain_graph`，第二次 backward 會報錯 `Trying to backward through the graph a second time`：兩個 loss 共用 backbone 那段計算圖，那段的中間值在第一次 backward 時就釋放了。

第三次反傳兩者的和。先替兩份 backbone 梯度取名字：記框 loss 為 \(L_{\text{box}}\)、分類 loss 為 \(L_{\text{cls}}\)，\(\theta\) 為共用 backbone 的權重。

- \(g_{\text{box}}=\partial L_{\text{box}}/\partial\theta\)，就是程式的 `box_backbone_grad`。
- \(g_{\text{cls}}=\partial L_{\text{cls}}/\partial\theta\)，就是程式的 `class_backbone_grad`。

總 loss 是 \(L=L_{\text{box}}+L_{\text{cls}}\)。和的導數等於導數的和，所以 \(\theta\) 的梯度是 \(g_{\text{box}}+g_{\text{cls}}\)。其中每一項都是從自己的 loss 出發，沿自己的分支用連鎖律（chain rule）一路乘回 \(\theta\) 得到的；兩條路在共用特徵 f 會合，梯度在那裡相加。完整程式裡這樣核對：

``` { .python data-excerpt="lesson_cases/12-decoupled-head.py" }
model.zero_grad()
(box_loss + cls_loss).backward()  # 最後一次反傳，不必 retain_graph
assert torch.allclose(model.backbone.weight.grad, box_backbone_grad + class_backbone_grad, atol=1e-7)
```

`torch.allclose` 逐元素比對兩邊，容許極小的浮點誤差（同樣的數用不同順序相加，最後幾位小數可能不同）。這個檢查也能抓到程式錯誤：例如第三次 backward 前忘了清梯度，總梯度就會多出第二次留下的舊梯度，斷言便失敗。接著完整程式做一次 `optimizer.step()`，確認 backbone 權重真的改變。沒有 step，只證明計算圖能反傳出梯度，還沒驗證參數真的被更新。

## 用 cosine 看兩份梯度的方向

兩份 backbone 梯度 \(g_{\text{box}}\)、\(g_{\text{cls}}\) 的方向一致嗎？用餘弦相似度（cosine similarity，以下簡稱 cosine）來看。backbone 權重的 shape 是 `[8,3,3,3]`，所以每份梯度有 8×3×3×3=216 個數（只取權重，不含 bias）。把它們攤平成兩個 216 維向量，算

\[
\cos\varphi=\frac{g_{\text{box}}\cdot g_{\text{cls}}}{\lVert g_{\text{box}}\rVert\,\lVert g_{\text{cls}}\rVert}
\]

\(\varphi\) 是兩個向量的夾角；分子是內積，分母是兩個向量長度的乘積。這就是高中的向量夾角公式，只是分量從 2、3 個變成 216 個。完整程式寫成：

``` { .python data-excerpt="lesson_cases/12-decoupled-head.py" }
# flatten() 把 [8,3,3,3] 攤平成 216 個數；dim=0 表示沿攤平後的這一軸計算
cosine = F.cosine_similarity(box_backbone_grad.flatten(), class_backbone_grad.flatten(), dim=0)
```

cosine 接近 1 表示兩份梯度在這批資料上方向較一致，接近 −1 表示較相反，接近 0 表示大致正交（夾角約 90°，兩個方向互不相干）。用二維向量手算：\((3,4)\) 的長度是 5。它和 \((4,-3)\) 的內積是 \(12-12=0\)，cosine 為 0（垂直）；和 \((-3,-4)\) 的內積是 \(-25\)，cosine 為 \(-25/(5\times5)=-1\)（正好相反）；和 \((6,8)\) 的內積是 50，cosine 為 \(50/(5\times10)=1\)（同方向）。

本次印出 −0.0073。它雖然帶負號，但非常接近 0（夾角約 90.4°），應讀成「大致正交」：兩個 loss 想要的 backbone 修改方向幾乎互不相干，不是明顯相反。若 cosine 明顯為負，表示兩個任務把共用的 backbone 往相反方向推，相加時抵銷一部分，這就是任務衝突。decoupled head 只分開 backbone 之後的卷積，本來就不保證消除這種衝突。所以就算看到負 cosine，也不能直接宣稱 decoupled head 無效，因為分支仍讓兩個任務在後段各有自己的特徵轉換。

這個值是固定 seed、人工 target 的一次診斷，不能代表整個任務的平均情況。

??? note "為什麼內積的正負代表「一致」或「衝突」？"

    用一階近似來看。步伐很小時，loss 的變化約等於「梯度和移動量的內積」，就像用切線估計曲線。照分類梯度走一小步，也就是把 \(\theta\) 改成 \(\theta-\eta\,g_{\text{cls}}\)（\(\eta\) 是學習率），框 loss 大約變成

    \[
    L_{\text{box}}(\theta-\eta\,g_{\text{cls}})\approx L_{\text{box}}(\theta)-\eta\,g_{\text{box}}\cdot g_{\text{cls}}
    \]

    內積為正時，框 loss 也跟著下降，兩個任務互相幫忙；內積為負時，這一步會讓框 loss 變大，這就是衝突；內積接近 0 時，這一步對框 loss 幾乎沒有影響。cosine 是內積除以兩個長度，正負號和內積相同，又不受梯度大小影響，所以適合拿來比較方向。

## 可核對的成本

本例含 bias 的參數共 1,446。一層卷積的參數數是「輸出 channel×輸入 channel×卷積核的高×寬」，再加上每個輸出 channel 一個 bias：

- backbone（3×3，3→8）：`8×3×3×3+8=224`
- 兩條分支的 3×3（8→8）：各 `8×8×3×3+8=584`
- 框分支的輸出層（1×1，8→4）：`4×8+4=36`
- 類別分支的輸出層（1×1，8→2）：`2×8+2=18`

合計 `224+584×2+36+18=1,446`，其中兩條分支本身共 1,222。最簡單的 coupled head 是一層 8→6 的 1×1 卷積，6 個輸出 channel 是 4 個框值＋2 個類別，只要 `6×8+6=54` 個參數。這層 1×1 其實等於並排的 8→4、8→2 兩層 1×1（`54=36+18`，每個輸出 channel 本來就有自己的一列權重），所以只把輸出層拆成兩個不算分工；分工指的是輸出層之前各有自己的卷積。這個基線不只沒有分工，容量也不同，所以兩者的效果差異不能都歸因於分工。公平比較應選相近的 channel 數、層數或成本，並說明對齊哪一項。

??? note "保留一層 3×3 的 coupled head 有多少參數？"

    讓 coupled head 也先經過一層 3×3：f → 3×3 卷積（8→8）→ ReLU → 1×1 卷積（8→6），前 4 個輸出 channel 給框、後 2 個給類別。參數是 `584+54=638`。它和每條分支一樣是一層 3×3 加一層 1×1、中間 8 個 channel，對齊的是層數與 channel 數；參數量仍只有兩條分支（1,222）的一半左右。

執行 `PYTHONPATH=. python lesson_cases/12-decoupled-head.py`，本例只需 CPU。應看到：

1. 框分支、類別分支的 shape：`(2, 4, 4, 4)` 與 `(2, 2, 4, 4)`（`box / class shapes` 那行）。
2. 分類 loss 單獨反傳後，框分支的 `.grad` 仍為 None（`classification-only backward leaves box branch grad=None`）。框 loss 單獨反傳的那個方向由完整程式的斷言檢查，不另外印出。
3. backbone 的總梯度等於兩份梯度之和：`verified`。
4. 參數數量：`1446`。

第 2、3 項印的英文是固定的文字，不是程式算出來的值。程式先跑完所有斷言才開始印，所以印得出這兩行，就表示對應的 `is None` 與 `torch.allclose` 斷言已經通過。另外會印出一次梯度 cosine（`shared-backbone gradient cosine` 那行）。它可作檢查訊號，但沒有通用的理想數字。

## 收益、代價與常見錯誤

分支讓定位與分類在最後幾層各用自己的卷積權重，也能分開決定每條分支的 channel 數與 loss。代價有三：

- 多出的卷積計算。
- 記憶體：訓練時各層的輸出要暫存給 backward 用，這些中間張量叫 activation（不是 ReLU 這種激勵函數本身）。分支越多，要暫存的越多。
- 多了要自己決定的設定：分支要幾層、幾個 channel。

當 backbone 已經很小，增加兩條 channel 數多的分支，可能比 coupled head 更耗時；當資料很少，更多參數也可能過擬合。速度必須實測整條推論路徑，不能只看分支數。

常見錯誤：

- **兩次 backward 之間忘了清梯度**：backward 預設把新梯度加在舊梯度上，第二次看到的就是兩次的累加。
- **把 `.grad` 是 None 和全 0 tensor 混為一談**：若把第二次反傳前的 `model.zero_grad()`（程式裡的第二個）改成 `model.zero_grad(set_to_none=False)`，框分支在第一次反傳得到的梯度會被填成全 0 tensor，而不是設回 None；第二次反傳沒走進框分支，它就一直是全 0，`model.box_branch[0].weight.grad is None` 的斷言便失敗。改第一個沒有作用，因為那時所有參數的 `.grad` 本來就是 None（兩者的差別見第 2 章〈[訓練診斷](02-diagnostics.md)〉）。
- **在分支前誤用 `detach()`**：例如把 `forward` 裡的 `self.class_branch(f)` 寫成 `self.class_branch(f.detach())`，分類 loss 的梯度就到不了 backbone，backbone 只剩框 loss 在訓練。一般的訓練迴圈很難發現這個錯：分類分支本身仍會更新，loss 也可能照樣下降。本節程式則會在第二次反傳後停下：分類 loss 沒有流進 backbone，`model.backbone.weight.grad` 仍是 None；None 沒有 `.clone()` 可呼叫，所以接著取 `class_backbone_grad` 的那行報 `AttributeError`。本節兩個任務都需要訓練共用特徵，所以不在分支前做 detach。

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

本節的完整程式已於 2026-10-02 用 PyTorch 2.9.1+cpu 在 CPU 上執行過，程式裡的 assert 檢查全部通過。下面是那次印出的原始輸出；每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-decoupled-head.json)

??? example "展開本次實際輸出"

    ```text
    box / class shapes: (2, 4, 4, 4) (2, 2, 4, 4)
    classification-only backward leaves box branch grad=None
    shared-backbone gradient cosine: -0.0073
    total backbone gradient = box gradient + class gradient: verified
    parameters: 1446
    ```

<!-- curriculum-evidence:end -->
