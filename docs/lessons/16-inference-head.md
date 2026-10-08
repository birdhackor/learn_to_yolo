# 16.2 YOLO26 推論 head：把訓練用的分支從部署模型真正拿掉

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/16-inference-head.ipynb){ .md-button }

訓練用的一對多 head 已完成「多教共用特徵」的工作；若推論選一對一 head，還有必要帶著兩組權重、每張圖算兩份輸出嗎？讀雙 head 回傳的 `['one']` 只是在最後選一份結果，另一份仍已白算。要放到伺服器或手機使用，可以另外建立只含必要部件的部署模型。

接著有兩個不同問題：保留的 backbone 和 one 是否仍算出相同數值？這些數值是否被正確解碼成框、分數與類別？本節用小 CNN 完成這次轉換，先比較同一輸入的 raw，再追同一組候選到最終回傳值。

## 先選部署路徑，再決定保留誰

沿用 13.1 的名字，many 是一對多輔助 head，one 是一對一 head。本文選 NMS-free 推論路徑，只保留 backbone 和 one；top-k 仍會限制輸出數，少重複要由 one 的分數學好，不能靠拿掉 many 自動得到。

官方 YOLO26 的 `end2end: True` 建立這兩組分支，訓練兩組都算 loss。它的 one 在衝突後還以 `topk2=1` 再篩，讓每個 GT 最終至多一個正候選、也可能零個；這和 13.1 YOLOv10 只限制初選 top-1 不同。兩者都不是「一張圖只輸出一框」，也不保證推論沒有漏檢或重複。

使用官方套件時，選路徑還有一個實際差別：[原論文](https://arxiv.org/abs/2606.03748) §3.2.1 把 one-to-one 列為預設，但本書固定版本的 `model.predict()` 預設是 many 加 NMS；要傳 `nms=False` 才走 one、不做 NMS（[固定版本的官方訓練說明](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md)）。這個預測函式的預設與論文的設計不同，自己部署要明確選擇。官方 `Detect.fuse()` 也依這個選擇拿掉不用的一組，本文小模型模擬的是保留 one。

## 小模型如何產生兩份候選輸出？

`DualToy` 接 B 張 64×64 RGB 圖，輸入 `[B,3,64,64]`。backbone 是 3→8 的 3×3 卷積、stride 2、padding 1，接 ReLU，再 `AdaptiveAvgPool2d(4)`，把每個 channel 平均成 4×4。所以特徵是 `[B,8,4,4]`，16 格各是一個候選；對這次 64×64 輸入，每格按 64/4=16 畫素解碼。

many、one 都是 8→6 的 1×1，各回傳 `[B,6,4,4]`。六 channel 依序為 `[l,t,r,b,class0,class1]`：前四個 raw 經 softplus 成為格單位正距離，後兩個是類別 logits，各做 sigmoid，沒有另外的 objectness。四距離各由一個數直接表示，沒有 DFL；softplus 是本例選擇，官方 16.1 的直接回歸不加它，能表達負距離。

`DualToy` 回傳 `{'many':...,'one':...}`。one 讀 `features.detach()`，只讓 one loss 更新 one，不回到 backbone；many loss 才教共用特徵。detach 不改前向數值，部署只推論時不需要它。

![訓練保留雙 head，部署只帶 backbone 與 one](../assets/diagrams/16-head-paths.svg)

實線是前向，虛線是梯度；one loss 到 detach 即停止。部署圖只留下原 backbone、one 的副本，點線所比的是兩邊對同一輸入的 one raw，還不是最後挑出的三框。

為了讓「複製更新前權重」和「複製當下權重」能被分辨，完整程式先做一次人工更新。圖像是 `[2,3,64,64]` 的隨機值，loss 是兩份 raw 的平方平均相加，要求輸出往 0 靠，沒有物件標註：

``` { .python data-excerpt="lesson_cases/16-inference-head.py" }
# main() 內
model = DualToy()
image = torch.rand(2, 3, 64, 64)  # B=2 張 64×64 的隨機圖
optimizer = torch.optim.SGD(model.parameters(), lr=.02)
optimizer.zero_grad()
output = model(image)  # DualToy 回傳的 dict：{'many': ..., 'one': ...}
loss = output['many'].square().mean() + output['one'].square().mean()
loss.backward()
before = model.one.weight.detach().clone()  # 更新前 one 權重的副本
optimizer.step()
# one 的權重必須真的變了；否則更新前的 one 副本也能通過後面的逐值比對
assert not torch.equal(before, model.one.weight)
```

最後斷言確認 one 的 weight 真的變了。`.detach().clone()` 保存更新前獨立副本；若沒有 step、學習率為 0 或 loss 漏掉 one，就不能把未變的初值稱為已更新權重。這一步只讓轉換檢查有內容，不是偵測訓練；沒有 assignment，沒有學「一個物件一框」。

## 建一個沒有 many 的模型，保留當下的 one

`DeployToy` 只有 backbone 和 head，head 從已更新的 one 複製。沒有 `self.many`，每次 forward 也沒有呼叫它：

``` { .python data-excerpt="lesson_cases/16-inference-head.py" }
# DeployToy.__init__ 內：trained 是傳進來、已更新的 DualToy
self.backbone = copy.deepcopy(trained.backbone)
self.head = copy.deepcopy(trained.one)  # 只複製 one；many 不複製
```

`copy.deepcopy` 連層與權重複製成獨立的一份。若只設 `self.head=trained.one`，兩模型會共用參數，之後訓練原模型，部署模型也跟著變。現在能獨立保存當下的推論模型。

先在**解碼前**比較兩邊。不能直接拿 `deploy(image)` 對 `model(image)['one']`，因為前者已解碼、已挑 top-3；後者仍是 16 格的 raw：

``` { .python data-excerpt="lesson_cases/16-inference-head.py" }
# main() 內：model 就是同一個 DualToy
deploy = DeployToy(model).eval()  # model 傳進去，就是 __init__ 裡的 trained
with torch.no_grad():
    reference = model(image)['one']  # 原雙 head 模型的 one 輸出：[B,6,4,4]
    deploy_raw = deploy.head(deploy.backbone(image))  # 部署模型解碼前的 raw：[B,6,4,4]
    assert torch.equal(reference, deploy_raw)  # 逐值完全相同，沒有容差
```

兩邊是同一張隨機 image、完全相同的權重與 PyTorch 運算，所以可用 `torch.equal` 要求每個 raw 都相同。這比只看 shape 更能抓出複製錯 head 或舊權重的錯。程式當場印出的 max abs difference=0.0 就是此比對的差值，並非準確度指標。

`deploy` 用 eval，`model` 仍是 train，但本例沒有 BN、Dropout，兩者前向行為相同。有這些層時兩邊都要先 eval；no_grad 只停止記錄反傳，不代替 eval。`.eval()` 也不刪分支，還要建立這個部署模型才能真的少帶 many。

再查參數名稱，防止沒用的 head 仍躲在模型裡：

``` { .python data-excerpt="lesson_cases/16-inference-head.py" }
assert not any('many' in name for name, _ in deploy.named_parameters())
```

部署只列 `backbone.0.weight`、`backbone.0.bias`、`head.weight`、`head.bias`。backbone 有 `3×3×3×8+8=224` 參數，每 head 有 `8×6+6=54`；因此訓練模型 332，部署 278，少 54，約 16%。332／278 由程式印出而非斷言；比例只屬於這個小 CNN。

## 距離、分類和候選編號要一起對上

保留 raw 還不夠，使用者要的是圖上的框。先看一個人工候選：中心 (24,40)、stride 16，softplus 後 ltrb=`[1,0.5,2,1.5]` 格。乘 16 成 `[16,8,32,24]` 畫素，解碼為 `[8,32,56,64]`。

類別 logits 是 `[0,ln3]`，sigmoid 分數為 `[0.5,0.75]`，取 label 1、score 0.75。它們不必加成 1，因為各類獨立 sigmoid；也不能沿用第 7 章 `obj×softmax(class)`，這裡沒有 obj。人工例子核對框、分數與類別的值，讓通過不只依賴 shape。

4×4 的完整候選圖在 x、y 各有中心 8、24、40、56。raw 按 row-major 從 `[B,6,4,4]` 變 `[B,16,6]`：先 `flatten(2)` 合併 H、W，再 `transpose(1,2)` 把 channel 放最後。第 k 個候選在第 k//4 列、第 k%4 欄，所以中心

\[
((k\bmod4+0.5)×16,\ (\lfloor k/4\rfloor+0.5)×16).
\]

例如 (24,40) 對應第 2 列第 1 欄，即 k=9。中心點也用同一排列產生，才不會把第 9 個 raw 解到別格。

部署 forward 先解全部 16 框，再挑前三名；保留以下原程式摘錄：

``` { .python data-excerpt="lesson_cases/16-inference-head.py" }
def forward(self, image):  # image：[B,3,64,64]
    # head 輸出 [B,6,4,4]；flatten(2) → [B,6,16]；transpose(1, 2) → [B,16,6]
    raw = self.head(self.backbone(image)).flatten(2).transpose(1, 2)  # B,16,6
    # 前四項經 softplus 變成正的距離：[B,16,4]，單位是格（教學選擇，不是官方 YOLO26 的做法）
    distance = F.softplus(raw[..., :4])
    axis = (torch.arange(4, device=image.device, dtype=image.dtype) + .5) * 16  # [8,24,40,56]
    y, x = torch.meshgrid(axis, axis, indexing='ij')  # 各 [4,4]：y 由上到下變，x 由左到右變
    points = torch.stack((x, y), -1).reshape(1, 16, 2)  # [1,16,2]：第 k 個中心點對應 raw 的第 k 列
    boxes = decode_ltrb(points, distance, stride=16)  # [B,16,4]：64×64 輸入的畫素 xyxy
    # 後兩項是兩類 logits：各自做 sigmoid，每個候選只留最高分的類別（簡化）；scores、labels 各 [B,16]
    scores, labels = raw[..., 4:].sigmoid().max(-1)
    top_scores, ids = scores.topk(3, dim=1)  # 分數前三名與它們的候選編號，各 [B,3]
    # 依 ids 取出那三個候選的框與類別：[B,3,4]、[B,3]（gather 見下方摺疊區）
    return boxes.gather(1, ids[..., None].expand(-1, -1, 4)), top_scores, labels.gather(1, ids)
```

`decode_ltrb` 使用 x1=x−l×stride、y1=y−t×stride、x2=x+r×stride、y2=y+b×stride。`raw[...,4:].sigmoid().max(-1)` 沿類別軸取最大值，讓每個候選取得一個分數與類別；`scores.topk(3,dim=1)` 再沿候選軸挑前三名，得到分數和原候選編號 ids。gather 用**同一份 ids**挑框和 labels，不能各自排序。

| 值 | B=2 時 shape | 含義 |
| --- | --- | --- |
| raw | `[2,16,6]` | 四個未轉換值、兩個類別 logit，中間結果 |
| 全部解碼框 | `[2,16,4]` | 64×64 圖的畫素 xyxy，中間結果 |
| 回傳框 | `[2,3,4]` | ids 指到的三框 |
| 回傳 scores／labels | `[2,3]`／`[2,3]` | 同三候選的 sigmoid 分數與整數類別 |

??? note "gather 怎麼從 16 個框挑出 3 個"

    `scores.topk(3, dim=1)` 回傳每張圖分數前三名的分數 `top_scores`，以及它們的候選編號 `ids`，shape 都是 `[B,3]`。假設第 0 張圖的 `ids[0]` 是 `[9,2,5]`，我們要的就是 `boxes[0,9]`、`boxes[0,2]`、`boxes[0,5]` 這三個框。

    `gather(1, index)` 沿第 1 軸（候選軸），依 index 裡的編號取值；輸出的 shape 和 index 相同。對三維的 boxes 來說，結果的 `[b,j,c]` 等於 `boxes[b, index[b,j,c], c]`。我們要的輸出是 `[B,3,4]`（每框 4 個座標），所以 index 也要是 `[B,3,4]`：

    - `ids[..., None]` 在最後加一軸：`[B,3]` → `[B,3,1]`。
    - `.expand(-1, -1, 4)` 把長度 1 的最後一軸擴成 4（-1 表示該軸長度不變）：`[B,3,1]` → `[B,3,4]`。四個座標用同一個候選編號。expand 不真的複製資料，4 個位置讀的都是 `ids` 裡同一個數；這裡只讀不改，所以沒問題，要改其中的值，得先 `clone()`。

    所以結果的 `[b,j,c]` 就是 `boxes[b, ids[b,j], c]`：第 0 張圖依序取出候選 9、2、5 的四個座標。labels 每個候選只有一個數，`labels.gather(1, ids)` 直接用 `[B,3]` 的 ids，得到 `[B,3]`。


人工解碼例子直接呼叫距離解碼和 sigmoid，不經過整個 forward，因此沒有檢查 16 格排列；排列的依據是前面 raw 與 points 同用 row-major。人工分數 0.75 由 `torch.allclose` 容許極小 float32 誤差，和複製模型要求完全一致的判準不同。

本例只留每候選最高分類別，官方預設 NMS-free top-k 則先依每候選最高分挑 k 候選，再攤平這 k 個候選的各類分數，挑最高 k 個（候選, 類別）。同一位置 class0=0.8、class1=0.7，官方兩筆都可能留下，本例只有 class0 一筆。兩種都不比兩框 IoU，top-k 不是 NMS。

## 哪些部署工作完成，哪些還需要另外處理？

本例已把 many 的權重和 forward 移除，保住原 one raw，並固定 `(boxes,scores,labels)` 介面。原始雙模型只讀 `['one']` 則仍白算 many。若誤複製 many，因兩 head shape 相同不會立即報錯，raw 逐值比對才會抓到。

中心點和 stride 16 仍寫死，只適用 64×64。換成 128×128，pool 仍輸出 4×4，正確 stride 應為 32；shape 斷言可照樣通過，框座標卻只有正確值一半。top-3 也一定回傳三框，即使都是背景；實用時要加 score 門檻，再檢查重複、漏檢等失敗。這個隨機模型沒有品質證據。

官方 `Detect.fuse()` 依路徑移除不用的 head，和卷積／BN 融合不同。若走 many 加 NMS，移除的反而是 one；走本文 one 路徑才移除 many。整體模型 `fuse()` 還可能把 BN 的縮放、平移併入前一層卷積，省去一層前向；本例沒有 BN、沒有做這種運算改寫。改寫運算或換執行工具時，浮點末位可能不同，便要改用有容差的 parity 比對。

執行 `PYTHONPATH=. python lesson_cases/16-inference-head.py`，對照人工框 `[8,32,56,64]`、分數 `[0.5,0.75]`、label 1；再看雙模型 keys、部署三個 shape、參數 332／278 和 raw 最大差 0.0。labels shape 與參數數只是印出，其他主要值由斷言核對，後面才印結果。這些輸出確認轉換與解碼，不能拿三框數量猜圖上有幾個物件。

??? note "選讀：人工 softplus 輸入，以及匯出後的比對"

    若希望 softplus 後恰好得到人工距離 d，要倒算 raw：`r=ln(exp(d)−1)`，程式用 `log(expm1(distance))`。expm1(d) 是 eᵈ−1，d 很小時比先算 exp 再減 1 精確。d=1 時 r≈0.541，d=0.5 時 r≈−0.433；raw 可負，softplus 後距離仍正。這些值是人工倒算，不是模型學到。

    [第 20 章](20-deployment.md)匯出的是第 7 章 GridDetector，並比較 PyTorch、ONNX Runtime 等執行工具的 raw，不是本節的部署 head。換工具可能改卷積實作與加總順序，所以用有容差比對；本節 deepcopy 後同工具同運算，才要求完全一致。兩者的目的都是確認保留的數值，只是誤差條件不同。

## 自主練習

自主練習：在完整程式（Colab 裡那份，或本機的 `lesson_cases/16-inference-head.py`）動手改，先預測結果，再執行核對。完整程式的斷言寫死了原題的答案；改了程式，答案不同的斷言要跟著改，否則程式會在那一行報錯停下。

1. 把 `DeployToy.forward` 裡的 `top_scores, ids = scores.topk(3, dim=1)` 改成取前 5 個。三個回傳值的 shape 會變成什麼？`main()` 裡哪個斷言要跟著改、改成什麼？參數數會變嗎？
2. 把 `main()` 裡的 `image = torch.rand(2, 3, 64, 64)` 改成 128×128，其他先不改；backbone 仍用 `AdaptiveAvgPool2d(4)`。程式會報錯嗎？這時每格對應幾畫素、stride 應是多少？16 個中心點的 x、y 變成哪些值？`DeployToy.forward` 要改哪幾處，才能解出正確的框？

??? note "參考答案"

    **第 1 題**：改成 `scores.topk(5, dim=1)` 後，三個回傳值的 shape 變成 `[2,5,4]`、`[2,5]`、`[2,5]`。`main()` 的斷言 `assert boxes.shape == (2, 3, 4) and scores.shape == (2, 3)` 要改成 `assert boxes.shape == (2, 5, 4) and scores.shape == (2, 5)`，否則程式會在這一行報錯停下。labels 的 shape 沒有斷言，印出會變成 `(2, 5)`。參數數不變，仍是 332／278，因為 top-k 只是挑選，沒有權重。

    **第 2 題**：不會報錯，所有斷言也都照樣通過。128×128 經 stride 2 的卷積變成 64×64，`AdaptiveAvgPool2d(4)` 仍輸出 4×4，所以每格對應 128/4＝32 畫素，stride 應是 32，中心點的 x、y 都變成 16、48、80、112。`DeployToy.forward` 要改兩處：

    - `axis = (torch.arange(4, device=image.device, dtype=image.dtype) + .5) * 16` 最後的 `16` 改成 `32`。
    - `boxes = decode_ltrb(points, distance, stride=16)` 的 `stride=16` 改成 `stride=32`。

    卷積本來就接受任何尺寸，不用改。若不改這兩處，中心點和距離都還按 16 換算，程式照樣執行，但所有框的座標都只剩正確值的一半。

參考來源：[YOLO26 配置的 end2end 與 reg_max](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml)、[Detect forward、postprocess、get_topk_index 及 fuse](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)、[整個模型的 fuse（BaseModel.fuse：卷積／BN 融合，再呼叫 Detect 的 fuse）](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/tasks.py)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-inference-head.json)

??? example "展開本次實際輸出"

    ```text
    manual ltrb decode box: [[8.0, 32.0, 56.0, 64.0]]
    manual class probabilities: [[0.5, 0.75]] ; winner label=1, score=0.75
    training output keys: ['many', 'one']
    deploy boxes / scores / labels: (2, 3, 4) (2, 3) (2, 3)
    parameters training / deploy: 332 278
    retained one-head raw output vs reference, max abs difference: 0.0
    top-k has no pairwise IoU/NMS; random toy predictions are not accuracy evidence
    ```

<!-- curriculum-evidence:end -->
