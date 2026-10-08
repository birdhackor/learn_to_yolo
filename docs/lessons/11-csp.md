# C.11.1 CSP：分一部分通道走較短的路

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/11-csp.ipynb){ .md-button }

上一章讓同一張圖在粗、細兩種特徵上各接一個 head。接下來先看產生特徵的成本：一個區段裡連做兩層 3×3 卷積，是否每個通道都必須走完整條路？我們希望輸出仍有原本的通道數，卻減少寬卷積反覆處理全部通道的工作。

材料先縮成一個 block：輸入、輸出都為 `[B,8,8,8]`，也就是 B 筆、8 通道、8×8 位置。Full 對照讓 8 通道全部經兩層 3×3，再做 1×1 混合。**CSP（Cross Stage Partial，跨階段的部分連接）**則讓一部分通道直接到區段末端，另一部分才經過那串卷積，最後合併。stage 是網路裡保持同一特徵解析度的一段；本例把整個 stage 縮成兩層卷積。

## 先把兩條路的 shape 對起來

輸入 `x` 的軸是 NCHW：第 0 軸 batch、第 1 軸 channel、第 2、3 軸高、寬。`x.chunk(2,dim=1)` 沿通道切半，前 4 通道叫 **bypass（旁路）**，後 4 通道交給 **branch（卷積支路）**，各為 `[B,4,8,8]`。通道編號只是排列，沒有預設前半是邊緣、後半是顏色。

旁路的值不做 3×3，直接等在末端；支路經兩個 4→4 的 3×3 卷積，各接 ReLU。stride 1、padding 1 保持 8×8 不變。沿 channel 串接後，4+4=8，再用 8→8 的 1×1 **fuse（融合層）**混合兩路，得到原本要求的 `[B,8,8,8]`。

![Full 與 CSP 的路徑：前 4 通道旁路、後 4 通道經兩個 3×3，串接後 1×1 融合](../assets/diagrams/11-csp.svg)

看 CSP 的分叉與接合：減少的是中間兩層 3×3 的輸入、輸出寬度，輸出總通道數沒有減半。fuse 每個位置讀 8 個通道，做加權和再加 bias；它讓下一個 block 讀到的通道可以混合旁路與處理過的訊息。若沒有 fuse，連續使用同型 block 時，前 4 個原值可能一路旁路，都沒有進入 3×3。

```python
# 卷積支路：3×3 卷積 → ReLU → 3×3 卷積 → ReLU，[B,4,8,8] → [B,4,8,8]
branch = nn.Sequential(nn.Conv2d(4,4,3,padding=1),nn.ReLU(),
                       nn.Conv2d(4,4,3,padding=1),nn.ReLU())
fuse = nn.Conv2d(8,8,1)  # 融合用的 1×1 卷積：[B,8,8,8] → [B,8,8,8]

bypass, transformed = x.chunk(2,dim=1)        # 各 [B,4,8,8]：前 4 個通道（旁路）、後 4 個通道（還沒轉換，要送進卷積支路）
processed = branch(transformed)               # 卷積支路的輸出：[B,4,8,8]
joined = torch.cat([bypass,processed],dim=1)  # 沿 channel 串接：[B,8,8,8]
y = fuse(joined)                              # [B,8,8,8]
```

上面依完整 `CSP` 類別攤開 forward，只省略 `self.` 並拆出中間變數。ReLU 沒有參數，Conv2d 保留預設 bias。本節的融合只混同一位置的通道；下一頁將合併不同尺度的特徵。

## 串接不是相加

把兩路各兩通道的數值寫成 a=`[1,2]`、b=`[10,20]`，shape 各 `[1,2,1,1]`。concat 沿通道接成 `[1,2,10,20]`，shape `[1,4,1,1]`；add 逐值相加成 `[11,22]`，shape 仍 `[1,2,1,1]`。程式先用斷言核對這個小例子。

CSP 保留兩路再交給 fuse，與第 3 章 [ResNet shortcut](03-identity.md) 的逐值相加不同，沒有 `y=x+F(x)` 保證。若支路結果為 0，joined 只剩 `[前4通道原值,0,0,0,0]`，後 4 個輸入原值已不在；fuse 還會改變這 8 個通道。所以「有旁路」不等於「整個 block 是 identity」，不能從第 3 章直接搬來 `F(x)=0時y=x` 的結論。

## 參數成本可以逐項手算

為什麼只處理半數通道能省成本？帶 bias 卷積參數為 `out×in×k²+out`，輸入、輸出通道一起減半，權重項就變成四分之一。Full 和 CSP 末端都用同一個 8→8 的 fuse，對照如下：

| 模組 | 兩個 3×3 | 最後 1×1 | 合計 |
| --- | --- | --- | --- |
| Full，3×3 為 8→8 | `2×(8×8×9+8)=1168` | `8×8+8=72` | 1240 |
| CSP，3×3 支路為 4→4 | `2×(4×4×9+4)=296` | 72 | 368 |

3×3 部分從 1168 降到 296，合計約剩三成。和把整個 block 改窄成 4 通道相比，CSP 仍把旁路 4 通道送給 fuse，輸出仍有 8 通道；不過只有一半經過空間卷積，能表示的轉換也與 Full 不同。相同輸出 shape 不保證相同容量或偵測品質。

這是單一 block 的參數計算，沒有量硬體速度。切分、串接要搬記憶體，整個偵測器還有其他層，不能把 368/1240 直接當成整網延遲比例。

## 梯度怎麼分回兩路

要確認這兩路真參與計算，把輸出各值平方後平均，當成以全 0 為目標的 MSE。這個 loss 沒有偵測意義，只是能從輸出求反傳。輸入設 `requires_grad=True`，所以 backward 後可讀 `x.grad`，查看每個輸入元素對 loss 的變化率。

反傳先經 fuse，再由 concat 按通道切回兩路：前 4 通道直接回到 bypass；後 4 通道經兩層卷積與 ReLU，回到 branch 輸入。程式對前半、後半梯度各取絕對值總和，避免正負抵銷；兩個都有限且非零，表示兩半輸入都連到 loss。

但這還不能證明 branch 有執行。若錯把後 4 通道直接接 fuse，兩半輸入仍都收到梯度，branch 參數卻沒有參與計算、梯度為 None。因此程式也檢查**每個參數**的梯度不是 None、有限且不全 0，再由 SGD 更新一步，核對 fuse 權重改變。

??? note "頁尾紀錄裡，CSP 的兩個數為什麼差很多？"

    頁尾紀錄中，Full 的兩個數很接近；CSP 的第一個數（前 4 個通道）卻比第二個數（後 4 個通道）大很多。差別來自兩路到 loss 要穿過的層數不同：

    - 前 4 個通道（旁路）只隔一層 1×1 的 fuse 就到 loss。
    - 後 4 個通道（卷積支路）還要穿過兩層 3×3 卷積與 ReLU。這裡 4→4 的 3×3 卷積，初始權重落在 ±1/6 之間；反傳時會用這些權重加總各路梯度，ReLU 在輸入為負的位置則傳回 0。本次初始化中，走過這條支路的梯度比旁路小很多；單一權重小，不表示梯度每經一層都一定縮小。

    所以剛初始化時，兩個數差很多。這是初始化當下的現象，不是品質指標，也不是 CSPNet 論文說的重複梯度資訊。Full 的 8 個通道都走同一種路（兩層 3×3 卷積與 ReLU，再到 fuse），所以兩數接近。另外，兩個模型各自重新抽一份隨機輸入，Full 與 CSP 的數字不能互相比較。

## 執行與核對

用頁首 Colab 或 `PYTHONPATH=. python lesson_cases/11-csp.py`。Full 與 CSP 各用一份隨機輸入 `[2,8,8,8]` 做 forward、平方平均 loss、backward 與一次 SGD。核對 concat/add 數值、參數數 `[1240,368]`、輸出 `[2,8,8,8]`、兩半輸入與全部參數梯度，以及 fuse 更新。輸出的 `finite backward and step` 是通過這些斷言後的固定摘要。

這個 block 沒有放進 MiniYOLO；這些觀察能確認成本和資料流，不能證明 CSPNet 論文所說的「重複梯度資訊」減少，也沒有 AP。若要測放入偵測器是否划算，應在同一位置替換 block，固定資料、訓練預算與評估方式；若問題是相同參數量下哪種結構好，還需另把 Full 調窄。兩模型此處各抽不同的隨機輸入，梯度總和也不能互相比品質。

??? note "歷史與版本"

    **CSPNet 的梯度說法**：CSPNet 論文認為，DenseNet（每一層都把前面各層的輸出串接起來當輸入的網路）這類網路在反向傳播時，不同層的權重更新會重複用到大量相同的梯度。論文稱這種現象為「重複的梯度資訊」（duplicate gradient information）。論文讓一部分通道直接跨到 stage 末端，說這樣能省下計算，也讓梯度路徑變成兩倍。要減少重複，論文認為關鍵在末端的融合順序：走完 dense block（DenseNet 裡層層串接的那一串卷積層）的那部分先經過一層 transition（過渡層，例如 1×1 卷積），再和跨過來的那部分串接。論文說這種排法截斷了梯度流（truncate the gradient flow），用來避免不同層學到重複的梯度資訊；這不是指梯度傳不回前面的層。論文也比較了先串接、再做 transition 的排法：也能省下計算，但論文說這樣仍會大量重用梯度資訊。本節的 block 是先串接、再用 1×1 融合，排法比較接近後者。本節只檢查兩條路都收得到梯度，不量測重複的程度。

    **YOLOv4**：[YOLOv4](https://arxiv.org/abs/2004.10934) 的 backbone（主幹）採用 CSPDarknet53，也就是在 YOLOv3 的主幹 Darknet-53 上加入 CSP 結構。YOLOv4 另外還有其他訓練技巧與 neck（夾在 backbone 與 head 之間整理特徵的部分）設計。

    **YOLOv5**：YOLOv5 由 Ultralytics 公司另行開發，以開源程式碼發布。可對照固定版本的[官方 v6.0 common.py](https://github.com/ultralytics/yolov5/blob/956be8e642b5c10af4a1533e09084ca32ff4f21f/models/common.py)（連結固定在 v6.0 tag 指向的 commit）：v6.0 是 YOLOv5 程式庫的版本號，不是 YOLOv6；common.py 是定義各種模組的程式檔。v6.0 的模型配置用的 CSP 模組是 C3，程式註解寫著「CSP Bottleneck with 3 convolutions」。C3 不用 chunk 切半，而是用兩個 1×1 卷積從同一個輸入各算出一支，每支的通道數是輸出通道數的一半（`e=0.5`），相當於本節的切半。其中一支再經過一個或多個 Bottleneck，然後兩支串接，由第三個 1×1 卷積融合；所以 C3 和本節一樣是先串接、再融合。Bottleneck 是先做 1×1、再做 3×3 卷積的小模組。建立 C3 時有個選項 `shortcut`，預設為 True，這時 Bottleneck 還會像第 3 章的 shortcut 那樣，把輸入逐值加到這兩層卷積算出的結果上。以 v6.0 的 [yolov5s 配置](https://github.com/ultralytics/yolov5/blob/956be8e642b5c10af4a1533e09084ca32ff4f21f/models/yolov5s.yaml)為例，backbone 的 C3 都用這個預設；backbone 之後的 C3 則設成 False，不做這個相加（YOLOv5 的設定檔把 neck 也寫在 `head:` 段落底下）。同一個檔案裡較早的 BottleneckCSP 與 YOLOv4 的 CSPDarknet53，則像論文那樣，在支路末端先接一層 1×1 卷積再串接。

    YOLOv4、YOLOv5 相對 YOLOv3 的改進不只 CSP 一項，不能全部歸功於 CSP。本節簡化為 8 通道的一次切分、兩個小卷積，再串接並用 1×1 融合，不重現完整的 CSPDarknet 或 C3。原版的每個卷積後面都接 BatchNorm（一種把每個 channel 的數值重新標準化的層）與激勵函數：YOLOv5 的 C3 由 `Conv` 模組組成（不帶 bias 的卷積 → BatchNorm → SiLU（Sigmoid Linear Unit，sigmoid 線性單元：把輸入 x 乘上 sigmoid(x)）），YOLOv4 的 CSPDarknet53 每層卷積後接 BatchNorm 與 Mish。BatchNorm 會減掉每個 channel 的平均值，卷積的 bias 加了也會被減掉，所以原版不帶 bias。本節不加 BatchNorm、改用 ReLU，卷積保留 `nn.Conv2d` 預設的 bias；1240 與 368 只算卷積的權重與 bias。

實際設計還需選切分比例與合併位置。通道原本很少、資料又難時，只讓少量通道進 3×3 可能限制表達；多出切分和 concat 也有記憶體成本。

常見接線錯誤是沿 dim 3 切寬而非 dim 1 切通道，得到 `[B,8,8,4]`，與 4→4 卷積不合；或 branch 的兩層 3×3 漏 padding，8×8 縮成 4×4，不能與旁路 8×8 串接。concat 規則是除接合軸以外，其餘軸長度都相同。

## 自主練習

手算即可，不必改程式；先自己算，再展開參考答案。

1. 輸入改成 C=10 個通道，平均切成兩路、每路 5 個通道，fuse 改成 10→10。CSP 的參數是多少？空間 shape 會變嗎？
2. 如果把串接改成逐值相加，兩個 `[B,4,8,8]` 相加後是什麼 shape？fuse 還能用 8→8 的 1×1 卷積嗎？

??? note "參考答案"

    **第 1 題**：兩個 5→5 的 3×3 卷積各是 5×5×9+5=230，fuse 是 10×10+10=110，共 230×2+110=570。空間 shape 仍不變。對照：同樣 C=10 的 Full 是 2×(10×10×9+10)+110=1930；570/1930≈0.30，和 C=8 時的 368/1240≈0.30 差不多，因為各部分的參數都大約和通道數的平方成正比。

    **第 2 題**：逐值相加後是 `[B,4,8,8]`，只剩 4 個通道。8→8 的 fuse 要求輸入 8 個通道，不能直接用；要把 fuse 的輸入通道改成 4，例如 `nn.Conv2d(4,8,1)`，輸出才會回到 `[B,8,8,8]`。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-csp.json)

??? example "展開本次實際輸出"

    ```text
    concat (1, 4, 1, 1) [1.0, 2.0, 10.0, 20.0] add (1, 2, 1, 1) [11.0, 22.0]
    full input gradient abs sums first4/last4 0.033093 0.029087
    full output (2, 8, 8, 8) finite backward and step
    CSP input gradient abs sums first4/last4 0.255476 0.019409
    CSP output (2, 8, 8, 8) finite backward and step
    parameters full/CSP [1240, 368] capacity differs; no accuracy comparison
    ```

<!-- curriculum-evidence:end -->
