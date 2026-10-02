# VGG 風格小 CNN：讓區域性圖樣重複使用

一張 32×32 RGB 圖有 3072 個數值。全連線層對不同輸入位置使用不同權重；卷積先看鄰近小區域，並在不同位置使用同一套權重。因此「紅色邊緣出現在左邊或右邊」可以由同一個濾鏡處理。前置是知道 forward、loss 和一次更新；不必先懂完整 VGG。

歷史機制：[VGG 原始論文](https://arxiv.org/abs/1409.1556) 研究堆疊小型 3×3 卷積的深層分類網路。本節保留「小卷積重複堆疊」的想法，縮成兩個 block、4／8 channels、global average pooling，省略原版的深度與大型全連線層。它是教學 CNN，不是 VGG16 的重現。

[在 Colab 執行](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/01-small-cnn.ipynb)，或執行 `PYTHONPATH=. python lesson_cases/01-small-cnn.py`。資料是程式畫的 8 張紅／藍矩形；輸入 32×32、batch 8、CPU 訓練 3 步。目的是確認形狀、梯度與參數更新，尚未訓練出可用分類器。

## 先約定圖片怎麼進模型

一般圖片陣列是 HWC：高、寬、顏色；PyTorch 卷積使用 NCHW：batch、channel、高、寬。本例 `images.shape=[8,3,32,32]`，dtype 為 float32，RGB 值在 0 到 1。uint8 的 0 到 255 應先轉 float 再除 255。`permute(2,0,1)` 是重新排列軸，`reshape` 不能代替它；後者只改容器形狀，可能把顏色與位置混在一起。

類別 0 是紅矩形、1 是藍矩形，標籤 shape `[8]`、dtype long。分類 head 輸出 `[8,2]` **logits**，也就是尚未轉成機率的分數；`cross_entropy` 接收 logits，內部處理 softmax。

## 一個 block 做了什麼

Block是一小組連續處理，例如「兩次Conv–ReLU再pooling」。Backbone是提取特徵的主幹；head是把特徵轉成任務答案的末端。本節head是一個linear，輸出兩個類別分數。

Conv2d 的一個 3×3 濾鏡在每個位置讀取區域性資料，乘權重並加總。RGB輸入時，一個輸出channel讀取R、G、B各一個3×3區塊，共27個值乘27個權重，再相加並加bias；不是每次只處理一種顏色。每個輸出 channel 都有自己的一套濾鏡；channel 是不同的特徵表示，不一定對應某個顏色或物件。ReLU 把負值變成 0，加入非線性。2×2 max pooling 以 stride 2 保留每區最大值，縮小空間尺寸。

padding=1 在四周補一圈，配合 kernel=3、stride=1，讓 32×32 維持 32×32。一般單軸輸出長度為 \(\lfloor(H+2p-k)/s\rfloor+1\)，其中 H 是輸入長度、p 是 padding、k 是 kernel、s 是 stride。例如 pooling 的 H=32、p=0、k=s=2，輸出就是16。

| 位置 | Tensor shape | 空間意義 |
| --- | --- | --- |
| 輸入 | `[8,3,32,32]` | 原影象素 |
| 兩次 Conv–ReLU | `[8,4,32,32]` | 每點 4 個特徵 |
| 第一次 pooling | `[8,4,16,16]` | 更疏的空間網格 |
| 兩次 Conv–ReLU | `[8,8,16,16]` | 每點 8 個特徵 |
| 第二次 pooling | `[8,8,8,8]` | 共 64 個位置 |
| 空間平均、linear | `[8,8]` → `[8,2]` | 每圖一個類別分陣列 |

感受野是某個特徵可能受原圖多大範圍影響。起始是一畫素；第一個3×3看3×3，第二個看5×5，第一次pool後看6×6。此後一格間距已是原圖2畫素，兩個3×3依序擴到10×10、14×14，第二次pool後到16×16。這是理論範圍，不表示其中每個畫素影響同樣強。

Global average pooling（GAP、空間平均）將每個channel的8×8共64個位置平均成一個值。例如同channel的兩個位置是2、6，平均變4，平均本身就沒有保留它們誰在左誰在右。`AdaptiveAvgPool2d(1)` 把每個channel縮成1×1，flatten去除這兩個長度1的空間軸，再交給head。

## 短程式與實際成本

```python
optimizer.zero_grad(set_to_none=True)
features = self.features(images)                    # [8,8,8,8]
summary = self.pool(features).flatten(1)             # [8,8]
logits = self.head(summary)                          # [8,2]
loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()
optimizer.step()
```

一個卷積的參數數量是 \(C_{out}(C_{in}k^2+1)\)，其中 \(C_{in},C_{out}\) 是輸入／輸出channel數，k是kernel長度，最後的 1 是 bias。第一層為 \(4(3\times9+1)=112\)，完整模型共 **1158** 個參數。案例計算每張圖 **479248** 次乘加，僅包含 conv 與 linear，不含 ReLU、pooling與資料搬移，因此不是完整耗時模型。第一層中間特徵 `[8,4,32,32]` 的 float32 數值本身需128 KiB（1 KiB=1024 bytes）；訓練還需儲存更多中間結果、梯度與參數。

一次乘加是「一個值乘權重、累加到答案」，不是把一次乘法和一次加法分別計成兩次。卷積的乘加數是輸出位置數×輸出channel數×每個輸出讀取的值數。第一層為 \(32\times32\times4\times(3\times9)=110592\)。其餘三層為147456、73728、147456，linear為 \(8\times2=16\)，相加即479248；batch8再乘8。這裡不把bias加法列入MAC。

更寬可以表達更多特徵，但相鄰卷積的輸入與輸出channel都增加時，成本常接近寬度的平方。更早下取樣省計算，同時可能抹去細小物件；這會在偵測任務變得關鍵。

## 核對與看錯誤

程式先印表中的逐層 shape，再核對參數和乘加數；三步都檢查首層權重梯度為有限值，首層權重確實改變。每一步先清上一步梯度，再算loss與更新。推論則使用eval與no_grad，對 `[8,2]` logits每列用 `argmax(1)` 選類別。最後輸出預測、標籤與 `error_indices`，並寫入 `artifacts/01-small-cnn.png`：紅色標題就是分類錯誤。讀圖時先確認矩形顏色與 GT（正確標籤）一致，再看預測；若全部猜同一類，不能宣稱模型學會了。

![固定seed三步推論圖板的前四張人工圖形，兩張紅矩形被錯分](../assets/diagrams/01-small-cnn.svg)

固定seed7的此次實跑全猜類別1，錯誤索引是0、2、4、6。上圖展示前四張；GT是真值，pred是預測類別。這個失敗正提醒我們：參數更新成功與分類學好是兩件事。

此圖板使用**同一批訓練資料**，只是完整推論路徑的展示。要判斷泛化，需要獨立資料與較完整訓練。不要把人工畫圖的簡單程度，當成真實圖片也容易的證據。

## 常見錯誤、自主練習與答案

若輸入 `[8,32,32,3]`，卷積會把32當成channel；先查軸，別先調learning rate。若 loss 報標籤越界，先確認類別id是0／1。若將softmax後的值再交給cross entropy，便改變了預期的輸入契約。

練習只把 width 從4改成8，維持資料、步數及seed。答案：各層channel變成8／16，四層conv及head參數依序224、584、1168、2320、34，合計 **4330**；參數約3.74倍，不是2倍。MAC依序221184、589824、294912、589824、32，合計 **1695776**。同步改案例的參數與乘加預期，再執行；觀察錯誤圖板可以，但3步、單seed、8張圖不足以判定寬模型比較準。
