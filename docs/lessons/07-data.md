# 7.1 Grid MiniYOLO：先讓資料可以被檢查

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/07-data.ipynb){ .md-button }

要讓第 5、6 章的格子偵測器真的從圖片學習，第一件事是確認：模型看到的物件，和標註交給它的答案是同一個。若藍色矩形的框寫到紅色矩形上，loss 仍可能下降，因為它只比較輸出與標註，不知道標註是否框對了物件。

先用一張能逐個畫素核對的圖，把圖片、框與類別接起來。讀完本節，你能將它們寫成模型需要的格式，保留沒有物件的圖片，並分辨「格式合法」與「標註對齊」這兩種檢查。

## 一個框如何對應到圖片

![兩色矩形與半開區間框](../assets/diagrams/07-data.svg)

這張 64×64 圖放大 4 倍顯示。紅框是 `[8,12,24,28]`，藍框是 `[40,36,56,52]`；背景的 RGB 是 (0,0,0)，紅色是 (1,0,0)，藍色是 (0,0,1)。框沿用〈[座標轉換](04-coordinates.md)〉的 pixel xyxy：約定先 x 後 y，四個數依序是左上角、右下角的座標。

紅框的 x 範圍是 [8,24)，含 8、不含 24，因此紅畫素從 x=8 到 23，共 16 個；y 從 12 到 27，也是 16 個。這種半開區間的面積是 `(24−8)×(28−12)=256`，不必加 1，也正好與 Python 的切片 `a:b` 對齊。

完整程式用下面幾行建立圖與標註；中文註解是本頁加的：

``` { .python data-excerpt="lesson_cases/07-data.py" }
import torch
...  # 省略：匯入 miniyolo.data、def main(): 與兩行不影響本節輸出的設定
image = torch.zeros(3, 64, 64)
image[0, 12:28, 8:24] = 1  # R 通道；tensor 索引是 channel,y,x
image[2, 36:52, 40:56] = 1  # B 通道
# 每個框配一個類別：0=紅，1=藍
target = {"boxes": torch.tensor([[8., 12., 24., 28.], [40., 36., 56., 52.]]),
          "labels": torch.tensor([0, 1], dtype=torch.long)}
```

這裡有兩套順序要分清。框先 x 後 y，但影像的索引先 y 後 x，所以紅框對應 `image[0,12:28,8:24]`。類別 id 與顏色通道也是兩套編號：藍色的類別是 1，卻要畫在通道 2（B），通道 1 是 G。背景沒有第三個類別；格子偵測器用 objectness 表達沒有物件。

可以先核對邊界：`image[0,12,8]=1` 是第一個紅畫素，`image[0,27,23]=1` 是最後一個。右邊的 `image[0,27,24]` 與下方的 `image[0,28,23]` 都是 0。紅、藍通道總和各是 `16×16=256`，綠通道是 0。

## 格式正確，還不代表框的位置正確

資料和模型／loss 之間約好的格式叫**資料契約**。本章採用：

手機上可左右滑動表格，查看完整欄位。

| 欄位 | shape／型別 | 意義 |
| --- | --- | --- |
| image | `[3,64,64]` float32 | RGB、CHW，畫素值在 `[0,1]` |
| boxes | `[N,4]` float32 | 輸入圖 pixel 的 xyxy 半開區間框 |
| labels | `[N]` int64（`torch.long`） | 和框同順序的類別 id：紅 0、藍 1 |
| batch 影像 images | `[B,3,64,64]` | B 張圖，可直接進 CNN |
| batch 標註 targets | 長度 B 的 list | 每張圖一個 `{boxes,labels}` dict，各張的 N 可以不同 |

N 是一張圖的物件數，B 是一批的圖片數；CHW 是通道、高、寬，批次的 BCHW 就是第 1 章的 NCHW。表中的框數 N 與 NCHW 的 N 含義不同。

shape、型別、座標範圍都需要檢查，但它們抓不到合法的錯位框。例如把紅框寫成 `[12,8,28,24]`，四個數仍在 0～64 內，寬高也為正，卻往右、往上各偏 4 pixel。要驗證對齊，必須把標註畫回圖上。這個人工場景還能更精確：由標註重畫一份預期影像，再逐值比較。

``` { .python data-excerpt="lesson_cases/07-data.py" }
expected_pixels = torch.zeros_like(image)
for box, label in zip(target["boxes"], target["labels"]):
    x1, y1, x2, y2 = box.to(torch.long).tolist()
    channel = {0: 0, 1: 2}[label.item()]  # 類別 0→R；類別 1→B
    expected_pixels[channel, y1:y2, x1:x2] = 1
assert torch.equal(image, expected_pixels), "Annotation and colored pixels disagree"
```

`assert` 是斷言：條件不成立就報錯停下；`torch.equal` 要求每個值都相等。若只把紅框 x1 從 8 改成 9，預期影像會少掉 x=8 那一欄的 16 個紅畫素，總和變成 240，原圖仍是 256。這個斷言就會報 `Annotation and colored pixels disagree`。相反地，寫死切片 `8:24` 的通道總和檢查只讀原圖，仍得到 256，抓不到標註被改錯。

## 兩張圖，為什麼標註保留成 list

第二張圖刻意使用全黑的空圖。它沒有框，應寫成 `[0,4]` 的 boxes 與 `[0]` 的 long labels，表示「零個框，每個框有四欄」。`[[0,0,0,0]]` 是一個零面積的假框，不是零個框；`build_targets` 會拒絕它。若不檢查，假框還會被當成中心在 (0,0) 的物件，教出錯誤答案。

``` { .python data-excerpt="lesson_cases/07-data.py" }
from miniyolo.data import ShapeDataset, collate
...  # 省略：上面的固定場景與逐值比對
empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
images, targets = collate([(image, target), (torch.zeros_like(image), empty)])
```

`collate` 是本書的批次打包函式：影像用 `torch.stack` 疊成 `[2,3,64,64]`，標註原樣放進 `[target,empty]`。stack 要求 shape 相同，影像都為 `[3,64,64]`，可以疊；兩張圖的框分別是 `[2,4]`、`[0,4]`，直接 stack 會報 `stack expects each tensor to be equal size`。保留逐圖 list，就能讓每張圖有不同框數，又維持第 i 張影像對應第 i 份標註。PyTorch 的 DataLoader 也能用 `collate_fn` 接這個函式。

空圖是有用的訓練材料：它要求模型在整張背景上都不要報物件。本章的 4×4 格會全部學 objectness=0；空圖沒有框與類別答案，因此不算這兩項 loss。兩物件圖也有 14 個背景格，但它不能代替整張都是背景的例子，也不能測到 N=0 的資料路徑。程式變數 `target`、`targets` 此時仍指原始標註（GT，ground truth／真值）；下一節的 `build_targets` 才把它們轉成每格的訓練目標。

## 從能手查的色塊，換到訓練資料

固定場景使用 0、1 畫素，便於精確核對。訓練需要更多位置和大小，所以改由 `ShapeDataset` 生成圖片與標註。它是 `miniyolo/data.py` 的 PyTorch Dataset：用 `dataset[i]` 取得第 i 筆；固定 seed（亂數種子）時，同一編號會得到同一張圖。

預設每張有 0～2 個矩形。本節取 seed=7 的 8 張，其中有 3 張空圖。背景是 `[0,0.04)` 的微小雜訊，紅色 RGB=(0.95,0.10,0.10)，藍色=(0.10,0.10,0.95)，不再只有 0、1。

每個矩形寬、高為 8～15 pixel，完整放在一個 16×16 格內，而且同圖的物件使用不同格。這避免了第 5 章的同格碰撞，先讓每格一框的流程能運作；中心也嚴格在格內，不會落到格線上。人工資料因此容易查錯，代價是只有矩形、近乎純色、近乎黑底。資料檢查通過，還不能推論模型會辨識真實照片。

這個 4×4、每格一框的模型沿用 [YOLOv1](https://arxiv.org/abs/1506.02640)「中心所在格負責」的想法。本章使用自己的網路、objectness 與 loss，是教學簡化。原版每格預測 2 框，confidence 的定義為 Pr(Object)×IoU；本章的 objectness 只回答這格是否分到物件。

## 執行並讀出檢查結果

在 Colab 執行頁首 notebook，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-data.py`。它不下載檔案、不建立模型，也不需要 GPU：

```text
batch (2, 3, 64, 64) counts [2, 0]
red/blue channel sums 256.0 256.0
dataset contract checked: 8 images
```

第一行核對批次與逐圖框數，第二行核對色塊總和。印出前，程式已完成標註重畫與原圖的逐值比對，以及 batch／空框 shape 檢查。第三行表示 8 張生成圖也通過格式與範圍檢查：影像 float32、shape `(3,64,64)`，框列數等於類別個數，座標在 0～64 且寬高為正。這三行是資料檢查，尚未更新任何模型參數。

接真實圖片時也要先畫標註疊圖，再查數值。Pillow 是常用的 Python 讀圖套件，延續 PIL（Python Imaging Library，Python 影像函式庫），匯入時仍寫 `PIL`。讀圖轉成陣列通常是 HWC，要 `permute(2,0,1)` 換軸；用 reshape 硬改 shape 會混淆位置與顏色。OpenCV 的 `cv2.imread` 讀出 BGR，要換回 RGB；0～255 的畫素要除以 255；resize 圖片時，框也要同步變換。這些錯都可能讓尺寸檢查通過，卻破壞影像與答案的關係。非正方形圖片和來源切分的實際操作放在第 8 章，這裡先保住同一個物件的畫素、框與類別對應。

## 自主練習

自主練習：在完整程式（Colab 或 `lesson_cases/07-data.py`）把 `target` 的紅框從 `[8., 12., 24., 28.]` 改成 `[8., 12., 28., 28.]`，畫紅色的那行 `image[0, 12:28, 8:24] = 1` 也要跟著改。

1. 畫紅色那行的切片要改成什麼，畫素和標註才一致？
2. 兩處都改好後執行，印出的紅色通道總和是多少？
3. `assert images[0, 0, 12:28, 8:24].sum().item() == 256` 這行不改，會失敗嗎？要檢查整塊紅色，該改成什麼？

附加題：畫素不改，只把紅框誤寫成上面那個 x、y 對調的 `[12., 8., 28., 24.]`。程式會停在哪一行？為什麼？

??? note "參考答案"

    1. 改成 `image[0, 12:28, 8:28] = 1`。新框寬 28−8=20、高 28−12=16。
    2. 第二行輸出變成 `red/blue channel sums 320.0 256.0`：紅色有 20×16=320 個值為 1 的畫素。
    3. 不會失敗。它用寫死的切片 `8:24`，只加總原來 16×16 那一塊，仍是 256，看不到新多出的 x=24～27 這 4 欄。要檢查整塊紅色，改成 `assert images[0, 0, 12:28, 8:28].sum().item() == 320`。

    如果只改標註、或只改畫素，畫素和標註就對不上，`assert torch.equal(image, expected_pixels)` 那行會失敗。

    附加題：停在 `assert torch.equal(image, expected_pixels), ...` 那行，錯誤訊息是 `Annotation and colored pixels disagree`。依標註重畫時，紅色被畫在 `expected_pixels[0, 8:24, 12:28]`（框 `[x1,y1,x2,y2]` 對應切片 `[y1:y2, x1:x2]`），和原圖的 `image[0, 12:28, 8:24]` 不重合，所以逐值比對失敗。



資料中的紅框仍用 pixel xyxy 描述；接下來要把它放進模型固定的 4×4 輸出位置。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-data.json)

??? example "展開本次實際輸出"

    ```text
    batch (2, 3, 64, 64) counts [2, 0]
    red/blue channel sums 256.0 256.0
    dataset contract checked: 8 images
    ```

<!-- curriculum-evidence:end -->
