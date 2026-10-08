# 7.1 Grid MiniYOLO：先讓資料可以被檢查

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/07-data.ipynb){ .md-button }

前置：[多物件責任分配](05-assignment.md)、[座標轉換](04-coordinates.md)。本節要解決的問題是「模型讀進去的畫素（pixel），是否仍與框描述同一個物件」。若藍色矩形的框移到紅色矩形上，訓練可以照常降低某個 loss，卻是在學錯的任務。這是因為 loss 只比較模型輸出和標註，不會檢查標註是否真的框在那個物件上；標註一致地錯，模型就一致地學錯。

本節還沒有訓練，只確認資料契約。**資料契約**是資料和模型／loss 之間事先約好的格式：資料照這個格式給，模型和 loss 照這個格式讀。本節的契約有五條：

- 影像、框、類別的 shape 和 dtype，照下面的表格。
- 影像是 RGB，畫素值在 0 到 1 之間。
- 框以畫素為單位，寫成 xyxy（左上角、右下角的座標），採半開區間。
- 紅色的類別 id 是 0，藍色是 1。
- 沒有物件的圖照樣保留：框的 shape 寫成 `[0,4]`，類別的 shape 寫成 `[0]`。

有些違反契約的錯不會讓程式報錯：程式照跑、loss 照降，模型學到的卻是錯的。所以要在訓練前先核對。讀完本節，你能把一張圖和它的標註寫成這個格式，並用程式核對兩者是否描述同一個物件。

歷史機制：YOLOv1 把圖片切成格子（grid），由物件中心所在的格子負責預測；原文是 [You Only Look Once (2016)](https://arxiv.org/abs/1506.02640)。本章簡化成 64×64 的兩色矩形、4×4 格、每格一個框、兩個互斥類別。網路、每格框數、confidence 和 loss 都不是原論文的設計；本章改用 objectness，只表示這格有沒有分配到物件。

??? note "原論文的每格框數與 confidence"

    原論文每格預測 2 個框。每個框各有一個 confidence，原文定義為 Pr(Object) × IOU：「有物件的機率」乘上「預測框與真值框的 IoU」（交集面積除以聯集面積）。本章每格只有一個框，也不使用這個定義。

![兩色矩形與半開區間框](../assets/diagrams/07-data.svg)

上圖把 64×64 的輸入放大 4 倍顯示。圖中的顏色只是為了方便看；實際的 (R,G,B) 值是背景 (0,0,0)、紅 (1,0,0)、藍 (0,0,1)。

## 同一個物件的第一站

固定場景是一張 64×64 的圖，裡面有紅框 `[8,12,24,28]` 與藍框 `[40,36,56,52]`。本章會帶著這個紅框依序走過資料 → 訓練目標（target）→ loss → 推論。本節是第一站：先確認畫素和標註一致。

框的順序是 `x1,y1,x2,y2`，單位是輸入圖片的畫素。框採半開區間：x 方向是 [8,24)，含 8、不含 24，所以紅色畫素的 x 是 8 到 23，共 24−8=16 個；y 方向是 [12,28)，紅色畫素的 y 是 12 到 27，也是 16 個。面積是 16×16=256，不必加 1。

紅色的類別 id 是 0，藍色是 1。背景不算一個類別，所以沒有類別 2；背景改由下一節的 objectness=0 表示。

完整程式這樣畫出這張圖，並寫下它的標註（中文註解是本頁加的）：

``` { .python data-excerpt="lesson_cases/07-data.py" }
import torch
...  # 省略：匯入 miniyolo.data、def main(): 與兩行不影響本節輸出的設定
image = torch.zeros(3, 64, 64)  # 背景全是 0
image[0, 12:28, 8:24] = 1  # 紅，索引順序是 channel,y,x
image[2, 36:52, 40:56] = 1  # 藍
# 一張圖一個標註 dict；N 是這張圖的框數（這裡 N=2）
# boxes 是 [N,4] 的 xyxy 框，labels 是 [N] 的類別 id
target = {"boxes": torch.tensor([[8., 12., 24., 28.], [40., 36., 56., 52.]]),
          "labels": torch.tensor([0, 1], dtype=torch.long)}
```

這段程式有三個容易寫錯的地方：

- Python 切片 `a:b` 取 a、a+1、…、b−1，不含 b，本身就是半開區間。所以框 `[x1,y1,x2,y2]` 可以直接寫成 `image[c, y1:y2, x1:x2]`，不必加減 1。
- 框是先 x 後 y，tensor 索引卻是先 y 後 x：紅框 `[8,12,24,28]` 對應 `image[0, 12:28, 8:24]`。
- 通道（channel）0、1、2 依序是 R、G、B。類別 id 和通道索引是兩套編號：藍色是類別 1，卻畫在 `image[2]`（B 通道），不是 `image[1]`（G 通道）。

注意名稱：程式變數 `target`（以及後面的 `targets`）在本節只是原始標註（GT，真值），每張圖一個 `{boxes, labels}` dict。下一節的 `build_targets` 才把它轉成每個格子的訓練目標（有哪些欄位，到時再逐一說明）；下一節說的 target 指的是這個。

下表列出資料契約規定的格式：

| 欄位 | shape／型別 | 意義 |
| --- | --- | --- |
| image | `[3,64,64]` float32 | RGB、CHW、畫素值在 `[0,1]` |
| boxes | `[N,4]` float32 | 這張圖片的框，以畫素為單位的 xyxy |
| labels | `[N]` int64（就是 `torch.long`） | 每個框對應一個類別 id |
| batch 影像（程式變數 `images`） | `[B,3,64,64]` | 即第 1 章的 NCHW（這裡的 N 就是 B，和框數 N 無關），可直接進 CNN |
| batch 標註（程式變數 `targets`） | 長度 B 的 list | 每個元素是一張圖的 `{boxes, labels}` dict；各張的 N 可以不同 |

表中 N＝這張圖的框（物件）數，可以是 0；B＝一個 batch 的圖片數；CHW＝通道、高、寬。

逐步核對：紅色畫素的 (R,G,B) 是 (1,0,0)，藍色是 (0,0,1)，背景是 (0,0,0)。你可以這樣手算：

1. 紅色左上角的畫素是 `(x=8,y=12)`，位於 `image[0,12,8]`，值是 1。
2. 最後一個紅畫素是 `(x=23,y=27)`，即 `image[0,27,23]=1`。它右邊的 `image[0,27,24]` 和下方的 `image[0,28,23]` 都是 0，符合「右邊界 24、下邊界 28 不含在內」。
3. 算通道總和：紅色是 16×16×1=256，藍色也是 256，綠色是 0。

這比只看圖片 shape 多核對了通道、x 與 y 的順序、框邊界和畫素內容。

## 為何標註不能像影像一樣疊起來

接著把兩張圖組成一個 batch（B=2）：第一張是上面的固定場景，第二張刻意放一張全黑、沒有任何物件的空圖。

空圖的框寫成 `torch.empty(0,4)`，它建立一個 shape 為 `[0,4]` 的 tensor：0 個框，每個框 4 個數，裡面一個元素也沒有。這和一個全 0 的框 `[[0,0,0,0]]`（shape `[1,4]`）不同。類別寫成 `torch.empty(0,dtype=torch.long)`，shape 是 `[0]`。dtype 仍要是 long，因為下一節的 `build_targets` 要求類別必須是 long。第 4 章〈[座標轉換與還原](04-coordinates.md)〉也提過：沒有框時仍保留 `[0,4]`。

這張空圖不能刪。它提供背景監督，也就是教模型「這裡沒有物件」的訓練訊號。下一節把標註轉成每格的訓練目標時，空圖 4×4 共 16 格的 objectness 目標都是 0。再下一節的 loss 用 BCE（binary cross entropy，二元交叉熵：目標是 0 或 1 時用的 loss）把這 16 格預測的 objectness 往 0 推。框和類別的 loss 只算有物件的格子，這張空圖不貢獻。第一張圖雖然也有 14 格背景，但刪掉空圖，模型就少了整張都是背景的例子，程式也測不到 N=0 的情況。

那為什麼不把兩張圖的框也疊成一個 tensor？`torch.stack` 會把 shape 完全相同的 tensor 沿新的第 0 軸疊起來：兩張 `[3,64,64]` 影像疊成 `[2,3,64,64]`。兩張圖的框卻是 `[2,4]` 與 `[0,4]`，shape 不同，`torch.stack` 會報 RuntimeError，訊息開頭是 `stack expects each tensor to be equal size`（stack 要求每個 tensor 大小相同）。

也不能用 `[0,0,0,0]` 這種假框占位（例如在空圖補假框，讓每張的框數一樣多）。它的寬高都是 0，下一節的 `build_targets` 會直接報 ValueError（標註框必須有正面積）。就算沒有這道檢查，假框也得配一個類別，會被當成中心在 (0,0) 的真物件：左上角那格變成正格，等於教模型「全黑的圖左上角有物件」。

所以本書在 `miniyolo/data.py` 寫了一個小函式 `collate`。它的輸入是一串（影像, 標註）配對：影像用 `torch.stack` 疊成 `[B,3,64,64]`；標註不疊，原樣放進長度 B 的 list；兩者一起回傳。PyTorch 內建的 `DataLoader`（負責把資料一批批取出的工具）也能用參數 `collate_fn` 接這種函式，第 8 章〈[用自己的資料](08-own-data.md)〉示範讀自己的資料時就這樣用。本章的程式則是直接呼叫 `collate`，完整程式裡是這幾行：

``` { .python data-excerpt="lesson_cases/07-data.py" }
from miniyolo.data import ShapeDataset, collate  # 在完整程式開頭；ShapeDataset 在本頁下文介紹
...  # 省略中間的程式，包括前文的畫圖、寫標註，以及後文的第 1 項檢查
empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
# torch.zeros_like(image)：和 image 同 shape、同 dtype 的全 0 tensor，也就是那張全黑的空圖
images, targets = collate([(image, target), (torch.zeros_like(image), empty)])
# images 是 [2,3,64,64] 的 tensor；targets 是 [target, empty]，長度 2 的 list
```

## 固定場景之外：生成資料 ShapeDataset

固定場景只用 0 和 1 兩種畫素值，所以能逐值精確核對。後續訓練用的是生成資料，不是反覆學本節這兩個 256 畫素的固定色塊。

`ShapeDataset` 是本書 `miniyolo/data.py` 裡的 PyTorch Dataset 類別，可以用 `dataset[i]` 取出第 i 筆資料。給它編號 i，它就依固定的亂數種子（seed）畫出第 i 張圖和標註；同一個編號每次都畫出同一張。它的圖有這些性質：

- 預設設定下，每張有 0～2 個矩形，所以會出現空圖。本節完整程式沿用預設，用 seed=7 取 8 張，其中就有 3 張空圖。
- 每個矩形整個落在 4×4 格的某一格（16×16 畫素）裡，寬、高是 8～15 畫素。因此中心嚴格位於格子內，不會落在格線上；下一節換算格內 xy 時，也不會得到端點 0。同一張圖的矩形一定在不同格子，避開第 5 章講的同格碰撞（兩個物件中心落在同一格）。
- 背景每個畫素是 [0, 0.04) 的微小隨機雜訊。紅色塊的 RGB 是 (0.95, 0.10, 0.10)，藍色是 (0.10, 0.10, 0.95)。

## 執行完整程式：三項檢查

可以用頁首的按鈕在 Colab 執行完整程式，或在專案根目錄執行 `PYTHONPATH=. python lesson_cases/07-data.py`。程式不下載任何檔案、不需要 GPU，也不建立模型。預期輸出三行：

```text
batch (2, 3, 64, 64) counts [2, 0]
red/blue channel sums 256.0 256.0
dataset contract checked: 8 images
```

第一行是 batch 影像的 shape，以及兩張圖各自的框數（固定場景 2 個、空圖 0 個）。第二行是第一張圖紅、藍兩個通道各自的總和。第三行表示 8 張生成資料都通過了檢查。

印出這三行之前，完整程式已經用斷言（assert：條件不成立就報錯停下）做完三項檢查：

1. 依標註重畫「預期影像」（程式變數 `expected_pixels`）：從全 0 開始，類別 0 的框內在通道 0（R）填 1，類別 1 的框內在通道 2（B）填 1。它必須和原圖每個值都相等（`torch.equal`）。
2. batch 格式：`images` 的 shape 是 (2,3,64,64)，空圖的 boxes 是 (0,4)；紅框、藍框所在的那塊切片，總和各是 256。
3. 8 張生成資料：影像是 float32、shape (3,64,64)；boxes 的列數等於 labels 的個數；每個框的座標都在 0～64 內，而且 x2>x1、y2>y1。

第 1 項在完整程式裡是這幾行：

``` { .python data-excerpt="lesson_cases/07-data.py" }
expected_pixels = torch.zeros_like(image)
for box, label in zip(target["boxes"], target["labels"]):
    x1, y1, x2, y2 = box.to(torch.long).tolist()  # 框座標轉成整數，才能當切片用
    channel = {0: 0, 1: 2}[label.item()]  # 類別 0（紅）→ 通道 0；類別 1（藍）→ 通道 2
    expected_pixels[channel, y1:y2, x1:x2] = 1
assert torch.equal(image, expected_pixels), "Annotation and colored pixels disagree"
```

為什麼要從標註反推畫素？試試在完整程式把紅框的 x1 從 8 改成 9，畫素不改。依標註重畫的預期影像少了 x=8 那一欄的 16 個紅畫素：預期影像的紅色總和是 240，原圖是 256。第 1 項的斷言會失敗，程式停在那裡，錯誤訊息是 `AssertionError: Annotation and colored pixels disagree`。第 2 項的紅色總和檢查用寫死的切片 `8:24`，只看畫素、不看標註；就算執行到也仍是 256，抓不到這種錯。

## 用人工矩形當資料：收益與代價

可控圖形讓畫素、標註、空圖與多物件能先被逐項驗證，失敗時不用猜是真實資料太難，還是程式錯了。代價是資料過於乾淨：永遠只有矩形、只有兩種接近純色的顏色、背景幾乎全黑（生成資料只多了極小的雜訊）。所以通過這裡的檢查，不能證明模型能辨識真實照片。

稍後接自己的圖片時，要沿用同一套資料契約，例如 RGB 順序、以畫素為單位的 xyxy 框，以及開頭契約的最後一條（沒有物件的圖照樣保留，框的 shape 寫成 `[0,4]`）。另外還要處理兩件事。一是原圖尺寸：要 resize 或 letterbox（等比縮放後補邊），見第 4 章〈[座標轉換與還原](04-coordinates.md)〉。二是按來源切分訓練／驗證／測試資料：同一個來源（例如同一段影片）的圖要整組放在同一邊，否則模型評估時遇到的，其實是訓練時看過的近似圖片。第 8 章〈[用自己的資料](08-own-data.md)〉會細講。

接自己的圖片時，常見錯誤有四種：

1. 把 Pillow 讀進來的圖直接當成 CHW。Pillow 是常用的 Python 讀圖套件，沿用 PIL（Python Imaging Library，Python 影像函式庫）作為匯入名稱。圖片轉成 NumPy 陣列後是 `[H,W,C]`，要先用 `torch.from_numpy` 轉成 tensor，再用 `permute(2,0,1)` 換成 `[C,H,W]`；用 `reshape` 硬改形狀會把顏色和位置混在一起（見第 1 章〈[VGG 風格小 CNN](01-small-cnn.md)〉）。
2. 把 RGB 讀成 BGR。OpenCV（另一個常用的影像套件）的 `cv2.imread` 讀出的通道順序是 B、G、R；當成 RGB 用，R 和 B 會對調。在本章，這等於紅色（類別 0）變成藍色（類別 1），標註全部對不上。
3. 0～255 的整數畫素沒有除以 255。數值比契約的 0～1 大了 255 倍；就算已轉成 float32，shape 和 dtype 都對，這兩項檢查抓不到。
4. 圖片 resize 了，框卻還用原圖座標，框就落在錯的位置（第 4 章）。

先畫標註疊圖（把框畫在圖片上，用眼睛看有沒有對齊），再檢查數值。框的座標範圍檢查（在 0～64 之內、x2>x1、y2>y1）抓不到「合法但位置錯」的框。例如把紅框誤寫成 x、y 對調的 `[12,8,28,24]`：四個數都在 0～64 內，x2>x1、y2>y1 也成立，範圍檢查全部通過，下一節的 `build_targets` 也照收；框卻和紅色方塊錯開了，往右、往上各偏 4 畫素。只有畫標註疊圖，或依標註重畫畫素再逐值比對，才抓得到。

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

下一節會把這個 16×16 的紅框，轉成某一個格子要學的訓練目標。

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
