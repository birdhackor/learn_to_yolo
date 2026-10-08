# C.11.3 增強：畫素怎麼變，框就怎麼變

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/11-augmentation.ipynb){ .md-button }

偵測器已經知道如何從一張圖建立 targets。若紅矩形總在同一位置，模型可能過度依賴位置或背景；我們希望用同一份標註，提供更多合理的位置、尺度與外觀變化。這次先不改模型與 loss，而是把訓練圖翻轉或裁切，再讓答案跟著變。

**資料增強（data augmentation）**把變換後的圖當成新的訓練樣本，不是改善畫質。偵測的答案含框和 labels，因此畫素移到哪裡，框也要移到哪裡；刪框時對應的 label 也要刪。以第 7 章的紅框 `[8,12,24,28]` 為例，圖若左右翻、框卻留原處，就把黑色背景標成紅物件，真正紅色反而沒有框，增加資料卻製造錯誤監督。

## Flip：框邊界用 W−x，畫素編號用 W−1−x

沿用 [資料契約](07-data.md) 的半開區間：64×64 圖中紅色畫素佔 x=8…23、y=12…27，框是 `[8,12,24,28]`。水平 flip（左右鏡射）後，紅色 x 變 55…40，仍是 16pixel 寬，新框為 **`[40,12,56,28]`**，y 不變。

![原圖、水平翻轉與裁切：紅色畫素及綠色標註框一起變](../assets/diagrams/11-augmentation.svg)

綠線是紅物件的框。①→②翻轉，負責格由淡藍 `(gx=1,gy=1)` 移到亮藍 `(3,1)`；①→③從白虛線區裁切，灰色是被裁掉的半個紅物件。

框邊界是畫素之間的格線，套 `u→W−u`；左右交換，所以新左界來自舊右界：`x1'=W−x2`、`x2'=W−x1`。代入 64−24=40、64−8=56，正好包含畫素 40…55。

畫素編號則是 `i→W−1−i`。畫素 i 佔格線 i 到 i+1，鏡射後佔 W−1−i 到 W−i，編號取左側格線，因此多一個−1。畫素 23→40、8→55；若把 W−1 用在框，會得到 `[39,12,55,28]`，偏左 1pixel。

``` { .python data-excerpt="lesson_cases/11-augmentation.py" }
def horizontal_flip(image,boxes):
    width = image.shape[-1]         # 圖寬，也就是本頁的 W；CHW 的最後一軸是寬
    result = boxes.clone()          # 讀舊 boxes，寫獨立副本；否則算 x2 時會讀到已改成 40 的 x1
    result[:,0] = width-boxes[:,2]  # 新 x1 = W − 舊 x2
    result[:,2] = width-boxes[:,0]  # 新 x2 = W − 舊 x1；y1、y2 不動
    return image.flip(-1),result    # 圖沿最後一軸（寬）左右鏡射，和新框一起回傳
```

先 clone 再寫副本，確保計算新 x2 時仍讀舊 x1；若直接改原框，x1 先變 40，再拿 40 算 x2 便變 24，左右倒置。圖用 `flip(-1)` 翻寬軸：CHW 的第 0 軸是通道，翻 0 會把紅藍對調；翻 1 是上下翻，都不能配水平框公式。

labels 仍是 class 0。若類別含方向，例如左轉箭頭，翻後可能應變右轉；要明訂類別對應，或不用這種增強。flip 後中心 `(48,20)`，除以 16 得 `(3,1.25)`，新的 target 應在格 `(3,1)`，不能沿用原格。

連翻兩次必須還原，但這個檢查不充分。錯用 W−1、忘交換左右、翻錯軸，都可能連翻兩次還原。完整程式另核對**翻一次**的新框與整張預期圖：紅色只在 x40…55、y12…27，框外沒有紅色。這才確認畫素、框和方向一起正確。

## Crop 會改座標，也會改可見面積

從原圖取 left=16、top=8、寬高 32 的 crop（裁切區）：原圖 x `[16,48)`、y `[8,40)`，新圖 shape `[3,32,32]`。這次先在 32×32 座標算答案，稍後再接回 64×64 模型。

紅框左半在裁切區外。先把原點移到裁切區左上：xyxy 減 `[16,8,16,8]`，得到 `[-8,4,8,20]`；再把坐標 clamp（夾回）0～32，得到 **`[0,4,8,20]`**。原框本來合法，是裁切使物件超出新圖，clamp 纔有意義；若原標註本來超出原圖，仍需修資料，不能用此步掩蓋第 7.5 節的資料錯誤。

可見寬 8、高 16，面積 128；原面積 256，可見比例 0.5。本例保留條件是**可見面積>0，而且可見比例≥0.5**。等號算保留，改門檻 0.6 則刪。這是同一物件剩餘的面積比例，不是 score 門檻或兩框 IoU。面積>0 也保護門檻設 0 的情況，否則會留下非法零面積框。

每框的判斷存成布林 `keep`。crop 函式一起完成平移、clamp、算比例、篩框：

``` { .python data-excerpt="lesson_cases/11-augmentation.py" }
def crop(image,boxes,left,top,width,height,min_visibility=.5):  # min_visibility 是可見比例門檻
    # 原點移到裁切區左上角：x 減 left、y 減 top
    shifted = boxes-torch.tensor([left,top,left,top],dtype=boxes.dtype)
    clipped = shifted.clone()
    # [:,[0,2]] 一次取 x1、x2 兩欄；clamp 把小於 0 的改成 0、大於 width 的改成 width
    clipped[:,[0,2]] = clipped[:,[0,2]].clamp(0,width)
    clipped[:,[1,3]] = clipped[:,[1,3]].clamp(0,height)  # y1、y2 同理
    # [x2,y2]−[x1,y1] 是 [寬,高]；prod(-1) 沿最後一軸相乘，得到寬×高
    area = (boxes[:,2:]-boxes[:,:2]).prod(-1)  # 原面積
    # 可見面積；clamp(min=0) 先把負的寬、高改成 0（框的 x2<x1 或 y2<y1 時才會是負的）
    visible = (clipped[:,2:]-clipped[:,:2]).clamp(min=0).prod(-1)
    keep = (visible > 0) & (visible/area >= min_visibility)  # 每個框一個 True／False
    # 裁圖：CHW 先切 y 再切 x；回傳裁切後的圖、只留 keep 為 True 的框，以及 keep
    return image[:,top:top+height,left:left+width],clipped[keep],keep
...
# main() 裡：本例切的是 image[:,8:40,16:48]，cropped 是 [3,32,32]；這裡的 clipped 是留下的框
cropped, clipped, keep = crop(image,boxes,16,8,32,32,.5)
cropped_labels = labels[keep]  # labels 用同一個 keep 篩
```

函式只適用裁切區完全在原圖內：`0≤left`、`left+width≤W`，y 同理。切片若越界，圖可能少於宣告的 width／height，框卻仍夾到宣告邊界，未必報錯；隨機取 crop 前要先保證範圍。

只查框內全是紅色，也不足以核對 crop。若切圖錯從 x12 而非 16 開始，紅色變成新圖 x0…11，框內 x0…7 仍全紅，卻多了框外一塊。程式用 `torch.equal` 比對**整張** `[3,32,32]` 預期圖，只在 x0…7、y4…19 塗紅，並核對框，才能抓到這個差異。

### 用同一個 keep 篩 labels，檢查剩下的監督

boxes 與 labels 的列是配對的，必須用同一 keep。若三框 `labels=[0,1,0]`、`keep=[True,False,True]`，框剩兩列、labels 也應變 `[0,0]`；不篩會數量不合或配錯類別，本書 build_targets 會拒絕長度不同。

刪標註不會刪畫素。門檻 0.6 會刪本例紅框，但 32×32 圖仍有 8×16 紅色。若用空標註建立第 7 章 targets，本例沒有 ignore，所有格的 objectness 都變 0，會把這塊紅色教成背景。這是未標註前景，不能當成真正空圖。門檻過低則保留只露小角、很難辨識的物件；取捨要依任務，而非刪得越多越好。

### 從 32×32 接回 64×64，重新分配格子

模型若仍要求 64×64，crop 後還需同步 resize 或 letterbox 圖與框，再建立 targets。若直接用 32×32，build_targets 要傳 `image_size=32`；預設 64 會默默算錯寬高比例與格子。letterbox 還需加補邊偏移，見 [座標轉換](04-coordinates.md)。

本程式沒有 resize，下圖是手算：32 放大到 64，比例 2，不補邊。框 `[0,4,8,20]` 乘 2 成 `[0,8,16,40]`，中心 `(8,24)`；按每格 16pixel，落在 `(gx=0,gy=1)`。同一塊可見紅色也放大 2 倍，既換位置也換尺度。

![crop 後 32×32 框放大到 64×64，再依新中心重建 4×4 targets](../assets/diagrams/11-crop-resize-target.svg)

上站是 crop 程式輸出的 32×32 座標，下站是手算 resize 後 64×64；只有下站疊 targets 網格，藍虛線標新中心 `(8,24)` 的格 `(0,1)`。先變圖、框，再算 target，不能沿用增強前 `(1,1)`。

## 執行完整程式

執行 `PYTHONPATH=. python lesson_cases/11-augmentation.py` 或頁首 Colab。本例固定 flip 與 crop 參數以便核對；MiniYOLO 訓練程式尚未接上增強。這是資料幾何檢查，沒有 backward 與 AP；seed 7 與 2 執行緒是共用設定，本例沒有抽亂數。

```text
flip box [[40.0, 12.0, 56.0, 28.0]] double flip is identity
crop box [[0.0, 4.0, 8.0, 20.0]] visible area 128/256=.5
labels after visibility .5/.6 [0] []
no-object image: boxes [0,4], labels Long[0], pixels stay zero
```

前兩行的框與第三行 labels 是算出的；`double flip is identity`、`visible area 128/256=.5`、整個第 4 行是通過斷言後的固定說明。所有 assert 在 print 前，包含新框、整張預期圖、兩次 flip 還原與空圖。

門檻 0.5 時 labels 為 `[0]`，是唯一框的 class 0；0.6 時是 `[]`，長度 0，boxes 保持 `[0,4]`，labels 保持 long 且 shape `[0]`。第 4 行測的是真正全零畫素、零物件的圖：變換後畫素仍零，空 shape 與 labels 保留。dtype 由篩選保持，完整程式亦核對 0.6 組篩後為 long。不要用仍有紅色但刪完標註的圖代替這項空圖測試。

## 增強放在資料讀取與 targets 之間

讀出圖與標註後，先同步增強，才按新框建 targets。訓練時常隨機抽，例如 YOLOv5 v6.0 每張水平翻轉機率 fliplr=0.5；validation／test 則用固定前處理，避免每次評估材料都變。

固定 seed 讓重跑的亂數序列一致，不要求同張圖每個 epoch 用同一變換。epoch 是一輪用完訓練資料；不同輪通常從序列抽到不同結果。本書 ShapeDataset 每個索引以固定 seed 重造同圖，若增強沿用那個生成器，每輪反而總抽到同一變換，需另安排亂數來源。

增強省下重新標註的工作，卻增加資料處理時間、類別方向規則與可見比例設定，也可能讓 partial 物件更難學。要驗證有用，應固定來源切分與訓練步數或計算預算，比有、無增強在 held-out 上的品質；本例只確認同步幾何。

??? note "歷史來源與進階增強"

    [YOLOv4](https://arxiv.org/abs/2004.10934) 討論了 bag of freebies（免費贈品包），包括 Mosaic 等訓練策略。YOLOv4 把 bag of freebies 定義成只改訓練方法或只增加訓練成本、但不增加推論成本的改進：訓練時多花工夫，模型實際使用時速度不變，所以像是免費的。資料增強就是最常見的一種。Mosaic 把 4 張訓練圖拼成 1 張，各張圖的框也跟著換算到拼接後的位置。

    [YOLOv5 v6.0 augmentations.py](https://github.com/ultralytics/yolov5/blob/956be8e642b5c10af4a1533e09084ca32ff4f21f/utils/augmentations.py) 提供程式實作，收的是 mixup、隨機透視、letterbox 等函式：

    - mixup：把兩張圖按比例疊在一起；YOLOv5 的版本會保留兩張圖的所有框。
    - 隨機透視（random perspective）：隨機做旋轉、平移、縮放、斜切、透視等幾何變形，框也跟著一起變換。變換後再用 `box_candidates` 刪框：寬、高都要大於 2 畫素，面積和照同樣倍率縮放的原框相比要大於 0.1，長寬比要小於 20；每個框和它的類別存在同一列，所以會一起被刪掉。
    - letterbox：等比例縮放後補邊（見第 4 章）。

    Mosaic 和訓練時整張圖的左右翻轉不在這個檔案，而在同版本的 [v6.0 datasets.py](https://github.com/ultralytics/yolov5/blob/956be8e642b5c10af4a1533e09084ca32ff4f21f/utils/datasets.py)：Mosaic 在 `load_mosaic`；左右翻轉在 `__getitem__`，依 `fliplr` 的機率執行。YOLOv5 在這一步用的是除以圖寬後的中心 x（正規化中心 x），所以翻轉時把它改成 1−x。這和本節的 W−x 是同一件事：中心 cx 翻轉後是 W−cx，除以 W 就是 1−cx/W。

    這些來源包含多種策略，不等於本節的兩個操作。本節簡化成人工固定的水平翻轉（horizontal flip）與裁切（crop），沒有 Mosaic 四圖拼接、mixup 或隨機透視。

    要不要在訓練中保留某種增強，應該用固定預算（budget）的對照來決定：在相同的訓練步數或計算量下，有、無這種增強分別訓練，再比較結果。本節沿用第 7 章 64×64 圖上的紅框例子，只檢查圖、框、labels 是否同步變換，沒有訓練；flip、crop 與 Mosaic、mixup 都沒在相同訓練步數下比較過效果。

自主練習（手算即可；先自己算，再展開答案）：

1. 寬 64 的圖上，整張圖大小的框 `[0,0,64,64]` 翻轉後是多少？
2. 紅框裁切後，可見面積只剩原本的一半時，是否一定會被刪除？
3. 同樣的裁切（left=16、top=8、32×32）套在藍框 `[40,36,56,52]` 上，框會變成多少？可見比例多少？門檻 0.5 時留不留？若紅、藍兩框在同一張圖上，keep 和 labels 會變成什麼？

??? note "參考答案"

    **第 1 題**：new_x1 = 64 − 64 = 0、new_x2 = 64 − 0 = 64，y 不變，所以仍是 `[0,0,64,64]`。

    **第 2 題**：不一定，取決於明寫的可見比例門檻。本例比例剛好 0.5：門檻 0.5（程式用 `>=`）時保留，門檻 0.6 時刪除。

    **第 3 題**：減掉 (16, 8) 得 `[24,28,40,44]`；夾回（大於 32 的改成 32）得 `[24,28,32,32]`。可見面積 8×4 = 32，原面積 16×16 = 256，比例 32/256 = 0.125，小於 0.5，所以刪除。紅、藍兩框在同一張圖上時，`keep=[True,False]`，boxes 只剩 `[[0,4,8,20]]`，labels（紅 0、藍 1）只剩 `[0]`。

    想用程式核對第 3 題，可以在 Colab 執行完完整程式後，新增一格執行；在自己的電腦上，則執行 `PYTHONPATH=. python -i lesson_cases/11-augmentation.py`，程式跑完出現 `>>>` 提示後再貼上：

    ```python
    both = torch.tensor([[8., 12., 24., 28.], [40., 36., 56., 52.]])
    _, kept, keep = crop(torch.zeros(3, 64, 64), both, 16, 8, 32, 32)
    print(kept.tolist(), keep.tolist())  # [[0.0, 4.0, 8.0, 20.0]] [True, False]
    print(torch.tensor([0, 1])[keep].tolist())  # labels（紅 0、藍 1）只留 [0]
    ```

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-augmentation.json)

??? example "展開本次實際輸出"

    ```text
    flip box [[40.0, 12.0, 56.0, 28.0]] double flip is identity
    crop box [[0.0, 4.0, 8.0, 20.0]] visible area 128/256=.5
    labels after visibility .5/.6 [0] []
    no-object image: boxes [0,4], labels Long[0], pixels stay zero
    ```

<!-- curriculum-evidence:end -->
