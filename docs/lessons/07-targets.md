# 7.2 Grid MiniYOLO：把框變成訓練目標

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/07-targets.ipynb){ .md-button }

一張圖只有兩個框，模型卻固定輸出 4×4=16 個位置的預測：哪一格該學哪個框？其他 14 格又該學什麼？本節把標註框換算成這 16 個位置各自的正確答案，也就是訓練目標（target）。讀完你能手算任何一個框由哪一格負責、那一格的 target 是哪四個數，也知道每一格要算哪些 loss。

前置：上一節的[資料契約](07-data.md)（圖片、框與類別要照什麼格式存）。

訓練時告訴模型「每個輸出位置的正確答案是什麼」的資料，叫做監督（supervision）；本節產生的 target 就是監督。本節還沒有模型參與運算：我們用人工設定的兩個框和已知答案，檢查 assignment（責任分配：哪個輸出位置負責哪個真實框）。assignment 決定訓練時的監督；它和推論時的 NMS（刪除重複預測框）、評估時的 matching（配對預測框與真實框）是三個不同的程式。

「物件中心落在哪一格，就由那一格負責」的概念來自 [YOLOv1](https://arxiv.org/abs/1506.02640)。本節用的是簡化版，不是原版。

??? note "和 YOLOv1 原版差在哪"

    YOLOv1 原版（PASCAL VOC 資料集的設定）把圖切成 7×7 格，每格預測 2 個框；訓練時，由和真實框 IoU（重疊比例）較大的那個框負責。本節簡化成每格只有 1 個框，不必挑選；類別用兩類 softmax。

    本節也維持 4×4 格、不加 anchor（預先指定的框寬高）的基本設定，也就是後續實驗拿來對照的基線（baseline）。anchor 到第 9 章才介紹。

## 由 pixel 一步一步走到 cell

本節的例子是一個 batch，共 2 張圖（B=2，B 是一個 batch 的圖片數）。第 0 張是上一節的固定場景，有紅框和藍框；第 1 張是上一節的空圖，沒有框。`b` 是圖片的索引，本例是 0 或 1。

64×64 圖分成 4×4 格，每格寬、高各 16 pixel。格子用兩個索引表示：gy 是第幾列（由上往下數）、gx 是第幾欄（由左往右數），都從 0 起算，範圍 0～3。原始框是 pixel 單位的 `[x1,y1,x2,y2]`，x 向右、y 向下。

![64×64 圖片切成 4×4 格；紅框左上角 (8,12)、右下角 (24,28)，中心黑點 (16,20) 落在淺藍色虛線的負責格](../assets/diagrams/object-journey.svg){ width="400" }

這張圖和首頁是同一張，只畫了紅框。淺藍色虛線格是紅框的負責格，不是本節的藍框。圖例寫的「第 2 列、第 2 欄」是從 1 數起；換成從 0 起算的索引，就是 gy=1、gx=1。

每個框都照下面五步換算，紅框、藍框並排對照：

| 步驟 | 紅框 `[8,12,24,28]` | 藍框 `[40,36,56,52]` |
| --- | --- | --- |
| 1. 算中心與寬高（pixel） | 中心 (16,20)，寬高 (16,16) | 中心 (48,44)，寬高 (16,16) |
| 2. 中心除以格寬 16 | (1, 1.25) | (3, 2.75) |
| 3. 取 floor，得到負責格 | gx=1、gy=1 | gx=3、gy=2 |
| 4. 減掉格子索引，剩下的小數是格內位置 | (0, 0.25) | (0, 0.75) |
| 5. 寬高除以整張圖 64 | (0.25, 0.25) | (0.25, 0.25) |
| box target：[格內 x, 格內 y, 寬, 高] | [0, 0.25, 0.25, 0.25] | [0, 0.75, 0.25, 0.25] |

第 3 步的 floor 是向下取整：取不超過它的最大整數，例如 floor(1.25)=1、floor(2.75)=2；對正數來說就是無條件捨去。第 4 步的 0.25 表示中心在格子上緣往下 0.25 格，也就是 0.25×16=4 pixel（20−16=4）；格內 x 是 0，表示中心剛好在格子的左緣。

寫成一般式（格寬、格高都是 64/4=16，cx、cy 是中心座標，⌊ ⌋ 就是 floor）：

- 負責格：gx = ⌊cx/16⌋、gy = ⌊cy/16⌋
- 格內位置：格內 x = cx/16 − gx、格內 y = cy/16 − gy
- 寬、高：各除以整張圖 64

target 張量的前三軸依序是圖片 b、格子列 gy、格子欄 gx。所以紅框的答案存在 `box[0,1,1] = [0,.25,.25,.25]`（b=0、gy=1、gx=1）。四項都是沒有單位的比例：xy 以格子為尺，wh 以整張圖為尺。

為什麼用兩種尺？哪一格已經由 gx、gy 記下，xy 只需記格內剩下的零頭，一定落在 0 到 1 之間。框卻可能比一格大：寬 40 pixel 的框除以 16 得 2.5，超出 sigmoid 能輸出的範圍（本節後面會看到，head 的四個框輸出都要先經 sigmoid）。改除以整張圖 64，合法框的寬高不會超過整張圖，結果一定在 0 到 1 的範圍內。YOLOv1 原文也是為了讓四個數都落在 0 與 1 之間，才這樣定義。

紅框中心的 x=16 正好落在第 2 欄（索引 gx=1，從 0 起算）的左邊界。每欄都是左閉右開：[0,16)、[16,32)、[32,48)、[48,64)。16 屬於 [16,32)，算式上就是 floor(16/16)=1。藍框中心的 x=48 同理歸 gx=3，所以兩個框的格內 x 都是 0。

索引先 y 再 x，向量內先 x 再 y；兩個順序同時存在，必須分清。例如藍框的答案存在 `box[0,2,3]`（gy=2、gx=3），向量的前兩個數卻依序是格內 x=0、格內 y=0.75。

## 正格、負格與 mask

物件中心落入的格子是正格，其餘是負格。紅框其實蓋到 4 格（列 0～1、欄 0～1，見上圖），只有中心 (16,20) 所在的 (gy=1,gx=1) 是正格；另外 3 格看得到紅色，objectness 目標仍是 0。模型要學的是「物件中心在不在這格」，不是「這格有沒有紅色」，也就是第 5 章說的「責任只看中心」。

objectness 目標表示這格是否被分配到物件。把第 0 張圖的 objectness 排成 4×4，列是 gy、欄是 gx：

|  | gx=0 | gx=1 | gx=2 | gx=3 |
| --- | --- | --- | --- | --- |
| gy=0 | 0 | 0 | 0 | 0 |
| gy=1 | 0 | 1（紅框） | 0 | 0 |
| gy=2 | 0 | 0 | 0 | 1（藍框） |
| gy=3 | 0 | 0 | 0 | 0 |

第 1 張空圖的 16 格全是 0。positive 是布林 mask（遮罩：由 True／False 組成的張量），用來選出正格。

下表的 S=4 是每邊的格數（第 5 章和下一節也寫成 S）；小寫 gy、gx 則是某一格的索引，範圍 0～3。

| target 欄位 | 本例 shape | 內容 | 哪些格子進 loss |
| --- | --- | --- | --- |
| box | `[2,4,4,4]`，即 `[B,S,S,4]` | 2 張圖 × 4 列 × 4 欄 × 4 個框數字；最後一軸依序是格內 x、格內 y、全圖寬比例、全圖高比例 | 只有正格 |
| objectness | `[2,4,4]` float | 要學的目標值：正格 1，其餘 0 | 32 格全部 |
| positive | `[2,4,4]` bool | 正格 True，其餘 False | 本身不進 loss，只用來挑出正格 |
| class_ids | `[2,4,4]` long | 正格存物件的類別 id | 只有正格（positive=True 時才讀取） |

本節的 objectness 剛好等於 positive 轉成 0／1，但兩者用途不同：objectness 是要學的答案，positive 是挑格子算框與類別 loss 的開關。其他設計兩者可以不同。例如 YOLOv1 原版裡對應 objectness 的是 confidence（信心分數）；負責物件的那個框，confidence 目標是預測框與真實框的 IoU，不是固定的 1。

第 0 張有 2 個正格、14 個負格；第 1 張空圖 16 格都是負格。整個 batch 正格 2、負格 30。本節沒有 ignore 狀態（第 5 章提過：暫時不算某項 loss 的格子），所有非正格都學背景。

本節 build_targets 的負格 box 填 0、class_ids 也填 0（第 5 章的示範填 −1），都只是建立張量時先填好的預設值，沒有意義。0 剛好也是紅色的類別 id，只看 class_ids 分不出紅色和背景；判斷正負格一律用 positive。負格不能放進框的 MSE（均方誤差）平均，也不能拿去算分類 loss；否則 30 個負格的無意義數值會混進平均，把結果汙染掉。

## Head 與 mask 如何接起來

head 的輸出是 `[B,4,4,7]`：每格 7 個數，依序是 `tx,ty,tw,th,obj,class0,class1`。七個數都是 logits（還沒轉成機率或比例的原始分數）。前四個代表框，要先經 sigmoid 變成 0 到 1 之間（不含端點）的數，才和 box target 比較。

??? note "target 剛好是 0，sigmoid 學得到嗎？"

    sigmoid 的輸出永遠在 0 與 1 之間，碰不到端點。本節紅、藍框的格內 x 剛好是 0（中心落在格線上），這種 target 只能逼近，無法剛好達到。後面訓練用的生成資料（ShapeDataset），物件中心不會落在格線上。後來的 YOLO 版本會調整 xy 的輸出範圍來處理這件事，YOLOv4 稱為 grid sensitivity。

每一段輸出對照哪個 target、用哪種 loss：

| head 每格的輸出 | 對照的 target（算哪些格子） | loss |
| --- | --- | --- |
| `tx,ty,tw,th`，先經 sigmoid | box（只算 2 個正格） | MSE |
| `obj` | objectness（32 格全算） | BCE（binary cross entropy，二元交叉熵），回答「有沒有物件」的是非題 |
| `class0,class1` | class_ids（只算 2 個正格） | 交叉熵：第 1 章的 `cross_entropy`，內部先做 softmax，回答「紅還是藍」 |

公式在下一節手算。背景不是第三個 class，objectness 已承擔「有沒有物件」。

下面是依完整程式改寫的簡化版，把本節的兩張圖轉成 target。輸入 `[scene, empty]` 是原始標註，格式就是上一節 `collate` 留下的 `targets` 清單：`scene` 含紅、藍兩框（類別 0、1），`empty` 沒有框。`grid_size`、`image_size`、`num_classes` 依序是每邊格數 S、圖片邊長（pixel）與類別數。輸出 `target` 是轉換後要送進 loss 的訓練目標，也就是前面表格的 box、objectness、positive、class_ids 四個欄位，每張圖都固定是 4×4。上一節的 `targets`（標註清單）和這裡的 `target`（訓練目標）只差一個 s：前者每張圖的框數可以不同，後者形狀固定。

簡化版只從完整程式取了建立 target 的那一行，其餘都是為了說明才加的。`pos`、`red` 這兩個名字完整程式沒有，它直接寫 `target["positive"]`、`target["box"][0, 1, 1]`（Python 的字串用單引號或雙引號都一樣）。最後三行註解示範 mask 怎麼用：`pred` 代表 head 的輸出，本例 shape `[2,4,4,7]`；它到下一節才會出現，本節完整程式沒有模型，所以只寫成註解。

```python
# 完整程式（Colab）裡先 import 了 build_targets，並建立 scene 與 empty
target = build_targets([scene, empty], grid_size=4, image_size=64, num_classes=2)
pos = target['positive']
red = target['box'][0, 1, 1]  # [0,.25,.25,.25]
# pred[..., :4]      → [2,4,4,4]：... 保留前三軸（b、gy、gx），:4 取最後一軸的前 4 個數
# pred[..., :4][pos] → [2,4]：布林索引只留 pos 為 True 的 2 格，依 (b,gy,gx) 由小到大排
# target['box'][pos] → [2,4]：同一個 pos、同樣順序，所以兩邊的第 i 列是同一格
```

objectness 不用 pos：`pred[..., 4]` 的 32 格全部和 `target['objectness']` 比較。類別也用 pos 挑出 2 格：`pred[..., 5:][pos]` 的 shape 是 `[2,2]`，對照 `target['class_ids'][pos]`，也就是 `[0,1]`。

可以用頁首的「在 Colab 執行本節」按鈕執行，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-targets.py`。

完整程式建好 target 後，先用斷言（assert：條件不成立就報錯停下）核對本節手算的答案：box 的 shape 是 `[2,4,4,4]`；整批正格共 2 個；`box[0,1,1]` 是 `[0,.25,.25,.25]`、`box[0,2,3]` 是 `[0,.75,.25,.25]`；這兩格的 class_ids 依序是 0 和 1；第 1 張空圖沒有任何正格。斷言通過時不會印出任何東西。

印出結果之前，程式還做一個碰撞測試：另外建一組標註，放紅框和 `[10,14,26,30]` 兩個框（類別都是 0）。後者的中心是 (18,22)，18/16=1.125、22/16=1.375，取 floor 後也落在 (gx=1,gy=1)，和紅框同一格。build_targets 因此拋出 ValueError（Python 表示「輸入值不合理」的錯誤），不會悄悄覆蓋第一個框。程式用 try/except 接住這個錯誤，印出 `same-cell collision rejected`，這就是輸出的第一行。要是 build_targets 沒有報錯，程式會自己丟出 AssertionError（斷言失敗時出現的那種錯誤）停下，訊息是 `Two objects silently overwrote one slot`（意思是兩個物件悄悄寫進同一格，後一個蓋掉前一個）。

之後依序印出正格 `[[0,1,1],[0,2,3]]`（每組是 b, gy, gx；程式用 `target["positive"].nonzero()` 列出所有 True 的位置）、紅框與藍框的 target，以及 `positive/negative counts 2 30`。最後這行的 2 是直接寫在 print 裡的數字，前面的斷言已確認正格確實是 2 個；30 則是程式從 positive 數出來的負格數。如果你自己把 `scene` 的第二個框改成 `[10,14,26,30]`，程式會在呼叫 build_targets 建立 target 時停下，錯誤訊息是 `ValueError: same-cell collision: image 0, cell (1,1)`（第 0 張圖，括號裡依序是 gy、gx）；這是預期行為，因為同一格放不下兩個框。

## 得到的能力與留下的限制

固定格子讓輸出數量與 mask 很簡單。建立 target 時，空圖片也不必特別處理：16 格都當負格，只學 objectness=0。（算 loss 時另有一件事要處理：若整批都沒有正格，例如整個 batch 只有一張圖（B=1），而且那張是空圖，框與類別 loss 會對 0 個數取平均，得到 NaN；下一節〈[Grid MiniYOLO loss](07-loss.md)〉會用 `pos.any()` 分支處理。）代價是格子容量：每格只有一個框槽（第 5 章的 slot：一個可以輸出框的位置），兩個物件中心落在同一格時，即使類別相同也放不下。類別數從 2 變 3 時，head 最後一軸只從 7 變 8，框仍只有一組，所以增加類別不會增加框槽。

上一節的生成資料（ShapeDataset）刻意把每個物件放在不同格子，只為先把「資料→target→loss→訓練」整條流程跑通；真實照片裡，兩個物件的中心可能落在同一格。第 9 章 anchor slot 與第 10 章多尺度會分別討論容量及解析度。

常見錯誤：

- **寬高除以格寬 16**：紅框會得到 (1,1)；正確是除以整張圖 64，得到 (0.25,0.25)。
- **用左上角 (8,12) 找格子**：floor(8/16)=0、floor(12/16)=0，會落到 (gy=0,gx=0)；正確是用中心 (16,20)，落在 (gy=1,gx=1)。
- **用 class_ids 的值判斷正負格**：例如把 `class_ids == 0` 當成背景，或沿用第 5 章填 −1 的習慣，把 `class_ids != -1` 當成正格。本節負格也填 0，前者會連紅框的正格一起當成背景，後者會把 32 格全當成正格。判斷正負格一律用 positive。
- **索引寫成 `pred[:,gx,gy]`**：正確是 `pred[:,gy,gx]`，先列 y、後欄 x。紅框在 (1,1)，寫反也看不出差別，要用藍框核對：正確取 `pred[0,2,3]`，寫反會取到 `pred[0,3,2]`。

用紅、藍框的數字逐項核對，遠比看 loss 是否變小可靠。

自主練習（先自己算，再展開答案）：

1. 框 `[4,4,12,12]`（類別 0）的 target 是什麼？由哪一格負責？
2. 框 `[40,2,48,10]` 由哪一格負責？`positive.nonzero()` 應列出什麼？box target 是什麼？這題中心的 x、y 不同，專門檢查索引順序。

想用程式核對時，不要改 notebook 裡原本的 `main()`（包住整個示範的函式）。先依序執行 notebook 原本的程式碼儲存格（code cell，和本頁的網格 cell 無關），再在最後一個儲存格下面新增一個程式碼儲存格，貼上參考答案裡的程式另外測試。這樣原本範例的檢查仍然有效。

??? note "參考答案"

    **第 1 題**：中心 (8,8)、寬高 (8,8)。8/16=0.5，floor 得 gx=0、gy=0；格內位置 (0.5,0.5)，寬高除以 64 得 (0.125,0.125)。所以由 (gx=0,gy=0) 負責：`box[0,0,0] = [.5,.5,.125,.125]`、objectness=1、class_ids=0；只有這一格的 positive 是 True，其餘 15 格都是負格。

    用下面的程式核對；兩個斷言（assert）都成立，程式才不會報錯：

    ```python
    new_box = {'boxes': torch.tensor([[4.,4.,12.,12.]]), 'labels': torch.tensor([0])}
    new_target = build_targets([new_box], grid_size=4, image_size=64, num_classes=2)
    # nonzero() 列出所有 True 的位置，每列是 [b, gy, gx]
    assert new_target['positive'].nonzero().tolist() == [[0,0,0]]
    # allclose：允許極小浮點誤差的相等比較
    assert torch.allclose(new_target['box'][0,0,0],torch.tensor([.5,.5,.125,.125]))
    ```

    **第 2 題**：中心 (44,6)、寬高 (8,8)。44/16=2.75、6/16=0.375，floor 得 gx=2、gy=0；格內位置 (0.75,0.375)，寬高 (0.125,0.125)。`positive.nonzero()` 應為 `[[0,0,2]]`（b=0、gy=0、gx=2），`box[0,0,2] = [.75,.375,.125,.125]`。若得到 `[[0,2,0]]`，就是把 x、y 寫反了。

    用程式核對時，把上面 `new_box` 裡的框 `[[4.,4.,12.,12.]]` 改成 `[[40.,2.,48.,10.]]`；第一個斷言改成 `== [[0,0,2]]`，第二個改成比較 `new_target['box'][0,0,2]` 和 `torch.tensor([.75,.375,.125,.125])`。

再看一個邊界情況：若某個標註的中心是 (64,64)，代入得 floor(64/16)=4，但格子索引只有 0～3。合法框（x1<x2≤64）的中心一定小於 64，所以這代表標註壞了；build_targets 會先檢查框有正面積、位於 0～64 內，不合格就拋出 ValueError。不能用 clamp（把數值硬夾回範圍內，例如把 4 改成 3）掩飾錯誤：程式不會報錯，卻會把壞標註偷偷塞進最後一格。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式已於 2026-10-02 用 PyTorch 2.9.1+cpu 在 CPU 上執行過，程式裡的 assert 檢查全部通過。下面是那次印出的原始輸出；每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-targets.json)

??? example "展開本次實際輸出"

    ```text
    same-cell collision rejected
    positive indices (b,y,x) [[0, 1, 1], [0, 2, 3]]
    red target [0.0, 0.25, 0.25, 0.25]
    blue target [0.0, 0.75, 0.25, 0.25]
    positive/negative counts 2 30
    ```

<!-- curriculum-evidence:end -->
