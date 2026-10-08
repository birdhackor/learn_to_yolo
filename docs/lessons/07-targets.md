# 7.2 Grid MiniYOLO：把框變成訓練目標

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/07-targets.ipynb){ .md-button }

上一節的第一張圖有紅、藍兩個框，第二張是空圖。模型卻對每張圖固定輸出 4×4=16 格的預測。現在要把變動數量的標註，寫成這 16 格各自應學的答案：誰負責紅框，誰負責藍框，其餘格子要學什麼？

第 5 章已用「中心所在格負責」解決這件事。本節把它接到 `miniyolo.targets.build_targets`，讀完能手算負責格、框 target 與正負格 mask，並核對套件產生的結果。全程只轉換已知標註，沒有模型訓練。

## 紅框的哪個位置決定負責格

![紅框中心與負責格](../assets/diagrams/object-journey.svg){ width="400" }

紅框仍是 `[8,12,24,28]`，中心黑點為 (16,20)，寬高為 (16,16) pixel。淺藍色虛線標出負責格，沒有畫藍色物件。圖例的「第 2 列、第 2 欄」從 1 數起；程式從 0 起算，對應 gy=1、gx=1。gy 是由上往下的列，gx 是由左往右的欄。

64×64 圖分成 4×4 格，每格 16×16 pixel。中心除以 16，整數部分找格子，小數部分記中心在格內的剩餘位置；框寬高則除以整張圖 64。紅、藍並排做一次：

手機上可左右滑動表格，查看完整欄位。

| 換算 | 紅框 `[8,12,24,28]` | 藍框 `[40,36,56,52]` |
| --- | --- | --- |
| 中心、寬高（pixel） | (16,20)、(16,16) | (48,44)、(16,16) |
| 中心÷16 | (1,1.25) | (3,2.75) |
| floor 找負責格 | gx=1、gy=1 | gx=3、gy=2 |
| 減格索引，取格內 xy | (0,0.25) | (0,0.75) |
| 寬高÷64 | (0.25,0.25) | (0.25,0.25) |
| box target | `[0,.25,.25,.25]` | `[0,.75,.25,.25]` |

floor 是向下取整，例如 floor(1.25)=1、floor(2.75)=2。紅框中心從負責格上緣 y=16 往下 4 pixel，所以格內 y=4/16=0.25；中心 x=16 剛好在左緣，所以格內 x=0。格子邊界同樣採半開區間：[0,16)、[16,32)、[32,48)、[48,64)，因此 x=16 歸 gx=1，x=48 歸 gx=3。

一般規則是 `gx=floor(cx/16)`、`gy=floor(cy/16)`；格內 xy 是 `[cx/16−gx,cy/16−gy]`，wh 是 `[w/64,h/64]`。target 的四項都無單位，但有兩種尺：xy 用格子、wh 用全圖。格子索引已記住大致位置，xy 只需記零頭；wh 用全圖，則讓跨格的大框也能落在 0～1 的輸出範圍。例如寬 40 pixel，除格寬得到 2.5，sigmoid 無法輸出；除全圖得到 0.625，就能表達。

存放答案時，索引為 `[b,gy,gx]`，向量內卻為 `[x,y,w,h]`。本批 B=2，b=0 是兩物件圖、b=1 是空圖，所以紅框存在 `box[0,1,1]`，藍框存在 `box[0,2,3]`。用藍框核對特別有用：把索引寫反成 `[0,3,2]`，就取到另一格；紅框的 (1,1) 看不出這個錯。

## 看得到色塊，為什麼仍可能是負格

紅框蓋到列 0～1、欄 0～1 的四格，但只有中心所在的 (1,1) 被分到它。這一格是正格，其餘都是負格；objectness target 表示「有沒有被分到物件」，所以有紅色畫素不一定代表 objectness=1。

第 0 張圖的 objectness 排成下面的 4×4 表：

手機上可左右滑動表格，查看完整欄位。

| | gx=0 | gx=1 | gx=2 | gx=3 |
| --- | --- | --- | --- | --- |
| gy=0 | 0 | 0 | 0 | 0 |
| gy=1 | 0 | 1（紅） | 0 | 0 |
| gy=2 | 0 | 0 | 0 | 1（藍） |
| gy=3 | 0 | 0 | 0 | 0 |

空圖的 16 格全為 0。因此整批有 2 個正格、30 個負格（14+16）。本節沒有 ignore，也就是暫不計某項 loss 的第三種狀態；每個負格都學背景。

要算框與類別 loss，還需要一張 True／False 表選出正格，這就是 `positive` mask。它與 objectness 在此能互換成 0／1，角色卻不同：objectness 是模型要學的答案，positive 是哪些格子進框／類別 loss 的開關。

手機上可左右滑動表格，查看完整欄位。

| target 欄位 | shape（B=2、S=4） | 如何使用 |
| --- | --- | --- |
| box | `[2,4,4,4]`，即 `[B,S,S,4]` float | 正格存格內 xy、全圖 wh；只有正格進框 loss |
| objectness | `[2,4,4]` float | 正格 1、負格 0；32 格全部進 loss |
| positive | `[2,4,4]` bool | 正格 True；用來挑格，不是 loss 答案 |
| class_ids | `[2,4,4]` long | 正格存類別 id；只有 positive=True 才讀取 |

`build_targets` 初始化負格 box=0、class_ids=0，這些是占位值，不是框或類別答案。第 5 章曾填 class_ids=−1，這裡填 0；0 同時也是紅色 id，因此不能用 `class_ids==0` 找背景，也不能用 `class_ids!=-1` 找正格。兩種做法都會選錯。判斷正負格一律讀 positive。

## 原始標註如何接到七個輸出

`build_targets` 把逐圖的 `{boxes,labels}` 清單轉成固定形狀的格子答案。下方 `scene` 是紅、藍兩框，`empty` 是空圖；後三個參數為每邊格數、輸入圖邊長（pixel）、類別數。`target` 現在改指轉換後的四欄位，而非原始標註。

```python
# 示意：完整程式已匯入 build_targets，並建立 scene、empty
target = build_targets([scene, empty], grid_size=4, image_size=64, num_classes=2)
pos = target['positive']
red = target['box'][0,1,1]  # [0,.25,.25,.25]
# pred 是稍後才會使用的模型輸出：[2,4,4,7]
# pred[..., :4][pos] → [2,4]；target['box'][pos] → [2,4]
# 同一個 mask、同一排序，兩邊第 i 列才是同一格
```

head 每格七個原始輸出（logits）依序為 `tx,ty,tw,th,obj,class0,class1`。框的四項先經 sigmoid，成為 0～1 的比例，再與 box 比 MSE（均方誤差）；obj 與全部 32 格的 objectness 比 BCE（binary cross entropy／二元交叉熵）；兩個類別 logits 與正格 class_ids 比交叉熵，回答紅或藍。背景由 objectness 處理，不新增第三類。

`...` 索引保留前三軸，`:4` 取框四項；`[pos]` 只留下正格，按 `(b,gy,gx)` 由小到大排成 `[2,4]`。類別同樣取 `pred[...,5:][pos]`，shape `[2,2]`，對照 class_ids 的 `[0,1]`。objectness 用 `pred[...,4]`，不用 mask。這樣固定輸出就能接變動框數，也避免 30 個負格占位值混進框與類別的平均。

??? note "格內位置恰好是 0"

    紅、藍框的格內 x 都為 0。sigmoid 的數學輸出不含 0、1，有限 logit 只能逼近它們，不能恰好命中。本章訓練資料的中心不落格線，避免端點；後來的 YOLO 會調整 xy 範圍，YOLOv4 稱相關問題為 grid sensitivity。

## 一格放不下兩個框時，直接拒絕

把另一個框 `[10,14,26,30]` 與紅框放在同張圖，兩者中心為 (18,22)、(16,20)，floor 後都屬於 (gy=1,gx=1)。這格只有一組框輸出，不能同時寫兩份答案，即使兩個都是紅色也一樣。`build_targets` 因此報 `ValueError: same-cell collision: image 0, cell (1,1)`，避免後寫的框悄悄覆蓋前一個。

增加類別只增加類別 logits：兩類的 7 項變成三類的 8 項，框仍只有一組。更多類別不能解決同格容量；NMS 只能刪重複預測，也補不回第二份框答案。ShapeDataset 先讓物件分居不同格，便於驗證流程；真實資料仍可能碰撞，後續 anchor slot 與多尺度才研究增加候選的方式。

標註壞掉則應在分配前修正。例如中心 (64,64) 會算出索引 (4,4)，超出 0～3；但圖內正面積框的中心必小於 64，所以合法標註不會發生這件事。程式先拒絕越界或零面積框，不用 clamp 把壞答案硬塞進最後一格。

## 執行，核對位置與容量

在 Colab 執行頁首 notebook，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-targets.py`。程式先斷言 box shape、兩個正格的四項數值與類別，以及空圖沒有正格；再用 try/except 接住上面的碰撞，印出 `same-cell collision rejected`。若碰撞未被拒絕，會用 `Two objects silently overwrote one slot` 報錯停下。

接著的 `positive indices (b,y,x)` 應是 `[[0,1,1],[0,2,3]]`，紅、藍 target 分別為 `[0,.25,.25,.25]`、`[0,.75,.25,.25]`，最後為 `positive/negative counts 2 30`。2 是 print 中的固定數，先前斷言已核實；30 由 mask 數出。這比看 loss 是否下降更直接：它核對的是同一個物件到底交給了哪個位置。

## 自主練習

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



現在每格要學的答案已經固定。下一節用全零 logits 手算 loss，確認正格、負格與空圖分別收到正確訊號。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-targets.json)

??? example "展開本次實際輸出"

    ```text
    same-cell collision rejected
    positive indices (b,y,x) [[0, 1, 1], [0, 2, 3]]
    red target [0.0, 0.25, 0.25, 0.25]
    blue target [0.0, 0.75, 0.25, 0.25]
    positive/negative counts 2 30
    ```

<!-- curriculum-evidence:end -->
