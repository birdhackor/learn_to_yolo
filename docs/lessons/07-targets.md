# Grid MiniYOLO：把框變成監督

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/07-targets.ipynb){ .md-button }

前置：[資料契約](07-data.md)。本節要回答「一張圖有兩個框，為什麼模型有 16 個輸出位置，誰應該學什麼」。模型尚未參與運算；我們先用人工答案確認 assignment（責任分配：哪個輸出位置負責哪個真實框）。這一步決定訓練監督，與推論 NMS（刪除重複預測框）、評估 matching（配對預測框與真實框）是三個不同程式。

歷史來源仍是 [YOLOv1](https://arxiv.org/abs/1506.02640) 的中心落格概念。本章簡化為每格一個框，兩類 softmax；不實作原版每格多框與依 IoU 選責任框。本次實驗保留 4×4 基線，不加入 anchor。

## 由 pixel 一步一步走到 cell

64×64 圖分成 4×4 格，每格寬、高各 16 pixel。原始框是 pixel 單位的 `[x1,y1,x2,y2]`，x 向右、y 向下。沿用紅框 `[8,12,24,28]`：中心 `(16,20)`，寬高 `(16,16)`。中心除以格寬得到 `(1,1.25)`，取 floor 決定 `(gx=1,gy=1)`。剩下的小數是格內位置 `(0,.25)`；寬高除以整張圖 64，得到 `(.25,.25)`。

因此 `box[b,gy,gx] = [0,.25,.25,.25]`，四項皆為無單位比例，xy 相對格子、wh 相對整張圖。x=16 正好在第二欄的左邊界，歸第二欄，不是第一欄的右邊界。藍框 `[40,36,56,52]` 中心 `(48,44)`，落 `(gx=3,gy=2)`，target 是 `[0,.75,.25,.25]`。索引先 y 再 x，向量內先 x 再 y；兩個順序同時存在，必須分清。

`B` 是 batch 圖片數，`b` 是其中一張的索引；前三軸依序是圖片、格子列 y、格子欄 x。本例 B=2。物件中心落入的格子是正格，其餘為負格；objectness 目標表示這格是否分配物件，positive 是布林 mask（遮罩），用來選出正格。

| target 欄位 | 本例 shape | 使用方式 |
| --- | --- | --- |
| box | `[2,4,4,4]` | `[B,Gy,Gx,4]`，最後軸依序格內 x/y、全圖寬/高比例 |
| objectness | `[2,4,4]` | 正格 1，其餘 0 |
| positive | `[2,4,4]` bool | 決定哪些位置計框與類別 loss |
| class_ids | `[2,4,4]` long | 只在 positive=True 時讀取 |

第一張 2 個正格、14 個負格；第二張空圖有 16 個負格。整個 batch 正格 2、負格 30。這個版本沒有 ignore 狀態，所有非正格都學背景。負格的 box/class 欄位即使填了數字也沒有語義，不能把它們拿來平均回歸或算分類 loss。

## Head 與 mask 如何接起來

head 的輸出 `[B,4,4,7]` 最後一軸為 `tx,ty,tw,th,obj,class0,class1`。前四個是 logits，經 sigmoid 才與 `[0,1]` 的 target 比較；objectness logit 用二元交叉熵（BCE）；class logits 在正格使用類別交叉熵。背景不是第三個 class，objectness 已承擔「有沒有物件」。

```python
target = build_targets([scene, empty], grid_size=4, image_size=64)
pos = target['positive']
red = target['box'][0, 1, 1]  # [0,.25,.25,.25]
# pred[..., :4][pos] 只取負責物件的格子
```

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/07-targets.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-targets.py`。應列出正格 `[[0,1,1],[0,2,3]]`、兩個 target 與 `positive/negative counts 2 30`。再把第二個框改成 `[10,14,26,30]`：它的中心也在同一格，函式必須拋 ValueError，不能悄悄覆蓋第一個框。

## 得到的能力與留下的限制

固定格子讓輸出數量與 mask 很簡單，也讓空圖片自然成立；成本是格子容量：同格兩個物件，即使類別相同也只能存一個框。增加 class 數不會增加框槽。資料生成器暫時避免碰撞是為了先接通管線，不能宣稱世界上的物件不會碰撞。第 9 章 anchor slot 與第 10 章多尺度會分別討論容量及解析度。

常見錯誤：把 normalized wh 誤除以 16；以左上角而非中心落格；用 class id=0 判斷背景；把 `pred[:,gx,gy]` 寫反。逐項用紅框數字核對，遠比看 loss 是否變小可靠。

自主練習：`[4,4,12,12]` 的 target 是什麼？答案：中心 `(8,8)` 在 `(0,0)`，格內 xy 為 `(.5,.5)`，wh 為 `(.125,.125)`，只有該格的 positive 變 True。若中心恰好是 `(64,64)`，它已在輸入外邊界，應先檢查標註合法性，不能用 clamp 掩飾錯誤。
