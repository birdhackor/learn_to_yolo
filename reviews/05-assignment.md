# 第 05 課陌生讀者審閱

審閱範圍：只讀 `docs/lessons/05-assignment.md`、`lesson_cases/05-assignment.py` 與 `docs/assets/diagrams/05-assignment.svg`，以已有基礎 PyTorch/CNN、未讀其他課的讀者為準。Colab 佔位略過。結論：核心概念與數值一致；有 1 項重要補充，沒有必須刪除的內容。

## 1. 補上 assignment 與 NMS 的時機及用途區別

- 位置：`docs/lessons/05-assignment.md:30–55`，建議接在第 55 行容量限制後。
- 問題：頁面完整說明訓練時的中心責任、正負格與 loss mask，但完全沒有提到 NMS。只讀本頁的讀者無法判斷「多個物件搶一個 slot」和「多個預測框重複指出同一物件」是不同問題，也容易以為 NMS 可以救回同 cell 被容量限制丟掉的標註。
- 具體修法：新增短段即可，例如：「Assignment 在訓練時依標註決定哪些候選負責哪個物件，以及哪些候選學背景。推論時，模型仍先輸出固定的候選框；NMS 則依預測分數與框的重疊程度刪除重複預測。NMS 不會替訓練分配標註，也不能把一個 slot 拆成兩個，因此無法解決本課的同 cell 容量衝突。」不必在本課展開 NMS 演算法或新增程式。

## 已核對且沒有問題

- `05-assignment.md:5,11–13,55`：每格一個 slot、固定 `[B,S,S,5+C]`、七個輸出值、slot 與 class 軸獨立，都能接到程式第 51–58 行的卷積與 `permute`；兩類沒有各自專用框。
- `05-assignment.md:21–28,32–49` 與 SVG：紅／藍中心、row/col、偏移、整圖寬高、正 2／負 14／ignore 0 一致；背景不算框與類別 loss、仍算 objectness loss，負格填 0／−1 與共享權重梯度的說明清楚。圖已渲染查看，文字清楚且沒有截字。
- `05-assignment.py:18–19,42–49`：同格同類明確報錯，正文第 53 行所述限制沒有被程式默默覆蓋。
- 實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/05-assignment.py` 成功；shape、batch 正 2／負 30／ignore 0、碰撞錯誤與第 59 行三項 loss 的兩步數字完全相符。
- 另用同一 `build` 核對第 61 行練習：左移藍框落在 `[row2,col1]`，target 仍為 `[0.75,0.75,0.25,0.25]`；`S=8` 時正格為 `[1,1]`、`[5,5]`，target 為 `[0.5,0.5,0.25,0.25]`，負格 62，符合正文。


## 作者修訂

已補assignment在訓練生成target、NMS在推論解碼後去重的時機；分清兩物件搶一slot與多slot重複同物件，說明NMS不能补回容量不足丟失的物件，並補常見錯誤。原CPU案例的target、碰撞及mask梯度檢查通過。
