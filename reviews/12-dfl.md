# 第 12 課陌生讀者審閱

只讀 `docs/lessons/12-dfl.md`、`lesson_cases/12-dfl.py`；本課未引用圖片。以具備基本 PyTorch/CNN 知識、較不熟大學數學的讀者角度，連續 target、相鄰 bin 權重、softmax 與期待值、初始梯度、stride 換算、範圍和 raw head 成本都能理解。兩份同期待值分布的例子有效說明額外監督的作用。

執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/12-dfl.py` 成功：初始期待值 `1.50`、loss `1.386294`、梯度 `[.25,-.5,0,.25]`，學得期待值 `1.2500` 格、`9.9999` 像素。正文數字與實際程式一致。發現以下兩項需要修改。

1. **教材中的 loss 程式片段在批次大於 1 時會算錯。** `docs/lessons/12-dfl.md:19–20` 的兩次 `cross_entropy` 未設定 `reduction='none'`，預設先把整批 loss 平均，再套每筆 target 權重；這與 `lesson_cases/12-dfl.py:13–14` 的正確實作不同。單筆範例掩蓋了問題。用 logits `[[3,0,0,0],[0,0,3,0]]`、target `[1.25,2.6]`，教材片段得到 `2.276706`，正確 DFL 是 `2.539206`。具體修改：在片段兩次 `cross_entropy` 都加上 `reduction='none'`，並用一句話解釋「先對每筆樣本套自己的左右權重，再平均」。

2. **「把 DFL 當成分類 loss」容易與前面的實作形成矛盾。** `docs/lessons/12-dfl.md:40` 把這列為錯誤，但第 13、19–20 行又正是加權 cross entropy。陌生讀者會不清楚到底哪一步不能視為分類。具體修改：改成「把距離 bin 當成物件類別，或將連續 target 四捨五入成單一 bin 後做 cross entropy」。在第 13 行 loss 說明後補一句：「這裡沿用 cross entropy 的計算方式，但 bins 代表距離位置，兩個相鄰位置共同構成監督；不是 detector 的物件分類分支。」

沒有發現其他實質數值錯誤。歷史來源和不確定性的限制各只占短句，未嚴重干擾主線，無須為了縮短而刪除。


## 作者修訂紀錄（2026-10-02）

頁面兩份cross_entropy已改reduction=none，先逐樣本加权再平均；明確區分距離bin的加權CE和物件類別分類loss。正式案例原本已採正確reduction，CPU驗證初始梯度[.25,-.5,0,.25]與期待距離1.2500格。
