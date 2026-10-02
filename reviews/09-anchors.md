# 第 09 課閱讀 review

以只懂基本 PyTorch/CNN、未讀其他教材的讀者視角檢查。anchor 是寬高先驗、槽位與類別獨立、尺寸 IoU 選責任槽，以及本章與完整 YOLOv2 的差異，整體說明清楚。沒有需要刪除的內容；本課沒有圖可核對。

執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/09-anchors.py` 通過所有 assertion：尺寸 IoU `[1, .25]`、best anchor `0`、log wh `[0, 0]`、positive/ignore/negative `1/1/30`；解碼結果 `[8.0016003, 12, 24.0016003, 28]`，與文字所述誤差一致。寬高練習答案正確。

## 實質問題

1. **中心的原始輸出與格內比例尚未明確分開。** `docs/lessons/09-anchors.md:9–11` 先用 `sigmoid(tx)` 解碼，接著說「tx/ty 對應格內比例 `(0,.25)`」，但沒有列出中心 encode，也沒有明確定義 `gx/gy`、`tx/ty`。讀者容易把 `(0,.25)` 直接當成 `tx/ty`，此時 `tx=0` 會解成格內比例 `.5`。建議在第 9–11 行定義 `gx/gy` 為整數格座標、`tx/ty` 為模型原始輸出，另以 `ox/oy` 表示格內比例；補上 `ox=cx/16-gx`、`tx=logit(clamp(ox, 1e-4, 1-1e-4))`（y 同理），並列出本例 `tx≈-9.21024`、`ty≈-1.09861`。接著說明 `lesson_cases/09-anchors.py:21` 的 logit 用於驗證 encode/decode，而第 40 行的中心 loss 比較 `sigmoid(raw tx/ty)` 與比例 target `(0,.25)`。這能讓編碼、解碼與訓練 target 接成同一條推導。


## 作者修訂與驗證（2026-10-02）

明分gx／gy格索引、ox／oy格內比例與tx／ty原始logits，列出logit encode式與−9.21024／−1.09861。case核對比例及encode值，並保留ignore零梯度與真SGD一步。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/09-anchors.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。
