# 第 07 課 training 閱讀檢查

僅閱讀 `docs/lessons/07-training.md` 與 `lesson_cases/07-training.py`，以會基本 PyTorch、尚未認識此專案的讀者角度檢查。未讀其他教材或大綱，未修改教材。略過 Colab 佔位與待 root 補入的 `ROOT_VERIFIED_SYNTHETIC_RESULT`。

## 執行結果

執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/07-training.py`，exit code 0。

| step | total | box | objectness | classification | positive | negative |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9817 | 0.0090 | 0.2525 | 0.6842 | 0.1202 | 0.1202 |
| 1 | 0.9248 | 0.0091 | 0.2508 | 0.6283 | 0.1190 | 0.1181 |
| 2 | 0.8323 | 0.0094 | 0.2491 | 0.5362 | 0.1200 | 0.1169 |

最後輸出為 `3 real CPU optimizer steps; parameters changed; no generalization claim`。

## 需要修改

1. **有限梯度的宣稱與實際斷言不一致。** 正文第 32 行把「有限梯度」列為通過條件；案例第 23 行只檢查 total 有限，第 25–26 行只檢查梯度絕對值總和大於零。`inf > 0` 仍會通過，所以有限 loss 並不足以實現該條件。建議在 `optimizer.step()` 前收集非空梯度並逐一檢查，再保留非零總量檢查：

   ```python
   grads = [p.grad for p in model.parameters() if p.grad is not None]
   assert grads and all(torch.isfinite(g).all().item() for g in grads)
   grad = sum(g.abs().sum().item() for g in grads)
   assert grad > 0
   ```

   正文第 24 行相應改為「每步檢查 total 與各非空梯度有限，且梯度絕對值總和大於零」。

2. **輸出與正負格 mask 的 shape 尚缺一段本地說明。** 正文第 9 行提供影像 shape，第 16 行提供 `[4,4,4,7]`，但未解釋最後的 7，或案例第 29–31 行為何選 `[..., 4]` 並以 `target['positive']` 索引。建議在第 9 行後補一句：「輸出 `[B,G,G,5+C]` 在本例是 `[4,4,4,7]`：依序為 batch、格列、格欄與每格的四個框參數、一個 objectness logit、兩個類別 logits；`target['positive']` 是 `[4,4,4]` 的布林 mask。4 張圖各有 16 格，因此共 64 格，其中 4 格負責物件、其餘 60 格為背景。」並在第 28 行補明 `labels=0` 是有效的物件類別編號，背景由正格 mask 表達。這能讓未讀過專案的 PyTorch 讀者接上案例的張量索引。

## 讀者能理解且應保留的內容

- 正文第 22 行把清梯度、forward、backward 與 step 的責任說清楚；第 40 行刪除 `step()` 的練習也能驗證兩者差別。
- 第 24 行明確說明分數取自更新前的 prediction，避免把 `step()` 後列印誤讀為更新後結果。
- 第 28–30 行提供了可執行的背景診斷順序：檢查資料與正格、人工梯度、參數更新，再追查分項 loss 與解碼框。沒有把 total 下降當作偵測成功。
- 第 36 行已把少量 overfit 操作化為「固定這 4 張，訓練到定位與分類明顯正確」，再到獨立圖片評估。可在第一次出現 overfit 時補上「先確認模型能記住這幾張固定訓練圖」，但不是實質阻礙。
- 第 5、36、38 行反覆限制三步實驗的結論；實際三步正格分數接近 0.12，也未被包裝成辨識效果。沒有發現需要刪除的實質內容。

上述兩處補齊後，三步案例的教學目的與實際行為一致。頁末尚未加入的 160 步結果不在本次評價範圍內。


## 作者修訂與驗證（2026-10-02）

補7通道、圖片-y-x軸、positive及objectness索引；case新增每個參數梯度全有限檢查，保留真實三步與參數變動assertion。root補入的160步合成資料結果、曲線與預測圖完整保留。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/07-training.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。
