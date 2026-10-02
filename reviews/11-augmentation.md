# 11-augmentation：陌生讀者審閱

範圍：只讀 `docs/lessons/11-augmentation.md` 與 `lesson_cases/11-augmentation.py`。章節未引用圖片，沒有額外圖片可核對；未讀前置頁面，忽略 Colab 佔位。讀者假設有基本 Python、PyTorch 與 CNN 知識。

整體可讀且數值正確。半開區間的框邊界與像素索引分開說明，flip 的 `[40,12,56,28]`、crop 的 `[0,4,8,20]`、128/256=.5，以及 .5 保留、.6 捨棄都能跟著手算。YOLOv4／YOLOv5 的歷史來源和本章不實作 Mosaic 的界線清楚。裁剪後尺寸、重建 targets 和可見前景被刪標註的代價也已有交代。沒有需要刪除的主題，無須湊足三個問題。

## 1. 「空圖」案例其實仍包含紅色物件，需分清空標註與沒有物件的圖

位置：`lesson_cases/11-augmentation.py:42–45`；`docs/lessons/11-augmentation.md:34`。

程式註解寫「Empty images remain legal」，但傳入的是第 27 行畫了紅色物件的 `image`，只是另給 `(0,4)` 的 boxes。這確實能驗證空 tensor 的 shape 能通過函式，卻不是一組正確的「沒有物件的圖＋空標註」。只讀本頁的初學者可能把「沒有框也能跑」當成允許有前景卻無標註；本頁第 23 行恰好又提醒了這類未標註前景的風險。

具體修法：第 43 行前新增 `empty_image = torch.zeros_like(image)`，flip 使用 `empty_image`，crop 使用 flip 回傳的圖片及空 boxes。將註解改成「A no-object image keeps an empty [0,4] boxes tensor through both operations」，並在第 34 行補一句「空圖是沒有目標的圖片，boxes 仍保留 `[0,4]` 形狀」。若只想測空 boxes 的函式支援，則明寫这是 shape 邊界測試，不能當成完整資料樣本。

## 2. labels 同步目前只有文字指示，執行案例沒有實際示範

位置：`docs/lessons/11-augmentation.md:17,23,29`；`lesson_cases/11-augmentation.py:36–48`。

文章已正確說明 flip 不改類別、crop 必須用同一個 keep mask 過濾 labels；但可執行案例從頭沒有建立 labels，第 48 行只是把這項要求印出來。讀者看到了框移除的結果，卻看不到原本一個 label 在 .6 閾值下也變成空陣列，因而少了一個最容易直接照做的同步步驟。這不是目前函式的計算錯誤，而是範例的教學缺口。

具體修法：在第 28 行後建立 `labels = torch.tensor([0], dtype=torch.long)`；在 .5 與 .6 的 crop 呼叫後分別執行 `cropped_labels = labels[keep]`、`strict_labels = labels[keep_strict]`，核對結果是 `[0]` 與長度 0，並在輸出中顯示。空圖案例可同時建立 `(0,)` 的 long labels，讓讀者看到無物件仍保留正確資料形狀。保留目前回傳 keep 的介面即可，不必加入資料集或訓練流程。

## 執行驗證

在 repo 根目錄執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/11-augmentation.py`，exit code 0，所有現有 assertions 通過。輸出為 flip 框 `[[40.0,12.0,56.0,28.0]]`、雙 flip 還原、crop 框 `[[0.0,4.0,8.0,20.0]]`、可見面積 128/256=.5，以及 .6 移除物件。這些結果支持本章的幾何結論；程式沒有實作 Mosaic，也沒有訓練收益或 AP 結論，文章已如實交代。


## 作者修訂與驗證（2026-10-02）

空圖改全零像素，配[0,4]空框及Long[0]空標籤；flip後圖片及框一起送crop。實際使用keep篩labels，.5留下[0]、.6變空；數值／dtype／空图像素均有assertion並列印。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/11-augmentation.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。
