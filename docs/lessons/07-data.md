# Grid MiniYOLO：先讓資料可以被檢查

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/07-data.ipynb){ .md-button }

前置：[多物件責任分配](05-assignment.md)、[座標轉換](04-coordinates.md)。本節要解決的問題是「模型喫到的畫素，是否仍與框描述同一個物件」。若藍色矩形的框移到紅色矩形上，訓練可以照常降低某個 loss，卻是在學錯的任務。

歷史機制：YOLOv1 把圖片切成 grid，由物件中心所在的格子負責預測；原文是 [You Only Look Once (2016)](https://arxiv.org/abs/1506.02640)。本章簡化：64×64 的兩色矩形、4×4 grid、每格一個框、兩個互斥類別；不是原論文的完整網路、框數、confidence 或 loss。本次實驗只確認資料契約，尚未訓練。

![兩色矩形與半開區間框](../assets/diagrams/07-data.svg)

## 同一個物件的第一站

固定場景有紅框 `[8,12,24,28]` 與藍框 `[40,36,56,52]`。框順序是 `x1,y1,x2,y2`，單位是輸入圖片的 pixel。採半開區間：紅色畫素的 x 是 8 到 23，y 是 12 到 27，因此寬、高都為 16；面積是 256，不加 1。紅色類別 id 是 0，藍色是 1，背景沒有第三個 class id。

| 欄位 | shape／型別 | 意義 |
| --- | --- | --- |
| image | `[3,64,64]` float32 | RGB、CHW、畫素範圍 `[0,1]` |
| boxes | `[N,4]` float32 | 當張圖片的 pixel xyxy |
| labels | `[N]` int64 | 每個框對應一個類別 |
| batch images | `[B,3,64,64]` | NCHW，可直接進 CNN |
| batch targets | 長度 B 的 list | 各張 N 可以不同 |

逐步核對：先找紅色左上畫素 `(x=8,y=12)`；它位於 `image[0,12,8]`。再找右下外側 `(24,28)`，它不屬於紅矩形。最後量通道總和：紅、藍各 256，綠色為 0。這比只看圖片 shape 多檢查了 channel、xy 順序、框邊界和畫素內容。

## 為何 targets 不直接 stack

第二張刻意放空圖片，框是 `torch.empty(0,4)`，類別是 `torch.empty(0,dtype=torch.long)`。這張仍會提供背景監督：在下一節建立 target 時，4×4 共 16 個候選位置的 objectness 都是 0，BCE 會要求它們學低 objectness，而框／類別不計 loss；不能刪除它，也不能塞入 `[0,0,0,0]` 假物件。第一張 N=2、第二張 N=0，直接 `torch.stack` 框會失敗，因此 `collate` 只疊影像，保留 targets list。

```python
image = torch.zeros(3, 64, 64)
image[0, 12:28, 8:24] = 1  # 紅，索引順序是 channel,y,x
image[2, 36:52, 40:56] = 1  # 藍
images, targets = collate([(image, target), (empty_image, empty)])
```

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/07-data.ipynb，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-data.py`。預期 `batch (2,3,64,64) counts [2,0]`、通道總和 `256.0 256.0`，再透過由 boxes／labels 重建彩色 mask 與原畫素逐值相等、以及 8 張生成資料的範圍與正面積檢查。把紅框 x1 改成 9 而不改畫素，此 assertion 應失敗。沒有下載、沒有 GPU，也沒有模型結果。

## 這個改動的收益與代價

可控圖形讓畫素、標註、空圖與多物件能先被逐項驗證，失敗時不用猜是真實資料太難，還是程式錯了。代價是資料過於乾淨：矩形、顏色、背景都固定，透過這裡不能證明能辨識照片。稍後接自己的圖片時，要保留同一套 RGB、座標與空圖契約，再處理原圖尺寸及來源切分。

常見錯誤是把 PIL 的 HWC 直接當 CHW、把 RGB 讀成 BGR、整數畫素未除以 255、框使用原圖座標而影像已 resize。先畫標註疊圖，再檢查數值；框的合法範圍檢查抓不到「合法但位置錯」的框。

自主練習：紅框改成 `[8,12,28,28]` 時，應修改哪個切片？通道總和是多少？答案：`image[0,12:28,8:28]`；寬 20、高 16，總和 320。只改標註或只改畫素會破壞契約。下一節將把原來 16×16 的紅框變成某一個 grid cell 的監督值。

固定scene用0／1畫素做精確核對；ShapeDataset則會變動尺寸／位置，背景有[0,.04)微量雜訊，紅／藍色塊用[.95,.10,.10]／[.10,.10,.95]。後續訓練用生成資料，不是反覆學本頁兩個256畫素的固定色塊。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-data.json)

??? example "展開本次實際輸出"

    ```text
    batch (2, 3, 64, 64) counts [2, 0]
    red/blue channel sums 256.0 256.0
    dataset contract checked: 8 images
    ```

<!-- curriculum-evidence:end -->
