# Grid MiniYOLO：把輸出接回圖片

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/07-inference.ipynb){ .md-button }

前置：[targets](07-targets.md)、[三步訓練](07-training.md)、[NMS](06-decode-nms.md)。本節把 dataset→model→decode→score filtering→分類別 NMS 接成可核對的推論介面。故障可能出現在每一段，因此先讓人工已知 logits 走同一個 decoder，再讓小模型輸出走它。

歷史背景是 [YOLOv1](https://arxiv.org/abs/1506.02640) 的 grid 框與推論；本章採用自己的 7 維 head、sigmoid wh 和 `sigmoid(obj)×softmax(class)` 分數，不使用原論文的 confidence 定義。本節沿用既有 head 定義，聚焦推論；人工 fixture 是答案檢查，不是模型學會的證據。

本節共用decode也把預測框裁切到输入畫布邊界，並排除非正面積框；第6章獨立decode不含這步。它處理推論越界，不代表可以clamp非法GT來掩飾標註錯誤。

## 從格內值回到 pixel 框

`raw[b,y,x,:]` 的前三軸是圖片、格子列 y、格子欄 x；B 是 batch 圖片數，各索引從0起算。最後七項依序為 `[tx,ty,tw,th,obj_logit,red_logit,blue_logit]`，logit 是尚未轉成機率的實數。class id `0=紅、1=藍`，softmax只沿最後兩個類別通道計算。

`sigmoid(tx,ty)` 是相對一個 cell 的格內中心 offset；`sigmoid(tw,th)` 是相對整張64×64輸入的寬、高比例。四項均無單位，但分母不同。中心公式是 `cx=(x+sigmoid(tx))/4×64`、`cy=(y+sigmoid(ty))/4×64`；最後框採輸入pixel的 `[x1,y1,x2,y2]` 邊界，x向右、y向下。

![固定紅／藍矩形的pixel框](../assets/diagrams/07-data.svg)

紅框的 cell=(x=1,y=1)，輸出前四維經 sigmoid 為 `[0,.25,.25,.25]` 時：

- 中心 x=`(1+0)/4×64=16`，y=`(1+.25)/4×64=20`。
- 寬高=`(.25,.25)×64=(16,16)`，因此 xyxy=`[8,12,24,28]`。
- 若 objectness=.8、紅類 softmax=.9，最終 score=.72。這個 score 用來排序與過濾，不能稱為 precision。

offset=0 是 sigmoid 的極限值，有限 logit 無法精確得到零。本節 fixture 使用 `logit(clamp(target,1e-4,1-1e-4))`，因此紅框 x 邊界有約 .0016 pixel 誤差，採 `.02 pixel` 容差驗證。這個細節也說明邊界 target 並不必然能用有限參數精確擬合。

## 同一介面支援零／一／多物件

模型永遠輸出 `[B,4,4,7]`，decoder 每格選最大機率類別，再產生最多 16 個候選。過濾後的每張結果則是 `boxes[N,4]、scores[N]、labels[N]`；N 可以為 0。案例把三張圖作為一個 batch，人工 fixture 分別放 2、1、0 個高分位置。第二張使用獨立的one_image並清除藍色畫素，確實只剩紅物件；不能只刪藍框標註卻留下藍畫素。

```python
model.eval()
with torch.inference_mode():
    raw = model(images)  # [3,4,4,7]
    result = decode_grid(raw, image_size=64,
                         score_threshold=.25, nms_iou=.5)
```

`eval()` 改變模型模式，`inference_mode()` 停止建立梯度；兩者用途不同。decoder 在每個類別內以 score 排序做 NMS，兩個不同類別即使重疊也不互相壓掉。score threshold=.25 決定留下哪些候選；NMS IoU=.5 決定哪些同類重複候選被刪。二者都不是 held-out matching 的 IoU 門檻。

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/07-inference.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-inference.py`。先列出未訓練模型的候選數，只證明介面相容；再列出 `artificial known-logit fixture counts [2,1,0]`，每個人工框還原到 .02 pixel 以內。另檢查 batch=1 仍回傳長度 1 的 list，不會因 `squeeze()` 丟失 batch 軸。最後額外在相鄰cell放一個低一點分數、幾乎重疊的紅候選；NMS IoU=1保留2框，改.5剩1框，顯示同類去重實際發生。這個人工對照也不代表模型學會。

## 圖片尺寸與顯示的最後一段

本節輸入和原圖都是 64×64，所以 pixel 框可直接疊回同一張圖。若使用非正方形照片，模型框仍在 padded 輸入座標，必須用第 8 節的 `undo_letterbox` 回原圖後再畫。畫圖時不把 normalized wh 當 pixel，也不把 xyxy 當 `(x,y,w,h)`。人工紅／藍場景可對照[資料頁的靜態圖](07-data.md)。

收益是訓練與展示共用一個 decode，空結果與 batch 都有固定契約；代價是固定 grid 仍可能漏同格物件，NMS 也不能補回沒有預測的框。提高 threshold 讓畫面更乾淨，同時可能降低 recall；必須用獨立評估判斷，不能憑畫框數量少就說進步。

常見錯誤是把訓練模式當推論、對 class 各自 sigmoid 卻仍宣稱互斥 softmax、NMS 跨類別、空結果索引 `[0]` 崩潰。自主練習：score=.72，threshold 改成 .75 會發生什麼？答案：框在 NMS 前就被刪；提高 NMS IoU 無法救回它。下一節將把這些結果交給評估 matching。

主fixture的score接近1，不會在.75被刪。想動手驗證.72這一題，另用下面獨立fixture，不改主例：

```python
score_raw = torch.zeros(1,4,4,7)
score_raw[...,4] = -20
score_raw[0,1,1,:4] = torch.logit(torch.tensor([.5,.25,.25,.25]))
score_raw[0,1,1,4] = torch.logit(torch.tensor(.8))
score_raw[0,1,1,5:] = torch.log(torch.tensor([.9,.1]))
low = decode_grid(score_raw,64,.70,.5)[0]
high = decode_grid(score_raw,64,.75,.5)[0]
assert len(low['boxes']) == 1 and len(high['boxes']) == 0
assert abs(low['scores'][0].item()-.72) < 1e-6
```

這裡改的是score門檻，NMS的IoU門檻仍是.5。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-inference.json)

??? example "展開本次實際輸出"

    ```text
    untrained model candidate counts (no quality claim) [0, 0, 0]
    artificial known-logit fixture counts [2, 1, 0]
    fixture red box [8.00160026550293, 12.0, 24.00160026550293, 28.0]
    same-class overlapping fixture NMS counts 2 -> 1
    ```

<!-- curriculum-evidence:end -->
