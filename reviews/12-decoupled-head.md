# 第 12 課審閱：Decoupled head

審閱範圍：只讀 `docs/lessons/12-decoupled-head.md`、`lesson_cases/12-decoupled-head.py` 及教材引用的 Ultralytics `head.py` 固定 commit；本課沒有引用圖片。未讀其他教材或大綱，未改教材。Colab 佔位依要求略過。

## 整體判斷

具備基本 PyTorch／CNN 知識的讀者，可以從第 23、34 行理解主要結論：box loss 更新 box branch 與共享 backbone；classification loss 更新 class branch 與共享 backbone；兩者相加時，共享 backbone 收到兩份梯度之和。第 23 行也正確指出，分支分開並不消除 backbone 上的任務衝突。

動機（第 5、46 行）與參數代價（第 40 行）基本清楚，沒有把人工案例誇大成精度提升證據。執行結果、shape、參數計算及分支來源均相符。建議修正一處軸描述，補上輸出到 loss 的轉換，並減少未定義的術語；不需要重寫整課。

## 建議修改

### 1. 必修：channel 軸的說法會指錯位置

- 位置：教材第 19 行。
- 原文：`[B,C,H,W]` 的「第 2 個 4 是 channel」。
- 問題：在 `[2,4,4,4]` 中，依出現次序數「第 2 個 4」是高度；channel 是第 2 個元素、也就是第 1 個 4。這句會讓讀者把 shape 對錯軸。
- 建議替換：`[B,C,H,W]` 依序是批次大小、channel 數、高度、寬度；box 的 `[2,4,4,4]` 中，第 2 個元素 `4` 是四個距離 channel，後兩個 `4` 才是空間高度與寬度。

### 2. 必修：補上 raw box → 正距離 → box loss，並避免稱回歸值為 logits

- 位置：教材第 7、15、19 行；程式第 24、28、29 行。
- 問題：教材說「距離 logits」，但實際 `boxes` 是未限制正負的連續回歸原始值；`softplus(boxes)` 才是與 `1.5` 比較的正距離。新讀者直接對照表格與 `boxes`，會以為負值也是距離，或以為 box 與 class 都要使用 sigmoid。`ltrb` 也未首次解釋。
- 建議第 15 行改成：四個 channel 是左、上、右、下（`l,t,r,b`）距離的原始回歸值；經 `softplus` 轉為正距離。
- 在第 19 行後補兩行程式及一句話：

  ```python
  box_loss = F.smooth_l1_loss(F.softplus(boxes), distances)
  cls_loss = F.binary_cross_entropy_with_logits(logits, classes)
  ```

  `softplus(z)=log(1+exp(z))` 把 box 原始值轉成正距離，再用 Smooth L1 比較 target；分類 logit 是 sigmoid 前的原始分數，`binary_cross_entropy_with_logits` 直接接收它，計算 loss 前不要再手動 sigmoid。`ltrb` 距離是從每個特徵位置到框的左、上、右、下邊界，本例以特徵格為單位。
- 第 7 行「兩個 sigmoid 類別 logits」可改成「兩個獨立分類 logits（sigmoid 前的分數）」。

### 3. 建議：在第一次出現時界定 head／backbone，刪掉本課不用的名詞

- 位置：教材第 5、7、19、48 行。
- 第 5 行建議補：`backbone` 是共同抽取特徵的前段 CNN；`head` 是把特徵轉成預測的後段。本課一個 detection head 內有 box 與 class 兩個分支，兩個分支各有自己的卷積參數。
- 第 7 行的 DFL／TAL 沒有定義，也不参与本次實驗；建議把「不加入 DFL 或 TAL」改成「只示範分支與梯度，省去完整偵測器的距離分布預測和正樣本選擇」。若要保留縮寫，需在此首次給中文意義。
- 第 19 行的 assignment 建議改成「正樣本分配（決定哪些特徵位置負責哪些真實框）」；正樣本可直接寫為「有物體、需要學習框的位置」。
- 第 48 行末尾突然引入「YOLOv10 一對一分支」卻沒有解釋。建議刪掉這一句，保留前面的 `detach()` 注意事項，並補「`detach()` 會切斷這條分支回到 backbone 的梯度」。本課不需要額外的 YOLOv10 背景才能成立。

### 4. 小幅程式改善：把清成 None 的行為寫明

- 位置：教材第 29 行；程式第 30、35、41 行。
- 目前在本環境 PyTorch 2.9.1 中執行通過；`zero_grad()` 預設使用 `set_to_none=True`。本課用 `.grad is None` 當作梯度路徑證據，建議全部寫成 `model.zero_grad(set_to_none=True)`，讓教學行為明確，也避免讀者在不同版本或自行改成零 tensor 清除時，誤把斷言失敗當成分支不獨立。
- 程式第 32–38 行只檢查各分支第一層的 weight；架構實際沒有問題。若正文要強調整條分支所有參數，可改用 `all(p.grad is None for p in branch.parameters())`，並對作用中的分支使用 `all(p.grad is not None ...)`。否則可保持簡短案例，註明第一層 weight 是代表性檢查。

## 已核對且無問題

- 共享特徵：程式第 10、15 行為 `3→8`、padding 1 的 `3×3` 卷積，輸入 `[2,3,4,4]` 得到 `[2,8,4,4]`。分支最後投影得到 box `[2,4,4,4]`、class `[2,2,4,4]`，與教材第 11–17 行相符。
- 梯度：補充檢查了兩條分支與 backbone 的所有參數。單獨 box backward 時，整個 class branch 的 gradient 都是 `None`；單獨 class backward 時，整個 box branch 的 gradient 都是 `None`。兩次 backbone 的 weight 與 bias 都有梯度。原案例驗證總 backbone weight gradient 等於兩項之和，也確認 `optimizer.step()` 改變權重。
- 參數：backbone 224、box branch 620、class branch 602，共 1,446。兩個 `8→8` 卷積各 584，box 投影 36，class 投影 18，與第 40 行完全相符。直接 `8→6` 的 `1×1` head 含 bias 為 54，計算正確。若想讓成本比較更直觀，可補一句「本例兩個分支的 head 合計 1,222 個參數；加同一 backbone 的直接 head 模型合計 278」，但目前已明確標示 54 是 head 參數，不是算錯。
- 人工 targets、全正樣本及 Smooth L1 的限制已在第 19、52 行交代，未宣稱是真實 YOLOv8 的完整訓練流程。第 36 行對 cosine 的解讀與限制也恰當；實測 `-0.0073` 接近正交，不代表明顯衝突。
- 固定來源 commit 的 `head.py` 第 124–138 行確實分別定義 `cv2` 框分支與 `cv3` 類別分支；legacy 路徑第 128–129 行對應文中 YOLOv8 類設計。原程式的層數、輸出四個連續距離及 Smooth L1 是明示的教學簡化，沒有把完整源碼的結構冒充成此案例。第 52 行可把連結加上 `#L124-L138` 方便查找。
- 第 46 行關於額外卷積、activation 記憶體、容量與耗時需要實測的描述合理；沒有必要刪除。

## 執行證據

命令：`PYTHONPATH=. .venv-model/bin/python lesson_cases/12-decoupled-head.py`，退出碼 0。

```text
box / class shapes: (2, 4, 4, 4) (2, 2, 4, 4)
classification-only backward leaves box branch grad=None
shared-backbone gradient cosine: -0.0073
total backbone gradient = box gradient + class gradient: verified
parameters: 1446
```

補充檢查沒有修改檔案：raw box 值約 `[-0.0388, 0.3038]`，`softplus` 後距離約 `[0.6739, 0.8565]`；box loss 約 `0.266253`，class loss 約 `0.760256`。這些數值也顯示第 2 項說明值得補上。


## 作者修訂紀錄（2026-10-02）

已用axis1/2/3明確區分channel與空間，補backbone/head/assignment中文定義、raw→softplus→正ltrb距離的轉換，刪除未說明的DFL/TAL/v10插句。CPU案例已核對分支隔離、共享梯度相加及1,446參數。
