# 16-inference-head 閱讀審查

以陌生專案、只具基礎 Python／PyTorch／CNN 的讀者為準，僅閱讀本課與 `lesson_cases/16-inference-head.py`；本課未引用專屬圖。Colab 佔位不列問題。未修改教材與範例。

實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/16-inference-head.py` 成功：train keys 為 `many`／`one`，輸出為 `[2,3,4]`／`[2,3]`／`[2,3]`，參數為 332／278，raw 比對與移除 many 參數的 assert 均通過。官方與玩具模型的界線、移除輔助 head 的目的、sigmoid 無 objectness、top-k 不等於 IoU 去重，已有清楚提醒；以下三項仍會妨礙獨立理解。

1. **缺少為何保留 one、它與 NMS-free 如何相連的解釋。** 位置：`docs/lessons/16-inference-head.md:5–7`、`:20`、`:40`；`lesson_cases/16-inference-head.py:12–17`、`:46`。目前只命名 many／one，宣布官方推論選 one，再說沒有 NMS。初學者無法知道這兩個名字指的是「每個標註物件對應多少個正樣本」而不是輸出框數，也無法理解為何刪 many 後仍能推論。建議在「兩種介面不能混讀」前補一小段：one-to-many 為一個物件分配多個正候選，以提供較密集的訓練訊號；one-to-one 為一個物件分配一個正候選，訓練推論分支減少重複預測；官方部署使用後者並以 top-k 選輸出，因此不需靠 NMS 做兩框 IoU 去重。接著明說本例只是兩個獨立卷積 head，平方 loss 沒有實作任一種 assignment，`detach()` 只切斷 one loss 回傳 backbone 的梯度，並不能讓隨機框自動變成一物件一框。可補一張小型箭頭圖顯示 features → many／detach → one，部署只留下 backbone → one。

2. **DFL-free 與解碼尚欠可核對的數值橋接。** 位置：`docs/lessons/16-inference-head.md:11–18`、`:38`、`:50`；`lesson_cases/16-inference-head.py:27–35`。`distance logits`、`reg_max1`、softplus 與 stride 在同段出現，但沒定義四 channel 順序，也未說 DFL-free 究竟移除了什麼；僅查 shape 的 top-5 練習不能發現左右／上下接反或距離漏乘 stride。建議先定義本例 raw 六個 channel 為 `[l,t,r,b,class0,class1]`，前四項經 softplus 後是「特徵格單位的四邊距離」。再用一句說明 DFL 常見路徑為每邊預測多個 bin 的分布並取期望，DFL-free 路徑每邊直接預測一個值；本例 softplus 正值限制是教學加工，不能照搬為官方 YOLO26 行為。補一個手算與 assert 的練習：中心 `(24,40)`、stride 16、變換後距離 `[1,0.5,2,1.5]`，應解碼為 xyxy `[8,32,56,64]`；類別 logits `[0,ln(3)]` 應得分 `[0.5,0.75]`，取 label 1、score 0.75。如此能同時確認座標單位、sigmoid 和單候選單標籤的選擇，而不只改輸出維度。

3. **介面表混合中間張量與真正回傳值。** 位置：`docs/lessons/16-inference-head.md:11–18`；`lesson_cases/16-inference-head.py:27–35`、`:51–54`。正文說 deploy 回傳三個 tensor，緊接表格卻列出 raw 候選與全部 decoded boxes，容易使读者以為 `deploy(image)` 也能取得 `[2,16,6]` 與 `[2,16,4]`。建議將表格加「取得方式／是否回傳」欄，清楚標示 raw 与 decoded boxes 是 `forward` 內部中間值，真正介面只有 `(top_boxes, top_scores, top_labels)`；或分成兩張短表。raw 一致性檢查是在解碼前另外呼叫 `deploy.head(deploy.backbone(image))`，應在表下明說，避免讀者把它與已 flatten 的 `[B,16,6]` 或最終 top-k 輸出混讀。

無需刪除整段內容；保留現有玩具模型限制與部署檢查，補上述定義、數值驗證與介面標示即可。


## 作者修訂紀錄（2026-10-02）

補many/one監督關係、官方one分支與NMS-free的因果條件，明示本例平方loss沒有assignment且未學唯一性。表格新增是否回傳欄，raw/decoded中間值與三個top-k回傳值分開。新增手算與CPU assert：點(24,40)、stride16、距離[1,.5,2,1.5]→[8,32,56,64]；類別logits[0,ln3]→[.5,.75]、label1。主案例raw等價與332/278參數檢查通過。
