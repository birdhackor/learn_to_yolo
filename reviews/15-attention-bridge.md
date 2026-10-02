# 第 15 課閱讀與正確性審查

閱讀範圍：`docs/lessons/15-attention-bridge.md`、`lesson_cases/15-attention-bridge.py`、`docs/assets/diagrams/15-attention-bridge.svg`。以只懂基本 Python／PyTorch／CNN、線代生疏的讀者角度檢查；未讀其他課頁，未修改課文、程式或圖片，忽略 Colab 佔位。

主線能讀懂：第 13–15 行直接列出兩個 channel 與四個空間 token，順序清楚；第 21–31 行把第一個 query 的內積、softmax 和兩個 channel 的加權和接起來；第 35 行及程式第 19 行說明反向轉置與還原。圖的五個 shape、接收列／來源欄及 softmax 軸與程式一致，沒有看到圖示錯誤。

## 1. 修正「漏掉 transpose 會得到 C×C」的錯誤說法

- 位置：`docs/lessons/15-attention-bridge.md:45`；對照 `lesson_cases/15-attention-bridge.py:17`。
- 問題：本例 Q、K 都是 `[1,4,2]`。刪掉 `k.transpose(-2,-1)` 後，`q @ k` 嘗試讓 `4×2` 乘 `4×2`，會直接維度不匹配，不會得到 `C×C`。這會讓正在靠 shape 判斷矩陣乘法的讀者形成錯誤規則。
- 具體修法：將「漏掉transpose導致得到C×C而不是N×N」改成「漏掉 K 的 transpose 會讓本例矩陣乘法維度不匹配；若將 channel 當作 token，則可能錯算成 channel 之間的 C×C 關係」。也可只保留前半句，避免與同句已有的「把channel當token」重複。
- 驗證：執行 `q @ k` 得到 `Expected size ... [1, 2] but got: [1, 4]`。

## 2. 讓自主練習能在提供的程式內完成

- 位置：`docs/lessons/15-attention-bridge.md:47`；`lesson_cases/15-attention-bridge.py:11`、`:20`、`:22`、`:24`、`:35`。
- 問題：讀者照練習將右下位置改成 `[2,0]`，程式會先在第 20 行原始 tokens 的固定 assert 失敗；後面的第一列權重和輸出 assert 也仍是原例答案。如果讀者改在 SGD step 後新增練習，QKV 又已不再是 identity。課文未告訴初學者在哪裡改、如何保留原例檢查，會妨礙用程式驗算。
- 具體修法：在第 28 行後、optimizer step 前增加獨立練習段，複製 `tokens.detach().clone()`、設定 `exercise_tokens[0,3]=torch.tensor([2.,0.])`，再經同一個仍為 identity 的 `qkv` 重算 Q/K/V、softmax 與加權和；保留原例 assert，另對練習結果核對。課文第 47 行指明執行這一段。若只要求紙筆練習，也應明說原程式的 assert 固定驗證原例，不能直接修改輸入後沿用。
- 核對答案：第一列權重約 `[0.2212,0.1091,0.2212,0.4486]`；第一個 output 約 `[1.3395,0.3302]`。第四位置權重最大、第一 channel 增加，課文的數值方向正確。

## 驗證結果與保留內容

`PYTHONPATH=. .venv-model/bin/python lesson_cases/15-attention-bridge.py` 成功：tokens 為 `[[1,0],[0,1],[1,1],[0,0]]`，第一列權重約 `[0.3349,0.1651,0.3349,0.1651]`，第一個 output 約 `[0.6698,0.5]`；shape 路徑為 `(1,2,2,2) → (1,4,2) → (1,4,4) → (1,2,2,2)`，reconstruction loss `0.1795`，SGD 更新成功。額外核對 Q、K、V 三份投影各自都有非零梯度。

第 41 行的 N² 成本例子正確（4,096 → 65,536，16 倍）；第 43、45 行對位置資訊與權重解釋的界線也適當。除上述兩處外，沒有需要刪除的關鍵內容。


## 作者修訂紀錄（2026-10-02）

已修漏K transpose會矩陣維度不匹配的說明。新增SGD之前的獨立tokens副本練習，保留原例assert與identity投影；CPU核對第四token[2,0]時權重[.2212,.1091,.2212,.4486]、輸出[1.3395,.3302]。Q/K/V三份投影各自有非零梯度，主案例通過。
