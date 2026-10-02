# 第 07 課審查：loss

審查角色：完全不熟悉此專案、Python／PyTorch／CNN 入門、久未接觸大學數學的讀者。只閱讀 `docs/lessons/07-loss.md` 與 `lesson_cases/07-loss.py`；未閱讀其他教材、大綱或 `miniyolo` 原始碼。Colab 佔位不列為問題。沒有修改教材。

## 結論與實測

loss 數值與程式一致，未發現計算錯誤。執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/07-loss.py` 成功：box `0.109375`、objectness `0.693147`、classification `0.693147`、total `1.933169`，正／負 objectness 梯度分別為 `-0.03125`、`0.03125`，空圖可以 backward。

另外透過執行公開函式（未閱讀其原始碼）核對：所有負格的框及 class 梯度均為 0；空圖的 total／objectness 均為 `0.693147`，框／class loss 與梯度均為 0；複製成兩張圖不改變三項 mean loss；class logits 共同加 10 不改變 CE。末尾練習答案正確。

對指定讀者，主要問題是符號與張量契約沒有交代，導致讀者能照抄數字，卻不易理解為何要取這些位置、這四個座標以及不同的平均分母。

## 實質問題與具體修改

### 1. shape、通道、符號與 Boolean mask 沒有定義

位置：教材第 9、16、19–23 行；程式第 11–13 行。

`[1,4,4,7]`、`B`、`S`、`Npositive`、`pred[..., :4]`、`pred[..., 5:][pos]` 都直接出現。初學者不知道哪一軸是圖片／格子／通道，七個值是什麼，也不知道 Boolean mask 會把三個前導維度選成「正格清單」。`build_targets([scene], 4, 64, 2)` 的三個數字用途同樣未說明。

建議在第 9 行前補一小段與 shape 表：本例圖片大小為 64×64、grid 為 4×4、類別數為 2；`B` 是 batch 圖片數、`S` 是每邊格數、`C` 是類別數、`Npositive = pos.sum()` 是整批正格數。每格輸出四個框 logit、一個 objectness logit、`C` 個 class logits。logit 是尚未轉成機率的實數；sigmoid 將一個值轉成 0 到 1，softmax 將多個類別值轉成總和為 1 的機率。

| 名稱 | 本例 shape／dtype | 經正格 mask 後 |
| --- | --- | --- |
| prediction | `[B,S,S,5+C] = [1,4,4,7]`，浮點數 | 框為 `[Npositive,4]`；類別為 `[Npositive,C]` |
| `target['positive']` | `[B,S,S]`，bool | `True` 的格才提供框與類別監督 |
| `target['box']` | `[B,S,S,4]`，浮點數 | `[Npositive,4]` |
| `target['objectness']` | `[B,S,S]`，浮點數 0／1 | 不取 mask，全格計算 |
| `target['class_ids']` | `[B,S,S]`，整數 `torch.long` | `[Npositive]` |

再以一句話解釋 `...` 是「保留前面的 batch、y、x 三個軸」，`:4` 取通道 0–3，通道 4 是 objectness，`5:` 取類別通道；本例 mask 只有 `[0,1,1]` 為 True。標題「七個零 logits」也宜改成「每格七個零 logits」，因為整張 prediction 有 16×7 個值。

### 2. 框 target 的四個分量與正規化單位不明

位置：教材第 9、11、16 行；程式第 10–14 行。

`[8,12,24,28] → [0,.25,.25,.25]` 是手算的起點，但這兩組數字各自的順序與轉換沒有寫出來。尤其第一個 0 不代表物件位於圖片最左側；前兩個分量是相對格子的中心偏移，後兩個分量是相對圖片的寬高。「座標已正規化」不足以讓讀者分清這兩種分母。

建議在第 9 行後完整算一次：原框為 `[x1,y1,x2,y2]`（pixel），中心 `(16,20)`、寬高 `(16,16)`；每格邊長 `64/4=16`，中心落在 `(y=1,x=1)`。框 target 的順序為 `[tx,ty,w/W,h/H]`，所以 `tx=16/16−1=0`、`ty=20/16−1=.25`、`w/W=h/H=16/64=.25`。四個框預測 logit 都是 0，sigmoid 後得到 `[.5,.5,.5,.5]`，誤差向量為 `[.5,.25,.25,.25]`，再平方、加總、除以 4，得到 `.109375`。

第 16 行的 pixel 比較宜跟著補「中心偏移除以格子邊長，寬高除以圖片邊長」，避免讀者誤以為四個分量都除以 64。平均與權重說明本身正確。

### 3. loss 片段沒有展示歸約設定和空正格處理

位置：教材第 18–24、30、36 行。

片段使用未定義的 `mse`、`bce_with_logits`、`cross_entropy`，也沒有明示 `reduction='mean'`。讀者無法判斷這是可執行 PyTorch 寫法或示意寫法。更重要的是，片段對空 mask 仍直接求 MSE／CE，與第 30 行宣稱的空圖處理不完整對應。

建議以 `import torch.nn.functional as F`、`F.mse_loss(..., reduction='mean')`、`F.binary_cross_entropy_with_logits(..., reduction='mean')`、`F.cross_entropy(..., reduction='mean')` 呈現，並在框／類別計算前加入 `if pos.any(): ... else: ...`。else 中可用 `box = pred[..., :4].sum() * 0`、`cls = pred[..., 5:].sum() * 0`，objectness 仍放在分支之外全格計算。順便說明 scalar 是 shape 為 `[]` 的單一值 tensor；乘 0 保留與 prediction 的計算關係，因此回傳零梯度。

在片段旁明示三個分母：框對選出的 `Npositive×4` 個元素平均，CE 先為每個正格算一次 loss 再對 `Npositive` 平均，BCE 對 `B×S×S` 格平均。`Npositive=0` 時不執行前兩個 mean。

### 4. 對久未接觸數學的讀者，BCE 與梯度方向仍差一步

位置：教材第 12–13、28 行。

MSE 有手算式，但 BCE 為何對正負 target 都是 `ln(2)`、`/16` 從何而來，以及負梯度為何提高 logit，只給結果。第 28 行的 `/16` 也只適用於本例，初學者容易當成通用常數。

建議補 `p=sigmoid(z)`、`t∈{0,1}`，單格 BCE 是 `−[t ln(p)+(1−t)ln(1−p)]`；`p=.5` 時正格取 `−ln(.5)`，负格取 `−ln(1−.5)`，兩者相同。CE 是正確類別的 `−ln(p[class_id])`。`ln`／`log` 在這裡都是自然對數。

接著寫全格 mean 的梯度為 `(p−t)/(B×S×S)`；本例分母才是 16。用一次更新 `z_new=z−學習率×gradient` 展示：正格梯度 `−.03125`，所以減去負數會提高 z；負格相反。不要求讀者先會微積分推導，可明示這是本例使用的導數結果。

### 5. 現有檢查比「mask 正確、空圖有有效梯度」的教學目標弱

位置：教材第 3、13、28、30–32 行；程式第 20–23、27–33 行。

目前僅驗證 `[0,0,0]` 一個背景格的框梯度為 0，未檢查其他背景格或任何背景 class 梯度。objectness 梯度僅 assert 正負號，精確數值靠 print；空圖只驗證框／class loss 為 0 與梯度有限，若空圖 objectness 被錯誤歸零，仍可能通過有限梯度檢查。這不是目前實作有錯，而是檢查沒有覆蓋文中聲稱要驗證的性質。

建議程式用 `pos = target['positive']`，assert 所有 `prediction.grad[~pos][..., :4]` 及 `prediction.grad[~pos][..., 5:]` 均為 0；用 `torch.allclose` 核對所有正／負 objectness 梯度為 `−1/32`、`1/32`。空圖再 assert objectness／total 等於 `math.log(2)`、所有 objectness 梯度等於 `1/32`、框／class 梯度全部為 0。如此可把「有限」與「正確」分開驗證。

## 其餘評估

- 沒有發現框、objectness、class 權重相加或數字四捨五入錯誤。
- 正格框／分類、全格 objectness 的監督範圍合理；實測也吻合。
- 權重 5 的設計說明、與 YOLOv1 的差異、空 tensor mean 得到 NaN、CE 對共同平移不變，均正確。
- 沒有明顯冗餘。建議優先補上同一個案例的 shape 與座標推導，無須另加不同場景的長篇例子。


## 作者修訂與驗證（2026-10-02）

補完整head／target shape、logit／mask、框encode兩種分母、BCE自然對數及一般梯度分母；正文F.*使用mean與空mask分支。case檢查所有背景框／class梯度0、正負obj精確±1/32，空圖obj=ln2且所有obj梯度1/32。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/07-loss.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。
