# 一次學習的超短暖身

先回答一個最小問題：程式跑完一次 `loss.backward()`，模型已經學到了嗎？還沒有。這一步只算出調整方向；`optimizer.step()` 才修改參數。本節用一個可手算的數字，讓 forward、loss、gradient 和更新接在一起。前置只需會建立 Python 變數，不用記得微積分推導。

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/00-warmup.ipynb)。本機在專案根目錄執行 `PYTHONPATH=. python lesson_cases/00-warmup.py`。實驗只有一筆資料、一個參數、CPU 一步 SGD；不用下載資料。這是算術驗證，不代表完成了任何圖片任務。

## 從輸入走到誤差

把模型想成一個旋鈕：輸入數字 \(x=2\)，模型將它乘上可學參數 \(w\)，輸出預測 \(\hat y=wx\)。希望答案是 \(y=4\)，初始旋鈕 \(w=1\)，所以預測是 2。

Tensor 是帶 shape 的數字容器。此處輸入 `[[2.0]]` 的 shape 是 `[1,1]`，第一軸是 batch 中一筆資料，第二軸是每筆一個特徵。權重也是 `[1,1]`。一般線性層把 `[B,D]` 輸入乘上 `[K,D]` 權重的轉置，得到 `[B,K]`；B 是本批資料筆數，D 是每筆輸入特徵數，K 是每筆輸出數。當 D 大於 1，一個輸出就是「每個輸入乘上對應權重，再加起來」，這便是內積。一般Linear還會加上每個輸出的bias（可學的常數偏移）；本例設定`bias=False`，只有w這一個參數。

我們定義平方誤差 \(L=(\hat y-y)^2\)。這裡是 \((2-4)^2=4\)。loss 為 scalar，也就是 shape `[]`；它表達目前答案離目標多遠，**不是**分類正確率。平方把正負誤差都轉成非負數。

## 梯度只是在回答「旋鈕往哪邊轉」

必懂的是梯度的意義：固定資料，將某個參數稍微增加，loss 會如何改變？\(\partial L/\partial w\) 就是這個變化率。不需要背「偏導」這個名字；模型有很多旋鈕時，它只表示一次看其中一個。

本例的兩段關係是 \(w\) 改變預測、預測改變 loss。把這兩段影響相乘，就是鏈式法則：

第一段 \(\hat y=wx\)：w 增加一點點 \(\Delta w\)，預測增加 \(x\Delta w\)，所以變化率是 x=2。第二段令誤差 \(e=\hat y-y\)：\((e+\Delta e)^2-e^2=2e\Delta e+(\Delta e)^2\)。當 \(\Delta e\) 很小，最後一項更小，因此平方誤差的瞬間變化率是 \(2e\)。這解釋了下面的 2、誤差和 x 各從哪裡來；不用背導數表。

\[
\frac{\partial L}{\partial w}
=2(\hat y-y)\times x
=2(2-4)\times2=-8.
\]

負號表示增加 w 能往降低 loss 的方向走。SGD 用學習率 \(\eta=0.1\) 決定走多大步：\(w_{new}=w-\eta g=1-0.1(-8)=1.8\)。更新後預測 \(1.8\times2=3.6\)，新 loss 是 \(0.16\)。這次下降有手算證據；一般模型每步或每批的 loss 不保證都下降。

## 對應到五行程式

先建立與手算相同的設定。`bias=False` 讓模型只有 w，避免多一個截距；線性層預設隨機初始化，因此要明確把它設為1：

```python
import torch
model = torch.nn.Linear(1, 1, bias=False)
with torch.no_grad():
    model.weight.fill_(1.0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x, target = torch.tensor([[2.0]]), torch.tensor([[4.0]])
```

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(x)                         # forward
loss = ((prediction - target) ** 2).mean()   # scalar
loss.backward()                              # 填入參數的 .grad
optimizer.step()                             # 修改參數
```

`zero_grad` 清除上一輪梯度，因為 PyTorch 的 `backward` 預設會累加梯度。`mean` 表示在多筆資料時取平均；本例一筆，所以與手算相同。可以用 `parameter.grad` 看方向，用更新前後的副本確認真的改了；儲存副本用 `detach().clone()`，不能只是儲存同一個參數物件的別名。

梯度計算需要從 loss 連到參數的計算路徑。若把預測先 `.detach()`、轉成 NumPy，或以 `.item()` 建立新的 loss，原路徑就斷了。記錄數值可以 `.item()`，計算 loss 時保留 tensor。

## 兩個常混在一起的開關

`model.train()` 與 `model.eval()` 控制某些層的行為，例如 Dropout 是否隨機遮掉值、BatchNorm 是否更新統計。這個單純線性層兩種模式結果相同。`eval()` 本身不停止梯度；本節有 assertion 驗證這件事。推論時另用 `with torch.no_grad():`，省去建立梯度圖的成本。

Optimizer 內部持有**當初交給它的參數物件**。Notebook 若重跑 `model = ...`，得到的是新模型，舊 optimizer 仍然指向舊參數。因此要一起重建 optimizer。只改模型名稱不會自動替換它內部的引用。

## 可核對輸出與停點

```text
x_shape=(1, 1), weight_shape=(1, 1)
prediction=2.00, loss=4.00, gradient=-8.00
weight: 1.00 -> 1.80; new_loss=0.16
```

透過上述數值、模式與 optimizer 引用檢查便可停止。收益是每一步都能追到具體數字；代價是這個線性問題不含影像、非線性與泛化，不能用它判斷 CNN 準確率。

## 自主練習與答案

把學習率改成 0.25，先預測新參數再執行。答案：\(w=1-0.25(-8)=3\)，預測 6，loss 又是 4。方向正確仍可能走過頭。修改案例中的對應 assertion 才能驗證新設定；不要刪除檢查來掩蓋不一致。再回答「只呼叫 backward 是否改變 w？」答案是否，除非你另有手動更新程式。

延伸原始檔案：[PyTorch autograd 教學](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) 與 [SGD API](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/00-warmup.json)

??? example "展開本次實際輸出"

    ```text
    x_shape=(1, 1), weight_shape=(1, 1)
    prediction=2.00, loss=4.00, gradient=-8.00
    weight: 1.00 -> 1.80; new_loss=0.16
    eval still tracks gradients; replacement optimizer points to replacement model
    ```

<!-- curriculum-evidence:end -->
