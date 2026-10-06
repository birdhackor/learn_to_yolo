# 0 一次學習的超短暖身

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/00-warmup.ipynb){ .md-button }

先用一個能手算的模型，看看「訓練一步」到底做了什麼。它收到數字 2，我們希望它回答 4。

模型只有一個可以調整的數字 \(w\)，叫參數。它的做法很簡單：**把輸入乘上 w**。所以 w=1 時回答 2；w=2 時才會回答 4。訓練要做的，就是根據目前的錯誤，把 w 調向比較好的值。

本節在 CPU 上用一筆資料更新一次，沒有圖片。前半只需四則運算與平方；想動手可點頁首的 Colab，操作步驟在頁末。

## 從輸入走到誤差

先把三件事分開：輸入 \(x=2\)、人給的答案 \(y=4\)、模型可調的參數 \(w=1\)。資料 x、y 保持不變，這一步只改 w。

模型的預測寫成 \(\hat y=wx\)。\(\hat y\) 讀作 y hat，表示模型的答案，和人給的 y 分開。現在預測是 \(1\times2=2\)，比目標 4 少了 2。

我們把誤差平方當成 loss（損失）：

\[
L=(\hat y-y)^2=(2-4)^2=4.
\]

loss 是衡量錯多少的一個數，越小越好。平方讓「太大」和「太小」都算錯：例如預測 6 和預測 2，離目標 4 一樣遠，loss 都是 4。

## 梯度在回答「旋鈕多轉一點，loss 怎麼變」

把 w 想成旋鈕。現在要決定：往大轉，還是往小轉？

試著讓 w 從 1 增加到 1.01：預測從 2 變成 2.02，loss 從 4 變成 3.9204。**w 增加一點，loss 反而減少**，所以這次應該把 w 調大。

梯度把這個方向寫成變化率。本例記為 \(g=\partial L/\partial w\)：固定資料，w 增加很小一點時，loss 的變化量除以 w 的變化量。上面的試算是

\[
\frac{3.9204-4}{1.01-1}=-7.96.
\]

把變動縮得更小，這個值會接近 **−8**，就是這裡的梯度。負號告訴我們增加 w 能讓 loss 在附近下降；正號則要往減小 w 的方向走。

這個 −8 也能沿著計算的兩段關係得到：w 改變預測，預測改變 loss。第一段的變化率是輸入 x=2；第二段的變化率是兩倍誤差 \(2(2-4)=-4\)。把兩段相乘，就是連鎖律（chain rule）：

\[
g=2(\hat y-y)\times x=2(2-4)\times2=-8.
\]

PyTorch 的 `loss.backward()` 會沿著 loss 的計算過程，自動算出這個梯度，存進參數的 `.grad`。**算出方向之後，參數還沒有改變。**

### 現在真的轉動旋鈕

optimizer（優化器）拿梯度修改參數。本例用 **SGD（Stochastic Gradient Descent，隨機梯度下降）**，學習率 \(\eta=0.1\) 控制每步走多遠。SGD 在一般訓練中可隨機抽取資料來算梯度；本例只有一筆固定資料，沒有這個抽樣步驟。\(\eta\) 讀作 eta。更新是往梯度的反方向走：

\[
w_{\mathrm{new}}=w-\eta g=1-0.1(-8)=1.8.
\]

減掉負數就是增加，所以 w 增加了 0.8。用新 w 再算一次：預測 \(1.8\times2=3.6\)，loss 是 \((3.6-4)^2=0.16\)。

| 這一步看到的數字 | 更新前 | 更新後 |
| --- | --- | --- |
| 參數 w | 1 | 1.8 |
| 模型預測 | 2 | 3.6 |
| loss | 4 | 0.16 |

這次更新讓答案接近目標。方向正確仍可能走太遠，練習 1 會讓你試一次。

## 對應到五行程式

先建立和手算相同的模型與資料。`Linear(1,1)` 表示每筆輸入 1 個數、輸出 1 個數；`bias=False` 表示只做乘法，沒有多加一個可學常數。權重就是 w，把它設成 1。

``` { .python data-excerpt="lesson_cases/00-warmup.py" }
import torch
...  # 完整程式的 main() 與固定設定省略
model = torch.nn.Linear(1, 1, bias=False)
with torch.no_grad():
    model.weight.fill_(1.0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x, target = torch.tensor([[2.0]]), torch.tensor([[4.0]])
```

`tensor` 是有形狀的數字容器。`[[2.0]]` 的 shape 是 `[1,1]`：1 筆資料，每筆 1 個輸入。`target` 裝目標 4。`torch.no_grad()` 讓手動設定初值的動作不記入計算過程；離開它的縮排後，恢復梯度記錄。`model.parameters()` 把要學的參數交給 optimizer，`lr` 是學習率。

下面五行就是一次訓練，和剛才的手算對上：

``` { .python data-excerpt="lesson_cases/00-warmup.py" }
optimizer.zero_grad(set_to_none=True)        # 清除舊梯度
prediction = model(x)                       # 算出預測 2
loss = ((prediction - target) ** 2).mean()  # 算出 loss 4
loss.backward()                            # 算出梯度 -8，w 仍是 1
...  # 完整程式在這裡記下梯度與更新前的 w，供之後核對
optimizer.step()                           # 真正把 w 改成 1.8
```

`zero_grad` 清掉上一步留下的梯度，因為 PyTorch 的 backward 會累加梯度。`.mean()` 把誤差平方平均成一個 loss；本例只有一筆，結果就是手算的 4。完整程式的其他設定和斷言用來核對這一步，細節放在下面的選讀區。

要記住的關係是：**forward 算答案，loss 算錯多少，backward 算該怎麼調，step 才真的調。**之後模型變大，這條關係仍然相同。

## 核對輸出：看到這些就完成本節

執行 Colab 的完整實驗格，會印出：

```text
x_shape=(1, 1), weight_shape=(1, 1)
prediction=2.00, loss=4.00, gradient=-8.00
weight: 1.00 -> 1.80; new_prediction=3.60; new_loss=0.16
eval still tracks gradients; replacement optimizer points to replacement model
```

第 2、3 行正是手算的更新前後數字。第 1 行記下容器的 shape；最後一行是兩項額外程式檢查的通過訊息，下方選讀區會解釋。完整程式也用 `assert` 自動核對答案，條件不成立就報錯。

這個例子確認了一次參數更新，還沒有學習圖片。下一節把輸入換成圖片，看看小 CNN 如何回答紅、藍分類；資料與模型會變大，但仍會用到這五行。

## 自主練習與答案

**練習 1**：把學習率改成 0.25，先用手算預測新的 w、更新後的預測值與 new_loss，再執行驗證。這三個數都印在輸出第 3 行；第 2 行的 `prediction=2.00` 是更新前的預測，改了學習率也不會變。在 Colab 裡改「本節可修改的完整實驗」下面那一格；本機則改 `lesson_cases/00-warmup.py`，兩者是同一份程式。要改的是建立主 optimizer 的那行 `optimizer = torch.optim.SGD(model.parameters(), lr=0.1)`：把 `lr=0.1` 改成 `lr=0.25`。程式後段還有一個 `lr=0.1`，屬於替換用的 `new_optimizer`，和下面的斷言無關，不要改。

完整程式印出數字之後，用這三行斷言核對答案：

``` { .python data-excerpt="lesson_cases/00-warmup.py" }
assert abs(gradient + 8.0) < 1e-6
assert abs(after - 1.8) < 1e-6 and abs(new_loss - 0.16) < 1e-5
assert abs(new_prediction.item() - 3.6) < 1e-5
```

`gradient` 是 backward 之後記下的梯度，`after` 是更新後的 w，`new_prediction` 是更新後的預測。`new_prediction` 是 tensor；第三行先用 `.item()`把它取成普通數字，和 `after`、`new_loss` 一樣用普通數字比較。`abs(a - b) < 1e-6` 表示 a 和 b 相差不到 10⁻⁶（`1e-5` 則是相差不到 10⁻⁵），用來容許小數計算的微小誤差。

??? note "為什麼比較小數要容許一點誤差"

    電腦用二進位、有限的位數存小數；PyTorch 預設的 float32（32 位元浮點數）約只有 7 位有效數字，1.8、3.6 這類十進位小數存不精確，只能存成最接近它們的數。所以 `after` 實際上是 1.7999999523…，`new_prediction` 是 3.5999999046…，寫成 `after == 1.8` 或 `new_prediction.item() == 3.6` 反而不成立。

只改學習率就執行，前 3 行照樣會印出（第 3 行已換成新的 w、更新後的預測與 new_loss），接著第二行斷言出現 AssertionError。這是預期中的錯誤：第二、三行斷言還在核對舊答案 1.8、0.16 與 3.6。把這三個數字換成你預測的新 w、new_loss 與更新後的預測，再執行一次；不再報錯、第 4 行也印出來，就表示你的預測對了。只改了第二行、忘了改第三行，就換成第三行斷言報錯。第一行核對梯度，不用改，因為梯度在更新之前就算好了，和學習率無關。不要刪掉斷言來讓錯誤消失。

**練習 2**：只呼叫 `backward`，會改變 w 嗎？先預測，再用程式確認：把完整程式裡 `optimizer.step()` 那一行刪掉，或在那一行最前面加 `#`，讓 Python 把它當成註解跳過，再執行。第 3 行會印出什麼？哪一行斷言會報錯？做完記得改回來。

??? note "參考答案"

    **練習 1**：\(w=1-0.25(-8)=3\)，預測 \(3\times2=6\)，loss 是 \((6-4)^2=4\)。第二行斷言要改成 `assert abs(after - 3.0) < 1e-6 and abs(new_loss - 4.0) < 1e-5`，第三行改成 `assert abs(new_prediction.item() - 6.0) < 1e-5`。三行斷言都通過時，輸出第 3 行是 `weight: 1.00 -> 3.00; new_prediction=6.00; new_loss=4.00`。前面說過，最理想的 w 是 2；w 從 1 跳到 3，越過 2 到了另一側同樣遠的地方，誤差大小同樣是 2，所以 loss 又是 4。方向正確仍可能走過頭。

    延伸：學習率多少，才能一步剛好走到 w=2？要讓 \(1-\eta(-8)=2\)，也就是 \(8\eta=1\)，所以 \(\eta=0.125\)。

    **練習 2**：不會。backward 只把梯度填進 `.grad`；要等 `optimizer.step()`，或你另外寫的手動更新程式，才會改變 w。拿掉 `optimizer.step()` 後，第 3 行印出 `weight: 1.00 -> 1.00; new_prediction=2.00; new_loss=4.00`：w、預測和 loss 都和更新前一樣。接著第二行斷言出現 AssertionError，因為 w 不是 1.8；第一行核對梯度，照樣通過，因為梯度在 backward 時就算好了。

延伸閱讀（英文官方文件）：[PyTorch autograd 教學](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)（autograd 是 PyTorch 自動算梯度的機制，`backward` 就靠它）與 [SGD 官方說明頁](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)。

## 選讀：需要時再回來查

主線已完成一次更新。以下分別補充推導、PyTorch 行為與執行方式；遇到相關問題時再讀對應項目。

??? note "−8 的推導：從變化量回想連鎖律"

    本例的兩段關係是：\(w\) 改變預測，預測改變 loss。把這兩段的變化率相乘，就是連鎖律（chain rule）。下面分段計算。

    第一段 \(\hat y=wx\)：w 增加一點點 \(\Delta w\)，預測增加 \(x\Delta w\)，所以變化率是 x=2。

    第二段令誤差 \(e=\hat y-y\)，於是 \(L=e^2\)。e 增加 \(\Delta e\) 時，L 的變化量是 \(\Delta L=(e+\Delta e)^2-e^2=2e\Delta e+(\Delta e)^2\)，所以 \(\Delta L/\Delta e=2e+\Delta e\)；\(\Delta e\) 趨近 0 時就是 \(2e\)，這是 L 對 e 的瞬時變化率。目標 y 固定，所以預測多多少，e 就多多少；\(2e\) 也就是 L 對預測 \(\hat y\) 的變化率。

    兩段接起來：\(\Delta\hat y=x\,\Delta w\)，所以 \(\Delta L=2e\,\Delta\hat y+(\Delta\hat y)^2\approx 2e\,x\,\Delta w\)。約等號 ≈ 表示略去了很小的 \((\Delta\hat y)^2\)：它等於 \(x^2(\Delta w)^2\)，除以 \(\Delta w\) 後是 \(x^2\Delta w\)，在 \(\Delta w\) 趨近 0 時也趨近 0。所以兩邊除以 \(\Delta w\)，再讓 \(\Delta w\) 趨近 0，就得到下面的式子。這說明了式子裡的 2、誤差和 x 各從哪裡來，不用背導數表。

    \[
    \frac{\partial L}{\partial w}
    =2(\hat y-y)\times x
    =2(2-4)\times2=-8.
    \]

    最前面的 2 來自平方，括號裡的 2 是預測 \(\hat y\)，最後的 2 是輸入 x。也可以用計算機驗證：把 w 從 1 改成 1.01，預測變成 2.02，loss 變成 \((2.02-4)^2=3.9204\)。loss 的變化量是 \(3.9204-4=-0.0796\)，除以 w 的變化量 0.01，得到 −7.96，很接近 −8；差的 0.04 正是前面略去的 \(x^2\Delta w=4\times0.01\)。

??? note "tensor、梯度累加與保存舊權重"

    `zero_grad` 清掉參數上留著的舊梯度。訓練通常要重複很多步，每步都要先清，因為 PyTorch 的 `backward` 預設會把新梯度加在舊梯度上（累加）。例如沒先清除，就在 w 還是 1 時再做一次 forward 和 backward，`model.weight.grad` 會變成 −8+(−8)=−16，更新就走兩倍遠。`set_to_none=True` 把 `.grad` 清成 None（表示還沒有梯度）。本課程用的 PyTorch 2.9.1 預設就是 True；程式仍明確寫出來，讀程式時不必記預設值。對下一次 backward 有算到梯度的參數來說，效果和歸零相同；曾有梯度、這次沒算到梯度的參數（例如這一步的 loss 沒用到它，計算圖沒經過它）則不同：它的 `.grad` 會維持 None，而不是歸零後的全 0，`optimizer.step()` 也會直接跳過它。第 2 章〈[訓練診斷](02-diagnostics.md)〉就靠這個差別，找出沒接上 backward 的參數。本例只跑一步，有沒有這行結果都相同。

    `mean` 在多筆資料時取平均，把每筆的誤差平方併成一個數；本例只有一筆，所以與手算相同。loss 要是一個數，才比得出「變小了沒」，`loss.backward()` 也要從這一個數往回算。有好幾筆資料卻沒用 `.mean()`（或 `.sum()`）併成一個數，`loss.backward()` 會報錯（RuntimeError: grad can be implicitly created only for scalar outputs）。

    backward 之後，可以用 `print(model.weight.grad)` 看梯度，應該印出 `tensor([[-8.]])`，和手算一致。

    想確認 step 真的改了參數，要先存一份更新前的副本。若只寫 `alias = model.weight` 來存舊值，得到的不是副本，而是別名：同一個物件多了一個名字。`optimizer.step()` 會直接改寫參數裡的數字，別名也就跟著變。就像 Python 的 `b = a` 不會複製 list：之後改 a 的內容（例如 `a[0] = 9`），b 也跟著變。

    可以動手看差別：在完整程式的 `optimizer.step()` 前面插入 `alias`、`snapshot` 兩行，後面加兩行 `print`，每一行都和 `optimizer.step()` 對齊縮排。副本不取名 `before`：完整程式已經用 `before` 存了更新前的 w（後面說明），第 3 行輸出要用它。

    ```python
    alias = model.weight                      # 別名：step 之後跟著變成 1.8
    snapshot = model.weight.detach().clone()  # 副本：step 之後仍是 1.0
    optimizer.step()                          # 完整程式原有的這一行，不要重複加
    print(alias)
    print(snapshot)
    ```

    執行後，原本的 4 行輸出前面會多出 3 行：

    ```text
    Parameter containing:
    tensor([[1.8000]], requires_grad=True)
    tensor([[1.]])
    ```

    前兩行是 `alias`：`model.weight` 是 Parameter（PyTorch 用來裝可學參數的 tensor），所以印出時多一行 `Parameter containing:`；`requires_grad=True` 表示 PyTorch 要替它算梯度。第三行是 `snapshot`，仍是更新前的 1。PyTorch 印 tensor 時，裡面的數都是整數，就只印到小數點，例如 `1.`；有小數時預設印到小數點後 4 位，所以 1.8 印成 `1.8000`。

    `detach()` 只把 tensor 從計算圖剪下來，數字仍和參數共用；`clone()` 才複製出自己的一份數字。所以只寫 `detach()` 存下的舊值，step 之後也會變成 1.8。完整程式則用 `before = model.weight.item()`：`.item()` 把只含一個數的 tensor 取成普通的 Python 數字，同樣不會跟著變。

    backward 要沿著計算圖從 loss 一路走回參數，所以算 loss 時計算圖不能斷。`.item()` 取出的普通數字不記得自己是怎麼算出來的；`.detach()` 會刻意把 tensor 從計算圖剪下；NumPy 是另一個常用的數值計算套件，它的陣列也不記錄計算圖。若把預測先 `.detach()`、轉成 NumPy，或先用 `.item()` 取成普通數字再算 loss，計算圖就斷了。在本例，這三種做法都會讓程式報錯。例如 `loss = (prediction.item() - 4.0) ** 2` 算出的是普通的 Python 浮點數（float），根本沒有 `.backward()` 可以呼叫；只用 `.detach()` 時 loss 仍是 tensor，但本例只有 w 一個參數，剪斷後 loss 不再連著任何要學的參數，backward 就報錯。模型有好幾段參數時，用 `.detach()` 剪斷中間一段不一定報錯：loss 仍連著後段的參數，backward 照常執行，只是被剪斷的前段參數 `.grad` 停在 None、學不到東西。第 2 章〈[訓練診斷](02-diagnostics.md)〉的失敗一就是這種情況。記錄數值可以用 `.item()`；要拿來 backward 的 loss，計算時要保留 tensor。

## 兩個常混在一起的開關

??? note "model.eval() 和 torch.no_grad() 各控制什麼"

    常被混在一起的兩個開關是 `model.eval()` 與 `torch.no_grad()`。前者（和 `model.train()` 成對）只切換某些層的行為，不會關掉梯度記錄；後者才讓 PyTorch 不記錄計算圖。推論（拿模型做預測、不更新參數）時，通常兩個一起用。

    `model.train()` 切到訓練模式，`model.eval()` 切到評估／推論模式。兩者只影響某些層：例如 Dropout（訓練時隨機把部分數值設成 0 的層）只在 train 模式遮值；BatchNorm（Batch Normalization，批次正規化，用平均與變異數把數值標準化的層）在 train 模式用這一批資料自己的平均與變異數來標準化，同時更新它記錄的平均與變異數；eval 模式改用記錄下來的值，所以同一筆輸入在兩種模式的輸出可能不同。本課程的程式沒用到這兩種層，只要記得：有些層在訓練與推論時行為不同。本節這個單純的線性層，兩種模式結果相同。

    完整程式在更新參數之後，兩個開關都用上：先切到 eval 模式，再在 `torch.no_grad()` 裡重算預測與 loss；接著用斷言（assert）驗證 `eval()` 本身不會關掉梯度記錄。斷言的意思是「條件不成立就報錯停下」，用來自動核對答案：

    ``` { .python data-excerpt="lesson_cases/00-warmup.py" }
    model.eval()                    # 切到評估／推論模式
    with torch.no_grad():           # 縮排內不記錄計算圖
        new_prediction = model(x)   # 更新後的預測
        new_loss = ((new_prediction - target) ** 2).mean().item()
    ...                             # 省略：印出前 3 行輸出、核對答案的斷言
    assert model.training is False  # 確認已切到 eval 模式
    assert model(x).requires_grad   # eval 模式下，輸出仍連著計算圖
    ```

    `requires_grad` 為 True，表示這個結果還連著計算圖，能拿來算梯度。最後一行寫在 `with torch.no_grad():` 的縮排外面，只有 eval 模式在作用，所以輸出仍連著計算圖。`torch.no_grad()` 常見兩種用途：一是像前面那樣手動改參數，二是推論時省下記錄計算圖的成本（記憶體與時間）。重算 `new_prediction` 與 `new_loss` 屬於第二種：只看更新後的結果，不再更新參數。`new_loss` 只用來印出和核對、不拿來 backward，所以可以直接用 `.item()` 取成普通數字。

## 重建模型時，optimizer 也要重建

??? note "在 notebook 裡重建模型時要注意"

    optimizer 內部記住的是**當初交給它的參數物件**，不是 `model` 這個名字。在 Colab 這類可以分格執行的 notebook 裡，若建模型與建 optimizer 寫在不同格，之後只重跑建模型的那一格，只是讓 `model` 這個名字改指向一個新模型；optimizer 手上仍拿著舊模型的參數。就像 Python 執行 `a = [1]; b = a; a = [2]` 之後，b 還是 `[1]`。這和前面「改 a 的內容，b 也跟著變」不同：`a = [2]` 沒有改原本 list 的內容，而是讓 a 改指向另一個新 list。所以重建模型時，要一起重建 optimizer。本節的完整程式把兩者寫在同一格的 `main()` 裡，每次執行都一起重建，所以不會遇到這個問題。

    若忘了重建，程式不會報錯，但 `optimizer.step()` 只處理舊模型的參數：新模型的權重一直不變，loss 也一直不降。完整程式最後建了一個替換用的新模型和新 optimizer，並用斷言確認新 optimizer 拿到的是新模型的參數。

## 執行方式

??? note "第一次用 Colab"

    1. 點頁首的按鈕，Colab 會在瀏覽器開啟本節的 notebook。執行程式前要先登入 Google 帳號。
    2. 執行第一個程式格（環境格）：點格子左側的執行鈕，或點進格子後按 Shift+Enter。Colab 可能先跳出警告，說這份 notebook 不是 Google 編寫的、是從 GitHub 載入的；它來自本教材的 GitHub 專案，選擇仍要執行即可（按鈕名稱以 Colab 當時的畫面為準）。
    3. 等環境格印出「固定教材版本： lessons-v0.6.1」這一行，環境才算準備好。PyTorch 不是 2.9.1 時，環境格要先下載、改裝，會多等一會兒。安裝套件時可能印出一些訊息，其中可能有 `ERROR:` 開頭、說其他預裝套件（例如 torchvision）需要別的版本的訊息；各節實驗用不到那些套件，只要之後有印出這一行，就是成功了。若環境格停在錯誤、沒有印出這一行（例如要你重新啟動工作階段），照訊息做完，再從第一格執行。
    4. 用同樣的方式執行最後一格「本節可修改的完整實驗」。剛打開 notebook 時，這一格下方已經有一份輸出：那是執行紀錄存下的結果，不是你跑出來的；你執行之後，它會換成這次印出的內容。之後改了程式，要再執行一次這一格才會生效。

??? note "在自己的電腦執行"

    先從 [GitHub](https://github.com/birdhackor/learn_to_yolo) 下載本專案，照 README 建立 `.venv-model`（專給本教材用、裝好固定版本套件的 Python 環境）並安裝 PyTorch。接著在專案根目錄（含 `lesson_cases` 資料夾的那一層）執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/00-warmup.py`。

    `.venv-model/bin/python` 是這個環境裡的 Python，PyTorch 就裝在這裡。只打 `python` 時，用到的是電腦原本的 Python：可能出現 `No module named 'torch'`（找不到 PyTorch），有的電腦甚至沒有 `python` 這個指令（出現 `command not found`）。之後各節寫的 `python …` 也一樣：先執行 `source .venv-model/bin/activate` 啟用這個環境，`python` 就會是這個環境的 Python（只對這個終端機視窗有效，開新視窗要再啟用一次）；或直接把 `python` 換成 `.venv-model/bin/python`。

    `PYTHONPATH=.` 讓 Python 找得到專案裡的 miniyolo（本專案共用的程式套件）；本節用不到，後面章節需要。Windows 的 PowerShell 不支援這種寫法，要照 README 先用 `$env:PYTHONPATH='.'` 設定，再執行 `.venv-model\Scripts\python.exe lesson_cases/00-warmup.py`；之後各節的 `python` 也換成 `.venv-model\Scripts\python.exe`。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-06 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/00-warmup.json)

??? example "展開本次實際輸出"

    ```text
    x_shape=(1, 1), weight_shape=(1, 1)
    prediction=2.00, loss=4.00, gradient=-8.00
    weight: 1.00 -> 1.80; new_prediction=3.60; new_loss=0.16
    eval still tracks gradients; replacement optimizer points to replacement model
    ```

<!-- curriculum-evidence:end -->
