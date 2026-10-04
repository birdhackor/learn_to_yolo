# 0 一次學習的超短暖身

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/00-warmup.ipynb){ .md-button }

先回答一個最小問題：程式跑完一次 `loss.backward()`，模型已經學到了嗎？還沒有。這一步只算出梯度；`optimizer.step()` 才修改參數。optimizer（優化器）是專門拿梯度去修改參數的物件。

本節用一個能手算的小例子，把訓練一步的四件事接起來：forward（前向計算：把輸入算成預測）→ loss（損失：預測離目標多遠）→ gradient（梯度：loss 隨參數怎麼變）→ 更新參數。讀完後，你能用紙筆算出這一步的每個數字，也能指出程式裡哪一行做哪件事。

前置知識：前半的手算只需四則運算與平方；後半的程式假設你會基本 Python（變數、函式、list），跑過一次 PyTorch 範例更好。沒寫過 PyTorch 也能先讀完手算。

不想設定環境，就用頁首的「在 Colab 執行本節」按鈕。

??? note "在自己的電腦執行"

    先從 [GitHub](https://github.com/birdhackor/learn_to_yolo) 下載本專案，依 README 安裝 PyTorch。接著在專案根目錄（含 `lesson_cases` 資料夾的那一層）執行 `PYTHONPATH=. python lesson_cases/00-warmup.py`。

    `PYTHONPATH=.` 讓 Python 找得到專案裡的 miniyolo（本專案共用的程式套件）；本節用不到，後面章節需要。Windows 的 PowerShell 不支援這種寫法，要照 README 先用 `$env:PYTHONPATH='.'` 設定，再執行 `python lesson_cases/00-warmup.py`。

實驗只有一筆資料、一個參數，在 CPU 上做一步 SGD（stochastic gradient descent，隨機梯度下降）；不用下載資料。「隨機」指一般訓練每次隨機抽一小批資料來算梯度；本例只有一筆資料，不必抽。這是算術驗證，不代表完成了任何圖片任務。

## 從輸入走到誤差

把模型想成只有一個旋鈕的機器：旋鈕的刻度就是可學參數 \(w\)，機器把輸入乘上 \(w\)。輸入數字 \(x=2\)，輸出預測 \(\hat y=wx\)。目標（希望得到的輸出）是 \(y=4\)。旋鈕一開始轉在 \(w=1\)，所以預測是 2。因為 \(2\times2=4\)，轉到 \(w=2\) 時預測剛好等於目標。

PyTorch 程式把資料和參數（本例的 x、目標、權重）裝在 tensor（張量）裡。tensor 是帶 shape（形狀）的數字容器；shape 記錄每個軸有多長，每多包一層中括號就多一個軸。程式把 x 存成 `[[2.0]]`，shape 是 `[1,1]`：第一軸長度 1，表示這一批（batch）只有 1 筆資料；第二軸長度 1，表示每筆只有 1 個輸入數字（這種輸入數字叫特徵）。權重也是 `[1,1]`。

??? note "一般情況：多筆資料、多個輸入"

    一般的線性層（PyTorch 裡就是 `torch.nn.Linear`）一次處理 B 筆資料，每筆有 D 個輸入特徵，每筆算出 K 個輸出。輸入是 `[B,D]`，權重是 `[K,D]`，輸出是 `[B,K]`。

    先看一筆資料的手算例。輸入是 [1,2,3]（D=3），要算 2 個輸出（K=2）。第一個輸出的權重是 [1,0,1]：每個輸入乘上對應權重再加起來，\(1\times1+2\times0+3\times1=4\)。第二個輸出的權重是 [0,1,0]：\(1\times0+2\times1+3\times0=2\)。所以輸出是 [4,2]。這種「對應相乘再相加」就是內積。

    PyTorch 把每個輸出的那組權重存成一列，所以權重的 shape 是「輸出數×輸入數」，這裡是 [2,3]。轉置就是把矩陣的列和行對調，[2,3] 轉置後變成 [3,2]。把這筆輸入看成 shape [1,3]（1 筆、3 個數），乘上 [3,2]，才得到 [1,2] 的輸出。B 筆資料時也一樣：`[B,D]` 輸入乘上 `[K,D]` 權重的轉置，得到 `[B,K]`。

    一般 `Linear` 還會加上每個輸出的 bias（可學的常數偏移）；本例設定 `bias=False`，只有 w 這一個參數。

預測減目標叫誤差，本例是 \(2-4=-2\)。我們定義 loss（損失）\(L=(\hat y-y)^2\)，也就是平方誤差；這裡是 \((2-4)^2=4\)。loss 表達目前的預測離目標多遠，越小越好；它和分類任務常看的正確率（答對幾成）是**兩回事**。

平方把正負誤差都轉成非負數，因為預測太大和太小都算錯。假設有兩筆資料，一筆預測太大 2（誤差 +2），另一筆太小 2（誤差 −2）；如果不平方就直接平均，會得到 0，看起來像完全沒錯。

在程式裡，loss 是 scalar（純量）：沒有任何軸的單一個數，shape 是空的 `[]`。`[[2.0]]` 雖然也只有一個數，但包了兩層括號，有兩個長度為 1 的軸。下面程式用 `.mean()` 把所有數平均成一個數，軸就不見了。

## 梯度在回答「旋鈕多轉一點，loss 怎麼變」

必懂的是梯度的意義：固定資料，將某個參數稍微增加，loss 會如何改變？\(\partial L/\partial w\) 就是這個變化率，精確地說是瞬時變化率：loss 的變化量除以 w 的變化量，在 w 的變化量趨近 0 時的值。

\(\partial\) 讀作 partial。這個符號只是提醒：模型有很多參數時，只動 w，其他先固定。每個參數各有一個這樣的變化率，排在一起就叫梯度；本例只有 w，梯度就是一個數。

本例的兩段關係是：\(w\) 改變預測，預測改變 loss。把這兩段的變化率相乘，就是連鎖律（chain rule）。下面分段計算。

第一段 \(\hat y=wx\)：w 增加一點點 \(\Delta w\)，預測增加 \(x\Delta w\)，所以變化率是 x=2。

第二段令誤差 \(e=\hat y-y\)，於是 \(L=e^2\)：\((e+\Delta e)^2-e^2=2e\Delta e+(\Delta e)^2\)。兩邊除以 \(\Delta e\) 得到 \(2e+\Delta e\)；\(\Delta e\) 趨近 0 時就是 \(2e\)，這是 L 對 e 的瞬時變化率。目標 y 固定，所以預測多多少，e 就多多少；\(2e\) 也就是 L 對預測 \(\hat y\) 的變化率。

兩段接起來：\(\Delta\hat y=x\,\Delta w\)，所以 \(\Delta L\approx 2e\,\Delta\hat y=2e\,x\,\Delta w\)。兩邊除以 \(\Delta w\)，再讓 \(\Delta w\) 趨近 0，就得到下面的式子。這說明了式子裡的 2、誤差和 x 各從哪裡來，不用背導數表。

\[
\frac{\partial L}{\partial w}
=2(\hat y-y)\times x
=2(2-4)\times2=-8.
\]

最前面的 2 來自平方，括號裡的 2 是預測 \(\hat y\)，最後的 2 是輸入 x。也可以用計算機驗證：把 w 從 1 改成 1.01，預測變成 2.02，loss 變成 \((2.02-4)^2=3.9204\)。loss 的變化量是 \(3.9204-4=-0.0796\)，除以 w 的變化量 0.01，得到 −7.96，很接近 −8。

PyTorch 的 `loss.backward()` 會自動做同一件事。forward 時，PyTorch 會記下 loss 是由 w 經過哪些運算算出來的，這份紀錄叫計算圖（computation graph）。backward 沿著計算圖從 loss 往回、一段一段套用連鎖律算出梯度，所以叫 backward。本例算出的 −8 會存進參數的 `.grad`。

記 \(g=\partial L/\partial w\)，本例 \(g=-8\)。梯度指向 loss 增加的方向：\(g>0\) 表示 w 變大時 loss 也變大，該讓 w 變小；\(g<0\) 正好相反，該讓 w 變大。兩種情況都要往梯度的反方向走，所以下一段的更新式用減號。沿梯度的反方向走、讓 loss 往下降，就叫梯度下降（gradient descent），也就是 SGD 名稱裡的 GD。本例 g 是負的，表示增加 w 能讓 loss 下降。

SGD 用學習率 \(\eta=0.1\) 控制步伐。學習率是倍率：每步從 w 減掉的量是 \(\eta\) 乘上梯度，\(\eta g=0.1\times(-8)=-0.8\)。減掉 −0.8 等於加 0.8，所以 w 實際增加 0.8（步伐是 0.8，不是 0.1）：\(w_{\text{new}}=w-\eta g=1-0.1(-8)=1.8\)。更新後預測 \(1.8\times2=3.6\)，新 loss 是 \(0.16\)。這次下降有手算證據；一般模型每步或每批的 loss 不保證都下降。

## 對應到五行程式

下面兩段程式摘自完整程式（Colab 裡的那份，也就是 `lesson_cases/00-warmup.py`）。完整程式還多了幾行，例如印出數字、自動核對答案。摘錄裡單獨一行的 `...` 表示那裡省略了完整程式的幾行。

先建立與手算相同的設定。程式用 `torch.nn.Linear`（線性層：把輸入乘上權重、通常再加上一個常數 bias 的層）當作那台只有一個旋鈕的機器。`bias=False` 讓模型只有 w，避免多一個截距；線性層的權重預設是隨機初始值，因此要明確把它設為 1：

``` { .python data-excerpt="lesson_cases/00-warmup.py" }
import torch
...                          # 省略：def main(): 和兩行與手算無關的設定
# Linear(1, 1)：每筆輸入 1 個數、輸出 1 個數；bias=False：只有 w 一個參數
model = torch.nn.Linear(1, 1, bias=False)
with torch.no_grad():        # 縮排內的動作不記進計算圖（不包會報錯）
    model.weight.fill_(1.0)  # 名稱以底線結尾：直接改寫原 tensor，把 w 設成 1
# parameters() 把可學參數（本例只有 w）交給 optimizer；lr 就是學習率 η
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x, target = torch.tensor([[2.0]]), torch.tensor([[4.0]])  # x=2，目標 y=4
```

`with torch.no_grad():` 底下縮排的程式，會在「不記錄計算圖」的設定下執行；離開縮排，就恢復記錄。手動設定初值不是模型計算的一部分，不該記進計算圖。而且 PyTorch 不允許在記錄計算圖時直接改寫需要梯度的參數；不包在 `torch.no_grad()` 裡，會出現 RuntimeError。

手算符號在程式裡的名字：\(w\)→`model.weight`、\(\hat y\)→`prediction`、\(y\)→`target`、\(L\)→`loss`、\(g\)→`model.weight.grad`（backward 之後才有值）、\(\eta\)→`lr`。下面五行就是訓練一步：

``` { .python data-excerpt="lesson_cases/00-warmup.py" }
optimizer.zero_grad(set_to_none=True)
prediction = model(x)                         # forward
loss = ((prediction - target) ** 2).mean()   # scalar
loss.backward()                              # 填入參數的 .grad
...                                          # 省略兩行：把梯度和更新前的 w 存成 gradient、before
optimizer.step()                             # 修改參數
```

下面逐一說明這幾行的細節，以及 PyTorch 常見的陷阱。第一次看不懂原因沒關係，先記住規則，後面章節會再遇到。

`zero_grad` 清掉參數上留著的舊梯度。訓練通常要重複很多步，每步都要先清，因為 PyTorch 的 `backward` 預設會把新梯度加在舊梯度上（累加）。例如沒先清除，就在 w 還是 1 時再做一次 forward 和 backward，`model.weight.grad` 會變成 −8+(−8)=−16，更新就走兩倍遠。`set_to_none=True` 把 `.grad` 清成 None（表示還沒有梯度），目前的 PyTorch 預設就是 True。對下一次 backward 有算到梯度的參數來說，效果和歸零相同；曾有梯度、這次沒算到梯度的參數則不同：它的 `.grad` 會維持 None，而不是歸零後的全 0，`optimizer.step()` 也會直接跳過它。第 2 章〈[訓練診斷](02-diagnostics.md)〉就靠這個差別，找出沒接上 backward 的參數。本例只跑一步，有沒有這行結果都相同。

`mean` 表示在多筆資料時取平均；本例一筆，所以與手算相同。

backward 之後，可以用 `print(model.weight.grad)` 看梯度，應該印出 `tensor([[-8.]])`，和手算一致。想確認 step 真的改了參數，要先存一份更新前的副本。若只寫 `alias = model.weight` 來存舊值，得到的不是副本，而是別名：同一個物件多了一個名字。`optimizer.step()` 會直接改寫參數裡的數字，別名也就跟著變。就像 Python 的 `b = a` 不會複製 list：之後改 a 的內容（例如 `a[0] = 9`），b 也跟著變。把下面兩行插在五行程式的 `optimizer.step()` 之前，step 之後再印出 `alias` 和 `snapshot`，就能看出差別。副本不取名 `before`，是因為完整程式已經有一個 `before`（見下一段），後面印出 `weight: 1.00 -> 1.80` 的那行要用它；若把這兩行和之後加的 print 插進 Colab 的完整程式，每一行都要和 `optimizer.step()` 對齊縮排：

```python
alias = model.weight                      # 別名：step 之後跟著變成 1.8
snapshot = model.weight.detach().clone()  # 副本：step 之後仍是 1.0
```

`detach()` 只把 tensor 從計算圖剪下來，數字仍和參數共用；`clone()` 才複製出自己的一份數字。所以只寫 `detach()` 存下的舊值，step 之後也會變成 1.8。完整程式則用 `before = model.weight.item()`：`.item()` 把只含一個數的 tensor 取成普通的 Python 數字，同樣不會跟著變。

backward 要沿著計算圖從 loss 一路走回參數，所以算 loss 時計算圖不能斷。`.item()` 取出的普通數字不記得自己是怎麼算出來的；`.detach()` 會刻意把 tensor 從計算圖剪下；NumPy 是另一個常用的數值計算套件，它的陣列也不記錄計算圖。若把預測先 `.detach()`、轉成 NumPy，或先用 `.item()` 取成普通數字再算 loss，計算圖就斷了。在本例，這三種做法都會讓程式報錯。例如 `loss = (prediction.item() - 4.0) ** 2` 算出的是普通的 Python 浮點數（float），根本沒有 `.backward()` 可以呼叫。記錄數值可以用 `.item()`；要拿來 backward 的 loss，計算時要保留 tensor。

## 兩個常混在一起的開關

常被混在一起的兩個開關是 `model.eval()` 與 `torch.no_grad()`。前者（和 `model.train()` 成對）只切換某些層的行為，不會關掉梯度記錄；後者才讓 PyTorch 不記錄計算圖。推論（拿模型做預測、不更新參數）時，通常兩個一起用。

`model.train()` 切到訓練模式，`model.eval()` 切到評估／推論模式。兩者只影響某些層：例如 Dropout（訓練時隨機把部分數值設成 0 的層）只在 train 模式遮值；BatchNorm（用平均與變異數把數值標準化的層）在 train 模式會更新它記錄的平均與變異數。本課程的程式沒用到這兩種層，只要記得：有些層在訓練與推論時行為不同。本節這個單純的線性層，兩種模式結果相同。

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

optimizer 內部記住的是**當初交給它的參數物件**，不是 `model` 這個名字。在 Colab 這類可以分格執行的 notebook 裡，若只重跑 `model = ...` 那一格，只是讓 `model` 這個名字改指向一個新模型；optimizer 手上仍拿著舊模型的參數。就像 Python 執行 `a = [1]; b = a; a = [2]` 之後，b 還是 `[1]`。這和前面「改 a 的內容，b 也跟著變」不同：`a = [2]` 沒有改原本 list 的內容，而是讓 a 改指向另一個新 list。所以重建模型時，要一起重建 optimizer。

若忘了重建，程式不會報錯，但 `optimizer.step()` 只處理舊模型的參數：新模型的權重一直不變，loss 也一直不降。完整程式最後建了一個替換用的新模型和新 optimizer，並用斷言確認新 optimizer 拿到的是新模型的參數。

## 核對輸出：看到這些就完成本節

上面「對應到五行程式」那兩段摘錄的程式沒有 `print`。下面 4 行是 Colab 裡完整程式印出的結果：

```text
x_shape=(1, 1), weight_shape=(1, 1)
prediction=2.00, loss=4.00, gradient=-8.00
weight: 1.00 -> 1.80; new_prediction=3.60; new_loss=0.16
eval still tracks gradients; replacement optimizer points to replacement model
```

第 1 行是正文講過的 shape，第 2、3 行就是本節手算的數字。第 2 行的 `prediction=2.00` 是更新前的預測；第 3 行的 `new_prediction=3.60` 是更新後的預測，也就是手算的 \(1.8\times2=3.6\)。第 1 行的 `(1, 1)`、正文寫的 `[1,1]`，和自己執行 `print(x.shape)` 會看到的 `torch.Size([1, 1])`，是同一個 shape 的不同印法。完整程式印出前 3 行後，會用斷言核對梯度、新權重、更新後的預測與新 loss。第 4 行要等所有斷言都通過才會印出，意思是：切到 eval 模式後，輸出仍連著計算圖；替換用的 optimizer 指向替換後的模型。

你的輸出和上面 4 行相同、而且沒有錯誤訊息，本節就完成了。之後章節的訓練，基本上就是把前面那五行程式（也就是訓練一步）重複很多次。模型和資料會變大；loss 可能改用更複雜的算法，optimizer 也可能換成別種（例如第 7 章的訓練改用 Adam）。收益是每一步都能追到具體數字；代價是這個線性問題不含影像、非線性與泛化（對沒看過的資料也做得好），不能用它判斷 CNN（卷積神經網路）的準確率。

## 自主練習與答案

**練習 1**：把學習率改成 0.25，先用手算預測新的 w、更新後的預測值與 new_loss，再執行驗證。這三個數都印在輸出第 3 行；第 2 行的 `prediction=2.00` 是更新前的預測，改了學習率也不會變。在 Colab 裡改「本節可修改的完整實驗」下面那一格；本機則改 `lesson_cases/00-warmup.py`，兩者是同一份程式。要改的是建立主 optimizer 的那行 `optimizer = torch.optim.SGD(model.parameters(), lr=0.1)`：把 `lr=0.1` 改成 `lr=0.25`。程式後段還有一個 `lr=0.1`，屬於替換用的 `new_optimizer`，和下面的斷言無關，不要改。

完整程式印出數字之後，用這三行斷言核對答案：

``` { .python data-excerpt="lesson_cases/00-warmup.py" }
assert abs(gradient + 8.0) < 1e-6
assert abs(after - 1.8) < 1e-6 and abs(new_loss - 0.16) < 1e-5
assert abs(new_prediction.item() - 3.6) < 1e-5
```

`gradient` 是 backward 之後記下的梯度，`after` 是更新後的 w，`new_prediction` 是更新後的預測。`new_prediction` 是 tensor；第三行先用 `.item()`（前面存 `before` 時介紹過）把它取成普通數字，和 `after`、`new_loss` 一樣用普通數字比較。`abs(a - b) < 1e-6` 表示 a 和 b 相差不到 10⁻⁶（`1e-5` 則是相差不到 10⁻⁵），用來容許小數計算的微小誤差。例如 `after` 實際上是 1.7999999523…，`new_prediction` 是 3.5999999046…，寫成 `after == 1.8` 或 `new_prediction.item() == 3.6` 反而不成立。

只改學習率就執行，前 3 行照樣會印出（第 3 行已換成新的 w、更新後的預測與 new_loss），接著第二行斷言出現 AssertionError。這是預期中的錯誤：第二、三行斷言還在核對舊答案 1.8、0.16 與 3.6。把這三個數字換成你預測的新 w、new_loss 與更新後的預測，再執行一次；不再報錯、第 4 行也印出來，就表示你的預測對了。只改了第二行、忘了改第三行，就換成第三行斷言報錯。第一行核對梯度，不用改，因為梯度在更新之前就算好了，和學習率無關。不要刪掉斷言來讓錯誤消失。

**練習 2**：只呼叫 `backward`，會改變 w 嗎？

??? note "參考答案"

    **練習 1**：\(w=1-0.25(-8)=3\)，預測 \(3\times2=6\)，loss 是 \((6-4)^2=4\)。第二行斷言要改成 `assert abs(after - 3.0) < 1e-6 and abs(new_loss - 4.0) < 1e-5`，第三行改成 `assert abs(new_prediction.item() - 6.0) < 1e-5`。三行斷言都通過時，輸出第 3 行是 `weight: 1.00 -> 3.00; new_prediction=6.00; new_loss=4.00`。前面說過，最理想的 w 是 2；w 從 1 跳到 3，越過 2 到了另一側同樣遠的地方，誤差大小同樣是 2，所以 loss 又是 4。方向正確仍可能走過頭。

    延伸：學習率多少，才能一步剛好走到 w=2？要讓 \(1-\eta(-8)=2\)，也就是 \(8\eta=1\)，所以 \(\eta=0.125\)。

    **練習 2**：不會。backward 只把梯度填進 `.grad`；要等 `optimizer.step()`，或你另外寫的手動更新程式，才會改變 w。

延伸閱讀（英文官方文件）：[PyTorch autograd 教學](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)（autograd 是 PyTorch 自動算梯度的機制，`backward` 就靠它）與 [SGD 官方說明頁](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式已於 2026-10-02 用 PyTorch 2.9.1+cpu 在 CPU 上執行過，程式裡的 assert 檢查全部通過。下面是那次印出的原始輸出；每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/00-warmup.json)

??? example "展開本次實際輸出"

    ```text
    x_shape=(1, 1), weight_shape=(1, 1)
    prediction=2.00, loss=4.00, gradient=-8.00
    weight: 1.00 -> 1.80; new_loss=0.16
    eval still tracks gradients; replacement optimizer points to replacement model
    ```

<!-- curriculum-evidence:end -->
