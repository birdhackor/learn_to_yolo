# 3.3 Plain／residual 對照：先控制比較條件

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/03-comparison.ipynb){ .md-button }

Residual 有直接路徑，是否就一定比 plain 更準？讀完你會知道：公平的對照要固定哪些條件、輸出怎麼讀，以及這種小實驗能寫出什麼程度的結論。前置是知道卷積與交叉熵；本節會重述 shortcut 的公式。

這裡的 plain 指沒有 shortcut、把 block 一個接一個疊起來的普通網路，每個 block 只輸出 \(F(x)\)；residual（殘差）網路則是每個 block 輸出 \(x+F(x)\)，也就是 [identity shortcut 那節](03-identity.md)的 residual block。「residual 是否一定更準」不能只比較兩個不同模型最後的 loss 來回答，因為很多條件都可能影響結果：容量（模型能表示多複雜規則的能力，大致隨層數、channel 數增加）、初始化、資料、訓練步數，以及計分方式（用 loss、accuracy 或其他指標判斷好壞）。所以本節只改「是否加入同 shape 的 shortcut」，做一次每個數字都能回頭核對來源的小比較。

設計來源是 [ResNet 原始論文](https://arxiv.org/abs/1512.03385) 對 plain 與 residual 的研究。本節用 4 個 channel、3 個 block 與人工色塊，並省略三樣東西：BatchNorm、原版的深度，以及相加後的 activation。BatchNorm（批次正規化）在原版每個卷積後都有，訓練時它用同一批資料算出的平均與標準差，把每個 channel 的數值調到穩定的尺度；activation 是層與層之間的非線性函數，本節指 ReLU。本節是局部的機制對照，不是重現論文在 ImageNet（約 128 萬張訓練照片、1000 類）上的結果。論文的 plain 網路也有 BatchNorm；本節沒有它，所以結果只代表本設定，加回後會怎樣，本節沒有測。

可以用頁首的按鈕在 Colab 執行，或在本機執行 `PYTHONPATH=. python lesson_cases/03-comparison.py`。兩種方式跑的是同一份完整程式：Colab 裡「本節可修改的完整實驗」下方那格，內容就是 `lesson_cases/03-comparison.py`；網頁上只摘錄了其中幾行。CPU、16×16 輸入、訓練 8 張／validation 4 張，兩個模型各用 SGD 更新 3 步。3 步只確認梯度傳得到 stem、參數有更新、紀錄完整，不拿來替架構排名。

## 唯一主要改動是什麼

兩個模型用相同的 stem、3 個結構相同的主分支 F，以及相同的分類 head。Stem 是輸入端的第一層，把 3 個 channel 變成 4 個；head 是一個 Linear（全連線層），把每個 channel 的空間平均轉成 2 類 logits。整個資料流如下，B 是圖片數：

輸入 `[B,3,16,16]` → stem（3×3 卷積＋ReLU）`[B,4,16,16]` → 3 個 block（都保持 `[B,4,16,16]`）→ 每個 channel 取空間平均 `[B,4]` → Linear `[B,2]`

第 k 個 block（k＝1、2、3）收到輸入 \(x\) 後，plain 輸出 \(F_k(x)\)，residual 輸出 \(x+F_k(x)\)。三個 \(F_k\) 結構相同、權重各自獨立，都是 Conv–ReLU–Conv：兩個 3×3 卷積都保持 `[B,4,16,16]`，block 輸出後不再接 ReLU。兩個模型所有可學參數完全相同，只有 residual 多了逐值加法：

```python
correction = self.branch(x)  # self.branch 就是 F：Conv–ReLU–Conv
# self.residual 為 True 時回傳 x+F(x)；為 False 時只回傳 F(x)
return x + correction if self.residual else correction
```

`self.residual` 是建模型時給的 True／False（plain 給 False，residual 給 True）。最後一行是 Python 的條件運算式 `A if 條件 else B`；它的優先順序比 `+` 低，所以 `x + correction` 會分成一組，整行等於 `(x + correction) if self.residual else correction`。plain 時，correction 就是整個 block 的輸出。

原版 ResNet 在相加之後還有一個 ReLU。若只替 residual 加上它，兩個模型就同時差了「有沒有 shortcut」和「多一個 ReLU」兩件事，結果不同時分不清是誰造成的。所以本節兩個模型的 ReLU 位置完全相同：stem 之後一個、每個 F 的兩個卷積之間一個，block 輸出後都不接 ReLU。若加入 projection，模型還會多出參數；那是另一個問題，應另做對照。

## 固定什麼，才能解讀觀察

公平比較就像理化實驗的控制變因。操縱變因只有「有沒有 shortcut」；控制變因是初始權重、資料、optimizer 與 learning rate、步數、ReLU 位置；應變變因是每步的 loss 與 stem 梯度，以及最後的 validation accuracy。另外也把參數數、乘加數（MAC）、加法次數與時間等成本一起記下來。

資料是程式畫的色塊圖。每張 16×16 黑底圖上有一個 8×8 的純紅（類別 0）或純藍（類別 1）方塊。8 張訓練圖只有 4 種，各重複兩次；4 張 validation 是同樣 4 種方塊往下移 2 列，沒有參與更新。分類只要看顏色，所以這是很容易的人工資料，4 張也不能代表真實圖片。每一步都把同一批 8 張訓練圖一次全部送入（不抽樣、不打亂），兩個模型完全一樣；SGD、learning rate 0.1、交叉熵與 3 次更新也完全一致。不過 optimizer 是兩個模型各建一個，分別引用各自的參數，不能共用同一個 optimizer 輪流訓練。

初始權重要相同，得先懂 seed。seed（亂數種子）固定後，每次執行產生的整串亂數都一樣；建模型時，各層依序從這串亂數取值當初始權重。完整程式只在開頭設一次 seed，接著連續建立兩個模型，residual 取到的是後面的亂數，權重和 plain 不同。就算在建每個模型前各重設同一個 seed，也要兩個模型建立各層的順序與大小完全相同才會對上；本例剛好相同，但只要其中一個多一層（例如 projection），後面每一層都會錯開。

所以完整程式不靠 seed 讓權重相同：先建 plain，再用 `residual.load_state_dict(plain.state_dict())` 把權重整份複製給 residual，並用斷言（assert）逐一檢查每個參數數值相等。`state_dict` 是記錄每個參數名稱（例如 `stem.weight`）與數值的字典。

紀錄裡的 loss 是每次更新前量的，validation accuracy 是 3 次更新後量的，時間是 3 步合計。梯度記的是 stem 梯度的 L2 長度（L2 norm）：把各梯度值平方相加再開根號，用一個數字概括梯度有多大，就像向量 \((3,4)\) 的長度是 \(\sqrt{3^2+4^2}=5\)。完整程式只算 stem 卷積的權重，不含 bias。

為什麼量 stem？stem 在最前面、離 loss 最遠，梯度要往回穿過全部 3 個 block：plain 只能經過 6 個卷積，residual 每個 block 還多一條直接路徑（見 [identity shortcut 那節](03-identity.md)），所以比較 stem 的梯度，最能看出梯度傳不傳得到最前面。L2 長度大只表示這一步參數被推得比較用力，不代表方向比較好，也不代表模型比較準。

## 參數數量相同，計算仍略有不同

參數數可以逐層算出來：

- stem：有 bias 的 3×3 卷積，3→4 channel，\(4\times(3\times9+1)=112\)；
- 每個 F：兩個無 bias 的 3×3 卷積，4→4 channel，\(2\times4\times4\times9=288\)；
- head：含 bias 的 Linear，4→2，\(4\times2+2=10\)。

因此兩個模型都是 \(112+3\times288+10=986\) 個參數。參數數量相同，不代表兩個模型算的是同一件事：同一組參數值，兩個模型算出的結果不同。例如把所有 F 的權重都設為 0：plain 每個 block 都輸出全 0，head 只剩 bias，每張圖得到一樣的分數；residual 每個 block 輸出 \(x+0=x\)，stem 的特徵原樣送到 head，紅圖和藍圖得到不同的分數。

兩個模型卷積與 Linear 的乘加數相同。一次乘加是把一個值乘上權重、再累加到答案，算法與 [VGG 風格小 CNN](01-small-cnn.md) 那節相同：

\[
16\times16\times4\times3\times9
+6\times16\times16\times4\times4\times9
+4\times2=248840.
\]

第一項是 stem（27648），第二項是 3 個 block 共 6 個卷積（每個 36864），第三項是 head（8）；bias 的加法不計入。Residual 每張圖另多 \(3\times4\times16\times16=3072\) 次逐值加法。

乘加數與加法次數是上面手算的結果，完整程式直接把它們寫成 print 裡的固定數字（也就是「寫死」）；參數數 986 才是完整程式實際數出、並用 assert 檢查的。完整程式裡對應的幾行如下，`name` 是模型名稱 `"plain"` 或 `"residual"`：

```python
# p 是一個參數 tensor，p.numel() 是它裡面的數值個數；全部加起來就是參數數
params = sum(p.numel() for p in model.parameters())
assert params == 986
# （中間省略：訓練 3 步、量 validation accuracy 等）
print(f"{name}: params={params}, MACs/image=248840, shortcut_adds/image="
      f"{3072 if name == 'residual' else 0}, validation_accuracy={acc:.2f}, "
      f"3_step_seconds={elapsed:.4f}")
```

乘加數不含 activation、空間平均與資料搬移，所以兩者「乘加數相同」不等於所有成本完全相同。Shortcut 要把輸入保留到相加為止，所以推論時的記憶體峰值（同一時刻最多占用多少記憶體）也要考慮：plain 的 block 輸入在第一個卷積算完後就能釋放，residual 卻要留到相加。訓練時則不同：第一個卷積在反向傳播時本來就要用這份輸入算梯度，有沒有 shortcut 都會保存，所以 identity shortcut 幾乎不增加訓練時的記憶體。

單次 CPU 時間容易受首次執行、快取（電腦暫存剛用過資料的高速記憶體）與系統負載影響；完整程式只記錄本次數值。本例 plain 先跑，很可能承擔了首次執行的額外成本，所以輸出中 plain 若比較慢，不代表它的計算比較多。正式的速度比較應先暖機（正式計時前先空跑幾次，丟掉這幾次的時間）、多次量測，固定 batch 與裝置，並分清 forward 和完整訓練步的時間。

## 可核對輸出與應如何下結論

兩個模型都應印出 `params=986`，這是完整程式數出並用 assert 檢查的；`MACs/image=248840` 與 shortcut 加法數（plain 為 0、residual 為 3072）則是固定印出的手算值。完整程式每一步都用 assert 檢查 loss 與 stem 梯度是有限值，也就是不是 inf（無限大）或 NaN（無法定義的數）；也檢查 stem 梯度的 L2 長度大於 0。3 步後再檢查 stem 參數確實改變。Validation accuracy 會印出來，但 assert 檢查不要求 residual 勝出，因為誰勝出不是算術上必然的結果。

讀 loss 時先記住一個基準。一張圖的交叉熵是 \(-\ln p\)，\(p\) 是模型給正確類別的機率；整批的 loss 再對所有圖平均。兩類分類時，若每張圖都給兩類各 50%，\(p=0.5\)，loss 就是 \(-\ln 0.5=\ln 2\approx0.693\)。所以 loss 停在 0.693 附近、accuracy 是 0.50，表示模型還在猜。

可能看到 residual 初期的梯度或 loss 變化較明顯，也可能兩者 accuracy 一樣。合格的結論要寫出條件與實際數字，例如：「在 seed 7、8 張人工色塊、SGD learning rate 0.1、只訓練 3 步的條件下，residual 的 stem 梯度 L2 長度約 0.15，plain 約 0.001；residual 印出的 loss 從 0.6922 降到 0.6855，plain 幾乎不變；兩者 validation accuracy 都是 0.50（4 張）。」這些數字取自本節最下方的執行紀錄，換電腦重跑時小數末位可能略有不同。不能寫「ResNet 總是更準」，也不能把單次梯度 L2 長度較大解釋成泛化較好。

收益是學會控制變因，並把參數數、乘加數等成本一起記下來；代價是這個快速實驗的證據很有限。需要比較效果時，要給兩個模型相同且更多的訓練步數、使用更多獨立資料、跑多個 seed，並先定好評估規則，再決定是否保留 shortcut。

## 常見錯誤、自主練習與答案

常見錯誤是 plain 較窄、residual 較深，卻把差異全歸給 shortcut；或只讓其中一個模型多訓練幾次，直到它贏。另一種是用 validation 更新參數，失去獨立性。

練習：把 block 數由 3 改成 1，其他設定不變。要改的是完整程式（Colab 裡「本節可修改的完整實驗」下方那格程式，或本機的 `lesson_cases/03-comparison.py`）第 23 行 `Classifier` 裡的 `for _ in range(3)`，改成 `range(1)`；第 57 行控制訓練步數的 `for step in range(3)` 不要動。接著先手算新的參數數、乘加數與加法次數，把 `assert params == 986` 的 986，以及 print 裡寫死的 `248840`、`3072`，改成你算出的值。執行前先預測兩個模型的 stem 梯度會怎樣變，再實跑核對。一樣不應把 3 步的勝負當成最終的架構選擇。

??? note "參考答案"

    參數數為 \(112+288+10=410\)，乘加數為 \(27648+73728+8=101384\)，residual 每張圖額外 1024 次加法。所以 `assert params == 986` 改成 `assert params == 410`，print 裡的 `248840` 改成 `101384`、`3072` 改成 `1024`。

    梯度的預測：只剩 1 個 block 時，plain 從 stem 到 head 只隔 2 個卷積，stem 梯度應明顯變大；residual 本來就有直接路徑，變化不大，所以兩者差距縮小。另外實跑一次的結果（不在本節的執行紀錄裡，數字會因電腦略有不同）：plain 的 stem 梯度約 0.04，比 3 個 block 時（約 0.001）大很多；residual 約 0.15～0.16，和原本差不多；兩者差距從約 150 倍縮到約 4 倍，validation accuracy 仍都是 0.50。

    還要注意：少建 2 個 block 後，head 取到的是亂數串裡不同位置的值，初始權重也跟著變（stem 與第 1 個 block 不變）。這正是前面講 seed 時說的情況：層的數量或順序一變，後面的層取到的亂數就錯開。所以 1 個 block 和 3 個 block 的結果之間，差的不只是 block 數。

## 延長到 40 步：這次真正學到了什麼

這是與上方 3 步檢查分開的補充實驗。本節 40 步沿用上方 3 步的全部設定（seed 7、CPU、同樣的初始權重、同樣 8 張訓練圖、SGD、learning rate 0.1），只把更新次數從 3 增加到 40；前 3 步的 loss 與本節最下方 3 步實驗的紀錄相同。

|模型|loss（第 1 次更新前→第 40 次更新前）|40 次更新後的訓練 accuracy（8 張）|40 次更新後的 validation accuracy（4 張）|
|---|---|---|---|
|plain|0.693791 → 0.692943|0.50|0.50|
|residual|0.692160 → 0.030739|1.00|1.00|

表中的 accuracy 是 40 次更新全部完成後，用 eval 模式量的。

![本次固定資料 40 步的實際 loss](../assets/diagrams/03-comparison-learning.svg)

圖說：藍線是 plain，幾乎水平停在 0.693（\(\approx\ln 2\)，等於亂猜）；橘線是 residual，前 15 步緩降，約第 20～30 步急降，第 40 次更新前約 0.031。橫軸是第幾次更新，每個點是該次更新「前」、在同一批 8 張訓練圖上量到的 loss：第 1 點還沒更新，第 40 點是第 40 次更新前的值。圖上的英文：Training step 是訓練步序，pre-update loss 是更新前的 loss，Fixed synthetic batch 是固定的一批人工資料。

想自己重跑：先執行本節 Colab 最上面的環境格（下載教材程式、安裝套件的那一格），再按「＋程式碼」（英文介面是「+ Code」）新增一格，貼上這行指令：`!python scripts/run_learning_extensions.py --section 03-comparison`。開頭的 `!` 表示把這行交給系統命令列執行，不是當成 Python 程式。原始完整紀錄：[40 步結果](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-comparison-learning.json)。

### 結果怎麼讀

**觀察。** 兩個模型都是 986 個參數、同樣的起點、同樣用 SGD 與 learning rate 0.1。plain 40 步後 loss 幾乎沒動，一直貼著 \(\ln 2\approx0.693\)，8 張訓練圖全部猜成類別 1，訓練與 validation accuracy 都是 0.50。residual 的 loss 降到約 0.031，表示模型給正確類別的機率已接近 1；訓練 8 張與 validation 4 張全對。兩個模型每次更新的梯度都是有限值、L2 長度不為 0，權重也確實改變；所以 plain 並不是程式沒在更新，而是更新幾乎沒有效果。

**機制。** 3 步實驗的紀錄裡，三步的 plain stem 梯度都只有 residual 的約 1/150。訊號（各層輸出的數值）會一層層變小，源頭是 PyTorch 預設初始化給的權重偏小：block 裡每個 4→4 的 3×3 卷積，一個輸出值是 4×9＝36 個「權重×輸入」相加，權重取自 −1/6 到 1/6 的均勻分布（這個範圍內每個值出現的機會都一樣），平方的平均只有 \((1/6)^2\div3=1/108\)。權重有正有負、彼此獨立，36 項相加時交叉相乘的部分平均會互相抵消，所以每過一個卷積，數值平方的平均約縮成 36×1/108＝1/3，中間的 ReLU 把負值變成 0，又再少約一半。一個 block 合起來約縮成 1/18，數值大小約剩 1/4。plain 沒有 shortcut 把 \(x\) 加回，本設定又沒有 BatchNorm 把尺度拉回，所以訊號每過一個 block 都越來越接近 0。傳到 head 時特徵已經非常接近 0，head 算出的分數幾乎只剩它自己的 bias，8 張圖拿到幾乎一樣的分數；這組 bias 稍微偏向類別 1，於是 8 張全猜類別 1。反過來，梯度從 loss 傳回 stem 時也要穿過同樣 6 個卷積，每過一個 block 也明顯變小。residual 每個 block 都把 \(x\) 原樣加回（identity shortcut 那節的直接路徑），stem 算出的紅／藍特徵能一路送到 head，梯度也能沿 shortcut 傳回 stem。

**限制。** 這只代表本設定：沒有 BatchNorm、PyTorch 預設初始化（建立層時自動給的隨機權重）、3 個 block、人工色塊。如開頭所說，加回 BatchNorm 會怎樣，本節沒有測；改用權重較大的初始化（例如 ResNet 論文用的 He 初始化）會怎樣，本節也沒有測。validation 全對只表示方塊往下移 2 列也答得對，離真實照片的泛化還很遠；曲線也只畫這批訓練圖的 loss。所以不能據此宣稱真實圖片或更深網路的泛化效果。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式已於 2026-10-02 用 PyTorch 2.9.1+cpu 在 CPU 上執行過，程式裡的 assert 檢查全部通過。下面是那次印出的原始輸出；每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-comparison.json)

??? example "展開本次實際輸出"

    ```text
    plain step=0, loss=0.6938, stem_grad_norm=0.000987
    plain step=1, loss=0.6937, stem_grad_norm=0.001004
    plain step=2, loss=0.6936, stem_grad_norm=0.001014
    plain: params=986, MACs/image=248840, shortcut_adds/image=0, validation_accuracy=0.50, 3_step_seconds=0.0124
    residual step=0, loss=0.6922, stem_grad_norm=0.146834
    residual step=1, loss=0.6888, stem_grad_norm=0.147695
    residual step=2, loss=0.6855, stem_grad_norm=0.146226
    residual: params=986, MACs/image=248840, shortcut_adds/image=3072, validation_accuracy=0.50, 3_step_seconds=0.0081
    Same initial weights/data/optimizer/steps; 3 steps and 4 validation images do not rank architectures.
    ```

<!-- curriculum-evidence:end -->
