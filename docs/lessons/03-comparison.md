# Plain／residual 對照：先控制比較條件

Residual有直接路徑，是否就一定更準？不能只看兩個不同模型的最後一個loss回答。容量、初始化、資料、訓練步數與計分方式都可能影響結果。本節只改「是否加入同shape shortcut」，做一次可追溯的小比較。前置是知道卷積與交叉熵；本頁重述shortcut的公式。

設計來源是 [ResNet 原始論文](https://arxiv.org/abs/1512.03385) 對plain與residual的研究。本節用4channel、3個block與人工色塊，省略BatchNorm、原版深度與相加後activation。它是區域性機制對照，不是重現論文的ImageNet結果。

[在 Colab 執行](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/03-comparison.ipynb)，或 `PYTHONPATH=. python lesson_cases/03-comparison.py`。CPU、16×16輸入、訓練8張／validation4張，兩模型各3步SGD。3步只驗證路徑與紀錄方法，不作架構排名。

## 唯一主要改動是什麼

兩模型用相同stem、3個相同結構的主分支F及相同分類head。Stem是輸入端3→4channel的卷積；head將每channel的空間平均轉成2類logits。每個block的兩個3×3卷積都保持 `[B,4,16,16]`，B為圖片數。

Plain輸出 \(y=F(x)\)，residual輸出 \(y=x+F(x)\)。兩者F都是Conv–ReLU–Conv，無相加後ReLU。所有可學參數完全相同，只有residual多逐值加法：

```python
correction = self.branch(x)
return x + correction if self.residual else correction
```

這個對照保留activation位置不變，避免同時改shortcut與非線性。若加入projection，模型還會多參數；那是另一個問題，應另做對照。

## 固定什麼，才能解讀觀察

資料是紅／藍色塊，類別0／1。Validation位置與訓練列表不同且未參與更新，但仍是很容易的人工分佈；4張不能代表真實圖片。相同batch按相同順序送入兩模型，SGD learning rate0.1、交叉熵、3次更新完全一致。

不能只在兩模型前各寫同一seed，就以為逐層參數必然相同：建構次序可能消耗不同的亂數。案例先建plain，再把它的 `state_dict` 複製到residual，並逐參數assert數值相等。兩個optimizer分別引用各自的參數，不能共用同一個optimizer輪流訓練。

比較紀錄包含更新前loss、stem梯度的長度、參數數、乘加數、額外加法、更新後validation accuracy與3步時間。梯度的長度用tensor的L2 norm：各梯度平方相加再開根號，用單一數字概括規模；它不是「梯度品質分數」。

## 參數數量相同，計算仍略有不同

Stem有112參數，每個F有288，head有10，因此兩者都是 \(112+3\times288+10=986\) 個參數。參數數量相同不代表兩者可表達的函式完全相同，shortcut改變了計算關係。兩者卷積與linear的乘加數相同：

\[
16\times16\times4\times3\times9
+6\times16\times16\times4\times4\times9
+4\times2=248840.
\]

Residual另多 \(3\times4\times16\times16=3072\) 次逐值加法／圖。程式的MAC估算不包含activation、空間平均及資料搬移；兩者「MAC相同」不等於所有成本完全相同。Shortcut通常需要保留輸入供相加，訓練記憶體也要考慮。

單次CPU時間容易受首次執行、快取、系統負載影響；案例只記錄本次數值。正式速度比較應先暖機、多次量測，固定batch及裝置，並分清forward和完整訓練步時間。

## 可核對輸出與應如何下結論

兩模型都應印 `params=986`、`MACs/image=248840`；plain的shortcut加法為0、residual為3072。每一步stem梯度應有限且非零，3步後stem參數改變。Validation accuracy會被印出，但assertion不要求residual勝出，因為那不是算術必然。

可能看到residual的初期梯度或loss變化較明顯，也可能兩者accuracy一樣。可寫的結論是「在這次seed、人工資料與3步預算下觀察到某數值」。不能寫「ResNet總是更準」，也不能把單次梯度norm更大解釋為泛化更好。

收益是學會控制變因與記錄成本；代價是這個快速實驗證據很有限。需要效果比較時，增加固定訓練預算與獨立資料，跑多個seed，先定評估規則，再決定是否保留shortcut。

## 常見錯誤、自主練習與答案

常見錯誤是plain較窄、residual較深卻把差異全歸給shortcut；或只讓其中一個模型多訓練幾次直到贏。另一種是用validation更新參數，失去獨立性。

練習把block數由3改成1，其他固定。答案參數為 \(112+288+10=410\)，MAC為 \(27648+73728+8=101384\)，residual額外1024次加法／圖。同步更新assertion與列印成本，先預測梯度路徑可能怎樣變，再實跑；一樣不應把3步勝負當成最終架構選擇。
