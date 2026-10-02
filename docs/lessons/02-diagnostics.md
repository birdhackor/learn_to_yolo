# 訓練診斷：把下一個檢查點說清楚

「loss 不動」不是一個完整診斷。可能是標籤錯、梯度斷、沒有更新、learning rate 不合適，也可能是資料太難。先查能直接核對的事項，比同時換 optimizer、模型與資料更容易找出原因。前置是能看懂 tensor shape 與 forward／backward／step；本節不需要記住 CNN 架構。

[在 Colab 執行](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/02-diagnostics.ipynb)，或執行 `PYTHONPATH=. python lesson_cases/02-diagnostics.py`。CPU 小實驗先製造兩個程式錯誤，再用8筆二維人工資料訓練20步。它不是影像模型的效果測試，而是刻意安排的診斷案例。

## 四個檢查點，各自回答不同問題

1. **一個 batch 的圖片與標籤。** 標籤（label）是每筆資料的正確類別id。畫出資料，核對類別、顏色、shape、dtype、畫素範圍。本課兩輸出的cross entropy用long的0／1類別id，不是1／2；其他二元loss可能用不同契約。偵測還要畫框，看空圖、越界與變換後的位置。
2. **一次 forward／loss／backward／step。** logits是尚未轉機率的原始類別分數，shape `[B,C]`；label是 `[B]`，B為batch筆數、C為類別數。loss與梯度應有限，至少一個預期可學參數應有非零梯度且更新後真的變了。
3. **少量資料 overfit。** 刻意反覆訓練幾筆資料，充分擬合或記住它們，檢查訓練管線；本例train accuracy達1.00，不只是loss小降一點。若仍失敗，優先查監督、容量、學習率及訓練路徑。
4. **held-out 評估。** 在沒有參與更新的資料上推論。eval切評估模式；no_grad暫停記錄反傳所需的運算。Validation是監測與選設定的驗證集，test是最後保留的測試集；不要用test反覆挑超參數。

前兩步可以在CPU完成，GPU錯誤也可先縮成這樣的批次揭露label或shape問題。不要先改一堆設定，卻沒有記錄哪一步讓問題消失。

## 失敗一：反傳成功，前半模型卻沒有學

案例把模型拆成body與head；body是 `[2]→[3]`，head是 `[3]→[2]`。錯誤程式如下：

```python
logits = head(body(x).detach())
loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()
```

`detach()` 保留數值、切斷前半段的梯度路徑。head仍然可以學，整體loss甚至可能下降，但 `body.weight.grad is None`。這與「有梯度但數值很小」不同：None通常表示該參數沒有接進本次反傳，或本次未使用。本例只驗證梯度連通與參數更新，沒有量測這個斷圖模型的loss下降。

拿掉detach後，程式檢查body的梯度非零，並比較更新前後的副本。只看loss不能抓到這個區域性凍結。另一方面，刻意凍結pretrained backbone時可能需要它；是否錯誤由實驗目的決定。本課模型從零訓練，body應參與學習。

## 失敗二：把資料契約當成超參數問題

兩類模型的label若是 `[0,2]`，2越界，因為輸出只有索引0與1。案例在loss前驗證 \(0\leq label<C\)，直接印出錯誤原因。降低learning rate不能修好類別id；把C改成3也未必對，應先核對類別對映表。

還要留意loss接收的單位：cross entropy接logits，不接argmax結果；argmax是離散決策，不能把它當作訓練預測。畫圖時可使用argmax，算loss時保留原始分數。

## 失敗三：少量資料記住了，新資料仍全錯

讓每筆資料有兩個特徵 \((a,b)\)。a是較弱但穩定的物件訊號；b是更強的角落背景訊號。訓練資料如下，validation只反轉b：

| 類別 | 訓練特徵 | held-out 特徵 |
| --- | --- | --- |
| 0 | `[-0.2,-1]` | `[-0.2,+1]` |
| 1 | `[+0.2,+1]` | `[+0.2,-1]` |

模型是無bias線性分類器，shape `[8,2]→[8,2]`，每類同一個點重複四次，共8筆、只有兩種不同輸入；權重全0開始，用SGD、learning rate0.2。由於b數值較大且與標籤完全相關，模型能快速依靠它。每步記錄的是**更新後**的train／validation loss，避免把不同時間點當同一條曲線比較。Step從0計，step19就是第20次更新。

20步後可核對 `train_accuracy=1.00`、`validation_accuracy=0.00`，且validation loss上升。這個差異是人工設計的分佈變化，不能聲稱真實資料一定有相同問題。它證明「小批次學會了」只是管線與容量的線索，還沒證明泛化。

## 收益、代價與停點

這套檢查順序把資料／程式問題、最佳化問題、泛化問題分開。成本是先花時間畫圖、儲存梯度與分項紀錄；收益是下一步有明確理由。一次梯度很小不能直接推論梯度消失：先看尺度、資料、layer與多個step。總loss下降也可能掩蓋某個分項停滯；後面定位與偵測會分別記錄分類、框、背景loss。

本節透過斷圖檢測、修復更新、label檢查及人工train／held-out差異即可停止。若小批次overfit失敗，尚不值得用大資料長訓練掩蓋它。

## 自主練習與答案

將訓練集新增8筆：4筆 `[-0.3,+1]` 標0、4筆 `[+0.3,-1]` 標1，與原validation的a=±0.2數值不同。先預測會依賴哪個訊號。答案：b不再穩定提供答案，模型更有理由使用a；但a幅度較小，固定20步不一定讓機率很有信心。維持原validation只用於評估，勿把它原封不動加入訓練；否則即使accuracy改善也只是看過同筆資料，不能再稱held-out。本練習仍是兩維人工資料的獨立點檢查，不代表真實圖片泛化。若 `body.weight.grad` 為None，下一個檢查點是detach／no_grad／參數是否參與forward，不是先增加模型寬度。

原始工具檔案：[PyTorch zero_grad](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)、[CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
