# 03-comparison 閱讀 review

範圍：只讀本課文字與案例程式；本頁沒有引用比較圖。以具基本 PyTorch／CNN 知識、首次接觸 ResNet 的讀者檢查。

實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/03-comparison.py` 通過；兩者均為 986 個參數、248840 MAC／圖，shortcut 加法為 0／3072，validation accuracy 均為 0.50。文中的 1-block 練習答案也正確。比較條件與「不能用 3 步結果排名」的結論清楚。以下兩點建議修正：

1. **「容量相同」容易被理解成表達能力完全相同。** `docs/lessons/03-comparison.md:13` 的「所有可學參數完全相同」及 `:30` 的「容量相同」沒有區分參數數量、初始值與模型能表示的函數。加入 shortcut 會改變模型的函數形式；相同參數數量不保證相同表達能力，而訓練後參數值也會分歧。建議把標題改成「參數數量相同，計算仍略有不同」，第 13 行改成「兩者可學參數的結構、數量與初始值相同；加入 shortcut 後，前向輸出與訓練更新可以不同」。這能保留公平控制的重點，避免新手把參數數量當成容量的完整定義。

2. **程式沒有驗證文字要求的梯度有限性。** `docs/lessons/03-comparison.md:46` 要求每一步 stem 梯度有限且非零，但 `lesson_cases/03-comparison.py:63` 只檢查 `grad_norm > 0` 與 loss 有限；`grad_norm` 若為正無窮仍會通過。建議在取 norm 前加入 `assert torch.isfinite(model.stem.weight.grad).all()`，再保留 norm 大於零與 loss 有限的檢查。第 28／46 行也可明寫「stem 權重梯度」，因為程式沒有將 stem bias 的梯度納入這個數字。


## 作者修訂

容量相同已改為參數數量相同，明說不代表可表達函數完全相同；新增首層gradient的isfinite.all()檢查。Validation位置與train不同，另有逐圖無重複assertion；三步CPU對照通過，兩者validation accuracy均0.5，不作架構排名。
