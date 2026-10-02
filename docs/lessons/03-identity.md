# ResNet identity shortcut：先確定真的能原樣透過

一串卷積每次都重算表示。如果某段目前不需要改變輸入，能否讓它直接透過，再慢慢學需要的修正？Residual block把輸入旁邊接一條短路徑，輸出變成「原本的值＋學到的修正」。前置只需知道卷積保留shape的方法與逐值相加；projection留到另一節。

歷史機制來自 [ResNet 原始論文](https://arxiv.org/abs/1512.03385)。原版常見block在相加後有ReLU。本節使用**相加後沒有ReLU**的教學block，刻意讓 \(F(x)=0\) 時可以對所有正負輸入精確驗證identity；這不等於完整原版ResNet配置。

[在 Colab 執行](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/03-identity.ipynb)，或 `PYTHONPATH=. python lesson_cases/03-identity.py`。CPU實驗先做人工零分支與梯度檢查，再用另一個隨機block訓練2步。前者證明算術，後者證明可更新，不是分類效果比較。

## 把「修正」與「完整答案」分開

用 \(x\) 表示輸入，\(F(x)\) 表示兩層卷積分支產生的修正，\(y\) 表示輸出：

\[
y=x+F(x).
\]

這裡的加法是逐個位置、逐個channel相加，不是把兩個tensor接起來。若x與F都是 `[B,4,8,8]`，y仍為 `[B,4,8,8]`；B是batch筆數。串接才會變成8個channel。

先看四個數字：x是 `[-2,-1,0,1]`，F全0，y仍是 `[-2,-1,0,1]`。若後面加ReLU，負數變0，便只能說相加前的值原樣透過，不能說整個block對任意x都是identity。這個差異會直接影響手算與程式是否一致。

## Shape 不能靠期待配對

本例每個卷積kernel3、padding1、stride1，channel數保持不變，所以兩支都有相同shape：

| 路徑 | 輸入 | 輸出 |
| --- | --- | --- |
| shortcut直接透過 | `[B,4,8,8]` | `[B,4,8,8]` |
| 3×3 Conv → ReLU → 3×3 Conv | `[B,4,8,8]` | `[B,4,8,8]` |
| 相加 | 兩個同shape張量 | `[B,4,8,8]` |

如果主分支把channel改成8，或用stride2縮到4×4，直接把x加上去就不再合法。PyTorch某些shape可以broadcast，但這不代表符合設計；在這種block應明確核對完整shape。

## 程式只多一個加號，學習目標卻不同

```python
def forward(self, x):
    return x + self.branch(x)
```

branch是兩個無bias的3×3卷積，中間一個ReLU。4channel時共 \(2\times4\times4\times9=288\) 個參數；直接shortcut沒有參數。它並沒有免除主分支計算，還多了一次逐值加法與儲存shortcut所需的記憶體。

若理想輸出是 \(H(x)\)，分支只需學 \(F(x)=H(x)-x\)。當H接近x時，修正可能比較好學；是否真的更容易，要用固定預算的對照實驗判斷，不能僅因公式好看就保證準確率提升。

## 不背微積分也能理解直接梯度路徑

把x增加很小一點 \(\Delta x\)，輸出變化是 \(\Delta x+[F(x+\Delta x)-F(x)]\)。第一項來自直接路徑，永遠存在；第二項來自主分支。這是「梯度可沿shortcut直接傳遞」的直覺來源，不表示全部梯度永遠等於1，或深層網路從此沒有最佳化問題。

人工零分支的實驗取loss為y中所有值的和。固定其他元素，只將x的某一個元素增加0.001，loss也增加0.001，因此每個輸入元素的梯度是1。案例印出 `input_gradient=[1,1,1,1]` 並assert相等。隨機F時還會有分支的影響，不能照搬這個答案。

把整個分支所有權重都設0是為了**機制驗證**。本例兩層無bias卷積全0，中間ReLU輸入也全0，兩層weight梯度都是0，分支無法靠這條路徑開始學；第二部分另建隨機初始化的block，避免把這個人工測試當成訓練建議。

第二部分的隨機block使用同shape全零target，MSE是每個輸出與零的差平方，再平均。它希望整個輸出接近零，F可能往−x修正；這個更新檢查的目標，刻意不同於第一部分F=0時保留x的identity算術測試。

## 核對、常見錯誤與收益

程式先核對負值保留，再確認隨機block輸出 `[2,4,8,8]`、末層權重有非零梯度且更新。這些透過便完成本節機制檢查。identity的收益是保留同shape資訊與提供直接路徑；限制是它只適合同shape，且兩步玩具MSE沒有回答圖片分類是否更好。

常見錯誤包括把addition寫成concatenation、忘了相加後ReLU會改負值、把「shortcut沒有參數」誤讀成整個block沒有參數，以及看到F=0測試便把正式網路全部初始化0。

## 自主練習與答案

在相加後加ReLU，保留人工x。答案y會變成 `[0,0,0,1]`，原本逐值identity assertion應失敗；在正值1處，sum loss對x的梯度為1，負值處則為0。修改相應assertion並解釋原因，別只刪除它。接著把主分支stride改2：答案是直接shortcut不再匹配，需要學下一節的projection。



<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/03-identity.json)

??? example "展開本次實際輸出"

    ```text
    x=[-2.0, -1.0, 0.0, 1.0], y=[-2.0, -1.0, 0.0, 1.0]
    input_gradient=[1.0, 1.0, 1.0, 1.0]; F=0 preserves negative values too
    step=0, shape=(2, 4, 8, 8), loss=1.0816
    step=1, shape=(2, 4, 8, 8), loss=1.0724
    random branch updated; this does not measure ResNet classification quality
    ```

<!-- curriculum-evidence:end -->
