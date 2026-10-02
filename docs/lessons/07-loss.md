# Grid MiniYOLO：loss 必須能手算

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/07-loss.ipynb){ .md-button }

前置：[targets](07-targets.md)。本節在訓練前用人工 logits 驗證三件事：數值是否吻合、哪些位置收到梯度、空圖能否 backward。只看到 total loss 是有限數字不足以證明 mask 正確。

歷史來源：[YOLOv1 原論文](https://arxiv.org/abs/1506.02640) 的聯合定位與分類目標。本章簡化為 normalized 座標 MSE、全格 objectness BCE、正格分類 CE；沒有原版平方根寬高、IoU confidence target 或多框責任規則。以下公式就是本 repo 的 `grid_loss`，不能當成完整 YOLOv1 loss。

## 一個紅框，每格七個零 logits

B 是 batch 圖片數、S 是每邊格數、C 是類別數；本例 B=1、S=4、C=2。logit 是尚未轉成機率的實數；sigmoid 把單一值轉成 0 到 1，softmax 把多類值轉成總和為 1 的機率。`Npositive=pos.sum()` 是整批正格數。mask 是布林遮罩，True 位置才學框與類別。

| 欄位 | 原 shape | 正格 mask 後 |
| --- | --- | --- |
| prediction | `[B,S,S,5+C]` | 框 `[Npositive,4]`、類別 `[Npositive,C]` |
| positive | `[B,S,S]` bool | 本例只有 `[0,1,1]` 為 True |
| box | `[B,S,S,4]` float | `[Npositive,4]` |
| objectness | `[B,S,S]` float | 不取 mask，全格計算 |
| class_ids | `[B,S,S]` long | `[Npositive]` |

`...` 保留 batch、y、x 三軸，`:4` 取通道0–3，通道4是objectness，`5:`取類別。Boolean indexing 將所有正格收成清單。`build_targets([scene],4,64,2)` 的後三參數是grid邊長、圖片pixel邊長、類別數。

沿用 `[8,12,24,28]`，其正格 `(y=1,x=1)` 的框 target 是 `[0,.25,.25,.25]`，class=0。原框順序是 pixel xyxy，中心 `(16,20)`、寬高 `(16,16)`。每格 `64/4=16` pixel，所以格內偏移 x=`16/16−1=0`、y=`20/16−1=.25`；wh 都是 `16/64=.25`。四個 target 無單位，前兩項相對格子、後兩項相對全圖。把整個 `[1,4,4,7]` prediction 填零。sigmoid(0)=.5，兩類 softmax 都是 .5。

1. 框 loss 只取正格，平均它的四座標平方誤差：`(.5²+.25²+.25²+.25²)/4=.109375`。座標已正規化，不是 pixel 平方。
2. objectness loss 用全部 16 格：1 正、15 負。零 logit 對 target=0 或 1 都得到 `ln(2)=.693147`；全格 mean 因而仍是 .693147。
單格 BCE 可讀成 `−[t ln(p)+(1−t)ln(1−p)]`，p=sigmoid(z)，t是0或1。p=.5時，正格取`−ln(.5)`，負格取`−ln(1−.5)`，所以相等。這裡 ln／log 都是自然對數。

3. 類別 loss 只取正格：正確類別機率 .5，所以 `-log(.5)=.693147`。負格不計 class loss。
4. 本章固定 `total=5×box+objectness+classification`，故 `1.933169`。權重 5 是教學基線的設計值，沒有自動找出最好的能力。

三項都無量綱，但其有效樣本數不同：box 平均 `Npositive×4`、class 平均 `Npositive`、objectness 平均 `B×S×S`。若改成 sum，batch 或物件數改變就會改變梯度尺度。若將框改成 pixel 座標，16 pixel 的誤差平方會比 normalized 誤差大很多，也需要重新考慮權重。

```python
import torch.nn.functional as F
pos = target['positive']
if pos.any():
    box = F.mse_loss(pred[..., :4].sigmoid()[pos],
                     target['box'][pos], reduction='mean')
    cls = F.cross_entropy(pred[..., 5:][pos],
                          target['class_ids'][pos], reduction='mean')
else:
    box = pred[..., :4].sum() * 0
    cls = pred[..., 5:].sum() * 0
obj = F.binary_cross_entropy_with_logits(pred[..., 4],
                                         target['objectness'], reduction='mean')
total = 5 * box + obj + cls
```

## 梯度讓數字成為方向

BCE 對 objectness logit 的梯度是 `(sigmoid(logit)-target)/(B×S×S)`，本例分母纔是16。因此正格為 `-.03125`，背景為 `+.03125`。更新是 `z_new=z−學習率×gradient`，optimizer 做減法，會提高正格 logit、降低背景 logit。負格的四個框梯度必須是零；若不為零，表示背景正在學某個無意義框。

scalar 是 shape `[]` 的單值 tensor；`sum()*0` 保留與 prediction 的計算關係，梯度為0。空圖沒有正格：box 與 class 回傳連到計算圖的 scalar 0，objectness 仍是 .693147。對空 tensor 直接做 mean 會得到 NaN，不能靠刪掉空圖解決。這個分支讓全背景 batch 仍有有效梯度。

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/07-loss.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-loss.py`，應核對 `box .109375 / objectness .693147 / classification .693147 / total 1.933169`、正負梯度 `-.03125/.03125`，並透過空圖 backward。這是 loss 的人工單元實驗，沒有 AP 結果。

收益是每一項監督和梯度方向都能被驗證；代價是 MSE 未直接表達框重疊品質，平均 BCE 在稀疏場景也可能使背景訊號佔優。不要在尚未確認資料與 mask 時先調權重。第 11 章才單獨替換定位目標，看 IoU 類 loss 改了什麼。

常見錯誤：先 sigmoid 再傳給 `BCEWithLogitsLoss`；先 softmax 再傳給 CE；把背景 class=-1 也送進 CE；空正格除以零。這些問題可能產生有限 loss，仍需梯度與人工數值檢查。

自主練習：把 batch 複製成兩張完全一樣的圖，三項 mean loss 變多少？答案：不變；若使用 sum 才會翻倍。把正格類別兩個 logits 都加 10，CE 是否變？答案：不變，softmax 對共同平移不變。
