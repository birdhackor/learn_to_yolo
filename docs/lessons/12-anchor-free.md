# C.12.1 Anchor-free：從候選點量出四條邊

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/12-anchor-free.ipynb){ .md-button }

到目前為止，框既可直接寫成中心與寬高，也可相對於 anchor 尺寸修正；兩種都能還原物件。若希望換資料長寬比時少一份尺寸模板設定，還能怎樣用特徵位置描述框？這次保留每格的候選位置，把答案改成**從這個點到框左、上、右、下四邊有多遠**。

**Anchor-free**指不用預設的寬高 anchor，不等於沒有參考位置。模型仍在固定候選點輸出框，也仍要決定哪個點學哪個物件。

這種點到四邊的表示，在 2019 年的 [FCOS（Fully Convolutional One-Stage Object Detection，全卷積單階段物件偵測）](https://arxiv.org/abs/1904.01355)已使用；Ultralytics 於 2023 年發布的 YOLOv8 也用它。YOLOv8 沒有正式論文，本頁依固定原始碼對照，不把這個小實驗當成完整 YOLOv8。

## 一個物件的數字旅程

假想輸入 64×64pixel，特徵 8×8，相鄰候選點隔 stride=8pixel。通常把每格中心當點，位置是 `((column+0.5)×stride,(row+0.5)×stride)`。第(3,3)格（欄、列從 0 算）的點因此為 `(28,28)`；程式用 `point` 存它，沒有真的計算圖與特徵。

真值框是 `[12,16,40,36]`。固定這個點，沿四個方向量到框邊，pixel 距離為 `[28−12,28−16,40−28,36−28]=[16,12,12,8]`。除以 8 換成特徵格單位，得到 **`ltrb=[2,1.5,1.5,1]`**，ltrb 依序是 left、top、right、bottom。1 格=8pixel，邊界不必落在格線，所以距離可以不是整數。

![框內點(28,28)的四距離，以及框外點(44,28)的負右距離](../assets/diagrams/12-anchor-free.svg)

黑點在第(3,3)格中央，卻不在框中心 `(26,26)`：左 16 比右 12 長，上 12 比下 8 長。這個不對稱例子讓左右、上下的對應可檢查；若四邊都相等，寫反順序也看不出來。點不必是物件中心，框內其他點也能量出四個正距離。

解碼反過來，把格距離乘 stride，再從點減左、上，加右、下。記點為 `(px,py)`，stride 為 s：

\[
x_1=p_x-ls,\quad y_1=p_y-ts,\quad x_2=p_x+rs,\quad y_2=p_y+bs.
\]

代入得 `[28−2×8,28−1.5×8,28+1.5×8,28+1×8]=[12,16,40,36]`。這個框不需要選 anchor 寬高，尺寸由 `(l+r)×s`、`(t+b)×s` 得到。

``` { .python data-excerpt="lesson_cases/12-anchor-free.py" }
def decode(points, distances, stride):
    left_top, right_bottom = distances[..., :2], distances[..., 2:]  # [l,t] 和 [r,b]
    return torch.cat((points - left_top * stride, points + right_bottom * stride), -1)
```

下面從 GT 造出訓練答案 `distance`，核對它與手算相同，再 decode 回原框。

``` { .python data-excerpt="lesson_cases/12-anchor-free.py" }
point = torch.tensor([[28., 28.]])         # 候選點的畫素 x,y：第 (3,3) 格的中心
gt = torch.tensor([[12., 16., 40., 36.]])  # 真值框 xyxy，畫素
# gt[:, :2] 是 (x1,y1)，gt[:, 2:] 是 (x2,y2)
# point - gt[:, :2] = (px-x1, py-y1)：左、上；gt[:, 2:] - point = (x2-px, y2-py)：右、下
# -1：沿最後一軸串成 [l,t,r,b]；/ 8：除以 stride，換成格單位
distance = torch.cat((point - gt[:, :2], gt[:, 2:] - point), -1) / 8  # 由真值算出的 target
assert torch.allclose(distance, torch.tensor([[2., 1.5, 1.5, 1.]]))
assert torch.allclose(decode(point, distance, 8), gt)
```

本頁 point 用 pixel，只有 distance 乘 stride，解出的整框不能再乘一次。Ultralytics 則先用格單位的 point 與 distance 解碼，再將整框乘 stride；兩種各自一致便可，不能混用。

完整 head 的規格與這個單點程式要分開：

| 材料 | 完整 head shape | 單位／角色 |
| --- | --- | --- |
| 候選 points | `[P,2]` | pixel x、y；位置固定、各圖共用 |
| 四距離輸出 | `[B,P,4]` | 特徵格 ltrb |
| 解碼框 | `[B,P,4]` | pixel xyxy |
| 類別 logits | `[B,P,C]` | 每點每類分數；本例沒有 |

B 是圖數、P 是點數、C 是類別數；8×8 每格一點有 P=64，多尺度可相加。本程式只有 P=1、沒有圖片與 batch 軸，distance 和 decoded 都是 `[1,4]`，1 是點數；gt 也是 `[1,4]`，1 則是 GT 數。

## 為何仍要 assignment

本例要求四距離都**大於 0**，因此 `x1<px<x2`、`y1<py<y2`，解出的框包住點、寬高為正。這個表示有明確責任條件：點必須在要學的 GT 內。

圖中另一紅點 `(44,28)` 在 GT 右側，對 GT 的 pixel ltrb 為 `[32,12,−4,8]`，右邊距離−4pixel、即−0.5 格。下文用 softplus 確保距離為正，這點就不能表示 GT，應從正樣本中排除；程式最後斷言至少一距離小於 0。拿掉尺寸 anchor，不能略過這個選點問題。

框內也可能有很多點，每點各可輸出同一個框；訓練仍要選哪些當正樣本、各學哪個 GT，其餘學背景，稱為 **assignment（責任分配）**。第 7 章只讓框中心所在格負責：sigmoid 中心偏移限在格內，只有那格能表示這個中心。改成四邊距離，框內任何點都能表示同一框，提供了多點學同一物件的可能，也增加選擇責任的工作。

最簡單可只選框內點，或再限縮到中心附近；YOLOv8 則按當下分類分數與框重疊挑合適點，每步可能改變，稱動態 assignment。下一頁先建立分類／定位兩分支，再由 [12.3](12-assignment.md) 示範挑點。此單點實驗不含分類或背景 loss。

第 7 章本來也不用預設尺寸，依此定義也是 anchor-free；與本頁的共同點只是沒有尺寸模板，表示與責任規則仍不同，不能都叫 YOLOv8。

## 真的更新四個距離

現在不是隻把答案編碼再還原，而是從錯誤的初值學回答案。程式沒有 CNN，把 `raw [1,4]` 四原始數設為 `nn.Parameter`，全部從 0 開始，由 SGD 更新；point 與 GT 不更新。

先用 **softplus(x)=ln(1+exp(x))** 把 raw 變正距離。任意 x 都得到正數，raw=0 時距離為 ln2≈0.693。這是本節確保正距離的教學選擇，實際 YOLOv8 用分佈表示，後面的 DFL 頁才換它。

??? note "softplus 和第 9 章的 exp 有什麼不同"

    兩者的輸出都恆為正。exp 的值和斜率都隨 x 指數暴增，第 9 章就提過 exp 可能解出很大的框。softplus 在 x 很大時約等於 x，在 x 很負時趨近 0；它的斜率是 sigmoid(x)：ln(1+exp(x)) 的導數是 exp(x)/(1+exp(x))＝1/(1+exp(−x))，介於 0 和 1 之間，不會超過 1。

為量這四個距離差多少，用 **Smooth L1**：對每邊誤差 d=預測−target，|d|<1 時是 0.5d²，|d|≥1 時是|d|−0.5，最後四邊平均。大誤差斜率固定±1，不會像 MSE 隨誤差一直變陡；接近答案時用平方段，斜率隨 d 接近 0，避免純 L1 在 0 的折角讓小步仍來回跳。分界 1 是 PyTorch 預設，兩段值與斜率在這裡接合。

初始 `[0.693,0.693,0.693,0.693]` 比四個 target 都短，每邊都需變長，並非因 target 不對稱纔有梯度。左邊誤差最大，初始 raw 梯度−0.125；上、右同為約−0.1009，下約−0.0384。SGD 減負梯度，四 raw 都增大，各自往各自答案靠近。

??? note "手算初始 loss 0.376236"

    初始四邊預測都是 softplus(0)≈0.6931，target 是 `[2,1.5,1.5,1]`，所以 d≈[−1.3069, −0.8069, −0.8069, −0.3069]。只有左邊 |d|≥1，取 |d|−0.5；其他三邊取 0.5×d²。各邊約 [0.8069, 0.3255, 0.3255, 0.0471]，加起來約 1.505，除以 4 約 0.376，就是執行紀錄的初值 0.376236。

??? note "初始梯度怎麼算"

    由連鎖律（chain rule），loss 對第 i 邊 raw 的梯度＝1/4 × Smooth L1 在 d 的斜率 × softplus 在 0 的斜率。1/4 來自四邊取平均；Smooth L1 的斜率在 |d|<1 時是 d，在 |d|≥1 時是 ±1；softplus 的斜率是 sigmoid(x)，在 0 是 0.5。

    - 左：1/4 × (−1) × 0.5 = −0.125（|d|≥1，斜率取 −1）
    - 上：1/4 × (−0.8069) × 0.5 ≈ −0.1009
    - 右：和上邊相同，≈ −0.1009
    - 下：1/4 × (−0.3069) × 0.5 ≈ −0.0384

    四個梯度都是負的。SGD 減去梯度，會把四個 raw 都調大，預測距離跟著變長，往 target 靠近。就算 target 是對稱的 `[1.5,1.5,1.5,1.5]`，梯度也是四個 −0.1009，不是零：梯度不為零是因為預測不等於 target，和 target 對不對稱無關。

``` { .python data-excerpt="lesson_cases/12-anchor-free.py" }
raw = nn.Parameter(torch.zeros(1, 4))      # 四個要學的原始值，從 0 開始
optimizer = torch.optim.SGD([raw], lr=.5)
before = F.smooth_l1_loss(F.softplus(raw), distance).item()  # 更新前的 loss
for _ in range(80):                        # 更新 80 次
    optimizer.zero_grad()                  # 先清掉上一次的梯度，否則會累加
    # F.softplus(raw) 是四個正距離（格單位），和 target distance 比較
    loss = F.smooth_l1_loss(F.softplus(raw), distance)
    loss.backward()
    assert torch.isfinite(raw.grad).all()  # 梯度的每個數都有限
    optimizer.step()
after = F.smooth_l1_loss(F.softplus(raw), distance).item()   # 更新後的 loss
assert after < before / 100
```

每步檢查 raw 梯度有限，80 次更新後 loss 需小於初始百分之一。保存結果為 **0.376236→0.000012**；學到距離再 decode，框約 `[12.01,16.03,39.97,35.94]`。這表示四個數能依此 loss 更新到接近 target，沒有圖片，所以沒有學會從圖找物件的證據。程式對 learned box 只印出、沒有斷言，需與原框自己比對。

執行 `PYTHONPATH=. python lesson_cases/12-anchor-free.py` 或頁首 Colab，依序核對 target `[2,1.5,1.5,1]`、decoded target `[12,16,40,36]`、loss、learned box 與框外點訊息。shape `[1,4]` 正確只是一部分；不對稱 target 的逐值 encode/decode 才檢查順序和單位。實際 YOLOv8 用 CIoU 加 DFL，本例 Smooth L1 只服務四距離更新。

## 收益、代價與容易搞錯的地方

不需尺寸清單，就少了第 9 章的聚類、保存與換分佈後重挑 anchor 工作；代價轉到點密度、可表示距離與選點。stride 太大，小框可能沒有任何內部候選：stride 16 的格中心只有 8、24、40、56，框 `[26,26,36,36]` 內沒有點；stride 8 有 `(28,28)`，纔可用正距離表示它。

本例 softplus 距離沒有固定上限；若換有限刻度的分佈，每邊可表示多長便取決於刻度數與 stride，[DFL](12-dfl.md)會處理這個成本。密集點仍可能同報一個物件，所以 anchor-free 不保證 NMS-free；尺寸表示與推論去重是不同工作，這次沒有 AP 或 NMS 實驗。

Ultralytics 變數 `anchor_points` 只指參考點，不是第 9 章有寬高的 anchor。其他常見錯誤是 ltrb 當 xyxy、左右對調、格當 pixel 或讓框外點學正距離。單位、順序與點的資格要在訓練前對齊。

自主練習：保持框不變，stride 改為 4，點改成 stride 4 的第 (5,5) 格中心 `(22,22)`（`(5+0.5)×4=22`）。先算距離，再修改程式。要改的是完整程式（`lesson_cases/12-anchor-free.py`，也就是 Colab 裡那份）`main()` 裡的這幾處，也可以先把 stride 集中成一個變數：

- `point` 改成 `[[22., 22.]]`；
- 算 `distance` 那一行的 `/ 8` 改成 `/ 4`；
- 第一個斷言 `assert torch.allclose(distance, ...)` 的預期值，改成你算出的格單位距離；
- 三處 `decode(..., 8)` 都改成 stride 4：一處在斷言裡，兩處在 `print` 裡。

其餘斷言不必改，本題沿用原本的 80 步就會通過；框外點 `outside` 那一段直接比較畫素座標，和 stride 無關，也不必改。若 `decode` 改用 stride 4，算 `distance` 時卻仍除以 8，target 只有正確值的一半，解出的框也縮小成 `[17,19,31,29]`，斷言會攔下這個錯。這是單位錯誤，不是模型能力問題。

延伸題：做完上題後，再把 `gt` 的下邊 y2 從 36 改成 62，第一個斷言的預期值也跟著改。先預測原本的 80 步還夠不夠讓 `after < before / 100` 通過，再執行核對。

??? note "參考答案"

    畫素距離：左 `22−12=10`、上 `22−16=6`、右 `40−22=18`、下 `36−22=14`，即 `[10,6,18,14]`。除以 stride 4 得格單位 `[2.5,1.5,4.5,3.5]`，所以第一個斷言的預期值改成 `[[2.5, 1.5, 4.5, 3.5]]`。解碼：`x1=22−2.5×4=12`、`y1=22−1.5×4=16`、`x2=22+4.5×4=40`、`y2=22+3.5×4=36`，仍是同一框 `[12,16,40,36]`。執行後第一行印出 `target ltrb in feature cells: [[2.5, 1.5, 4.5, 3.5]]`，第二行仍是 `decoded target pixels: [[12.0, 16.0, 40.0, 36.0]]`，第四行 `learned box pixels` 應接近 `[12,16,40,36]`。若漏改印第四行的那處 `decode(..., 8)`，會印出約 `[2,10,58,50]`：距離乘了 8 而不是 4，格和畫素混用；這一行沒有斷言，程式照樣通過，只能靠你對照。

??? note "延伸題的參考答案"

    下邊距離是 `(62−22)/4=10` 格，target 變成 `[2.5,1.5,4.5,10]`。框外點仍在框的右側，最後的斷言照樣通過；但 80 步不夠：`assert after < before / 100` 會失敗，把 `for _ in range(80)` 改成例如 `range(100)` 就會通過。原因是 Smooth L1 在誤差大於 1 時梯度大小固定，距離每步的推進量約為 0.125×sigmoid(raw)²：0.125 來自 lr 0.5 × 四邊平均的 1/4 × 斜率 1；softplus 的斜率 sigmoid(raw)（不超過 1）則出現兩次，一次在連鎖律算梯度時，一次在把 raw 的變化換成距離的變化時。所以每步最多隻推進約 0.125 格，而且起步遠低於這個上限：raw=0 時只有 0.125×0.5²≈0.031 格，要等 raw 變大才接近 0.125。光看上限，80 步最多能走 10 格，好像夠用；實際上每步走不滿，所以不夠。

參考來源：[Ultralytics Detect 的距離解碼與分支](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)、[bbox2dist／dist2bbox 與候選點](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py)。本節的 softplus 小模型是教學選擇。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/12-anchor-free.json)

??? example "展開本次實際輸出"

    ```text
    target ltrb in feature cells: [[2.0, 1.5, 1.5, 1.0]]
    decoded target pixels: [[12.0, 16.0, 40.0, 36.0]]
    distance loss 0.376236 -> 0.000012
    learned box pixels: [[12.01, 16.03, 39.97, 35.94]]
    outside point requires a negative distance: assignment is still necessary
    ```

<!-- curriculum-evidence:end -->
