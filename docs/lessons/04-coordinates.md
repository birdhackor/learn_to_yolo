# 座標轉換與還原：框跟圖片一起移動

模型在64×64輸入上畫出的框，不能直接畫回80×40原圖。縮放與補邊改變了座標系；若只記住圖片變成正方形，忘記padding，框會上下偏移。前置只需知道xyxy四數表示左上／右下，本頁從一張非正方形圖片走完往返。

[在 Colab 執行](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/04-coordinates.ipynb)，或 `PYTHONPATH=. python lesson_cases/04-coordinates.py`。本節CPU只做幾何變換，不需訓練。案例同時驗證整數比例、奇數尺寸與空框；成功代表變換契約成立，不代表模型定位準確。

## 三個座標系，先寫上單位

原圖高H=40、寬W=80，RGB張量 `[3,40,80]`、float32、值0到1。框計算使用浮點tensor，因為scale可能是0.8、轉換後也可能有小數；本節helper遇到整數框會先轉float32。不能在整數dtype中建立比例向量，0.8會被截成0。真值框是 `[10,5,50,25]` pixels：x向右、y向下、半開區間，所以框寬40、高20。模型輸入是64×64；其框仍可用pixels表示，但pixels是**輸入畫布**的pixels。再除64可以得到輸入畫布的正規化座標；「正規化」不會自動指回原圖。

| 框所在空間 | xyxy | 意義 |
| --- | --- | --- |
| 原圖 | `[10,5,50,25]` px | 80×40的座標 |
| Letterbox輸入 | `[8,20,40,36]` px | 64×64的座標 |
| 輸入正規化 | `[0.125,0.3125,0.625,0.5625]` | 每軸除64 |

## Stretch與letterbox的差別

Stretch把原圖直接拉成64×64，水平方向比例 \(s_x=64/80=0.8\)，垂直方向 \(s_y=64/40=1.6\)。框成為 `[8,8,40,40]`；原先40×20的矩形變成32×32，物件形狀也被拉長。

Letterbox先選能放進畫布的等比例縮放 \(s=\min(64/80,64/40)=0.8\)，原圖縮成64×32。剩下32pixel高度，上下各補16；水平方向不用補。框先乘0.8成 `[8,4,40,20]`，再加y方向16，得到 `[8,20,40,36]`。

![縮放0.8與上下padding16的座標往返](../assets/diagrams/04-coordinates.svg)

保持長寬比是letterbox的收益；代價是部分輸入為padding，物件實際可用的pixels較少，也必須記住每圖的變換資料。背景padding值也應與訓練前處理一致。

## 逆變換要倒過來做

若去程是 \(x'=s_x x+p_x,\ y'=s_y y+p_y\)，回程就是：

\[
x=(x'-p_x)/s_x,\qquad y=(y'-p_y)/s_y.
\]

本例y1回程 \((20-16)/0.8=5\)，y2回程 \((36-16)/0.8=25\)。順序是先扣padding、再除scale。若先除0.8再扣16，會得到9而不是5。

```python
scales = torch.tensor([sx, sy, sx, sy])
padding = torch.tensor([left, top, left, top])
input_boxes = original_boxes * scales + padding
original_boxes_again = (input_boxes - padding) / scales
```

框shape `[N,4]`，N是這張圖的物件數；向量 `[4]` 在每個框重複使用。N=0時仍保留 `[0,4]`，不要把空框寫成無軸的空list，免得下游無法辨識四座標契約。

## 奇數尺寸為何值得另外驗證

圖片resize必須使用整數寬高。37×83原圖放進64×64，理想比例64/83約0.7711，height四捨五入成29；實際resize為29×64。因整數取整，水平實際比例64/83、垂直29/37約0.7838，略有差異。

本節helper儲存實際 `new_w/W` 與 `new_h/H`，框使用這兩個比例，確保依實際pixel邊界定義能往返。這是教學的明確約定；其他實作也可能記錄理想比例，再容忍不到一pixel的rounding差，使用時必須確認自己的契約。補邊奇數時left／top取下整數，另一側可能多1pixel，不要假定每邊總是完全相同。

## 可核對輸出與常見錯誤

輸出應是original_hw `(40,80)`、resized_hw `(32,64)`、letterbox框 `[[8,20,40,36]]`、restored框 `[[10,5,50,25]]`。案例對往返使用float tolerance而不是字串比對，還寫出 `artifacts/04-coordinates.png` 的原圖／stretch／letterbox圖板。完成這些檢查即可結束本節。

若把輸入正規化框直接乘原圖W、H，會得到 `[10,12.5,50,22.5]`，與真值的y=5／25不同：padding沒有消失。若batch有不同原圖尺寸，每張都要儲存自己的metadata，不能全部套用第一張的比例。裁切越界預測應在所選座標系按明確規則做；裁切不是逆轉換的替代品。

## 自主練習與答案

將原圖改成H=80、W=40，框改為 `[5,10,25,50]`。答案resize成64×32（高×寬），左右各補16，框變成 `[20,8,36,40]`；回程x先減16，再除0.8。再問為何不使用加1計算框寬？答案是本課採連續pixel邊界／半開區間契約，加1會更改標註與IoU幾何。

直式題可保留原例，在本節notebook最後新增獨立cell，不必逐一猜主例的固定答案：

```python
portrait = torch.zeros(3, 80, 40)
portrait[0, 10:50, 5:25] = 1
portrait_box = torch.tensor([[5., 10., 25., 50.]])
portrait_canvas, mapped, meta = letterbox(portrait, portrait_box)
assert meta['resized_hw'] == (64, 32)
assert torch.allclose(mapped, torch.tensor([[20., 8., 36., 40.]]))
assert torch.allclose(undo(mapped, meta), portrait_box)
stretched = portrait_box * torch.tensor([1.6, .8, 1.6, .8])
assert torch.allclose(stretched, torch.tensor([[8., 8., 40., 40.]]))
```

若選擇修改原main，也需同步圖片切片、letterbox固定答案、stretch比例；圖題原本是`Original 80x40`（寬×高），直式題要改成`Original 40x80`。不要刪掉往返、整數標註與空框檢查。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-coordinates.json)

??? example "展開本次實際輸出"

    ```text
    integer annotation converted to float; finite exact round trip passed
    original_hw=(40, 80), resized_hw=(32, 64)
    original=[[10.0, 5.0, 50.0, 25.0]], letterbox=[[8.0, 20.0, 40.0, 36.0]], restored=[[10.0, 5.0, 50.0, 25.0]]
    stretch=[[8.0, 8.0, 40.0, 40.0]], padding=[0.0, 16.0, 0.0, 16.0]
    odd rounding: resized_hw=(29, 64), scales=[0.7710843086242676, 0.7837837934494019, 0.7710843086242676, 0.7837837934494019]
    odd-size and empty-box round trips passed; geometry only, no training required
    transform_panel=artifacts/04-coordinates.png
    ```

<!-- curriculum-evidence:end -->
