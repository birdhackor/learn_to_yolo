# 14 YOLO11 特徵模組：拆路徑、保留中間成果、再融合

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/14-feature-module.ipynb){ .md-button }

偵測 head 要判斷物件，也要定位邊界；送給它的特徵既需要局部資訊，也需要周圍較大的範圍。疊卷積可以逐層擴大感受野，但如果模組最後只交出末層結果，先前的訊號就必須一路被保留下來，才能供後面使用。[CSP](11-csp.md) 已讓一部分 channel 走短路、另一部分走卷積。本節再問：能不能把卷積途中算出的結果也保留，最後一起交給融合層選用？

YOLO11 的 C3k2 模組採用這種切分、逐步轉換、串接的安排。我們把它縮成輸入輸出同為 8 channel 的小模組，追出四份特徵如何共存、哪裡省了卷積成本，又在哪裡限制了資訊。這個安排改的是特徵的路徑；是否改善偵測，還取決於路徑真正學到什麼。

## 末端除了最新結果，還能讀哪些中間結果？

本節自訂模組叫 `SplitAggregate`，意思是先切分、再彙整，不是官方模組名。先用 1×1 卷積把 8 個輸入 channel 重新混合，輸出 8 個 channel，再切成 a、b 各 4 個。這和直接固定取輸入的前後半不同：a、b 是投影層學出的組合。

a 直接保留；b 經第一個小塊得 b1，再經第二個得 b2。每個小塊都用 [residual](03-identity.md) 的 `x+F(x)`：F 是 3×3 卷積→ReLU→3×3 卷積，兩層都是 4→4、stride 1、padding 1，所以加回 x 時 shape 不變。本頁稱它 bottleneck（瓶頸小塊）；它在整體 8 channel 裡只處理 4 channel，內部兩層沒有再縮窄。

![a、b、b1、b2 保留後串接的路徑](../assets/diagrams/14-split-paths.svg)

沿圖中 b 的路徑看：b1 不只餵給下一個小塊，也分一路到最後串接；b2 算完時，a、b、b1 都仍在。a、b、b1、b2 沿 channel 串成 16 channel，再以 1×1 融合到 8。圖中的 F₁、F₂ 分別是兩個小塊的卷積轉換；hidden=4 表示每份保留特徵的 channel 數。

直接疊兩層卷積也能學著保留局部資訊。本設計的不同，是讓融合層直接讀到不同轉換深度的結果，不必全部只經由末層結果取得。短路徑提供轉換較少的值，長路徑提供經過鄰格互動的值；融合權重再依任務學如何組合。這是路徑安排的理由，不是每種訊號都一定有用的保證。

## 四份特徵同位置串接，空間尺寸不變

輸入是 `[B,C,H,W]=[2,8,8,8]`：2 筆特徵、8 channel、高寬各 8。三個 8 意義不同，channel 是第 1 軸。每份 a、b、b1、b2 的 shape 都是 `[2,4,8,8]`；同一格的四份 channel 接在一起，得到 `[2,16,8,8]`。最後 fuse 是 16→8 的 1×1，回到 `[2,8,8,8]`。

以下摘自 `SplitAggregate`。最後一行把完整程式的 `joined`、`output` 和回傳合併；預設 `inspect=False` 時結果相同：

```python
# __init__：建立模組時執行一次（本例用預設值 channels=8、hidden=4、blocks=2）
self.project = nn.Conv2d(channels, 2 * hidden, 1)  # 1×1 投影：8→8
self.blocks = nn.ModuleList(Bottleneck(hidden) for _ in range(blocks))  # 兩個 bottleneck，各 4→4
self.fuse = nn.Conv2d((2 + blocks) * hidden, channels, 1)  # 1×1 融合：16→8

# forward：每次計算時執行
a, b = self.project(x).chunk(2, dim=1)  # a、b 各 [2,4,8,8]
paths = [a, b]
for block in self.blocks:
    paths.append(block(paths[-1]))  # 拿 paths 的最後一份，算出下一份
return self.fuse(torch.cat(paths, dim=1))  # 串接成 [2,16,8,8]，再融合回 [2,8,8,8]
```

`paths[-1]` 取最後一份：第一輪取 b 算 b1，第二輪取 b1 算 b2。所以迴圈結束時 `paths=[a,b,b1,b2]`。`torch.cat(...,dim=1)` 是沿 channel 串接，不是相加，也不是讓圖變寬；fuse 學的是同一位置上 16 個 channel 的組合，不是直接平均四條路。

四份特徵的差別可以用感受野看見。以下範圍相對於**模組輸入特徵圖**，不是原圖：

| 特徵 | 經過的空間轉換 | channel | 理論感受野 |
| --- | --- | --- | --- |
| a | 只有 1×1 投影 | 4 | 1×1 |
| b | 只有 1×1 投影 | 4 | 1×1 |
| b1 | 一個小塊，兩個 3×3 | 4 | 5×5 |
| b2 | 兩個小塊，四個 3×3 | 4 | 9×9 |

每個 stride 1 的 3×3 讓感受野邊長加 2；residual 同時保留更短的路。融合層可以把同一格的局部值與較廣鄰域的結果一起用，這正是保存中間結果所增加的選擇。本例輸入只有 8×8，9×9 視窗會碰到 padding：b2 的中央 2×2 格看得到全圖，角落只看得到 5×5 真實位置。理論 9×9 不能當成每格都實際讀到 81 個輸入值。

## 窄路徑省什麼，又擋住什麼？

拿輸入輸出同為 8 channel 的 plain 作成本對照：plain 是兩個 8→8 的 3×3，中間一個 ReLU。兩者都含卷積 bias；參數數公式為 `out×in×kh×kw+out`，沒有 BN 參數。

| 模組 | 計算 | 總參數 |
| --- | --- | --- |
| plain | `2×(8×8×9+8)` | 1,168 |
| split 投影 | `8×8+8` | 72 |
| split 四個 3×3 | `4×(4×4×9+4)` | 592 |
| split 融合 | `16×8+8` | 136 |
| split 合計 | `72+592+136` | 800 |

split 的 3×3 數量更多，但都在 hidden=4 上做，592 比 plain 的 1,168 少 576；1×1 投影、融合另加 208，合計仍少。每個位置的乘加次數不含 bias：split 是 `64+4×144+128=768`，plain 是 `2×576=1,152`。每筆有 64 位置，所以分別為 49,152、73,728 次。這些數字只屬於 hidden=4、兩小塊的設定。

窄路徑同時限制了資訊：鄰格的 8 channel 原值要先壓成 b 的 4 channel，才能經 3×3 傳來；a 沒有讀鄰格的路。多留 b1、b2 並沒有恢復已在這個入口捨去的任意 8 維鄰格資訊。因此不能把「中間結果較多」直接解讀成什麼任務都更容易。

保存路徑也有暫存成本。a、b、b1 要等 b2 完成；串接又建立 `[2,16,8,8]`，共 2,048 個 float32、8 KiB。這些中間張量叫 activation，和激勵函數不同。參數與乘加較少，也未必延遲較短：小卷積、串接和搬資料在不同裝置上的成本不同。

## 用右移任務檢查可訓練，失敗也要解讀

完整程式用 seed=7 產生一組隨機 x，target=`x.roll(1,dims=-1)`：每個 channel 沿寬向右循環移一格。`[1,2,3,4]` 的答案是 `[4,1,2,3]`；每格要輸出左鄰值，最左欄則要最右欄。這是一組固定特徵的人工轉換任務，沒有圖片物件或 GT。

它可以檢查 MSE、backward 和 SGD 是否接通，卻不是本模組能通用完成的右移規則。除了上述 8→4 的鄰格資訊瓶頸，roll 的邊界也是循環的，卷積卻補 0；最左欄看不到最右欄的值。因此本例只要求更新 30 次後 MSE 比初值小，沒有要求等於 0。

保存結果的 MSE 降到 1.0049，但同一資料若全部輸出 0，MSE 約 0.97，還比較低。下降支持「這個模組能被 loss 更新」，沒有支持「已學會位移」。想核算 0.97，可在 `target = x.roll(1, dims=-1)` 後加 `print(target.square().mean())`。plain 在程式裡只統計參數，沒有訓練；不能拿它的隨機 loss 與訓練過的 split 比品質。

串接路徑則另有檢查：程式確認 concat 的四段依序就是 a、b、b1、b2，也檢查各段到 fuse 的直接梯度。不能只看 b1 本身有梯度就說它已送進 fuse：即使省掉 b1 的串接，它仍能經 b2 反傳。詳細檢查放在下方選讀。

執行 `PYTHONPATH=. python lesson_cases/14-feature-module.py`。對照相同的輸入輸出 `(2,8,8,8)`、四份 4 channel 串接為 16→8、參數 1168／800、MSE 下降與路徑檢查。CPU 小張量沒有量測偵測延遲或 AP。

## 從小模組回到 YOLO11

官方 C3k2 和本例都保留逐步轉換的特徵，再串接融合；官方還有 BN、激勵函數、內部寬度與小塊選項。下面兩份選讀交代名稱與差異，避免把 800 個參數當成官方配置的數字。

??? note "名字小抄：C2f、C3k2、C3k"

    這些都是 Ultralytics 程式裡的類別名稱，不必背。

    - **C2f**：YOLOv8 用的模組。1×1 卷積後切兩半，一半依序經過多個 Bottleneck，每一步的輸出都串接起來，再用 1×1 融合。下方程式裡 `forward` 那五行，幾乎就是它的 forward。
    - **C3k2**：YOLO11 用的 C2f 變體，整體結構和 C2f 相同。參數 `c3k` 主要決定內部小塊用什麼：`False` 時用 Bottleneck，`True` 時改用 C3k（還有其他選項，本節不談）。
    - **C3k**：可自訂 kernel 大小的 C3。C3 是第 11 章 CSP 那節提過的 YOLOv5 模組。

??? note "與官方 C3k2 的差別"

    **歷史機制**：YOLO11 的官方配置採用 C3k2 等模組，但整體還有其他 backbone（主幹）、neck（夾在主幹與 head 之間整理特徵的部分）與 attention（注意力）設定，所以 YOLO11 的版本收益不能全部歸因於 C3k2。

    `SplitAggregate` 參考 C3k2／C2f 的做法，但簡化了下面幾項，所以不等同完整的 C3k2：

    - **官方的 Conv**：這是 Ultralytics 自己包的一層，依序是卷積（不含 bias）→ BN（BatchNorm，批次正規化，第 3 章介紹過）→ 激勵函數（activation，例如 ReLU、SiLU；SiLU 是 [11.1](11-csp.md) 選讀已介紹的 Sigmoid Linear Unit，sigmoid 線性單元，官方預設用它）。本例改用一般的 `nn.Conv2d`（含 bias），不加 BN；投影與融合後面也不加激勵函數，整個模組只有 bottleneck 的 F 中間有 ReLU。
    - **Bottleneck 中間的 channel 數**：官方 C3k2（`c3k=False` 時）的 Bottleneck 預設把中間縮成一半 channel（e=0.5）；本例沿用 C2f 的寫法（e=1.0），兩層都是 hidden→hidden。
    - **C3k 內部結構**：官方 C3k2 可以用 `c3k` 參數把內部小塊換成 C3k；本例沒有這個選項。


若要放進 MiniYOLO，先選輸入輸出相同的部位。例如 [7.4](07-training.md) 表格裡 adaptive pool 後的 3×3，shape `[B,32,4,4]→[B,32,4,4]`，可換成 `SplitAggregate(channels=32,hidden=16)`。再固定資料切分、步數與 decode，只改這個部位，記錄 AP50、參數、端到端時間和失敗圖。完整 YOLO11 另外還改了 backbone、neck 與 attention 配置，局部替換的結果只回答本模組是否適合這個位置。

自己改模組時，最先檢查三個介面：chunk 是連續前後半，不是交錯 channel；concat 後 fuse 要讀所有保留的 channel；residual 的 x 和 F(x) 要同 shape。若某個不相同的軸長恰好是 1，broadcast 可能不報錯，更需要逐軸核對。把小塊改成 4→2→4 仍可相加；改成不同輸出寬度或 stride，就不能直接照搬 `x+F(x)`。

bottleneck 這個名字在 ResNet、Ultralytics 和本例指的內部層數與寬度不同，要比較時先展開實作。加小塊、改 hidden、關捷徑或在官方設 `c3k=True` 都會改變工作或成本；下面練習只加一個小塊，看「拆路徑一定較省」為何不是通則。

??? note "進階：確認四份特徵都接進融合層（可先跳過）"

    訓練結束後，完整程式另做一次檢查用的 forward：`module(x, inspect=True)`。`inspect` 是完整程式的 `forward` 才有的參數（網頁摘錄省略了），設成 `True` 時會多傳回 `paths`（a、b、b1、b2）和串接結果 `joined`。程式對這五個 tensor 呼叫 `retain_grad()`。PyTorch 預設會替參數保留 `.grad`，中間算出來的 tensor 則不保留；要先呼叫 `retain_grad()`，backward 後才看得到它的梯度（[第 5 章](05-assignment.md)用過）。

    **段**（segment）：串接後的 16 個 channel 中，第 0–3 個是 a、4–7 是 b、8–11 是 b1、12–15 是 b2，每份特徵占一段。完整程式裡的變數 `segment`（一段的 channel 範圍）和印出的 concat-segment，指的都是這裡的段。程式會核對每份特徵和自己那一段的數值完全相同，確認串接順序是 a、b、b1、b2。

    **一個值用在兩處，梯度要相加**：例如 L=2u+3u，u 用在兩個地方，L 對 u 的梯度是 2+3=5，兩條路各貢獻一項。[第 11 章特徵融合那節](11-fusion.md)的 nearest 上取樣也是同一個道理：一個值被複製成 4 格，loss 是全部輸出的總和時，它的梯度是 1+1+1+1=4。

    **直接梯度與總梯度**：

    - 直接梯度：loss 對 concat 裡某一段的梯度，只算 concat → fuse 這一條路。
    - 總梯度：loss 對該特徵本身的梯度。

    a、b2 只進 concat，所以它們各自的直接梯度和總梯度相等。b、b1 還要餵給下一個 bottleneck，所以總梯度＝直接梯度＋經下一個 bottleneck 傳回的部分。backward 後，程式逐段檢查直接梯度不為 0，也檢查每份特徵的總梯度不為 0。

    為什麼要實際檢查？沒串接、也沒被後面用到的輸出，對 loss 沒有貢獻，數學上的偏導為 0；在 PyTorch 中，這類未連線 tensor 的 `.grad` 可能是 `None`，不是一個全零 tensor。但反過來不成立：有連線也不保證梯度不為 0，所以程式另用斷言（assert）檢查算出來的值。

    為什麼不能只查總梯度？假設設計成不把 b1 交給 concat（例如只串 a、b、b2，fuse 改成 12→8），b1 仍會經 b2 收到梯度，第一個 bottleneck 也一樣。所以「b1 的總梯度不為 0」或「第一個 bottleneck 有梯度」，都證明不了 b1 直接接進了融合層。要確認這件事，得看 concat 裡 b1 該在的那一段（第 8–11 個 channel），而且下面兩項都要成立：

    - 這一段的數值和 b1 完全相同，表示這裡放的確實是 b1。上面只串 a、b、b2 的設計，第 8–11 個 channel 放的是 b2，這一項就不成立。
    - 這一段的直接梯度不為 0，表示 fuse 的輸出確實受這一段影響。若 b1 留在 concat 裡，fuse 卻用不到這 4 個 channel（例如先把它們乘上 0 再融合），第一項仍成立，b1 也仍經 b2 收到梯度，只有這一項不成立。

    所以「b1 那一段有直接梯度」和「b1 的總梯度不為 0」是兩個不同的檢查。

    執行紀錄最後一行的 gradient L1，是每一段直接梯度的絕對值加總，用來確認不為 0；它只是檢查用的數字，不是加進訓練的 L1 loss。

## 自主練習

**自主練習**：在 Colab 裡改「本節可修改的完整實驗」下面那一格；本機則改 `lesson_cases/14-feature-module.py`，兩者是同一份程式。把 `main()` 裡的 `module = SplitAggregate()` 改成 `module = SplitAggregate(blocks=3)`，hidden 維持 4。執行前先預測：

1. concat 後有幾個 channel？fuse 是幾→幾？
2. 總參數是多少？還比 plain 的 1,168 省嗎？

`__init__` 已用 `(2+blocks)×hidden` 自動設定 fuse 的輸入 channel；完整程式的斷言（assert）與印出的字串也依實際 block 數計算，都不必手改。

??? note "參考答案"

    多了一個 bottleneck，共五份 4 channel 的特徵：concat 後是 (2+3)×4=20 個 channel，fuse 是 20→8。印出的那行會是 `concatenated channels: 4 + 4 + 4 + 4 + 4 = 20; fuse 20 -> 8`。

    參數：投影 72＋六個 3×3 卷積 6×148=888＋融合 20×8+8=168，合計 1,128，印出 `parameters plain / split: 1168 1128`。1,128 已接近 plain 的 1,168，而且要保留更多中間特徵。MSE 那行的數字也會和 blocks=2 時不同，因為模型變了；斷言照樣通過。

    一般來說（channels=8、hidden=4），總參數＝投影 72＋bottleneck 296×blocks＋融合 (32×(2+blocks)+8)＝144+328×blocks：blocks=2 得 800，blocks=3 得 1,128，blocks=4 得 1,456，已經超過 plain。「拆路徑就更省」不是可用的普遍結論。

參考來源：[YOLO11 官方配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/11/yolo11.yaml)、[C3k2／C2f／Bottleneck 定義](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/14-feature-module.json)

??? example "展開本次實際輸出"

    ```text
    input/output: (2, 8, 8, 8) (2, 8, 8, 8)
    concatenated channels: 4 + 4 + 4 + 4 = 16; fuse 16 -> 8
    parameters plain / split: 1168 800
    local transformation MSE 1.0722 -> 1.0049
    concat order, each direct concat-segment gradient, and each path total gradient: verified
    direct concat-segment gradient L1: [0.2967, 0.3349, 0.3481, 0.2779]
    ```

<!-- curriculum-evidence:end -->
