# 4.1 單物件分類與定位：類別之外，還要回答在哪裡

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/04-localization.ipynb){ .md-button }

A 章的分類器已能給出紅、藍兩類的分數。現在把紅方塊從左邊搬到右邊：分類答案仍是「紅」，位置答案卻必須跟著變。**同一次讀圖，除了回答是什麼，還要交出一個框，指出物件在哪裡。**

先保留每張圖剛好一個物件的設定，讓這個新增答案容易追蹤。本例是兩張黑底的 32×32 人工圖：第 0 張有紅方塊，類別 0；第 1 張有藍方塊，類別 1。每張的正確答案現在都有兩部分：類別與位置。從這裡出發，我們先寫出框，再決定模型怎麼輸出它、用什麼誤差訓練，最後檢查預測是否真的框住物件。

## 先替方塊寫出位置答案

框可以用兩個角表示。圖片左上角是原點，x 向右、y 向下，單位是畫素（pixel，簡寫 px）。**xyxy** 的四個數依序是左上角與右下角：\((x_1,y_1,x_2,y_2)\)。

![紅色物件的真值框，與人工向右下偏移的預測框](../assets/diagrams/04-localization.svg)

先看圖中的紅方塊與綠色實線：GT（ground truth，真值）是正確標註，框為 `[4,6,16,18]` px。黃色虛線 pred 是**人工設定**的預測框 `[6,8,18,20]`，用來手算誤差，還不是模型輸出。圖中的白色斜線區是兩框重疊的部分；稍後會用它計算重疊程度。

本書的框座標記的是畫素之間的邊界線，採**半開區間**，含左上、不含右下。例如 x 從 4 到 16，蓋住第 4～15 號畫素，共 12 個，所以框寬是 \(x_2-x_1\)，不加 1。這和 Python 切片 `4:16` 相同。畫圖時，tensor 索引先列後欄，寫成 `images[..., y1:y2, x1:x2]`；框的四個數仍先 x 後 y。

框也可以改記**中心與寬高（cxcywh）**。它和 xyxy 表示同一個矩形：

\[
\begin{aligned}
c_x&=(x_1+x_2)/2,\quad c_y=(y_1+y_2)/2,\\
w&=x_2-x_1,\quad h=y_2-y_1.
\end{aligned}
\]

| 圖片／類別 | 真值 xyxy（px） | 同一框的 cxcywh（px） |
| --- | --- | --- |
| 第 0 張，紅／0 | `[4,6,16,18]` | `[10,12,12,12]` |
| 第 1 張，藍／1 | `[16,10,28,22]` | `[22,16,12,12]` |

有些舊格式把右下角記成「最後一個畫素的編號」，寬高才要加 1；若混用，框的面積與後面的重疊分數都會變。本書始終沿用上面的半開區間。

## 把框答案換成模型可輸出的比例

框 head（把特徵轉成框答案的末端）要輸出 cx、cy、w、h 四個值。本例讓四個原始輸出各經過 **sigmoid**：

\[
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

它把任意實數壓到 0 與 1 之間，例如 \(\sigma(0)=0.5\)、\(\sigma(2)\approx0.88\)。這裡壓出的數字是四個**座標比例**，不是四種類別的機率。

因此正確答案也要換成比例，才能直接和輸出比較。把 cx、w 除以圖寬 W，把 cy、h 除以圖高 H，這叫**正規化**。本例 W=H=32：紅框的 target（訓練目標）是 `[0.3125,0.375,0.375,0.375]`，藍框是 `[0.6875,0.5,0.375,0.375]`。換到別的尺寸，比例仍有同一個意義；非正方形圖片則須分別用 W、H。

要畫回圖片，先求左右界 \(c_x\pm w/2\)、上下界 \(c_y\pm h/2\)，再各乘 W、H。圖中的人工預測 cxcywh 正規化後是 `[0.375,0.4375,0.375,0.375]`，因此：

\[
x_1=(0.375-0.1875)\times32=6,\qquad
x_2=(0.375+0.1875)\times32=18.
\]

同樣算出 y1=8、y2=20，回到黃色框 `[6,8,18,20]` px。換表示時，先核對順序與單位：xyxy 不可直接和 cxcywh 相減，px 也不可直接和比例相減。

sigmoid 只限制四個比例各自在 0～1，**不保證整個框在圖內**。例如 cx=0.05、w=0.3，左界是 −0.10，乘 32 得 −3.2 px。若使用時要將框限制在圖片內，得在還原框之後另訂截斷規則；本節不靠截斷把預測變準。

??? note "選讀：截斷預測與修正標註是不同工作"

    [完整圖片推論](07-inference.md)的 `decode_grid` 會把預測框的座標截到圖內。截斷 GT 並不能限制模型輸出；原始 GT 若越界，應先檢查並修正資料。[圖與框同步增強](11-augmentation.md)刻意裁圖後，則可依增強規則截斷 GT、刪掉露出太少的框，因為圖片本身也已改變。

## 分類與定位，該讀同一份特徵的哪部分

A 章分類 head 用 GAP（Global Average Pooling，全域平均池化），把每個通道的空間位置取平均。顏色出現在哪裡不改變紅／藍答案，這種摘要適合本次分類；框卻需要位置。

用一張人工 4×4 特徵圖看這個差別：只有一格是 1，其餘是 0。不論 1 在第 1 列第 0 欄，還是第 1 列第 3 欄，平均都是 1/16=0.0625。平均後，後面的層無法再由這個數分出左右。

改成**展平（flatten）**，16 個數仍依位置排列。第 k 個數固定來自第幾格：\(k=\text{列}\times4+\text{欄}\)，編號從 0 起算。左側的 1 位在 k=4，右側位在 k=7。線性層計算 \(y=\sum_k w_kz_k+b\)，兩種輸入便分別得到 \(w_4+b\) 與 \(w_7+b\)；只要兩個位置的權重不同，輸出就能不同。這把定位需要的「反應在哪裡」直接交給框 head。

本例讓兩個 head 共用 backbone（抽取特徵的主幹）：3×3 卷積把 `[B,3,32,32]` 變成 4 通道，padding=1 保持尺寸，接 ReLU，再用 2×2 max pooling，得到 `[B,4,16,16]`。B 是這批圖片的張數。

- 分類 head 對高、寬平均成 `[B,4]`，輸出 `[B,2]` 紅、藍 logits。
- 框 head 保留全部 4×16×16=1024 個特徵，輸出 `[B,4]`，經 sigmoid 得到正規化 cxcywh。

`Localizer` 就是這個接法；`self.body` 是 backbone：

``` { .python data-excerpt="lesson_cases/04-localization.py" }
class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone：3×3 卷積（3→4 channel，padding=1）→ ReLU → 2×2 max pooling
        self.body = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.class_head = nn.Linear(4, 2)  # 分類 head：平均後的 4 個數 → 2 個類別分數
        self.box_head = nn.Linear(4 * 16 * 16, 4)  # 框 head：展平的 1024 個數 → 4 個框數字

    def forward(self, images):
        features = self.body(images)  # [B,4,16,16]
        # 分類 head：mean((2, 3)) 對高、寬取平均 → [B,4]，再經 class_head → [B,2] logits
        # 框 head：flatten(1) 展平 → [B,1024]，再經 box_head → [B,4]，sigmoid 壓到 0～1
        return self.class_head(features.mean((2, 3))), self.box_head(features.flatten(1)).sigmoid()
```

位置資訊的代價也可直接算出：框 head 有 1024×4＋4=4100 個參數；若先平均，只需 4×4＋4=20 個。分類 head 是 4×2＋2=10 個參數。展平還把模型綁在固定特徵尺寸上：輸入改為 64×64，pooling 後變成 `[B,4,32,32]`，展平有 4096 個數，原本只收 1024 個的框 head 就會報 shape 不合；分類 head 平均後仍是 `[B,4]`。即使 33×33 經 pooling 仍成 16×16、沒有報錯，最後一列與一欄也已被丟掉，不能只改輸入尺寸就照用。

??? note "選讀：GAP 前的特徵仍可能間接帶位置"

    上面的算例證明的是「這兩份特徵取平均後相同」，不是所有 GAP 模型都無法定位。靠近圖邊的卷積窗會讀到 padding，特徵可能和中央不同；物件平移後落入不同 max pooling 區塊，特徵值也可能改變。本例黑底與 padding 都是 0，邊界線索主要在物件碰到圖邊時出現。我們選展平，是直接保留位置對應，不依賴這些間接線索。

## 兩種答案，分別用 loss 評分

類別沿用 A 章的交叉熵，評分模型給正確類別多少機率；框是連續數值，沿用數值誤差的想法，用 **MSE（Mean Squared Error，均方誤差）**：四個比例各自相減、平方，再取平均。

黃色人工框與紅色 GT 的正規化 cxcywh 分別是 `[0.375,0.4375,0.375,0.375]` 和 `[0.3125,0.375,0.375,0.375]`。中心各差 0.0625，寬高相同，所以單框 MSE 是

\[
\frac{0.0625^2+0.0625^2+0+0}{4}=0.001953125.
\]

完整程式一批有兩張圖，`mse_loss` 對 2×4=8 個座標平均。每步把分類 loss 與 5×框 loss 相加，兩個 head 與共用 backbone 都由這個總 loss 更新：

``` { .python data-excerpt="lesson_cases/04-localization.py" }
for step in range(3):
    optimizer.zero_grad(set_to_none=True)  # 清掉上一步的梯度
    logits, boxes = model(images)  # logits [B,2]；boxes [B,4] 是正規化 cxcywh
    class_loss = nn.functional.cross_entropy(logits, labels)
    box_loss = nn.functional.mse_loss(boxes, targets)  # targets：正規化 cxcywh 的真值框 [B,4]
    loss = class_loss + 5 * box_loss  # 框 loss 乘權重 5 再相加
    loss.backward()
    assert model.box_head.weight.grad.abs().sum() > 0  # 框 head 收到的梯度不全是 0
    optimizer.step()
    print(f"step={step}, classification={class_loss.item():.4f}, "
          f"box={box_loss.item():.4f}, total={loss.item():.4f}")
```

這段之前已建立兩張圖 `images`（`[2,3,32,32]`）、`labels=[0,1]` 與框 `targets`；`optimizer` 是學習率 0.1 的 SGD。`nn.functional` 是 PyTorch 的現成 loss 函式，也常簡寫成 `F`。

權重 5 把**框 loss 的梯度**放大 5 倍，讓它在共同更新中有較大影響。分類與框的量尺不同：step=0 分類約 0.7173，框約 0.0185，相差約 39 倍；但 loss 的數值占比不等於它推動參數的力道占比。因此程式分別印出兩項，不能只看總和。5 是固定示範設定，沒有比較證據說它最合適。

??? note "選讀：核對兩項 loss 的起點與降幅"

    兩類 logits 相等時，交叉熵是 −ln 0.5≈0.693。框若暫假設四個 logits 都是 0，sigmoid 後都是 0.5；和兩張 target 的八個誤差平方取平均，得到

    \[
    (2\times0.1875^2+5\times0.125^2+0^2)/8\approx0.0186.
    \]

    這解釋了為什麼起點是分類約 0.7、框約 0.02。step=0 框項乘 5 後約 0.0925，只占總 loss 0.8099 約 11%；前兩次更新的總降幅卻約 0.052，其中約 0.047 來自框項降到 0.0455，分類只降約 0.005。梯度是 loss 對參數的斜率，不能從這個 11% 推出梯度也只占 11%。

## 誤差很小，框就貼得很準嗎

回看開頭的人工圖。**IoU（Intersection over Union，交併比）**用兩框的交集面積除以聯集面積，直接回答整個框重疊多少：0 是不重疊，1 是完全一致。

交集左右界取兩個左界的較大者、兩個右界的較小者，上下同理。黃色框與 GT 的交集寬為 min(16,18)−max(4,6)=10，高為 min(18,20)−max(6,8)=10；圖中的斜線區因此是 100 px²。兩框各 144 px²，聯集是 144＋144−100=188 px²，扣一次交集，避免重複計數：

\[
\operatorname{IoU}=100/188\approx0.532.
\]

若兩框不相交，寬或高算出的差小於或等於 0，就取 0，交集與 IoU 都是 0。

MSE 與 IoU 看的是不同關係。仍向右、向下各偏 2 px，但將框縮成 4×4：GT `[4,6,8,10]` 對 pred `[6,8,10,12]`，交集只有 2×2=4，聯集 16＋16−4=28，IoU 約 0.14。兩組的正規化 MSE 卻同是 0.001953125，因為座標差相同。**同樣偏幾個畫素，小框失去的重疊比例更大。**本節用 MSE 訓練、IoU 評估；[IoU 類 loss](11-iou-loss.md)再教如何將這種幾何關係放進 loss。

## 模型真正交出的框：先看三次更新

可以在 Colab 執行完整實驗，或在本機跑 `PYTHONPATH=. python lesson_cases/04-localization.py`。程式先核對兩份人工特徵的平均都是 0.0625，再讓模型在這兩張圖上做 3 次 SGD 更新。它確認框 head 收到非零梯度、權重改變，並把**第三次更新後的輸出**畫在同樣兩張圖上：

![固定兩張人工圖、SGD 更新三步後的模型預測與真值](../assets/diagrams/04-localization-model.png)

左圖紅方塊、右圖藍方塊；綠框是 GT，黃框才是模型的 pred。比較四條邊，黃框仍有明顯偏差，尤其右圖的左界。這證明新增的框輸出確實能畫成位置，也顯示三次更新還沒有擬合好這兩個框。

頁尾的分類、框與總 loss 都量在**當次更新之前**：step=0 尚未更新，step=2 已更新兩次，總 loss 約 0.8099→0.7576、分類 0.7173→0.7123、框 0.0185→0.0091。第三次更新後沒有再量 loss；上圖則取自更新後。這些是兩張訓練圖的結果，不能當作新圖上的定位能力。

??? note "選讀：完整程式的 shape 與存檔"

    第 0 張 target 應是 `[0.3125,0.375,0.375,0.375]`。`class_shape=(2,2)` 表示兩圖各有兩類分數；`box_shape=(2,4)` 表示兩圖各有四個正規化框數字。疊圖存成 `artifacts/04-localization.png`。開頭的 IoU 圖使用人工固定框，和這份模型疊圖是不同材料。

## 訓練 40 步：模型能學會這兩張圖的類別與框嗎 { #forty-steps }

接著從相同初始權重重來，在同樣兩張圖上更新 40 次，觀察它能把這批答案學到什麼程度。這次使用 Adam、學習率 0.01；三步實驗使用 SGD、0.1。步數與 optimizer 同時改了，因此結果差異不能全歸因於步數。

手機上可左右滑動表格，查看完整欄位。

|模型|第 1 步（未更新）→ 第 40 次更新前的總 loss|訓練分類正確率（accuracy）|獨立驗證資料（validation）分類正確率|
|---|---|---|---|
|localizer|0.809883 → 0.430213|1.00|未評估|

分類正確率在 40 次更新做完後量，1.00 表示兩張訓練圖都分對類別。框是否準確，還要另看。這次沒有獨立驗證圖，紀錄的 `validation_accuracy` 是 `null`。

![同樣兩張訓練圖上，40 次更新前量到的總 loss 與分項](../assets/diagrams/04-localization-learning.svg)

兩圖橫軸都是第幾次更新，每點量在該次更新之前，第 1 點就是三步實驗的 step=0。左圖藍色虛線是總 loss，紫線是分類 loss，紅線是 5×框 loss；每點都滿足「總和＝紫線＋紅線」。右圖把未乘 5 的框 MSE 單獨畫出，才看得清它接近 0 時的起伏。

開頭約 0.10 的總降幅主要來自框項；之後紅線貼近 0，總 loss 的下降主要來自分類。整段總 loss 約降 0.38，起點框項只有約 0.09，即使降到 0，也只能解釋總降幅的四分之一左右。右圖的框 MSE 在很低的範圍起伏，沒有一路下降。**總 loss 一路變小，不表示位置也一直變準。**

40 次更新後，兩個預測 xyxy 框約為 `[4.48,4.77,18.13,18.93]`、`[16.33,10.58,29.18,23.12]` px；各自和同張圖 GT `[4,6,16,18]`、`[16,10,28,22]` 比較，IoU 是 `[0.6945,0.7752]`。它們比人工偏移框的 0.532 高，仍未完全重合。這裡沒有第 40 次更新後的 loss，卻有更新後框的評估，兩種量測不要混在一起。

這只支持模型能擬合這兩張圖。尤其兩圖顏色不同，讀平均特徵的框 head 也可能靠顏色記住兩個位置，所以此結果不能證明展平比平均好；比較位置資訊時，應使用同色、只有位置不同的圖。

??? note "選讀：40 步的設定、檢查與重跑"

    `scripts/run_learning_extensions.py` 直接使用本節 `Localizer` 與 `to_xyxy`，畫同樣兩圖、使用同樣 target，以及交叉熵＋5×框 MSE。兩個實驗皆固定 seed=7，在 CPU 上執行；40 步是重新初始化，不接續三步結果。Adam（Adaptive Moment Estimation，自適應矩估計）在 A.1 已介紹，它利用過去梯度與梯度平方的平均調整各參數步幅。

    腳本逐步確認 loss、梯度皆為有限數，梯度的整體 L2 長度（所有梯度平方相加再開根號）大於 0，最後確認權重改變，也逐點核對 loss 加總。這些檢查與訓練圖上的結果，不代表泛化到真實圖片或更深架構。

    先執行 Colab 的準備環境格，再新增一格：

    ```python
    !python scripts/run_learning_extensions.py --section 04-localization
    from IPython.display import SVG, display
    display(SVG(filename='artifacts/runs/learning/04-localization/learning.svg'))
    ```

    `!` 讓 notebook 執行終端機指令。本機在專案根目錄執行第一行時去掉 `!`。結果寫到 `artifacts/runs/learning/04-localization/`：`report.json` 保存完整結果，`learning.svg` 是分項 loss 圖，不覆寫網站圖與紀錄。印出的 JSON 省略四串逐步 loss，其餘和 report 相同；重跑小數可能略有差異。[保存的 40 步結果](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-localization-learning.json)供核對。

## 停一下：位置差了多少，和重疊少了多少

將開頭的人工預測改成向右移 6 px、y 不變，即 xyxy `[10,6,22,18]`。不用重訓，請算交集、聯集、IoU，以及正規化 cxcywh 的 MSE。

??? note "參考答案"

    交集寬是 min(16,22)−max(4,10)=6，高是 12，面積 72；聯集 144＋144−72=216，IoU=72/216=1/3。

    預測 cxcywh 是 `[16,12,12,12]` px，正規化後只有 cx 差 6/32，MSE=(6/32)²/4=0.0087890625。先統一表示與單位，才能比較。疊圖時也要分清 GT 與模型輸出，畫出標註不等於預測成功。

現在一張圖已能交出「類別＋框」兩種答案。接下來先處理[圖片尺寸改變時，框如何同步轉換與還原](04-coordinates.md)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/04-localization.json)

??? example "展開本次實際輸出"

    ```text
    two different positions -> same global mean=0.0625
    step=0, classification=0.7173, box=0.0185, total=0.8099
    step=1, classification=0.7146, box=0.0127, total=0.7779
    step=2, classification=0.7123, box=0.0091, total=0.7576
    first_target_cxcywh=[0.3125, 0.375, 0.375, 0.375], class_shape=(2, 2), box_shape=(2, 4)
    predicted pixel xyxy=[[7.4885783195495605, 7.199568748474121, 20.595396041870117, 19.995590209960938], [9.941701889038086, 8.684192657470703, 23.892370223999023, 22.775699615478516]]
    overlay=artifacts/04-localization.png; no held-out detection claim
    ```

<!-- curriculum-evidence:end -->
