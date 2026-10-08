# C.11.2 特徵融合：把深層資訊送回細網格

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/11-fusion.ipynb){ .md-button }

第 10 章的 fine head 讀 8×8 淺層特徵，coarse head 讀 4×4 深層特徵。細位置能保留較小的變化，但判斷這一塊是物件還是背景，也可能需要較大的周邊範圍。現在仍要讓細 head 輸出框，想把深層已算出的資訊也送給它，避免細 head 只能使用較早的特徵。

以 TwoScale 為例，淺層每格的理論感受野是 15×15pixel，深層是 31×31pixel；後者有較廣的材料可辨識物件，位置卻較粗。**特徵融合**把兩者放在同一細網格，讓後面的卷積學如何一起使用它們。「語義」在這裡指辨識這一塊是什麼的資訊，並非深層每格已經是一個類別答案；實際是否更會辨識仍需要訓練與評估。

## 融合放在哪裡：neck 與特徵金字塔

backbone（主幹）把圖片變成特徵，head 把特徵變成框與類別；中間整理特徵的部分叫 **neck**。本頁的迷你 neck 把深層送到淺層的位置，再交給細 head；粗 head 仍可直接讀 deep，不必跟著換輸入。

接線檢查先把兩路材料固定為 `shallow [1,8,8,8]` 與 `deep [1,16,4,4]`，以互不相關的隨機張量代替真正 backbone 特徵，channel 是 TwoScale 的一半；本次只計算 neck，尚不接 head。下面沿用第 10 章的深、淺尺度關係畫出它放在模型中的位置。

不同深度的特徵圖由大到小排列，像下大上小的**特徵金字塔**。深、粗在塔頂，因此深→淺叫 **top-down（由上而下）**；反向叫 bottom-up。backbone 同一網格大小的特徵從旁邊接入，叫 **lateral connection（橫向連接）**，本例就是 shallow。

![本節四步融合：deep 在上、shallow 在下，top-down 送到細網格，lateral 從旁接入](../assets/diagrams/11-fusion.svg)

實線畫本節真正計算的路徑。先看兩張材料如何對上位置，再看為什麼有兩層卷積：

| 步驟 | 輸入→輸出 | 做的工作 |
| --- | --- | --- |
| 1×1 reduce | `[B,16,4,4]→[B,8,4,4]` | 在粗網格減少通道 |
| nearest 上取樣 | `[B,8,4,4]→[B,8,8,8]` | 把粗格值複製到細網格 |
| 沿 channel concat | 兩份`[B,8,8,8]→[B,16,8,8]` | 並排保留兩路通道 |
| 3×3 mix | `[B,16,8,8]→[B,8,8,8]` | 混兩路通道與相鄰位置 |

**必要對齊的是 B、H、W，channel 不必相同。** 不做 reduce 也能把 deep 上取樣成 `[B,16,8,8]`，與 shallow 接成 24 通道；完整程式也核對這件事。先 reduce 的理由是降低後段成本：不減時 mix 參數 1736，先減時 reduce＋mix 共 1296。這個選擇不是 concat 的合法性要求。

concat 讓兩路各保留自己的通道，再交給 mix；若改逐值相加，整個 shape 包括 channel 都要相同，合併後仍 8 通道，mix 輸入也要改成 8。上一節的兩路串接算例在這裡同樣適用。

## 用四個數看上取樣

nearest（最近鄰）讓每個新格抄一個來源值。2×2 單通道 `[[1,2],[3,4]]` 放大到 4×4：

```text
1 1 2 2
1 1 2 2
3 3 4 4
3 3 4 4
```

四個來源值各被複製四次，沒有生成新的細節。deep 放大後相鄰 2×2 位置仍是同一值；細位置資訊要從 shallow 來，mix 纔有兩種材料可用。這就是兩路分工：細網格不只是放大的深層值，也保留了原本細特徵。

若 loss 是所有輸出總和，把來源的 1 改成 1.001，四個輸出各增加 0.001，loss 增加 0.004，所以來源梯度為 4；另外三值相同。完整程式先核對 nearest 矩陣與這四個梯度。這個檢查只教複製與反傳，沒有證明融合提高辨識力。

??? note "先 1×1 或先上取樣，有差嗎？"

    nearest 只是複製，1×1 卷積又是每一格各自計算，所以兩者交換先後，結果完全相同。差別在計算量。乘加次數＝格數×輸入 channel×輸出 channel：先做 1×1，只要在 4×4 的 16 格上算，16×16×8=2048 次；先上取樣再做 1×1，就要在 8×8 的 64 格上算，64×16×8=8192 次，是 4 倍。

另一種 bilinear（雙線性插值）會依距離混合相鄰來源值，產生中間值；它改變值與梯度的分配。這裡的插值是算新特徵格，與第 6 章 AP 的插值不同。主例固定用 nearest，以下是選讀比較。

??? note "選讀：bilinear 的來源座標、align_corners 與梯度"

    仍用 `[[1,2],[3,4]]` 這張 2×2 特徵圖，放大到 4×4。來源格子的中心座標是 0、1；新格子的中心要先換回來源座標，再依距離分配權重。`align_corners` 決定的是這個對應，不是有沒有保留原本的數字。

    ![兩格放大成四格時，兩種 align_corners 設定對應的來源中心座標](../assets/diagrams/11-bilinear-coordinates.svg)

    **`align_corners=False`（PyTorch 預設）**：第一列四個新格的中心換回來源座標是 −0.25、0.25、0.75、1.25。超出來源中心範圍時用邊上的值；介於 0 和 1 時依距離加權。因此第一列是 1、0.75×1+0.25×2＝1.25、0.25×1+0.75×2＝1.75、2。

    **`align_corners=True`**：第一個與最後一個新格中心對齊來源的兩端中心，中間兩點位於 1/3、2/3，所以第一列是 1、4/3、5/3、2。圖只畫水平方向；垂直方向也照相同規則換算。

    **loss 換了，梯度也會換。** 若 loss 剛好是全部輸出總和，bilinear 每個來源的梯度也仍是 4。這是這個例子的特例，不能推成兩種方法的梯度永遠相同。

    若 loss 改成 16 個輸出平方的平均，nearest 的每個來源值 s 被複製 4 次，梯度是 4×2s/16＝s/2，也就是 `[[0.5,1],[1.5,2]]`。預設設定的 bilinear 則是 `[[0.78125,1.09375],[1.40625,1.71875]]`。完整程式對融合輸出用的就是平方平均，但沒有拿這個 2×2 例子比較 bilinear；這裡是補充算例。因此實驗紀錄要同時寫插值方法與 `align_corners` 等設定。

## 對照完整程式的 Fusion

`Fusion` 建立 `reduce=Conv2d(16,8,1)` 與 `mix=Conv2d(16,8,3,padding=1)`；後者保持 8×8 不變，沒有 padding 則變 6×6。forward 依圖中的四步算：

``` { .python data-excerpt="lesson_cases/11-fusion.py" }
def forward(self,shallow,deep,inspect=False):
    # shallow：[B,8,8,8]（8 channel、8×8 格）；deep：[B,16,4,4]（16 channel、4×4 格）
    reduced = self.reduce(deep)  # [B,8,4,4]：1×1 只改 channel，4×4 格不變
    # shallow.shape[-2:] 取最後兩軸 (H,W)，也就是 (8,8)
    up = F.interpolate(reduced,size=shallow.shape[-2:],mode='nearest')  # [B,8,8,8]
    # 沿 dim=1（channel 軸）接起來：[B,8+8=16,8,8]
    concat = torch.cat([shallow,up],dim=1)
    # mix 是 3×3、padding=1，輸出仍是 8×8：[B,8,8,8]
    output = self.mix(concat)
    # inspect=True 時連三個中間張量一起傳回，否則只傳回 output
    return (output,reduced,up,concat) if inspect else output
```

`size=shallow.shape[-2:]` 取淺層高、寬，直接指定上取樣結果與它相同。若只寫 `scale_factor=2`，奇數網格未必對上：100×100 經 TwoScale 前三層變 50、25、13，深層再降成 7；7 放大 2 倍是 14，不能與 13×13 接，指定 size 則可。

`inspect=True` 會依序傳回 output、reduced、up、concat。完整程式用這一次 forward 的 output 算 loss，也量這一次四個張量的 shape；不是另外造一份 shape 示意。平常 `inspect=False` 則只回 output。

shape 合法也不自動證明圖片上的位置對齊。實際接 backbone 時，兩路 stride、padding 或裁切可能使同名格偏移；需要追每格在原圖的位置。本節隨機張量沒有原圖座標，所以只檢查張量連線。

??? note "進階：shape 對了，位置不一定對"

    shape 一致，不代表兩張特徵圖的同一格對到原圖的同一個位置（pixel 對齊）。要追蹤的是每一層每一格的中心落在原圖哪裡（格點位置），以及各層的 stride 與 padding。

    例如某條路上有一層 3×3 卷積沒有 padding：8×8 會變成 6×6，新的第 0 格的中心其實在原本的第 1 格。再用 `size=(8,8)` 把它放大回 8×8，shape 對了，但有些格子會對到相鄰一格的位置。裁切（把特徵圖邊緣切掉幾格；和前面章節把框夾回原圖範圍的裁切不是同一件事）造成的位移也一樣，不會因為 shape 一致就消失。

    本節的 `shallow`、`deep` 是隨機張量，沒有對應的原圖位置，所以不會遇到這個問題。接上真實的 backbone 時，就要逐層追蹤 stride、padding 與裁切。

??? note "延伸：FPN 也在合併後接 3×3 卷積"

    只是混合 channel 的話，1×1 卷積就做得到（上一節 CSP 就用 1×1 混合兩路）；3×3 除了混合兩路的 channel，還會看相鄰的格子。FPN 原論文在合併後也接一個 3×3 卷積，論文給的理由只有一句：減少上取樣造成的混疊（aliasing）效應，沒有再說明。常見的理解是放大後出現的假紋路，例如 nearest 複製出的 2×2 方塊邊界。本節沒有比較 mix 用 1×1 或 3×3 的差別，也沒有驗證這個效果。

## 參數與記憶體可以手算

沿用帶 bias 卷積的 `輸出×輸入×k²＋輸出`：reduce 是 `8×16+8=136`，mix 是 `8×16×9+8=1160`，合計 1296。這是本頁小 neck 的成本；接回 TwoScale 需要 32→16 的 reduce（528 參數），concat 成 32 通道，再用 32→16 的 3×3mix（4624），合計 5152，輸出 16 通道纔可直接交給原 fine_head。

concat 要存 `B×16×8×8` 個數。B=1、float32 每數 4bytes 時，這一張為 4096bytes（4KiB），還沒有算其他層與反傳的暫存。這些中間輸出也叫 activation，指張量，不是 ReLU 這種 activation function；本節 Fusion 其實沒有 ReLU。

## 執行與核對

執行 `PYTHONPATH=. python lesson_cases/11-fusion.py` 或頁首 Colab。第一行核對 nearest 的小矩陣和來源梯度；第二行依序列 shallow、deep、reduce、nearest、concat、mix 的 shape；第三行列 1296 參數。後四個 shape 量的是 `inspect=True` 回傳值，最後 mix 就是參與 loss 的 output。

這次對融合輸出用平方平均 loss，兩輸入與 reduce／mix 所有參數都需有有限、不全 0 的梯度，再做一次 SGD，核對 mix 權重改變。只看兩輸入有梯度還不夠：若錯把 up 的來源改成 `deep[:,:8]`，兩輸入仍到 loss，reduce 卻被繞過，參數梯度為 None。額外的參數檢查才抓得到它。

所有斷言都在三個 print 之前；`both branches backward and one step` 是通過後的固定摘要。這裡沒有偵測任務，所以不報 AP 或速度提升。融合可能帶入可用的廣域資訊，也可能帶入背景誤報；deep 若已把小物件訊號壓掉，nearest 不能補回它。正式比較要固定輸入、來源切分、訓練預算及 decode／NMS，只換 neck，分別看小物件與背景誤報。

本節示範的深→淺融合是 YOLOv4／v5 neck 的一段常見連線。完整架構還會串起更多尺度及淺→深路徑；我們只核對這一段，不能以它代表整個版本的品質或速度。

??? note "選讀：FPN、PAN 與 YOLO 的完整連線"

    歷史機制：

    - [FPN](https://arxiv.org/abs/1612.03144)（Feature Pyramid Network，特徵金字塔網路）：用 top-down 路徑與 lateral connection 建構特徵金字塔。它把深層特徵放大後，和 lateral 那一支逐值相加，所以兩支的 channel 數必須相同；每次合併前，新接進來的 lateral（淺層）那一支先經過 1×1 卷積調整 channel。
    - [PANet](https://arxiv.org/abs/1803.01534)（Path Aggregation Network，路徑聚合網路；本頁文字與圖中簡稱 PAN）：在 FPN 之外再增加一條 bottom-up 路徑。
    - [YOLOv4](https://arxiv.org/abs/2004.10934) 與固定版本 [YOLOv5 v6.0 模型配置](https://github.com/ultralytics/yolov5/blob/956be8e642b5c10af4a1533e09084ca32ff4f21f/models/yolov5s.yaml)：都在 FPN 式的 top-down 路徑之後，再接一條 PAN 式的 bottom-up 路徑；YOLOv4 還把 PAN 原本的相加改成 concat。YOLOv5 v6.0 的配置檔裡（寫在 `head:` 底下），每次 top-down 合併都是 1×1 卷積 → nearest 放大 2 倍 → concat（直接接 backbone 的淺層特徵）→ C3 模組，和本節的四步同型。

    ??? note "PAN 為什麼還要一條淺→深的路？"

        backbone 本身就是從淺層算到深層，但這條路很長。PANet 論文指出，在 FPN 的 backbone 裡，淺層的資訊要傳到最深層，可能得經過上百層。PANet 另加一條不到 10 層的 bottom-up 短路徑，讓淺層較準的位置資訊比較容易傳到深層。本節沒有實作這條路徑。

    上面這幾個設計不是同一篇論文提出的一個模組。本節簡化成一條深→淺的路徑：先用 1×1 減少深層的 channel，再放大、concat，最後用 3×3 卷積混合；上方四步主例已逐步示範。也就是隻做一次 FPN 式的深→淺融合，不是完整的 PANet，也不聲稱重現完整的 YOLOv4／v5 neck：相對 YOLOv5，本節的 mix 只是一個 3×3 卷積而不是 C3，卷積後沒有 BatchNorm 與 SiLU，也沒有第三個尺度與 bottom-up 路徑。

    和原始 FPN 相比，有幾處不同要分清楚。一是用 concat 代替相加：兩路的 channel 各自保留，交給後面的 3×3 卷積學怎麼混。二是 lateral 那一支沒有 1×1，淺層特徵直接接進 concat。這兩處都和 YOLOv3、YOLOv5 v6.0 的 top-down 合併相同：深層先接 1×1、再上取樣，然後直接和 backbone 的淺層特徵 concat。深層那一支的 1×1（本節的 reduce）FPN 也有：FPN 讓 backbone 每個尺度（stage）的輸出各接一個 1×1，最深那個尺度的 1×1 輸出就是 top-down 路徑的起點；之後每次合併，只有新接進來的 lateral 那一支接 1×1，從上面送下來的那一支不再接。另外，FPN 最深那個尺度的 1×1 輸出，也會再經過 3×3 卷積交給它自己的 head；本節的 reduce 只用在 top-down，粗 head 仍直接接 deep。相加與 concat 哪個比較好，本節沒有比較。

## 常見錯誤

沿 channel 接應為 dim 1。若寫 dim 3，兩個 `[B,8,8,8]` 會變 `[B,8,8,16]`，cat 本身合法，直到 mix 需要 16 輸入通道才報錯。reduce 輸出若改 12，concat 是 8+12=20，mix 也要同步改 20 輸入。

上取樣變的是特徵網格，框座標不因此乘 2。fine head 仍用 stride 8：格(1,1)、格內偏移 0.5 的中心是 12pixel，誤用深層 stride 16 會變 24。也不要先做 global pooling 把 H×W 壓成 1×1 再期待上取樣定位：所有 64 格拿到相同深層值，已沒有哪一格的排列資訊；第 4 章已用左右單點示範這種資訊損失。

## 自主練習

手算即可，不必改程式。先自己算，再展開答案。

1. 若淺層特徵是 `[2,12,10,10]`，深層先用 1×1 減到 12 channel，再上取樣，concat 之後的 shape 是多少？mix 的輸入 channel 要設成多少？
2. 承上題，若深層原本是 `[2,32,5,5]`：reduce 用 1×1 把 32 channel 減到 12，mix 用 3×3（padding=1），輸入 channel 用第 1 題的答案，輸出 12 channel。這個 neck 共有幾個參數？

??? note "參考答案"

    **第 1 題**：上取樣後深層是 `[2,12,10,10]`，沿 channel 軸 concat 得到 `[2,24,10,10]`；mix 的輸入 channel 必須是 24。

    **第 2 題**：reduce 是 12×32+12=396；mix 是 12×24×9+12=2604；合計 3000 個參數。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-fusion.json)

??? example "展開本次實際輸出"

    ```text
    nearest example [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]] source gradient [[4.0, 4.0], [4.0, 4.0]]
    shapes shallow (1, 8, 8, 8) deep (1, 16, 4, 4) -> reduce (1, 8, 4, 4) -> nearest (1, 8, 8, 8) -> concat (1, 16, 8, 8) -> mix (1, 8, 8, 8)
    parameters 1296 both branches backward and one step
    ```

<!-- curriculum-evidence:end -->
