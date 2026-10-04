# 11.2 特徵融合：把深層資訊送回細網格

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/11-fusion.ipynb){ .md-button }

第 10 章加了 8×8 的細 head，讓細格子也能輸出框。但細 head 只拿到淺層特徵，而淺層特徵的「語義」可能不足。本節把深層特徵的資訊送回 8×8 的細網格，和淺層特徵合在一起，這就是特徵融合。讀完你能一步步追出融合時每一步的 shape，說出 concat（串接）需要什麼條件，並手算這個融合模組的參數與記憶體。

這裡的「語義」指特徵能判斷「這一塊是什麼東西」的程度。以第 10 章的 TwoScale 模型為例：64×64 的輸入經過三次 3×3、stride 2 的卷積，得到 8×8 的淺層特徵（第 10 章叫 fine feature），每一格的理論感受野約 15×15 pixel（這和下面的 31×31 都只算理論範圍；最上一列與最左一欄的格子，這個範圍有一部分落在 padding 補的 0 上；右、下兩邊也補了 0，只是用不到）。感受野是原圖中可能影響這一格的範圍（第 1 章）。再多一層卷積，得到 4×4 的深層特徵（coarse feature），每格的感受野約 31×31 pixel。深層每格看得較廣，比較有機會分辨這一塊是物件還是背景；但每格間距 16 pixel，位置比較粗。FPN（Feature Pyramid Network，特徵金字塔網路，下面會介紹）論文也這樣描述：淺層特徵的語義較弱，但位置較準。融合就是讓 8×8 的每一格，同時拿到淺層的細位置和深層看得較廣的判斷。

本節只加入這一條從深層到淺層的融合，不同時改 anchor、資料增強（augmentation：訓練時對圖片做翻轉等變化，框也跟著變；下一節介紹）或 loss。

前置：第 10 章的[多尺度 head](10-multiscale.md)（8×8 細 head 與 4×4 粗 head 怎麼接），以及第 1 章的[卷積 shape](01-small-cnn.md)（卷積後的 shape、感受野與參數怎麼算）。

## 融合放在哪裡：neck 與特徵金字塔

偵測模型可以分成三段：backbone（特徵提取網路）把圖片變成特徵；head（輸出框與類別的預測器）把特徵變成每格的輸出；neck 夾在兩者之間，負責整理或融合特徵。本節的融合模組就是一個很小的 neck。

談 neck 時常說「特徵金字塔」。同一張圖在網路不同深度會得到多張特徵圖，越深的越小、格子越粗。把它們由大到小疊起來，就像一座下大上小的金字塔，最深、最粗的一層在塔頂。下面幾個方向名稱，都是照這座金字塔的上下來取的：

- **top-down（由上而下）**：從塔頂（深、粗）往塔底（淺、細）送資訊，也就是本節的深→淺方向。
- **bottom-up（由下而上）**：反過來，從淺層往深層送。
- **lateral connection（橫向連接）**：把 backbone 裡同一網格大小的特徵橫著接過來，和 top-down 送來的特徵合併。本例就是 8×8 的淺層特徵，也就是下圖中「lateral 同尺度接入」那條線。

歷史機制：

- [FPN](https://arxiv.org/abs/1612.03144)：用 top-down 路徑與 lateral connection 建構特徵金字塔。它把深層特徵放大後，和 lateral 那一支逐值相加，所以兩支的 channel 數必須相同；每次合併時，用 1×1 卷積調整 channel 的是 lateral（淺層）那一支。
- [PANet](https://arxiv.org/abs/1803.01534)（Path Aggregation Network，路徑聚合網路；本頁文字與圖中簡稱 PAN）：在 FPN 之外再增加一條 bottom-up 路徑。
- [YOLOv4](https://arxiv.org/abs/2004.10934) 與固定版本 [YOLOv5 v6.0 模型配置](https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml)：各有自己的 neck 設計。

??? note "PAN 為什麼還要一條淺→深的路？"

    backbone 本身就是從淺層算到深層，但這條路很長。PANet 論文指出，在 FPN 的 backbone 裡，淺層的資訊要傳到最深層，可能得經過上百層。PANet 另加一條不到 10 層的 bottom-up 捷徑，讓淺層較準的位置資訊比較容易傳到深層。本節沒有實作這條路徑。

上面這幾個設計不是同一篇論文提出的一個模組。本節簡化成一條深→淺的路徑：把深層特徵放大、concat、再卷積，下一小節會逐步說明。也就是只做一次 FPN 式的深→淺融合，不是完整的 PANet，也不聲稱重現完整的 YOLOv4／v5 neck。

和原始 FPN 相比，有兩處不同要分清楚。一是用 concat 代替相加：兩路的 channel 各自保留，交給後面的 3×3 卷積學怎麼混（YOLOv3 也是用 concat 合併）。二是 lateral 那一支沒有 1×1，淺層特徵直接接進 concat；FPN 則在每次合併前，先用 1×1 處理 lateral 那一支。深層那一支的 1×1（本節的 reduce）FPN 也有：最深的一層在走 top-down 路徑之前，同樣先接一個 1×1。相加與 concat 哪個比較好，本節沒有比較。

本節借用第 10 章 stride 8／16 兩種尺度的概念，但兩個輸入改用 8 與 16 channel 的隨機張量（見圖下說明），並沒有直接接上第 10 章 16／32 channel 的特徵。融合輸出的特徵供細 head 使用，沒有整套雙向路徑。若接回第 10 章，只有細 head 改吃融合特徵；粗 head（4×4）沿用第 10 章的接法，直接接深層特徵，本節不動它（圖中也沒有畫粗 head）。本節只確認融合接得通、梯度傳得到；這個融合模組值不值得留在模型裡，要靠品質與成本的對照實驗判斷（見後面「收益與代價」），本節沒有做這個對照。

![同一輸入的兩種 stride 與本節融合路徑](../assets/diagrams/11-fusion.svg)

看圖說明（圖中 Shallow 是淺層特徵，Deep 是深層特徵）：

- 圖的左半（輸入圖片 → Backbone → Shallow → Deep）畫的是真實網路裡這兩層特徵從哪裡來。本節程式沒有圖片，也沒有 backbone：`shallow`、`deep` 是用 `torch.randn` 直接造出的 `[1,8,8,8]` 與 `[1,16,4,4]` 隨機張量，兩者互不相關。所以本節只能檢查 shape、梯度與參數，看不出融合的效果。
- 圖中淺層畫在上方、深層畫在下方，所以 top-down 路徑（Deep → 1×1 reduce → Nearest → Concat）在圖上是由下往上走，和名字的方向相反。
- 從融合輸出往下的橘色虛線，代表 PAN 另加的 bottom-up 路徑（淺→深）；它在圖上由上往下走，同樣和名字的方向相反。本例沒有實作這條路徑。

## 為什麼不能直接 concat

在真實網路裡，同一張 64×64 的輸入經過 backbone，總 stride 8 的淺層特徵有 8×8 格，總 stride 16 的深層特徵有 4×4 格。總 stride 是從輸入到這一層、各層 stride 的乘積：例如三次 stride 2 就是 2×2×2=8，代表相鄰兩格在原圖相隔 8 pixel。兩層的總 stride 本來就不同（8 和 16）。

本節程式裡，淺層特徵是變數 `shallow`，形狀 `[B,8,8,8]`，也就是 8 個 channel、8×8 格（`[B,C,H,W]` 依序是 batch、channel、高、寬）。它的角色對應第 10 章 stride 8 的 fine feature，但只有 8 channel（第 10 章是 16）。深層特徵是變數 `deep`，形狀 `[B,16,4,4]`：16 個 channel、4×4 格，角色對應第 10 章 stride 16 的 coarse feature（第 10 章是 32 channel）。淺層格子細、位置準；深層 channel 較多、網格較粗。

沿 channel 軸（dim=1）concat 時，其他軸都要一樣長。兩者的最後兩軸 H、W 不同（8×8 對 4×4），所以不能直接接。必要的一步是把深層特徵放大到 8×8，這叫上取樣（upsample）：把較粗的特徵圖放大成更多格。本節用最簡單的 nearest（最近鄰）：每個新格子直接抄來源對應格的值。放大 2 倍時，輸出的第 0、1 列都抄來源第 0 列，第 2、3 列都抄來源第 1 列，以此類推，欄也一樣。放大後格子數對上了，但深層的內容仍是每 2×2 格同一個值，細節並沒有變多（後面「用四個數看上取樣」會示範）。

整個融合分成四步：

1. **reduce**（1×1 卷積）：`[B,16,4,4]` → `[B,8,4,4]`。1×1 只改 channel，不改 4×4 格。這一步不是 concat 的必要條件，下方說明為什麼仍然要做。
2. **nearest 上取樣**：`[B,8,4,4]` → `[B,8,8,8]`，把深層特徵放大到淺層的網格大小。
3. **concat**：`[B,8,8,8]` 接 `[B,8,8,8]` → `[B,16,8,8]`。channel 是 8+8=16，H、W 仍是 8×8。
4. **mix**（3×3 卷積）：`[B,16,8,8]` → `[B,8,8,8]`，把兩路混合成 8 channel。

concat 的必要條件是 H、W 相同，所以一定要先上取樣到 8×8；channel 數則不必相同。不先減 channel 也能合法 concat：深層直接上取樣成 `[B,16,8,8]`，和淺層接成 `[B,24,8,8]`；只是 mix 要改成 24 channel 輸入，成本也跟著改變。完整程式裡的斷言（assert）也確認了這件事。

那為什麼還要先減 channel？好處之一是參數比較少：不減的話，mix 要從 24 channel 混成 8，參數 8×24×9+8=1736；先減的話，reduce 加 mix 只要 136+1160=1296（後面「參數與記憶體可以手算」會逐項算）。減完之後，兩路也各占 8 個 channel。

??? note "先 1×1 或先上取樣，有差嗎？"

    nearest 只是複製，1×1 卷積又是每一格各自計算，所以兩者交換先後，結果完全相同。差別在計算量。乘加次數＝格數×輸入 channel×輸出 channel：先做 1×1，只要在 4×4 的 16 格上算，16×16×8=2048 次；先上取樣再做 1×1，就要在 8×8 的 64 格上算，64×16×8=8192 次，是 4 倍。

若改用相加（add），兩支的整個 shape 都必須相同，包括 channel 數：逐值相加後仍是 `[B,8,8,8]`，mix 要改成 8 channel 輸入。concat 則把兩組並排保留成 16 channel。

## 對照完整程式的 Fusion

完整程式把這四步寫成 `Fusion` 類別，在 `__init__` 裡建了兩層：

- `self.reduce = nn.Conv2d(16,8,1)`：1×1 卷積，把 16 channel 減成 8。
- `self.mix = nn.Conv2d(16,8,3,padding=1)`：3×3 卷積，把 16 channel 混成 8。`padding=1` 讓 8×8 仍是 8×8；沒有 padding 會變成 6×6。

下面是它的 `forward()`，和完整程式相同，只加了中文註解：

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

第一行除了兩個輸入，還有一個檢查用的參數 `inspect`，預設是 `False`。最後一行的意思是：`inspect` 是 `False` 時，只傳回融合輸出 `output`，所以平常寫 `model(shallow,deep)`，拿到的就是它；設成 `True` 時，會把 `output` 連同中間的 `reduced`、`up`、`concat` 一起傳回。完整程式寫的是 `output,reduced,up,concat = model(shallow,deep,inspect=True)`，等號左邊的四個變數依序接住傳回的四個張量。後面算 loss、做 backward 用的就是這次的 `output`；執行時印出的每一步 shape，量的也是這四個張量。所以印出的 shape，就是真正算出 loss 的那次 forward 裡的 shape（見後面「執行與核對」）。

`size=shallow.shape[-2:]` 直接要求「放大成和淺層一樣的高、寬」，比假定放大 2 倍更明確。另一種寫法是 `scale_factor=2`：不指定輸出大小，而是長、寬各放大 2 倍。有些輸入尺寸會讓兩種寫法的結果不同。例如輸入改成 100×100，照第 10 章 TwoScale 的做法經過三次 3×3、stride 2、padding 1 的卷積，依序得到 50、25、13，淺層是 13×13；再一次得到深層 7×7。`scale_factor=2` 會放大成 14×14，和 13×13 concat 時會報錯；寫 `size=shallow.shape[-2:]` 則直接得到 13×13。

??? note "延伸：FPN 也在合併後接 3×3 卷積"

    只是混合 channel 的話，1×1 卷積就做得到（上一節 CSP 就用 1×1 混合兩路）；3×3 除了混合兩路的 channel，還會看相鄰的格子。FPN 原論文在合併後也接一個 3×3 卷積，論文給的理由是減少上取樣造成的混疊（aliasing）效應，大致是指放大後出現的假紋路，例如 nearest 複製出的 2×2 方塊邊界。這是 FPN 論文的說法；本節沒有比較 mix 用 1×1 或 3×3 的差別，也沒有驗證這個效果。

??? note "進階：shape 對了，位置不一定對"

    shape 一致，不代表兩張特徵圖的同一格對到原圖的同一個位置（pixel 對齊）。要追蹤的是每一層每一格的中心落在原圖哪裡（格點位置），以及各層的 stride 與 padding。

    例如某條路上有一層 3×3 卷積沒有 padding：8×8 會變成 6×6，新的第 0 格的中心其實在原本的第 1 格。再用 `size=(8,8)` 把它放大回 8×8，shape 對了，但有些格子會對到相鄰一格的位置。不同裁切造成的位移也一樣，不會因為 shape 一致就消失。

    本節的 `shallow`、`deep` 是隨機張量，沒有對應的原圖位置，所以不會遇到這個問題。接上真實的 backbone 時，就要逐層追蹤 stride、padding 與裁切。

## 用四個數看上取樣

單 channel 的 2×2 特徵 `[[1,2],[3,4]]`，用 nearest 放大成 4×4：

```text
1 1 2 2
1 1 2 2
3 3 4 4
3 3 4 4
```

每個來源值被複製成 2×2 共四格。若 loss 是所有輸出的總和，每個來源的梯度恰為 4。可以用第 3 章的推法：把某個輸入加一點點，看 loss 增加多少。把來源的 1 改成 1.001，4×4 裡的四個 1 都變成 1.001，總和增加 0.004，所以這個來源的梯度是 0.004÷0.001=4；四個來源都一樣。

這裡的「複製」不是生成四份新細節：深層特徵仍只有原本四個值。細位置的資訊來自淺層特徵，3×3 的 mix 學習怎麼使用兩路。

若改用 bilinear（雙線性插值：新格是相鄰來源值依距離的加權平均），放大結果會出現 1.25、1.75 這類中間值，不再只是複製。在 loss 剛好是全部輸出總和這個特例下，bilinear 每個來源的梯度也仍是 4；換成別的 loss，nearest 與 bilinear 的梯度通常就不同，例如本節完整程式用的「輸出平方的平均」。所以必須記錄用的是哪一種插值方法，不能把兩者當成完全一樣。這裡的插值指放大特徵圖時，由已知格算出新格的方法，和第 6 章算 AP 時的插值是兩回事。

## 參數與記憶體可以手算

卷積的參數用第 1 章的公式 \(C_{out}(C_{in}k^2+1)\)。乘開就是 \(C_{out}\times C_{in}\times k^2\) 個權重，加上 \(C_{out}\) 個 bias（每個輸出 channel 一個），所以下面都寫成「輸出×輸入×k²＋輸出」（1×1 時 k²=1，省略不寫）。本例：

- reduce（1×1，16→8）：`8×16+8=136`
- mix（3×3，16→8）：`8×16×9+8=1160`
- 合計 1296。

1296 是本節這個教學用迷你 neck 自己的參數，不是直接加在第 10 章模型上的成本。若接回第 10 章（淺層 16 channel、深層 32 channel）：

- reduce 把 32 channel 減成 16：`16×32+16=528`
- concat 後是 16+16=32 channel
- mix 把 32 channel 混成 16：`16×32×9+16=4624`
- 這份 neck 合計 5152 個參數。

mix 輸出 16 channel，原本吃 16 channel 的 `fine_head` 才不必改。

記憶體也會增加。concat 後的中間特徵有 `B×16×8×8` 個數，比單路 8 channel 多。以 float32、B=1 為例：16×8×8=1024 個數，每個 float32 占 4 bytes，僅這張 concat 就是 4096 bytes（4 KiB）。實際訓練時，還要暫存其他層的中間輸出和梯度。這些中間輸出也叫 activation，這裡指各層輸出的張量，反向傳播時要用到。它和第 3 章的激勵函數（activation function，例如 ReLU）不是同一個意思；本例的 `Fusion` 其實沒有 ReLU。

## 執行與核對

可以用頁首的「在 Colab 執行本節」按鈕執行，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/11-fusion.py`。應核對四件事：

1. 輸出第一行的 `nearest example`：2×2 例子放大後的 4×4 矩陣和手算相同。
2. 同一行的 `source gradient`：2×2 例子的來源梯度全為 4。
3. 第二行的 `shapes`：先是兩個輸入 shallow `(1, 8, 8, 8)`、deep `(1, 16, 4, 4)`，接著依序是 reduce `(1, 8, 4, 4)`、nearest `(1, 8, 8, 8)`、concat `(1, 16, 8, 8)`、mix `(1, 8, 8, 8)`，和上面「整個融合分成四步」一致（這裡 B=1）。後四項量的是 `inspect=True` 那次 forward 傳回的 `reduced`、`up`、`concat`、`output`。最後的 mix `(1, 8, 8, 8)` 就是融合輸出 `[1,8,8,8]`（同一個 shape，只是印法不同），程式也用斷言 `assert output.shape == (1,8,8,8)` 核對。
4. 第三行的 `parameters`：參數共 1296。

此外，完整程式用「輸出平方的平均」當 loss，對融合輸出做一次 backward。這個 loss 沒有偵測上的意義，只是把輸出收成一個數，好呼叫 `backward()` 算梯度。斷言確認 `shallow`、`deep` 兩個輸入的梯度都是有限值（不是 inf 或 NaN）而且不全為 0，表示兩條路都真的連到輸出。接著實際做一次 SGD 更新，並用斷言確認 mix 的權重確實改變了。

只看輸入的梯度還不夠。假如把 `forward()` 裡 `F.interpolate(reduced,…)` 的 `reduced` 換成 `deep[:,:8]`（深層的前 8 個 channel），融合輸出就不再經過 reduce；可是 shape 照樣對得上，兩個輸入也照樣收到梯度。所以程式也斷言 reduce 與 mix 的每個參數（權重與 bias）的梯度都不是 `None`、都是有限值，而且不全為 0。第 2 章〈[訓練診斷](02-diagnostics.md)〉說過，backward 後梯度仍是 `None` 的參數，表示它這次沒有接進反向傳播；上面那樣改，reduce 的參數就是這種情況，這個斷言會失敗。

完整程式的斷言全都寫在三個 `print` 之前，所以三行都印出來，就表示斷言全部通過。第三行後半的 `both branches backward and one step` 是固定印出的字，摘要的就是上面這些梯度與更新的斷言。

這個實驗只檢查特徵怎麼接，沒有訓練偵測器，所以沒有小物件 AP 或速度的結論。

## 收益與代價

收益是細 head 可以同時利用淺層的空間細節，以及深層看得較廣的判斷。代價有三項：多了卷積層的參數與計算；多了要暫存的中間特徵（activation）；還要追蹤兩路的格子有沒有對齊（見上方〈進階：shape 對了，位置不一定對〉）。

深層的低解析度資訊，不一定能救回已經丟失的小物件。融合做得不好，也可能把背景的訊號帶進細 head，使它在背景上多報框。正式比較時，應固定輸入、資料、訓練預算（例如步數）與 decode／NMS 的設定，只換 neck，再分別看小物件的表現與背景誤報。

## 常見錯誤

- **`torch.cat` 寫成 `dim=3`**：兩個 `[B,8,8,8]` 會沿寬度接成 `[B,8,8,16]`，這一步不會報錯；要到 mix 才因為輸入 channel 是 8、不是 16 而報錯。
- **改了 reduce 的輸出 channel，卻沒同步改 mix 的輸入 channel**：例如 reduce 改成輸出 12 channel，concat 後變成 8+12=20 channel，mix 卻仍寫 16 輸入，執行到 mix 就報錯。
- **把上取樣當成框座標的換算**：上取樣只是把特徵圖放大成更多格，框的座標不會因此跟著乘 2。接在融合特徵後面的細 head，解碼仍照它自己的 stride 8。
- **先對整張特徵圖做 global pooling，又期待之後能定位**：global pooling（全域池化，例如第 4 章的 global average pooling）把每個 channel 的整張 H×W 壓成一個數，不再保留「哪一格」的空間排列。第 4 章示範過：特徵圖只有一格是 1、其餘是 0 時，不論這格在左邊或右邊，平均出來都一樣。pooling 前的特徵就算因為圖邊、padding 等原因帶著一些位置資訊，也只是間接線索，不能靠它定位。

## 自主練習

手算即可，不必改程式。先自己算，再展開答案。

1. 若淺層特徵是 `[2,12,10,10]`，深層先用 1×1 減到 12 channel，再上取樣，concat 之後的 shape 是多少？mix 的輸入 channel 要設成多少？
2. 承上題，若深層原本是 `[2,32,5,5]`：reduce 用 1×1 把 32 channel 減到 12，mix 用 3×3（padding=1）把 24 channel 混成 12。這個 neck 共有幾個參數？

??? note "參考答案"

    **第 1 題**：上取樣後深層是 `[2,12,10,10]`，沿 channel 軸 concat 得到 `[2,24,10,10]`；mix 的輸入 channel 必須是 24。

    **第 2 題**：reduce 是 12×32+12=396；mix 是 12×24×9+12=2604；合計 3000 個參數。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式已於 2026-10-02 用 PyTorch 2.9.1+cpu 在 CPU 上執行過，程式裡的 assert 檢查全部通過。下面是那次印出的原始輸出；每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-fusion.json)

??? example "展開本次實際輸出"

    ```text
    nearest example [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]] source gradient [[4.0, 4.0], [4.0, 4.0]]
    8ch shallow + 8ch upsampled deep -> 16ch concat -> 8ch mixed
    parameters 1296 both branches backward and one step
    ```

<!-- curriculum-evidence:end -->
