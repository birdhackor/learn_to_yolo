# 審查紀錄：特徵融合

審查範圍：`docs/lessons/11-fusion.md`、頁面上的圖（`docs/assets/diagrams/11-fusion.svg`），以及 `lesson_cases/11-fusion.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：docs/lessons/11-fusion.md 這次的修改沒有必要等級的問題，也沒有需要另外提的建議，判定通過。所有檢查都在暫存副本進行，沒有在 repo 內執行任何東西。

1. 頁上關於程式的說法都和定稿的 lesson_cases/11-fusion.py 一致。
- 程式 exit 0，三行輸出和頁面〈執行與核對〉第 1 到 4 點逐項相符。用 OMP 執行緒 1、2、4 跑，輸出都逐位元組相同。
- forward() 區塊和程式逐行相同，只加了中文註解、拿掉 class 縮排。
- 行內引用的 `output,reduced,up,concat = model(shallow,deep,inspect=True)` 和 `assert output.shape == (1,8,8,8)` 都和程式逐字相同。
- 「斷言全都在三個 print 之前」屬實；第三行後半確實是固定印出的字。

2. 照頁面寫的方式，在暫存副本實際改程式驗證（腳本在 mut/run_mut.py、mut/probe.py）：
- 把 `F.interpolate(reduced,…)` 的 reduced 換成 `deep[:,:8]`：shape 不變，兩個輸入的梯度有限且不為 0；reduce.weight 和 reduce.bias 的梯度是 None，程式停在參數斷言（AssertionError），和頁面說的一樣。
- 常見錯誤 `dim=3`：mix 報「expected input[1, 8, 8, 16] to have 16 channels」。
- 常見錯誤「reduce 改輸出 12 channel」：mix 報「to have 16 channels, but got 20」。
- 以下說法都實算確認：bilinear 會出現 1.25、1.75；總和 loss 下梯度為 4，平方平均 loss 下 nearest 和 bilinear 的梯度不同；100×100 依序得到 50、25、13、7；scale_factor=2 放大成 14×14，concat 時報錯，size= 則得到 13×13。
- 參數數字 136、1160、1296、1736、528、4624、5152 和兩題練習的答案（[2,24,10,10]、396+2604=3000）都用程式算過，沒有錯。

3. 先前審查意見和受程式改動影響的段落清單：
- Colab 連結已經是 lessons-v0.4.0，沒有手改。
- 第 148 行那段補丁說明已刪除。
- 11-fusion 部分提的「重跑並用 torch.equal 斷言」寫法已被 inspect 取代，頁上沒有殘留。
- inspect 部分的第 82、89、92、148 行和選做的第 151 行都已處理。
- 第 185、191 行（執行紀錄區塊）留給 verify_curriculum.py 重產，修改者已列在待重產清單。
- 整頁沒有修訂或審查經過的敘述。第 37 行改成現在式的範圍說明，符合審查用的事實與寫作規範清單。

4. 數字和摘錄：
- 新加的數字都是確定值，沒有任何一個來自這台 Mac。
- 摘錄比對工具結果是 []。故意把頁面區塊改成 scale_factor=2，或把 return 改成 else concat，都會被報出來，表示檢查有作用。
- 頁上只有這一個 Python 區塊。暫存副本裡的 validate_lessons 只在別頁（09-anchor-clustering、20-deployment）報摘錄問題，11-fusion 沒有被報。

5. 建置與圖：
- 暫存副本裡 zensical build --clean --strict 通過（No issues found），validate_site.py 也通過，兩者都是 exit 0。
- 渲染後的 HTML 帶有 data-excerpt，新段落顯示正常，連到第 2 章的連結有效。
- 11-fusion.svg 沒有改動，有 viewBox、title 和 desc。用 qlmanage 轉成圖片看過，圖面乾淨，reduce 16→8、nearest 4×4→8×8、concat 8+8=16、mix 16→8 都和程式一致。

6. 改動範圍：只改了 docs/lessons/11-fusion.md。SVG、程式、notebook 和紀錄 JSON 都和 HEAD 相同。

7. 修改者提的疑慮我也查過，都不構成問題：
- inspect=False 的預設路徑沒有被執行，但頁上「平常寫 model(shallow,deep) 拿到 output」那句由已標記的 return 行守著，return 一改，摘錄檢查就會報。
- 第 2 章第 38 行目前仍有頁面引用的那句 None 說明。

8. 預期中尚未完成、和這頁無關的事：11-fusion 的紀錄 JSON 和頁尾紀錄區塊仍是舊的第二行，reviews/coverage 顯示 11-fusion「no review」。這些在發布流程中重產和重審。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：在暫存副本建立 rsync 複本，所有執行都在複本裡做。

**讀了什麼**
- 審查用的事實與寫作規範清單。
- docs/lessons/11-fusion.md 全文。
- 在複本執行 zensical build --clean --strict（exit 0，No issues found），把 site/lessons/11-fusion/index.html 轉成文字逐段讀。確認 5 個 note 摺疊區、1 個 example 摺疊區、3 個 MathJax 公式、程式摘錄與圖都照原意呈現。Playwright 瀏覽器沒有安裝，所以沒有整頁截圖。
- docs/assets/diagrams/11-fusion.svg：讀了原始碼（含 desc），也用 qlmanage -t -s 1200 轉成 PNG 看過。
- lesson_cases/11-fusion.py 與 notebooks/11-fusion.ipynb。

**跑了什麼**
- 照頁面的指令執行課程程式（PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg …/python lesson_cases/11-fusion.py），exit 0。三行輸出和「執行與核對」四項一致：nearest 4×4、來源梯度全為 4、shapes 六項、parameters 1296。

**練習與改法**
- 練習第 1、2 題先手算（[2,24,10,10]、24；396+2604=3000），再用 PyTorch 建同樣的層核對，結果一致。
- 依頁面描述在程式複本上做了三種改法，結果都和頁面說的一樣：
  - 把 reduced 換成 deep[:,:8]：第 35 行的逐參數斷言出現 AssertionError。
  - torch.cat 改成 dim=3：要到 mix 才 RuntimeError（expected input[1, 8, 8, 16] to have 16 channels）。
  - reduce 改成輸出 12：concat 成 20 channel，mix 報 RuntimeError。

**另外核對的數值**（暫存副本腳本 11-fusion-reader-work/checks.py）
- 1736、5152、4096 bytes。
- 1×1 與 nearest 交換先後：torch.equal 為 True。
- bilinear 2×2→4×4：第一列是 [1,1.25,1.75,2]；align_corners=True 時是 1.333／1.667。sum loss 的梯度是 4；平方平均時 nearest 是 [[0.5,1],[1.5,2]]，bilinear 是 [[0.78125,1.09375],[1.40625,1.71875]]。
- 100×100：50→25→13→7；scale_factor=2 得到 14，concat 報錯；size= 得到 13。
- 用全 1 權重的四層 3×3、stride 2 卷積對輸入求梯度：感受野是 15 與 31，而且只有上、左邊緣的格子碰到 padding。
- nearest 6→8 會錯位，7→13 對位正確。

**交叉查核**
- docs/glossary.md。
- 00-warmup、01-small-cnn（公式、感受野、float32、KiB、MAC、__init__）。
- 02-diagnostics（None 梯度）。
- 03-identity／03-projection（加 0.001 的推法、捷徑的定義、激勵函數）。
- 04-localization（GAP 左右的例子）；04-coordinates、08-own-images（「裁切」的用法）。
- 06-evaluation（AP 插值）。
- 10-multiscale 頁面與程式（TwoScale 是 16／32 channel，fine_head 吃 16 channel）。
- 11-csp（1×1 fuse）、14-feature-module（引用本頁的梯度 4）。
- raw.githubusercontent.com/ultralytics/yolov5/v6.0/models/yolov5s.yaml 的 head: 段。

**排除的項目**
- 頁尾紀錄區塊的第二行仍是「8ch shallow + 8ch upsampled deep -> 16ch concat -> 8ch mixed」，紀錄 JSON 的 case_sha256（1da3eb…）也和目前程式（4de6c1…）不同。這是正在重產的暫時狀態，不列為發現。notebook 第一格的舊模板文字同理。
- validate_lessons 在複本裡只因 reviews/ 還沒有審查而失敗；程式摘錄檢查與 notebook 最後一格一致性的檢查都通過。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 第 29–31 行摺疊區〈PAN 為什麼還要一條淺→深的路？〉：「PANet 另加一條不到 10 層的 bottom-up 捷徑」 | 本書的「捷徑」是固定術語。3.1 節、術語表和上一節 11.1 的前置，都把 shortcut（捷徑）定義成「把輸入逐值加回輸出」，也就是 y=x+F(x)。照順序讀到這裡的讀者，會把 PANet 這條路讀成 ResNet 式的 shortcut，以為 PAN 把淺層特徵原值加到深層輸出上。但這裡指的只是「層數比較少的一條路」。 |
| 2 | 建議 | 第 25 行「每次合併時，用 1×1 卷積調整 channel 的是 lateral（淺層）那一支」與第 35 行「深層那一支的 1×1（本節的 reduce）FPN 也有」 | 兩句先後讀起來互相矛盾：第 25 行讓讀者以為 FPN 的 1×1 只在淺層那支，第 35 行又說深層那支也有。化解矛盾的關鍵沒有寫清楚：FPN 對 backbone 的每一層都接 1×1，最深那層的 1×1 輸出就是 top-down 路徑的起點，之後從上面送下來的那支不再接 1×1。這層意思只濃縮在「最深的一層在走 top-down 路徑之前」半句裡，初學者看不出來。 |
| 3 | 建議 | 第 62 行「concat 的必要條件是 H、W 相同，所以一定要先上取樣到 8×8」 | 開頭說讀完要能「說出 concat 需要什麼條件」，這裡給的答案卻只有 H、W。第 53 行寫的是「其他軸都要一樣長」，術語表也寫「其他軸（B、H、W）要一樣長」。讀者若記成「只要 H、W 相同」，之後把 batch 數不同的兩個張量接起來，就會意外報錯。 |
| 4 | 建議 | 第 17–21 行的金字塔說明，與第 39 行的圖、第 44–45 行兩條看圖說明 | 正文先建立「深、粗的一層在塔頂，top-down 是由上往下」的心像。圖卻把淺層畫在上、深層畫在下，於是 top-down 在圖上往上走，PAN 的 bottom-up 在圖上往下走。頁面只能用兩條說明提醒「和名字的方向相反」，讀者要同時記住兩套相反的上下，看圖時很容易把 top-down 和 bottom-up 對錯。 |
| 5 | 建議 | 第 126 行「若改用 bilinear（雙線性插值：新格是相鄰來源值依距離的加權平均），放大結果會出現 1.25、1.75 這類中間值」 | 數學好的讀者若照「依距離的加權平均」自己算，最自然的做法是把 4 個新格等距放在兩個來源格之間（位置 0、1/3、2/3、1），得到 4/3、5/3，和頁面的 1.25、1.75 對不上，會以為自己算錯或頁面寫錯。1.25、1.75 來自 PyTorch 預設的 align_corners=False：新格中心換回來源座標是 −0.25、0.25、0.75、1.25，兩端再夾到邊界。我在暫存副本實測，第一列是 [1, 1.25, 1.75, 2]；align_corners=True 時則是 [1, 1.333, 1.667, 2]。 |
| 6 | 建議 | 第 126 行「換成別的 loss，nearest 與 bilinear 的梯度通常就不同，例如本節完整程式用的「輸出平方的平均」」 | 這句容易讀成「完整程式有示範這個差異」。但程式只對 Fusion 的輸出用平方平均，從沒拿這個 2×2 例子比較 nearest 與 bilinear，輸出裡也找不到相關數字，讀者無法核對「通常就不同」。其實這個例子可以手算。 |
| 7 | 建議 | 第 27 行「YOLOv4 與固定版本 YOLOv5 v6.0 模型配置：各有自己的 neck 設計。」（連帶第 33 行「也不聲稱重現完整的 YOLOv4／v5 neck」） | 這一條沒有給讀者任何可以抓住的內容。讀者不知道 YOLOv4／v5 的 neck 長什麼樣，第 33 行的「不聲稱重現」也就無從理解；連結指向一個 YAML 設定檔，頁面也沒說要看哪裡。其實 YOLOv5 v6.0 的 yolov5s.yaml 在 `head:` 底下，每次 top-down 合併都是 `Conv [512,1,1]` → `nn.Upsample [None,2,'nearest']` → `Concat`（接 backbone P4）→ `C3`，和本節的 reduce→nearest→concat→mix 同型，lateral 那支也沒有 1×1；之後再接 3×3 stride 2 → Concat → C3 的 bottom-up 路徑。讀者錯過了最直接的對照。 |
| 8 | 建議 | 第 107 行〈進階：shape 對了，位置不一定對〉「不同裁切造成的位移也一樣」，以及第 109 行「逐層追蹤 stride、padding 與裁切」 | 「裁切」在前面各節（4.2、8.1）指的是把框夾回原圖範圍內；這裡指的卻是把特徵圖或圖片的邊緣切掉。兩種意思不同，頁面也沒有說明或舉例。讀者會帶著舊的意思讀，看不懂裁切為什麼會造成位移。 |
| 9 | 建議 | 第 7 行括號「（這和下面的 31×31 都只算理論範圍；最上一列與最左一欄的格子，這個範圍有一部分落在 padding 補的 0 上；右、下兩邊也補了 0，只是用不到）」 | 第 1 章的結論是「靠圖邊的格子」範圍都會落到 padding 上，這裡卻說只有上、左兩邊，右、下補的 0「用不到」，而且沒給理由。數學好的讀者想驗證也無從下手（我用梯度實測確認：8×8 的第 (7,7) 格看到原圖第 49～63 列，第 (0,0) 格只看到原圖 0～7）。這段括號又插在開場最關鍵的一段中間，打斷「語義 vs 位置」的主線，和融合本身無關。 |
| 10 | 建議 | 第 33 行「本節簡化成一條深→淺的路徑：把深層特徵放大、concat、再卷積」 | 這句摘要漏了第一步 reduce（1×1 減 channel）。後面的「整個融合分成四步」和圖都把 1×1 reduce 放在最前面，讀者對照時會以為摘要和四步是兩種不同的做法。 |
| 11 | 建議 | 第 64 行「好處之一是參數比較少……減完之後，兩路也各占 8 個 channel。」 | 「好處之一」暗示還有別的好處，但最後一句「兩路也各占 8 個 channel」沒說這有什麼用，讀者讀完會問「所以呢？」。程式第 45 行的註解顯示它和第 70 行「改用相加時 channel 數必須相同」有關，頁面卻沒有把兩者接起來。 |
| 12 | 建議 | 第 168 行「深層的低解析度資訊，不一定能救回已經丟失的小物件。」 | 「已經丟失」沒說丟在哪裡：是輸入圖縮小時就消失了（第 10 章說的「已被 resize 到 2 pixel」），還是在 4×4 的深層裡被一格 16×16 的範圍稀釋掉？讀者無法判斷這句在警告什麼，也看不出它和融合的關係。 |
| 13 | 建議 | 第 174 行常見錯誤「把上取樣當成框座標的換算：……框的座標不會因此跟著乘 2」 | 讀者不清楚這條錯誤實際是什麼動作：要把什麼座標乘 2？會用錯哪個 stride 解碼？這條只有結論、沒有情境，讀者不容易對應到自己可能犯的錯。 |
| 14 | 建議 | 第 175 行常見錯誤「先對整張特徵圖做 global pooling，又期待之後能定位」 | 本節的模型沒有任何 pooling，這條錯誤和融合有什麼關係沒有交代，讀者會不明白它為什麼列在這一節。 |
| 15 | 建議 | 第 182 行練習 2「mix 用 3×3（padding=1）把 24 channel 混成 12」 | 頁面要讀者「先自己算，再展開答案」，但第 2 題的題目已經寫出第 1 題第二問的答案（mix 輸入 24 channel）。讀者一眼看到兩題，第 1 題就被破梗了。 |
| 16 | 建議 | 第 15 行「backbone（特徵提取網路）」；圖中「每格間距 8 畫素」「每格間距 16 畫素」 | 術語表、第 1 章、第 10 章和 11.1 節都把 backbone 譯成「主幹」，本頁卻改稱「特徵提取網路」。讀者在術語表用 Ctrl+F 找「特徵提取網路」會找不到，因為術語表寫的是「從圖片提取特徵的主幹」。另外，本頁正文一律寫 pixel（15×15 pixel、相隔 8 pixel），圖中卻寫「畫素」，同一個量在同一頁有兩種寫法。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：Read files (暫存副本:
- 審查用的事實與寫作規範清單; docs/lessons/11-fusion.md; lesson_cases/11-fusion.py (imports only torch, no miniyolo); docs/assets/diagrams/11-fusion.svg.
- For cross references: lesson_cases/10-multiscale.py (TwoScale: three Conv2d(·,·,3,2,1) layers, deep 16→32, fine_head Conv2d(16,7,1)); docs/lessons/10-multiscale.md, 01-small-cnn.md, 02-diagnostics.md, 03-identity.md, 03-projection.md, 04-localization.md, 06-evaluation.md, 11-csp.md and lesson_cases/11-csp.py; docs/glossary.md; section-map.json; scripts/validate_lessons.py; artifacts/checks/curriculum/11-fusion.json.

Commands run (all in the scratch copy):
1. The specified rsync, to build the scratch copy.
2. `PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/11-fusion.py`: exit 0, three lines of output.
3. 摘錄比對: printed [].
4. `qlmanage -t -s 1200 -o .../11-fusion-tech-render docs/assets/diagrams/11-fusion.svg`, then looked at the PNG.
5. Scratch scripts in 暫存副本:
   - check1.py: swap order of 1×1 and nearest (torch.equal True); bilinear with align_corners False and True (output values, sum-loss gradient, mean-square gradient); nearest mean-square gradient; dim=3 concat error; reduce→12 channels error; sizes for a 100×100 input; scale_factor=2 concat error; 6→8 nearest index map [0,0,1,2,3,3,4,5].
   - fusion_bypass.py + check2.py: replace reduced with deep[:,:8]; reduce.weight/bias grad None; main() AssertionError at line 35.
   - check3.py: receptive field through gradients (8×8 grid: 15×15, cell (0,0) clipped; 4×4 grid: 31×31).
   - check4.py: parameter counts and exercise answers; 4096 bytes; no-padding 3×3 gives 6×6.
6. `.venv-docs/bin/zensical build --clean --strict` (exit 0) and `python3 scripts/validate_site.py` (exit 0, every check passed).
7. A punctuation and spacing scan of the page.
8. A comparison of the sha256 of lesson_cases/11-fusion.py with the record's case_sha256.
9. The local PyTorch 2.9.1 F.interpolate docstring: "align_corners ... Default: False".

Sources:
- FPN arXiv 1612.03144, https://ar5iv.labs.arxiv.org/html/1612.03144, §3 paragraphs "Bottom-up pathway" and "Top-down pathway and lateral connections". Quotes: "The bottom-up feature map is of lower-level semantics, but its activations are more accurately localized as it was subsampled fewer times."; "...merged with the corresponding bottom-up map (which undergoes a 1×1 convolutional layer to reduce channel dimensions) by element-wise addition."; "To start the iteration, we simply attach a 1×1 convolutional layer on C5 to produce the coarsest resolution map."; "Finally, we append a 3×3 convolution on each merged map ... to reduce the aliasing effect of upsampling."; "There are no non-linearities in these extra layers".
- PANet arXiv 1803.01534, https://ar5iv.labs.arxiv.org/html/1803.01534, §3.1 Motivation. Quote: "a 'shortcut' ... which consists of less than 10 layers ... the CNN trunk in FPN gives a long path ... passing through even 100+ layers". Also the Augmented Bottom-up Structure paragraph.
- YOLOv3 arXiv 1804.02767, https://ar5iv.labs.arxiv.org/html/1804.02767, §2.3. Quote: "merge it with our upsampled features using concatenation".
- YOLOv4 arXiv 2004.10934, https://ar5iv.labs.arxiv.org/html/2004.10934. §2.1 gives the neck definition; §3.1: "We use PANet ... instead of the FPN used in YOLOv3"; §3.3: "replace shortcut connection of PAN to concatenation"; §3.4: Neck: SPP, PAN.
- ultralytics/yolov5 tag v6.0 = commit 956be8e642b5c10af4a1533e09084ca32ff4f21f (resolved with gh api):
  - models/yolov5s.yaml head lines 29–47: `[-1, 1, Conv, [512, 1, 1]]`, `[-1, 1, nn.Upsample, [None, 2, 'nearest']]`, `[[-1, 6], 1, Concat, [1]]  # cat backbone P4`, `[-1, 3, C3, [512, False]]`.
  - models/common.py lines 36–42 (Conv = Conv2d(bias=False) + BatchNorm2d + SiLU), lines 125–132 (C3), line 266 (Concat).
  - models/yolo.py.
- pjreddie/darknet commit f6afaabcdf85f77e7aff2ec55c020c0e297c77f9, cfg/yolov3.cfg lines 618–633: [route] layers=-4 → [convolutional] size=1 filters=256 → [upsample] stride=2 → [route] layers = -1, 61. Layer 61 is the stride-16 [shortcut] output.
- facebookresearch/Detectron commit 04155a01a6ea68f22ac27c79a822066457941ece, detectron/modeling/FPN.py:
  - add_fpn lines 141–215: "# For the coarsest backbone level: 1x1 conv only seeds recursion"; "# Post-hoc scale-specific 3x3 convs / for i in range(num_backbone_stages)", which includes the coarsest level.
  - add_topdown_lateral_module lines 259–297: lateral 1x1, UpsampleNearest scale=2, Sum.

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 「歷史機制」YOLOv4／YOLOv5 那一項（第 27 行），以及「和原始 FPN 相比，有兩處不同」段落（第 35 行） | 頁面只說 YOLOv4 和 YOLOv5 v6.0「各有自己的 neck 設計」，又把「lateral 那一支沒有 1×1」列為和 FPN 的差異，但沒有告訴讀者：本節 reduce→nearest→concat→mix 的接法，正是 YOLOv3 和 YOLOv5 v6.0 top-down 一步的實際接法。 - darknet yolov3.cfg@f6afaab 第 618–633 行：`[route] layers=-4` → 1×1 `[convolutional] filters=256` → `[upsample] stride=2` → `[route] layers = -1, 61`。第 61 層是淺層，直接串接，沒有經過 1×1。 - yolov5s.yaml@956be8e head 前四行：`[-1, 1, Conv, [512, 1, 1]]`、`[-1, 1, nn.Upsample, [None, 2, 'nearest']]`、`[[-1, 6], 1, Concat, [1]] # cat backbone P4`、`[-1, 3, C3, [512, False]]`。 結果有兩個：讀者會以為「lateral 不做 1×1」是本課自己的簡化；而本節相對 YOLOv5 真正的簡化又都沒有列出，包括 mix 只是一個 3×3 而不是 C3、沒有 BN 與 SiLU（common.py 第 36–42 行的 Conv 是 Conv2d(bias=False)+BatchNorm2d+SiLU）、只有一次深→淺、沒有三尺度與 bottom-up。 |
| 2 | 建議 | 「和原始 FPN 相比」段落末句「深層那一支的 1×1（本節的 reduce）FPN 也有」（第 35 行），以及第 37 行粗 head 的說明 | FPN 接在 C5 上的 1×1，產生的是最粗一層的特徵圖。論文原文是 “To start the iteration, we simply attach a 1×1 convolutional layer on C5 to produce the coarsest resolution map.”。這張圖也是 P5 的來源：官方實作 Detectron FPN.py@04155a0 的 “Post-hoc scale-specific 3x3 convs” 迴圈涵蓋最粗層在內的每一層，接上 3×3 後才交給 head。 本節的 reduce 只供 top-down 使用；接回第 10 章時，粗 head 仍直接吃未經處理的 deep。頁面把 reduce 對應成 FPN 的 C5 1×1，又寫「有兩處不同要分清楚」，讀者會以為除了這兩處，本節 neck 其餘部分都和 FPN 相同。 |
| 3 | 建議 | 〈延伸：FPN 也在合併後接 3×3 卷積〉（第 101 行） | FPN 論文給的理由只有一句：“we append a 3×3 convolution on each merged map to generate the final feature map, which is to reduce the aliasing effect of upsampling.”，沒有說明 aliasing 指什麼。頁面接著寫「大致是指放大後出現的假紋路，例如 nearest 複製出的 2×2 方塊邊界。這是 FPN 論文的說法」，讀者會把「2×2 方塊邊界」這個例子也當成論文本身的說法。 |
| 4 | 建議 | 「用四個數看上取樣」中 bilinear 那一段（第 126 行） | 1.25、1.75 只在 PyTorch `F.interpolate(..., mode='bilinear')` 用預設 `align_corners=False` 時出現（PyTorch 2.9.1 docstring：Default: False）。實測設成 `align_corners=True` 時，同一個例子第一列是 `[1.0, 1.333…, 1.667…, 2.0]`。 這一段正要強調「必須記錄用的是哪一種插值方法」，卻沒提到同一種 bilinear 還有這個會改變數值的設定。讀者自己試 `align_corners=True`，會對不上頁面的數字。 |
| 5 | 建議 | 「常見錯誤」global pooling 那一項的最後一句（第 175 行） | 「也只是間接線索，不能靠它定位」比第 4 章的說法更絕對。第 4 章〈單物件分類與定位〉明說「『平均這一步分不出位置』不等於『只用 GAP 的模型一定不能定位』」，並說這種線索「間接又不可靠」；第 1 章也只說「不宜只靠 GAP 之後的特徵」。這裡的「不能」容易被讀成「不可能」，和第 4 章的區分相矛盾。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 技術查核 | 沒指出本節接法和 YOLOv3／v5 的 top-down 相同，也沒列出相對 v5 的簡化 | 已修正：第 35 行補上：兩處差異都和 YOLOv3、YOLOv5 v6.0 的 top-down 合併相同；第 33 行補上相對 v5 的簡化（mix 不是 C3、沒有 BN／SiLU、沒有第三尺度與 bottom-up）。已對照 956be8e 的 yolov5s.yaml head |
| 2 | 技術查核 | FPN 最深層的 1×1 輸出也是 P5 的來源 | 已修正：補一句：FPN 最深層的 1×1 輸出會再經 3×3 交給它自己的 head；本節的 reduce 只用在 top-down，粗 head 直接接 deep。「有兩處不同」改成「有幾處不同」 |
| 3 | 技術查核 | aliasing 的解讀被寫成論文的說法 | 已修正：分開寫：論文的理由只有一句；「2×2 方塊邊界」標明是常見的理解 |
| 4 | 技術查核 | bilinear 的 1.25／1.75 取決於 align_corners | 已修正：寫明 PyTorch 預設 align_corners=False，並列出 True 時的值（4/3、5/3）；結論句補上「以及 align_corners 這類設定」。暫存副本實測核對 |
| 5 | 技術查核 | global pooling 說成「不能定位」，和第 4 章不一致 | 已修正：改成「間接、不可靠的線索，不宜靠它定位（見第 4 章）」 |
| 6 | 讀者審查 | 「捷徑」和 shortcut 的術語衝突 | 已修正：改成「短路徑」 |
| 7 | 讀者審查 | FPN 的 1×1 在哪一支，前後說法看似矛盾 | 已修正：第 25 行改成「新接進來的 lateral 那一支先經過 1×1」；第 35 行把 FPN 每層 1×1、top-down 起點、之後只有 lateral 接 1×1 的順序講開 |
| 8 | 讀者審查 | concat 條件只寫 H、W | 已修正：改成 channel 以外的軸（B、H、W）都要相同，本例要處理的是 H、W |
| 9 | 讀者審查 | 圖的上下和金字塔方向相反 | 未改：翻轉整張圖是版面重畫，不算小修改；頁面已有兩條看圖說明交代方向相反。在現有的版面加方向標籤，寬度也放不下 |
| 10 | 讀者審查 | bilinear 的數值算不出來 | 已修正：補上算法：新格中心換回來源座標是 −0.25、0.25、0.75、1.25，超出範圍用邊上的值，所以是 0.75×1+0.25×2＝1.25 等 |
| 11 | 讀者審查 | 「本節完整程式用的平方平均」像是程式有示範 | 已修正：註明程式沒拿 2×2 例子比較，並給出可手算的數字：nearest 是 s/2＝[[0.5,1],[1.5,2]]，bilinear 是 [[0.78125,…]]。暫存副本實測核對 |
| 12 | 讀者審查 | YOLOv4／v5 那一條沒有內容 | 已修正：改寫成具體內容：兩者都有 FPN＋PAN；YOLOv4 把 PAN 的相加改成 concat；v5 每次 top-down 合併是 1×1→nearest→concat→C3，和本節四步同型 |
| 13 | 讀者審查 | 「裁切」的意思和前面章節不同 | 已修正：加括號定義：把特徵圖邊緣切掉幾格，和前面把框夾回原圖不是同一件事 |
| 14 | 讀者審查 | padding 的括號打斷主線，「約」也不必要 | 已修正：刪掉括號，改成「理論感受野是 15×15／31×31 pixel」（理論值剛好是這兩個數） |
| 15 | 讀者審查 | 摘要句漏了 reduce | 已修正：改成「先用 1×1 減少深層 channel，再放大、concat，最後 3×3 混合」，和四步對應 |
| 16 | 讀者審查 | 「兩路各占 8 個 channel」沒說有什麼用 | 已修正：補上用處：改用相加時 channel 數已經對齊（見後面說明） |
| 17 | 讀者審查 | 「已經丟失的小物件」講得太模糊 | 已修正：寫成具體情境：深層每格間距 16 pixel，小物件的訊號被背景沖淡；放大只是複製，補不回細節 |
| 18 | 讀者審查 | 上取樣與框座標的錯誤缺少情境 | 已修正：補上數字例子：格 (1,1)、偏移 0.5，用 stride 8 是 12 pixel，誤用 16 會得到 24 pixel（和第 10 章 decode 的寫法一致） |
| 19 | 讀者審查 | global pooling 那條和本節的關係沒交代 | 已修正：補一句接回本節：深層 pooling 成 1×1 再上取樣回 8×8 時，64 格拿到的深層值全都一樣 |
| 20 | 讀者審查 | 練習 2 洩漏了練習 1 的答案 | 已修正：改成「輸入 channel 用第 1 題的答案，輸出 12 channel」 |
| 21 | 讀者審查 | backbone 譯名、圖中「畫素」用字 | 已修正：正文改成「backbone（主幹，從圖片提取特徵的前段網路）」；SVG 的「畫素」改成 pixel（正文與 desc 各兩處） |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/11-csp.md`、`docs/lessons/11-fusion.md`、`docs/lessons/11-augmentation.md`、`docs/lessons/11-iou-loss.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/11-fusion.md 第 35 行 | 「FPN 讓 backbone 的每一層各接一個 1×1」範圍說大了。FPN 只在每個 stage 的輸出（C2～C5，也就是金字塔的每個尺度）接 1×1，不是每一層卷積都接。同一頁的 PAN 摺疊區用「上百層」指卷積層，初學讀者可能以為 FPN 在上百層上各加一個 1×1。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

對照 FPN 論文第 3 節：C5 先接 1×1 作為 top-down 起點，每次合併只有 lateral 那支接 1×1，合併用逐元素相加，再接 3×3 減少 aliasing。「每個尺度（stage）的輸出各接一個 1×1」與「只有新接進來的 lateral 那支接 1×1」正確；YOLOv3、YOLOv5 v6.0 都是先 1×1、再上取樣、再 concat 淺層，也正確。本頁也確實教了術語表 concat／add 列寫的條件。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

讀了 reviews/11-fusion.md 的讀者審查、技術查核、修正後檢查與後續編輯檢查。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/11-fusion.md | 21 項 should 只有一項（圖的上下方向）在修正後檢查裡提到沒採納，其餘都沒有處理，指向的〈定稿修正〉不存在（原因同上）。 | 已修正：產生器改以頁名、節名、萬用字元、頁面上的圖與該節 notebook 把處理對應到頁面，〈定稿修正〉列出這一頁每一項的處理。 |
| 2 | 建議 | 〈技術查核〉方法 | 方法是英文，含內部路徑與殘句：「Read files (暫存副本: …」「`python3 .../摘錄比對工具 <暫存副本> docs/lessons/11-fusion.md`: printed []」「暫存副本 scripts in .../暫存副本: - check1.py: …」，而且在列出查閱的論文與固定 commit 之前就截斷，來源只散見於發現欄。 | 已處理：英文的方法段是查核者的原文，保留不譯。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：讀者 16 項與技術 5 項都在〈定稿修正〉處理（上一輪指出缺漏，已補）；技術查核列出 FPN、PANet、YOLOv4 論文與 yolov5 固定 commit；英文方法段依處理欄決定保留原文，這點不再重報。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/11-fusion.md 第 26、119、126、128 行 | 〈上述處理〉第 2 項寫「部分修正：…路徑與殘句已清理」，但它點名的三句都原樣留著：「方法：Read files (暫存副本:」「3. `python3 .../摘錄比對工具 <暫存副本> docs/lessons/11-fusion.md`: printed [].」「5. 暫存副本 scripts in .../暫存副本:」；第 26 行「3. 清單先前審查意見和受程式改動影響的段落清單：」也是逐字替換造成的病句。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項（部分修正）：第 26 行病句已改通順，第 126 行指令已改好；但處理說明對保留了什麼寫得不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/11-fusion.md 第 119、128 行與第 3 輪第 1 項處理 | 處理寫「部分修正：路徑與工具殘句已清理…；英文方法段是查核者的原文，保留」，但點名的路徑殘句「方法：Read files (暫存副本:」（第 119 行，括號沒有關）與「5. 暫存副本 scripts in 暫存副本:」（第 128 行）仍在。處理沒說這兩句保留，反而說路徑殘句已清理；而且這兩句已被替換工具改過，並不是查核者的原文。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

第 4 輪第 1 項屬實：「方法：Read files (暫存副本:」在第 119 行，「5. 暫存副本 scripts in 暫存副本:」在第 127 行，舊的處理說法已重寫。第 3 輪第 1 項中，第 26 行病句和原第 126 行的指令確實已改寫。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 3 輪第 1 項處理欄（第 218 行） | 「5 處中 4 處已改寫或刪除」把發現引用的第 2 輪處理說法「部分修正：…路徑與殘句已清理」也算成已改寫，但它仍在第 2 輪第 2 項處理欄（第 210 行）。實際改寫的紀錄文字是 3 處。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[evolution當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/evolution.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/evolution.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/evolution.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_a` 實讀本頁 11 個凍結單元；首次使用／前文方法範圍四題位置為 11-fusion/00:first_use, 11-fusion/02:first_use, 11-fusion/04:first_use, 11-fusion/07:first_use，頁末為 11-fusion/10。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_a.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_a-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_a.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁未有需要改寫的已裁定問題，保留原教學內容；仍完整重讀與核對。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 c1 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `c_foundation` 實讀本頁 9 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/c_foundation/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：R007。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/11-fusion.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
