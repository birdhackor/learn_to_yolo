# 審查紀錄：CSP

審查範圍：`docs/lessons/11-csp.md`、頁面上的圖（`docs/assets/diagrams/11-csp.svg`），以及 `lesson_cases/11-csp.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要或建議等級的問題。全部在暫存副本檢查，repo 內沒有執行任何程式、也沒有修改。

1. 程式敘述（must，通過）
- 逐句對照最終版 lesson_cases/11-csp.py，第 51–52 行是新的參數梯度斷言。斷言清單、印出內容、名稱都與程式一致。
- 課程程式 exit 0，stdout 與 artifacts/checks/curriculum/11-csp.json 的 6 行逐字相同。
- 突變實測（mutate.log、details.log）：
  - 漏呼叫 branch：兩個輸入梯度總和仍非零，branch 四個參數的 grad 都是 None，在第 51 行報 AssertionError，與新增第 78 行的說法一致。
  - branch 輸出乘 0：在同一道斷言報錯。
  - dim=3：在卷積報 RuntimeError（expected 4 channels, got 8），與「常見錯誤」一致。
  - 兩半對調：全部通過；頁面沒有宣稱這種錯抓得到。
- 第 118 行括號「梯度全為 0 時 SGD 不改參數」：用 SGD lr=.01 實測成立。
- 兩題自主練習照題意計算：C=10 時 CSP 570、Full 1930、比值約 0.30；逐值相加後 [B,4,8,8]，8→8 fuse 報錯，Conv2d(4,8,1) 輸出 [B,8,8,8]。與參考答案相符。
- 交叉引用：02-diagnostics.md 第 38 行「backward 後仍是 None 就是沒接上」、03-identity 的 requires_grad=True 與 [1,2]/[10,20] 範例，在現行工作樹都成立；新連結格式與全站一致。

2. trace／受程式改動影響的段落（must，通過）
- 第 3、104 行的 Colab tag 在 HEAD 已是 lessons-v0.4.0，validate_site 的 colab_pairs 通過。
- 第 116、125 行替弱斷言補的說明已移除並改寫。
- 第 72、76、121 行照受程式改動影響的段落與查核處理；第 163 行以下的執行紀錄區塊未改。
- SVG 兩條先前審查意見（第 65 行與 &lt;desc&gt;）已改。
- 全頁沒有修訂或審查敘事，也沒有待辦口吻；頁面與 SVG 都沒有宣稱斷言能分辨哪一半進卷積支路。

3. 數字（must，通過）：新增文字沒有引入數字。確定值 1240、368、1168、296、72、570、1930 都正確；依紀錄而定的三處，編輯者都已列出。

4. 摘錄（must，通過）：唯一的 python 區塊是簡化改寫版，沒有標記，前導句明說是簡化；摘錄比對工具輸出 []；正文沒有程式行號。

5. 可讀性（should，通過）：新段落用「漏呼叫 branch」這個具體情境說明為什麼還要查參數梯度，段落順序合理。

6. 建置（must，通過）
- zensical build --clean --strict 與 validate_site.py 都通過。
- SVG 用 qlmanage 渲染正常，viewBox、title、desc 齊全，內容與程式一致。
- 11-csp 相關檔中，只有頁面與 SVG 和 HEAD 不同。

附帶觀察（不影響判定）：編輯者疑慮第 3 點說 notebooks/11-csp.ipynb「還沒重建」，這不完全準確。最後一格程式已等於現行 11-csp.py，source_ref 也是 lessons-v0.4.0，只是 outputs 是空的，要等紀錄重產後填入。暫存副本裡 validate_lessons.py 失敗是在 09-anchor-clustering 與 20-deployment 的摘錄，不是本頁；本頁只有審查涵蓋缺件（no review），屬預期。

證據在暫存副本，包括 mutate.py、mutate.log、details.py、details.log、build.log、validate_site.log、render/11-csp.svg.png。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：讀了哪些檔案：
- 審查用的事實與寫作規範清單
- docs/lessons/11-csp.md：讀了 Markdown，也讀了在暫存副本用 `zensical build --clean --strict` 嚴格建置後的 site/lessons/11-csp/index.html 正文。建置 exit 0、No issues found。摺疊區、表格、MathJax、圖片路徑都正確。
- lesson_cases/11-csp.py
- notebooks/11-csp.ipynb：最後一格與 lesson_cases 逐字相同。
- docs/assets/diagrams/11-csp.svg：讀了原始碼，並用 qlmanage 轉成 1200px PNG 目視。
- docs/glossary.md
- 前置與前後節：01-small-cnn、02-diagnostics、03-identity、03-projection、03-comparison、10-multiscale、11-fusion 開頭、14-feature-module 的相關段落，以及 learning-path、planning/outline。

執行了什麼：
- 在暫存副本跑 `PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/11-csp.py`，exit 0，6 行輸出與頁尾紀錄逐字相同。

練習（頁面指示手算）：
- 練習 1：先手算出 5×5×9+5=230、230×2+110=570、Full 為 1930、比值 0.2953 對 0.2968。再寫 C=10 的 CSP 與 Full，用 PyTorch 數參數，得到 570、1930，輸出 (2,10,8,8)，與參考答案一致。
- 練習 2：手算後實測。相加得到 (2,4,8,8)；接 8→8 的 fuse 會報 RuntimeError（expected input[2, 4, 8, 8] to have 8 channels, but got 4）；改用 Conv2d(4,8,1) 得到 (2,8,8,8)。與參考答案一致。

核對正文的主張（都在暫存副本中修改後執行）：
- chunk 改成 dim=3：報 RuntimeError（input[2, 8, 8, 4]… got 8 channels）。
- forward 漏呼叫 branch：參數梯度斷言報 AssertionError。這時輸入梯度兩個總和仍非零（0.372767／0.454024），branch 參數的 grad 都是 None。
- 卷積支路改成 padding=0：輸出 (2,4,4,4)，torch.cat 報「Sizes of tensors must match except in dimension 1」。
- 支路輸出歸零、fuse 設成單位矩陣：輸出不等於 x。
- seed 0–49：Full 的前／後 4 通道比值落在 0.84–1.33，CSP 落在 5.13–26.03。
- 改用 He 初始化（fan_out、ReLU）：CSP 比值的中位數從約 9.6 降到約 0.75，確認差距來自預設初始權重。4→4 的 3×3 權重落在 ±1/6，1×1 fuse 落在 ±1/√8。
- 用 grep 確認 miniyolo/ 裡沒有 CSP；頁面引用的第 1、2、3.1、3.3、10 章內容（4 倍、[1,2]/[10,20]、requires_grad、ImageNet、None 梯度、Darknet-53）都確實存在。

限制：
- Playwright 的瀏覽器沒有安裝，所以沒有截圖，只讀了建置後的 HTML。
- 依指示，沒有把本機的計時或訓練數字當成紀錄機器的值；本節正文也沒有引用紀錄裡的數字。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/11-csp.md 第 17 行〈歷史與版本〉CSPNet 的梯度說法：「至於截斷梯度流、減少這種重複，論文歸功於末端的融合順序」 | 「截斷梯度流」沒有解釋。照字面，讀者會以為是「梯度傳不回去」。這和本節主文反覆強調、程式也用斷言確認的「兩路都接通、都收到梯度」正好相反。摺疊區最後只說「本節只檢查兩條路都收得到梯度」，沒有化解這個看似矛盾的地方。另外，「至於 X，論文歸功於 Y：…」的句構很繞，高中讀者讀到冒號後已經抓不到主詞。 |
| 2 | 建議 | docs/lessons/11-csp.md 第 36 行「fuse（融合用的 1×1 卷積，8→8）」、第 44 行程式註解「融合用的 1×1 卷積」；對照第 10 章第 15 行「融合留到第 11 章再單獨加入」、術語表 neck 列（連到 11.2） | 讀者剛讀完第 10 章「融合留到第 11 章」，一進 11.1 就看到「融合」，容易以為這就是第 10 章預告的特徵融合。其實本節的「融合」只是在同一位置用 1×1 卷積混合 8 個通道；下一節〈特徵融合〉才是把不同尺度的特徵放大後合併。同一個詞在相鄰兩節意思不同，頁面沒有提醒，術語表也只把「融合」連到 11.2。 |
| 3 | 建議 | docs/lessons/11-csp.md 第 5 行「最後再把兩半接回來、混合一次」、第 36 行、第 100 行 | 頁面說明了 fuse 怎麼算（每個輸出通道是 8 個輸入通道的加權和），卻沒說為什麼需要它。讀者自然會問：既然串接後已經回到 8 個通道，為什麼還要多一層 1×1？少了這個理由，讀者也看不出 CSP 和「一半通道永遠不處理」有什麼差別。 |
| 4 | 建議 | docs/lessons/11-csp.md 第 13 行「計算量約減少 20%」；第 98 行「約剩三成」、第 135 行「本例的參數從 1240 降到 368」 | 同一頁先引用論文「計算量約減少 20%」，後面的玩具 block 卻省了約七成。讀者容易以為本節比論文還省，或以為 CSP 一般能省七成。頁面沒說明兩個數字量的範圍不同：論文是整個網路的計算量，含沒套 CSP 的層；本節只是單一 block 的參數。 |
| 5 | 建議 | docs/assets/diagrams/11-csp.svg 第 65 行圖底「本節簡化：卷積帶 bias。」；同圖第 36–37、59–60 行「3×3＋ReLU」、第 55–56 行「前 4 channels／後 4 channels」、第 58 行「split：4+4」；頁面第 56 行「看圖時注意」 | 「本節簡化：卷積帶 bias」讀起來像「帶 bias 就是簡化」。正文只在第 52 行說 nn.Conv2d 預設帶 bias，完全沒提相對什麼而簡化：官方 C3 的 Conv 是不含 bias 的卷積→BN→SiLU，這點第 14 章才講，本節的正文與摺疊區都沒提。另外，這頁的核心是「串接不是相加」，圖卻用「＋」表示「接在後面」，得靠第 56 行另加警語。圖中的英文複數「channels」與「split」，也和正文的「前 4 個通道」「chunk／切分」用字不同。 |
| 6 | 建議 | docs/lessons/11-csp.md 第 11 行「要在固定預算下做對照實驗才知道：固定資料、訓練步數與評估方式，只換這個 block」；〈本節能驗證／不能驗證〉第 129 行「這是容量成本的比較，不是在相同參數量下比較準確度的公平實驗」、第 131 行 | 「固定預算」沒說是什麼預算。冒號後列的是資料、步數與評估方式，但「預算」也常指參數量或計算量。第 129 行又出現另一種「公平實驗」（相同參數量下比準確度），和第 11、131 行「同一位置換 block、固定資料與步數」不是同一種，頁面沒說兩者各回答什麼問題。「容量成本的比較」也不成詞，讀者看不出比的是什麼。 |
| 7 | 建議 | docs/lessons/11-csp.md 第 85 行摺疊區「在 PyTorch 預設的初始權重下，梯度穿過這兩層後會小很多。」 | 這裡只給結論、沒給理由。數學好的讀者會想知道為什麼穿過兩層就變小，也可能誤以為這是 CSP 結構本身的缺點。我在暫存副本改用 He 初始化（fan_out、ReLU），30 個 seed 的前／後 4 通道比值中位數從約 9.6 降到約 0.75，可見原因確實是預設初始權重的尺度。 |
| 8 | 建議 | docs/lessons/11-csp.md 第 21 行〈歷史與版本〉YOLOv5 段 | 這段有三個地方會讓讀者卡住： 1. 一句話中間塞了一大段括號（Bottleneck 的定義、shortcut 的預設值、逐值相加），括號結束才接回主句「兩支串接後由第三個 1×1 卷積融合」，讀到這裡已經忘了主句在講什麼。 2. 「參數 `shortcut`」的「參數」是建構時傳入的選項，但本頁其他地方的「參數」都指可學的權重（1240／368），容易讀成 shortcut 是可學參數。 3. 「設定檔寫在 `head:` 底下」的 C3，和本書「head 是輸出預測的末端」不符，讀者會疑惑 C3 怎麼在 head 裡。 |
| 9 | 建議 | docs/lessons/11-csp.md〈執行與核對〉第 113–119 行斷言清單；頁尾紀錄第 3、5 行 `full/CSP output (2, 8, 8, 8) finite backward and step` | 頁面逐一說明了紀錄第 1 行（concat／add）、兩行梯度總和與最後的參數數，卻沒說每個模型第二行 `finite backward and step` 代表什麼。讀者對照紀錄時，不知道這一行表示哪些檢查通過了。 |
| 10 | 建議 | docs/lessons/11-csp.md 第 139 行「通道本來就很少、資料又很難時，切得太多未必合適。」 | 「切得太多」有兩種讀法：切成太多份，或讓太多通道走旁路。本節只切一次、切成兩份，讀者不確定指的是哪一種。 |
| 11 | 建議 | docs/lessons/11-csp.md〈本節能驗證／不能驗證〉第 127 行與第 130 行 | 「不能證明重複梯度資訊減少」第 127 行已經說過，第 130 行又說一次（第 17 行摺疊區也說過）。而且第 130 行用「所以」連接：「平方 loss 和偵測無關，所以不報告 AP，也不量測重複的梯度資訊」。loss 和偵測無關可以解釋為何不報告 AP，卻不是不量測重複梯度的理由，因果不通。 |
| 12 | 建議 | docs/lessons/11-csp.md 第 46 行程式摘錄 `bypass, transformed = x.chunk(2,dim=1)`（完整程式第 13 行同名） | 對程式新手來說，`transformed`（已轉換的）聽起來像卷積後的結果，其實是還沒進卷積支路的後 4 個原始通道；下一行的 `processed` 才是轉換後的結果。註解雖然寫了「送進卷積支路」，名字本身仍會誤導。 |
| 13 | 建議 | docs/lessons/11-csp.md 第 144 行常見錯誤「兩支的高、寬不同還要串接」 | 本節的兩支都來自同一個 x，而且第 32 行說了 stride 1、padding 1 會保持 8×8。讀者看不出高、寬怎麼會不同，這條錯誤顯得抽象。 |
| 14 | 建議 | docs/lessons/11-csp.md 第 91 行「k 是卷積核邊長」與公式 `out×in×k²+out`；第 38 行「依完整程式的 `CSP` 類別」 | 有兩處用字和前面的章節或本書慣用意思不一致： 1. 照課程順序，「卷積核」是在這裡第一次出現；第 1 章一直用「濾鏡（kernel）」，讀者可能以為是新東西。公式寫法也和第 1 章的 C_out(C_in k²+1) 不同，頁面沒說是同一條。 2. 「CSP 類別」的「類別」在本書多半指物件類別（類別分數、類別數），這裡卻是 Python 的 class。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：工作目錄：暫存副本（照指示的 rsync 建立）。來源檔放在同層的 11-csp-tech-src2，探測程式放在 11-csp-tech-work。沒有在 repo 根目錄裡執行或寫入任何東西。

一手來源：
(1) CSPNet，arXiv 1911.11929v1。https://arxiv.org/abs/1911.11929 （摘要、submission history），PDF 為 https://arxiv.org/pdf/1911.11929v1 （sha256 34911bf1…），用 macOS PDFKit 抽文字並渲染第 3、5 頁。讀了 Abstract、§1 Introduction、§3.1 的 DenseNet／Cross Stage Partial DenseNet／Partial Dense Block／Partial Transition Layer、Figure 2、Figure 3、§4.2 Table 1、§4.3。
(2) YOLOv4，arXiv 2004.10934v1。https://arxiv.org/abs/2004.10934 ，PDF 為 https://arxiv.org/pdf/2004.10934v1 。讀了 Abstract、§2.1 backbone 列表、§3.1 Selection of architecture、§3.4 YOLOv4（Backbone: CSPDarknet53；BoS for backbone: Mish、CSP）、表中 YOLOv3 Darknet-53 各列。
(3) ultralytics/yolov5：用 gh api 把 tag v6.0 解析為 commit 956be8e642b5c10af4a1533e09084ca32ff4f21f。讀了 models/common.py（Conv 第 36–49 行、Bottleneck 第 93–103 行、BottleneckCSP 第 106–122 行、C3 第 125–137 行）、models/yolov5s.yaml（第 12–48 行）、models/yolo.py 的 parse_model（第 249–289 行，C3 參數的插入方式）。
(4) WongKinYiu/CrossStagePartialNetworks，commit 8786894eec2bc8e7d78572d5a770f091baa6f4fb 的 README.md（第 29–30 行：DarkNet-53 與 CSPDarkNet-53 的對照）。
(5) AlexeyAB/darknet，commit 59596d7880f6504768df41d6daa586f5cb2b932f 的 cfg/yolov4.cfg（第 20–140 行：CSP stage 的 1×1、route、殘差、1×1、route -1,-7、1×1；batch_normalize=1、activation=mish）。
另用 curl 確認頁面上的 arXiv 與 GitHub 連結都回 200。

課程檔案：docs/lessons/11-csp.md、lesson_cases/11-csp.py、docs/assets/diagrams/11-csp.svg、notebooks/11-csp.ipynb（最後一格與程式逐字相同，source_ref 是 lessons-v0.4.0）、審查用的事實與寫作規範清單、scripts/validate_lessons.py、miniyolo/models.py（沒有 CSP）、docs/glossary.md 第 60 行。交叉引用核對：01-small-cnn.md 第 141、217 行；02-diagnostics.md 第 38 行；03-identity.md 第 11、23、29、79 行；03-comparison.md 第 9 行；03-projection.md 第 11 行；14-feature-module.md 第 5–39 行。reviews/11-csp.md 只讀過、沒有採用。

執行過的指令：
- rsync 建立暫存副本。
- `PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/11-csp.py`：exit 0，輸出和頁尾區塊逐字相同。
- 摘錄比對：結果 []。
- `qlmanage -t -s 1200` 渲染 SVG，看過 PNG。
- `.venv-docs/bin/zensical build --clean --strict`：exit 0，No issues found。
- `python3 scripts/validate_site.py`：exit 0，全部 passed。
- 自寫的 probe.py 實測：漏呼叫 branch 時，branch 四個參數的 grad 都是 None，程式的檢查會失敗；chunk(dim=3) 產生 RuntimeError（通道數 8≠4）；用同樣的 RNG 順序重現 0.033093／0.029087／0.255476／0.019409，並量到 concat 處兩邊的梯度總和（CSP 0.255476／0.180649、Full 0.164523／0.147327）；C=8 與 C=10 的參數數 1240、368、1930、570；練習 2 的 shape；卷積支路輸出全 0 時串接結果不等於 x。
- probe_mem.py：chunk 傳回 view；卷積遇到不連續的輸入時會觸發 aten::contiguous 複製。
- probe_excerpt.py：頁面改寫的程式片段與 CSP.forward 輸出完全相等。
- 用 grep 檢查頁面沒有敘述修訂經過的字眼，也檢查了中英文間的空格與標點。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/lessons/11-csp.md 第 21、23 行（〈歷史與版本〉的 C3 描述與「本節簡化為…」）、第 52 行；docs/assets/diagrams/11-csp.svg 第 65 行與第 3 行 desc 的「本節簡化：卷積帶 bias」 | 頁面沒有揭露本節省略了 BatchNorm、並改用 ReLU。原版每個卷積後都接 BatchNorm 與激勵函數。YOLOv5 v6.0（commit 956be8e）models/common.py 的 C3 由 `Conv` 組成，第 40–45 行是：`self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)`、`self.bn = nn.BatchNorm2d(c2)`、`self.act = nn.SiLU() ...`，forward 是 `return self.act(self.bn(self.conv(x)))`。YOLOv4 的 CSPDarknet53 在 darknet 的 cfg/yolov4.cfg 裡，每層卷積都是 `batch_normalize=1`、`activation=mish`；論文 §3.4 也把 Mish 列為 backbone 的 Bag of Specials（BoS）。第 21 行把 C3 的組成寫成單純的「1×1 卷積」「3×3 卷積」，第 23 行列出的簡化只有切分次數與卷積個數。圖上「本節簡化：卷積帶 bias」只寫出結果，沒說原版為什麼不帶 bias（因為後面接 BN），初學者看不出這為什麼算簡化。本課程其他頁都明說省略了 BN（03-identity 第 11 行、03-projection 第 11 行、03-comparison 第 9 行、14-feature-module 第 39 行），本頁和它們不一致。 |
| 2 | 建議 | docs/lessons/11-csp.md 第 17 行末句「本節的 block 是先串接、再用 1×1 融合，排法比較接近後者」與第 21 行的 C3 結構描述 | 第 17 行說本節 block 比較接近論文所說「仍會大量重用梯度資訊」的先串接排法（fusion first），但第 21 行描述 C3 時沒有指出 C3 也是這個排法。v6.0 common.py 第 137 行是 `return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))`：Bottleneck 的輸出直接串接，之後才由 cv3 融合。反而是同檔較早的 `BottleneckCSP`，以及 YOLOv4 的 CSPDarknet53，才符合論文 Figure 3(b) 的「transition → concatenation → transition」。BottleneckCSP 第 120 行先做 `y1 = self.cv3(self.m(self.cv1(x)))`，第 122 行串接後再經 cv4。CSPDarknet53（yolov4.cfg 第 86–104 行）在殘差塊後先接一層 1×1，再 `[route] layers = -1,-7` 串接，最後再接一層 1×1。照現在的寫法，讀者容易以為實際 YOLO 的 CSP 模組都用論文推薦的排法，只有本節是退化版。另外，C3 的 cv1、cv2 輸出通道是 `c_ = int(c2 * e)`（第 129 行），預設 `e=0.5`，各為輸出的一半。這才是本節「切半」在 C3 裡的對應，頁面沒寫。「C3 是 YOLOv5 的 CSP 模組」這句也沒提到同檔還有 BottleneckCSP。 |
| 3 | 建議 | docs/lessons/11-csp.md 第 17 行「論文的做法是讓一部分通道直接跨到 stage 末端，藉此省下計算；至於截斷梯度流、減少這種重複，論文歸功於末端的融合順序」 | 這句把切分只說成省計算，把梯度方面的好處全歸給融合順序，比論文的說法單純。CSPNet §3.1〈Partial Dense Block〉列出切分的三個目的：「1.) increase gradient path: Through the split and merge strategy, the number of gradient paths can be doubled. … 2.) balance computation of each layer … 3.) reduce memory traffic」。〈Partial Transition Layer〉段末也寫：「By using the split and merge strategy across stages, we are able to effectively reduce the possibility of duplication during the information integration process.」後半句把截斷梯度流歸給 partial transition 是正確的（原文：「uses the strategy of truncating the gradient flow to prevent distinct layers from learning duplicate gradient information」）。但讀者對照原文時會發現，切分在論文裡也被說成有梯度方面的作用，和頁面的分工對不上。 |
| 4 | 建議 | docs/lessons/11-csp.md 第 21 行的「官方 v6.0 common.py」連結與「以 v6.0 的 yolov5s 配置為例」 | 連結用的是 tag（`blob/v6.0/…`），不是 commit。tag 可以被移動或刪除，和本課程連 ultralytics/ultralytics（441632c…）、THU-MIG/yolov10（453c6e3…）時固定 commit hash 的做法不一致。v6.0 tag 目前指向 commit 956be8e642b5c10af4a1533e09084ca32ff4f21f（「YOLOv5 release v6.0 (#5141)」）。另外，yolov5s 配置的說法（backbone 的 C3 用預設，`head:` 底下設 False）在本頁沒有來源連結，讀者無法自己核對。 |
| 5 | 建議 | docs/lessons/11-csp.md 第 11 行「要檢查的是 shape、參數數、梯度有沒有接通，以及參數會不會更新」 | 程式只比對 fuse 權重在更新前後是否不同（lesson_cases/11-csp.py 第 44、56 行：`before = model.fuse.weight.detach().clone()`、`assert not torch.equal(before,model.fuse.weight.detach())`）。其他參數只檢查梯度不是 None、是有限值且不全為 0，沒有確認它們真的改變。第 119、123 行寫得精確（「fuse 的權重確實改變」），第 11 行的概括則比程式實際檢查的範圍大。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 技術查核 | 沒揭露省略 BatchNorm、改用 ReLU（must） | 已修正：〈歷史與版本〉末段補上：原版每個卷積後接 BatchNorm 與激勵函數（C3 的 Conv＝不帶 bias 的卷積→BN→SiLU；CSPDarknet53 是 BN＋Mish）、為什麼原版不帶 bias，以及本節不加 BN、改用 ReLU、保留 bias。SVG 圖底與 desc 改成「本節簡化：不加 BatchNorm、改用 ReLU，卷積帶 bias」，並拆成三行、加高 viewBox。已對照 956be8e 的 common.py 第 36–42 行 |
| 2 | 技術查核 | C3 也是先串接再融合、e=0.5 半寬沒寫 | 已修正：YOLOv5 段補上：cv1、cv2 各輸出一半通道（e=0.5），C3 和本節一樣先串接、再融合；BottleneckCSP 與 CSPDarknet53 則先接 1×1 再串接。已對照 common.py 第 120–137 行 |
| 3 | 技術查核 | 把切分只說成省計算 | 已修正：改成「論文說這樣能省下計算，也讓梯度路徑變成兩倍」，截斷梯度流仍歸給融合順序 |
| 4 | 技術查核 | 官方程式用 tag 連結、yolov5s 配置沒有連結 | 已修正：common.py 改連 v6.0 tag 指向的 commit 956be8e（已用 git ls-remote 確認），另加同一 commit 的 yolov5s.yaml 連結。為了一致，本批四頁的 YOLOv5 連結都改成這個 commit |
| 5 | 技術查核 | 第 11 行說「參數會不會更新」，範圍說大了 | 已修正：改成「fuse 的權重會不會更新」 |
| 6 | 讀者審查 | 「截斷梯度流」沒解釋、句構太繞 | 已修正：拆成短句，並補上：意思是避免不同層學到重複的梯度資訊，不是梯度傳不回前面的層 |
| 7 | 讀者審查 | 本節的「融合」和下一節的特徵融合容易混淆 | 已修正：fuse 第一次出現的地方補一句：這裡只在同一位置混合通道，下一節〈特徵融合〉合併的是不同尺度的特徵（附連結） |
| 8 | 讀者審查 | 沒說為什麼需要 fuse | 已修正：補上理由：沒有 fuse 時，前 4 個通道在下一個 block 又走旁路，就一直沒經過 3×3 |
| 9 | 讀者審查 | 論文的 20% 和本例的七成容易混淆 | 已修正：「約剩三成」後補上：本例是單一 block 的參數比例，論文的 20% 是整個網路的計算量，兩者不能直接比 |
| 10 | 讀者審查 | 圖上用「＋」、channels、split 等用字 | 已修正：SVG 改成「3×3→ReLU」，並刪掉正文那句「＋不是相加」的警語；「前／後 4 channels」改成「前／後 4 個 channel」，「split：4+4」改成「切成 4+4」；bias 的原因已寫進〈歷史與版本〉 |
| 11 | 讀者審查 | 「固定預算」與「容量成本的比較」意思不清 | 已修正：第 11 行改成「條件固定的對照實驗：資料、步數、評估方式都相同」；第 129 行改成：本節只比參數數，參數量相同時比準確度的實驗本節沒做 |
| 12 | 讀者審查 | 梯度穿過兩層變小，沒給理由 | 已修正：補上：預設初始權重偏小（4→4 的 3×3 卷積是 ±1/6），每穿過一層卷積梯度就縮小一截，ReLU 在負輸入處傳回 0。已用 kaiming_uniform 的上下界核對 |
| 13 | 讀者審查 | YOLOv5 段括號太長、「參數 shortcut」有歧義、C3 寫在 head: | 已修正：整段拆成短句；「參數」改成「建立 C3 時有個選項 `shortcut`」；補上 YOLOv5 設定檔把 neck 也寫在 head: 底下 |
| 14 | 讀者審查 | 「finite backward and step」那一行沒解釋 | 已修正：斷言清單後補一句說明這一行的意思 |
| 15 | 讀者審查 | 「切得太多」有歧義 | 已修正：改成「讓太多通道走旁路（只留少數通道做 3×3）」 |
| 16 | 讀者審查 | 重複梯度限制講了兩次，「所以」的因果不通 | 已修正：第 130 行只留「所以不報告 AP」 |
| 17 | 讀者審查 | 變數名 transformed 容易誤解 | 已修正：程式不能改，所以在摘錄註解寫明「後 4 個通道（還沒轉換，要送進卷積支路）」 |
| 18 | 讀者審查 | 高寬不同的錯誤太抽象 | 已修正：補上實例：漏寫 padding=1 時 8×8 會縮成 4×4，[B,4,4,4] 無法和 [B,4,8,8] 串接。暫存副本實測確實報錯 |
| 19 | 讀者審查 | 「卷積核」「CSP 類別」用字不一致 | 已修正：改成「kernel（濾鏡）的邊長」（和第 1 章一致）、「`CSP` class」；公式和第 1 章寫法的對照屬風格偏好，沒有加 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/11-csp.md`、`docs/lessons/11-fusion.md`、`docs/lessons/11-augmentation.md`、`docs/lessons/11-iou-loss.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

用腳本比對發現與處理列數並讀了修正後檢查。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/11-csp.md | 19 項發現中，must 的處理寫在修正後檢查裡，其餘 18 項 should 沒有處理，指向的〈定稿修正〉不存在（原因同上）。另有「2. trace／impact（must，通過）」。 | 已修正：產生器改以頁名、節名、萬用字元、頁面上的圖與該節 notebook 把處理對應到頁面，〈定稿修正〉列出這一頁每一項的處理。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：讀者 14 項與技術 5 項都在〈定稿修正〉處理（上一輪指出整節缺漏，已補）；技術查核列出 yolov5 固定 commit 的 common.py、yolov5s.yaml；批次檢查掛在本頁。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/11-csp.md 第 114 行 | 指令殘句：「`摘錄比對工具 <暫存副本> docs/lessons/11-csp.md`：結果 []」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 114 行點名的指令殘句已移除，處理說明屬實。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[evolution當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/evolution.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/evolution.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/evolution.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_a` 實讀本頁 9 個凍結單元；首次使用／前文方法範圍四題位置為 11-csp/00:first_use, 11-csp/01:first_use，頁末為 11-csp/08。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_a.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_a-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_a.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R032；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 c1 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁相關處置：DEC-020；包含採用、保留或後文撤回的來源與理解收益。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `c_foundation` 實讀本頁 7 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/c_foundation/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：無額外局部修正。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/11-csp.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
