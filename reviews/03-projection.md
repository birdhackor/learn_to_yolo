# 審查紀錄：ResNet projection shortcut

審查範圍：`docs/lessons/03-projection.md`、頁面上的圖（`docs/assets/diagrams/03-projection.svg`），以及 `lesson_cases/03-projection.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題。只有一條 should：參考答案第 3 題的「再往下」少一步說明。

我在暫存副本查了以下各項。

1. 程式的敘述都成立。
   - lesson_cases/03-projection.py 跑完 exit 0，stdout 剛好六行，stderr 是空的。第三行是 `position probe (input channel 0 = 10*row + col): first output channel=[[0.0, 2.0], [20.0, 22.0]]`。
   - 頁上列的七個 assert 和程式一一對應。六行輸出的順序與內容、`probe`／`probe_output` 等名稱也都相符。
   - 練習照頁面指示逐步做過，變體放在暫存副本：
     - E0：只改 `expand` → 停在 `assert output.shape == (1, 6, 2, 2)`。
     - E1：照答案 2 改兩個 assert → 全部通過，第三行仍是 [[0.0, 2.0], [20.0, 22.0]]，兩支都是 3×3。
     - E2：P 改成 stride 1 → 停在 forward 的 shape assert；改回 stride 2 就通過。
     - E3：F 第一層也改成 stride 1 → 停在 `(1, 6, 3, 3)` 那行。
     - E4：兩個 assert 改成 5×5 → 停在探針 assert，`probe_output` 就是頁上那張 4×4。
     - E5：探針期望值也改掉 → 停在 `mse_loss`，報 RuntimeError。
   - 正文第 95、110 行說的情況也實測過：
     - P 用 stride 1 時，forward 的 assert 會擋下。
     - stride 3 時探針得 [[0.0, 3.0], [30.0, 33.0]]。
     - 平均池化後接 1×1 卷積，探針得 [[5.5, 7.5], [25.5, 27.5]]。
     - 後兩種情況下 321 仍通過；拿掉探針 assert 後程式 exit 0，所以確實「只有探針的 assert 會失敗」。

2. 追溯與影響清單都處理了。
   - 第 3 行的 Colab 連結已經是 lessons-v0.4.0，和 section-map.json 一致。
   - 第 93 行的但書已刪，改寫成探針說明。數值依定稿程式寫 [[0,2],[20,22]]，不是追溯建議的 [[0,2],[8,10]]；程式與審查用的事實與寫作規範清單優先，所以這樣是對的。
   - 受程式改動影響的段落列的第 13、57、93、115、123、126、132、133 行都改了。之前查核補充的第 77、84 行，以及「不是原值」的說明衝突，也都處理了。
   - 全頁找不到修訂、審查或製作經過的字眼，也沒有引用程式行號。

3. 數字都是可手算的確定值，正文沒寫 loss 或任何在這台 Mac 上量到的數字。要等紀錄重產才能核對的項目，編輯都列了：第 2、3 行輸出、行數與行序、頁尾舊的五行執行紀錄。頁尾目前還是五行，依審查用的事實與寫作規範清單屬於發布前的暫時狀態，不算頁面問題。

4. 摘錄檢查通過。
   - 摘錄比對工具對暫存複本和 repo 的結果都是 `docs/lessons/03-projection.md []`。
   - 我故意改壞三處都被抓到：改探針 assert 的值、把 `return` 改成 `y =`、拿掉第二塊的 data-excerpt 標記。
   - 這個檢查只比對每一行在不在原檔，不管順序，所以三塊的順序和連續性我另外人工核對：都是原檔連續的一段。第一塊把 `__init__` 裡那行和 `forward` 都靠左對齊，正文已說明它在 `__init__` 裡，不算誤導。
   - 暫存區跑 validate_lessons.py：42 頁的逐頁檢查與摘錄檢查全部通過，只在審查涵蓋那一關失敗（`no review`）。這是預期內的，發布前會重審。

5. 可讀性：新名詞「位置探針」有解釋，`detach()` 和計算圖在 00-warmup、02-diagnostics 已經教過，講解順序也合理。

6. 呈現與改動範圍。
   - zensical build --clean --strict 與 validate_site.py 在暫存區都 exit 0。三個摘錄區塊都正常顯示成 language-python 程式碼，數學式也都正常顯示。
   - 03-projection.svg 和 HEAD 相同，有 viewBox、title、desc，用 qlmanage 渲染乾淨，486＋18 個權重與 s／p 設定都和程式一致。
   - 這頁的編輯只改了 docs/lessons/03-projection.md。lesson_cases、notebook、這頁的執行紀錄 JSON 和 reviews 都沒動。

在 repo 裡只跑過唯讀的 git 指令，以及題目指定的唯讀摘錄比對工具。紀錄檔在暫存副本。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/03-projection.md 第 150 行，參考答案第 3 題：「改完後，換位置探針的 assert 失敗：……取樣位置變了，探針要抓的正是這件事。再往下，第二部分的輸出也會變成 `[2,6,4,4]`……`mse_loss` 會報錯」 | 這條替代路徑的前後幾步都寫了「要改什麼才能往下走」：兩個 output assert 要再改成 5×5，`mse_loss` 的目標也要跟著改。只有探針 assert 失敗之後，沒說要怎麼處理才會到「再往下」的 `mse_loss`。照著做的讀者會停在探針 assert，不確定該改期望值還是刪掉它；前文又說「不是移除 assert」，有人可能就直接刪掉這道檢查。我在暫存區實測（E5）：把探針的期望值改成頁上那張 4×4 之後，程式才會停在 `mse_loss`，錯誤是 RuntimeError: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 3。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：讀了這些：審查用的事實與寫作規範清單；docs/lessons/03-projection.md 從頭到尾；在暫存副本執行 zensical build --clean --strict（exit 0，No issues found）後，讀 site/lessons/03-projection/index.html，逐段核對表格、MathJax 公式區塊、三個 data-excerpt 程式摘錄、三個摺疊區與練習清單的轉換；docs/assets/diagrams/03-projection.svg 原始碼，並用 qlmanage -t -s 1200 轉成 PNG 檢視；lesson_cases/03-projection.py；notebooks/03-projection.ipynb（4 格，最後一格與程式相同）；docs/glossary.md（符號 P、residual block、concat／add、activation、stride 等列）；照順序讀的前幾節 00-warmup、01-small-cnn、03-identity（確認內積、Linear 權重 [K,D]、輸出邊長公式、MAC、視窗、兩點收益、直接梯度路徑、零權重梯度在哪裡教過，以及前面從沒教過多軸索引與冒號）；03-comparison（確認下一節只用同 shape 的 identity shortcut）；learning-path.md 的 3.2 描述；scripts/validate_lessons.py。

跑了這些：在暫存副本用 PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg 執行本節程式，exit 0，印出 6 行，與正文〈核對〉列出的內容一致。照頁面指示做完練習全部三題，還順著答案描述的另一條路做下去：(A) 只把 expand 改成 5×5：什麼都沒印，停在 assert output.shape == (1, 6, 2, 2) 的 AssertionError；(B) 再把兩個 assert 改成 (1,6,3,3) 與 torch.full((3,3),321.0)：印出 3×3 的 shape、3×3 全是 321 的格子，探針仍是 [[0,2],[20,22]]，loss 不變；(C) 再把 projection 改成 stride=1：在 output = block(image) 時，forward 裡的 assert main.shape == shortcut.shape 失敗；(D) 照答案的另一條路把 F 第一層改成 stride 1：停在 (1,6,3,3) 的 assert；改成 5×5 後換探針 assert 失敗，probe_output 正是答案寫的整張 4×4；再改掉探針 assert，先出 UserWarning，接著 mse_loss 拋出 RuntimeError（size 4 對 2）。另外實跑了 stride 3 與「平均池化＋1×1」兩種變體，確認 image 仍算出 321、探針值如頁面所寫。也核對了 123、[6,3,1,1]、162+324+18=504、72/1944，以及 H=1～8 時 F、P、pooling 與 F 沒有 padding 時的邊長。在暫存副本執行 validate_lessons.py：程式摘錄檢查通過，只停在審查涵蓋檢查（各頁還沒有審查紀錄，屬發布前的暫時狀態）。

沒有列入的：artifacts/checks/curriculum/03-projection.json 的 case_sha256（9cdad85…）和目前程式（7239239…）不同，頁尾自動產生的區塊因此只有 5 行。依指示，這份紀錄正在重產，所以不報。Playwright 的 Firefox 沒有安裝，無法用瀏覽器截圖；畫面檢查是靠建置出的 HTML 與 SVG 轉成的 PNG。沒有改動 repo 根目錄裡的任何檔案。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈自主練習與答案〉開頭的操作說明與第 2 題 | 照指示只把 `expand(1, 3, 4, 4)` 改成 `expand(1, 3, 5, 5)` 就執行，程式一行都沒印，直接停在 `assert output.shape == (1, 6, 2, 2)` 的 AssertionError（實跑確認）。頁面沒說這是預期中的錯誤。第 0、1 章的練習都明講「出現 AssertionError 是預期的，改斷言、別刪斷言」，這裡沒有；初學者看到 traceback 又沒有任何輸出，容易以為自己改壞了。另外 Colab 只寫「完整程式（Colab…）」，沒有像其他節那樣指明是〈本節可修改的完整實驗〉下面那一格。 |
| 2 | 建議 | 〈Shape 與程式對照〉「第二部分：隨機權重更新檢查」段的「程式確認 P 的梯度不是 0」，以及〈核對與常見錯誤〉第一段的「P 的梯度不是 0」；對照前面的摺疊區〈換成 P 之後，上一節的直接梯度路徑還在嗎？〉 | 摺疊區說的「梯度沿 P 回傳」，是 loss 對輸入的梯度（輸入增加 0.001，輸出增加 0.003）。第二部分檢查的卻是 loss 對 P 那 18 個權重的梯度（`learned.projection.weight.grad`）；程式沒有檢查輸入的梯度（inputs 沒有 requires_grad）。兩處都只寫「P 的梯度」，讀者容易以為程式驗證了摺疊區說的直接梯度路徑。3.1 節特別區分過「對參數的梯度」與「對輸入的梯度」，也寫成「主分支末層的權重收得到非零梯度」，這裡卻沒標出是哪一種。 |
| 3 | 建議 | 〈Shape 與程式對照〉第一部分的程式摘錄：`block.projection.weight[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])`（以及同段的 `output[0, 0]`） | 這是全書第一次出現多軸索引加冒號 `:` 的寫法（前面各節只用過 3.1 的 `y[x < 0]` 布林索引）。註解只說「設 P 第一個輸出 channel（索引 0）的權重」，沒說四個索引各對應哪一軸、`:` 是什麼意思；程式新手看不出這一行為什麼剛好選到 3 個數。 |
| 4 | 建議 | 〈1×1 卷積看不到鄰居，卻能混合 channel〉最後一段「P 和 2×2、stride 2 平均池化（average pooling）的差別」；〈Shape 與程式對照〉探針後一段「先做 2×2、stride 2 平均池化，再做 stride 1 的 1×1 卷積」；〈核對與常見錯誤〉最後一段「若 shortcut 改用 2×2、stride 2 的 pooling」 | 前面各節只教過 max pooling 與全域平均池化（GAP）。2×2 平均池化在這裡第一次出現，卻沒有定義，也沒說為什麼拿它和 P 比。最後一段只比邊長，說 pooling 在偶數 H 時「對齊」；但 pooling 不改 channel 數，輸出仍是 3 個 channel，還是不能和 F 的 6 個 channel 相加。讀者可能以為 pooling 本身就能當這裡的 shortcut。 |
| 5 | 建議 | 〈自主練習與答案〉參考答案第 3 題 | 一段裡塞了四件事：在哪裡失敗、通用修法、實際修法、改 F 的另一條路。另一條路還要連續經過三處錯誤（`output.shape` 的 assert → 探針的 assert → `mse_loss` 報錯）。「再往下」要先把探針的 assert 改掉才到得了，答案沒說。初學者很難照著一步步核對（我照做時得自己改探針 assert，才看到 `mse_loss` 的 RuntimeError，之前還先跳出一個 UserWarning）。 |
| 6 | 建議 | 〈先把兩支畫出來〉的圖（03-projection.svg） | 圖例只說明 s=stride、p=padding；括號裡的「3→6」「6→6」是 channel 數，圖上沒寫。同一框裡「ReLU → 3×3」的箭頭卻是「接著做」，兩種箭頭意思不同。「486 個可學權重」「18 個可學權重」先出現在圖上，算法要到〈收益與代價〉才給，讀者看圖時不知道 486 從哪來。 |
| 7 | 建議 | 開頭第三段「前置只需知道 NCHW 與卷積的 stride（卷積窗每次滑動的格數）」 | 本頁實際還用到：第 1 章的輸出邊長公式與 MAC 算法；暖身節的內積與 Linear 權重 `[K,D]`；3.1 節 identity shortcut 的兩點收益、「F 全為 0 時權重梯度為 0」（頁內寫「上一節說過」）與 MSE。寫「只需」兩項，低估了前置，和 3.1 節（明列要會用輸出長度公式）的寫法也不一致。另外，這裡的「卷積窗」與後文的「3×3 窗」，和第 1 章定義的「視窗」用詞不同。 |
| 8 | 建議 | 〈Shape 與程式對照〉「第一部分：手設權重驗算」段開頭 | 原文是：「F 的權重全設 0，所以 F(x) 全為 0。P 的權重矩陣 W 只有第一列……其餘 5 列全是 0。因此 y=P(x)。」y=P(x) 是由 F(x)=0 推出的，和 P 的權重怎麼設無關；中間夾了 P 權重那一句，讓「因此」看起來像是由 P 的權重推出來的。 |
| 9 | 建議 | 開頭第五段（「CPU 實驗分兩部分」）的「先驗算一個 channel 混合後的數值」 | 字面上像是「把一個 channel 拿去混合」；但 channel 混合指的是把三個輸入 channel 加權相加，讀者要讀到下一節才知道這裡指的是 321。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：讀過的檔案（都在暫存副本內）：docs/lessons/03-projection.md、lesson_cases/03-projection.py、docs/assets/diagrams/03-projection.svg；交叉引用的 docs/lessons/03-identity.md、03-comparison.md（含 lesson_cases/03-comparison.py 摘錄）、00-warmup.md（內積、Linear 權重 [K,D]）、01-small-cnn.md（邊長公式、MAC 定義）；docs/glossary.md、section-map.json、zensical.toml 導覽、artifacts/checks/curriculum/03-projection.json（stdout 與 case_sha256）、reviews/03-projection.md；以及審查用的事實與寫作規範清單。

一手來源：
(1) ResNet，He et al. 2015，arXiv:1512.03385。用 https://ar5iv.labs.arxiv.org/html/1512.03385 抽出全文（https://arxiv.org/pdf/1512.03385v1 也下載了，但本機無法渲染 PDF）。核對的段落：
- §3.2：式 (1)(2)、"we can perform a linear projection W_s by the shortcut connections to match the dimensions"、"We adopt the second nonlinearity after the addition"。
- §3.3：Plain Network 的 "(ii) if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer"；Residual Network 的 "(A) ... extra zero entries padded for increasing dimensions. This option introduces no extra parameter; (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2."
- Table 1 caption："Downsampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2"。
- §3.4："We adopt batch normalization (BN) right after each convolution and before activation"。
- §4.1 Identity vs. Projection Shortcuts 與 Table 3（ResNet-34 A／B／C 的 top-1 error 為 25.03／24.52／24.19，10-crop）："B is slightly better than A ... projection shortcuts are not essential"。
- §4.2 CIFAR："we use identity shortcuts in all cases (i.e., option A)"。

(2) 官方程式：https://github.com/KaimingHe/deep-residual-networks，commit a7026cb6d478e131b765b898c312e25f9f6dc031，prototxt/ResNet-50-deploy.prototxt。
- 第 487–520 行：res3a_branch1 是 kernel_size 1、pad 0、stride 2、bias_term false 的卷積，後接 bn3a_branch1、scale3a_branch1。
- 第 636–650 行：Eltwise res3a 之後接 res3a_relu。

(3) Bag of Tricks，He et al. 2019，arXiv:1812.01187，用 https://ar5iv.labs.arxiv.org/html/1812.01187：§4.1 的 stage 與 downsampling block，§4.2 的 ResNet-B 與 ResNet-D（"also ignores 3/4 of input feature maps ... adding a 2×2 average pooling layer with a stride of 2 before the convolution, whose stride is changed to 1"）。

(4) timm：https://github.com/huggingface/pytorch-image-models，commit f5780845d8ea714810ce850d8fe43e4d080ebf6d，timm/models/resnet.py 的 downsample_avg（第 306–330 行；第 324 行是 ceil_mode=True, count_include_pad=False）。commit hash 經 GitHub API 取得。

執行的指令：
- 用 rsync 建立暫存副本。
- 執行課程程式：PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/03-projection.py。結果為 exit 0，印出六行。
- 摘錄比對，輸出 []。
- qlmanage -t -s 1200 渲染 SVG，並看過 PNG。
- .venv-docs/bin/zensical build --clean --strict（exit 0，"No issues found"），之後跑 python3 scripts/validate_site.py（全部 passed），再檢查渲染後 HTML 的表格、details、數學與粗體。
- 用 sed 產生六個練習變體 ex/a–f.py 並逐一執行：
  - 只改成 5×5 時，停在 output.shape 的 assert。
  - 改好 assert 後 exit 0，探針那行不變。
  - P 改 stride=1 時，停在 forward 的 assert。
  - F 也改 stride 1 時，停在 (1,6,3,3) 的 assert。
  - 改成 5×5 版之後，探針 assert 失敗，探針輸出是整張 4×4。
  - 最後 mse_loss 先發 UserWarning，再報 RuntimeError。
- 自寫 ex/checks.py 驗證：
  - 其餘 5 個輸出 channel 為 0；channel 順序反過來得 123。
  - stride 3 的探針輸出 [[0,3],[30,33]]；平均池化版 [[5.5,7.5],[25.5,27.5]]，兩者對 image 都仍算出 321。
  - 不包 no_grad 會報錯；參數依序是 162／324／18。
  - 慣例 MAC 為 648／1296／72（比例 3.7%）；扣掉 padding 後實際乘到輸入的是 450／576／72（比例 7.0%）。
  - F 第一層的窗涵蓋每一格；P 的梯度只到 4 個位置。
  - H＝3 到 8 時，無 padding 的 F 比 P 少 1；pooling 在奇偶 H 的對齊情形。
  - broadcast、reshape 會不會報錯；假設層的 MAC 1296。
- 計算現行程式的 shasum，並和紀錄裡的 case_sha256 比對。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈收益與代價，要分開算〉第一段「收益是主分支與 shortcut 在 stage 交界也能相加，而且 P 能學習 channel 轉換」（第 116 行），以及第二段「P 的 18 個參數與 72 次乘加，就是 (B) 比不增加參數的 (A) 多付的代價」（第 118 行） | 收益和代價用的基準不同。收益是跟上一節的 identity 比（「在 stage 交界也能相加」），代價卻是跟 (A) 比。(A) 先用 stride 2 取樣再補 0，一樣能讓兩支相加，而且沒有參數，所以「能相加」不是 (B) 勝過 (A) 的地方。頁面也沒說 (B) 比 (A) 多換到什麼。論文 §4.1 與 Table 3 的結果：ResNet-34 在 ImageNet validation（10-crop）的 top-1 error，A 是 25.03%，B 是 24.52%。論文寫 "B is slightly better than A. We argue that this is because the zero-padded dimensions in A indeed have no residual learning"，結論是 "projection shortcuts are not essential for addressing the degradation problem"；論文的 CIFAR-10 實驗也全部用 option A。另外，官方 prototxt 的 projection 後面還接 bn3a_branch1／scale3a_branch1，所以「(B) 只多付 18 個參數」只適用本節拿掉 BN 的 block。 |
| 2 | 建議 | 第 110 行「`image` 的檢查抓不到、探針抓得到的例子：……把 P 換成『先做 2×2、stride 2 平均池化，再做 stride 1 的 1×1 卷積』……只有探針的 assert 會失敗。」 | 這個改法和 stride 3 一起被列為「抓得到」的例子，初學者容易以為它也是寫錯了。實際上，它就是已發表的 ResNet-D 下取樣 shortcut。Bag of Tricks（arXiv:1812.01187）§4.2 寫："the 1×1 convolution in the path B of the downsampling block also ignores 3/4 of input feature maps ... adding a 2×2 average pooling layer with a stride of 2 before the convolution, whose stride is changed to 1, works well in practice"。它的動機正是本頁第 57 行說的「16 格中有 12 格沒進入 shortcut」。探針在這裡失敗，只代表它的取樣方式和本節的 P 不同，不代表設計有錯。 |
| 3 | 建議 | 第 136 行「若 shortcut 改用 2×2、stride 2 的 pooling，邊長在偶數 H 時對齊、奇數 H 時差 1（H=5 時 pooling 輸出 2×2、F 輸出 3×3）」 | 這句只在 pooling 用 PyTorch 預設 ceil_mode=False 時成立，這時邊長照 ⌊(H−2)/2⌋+1 計算。常見的 ResNet-D 實作用的是 ceil_mode=True，例如 timm（commit f5780845d8ea714810ce850d8fe43e4d080ebf6d，timm/models/resnet.py 第 324 行 `avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)`）。這時邊長是 ⌈H/2⌉，奇數 H 也和 F 對齊；H=5 時兩支都是 3×3。頁面沒說是哪一種 pooling，讀過 timm 程式的讀者會得到相反的結論。 |
| 4 | 建議 | 第 118 行乘加次數一段（「主分支兩層是 2×2×6×(3×9)=648 與 2×2×6×(6×9)=1296，合計 1944；加上 P 的 72 次，只比主分支多約 3.7%」），以及第 124 行「每個輸出讀取的值數」 | 這裡照常用慣例計數：每個輸出都算滿 C_in×k² 次，padding 補的 0 也算在內。本節的特徵圖只有 2×2，padding 占的比例很大。F 第二層每個輸出的 54 次裡，只有 24 次乘到真正的輸入，全層是 576 次，不是 1296。F 第一層實際是 450 次，不是 648（暫存副本用全 1 卷積數過）。只算真正乘到輸入的次數時，P 的 72 次約占主分支的 7.0%，不是 3.7%。頁面只寫「每個輸出讀 3×9 個值」，數學好的讀者照字面去數格子，會得到不同的答案，也不會知道 3.7% 是由計數方式決定的。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 讀者審查 | 練習只改 expand 就跑，會在印出任何東西前出現 AssertionError，頁面沒預告 | 已修正：補上：這是預期中的錯誤，答完第 1、2 題改好 assert 再執行，不要刪 assert；並寫明 Colab 要改最後一格 |
| 2 | 讀者審查 | 「P 的梯度」沒分清是權重的梯度還是輸入的梯度 | 已修正：兩處都改成「P 權重的梯度」，並補一句：程式只檢查 P 學得動，沒有另外檢查梯度沿 P 傳回輸入 |
| 3 | 讀者審查 | [0, :, 0, 0] 這種多軸索引沒有解釋 | 已修正：在摘錄後補一段，說明四個索引各對應哪一軸、冒號代表全取，以及 output[0, 0] 是什麼 |
| 4 | 讀者審查 | 平均池化沒定義；pooling 不改 channel 數 | 已修正：第一次出現時補上定義（和 max pooling 一樣切塊，只是取平均）；最後一段補上 pooling 不改 channel 數，仍要接 1×1 卷積 |
| 5 | 讀者審查＋先前查核留下的項目 | 練習第 3 題答案太擠，探針那一步沒說要怎麼處理 | 已修正：補上「探針的期望值也得改成這張 4×4」再往下走，並註明 mse_loss 會先出現 UserWarning 再報 RuntimeError（已在暫存副本實跑確認）；沒有拆成小項 |
| 6 | 讀者審查 | 圖例沒說括號是 channel 數；486、18 的算法沒交代出處 | 已修正：SVG 圖例與 desc 補上「括號內是輸入→輸出 channel 數」；表格後補一句：486、18 的算法見〈收益與代價，要分開算〉 |
| 7 | 讀者審查 | 前置寫「只需」兩項，低估了前置；「卷積窗」與「視窗」用詞不一 | 已修正：前置改列 NCHW、stride、輸出邊長公式與上一節的結論；「卷積窗」「3×3 窗」統一成「視窗」 |
| 8 | 讀者審查 | 「因此 y=P(x)」放錯位置，看起來像是由 P 的權重推出 | 已修正：移到「F(x) 全為 0」後面 |
| 9 | 讀者審查 | 「驗算一個 channel 混合後的數值」容易誤讀 | 已修正：改成「驗算一次 channel 混合（三個輸入 channel 的加權和）」 |
| 10 | 技術查核 | 收益和代價的比較基準不同（一個跟 identity 比，一個跟 (A) 比） | 已修正：收益分兩層寫：相對 identity，以及相對 (A)。附論文表 3 的 25.03% 與 24.52%，並說明論文認為 projection 不是必要的；代價那句註明本節沒有 BatchNorm |
| 11 | 技術查核 | 平均池化版其實是已發表的 ResNet-D 設計，不是寫錯 | 已修正：補一句：stride 3 是寫錯，平均池化版是 Bag of Tricks §4.2 的 ResNet-D 刻意採用的設計，探針失敗只代表讀法不同 |
| 12 | 技術查核 | pooling 邊長的說法只在 ceil_mode=False 時成立 | 已修正：註明是 PyTorch 預設的 nn.AvgPool2d(2)，並補上 ceil_mode=True 時邊長是 ⌈H/2⌉ |
| 13 | 技術查核 | MAC 的算法連 padding 的 0 也算進去，但頁面沒說 | 已修正：補一句說明這個計數慣例，並列出只算真正乘到輸入時的次數 450 與 576（已手算核對） |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/03-identity.md`、`docs/lessons/03-projection.md`、`docs/lessons/03-comparison.md`、`docs/lessons/04-localization.md`、`docs/lessons/04-coordinates.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/03-projection.md 第 120 行「ImageNet validation 的 top-1 錯誤率」 | 這次新增的句子用了「top-1」，課程其他地方和術語表都沒有定義這個分類用語，初學讀者看不懂。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

核對 ResNet 論文 Table 3（10-crop，ImageNet val）：ResNet-34 選項 A 的 top-1 錯誤率 25.03%、B 為 24.52%，論文說 A/B/C 的小差異表示「projection shortcuts are not essential for addressing the degradation problem」，頁面數字與結論正確。MAC 與 padding 的補充、Bag of Tricks §4.2 的 ResNet-D 敘述也正確。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 〈收益與代價〉：「top-1 錯誤率（分數最高的那一類猜錯的比例）」 | 定義沒有錯，但「分數最高的那一類猜錯的比例」容易被讀成「某一類別的錯誤率」，沒有講清楚分母是全部圖片。 | 已修正：改成「模型給分最高的類別不是正確類別的圖片，占全部圖片的比例」。 |

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

top-1 定義「模型給分最高的類別不是正確類別的圖片，占全部圖片的比例」正確（ResNet 表 3，10-crop、ImageNet val，(A) 25.03%、(B) 24.52%），講清了分母，修好發現，放在括號裡仍讀得通。本頁紀錄有必要問題。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/03-projection.md | 紀錄的問題：列了 14 項發現（第 1 次查核 1、讀者審查 9、技術查核 4），寫「建議事項在下方〈定稿修正〉逐項處理」「各項的處理見下方〈定稿修正〉」，但整份沒有〈定稿修正〉，每一項的處理都沒列。原因：lean 工作流程 fix:b1b 的 handled 以「03-projection」為 page，生成器只比對 docs/lessons/03-projection.md 與 03-projection.md。 | 已修正：產生器改以頁名、節名、萬用字元、頁面上的圖與該節 notebook 把處理對應到頁面，〈定稿修正〉列出這一頁每一項的處理。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

全文讀過紀錄。第 1 次查核的 1 項建議、讀者審查 9 項、技術查核 4 項，都在〈定稿修正〉13 列處理（先前查核那項和讀者第 5 項合併）；技術查核列出 ResNet 論文、官方 prototxt（a7026cb）、Bag of Tricks 與 timm（f578084）；批次檢查留下的 top-1 意見，經〈後續編輯的檢查〉兩輪改好並確認。上一輪「沒有〈定稿修正〉」的必要問題已解決。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/03-projection.md 第 31、34、49、52、104、106、145 行 | 內部名稱與殘句：「2. 追溯與影響清單都處理了」「之前 check 補充的第 77、84 行」「以及題目指定的唯讀摘錄比對工具。紀錄檔在暫存副本、暫存副本、暫存副本、暫存副本.stdout（都在 …/暫存副本）」「（暫存副本03-projection.svg.png）」「用 rsync 建立暫存副本 03-projection-tech」「python3 摘錄比對工具 <暫存副本> docs/lessons/03-projection.md」；來源欄是「reader+先前查核留下的項目」。 | 來源欄改成中文。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：「之前查核補充」、摘錄指令、svg.png、03-projection-tech 與來源欄都改好了；但另兩句點名的殘句仍在，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/03-projection.md 第 30、51 行 | 點名的「2. 追溯與影響清單都處理了」在第 30 行逐字還在（內部清單名）；點名的「以及題目指定的唯讀摘錄比對工具。紀錄檔在暫存副本…（都在 …/暫存副本）」在第 51 行只少了幾個「暫存副本」，仍是殘句。處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 6 輪：上一輪的處理：有必要問題

第 3 輪 #1 說來源欄已改成中文，但〈定稿修正〉#5 的來源欄仍有英文的先前查核留下的項目。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 3 輪 #1 的處理欄（第 187 行）；〈定稿修正〉#5 的來源欄（第 145 行） | 處理欄寫「來源欄改成中文」，但 #5 的來源是「讀者審查＋leftover」，原始標籤是 reader+leftover。第 3 輪發現點名的正是這個來源，當時顯示為「reader+先前查核留下的項目」。 | 已修正：組合來源裡的 leftover 也換成中文（讀者審查＋先前查核留下的項目）。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/foundations.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `foundations` 實讀本頁 7 個凍結單元；首次使用／前文方法範圍四題位置為 03-projection/00:first_use, 03-projection/03:first_use，頁末為 03-projection/06。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/foundations.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/foundations-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/foundations.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS04。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。


## 2026-10-07 新 SKILL 改寫：作者自查

本次範圍：本節新版正文、必要圖、程式摘錄與配對 notebook。先前的獨立查核結論對應當時舊稿，不能拿來宣稱本次大幅改寫已獨立通過。此次由參與撰寫的作者核對概念順序、主文是否依賴選讀、數字、圖文對應與前後銜接；沒有逐段盲讀、獨立審查者或真人讀者測試，新稿仍待使用者閱讀回饋。

- [概念覆蓋與本文路線](clear-tutorial/opening-a-2026-10-07/concepts-and-author-check.md)：38 個必要概念／轉換檢查點，逐項列先備、位置、角色與小變化。
- [首次使用原句](clear-tutorial/opening-a-2026-10-07/first-use-excerpts.md)：當場的實際原文，明記為作者自查。
- [來源核對](clear-tutorial/opening-a-2026-10-07/sources.md)、[14 個手算與變化實測](clear-tutorial/opening-a-2026-10-07/example-checks.json)。
- [網站實際頁面](clear-tutorial/opening-a-2026-10-07/browser-check.json)：8 個入口／A 頁在 1280 與 390 像素寬的檢查；圖片載入、MathJax 與整頁寬度正常，選讀預設收折。作者另視覺檢查新增圖、手機捷徑圖及主要數學段落。
- [修改範圍](clear-tutorial/opening-a-2026-10-07/scope-check.json)：B 以後課文與 notebook、所有實驗與模型原始碼未改；閱讀路線 B 起維持原文。

六節主實驗已在 CPU 重跑，僅更新其執行證據；A.1 與 A.3.3 的40步實驗也重跑並核對保存值。這些檢查支持數字與實作一致，不替代首次閱讀驗收。覆蓋指紋記錄的是本次作者實際查過的版本。

## 2026-10-08：定版 SKILL 與 A 章修後複查

本次範圍：本節修改處、必要前文與過渡；projection 正文不變，僅手機目錄。五輪完整來源覆核後選第四輪實測最佳版本，仍只抓回 9 項既知必要問題中的 3 項，三項使用者優先問題均漏抓；不宣稱穩定全抓。比較與選版見[本次決定](clear-tutorial/final-selection-2026-10-08/final-decision.md)，十份原始報告與指紋保留。

教材修正明確使用既知問題提示，與獨立抓漏分開記錄。作者按定版 SKILL 修正後，另由未參與作者的[技術／銜接審閱者](clear-tutorial/final-selection-2026-10-08/post-repair/technical.json)及[實際頁面審閱者](clear-tutorial/final-selection-2026-10-08/post-repair/visual.json)覆核指定焦點；圖中字體調整另有[小變更紀錄](clear-tutorial/final-selection-2026-10-08/post-repair/technical-delta.json)。所查焦點無未解必要問題，不是逐段盲讀、真人學生驗收或全書新審閱。

[實際 Chromium 頁面與操作](clear-tutorial/final-selection-2026-10-08/post-repair/browser.json)涵蓋 8 頁的 1280×844 與 390×844 尺寸、公式、圖片、手機內嵌目錄與橫向表格；其他尺寸／瀏覽器及完整選讀未驗。[手工機制核對](clear-tutorial/final-selection-2026-10-08/post-repair/hand-checks.json)支持新增算例一致，不當成訓練效果證據。六節 A 主實驗在 CPU 重跑通過；兩份 40 步結果沿用仍符合原程式指紋的保存紀錄。

[修正處置](clear-tutorial/final-selection-2026-10-08/repair-decisions.json)與[範圍比對](clear-tutorial/final-selection-2026-10-08/post-repair/scope-check.json)保存依據。B 以後的 92 個課文／notebook、路線 B 起段落及所有原實驗程式不變；公開程式 tag 仍是 lessons-v0.6.1。覆蓋指紋更新只綁定本次實際查核版本與上述範圍。
