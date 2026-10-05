# 審查紀錄：ResNet identity shortcut

審查範圍：`docs/lessons/03-identity.md`、頁面上的圖（`docs/assets/diagrams/03-identity.svg`），以及 `lesson_cases/03-identity.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要或建議問題。實際有改動的只有 docs/lessons/03-identity.md 的四處：兩個程式區塊開頭改成 data-excerpt 標記、第 103–105 行、第 155 行。所有結果都在我自己的暫存副本跑過

1. 頁面對程式的說法都成立。
   - 原程式 exit 0，印出的前兩行和第 103 行引用的一字不差：`x=[-2.0, -1.0, 0.0, 1.0], y=[-2.0, -1.0, 0.0, 1.0]` 與 `input_gradient=[1.0, 1.0, 1.0, 1.0]; negatives_preserved=True`。
   - 兩題練習都照頁面指示一步一步改（腳本與輸出在同層的暫存副本）。
     - 練習 1 只加 ReLU：停在斷言一。只改好斷言一：停在斷言二。兩個斷言都照答案改好：exit 0，第一部分印出 `y=[0.0, 0.0, 0.0, 1.0]` 與 `input_gradient=[0.0, 0.0, 0.0, 1.0]; negatives_preserved=False`，和第 155 行一致。第二部分 loss 變成 0.5366／0.5240，確實和紀錄的 1.0816 不同，更新相關的斷言仍通過。
     - 練習 2（stride=2）：第一部分兩行和原程式相同，仍是 True；第二部分在第一次 forward 就丟出頁面引用的 RuntimeError，內容一字不差。再加上 shape 斷言後，程式停在第一部分的那個斷言。
   - 第 105 行講的語法和函式行為也另外確認過：`x < 0` 得到和 x 同 shape 的 True／False tensor，`x[x < 0]` 取出 [-2.0, -1.0]，`torch.equal` 在 shape 或值不同時回傳 False。頁面也明說它只比對 x 為負的兩個位置，整個 y 是否等於 x 由斷言一核對，沒有寫成整個 block 是 identity。

2. 清單上的項目都處理了。
   - 先前審查意見第 3 行：HEAD 裡已經是 lessons-v0.4.0。
   - 第 153 行那句寫死字串的附註已刪掉，換成實際印出的內容。受程式改動影響的段落第 103、153 行都照做。第 179、185 行在自動產生的紀錄區塊內，編輯已列為等重產紀錄。
   - 全頁搜過修訂經過或審查經過的字眼，沒有找到。頁面裡出現的「原本」都在講內容本身，不是在講教材怎麼改過。

3. 數字沒問題。新加的都是決定性的值（x、y、梯度、True／False），沒有帶進這台 Mac 跑出來的訓練數字或計時。要等重產紀錄的值，編輯都列了。
   - 我在暫存副本用 verify_curriculum 的 run／attach 只重產本節，模擬發布時的狀態。結果只有頁尾紀錄區塊改變：第二行變成 `negatives_preserved=True`，和正文一致，正文沒有任何變動。
   - 這份模擬紀錄是在 Mac 上產生的，只留在暫存副本，不能當作正式紀錄。

4. 摘錄檢查通過。摘錄比對工具印出 `docs/lessons/03-identity.md []`。
   - 我在暫存副本裡把兩個區塊各改壞一行，檢查兩個都會報錯，表示標記真的有在比對。
   - 兩段練習答案是讀者要自己寫的片段，不必標記。正文沒有引用程式行號。

5. 可讀性：本頁第一次用到 True／False 索引，第 105 行就地解釋了，順序也合理：先說印出什麼，再解釋 negatives_preserved，練習 1 的答案再接著用 False。

6. 建置與圖都正常。
   - zensical build --clean --strict 與 scripts/validate_site.py 都 exit 0；模擬重產紀錄後再跑一次也都 exit 0。
   - 建好的 HTML 裡，兩段摘錄都有 Python 語法上色並帶 data-excerpt 屬性，`&lt;` 跳脫正確。
   - 03-identity.svg 和 repo 裡的檔案逐位元相同，有 viewBox、title、desc。用 qlmanage 算繪後畫面乾淨，內容和程式一致：Conv→ReLU→Conv、identity shortcut、相加後沒有 ReLU。
   - 和本節有關的檔案（lesson_cases/03-identity.py、notebooks/03-identity.ipynb、03-identity.json、03-identity.svg）相對 HEAD 都沒動，只有頁面改了。

僅供參考，不算問題：
- validate_lessons 在暫存副本裡只因審查涵蓋檢查失敗（所有頁面都是 no review）；validate_curriculum_evidence 因紀錄過期失敗，第一個報錯的是 00-warmup。兩者都是審查用的事實與寫作規範清單列出的暫時狀態，摘錄與 notebook 一致性那幾項已經通過。
- 我檢查到一半時，工作樹裡出現一批暫存的刪除（artifacts/checks/ 下多份舊紀錄，含 lesson-runtime.json）。我第一次看 git status 時還沒有，所以是同時在跑的其他工作做的，不是這次的編輯。本頁沒有連到任何被刪的檔案。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：讀了以下檔案：
- 審查用的事實與寫作規範清單
- docs/lessons/03-identity.md 全文
- lesson_cases/03-identity.py
- notebooks/03-identity.ipynb 各格（確認最後一格就是完整程式）
- docs/assets/diagrams/03-identity.svg：讀原始碼，也用 qlmanage -t -s 1200 轉成 PNG 看圖
- docs/glossary.md
- 前置與相鄰各節 00-warmup、01-small-cnn、02-diagnostics、03-projection、03-comparison：核對前置知識（連鎖律、ŷ=wx 的梯度、輸出長度公式、參數公式），以及「直接梯度路徑／直接路徑」、BatchNorm、記憶體說法的交叉引用

建置與渲染：在暫存副本執行 zensical build --clean --strict，沒有錯誤。讀了 site/lessons/03-identity/index.html 的 article，另用系統 headless Chrome 截整頁圖（Playwright 的瀏覽器沒裝），確認數學式、表格、摺疊區塊和圖都正常顯示。

執行課程程式（PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg，用 .venv-model 的 python）：前兩行與正文逐字相同，第二部分的 loss 是 1.0816／1.0724，結束碼 0。

照頁面做了兩題練習：
- 練習 1 分三步。只加相加後的 ReLU，斷言一 AssertionError；只改斷言一，換斷言二失敗；兩個斷言都照答案改，印出 `y=[0.0, 0.0, 0.0, 1.0]`、`input_gradient=[0.0, 0.0, 0.0, 1.0]; negatives_preserved=False`，第二部分 loss 變成 0.5366／0.5240，更新相關的斷言照樣通過。每一步都和答案相符。
- 練習 2：第一個卷積改成 stride=2。第一部分靠 broadcast 照樣通過，印出的兩行和原本相同；第二部分出現 `RuntimeError: The size of tensor a (8) must match the size of tensor b (4) at non-singleton dimension 3`，和頁面逐字相同。再照答案在 forward 加上 shape assert，第一部分就停在這個 assert。和答案相符。

另外用小段程式核對：全零權重時兩層卷積的權重梯度都是 0；兩種 channel 數的參數數是 18 與 288。也自行推導了「中間 channel 數和輸入相同、沒有 bias 的普通 block，無法對有正有負的輸入做 identity」這個說法，結論成立。

最後確認：審查的四個檔案與真實工作樹逐位元組相同，我沒有在 repo 根目錄內執行或寫入任何東西。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | 〈把「修正」與「完整答案」分開〉第 4、5 段：「若相加後再接 ReLU…原版 ResNet 的 block 相加後有 ReLU，正是這種情況。」「本節拿掉相加後的 ReLU，才能用正負輸入精確檢查…」；相關的另一處是〈程式只多一個加號，學習目標卻不同〉最後一段「論文網路裡這幾層的輸入都剛經過 ReLU、沒有負數」 | 頁面說原版 ResNet 的 block「正是這種情況」，也就是主分支輸出 0 時，整個 block 的輸出不等於 x。但頁面沒有說：原版網路裡每個 block 收到的 x 都剛經過上一個 ReLU，沒有負數。對這種 x，ReLU(x+0)=x，所以主分支輸出 0 時，原版 block 對它實際收到的輸入仍是原樣輸出。「輸入沒有負數」要到後面一段才出現，而且放在談普通層的脈絡裡，讀者接不起來。讀完這裡，讀者會以為原版 ResNet 的 block 在 F(x)=0 時不會原樣輸出，也會以為收益第 1 點（主分支權重為 0 時整段等於輸入）只對本節拿掉 ReLU 的版本成立。這是對論文設計的錯誤印象。 |
| 2 | 建議 | 第一段：「論文把這種『更深反而連訓練誤差都較高』的現象稱為退化（degradation）。」 | 讀者在第 2 章剛學過 overfit 和「最佳化問題」是兩回事。看到「網路更深反而變差」，最自然會先想到 overfit。頁面只用「連訓練誤差都」暗示這不是 overfit，沒有明說。頁末「沒驗證」清單提到最佳化問題，也沒和開頭的退化接起來，讀者不容易知道 shortcut 要解決的是哪一類問題。 |
| 3 | 建議 | 〈相加前，兩條路的 shape 必須完全相同〉最後一段「所以這種 block 應該用 assert 明確核對兩條路的完整 shape」，以及緊接著的 `IdentityBlock` 程式摘錄 | 頁面剛建議 residual block 要用 assert 核對兩條路的 shape，下一段摘錄的 `IdentityBlock.forward` 卻沒有這行，頁面也沒說為什麼。讀者會懷疑自己看漏了，或以為這個建議可有可無；要到練習 2 的答案，才第一次看到加 assert 的寫法。 |
| 4 | 建議 | 〈程式只多一個加號，學習目標卻不同〉最後一段：「論文網路裡這幾層的輸入都剛經過 ReLU、沒有負數，普通層用特殊的權重還能原樣輸出；本節的 x 有正有負…都做不到對所有輸入原樣輸出…」 | 一段裡同時放了四件事：論文的假說、論文網路的輸入沒有負數、普通層用「特殊的權重」可以原樣輸出、本節帶負數輸入的普通 block 完全做不到。「特殊的權重」沒有例子，讀者想不出是什麼。這段真正要分清的是：論文說的困難是「不容易學到」，不是「表示不了」；本節的設定則是根本表示不了。這一點沒有明說，讀者容易把兩個說法混成一個。 |
| 5 | 建議 | 〈不背微積分也能看懂：梯度沿 shortcut 直接傳回〉的 Δy/Δx 式子與下一段「所以每個 block 往回傳的變化率都含一個不經過權重的 1」 | 頁面只推到單一 block 的變化率是「1＋主分支那一項」，就接著說梯度能沿 shortcut 直接傳回前面的層、讓網路疊得更深。從「每個 block 都有一個 1」到「疊很多個 block 時，這個 1 還一路留著」，需要用連鎖律把各 block 的變化率相乘，這一步沒寫。讀者因此看不出 shortcut 為什麼對深網路特別重要，也看不出 3.3 節 plain 的 stem 梯度為什麼小得多。另外，式子用的是有限差商 Δy/Δx，沒提 Δx 趨近 0；暖身節的梯度定義是取這個極限。 |
| 6 | 建議 | 〈程式只多一個加號〉最後一段（第二部分的說明），以及頁尾執行紀錄 | 頁尾紀錄寫著「每個數字的意思，以本頁正文的說明為準」，但正文只解釋了第一部分印出的兩行。後三行沒有說明：`step=0, shape=(2, 4, 8, 8), loss=…`、`step=1…`、`random branch updated; …`。讀者不知道 loss 是那次更新前還是更新後量的、為什麼大約是 1、最後一行代表什麼，對照自己的輸出時，也判斷不了結果算不算正常。 |
| 7 | 建議 | 〈程式只多一個加號〉最後一段「第二部分只確認主分支末層的權重收得到非零梯度」，以及完整程式第二部分的 `trained.branch[2].weight` | 正文寫「主分支末層」，完整程式寫的是 `trained.branch[2]`。頁面沒說 `nn.Sequential` 可以用索引取出某一層，也沒說索引從 0 數、而且 ReLU 也佔一個位置。程式新手想找「第二個卷積」時會以為是 `branch[1]`，或把 `[2]` 誤讀成別的東西，看不懂第二部分的斷言在檢查哪一層。 |
| 8 | 建議 | 〈自主練習與答案〉第一段：「在本機則先複製一份再改。兩題各自從原始程式開始。」 | 頁面要本機讀者先複製一份再改，卻只給過原檔的執行指令 `PYTHONPATH=. python lesson_cases/03-identity.py`。讀者改好副本後照頁首指令執行，跑的是沒改過的原檔，輸出和原本一樣，會以為修改沒有效果。Colab 讀者做完練習 1 之後，頁面也沒說怎麼回到原始程式再做練習 2。 |
| 9 | 建議 | 〈自主練習與答案〉練習 1 參考答案第二段：「x=0 那格正好在 ReLU 的折點，PyTorch 在這一點取 0」 | 題目要讀者特別注意 x=0 那一格、先自己預測，但本頁和前面各節都沒說過 ReLU 在 0 這一點的梯度怎麼算。這一格又剛好 y＝x（都是 0），讀者很可能照「原樣通過」預測梯度是 1。答案只說「PyTorch 在這一點取 0」，沒點出這是 PyTorch 的規定、不是能從數學推出的結果，讀者會以為是自己算錯。同一句的「sum loss」也是本頁其他地方沒用過的英文叫法。 |
| 10 | 建議 | 〈自主練習與答案〉練習 2 參考答案第一點：「主分支輸出 `[1,1,1,1]`。它和 `[1,1,2,2]` 的 x 相加時…F(x) 又全是 0」 | 本頁的方括號同時用來寫數值（`[-2,-1,0,1]`、`[0,0,0,1]`）和 shape（`[1,1,2,2]`）。這裡的 `[1,1,1,1]` 前面沒寫「shape」，很容易被讀成「主分支輸出四個 1」，又和同一句的「F(x) 又全是 0」看起來互相矛盾。 |
| 11 | 建議 | 〈自主練習與答案〉練習 2 參考答案第二點的錯誤訊息 `RuntimeError: The size of tensor a (8) must match the size of tensor b (4) at non-singleton dimension 3` | 錯誤訊息照抄出來了，但沒有解讀。程式新手看不出 tensor a、tensor b 各指哪一個，dimension 3 是哪一軸，non-singleton 又是什麼意思；以後自己遇到同樣的訊息，也不知道該回頭查哪一條路、哪一軸的 shape。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：暫存副本依指示用 rsync 複製到暫存副本。所有程式都只在這份副本裡跑，沒有在 repo 內寫入任何東西。

【讀過的 repo 檔案】
- docs/lessons/03-identity.md
- lesson_cases/03-identity.py（只 import torch，沒有 import miniyolo）
- docs/assets/diagrams/03-identity.svg
- 交叉引用：docs/lessons/00-warmup.md（ŷ=wx、連鎖律）、01-small-cnn.md（輸出長度公式、C_out(C_in k²+1)）、03-projection.md（第 35 行的選項 A/B）、03-comparison.md（BN 說明、記憶體、直接路徑）
- docs/glossary.md、section-map.json（各節順序）
- scripts/validate_lessons.py（excerpt 比對邏輯）
- artifacts/checks/curriculum/03-identity.json（只用來確認它是舊輸出）
- 審查用的事實與寫作規範清單

【執行過的程式與指令】
1. 在暫存副本執行 PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/03-identity.py，exit 0。印出：x=[-2.0, -1.0, 0.0, 1.0], y=[-2.0, -1.0, 0.0, 1.0] / input_gradient=[1.0, 1.0, 1.0, 1.0]; negatives_preserved=True / step=0, shape=(2, 4, 8, 8), loss=1.0816 / step=1, … loss=1.0724 / random branch updated…。loss 是本機數字，不當作標準值。
2. 摘錄比對，輸出 []。另外人工逐行核對兩段摘錄的順序。
3. 練習的變體程式都放在暫存副本：
   - ex1_raw.py（相加後加 ReLU）：停在 assert torch.equal(x, y)。
   - ex1_only_first.py（只改斷言一）：停在斷言二。
   - ex1_fixed.py（兩個斷言都改）：印出 y=[0.0, 0.0, 0.0, 1.0]、input_gradient=[0.0, 0.0, 0.0, 1.0]、negatives_preserved=False，第二部分的斷言都通過。
   - ex2_stride.py（第一個卷積改 stride=2）：第一部分印出與原程式相同的兩行；第二部分報 RuntimeError: The size of tensor a (8) must match the size of tensor b (4) at non-singleton dimension 3，與頁面逐字相同。
   - ex2_guard.py（加 shape 斷言）：第一部分停在 shape 斷言。
   - ex2_probe.py：主分支輸出 (1,1,1,1)，梯度全為 1，第二部分主分支輸出 (2,4,4,4)。
4. 用 Python 另外檢查：torch.equal 的 docstring（2.9.1）、ReLU 在 0 的梯度是 0、主分支全為 0 時兩層權重梯度都是 0、參數數 18 和 288。
5. 用 qlmanage -t -s 1200 把 SVG 轉成 PNG 並看過：每段線的 shape 標註、虛線框指向相加之後那段線，都與正文一致。
6. 在暫存副本執行 .venv-docs/bin/zensical build --clean --strict，exit 0，回報 No issues found；python3 scripts/validate_site.py，exit 0，全部 passed。也檢查了渲染後 HTML 的表格、details、arithmatex 和粗體。
7. 用 gh api 和 curl 取得下列外部來源。

【外部來源】
- He et al. 2015，Deep Residual Learning for Image Recognition，https://arxiv.org/abs/1512.03385。PDF 抓 https://arxiv.org/pdf/1512.03385v1，全文從 https://ar5iv.labs.arxiv.org/html/1512.03385 解析。關鍵原文：
  - §1 圖 1 說明："Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error"
  - §1："a degradation problem…adding more layers to a suitably deep model leads to higher training error"
  - §1："There exists a solution by construction…the added layers are identity mapping"
  - §3.1："we explicitly let these layers approximate a residual function F(x):=H(x)−x"
  - §3.1："the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings"
  - §3.2："We adopt the second nonlinearity after the addition (i.e., σ(y))"
  - §3.2："(except for the negligible element-wise addition)"
  - §3.2："The element-wise addition is performed on two feature maps, channel by channel."
  - §3.3："(A) The shortcut still performs identity mapping, with extra zero entries padded…(B) The projection shortcut…(done by 1×1 convolutions)"
  - §3.4："We adopt batch normalization (BN) right after each convolution and before activation"
  - §4.1："We argue that this optimization difficulty is unlikely to be caused by vanishing gradients…the backward propagated gradients exhibit healthy norms with BN"
  - §4.2："On this dataset we use identity shortcuts in all cases (i.e., option A)"
- He et al. 2016，Identity Mappings in Deep Residual Networks，https://ar5iv.labs.arxiv.org/html/1603.05027：
  - §2 式 (5)：∂E/∂x_l = ∂E/∂x_L(1 + ∂/∂x_l ΣF)，以及 "a term…that propagates information directly without concerning any weight layers"
  - §4："both being derived under the assumption that the after-addition activation f is the identity mapping. But…f is ReLU as designed in [1], so Eqn.(5) and (8) are approximate"
- Goyal et al. 2017，https://ar5iv.labs.arxiv.org/html/1706.02677，§5.1："Setting γ=0 in the last BN of each residual block causes the forward/backward signal initially to propagate through the identity shortcut"
- 原作者官方模型：https://github.com/KaimingHe/deep-residual-networks/blob/a7026cb6d478e131b765b898c312e25f9f6dc031/prototxt/ResNet-50-deploy.prototxt。卷積是 bias_term: false，後面接 BatchNorm 和 Scale；res2a 的 Eltwise（相加）後面接 res2a_relu（ReLU）。
- torchvision v0.24.1（commit d801a34632023859a0a274803d6abaf0a45d77a5）：https://github.com/pytorch/vision/blob/d801a34632023859a0a274803d6abaf0a45d77a5/torchvision/models/resnet.py。conv3x3 用 bias=False；BasicBlock.forward 是 conv1→bn1→relu→conv2→bn2、out += identity、out = self.relu(out)；zero_init_residual 的註解："Zero-initialize the last BN in each residual branch, so that the residual branch starts with zeros, and each residual block behaves like an identity."
- PyTorch v2.9.1（commit d38164a545b4a4e4e0cf73ce67173f70574890b6）：https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/tools/autograd/derivatives.yaml
  - relu 的反向是 "self: threshold_backward(grad, result, 0)"，所以在 0 的梯度取 0。
  - add.Tensor 的反向直接把 grad 傳回兩邊，不保存輸入。
  - convolution 的反向用到 input，這支持「訓練時本來就要存 x」的說法。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/03-identity.md 第 125–127 行（收益第 2 點、「這兩點是 residual block 希望讓網路能疊得更深的理由」）與第 77–87 行梯度段 | 頁面只引用 2015 年的 ResNet 原始論文，但「梯度裡多出的 1，讓梯度沿 shortcut 直接傳回」這個論證不在那篇論文裡。它出自同一批作者 2016 年的〈Identity Mappings in Deep Residual Networks〉（arXiv 1603.05027）第 2 節式 (5)。而且推導前提是相加後的 activation 是 identity；作者在第 4 節明說，原版相加後是 ReLU 時，式 (5) 只是近似。原始論文 4.1 節的立場正好相反："this optimization difficulty is unlikely to be caused by vanishing gradients…the backward propagated gradients exhibit healthy norms with BN"。頁面把兩點並列成理由，又緊接在第 75 行「論文的理由是…」之後，讀者容易以為原始論文提出 shortcut 是為了解決梯度傳不回去。 |
| 2 | 建議 | docs/lessons/03-identity.md 第 127 行「identity shortcut 也只適用於兩條路 shape 相同的情況。」 | 在本頁的嚴格定義（輸出＝輸入）下，這句成立。但它與論文用語、也與下一節直接衝突。原論文 3.3 節的選項 (A) 寫 "The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter"；4.2 節的 CIFAR-10 實驗寫 "we use identity shortcuts in all cases (i.e., option A)"，包括維度增加的地方。03-projection 第 35 行也寫「(A) 仍用 identity，以 stride 2 取樣後，多出的 channel 補 0，不增加參數」。讀者連讀兩節會看到矛盾，也可能誤以為 shape 一變就只能用有參數的 projection。 |
| 3 | 建議 | docs/lessons/03-identity.md 第 107 行（「所以正式網路不能把主分支全部設成 0；…就是為了避免把這個人工測試當成訓練建議」）與第 134 行常見錯誤 | 「整個主分支都設成 0 就學不動」是對的，但讀者可能推廣成「讓 block 一開始就是 identity 不是訓練做法」。實務上確實有讓 F(x) 從 0 開始的初始化。torchvision v0.24.1（d801a34，torchvision/models/resnet.py）的 zero_init_residual 只把每個 residual branch 最後一個 BN 的 γ 設成 0，註解寫 "so that the residual branch starts with zeros, and each residual block behaves like an identity"，依據是 Goyal et al. 2017（arXiv 1706.02677 §5.1）。這和本頁的人工零分支差在只歸零最後一層：最後一層的輸入不是 0，所以收得到梯度。這正好可以用本頁折疊說明「為什麼兩層的權重梯度都是 0」的同一套推理解釋。 |
| 4 | 建議 | docs/lessons/03-identity.md 第 27 行「原版 ResNet 的 block 相加後有 ReLU，正是這種情況。」（並與第 75 行對照） | 原版每個 block 的輸入都是上一個 ReLU 的輸出（第一個 block 則是 stem 的 ReLU 加 max pooling），沒有負數。所以對它實際收到的輸入，F(x)=0 時 ReLU(x+0)=x，整個 block 仍原樣輸出；torchvision 的 zero_init_residual 註解也寫 "each residual block behaves like an identity"。本頁只在第 75 行談 plain 層時提到輸入非負。第 27 行的寫法容易讓讀者以為，第 75 行引述的論文論證（權重推向 0 就得到 y≈x）在原版會因為相加後的 ReLU 而不成立。 |
| 5 | 建議 | docs/lessons/03-identity.md 第 13 行「第二部分另建一個權重隨機的 block，訓練 2 步，只確認權重真的會更新。」 | 程式只斷言主分支末層（trained.branch[2]）的權重梯度非零、2 步後有改變，第一個卷積沒有檢查。第 118 和 131 行寫得精確，第 13 行說的範圍比程式實際檢查的寬。 |
| 6 | 建議 | docs/lessons/03-identity.md 第 69 行「`nn.Sequential` 把幾層依序串起來」 | 本頁第 23 行把「串接」定義成 concatenation，第 134 行又把「相加寫成串接」列為常見錯誤。緊接在程式摘錄後用「串起來」描述 Sequential（逐層接續計算），初學者容易和 torch.cat 的「串接」混在一起。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 讀者審查 | 原版 block「正是這種情況」，卻沒說原版的輸入沒有負數（must） | 已修正：補一句：原版每個 block 的輸入都剛經過 ReLU、沒有負數，所以 F(x)=0 時，原版 block 對實際收到的輸入仍原樣輸出；並寫明本節的 x 刻意含負數，才拿掉相加後的 ReLU |
| 2 | 讀者審查 | 退化沒點明不是 overfit | 已修正：在退化定義後補一句：退化不是 overfit，連訓練資料都學得比較差，屬於第 2 章說的最佳化問題（附連結） |
| 3 | 讀者審查 | 頁面建議用 assert 核對 shape，摘錄程式卻沒有這行 | 已修正：補一句：IdentityBlock 沒有這行，練習 2 會看到少了它時 broadcast 放過哪種錯誤；下一節的 block 有這行（已查 03-projection.py 確實有） |
| 4 | 讀者審查 | 「特殊的權重」沒舉例，也沒分清「學不到」與「表示不了」 | 已修正：拆成兩段：給出 kernel 中心為 1、只讀同編號 channel 的例子（不計 BatchNorm），並明說論文談的是不容易學到，本節帶負數的輸入則是連表示都做不到 |
| 5 | 讀者審查 | 梯度只推到一個 block，沒說疊多個 block 時那個 1 怎麼留下來 | 已修正：補兩個 block 相乘的算式 (1+a)(1+b)=1+a+b+ab（Δx 趨近 0），並對照普通 block 只剩 ab |
| 6 | 讀者審查 | 第二部分印出的三行沒有解釋 | 已修正：補一段：shape 是什麼、loss 是更新前量的 MSE、為什麼在 1 附近（小數因電腦而異）、最後一行要等斷言全部通過才印出（已對照程式） |
| 7 | 讀者審查 | 沒解釋 branch[2] | 已修正：在 nn.Sequential 那段補上索引從 0 數起，branch[0]、[1]、[2] 各是哪一層，並對應到「主分支末層」 |
| 8 | 讀者審查 | 沒說本機副本怎麼執行、Colab 怎麼還原 | 已修正：補上副本檔名與執行指令的例子，並提醒 Colab 做下一題前先改回原樣 |
| 9 | 讀者審查 | ReLU 在 0 的梯度沒點明是 PyTorch 的規定；「sum loss」用語不一致 | 已修正：改寫成：這一點左右的變化率不同、數學上沒有唯一的導數，PyTorch 規定取 0；「sum loss」改成「y.sum() 這個 loss」 |
| 10 | 讀者審查 | [1,1,1,1] 沒標明是 shape | 已修正：改成「主分支輸出的 shape 是 [1,1,1,1]（只剩一個值，而且是 0）」，x 那邊也寫明是 shape |
| 11 | 讀者審查 | 練習 2 的錯誤訊息照抄、沒解讀 | 已修正：補一句解讀：tensor a、b 各指哪一個，dimension 3 是 W 軸，non-singleton 是什麼意思 |
| 12 | 技術查核 | 梯度裡那個 1 的論證出自 2016 年的論文，原始論文反而認為問題不是梯度消失 | 已修正：收益段寫明出處不同：第 1 點是原始論文的假說，第 2 點出自 Identity Mappings（arXiv 1603.05027），前提是相加後沒有 activation；並補上原始論文認為有 BatchNorm 的 plain 變差不太可能是梯度消失 |
| 13 | 技術查核 | 「identity shortcut 只適用 shape 相同」與論文的 option (A) 衝突 | 已修正：改成「原樣通過的 shortcut 只適用於 shape 相同的情況；shape 不同時的兩種做法見下一節」 |
| 14 | 技術查核 | 讀者可能誤以為讓 block 一開始是 identity 不是正規的訓練做法 | 已修正：補一句：torchvision 的 zero_init_residual 只把最後一個 BatchNorm 的縮放係數設成 0，最後一層收得到梯度；不能用的是整個主分支都設成 0 |
| 15 | 讀者審查＋技術查核 | 第 27 行未提原版輸入沒有負數（與 reader must 同一件事） | 已修正：與讀者審查的必要一起修，用同一句話處理 |
| 16 | 技術查核 | 開頭說「只確認權重會更新」，範圍比程式實際檢查的寬 | 已修正：改成「只確認主分支末層的權重收得到梯度、真的會更新」 |
| 17 | 技術查核 | Sequential「串起來」容易和串接（concatenation）混淆 | 已修正：改成「依序連成一條流水線」 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/03-identity.md`、`docs/lessons/03-projection.md`、`docs/lessons/03-comparison.md`、`docs/lessons/04-localization.md`、`docs/lessons/04-coordinates.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

全文讀了 reviews/03-identity.md。獨立查核與方法齊全；1 項必要的處理寫在修正後檢查裡。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/03-identity.md | 讀者審查 10 項與技術查核 6 項 should 都沒有處理；紀錄寫「各項的處理見下方〈定稿修正〉」，但沒有這一節（原因同 03-projection）。 | 已修正：產生器改以頁名、節名、萬用字元、頁面上的圖與該節 notebook 把處理對應到頁面，〈定稿修正〉列出這一頁每一項的處理。 |
| 2 | 建議 | 〈技術查核〉方法 | 方法在列出查閱來源之前就被截斷（「3. 練習的變體程式都放在暫存副本： - ex1_raw.py…」），論文與 torchvision 的版本只散見於發現欄，讀者看不到完整的來源清單；另有「暫存副本」與「python3 摘錄比對工具 <暫存副本>」這類殘句。 | 已處理：查閱的來源寫在方法段與各項發現裡，沒有另外整理成清單。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：讀者審查 11 項（含 1 項必要）與技術查核 6 項都在〈定稿修正〉17 列處理（上一輪指出整節缺漏，已補）；批次檢查掛在本頁；技術查核列出 ResNet 兩篇論文（1512.03385、1603.05027）。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/03-identity.md 第 11、15、100、169 行 | 〈上述處理〉第 2 項寫「殘句已清理」，但它點名的「python3 摘錄比對工具 <暫存副本> docs/lessons/03-identity.md，輸出 []」仍在第 100 行；另有「所有結果都在我自己的暫存副本跑過：暫存副本」「腳本與輸出在同層的暫存副本」，〈定稿修正〉來源欄還有英文內部標籤「reader+tech」。 | 來源欄改成中文。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 100 行指令與來源欄已改好；但點名的另一句沒改，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/03-identity.md 第 15 行 | 點名的「（腳本與輸出在同層的暫存副本）」逐字還在，處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 2 | 建議 | reviews/03-identity.md 第 11、86 行 | 第 11 行句末「所有結果都在我自己的暫存副本」斷在半句；第 86 行「方法：暫存副本 rsync 複製到暫存副本。」是殘句。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

第 4 輪第 1、2 項屬實：第 15 行「（腳本與輸出在暫存副本）」與第 11 行「所有結果都在我自己的暫存副本」仍在，第 86 行的殘句已改寫。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 3 輪第 1 項處理欄（第 194 行） | 「5 處中 4 處已改寫或刪除」把引用的第 2 輪處理說法「殘句已清理」也算成已改寫，但它仍在第 2 輪第 2 項處理欄（第 186 行）。實際改寫的紀錄文字是 3 處。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/foundations.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。
