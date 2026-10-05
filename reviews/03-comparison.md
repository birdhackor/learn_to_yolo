# 審查紀錄：Plain／residual 對照

審查範圍：`docs/lessons/03-comparison.md`、頁面上的圖（`docs/assets/diagrams/03-comparison-learning.svg`），以及 `lesson_cases/03-comparison.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題，只有 3 個 should（見下表）。

所有檢查都在暫存副本執行，輸出放在其中的暫存副本，repo 內沒有執行任何程式。

1. 對程式的敘述全部屬實（lesson_cases/03-comparison.py，sha 8acae191…）。
- 逐句核對過：資料排法（上緣 3、左緣 2～5、每個位置一紅一藍；validation 上緣 5、左緣 4、5）、data() 從像素讀回位置後比對排序清單、train_step 的步驟、forward hook 的計數規則（只看 output[0]、加法依 residual 旗標）。
- deepcopy 副本另建 SGD 暖機、暖機後的斷言，以及斷言清單：和程式完全對應，沒有漏掉的斷言。
- lesson 執行 exit 0、stderr 為空。

2. 照頁面練習實跑，四個階段都和頁面描述一致。
- 只改 range(1)：停在 `assert params == 986`。
- 再改 410：停在 `assert macs == 248840`。
- 再改 101384：plain 跑完 3 步並印出摘要，輪到 residual 才停在 shortcut_adds 那一行。
- 三個都改完：通過，印出 params=410、MACs/image=101384、shortcut_adds/image 0／1024。
- 梯度：plain stem 梯度 0.036–0.039，3 個 block 時約 0.001，大了一個數量級以上；residual 0.153–0.161，3 個 block 時 0.146–0.148，差不多。兩者 validation 都是 0.50，logit 差距約 0.2，換電腦也不會翻轉。參考答案描述的趨勢都成立。

3. 40 步段與機制段經實跑確認。
- run_learning_extensions.py exit 0。前 3 個 loss 四捨五入後和 lesson 印出的相同。
- plain 8 張全猜類別 1；residual 的 validation 4/4 全對，而且每組一紅一藍只差顏色。
- head bias 在初始和 40 步後都偏向類別 1。block 權重都在 ±1/6 內，plain 每過一個 block，訊號約剩 0.19–0.27 倍。

4. 數字。
- 確定值（986、248840、3072、410、101384、1024，以及 27648、36864、1/108、1/18）都對。
- 沒有出現 Mac 實測的數字，1 個 block 的結果只寫定性描述。
- 依紀錄而定的數字沿用 HEAD 紀錄，編輯者已列入清單。提醒：重產紀錄後，residual 的 step=0 loss 很可能變成 0.6921（Mac 上是這個值），第 171 行的「0.6922」到時要跟著改。

5. 程式摘錄。
- 摘錄比對工具在 repo 和副本都是 []。三段 data-excerpt 的順序和內容我人工核對過；省略處有註解說明；正文沒有寫程式行號。
- validate_lessons 在副本中通過 notebook 與 42 頁的摘錄檢查，只在審查紀錄覆蓋那一步失敗，這是審查用的事實與寫作規範清單預期的發布前狀態。

6. 建置與範圍。
- `zensical build --clean --strict` exit 0，結果是 No issues found；validate_site.py exit 0。表格、摺疊區塊、程式區塊都正確渲染。
- 頁尾執行紀錄區塊和 HEAD 逐字相同。和 03-comparison 有關的檔案中，只有這一頁在編輯期間被改過。
- 18-video:86 與 17-capstone:42 引用的暖機定義仍然成立。
- 清單上的先前審查意見（3、65、79、83、95、101、105、107）和受程式改動影響的段落各條都已處理，全頁沒有寫作或修訂經過的敘述。
- 本頁的 lesson unit 沒有列 SVG。我用 qlmanage 看了重新產生的 03-comparison-learning.svg：中文標籤、顏色和曲線形狀都和圖說一致，有 viewBox、`<title>` 和 aria-label，但沒有 `<desc>`。這張圖由 scripts/run_learning_extensions.py 產生，不屬於本頁範圍。工作樹裡目前仍是舊的英文版，要等 --record 重產。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/03-comparison.md:109（count_per_image 摘錄之後的說明段） | 摘錄裡的 `if … elif …` 是課程頁面第一次出現 if 敘述：00～03-projection 各頁的程式區塊都沒有 if 敘述，`elif` 也只出現在本頁。說明段解釋了 `isinstance`、`model.modules()` 和 handle，卻沒說 `elif` 是什麼，也沒說三個分支只會執行第一個成立的那一段。程式新手看註解能懂每一支在做什麼，但不知道 `elif` 的意思，也看不出 plain 的 Block 為什麼什麼都不記。 |
| 2 | 建議 | docs/lessons/03-comparison.md:113–117（計時與暖機的三段） | 18-video.md:86 和 17-capstone.md:42 都引用本頁對暖機的定義，18-video 還直接連到本頁。但這三段放在標題「參數數量相同，計算仍略有不同」底下，頁面目錄裡看不到任何和時間或暖機有關的標題。從 18-video 點過來的讀者得自己往下翻或用搜尋，才找得到定義。 |
| 3 | 建議 | docs/lessons/03-comparison.md:173（「這段結論能支持的範圍是：…」） | 這句範圍說明只列了「沒有 BatchNorm、只有 3 個 block、用 PyTorch 預設初始化」，漏掉同頁〈限制〉段自己列出的「人工色塊」和「只有一個 seed」。讀者最可能照抄成結論的就是這一句；照抄的結果會變成「這種 plain 網路 40 步學不動」這樣的通則，和本頁「結論要寫出條件」的教法不一致，也和下一段「還要跑多個 seed」對不上。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：讀的檔案：審查用的事實與寫作規範清單、docs/lessons/03-comparison.md（原始 Markdown），以及在暫存副本用 zensical build --clean --strict 建出的 HTML（無錯誤），再用 headless Chrome 截 1280px 整頁圖逐段看。390px 的截圖被 Chrome 的最小視窗寬度裁切，不拿來判斷版面。另外讀了 lesson_cases/03-comparison.py、notebooks/03-comparison.ipynb（最後一格與程式逐字相同）、docs/glossary.md、03-identity.md、03-projection.md，以及 00-warmup、01-small-cnn、02-diagnostics 的相關段落、scripts/run_learning_extensions.py、miniyolo/figures.py、scripts/build_lesson_notebooks.py 第 17 行、artifacts/checks/curriculum/03-comparison*.json。

實跑：
- 課程程式：斷言全過，params、MAC、加法數都與頁面一致。
- `scripts/run_learning_extensions.py --section 03-comparison`：把 artifacts/runs/learning/03-comparison/learning.svg 和 repo 裡的舊 SVG 都用 qlmanage 轉成 PNG 對看；新圖的中文標籤、顏色、圖例與圖說一致。

照頁面做的練習（block 改成 1，逐步）：
- A：只改 range(1)，程式停在 `assert params == 986`，什麼都沒印。
- B：改了 986，停在 `assert macs`。
- C：再改 248840，plain 跑完 3 步並印出後，residual 停在加法數那行。
- D：三個值都改好，兩個模型都印出 410／101384／0、1024。plain stem 梯度約變成 36 倍，residual 差不多，validation 都是 0.50，與參考答案一致。
- `print(macs)` 提示：縮排對齊時印出 101384；寫在行首則出現 IndentationError。

另寫小程式核對「機制」段：
- block 卷積權重落在 ±1/6，平方平均約 1/108。
- plain 每過一個 block，數值大小約剩 0.19～0.27 倍；residual 約 1 倍。
- head bias 偏向類別 1，plain 的 8 張圖分數幾乎相同，40 次更新後仍是這樣。
- 全部 F 權重設 0：plain 每張圖分數相同，residual 紅、藍圖分數不同。
- 每個模型建立前重設同一個 seed，權重相同。
- 1 個 block 時 stem 與第 1 個 block 的初始權重不變，head 改變。

把原檔改成 1 個 block 再跑 40 步腳本：不會報錯，parameters=410，plain 也全對。之後已還原，並用 cmp 確認與原檔相同。

查證資料：用 WebFetch 讀 ResNet 論文 https://ar5iv.labs.arxiv.org/html/1512.03385 的 4.1 節〈Plain Networks〉（BN 讓前向、反向訊號都沒有消失；推測原因是收斂極慢）與 3.4 節（使用 He 初始化）。

沒做到的：沒有在 Colab 上執行；IPython 沒有安裝，我也沒有安裝，所以沒跑 display(SVG(...))。這台 Mac 量到的機器相關數字（loss 小數、梯度、時間），都不當作正確值。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | 〈結果怎麼讀〉的「機制」與「限制」兩段（docs/lessons/03-comparison.md 第 165、167 行），以及〈應如何下結論〉第 173 行 | 本頁把 plain 學不動完全歸因於預設初始化偏小、又沒有 BatchNorm，使訊號與梯度每過一個 block 縮成約 1/4，但沒有說明這和 ResNet 論文裡的退化不是同一件事。論文 4.1 節〈Plain Networks〉明說它的 plain 網路用 BN 訓練，前向訊號的變異數不為 0、反向梯度也正常（neither forward nor backward signals vanish），深層 plain 訓練誤差較高另推測為收斂極慢。3.1 節第 132 行把「加了 shortcut 的深層網路是否就不再有最佳化問題」交給本頁回答。照順序讀到這裡的讀者，會以為本頁示範了論文的退化機制，得到「ResNet 有效是因為防止訊號消失」的錯誤印象。現有限制只寫「加回 BatchNorm 會怎樣，本節沒有測」，擋不住這個推論。 |
| 2 | 建議 | 三段程式摘錄：第 21–25 行（Block 的 return）、第 71–81 行（main() 的斷言與 print）、第 85–107 行（count_per_image） | 摘錄裡的中文註解在完整程式裡都不存在，其中有整行的 `# self.residual 為 True 時回傳 x+F(x)…`、`# p 是一個參數 tensor…`、`# 把同一個 hook 掛到每個卷積…`；程式裡 count_per_image 的註解其實是英文（例如 `# values this layer outputs for image 0`）。頁面又說 Colab 那格「內容就是 lesson_cases/03-comparison.py；網頁上只摘錄了其中幾段」，卻沒有像 3.1、3.2 和其他各節那樣註明「中文註解是本頁加的」。讀者拿頁面對照 Colab 時會找不到這些行，以為自己的程式版本不對，或以為要自己補上。 |
| 3 | 建議 | 〈結果怎麼讀〉「機制」段（第 165 行），以及「限制」段的「He 初始化」（第 167 行） | 推導對數學好的初學者仍跳了幾步：(1) 沒說 −1/6～1/6 從哪來；(2)「平方的平均是 (1/6)²÷3」沒說 ÷3 怎麼來；(3)「交叉相乘的部分平均會互相抵消」沒有例子，讀者不知道指展開平方後的哪些項；(4) 從「平方的平均縮成 1/18」到「數值大小約剩 1/4」要開根號，文中沒寫；(5)「He 初始化」沒有定義，只說「權重較大」，讀者不知道 He 是人名，也不知道大多少、為什麼有幫助。整段五、六步推理擠在一個長段落裡，不易回頭核對。 |
| 4 | 建議 | 〈固定什麼，才能解讀觀察〉最後一段（第 117 行）與〈可核對的輸出〉第 130 行 | 頁面講了暖機，也說「兩個時間差距很小時，不能拿來說誰比較快」，但沒告訴讀者怎麼讀印出的 `3_step_seconds`。「差距很小」沒有尺度：是相差幾毫秒，還是相差幾成？兩個值都只有幾毫秒，單次量測時先計時的模型仍可能慢上好幾成。讀者剛讀過「暖機是為了不讓時間算到先計時的模型頭上」，看到這種結果會以為暖機沒生效，或得出 plain 比較慢的結論。另外，輸出最後一行英文 `Same initial weights/data/optimizer/steps; 3 steps and 4 validation images do not rank architectures.` 頁面沒有翻譯或說明；00-warmup 等節會逐行說明這種結尾行。 |
| 5 | 建議 | 〈常見錯誤、自主練習與答案〉第 183 行「可以在失敗的那行斷言前面暫時加一行 print，例如 print(macs)」 | 沒有提醒縮排。斷言在 main() 的 for 迴圈裡，開頭有 8 個空格；初學者若從行首寫 `print(macs)`，Python 會在下一行報 `IndentationError: unexpected indent`，不會印出數字（暫存副本實測）。00-warmup、02-diagnostics 叫讀者插入程式時都寫了「縮排和那行對齊」，本頁沒有。 |
| 6 | 建議 | 〈常見錯誤、自主練習與答案〉第 181 行（練習要改的檔案），以及〈訓練 40 步〉的重跑段（第 151–159 行） | 練習讓本機讀者直接改 `lesson_cases/03-comparison.py`，但 40 步腳本 run_learning_extensions.py 是從這個檔案匯入 Classifier。做完練習沒改回就跑 40 步腳本，它不會報錯，而是默默訓練 1 個 block 的模型。暫存副本實測：report 顯示 `parameters: 410`，plain 的訓練與 validation accuracy 也都是 1.00，和本頁的表格與結論相反，讀者會以為頁面錯了。3.1 節的練習有寫「在本機則先複製一份再改」，本頁沒有。Colab 裡改的是 notebook 格，不動磁碟上的檔案，所以只有本機會遇到。 |
| 7 | 建議 | 〈訓練 40 步〉重跑說明（第 151–159 行）與「觀察」段（第 163 行） | 第一行印出的「結果摘要」其實是約一百行的 JSON，含電腦資訊和一長串 SHA-256，頁面沒說要看哪幾個欄位才能和表格對照。「8 張訓練圖全部猜成類別 1」也只能從 JSON 的 `predicted_classes` 看出來，頁面沒提這個欄位。另外頁面只給 Colab 寫法，本機讀者把 `!python …` 照抄進終端機會出錯（zsh 會報 event not found），頁面沒說本機要去掉 `!`。 |
| 8 | 建議 | 第 151、159 行，與 notebook「本節可修改的完整實驗」說明格（由 scripts/build_lesson_notebooks.py 第 17 行產生） | 頁面說只要先執行環境格，再新增一格貼上三行；頁面又請讀者參考 notebook 的說明格，但那格寫的是「跑完下面的完整實驗後，另開一個 code cell 執行」。兩處說法不同，讀者會不確定自己是否漏了一步。其實 40 步腳本不需要先跑完整實驗，它自己載入程式並訓練。 |
| 9 | 建議 | 第 9、11、15、17、127 行 | (1) 第 15、17 行的「每個 channel 的空間平均」就是第 1 章教過的全域平均池化（GAP），頁面沒點明，讀者不易連回。(2) 第 127 行把 NaN 解釋成「無法定義的數」，術語表與第 1 章寫的是「Not a Number，算壞了的非數字」，說法不一致。(3) 第 11 行「3 步只確認梯度傳得到 stem、參數有更新、紀錄完整」裡的「紀錄完整」意思不明：是指每一行都印得出來，還是斷言都通過？(4) 第 9 行「[ResNet 原始論文](…) 對 plain」在連結後多一個空格，畫面上變成「原始論文對」；3.1 同一個連結後面沒有空格。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：**一手來源：**
- ResNet 論文 arXiv 1512.03385v1：PDF https://arxiv.org/pdf/1512.03385v1；文字取自 HTML 版 https://ar5iv.labs.arxiv.org/html/1512.03385。查了以下段落：
  - §1：梯度消失「has been largely addressed by normalized initialization … and intermediate normalization layers」。
  - §3.2 式 (1)：「We adopt the second nonlinearity after the addition」；「We can fairly compare plain/residual networks that simultaneously have the same number of parameters, depth, width, and computational cost」。
  - §3.3：Plain／Residual Network、option A／B。
  - §3.4：「We adopt batch normalization (BN) right after each convolution and before activation」；「We initialize the weights as in [13]」；mini-batch 256、learning rate 0.1、weight decay 0.0001、momentum 0.9。
  - §4.1：1000 類、1.28 million training images；表 2 為 27.94／27.88、28.54／25.03；「We argue that this optimization difficulty is unlikely to be caused by vanishing gradients. These plain networks are trained with BN…」。
  - 參考文獻 [13]。
- He 等人 arXiv 1502.01852（https://ar5iv.labs.arxiv.org/html/1502.01852）§2.2：std 為 √(2/n_l)。
- 官方 Caffe 定義：https://github.com/KaimingHe/deep-residual-networks/blob/a7026cb6d478e131b765b898c312e25f9f6dc031/prototxt/ResNet-50-deploy.prototxt。res2a 一段依序是 Convolution → BatchNorm → Scale → Eltwise(res2a) → ReLU(res2a_relu)。
- PyTorch tag v2.9.1（commit d38164a545b4a4e4e0cf73ce67173f70574890b6）：
  - torch/nn/modules/conv.py L179–188：_ConvNd.reset_parameters 用 kaiming_uniform_(a=√5)，註解說等同 uniform(-1/sqrt(k), 1/sqrt(k))。
  - torch/nn/init.py L511–573：kaiming_uniform_ 的 bound＝√3·gain/√fan；calculate_gain 的 leaky_relu gain 為 √(2/(1+a²))。
  - torch/nn/modules/linear.py L117–128。
  - torch/nn/modules/module.py 的 register_forward_hook docstring：「called every time after forward has computed an output」、「hook(module, args, output)」。
- Python 3.12 Language Reference §6.17 Operator precedence：https://docs.python.org/3.12/reference/expressions.html#operator-precedence。

**讀過的 repo 檔案：**
- 主要對象：docs/lessons/03-comparison.md、lesson_cases/03-comparison.py、scripts/run_learning_extensions.py。
- miniyolo/figures.py：use_svg_text、save_svg。
- scripts/validate_lessons.py：摘錄比對規則。
- notebooks/03-comparison.ipynb：4 格，最後一格與 lesson 檔逐字相同。
- 對照用頁面：docs/lessons/03-identity.md、docs/lessons/01-small-cnn.md（MAC 算法）。
- artifacts/checks/curriculum/03-comparison-learning.json：只用來了解格式，不當作正確值。
- 審查用的事實與寫作規範清單。

**執行的指令（都在暫存暫存副本）：**
- rsync 建立副本。
- 摘錄比對：輸出 []。
- `PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/03-comparison.py`：exit 0；params=986、MACs/image=248840、shortcut_adds/image=0／3072，與頁面一致。
- `.venv-model/bin/python scripts/run_learning_extensions.py --section 03-comparison`：exit 0，產生 report.json 與 learning.svg。
- 用 `qlmanage -t -s 1200` 把新 learning.svg 與倉庫 docs/assets/diagrams/03-comparison-learning.svg 轉成 PNG 檢視；倉庫那張是舊的英文版。
- `.venv-docs/bin/zensical build --clean --strict`：exit 0，No issues found；另檢查了渲染後的表格、數學與摺疊區塊。
- 自寫探針，放在 .../review/03-comparison-tech-probe：
  - probe_mech.py：初始化範圍、逐 block RMS、logit 拆解、F＝0 例子。
  - probe_grad.py：逐 block 的反向梯度，以及 40 步後的 logit 拆解。
  - 練習改成 range(1) 的 3 個變體：
    - 斷言改對：印出 410／101384／0、1024；plain stem 梯度約 0.036–0.039，residual 約 0.153–0.161；兩者 val acc 都是 0.50。
    - 斷言都沒改：plain 在 `assert params == 986` 失敗。
    - 忘了改加法數：plain 跑完 3 步後，residual 在 shortcut_adds 那行失敗。
  - probe_lr.py：seed 7／1／2／3 各配 learning rate 0.1／1.0／3.0，跑 40 步。

這台 Mac（Apple M1）量到的數字只用來看量級與變動幅度，沒有當作正確的機器相依值。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | 第 9 行「並省略三樣東西：BatchNorm、原版的深度，以及相加後的 activation」；第 29 行「原版 ResNet 在相加之後還有一個 ReLU。若只替 residual 加上它……所以本節兩個模型的 ReLU 位置完全相同」 | 論文的 plain 網路每個卷積後都先接 BN 再接 ReLU（arXiv 1512.03385 §3.4：「We adopt batch normalization (BN) right after each convolution and before activation」），所以每對 3×3 卷積的第二個卷積後面也有 ReLU。residual 只是把這個 ReLU 移到相加之後（§3.2：「We adopt the second nonlinearity after the addition (i.e., σ(y), see Fig. 2)」）。官方 ResNet-50-deploy.prototxt（KaimingHe/deep-residual-networks@a7026cb）也是 Eltwise「res2a」之後接 ReLU「res2a_relu」。 所以原版兩個模型在每個 block 末端各有一個 ReLU，數量本來就相同；§3.2 也正是用「same number of parameters, depth, width, and computational cost」來說明兩者可以公平比較。 本節的 plain 拿掉了第二個卷積後的 ReLU，相鄰兩個 block 之間變成兩個卷積直接相連。這是相對於論文 plain 網路的簡化，頁面沒有揭露。第 9 行只列「相加後的 activation」，第 29 行又把它寫成只有 residual 才有的元件。讀者會以為論文的 plain 在那個位置本來就沒有 ReLU；或以為照原版就是只替 residual 加回，而那正是本頁警告的混淆。 |
| 2 | 建議 | 第 165 行〈結果怎麼讀〉的「機制」段；第 9 行「論文的 plain 網路也有 BatchNorm；本節沒有它……加回後會怎樣，本節沒有測」；第 167 行〈限制〉 | 本頁觀察到的是 plain 的訊號與梯度逐 block 變小，也就是梯度消失。我在暫存副本用初始權重量到：plain 每過一個 block，輸出 RMS 約變成 0.19、0.20、0.27 倍，反向梯度約變成 0.27、0.23、0.21 倍；residual 都約 1 倍。 論文明說它的 plain 不是這個問題。§4.1：「We argue that this optimization difficulty is unlikely to be caused by vanishing gradients. These plain networks are trained with BN, which ensures forward propagated signals to have non-zero variances. We also verify that the backward propagated gradients exhibit healthy norms with BN.」§1 也說，梯度消失已經「largely addressed by normalized initialization … and intermediate normalization layers」。表 2 裡，18 層 plain 與 ResNet 的 top-1 錯誤率幾乎相同（27.94% 對 27.88%），差距到 34 層才出現（28.54% 對 25.03%）。 頁面雖然說結果只代表本設定，卻沒有點明這與論文研究的退化（degradation）是不同的機制。初學者讀完「機制」段，很容易把「shortcut 解決梯度消失」當成 ResNet 論文的結論。 |
| 3 | 建議 | 第 9 行「本節用 4 個 channel、3 個 block 與人工色塊，並省略三樣東西：……」 | 這句讀起來像完整列出與原版的差別，但還有三類沒列： (a) 初始化：論文 §3.4 寫「We initialize the weights as in [13]」，[13] 是 He 等人的〈Delving deep into rectifiers〉，§2.2 的 std 為 √(2/n_l)。本節用 PyTorch 預設：4→4 的 3×3 卷積是 U(−1/6, 1/6)，變異數只有 He 初始化的 1/6。「機制」段的解釋正是建立在這一點上，第 9 行卻沒提，要到〈限制〉才出現。 (b) optimizer：論文用 mini-batch 256、momentum 0.9、weight decay 0.0001，learning rate 從 0.1 開始，停滯時除以 10。本節是沒有 momentum、沒有 weight decay、整批 8 張、learning rate 固定 0.1 的 SGD。「SGD、learning rate 0.1」剛好和論文的起始值相同，讀者容易以為訓練設定也照原版。 (c) 結構：原版 stem 是 7×7、stride 2 的卷積加 max pooling，而且分段下採樣、channel 加倍（維度改變處用 zero-padding 或 projection shortcut）；本節全程維持 [B,4,16,16]。 |
| 4 | 建議 | 第 21–25、71–81、85–107 行的三段 data-excerpt 程式 | 摘錄裡的中文註解不在 lesson_cases/03-comparison.py 裡，例如：「# self.branch 就是 F：Conv–ReLU–Conv」、「# self.residual 為 True 時回傳 x+F(x)；為 False 時只回傳 F(x)」、「# p 是一個參數 tensor……」、「# 把同一個 hook 掛到每個卷積、Linear 與 Block 上，留下每個 handle」。原檔只有英文註解，有幾行完全沒有註解。摘錄比對工具不比對註解，所以檢查會通過。 頁面沒說註解是本頁加的，讀者到 Colab 最後一格找這些行會找不到。03-identity、03-projection、04-localization 等頁都有寫「中文註解是本頁加的」。 |
| 5 | 建議 | 第 171 行範例結論「訓練 40 步後，plain 的 loss 仍停在 ln 2 附近（0.693791 → 0.692943）……residual 的 loss 降到約 0.031」 | 0.692943 和「約 0.031」都是「第 40 次更新前」的 loss，也就是 39 次更新後量的（第 140 行表頭與第 149 行圖說都這樣寫）。同一句裡的訓練與 validation accuracy，才是 40 次更新全部完成後量的。這段要示範「寫出條件與實際數字」，卻把兩個時間點都寫成「訓練 40 步後」。 |
| 6 | 建議 | 第 171 行「換電腦重跑時小數末位可能略有不同」 | 40 步訓練會把不同電腦間的浮點差異放大，差異不只在末位。我在 Apple M1（PyTorch 2.9.1）上重跑 40 步：residual 第 40 次更新前的 loss 是 0.030958，頁面表格（記錄機器）是 0.030739，從小數第 4 位就不同。3 步的 residual stem 梯度也從第 4 位有效數字開始不同（0.146701 對 0.146834）。 這裡不是說 Mac 的值才對，而是「末位」低估了變動幅度。讀者在 Colab 看到 0.0310，可能會以為自己跑錯了。 |
| 7 | 建議 | 第 173 行「這段結論能支持的範圍是：在這個沒有 BatchNorm、只有 3 個 block、用 PyTorch 預設初始化的設定下，plain 40 步內學不動，residual 學得動」 | 範圍裡少了 learning rate 0.1，而結果對它很敏感。我在暫存副本用同一份模型與資料試跑 40 步： - learning rate 0.1：seed 1、2、3 的定性結果都和 seed 7 相同。 - learning rate 1.0：seed 7 的 residual 停在 loss 0.6931、accuracy 0.50；seed 1、3 的 plain 訓練與 validation accuracy 反而到 1.00（loss 仍約 0.693）。 所以「plain 學不動、residual 學得動」要靠 learning rate 0.1 這個條件，第 173 行列的三個條件不夠。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 讀者審查 | 把本節的訊號消失當成論文的退化機制（must） | 已修正：「限制」段新增一段：本節的直接原因是梯度消失；論文的 plain 有 BatchNorm 和 He 初始化，訊號與梯度都沒有消失，所以本節不能用來解釋論文的退化。結論範圍也補上同樣的限制 |
| 2 | 技術查核 | 論文的 plain 在第二個卷積後也有 ReLU，頁面把它寫成只有 residual 才有（must） | 已修正：開頭的差異清單改成「每個 block 末端的 activation」；第 29 行重寫：原版兩者都有，本節兩邊都拿掉，要照原版就得兩邊一起加回 |
| 3 | 技術查核 | 沒點明本節的機制和論文的退化不同（與讀者審查的必要同一件事） | 已修正：與讀者審查的必要用同一段處理，並附上論文 4.1 節的說法與 18 層、34 層的對比 |
| 4 | 讀者審查＋技術查核 | 摘錄裡的中文註解在原檔中沒有，頁面也沒說明 | 已修正：三段摘錄前都補上「中文註解是本頁加的」；main() 那段也註明「...」那行是本頁加的 |
| 5 | 讀者審查 | 機制推導跳步，He 初始化也沒定義 | 已修正：補上 ±1/√36 的來源、均勻分布平方平均是 a²/3、交叉項的例子、開根號得到約 1/4，並把段落拆開；He 初始化在「限制」段補上定義 |
| 6 | 讀者審查 | 沒說怎麼讀 3_step_seconds，最後一行英文也沒翻譯 | 已修正：補一句：時間不拿來比較，單次量測可能顛倒；並翻譯最後一行英文 |
| 7 | 讀者審查 | 插入 print(macs) 時沒提醒縮排 | 已修正：補上「縮排和下面那行斷言對齊，也就是開頭 8 個空格」（已核對程式的縮排） |
| 8 | 讀者審查 | 本機直接改原檔，會讓 40 步腳本默默改跑 1 個 block 的模型 | 已修正：補上：本機先複製一份再改，或做完改回，並說明原因（已確認腳本從原檔路徑匯入 Classifier） |
| 9 | 讀者審查 | 沒說 JSON 摘要要看哪些欄位；本機執行要去掉 ! | 已修正：補上欄位對照（initial_loss、last_pre_update_loss、train_accuracy、validation_accuracy、predicted_classes，已核對腳本），以及本機的執行指令 |
| 10 | 讀者審查 | 頁面與 notebook 說明格對執行順序的說法不同 | 已修正：頁面註明不需要先跑完整實驗，先跑也可以；build_lesson_notebooks.py 不在可編輯範圍內，沒有改 |
| 11 | 讀者審查 | 沒點明 GAP、NaN 說法與術語表不一、「紀錄完整」意思不明、連結後多一個空格 | 已修正：四處都改：補上 GAP、NaN 改成與術語表一致、「紀錄完整」換成具體說法、刪掉連結後的空格 |
| 12 | 技術查核 | 開頭的省略清單不完整（缺初始化、optimizer、stem） | 已修正：清單改成「主要差別」，補上預設初始化（原版是 He）、簡單 SGD（原版另有 momentum、weight decay），以及 stem 與分段下採樣 |
| 13 | 技術查核 | 範例結論把第 40 次更新前的 loss 寫成「40 步後」 | 已修正：範例結論與觀察段都寫明「第 40 次更新前」與「40 次更新後」 |
| 14 | 技術查核 | 「小數末位略有不同」低估了換電腦後的差異 | 已修正：改成：訓練後的小數可能從第 3、4 位起就不同，定性結論不受影響 |
| 15 | 技術查核＋先前查核留下的項目 | 結論範圍漏了 learning rate 0.1、人工色塊、單一 seed | 已修正：範圍句補齊這些條件；「限制」段補上「也只試了 learning rate 0.1」 |
| 16 | 先前查核 | if … elif … 沒有解釋 | 已修正：在 isinstance 那句旁補一句：由上往下檢查，只執行第一個成立的那段；三個條件都不成立時什麼都不記 |
| 17 | 先前查核 | 暖機的定義在目錄裡看不到 | 已修正：加上小標題「### 時間怎麼量才公平：暖機」 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/03-identity.md`、`docs/lessons/03-projection.md`、`docs/lessons/03-comparison.md`、`docs/lessons/04-localization.md`、`docs/lessons/04-coordinates.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/03-comparison.md 第 9 行「stem 與分段下採樣也都簡化了」 | 本節程式完全沒有分段下採樣：3 個 block 全程都是 [B,4,16,16]。寫「簡化了」會讓讀者以為程式裡有某種簡化過的下採樣。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |
| 2 | 建議 | docs/lessons/03-comparison.md 第 173 行「論文第 4.1 節檢查過它們的前向訊號與反向梯度都沒有消失」 | 論文 4.1 節的說法是：BatchNorm 保證前向訊號的變異數不為 0，作者實際檢查的只有反向梯度的大小。頁面把兩者都寫成「檢查過」，和來源不完全一致。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

對照 lesson_cases/03-comparison.py：stem 是一個 Conv2d(3,4,3,padding=1)，block 都是 stride 1、padding 1，全程 16×16，沒有分段下採樣；SGD lr=0.1，沒有 momentum。對照 ResNet 論文：§3.4 寫每個卷積後接 BN、用 He 初始化，SGD 用 momentum 0.9、weight decay 1e-4、lr 逐步除以 10；§4.1 寫「BN ensures forward propagated signals to have non-zero variances… backward propagated gradients exhibit healthy norms」，並推測深層 plain 是「exponentially low convergence rates」；Table 2 中 18 層 27.94 對 27.88、34 層 28.54 對 25.03，頁面敘述正確。PyTorch 預設初始化 ±1/√36、He 初始化 2/36、1/18 與 0.24 的推導都對；〈結果怎麼讀〉的交叉引用存在。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

用腳本比對 reviews/03-comparison.md 的發現與處理列數並讀了相關段落。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/03-comparison.md | 列了 19 項發現（2 項 must 的處理在修正後檢查裡有寫），其餘 17 項 should 的處理都沒列；紀錄寫「各項的處理見下方〈定稿修正〉」，但沒有這一節（fix:b1b 的 page 寫成「03-comparison」，生成器沒比對到）。 | 已修正：產生器改以頁名、節名、萬用字元、頁面上的圖與該節 notebook 把處理對應到頁面，〈定稿修正〉列出這一頁每一項的處理。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核 3 項建議、讀者 9 項、技術 7 項，都在〈定稿修正〉17 列處理（有合併列）；批次檢查留下的 2 項意見，由〈後續編輯的檢查〉核對；技術查核列出 ResNet 論文來源。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/03-comparison.md 第 13、127、166、177 行 | 殘句與內部標籤：「輸出放在其中的暫存副本」「**執行的指令（都在暫存副本 .../暫存副本）：**」；〈定稿修正〉來源欄是「reader+tech」「tech+先前查核留下的項目」。 | 來源欄改成中文。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：〈定稿修正〉來源欄已改成中文，屬實；但點名的路徑殘句仍在，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/03-comparison.md 第 13、127 行 | 處理寫「點名的路徑殘句…已清理」，但點名的「輸出放在其中的暫存副本」在第 13 行逐字還在；點名的「**執行的指令（都在暫存副本 .../暫存副本）：**」只變成「**執行的指令（都在暫存暫存副本）：**」，仍是殘句。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

第 4 輪第 1 項屬實：第 13、127 行兩段仍在。第 3 輪第 1 項不實，而且和第 4 輪的說明互相矛盾。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 3 輪第 1 項處理欄（第 210 行） | 處理欄寫「已修正：點名的文字已不在紀錄裡」，但點名的「輸出放在其中的暫存副本」仍一字不改地留在第 13 行；同一份紀錄第 4 輪第 1 項的說明也寫它仍在。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |

### 第 6 輪：上一輪的處理：有必要問題

第 3 輪 #1 說來源欄已改成中文，但〈定稿修正〉#15 的來源欄仍有英文的先前查核留下的項目。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 3 輪 #1 的處理欄（第 210 行）；〈定稿修正〉#15 的來源欄（第 177 行） | 處理欄寫「來源欄改成中文」，但 #15 的來源仍是「技術查核＋leftover」。原始的來源標籤是 tech+leftover：產生器的 origin 對照表只認單一來源，組合來源直接交給 cell()；而 leftover 的替換規則只在緊鄰中文時才生效，「＋」後面的 leftover 因此沒被換掉。第 3 輪發現點名的正是這一欄，當時顯示為「tech+先前查核留下的項目」。 | 已修正：組合來源裡的 leftover 也換成中文（技術查核＋先前查核留下的項目）。 |
