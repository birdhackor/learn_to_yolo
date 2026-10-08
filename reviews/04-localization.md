# 審查紀錄：單物件分類與定位

審查範圍：`docs/lessons/04-localization.md`、頁面上的圖（`docs/assets/diagrams/04-localization-learning.svg`、`docs/assets/diagrams/04-localization.svg`），以及 `lesson_cases/04-localization.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題。只有一個 should：`to_xyxy` 沒有說明。所有程式都在暫存副本執行，repo 內沒有執行任何程式。

1. 程式敘述都成立
- lesson_cases/04-localization.py：exit 0，stdout 和 artifacts/checks/curriculum/04-localization.json 逐字相同。
- 兩個 data-excerpt 區塊和程式原文是連續的行，只多了中文註解；註解內容也和程式相符（梯度不全為 0、SGD lr 0.1、targets 是正規化 cxcywh）。
- 以下數字都用程式算過，全部正確：參數數 4100／10／4222；輸入 64×64 時的形狀錯誤（1x4096 對 1024x4）；IoU 100/188；4×4 例的 IoU≈0.143；兩例的 MSE 都是 0.001953125；摺疊說明的估計 0.01855；σ(2)≈0.88。
- 照頁面做了練習：[10,6,22,18] 的 IoU＝1/3，cxcywh 是 [16,12,12,12]，MSE＝0.0087890625，和參考答案相同。
- 照「想自己重跑」執行 run_learning_extensions.py --section 04-localization（不加 --record）：exit 0，只寫出 artifacts/runs/learning/04-localization/ 底下的 report.json 和 learning.svg，網站用的紀錄和圖都沒有變。印出的 JSON 等於 report.json 去掉四串 *_history。
- notebook 第 3 格確實有「可選：40 步學習實驗」，列的是同樣三行。

2. 先前審查意見與受程式改動影響的段落都處理了
- 先前審查意見第 3 行：tag 已經是 lessons-v0.4.0。
- 先前審查意見第 84 行：權重段照提案重寫，不再先講動機再自己反駁。
- 先前審查意見第 154 行：「只畫總 loss」的補註已刪，改成分項讀圖。
- 先前審查意見第 160 行：limits／ResNet 的說明已刪；「validation_accuracy 是 null」保留；補上輸出位置和 display。
- 受程式改動影響的段落第 143、145、154、160 行都處理了；notebook 的 display 路徑本來就是 artifacts/runs。
- 全頁沒有修訂或審查口吻。「中文註解是本頁加的」是全站摘錄的固定寫法，不算。

3. 數字
- 頁面上的數字都來自現有紀錄，沒有這台 Mac 跑出的數字。
- 分項的定性說法，我用 Mac 跑出的分項紀錄核對過：
  - 前 3 次更新總 loss 降 0.0996，其中 5×框降 0.081，分類降 0.019。
  - 第 4 點之後，5×框最大約 0.0116。
  - 框 MSE 在 0.0003～0.0023 之間上下起伏，不是一路降。
  - 總 loss 嚴格遞減。
- Mac 和現有紀錄的總 loss 最大只差 1.2e-7，所以重錄之後這些說法仍會成立。editor 已列出等待重錄的值。
- 這件事有守衛：validate_curriculum_evidence.py 會對 04-localization-learning.json 檢查 is_current（紀錄是否對得上目前的程式）。腳本加 --record 時先寫網站 SVG、最後才寫紀錄，所以發布時網站上一定是兩格圖。

4. 摘錄
- 摘錄比對工具輸出 `docs/lessons/04-localization.md []`。
- validate_lessons 的摘錄斷言通過，最後停在 reviews 那一步（no review），這是發布前的預期狀態。
- 正文沒有引用程式行號。

5. 可讀性
- 段落順序和 01-small-cnn 一致：40 步實驗在練習前面，錨點都是 #forty-steps。
- seed、Adam、L2 長度、NaN、nn.functional／F 第一次出現時都有說明，只有 to_xyxy 沒有（見下表）。

6. 渲染
- 在「目前工作樹」和「HEAD 只換本頁」兩份副本上，zensical build --clean --strict 和 validate_site.py 都 exit 0。
- 輸出的 HTML 有一個表格、兩個 data-excerpt、id=forty-steps，兩個頁內連結都正常。
- docs/assets/diagrams/04-localization.svg 沒有改動，有 viewBox、title、desc，用 qlmanage 看過，座標和程式一致。
- 新程式產生的兩格圖用 qlmanage 看過，和讀圖段的描述一致：左圖藍虛線是總 loss、紫線是分類、紅線是 5×框；右圖是框 MSE，用自己的刻度。
- 這張圖由 save_svg 產生，有中文 title 和 aria-label，但沒有 desc。這屬於腳本那一份修改，不是本頁的事。
- 這份修改只動了 docs/lessons/04-localization.md，執行紀錄區塊和 HEAD 逐位元相同。repo 裡其他修改或 staged 刪除都不是這份修改造成的。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/04-localization.md 第 129 行（〈訓練 40 步〉第一段）「直接取用完整程式裡的 `Localizer` 與 `to_xyxy`」 | `to_xyxy` 是這次新加進頁面的名字，全頁只出現這一次，也沒有說明。`Localizer` 前面有摘錄，`to_xyxy` 卻沒有。初學讀者會不知道它是什麼，也不知道腳本為什麼要用它。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：工作目錄：暫存副本（用任務指定的 rsync 建立）；沒有在 repo 根目錄內執行或寫入任何東西，只讀了 .gitignore 與 .venv-model 裡的 PyTorch 原始碼。

查核的外部來源：
1. Adam：https://arxiv.org/pdf/1412.6980v9。§1 Introduction："The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients"；另查 Algorithm 1 的 m_t、v_t、偏差修正與 θ_t 更新式。頁面「依每個參數過去梯度的大小，自動調整每一步走多遠」與此相符。
2. PyTorch 2.9.1（已安裝版本，git_version 5811a8d7da873dd699ff6687092c225caffcf1bb）的 torch/nn/functional.py：mse_loss 在 L3815–3835，reduction 預設 "mean"，文件寫 "'mean': the mean of the output is taken."；cross_entropy 在 L3375–3382，reduction 預設 "mean"。用來確認「對 2×4＝8 個數平均」。
3. 兩端都含、寬要加 1 的舊慣例：https://github.com/facebookresearch/detectron2/blob/1e3e13bbf607b54f62205c4c33922521822fb298/docs/notes/compatibility.md L11–13（"width = x2 - x1 ... In Detectron, a \"+ 1\" was added both height and width."）；https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/datasets/voc_eval.py L168–175（`iw = np.maximum(ixmax - ixmin + 1., 0.)`）與 lib/datasets/pascal_voc.py L207–211（"# Make pixel indexes 0-based"）。

頁面本身沒有引用論文或 YOLO 官方程式。

讀過的 repo 檔案：
- 審查用的事實與寫作規範清單
- docs/lessons/04-localization.md（全文）
- lesson_cases/04-localization.py
- scripts/run_learning_extensions.py
- miniyolo/geometry.py（box_iou、cxcywh 轉換、restore 的 clamp）
- miniyolo/inference.py（decode_grid L26–27 把框截進圖內）
- miniyolo/targets.py L31（GT 超出圖片時拋出 ValueError）
- scripts/build_lesson_notebooks.py，以及 notebooks/04-localization.ipynb 的四格（環境格、「可選：40 步學習實驗」三行）
- scripts/validate_lessons.py 的摘錄比對邏輯、scripts/validate_curriculum_evidence.py
- 連結頁 docs/lessons/05-assignment.md、07-data.md（空圖）、07-inference.md、11-augmentation.md（clamp 與可見比例 0.5）、11-iou-loss.md
- zensical.toml 導覽標題、docs/glossary.md 框約定、.gitignore（artifacts/*.png 已忽略）

執行的程式與指令：
1. rsync 建立暫存副本。
2. 執行本節程式：`PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/04-localization.py`，exit 0。固定的輸出行（mean=0.0625、first_target、shape）和頁面一致，step 損失也與紀錄相同。
3. 摘錄比對印出 []；我也逐行人工比對，兩段摘錄除了中文註解外與原檔完全相同。
4. `scripts/run_learning_extensions.py --section 04-localization`，exit 0。檢查 report.json 與四串歷史值：起點 0.809883、終點 0.430213、IoU [0.6945, 0.7752]、預測框、parameters 4222；印出的 JSON 只少四串 *_history。
5. `qlmanage -t -s 1200` 轉出以下圖片並檢視：docs/assets/diagrams/04-localization.svg（另核對 SVG 原始座標：1 px＝10 單位，GT、pred、交集位置都精確）、新產生的 artifacts/runs/learning/04-localization/learning.svg（左右兩張小圖，圖例、顏色、標籤都與第 139 行描述相符）、repo 舊版 learning svg（舊的單線圖，屬待重繪，未列為問題），以及 artifacts/04-localization.png 疊圖。Playwright 因 firefox 未安裝而無法使用。
6. `.venv-docs/bin/zensical build --clean --strict`，exit 0，No issues found。解析輸出的 HTML，確認表格、4 個摺疊區塊、26 個數學式與 #forty-steps 錨點都正確。
7. 自寫的 probe 腳本（放在暫存副本的 04-localization-tech-probe/）：
   - step 0 時各項 loss 對參數的梯度 L2：分類 0.159，5×框 0.574，支持第 95 行的論點。
   - 框 head 改讀平均特徵（GAP）的對照：40 步 IoU [0.58, 0.39]，400 步 [0.9994, 0.9993]。
   - 黑底圖上平移與碰邊對平均特徵的影響：內部平移偶數格差 1.5e-8、離邊 2 px 差 0、碰邊差 6.7e-3、平移奇數格差 7.7e-3，與摺疊說明相符。
   - 三步實驗的精確值：框項降 0.0473、分類降 0.0050、總降 0.0523。
   - 輸入 64×64 時拋出 RuntimeError；輸入 33×33 不報錯。

手算覆核（全部正確）：cxcywh [10,12,12,12]、正規化 target 兩組、還原 [6,8,18,20]、MSE 0.001953125、IoU 100/188、4/28、72/216、(6/32)²/4、初始框 loss 0.01855、ln2、參數數 10／4100／20／4222、展平索引 4 與 7、σ(2)≈0.88。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈訓練 40 步〉第 129–145 行（特別是第 143 行的限制段與第 145 行的 IoU 解讀） | 兩張訓練圖的顏色和位置一一對應：紅色方塊在左上，藍色方塊在右下。所以只讀平均特徵（GAP）的框 head 也能靠顏色把兩個框背起來，頁面沒有說明這一點。讀者剛讀完「框 head 為什麼要讀展平後的特徵」，很容易把 IoU [0.6945, 0.7752] 當成「要展平才學得到框」的證據，但這個實驗其實分不出展平和平均。我在暫存副本用同樣的 seed 7、Adam(lr=0.01) 和 loss，把 box_head 換成讀平均特徵的 Linear(4,4)：40 步的 IoU 約為 [0.58, 0.39]，400 步約為 [0.9994, 0.9993]。這些是本機數字，只用來說明這樣做可行。 |
| 2 | 建議 | 〈核對、收益與代價〉第 121 行「class shape `[2,2]`、box shape `[2,4]`」 | 這句以「完整程式應輸出」開頭，但程式實際印出的是 `class_shape=(2, 2), box_shape=(2, 4)`，也就是 Python tuple 的圓括號，逗號後還有空格；暫存副本執行結果和頁尾紀錄都是這樣。數值相同，但初學者照頁面去找 `[2,2]`，會在輸出裡找不到。 |
| 3 | 建議 | 〈核對、收益與代價〉第 125 行「輸入尺寸改變也會讓展平的框 head 對不上」 | 這句寫成通則，但不是所有尺寸改變都會報錯。`MaxPool2d(2)` 預設捨去除不盡的部分（ceil_mode=False）：輸入 33×33 時，pooling 後仍是 16×16，展平後還是 1024 個數，所以程式不報錯，只是默默丟掉最後一列與一欄。暫存副本實測 `model(torch.zeros(1,3,33,33))` 回傳 shape (1,4)；輸入 64×64 才會出現 `mat1 and mat2 shapes cannot be multiplied (1x4096 and 1024x4)`。結論「不能只改圖片尺寸、不改模型」仍然成立。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 技術查核 | 兩張圖顏色不同，這個實驗分不出展平和平均哪個好 | 已修正：IoU 段末補一句：只讀平均特徵的框 head 也能靠顏色記住兩個框，所以本實驗不能證明展平比平均好 |
| 2 | 技術查核 | 輸出的 shape 格式和頁面寫法不同 | 已修正：照實際輸出寫成 class_shape=(2, 2)、box_shape=(2, 4)，並註明就是頁面寫的 [2,2]、[2,4] |
| 3 | 技術查核 | 「輸入尺寸改變就對不上」寫成了通則，但有例外 | 已修正：改成「pooling 後不是 16×16 就對不上」；補上 33×33 不報錯但會丟掉邊緣（已在暫存副本實測輸出 shape 為 (1,4)） |
| 4 | 先前查核 | to_xyxy 沒有說明 | 已修正：補上簡短說明：它把 cxcywh 換成 xyxy，腳本在 40 次更新後用它算出預測框（已對照程式與腳本） |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/03-identity.md`、`docs/lessons/03-projection.md`、`docs/lessons/03-comparison.md`、`docs/lessons/04-localization.md`、`docs/lessons/04-coordinates.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

用腳本比對發現與處理列數並讀了紀錄。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/04-localization.md | 4 項 should（第 1 次查核 1、技術查核 3）都沒有處理，指向的〈定稿修正〉不存在（原因同上）；另有「2. trace 與 impact 都處理了 - trace 第 3 行」。 | 已修正：產生器改以頁名、節名、萬用字元、頁面上的圖與該節 notebook 把處理對應到頁面，〈定稿修正〉列出這一頁每一項的處理。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核 1 項建議與技術查核 3 項都在〈定稿修正〉4 列處理（上一輪指出缺這一節，已補）；技術查核寫明頁面沒有引用論文或 YOLO 官方程式；批次檢查掛在本頁。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/04-localization.md 第 91 行 | 指令殘句：「3. `python3 摘錄比對工具 <暫存副本> docs/lessons/04-localization.md`」，已經不是能執行的指令。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 90 行指令改成「摘錄比對印出 []」，處理說明屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/04-localization.md 第 88–89 行 | 「執行的程式與指令：」底下的清單從「2.」開始，原本的第 1 項刪掉後沒有重新編號。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

第 3 輪第 1 項屬實。第 4 輪第 1 項的「未改」屬實：第 88–89 行的清單仍從「2.」開始。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 4 輪第 1 項處理欄（第 150 行） | 發現點名的是「執行的程式與指令：」底下的清單從「2.」開始。處理欄只寫「點名的 1 處文字仍在紀錄裡：「執行的程式與指令：」」，略過了真正的問題：編號從 2. 起。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |

## 紀錄重產後的檢查

2026-10-05，由另一位 AI 在獨立 rsync 副本 `/tmp/lessons-v0.4.0-reviews/review-04/` 查核 `docs/lessons/04-localization.md` 全文、兩張頁面 SVG、三步案例及 40 步補充實驗。已讀 `AGENTS.md`、發布步驟第 7 項及原審查紀錄。結論：通過，沒有必要問題，也沒有新增建議；正文目前的六位小數 loss、兩位小數框座標與四位小數 IoU 都正確，不需要修改。

### 實際執行與對照

- 使用原工作區 `.venv-model/bin/python`，但 cwd 與 `PYTHONPATH=.` 都在獨立副本。執行 `lesson_cases/04-localization.py` 與 `scripts/run_learning_extensions.py --section 04-localization`，均 exit 0；案例的每一行 stdout 與現行 `04-localization.json` 相同，補充實驗完整 `report.json` 與現行 `04-localization-learning.json` 完全相同，包括四串各 40 點的歷史值。沒有使用 `--record`。補充實驗重畫的 `learning.svg` 和網站 SVG 逐 byte 相同，三步疊圖也和副本建立前的 PNG 逐 byte 相同。
- 逐句對照頁面、案例、補充腳本與其紀錄所綁定的 12 個 repo 檔案；全部 SHA-256 與紀錄相符。兩個程式摘錄去掉中文註解後，都是案例原文的連續行。notebook 四格已讀，最後一格程式與案例相同、stdout 與現行紀錄相同；環境格固定 `lessons-v0.4.0`，可選實驗的三行及 `artifacts/runs/learning/04-localization/learning.svg` 路徑與正文相同。本次不執行會下載遠端程式的環境格。
- 使用原工作區 `.venv-docs/bin/zensical build --clean --strict` 建置獨立副本，exit 0、`No issues found`；在副本執行 `python3 scripts/validate_site.py`，exit 0，60 頁及 42 課的本地連結、錨點、Colab 配對、Markdown 轉換與資產檢查通過。
- Playwright 使用 `/usr/bin/chromium`（151.0.7922.173）及 `--no-sandbox --disable-gpu --disable-dev-shm-usage`，只讀副本網站。實際打開本頁並展開四個摺疊說明：26 個公式全部轉換成 MathJax、未轉換公式 0，兩張圖載入、兩個摘錄、一個表格及 `#forty-steps` 正常，browser page error 0。桌面 1440 px、手機 375／320 px 都沒有整頁橫向溢出，寬公式、程式與表格可在自身容器內橫向捲動；SVG 依閱讀欄縮放。

### 手算、圖與結論

- 半開區間、x／y 方向、xyxy→cxcywh→正規化與反向還原都正確：紅框 `[4,6,16,18]` 得 `[10,12,12,12]` px 與 `[0.3125,0.375,0.375,0.375]`；藍框得 `[0.6875,0.5,0.375,0.375]`；人工預測的還原是 `[6,8,18,20]`。越界例的左界為 −0.10，即 −3.2 px。
- 人工大框的交集 100、聯集 188、IoU＝25/47≈0.532；4×4 小框為 4/28＝1/7≈0.14，兩者正規化 MSE 都是 1/512＝0.001953125。已照自主練習獨立計算右移 6 px：交集 72、聯集 216、IoU＝1/3；中心為 `[16,12]`，MSE＝9/1024＝0.0087890625，與答案一致。
- 初始估算 MSE＝19/1024＝0.0185546875，ln 2＝0.693147、sigmoid(2)＝0.880797；GAP 手例均為 1/16，展平索引為 4 與 7。手算並用參數 tensor 數覆核：分類 head 10、框 head 4100、平均後框 head 20、backbone 112、整個模型 4222。實際輸入 64×64 出現 `1x4096 and 1024x4` 形狀錯誤；33×33 輸出 `[1,4]`，與文中例外相符。
- 重算三步精確 loss，前兩次更新的總降幅為 0.0523132086，其中分類降 0.0050376654，5×框項降 0.0472755823；step=0 框項為 0.0925453473，約占總 loss 11%。正文的量尺、39 倍、梯度權重、更新前量測及近似數字均成立。
- 40 步的第 1 點與三步 step=0 相同；第 40 點是在第 40 次更新之前。六位小數確實是 `0.809883 → 0.430213`。前 3 次更新總降 0.0995911956，分類降 0.0186439753，框項降 0.0809472124；全段總降 0.3796699047，分類降 0.2929784656，框項降 0.0866914558。起始框項即使全降到 0，也只能解釋全段降幅的 24.38%，所以「全段大部分來自分類」成立。總 loss 嚴格下降、框 MSE 有上下起伏，每點的 `total ≈ classification + 5×box` 都成立。
- 從模型預測 xyxy 以獨立的純算術交集／聯集公式計算兩個 IoU，得 0.6944893537、0.7752413204，與 float32 紀錄相符，四位小數為 `[0.6945,0.7752]`。兩位小數的框仍是 `[4.48,4.77,18.13,18.93]`、`[16.33,10.58,29.18,23.12]`。兩張圖的類別都正確，所以 accuracy 1.00 成立；validation 為 null，沒有獨立評估。
- 已實際檢視 Chromium 畫出的兩張 SVG 與三步疊圖。人工 IoU 圖按 1 px＝10 SVG 單位繪製，GT、pred、100 px² 交集的原始 rect 座標與手算精確一致。學習圖的左圖確實是藍虛線 total、紫色 classification、紅色 5×box；右圖是未加權框 MSE、獨立刻度。由 SVG 刻度反推出資料轉換，逐點核對四條路徑的 160 個點，最大誤差僅 3.1×10⁻⁶ SVG 單位。圖中文字（計入旋轉變換後）全部在 viewBox 內，沒有裁切。
- 初學閱讀順序與首次術語說明足夠。原審查指出的 `to_xyxy`、shape tuple、33×33 例外及「不同顏色可讓 GAP 記住兩個框」都仍有正確處理。頁面區分人工固定框、三步 smoke test 與 40 步訓練圖結果；明說步數與 optimizer 同時變動、沒有驗證圖、不能證明展平比 GAP 好、不能推論真實圖片或更深架構，沒有超出實驗支持範圍的結論。

### 本次查閱的原始來源與版本

本頁沒有引用原始論文或特定 YOLO 官方程式。函式庫說法以已安裝的 PyTorch `2.9.1+cpu` 官方原始碼隨附文件核對，git version `5811a8d7da873dd699ff6687092c225caffcf1bb`：`torch/nn/functional.py` 的 `cross_entropy`（3375）與 `mse_loss`（3815）、`torch/nn/modules/loss.py` 的 `MSELoss`（568，明說 mean 對全部元素平均）、`linear.py` 的 `Linear`（53）、`conv.py` 的 `Conv2d`、`pooling.py` 的 `MaxPool2d`（157，預設 floor 與 stride）、`activation.py` 的 `Sigmoid`（335），以及 `torch/optim/adam.py` 的 `Adam`（34，逐參數的一／二階梯度矩估計與更新式）。括號是此固定安裝版本的起始行，沒有用先前審查代替本次開啟來源。

另對照本 repo 的 `miniyolo/inference.py` 預測框 clamp、`miniyolo/targets.py` 不合法 GT 報錯及 `lesson_cases/11-augmentation.py` 裁切後的可見比例規則，並查閱連結頁 `05-assignment.md`、`07-data.md`、`07-inference.md`、`11-augmentation.md`、`11-iou-loss.md` 的相關段落，與本頁延伸閱讀敘述相符。

### 本次檢查的 SHA-256

以下是實際查核的副本原始檔 bytes；不以它們取代 `reviews/coverage.json` 的涵蓋寫入。逐點計算與瀏覽器結果另存在副本的 `artifacts/runs/review-04/numeric-report.json`、`render-report.json`；截圖同目錄。原審查與 coverage 沒有改寫。

```text
lesson_cases/04-localization.py  08848d3095ceb88d3c487caccb31672901bb8d45b20fcea720155473dc06b0f7
miniyolo/__init__.py  785b058b2b011124243b06ee59df8e59dfa75b77f050b897479e5be636763fc9
miniyolo/data.py  cccad00e2c4f96eb6567eafc9e12248379c6b715fc1790d75518a253baa6181d
miniyolo/figures.py  155222c5bb0d6942bcc231d5c32c6f1a4db662293c8cf909560f1f9b75a3ea15
miniyolo/geometry.py  6a6b57d3493888e99dae4a012dab78127b8b543a0e01d8107963a3d6e63d8483
miniyolo/inference.py  995ac8f942d0c1e43d94f2efb3b7adcb91a9feb6b9bab923615209f0c190c313
miniyolo/losses.py  81fa9331c9a2aebe2e9c6c453a5313e64566bb20c3fe77ca45ec9df53424d4e4
miniyolo/metrics.py  53ce982e37cbd96c784f75c7d30faf99d52f79ab83ca7b8114eb21b4327330e0
miniyolo/models.py  49d029ea4ba2650ce8933cf97e3d25dc7aff2ca4e972eec19cdda17b0f4900e6
miniyolo/provenance.py  21769813c36b45f513c9f0d6125194f3baa5ae265278ffd44a796d298d124ed3
miniyolo/targets.py  2c8e32f2845b3bf970c77304c5cca0f999083d36a4d5af2f75f14aa89b823f16
scripts/run_learning_extensions.py  b84e3317f95374df87d759e562db9b100ee14628b9cf5bb111238cd4a2e8e6cf
docs/lessons/04-localization.md  a9c3bb8e94cb38cb079e0f0e2bae3884b688187ed5f1566c0652c268fa09aace
docs/assets/diagrams/04-localization.svg  a4d35a3b11a2cae489f45a232a0343c73bfbb09aabbd1be4acacca9f567b700e
docs/assets/diagrams/04-localization-learning.svg  dc8c7a1d69ef0a22a6e8a399bed8349ddd5eacd7cef1f57a45c00cd2de388baf
artifacts/checks/curriculum/04-localization.json  c8b66584a63a8d37b47bcad567442c62a8d21b7c8730fe859af4cab42d734157
artifacts/checks/curriculum/04-localization-learning.json  2896307a4c01b6701d4ddd264562d2a329fe77c2d1bcad8b347d9481a8ab5a39
notebooks/04-localization.ipynb  f0cf213fe6db7c238a75c57449e77468d6508e28c4899dcc2b2d89ee4214dd02
reviews/04-localization.md  7963071cf8430fffa58965168dcc1d5026d86316881de784f4342c4573e97a91
docs/assets/stylesheets/extra.css  bd285e60fca6236ecca730a05c9ebe0320645bd2d6261a973a97dcf8f123aca6
docs/assets/javascripts/mathjax.js  d52c29c2d786580e820ccad837cf1d6fecbbc4ce89610a0704095f96caf998c6
```


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/foundations.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `detector` 實讀本頁 8 個凍結單元；首次使用／前文方法範圍四題位置為 04-localization/00:first_use, 04-localization/01:first_use, 04-localization/02:first_use, 04-localization/03:first_use, 04-localization/04:first_use，頁末為 04-localization/07。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/detector.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/detector-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/detector.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R007、R039、R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-foundations.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。00 的 BN 首次命名及 04 的最後公式／紀錄變更另由偵測與演進技術報告補核。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS01。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 b 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁相關處置：DEC-012、DEC-014、DEC-026；包含採用、保留或後文撤回的來源與理解收益。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `b_foundation` 實讀本頁 9 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/b_foundation/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：R025。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/04-localization.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
