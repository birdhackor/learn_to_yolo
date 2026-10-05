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
