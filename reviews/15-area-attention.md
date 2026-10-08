# 審查紀錄：YOLOv12 Area Attention

審查範圍：`docs/lessons/15-area-attention.md`、頁面上的圖（`docs/assets/diagrams/15-area-layout.svg`），以及 `lesson_cases/15-area-attention.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。六項檢查都沒有必要或建議等級的問題。所有程式都在暫存副本執行，沒有在 repo 裡跑任何東西。

1. 頁面對程式的敘述都正確。
- 原程式實跑 exit 0。印出的六行，包括 `first-token output full=0.46875, area=0.09375` 與 `after intervention: full=0.78125, area=0.09375`，都和頁面的清單一致，清單也涵蓋了全部六行。`changing token 15…` 那一行依序印出 full_affected、area_affected、same_area，頁面寫的順序是對的。
- 三題練習都照頁面指示，從原程式只改一行來跑：
  - areas=2：印出 `(2, 8, 8) 128`、area=0.21875、area=False。
  - areas=1：印出 256，0.46875→0.78125，`same area=True`。
  - changed_index=3：印出 full=0.78125、area=1.34375。
  - assert 全部通過，結果和參考答案一致。
- 第一個程式框：我把它原樣包進函式，用 areas=1／2／4 各跑一次，output 與 weights 都和 `attend` 的回傳值 torch.equal 相同。所以「依 `attend` 改寫的簡化版」這個說法成立。框前列的三個差異（大寫 B、N、C、省略 assert、只存 output）也都屬實。
- 摘錄框裡的 `...`：略去的正是程式中算 full_affected、area_affected 的那兩行，前後行在程式裡是連續的。

2. 先前審查意見與受程式改動影響的段落都處理了。
- trace line 3：頁首在 HEAD 就已經是 lessons-v0.4.0，validate_lessons 的 source_ref 檢查也通過。
- trace line 153 與受程式改動影響的段落 114、116、149、153：取偶數的說明、「程式印成 0.2188」、「不是筆誤」、指向 4.2 `round()` 的連結都刪掉了。我在全頁找 0.4688、0.7812、0.2188、1.3438、四位、取偶、正中間、筆誤，只剩頁尾自動產生的區塊（第 171、173 行）。
- impact 161、168、170 在頁尾，沒有動過，和 HEAD 逐位元相同，之後由 verify_curriculum.py 重產。
- 全頁沒有修訂、審查或製作經過的敘述。

3. 數字都對，也不受機器影響。新寫入的 0.46875、0.09375、0.78125、0.21875、1.34375 都是 1/32 的倍數，和手算、實跑結果相同，換哪台機器跑都一樣。正文沒有寫進 loss 或計時這類會因機器而變的數字。要等重產的值（頁尾的日期、版本與兩行輸出）都已列在修正者的待重錄數值清單。

4. 摘錄檢查通過。
- 摘錄比對工具對 repo 和暫存副本都印出 `[]`。
- 突變測試：把 `==` 改成 `!=`、把 `shape[1]` 改成 `shape[0]`，兩種都被抓到；拿掉 data-excerpt 標記、改成一般的 ```python，則被判為未標記的逐字摘錄。
- 第一個框沒有標記，框前已說明它是簡化版，檢查也不會把它誤判成逐字摘錄。
- 正文沒有引用程式行號。

5. 可讀性沒有問題。新加的兩句框前說明，寫法和其他頁一致。`...` 的意思，以及 full_affected、area_affected、same_area 這三個名字，都在使用前解釋過。刪掉取偶數說明不影響教學順序。

6. 渲染沒有問題。
- 暫存副本中 `zensical build --clean --strict` exit 0（No issues found），validate_site.py 也是 exit 0。建出的 HTML 裡，摘錄框是正常的 language-python 高亮區塊。
- SVG 和 HEAD 相同，有 viewBox、<title>、<desc>。用 qlmanage 算圖，畫面乾淨：A=4 是四條帶，A=2 是兩段，紅框標 token0，紫框標 token3 和 token15，都和程式一致。
- 本節的相關檔案中，只有頁面和 HEAD 不同；git diff --check 沒有錯誤。

以下備註不影響通過，屬於發布流程，不是這次修改的問題：
- 頁尾與 artifacts/checks/curriculum/15-area-attention.json 的 stdout 還是舊的 0.4688、0.7812，notebook 最後一格目前也沒有存輸出。要等紀錄重產後才會和正文一致，所以重產必須和本頁一起發布。
- 在暫存副本整份跑 validate_lessons.py：notebook 與程式逐字相同、全站摘錄檢查都通過，只在「審查紀錄」那一步失敗，而且全站每一頁都一樣是 no review。發布前要針對本頁目前的文字重審 reviews/15-area-attention.md。
- areas=16 會在 grad 的 assert 失敗，HEAD 也一樣。頁面沒有要讀者試這個值，所以沒有任何敘述因此變成錯的。

佐證檔：
- 實跑輸出
- 練習與簡化框測試
- 建置與驗證紀錄：暫存副本、.validate-site.log、.validate-lessons.log
- SVG 算圖

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py（Bottleneck、C3、C2f、C3k、C3k2）
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/conv.py（Conv：bias=False、BatchNorm2d、default_act=SiLU）
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/11/yolo11.yaml（backbone/head 的 C3k2、SPPF、C2PSA）
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/tasks.py（parse_model：m/l/x 尺度把 C3k2 的 c3k 強制設為 True）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/block.py（AAttn、ABlock、A2C2f）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/conv.py（Conv）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/cfg/models/v12/yolov12.yaml（第 6 層 A2C2f area=4、第 8 層 P5/32 area=1，head 的 A2C2f a2=False）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/tasks.py（parse_model：A2C2f 的 n 插在 args[2]）
- https://arxiv.org/abs/2502.12524（Submission history：只有 v1）
- https://arxiv.org/html/2502.12524 §3.2 Area Attention（分成 l 段，(H/l,W) 或 (H,W/l)；只需一次 reshape、速度較快；預設 l=4，感受野變 1/4 但仍然夠大）、§3.4 Architectural Improvements（移除 positional encoding，改用 7×7 large separable convolution『position perceiver』）
- https://arxiv.org/abs/1706.03762 → https://arxiv.org/html/1706.03762 §3.2.1 Scaled Dot-Product Attention 正文（We suspect…）與腳註（變異數 d_k）
- https://ar5iv.labs.arxiv.org/html/1512.03385 §4.1 Deeper Bottleneck Architectures（1×1、3×3、1×1 三層）

這一頁沒有發現與來源不符的說法。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」。核對釘選的 yolov12 block.py：AAttn 的 `self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)`，與正文「程式版本是 5×5」一致。N²/A 的推導支持術語表 attention 列的說法。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：獨立查核通過，沒有待處理的建議；〈來源對照〉列出 YOLOv12 論文與官方程式，沒有不符；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/15-area-attention.md 第 50–53 行 | 空洞的路徑清單：「- 實跑輸出：.../暫存副本」「- 練習與簡化框測試：.../暫存副本」「- 建置與驗證紀錄：.../暫存副本、.validate-site.log、.validate-lessons.log」「- SVG 算圖：.../暫存副本」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 50、51、53 行已改成類別標籤；但第 52 行點名的殘句沒有清理，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/15-area-attention.md 第 52 行 | 點名的「- 建置與驗證紀錄：.../暫存副本、.validate-site.log、.validate-lessons.log」只拿掉「.../」，變成「- 建置與驗證紀錄：暫存副本、.validate-site.log、.validate-lessons.log」，仍是殘缺的檔名清單。處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[modern當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/modern.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_b` 實讀本頁 5 個凍結單元；首次使用／前文方法範圍四題位置為 15-area-attention/00:first_use，頁末為 15-area-attention/04。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_b.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_b-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_b.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS04。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 c2 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `c_modern` 實讀本頁 8 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/c_modern/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：無額外局部修正。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/15-area-attention.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
