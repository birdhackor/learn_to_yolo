# 審查紀錄：Grid MiniYOLO loss

審查範圍：`docs/lessons/07-loss.md`，以及 `lesson_cases/07-loss.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要，也沒有建議。所有程式都只在暫存副本裡執行，repo 沒有動。查核用的腳本和 log 在同目錄的暫存副本（mutate.py、nums_check.py、extra_check.py、build.log、validate_site.log、validate_lessons.log）。

1. 頁面對程式的敘述都正確
   - lesson 執行 exit 0，輸出四行，和〈執行與核對〉的四條依序對得上，第 3 行是 `positive-cell class gradients [-0.5, 0.5]`。
   - 頁面說三種錯「每一種都會讓其中一個斷言失敗」。我只改暫存副本的 miniyolo/losses.py 做突變測試，每次跑完都還原並用 cmp 確認：
     - 先 sigmoid 再傳給 BCE：objectness 變成 0.942827，程式停在「objectness 等於 ln 2」的斷言。
     - 先 softmax 再傳給 CE：類別 loss 仍是 0.693147，梯度變成 [-0.25, 0.25]，程式停在正格類別梯度的斷言。
     - 負格也送進 CE：類別 loss 仍是 0.693147，程式停在「負格類別梯度為 0」的斷言。
     - 拿掉空圖分支：程式停在「空圖 box/class 等於 0」的斷言。這時 loss 是 NaN，但梯度全是有限值，和頁面說的「壞掉的是 loss 值」一致。
     - class_ids 填 −1 再送進 CE，確實會報錯（IndexError）。
   - 練習 3：我從頁面直接抽出程式，照頁面說的存成 .py，在 repo 根目錄用 PYTHONPATH=. 執行。exit 0，印出的四項 loss 和參考答案一致。
   - 照最後一段改成 [scene, scene]，並改那三個斷言，也是 exit 0。類別斷言如果沒有改成 [-.25, .25] 就會失敗，所以這個變體真的能核對練習 1。
   - 其他實測都相符：練習 1 每個正格的類別梯度是 [-0.25, 0.25]，框梯度減半；練習 2 加 10 之後類別梯度仍是 [-0.5, 0.5]；負格的 class_ids 預設確實是 0。
   - notebook 的環境格會切到 repo 目錄（chdir），並把它加進 sys.path，所以在最後新增的儲存格可以 import miniyolo。

2. 數字
   - 新增的數字都是確定值，沒有從這台 Mac 帶進會隨機器改變的數字：數值驗算的 0.692647／0.693647、softmax([.001, 0]) 兩數相差約 0.0005（雙重 softmax 的 CE 是 0.692897）。
   - 要等紀錄重產才會吻合的值，編輯都已列出。

3. 先前審查意見與受程式改動影響的段落項目全部處理完
   - 兩條先前審查意見：頁首按鈕不手改；正文重複的 Colab 連結改成「用頁首的按鈕」。
   - 總評的兩條：程式行號引用、BCEWithLogitsLoss 的寫法。
   - 受程式改動影響的段落第 5、134、185、196、199、217、286 行各條。
   - 沒有留下「程式抓不到」這類過時說明。全頁沒有修訂或審查的敘事，正文也沒有程式行號。

4. 程式摘錄
   - 摘錄比對工具在 repo 和暫存副本都輸出 []；我故意改壞摘錄一個字元，會被抓到。
   - grid_loss 區塊已寫明是簡化版。練習程式不是逐字摘錄，不需要標記。

5. 渲染
   - zensical build --clean --strict 沒有任何問題，validate_site.py 通過（連結與錨點、Colab 配對、數學、摺疊區塊都過）。
   - 新增的推導 note 裡，列表和數學式都正確轉成 HTML；標了 data-excerpt 的區塊照常以 Python 高亮顯示。
   - 本頁沒有 SVG。
   - validate_lessons 只在審查涵蓋檢查失敗，因為 reviews 還沒做，這在預期之內。前面的 notebook／程式一致檢查和摘錄檢查都過了。
   - 和 07-loss 有關的檔案（lesson code、notebook、紀錄 JSON、reviews）都和 HEAD 相同，只有本頁被改。

順帶兩點，都不算頁面問題：
- 編輯疑慮裡說「notebooks/07-loss.ipynb 也還對應舊程式」，這不精確。notebook 最後一格已經和目前的 lesson_cases/07-loss.py 逐字相同（validate_lessons 的比對已通過）。過期的只有 artifacts/checks/curriculum/07-loss.json（case_sha256 仍是 de8c…）和頁尾的紀錄區塊。
- 「除以 Npositive」確實沒有人守：CE 改成 reduction='sum' 時，完整程式照樣通過，我重現了。頁面沒有說程式守著這一點，而讀者跑練習的 [scene, scene] 變體就會抓到。要不要補一個測試，屬於程式範圍，需要另外決定。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://arxiv.org/abs/1506.02640（YOLOv1，以 ar5iv HTML https://ar5iv.labs.arxiv.org/html/1506.02640 讀全文，即 arXiv 最新版 v5）：§2 Unified Detection（中心所在格負責；每格 B 個框與 confidence=Pr(Object)*IOU；沒有物件時 confidence 為 0、有物件時等於 IOU；每格一組條件類別機率；Pascal VOC 的設定 S=7、B=2、7×7×30）、式 (1)（測試時把類別條件機率乘上 confidence）、§2.1（最後一層用線性輸出）、§2.2 Training（xy 以格為基準、wh 以整張圖正規化，都落在 0 到 1；sum-squared error；λcoord=5、λnoobj=.5 的理由；預測寬高的平方根；由當下 IOU 最高的預測器負責）
- https://arxiv.org/abs/2004.10934（YOLOv4，以 ar5iv HTML https://ar5iv.labs.arxiv.org/html/2004.10934 讀全文）：§3.4〈Eliminate grid sensitivity〉，sigmoid 乘上大於 1 的係數（07-targets 提到這件事，但沒有附連結）
- https://pytorch.org/docs/stable/generated/torch.optim.Adam.html（轉址到 https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html）：演算法框（m_t、v_t、偏差修正、θ_t 更新式）與 betas 參數說明「running averages of gradient and its square」，預設值 (0.9, 0.999)
- https://github.com/cocodataset/cocoapi 的 commit 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9，PythonAPI/pycocotools/cocoeval.py：第 506–508 行（iouThrs 是 .50:.05:.95 共 10 個門檻，recThrs 是 101 點）、第 370–376 行（沒有真值的類別直接跳過，precision 留在 -1）、第 396–410 行（先取 precision 包絡線，再在 101 個 recall 位置取值）、第 452–455 行（取平均時排除 -1）
- 本機 PyTorch 2.9.1（.venv-model）實測函式庫行為：cross_entropy 的 target 為 -1 時報 IndexError；類別放在最後一軸時報 RuntimeError（shape 不合）；mse_loss 與 cross_entropy 對空 tensor 取平均得到 NaN；torch.stack 的錯誤訊息開頭；對常數 0 呼叫 backward 會報錯；float32 的 sigmoid(-18)≈1.52e-8，sigmoid(-89)=0；24 附近的 ulp 是 1.9e-6，所以 24±w/2 會捨入成同一個數；inference_mode 產生的 tensor 不能拿去算梯度；squeeze 會刪掉所有長度為 1 的軸

這一頁沒有發現與來源不符的說法。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：獨立查核通過，沒有待處理的建議，所以沒有〈定稿修正〉；〈來源對照〉列出來源；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/07-loss.md 第 11、32 行 | 內部名稱：「查核用的腳本和 log 在同目錄的暫存副本（mutate.py、nums_check.py…）」「verdict 的兩條：程式行號引用、BCEWithLogitsLoss 的寫法」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 32 行的總評已改成「總評」；但點名的路徑殘句沒改，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/07-loss.md 第 11 行 | 點名的「查核用的腳本和 log 在同目錄的暫存副本（mutate.py、nums_check.py…）」逐字還在，處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

第 4 輪第 1 項屬實。第 3 輪第 1 項不實，而且和第 4 輪的說明互相矛盾。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 3 輪第 1 項處理欄（第 73 行） | 處理欄寫「已修正：點名的文字已不在紀錄裡」，但第 11 行仍是「查核用的腳本和 log 在同目錄的暫存副本（mutate.py、nums_check.py、extra_check.py、…）」，正是發現用「…」省略引用的那一句；已改寫的只有第 32 行（「verdict 的兩條」改成「總評的兩條」）。第 4 輪第 1 項的說明也寫這句仍在。看來產生器把引文裡的「…」當成了字面字元比對。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[grid當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/grid.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/grid.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/grid.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `detector` 實讀本頁 7 個凍結單元；首次使用／前文方法範圍四題位置為 07-loss/00:scope_overview, 07-loss/01:first_use, 07-loss/02:first_use, 07-loss/03:first_use，頁末為 07-loss/06。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/detector.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/detector-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/detector.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁未有需要改寫的已裁定問題，保留原教學內容；仍完整重讀與核對。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 b 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `b_grid` 實讀本頁 11 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/b_grid/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：R019。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/07-loss.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。


## 2026-10-08：章節字母前綴統一

本次僅將H1節號補上所屬B／C／D／E，並同步側邊目錄與閱讀路線。正文（第一行之後）、章號、路徑、圖、程式與實驗設定未變；沒有重新冒稱概念首讀。來源逐頁比對與標題／導航實頁檢查另存於[格式核對](formatting/section-prefixes-2026-10-08/source-check.json)。
