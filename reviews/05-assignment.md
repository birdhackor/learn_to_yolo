# 審查紀錄：多物件輸出與責任分配

審查範圍：`docs/lessons/05-assignment.md`、頁面上的圖（`docs/assets/diagrams/05-assignment.svg`），以及 `lesson_cases/05-assignment.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要等級的問題，只有一條建議。

我在自己的暫存副本驗證。所有輸出和腳本都在同一層的暫存副本。repo 裡我只照查核指示單跑了唯讀的摘錄比對工具。

1. 程式敘述（must，通過）
- 程式和 HEAD 相同。跑一次 exit 0，印出 8 行，第 4 行是 asymmetric boxes：正格 [[0,0],[1,2]]，target 依序是 [.75,.625,.3125,.1875] 和 [.75,.25,.25,.25]，類別 [0,1]。核對清單的 8 項逐項與輸出相符。
- 以下各項和程式逐一相符：build 的標註斷言、回傳順序、main() 的變數名、主例的定義、t[positive] 的排序、三次 build 呼叫的用途。
- 我實跑確認的行為：
  - 只拿掉兩個 [positive] 時報 RuntimeError size mismatch；攤平成 [32,2]／[32] 時報 IndexError 'Target -1 is out of bounds'。
  - 整批都是空圖時，mse 和 cross_entropy 都是 NaN。
  - objectness 梯度公式誤差為 0。
  - 遮罩取值的順序是先依圖片、再逐列由左到右。
- 練習照頁面做：先在 notebook 式的命名空間執行完整程式，再貼上參考答案那一格，所有斷言都通過。
- 突變測試，每一種都在頁面說的那道斷言失敗：
  - 算格子時 row／col 寫反、x／y 偏移寫反、寬／高寫反：失敗在 odd 的兩道斷言。
  - box target 存到 [col,row]、類別存到 [col,row]：失敗在 odd 的斷言。
  - 寬高除以 16：失敗在 box_targets[0,0,0] 的斷言。
  - 負格也學框：失敗在框梯度斷言；負格也學類別：失敗在類別梯度斷言，若用 -1 則先報 IndexError。
  - 拿掉碰撞檢查：丟出 AssertionError。
- 碰撞檢查本身把 row／col 寫反時，沒有任何斷言抓得到（editor 已登記這個缺口）。頁面沒有宣稱涵蓋它。
- S=8 的三種情境：
  - 照頁面只在主例呼叫寫 grid=8，並改好 head 和答案：exit 0，印出 [[1,1],[5,5]]、[.5,.5,.25,.25]、62、126，不對稱框和碰撞兩行不變。
  - 改成 build 的預設值：先在 odd_cells 斷言失敗，兩框落在 [[1,1],[2,5]]。
  - 再把 odd 的答案也改成 S=8 的值：碰撞測試丟出 AssertionError。
- 跨頁的說法都核對過：4.2 用 torch.allclose（0.8 存不精確）、第 9 章 obj_target 是 0 但用 valid 遮罩排除且門檻 0.2、train.py 每步重建 target、第 4 章框 loss 乘 5、第 7 章空正格回傳仍連著計算圖的 0、首頁的 (列=1,欄=1) 和格線約定。

2. 先前審查意見與受程式改動影響的段落條目（must，通過）
- 六條先前審查意見都處理了。L157、L190 的 patch note 已刪，改寫成「由 target 算出」的說明。不改 build 預設值那段仍然需要，因為程式沒有改成明確傳 grid=4。
- 受程式改動影響的段落中假設練習藍框的舊條目（159、166、182 等）照指示沒有套用，其餘都用 box 部分的新值。
- 全頁沒有本版、修訂、新增這類製作或修訂敘述。

3. 數字（must，通過）
- 確定值全部和程式相符。loss 數字 0.0871→0.0861、0.7031→0.6839、0.5235→0.4744 取自既有紀錄，不是這台 Mac 跑出來的。
- 待重產紀錄後要核對的值，editor 都已列出：頁尾執行紀錄區塊仍是舊的 7 行（2026-10-02），屬於預期中的暫時狀態。

4. 摘錄（must，通過）
- 摘錄比對工具對 repo 和暫存副本都印出 []。
- 兩個逐字摘錄都有 data-excerpt 標記，並說明了中文註解是頁面加的，以及 ... 省略的是哪兩行。
- 參考答案那格是讀者自己的程式，不是摘錄。正文沒有程式行號。

5. 可讀性
- 順序仍然能教，新詞大多有解釋。只有「不對稱的框」缺一句定義，見下表。

6. 渲染（must，通過）
- 在暫存副本跑 zensical build --clean --strict 和 validate_site 都通過。用 headless Chrome 把摺疊區塊展開後截圖，排版正常。
- 05-assignment.svg 沒有改動。qlmanage 渲染正常，有 viewBox／title／desc，內容與程式一致（正格 2、負格 14、ignore 0）。
- 這位 editor 只改了 docs/lessons/05-assignment.md。程式、notebook、紀錄、SVG、reviews 都沒有變。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/05-assignment.md 第 24 行（「再用兩個不對稱的框核對…」），以及第 143 行第一次正式介紹「兩個不對稱的框」處 | 「不對稱的框」在第 24 行就出現，但全頁沒有一句明確定義，第 143 行也只是拿來和「主例兩框的中心 x＝y、寬＝高」對照。初學者很容易把「不對稱」讀成「不是正方形」，可是第二框 [36,12,52,28] 剛好是 16×16 的正方形，要到摺疊說明裡的「它寬＝高」才會發現理解錯了。這只會讓讀者短暫困惑，不會讓他照著做錯，所以列為建議。這個詞本身應該保留，因為它對應輸出裡的 asymmetric boxes。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://arxiv.org/abs/1506.02640（YOLOv1，以 ar5iv HTML https://ar5iv.labs.arxiv.org/html/1506.02640 讀全文，即 arXiv 最新版 v5）：§2 Unified Detection（中心所在格負責；每格 B 個框與 confidence=Pr(Object)*IOU；沒有物件時 confidence 為 0、有物件時等於 IOU；每格一組條件類別機率；Pascal VOC 的設定 S=7、B=2、7×7×30）、式 (1)（測試時把類別條件機率乘上 confidence）、§2.1（最後一層用線性輸出）、§2.2 Training（xy 以格為基準、wh 以整張圖正規化，都落在 0 到 1；sum-squared error；λcoord=5、λnoobj=.5 的理由；預測寬高的平方根；由當下 IOU 最高的預測器負責）
- https://arxiv.org/abs/2004.10934（YOLOv4，以 ar5iv HTML https://ar5iv.labs.arxiv.org/html/2004.10934 讀全文）：§3.4〈Eliminate grid sensitivity〉，sigmoid 乘上大於 1 的係數（07-targets 提到這件事，但沒有附連結）
- https://pytorch.org/docs/stable/generated/torch.optim.Adam.html（轉址到 https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html）：演算法框（m_t、v_t、偏差修正、θ_t 更新式）與 betas 參數說明「running averages of gradient and its square」，預設值 (0.9, 0.999)
- https://github.com/cocodataset/cocoapi 的 commit 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9，PythonAPI/pycocotools/cocoeval.py：第 506–508 行（iouThrs 是 .50:.05:.95 共 10 個門檻，recThrs 是 101 點）、第 370–376 行（沒有真值的類別直接跳過，precision 留在 -1）、第 396–410 行（先取 precision 包絡線，再在 101 個 recall 位置取值）、第 452–455 行（取平均時排除 -1）
- 本機 PyTorch 2.9.1（.venv-model）實測函式庫行為：cross_entropy 的 target 為 -1 時報 IndexError；類別放在最後一軸時報 RuntimeError（shape 不合）；mse_loss 與 cross_entropy 對空 tensor 取平均得到 NaN；torch.stack 的錯誤訊息開頭；對常數 0 呼叫 backward 會報錯；float32 的 sigmoid(-18)≈1.52e-8，sigmoid(-89)=0；24 附近的 ulp 是 1.9e-6，所以 24±w/2 會捨入成同一個數；inference_mode 產生的 tensor 不能拿去算梯度；squeeze 會刪掉所有長度為 1 的軸

這一頁沒有發現與來源不符的說法。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 「不對稱的框」第一次出現時沒有定義，第二框其實是正方形，讀者容易誤讀成「不是正方形」 | 已修正：查證屬實：第二框 [36,12,52,28] 寬＝高＝16，兩框共同的性質是中心 x≠y，只有第一框寬≠高。在第一次出現處（程式說明段）補上括號定義「（中心的 x、y 不相等；其中一框的寬、高也不相等）」，剛好對應同一句列出的三種寫反。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/05-assignment.md`、`docs/lessons/07-data.md`、`docs/lessons/07-targets.md`、`docs/lessons/07-loss.md`、`docs/lessons/07-inference.md`、`docs/lessons/07-training.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核的建議在〈定稿修正〉處理；〈來源對照〉列出查閱的來源，沒有不符；批次檢查掛在本頁；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/05-assignment.md 第 13、39 行 | 殘句與內部名稱：「所有輸出和腳本都在同一層的暫存副本。repo 裡我只照查核指示跑了唯讀的摘錄比對工具」「其餘都用 box 頁面組的新值」（讀者不知道「box 頁面組」是什麼）。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：點名的兩處都沒有真正改掉，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/05-assignment.md 第 13、39 行 | 點名的「所有輸出和腳本都在同一層的暫存副本。repo 裡我只照查核指示跑了唯讀的摘錄比對工具」在第 13 行幾乎逐字保留（只把「查核指示」換成「查核指示單」），前面還多了殘句「我在自己的暫存副本。」。第 39 行的「box 頁面組」只換成「box 部分」，讀者仍不知道是什麼，發現指出的問題沒有解決。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/foundations.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `detector` 實讀本頁 6 個凍結單元；首次使用／前文方法範圍四題位置為 05-assignment/00:first_use, 05-assignment/02:first_use, 05-assignment/03:first_use，頁末為 05-assignment/05。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/detector.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/detector-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/detector.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁未有需要改寫的已裁定問題，保留原教學內容；仍完整重讀與核對。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 b 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁相關處置：DEC-015；包含採用、保留或後文撤回的來源與理解收益。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `b_foundation` 實讀本頁 7 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/b_foundation/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：無額外局部修正。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/05-assignment.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
