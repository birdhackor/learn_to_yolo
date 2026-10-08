# 審查紀錄：NMS-free 推論

審查範圍：`docs/lessons/13-nms-free.md`、頁面上的圖（`docs/assets/diagrams/13-nms-free.svg`），以及 `lesson_cases/13-nms-free.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要也沒有建議。所有程式都在我自己的暫存副本執行，沒有寫入 repo。

1. 程式敘述都與定稿程式相符。lesson_cases/13-nms-free.py 的 sha256 是 7697044f…，就是兩個 unit 修完後的版本。
- 程式 exit 0，印出 5 行，和第 115–118 行的核對清單逐項相同。
- 第 19、82、85 行說留下 p0，依據是 stable=True 加上三分完全平手，逐輪走過的結果是 [0, 2]，屬實。
- 第 87 行說「p1 排前面就留 p1」。我把送進 nms 的順序改成 [1, 0, 2] 實測，印出 many/NMS: [1, 2]，斷言全過。
- torch 2.9.1 的 docstring 確認兩件事：argsort 預設不保證同分元素的順序；stable=True 保留原本先後。topk(stable=True) 實測得到 TypeError，所以第 91 行的「沒有保證」保留得正確。
- 「第 6 章 class_nms、第 7 章 decode_grid 用的 NMS 也這樣排序」：06-decode-nms.py 的 class_nms 和 miniyolo/geometry.nms 都用 argsort(descending=True, stable=True)。這兩個名稱也都在第 6、7 章頁面上出現過。
- 兩題練習完全照頁面步驟做：
  - 第 1 題：只改 target 時，程式停在 assert one_ids == [0, 2]；把斷言改成 [1, 2] 後印出 one/no NMS: [1, 2]。
  - 第 2 題：只改門檻時，程式停在框數斷言；三條斷言照參考答案改完後，三條路徑全空，exit 0。
- Colab 的格子描述（「本節可修改的完整實驗」下方那格）也核對過，相符。

2. 先前審查意見與受程式改動影響的段落全部處理。
- 第 3 行的 tag 已是 lessons-v0.4.0，沒有手改。第 28 行照先前審查意見的條件保留，因為輸出標籤沒有改。
- 第 19、82、85、87、115 行都照受程式改動影響的段落改寫。第 170、175、176 行在紀錄標記之下，一字未動。第 91 行照 unit 疑慮保留「沒有保證」。
- 頁面上已不剩平手「沒有保證」的補丁說法。全頁 grep 不到修訂敘事，也沒有程式行號。

3. 數字：0.959、0.8182、[0, 1, 2]／[0, 2]／[0, 2] 都是確定值。沒有從這台 Mac 引入任何數字。待重錄的值，編輯都已在待重錄數值清單列出。我在複本裡只對本節呼叫 verify_curriculum 的 run 與 attach，重產後頁尾印出 [0.959, 0.959, 0.959, 0.041]／[0.959, 0.041, 0.959, 0.041]，和第 115 行的新敘述、SVG 一致。

4. 摘錄：摘錄比對工具印出 []。我分別在兩段各做一次突變，各自都被抓到。我也逐行比對過：第一段是程式第 23–31 行的連續逐字複製，第二段是第 37–46 行，只是去掉 4 格縮排；兩段都只多了中文註解，引言也寫明了這一點。在複本中，validate_lessons.py 通過所有頁面檢查，只停在預期中「審查紀錄尚未建立」的那一關。

5. 易讀性：「穩定排序」在第 85 行第一次出現就有定義，第 19 行刻意不用術語，前後順序合理。

6. 建置與 SVG：在複本中，zensical build --clean --strict 與 validate_site.py 都通過，兩段摘錄渲染成 language-python。SVG 有 viewBox、title、desc，qlmanage 渲染正常，數字與程式輸出相同。這次只動了 docs/lessons/13-nms-free.md；SVG、notebook、紀錄 JSON、reviews 與程式都沒有差異。

不影響通過的備註：
(a) 第 85 行跨檔的第 6、7 章敘述目前屬實。但本頁的審查紀錄只綁本節程式，日後 geometry.py 改了排序也不會觸發本頁重審。編輯已在疑慮列出，受程式改動影響的段落也允許這樣寫。
(b) 沒有斷言固定「留下 p0」：拿掉 stable=True 後輸出不變、斷言全過。這一點編輯已在疑慮列出，要不要加斷言交給總編輯決定。
(c) 編輯報告說「notebook 最後一格要等重建」，這一句有一半已不成立：notebooks/13-nms-free.ipynb 最後一格的原始碼已經和程式相同，只有輸出還要等重錄。
(d) 第 115 行的「和上圖右半的數字相同」寫成「分數」會更精確，因為右半也有 target 欄。不至於誤導讀者，所以不列為問題。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://arxiv.org/abs/2405.14458 — abs page (v1 2024-05-23, v2 2024-10-30; the page links the unversioned abs, which resolves to v2)
- https://arxiv.org/html/2405.14458v2 — Abstract (NMS hampers end-to-end deployment and adds latency); §1 Introduction (slows inference, performance sensitive to NMS hyperparameters, prevents optimal end-to-end deployment); §2 Related Work (DETR adopts Hungarian loss for one-to-one matching); §3.1 Dual label assignments (weak supervision of one-to-one, identical-structure one-to-one head, backbone and neck get one-to-many supervision, discard one-to-many head at inference without extra cost, top-one selection performs the same as Hungarian matching); §3.1 Consistent matching metric (Eq. 1 uniform matching metric, analysis assumption of identical predictions, Eq. 2 supervision gap on soft targets, α_o2o=r·α_o2m and β_o2o=r·β_o2m imply m_o2o=m_o2m^r, r=1 default, Fig. 2(b) alignment frequency after training); §4 Analyses for NMS-free training (α_o2m=0.5, β_o2m=6.0); Appendix A.2; whole text searched for detach/stop gradient (none)
- https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/nn/modules/head.py — Detect cv2/cv3 L39–42; v10Detect L497–525 (max_det=300, deepcopy one2one_cv2/cv3, xi.detach() for one-to-one input, export-only v10postprocess)
- https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/loss.py — v8DetectionLoss L150–166 (TaskAlignedAssigner topk=tal_topk, alpha=0.5, beta=6.0), __call__ (assigner fed with that head's own detached pred_scores and pred_bboxes; box, cls BCE, DFL losses), v10DetectLoss L717–727 (tal_topk=10 and 1)
- https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/tal.py — target normalization L83–86, get_pos_mask L90–100, get_box_metrics L102–121, iou_calculation CIoU clamp(0) L123–125, select_highest_overlaps L232–258
- https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/ops.py — v10postprocess L851–864
- https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/models/yolov10/predict.py — postprocess (one2one only, v10postprocess with self.args.max_det, then preds[...,4] > self.args.conf)
- https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/cfg/default.yaml — L50 conf, L52 max_det: 300
- https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/engine/exporter.py — L226–233 (sets v10Detect.max_det = args.max_det on export)
- PyTorch 2.9.1 (version pinned in requirements) docstrings of the installed package, same text as https://docs.pytorch.org/docs/2.9/generated/torch.argsort.html (stable=False does not guarantee the order of equal elements) and https://docs.pytorch.org/docs/2.9/generated/torch.topk.html (no stable option; indices of tied elements are not guaranteed to be stable)
- https://docs.python.org/3.12/library/itertools.html#itertools.permutations — order and the empty result when r > n checked by running Python 3.12.15; the max() empty-iterable ValueError message checked by execution in the same interpreter

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 「YOLOv10 的做法」段：「先取分數最高的前 k 個（官方預設最多 300 個），再刪掉低於 score 門檻的，就直接輸出」，以及頁末連到 v10postprocess 的「postprocess 的 top-k」 | 和 13.1 同一個簡化。v10postprocess 是兩段 top-k：先依最高類別分數取 k 個候選，再取 k 個（候選, 類別）分數。所以官方輸出裡，同一個框可以帶不同類別出現多次，並不是「每個候選最多一筆」。本頁正在討論 NMS-free 輸出會不會重複，這一點和主題直接相關。 |
| 2 | 建議 | 頁末參考來源：「官方 v10Detect 推論（……`max_det = 300` 寫在這裡）」 | 預測路徑傳給 v10postprocess 的 k 是 `self.args.max_det`（predict.py），它的預設值來自 ultralytics/cfg/default.yaml L52 的 `max_det: 300`。head.py 的類別屬性 `max_det = 300` 只在 export 路徑使用（head.py L521–522），而且 exporter.py L232–233 會用 `self.args.max_det` 覆寫它。把「官方預設最多 300 個」的出處指向 head.py，讀者會以為預測時的上限是在那裡設定的。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 來源對照 | 「取前 k 個」沒說明官方排的是（框, 類別）組合，同一個框可能重複出現 | 已修正：在「YOLOv10 的做法」段補一句：官方排的是每個（框, 類別）組合的分數，多類別時同一個框可能以不同類別各出現一次。本節程式的 logits 是 [4]，每個框只有一個分數，所以取前 k 個分數就等於取前 k 個框。 |
| 2 | 來源對照 | 300 的出處指向 head.py，但預測時的 k 來自設定裡的 max_det | 已修正：已確認三件事：predict.py 傳入 self.args.max_det；default.yaml 設定 max_det: 300；head.py 的 max_det 只在 export 分支使用，exporter.py 匯出時會用 args 覆寫它。參考來源改成：head.py 的 max_det 只在匯出模型時使用；predict.py 的 k 取自設定的 max_det；另加 default.yaml（max_det: 300）的固定 commit 連結。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/13-dual-assignment.md`、`docs/lessons/13-nms-free.md`、`docs/lessons/14-feature-module.md`、`docs/lessons/15-attention-bridge.md`、`docs/lessons/15-area-attention.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」。v10Detect 的 max_det=300 只在 export 分支呼叫 v10postprocess，與括號說明相符；predict.py、default.yaml 的連結內容也符合。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：獨立查核通過；〈來源對照〉2 項在〈定稿修正〉處理；批次檢查掛在本頁；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/13-nms-free.md 第 11、26 行 | 內部名稱：「第 91 行照 unit concern 保留「沒有保證」」，以及殘句「所有程式都在我自己的複本暫存副本執行」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 11 行的殘句已清理；但處理聲稱的改字沒有發生。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/13-nms-free.md 第 26 行 | 處理寫「「unit 疑慮」改成「先前查核提出的疑慮」」，但第 26 行仍是「第 91 行照 unit 疑慮保留「沒有保證」」。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

第 4 輪第 1 項屬實：第 26 行仍是「第 91 行照 unit 疑慮保留「沒有保證」」，「先前查核提出的疑慮」不在紀錄裡。第 3 輪第 1 項中，第 11 行的殘句確實已改寫。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 3 輪第 1 項處理欄（第 91 行） | 產生器取最內層的引號，把頁面原文「沒有保證」當成仍在的點名文字。發現真正點名的是內部名稱「第 91 行照 unit concern 保留「沒有保證」」，現在只把 concern 換成了「疑慮」，「unit」仍在第 26 行，處理欄沒寫這件事。讀者會以為剩下的問題是「沒有保證」。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[modern當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/modern.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_b` 實讀本頁 4 個凍結單元；首次使用／前文方法範圍四題位置為 13-nms-free/00:scope_overview，頁末為 13-nms-free/03。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_b.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_b-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_b.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS04。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 c2 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁相關處置：DEC-024；包含採用、保留或後文撤回的來源與理解收益。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `c_modern` 實讀本頁 7 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/c_modern/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：無額外局部修正。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/13-nms-free.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
