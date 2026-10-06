# 審查紀錄：YOLOv10 dual assignment

審查範圍：`docs/lessons/13-dual-assignment.md`，以及 `lesson_cases/13-dual-assignment.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題。頁面對定稿程式 lesson_cases/13-dual-assignment.py（sha256 5ba8cda9…，已核對）的每一句描述都成立。清單上的項目都處理了，也沒有殘留修訂敘事。只有一條可選的 should（:66 的 i 沒有定義）。

驗證都在我自己的複本裡做，腳本放在暫存副本底下。沒有在 repo 裡執行任何東西。

1. 程式敘述
- 原程式 exit 0。stdout 和 artifacts/checks/curriculum/13-dual-assignment.json 的 stdout 逐位元組相同。
- 練習 1 照頁面指示把 `[.88, .10, .20]` 改成 `[.88, .10, .95]`（程式裡只出現一次），各種改法的結果：
  - 斷言都不改：停在 one_owner 斷言。
  - 只改 one_owner：停在 greedy 斷言。
  - 照舊答案只改兩個：仍停在 greedy 斷言。
  - 改 one_owner 和 greedy：停在 optimum 斷言。
  - 三個都改：通過，印出 `global quality=1.85; greedy quality=1.85`。optimum 和 greedy_value 都是 1.8499999642372131，逐位元相同。many loss 不變，one loss 會變，頁面只寫「會跟著變」，沒有寫這台 Mac 上的數字。
- 練習 2（4×3 的表）：
  - 原程式和做完練習 1 的程式，都在 exact_one_to_one 報 `ValueError: max() iterable argument is empty`（Python 3.12.15）。
  - 只放寬 exact：greedy_one_to_one 報同一個 ValueError。
  - 兩個都放寬：targets() 報 `IndexError: index 3 is out of bounds…`。
- 突變測試：拿掉 `owner[~selected.any(0)] = -1`，被 `[[.9,.8,.1],[.7,.6,.2]]→[0,0,-1]` 這個斷言擋下；把貪心改成「每個 GT 依序挑」，被 B 的 p0=.95 那張小表斷言擋下。頁面 :36、:42 說這兩個斷言各守什麼，都成立。
- 其他敘述也成立：:84 的官方 top-1 示意 `[0,-1,-1]`、:128 的「代價」例子、兩個簡化區塊列出的差異、:117–118 的梯度斷言、:122 新增的「最後一行是 step 之前算出的兩個 BCE loss」。

2. 清單項目
- trace line 3：HEAD 已經是 lessons-v0.4.0，和 section-map 一致。
- argmax 警告（:140）和寫死 greedy_value 那段（:155）都刪了，改成現在式的說明（:36、:42）。
- 受程式改動影響的段落的 :38、:148（改三個斷言、照出現順序列）、:153（「三個都改好後」）都處理了；:165 的紀錄區塊沒動。
- 程式修改清單補的 :157 也寫進 :162 了。
- 全頁沒有修訂或審查的敘事（「原表」「寫死」都是練習的語境）。

3. 數字：1.10、1.73、1.85、1.8499999642…、六種配法的總和、≈1.7×10³⁹（實算 1.7397e39）都是確定值，和程式一致。沒有引入這台 Mac 的數字。要等紀錄重產的值，編輯已經列出來了。

4. 摘錄：摘錄比對工具輸出 []，repo 和複本都一樣。兩個 Python 區塊都不是逐字摘錄，正文都標明是簡化版。正文沒有引用程式行號（「第三行」「第四行」指的是輸出的行）。validate_lessons 在摘錄和 notebook 檢查都通過，只停在 reviews「no review」，這是全站目前的狀態，不是本頁的問題。notebook 最後一格和 .py 逐字相同。

5. 渲染：zensical build --clean --strict 回報 No issues found。validate_site.py exit 0。HTML 裡參考答案的摺疊區、條列和行內程式碼都正確。本頁沒有 SVG。對照 mtime，本頁只動了 docs/lessons/13-dual-assignment.md；notebook、.py、紀錄 JSON 都早於這次編輯。

非必要的觀察，不用處理：:162 提醒放寬貪心時，如果學生只把迴圈改成 `range(min(g, p))`，`cols` 裡的 −1 會讓 `quality[i, -1]` 讀到最後一欄。結果貪心總和會大於全域最優（我測到 2.75 對 2.45），學生會在 optimum 斷言卡住。頁面沒有教這種改法，句子本身沒有錯，所以沒列成問題。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/13-dual-assignment.md:66（以及 :44–49 的簡化區塊） | 簡化區塊用 g 當 GT 編號，:54 也這樣說明；但 :66 引用的是完整程式的 `owner[c] = i`，頁面上從來沒定義 i。讀者讀到這裡要自己猜 i 是什麼，打開完整程式時又會看到小寫 g 在那裡代表 GT 數。這段不是本輪改的，句子本身也沒有錯，但編輯在疑慮裡問過要不要補，我的判斷是值得補一句，成本很低。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

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
| 1 | 建議 | 〈收益、代價與留下的問題〉的「代價」段：「所以 YOLOv10 讓兩邊用同一個品質公式，一對一再取每 GT top-1，讓每個 GT 在一對一選中的，就是它在一對多的第一名，縮小兩邊的差距。」 | 這句寫成無條件成立，但它只在論文的分析假設下成立：兩個 head 初始化相同，對每一對候選與 GT 算出相同的 p 與 IoU。官方訓練時，v10DetectLoss（loss.py L717–727）分別呼叫兩個 v8DetectionLoss，每個 assigner 收到的是自己那個 head 的 pred_scores 與 pred_bboxes，所以兩張品質表會隨訓練分開。論文 §3.1 與 Fig. 2(b) 也只報告：訓練後，一對一選中的候選落在一對多 top-1/5/10 內的比例提高了，並沒有說每次都相同。 |
| 2 | 建議 | 第五段：「論文把這套做法叫 consistent dual assignments……「一致」是指兩個 head 挑候選時，用同一個品質公式排名」 | 論文 §3.1 把「兩邊都用式 (1) 的 m(α,β)」叫 uniform matching metric。consistent matching metric 專指 α_o2o=r·α_o2m、β_o2o=r·β_o2m，這樣才有 m_o2o=m_o2m^r，兩邊排名才相同。Fig. 2(b) 的 inconsistency 對照組（α_o2o=0.5、β_o2o=2）用的也是同一個公式，只是參數不成比例。所以光說「用同一個品質公式」，不足以定義論文的「一致」。 |
| 3 | 建議 | 摺疊區〈本例與官方 YOLOv10 的差別〉表格的「head 與 loss」列 | 表裡只寫官方「分類與框都算 loss」，沒有說明官方的分類 target 不是 0／1。官方 TaskAlignedAssigner（tal.py L83–86）會把正樣本的類別 target 乘上 norm_align_metric＝m×（該 GT 正樣本中最大的 CIoU）÷（該 GT 正樣本中最大的 m）。因此 top-1 的一對一正樣本，target 等於它和 GT 的 CIoU，也就是論文的 t_o2o,i=u*。論文式 (2) 的 supervision gap 也是用這種軟 target 定義的。讀者看這張表，容易以為官方和本例一樣用 0／1 target。 |
| 4 | 建議 | 〈收益、代價與留下的問題〉的「收益」段：「官方程式直接從一對一 head 取分數最高的 k 個框（預設最多 300 個），再用 score 門檻……濾掉低分框」 | 官方 ops.v10postprocess（ops.py L851–864）的 top-k 分兩段：先依每個候選的最高類別分數取 k 個候選，再從這 k 個候選×nc 個類別的分數中取 k 個（候選, 類別）組合。所以多類別時，同一個框可以帶著不同類別出現好幾次。「取 k 個框」只是單一類別時的簡化。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 來源對照 | 「代價」段把「一對一選中的就是一對多第一名」寫成無條件成立 | 已修正：已對照官方 loss.py 的 v10DetectLoss：兩個 head 各自呼叫一次 v8DetectionLoss，用各自的預測。改成有條件的寫法：兩個 head 預測相同時（論文分析的假設）才成立；官方訓練時兩邊的第一名不一定相同，論文的實驗只顯示「和參數不成比例時相比」更常落在一對多的前幾名。表格的「品質分數」列也補上「各用自己的預測計算」。第一小節那句本來就有「在同一張品質表上」的條件，所以沒改。 |
| 2 | 來源對照 | 「一致」只寫成「同一個品質公式」，沒有涵蓋論文的 consistent matching metric | 已修正：已對照論文 §3.1：共用公式叫 uniform matching metric；consistent 指的是 α_o2o＝r·α_o2m、β_o2o＝r·β_o2m。改成「用同一個品質公式排名，而且一對一公式的 α、β 是一對多的同一倍數（下面說明）」，後文本來就有 r 倍的說明。uniform 這個名稱沒有加進頁面，以免頁面更密。 |
| 3 | 來源對照 | 差別表沒有說明官方的分類 target 是軟 target，不是 0／1 | 已修正：已對照官方 tal.py 的正規化段落（norm_align_metric），在表格加一列「分類 target」：本例填 0／1；官方填 0～1 之間的軟 target，算法是 m×該 GT 正樣本中最大的 CIoU÷最大的 m；一對一的 target 就是它和 GT 的 CIoU。附上固定 commit 的 tal.py 連結。 |
| 4 | 來源對照 | 「收益」段的「取 k 個框」忽略了 v10postprocess 是兩段 top-k | 已修正：已確認 ops.v10postprocess 的兩段 top-k 等於對（框, 類別）組合取前 k 名。改成「取分數最高的 k 個（框, 類別）組合」，並註明多類別時同一個框可能以不同類別各出現一次。 |
| 5 | 先前查核 | `owner[c] = i` 的 i 在頁面上沒有定義 | 已修正：已確認完整程式用 i 當 GT 編號、g 當 GT 數。照建議補一句括號：完整程式把 GT 編號寫成 i，就是上面簡化版的 g。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/13-dual-assignment.md`、`docs/lessons/13-nms-free.md`、`docs/lessons/14-feature-module.md`、`docs/lessons/15-attention-bridge.md`、`docs/lessons/15-area-attention.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/13-dual-assignment.md〈收益、代價與留下的問題〉「收益」段 | 同一句的前半已改成「取分數最高的 k 個（框, 類別）組合」，並說明同一個框可能以不同類別各出現一次。後半卻還寫「拿每個框自己的分數去比」「濾掉低分框」。多類別時一個框有好幾個分數，讀者不知道比的是哪一個。官方 predict.py 的 `preds[..., 4] > conf` 是拿每一列（框, 類別）的分數去篩。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |
| 2 | 建議 | docs/lessons/13-dual-assignment.md「常見錯誤」最後一條 | 「推論的 top-k 是整張圖依分數取前 k 個框」和同頁已更正的收益段（取前 k 個（框, 類別）組合）不一致。這次的修正處理的就是「取 k 個框」這個簡化，但同一頁只改了一處。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |
| 3 | 建議 | docs/lessons/13-dual-assignment.md〈本例與官方 YOLOv10 的差別〉表格新增的「分類 target」列 | 「一對一的每個 GT 只有一個正樣本」寫成恰好一個。但同一個摺疊區的下一段就示範：兩個 GT 搶同一個候選時，B 沒有一對一正樣本。官方 select_highest_overlaps 也會讓搶輸的 GT 沒有正樣本。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」。對照 THU-MIG/yolov10 453c6e3：ops.py 的 v10postprocess 先依每框最高分取 max_det 個框，再把這些框的（框, 類別）攤平取 top-k，結果等同在全部組合中取前 k，同一框可能以不同類別各出現一次；predict.py 只取 one2one 的輸出，再用 conf 門檻逐組合篩；default.yaml 的 max_det 是 300。tal.py L82–L86 把 target 正規化為 m×（該 GT 正樣本最大 CIoU）÷（最大 m）；一對一 topk=1 加上 select_highest_overlaps，每個 GT 至多一個正樣本，target 約等於它的 CIoU（CIoU 已 clamp 到 ≥0）。「一致」的 r 倍數說明，以及論文只顯示一致參數讓一對一更常落在一對多前幾名，都與論文相符。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：先前查核 1 項與來源對照 4 項，都在〈定稿修正〉處理；THU-MIG/yolov10 453c6e3 在來源清單；批次檢查留下的 3 項，由〈後續編輯的檢查〉核對。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/13-dual-assignment.md 第 13、34 行 | 殘句與內部名稱：「驗證都在我自己的複本裡做：暫存副本，腳本放在暫存副本底下」「查核範圍清單補的 :157 也寫進 :162 了」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 13 行刪掉「：暫存副本」後是一般說法；第 34 行已改成「程式修改清單」。處理說明屬實。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[modern當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/modern.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_b` 實讀本頁 5 個凍結單元；首次使用／前文方法範圍四題位置為 13-dual-assignment/00:first_use, 13-dual-assignment/01:first_use, 13-dual-assignment/02:first_use，頁末為 13-dual-assignment/04。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_b.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_b-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_b.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R026；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。
