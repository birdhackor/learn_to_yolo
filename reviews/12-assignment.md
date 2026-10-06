# 審查紀錄：Sample assignment

審查範圍：`docs/lessons/12-assignment.md`、頁面上的圖（`docs/assets/diagrams/12-assignment.svg`），以及 `lesson_cases/12-assignment.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。docs/lessons/12-assignment.md 沒有必要或建議問題。所有指令都在暫存副本執行，沒有在 repo 內跑任何東西。

1. 頁面對程式的敘述都成立（對照定稿的 lesson_cases/12-assignment.py）。
   - 程式 exit 0，依序印出頁面列的六行。第 1 行是 `[[0.225, 0.311, 0.0, 0.0], [0.0, 0.147, 0.567, 0.0]]`，第 4 行 gradient 是 `[-0.125, -0.125, -0.125, 0.125]`。
   - 程式正好有四個斷言，依序核對 owner、空影像 owner、梯度、練習 1 的 owner。品質表確實沒有斷言。
   - 重算過的數字都對：IoU 表、遮罩前品質 [0.225, 0.3111, 0.0035, 0]、遮罩後品質、k=1 的兩個 owner（[-1,0,1,-1]、[0,1,-1,-1]）、SGD 一步後的 logits [0.125, 0.125, 0.125, -0.125]（程式確實沒印），以及逐行走查各步的值。
   - 練習 2 照頁面指示把 `.7` 改成 `.1`：exit 0，第一行印 0.044，owner 仍是 [0,0,1,-1]，所有斷言通過。
   - 改成 0：第一個斷言失敗，owner 是 [0,1,1,-1]。
   - 延伸題（衝突改比 metric）：先在第一個斷言失敗；把它改成 [0, 1, 1, -1] 後全部通過，其他斷言不必改。
   - 練習 1 的答案與程式結尾的斷言逐字相同。Colab 指示裡的「本節可修改的完整實驗」在 notebook 裡確實有這個標題。13.1 程式解衝突確實比品質。

2. 各項痕跡都處理了。
   - 第 3 行已經是 lessons-v0.4.0。
   - 第 76 行行尾註解與程式第 22 行逐字相同。
   - 第 109 行的括號保留，因為程式仍然不印 logits，這句是真的；梯度字面值也改成實際印出的格式。
   - 第 113 行刪掉 float32 說明，改成小數第 3 位，與程式的 `.double().round(decimals=3)` 一致。
   - 第 125 行改成「官方」。摺疊區開頭已把範圍固定在頁尾 commit，「官方」也是整個摺疊區的用詞。
   - 受程式改動影響的段落第 164、169 行在自動產生的區塊內，沒動；我用 diff 確認這個區塊與 HEAD 逐字相同。
   - 全頁沒有修訂或審查的敘述，也沒有引用程式行號。

3. 新加的數字（0.311、0.147、0.044、±0.125）都是定值，不是這台 Mac 跑出來才有的。等紀錄重產後才會一致的值，修改者已全部列出。

4. 唯一的 Python 區塊已標 data-excerpt。摘錄比對工具對 repo 和暫存副本都印出 []；validate_lessons 的摘錄與 notebook 檢查 42 頁都通過。檢查器比對時會去掉註解，所以我另外逐行手動比對：程式碼與程式第 21–32 行一致（只少一層縮排），英文行尾註解相同，其他註解都是中文。

5. 我查證了改過的摺疊區那一句，對照 Ultralytics 441632c 的 tal.py 與 loss.py。`stride_val = stride[1]` 同時是判斷門檻和放大後的邊長，條件是 `wh < stride_val`。常見的 stride 8、16、32 下就是「邊長小於 16 放大成 16」，和 16.3 節的寫法一致。摺疊區其他項也相符：topk=10、α=0.5、β=6、CIoU 截到 0、在所有框內 GT 間取 CIoU 最大、target 依品質縮放、detach 後才交給 assigner。

6. 渲染與檔案範圍。
   - zensical build --clean --strict exit 0，validate_site.py exit 0。摘錄區塊渲染成 Python 高亮區塊，4 個摺疊區都正常。
   - 12-assignment.svg 與 HEAD 相同，有 viewBox、title、desc。依 48+12x 換算，座標與程式的 gt、points、predicted_boxes 一致；qlmanage 渲染正常。
   - 本節相關檔案中只有頁面改動；程式、notebook、紀錄 JSON、SVG 都沒動。

不影響判定、發布前要做的事（審查用的事實與寫作規範清單已說明是發布時才會成立的狀態，不是頁面問題）：
- 本節紀錄 artifacts/checks/curriculum/12-assignment.json 和頁尾執行紀錄仍是舊的 float32 第 1 行，要用 verify_curriculum.py 重產。
- validate_lessons 目前只卡在審查涵蓋（42 頁都還沒有審查紀錄）。
- 程式第 44 行的註解不在本頁範圍，修改者已列在疑慮。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://arxiv.org/abs/2107.08430 (YOLOX, arXiv PDF): §2.1 'Decoupled head' 段落（'Replacing YOLO's head with a decoupled one…'）、Fig. 2 說明
- https://arxiv.org/abs/2108.07755 (TOOD, arXiv PDF): §3.2 Task Alignment Learning（'comprises a sample assignment strategy and new losses'）、§3.2.1 式 (9) t=s^α×u^β 與 'Training sample assignment'（每個物件取 t 最大的 m 個）、§3.2.2 normalized t、§4 'On hyper-parameters'（α=1、β=6、m=13）、附錄 'Optimization'（多物件衝突時交給面積最小的物件）
- https://arxiv.org/abs/2006.04388 (Generalized Focal Loss v1, arXiv PDF): §1 與 Fig. 3（Dirac delta、遮擋／模糊造成的邊界模糊、攤平的分佈）、§3 式 (6) DFL 定義與 'rapidly focus…nearest two to y'、式 (7) GFL 統一式與 'FL, QFL and DFL are all special cases of GFL'、Table 2 與 §4（實驗在 mmdetection 上進行，n=16、Δ=1）
- https://arxiv.org/abs/1804.02767 (YOLOv3, arXiv PDF): §2.2 Class Prediction（independent logistic classifiers、binary cross-entropy）
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/nn/modules/head.py: Detect.__init__（nc=80、reg_max=16、no=nc+4*reg_max、cv2／cv3 定義、legacy 分支）、inference 的 scores.sigmoid()
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/nn/modules/block.py: class DFL（arange(c1) 固定權重、沿 bin 軸 softmax 後求期望值）
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/utils/loss.py: DFLoss（clamp 到 reg_max-1-0.01、左右 bin 權重）、BboxLoss（只對 fg_mask 算 CIoU 與 DFL、bbox2dist 的 reg_max-1 上限）、v8DetectionLoss（TaskAlignedAssigner topk=10、alpha=0.5、beta=6.0、stride；pred_scores.detach().sigmoid() 與 pred_bboxes.detach() 交給 assigner；BCE 涵蓋所有 anchor；preprocess 補齊 GT；bbox_decode 用 view(b,a,4,K).softmax(3)）
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/utils/tal.py: TaskAlignedAssigner（stride_val=stride[1]、get_pos_mask、get_box_metrics 的 score^α×overlap^β、iou_calculation 為 CIoU clamp(0)、select_topk_candidates、select_candidates_in_gts 的小框放大與嚴格框內、select_highest_overlaps 在框內 GT 中取 CIoU 最大、_forward 的 normalized align metric、get_targets）與 bbox2dist
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/nn/tasks.py: DetectionModel.init_criterion（非 end2end 用 v8DetectionLoss）、parse_model 的 legacy 判定（C2f 系列模型維持 legacy=True；C3k2／A2C2f／C2fCIB 時設為 False）
- https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/gfl_head.py: Integral（linspace(0, reg_max, reg_max+1)）、GFLHead 的 reg_max 文件字串（'Max value of integral set {0, ..., reg_max}'）與 gfl_reg 輸出 4*(reg_max+1)
- PyTorch 2.9.1（本機 .venv-model 實測）：nn.Module.zero_grad(set_to_none=True) 預設與結果為 None、torch.allclose 預設 rtol=1e-05／atol=1e-08、第二次 backward 的 'Trying to backward through the graph a second time' 錯誤、F.cross_entropy 目標索引越界時拋出 IndexError 'Target 4 is out of bounds.'

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 開頭「本節的規則分四步」清單（第 11–18 行），特別是第 4 步「衝突：交給 IoU 最大的 GT」與其後「本例只示範 TAL 的部分結構」 | 頁面把四步規則說成 TOOD 論文所提 TAL 的「部分結構」，但論文本身只涵蓋第 2、3 步。TOOD §3.2.1 只寫：每個物件取 t=s^α×u^β 最大的 m 個 anchor 當正樣本，其餘當負樣本（§4 定 m=13）；正文沒有「參考點須落在框內」這條資格。至於一個 anchor 同時是多個物件的正樣本時，論文附錄〈Optimization〉寫的是「只交給面積最小的物件」，不是 IoU 最大。框內資格與以 IoU 解衝突都是實作層的選擇：Ultralytics tal.py 的 select_candidates_in_gts 做框內資格，select_highest_overlaps 用 CIoU 取最大來解衝突。讀者照本頁去讀論文，或依「TAL」自行實作時，會以為以 IoU 解衝突是論文的原始設計。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 來源對照 | 四步規則被說成 TAL 的結構，但 TOOD 正文只有第 2、3 步 | 已修正：已在 ar5iv 查證 TOOD §3.2.1：正文只寫每個物件取 t 最大的 m 個當正樣本，其餘當負樣本，§4 取 m=13，沒有框內資格。另確認 Ultralytics 的 select_highest_overlaps 以 CIoU 取最大來解衝突。補一句：論文正文只寫第 2、3 步；第 1 步與第 4 步來自實作，Ultralytics 解衝突比的是 CIoU。附錄「交給面積最小的物件」這條，因 ar5iv 沒有收錄補充材料而無法查證，所以不寫。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/12-anchor-free.md`、`docs/lessons/12-decoupled-head.md`、`docs/lessons/12-assignment.md`、`docs/lessons/12-dfl.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」，位置正確；tal.py、loss.py 都是 441632c 的固定連結。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：獨立查核通過；〈來源對照〉1 項建議在〈定稿修正〉處理；Ultralytics 441632c 的 tal.py、loss.py 都在來源清單；批次檢查掛在本頁；〈後續編輯的檢查〉齊全。沒有發現問題。

### 第 4 輪：上一輪的處理：通過

第 3 輪沒有發現，沒有處理說明需要核對。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[evolution當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/evolution.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/evolution.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/evolution.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_a` 實讀本頁 5 個凍結單元；首次使用／前文方法範圍四題位置為 12-assignment/00:first_use, 12-assignment/02:first_use, 12-assignment/03:first_use，頁末為 12-assignment/04。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_a.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_a-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_a.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R037；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-06：v0.6.1 有界修正複查

TOOD 歷史引文補完整論文名稱與中文意義，仍區分 TOOD、TAL 與本例任務對齊分配。

本輪方法、逐批閱讀原始紀錄、非作者技術核對、另一位讀者銜接與實際 Zensical 修改段落截圖見 [v0.6.1 局部複查](clear-tutorial/release-v0.6.1-2026-10-06/README.md)。本次只重新檢查修改處、必要上下文與版本一致性；其餘正文、程式及圖的既有審閱保留原範圍，不改標為整頁或全書新的首次盲讀驗收。
