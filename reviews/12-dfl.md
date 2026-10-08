# 審查紀錄：DFL

審查範圍：`docs/lessons/12-dfl.md`，以及 `lesson_cases/12-dfl.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要也沒有建議等級的問題。所有檢查都在我自己的暫存副本裡做，主 repo 只讀過 git diff、git status，並依指示對它跑了一次唯讀的摘錄比對工具。

1. 程式敘述都正確（對照 lesson_cases/12-dfl.py）
- 主程式結束碼 0，第 3 行輸出是 [[0.0025, 0.7475, 0.2474, 0.0025]]，其餘四行與紀錄相同。
- 練習 1 照頁面四行改法實跑，結束碼 0。初始梯度印出 [[0.25, 0.25, -0.15000009536743164, -0.34999990463256836]]，學到的機率是 [[0.0025, 0.0025, 0.3975, 0.5975]]，期望值 2.5899。頁面寫的「斷言全部通過」「後兩個值是 −0.15000009…、−0.34999990…」「機率接近 [0, 0, 0.4, 0.6]」「期望值略小於 2.6，仍在 0.02 容差內」全部屬實。
- 練習題說「跟著要改三處」，我逐處套用驗證：只改 target 時，梯度的 allclose 失敗；再改 wanted_grad，換期望值斷言失敗；再改期望值斷言，換 probs[0, 1] > .73 那一行失敗；四處都改完才通過。題目說法正確。
- target 3.2 時，dfl() 的範圍斷言報 AssertionError。拿掉斷言、讓 target 為 3.0 時，cross_entropy 報 IndexError（Target 4 is out of bounds.）。兩者都與頁面一致。
- 第 39 行說程式片段是簡化版，描述正確：完整程式用 weight_right、logits.shape[-1]，並多一行範圍檢查。

2. 盤點項目都處理了
- 先前審查意見第 3 行：HEAD 的 Colab 連結已經是 lessons-v0.4.0。
- 受程式改動影響的段落第 175、182 行：都在自動產生的紀錄區塊內，沒有手改，這是對的。
- 程式修改清單提到練習梯度的 float32 尾數：已在參考答案中說明。
- 頁面沒有修正說明式的附註，也沒有改版、審查或寫作過程的敘述。grep 到的「原本」都指 12.1／12.2 的原設計或練習前的原值，屬教學用語。

3. 數字
- 新寫進的數字都可以手算重現，不依賴這台 Mac：
  - float32(2.6)＝2.5999999046…，兩個權重各差 9.54e-8，確實「不到一千萬分之一」。
  - 我用 numpy float32 獨立推導，梯度和 Mac 實跑逐位相同；每一步中間值都能被 float32 精確表示，所以運算順序或 FMA 不影響結果。
  - 唯一前提是 softmax(0) 算出剛好 0.25。紀錄機器的主程式梯度印出剛好的 0.25 與 0.0，可見前提在那台機器上也成立。
  - allclose 的預設容差可以讓這組梯度通過。
- 練習答案中依賴訓練結果的數字（[0.0025, 0.0025, 0.3975, 0.5975]、2.59）已改成定性描述，符合「本版改過的程式，練習答案不放機器相依數字」的規則。
- 其他手算數字我都重算過：1.385794、0.562、0.570、0.785、0.897、0.8415、0.857、1.151、0.9823。
- 要等紀錄重產後再核對的值，editor 都列出來了：紀錄區塊、第 90、103、60 行。

4. 程式摘錄
- 範圍斷言那個區塊已標 data-excerpt，渲染成正常的 Python 程式區塊。
- 摘錄比對工具對主 repo 和暫存副本都印出 docs/lessons/12-dfl.md []。
- 簡化版片段與讀者自己要打的片段都沒有標記，正確。
- 正文沒有引用程式行號。

5. 易讀性
- float32、float64、torch.allclose 第一次出現時都有白話解釋。
- 第 4 章〈座標轉換與還原〉的交叉引用成立：04-coordinates.md 第 101 行確實講了 float32 存不準大多數小數，寫法和其他頁一致。
- 參考答案依序是：手算 → 改程式 → 看輸出尾數的原因 → 定性結果，順序合理。

6. 建置與檔案範圍
- zensical build --clean --strict 結束碼 0；validate_site.py 結束碼 0（連結、錨點、摺疊區塊、數學都通過）。
- 各頁圖檔清單中本頁沒有 SVG。
- 與 12-dfl 相關的檔案只有 docs/lessons/12-dfl.md 有變動。

不屬於本頁問題、但發布流程要注意的兩件事（editor 也已提出）：
- artifacts/checks/curriculum/12-dfl.json 已過期：紀錄的 case_sha256 是 d100d38f…，目前程式是 2f3ec889…。notebook 重建後要執行 verify_curriculum.py --section 12-dfl 重產。
- 本頁文字改了，reviews/coverage.json 對本頁的審查紀錄失效，需要重審。

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
| 1 | 建議 | 摺疊區「名字裡的 Focal 是什麼意思」（第 11–13 行） | 摺疊區只用「聚焦」解釋名字。論文確實寫 DFL「forces the network to rapidly focus on the values near label y」，但論文對這個名字最直接的交代是另一件事：§3 與式 (7) 把 QFL、DFL 統一成 Generalized Focal Loss（GFL，Focal Loss 的推廣），並寫明原始 FL、QFL、DFL 都是 GFL 的特例。DFL 就是不帶調整項的那一種；論文的理由是框只在正樣本上學、沒有類別不平衡問題，所以「直接沿用 QFL 的完整 cross entropy 部分」（式 6 前一句）。只讀本頁的人會以為這個名字和 Focal Loss 家族無關，只是取「聚焦」的字面意思。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 來源對照 | 「Focal」只解釋成「聚焦」，漏了 GFL 的統一框架 | 已修正：已在 ar5iv 查證 GFL 論文的原文（FL、QFL、DFL 都是 GFL 的特例；框只對正樣本學、沒有類別不平衡，所以沿用完整的 cross entropy 部分）。在摺疊區保留原兩句，補一句這個說明。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/12-anchor-free.md`、`docs/lessons/12-decoupled-head.md`、`docs/lessons/12-assignment.md`、`docs/lessons/12-dfl.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」；GFL 論文、Ultralytics 441632c 的 loss.py／block.py、mmdetection v3.3.0 的 gfl_head.py 都可取得（HTTP 200）。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：〈來源對照〉1 項在〈定稿修正〉處理；GFL 論文與 Ultralytics 441632c 的 block.py、loss.py 都在來源清單；批次檢查掛在本頁。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/12-dfl.md 第 23 行 | 內部名稱：「查核範圍清單提到練習梯度的 float32 尾數」，讀者不知道「查核範圍清單」是什麼。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 23 行已改成「程式修改清單」，處理說明屬實。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[evolution當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/evolution.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/evolution.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/evolution.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_a` 實讀本頁 6 個凍結單元；首次使用／前文方法範圍四題位置為 12-dfl/00:first_use, 12-dfl/01:first_use，頁末為 12-dfl/05。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_a.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_a-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_a.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁未有需要改寫的已裁定問題，保留原教學內容；仍完整重讀與核對。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 c2 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `c_foundation` 實讀本頁 6 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/c_foundation/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：R011。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/12-dfl.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
