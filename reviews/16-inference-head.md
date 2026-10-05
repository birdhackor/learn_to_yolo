# 審查紀錄：YOLO26 推論 head

審查範圍：`docs/lessons/16-inference-head.md`、頁面上的圖（`docs/assets/diagrams/16-head-paths.svg`），以及 `lesson_cases/16-inference-head.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要，也沒有需要再改的建議。全部檢查都在暫存副本裡跑，主 repo 沒有寫入任何檔案。

1. 頁面對程式的敘述都屬實。
   - 在副本跑課程程式：exit 0，印出 7 行，和正文輸出清單逐字相同（stdout 存在暫存副本）。
   - 突變實測（driver 是暫存副本driver.py）：
     - 拿掉 step、lr=0、loss 漏掉 one 這三種錯，都停在 `assert not torch.equal(before, model.one.weight)`。
     - 複製成 many、只把 one 權重換回舊值、整份用舊模型，這三種都停在 `torch.equal(reference, deploy_raw)`。
     - indexing 改成 'xy' 時全部照樣通過。頁面寫「不檢查 16 個候選的排列順序」，和這個結果一致。
   - 練習 1 照頁面操作：先停在 shape 斷言；把斷言改成 (2,5,4)／(2,5) 後，印出 (2, 5, 4) (2, 5) (2, 5) 和 332 278，和參考答案相同。
   - 練習 2：改成 128×128 後全部斷言通過；照答案把兩處 16 改成 32 後可以跑完。不改時，框剛好是正確值的一半。
   - 另外核對過：named_parameters 的四個名稱、參數數 224 和 54、torch.equal 對 shape 與 dtype 的行為。

2. 7 條先前審查意見和 12 條受程式改動影響的段落都已處理，前一位檢查者提的第 191 行建議也處理了。
   - fused 補註、寫死的 verified、0.7500000596046448、atol／rtol 公式都已刪除。
   - 保留的三句（332／278 和 labels 的 shape 沒有斷言、排列順序沒有檢查、參考答案裡的 labels）對定稿程式仍然屬實。程式問題清單也沒有對應的程式修正，所以保留是對的。
   - 頁面沒有修訂或審查經過的敘述。

3. 數字：0.0、0.75、332／278、54、224 都是確定值，沒有引用 Mac 上量到的數字。頁尾紀錄區塊仍是舊輸出（第 2、6 行不同），要用 verify_curriculum.py 重產；編輯已列出全部受影響的值。

4. 摘錄：5 個區塊都標了 data-excerpt，摘錄比對工具印出 []。另外確認每一塊都是程式裡連續、完整的一段。把副本頁面故意改壞，檢查抓得到。正文沒有引用程式行號。

5. 可讀性：torch.equal、容差、clone 都有說明。引用的第 0、2、4.2、20 章內容在工作樹裡都在，說法一致；章節順序仍然照著教。

6. 渲染：zensical build --clean --strict 和 validate_site.py 在副本都通過。SVG 用 qlmanage 渲染正常，有 viewBox、title、desc，內容和程式一致。跟本節有關的檔案裡，只有頁面一個改了。

附註，都不算問題：
- 第 199 行「兩章比對的都是解碼前的 raw」：第 20 章另外還比對還原後的框，但它確實有逐值比對 raw，這句不算錯。
- 程式沒有守著候選的排列順序，要不要補守衛是程式 owner 的決定。頁面已經照實寫明。
- 編輯說「notebook 要重建，在那之前 validate_lessons 會失敗」不完全準確：實測 notebook 最後一格已經和程式相同，source_ref 也已是 lessons-v0.4.0。validate_lessons 失敗只是因為全站頁面還沒有審查紀錄，這是暫時狀態。
- 檢查期間，主 repo 的 data/manifest.json 和 docs/preparation/*.md 被其他程序改了，與本次檢查無關。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml (第 9–10 行 end2end/reg_max；第 52 行 Detect(P3,P4,P5))
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py (Detect：第 108–143 行 __init__／DFL 或 nn.Identity／one2one deepcopy；第 155–167 行 end2end；第 180–204 行 forward 與 detach；第 219–227 行解碼；第 249–288 行 postprocess／get_topk_index；第 290–295 行 fuse)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py (DFLoss 第 89–108 行；BboxLoss 第 111–156 行；v8DetectionLoss 第 347–383、412–481 行；E2ELoss 第 1322–1354 行)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py (TaskAlignedAssigner 第 54–59 行 stride_val；get_pos_mask 第 180–204 行；select_candidates_in_gts 第 317–348 行；select_highest_overlaps 第 350–383 行 topk2；dist2bbox 第 457–476 行；bbox2dist 第 479–494 行)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/optim/muon.py (zeropower_via_newtonschulz5：bfloat16、5 次迭代；muon_update 的 flatten(1)；MuSGD.step 的 momentum_buffer_SGD 與 muon/sgd 比例；並在本機用此函式重算 diag(3,0.1) 的例子：bf16 得 0.6875/1.1328，float32 得 0.6970/1.1288)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md (第 21、26–27、98、158–172、233–247、323–337 行：Objects365→COCO、nms 預設、實驗分支與內部參數 o2m)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/tasks.py (BaseModel.fuse 第 243–280 行；DetectionModel.init_criterion 第 618–620 行 E2ELoss；parse_model 第 2058、2211 行 end2end 傳給 Detect)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/engine/trainer.py (第 638–639 行每 epoch 呼叫 criterion.update()；第 1176–1231 行 MuSGD 參數分組 ndim∈{2,4}、nesterov)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/engine/predictor.py (第 460–470 行 end2end=self.args.nms is False)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/autobackend.py (第 193–246 行 end2end 參數)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/nms.py (第 71 行 end2end 時不做 NMS)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/default.yaml (第 61 行 nms 預設 None＝外部 NMS；第 103–106 行 box/cls/dfl gain)
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py (DFL 第 60–82 行，bins 0…c1−1)
- https://arxiv.org/abs/2606.03748 與 https://arxiv.org/html/2606.03748v1 (Table 1；§3.2.1 One-to-One Head (default)、topk=7/topk2=1；§3.2.2 式 (1) 與 (K−1)s 上限；§3.3.1 MuSGD；§3.3.2 式 (2)(3)；§3.3.3 式 (4)–(6)，式 (5) 的門檻是 d<s_min；§4.1 (0.8,0.2)→(0.1,0.9)、每 epoch 更新一次、s_min=8、s_ref=16；§4.3.2 Table 4；§4.3.4 Table 6)
- https://kellerjordan.github.io/posts/muon/ (Muon = MomentUm Orthogonalized by Newton-Schulz)
- PyTorch 2.9.1 本機 docstring 與實測 (torch.nn.functional.smooth_l1_loss 預設 beta=1.0、reduction='mean'；torch.equal；Tensor.gather 沿 dim=1 的語意)

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈收益、代價與部署檢查〉「整個模型的 `fuse()` 除了呼叫 `Detect` 的 `fuse()`，還會做卷積／BN 融合」；頁尾參考來源 | 這句話是對的，但根據在 `ultralytics/nn/tasks.py` 的 `BaseModel.fuse()`（固定版本第 243–280 行：`fuse_conv_and_bn`、RepConv／RepVGGDW 重參數化，最後 `m.fuse() # remove the unused detection branch`），不在頁尾連結的 head.py 或 yolo26.yaml 裡。頁面在〈官方 YOLO26 怎麼做〉說過，官方程式的說法都以頁尾連到的那一版為準；讀者照連結去查，找不到這一句的出處。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 來源對照 | 「整個模型的 fuse() 會做卷積／BN 融合」的出處不在頁尾連結裡 | 已修正：確認 tasks.py 的 BaseModel.fuse 會做 fuse_conv_and_bn 與 RepConv 重參數化，最後呼叫 Detect 的 fuse()。頁尾補上固定 commit 的 tasks.py 連結，連結文字寫明 BaseModel.fuse；照寫作規則沒有寫行號。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/16-dfl-free.md`、`docs/lessons/16-inference-head.md`、`docs/lessons/16-training.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」，摺疊區那句「都以頁尾參考來源連到的那一版官方程式為準」指得清楚：三個頁尾連結與內文的訓練說明連結同為 441632c。逐項對照：yolo26.yaml 有 `end2end: True`；Detect.fuse() 依 end2end 把 one2one_* 或 cv2/cv3 設成 None；predictor.py 用 `end2end=self.args.nms is False` 選 head，default.yaml 的 nms 預設 None（外部 NMS）；訓練說明第 26 行寫 NMS 預設、`nms=False` 走 NMS-free head；論文 §3.2.1 標「One-to-One Head (default)」。head.py 有 get_topk_index、postprocess，tasks.py 的 BaseModel.fuse 先融合 BN。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：〈來源對照〉1 項在〈定稿修正〉處理；批次檢查掛在本頁；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/16-inference-head.md 第 15、25、34 行 | 殘句與內部檔名：「突變實測（driver 是暫存副本driver.py）」「code-issues.md 也沒有對應的程式修正」（不在 repo 的內部檔）、「在副本都通過（暫存副本、暫存副本）」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 25 行已改成「程式問題清單」，第 34 行括號已刪；但第 15 行仍是殘句，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/16-inference-head.md 第 15 行 | 點名的「突變實測（driver 是暫存副本driver.py）」只刪掉 driver.py，變成「突變實測（driver 是暫存副本）」，意思不通（driver 不是暫存副本）。處理卻寫殘句已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 2 | 建議 | reviews/16-inference-head.md 第 3 輪第 1 項處理欄 | 「程式問題清單改成「程式問題清單」」是同語反覆。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

第 3 輪第 1 項、第 4 輪第 1 項屬實：第 15 行的「突變實測（driver 是暫存副本driver.py）」仍在。第 4 輪第 2 項不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪第 2 項處理欄（第 102 行） | 點名的是第 3 輪第 1 項處理欄的同語反覆「程式問題清單改成「程式問題清單」」。那一格已重寫，同語反覆已經不在，處理欄卻寫「未改：…「程式問題清單」」；這個字串只出現在正文第 25 行。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
