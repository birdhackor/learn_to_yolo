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

## 紀錄重產後的檢查

2026-10-05，由另一位 AI 在獨立暫存副本重新查核 `16-inference-head`。結論：通過；沒有必要修正，也沒有新增建議。完整讀過本頁正文、頁尾紀錄、舊審查、case、SVG、JSON 與 notebook，沒有沿用舊審查的「通過」作為本輪證據。

本輪在 `/tmp/lessons-v0.4.0-reviews/review-16/` 執行，使用 Python 3.12.14、PyTorch 2.9.1+cpu、程式指定的兩個執行緒，以及 Zensical 0.0.67。沒有改主工作區、coverage 或既有紀錄，沒有使用 Git、GPU、遠端 workflow 或下載資料集。原 case 只 import `copy`、`math` 與 PyTorch，沒有 repo 模組依賴；逐項核對的 dependency hash 因此只有 case 本身。本節沒有綁定它的補充實驗。

查核方法與結果：

- **預設、摘錄與紀錄**：從副本執行 `PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python lesson_cases/16-inference-head.py`，exit 0。七行實際 stdout 與 JSON、notebook 最後一格、頁尾區塊逐字相同；另外在本機獨立執行 notebook 最後一格，也得到相同 stdout。五份 `data-excerpt` 摘錄除翻譯註解外，都是 case 內連續、完整的程式段。檢查 section-map 的 `ready`／`lessons-v0.4.0`、notebook 的版本與最後一格 source、JSON/index 的 case/dependency SHA-256、UTC 時間、CPU、版本、passed、exit_code 與空 stderr，全部一致。2.551 秒是作者原紀錄的整個子程序時間，本輪沒有把重跑時間當成模型效能證據。
- **練習 1**：只改成 top-5 時，實際停在 case 第 77 行原 shape 斷言。照答案更新斷言後跑完，三個 shape 為 `(2,5,4)`、`(2,5)`、`(2,5)`，參數數仍為 332／278。
- **練習 2**：只把輸入改成 128×128，原程式仍全部通過；照答案把中心點倍率與 `decode_ltrb` stride 兩處改成 32，也能跑完。用同一份模型、同一個 128×128 輸入逐值比較，正確框等於錯誤框乘 2，scores／labels 相同；中心點各軸為 16、48、80、112。答案沒有漏改處。
- **手算與排列**：卷積輸出長度為 `floor((64+2−3)/2)+1=32`，pool 後為 4；16 個中心點按列攤平，第 9 個為 `(24,40)`。距離 `[1,0.5,2,1.5]×16=[16,8,32,24]`，解框 `[8,32,56,64]`；sigmoid 分數 `[0.5,0.75]`，label 1。逆 softplus 的 raw 約為 `[0.541325,−0.432752,1.854587,1.247518]`，float32 的 `sigmoid(ln3)` 實得 `0.7500000596046448`，與正文對容差／四捨五入的說明相符。用人工序號框驗證 `gather` 確實依 `[9,2,5]` 取完整四座標。逐個核對 16 個中心點和 `flatten` 順序。
- **權重、梯度與錯誤變體**：實跑刪除 step、lr=0、漏掉 one loss，均在 one 權重更新斷言失敗；複製 many、換回更新前 one 權重，均在 raw 完全相等斷言失敗。單獨反傳兩種 loss，one 不把梯度送回 backbone，many 則會；修改原模型 one 權重不影響 deepcopy 的部署副本。參數手算 224＋54＋54＝332、224＋54＝278，少 54/332≈16.265%；四個部署參數名稱與正文相同，`.eval()` 後原模型仍帶 many。把 meshgrid 改成 `indexing='xy'` 時完整程式仍會通過，與正文第 167 行明說人工例子不檢查排列順序一致，本輪沒有將其誤報成已受斷言保障。
- **教學與官方模型邊界**：逐句核對 softplus、單尺度／兩類／16 候選、平方 loss、沒有 assignment、每候選只留一類、top-3 不過門檻、沒有偵測品質證據等限定。本例只示範部署流程，沒有把 332／278 或約 16% 的比例套給完整 YOLO26；第 20 章匯出的 GridDetector 與本例不同，正文已明說。抽出固定版本官方 head 的原始 `postprocess`、`get_topk_index`、輔助方法及 `fuse`，在 CPU 真正執行：候選分數 `[0.8,0.7]` 會以兩個類別佔據官方 top-2，本例則挑兩個不同候選；`end2end=True` 移除 cv2/cv3，False 移除 one2one 分支。這是執行原始方法的局部核對，沒有宣稱跑過完整官方 YOLO26。
- **陌生讀者理解**：以高中數學程度、程式新手閱讀順序重讀，核對前置與相關章節連結。many／one、部署、raw、stride、label、容差、deepcopy、gather 的意思與用途在需要時交代；先說「只讀 one 仍白算 many」，再說實際移除、raw 比對與解碼，理由連得起來。候選排列式、逆 softplus 推導與兩題可直接照改的答案沒有省掉必要步驟；教學模型、官方 head 和第 20 章模型也有明確區別，沒有新增理解障礙。
- **圖與實際網頁**：副本 `zensical build --clean --strict` 通過；`scripts/validate_site.py` 通過。以自己的 HTTP server（8816）和 Chromium 實際看本頁，檢視 1280×900 桌面與 390×844 手機截圖，確認正文、表格、四個摺疊區、練習答案與 SVG 確實呈現。SVG 的前向實線、反向虛線、one 梯度停止於 detach、部署沒有 many／detach、兩個 raw 輸出間的紫色點線均與課文／程式一致，文字沒有裁切或箭頭歧義；本頁沒有水平頁面溢出與 JavaScript pageerror。自己的 server 已停止，未接觸 8794。

本輪重新開啟的來源與版本（官方程式以 raw GitHub 讀取相同固定 commit，均回 HTTP 200）：

- [YOLO26 原論文 v1](https://arxiv.org/html/2606.03748v1)，並開啟 [abs 頁](https://arxiv.org/abs/2606.03748)：閱讀 §3.2.1 的「One-to-One Head (default)」與雙 head 說明，以及 §3.2.2 DFL 距離期望。論文的預設 head 與 Python predict 預設值確實需要分開。
- Ultralytics commit `441632cdfd19e22e60a4b1b1999d46326ca51ec4`：[yolo26.yaml](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml)、[head.py](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)、[tasks.py](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/tasks.py)、[training recipe](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md)。核對 end2end=True、reg_max=1／Identity、兩支訓練、detach、sigmoid 與直接帶符號距離、雙階段 top-k、Detect 分支移除、BaseModel 的卷積／BN 融合與 Detect.fuse 呼叫。
- 同 commit 的 `engine/predictor.py`、`nn/autobackend.py`、`nn/backends/pytorch.py`、`cfg/default.yaml`、`utils/nms.py`、`utils/loss.py` 與 `utils/tal.py`：沿著 `nms=None` → `self.args.nms is False` → backend 在 fuse 前設定 end2end 的路徑讀過原始程式，核對預設 many＋NMS、明確 `nms=False` 改走 one；E2ELoss 算兩支 loss，dist2bbox 是中心點減左上／加右下。正文沒有把設定檔的 `end2end: True` 誤當成 predict 預設。
- PyTorch 2.9 的 equal／gather／expand 官方 HTML 三頁均回 HTTP 403；改讀官方 [v2.9.1 `_torch_docs.py`](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_torch_docs.py) 與 [v2.9.1 `_tensor_docs.py`](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_tensor_docs.py)，並讀本機 2.9.1 的 eval、detach、clone、softplus、AdaptiveAvgPool2d docstring。核對 torch.equal 的 size／elements（沒有聲稱它檢查 dtype）、gather 的 dim=1 索引式與 expand 共用儲存／先 clone 再寫的說明。

必要問題：無。建議事項：無新增；沒有為了產生發現而要求改寫正確的正文。

既有意見與本輪觀察的處理：

| 位置／項目 | 本輪核對與處理 |
| --- | --- |
| 舊〈來源對照〉要求補 BaseModel.fuse 的出處 | 已確認正文第 217 行有固定 commit 的 tasks.py 連結，來源的 BaseModel.fuse 確實融合 BN 並呼叫 Detect.fuse；原建議已處理，不需再改。 |
| 舊〈獨立查核〉說當時頁尾還是舊輸出 | 本輪當作歷史描述；目前 JSON、最後一格與頁尾七行已同步，預設重跑也相同，不需修正。 |
| 332／278、labels shape 沒有斷言；人工例子不檢查候選排列 | 重新讀 case、實跑變體後仍屬實，正文已交代限制，維持原文。 |
| stale 僅由自動產生器移除正文尾兩空行 | 不因此略過審查。按 coverage 的規則排除頁尾／正規化 Colab tag，現行正文 hash 是 `fb40f2f41649570b2c14c488efbbc34680516554aeee1e2e88dc4edba6422161`；只加回兩個換行就得到原 coverage 的 `b88b6b823d30693cb7a40ffd7d05408294f843b9b3c3e7506a934a1b4900e421`。圖與 case hash 沒變。本輪僅交付審查紀錄，由主流程另行更新 coverage。 |

查核快照 SHA-256（整個檔案的 bytes；正文 coverage hash 另列於上）：

| 檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/16-inference-head.md` | `5bf4085ba440e4e2dcd53731e216b300dc599a3103c2e6ea4af2fb76f91f5519` |
| `docs/assets/diagrams/16-head-paths.svg` | `1acac44ac36b5b847e55a22e24d2a97fc2095a25800e3ed0df71e75dbcebf2c4` |
| `lesson_cases/16-inference-head.py` | `e76f4949c12b64c8f1470fda2a8b23efff198eaa342950aaba9cd3dd924c9c3d` |
| `notebooks/16-inference-head.ipynb` | `b134a8a6854eb1cc22862ba878a9b20d2ca56eab0b67a13d4ca5225a957e740c` |
| `artifacts/checks/curriculum/16-inference-head.json` | `95b2f301b0b45e22ba7ec1b1aff8f4a21d7d2f5aae8e8706f00fb571c8ccb4cb` |
| 本輪讀取的 `reviews/16-inference-head.md` | `687e7b03aeb48f8c2d959df54d94abbfb8906d9ee463b7e8a3b701f3b90f3b77` |

限制：本輪沒有登入 Colab、執行 notebook 的 clone／安裝環境格、下載預訓練權重或驗證偵測準確度；最後一格是在指定的本機 CPU 環境真正執行。沒有真人學生測試。本頁瀏覽器檢查已做，整站 MathJax／CSS 不在本輪範圍。原始來源、下載狀態／hash、重跑輸出、練習與錯誤變體結果及截圖保留在暫存副本的 `artifacts/review-16/`，沒有寫回作者證據。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[modern當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/modern.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_b` 實讀本頁 5 個凍結單元；首次使用／前文方法範圍四題位置為 16-inference-head/00:first_use, 16-inference-head/02:first_use，頁末為 16-inference-head/04。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_b.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_b-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_b.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS04。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。
