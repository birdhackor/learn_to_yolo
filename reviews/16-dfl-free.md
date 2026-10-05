# 審查紀錄：YOLO26 DFL-free

審查範圍：`docs/lessons/16-dfl-free.md`、頁面上的圖（`docs/assets/diagrams/16-distance-decode.svg`），以及 `lesson_cases/16-dfl-free.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要或建議問題。頁面正文、練習題和參考答案，都和目前的 lesson_cases/16-dfl-free.py（sha256 fc9aeedf…）一致。所有程式都在暫存副本執行，沒有在 repo 裡跑任何東西。

1. 程式相關的敘述都正確。
- 程式 exit 0。第 5、6 行輸出是 [[74.0, 64.0, 228.0, 108.0]] 和 [[92.0, 68.0, 108.0, 116.0]]，其餘 11 行和紀錄逐字相同。
- 第 2 題我照頁面指示一步步改：
  - 只改 target：停在 initial_terms 斷言。
  - 再改 initial_terms：停在範圍斷言。
  - 再改範圍斷言：停在 decoded 斷言。
  - 四處都改、x2 填 148.：exit 0，印出 `Smooth L1 3.187500 -> 0.000000`。
  - 漏改任一個斷言，都在那一行報錯；x2 填 144. 也停在 decoded 斷言。
  這些都和參考答案說的一樣。
- 第 3 題用程式自己的 decode 驗算：[-1,2,3,4] 得到 [92,68,108,116]，是合法框；[-3,2,2,4] 得到 [108,68,100,116]，x1>x2，不合法。兩個都和答案一致。
- 沒改到的段落也順便驗了：右邊第 137 步後才進入平方段，左邊 3 步；平方段每一步誤差乘 0.875。

2. 先前審查意見和受程式改動影響的段落都處理了。
- 第 3 行的 Colab 按鈕在 HEAD 已經是 lessons-v0.4.0，和 section-map.json 的 source_ref 一致。
- 受程式改動影響的段落列的第 29、33、84、120、131、135、137 行都已改寫。「人工參考點……距離公式對任何點都一樣」整句已刪，頁面沒有留下替舊缺陷辯解的文字。
- 第 145、154、155 行在自動產生的紀錄區塊裡，沒有手改，這樣是對的。
- 全頁沒有修訂或審查經過的敘述。搜尋到的「改成」都是練習的操作指示，「新值＝舊值」是 SGD 公式。

3. 數字
- 新出現的數字（84、74、64、228、108、204、148、92、68、116、100）都由算式直接算出，不是這台 Mac 跑出來的。要等紀錄重產才能確定的值，修改者都列出來了。
- 我在暫存副本只對這一節模擬 verify_curriculum 的 run 加 attach：頁面只有紀錄區塊的標頭和兩行解碼輸出會變，變完就和正文一致。這份模擬紀錄只在暫存副本，不是正式紀錄。

4. 摘錄
- 頁面唯一的 Python 區塊已標 data-excerpt，摘錄比對工具印出 []。
- 這個檢查只看每一行有沒有出現在原檔，不看順序，所以我另外逐行對照。三個 `...` 依序代表：initial_terms、它的斷言和 before 三行；迴圈裡第一段 `if step == 0:`；第二段 `if step == 0:`。這和區塊前的說明一致。
- 正文沒有寫程式行號。

5. 可讀性
- 第 29 行用 12.1 的講法（從 0 起算第幾欄、第幾列那一格的中心），沒有用到 16.3 才定義的「格心」。12.1 在 HEAD 和本版都寫了「密集 head 通常取各格中心當候選點」，所以「和 12.1 節一樣取格的中心」成立。
- 第 108 行的新句子和第 20 章內容相符：CPU 段比較 PyTorch 和 ORT 並計時；L4 段把 TensorRT 的 FP32／FP16 engine 和同一張 L4 上的 PyTorch 比對並計時。用「同一種模型」而不是「同一個模型」是對的，因為第 20 章說兩份模型架構相同、權重不同。

6. 建置、圖和檔案範圍
- 在暫存副本跑 zensical build --clean --strict（回報 No issues found）和 validate_site.py，兩者都 exit 0。
- SVG 用 qlmanage 算圖，文字沒有互相重疊；xmllint 通過；viewBox、title、desc 都在。
- 用 desc 寫的換算 (3.25x−50, 3.25y−104) 從圖反推回畫素：藍框 [74,64,228,108]、紅點 (84,84)、虛線 x=204、刻度 74/84/204/228，四個雙箭頭都停在框線內緣，全部和程式一致。
- 這一節相關的檔案裡，只有 docs/lessons/16-dfl-free.md 和 HEAD 不同；SVG、程式、notebook、紀錄 JSON、review 都沒有動。git diff --check 也通過。

附註，不算問題：
- 修改者在疑慮第 1 條說 validate_lessons 會因為 notebook 和 lesson case 不一致而失敗。這對本節不成立：notebooks/16-dfl-free.ipynb 最後一格已經和目前的程式逐字相同。真正還沒做完的是兩件事：紀錄重產（JSON 裡的 case_sha256 還是 1c0ef4db…），以及 reviews 重審並重新記錄涵蓋範圍。
- 疑慮第 4 條說 20-deployment.md 寫 0.160／0.162 ms。那是 HEAD 的值；工作樹已改成 0.147 ms，和 deployment-gpu.json 一致。

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
| 1 | 建議 | 〈官方訓練仍保留什麼〉第一段「正規化 L1 的做法是：……再對四邊取 ｜預測−目標｜ 的平均（這就是 L1 loss）」，以及第二段「權重、正樣本品質……也會影響訓練」 | 描述只講到「每個正樣本的四邊取平均」。固定版本 loss.py 的 `BboxLoss.forward`（第 143–154 行）接下來還會把每個正樣本的 L1 乘上 `weight`（該候選 target_scores 的和），全部加總後除以 `target_scores_sum`。`v8DetectionLoss` 再把它乘上 `self.hyp.dfl`（第 477 行）。default.yaml 第 106 行是 `dfl: 1.5`，註解直接寫 DFL-free 時這個值就是 L1 的 gain；訓練紀錄的名稱則是 `l1_loss`（loss.py 第 367 行）。頁面只提到變數叫 `loss_dfl`，再用一句籠統的「權重……也會影響」帶過。讀者照這段重寫時會算成不加權的平均；想調 L1 的權重時，也不知道要調的是 `dfl`。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 來源對照 | 正規化 L1 漏了 target score 加權、除以總和與超參數 dfl 的倍數 | 已修正：對照固定版本的 loss.py（BboxLoss 乘 weight、加總後除以 target_scores_sum；v8DetectionLoss 再乘 hyp.dfl；loss 名稱為 l1_loss）與 default.yaml（dfl: 1.5，註解寫明 DFL-free 時是 L1 的 gain），確認問題成立。在定義後補一段：每個正樣本乘上它的類別 target（引用 12.2 節的說法），加總後除以總和，CIoU 也同樣加權，最後乘 dfl（預設 1.5）。下一段補上超參數也叫 dfl、印出的名稱是 l1_loss。頁尾加上 default.yaml 的連結。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/16-dfl-free.md`、`docs/lessons/16-inference-head.md`、`docs/lessons/16-training.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」。yolo26.yaml 的 `reg_max: 1 # DFL bins`；Detect 的 `self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()`，與正文相符，也支持術語表 DFL 列連到 16.1。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：〈來源對照〉1 項在〈定稿修正〉處理；批次檢查掛在本頁；〈後續編輯的檢查〉齊全；結構檢查通過。沒有發現問題。

### 第 4 輪：上一輪的處理：通過

第 3 輪沒有發現，沒有處理說明需要核對。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[modern當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/modern.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。
