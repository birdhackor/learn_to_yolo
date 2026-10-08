# 審查紀錄：Grid MiniYOLO 資料

審查範圍：`docs/lessons/07-data.md`、頁面上的圖（`docs/assets/diagrams/07-data.svg`），以及 `lesson_cases/07-data.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題，只有一條 should（第 39 行省略註解的用詞）。

每一項都在暫存副本裡重驗過。

1. 程式敘述：lesson_cases/07-data.py 在暫存副本執行，exit 0，stdout 和 artifacts/checks/curriculum/07-data.json 逐字相同，紀錄的 case_sha256 也對得上目前的程式。頁上依賴程式的說法逐項實測都成立：
   - torch.stack 的錯誤訊息開頭。
   - build_targets：[0,0,0,0] 報「標註框必須有正面積」；類別不是 long 時報 ValueError；[12,8,28,24] 照收。
   - 固定場景 2 個正格、14 個負格；空圖 16 格 objectness 都是 0；空圖的 box／class 梯度為 0。
   - collate 回傳 Tensor 加上原物件組成的 list。
   - ShapeDataset seed=7 的框數是 [2,0,2,0,0,2,2,1]（3 張空圖）；寬高 8～15、每個框整個在一格內且格子互異；背景小於 0.04；顏色精確。
   - 練習照頁面指示實際改程式跑過：x1 改成 9 時停在 torch.equal，訊息是 Annotation and colored pixels disagree；紅框改 [8,12,28,28]、切片改 8:28 後印出 `red/blue channel sums 320.0 256.0`；原本 8:24==256 的 assert 仍通過，改成 8:28==320 也通過；只改標註、只改畫素、附加題 [12,8,28,24] 都停在 torch.equal 那行。
   - 交叉引用（首頁紅框、第 1 章 permute／reshape、第 4 章 [0,4]、第 5 章同格碰撞、08 頁的 DataLoader(collate_fn=collate) 片段、第 7 章各程式都直接呼叫 collate）目前都成立。

2. 先前審查意見處理：
   - 第 3 行頁首按鈕已指向 lessons-v0.4.0，沒有手改。
   - 第 112 行照先前審查意見指定的句子開頭改寫，頁上現在只剩一個 Colab 連結。
   - SVG desc 的括號已刪。
   - 選做的 zeros_like 修飾也做了。
   - 受程式改動影響的段落沒有 07-data 的條目；lesson_cases/07-data.py 與 miniyolo/data.py 都和 HEAD 相同。
   - 全頁和 SVG 都沒有修訂或審查的敘述。

3. 數字：都是確定值（256、240、320、14、16、[2,0,2,0,0,2,2,1] 等），沒有從這台 Mac 帶進任何數字。editor 已列出三行要等重產紀錄的輸出。

4. 摘錄：三個 Python 區塊都標了 data-excerpt，摘錄比對工具對本頁回傳 []。我另外核對了行序和 `...` 註解的內容，都正確。去縮排的做法和 00-warmup 一致；正文沒有引用程式行號。

5. 可讀性：改寫後的句子清楚，教學順序沒變。只有第 39 行的「執行緒數／隨機種子」列為建議。

6. 建置：zensical build --clean --strict 與 validate_site.py 都通過（colab_pairs passed）；產出的 HTML 有三個帶 data-excerpt 的 python 區塊，Colab 連結只有一個。validate_preparation 通過。validate_lessons 只停在審查涵蓋檢查（07-data 還沒有審查），摘錄檢查已通過；這是發布前預期的狀態。SVG 有 viewBox、title、desc，用 qlmanage 轉成 PNG 畫面乾淨，desc 逐項對照座標都對。和 07-data 相關的改動只有這兩個列出的檔案。

提醒：
- 07-data.svg 也出現在 docs/lessons/07-inference.md 第 25 行。這次改了 desc，07-inference 的審查涵蓋會過期（editor 已提出）。
- 第 89 行對第 8 章的引用要成立，08-own-data.md 必須保留那段「換成自己資料時的寫法」片段；以目前的版本來說是成立的。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/07-data.md 第 39 行（第一個摘錄區塊的 `...` 省略註解） | 這次新加的註解「兩行設定（隨機種子、執行緒數）」帶進了「執行緒數」這個新詞。第 0～6 章都沒出現過這個詞，要到 7.4〈訓練〉才解釋（threads=2 表示只用 2 個 CPU 執行緒）。另外「隨機種子」和術語表的「seed（亂數種子）」不一致；01、03、07-heldout、07-training 都用「亂數種子」。這只是被省略那兩行的標籤，讀者不會因此卡住或被誤導，所以算建議。 |

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
| 1 | 先前查核 | 摘錄裡 `...` 那行的註解提早帶出「執行緒數」，「隨機種子」也和術語表的 seed（亂數種子）不一致 | 已修正：查證屬實，也確認改寫後的說法成立：程式完全沒用到 torch 的全域亂數（ShapeDataset 用自己的 Generator），印出的總和都是精確的 256.0，執行緒數影響不了。註解改成和 00-warmup 同樣的寫法「與兩行不影響本節輸出的設定」。同頁 ShapeDataset 段落的「隨機種子（seed）」也改成術語表用詞「亂數種子（seed）」。摘錄比對工具結果為 []。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/05-assignment.md`、`docs/lessons/07-data.md`、`docs/lessons/07-targets.md`、`docs/lessons/07-loss.md`、`docs/lessons/07-inference.md`、`docs/lessons/07-training.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核唯一的建議在〈定稿修正〉處理；〈來源對照〉列出來源（PyTorch 文件），沒有不符；批次檢查掛在本頁；沒有空清單、截斷或指向錯誤；工作流程紀錄裡本頁的發現都收進紀錄。沒有發現問題。

### 第 4 輪：上一輪的處理：通過

第 3 輪沒有發現，沒有處理說明需要核對。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/grid.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `detector` 實讀本頁 6 個凍結單元；首次使用／前文方法範圍四題位置為 07-data/00:first_use, 07-data/02:first_use, 07-data/03:first_use，頁末為 07-data/05。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/detector.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/detector-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/detector.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS04。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 b 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁相關處置：DEC-016；包含採用、保留或後文撤回的來源與理解收益。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。
