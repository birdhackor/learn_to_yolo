# 審查紀錄：Grid MiniYOLO targets

審查範圍：`docs/lessons/07-targets.md`、頁面上的圖（`docs/assets/diagrams/object-journey.svg`），以及 `lesson_cases/07-targets.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要，也沒有需要列出的建議。

1. 程式敘述（所有實測都在暫存暫存副本）
- 跑 lesson_cases/07-targets.py，exit 0，stdout 和 artifacts/checks/curriculum/07-targets.json 逐字相同。
- 頁面列的七個斷言，逐條對到程式：shape、正格 2 個、box[0,1,1]、box[0,2,3]、兩格的 class_ids、空圖沒有正格。執行順序也和頁面一致：先斷言，再做碰撞測試，最後印出結果。
- 碰撞測試的 else 分支：我把 build_targets 的碰撞檢查拿掉再跑，程式停在 `AssertionError: Two objects silently overwrote one slot`，和頁面說的一樣。
- 照頁面把 scene 的第二個框改成 [10,14,26,30]，程式停在 build_targets 那一行，訊息是 `ValueError: same-cell collision: image 0, cell (1,1)`，括號裡依序是 row、col，也就是 gy、gx。
- 計數那行的 2 確實是寫死在 print 裡的數字，30 由 `~positive` 數出來。
- 練習：在模擬 notebook 的 namespace（照 notebook 執行最後一格完整程式，`__name__=='__main__'`）裡，貼上頁面參考答案的程式。第 1 題原樣通過；第 2 題照頁面說的三處修改後也通過，得到 [[0,0,2]] 和 [.75,.375,.125,.125]。
- 頁面其他說法也逐一實測過：四個欄位的 shape 和 dtype、objectness 等於 positive 轉成 0／1、負格的 box 和 class_ids 都填 0、`class_ids != -1` 得 32 格、`pred[...,5:][pos]` 是 [2,2]、`class_ids[pos]` 是 [0,1]、只有一張空圖時取平均得 NaN、壞框拋出 ValueError。另外核對了 ShapeDataset 的中心不會落在格線上、第 5 章的 build() 把 class_ids 填 −1、07-loss 用的是 `pos.any()`、head 的 7 個數的順序。

2. 先前審查意見與影響清單
- L80「本版」、L82「這個版本」都已改掉。
- L3 的 Colab tag，HEAD 裡本來就是 lessons-v0.4.0，和 section-map 一致，所以不必手改。
- L27 關於首頁沿用那張圖的說明正確，保留不改是對的。
- 受程式改動影響的段落裡沒有這一頁的條目，其他部分的說明也沒有點到這一頁。
- 全頁沒有描述教材修訂經過的字眼。第 148 行的「原本的 main()」指的是 notebook 裡原有的程式，不是教材的修訂歷史。

3. 數字：全部是固定值，沒有從這台 Mac 帶進來的數字。需要等重產紀錄核對的值，editor 都已列出。

4. 程式摘錄：摘錄比對工具印出 `docs/lessons/07-targets.md []`。簡化版區塊在正文裡寫明了是簡化版，寫法和 09-anchors、13-dual-assignment 一致。正文沒有引用程式行號。

5. 渲染：在暫存副本裡 zensical build --clean --strict 回報 No issues found，validate_site 全部通過。object-journey.svg 有 viewBox、title、desc；用 qlmanage 渲染正常，紅框、中心點和淺藍負責格都和程式一致。這張 SVG、lesson、miniyolo/targets.py、geometry.py、notebook 和紀錄，跟 HEAD 比都沒有改；這一節相關的檔案裡，只有頁面本身被改了。

6. 看到但沒列為問題的兩點（照實寫，都沒有讀者會因此做錯）：
- 第 124 行同一段裡，ValueError 用「拋出」、AssertionError 用「丟出」。全書本來就兩種說法並用，第 5 章也用「丟出」。
- 練習題用簡寫 `positive.nonzero()`，正文描述程式時用 `target["positive"].nonzero()`。參考答案的程式寫法是正確的，讀者照著跑不會出錯。

證據檔在暫存副本：sim_exercise.log、collide.log、else.log、claims.log、build.log、validate_site.log、object-journey.svg.png。

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

以腳本核對紀錄：獨立查核通過，沒有待處理的建議，所以沒有〈定稿修正〉；〈來源對照〉列出來源，沒有不符；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/07-targets.md 第 13 行 | 殘句：「1. 程式敘述（所有實測都在暫存副本 …/暫存副本）」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：點名的殘句沒改，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/07-targets.md 第 13 行 | 點名的「1. 程式敘述（所有實測都在暫存副本 …/暫存副本）」逐字還在，處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
