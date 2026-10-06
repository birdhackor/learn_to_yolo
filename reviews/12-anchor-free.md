# 審查紀錄：Anchor-free

審查範圍：`docs/lessons/12-anchor-free.md`、頁面上的圖（`docs/assets/diagrams/12-anchor-free.svg`），以及 `lesson_cases/12-anchor-free.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要問題，只有一條 should：第 11 行導言的「只確認兩件事」，這句在本次編輯前就有。

1. 程式敘述都成立。定稿程式在暫存副本執行，exit 0，5 行輸出都和頁面描述一致。
   - 練習：照頁面逐條改（point 改成 [[22.,22.]]、/8 改成 /4、第一個斷言改成 [[2.5,1.5,4.5,3.5]]、三處 decode(...,8) 改成 4；確實是一處在斷言、兩處在 print）。80 步就通過，前兩行和參考答案逐字相同。
   - 單位錯誤情境（distance 仍除以 8）會被斷言擋下；[17,19,31,29] 算得對。
   - 延伸題（y2 改成 62、預期值改成 [2.5,1.5,4.5,10]）：80 步在 after < before/100 失敗。after 約是門檻的 3.95 倍，float32 與 float64 結果相同，不是邊緣值。81 步仍然失敗，86 步起通過，框外點的斷言照樣通過。
   - 編者沒用受程式改動影響的段落建議的 [2.5,1.5,10,3.5]，理由成立：把 x2 改成 62 時，框外點的畫素距離變成 [32,12,18,8]，四個全正，最後一個斷言會失敗。
   - 突變測試：左右對調、少除 stride、少乘 stride、點多乘 stride、上下對調都會被擋；上與右對調擋不住，頁面也沒有這樣宣稱。
2. 四條先前審查意見和 19 條受程式改動影響的段落都已處理。(24,24) 人工點的例外說明、藍點、(20,20)「同樣只是人工設定的點」都刪掉了。用 grep 找過全頁，沒有修訂或流程敘述。
3. 數字：d、各邊 loss、初始梯度 [−0.125,−0.100857,−0.100857,−0.038357]、對稱 target 的梯度、[32,12,−4,8]、stride 16 的格中心都重算過，全部一致。正文沒有引入這台 Mac 的訓練數字，等紀錄重產的值編者都已列出。頁尾 evidence 區塊現在還是舊紀錄，屬發布前的暫時狀態。
4. 摘錄：摘錄比對工具對真實工作樹和暫存副本都印出 []。三段摘錄我逐行對照過，程式碼的內容與順序都和 .py 相同，只換了註解。正文沒有程式行號。
5. 交叉引用都對得上：4.2 節有 torch.allclose、第 2 章有 isfinite、第 0 章有 .item()、12.2 節有分類分支，槽（slot）和術語表一致。
6. 渲染：暫存副本裡 zensical build --clean --strict 與 validate_site.py 都 exit 0。HTML 裡三段 data-excerpt、六個摺疊區、表格、alt 都正確。SVG 用 qlmanage 算圖，座標、箭尖、標籤、viewBox、title、desc 都和程式一致。validate_lessons 與 validate_curriculum_evidence 在暫存副本會失敗，原因只是全站的 review 與紀錄還沒產生，屬預期。程式、SVG、notebook、紀錄、術語表相對 HEAD 都沒變。

範圍外：docs/glossary.md:113 仍寫 (24,24)、[1.5,1,2,1.5]，和本頁不一致，應由該檔負責人處理。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/12-anchor-free.md 第 11 行（導言）「實驗只確認兩件事：框和四個距離能互相換算（…），以及四個距離真的能用梯度下降學到」 | 程式實際上確認了三件事。除了往返換算和距離學得到，最後一個斷言還確認框外點 (44,28) 至少有一個距離是負的。本次改寫的第 61 行也明寫「完整程式最後一個斷言檢查的就是這件事」，和導言的「只…兩件事」互相矛盾。這句在本次編輯前就有（舊程式也有框外點斷言），不在先前審查意見與受程式改動影響的段落清單內。讀者不會因此做錯事，所以列為建議。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：讀過的檔案：審查用的事實與寫作規範清單；docs/lessons/12-anchor-free.md 全文（含摺疊區）；lesson_cases/12-anchor-free.py；docs/assets/diagrams/12-anchor-free.svg 的原始碼，並用 qlmanage -t -s 1200 轉成 PNG 逐項核對座標、箭頭與標籤；docs/glossary.md（ltrb、logits、anchor（先驗）、DFL、候選、slot、符號 P／d／t）。

查證前後文時對照了這些頁面：07-targets（sigmoid 偏移、每格一個槽）、07-loss（ln u 的導數、sigmoid 的導數）、07-inference 與 09-anchors（offset、logits、exp 解出很大的框、回歸的定義）、06-decode-nms（[row,col] 與 (col,row)）、11-iou-loss（「常見錯誤」列出把 GIoU 限制到 [0,1]）、12-decoupled-head、12-assignment（不用「動態」一詞；@torch.no_grad 的理由）、12-dfl（16 個 bin＝0～15 格、120 畫素）、00-warmup、02-diagnostics、04-coordinates，以及 zensical.toml 導覽的節次編號。也讀了 scripts/validate_lessons.py 的摘錄規則：它會去掉註解再比對，所以頁面程式塊裡加的中文註解是合法的。

建置與執行：在暫存副本用 zensical build --clean --strict 建置，exit 0，再從 site/lessons/12-anchor-free/index.html 抽出正文，逐段讀標題、表格、摺疊區與程式塊。Playwright 的瀏覽器沒有安裝，所以沒有截手機寬度的畫面。用 PYTHONPATH=. OMP_NUM_THREADS=2 執行本節程式，exit 0，五行輸出裡的確定值都和正文一致。

做過的練習（全部在暫存副本）：
- ex1：照四點清單改成 stride 4、點 (22,22)。前兩行和參考答案一字不差，80 步就通過。
- ex1_wrong：仍除以 8。第一個斷言失敗；另外算出一半大小的 target [1.25,0.75,2.25,1.75]，解碼得 [17,19,31,29]。
- ex2：延伸題 y2=62。80 步時 after<before/100 失敗；在這台 Mac 上最少要 86 步；改成 100 步（ex2b）整支程式通過，框外點斷言也通過。
- ex1_forgot_third：漏改印學得框的那處 decode。全部斷言通過，第四行印出約 [2,10,58,50]。

其他驗證：用 autograd 核對初始梯度與對稱 target 的梯度；量了每步推進量（第一步約 0.0317，上限 0.125）。另外寫了一段小程式比較：把 encode／decode 的左右一起寫反時，點在框中心 (26,26) 斷言抓不到，點在 (28,28) 抓得到。

最後掃描了修訂經過的字眼，以及中英文之間的空格與全形標點，都沒有發現問題。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | 〈真的更新四個距離〉倒數第二段「如果只檢查輸出 shape 是 `[1,4]`……這個實驗多確認了『真值框 → 距離 target → loss → 梯度 → 參數 → 解碼回框』整條路的方向和單位都對」；連帶自主練習〈參考答案〉 | 這句說整條路的單位都確認過，包括最後的「參數 → 解碼回框」。但程式只對兩件事下斷言：真值框換成 target 再解碼回原框，以及 loss 降到初值的百分之一以下。學到的距離解碼成框（`learned box pixels` 那行）只有印出，沒有任何斷言檢查。我照練習修改，但故意漏改第三處 `decode(..., 8)`（印學得框的那處）：程式全部通過、正常結束，第一、二行和參考答案一字不差，第四行卻印出約 `[[2.01, 10.03, 57.97, 49.99]]`，正是本頁「常見錯誤」說的格與畫素混用。參考答案只寫了第一、二行該印什麼，讀者沒有東西可以對照第四行，會以為自己做對了。 |
| 2 | 建議 | 開頭第三段「實驗只確認兩件事：框和四個距離能互相換算……以及四個距離真的能用梯度下降學到」 | 程式其實還核對第三件事：框外點 `outside` 至少有一個負距離。〈為何仍要 assignment〉也稱它是「完整程式最後一個斷言」。讀者對照程式裡的 assert 會數不攏，以為自己漏看了什麼。 |
| 3 | 建議 | 〈一個物件的數字旅程〉表格下方那段：「表格列出完整 head 的輸出規格」與末句「這個 1 是 P，不是四邊中的一項」 | 表格第一列 candidate points 不是 head 的輸出，下一句也說它只由特徵圖大小和 stride 決定，所以「輸出規格」前後矛盾。末句的對比也讓人摸不著頭緒：前一句剛說程式省去 batch 軸，讀者自然會問「`[1,4]` 的 1 是 B 還是 P」，不會以為 1 是四邊中的一項。 |
| 4 | 建議 | 〈為何仍要 assignment〉第二段「本節的輸出經過 softplus，永遠大於 0」 | softplus 在這裡第一次出現，也是課程裡第一次出現，但定義要到下一小節「softplus 把任意實數變成正數……」才給。讀者在這裡只能先接受「永遠大於 0」。 |
| 5 | 建議 | 〈為何仍要 assignment〉第三段 YOLOv8 那句的括號：「原始碼算的是第 11 章〈IoU 類 loss〉的 CIoU，並把負值截成 0，不是普通 IoU」 | 讀者剛在 11.4 節的「常見錯誤」讀到「把 GIoU 限制到 [0,1]，丟掉兩框分開時的負值訊號」。這裡說 YOLOv8 把 CIoU 的負值截成 0，卻沒說為什麼可以，讀者容易以為 YOLOv8 犯了 11.4 的錯，或以為 11.4 的提醒不成立。真正的理由（挑點只用來排名、不求梯度）要到 12.3 節才講。這個長括號夾在句子中間，也讓整句很難讀。另外同一頁對 11.4 的稱呼不一致：這裡寫「第 11 章〈IoU 類 loss〉」，後文寫「11.4 節」。 |
| 6 | 建議 | 同段「這就是下一段說的動態 assignment，12.3 節會用簡化版示範」 | 下一段只在列舉 YOLOv8 的差異時寫了「動態 assignment（12.3 節）」幾個字，沒有任何說明，讀者往下找不到解釋。12.3 節通篇也沒有「動態」一詞，而是稱為 sample assignment、task-aligned assignment，讀者對不上名字。 |
| 7 | 建議 | 摺疊區〈softplus 和第 9 章的 exp 有什麼不同〉「它的斜率是 sigmoid(x)」 | 這裡直接給結論，但後面的初始梯度（softplus 在 0 的斜率 0.5）和延伸題的 0.125×sigmoid(raw)² 都靠它。數學好的讀者會想自己驗證；7.3 節已經教過「ln u 的導數是 u'/u」，只差一行就能推出來。 |
| 8 | 建議 | Smooth L1 的說明：「Smooth 是把誤差接近 0 的地方改成平滑的平方」及「誤差大時斜率固定為 ±1……」那段 | 頁面說了大誤差為什麼不用平方，卻沒說小誤差為什麼不用純 L1。讀者會問：既然 L1 好，為什麼接近 0 的地方要改？ |
| 9 | 建議 | 摺疊區〈初始梯度怎麼算〉末句「就算 target 是對稱的 [1.5,1.5,1.5,1.5]……和 target 對不對稱無關」；以及「這個實驗多確認了……（左右沒有對調……）」 | 讀者沒有理由以為對稱的 target 梯度會是零，這句要反駁的誤解來得沒頭沒腦。真正該講的反而沒講：本例刻意讓點偏離框中心（l≠r、t≠b），斷言才抓得到左右寫反。我實測過：點放在框中心 (26,26) 時 target 是 [1.75,1.25,1.75,1.25]，把 encode 與 decode 的左右一起寫反，兩個斷言照樣通過；用本例的 (28,28)，第一個斷言就失敗。 |
| 10 | 建議 | 〈收益、代價與容易搞錯的地方〉第一句「不需因長寬比分佈變化重新聚類先驗（第 9 章的尺寸聚類）」 | 全頁都說「anchor 尺寸」，只有這裡突然用「先驗」。術語表把「先驗」當 anchor 的別名，但本頁沒有說明。句子又很擠，讀者不容易看出它是在回答開頭的問題：「資料的長寬比改變，尺寸還合適嗎」。 |
| 11 | 建議 | 摺疊區〈三項代價的例子〉「距離範圍」：「例如 12.4 節的 DFL 用 16 個 bin（距離刻度）時，stride 8 下最多只能表示 120 畫素」 | 少了關鍵一步：讀者會算 16×8＝128，看不出 120 從哪來（16 個 bin 代表 0～15 格）。另外 12.4 節自己的實驗用 4 個 bin，16 是 Ultralytics 的預設值；寫成「12.4 節的 DFL 用 16 個 bin」，讀者到 12.4 會找不到。「120 畫素」指的是每條邊的距離，這點也沒寫清楚。 |
| 12 | 建議 | 「常見錯誤是……或忘記框外候選點不可直接使用正距離 target」 | 句意難懂。框外的點根本算不出全正的距離 target，「不可直接使用正距離 target」卻讀起來像是它有一個正的 target，只是不能用。 |
| 13 | 建議 | 自主練習〈參考答案〉摺疊區裡的「延伸」 | 延伸題的題目和答案（「80 步就不夠」與原因）放在同一個摺疊區。讀者打開參考答案時，同時看到新題目和它的結論，沒機會像主練習那樣先預測再執行。「要增加步數才會通過」也沒說加到多少、改哪一行。我在暫存副本實測：80 步時 `after < before / 100` 失敗（after 約 0.136，門檻約 0.034），86 步起通過；100 步時 after 約 0.0007，整支程式連同框外點斷言都通過。訓練數字會因電腦略有不同，但 100 步的餘裕很大。 |
| 14 | 建議 | 圖 docs/assets/diagrams/12-anchor-free.svg 與「看圖時注意」段 | 正文要讀者在圖上找「第 (3,3) 格」「第 (5,3) 格」和框中心 (26,26)，但圖上只有畫素刻度，沒有格的編號，也沒標框中心，讀者得自己數格子。至於 (5,3) 是 (column,row) 還是 (row,column)，正文只能靠「同一列」反推；6.1 節剛提醒過 tensor 索引是 [row,col]，讀者很容易讀反。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：讀過的檔案：
- docs/lessons/12-anchor-free.md、lesson_cases/12-anchor-free.py、docs/assets/diagrams/12-anchor-free.svg、審查用的事實與寫作規範清單、scripts/validate_lessons.py（excerpt_problems）、artifacts/checks/curriculum/12-anchor-free.json、section-map.json、docs/glossary.md。
- miniyolo/targets.py、miniyolo/inference.py（第 7 章 grid 的 target 與解碼）。
- 交叉引用的課程頁：00-warmup、02-diagnostics、04-coordinates、05-assignment、07-loss、09-anchors、09-anchor-clustering、10-multiscale、11-iou-loss、12-decoupled-head（含 lesson_cases/12-decoupled-head.py）、12-assignment、12-dfl、13-nms-free、16-training。

原始來源：
- Ultralytics 固定 commit 441632cdfd19e22e60a4b1b1999d46326ca51ec4（從 raw.githubusercontent.com 下載）：
  - ultralytics/__init__.py：__version__ = "8.4.171"。
  - ultralytics/nn/modules/head.py：Detect.__init__（cv2 輸出 4*reg_max、cv3、DFL、reg_max=16）、forward_head 的輸出形狀 (B,4*reg_max,A)／(B,nc,A)、_inference 輸出 (B,4+nc,A)、_get_decode_boxes「dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0), ...) * self.strides」、decode_bboxes 的「xywh=not self.end2end and not self.xyxy」。
  - ultralytics/utils/tal.py：make_anchors（docstring 寫 "Anchor points in grid units"，offset 0.5）、dist2bbox、bbox2dist（clamp 到 reg_max−0.01）、TaskAlignedAssigner（iou_calculation 是 "bbox_iou(..., CIoU=True).squeeze(-1).clamp_(0)"、get_box_metrics、select_candidates_in_gts 的 stride_val 放大 "Valid boxes with a side smaller than stride_val are enlarged..."、self.stride_val = self.stride[1]）。
  - ultralytics/utils/loss.py：DFLoss、BboxLoss（CIoU 加 DFL，bbox2dist(..., reg_max-1)）、v8DetectionLoss（TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0, stride=self.stride.tolist())、bbox_decode、assigner 的輸入乘 stride）。
  - ultralytics/nn/modules/block.py：DFL（softmax 後與 arange 權重的 conv，即期望值）。
  - ultralytics/utils/metrics.py：bbox_iou 的 CIoU。
  - ultralytics/nn/tasks.py：DetectionModel.init_criterion（沒有一對一 head 時用 v8DetectionLoss）。
- Ultralytics 歷史版本：tal.py 在 commit 9c92716216（8.3.252）與 f2d3aed634a5b0e4828024718d4a61ab2f83fb19（8.4.0，YOLO26 發布）之間做 diff，確認小框放大是 8.4.0 才加入，8.3.252 是 bbox_deltas.amin(3).gt_(eps)；兩版的 __init__.py 版本字串。另用 gh api 列出 repos/ultralytics/ultralytics/commits?path=ultralytics/utils/tal.py&sha=441632c 的提交（含 e82c8452d6 #25912）。
- FCOS 論文：https://arxiv.org/abs/1904.01355（ar5iv HTML，https://ar5iv.labs.arxiv.org/html/1904.01355）。3.1 節：Eq.(1) l*=x−x0, t*=y−y0, r*=x1−x, b*=y1−y；"location (x,y) is considered as a positive sample if it falls into any ground-truth box"；"(⌊s/2⌋+xs, ⌊s/2⌋+ys)"；"we employ exp(x) to map any real number to (0,∞)"。第 2 節："The most popular anchor-free detector might be YOLOv1"、DenseBox 的 4D 向量。
- Ultralytics 官方文件 https://docs.ultralytics.com/models/yolov8/："YOLOv8 was released by Ultralytics on January 10, 2023"、"Ultralytics has not published a formal research paper for YOLOv8"。
- PyTorch 2.9.1 原始碼（.venv-model 裡安裝的版本）：torch/nn/modules/loss.py 的 SmoothL1Loss docstring（beta 分段、預設 mean）、torch/nn/modules/activation.py 的 Softplus docstring。

執行的指令：
- 建立副本：rsync -a --delete --exclude .git --exclude site --exclude '.venv*' --exclude artifacts/runs --exclude data/curated --exclude data/downloads 到 12-anchor-free-tech。
- 跑本節程式：PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/12-anchor-free.py，exit 0，印出 [[2.0,1.5,1.5,1.0]]、[[12.0,16.0,40.0,36.0]]、0.376236 -> 0.000012、[[12.01,16.03,39.97,35.94]] 和框外點訊息。
- 摘錄檢查：摘錄比對，結果 []。
- 用 sed 產生並執行的變體：exp/stride4.py（練習：exit 0，印 [[2.5,1.5,4.5,3.5]]）、exp/ext.py（延伸：80 步時 AssertionError）。
- 驗算腳本：exp/sim.py（延伸各步的推進量、86 步才通過、錯用 stride 時解出 [17,19,31,29]）、exp/grad.py（用 autograd 核對四邊 loss、初始梯度、對稱 target 的梯度）。
- 建置網站：.venv-docs/bin/zensical build --clean --strict，exit 0，"No issues found"；再檢查 site/lessons/12-anchor-free/index.html 的 6 個摺疊區、表格、程式區塊和 data-excerpt。
- 渲染圖：qlmanage -t -s 1200 -o render docs/assets/diagrams/12-anchor-free.svg，看過 PNG，並逐一核對 SVG 座標（畫素 (x,y) 對應到 (64+9x,104+9y)）。
- 搜尋：用 grep 查交叉引用頁和用語表，也查了頁面有沒有敘述製作或修訂經過的字眼。

沒有在 repo 根目錄裡執行任何程式或建置。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈為何仍要 assignment〉第 3 段（第 63 行）「YOLOv8 則在框內的點中，依模型當下的分類分數…動態挑選負責的點」；連帶第 9 行「本節對照的是它的原始碼（頁尾連結）」與〈三項代價的例子〉的「候選密度」 | 頁尾固定的 commit 441632c 是 Ultralytics 套件 8.4.171（ultralytics/__init__.py：__version__ = "8.4.171"），不是 2023 年 YOLOv8 發布時的程式。 這一版的選點規則和頁面寫的不同： - ultralytics/utils/tal.py 的 TaskAlignedAssigner.select_candidates_in_gts，會先把寬或高小於 stride_val 的 GT 邊以中心放大到 stride_val（self.stride_val = self.stride[1]，stride 為 [8,16,32] 時是 16 畫素；docstring 原文："Valid boxes with a side smaller than stride_val are enlarged to stride_val about their center."），再判斷候選點是否在框內。 - loss.py 的 v8DetectionLoss 以 stride=self.stride.tolist() 建這個 assigner；tasks.py 的 DetectionModel.init_criterion 對沒有一對一 head 的模型（YOLOv8）用的就是 v8DetectionLoss。所以用這份程式訓練 YOLOv8 也會套用這條規則。 - 被選中的框外點，BboxLoss 以 bbox2dist(anchor_points, target_bboxes, reg_max - 1) 把負距離夾成 0。 這條規則是 8.4.0（YOLO26 發布，commit f2d3aed634）才加入的；8.3.252（commit 9c92716216）的同一函式只檢查點是否嚴格在框內（bbox_deltas.amin(3).gt_(eps)）。 所以「在框內的點中」描述的是 YOLOv8 發布時的規則，和頁面說要對照的那份原始碼不一致。頁內「候選密度」例子裡 stride 16 下沒有框內點的 10×10 小框，正是這份程式用 STAL 補救的情況。12.3 節摺疊區與 16.3 節都寫了這條規則，本頁沒有提示。 |
| 2 | 建議 | 第 11 行「實驗只確認兩件事：框和四個距離能互相換算…以及四個距離真的能用梯度下降學到」，以及〈真的更新四個距離〉第 124 行「這個實驗多確認了『真值框 → 距離 target → loss → 梯度 → 參數 → 解碼回框』整條路的方向和單位都對」 | 頁面對「實驗確認了什麼」的說法前後不一致，而且兩處都和程式實際的斷言對不上： - 第 11 行說只確認兩件事，但程式還有第三個斷言：框外點的畫素距離至少一項小於 0。第 61 行也把它當成程式的檢查來介紹。 - 第 124 行把「參數 → 解碼回框」也算成確認過。但學到的距離解碼成框（decode(point, F.softplus(raw).detach(), 8)）只有 print，沒有斷言。程式的斷言只有：target 等於手算值、decode(target) 回到 gt、每步梯度有限、after < before / 100、框外點有負距離。例如只把那行 print 的 stride 寫錯，所有斷言照樣通過，只有印出的框不對。 第 126 行寫的「斷言的重點是 loss 下降，以及真值框換成距離再解碼回來，和原框一致」才是準確的說法。 |
| 3 | 建議 | 〈收益、代價與容易搞錯的地方〉第 138 行「或忘記框外候選點不可直接使用正距離 target」 | 「正距離 target」指什麼不清楚： - 框外點照公式算出的 target 本來就含負值（本例右邊是 −0.5 格），不是「正距離」。這句容易被讀成「框外點也能用正距離 target，只是不能『直接』用」。 - 頁尾 commit 的官方程式，對 STAL 放大小框後選到的框外點，確實拿夾成 0 的非負 target 去學。這句和官方做法的關係沒有交代。 |
| 4 | 建議 | 〈一個物件的數字旅程〉的表格與其後說明（第 25–32 行），特別是「表格列出完整 head 的輸出規格」與「距離與框實際的 shape 都是 [1,4]，這個 1 是 P」 | (a) 表格把 distance output、class logits 寫成 [B,P,4]、[B,P,C]，class logits 還寫「見 12.2 節」。但 12.2 節程式的兩個輸出是卷積的 [B,4,H,W]、[B,C,H,W]（印出 (2,4,4,4)、(2,2,4,4)），本頁沒說 [B,P,·] 是把 H、W 攤平成 P、再把 channel 移到最後的排法。Ultralytics Detect 也把 channel 放在候選軸之前：訓練時 boxes 是 (B,4×reg_max,A)、scores 是 (B,nc,A)；推論輸出 (B,4+nc,A)，非 end2end 時框是 xywh（decode_bboxes 的 xywh=not self.end2end and not self.xyxy）。 (b) 表格的「候選點」那一列不是 head 的輸出，和「完整 head 的輸出規格」的說法不合。 (c) 程式的 gt 也是 [1,4]，但它的 1 是真值框的個數，不是 P。這句緊接在「只有一個點和一個真值框」之後，「框」容易被讀成真值框。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 讀者審查 | 「整條路的單位都確認過」與實際斷言不符（學得的框沒有斷言） | 已修正：程式不在可改範圍內，所以縮小說法：斷言確認「真值框 → 距離 target → 解碼回原框」的方向與單位，以及 loss 降到初值的百分之一以下；並註明學得的框只印出來、沒有斷言，要自己比對。參考答案補上第四行應接近 [12,16,40,36]；若漏改那處 decode(..., 8)，會印出約 [2,10,58,50]，程式照樣通過。這兩個結果都在暫存副本實測過。 |
| 2 | 讀者審查 | 開頭說「只確認兩件事」，程式實際核對三件 | 已修正：改成「實驗確認三件事」，第三件是「框外的點需要負距離」。 |
| 3 | 讀者審查 | 「表格列出完整 head 的輸出規格」不精確，「不是四邊中的一項」讓人摸不著頭緒 | 已修正：改成「完整 head 用到的候選點，以及 head 輸出的規格」；最後一句改寫成：[1,4] 的 1 是 P、4 是四個數，真值框 gt 的 1 則是真值框的個數。 |
| 4 | 讀者審查 | softplus 在定義之前就出現 | 已修正：第一次出現時加括號：把任意實數變成正數的函數，定義見下方〈真的更新四個距離〉。 |
| 5 | 讀者審查 | YOLOv8 把 CIoU 的負值截成 0，看似和 11.4 節的常見錯誤衝突 | 已修正：刪掉這個夾在句中的長括號，改成指向 12.3 節摺疊區「本例和官方實作差在哪」。CIoU 截成 0、只用來排名，那裡已有完整說明。本頁也就不再出現「第 11 章〈IoU 類 loss〉」與「11.4 節」兩種稱呼。 |
| 6 | 讀者審查 | 「下一段說的動態 assignment」在下一段找不到說明，12.3 節也沒有這個名稱 | 已修正：改成直接定義：每個訓練步都依模型當下的輸出重新挑點的做法叫動態 assignment；YOLOv8 用的是 task-aligned assignment（任務對齊分配），由 12.3 節示範。 |
| 7 | 讀者審查 | 「softplus 的斜率是 sigmoid(x)」缺推導 | 已修正：補一行導數：ln(1+exp(x)) 的導數是 exp(x)/(1+exp(x))＝1/(1+exp(−x))。 |
| 8 | 讀者審查 | 沒說小誤差為什麼不用純 L1 | 已修正：補一句：L1 在 d=0 有折角，斜率從 −1 直接跳到 +1，接近 target 時仍走一樣大步，會在兩側來回跳；平方段的斜率是 d，越接近 target 步子越小。 |
| 9 | 讀者審查 | 對稱 target 那句來得突兀；沒說點為什麼刻意偏離框中心 | 已修正：在「這個實驗的斷言多確認了」那段補上：本例刻意讓點不在框中心，若左右距離相等，把左右寫反也看不出來。摺疊區裡對稱 target 那句是真話，用來對照前文「學成不對稱的四邊長度」，刪不刪屬於風格判斷，所以保留。 |
| 10 | 讀者審查 | 「重新聚類先驗」用語突兀、句子擠 | 已修正：改成「資料的長寬比改變時，不必再用第 9 章的尺寸聚類重新挑 anchor 尺寸」。 |
| 11 | 讀者審查 | 「12.4 節的 DFL 用 16 個 bin、最多 120 畫素」少了關鍵一步，出處也不對 | 已修正：改成「Ultralytics 的 DFL 預設每條邊 16 個 bin，代表 0～15 格（12.4 節說明），stride 8 時每條邊最多 15×8＝120 畫素」，和 12.4 節的說法一致。 |
| 12 | 讀者審查 | 「框外候選點不可直接使用正距離 target」句意難懂 | 已修正：改成具體的錯誤：讓框外的候選點也負責這個框；它的四個距離至少一項是負的（本例 r＝−0.5 格），本節的 softplus 輸出學不到。 |
| 13 | 讀者審查 | 延伸題的題目和答案放在同一個摺疊區，也沒說要加到幾步 | 已修正：題目移到摺疊區外，請讀者先預測 80 步夠不夠；答案另放一個「延伸題的參考答案」摺疊區，並寫明改成例如 range(100) 就會通過。暫存副本實測：80 步斷言失敗，100 步通過，餘裕很大。 |
| 14 | 讀者審查 | 圖上沒有格的編號，也沒標框中心 | 已修正：12-anchor-free.svg 下方加 column 0～7、右側加 row 0～7，框中心 (26,26) 加綠色小十字與標籤，圖例往下移，desc 同步更新；已渲染檢查畫面。正文第一次提到 (5,3) 時補上「column=5、row=3」。 |
| 15 | 技術查核 | 頁尾固定版本的官方程式會先放大小框再挑點，和「在框內的點中」不一致 | 已修正：已確認：固定 commit 是 Ultralytics 8.4.171，tal.py 的 stride_val＝stride[1]，select_candidates_in_gts 會放大小框；YOLOv8 也經由 v8DetectionLoss 套用。原句後面改成指向 12.3 節摺疊區的官方細節，那裡寫了小框放大。第 9 行與「候選密度」不另改：小框放大後，那些點到原框仍有負距離，所以「沒有點能用四個正距離表示它」仍然成立。 |
| 16 | 技術查核 | 「實驗確認了什麼」前後說法不一致，也和斷言對不上 | 已修正：和讀者審查的同一問題一起改：開頭改成三件事；後段縮小成實際有斷言的範圍，並寫明學得的框沒有斷言。 |
| 17 | 技術查核 | 「正距離 target」指什麼不清楚 | 已修正：已改寫成具體錯誤，並限定在本節 softplus 的設計下。官方把負距離夾成 0 的做法，12.3 與 16 章已經交代，本頁不重複。 |
| 18 | 技術查核 | [B,P,·] 與卷積 [B,·,H,W] 的關係沒說；候選點那列不是 head 輸出；gt 的 1 不是 P | 已修正：(b)(c) 在本頁修正（表格說明句與 [1,4] 那句）。(a) 的對照句放在 12.2 節，也就是卷積排法第一次出現、讀者需要對照的地方：16 個位置就是 P，把 channel 移到最後就是 [B,P,4]、[B,P,C]。 |
| 19 | 先前查核 | 導言「只確認兩件事」與程式的三個檢查矛盾 | 已修正：照要求的修正 (b) 改成三件事，最後一件是「框外的點需要負距離」。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/12-anchor-free.md`、`docs/lessons/12-decoupled-head.md`、`docs/lessons/12-assignment.md`、`docs/lessons/12-dfl.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」，是執行紀錄區塊前的最後一段；連到的 Ultralytics 441632c head.py、tal.py 都可取得。沒有其他標籤殘留，正文兩處「頁尾連結」說的就是這一行。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：先前查核 1 項、讀者 14 項（含 1 項必要）與技術 4 項，都在〈定稿修正〉處理；技術查核列出 Ultralytics 441632c 的 head.py、tal.py；批次檢查掛在本頁。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/12-anchor-free.md 第 41、72 行 | 殘句：「建置與執行：在暫存副本（…/暫存副本）用 zensical build…」「方法：讀過的檔案（都在暫存副本：」（括號沒有關）。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 72 行沒關的括號已改好；但第 41 行點名的殘句沒改，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/12-anchor-free.md 第 41 行 | 點名的「建置與執行：在暫存副本（…/暫存副本）用 zensical build…」逐字還在，處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[evolution當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/evolution.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/evolution.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/evolution.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_a` 實讀本頁 5 個凍結單元；首次使用／前文方法範圍四題位置為 12-anchor-free/00:first_use, 12-anchor-free/01:first_use, 12-anchor-free/03:first_use，頁末為 12-anchor-free/04。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_a.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_a-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_a.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁未有需要改寫的已裁定問題，保留原教學內容；仍完整重讀與核對。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。
