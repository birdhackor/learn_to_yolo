# 審查紀錄：尺寸聚類

審查範圍：`docs/lessons/09-anchor-clustering.md`、頁面上的圖（`docs/assets/diagrams/09-anchor-clustering.svg`），以及 `lesson_cases/09-anchor-clustering.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

第 1 輪檢查通過，沒有必要問題，只有一條建議建議。

程式與數字：lesson_cases/09-anchor-clustering.py 和 HEAD 相同，SHA-256 1262a5b3… 和紀錄的 case_sha256 一致。在暫存副本執行，exit 0，stdout 和 artifacts/checks/curriculum/09-anchor-clustering.json 逐字相同。頁面上每個數字都用程式本身的 size_iou 與 cluster 在暫存副本重算過，全部相符：
- 第 1 輪距離表、群平均 [25/3,25/3] 與 [94/3,50/3]，以及「第 2 輪不變就停止」。
- 弱基線各值與平均 0.3817。
- 上一節 anchor 的 1、0.8889、0.8889、0.5、0.5333、0.4444，平均 0.70926；表後那句的「約 0.44～0.53」也對。
- tw/th 的 0.021、−0.041、0.077、0.693。
- 新來源的 0.170、0.225、0.1975。
- 目標總和 0.529 與 0.396，而且每個框的最佳 anchor 確實就是自己那群的 anchor。
- [1,10]/[10,1] 反例 0.9474→1.6835。
- 練習 2 的 0.941→0.255、(0.714,0.652)，anchor 乘 2 後回到 0.941。

練習 2 另外照題意用 cluster 對放大 2 倍的尺寸重跑：anchor 正好變成 2 倍，平均最佳尺寸 IoU 仍是 0.9119；不改 anchor 則掉到 0.381，和參考答案一致。斷言、變數名（sizes、groups/assignment）、註解與程式的描述都屬實。沒有引入在這台 Mac 上量出來的數字；編輯列出的紀錄相依值都是確定性輸出。

先前審查意見與受程式改動影響的段落：
- line 3：頁首 Colab 連結在 HEAD 已經是 lessons-v0.4.0，不用改。
- line 124：0.7093 那一列已移出表格。表格現在只放程式印出的四個值，分別對應輸出第 2、2、3、4 行；0.7093 移到表後，當成前面手算的結果來比較。沒有殘留「程式沒有印出」這類補丁說明。先前審查意見原本建議只留前面的手算段落，表後新增的那句是延伸，不算補丁說明。
- 受程式改動影響的段落裡沒有本頁、本頁 SVG 或本節 unit 的項目。
- 全頁找不到修訂或審查經過的敘述。

摘錄：cluster() 區塊和程式逐行相同，13 行、只加了註解，已用 data-excerpt 標記。摘錄比對工具對 repo 和暫存副本都印出 []。我把其中一行的 mean 改成 median 試過，檢查會抓到。正文裡只提到輸出的行號，沒有提程式的行號；00-warmup 也是這樣寫。

渲染：zensical build --clean --strict 和 scripts/validate_site.py 在暫存副本都通過，產出的 HTML 裡表格與 Python 區塊都正常。SVG 沒有改，viewBox、<title>、<desc> 都在，qlmanage 渲染乾淨；總覽圖與兩張放大圖的座標都對得上程式的結果。和本頁相關的檔案只有這個 .md 被改。

另外：validate_lessons 在暫存副本會失敗，原因只是全站 59 頁都還沒有定稿審查紀錄，這是審查用的事實與寫作規範清單說的發布前暫時狀態；它的摘錄檢查本身通過。

建議（should）：表前那句提到「輸出第 2 行」時，補一句說明輸出可在頁尾〈實際執行紀錄〉展開。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/09-anchor-clustering.md 第 121 行（〈執行結果與比較〉表格前新增的那一句） | 這句用「輸出第 2 行箭頭前後的兩個數」「後兩列分別在第 3、4 行」來對照程式輸出，卻沒說讀者在哪裡看得到這份輸出。正文沒有貼出輸出，頁面上唯一的輸出在頁尾摺疊的〈實際執行紀錄〉裡。沒有自己執行程式的讀者讀到這裡，得自己猜「輸出」指的是哪裡。其他頁遇到同樣情況都寫明出處，例如 03-comparison、07-heldout、11-csp 的「頁尾的執行紀錄」。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://arxiv.org/abs/1612.08242（全文為 https://arxiv.org/pdf/1612.08242v1）：Abstract；2. Better 的 Convolutional With Anchor Boxes、Dimension Clusters（含 Figure 2、Table 1）、Direct location prediction（b_x、b_y、b_w、b_h 公式）；3. Faster 的 Darknet-19；另全文搜尋 ignore／threshold，確認論文沒有描述 ignore 規則
- https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/src/region_layer.c 第 232–252 行（以預測框對任一 GT 的 best_iou > l.thresh 決定 ignore）、第 266–292 行（best anchor：truth_shift 與 bias_match，只比寬高）
- https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/cfg/yolov2-voc.cfg [region] 段：第 243 行 bias_match=1、第 249 行 rescore=1、第 257 行 thresh = .6

這一頁沒有發現與來源不符的說法。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 表格前那句說「輸出第 2 行」，卻沒說輸出在哪裡看 | 已修正：補上「（頁尾〈實際執行紀錄〉可展開本次輸出）」，數字與摘錄都沒動。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/09-anchors.md`、`docs/lessons/09-anchor-clustering.md`、`docs/lessons/10-multiscale.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核的建議在〈定稿修正〉處理；〈來源對照〉列出 YOLO9000 與 Darknet 固定 commit，沒有不符；批次檢查掛在本頁。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/09-anchor-clustering.md〈定稿修正〉最後的〈留下的意見〉表 | 唯一一列是 docs/lessons/10-multiscale.md 第 9 行的意見，不是本頁的發現。處理寫「這項修正由下方〈後續編輯的檢查〉核對」，但本紀錄沒有〈後續編輯的檢查〉，等於指向不存在的段落。 | 已修正：產生器改以位置欄最先提到的頁面歸屬意見，這一列已移回 10-multiscale 的紀錄。 |

### 第 4 輪：上一輪的處理：通過

〈定稿修正〉之後已沒有〈留下的意見〉表，10-multiscale 的意見已移走，也不再指向不存在的〈後續編輯的檢查〉；第 3 輪第 1 項處理屬實。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[evolution當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/evolution.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/evolution.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/evolution.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_a` 實讀本頁 9 個凍結單元；首次使用／前文方法範圍四題位置為 09-anchor-clustering/00:first_use, 09-anchor-clustering/01:first_use, 09-anchor-clustering/02:first_use, 09-anchor-clustering/03:first_use，頁末為 09-anchor-clustering/08。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_a.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_a-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_a.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁未有需要改寫的已裁定問題，保留原教學內容；仍完整重讀與核對。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 c1 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁相關處置：DEC-005、DEC-010、DEC-011；包含採用、保留或後文撤回的來源與理解收益。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。
