# 審查紀錄：人工框評估與 AP50

審查範圍：`docs/lessons/06-evaluation.md`、頁面上的圖（`docs/assets/diagrams/06-evaluation.svg`），以及 `lesson_cases/06-evaluation.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要問題，只有一個 should：第 144 行的 `clean_result["map"]` 少一句說明。

查證全部在暫存副本裡做，沒有在 repo 裡執行任何東西。

1. 正文對程式的描述都正確。
   - 程式結束碼 0。第 124、126、144 行引用的三行輸出，和實際輸出的第 5–7 行逐字相同（`TP=2, FP=2, FN=1, mAP50=0.333333, ap_per_class=[0.333333, None]`、`candidate threshold .85: mAP50=0.166667, recall=0.3333`、`remove known high-score FP: mAP50=0.555556`）。
   - 「前四行依排名印出 score、判定、P、R」和實際輸出、表格都對得上。`cleaned`、`clean_result`、`ap_per_class`、`evaluate`、`num_classes=2`、`match_iou_threshold=.5`、`interpolated_ap` 都在程式裡。
   - 練習照頁面做：拿掉 0.95 那筆後，用 evaluate() 實算得到判定 TP、FP、TP；P 依序是 1、1/2、2/3；R 依序是 1/3、1/3、2/3；AP 是 5/9。完整程式也確實印出 0.555556。我在副本另存一份把 `cleaned` 改成不刪框的程式，執行後停在第 96 行的 5/9 斷言，表示正文說的斷言確實守著這個值。
   - 新加的兩類例子（AP50 是 1/3 和 1，mAP50 是 2/3）：我加了第三張圖，放一個類別 1 的 GT 和一個完全重合的預測，實跑 mAP 是 0.6667。
   - 其他手算說法也都用程式實跑過，全部成立：第 51 行刪掉各筆後的 P、R；第 100 行多加一筆低分 FP 後 AP 仍是 1/3；第 114 行 11 點平均 0.318；第 118 行 None 與 0 的規則；摺疊區的 VOC 數字例。
2. 兩條先前審查意見和受程式改動影響的段落的各項都處理了。
   - 第 3 行 Colab 連結在 HEAD 已經是 lessons-v0.4.0。
   - 第 124 行的更正句已刪掉。
   - 第 126、144 行照 (b) 改寫。
   - 第 154、163–165 行在執行紀錄標記以下，沒有手改。
   - 我用關鍵字搜過全頁，沒有修訂經過的敘述。第 120 行的「其實是 mAP」講的是 COCO 的名稱，不是替程式缺陷打的補丁。
3. 數字：新引用的全是固定不變的值，沒有從這台 Mac 帶進會隨機器改變的數字。等新紀錄產生才會更新的值，編輯都列出來了。
4. 程式摘錄：摘錄比對工具印出 `docs/lessons/06-evaluation.md []`。頁面唯一的 Python 區塊是虛擬碼，正文和程式註解都寫明了。正文沒有引用程式行號。
5. 好不好讀：mAP50 在第 118 行先定義，後面才使用；`ap_per_class` 有說明；沒有讀者會被誤導或卡住的地方。唯一可改的就是上面那個建議。
6. 建置與圖：
   - `zensical build --clean --strict` 結果「No issues found」，`validate_site.py` 結束碼 0；改過的段落在 HTML 裡顯示正常。
   - SVG 沒改，用 qlmanage 轉圖顯示正常，viewBox、title、desc 都在。圖上畫的是類別 0 的 PR 曲線，「AP50＝1/3」仍然正確。
   - 06-evaluation 相關的檔案只有頁面和 HEAD 不同（4 行刪、4 行增），執行紀錄標記以下逐字沒動。

給維護者的備註（不影響判定）：頁尾執行紀錄還是舊的 `AP50=` 輸出；notebooks/06-evaluation.ipynb 最後一格的輸出目前是空的。審查用的事實與寫作規範清單把這兩件列為發布前的暫時狀態，重產紀錄後就會一致。編輯的疑慮說 notebook 最後一格「要等重建才會更新」，這句只對一半：那格的程式碼已經和現行的 lesson_cases 完全相同（validate_lessons 的比對會通過），缺的只有輸出。

檢查用的檔案都在暫存副本：
- 執行輸出
- 建置與檢查紀錄
- 驗證腳本
- 改掉刪框的程式和它的輸出
- SVG 轉出的圖

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/06-evaluation.md 第 144 行（〈自主練習與答案〉的參考答案） | 這次修改新加了 `clean_result["map"]`，可是頁面沒有說 `"map"` 是什麼。全頁沒介紹過 `evaluate` 回傳的結果長什麼樣子；我查了 00 到 06 各頁，正文裡也是第一次出現 `x["key"]` 這種取值寫法。程式新手可能把 `map` 看成 Python 內建的 map 函式，或「地圖」的意思，要靠下一句的「mAP50 就是這題的 AP」才倒推得出它就是 mAP。讀者還是讀得下去，不會被誤導，所以只列建議。另外「評估它的結果存在 `clean_result`」裡的「存在」也可能先被讀成「存在（exist）」。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html — 3.4.1 Average Precision (AP)（錨點 SECTION00044100000000000000，含「prior to 2010 … 0,0.1,…,1」「VOC2010-2012 … all unique recall values」）、4.4 Evaluation（錨點 SECTION00054000000000000000，含 multiple detections 與 difficult 的說明）、2.5 Ground Truth Annotation 與 10.x 的 difficult 說明
- http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar — VOCcode/VOCevaldet.m（ovmax／jmax 迴圈、diff／det 判斷、npos 計算、+1 像素 IoU）、VOCcode/VOCap.m（補 (0,0)／(1,0)、右側最大值包絡、在 recall 變化處累加面積）
- https://cocodataset.org/#detection-eval（頁面原始檔：https://github.com/cocodataset/cocodataset.github.io commit 5e1c4da72464b1c6f068df0c02c91e3000ea62c4, dataset/detection-eval.htm）— 2. Metrics 註 1（10 個 IoU 門檻 .50:.05:.95）、註 2（不區分 AP 與 mAP）、註 6（每張圖跨所有類別最多 100 個偵測）、評估參數 iouThrs／recThrs（R=101）／maxDets
- https://github.com/cocodataset/cocoapi commit 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9, PythonAPI/pycocotools/cocoeval.py — L108-L109（crowd→ignore）、L163-L176（computeIoU 逐圖逐類截斷 maxDets）、L251-L294（evaluateImg：GT 依 ignore 排序、dt[0:maxDet]、跳過已配對非 crowd GT 後取 IoU 最高者、dtIg=gtIg[m]）、L372-L405（npig 排除 ignore GT、右往左取最大值的包絡、searchsorted 讀 101 個 recall 點）、L452-L455（只平均 >-1 的項目）、L506-L507（iouThrs、recThrs 定義）
- https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html — 函式說明（移除與較高分框 IoU > iou_threshold 的框；回傳索引依分數遞減排序）
- https://pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html — 函式說明（idxs 為每個框的類別索引；不同類別之間不做 NMS；回傳索引依分數遞減排序）

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 「本書這套配對規則是教學簡化版…原始規則可讀 Pascal VOC 官方評估說明（…4.4 節…）與 COCO detection evaluation」一段，以及其後的摺疊區〈本書配對與官方 VOC／COCO 的差別〉：第 1 段（VOC 配對順序）、第 3 段（pycocotools 配對順序）、ignore 與「候選數上限」兩個條目 | 這些說法經核對都正確，但讀者照頁面連結去查，找不到依據。(1) VOC2012 開發套件文件 4.4 節只寫了兩件事：同一物件的多個偵測算錯誤偵測；difficult 物件目前不納入評估。文件沒有寫「先在全部 GT 中找 IoU 最高的，再檢查它是否已被配對」這個順序，也沒寫 difficult 不算進 GT 總數、配到它的預測不算 TP 也不算 FP。這些只寫在 devkit 的 VOCevaldet.m 裡：ovmax 迴圈先跑過同圖同類的全部 GT，之後才檢查 gt(i).diff(jmax) 與 gt(i).det(jmax)；GT 總數則是 npos=npos+sum(~gt(i).diff)。(2) 另外三點只寫在 cocoeval.py 裡：「pycocotools 先跳過已配對的 GT，再取 IoU 最高的」「每張圖、每個類別最多計 100 個」「crowd 視為 ignore」。COCO 網站寫的反而是 "at most 100 top-scoring detections per image (across all categories)"。所以讀者照連結查證時，會讀到和本頁相反的說法，卻找不到本頁說法的出處。07-heldout.md 的「配對順序」條目也指向這裡，同樣受影響。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 參考答案的 clean_result["map"] 沒說是什麼，「存在」也可能讀成「存在（exist）」 | 已修正：已確認 evaluate 回傳的是 dict，"map" 是 mAP 欄位。改成「評估它的結果存進 `clean_result`，斷言核對其中的 mAP 欄位 `clean_result["map"]` 是 5/9」。實跑 06-evaluation.py 印出 mAP50=0.555556，和頁面一致。 |
| 2 | 來源對照 | VOC／COCO 配對順序、difficult 計數、crowd 當 ignore、每類 100 個上限，讀者照頁面連結查不到出處 | 已修正：已下載官方 VOCdevkit_18-May-2011.tar，也取得固定 commit 的 cocoeval.py，逐項核對：VOCevaldet.m 先跑完全部 GT 求 ovmax／jmax，再看 diff、det，npos 只算非 difficult 的物件；evaluateImg 依類別用 dt[0:maxDet] 截斷，配對時用 gtm 跳過已配對且不是 crowd 的 GT；_prepare 把 crowd 設成 ignore。頁面改了兩處：第一處，「原始規則可讀…」改成「規則文字見…；配對順序、difficult 怎麼計數、候選數上限這類細節，以官方評分程式為準」；第二處，摺疊區補一段出處，附 devkit tar 的連結與 cocoeval.py 固定 commit 的 #L235-L296、#L106-L109。行號範圍沒照建議用 L251-L294，因為那一段沒有含到配對的寫入（dtm、gtm）。07-heldout 的「配對順序」條目只補一個短語，指出出處在 06-evaluation。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/06-decode-nms.md`、`docs/lessons/06-evaluation.md`、`docs/lessons/07-heldout.md`、`docs/lessons/08-own-images.md`、`docs/lessons/08-own-data.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/06-evaluation.md 第 55 行「配對順序、difficult 怎麼計數、候選數上限這類細節，以官方評分程式為準」 | difficult 第一次出現在正文，就是這次加的這句，但頁面沒有說它是什麼。摺疊區第 69 行只說 difficult 物件怎麼計分，也沒有說它的意思。摺疊區預設收合，所以不展開的初學讀者只會看到一個沒解釋的英文字。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

下載 VOCdevkit_18-May-2011.tar（頁面網址 301 轉到 thor.robots.ox.ac.uk，可用）。VOCevaldet.m 第 32–40 行的 npos 只加非 difficult；第 75–104 行先找 ovmax／jmax，再看 diff 與 det，配到 difficult 的預測不算 TP 也不算 FP，與頁面一致。VOC2012 文件的 #SECTION00054… 錨點正是 4.4 Evaluation，內文也指向 3.4.1，「difficult（難以辨認）」符合 2.5 節定義。pycocotools 8c9bcc3：L106–L109 是設定 ignore 旗標，L235 起是 evaluateImg，裡面有 `dtind[0:maxDet]` 和「已配對且不是 crowd 就跳過」，錨點與說明正確。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

讀了 reviews/06-evaluation.md 並比對定稿修正的處理清單。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/06-evaluation.md〈來源對照〉與〈定稿修正〉 | 來源對照那項 should（VOC／COCO 配對順序、difficult、crowd、每類 100 個上限照頁面連結查不到出處）沒有處理列；〈定稿修正〉只列先前查核那一項。這項的處理在 fix:b3 裡，page 寫成「docs/lessons/06-evaluation.md（連帶 07-heldout.md）」，生成器沒比對到。 | 已修正：產生器去掉「（連帶 …）」後比對頁面，〈定稿修正〉列出這一項的處理，07-heldout 的紀錄也列入同一項。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：先前查核 1 項與來源對照 1 項都在〈定稿修正〉處理（上一輪指出來源對照那項缺處理，已補，07-heldout 也列入同一項）；cocoeval.py（8c9bcc3）與 VOCdevkit 都在來源清單；〈後續編輯的檢查〉齊全。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/06-evaluation.md 第 35–42、92 行 | 內部名稱與殘句：「給派工者的備註（不影響判定）」、一串空洞的路徑清單「- 執行輸出：暫存副本」「- 建置與檢查紀錄：暫存副本、暫存副本」「- 驗證腳本：暫存副本」「- 改掉刪框的程式和它的輸出：暫存副本、暫存副本」「- SVG 轉出的圖：暫存副本06-evaluation.svg.png」，以及〈上述處理〉摘要「比對 lean 工作流程的處理清單」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 35 行改成「給維護者的備註」；第 37–42 行改成檔案類別清單；第 92 行改成「比對定稿修正的處理清單」。處理說明屬實。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/foundations.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `detector` 實讀本頁 7 個凍結單元；首次使用／前文方法範圍四題位置為 06-evaluation/00:first_use, 06-evaluation/01:first_use, 06-evaluation/02:first_use, 06-evaluation/03:first_use, 06-evaluation/04:first_use，頁末為 06-evaluation/06。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/detector.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/detector-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/detector.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R014、R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-applications.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS04。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 b 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `b_foundation` 實讀本頁 9 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/b_foundation/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：無額外局部修正。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/06-evaluation.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
