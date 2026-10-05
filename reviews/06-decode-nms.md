# 審查紀錄：人工框解碼與 NMS

審查範圍：`docs/lessons/06-decode-nms.md`、頁面上的圖（`docs/assets/diagrams/06-decode-nms.svg`），以及 `lesson_cases/06-decode-nms.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題。只有一條選擇性的 should：torch.arange 沒有說明，這是改版前就有的。

所有查核都在我自己的暫存副本做，輸出放在其中的暫存副本。

1. 頁面對程式的敘述都成立。
   - 程式結束碼 0，stderr 空，印出 8 行。
   - 練習照頁面做：不改程式，直接讀 `exercise NMS .6 …` 和 `exercise score .70 …` 兩行，得到 [2, 1, 0] 和 [0.72, 0.855]，和參考答案一致。
   - 頁面上的說法我另寫 replicate.py 逐一重算，結果都和頁面相同：
     - 依分數排序是 [2,1,0]。紅框和藍、黃兩框的 IoU 都是 0；藍黃是 0.5385，不超過 0.6。
     - boxes70[0] 等於 boxes[1]，boxes70[1] 等於 boxes[2]；labels70 是 [0,0]。門檻 0.75 時只剩 boxes[2]。
     - 格號表的 shape 是 [4,4,2]，位置 [1,2] 取出 (2,1)，加上 sigmoid 後是 (2.25, 1.75)。
     - 2×2 小例子的結果正確；改用 indexing="xy" 時兩張表確實會對調。
     - σ(−12) ≈ 6.1e-6；allclose 的實際容許差是 8.2e-6 到 9.55e-6。

2. 先前審查意見和受程式改動影響的段落的每一項都處理了。
   - line 3：頁面已是 lessons-v0.4.0，validate_lessons 的 source_ref 檢查通過。
   - line 63／67：改採先前審查意見提的替代做法（逐字摘錄 decode()，在正文解釋 stack）。
   - line 230：那行 assert 已放進摘錄，開頭說明也縮短了。
   - impact 212、233、234 已改；252、257 屬自動產生的紀錄區塊，等重產。
   - 舊名殘留：整個 docs/ 已經沒有 cell_xy、keep06，也沒有「06 代表…」那句。
   - 修訂敘事：全頁沒有。頁面裡的「改成」「原本」都是在講練習的門檻或候選索引，屬教學用語。

3. 數字沒有問題。新加的數字只有三個候選框：它們是確定性的，由 expected_boxes 的 assert 守著，程式印出時先轉 float64 再取 3 位小數，換電腦也一樣。沒有從這台 Mac 量來的數字。頁尾紀錄第 1 行還是 float32 雜訊（23.200000762939453），editor 已列為要等重產的值。

4. 摘錄沒有問題。
   - 摘錄比對工具輸出 `docs/lessons/06-decode-nms.md []`。
   - 這個檢查器只看每一行有沒有出現在程式裡，不看順序，所以我另外逐行對照：decode 摘錄就是程式第 28–33 行，連續、順序相同；參考答案是 72–73 行、`...`（省略第 74 行的 print）、75–77 行。
   - 在暫存副本把兩段摘錄各改壞一行，檢查器都抓到；還原後再跑仍是 []。
   - 2×2 小例子寫明了「把 4 換成 2」，只有 2 行，不標記是對的。正文沒有出現程式行號。

5. 可讀性：順序是公式、手算、摘錄、解釋 meshgrid 和 stack、2×2 小例子，仍然好教。唯一的建議見問題清單。

6. 渲染沒有問題。
   - zensical build --clean --strict 顯示 No issues found；validate_site 通過。
   - 建好的 HTML 裡，兩段摘錄的 data-excerpt 是 HTML 屬性，沒有把 `{ .python` 當文字印出來；新加的行內數學有正確轉成數學式。
   - 06-decode-nms.svg 沒有 diff，viewBox、title、desc 都在，用 qlmanage 轉成 PNG 看起來乾淨。依 24+4.5x、52+4.5y 換算，三個框的 SVG 座標和程式算出的框一致。
   - 只有本頁被改：SVG、程式、紀錄 JSON、reviews 都沒動。42 本 notebook 的修改時間都是 00:29:27，是一次全部重建，不是這位 editor 做的。

給下一步的資訊（不算問題）：
- validate_lessons 停在「審查是否涵蓋目前內容」那一步，原因是 reviews/coverage.json 不存在，所有頁面都顯示 no review。在那之前，42 頁的 Colab tag、notebook 與程式逐字相同、摘錄檢查都通過了。
- validate_curriculum_evidence 一開始就在 00-warmup 失敗，原因是紀錄過期。這兩項都是審查用的事實與寫作規範清單列的發布前暫時狀態。
- editor 報告裡「notebooks/06-decode-nms.ipynb 還是舊的程式」這句已經過時：notebook 的最後一格現在和程式逐字相同，只剩輸出要等重產紀錄。
- reviews/06-decode-nms.md 仍有 4 處 keep06，要等重新審查本頁時處理。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/06-decode-nms.md 第 56–57 行（decode 摘錄裡 meshgrid 那行和它上面的註解） | 這段摘錄幾乎每個寫法都有註解（shape[0]、logits[..., :2]、image_size、[..., 2:4]、torch.cat），只有 torch.arange(grid) 沒說明。第 0–5 節和術語表都沒有介紹過 torch.arange，所以這是讀者第一次看到它。讀者從下方「每一列都是 [0,1,2,3]」和 2×2 小例子大致猜得出來，不會卡住，因此只列建議。這個缺口改版前就有（HEAD 第 66、77 行），不是這次編輯造成的。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html — 3.4.1 Average Precision (AP)（錨點 SECTION00044100000000000000，含「prior to 2010 … 0,0.1,…,1」「VOC2010-2012 … all unique recall values」）、4.4 Evaluation（錨點 SECTION00054000000000000000，含 multiple detections 與 difficult 的說明）、2.5 Ground Truth Annotation 與 10.x 的 difficult 說明
- http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar — VOCcode/VOCevaldet.m（ovmax／jmax 迴圈、diff／det 判斷、npos 計算、+1 像素 IoU）、VOCcode/VOCap.m（補 (0,0)／(1,0)、右側最大值包絡、在 recall 變化處累加面積）
- https://cocodataset.org/#detection-eval（頁面原始檔：https://github.com/cocodataset/cocodataset.github.io commit 5e1c4da72464b1c6f068df0c02c91e3000ea62c4, dataset/detection-eval.htm）— 2. Metrics 註 1（10 個 IoU 門檻 .50:.05:.95）、註 2（不區分 AP 與 mAP）、註 6（每張圖跨所有類別最多 100 個偵測）、評估參數 iouThrs／recThrs（R=101）／maxDets
- https://github.com/cocodataset/cocoapi commit 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9, PythonAPI/pycocotools/cocoeval.py — L108-L109（crowd→ignore）、L163-L176（computeIoU 逐圖逐類截斷 maxDets）、L251-L294（evaluateImg：GT 依 ignore 排序、dt[0:maxDet]、跳過已配對非 crowd GT 後取 IoU 最高者、dtIg=gtIg[m]）、L372-L405（npig 排除 ignore GT、右往左取最大值的包絡、searchsorted 讀 101 個 recall 點）、L452-L455（只平均 >-1 的項目）、L506-L507（iouThrs、recThrs 定義）
- https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html — 函式說明（移除與較高分框 IoU > iou_threshold 的框；回傳索引依分數遞減排序）
- https://pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html — 函式說明（idxs 為每個框的類別索引；不同類別之間不做 NMS；回傳索引依分數遞減排序）

這一頁沒有發現與來源不符的說法。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | decode 摘錄中 torch.arange(grid) 沒有說明 | 已修正：確認 0–5 節與術語表都沒介紹過 torch.arange。只在 meshgrid 上方原有的中文註解前補「torch.arange(grid) 是 [0,1,2,3]；」，摘錄比對工具結果仍是 []。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/06-decode-nms.md`、`docs/lessons/06-evaluation.md`、`docs/lessons/07-heldout.md`、`docs/lessons/08-own-images.md`、`docs/lessons/08-own-data.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核的建議在〈定稿修正〉處理；〈來源對照〉列出來源，沒有不符；批次檢查掛在本頁；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/06-decode-nms.md 第 13 行 | 殘句：「所有查核都在我自己的暫存副本做：暫存副本，輸出放在其中的暫存副本。」 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 13 行點名的殘句已刪除，處理說明屬實。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/foundations.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。
