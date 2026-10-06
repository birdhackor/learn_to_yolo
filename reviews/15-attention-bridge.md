# 審查紀錄：Feature map 到 attention

審查範圍：`docs/lessons/15-attention-bridge.md`、頁面上的圖（`docs/assets/diagrams/15-attention-bridge.svg`），以及 `lesson_cases/15-attention-bridge.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要問題，有三個建議，都是用詞涵蓋的範圍沒寫準，頁面對程式行為的敘述本身都正確。所有程式都只在暫存副本裡執行，沒有在 repo 內跑任何東西。

1. 程式敘述
- 課程程式 exit 0，七行輸出和頁面的核對清單一致。第 7 行是 loss=0.1595，正文沒有寫這個數字。
- 練習 2 照頁面的步驟做，只把 target 那一行改成 `target = feature`（probe/ex2.py）。結果是照常印出前六行，接著停在 `assert not torch.allclose(grad_q, grad_k)`，報 AssertionError，第七行沒有印出，和參考答案完全相同。
- 漏寫 `.view` 的版本（probe/noview.py）也停在同一條斷言。
- 數值探針 probe/sym.py：
  - 程式用的目標，四個 token 是 [1,0]、[0,0.5]、[1,0.5]、[0,0]。三塊梯度兩兩差 1.96e-2、0.169、0.187；更新一步後，三塊投影也兩兩不同。
  - target=feature 時，Q、K 梯度在 float64 差 1.7e-18，在 float32 差 9.3e-10。
  - 漏寫 `.view` 時，目標 token 是 [1,0]、[0,0.5]、[1,1]、[0,0]；Q、K 梯度在 float64 差 3.5e-18，在 float32 差 2.8e-9。
  - 這兩種情況下，`torch.equal` 都判成 False，allclose 都判成 True，三塊梯度也都不全為 0。頁面上對應的敘述全部成立，包括「改用 `torch.equal` 可能照樣通過」那句。
- 交叉引用都查過，確實存在：第 2 章的「不全為 0」與「比較更新前後的副本」、4.2 節的 allclose、3.1 的 broadcast、notebook 的「本節可修改的完整實驗」標題。notebook 最後一格和程式逐字相同。

2. 先前審查意見與受程式改動影響的段落
- 每一項都處理了：Colab 連結已經是 lessons-v0.4.0；第 22、120、125、126 行都改寫了；舊的「Q、K 梯度相同、只有 V 不同」整段已刪除，換成 channel1 乘 0.5 的理由、漏寫 `.view` 的反例、allclose 和 equal 的分工，以及練習 2。頁尾紀錄區塊留給 verify_curriculum.py 重新產生。
- 全頁沒有修訂或審查過程的敘述。出現的「原本」「改成」都是指讀者做練習時自己改程式。

3. 數字
- 新加的數字都是固定值（lr=0.1、8 個值、目標 token、約 7 位有效數字），沒有一個是在本機量出來的。
- 要等紀錄重新產生的值，editor 都列在待重錄數值清單裡了。

4. 摘錄
- 摘錄比對工具的輸出是 `docs/lessons/15-attention-bridge.md []`。
- 檢查器只比對每一行程式碼有沒有出現在原檔，不管順序。所以我另外逐行核對：兩個 data-excerpt 區塊和程式第 12 行、第 43–58 行連續且順序相同，只有註解不一樣。
- 第 61 行沒有標記的區塊和程式不同，第 68 行已經說明差在哪裡。
- 正文沒有寫程式行號。

5. 建置與 SVG
- 暫存副本中 `zensical build --clean --strict` 和 `validate_site.py` 都 exit 0，log 在 probe/build.log 和 probe/validate_site.log。
- 兩個 data-excerpt 區塊都渲染成 Python 語法高亮；練習 1、練習 2 的標籤確實成為 `<strong>`。
- 全站沒有任何連結指向本頁的錨點。新標題讓「實際執行紀錄」的錨點從 #_4 變成 #_5，在 repo 內不影響任何連結。
- validate_lessons.py 已經通過摘錄檢查，最後停在審查覆蓋：59 頁都是 no review，因為 reviews/coverage.json 還不存在。這是發布前預期的狀態，審查用的事實與寫作規範清單有列出。
- SVG 沒有修改。qlmanage 渲染正常（probe/render/15-attention-bridge.svg.png），viewBox、title、desc 都在，「Q＝K＝V＝token」和 0.3349、0.1651 與程式一致。
- 這位 editor 只改了 docs/lessons/15-attention-bridge.md。工作樹裡 15-area-attention.md 的改動是那一頁自己部分的工作。

三個建議都很小，各改一句就能解決，見下表欄位。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/15-attention-bridge.md 第 144 行最後一句「這些斷言不表示 attention 學到了對偵測有用的東西，只證明加權讀取、還原和 loss 都在同一張計算圖上。」 | 這句的範圍是舊程式寫的：舊程式只有「梯度非零」和「權重有改變」兩條斷言，兩條都只是在查梯度有沒有接通。現在多了「三塊梯度兩兩不同」，而「在同一張計算圖上」涵蓋不了這件事。句中用了「只證明」，等於把它排除在外，結果和同一頁的兩處對不上：第 22 行說實驗驗證「Q、K、V 三塊投影收到的梯度各不相同」，第 146 行說「直接用斷言核對三塊梯度兩兩不同」。讀者讀到這句，會以為兩兩比較的斷言也只是在查梯度有沒有接通。下一段會說明它真正的用途，所以不至於讓人做錯事。 |
| 2 | 建議 | docs/lessons/15-attention-bridge.md 第 146 行「這時 Q、K 兩塊收到的梯度完全相同，更新一步後 Q、K 的投影也仍然相同。」 | 「完全相同」只在數學上成立。暫存副本實測 target=feature 時，float32 算出的 grad_q 和 grad_k 差 9.3e-10，`torch.equal` 回傳 False；更新一步後兩塊投影差 5.8e-11。第 148 行和練習 2 參考答案都講了 float32 的捨入差，第 146 行卻先用「完全」把話說死。讀者做練習 2 時若用 `torch.equal` 自己核對，會先以為頁面寫錯，要讀到下一段才弄清楚。 |
| 3 | 建議 | docs/lessons/15-attention-bridge.md 第 127 行摘錄裡的 `loss = F.mse_loss(output, target)`，以及第 118、142 行 | 這是新加的摘錄，裡面用到變數 `output`，但頁面從沒交代過它。前面只出現過 `output_tokens`（shape [1,4,2]）；第 118 行講到還原成 [1,2,2,2] 時，沒說那個結果叫什麼名字。第 142 行說「輸出與目標的 8 個值」，可是 `output_tokens` 和還原後的輸出都剛好是 8 個值，這句分不出讀者看的是哪一個。初學者可能以為 `output` 就是 `output_tokens`。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py（Bottleneck、C3、C2f、C3k、C3k2）
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/conv.py（Conv：bias=False、BatchNorm2d、default_act=SiLU）
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/11/yolo11.yaml（backbone/head 的 C3k2、SPPF、C2PSA）
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/tasks.py（parse_model：m/l/x 尺度把 C3k2 的 c3k 強制設為 True）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/block.py（AAttn、ABlock、A2C2f）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/conv.py（Conv）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/cfg/models/v12/yolov12.yaml（第 6 層 A2C2f area=4、第 8 層 P5/32 area=1，head 的 A2C2f a2=False）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/tasks.py（parse_model：A2C2f 的 n 插在 args[2]）
- https://arxiv.org/abs/2502.12524（Submission history：只有 v1）
- https://arxiv.org/html/2502.12524 §3.2 Area Attention（分成 l 段，(H/l,W) 或 (H,W/l)；只需一次 reshape、速度較快；預設 l=4，感受野變 1/4 但仍然夠大）、§3.4 Architectural Improvements（移除 positional encoding，改用 7×7 large separable convolution『position perceiver』）
- https://arxiv.org/abs/1706.03762 → https://arxiv.org/html/1706.03762 §3.2.1 Scaled Dot-Product Attention 正文（We suspect…）與腳註（變異數 d_k）
- https://ar5iv.labs.arxiv.org/html/1512.03385 §4.1 Deeper Bottleneck Architectures（1×1、3×3、1×1 三層）

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈為什麼除以 √d〉摺疊區最後一段（第 107 行「出處：Attention Is All You Need 3.2.1 節（Scaled Dot-Product Attention）的腳註」），以及同一摺疊區「這時 softmax 已經飽和：…梯度也接近 0，Q、K 很難學」那一句 | 論文 3.2.1 的腳註只寫了一件事：假設 q、k 的元素是獨立、平均 0、變異數 1 的隨機變數，內積的平均是 0、變異數是 d_k。「d_k 大時內積變大，把 softmax 推到梯度極小的區域」寫在 3.2.1 正文，而且原文用 "We suspect"，表示這是作者的推測；它的實證依據是引用 [3]：d_k 大時，不縮放的 dot-product attention 輸給 additive attention。d=64 時 softmax([8,0]) 的數值例，以及「為什麼不除以 d」，則是本頁自己的推論。現在出處只標「腳註」，又把「飽和→梯度接近 0→Q、K 很難學」寫成確定的因果。讀者會以為整個摺疊區都出自腳註，也會以為論文已證實這個機制。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 來源對照 | √d 摺疊區只標「腳註」為出處，把作者的推測寫成確定的事 | 已修正：已對照論文 3.2.1：正文說「We suspect」，腳註只有變異數推導。出處改成：正文是作者的推測（d 大時內積變大，把 softmax 推到梯度極小的區域，所以乘上 1/√d）；變異數推導出自腳註；d=64 的數值例與「為什麼不除以 d」是本頁的補充。數值例本身是正確的局部計算（softmax([8,0]) 的梯度約 3e-4），所以沒改。 |
| 2 | 先前查核 | 「只證明……在同一張計算圖上」漏掉了三塊梯度兩兩不同的斷言 | 已修正：補上後半句：「而且 Q、K、V 三塊投影收到的梯度各不相同」，和同頁其他兩處一致。 |
| 3 | 先前查核 | 「Q、K 梯度完全相同」把話說死，沒顧到 float32 的捨入 | 已修正：在暫存副本實測 target=feature：梯度差 9.3e-10，torch.equal 回傳 False；更新後投影差 5.8e-11。改成「數學上完全相同」，並加註「float32 計算時只差捨入誤差，見下一段」。 |
| 4 | 先前查核 | 摘錄裡的 `output` 在頁面上沒有交代 | 已修正：在「還原圖片佈局」段補上完整程式的寫法 `output = output_tokens.transpose(1, 2).reshape_as(feature)`，說明 reshape_as 的意思，並指明下面的 output 就是它。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/13-dual-assignment.md`、`docs/lessons/13-nms-free.md`、`docs/lessons/14-feature-module.md`、`docs/lessons/15-attention-bridge.md`、`docs/lessons/15-area-attention.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」；yolov12 2abab71 的 block.py 回 200，AAttn 的括號說明正確。術語表引用的權重 [0.3349,0.1651,0.3349,0.1651] 與本頁第 55、82 行一致。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：先前查核 3 項與來源對照 1 項，都在〈定稿修正〉處理；批次檢查掛在本頁；〈後續編輯的檢查〉齊全；結構檢查通過。沒有發現問題。

### 第 4 輪：上一輪的處理：通過

第 3 輪沒有發現，沒有處理說明需要核對。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[modern當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/modern.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-05：v0.6.0 有界更新審閱

新增ViT選讀入口及主線15.2續讀導覽；兩首讀者讀實際原段，導讀及第三輪核新分岔。

本次僅重查以上變更，既有正文的歷史審閱保留；沒有把全頁或全書重新標成首次盲讀。[導讀／實際網站審閱](clear-tutorial/vision-v0.6.0/guide-visual-review.md)、[新增支線第三輪及實際前置](clear-tutorial/vision-v0.6.0/transitions-review.md)、[首讀修後複查](clear-tutorial/vision-v0.6.0/vit-recheck.md)記錄各自範圍。全版52本notebook的本機cell執行另見[實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)；不是Google Colab登入執行。來源與圖／程式指紋更新於[coverage.json](coverage.json)。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_b` 實讀本頁 6 個凍結單元；首次使用／前文方法範圍四題位置為 15-attention-bridge/00:first_use, 15-attention-bridge/01:first_use, 15-attention-bridge/02:first_use，頁末為 15-attention-bridge/05。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_b.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_b-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_b.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS04。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。
