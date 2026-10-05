# 審查紀錄：IoU 類 loss

審查範圍：`docs/lessons/11-iou-loss.md`、頁面上的圖（`docs/assets/diagrams/11-iou-loss.svg`），以及 `lesson_cases/11-iou-loss.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

通過，沒有必要或建議問題。所有執行都在暫存副本進行；log、模擬腳本 sim_colab_11iou.py 與精算腳本 nums_11iou.py 都放在上一層暫存副本。

1. 程式敘述全部屬實。逐句對照定稿的 lesson_cases/11-iou-loss.py（sha ddce1264…，與 HEAD 相同）。斷言、印出內容、鍵名 'iou'／'giou'／'diou'／'ciou'（值都是 loss）都正確。shifted_center 實際斷言的是 L_IoU、它的中心梯度與 L_DIoU，頁面列的正是這三項。DIoU 斷言清單（梯度為有限數值且 center.grad[0]>0、x 中心小於 40、loss 下降）與程式一致。
   - 照 161 行在 repo 根目錄執行程式：exit 0，stderr 是空的，5 行輸出和核對清單的 [0.02,0]、(38.7515,20) 等值相符。
   - 照答案摺疊區的指示模擬 Colab：先以 __name__=='__main__' 執行完整程式，再執行從頁面直接抽出的 4 行片段。印出 {'iou': 1.0, 'giou': 1.111111, 'diou': 1.257732, 'ciou': 1.257732} 與 {'iou': 1.0, 'giou': 1.076923, 'diou': 1.264865, 'ciou': 1.266554}，第 2、3 題手算過的項目都對得上。
   - 頁面對 miniyolo 的敘述也查過：decode_grid 在 @torch.no_grad() 下執行，含 score 門檻與 NMS；total=5*box+…；第 7 章框 MSE 是 0.109375。

2. 先前審查意見與受程式改動影響的段落都處理了。
   - 第 3、161 行：HEAD 已經是 lessons-v0.4.0，與 section-map 的 source_ref 一致。
   - 163 行：程式沒改鍵名，這句是現在式的讀法說明，內容為真，保留是對的。
   - 202 行：長括號已拿掉，題目只留「求 v、α 與 L_CIoU」，操作說明移到參考答案摺疊區。措辭和 11.3（11-augmentation.md:163）相同；「新增一格」在 03-comparison.md:151 已經教過。
   - 208 行：只列實際存在的斷言，L_GIoU 改用片段核對，沒有聲稱不存在的斷言。
   - impact 165 的括號已刪除；216、221、224 屬於 verify_curriculum.py 自動產生的區塊。
   - 全頁沒有修訂或審查敘事。grep 命中的「改成 0」講的是 clamp，「改寫自完整程式的示意」講的是示意程式，都不是修訂痕跡。

3. 數字正確。
   - 用 Fraction 精算：43008/3444736＝21/1682≈0.012485；精確步長 1.248514；40−100·21/1682＝38.7514863，取 4 位是 38.7515；間隙 6.7515（頁面寫約 6.75）；L_DIoU 1.294497；1.179487、0.0205；第 2 題 1.111111、1.257732；第 3 題 v≈0.041956、α≈0.040267、αv≈0.001689、L_CIoU≈1.266554。全部相符。
   - 沒有從這台 Mac 帶進機器相關的數字。0.02 與 38.7515 的計算路徑不經過 atan，只有 IEEE 的確定性運算；y 分量在 tie 時對半分，所以是確定的 0。
   - 需要等重錄的值，editor 都列了。目前紀錄的 case_sha256 仍是舊的 ce2b65ec…，頁尾區塊還印著 0.0199999…／38.751487…。這是審查用的事實與寫作規範清單所列發布前會補齊的狀態，不是頁面問題。

4. 程式摘錄沒有問題。摘錄比對工具印出 docs/lessons/11-iou-loss.md []。CIoU 區塊已明說是改寫的示意、不能單獨執行；新片段是讀者自己打的，不在任何 repo 檔案裡；內文沒有引用程式行號。

5. 可讀性：新文字的順序是先算、再展開答案、再用程式核對。新增的術語在前文都解釋過（分量、斷言、新增一格），讀者不會被誤導或卡住。

6. 渲染與變動範圍。
   - 暫存副本裡 zensical build --clean --strict 與 validate_site.py 都是 exit 0（markdown_rendering passed）；HTML 裡的摺疊區、程式區塊與數學都正確。
   - docs/assets/diagrams/11-iou-loss.svg 和 HEAD 相同。用 qlmanage 渲染很乾淨，viewBox、title、desc 都有；G/P/C 座標、union 512、area(C) 640、ρ＝24、c²＝1856、1.2→1.179487 都和程式一致。標題第一字小圖上看起來像「末」，查過是 U+672A「未」，正確。
   - 只有這一頁變動。相關的 program、notebook（最後一格與程式逐字相同）、紀錄 JSON、reviews、SVG、glossary、section-map、coverage.json 都和 HEAD 相同，其他檔案的 diff 也沒有動到指向本頁的引用。validate_lessons.py 在暫存副本失敗，原因只有另一頁 09-anchor-clustering.md 的摘錄問題，本頁的逐頁檢查都通過。

非阻擋的已知狀態（editor 已列）：reviews/11-iou-loss.md 不再涵蓋目前的頁面文字，發布前要重審這一頁；頁尾執行紀錄要由 verify_curriculum.py 依新程式重錄。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：讀過的資料：審查用的事實與寫作規範清單、docs/lessons/11-iou-loss.md 全文、docs/glossary.md、lesson_cases/11-iou-loss.py、notebooks/11-iou-loss.ipynb、miniyolo/inference.py（確認 decode_grid 有 @torch.no_grad、score 門檻與 NMS）、miniyolo/losses.py（確認框 loss 是 MSE），以及本頁引用的前文：00-warmup、02-diagnostics、04-localization、06-evaluation、07-data、07-loss、07-inference、09-anchors；另外也看了 12-anchor-free 與 12-decoupled-head，用來查 retain_graph 第一次在哪裡出現。

網站：在暫存副本執行 zensical build --clean --strict，exit 0，訊息是 No issues found；scripts/validate_site.py 也是 exit 0。我直接讀了 site/lessons/11-iou-loss/index.html，確認表格、數學式、摺疊區都正常轉換。Playwright 的瀏覽器沒有安裝，所以沒有截圖，改讀 HTML。

圖：用 qlmanage -t -s 1200 把 docs/assets/diagrams/11-iou-loss.svg 轉成 PNG 看過，再逐一對照 SVG 原始碼，檢查座標、面積、ρ、c² 和圖底那行數字。

程式：在暫存副本用 PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg 跑 lesson_cases/11-iou-loss.py，exit 0。印出的 5 行（GIoU [0.02, 0.0]／1.179487、initial losses 1/1.2/1.310345/1.310345、IoU 梯度 [0.0, 0.0]、DIoU [38.7515, 20.0]／1.294497、exact boxes）都和正文一致。頁尾自動產生的紀錄區塊依指示沒有比對。

練習：三題都先手算。接著在同一個 namespace 依序執行完整程式和頁上的核對格，模擬 notebook，印出 {'iou': 1.0, 'giou': 1.111111, 'diou': 1.257732, 'ciou': 1.257732} 與 {'iou': 1.0, 'giou': 1.076923, 'diou': 1.264865, 'ciou': 1.266554}，和參考答案相符。第 1 題用 losses(G,G) 核對，四項都是 0。本機的做法另外試了兩種：python -i 的路線可行；from lesson_cases.11-iou-loss import 會出現 SyntaxError。

其他驗算（用程式的 losses 函式）：
- 框在框裡的例子：0.75／0.765625，中心梯度分別是 [0,0]、[0,0]、[-0.0078,-0.0078]。
- 交叉框：GIoU 為 -0.528889。
- 剛好相接：GIoU 為 0，IoU 梯度是 [0.03125,0]。
- c_y=20 時 GIoU 對 y 的兩側斜率是 ±0.05。
- DIoU 走一步：38.751486、1.294497，間隙 6.7515。
- 第 4 章的兩個 IoU：0.5319 與 0.1429。
- 第 3 題：v 為 0.041956，α 為 0.040267。
- 常見錯誤：預測框左右對調時不報錯，因為寬高被 clamp；真值框對調時出現 AssertionError。

用詞：用 grep 查過修訂敘事用語（沒有找到）、術語在各頁的用法、全形標點與中英文之間的空格。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈兩次單步更新的核對〉（docs/lessons/11-iou-loss.md 第 159–171 行）；lesson_cases/11-iou-loss.py 第 40、42、60、64 行 | 頁上的關鍵數字「中心梯度 [0,0]、[0.02,0]」，在完整程式裡是用 `torch.autograd.grad(initial['iou'],center,retain_graph=True)[0]` 算出來的。這個函式在本頁和之前所有課程頁都沒出現過：全書只有本節的程式用到它，`retain_graph` 到 12.2 節才有說明。照順序讀到這裡的讀者只學過「先 `loss.backward()` 再讀 `.grad`」，看程式時會找不到梯度從哪裡來，也看不懂為什麼要寫 `[0]`、為什麼之後還能對同一個 `initial['diou']` 呼叫 `backward()`。另外，程式用 `torch.minimum`／`torch.maximum` 搭配 `[2:]`、`[:2]` 一次算 x、y 兩個方向，再用 `.prod()` 算面積，這幾處都沒有和正文的手算式子對上。 |
| 2 | 建議 | 〈兩次單步更新的核對〉的條列（第 163–167 行），對照頁尾執行紀錄的輸出 | 程式最先印出的是 GIoU 那一行（`GIoU gradient [0.02, 0.0] after 1-pixel real SGD move 1.179487`），接著才是 `initial losses`、`non-overlap plain IoU center gradient`、DIoU、`exact boxes`。條列的順序卻是「初始 loss → IoU 梯度 → GIoU → DIoU → P＝G」。頁面要讀者「對照下方執行紀錄逐項核對」，照順序對的初學者會拿第一條去對第一行，結果對不上。另外，第一行結尾的 `after 1-pixel real SGD move 1.179487` 很容易被讀成「移動了 1.179487」。 |
| 3 | 建議 | 參考答案的最後一段（第 212–221 行），對照第 161 行的本機執行方式 | 第 161 行把在本機執行 `PYTHONPATH=. python lesson_cases/11-iou-loss.py` 和 Colab 並列，但核對第 2、3 題的程式只寫了「在 Colab 執行完完整程式後，新增一格執行」。在自己電腦上跑的讀者沒有「格」可以新增。`losses` 定義在 `lesson_cases/11-iou-loss.py` 裡，這個檔名以數字開頭又有連字號，寫 `from lesson_cases.11-iou-loss import losses` 會出現 SyntaxError（我實測過），初學者沒辦法照做。 |
| 4 | 建議 | 第 35 行「IoU 類 loss 由比例組成，沒有單位，約為 1」；〈放進 MiniYOLO 時要注意〉第 185 行「本節的 IoU 類 loss 約 1 到 1.3，大小差很多」 | 1 到 1.3 只是本例兩框不重疊時的值。兩框重疊得越多，這些 loss 就越小，完全重合時是 0（第 1 題就是這種情況）。\(L_{\text{IoU}}\) 的範圍是 0 到 1，GIoU、DIoU loss 是 0 到 2。拿第 7 章未訓練例子的 0.109 去和本例不重疊時的 1～1.3 比，初學者很容易以為 IoU 類 loss 一向比框 MSE 大十倍左右，然後照這個比例去調權重。 |
| 5 | 建議 | 第 17 行圖說；docs/assets/diagrams/11-iou-loss.svg；第 5 行「2 px」 | 圖上標了「ρ = 24」，但圖說只解釋了小寫 c 和 union。ρ 要到大約 70 行之後的 DIoU 小節才有定義，讀者第一次看圖時不知道它是什麼。單位方面，圖裡寫「畫素」「畫素²」，正文寫 pixel、pixel²，第 5 行又寫 px，同一頁有三種寫法，而且沒有對照說明（7.5 節就有「圖中的『畫素』就是本頁的 pixel」這種說明）。圖底那句「圖中 1 畫素畫成 10 個單位」裡的「單位」是 SVG 的內部座標，讀者在畫面上看不到這種單位。 |
| 6 | 建議 | 〈DIoU：把兩個中心拉近〉第 102–114 行；第 80 行「用差分對照也差不多」 | 頁面從 \(L_{\text{DIoU}}=1+(c_x-16)^2/(c_x^2+256)\) 直接跳到導數，沒有寫出商的微分這一步，也沒有像 GIoU 那樣用實際移動來對照。第 0 章特別強調「不用背導數表」，7.3 節也提供「不會微分也能驗算」的做法；本節的讀者不一定會商的微分，卡在這裡就沒辦法自己確認 0.012485。另外，「差分」這個詞全書只出現在這裡，沒有說明是什麼。 |
| 7 | 建議 | 〈CIoU：再比較寬高比〉摺疊區的標題「（本例 v＝0，可先跳過）」；自主練習第 3 題（第 202 行）；正文第 171 行 | 摺疊區的標題告訴讀者可以先跳過，但第 3 題要用到裡面 v、α 的式子，正文第 171 行也提到「α 的 detach」，所以跳過的讀者在這兩個地方都會卡住。第 3 題標成「紙筆題」，卻要算 arctan 2，這個值用紙筆算不出來；而 arctan 在頁上只用「反正切」帶過，對沒學過反三角函數的讀者來說，不夠用來理解它的意思。 |
| 8 | 建議 | 第 21 行的 \((c_x,c_y)\)；第 88 行；第 100 行「\(c^2=c_x^2+16^2\)」 | 同一頁裡有幾個長得很像的符號：\(c_x,c_y\) 是 P 的中心，小寫 c 是 C 的對角線長，大寫 C 是包圍框，而第 7 章的 C 又是類別數。\(c^2=c_x^2+16^2\) 看起來就像「c 的 x 分量平方加 y 分量平方」，再加上 \(c_x\) 剛好等於 C 的寬，更容易被讀成分量；可是 \(c_y=20\) 並不是 C 的高 16。第 88 行的提醒只說小寫 c「和第 7 章的類別數 C 無關」，真正容易和類別數混淆的大寫 C（包圍框），在第一次出現時反而沒有說明。 |
| 9 | 建議 | 常見錯誤第 1 條（第 193 行）；lesson_cases/11-iou-loss.py 第 7 行 | 第 1 條說 xyxy 次序寫反會「算出負的寬高與面積」。但完整程式會把預測框的寬高 `clamp(min=1e-6)`，斷言只檢查真值框。我把預測框左右對調成 `[48,12,32,28]` 代入 `losses`，既沒有出現負數也沒有報錯，印出 \(L_{\text{GIoU}}\approx1.333333\)、\(L_{\text{CIoU}}\approx1.742308\)。讀者如果拿完整程式試這個錯誤，看到的結果和頁上說的不一樣，還可能誤以為程式也會擋下寫錯的預測框。 |
| 10 | 建議 | CIoU 摺疊區第 155 行「完整程式用它明確展示一種常見的實作選擇，不把它當成所有函式庫的唯一寫法。」 | 這句只說還有別的寫法，卻沒說是哪一種、和本頁引用的 YOLOv5 v6.0 是不是同一種、兩種寫法差在哪裡。讀者讀完無從判斷，也沒有任何可以接著做的事。 |
| 11 | 建議 | 〈收益與代價〉第 177 行「代價是要多處理幾件事：剛好相接、上下緣對齊這類轉折點；…梯度也要重新檢查」 | 前文說 PyTorch 在轉折點會自動給一個值，這裡卻把轉折點列為「要多處理」的代價，又沒說要處理什麼、怎麼處理。初學者不知道是要加程式、改資料，還是只要知道有這回事就好。 |
| 12 | 建議 | 第 11 行「本節只換定位 loss，head、assignment…與資料增強都不動」；〈放進 MiniYOLO 時要注意〉第 181 行「正式整合時」 | 頁面沒有說本書的 MiniYOLO 到底有沒有改用 IoU 類 loss。第 11 行讀起來像本節在偵測器裡換掉了 loss，但實驗其實只有一對框；「正式整合時」也容易被讀成後面章節會接著做。實際上 miniyolo/ 裡沒有任何 IoU 類 loss（losses.py 的框 loss 是 `F.mse_loss`），12.1 節用的也是 Smooth L1。讀者往後讀時，可能以為模型已經換成 CIoU 了。 |
| 13 | 建議 | 參考答案第 2 題的最後一句（第 208 行）「再左移到 8 pixel 時會剛好碰到 G」 | 這句接在第 2 題（已經左移 4 pixel）後面，「再左移到 8 pixel」可以讀成從第 2 題的位置再往左移 8 pixel（中心從 36 到 28）。那時 P＝[20,12,36,28]，交集寬已經是 4，兩框重疊了，並不是「剛好碰到」。 |
| 14 | 建議 | 第 187 行「train/validation/test」 | 術語表和本頁其他地方都用全形斜線（例如「train／validation／test」「DIoU／CIoU」「objectness／class」），只有這裡用半形的「/」。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：只在暫存副本工作：rsync 到暫存副本，沒有在 repo 根目錄裡執行任何會寫入的東西。

讀過的檔案：docs/lessons/11-iou-loss.md；lesson_cases/11-iou-loss.py（只 import math、torch，沒有 import miniyolo 或 scripts 模組）；docs/assets/diagrams/11-iou-loss.svg；審查用的事實與寫作規範清單。為了核對頁面的跨章引用，也讀了 docs/lessons/04-localization.md（第 115、117 行：0.53／0.14、「小於 0 就取 0」）、07-loss.md（第 27、85、89、92、195 行：類別數 C、紅框 [8,12,24,28]、框 MSE 0.109375、權重 5、〈空圖：沒有正格時〉）、06-evaluation.md（第 19、118 行：IoU≥0.5）、02-diagnostics.md（第 36 行：detach）、00-warmup.md（第 23 行）、docs/glossary.md（第 16 行）、miniyolo/inference.py 的 decode_grid（第 7–41 行：@torch.no_grad、score 門檻、NMS）、miniyolo/losses.py、miniyolo/targets.py、section-map.json（11.4 編號）、scripts/validate_lessons.py（excerpt 規則）。

查過的原始來源：
- GIoU：https://arxiv.org/abs/1902.09630（PDF 與 https://ar5iv.labs.arxiv.org/html/1902.09630）。§3 性質 4「−1 ≤ GIoU(A,B) ≤ 1」；§3.1 Algorithm 2（L_IoU＝1−IoU、L_GIoU＝1−GIoU，包圍框用 min/max 求）；§3.1 \"in all non-overlapping cases, IoU has zero gradient\"；§3.1〈L_GIoU behaviour when IoU = 0〉（2−U/A^c，以及 \"A^c is minimized while … A^p, is maximized\"）。
- DIoU／CIoU：https://arxiv.org/abs/1911.08287（PDF 與 ar5iv HTML）。Abstract（\"slow convergence and inaccurate regression\"）；Introduction 的 Eq. (2)(3)、\"IoU loss … would not provide any moving gradient for non-overlapping cases\"、\"GIoU loss intends to increase the size of predicted box at first\"、\"GIoU loss will totally degrade to IoU loss for enclosing bounding boxes\"、Fig. 2；〈Limitations of IoU and GIoU Losses〉；〈Distance-IoU Loss〉Eq. (6)(7)，c 是最小包圍框的對角線；〈Comparison with IoU and GIoU losses〉（scale invariant）；〈Complete IoU Loss〉Eq. (8)–(12) 與 \"the dominator w²+h² is simply removed\"。
- YOLOv4：https://arxiv.org/abs/2004.10934 §3.4 YOLOv4，\"Bag of Freebies (BoF) for detector: CIoU-loss, …\"。
- 官方程式：ultralytics/yolov5 的 tag v6.0，用 GitHub API 解析為 commit 956be8e642b5c10af4a1533e09084ca32ff4f21f。utils/loss.py 第 135–136 行（CIoU=True、lbox += (1.0 - iou).mean()）；utils/metrics.py 的 bbox_iou 第 190–232 行（第 224–226 行：v 用 autograd，alpha 在 torch.no_grad() 下計算）。Zzh-tju/DIoU-SSD-pytorch commit cec038bc1057f0cd532752413b24924fde427f09，utils/box/box_utils.py 第 85–89 行（α 在 torch.no_grad() 下計算）。另用 curl 確認頁面上 4 個外部連結都回 200。

執行過的程式與指令：
1. 在暫存副本執行 PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/11-iou-loss.py：exit 0，5 行輸出依序是 GIoU [0.02, 0.0]／1.179487、initial {'iou':1.0,'giou':1.2,'diou':1.310345,'ciou':1.310345}、[0.0, 0.0]、[38.7515, 20.0]／1.294497、exact boxes。正文的確定值全部和輸出一致；依指示，沒有拿正文去比對生成的紀錄區塊。
2. python3 摘錄比對工具 &lt;scratch&gt; docs/lessons/11-iou-loss.md：結果是 []。頁面沒有 data-excerpt 區塊，兩個 python 區塊都不是逐字摘錄。
3. qlmanage -t -s 1200 把 SVG 轉成 PNG 並檢視；依 (10x−40, 10y−10) 逐一核對 G、P、C、空白區、兩個中心、對角線與所有標註數字。
4. zensical build --clean --strict：exit 0，No issues found。檢查 site/lessons/11-iou-loss/index.html：表格 1 個、摺疊區 4 個、87 個數學式，LaTeX 指令與括號都配對。想用 Playwright 截圖，但瀏覽器沒有安裝，所以沒做。
5. 暫存副本腳本：
   - src11/check11.py：原樣執行頁面練習格（第 2 題 1.0/1.111111/1.257732/1.257732；第 3 題 1.0/1.076923/1.264865/1.266554），並手算或程式核對 v＝0.041956、α＝0.040267、交叉細長框 GIoU＝−0.528889、內含例 0.75/0.765625 與梯度 [0,0]、相接點梯度、MSE 288、IoU 0.5319／0.1429、DIoU 解析梯度 0.0124851 與一步後的 38.7515／1.294497。
   - src11/check11b.py、src11/check11d.py：寬高也可學時的梯度與 SGD 軌跡。
   - src11/check11c.py：c_y 方向折角兩側的 GIoU／DIoU 值與梯度。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/lessons/11-iou-loss.md 第 171 行（「本例只學中心、寬高固定…這個實驗測不到 v 的梯度」），連帶〈GIoU：扣掉包圍框裡的空白〉第 50、68–84 行的機制說明、圖標題「C 隨 P 往左而縮小」，以及第 7 行的學習目標「說出…會把框往哪裡推」 | 寬高固定這個簡化本身有寫出來（第 21 行），但頁面只說它的後果是「測不到 v 的梯度」。它同時藏起了 GIoU、DIoU 對寬高的梯度，頁面沒有說。兩篇原始論文都把「加大預測框」列為 IoU＝0 時的主要行為之一。GIoU 論文 §3.1〈L_GIoU behaviour when IoU = 0〉推出和本頁相同的 2−U/A^c，並說它 "is maximized when the area of the smallest enclosing box A^c is minimized while … the area of predicted bounding box A^p, is maximized"。DIoU 論文 Introduction 寫 "GIoU loss intends to increase the size of predicted box at first"，〈Limitations of IoU and GIoU Losses〉寫 "GIoU actually increases the predicted box size to overlap with target box … yielding a very slow convergence"。本頁卻把 GIoU 在不重疊時的訊號全部歸給「C 隨 P 往左而縮小」，讀者照學習目標回答「會把框往哪裡推」時，只會答「往左移」。我在暫存副本實測：在本例起點把 (c_x,c_y,w,h) 都設成可學，∂L_GIoU/∂w＝−0.015，和 ∂L_GIoU/∂c_x＝0.02 同一個量級；DIoU 則是 ∂L/∂w＝−0.006688、∂L/∂h＝−0.002675。也就是兩種 loss 的梯度下降都會一邊左移、一邊把 P 加寬（SGD lr=5 時，GIoU 把 w 撐到約 23.9、DIoU 撐到約 22.4，之後才縮回）。讀者照〈放進 MiniYOLO 時要注意〉整合時，寬高是可學的，會看到預測框先變大，頁面卻沒告訴他這是 loss 本身的行為，很可能誤以為座標換算寫錯。 |
| 2 | 建議 | docs/lessons/11-iou-loss.md〈DIoU：把兩個中心拉近〉第 100–116 行，以及〈兩次單步更新的核對〉第 166 行（DIoU 走一步後中心約 (38.7515,20)） | 第 82 行替 GIoU 說明了 c_y=20 是上下緣對齊的轉折點，y 分量的 0「是轉折點上取的值，不代表 y 方向沒有影響」。DIoU 用的是同一個 c_y=20，頁面只推導 x 分量，並回報 y 走完一步仍是 20，卻沒有同樣的提醒。而且 DIoU 在這個折角兩側的斜率和 GIoU 相反：P 往上或往下移時，c² 線性增加，ρ² 卻只按平方增加，所以 ρ²/c² 反而變小。用完整程式的 losses 實測：c_y＝21 或 19 時，L_DIoU＝1＋577/1889≈1.305453，小於 1.310345；c_y＝20.001 時，autograd 的 y 分量是 −0.00535，梯度下降會把 P 推離上下對齊。c_y 偏 8 pixel 時 L_DIoU 甚至降到 1.294118。GIoU 在同一點則是 V 形最低點，c_y＝20.001 的 y 分量是 +0.05，會把 P 拉回對齊。所以 y 留在 20，只是 PyTorch 在折角上取 0 的結果。讀者若依標題「把兩個中心拉近」推論 DIoU 在每個方向都縮小中心距離，放到本例的 y 方向就不成立。 |
| 3 | 建議 | docs/lessons/11-iou-loss.md CIoU 摺疊區第 155 行（「完整程式用它明確展示一種常見的實作選擇，不把它當成所有函式庫的唯一寫法」） | 這句沒有出處，讀者無從判斷「常見」指的是什麼；另外還漏了論文自己的另一個實作差異。DIoU 論文 Eq. (12) 只給出 ∂v/∂w、∂v/∂h，也就是優化時把 α 當常數；接著又說 "in our implementation, the dominator w²+h² is simply removed for stable convergence"。本課程式則和 YOLOv5 v6.0（commit 956be8e，utils/metrics.py 第 224–226 行）、作者的 DIoU-SSD-pytorch（commit cec038b，utils/box/box_utils.py 第 85–88 行）一樣：α 在 torch.no_grad() 下算，v 用 autograd 的完整梯度，沒有拿掉 1/(w²+h²)。頁面交代了 detach 這個選擇，卻沒說 v 的梯度也和論文自己的實作不同。論文說這個分母在 w、h 落在 [0,1] 時容易讓梯度爆炸，若〈放進 MiniYOLO 時要注意〉改用正規化座標算 CIoU，正好會遇到。 |
| 4 | 建議 | docs/lessons/11-iou-loss.md〈兩次單步更新的核對〉第 161–167 行（「對照下方執行紀錄逐項核對」與其後清單） | 程式印出的第一行是 GIoU 那一行（`GIoU gradient [0.02, 0.0] after 1-pixel real SGD move 1.179487`），之後才依序是 initial losses、plain IoU 梯度、DIoU、exact boxes。清單卻從初始 loss 開始，GIoU 排在第三項。初學者照「逐項」由上往下對，第一項就對不上。 |
| 5 | 建議 | docs/lessons/11-iou-loss.md 第 13 行（「固定版本 YOLOv5 v6.0 的定位 loss 也用 CIoU（見 v6.0 loss.py）」） | 主張本身正確：tag v6.0 指向 commit 956be8e642b5c10af4a1533e09084ca32ff4f21f，utils/loss.py 第 135–136 行是 `iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)` 與 `lbox += (1.0 - iou).mean() # iou loss`。但連結只指到整個檔案，讀者得自己在兩百多行裡找 `CIoU=True`。tag 也可能被移動；第 12 章以後的頁面引用官方程式時都用 commit hash。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 技術查核 | 寬高固定，也藏起了 GIoU／DIoU 會放大框的梯度（must） | 已修正：第 171 行補上：寬高可學時，GIoU 對 w 的梯度是 −0.015、DIoU 是約 −0.006688，兩者都會一邊左移一邊加寬；也補上 GIoU 論文 IoU＝0 時的說法，並說明預測框先變大是 loss 本身的行為。DIoU 論文的引述加上「GIoU 傾向先放大預測框」。暫存副本用 autograd 實測核對 |
| 2 | 技術查核 | DIoU 在 c_y=20 的轉折點方向和 GIoU 相反 | 已修正：DIoU 小節末補一段：PyTorch 給 0，但往上或往下移 loss 都會下降（c_y=21 時約 1.305453），DIoU 縮小的是 ρ²/c²，不一定縮小 ρ。暫存副本實測核對 |
| 3 | 技術查核 | 「常見的實作選擇」沒有出處 | 已修正：換成具體出處：YOLOv5 v6.0 的 bbox_iou 在 no_grad 下算 α（附 commit 連結 L223–227，已核對）、論文把 α 當常數，以及論文自己的實作把 1/(w²+h²) 換成 1，本節沒有這樣做 |
| 4 | 技術查核 | 核對清單的順序和輸出不同 | 已修正：清單改成和輸出同樣的順序，每條標明「第 n 行」和那一行的開頭文字 |
| 5 | 技術查核 | loss.py 連結沒有行號、用的是 tag | 已修正：改成 commit 956be8e 的 loss.py#L135-L136（已核對是 CIoU=True 那兩行） |
| 6 | 讀者審查 | torch.autograd.grad 沒有說明 | 已修正：核對清單後補一段，說明 autograd.grad、[0]、retain_graph 各是什麼意思；變數名稱對照表屬風格偏好，沒有加 |
| 7 | 讀者審查 | 清單順序不同、1.179487 容易看成移動量 | 已修正：重排清單，並寫明行尾的 1.179487 是走一步之後的 L_GIoU，不是移動量 |
| 8 | 讀者審查 | 在自己的電腦上無法照著核對 | 已修正：補上 `PYTHONPATH=. python -i lesson_cases/11-iou-loss.py`，跑完再貼上程式碼 |
| 9 | 讀者審查 | 「約 1 到 1.3」容易被當成通則 | 已修正：第 35 行改成「本例約為 1」；第 185 行補上各 loss 的範圍（IoU 0～1，GIoU、DIoU 0～2，完全重合時是 0），並說明數字只來自單一例子，不能照比例換算權重 |
| 10 | 讀者審查 | ρ 沒說明，畫素／px／pixel 混用 | 已修正：圖說補上 ρ 和「圖中的畫素就是 pixel」；第 5 行的 px 改成 pixel；SVG 圖底與 desc 改成「依比例放大 10 倍繪製」 |
| 11 | 讀者審查 | DIoU 導數缺少推導，「差分」沒解釋 | 已修正：補上商的微分那一步，以及 c_x 從 40 改成 39 的驗算（約 0.0127）；第 80 行寫明差分就是移動前後的 loss 相減 |
| 12 | 讀者審查 | CIoU 摺疊區標「可先跳過」，第 3 題卻用得到 | 已修正：標題改成「本例 v＝0；第 3 題會用到」；第 3 題標出式子在哪裡、arctan 2≈1.107149；補上 arctan 的定義 |
| 13 | 讀者審查 | c_x 和 c、C 容易混淆 | 已修正：c² 那句註明 c_x 不是 c 的分量；C 第一次出現的地方註明它是框，不是類別數 |
| 14 | 讀者審查 | xyxy 寫反時，完整程式不會出現負數 | 已修正：第 1 條補上：程式只檢查真值框，預測框的寬高會 clamp 到 1e-6，不會報錯，但意思已經錯了 |
| 15 | 讀者審查 | 「常見實作選擇」那句沒有內容 | 已修正：和技術查核第 2 條一起改成具體出處 |
| 16 | 讀者審查 | 轉折點列為代價，卻沒說要做什麼 | 已修正：補上做法：PyTorch 在這些點只給其中一個值，檢查梯度方向時要避開或註明 |
| 17 | 讀者審查 | 沒說 MiniYOLO 有沒有改用 IoU 類 loss | 已修正：〈放進 MiniYOLO 時要注意〉開頭寫明現況：MiniYOLO 仍用框 MSE（已確認 losses.py），並把「正式整合時」改成「若要自己換」 |
| 18 | 讀者審查 | 「再左移到 8 pixel」有歧義 | 已修正：改成「從起點總共左移 8 pixel（中心 (32,20)、P＝[24,12,40,28]）」 |
| 19 | 讀者審查 | train/validation/test 用了半形斜線 | 已修正：改成全形「train／validation／test」 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/11-csp.md`、`docs/lessons/11-fusion.md`、`docs/lessons/11-augmentation.md`、`docs/lessons/11-iou-loss.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/11-iou-loss.md 第 199 行（〈常見錯誤〉xyxy 次序反了那一條） | 新加的「只會被當成寬度幾乎是 0 的框」不符合程式行為。lesson_cases/11-iou-loss.py 只把預測框的寬高 pwh clamp 到 1e-6；交集 inter_wh 與包圍框 outer_wh 直接用對調後的座標計算。實測 losses(torch.tensor([48.,12.,32.,28.]), G) 得 L_GIoU=1.333333、L_DIoU=1.692308（C 被算成 [8,12,32,28]，寬 24）；同一中心、寬度幾乎為 0 的框 [40,12,40.000001,28] 卻是 L_GIoU=1.5、L_DIoU=1.45，兩者並不相同。讀者若照這句去理解對調後的 loss，算出來會對不上。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

對照 lesson_cases/11-iou-loss.py 的 losses()：只 assert 真值框座標有限、寬高為正；預測框寬高先 `.clamp(min=1e-6)`，所以左右對調的預測框不會報錯、照樣算出 loss，句子正確，前後也讀得通。順帶核對新加的 YOLOv5 v6.0 錨點：loss.py L135–L136 是 `CIoU=True`，metrics.py L223–L227 在 torch.no_grad() 下算 alpha。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

用腳本比對發現與處理列數並讀了修正後檢查。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/11-iou-loss.md | 19 項發現中，must 的處理寫在修正後檢查裡，其餘 18 項 should 沒有處理，指向的〈定稿修正〉不存在（原因同上）。另有「2. trace 與 page-impact 都處理了」。 | 已修正：產生器改以頁名、節名、萬用字元、頁面上的圖與該節 notebook 把處理對應到頁面，〈定稿修正〉列出這一頁每一項的處理。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：讀者 14 項與技術 5 項都在〈定稿修正〉處理（上一輪指出缺漏，已補）；技術查核列出 yolov5 固定 commit 的 loss.py、metrics.py；〈後續編輯的檢查〉齊全。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/11-iou-loss.md 第 11 行 | 殘句：「模擬腳本 sim_colab_11iou.py 與精算腳本 nums_11iou.py 都放在上一層暫存副本與暫存副本。」 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 11 行已改寫（加上 log、去掉重複的「與暫存副本」），成為一般的位置說明。處理說明大致屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/11-iou-loss.md 第 11 行 | 「…都放在上一層暫存副本」仍沒有實際位置。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
