# 審查紀錄：Decoupled head

審查範圍：`docs/lessons/12-decoupled-head.md`，以及 `lesson_cases/12-decoupled-head.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：判定通過，沒有必要問題。只有兩條建議，都是讀者容易看不懂的地方，不是錯誤：一是新段落同時用了清單的「項」和輸出的「行」兩套編號；二是新寫法裡的「commit」沒有解釋。所有檢查都在自己的暫存副本進行，沒有動到 repo。

1. 關於程式的敘述都符合目前的程式碼。
   - 副本裡跑出的 5 行輸出，和 artifacts/checks/curriculum/12-decoupled-head.json 的 stdout 逐字相同。程式的 SHA-256 等於紀錄的 case_sha256，程式自 baed4ff 之後沒改過。
   - 突變測試（mut/run_mutants.py）：
     - 拿掉第一次的 retain_graph：第二次 backward 報 `Trying to backward through the graph a second time`。
     - 第三次 backward 前不清梯度：allclose 斷言失敗。
     - 照練習 1 原文的三處修改逐字改：通過，印出 `... box gradient + 2 * class gradient: verified`。只改第一處，或連第二次單獨反傳也乘 2：allclose 都失敗。
     - 第二次 zero_grad 改成 set_to_none=False：`is None` 斷言失敗。改在第一次則沒有影響，和頁面「上一次有梯度」的限定相符。
     - 類別分支改成 `self.class_branch(f.detach())`：在取 `class_backbone_grad` 那行報 `AttributeError: 'NoneType' object has no attribute 'clone'`，和新寫的句子一致。
     - 拿掉 step：權重改變的斷言失敗。
   - 程式確實先跑完所有斷言才開始印，所以新段落「印得出這兩行，就表示斷言已通過」成立。
   - 參考答案（mut/answers.py）：反傳 L_box+2L_cls 時，類別分支梯度正好是 2 倍（torch.equal），框分支逐位元不變，backbone 梯度 allclose g_box+2g_cls。含一層 3×3 的 coupled head 有 638 個參數；只反傳框 loss 時，1×1 權重的梯度不是 None，類別那 2 列全為 0。
   - 參數算式 224／584／36／18、合計 1446、兩條分支 1222、最簡 coupled head 54，全部正確。
   - 跨章引用也都成立：第 7 章 head 輸出 5+num_classes=7、第 7 章的 `total`、12.1 的 stride 8、第 10 章 YOLOv3 對每類各做 sigmoid、第 3 章的 nn.Sequential、第 2 章 None 與全 0 的差別。
   - cv2／cv3 那句對照過固定在 441632c 的 head.py：v8 的 legacy=True，兩條分支都是 Conv 3×3 → Conv 3×3 → Conv2d 1×1。

2. 兩條先前審查意見都處理了。
   - 第 3 行 Colab：HEAD 裡已經是 lessons-v0.4.0（由 463d3f5 的 build 腳本寫入），這次沒有手改。
   - 第 194 行：照建議逐字改寫，後半句沒動。
   - 受程式改動影響的段落沒有本頁或 SVG 的條目。16-inference-head 部分提到程式寫死印出 'verified'，這點已在頁面照實說明。
   - 正文沒有製作或修訂經過的字眼，也沒有寫死日期或機器。

3. 數字：這次沒有新增任何數字。−0.0073 和紀錄相同（這台 Mac 上是 −0.00728）；90.4° 是 arccos(−0.0073)=90.418°。record_evidence.py 把本節列為過期（舊紀錄缺 machine／dependencies 欄位），發版時會重錄。修改者已把這三個依賴紀錄的值列出。

4. 程式摘錄：摘錄比對工具印出 `docs/lessons/12-decoupled-head.md []`。三段都是程式裡連續的原文，只加了中文註解。我在副本裡改壞程式中兩行，摘錄檢查兩行都抓到，之後已還原，SHA 回到 38f6e646…。正文沒有引用程式行號。

5. 可讀性：見兩條建議。detach 那條改得具體清楚，頁面的教學順序沒有變。

6. 建置：在副本的完整工作樹上，`zensical build --clean --strict` 結束碼 0（No issues found），validate_site.py 結束碼 0。修改者說的 04-localization 錨點錯誤，在我的副本裡已經不會出現。
   - 算出的 HTML 裡，三段都是帶 data-excerpt 的 language-python 區塊，沒有屬性文字漏到頁面上。本頁沒有 SVG。
   - 本頁的 diff 只有 6 行，都在執行紀錄區塊之上，執行紀錄區塊沒動。相關的程式、notebook、執行紀錄、審查檔都沒有改。
   - validate_lessons.py 只在審查涵蓋那一步失敗；validate_curriculum_evidence.py 在 00-warmup 就停下，屬於別節的過期紀錄。依審查用的事實與寫作規範清單，這兩項都是發版前的預期狀態。紀錄檔都在副本根目錄：build-full.log、validate-site-full.log、validate-lessons.log、vce.log、record-evidence-list.log。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/12-decoupled-head.md 第 156 行（「應看到」清單後新增的段落） | 同一段裡用了兩套編號。「第 2、3 項」指上面清單的項目，「輸出的第 3 行」指實際輸出的行。清單第 3 項（`verified`）其實印在輸出的第 4 行，輸出的第 3 行是 cosine。初學者容易把「第 3 項」和「第 3 行」當成同一行：剛讀完「第 2、3 項印的英文是固定的文字」，接著讀到 cosine 是「第 3 行」，會一時以為 cosine 那行也是固定文字。內容本身沒有錯，只是容易看混。 |
| 2 | 建議 | docs/lessons/12-decoupled-head.md 第 194 行（「參考原始碼（Ultralytics，固定在 commit 441632c）」那行） | 新的統一寫法帶進了「commit」這個詞（舊寫法沒有），但本頁、術語表和前面各節都沒有解釋。20-deployment 提到的 commit 是 Modal Volume 的寫入，意思不同。目標讀者是程式新手，看不出「固定在 commit」要告訴他什麼：連結打開的是官方程式的某一個固定版本，可能和官方現在的最新版不同。讀者不會因此卡住（連結照樣能點），所以只列建議。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：先讀審查用的事實與寫作規範清單，再用 rsync 建暫存副本（沒有在 repo 根目錄執行任何東西）。

一、讀頁面與相關教材：在暫存副本用 zensical build --clean --strict 建站（exit 0），讀 site/lessons/12-decoupled-head/index.html 的 article。另起本機 http.server，用 headless Chrome 截 1280px 全頁，切成 7 段逐段看渲染：數學式、三個摺疊區、表格、程式摘錄、上下頁連結都正常，article 裡沒有任何 img 或 svg，所以本頁沒有圖可看。逐行對照 Markdown 原文，並讀了 notebooks/12-decoupled-head.ipynb（標題「本節可修改的完整實驗」在第 3 格說明格，程式在第 4 格）。對照的內容：docs/glossary.md（logits、ltrb、activation、objectness、BCE 等詞）；12-anchor-free.md（stride 8、格單位、softplus、Smooth L1、[B,P,4]／[B,P,C] 表）；07-targets／07-loss／07-training 與 miniyolo/models.py（7 個輸出的 1×1 head、total、紅色矩形）；00-warmup（∂ 符號）；01-small-cnn（參數公式、「第 0 軸」的寫法慣例）；02-diagnostics（None 與全 0 的差別）；03-identity、03-comparison（nn.Sequential、numel）；10-multiscale（YOLOv3 每類各做 sigmoid）；12-dfl、13-dual-assignment（後面的 head 仍有分支、[B,P,4×K]）；section-map.json（12.1～12.4 的編號）。也查了 scripts/validate_lessons.py 怎麼比對摘錄：它會去掉註解再比，所以摘錄裡多加的中文註解是設計允許的。

二、實跑：在暫存副本用 PYTHONPATH=. OMP_NUM_THREADS=2 執行 lesson_cases/12-decoupled-head.py，5 行輸出和頁面相同。

三、照頁面做練習與變體（檔案在暫存副本的 exercises/）：
- 練習 1 照頁面三處修改：通過並印出「+ 2 * class gradient: verified」。
- 只改 backward：斷言失敗。
- 第二次反傳也改成 2*cls_loss：斷言失敗，和參考答案相同。
- 用 named_parameters 比較 L_box+L_cls 與 L_box+2L_cls 的梯度：類別分支整條變 2 倍，框分支不變，backbone 兩者都不是。
- 練習 2 參考答案：自建 coupled head（3×3 8→8、ReLU、1×1 8→6），head 有 638 個參數，只反傳框 loss 時 1×1 類別列的梯度全 0 而不是 None，共用的 3×3 對兩個 loss 都有梯度；另驗證一層 8→6 的 1×1 等於拆成 8→4 與 8→2 兩層（54＝36＋18）。

四、試〈常見錯誤〉與 retain_graph 段落的說法：
- 拿掉第一次的 retain_graph：第二次 backward 報 RuntimeError「Trying to backward through the graph a second time…」；拿掉第二次的，則在第三次報同一個錯。
- 拿掉第三次前的 zero_grad：斷言失敗；拿掉第二次前的：停在 box_branch 的 is None 斷言。
- set_to_none=False 逐一改三個 zero_grad 與三個全改：只改第 1 或第 3 個時程式照常通過，改第 2 個或全改才失敗。
- 在 class_branch 前加 f.detach()：在 class_backbone_grad 那行報 AttributeError。
- 拿掉 clone()：程式照常通過。
- 另算 acos(−0.0073)＝90.42°。

五、validator：validate_lessons 只在審查涵蓋檢查失敗（審查紀錄正在撰寫，是工作樹的暫時狀態），摘錄檢查已通過；validate_site 全部通過。

六、參考連結：用 curl 取 Ultralytics commit 441632cd 的 ultralytics/nn/modules/head.py 與 ultralytics/nn/tasks.py。Detect 預設 legacy=False，cv3 有 legacy 與 DWConv 兩種寫法；parse_model 一開始設 legacy=True，遇到 C3k2、A2C2f、C2fCIB 才改成 False，所以 YOLOv8（用 C2f）走的是 legacy 那條：兩個 3×3 Conv 加一個 1×1。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈可核對的成本〉「最簡單的 coupled head」那段（第 143 行），以及參考答案第 2 題（第 192 行） | 頁面把 coupled head 的特徵寫成「最後由同一層 1×1 卷積同時輸出」，又說只有一層 8→6 的 1×1 基線「沒有分工」。參考答案第 2 題舉的「梯度路徑的差別」，則是 1×1 權重的類別列梯度是全 0 而不是 None。可是一層 8→6 的 1×1，和拆成 8→4、8→2 兩層 1×1，在數學上完全相同：每個輸出 channel 本來就有自己的一列權重，參數也一樣多（54＝36＋18，我在暫存副本驗證輸出相同）。框 loss 對類別列的梯度在兩種寫法裡都是 0，None 和全 0 只是 PyTorch 記錄方式不同。真正不同的是輸出層之前的層：保留一層 3×3 的 coupled 版本，那層 3×3 只反傳框 loss 或只反傳分類 loss 時都有梯度（實測兩者都非零），和 backbone 一樣是兩個任務共用、可能互相拉扯；decoupled 的兩層 3×3 則各只收到一種梯度。照現在的寫法，讀者容易以為「把最後的 1×1 拆成兩個」就算解耦，或以為 None 和全 0 是兩種設計在學習上的差別。 |
| 2 | 建議 | 〈常見錯誤〉第二項「把 .grad 是 None 和全 0 tensor 混為一談」（第 171 行） | 頁面沒說要改哪一個 model.zero_grad()，而程式裡有三個。我在暫存副本逐一改成 set_to_none=False：只改第一個，或只改第三個，程式都照常跑完並印出 verified；只有改第二個（或三個全改），才會在 assert model.box_branch[0].weight.grad is None 失敗。第一個 zero_grad 在摘錄裡帶著「所有參數的 .grad 設成 None」的註解，是最顯眼的一個。照字面試的初學者很可能改到它，結果看不到失敗，反而以為頁面錯了，或以為 None 和全 0 沒有差別。 |
| 3 | 建議 | 〈先把各軸說清楚〉框分支輸出的軸說明（第 42 行附近） | 12.1 的表格把框的距離輸出寫成 [B,P,4]、類別 logits 寫成 [B,P,C]，並註明「見 12.2 節」；12.4 也說「12.2 節的框分支…shape 是 [B,P,4×K]」。本頁卻只給 [2,4,4,4] 與 [2,2,4,4]（channel 在第 1 軸，位置在最後兩軸），沒有說這和 [B,P,4] 是什麼關係，也沒說 4×4 的每個位置就是一個候選點。照順序讀的讀者在 12.1、12.2、12.4 之間會看到兩種排法，不知道指的是不是同一個東西。 |
| 4 | 建議 | 頁尾〈參考來源〉的 Ultralytics 連結與說明（第 194 行） | 頁面說「YOLOv8 的每條分支是兩層 3×3 卷積再接一層 1×1 卷積」，這對 YOLOv8 是正確的。但連結的 head.py 在這個 commit 預設 legacy = False，而 cv3 有兩種寫法：只有 if self.legacy 那一種（Conv(x, c3, 3)、Conv(c3, c3, 3)、nn.Conv2d(c3, nc, 1)，註解寫 v3/v5/v8/v9）才是 YOLOv8，另一種用 DWConv。tasks.py 的 parse_model 對 YOLOv8 會設 legacy=True，但頁面沒提。照連結去讀原始碼的讀者，第一眼看到的是預設值和另一種寫法，會以為頁面寫錯了。 |
| 5 | 建議 | 〈先把各軸說清楚〉「這裡沿用前幾節的叫法，也稱它為 logits」（第 42 行） | 同一段裡，先說框分支的 boxes 也叫 logits，下一句又說程式裡叫 logits 的變數是類別分支的輸出。框的這個別名在本頁之後再也沒用到，卻讓「logits」在同一段指兩種東西，初學者得停下來分辨。後文「只用到類別分支的輸出 logits」也要靠上下文，才知道不包含框。 |
| 6 | 建議 | 第三次反傳的說明「每一項再各自沿自己的分支，用連鎖律（chain rule）乘回 θ」（第 93 行） | 前半句已經說 θ 的梯度是 g_box+g_cls，後半句的「再」讀起來像是加總之後還有下一步要做。其實 g_box 和 g_cls 本身就是各自沿著分支、用連鎖律算到 θ 的結果。數學好的讀者會卡在「已經是對 θ 的梯度了，還要再乘什麼」。 |
| 7 | 建議 | 第 158 行標題「收益、代價與下一項判斷」（右側目錄也顯示這個標題） | 這一節的內容是收益、三項代價、何時不划算、常見錯誤、自主練習、參考答案與參考來源，找不到標題說的「下一項判斷」是哪一項。從目錄看到這個標題的讀者，會預期這裡有一個要做的決定，或銜接下一節的說明。其他頁多半用「收益、代價與常見錯誤」。 |
| 8 | 建議 | 自主練習第 1 題「完整程式就是 Colab 裡「本節可修改的完整實驗」那一格」（第 176 行） | notebook 裡標題是「本節可修改的完整實驗」的那一格是說明格（Markdown），程式在它的下一格。其他頁都寫「下面那一格（程式）」。照字面找的初學者可能點開說明格去改。 |
| 9 | 建議 | 〈先把各軸說清楚〉「axis0 是 B=2（兩張圖），axis1 是四個 channel，axis2、axis3 才是高度、寬度」（第 42 行） | 第 1 章〈VGG 風格小 CNN〉明訂本書寫軸的編號用「第 0 軸、第 1 軸」，其他頁也都這樣寫，只有本頁用 axis0～axis3。初學者可能以為 axis0 是程式裡的某個名稱。 |

各項的處理見下方〈定稿修正〉。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://arxiv.org/abs/2107.08430 (YOLOX, arXiv PDF): §2.1 'Decoupled head' 段落（'Replacing YOLO's head with a decoupled one…'）、Fig. 2 說明
- https://arxiv.org/abs/2108.07755 (TOOD, arXiv PDF): §3.2 Task Alignment Learning（'comprises a sample assignment strategy and new losses'）、§3.2.1 式 (9) t=s^α×u^β 與 'Training sample assignment'（每個物件取 t 最大的 m 個）、§3.2.2 normalized t、§4 'On hyper-parameters'（α=1、β=6、m=13）、附錄 'Optimization'（多物件衝突時交給面積最小的物件）
- https://arxiv.org/abs/2006.04388 (Generalized Focal Loss v1, arXiv PDF): §1 與 Fig. 3（Dirac delta、遮擋／模糊造成的邊界模糊、攤平的分佈）、§3 式 (6) DFL 定義與 'rapidly focus…nearest two to y'、式 (7) GFL 統一式與 'FL, QFL and DFL are all special cases of GFL'、Table 2 與 §4（實驗在 mmdetection 上進行，n=16、Δ=1）
- https://arxiv.org/abs/1804.02767 (YOLOv3, arXiv PDF): §2.2 Class Prediction（independent logistic classifiers、binary cross-entropy）
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/nn/modules/head.py: Detect.__init__（nc=80、reg_max=16、no=nc+4*reg_max、cv2／cv3 定義、legacy 分支）、inference 的 scores.sigmoid()
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/nn/modules/block.py: class DFL（arange(c1) 固定權重、沿 bin 軸 softmax 後求期望值）
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/utils/loss.py: DFLoss（clamp 到 reg_max-1-0.01、左右 bin 權重）、BboxLoss（只對 fg_mask 算 CIoU 與 DFL、bbox2dist 的 reg_max-1 上限）、v8DetectionLoss（TaskAlignedAssigner topk=10、alpha=0.5、beta=6.0、stride；pred_scores.detach().sigmoid() 與 pred_bboxes.detach() 交給 assigner；BCE 涵蓋所有 anchor；preprocess 補齊 GT；bbox_decode 用 view(b,a,4,K).softmax(3)）
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/utils/tal.py: TaskAlignedAssigner（stride_val=stride[1]、get_pos_mask、get_box_metrics 的 score^α×overlap^β、iou_calculation 為 CIoU clamp(0)、select_topk_candidates、select_candidates_in_gts 的小框放大與嚴格框內、select_highest_overlaps 在框內 GT 中取 CIoU 最大、_forward 的 normalized align metric、get_targets）與 bbox2dist
- https://github.com/ultralytics/ultralytics @ 441632cdfd19e22e60a4b1b1999d46326ca51ec4 ultralytics/nn/tasks.py: DetectionModel.init_criterion（非 end2end 用 v8DetectionLoss）、parse_model 的 legacy 判定（C2f 系列模型維持 legacy=True；C3k2／A2C2f／C2fCIB 時設為 False）
- https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/gfl_head.py: Integral（linspace(0, reg_max, reg_max+1)）、GFLHead 的 reg_max 文件字串（'Max value of integral set {0, ..., reg_max}'）與 gfl_reg 輸出 4*(reg_max+1)
- PyTorch 2.9.1（本機 .venv-model 實測）：nn.Module.zero_grad(set_to_none=True) 預設與結果為 None、torch.allclose 預設 rtol=1e-05／atol=1e-08、第二次 backward 的 'Trying to backward through the graph a second time' 錯誤、F.cross_entropy 目標索引越界時拋出 IndexError 'Target 4 is out of bounds.'

這一頁沒有發現與來源不符的說法。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 讀者審查 | 一層 8→6 的 1×1 等於兩層 1×1；參考答案拿 None 對全 0 當梯度路徑的差別 | 已修正：成本段補一句：這層 1×1 等於並排的 8→4、8→2（54=36+18），只拆輸出層不算分工。參考答案第 2 題改用共用 3×3 收到兩種梯度當主要例子，並註明全 0 和 None 在數值上都等於沒有梯度，只是 PyTorch 記錄方式不同。 |
| 2 | 讀者審查 | 沒說要改程式裡三個 zero_grad 的哪一個 | 已修正：寫明要改第二次反傳前那個（程式裡的第二個），點出失敗的斷言是 model.box_branch[0].weight.grad is None，並說明改第一個沒有作用的原因。暫存副本實測：只有改第二個會失敗。 |
| 3 | 讀者審查 | [2,4,4,4] 和 12.1 節的 [B,P,4] 是什麼關係沒說 | 已修正：補一句：16 個位置各是一個候選點（P＝16），攤平高、寬並把 channel 移到最後，就是 [B,P,4] 與 [B,P,C]。 |
| 4 | 讀者審查 | 連結的 head.py 中 cv3 有 legacy 與 DWConv 兩種寫法 | 已修正：已確認：固定 commit 的 legacy 預設是 False，parse_model 只有遇到 C3k2、A2C2f、C2fCIB 才把它設成 False，所以 YOLOv8 走 legacy 分支。參考來源補一句：YOLOv8 用 self.legacy 為真的那一種，另一種是較新版本的類別分支。 |
| 5 | 讀者審查 | 「也稱它為 logits」讓 logits 在同一段指兩種東西 | 已修正：改成「還沒經過 softplus 的原始數（術語表的 logits 也包含這種框的原始輸出）」，並明說本頁單說 logits 時，都指類別分支的輸出變數 logits。 |
| 6 | 讀者審查 | 「每一項再各自…乘回 θ」讀起來像加總後還有下一步 | 已修正：改寫成：每一項都是從自己的 loss 沿自己的分支用連鎖律一路乘回 θ 得到的，兩條路在共用特徵 f 會合，梯度在那裡相加。 |
| 7 | 讀者審查 | 標題「收益、代價與下一項判斷」對不上內容 | 已修正：改成和其他頁一致的「收益、代價與常見錯誤」，並確認沒有連結指向舊的錨點。 |
| 8 | 讀者審查 | 「本節可修改的完整實驗」那一格其實是說明格 | 已修正：改成「下面那一格程式」，和 00、02 等頁的寫法一致。 |
| 9 | 讀者審查 | 軸編號用 axis0～axis3，和全書寫法不一致 | 已修正：改成第 0 軸到第 3 軸。 |
| 10 | 先前查核 | 同一段混用「第 2、3 項」和「輸出的第 3 行」兩套編號 | 已修正：照要求的修正刪掉「輸出的第 3 行，」，改回「（`shared-backbone gradient cosine` 那行）」。 |
| 11 | 先前查核 | 「固定在 commit」中的 commit 沒有解釋 | 未改：已過時：本頁和其他三頁的正文現在都沒有「commit」或「固定在 commit」的字樣，參考來源只是連結文字，所以不需要補解釋。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/12-anchor-free.md`、`docs/lessons/12-decoupled-head.md`、`docs/lessons/12-assignment.md`、`docs/lessons/12-dfl.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/12-decoupled-head.md 參考答案〈第 2 題〉最後一句「全 0 和 None 在數值上都等於沒有梯度，差別只在 PyTorch 的記錄方式。」 | 這句是這次修改新加的，「差別只在記錄方式」說得太滿。PyTorch `Optimizer.zero_grad` 的文件明說，optimizer 對這兩種情況處理不同：`.grad` 是 None 的參數，那一步直接跳過；全 0 的參數照樣執行一步。用了 momentum 或 weight decay（例如 AdamW）時，梯度全 0 的權重仍然會變。讀者若因此以為「梯度清成 0 就等於凍結參數」，之後訓練時會看到沒有梯度的權重還在變動。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」。對照釘選的 Ultralytics 441632c head.py：cv3 在 `self.legacy` 為真時是兩個 `Conv(…,3)` 再接 `nn.Conv2d(…,1)`；tasks.py 只在模型用到 C3k2、A2C2f、C2fCIB 時把 legacy 設成 False，所以 YOLOv8 走 legacy 分支，敘述正確。常見錯誤裡「把第二個 `model.zero_grad()` 改成 `set_to_none=False` 會讓 `box_branch[0].weight.grad is None` 失敗、改第一個沒有作用」，與 lesson_cases/12-decoupled-head.py 第 30–37 行的呼叫順序相符。第 2 題答案的 638 個參數（584＋54），以及全 0 和 None 的說明，在該例的類別列上成立，也與第 2 章的說法一致。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：先前查核 2 項與讀者 9 項，都在〈定稿修正〉處理；〈來源對照〉列出 YOLOX 論文與 Ultralytics 441632c head.py；批次檢查留下的 1 項，由〈後續編輯的檢查〉核對。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/12-decoupled-head.md 第 11、36 行 | 殘句與內部稱呼：「所有檢查都在自己的副本暫存副本進行」「我在副本裡改壞程式中兩行，checker 兩行都抓到」（checker 指摘錄比對，讀者看不出來）。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 11 行已清理；第 36 行的查核者已改成「摘錄檢查」。處理說明屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/12-decoupled-head.md 第 3 輪第 1 項處理欄 | 「「摘錄檢查兩行都抓到」改成「摘錄檢查兩行都抓到」」是替換工具造成的同語反覆。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

第 3 輪第 1 項屬實。第 4 輪第 1 項不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪第 1 項處理欄（第 159 行） | 發現點名的是第 3 輪第 1 項處理欄的同語反覆「「摘錄檢查兩行都抓到」改成「摘錄檢查兩行都抓到」」。那一格已由產生器重寫成「已修正：點名的文字已不在紀錄裡（改寫或刪除）。」，同語反覆已經不在。處理欄卻寫「未改：點名的 2 處文字仍在紀錄裡」，只因為正文第 36 行有改好後的「摘錄檢查兩行都抓到」。同一格開頭又說處理說明已重寫，前後矛盾。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[evolution當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/evolution.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/evolution.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/evolution.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。
