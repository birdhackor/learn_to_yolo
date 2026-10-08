# 審查紀錄：完整閱讀路線

審查範圍：`docs/learning-path.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

（這一輪同時查核：`docs/index.md`、`docs/learning-path.md`、`docs/planning/outline.md`；下表只列和本頁有關的發現。）

結論：沒有必要問題，通過。只剩一個 should：首頁「再散佈」和全站「再散布」用字不一致。

我做了哪些檢查（都在暫存副本裡做，log 在同層的暫存副本）：

1. 建置與驗證：zensical build --clean --strict 回報 No issues found、exit 0；validate_site.py exit 0，連結、錨點、Colab 配對、Markdown 呈現都通過；validate_preparation.py exit 0。README 的相對連結都指得到檔案，43 本 notebook 都是合法 JSON。另外跑了 validate_lessons.py，只因為還沒有定稿審查而失敗，這是審查用的事實與寫作規範清單列出的暫時狀態，不算這次查核的頁面的問題。repo 裡這三頁和編者暫存副本的同名檔逐位元相同；同一時段其他被改動的檔案，各屬於其他部分。

2. 頁面敘述和程式、指令對照（實際跑過的有）：
   - 首頁的格子算法：用 build_targets 驗證，紅框 [8,12,24,28] 分到 (列 1, 欄 1)，中心在 (40,20) 的框分到 (列 1, 欄 2)。
   - 7.4：跑了 lesson_cases/07-training.py，確實做了 3 次參數更新；miniyolo.train 的 --help 確實有 --steps、--samples、--device。
   - 大綱 7.3「整批沒有正格」：grid_loss 在這種情況下 box 與 classification 都是 0。
   - 大綱 8.1：跑了 08-own-images.py，非正方形圖 → CHW → letterbox → 推論 → 框還原回原圖，全部成立。
   - 環境檢查 notebook 確實只印 Python、PyTorch、GPU 與 git 資訊。
   - lesson_cases 裡沒有用到 cuda，也沒有下載；08-own-images 與 18-video 讀的檔案都是程式自己先寫出的。所以「資料由程式自己產生，不必下載資料集」成立。

3. 遺留項目與額外指示：
   - outline 第 154 行的遺留項目已修好，寫法和 feedback.md 第 26 行一致。
   - 受程式改動影響的段落裡確實沒有這三頁的項目；拿程式修改清單的程式修正逐一對照三頁內容，沒有任何一句因此變成錯的。
   - 首頁「還沒確認」已改成現在式的「沒有驗證」。它是 status.md〈沒有驗證的事〉的子集，沒有互相矛盾。
   - 改標題前確認過：docs、README、notebooks、scripts、tests、overrides 都沒有連到 _3 錨點，改名後錨點仍是 _3。
   - 摺疊區的新標題不再讓人以為是在講網站的製作經過。
   - 閱讀路線 42 節的連結文字和導覽、section-map 完全一致，一句話描述和各課頁 H1、內容相符。只把 18–20 章標成選修，和大綱、feedback.md 一致。

4. 數字與時態：這次沒有加入任何 Mac 量到的訓練或計時數字。編者列出的依賴紀錄的值，對應的紀錄都在，內容也相符，例如 custom-data-160-step 的 train mAP50 是 0.0068、validation 是 0，gpu-smoke 的 status 是 passed。三頁沒有修訂經過的敘述，也沒有「仍待」「之後會」這類計畫語氣，和審查用的事實與寫作規範清單、status.md、README、課頁都一致。

大綱裡沒被改到的各章敘述，我也對照了課頁和程式：完成條件、表格各列、[8,12,24,28] 沿用到 9.1／11.3／11.4、模型沒有用 Dropout／BatchNorm、3.3 兩個模型從同一份初始權重開始、第 2 與 17 章寫明可以停在哪裡。全部成立。

## 讀者審查與技術查核

### 讀者審查（AI 以這一頁的讀者身分閱讀）

方法：讀了哪些：
- 定稿事實清單審查用的事實與寫作規範清單。
- docs/learning-path.md 全文（在自己的暫存副本讀）。
- 照閱讀順序讀了 docs/index.md、docs/glossary.md、docs/status.md 全文，以及 docs/planning/outline.md 的相關段落。
- 42 節課文頁的開頭與前置段落。完整或分段細讀了：00-warmup、01-small-cnn（模型定義、softmax、交叉熵）、02-diagnostics（四個檢查與錯誤處理）、06-decode-nms、07-training（3 步與 160 步補充）、09-anchors、10-multiscale、16-training、17-capstone，以及 18、19、20 的前置。

建置與渲染：
- 在暫存副本用 rsync 複製 repo，跑 .venv-docs/bin/zensical build --clean --strict，結果 No issues found。
- 抽出 site/learning-path/index.html 的 article 逐段核對渲染結果（摺疊區、粗體、連結都正常）。
- Playwright 的瀏覽器沒有安裝，改用 qlmanage -t 把頁面渲染成 PNG 看版面。
- 在暫存副本跑 scripts/validate_site.py：連結與錨點、Colab 配對都 passed。
- 寫了一段小程式，比對本頁 42 項的編號、標題、連結和 zensical.toml 導覽及各節 H1：42 項全部存在且依序。
- 這頁沒有圖。

程式碼對照：
- miniyolo/losses.py 的 grid_loss 是 box、objectness、classification 三項。
- miniyolo/metrics.py 用 overlap >= iou_threshold，所以「至少 0.5」正確。
- miniyolo/geometry.py 的 nms。

在暫存副本跑的 lesson（PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg）：
- lesson_cases/00-warmup.py：四行輸出與正文相同。
- lesson_cases/07-loss.py：印出 box、objectness、classification 三項。
- lesson_cases/07-training.py：印出「3 real CPU optimizer steps」。
- 不設 PYTHONPATH 再跑 07-loss.py：得到 ModuleNotFoundError: No module named 'miniyolo'。

grep 查證：
- 課文程式裡的 class／self／super、dict 索引、f-string、list comprehension。
- e 與 ln 第一次出現的位置：第 1 章用到，9.1 節才說明。
- NCHW 與 [B,C,H,W] 的寫法、全站對 runtime 的譯名、第 11、12 章提到的 YOLO 版本。
- 術語表有沒有收增強、MiniYOLO、GridDetector、STAL 等詞。

網路查證：
- Google 繁中 Colab 文件把 runtime 譯作「執行階段」「變更執行階段類型」（docs.cloud.google.com/colab/docs/manage-runtimes?hl=zh-TW）。
- Python 官方繁中教學 https://docs.python.org/zh-tw/3/tutorial/ 的章節：第 4 章流程控制、第 5 章資料結構、第 7 章輸入輸出、第 8 章錯誤和例外、第 9 章 Class。

沒有在 repo 根目錄裡執行或寫入任何東西。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/learning-path.md 第 17 行：「第 9–16 章（版本節）……每節都寫明自己的起點（從哪個簡化設定出發），以及只改了哪一項機制，方便看出這一項改動本身的作用。」 | 這句和各節自己的說法相反，會讓讀者以為每節都是「只差一項」的對照，而且看得到這項改動的效果。 不是每節都只改一項： - 9.1 節寫「本節的設計只改兩件事：寬高怎麼表示，以及每格有幾個槽、由哪個槽負責」，另外四項 loss 直接相加，沒沿用第 7 章框 loss 乘 5。 - 第 10 章寫明模型「換成新寫的小網路」，channel 數、得到 4×4 的方式、head 初始化都和 GridDetector 不同，「不是對 GridDetector 原封不動多接一個 head 的對照」。 - 16.3 節一次介紹三個技巧，只對 Progressive Loss 做實驗。 「看出作用」也和各節的範圍說明衝突： - 9.1 節：「anchor 是否真能讓偵測變好，本節沒有證據」。 - 11.1 節：「本節不做這個實驗」。 - 12.4 節：「本節只驗證計算，不證明準確度會提高」。 - 驗證範圍頁也把「比較各機制的效果」列為沒有驗證。 照本頁去讀，讀者會把各節的小實驗當成「這個機制有效」的證據。 |
| 2 | 必要 | docs/learning-path.md 第 7 行：「必要：會基本 Python（變數、if／for、函式、list）。」（首頁 docs/index.md 第 9 行是同一句） | 括號裡看起來像完整清單，但課文程式從第 1 章起就需要更多： - 每個模型都寫成 `class SmallCNN(nn.Module):`，裡面有 `super().__init__()`、`self.features = …`、`def forward(self, x)`。第 1 章只說「`__init__` 建立這些層，`forward` 寫資料依序經過它們的路線」，全書沒有說明 class、self、繼承。 - dict 也常用，例如 4.2 節的 `meta['resized_hw']`、7.1 節的 `zip(target["boxes"], target["labels"])`、7.2 節的 `target['positive']`。 - 還有 f-string（3.3、4.1、4.2 節的 `print(f"…")`）和 list comprehension（3.3 節練習要改的 `[Block(residual) for _ in range(3)]`）。 只照這份清單準備的程式新手，到第 1 章就讀不懂模型定義，本頁也沒有指出要去哪裡補。 |
| 3 | 建議 | docs/learning-path.md 第 9 行：「數學：高中程度的函數、對數、向量與矩陣乘法即可。」 | 第 1 章的 softmax 就用到 e^z，交叉熵用到 −ln p。第 2 章與 4.1 節也一樣，例如 sigmoid 1/(1+e^−z)、ln 2≈0.693。 但 e 與 ln 要到 9.1 節才說明。9.1 節自己寫：ln 是以 e≈2.718 為底，「不是高中課本不寫底數時的以 10 為底」。所以只學過常用對數的讀者，第 1 章就看不懂 ln 2≈0.693 是怎麼來的，本頁的清單卻讓他以為已經準備夠了。 |
| 4 | 建議 | docs/learning-path.md 第 15 行：「第 17 章是結業任務，前置在第 6–7 章，不必先讀第 9–16 章；第 18–20 章是選修，可按需求挑讀。」 | 「可按需求挑讀」讀起來像這三章各自獨立、可以直接跳進去，實際上有依賴，也有設備限制： - 第 19 章的前置寫了第 18 章的逐幀處理。 - 第 18、20 章的前置都有 8.1 節（非正方形圖片的 letterbox 與框還原）。 - 第 20 章的 TensorRT 需要 NVIDIA GPU。Colab／CPU 範例只跑 ONNX Runtime，TensorRT 部分是作者在雲端 L4 上的實測。 只想學 tracking 或 TensorRT 的讀者，要等進了該節才發現這些。另外，「前置在第 6–7 章」容易被讀成只讀第 6–7 章就夠，但第 6–7 章本身是建立在第 0–5 章上。這三條路線擠在同一段裡，也不容易一眼看清。 |
| 5 | 建議 | docs/learning-path.md 第 84 行，〈遇到卡住的地方〉第 4 點：「實驗失敗時，先保存完整的錯誤訊息，再照訓練診斷的順序排查。」 | 第 2 章的順序是「loss 不降時」的四個檢查：資料與標籤、一次 forward／backward／step 的梯度、少量資料 overfit、held-out。它處理的是程式跑得完、但結果不對的情況。 新手最常遇到的「實驗失敗」卻是程式直接報錯停下，例如： - 在自己電腦上沒加 `PYTHONPATH=.` 時，出現 `ModuleNotFoundError: No module named 'miniyolo'`。我在暫存副本不設 PYTHONPATH 跑 07-loss.py，就是這個錯。 - 做練習時改了程式，斷言還在核對原題答案，出現 `AssertionError`。 照第 2 章的四步查不到這兩種錯，本頁也沒說保存錯誤訊息之後要看哪裡。 |
| 6 | 建議 | docs/learning-path.md 第 68 行：「16.3 YOLO26 訓練補強：Progressive Loss、STAL 與 MuSGD」 | 其他項目的冒號後面都有一句白話說明，或在括號裡解釋術語。這一項只列出三個沒解釋的英文名稱，術語表也沒有，讀者看不出這節在學什麼，也看不出三項裡只有 Progressive Loss 有做實驗。 |
| 7 | 建議 | docs/learning-path.md 第 55 行：「11.3 圖與框同步增強：畫素怎麼變，框就怎麼變」 | 這裡的「增強」是資料增強（data augmentation：訓練時把圖片翻轉、裁切等，再當成新的訓練樣本），但本頁沒有說明，術語表也沒有這一條。 初學者很容易讀成「把影像畫質變好」。11.3 節自己就得特別澄清「資料增強不是把畫質變好」。 |
| 8 | 建議 | docs/learning-path.md 第 24、38、52、57、62、65、73 行的說明句（1、6.2、10、12.1、13.2、15.2、18） | 這些句子照搬各節的副標，沒讀過那一節的人看不懂或會誤解： - 1「讓局部圖樣重複使用」：重複使用的是同一組小權重（濾鏡），不是圖樣。 - 6.2「把預測逐筆算成證據」：沒說「證據」指什麼。 - 10「同一個 pixel 框看兩種尺度」：不容易斷句。 - 12.1「量出四條邊」：實際量的是候選點到框四邊的距離。 - 13.2「拿掉 NMS 前，重複候選學會了什麼」：看不出這節做什麼。 - 15.2「互動範圍是一筆預算」：只是比喻，沒說內容。 - 18「分清 FPS（每秒幀數）與延遲」：術語表提醒來源幀率、處理速率、延遲是三個不同的量；第 18 章比的是「每秒處理完幾幀」和延遲，只寫「每秒幀數」容易被當成影片本身的幀率。 |
| 9 | 建議 | docs/learning-path.md 第 48 行：〈C. YOLO 的演化機制（第 9–16 章）〉 | A、B 兩組都有開頭說明，C 組卻直接列出 19 節。 標題只有第 9、10、13–16 章寫出 YOLO 版本，沒寫的有： - 第 11 章（CSP、特徵融合、增強、IoU loss），各節對照的是 YOLOv4／v5。 - 第 12 章（anchor-free、decoupled head、assignment、DFL），對照的是 YOLOv8。 - 15.1 節是 attention 的基礎，不是某一版的改動。 讀者想照「演化」的順序讀，會以為課程從 YOLOv3 直接跳到 YOLOv10。 |
| 10 | 建議 | docs/learning-path.md 第 3 行：「不必啟動 Colab 的執行環境（runtime，Colab 替你開的雲端機器）」 | 同站其他頁都把 Colab 的 runtime 叫「執行階段」：驗證範圍頁寫「不必選 GPU 執行階段」，8.1 節寫「同一個執行階段只要跑一次」「Colab 執行階段結束後」。Google 的繁中 Colab 說明也用「執行階段」「變更執行階段類型」。只有本頁叫「執行環境」。 此外，每本 notebook 最上面那一格叫「環境格」，4.2 節還寫「執行環境格和…那一格」。讀者容易把「執行環境」和「環境格」當成同一件事，在 Colab 選單上也找不到「執行環境」這個詞。 |
| 11 | 建議 | docs/learning-path.md 第 13 行，〈形狀記號〉摺疊區：「C=3 是每個位置的數值個數（channel，通道）」 | 括號把 channel 接在「數值個數」後面，讀起來像 channel 就是「個數」。 術語表的定義是：channel 是特徵圖的一層（每個位置在這一層有一個值），channel 數才是同一個位置有幾個值。後面課文常說「第一個 channel」「把 3 個 channel 變成 6 個」，照本頁的說法會讀不通。 |
| 12 | 建議 | docs/learning-path.md 第 3 行：「可訓練的格子偵測器（grid detector）：它把圖切成格子，每格各自預測框，也就是第 7 章的 MiniYOLO」 | 同一個模型在本頁與各節有四種叫法，術語表都沒有收： - 格子偵測器（grid detector）：只出現在本頁，課文不再用這個中文名。 - MiniYOLO。 - Grid MiniYOLO：導覽與第 7 章各節的標題。 - GridDetector：程式的 class 名稱，出現在 7.5 節前置「三步訓練（GridDetector 模型）」、第 10、17、20 章。 讀者讀到 7.5 節或第 10 章的 GridDetector 時，無從確認它就是本頁說的格子偵測器。 |

### 事實查核（AI 對照 repo 的程式、紀錄與頁面引用的來源）

方法：一、Repository 來源
本機 repo 根目錄（https://github.com/birdhackor/learn_to_yolo），branch release/lessons-v0.4.0，commit 31527417d328e18418c438cb76b4630a8c695f33，工作樹乾淨。所有程式都在暫存副本執行：暫存副本（以 rsync 建立）。

讀過的檔案：
- 被審頁面與事實清單：docs/learning-path.md（全文）、審查用的事實與寫作規範清單（全文）。
- 設定與對照：zensical.toml（nav）、section-map.json、docs/index.md、docs/glossary.md（常用符號，以及 IoU、NMS、AP、AP50、DFL、FPS、tracking 各列）、docs/status.md（前 40 行）、docs/planning/outline.md（第 3、9、17、19、21、132、134 行）、docs/planning/feedback.md（第 26 行）。
- 全文讀過的課程頁：docs/lessons/00-warmup.md、02-diagnostics.md。
- 讀過部分段落的課程頁：01-small-cnn.md（1–120 行）、06-decode-nms.md（1–80 行）、06-evaluation.md（1–60 行）、07-training.md（1–60 行與 160 步段落）、08-own-data.md（來源切分段落）。
- 讀開頭（起點、只改什麼、前置）的課程頁：03-identity、03-projection、03-comparison、07-heldout、07-inference、09-anchors、09-anchor-clustering、10-multiscale、11-csp、11-fusion、11-augmentation、11-iou-loss、12-anchor-free、12-decoupled-head、12-assignment、12-dfl、13-dual-assignment、13-nms-free、14-feature-module、15-attention-bridge、15-area-attention、16-dfl-free、16-inference-head、16-training、17-capstone、18-video、19-tracking、20-deployment。
- 全部 42 頁的 H1。
- 程式與其他檔案：lesson_cases/06-evaluation.py（match_iou_threshold=.5，比較用 >=）、06-decode-nms.py、17–20 四節的 import、19-tracking.md 與 19-tracking.py 對 18 的依賴、miniyolo/losses.py 的 grid_loss（total=5*box+objectness+classification）、notebooks/07-training.ipynb（四格結構，環境格 clone lessons-v0.4.0，說明寫「各節互相獨立」）、scripts/validate_site.py。

二、外部來源
- https://arxiv.org/abs/1409.1556：摘要寫「very small (3x3) convolution filters」。
- https://www.robots.ox.ac.uk/~vgg/research/very_deep/：作者 Simonyan & Zisserman，Visual Geometry Group, University of Oxford；「very small 3×3 filters」。
- https://onnx.ai/：首頁寫 Open Neural Network Exchange、「an open format built to represent machine learning models」。
- https://developer.nvidia.com/tensorrt：首段寫「ecosystem of tools … high-performance deep learning inference」。
- https://arxiv.org/abs/1911.11929：摘要寫「Cross Stage Partial Network (CSPNet)」。
- GFL 論文：https://arxiv.org/abs/2006.04388 與 NeurIPS 2020 摘要頁。
- https://raw.githubusercontent.com/ultralytics/ultralytics/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py：DFLoss docstring 寫「Distribution Focal Loss (DFL)」。
- https://docs.ultralytics.com/models/yolo26/：Key Features 列出 DFL Removal、nms=False、Progressive Loss & STAL、MuSGD。
- https://arxiv.org/abs/2405.14458：摘要寫「consistent dual assignments for NMS-free training」。
- https://arxiv.org/html/2502.12524：§3.2 寫「area attention module」。

三、執行的程式與指令（都在暫存副本，原 repo 只做唯讀的 git log、git tag）
1. 九節課程程式。指令形式：PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/<id>.py。依序跑 00-warmup、06-decode-nms、06-evaluation、07-targets、07-loss、07-training、07-inference、10-multiscale、15-attention-bridge，全部 exit 0。核對到的確定值：
   - w 1→1.80、gradient=-8；
   - 「3 real CPU optimizer steps」；
   - loss 分成 box、objectness、classification 三項；
   - 同一個紅框從標註 [8,12,24,28] 換成 target [0,0.25,0.25,0.25]，再解碼回 [8.0016,12,24.0016,28]；
   - 兩個 head 的 shape 是 (1,8,8,7) 與 (1,4,4,7)；
   - tokens 是 4 個位置。
2. python -m miniyolo.train --help：確認有 --steps。
3. .venv-docs/bin/zensical build --clean --strict：exit 0，「No issues found」。
4. python3 scripts/validate_site.py：exit 0。連結與錨點、Colab 配對、Markdown 轉換都通過。另外直接檢查了 learning-path 頁輸出的 HTML。
5. python3 scripts/validate_preparation.py：exit 0。
6. python3 scripts/validate_lessons.py：exit 1，失敗原因只有 59 頁「no review」（含 learning-path），屬審查迴圈進行中的暫時狀態，不是頁面文字的問題。
7. grep 檢查：
   - 42 頁課程頁各自連到 lessons-v0.4.0 的 notebooks/<id>.ipynb；
   - C 作類別數時，各節都有寫明；
   - 「版本節」「版本章」「選修」在全站的用法；
   - 「分佈」「分布」、「平均精確率」的一致性。
8. 唯讀查詢原 repo 的 git tag：lessons-v0.4.0 尚未建立，屬發布前的狀態。

四、照頁面走過的程序
1. 從第 0 章開始讀：執行暖身程式，核對一個參數的手算。
2. 卡關建議第 1 條：用 07-targets 與 07-inference，追同一個物件從標註、target 到解碼結果。
3. 卡關建議第 2 條：用 07-loss 與 07-training，確認三項 loss 分開印出。
4. Colab 按鈕：確認連到該節 notebook，環境格 clone 固定 tag，而且各節互相獨立。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/learning-path.md 第 17 行：「每節都寫明自己的起點（從哪個簡化設定出發），以及只改了哪一項機制，方便看出這一項改動本身的作用。」 | 這句把第 9–16 章每一節都說成「只改一項機制、看得出這項改動本身作用」的對照，但被它摘要的頁面自己寫了例外： - 9.1〈YOLOv2 anchor 與框參數化〉：寫「本節的設計只改兩件事：寬高怎麼表示，以及每格有幾個槽、由哪個槽負責」。另外四項 loss 直接相加、沒有沿用第 7 章框 loss 乘 5 的權重，頁面還說「anchor 是否真能讓偵測變好，本節沒有證據」。 - 第 10 章：寫模型「換成新寫的小網路」（TwoScale），channel 數、得到 4×4 的方式、head 初始化都和 GridDetector 不同，「所以本節不是『對 GridDetector 原封不動多接一個 head』的對照」。 - 16.3：一節介紹 Progressive Loss、STAL、MuSGD 三個技巧，只對第一個做實驗。 〈課程大綱〉第 19 行對同一件事的寫法有保留：「每個實驗只改一項主要機制……和起點還有其他差異時，該節會寫明，例如第 10 章的兩尺度模型是另外寫的小網路」。讀者若照本頁的說法讀第 10 章，會把 TwoScale 的結果整個歸給「多一個尺度」，而這正是那一節特別提醒不能做的歸因。 |
| 2 | 建議 | docs/learning-path.md 第 15 行：「第 17 章是結業任務，前置在第 6–7 章，不必先讀第 9–16 章；第 18–20 章是選修，可按需求挑讀。」 | 同一句替第 17 章寫了前置，第 18–20 章卻沒寫，容易讀成這三章互不相依、可任挑一章直接讀。實際的前置關係如下： - 19〈簡易 tracking〉開頭把「第 18 章影片管線的逐幀處理」列為前置，頁末還用 runpy 載入 lesson_cases/18-video.py，接上第 18 章模型的真實輸出。 - 18〈影片串流〉和 20〈ONNX／TensorRT〉都把 8.1〈自己的圖片推論〉（letterbox 補邊與框還原）列為前置。 - 第 17 章的前置（07-heldout、07-loss、06-evaluation、07-inference）不含第 8 章。 所以只照本頁讀完第 17 章就跳到 19，或跳過第 8 章直接讀 18、20，都會缺前置。 |
| 3 | 建議 | docs/learning-path.md 第 32 行：「IoU（Intersection over Union，重疊面積÷聯集面積）」 | 以下幾處都用「交集面積」，本頁換成「重疊面積」，用詞不一致；寫作規則要求術語和術語表一致： - 術語表 IoU 列：「兩框的交集面積除以聯集面積」 - 6.1：「兩框交集面積除以聯集面積」 - 11.4、19：「交集面積÷聯集面積」 此外 Intersection 對應的是「交集」，正好和後面的「聯集」成對。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 讀者審查 | 「每節只改一項機制、看得出作用」與各節不符（must） | 已修正：已和 9.1、第 10 章原文及課程大綱核對。改寫成：每節寫明起點和這一節改了什麼；和起點還有其他差異時該節會列出（舉 9.1 同時改寬高寫法與槽數、第 10 章另寫小網路為例）。並註明這些小實驗不是公平的效果比較，連到驗證範圍頁。 |
| 2 | 事實查核 | 同上，與課程大綱的保留寫法不一致（must） | 已修正：與讀者必要一起改。新寫法和 outline.md 一致，不再說「看出這一項改動本身的作用」。 |
| 3 | 讀者審查 | Python 必要條件漏了 class、dict（must） | 已修正：改成「變數、if／for、函式、list 與 dict，也看得懂 class 的寫法（class、self 與方法；這個 class 和物件的「類別」是兩回事）」，並附 Python 官方教學第 3–5 章與第 9 章。首頁同一句也一起改了，讓兩頁一致。f-string 沒有列入，因為從輸出就看得懂。 |
| 4 | 讀者審查 | 數學前置沒提到 e 與 ln | 已修正：改成「指數與對數」，並補一句：課文的 ln 以 e≈2.718 為底，例如 ln 2≈0.693，第 1 章就會用到（已和第 1 章交叉熵那段核對）。 |
| 5 | 讀者審查 | 第 18–20 章「可按需求挑讀」沒寫依賴關係 | 已修正：三條路線改成條列。第 17 章改成「讀完第 0–7 章就能做」，並寫明第 18、20 章要先讀 8.1 節、第 19 章要先讀第 18 章（已和各章的前置核對）。TensorRT 需要 GPU 這點第 20 章自己就有說明，所以沒有加。 |
| 6 | 事實查核 | 第 18–20 章的前置沒有寫出 | 已修正：與讀者 should 4 一起改，用的是同一句前置說明。 |
| 7 | 讀者審查 | 「實驗失敗」只導向訓練診斷 | 已修正：分成兩種情況。程式報錯時先看錯誤訊息的最後一行：ModuleNotFoundError 是沒加 PYTHONPATH，見第 0 章〈在自己的電腦執行〉；練習時的 AssertionError 見第 0 章練習 1。程式跑得完但 loss 不降時才照訓練診斷排查。已在暫存副本不設 PYTHONPATH 執行 07-loss.py，確實出現這個 ModuleNotFoundError。 |
| 8 | 讀者審查 | 16.3 只列三個英文名稱 | 已修正：照 16.3 節原文改寫：訓練中把 loss 比重從一對多分支逐步移到推論用的一對一分支（Progressive Loss）；另外說明 STAL 與 MuSGD，只對第一項做實驗。 |
| 9 | 讀者審查 | 11.3「增強」沒有說明 | 已修正：照 11.3 節的定義加上括號說明「資料增強：訓練時把圖片翻轉、裁切等，當成新的訓練樣本」。術語表沒有另加一條，因為 11.3 節自己有定義。 |
| 10 | 讀者審查 | 幾節說明句照搬副標，讀不懂 | 已修正：只改兩處不精確的：第 1 章改成「同一套小權重（濾鏡）在每個位置重複使用」（與第 1 章原文一致）；12.1 改成「量出到框四條邊的距離」。6.2、10、13.2、15.2、18 沒有改，這些句子和各節標題一致，內容也沒有錯，屬於用詞偏好。 |
| 11 | 讀者審查 | C 組沒有開頭說明，看不出各章對照的版本 | 已修正：加一句各章對照的版本：9 是 YOLOv2、10 是 YOLOv3、11 是 v4／v5、12 是 v8、13 是 v10、14 是 YOLO11、15 是 v12、16 是 YOLO26。每一章都和該章頁面的對照說明核對過。 |
| 12 | 讀者審查 | 「執行環境」與全站的「執行階段」不一致 | 已修正：改成「不必連上 Colab 的執行階段（runtime：…）」。 |
| 13 | 讀者審查 | channel 的括號讓人以為 channel 就是個數 | 已修正：改成「C=3 是 channel（通道）數，也就是每個位置有幾個數值」，並說明 RGB 的 3 個值各排成一層，就是 3 個 channel。 |
| 14 | 讀者審查 | 格子偵測器、MiniYOLO、GridDetector 的叫法對不起來 | 已修正：改成「第 7 章可訓練的 grid MiniYOLO：把圖分成格子，由物件中心所在的那一格預測框（程式裡的 class 叫 GridDetector）」，也順便避開首頁已不再使用的「切成」。術語表沒有加條目。 |
| 15 | 事實查核 | IoU 寫成「重疊面積」 | 已修正：改成「交集面積÷聯集面積」，和術語表、6.1、19 一致。 |

修正後由另一位 AI 檢查這一批頁面（`docs/index.md`、`docs/learning-path.md`、`docs/glossary.md`、`docs/status.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：標題與審查範圍對應導覽；第 1 次查核（和首頁、課程大綱同輪）列出本頁發現；讀者審查 12 項（含 2 項必要）與事實查核 3 項都在〈定稿修正〉逐項處理；本頁所屬批次（index、learning-path、glossary、status）的修正後檢查通過，而且本頁有改動；沒有空清單、截斷或指向不存在的段落；工作流程紀錄裡本頁的發現都收進紀錄。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/learning-path.md 第 15、29 行 | 殘句與內部名稱：「我做了哪些檢查（都在暫存副本裡做，log 在同層的暫存副本）」「拿查核範圍清單的程式修正逐一對照三頁內容」。和首頁紀錄是同一段文字，首頁那份的處理說明聲稱已清理，這裡同樣沒清理。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 15 行改成「我做了哪些檢查：」，第 29 行改成「程式修改清單」。處理說明屬實。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/reference.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-05：v0.6.0 有界更新審閱

新增選讀E、21.1入口及23.2實際4.1／6.2前置。

本次僅重查以上變更，既有正文的歷史審閱保留；沒有把全頁或全書重新標成首次盲讀。[導讀／實際網站審閱](clear-tutorial/vision-v0.6.0/guide-visual-review.md)、[新增支線第三輪及實際前置](clear-tutorial/vision-v0.6.0/transitions-review.md)、[首讀修後複查](clear-tutorial/vision-v0.6.0/vit-recheck.md)記錄各自範圍。全版52本notebook的本機cell執行另見[實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)；不是Google Colab登入執行。來源與圖／程式指紋更新於[coverage.json](coverage.json)。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次全52節的實際方法、處置、證據與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。首頁與路線先於各組目標課實際閱讀；第三輪另讀首頁／路線／詞表並看其實際站點。詞表新增條目與真正首次正文介紹另由相應技術報告核對。

本頁對應處置為 無需要改寫的已裁定問題，見[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。[基礎技術報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-foundations.json)與[最後維護delta核回](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)保存各自時點；[第三輪](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)與[最後綁定](clear-tutorial/full-review-2026-10-06/final-bindings.json)區分實際查核範圍。

沒有真人學生驗收，四題亦不能保證不漏報；本次 DINO 漏報、引用缺陷及手機細字限制已明列。[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)保存實際命令結果；此頁最新來源綁定於[coverage.json](coverage.json)，歷史審閱保留。

## 2026-10-06：v0.6.1 有界修正複查

僅將當前教材版本與環境提示從 lessons-v0.6.0 同步到 lessons-v0.6.1；舊驗證與歷史版本的範圍保留。

本輪方法、逐批閱讀原始紀錄、非作者技術核對、另一位讀者銜接與實際 Zensical 修改段落截圖見 [v0.6.1 局部複查](clear-tutorial/release-v0.6.1-2026-10-06/README.md)。本次只重新檢查修改處、必要上下文與版本一致性；其餘正文、程式及圖的既有審閱保留原範圍，不改標為整頁或全書新的首次盲讀驗收。


## 2026-10-07 新 SKILL 改寫：作者自查

本次範圍：閱讀路線入口與 A 區；B 區以後逐字保留。先前的獨立查核結論對應當時舊稿，不能拿來宣稱本次大幅改寫已獨立通過。此次由參與撰寫的作者核對概念順序、主文是否依賴選讀、數字、圖文對應與前後銜接；沒有逐段盲讀、獨立審查者或真人讀者測試，新稿仍待使用者閱讀回饋。

- [概念覆蓋與本文路線](clear-tutorial/opening-a-2026-10-07/concepts-and-author-check.md)：38 個必要概念／轉換檢查點，逐項列先備、位置、角色與小變化。
- [首次使用原句](clear-tutorial/opening-a-2026-10-07/first-use-excerpts.md)：當場的實際原文，明記為作者自查。
- [來源核對](clear-tutorial/opening-a-2026-10-07/sources.md)、[14 個手算與變化實測](clear-tutorial/opening-a-2026-10-07/example-checks.json)。
- [網站實際頁面](clear-tutorial/opening-a-2026-10-07/browser-check.json)：8 個入口／A 頁在 1280 與 390 像素寬的檢查；圖片載入、MathJax 與整頁寬度正常，選讀預設收折。作者另視覺檢查新增圖、手機捷徑圖及主要數學段落。
- [修改範圍](clear-tutorial/opening-a-2026-10-07/scope-check.json)：B 以後課文與 notebook、所有實驗與模型原始碼未改；閱讀路線 B 起維持原文。

六節主實驗已在 CPU 重跑，僅更新其執行證據；A.1 與 A.3.3 的40步實驗也重跑並核對保存值。這些檢查支持數字與實作一致，不替代首次閱讀驗收。覆蓋指紋記錄的是本次作者實際查過的版本。

## 2026-10-08：定版 SKILL 與 A 章修後複查

本次範圍：路線開頭與 A；B 起段落保持原文。五輪完整來源覆核後選第四輪實測最佳版本，仍只抓回 9 項既知必要問題中的 3 項，三項使用者優先問題均漏抓；不宣稱穩定全抓。比較與選版見[本次決定](clear-tutorial/final-selection-2026-10-08/final-decision.md)，十份原始報告與指紋保留。

教材修正明確使用既知問題提示，與獨立抓漏分開記錄。作者按定版 SKILL 修正後，另由未參與作者的[技術／銜接審閱者](clear-tutorial/final-selection-2026-10-08/post-repair/technical.json)及[實際頁面審閱者](clear-tutorial/final-selection-2026-10-08/post-repair/visual.json)覆核指定焦點；圖中字體調整另有[小變更紀錄](clear-tutorial/final-selection-2026-10-08/post-repair/technical-delta.json)。所查焦點無未解必要問題，不是逐段盲讀、真人學生驗收或全書新審閱。

[實際 Chromium 頁面與操作](clear-tutorial/final-selection-2026-10-08/post-repair/browser.json)涵蓋 8 頁的 1280×844 與 390×844 尺寸、公式、圖片、手機內嵌目錄與橫向表格；其他尺寸／瀏覽器及完整選讀未驗。[手工機制核對](clear-tutorial/final-selection-2026-10-08/post-repair/hand-checks.json)支持新增算例一致，不當成訓練效果證據。六節 A 主實驗在 CPU 重跑通過；兩份 40 步結果沿用仍符合原程式指紋的保存紀錄。

[修正處置](clear-tutorial/final-selection-2026-10-08/repair-decisions.json)與[範圍比對](clear-tutorial/final-selection-2026-10-08/post-repair/scope-check.json)保存依據。B 以後的 92 個課文／notebook、路線 B 起段落及所有原實驗程式不變；公開程式 tag 仍是 lessons-v0.6.1。覆蓋指紋更新只綁定本次實際查核版本與上述範圍。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本附頁由獨立銜接與技術審閱者核對當前來源；頁內既有審閱歷史的接觸另記，未冒稱完全無歷史提示。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。
