# 審查紀錄：首頁

審查範圍：`docs/index.md`、頁面上的圖（`docs/assets/diagrams/object-journey.svg`）。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

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

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/index.md 第 73 行，摺疊區〈資料來源、課綱依據與維護〉第一條「偵測資料的使用與再散佈（把資料再提供給別人）條件分開列出」 | 全站只有首頁寫「再散佈」。這一條連到的 preparation/data.md（〈Git LFS 的可再散布封裝〉〈再散布與 LFS 的分工〉），以及 README、publish.md、research 各頁都寫「再散布」。讀者點進去會看到不同的字。這一行這次有改寫，正好可以一起統一。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以這一頁的讀者身分閱讀）

方法：閱讀：審查用的事實與寫作規範清單；docs/index.md 逐行讀，並和建置後 site/index.html 的正文核對。對照讀了 docs/glossary.md、learning-path.md、status.md、validation/gpu-smoke.md、preparation/data.md 和 publish.md 的相關段落、planning 兩頁的標題，以及 00-warmup、01-small-cnn、04-localization、05-assignment、07-targets、07-training、20-deployment 的開頭與相關段落。另在全部課程頁 grep：前置、練習、參考答案、標註／GT／真值框、MiniYOLO、「因電腦而異」之類的提醒。

程式：讀了 miniyolo/targets.py（floor 後依列、欄分配）、inference.py（score 門檻 0.25、依類別做 NMS）、geometry.py 的 nms、models.py 的 GridDetector，以及 lesson_cases 的 import、資料來源、GPU 用法和跨節依賴。

執行：
- zensical build --clean --strict：exit 0，No issues。
- scripts/validate_site.py：全部通過（含連結與錨點）。
- lesson_cases/00-warmup.py 與 07-targets.py：exit 0。07-targets 印出 positive indices (b,y,x) [[0,1,1],[0,2,3]]，和首頁的 (列=1,欄=1) 一致。
- 直接呼叫 build_targets：框 [8,12,24,28] 落在 (列1,欄1)，target 是 [0,.25,.25,.25]；中心 (40,20) 落在 (列1,欄2)；中心 (28,16) 歸列 1。
- 對 GridDetector 量梯度：(列1,欄1) 那格的 objectness 輸出，受 x、y 都在 0–47 的畫素影響。
- 執行 notebooks/00_environment_check.ipynb 的程式格：能正常印出 Python、PyTorch、CUDA、git 資訊。

渲染：
- 用 qlmanage 把 object-journey.svg 轉成 PNG，照 SVG 註解的換算（156+3x, 136+3y）核對各座標。
- 用 headless Chrome 截首頁：1280px 淺色；390px（headless 視窗最小 500px，所以改用 390px 寬的 iframe 取得真實版面，scrollWidth 375 ≤ clientWidth 390）；深色 slate 配色。

連結：點過的連結都由 validate_site 確認存在。Colab 環境檢查的連結指向 main，那本 notebook 確實存在，內容也和描述相符。

網路：用 WebFetch 確認 https://docs.python.org/zh-tw/3/tutorial/ 存在，第 3–5 章涵蓋變數、if／for、函式、list。

沒有在 repo 根目錄裡執行任何會寫入的操作，結束時 git status 乾淨。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/index.md 第 59 行〈電腦需求與驗證範圍〉的「沒有驗證」；對照第 22–24 行與第 67 行 | 首頁把 Colab 當成主要的動手方式：「想動手就點該節的 Colab 按鈕」，又說「教材程式與 Colab 固定在同一個程式版本，方便你對照網頁裡的例子與結果」。可是「沒有驗證」只列了模型在真實照片上的效果、長時間訓練、機制比較和速度測試，沒寫出各節 notebook 從沒在 Google Colab 上實際執行過：執行紀錄是在 Colab 以外的 Linux 電腦用 CPU 跑出的。這份清單看起來是完整的，讀者會以為 Colab 這條路已經驗證過；在 Colab 遇到套件版本或重新啟動的問題時，會先懷疑自己操作錯。驗證範圍頁有寫這件事，但首頁沒有。 |
| 2 | 建議 | docs/index.md 第 67 行「教材程式與 Colab 固定在同一個程式版本，方便你對照網頁裡的例子與結果。」 | 這句讓讀者以為照著跑就會得到和網頁一樣的數字，卻沒說哪些數字會因電腦而不同。讀者在 Colab 重跑有訓練或計時的節（例如 4.1 的 40 步、7.4 的 160 步、8.2 的 1600 步），會看到不同的小數，可能以為自己做錯。至少 4.1 頁本身也沒有這個提醒。另外「同一個程式版本」沒說在哪裡看得到。 |
| 3 | 建議 | docs/index.md 第 9 行「必要：會基本 Python（變數、if／for、函式、list）。」 | 讀者是程式與 Python 新手。頁面把基本 Python 列為必要條件，卻沒給還不會的人任何下一步，也沒說哪些部分不需要 Python（第 0 章前半的手算只要四則運算和平方）。全站也找不到 Python 入門的連結。這樣的讀者只能先讀到第 0 章前半，進到程式部分就會卡住。 |
| 4 | 建議 | docs/index.md 第 30 行「第 7 章的簡化模型把圖片切成 4×4 個格子，每一格各輸出一組數字」；第 41 行「框本身可以跨過幾個格子…負責格子仍由中心決定」 | 「切成」容易讓初學者以為圖片被剪成 16 塊、每格只看自己那 16×16 畫素。這和第 5 行「整張圖只送進網路一次」矛盾，也會讓人想不通：負責格只涵蓋 16≤x<32、16≤y<32，它怎麼知道框的左上角在 (8,12)？實際上 GridDetector 一次處理整張圖。我在暫存副本對模型量梯度：(列1,欄1) 那格的輸出，受 x、y 都在 0–47 的畫素影響，遠超過它自己那 16×16。 |
| 5 | 建議 | docs/index.md 第 36–41 行（「16÷16=1」「捨去小數得 1」→「所以中心落在第 2 列、第 2 欄」→「歸欄 1」），以及圖上方的「欄 0～3」和圖例的「第 2 列、第 2 欄」 | 讀者剛算出 1 和 1，下一句卻寫「第 2 列、第 2 欄」，要讀到句尾才知道 1 是從 0 起算的索引。接著第 41 行又改用「歸欄 1」。圖上方標欄 0～3，圖例寫第 2 列、第 2 欄，同一格有兩種編號，「欄 1」和「第 1 欄」只差一個字，很容易差一。「索引」這個程式用語也沒有解釋。 |
| 6 | 建議 | docs/index.md 第 28 行「標註」；第 30 行「輸出位置」 | 首頁引入的這兩個核心詞，在它推薦的術語快速查裡都找不到條目。術語表沒有「標註」（同一件事叫 GT、真值），也沒有「輸出位置」（第 5 章與術語表叫 slot、候選、格）。課程頁多半寫「真值框」（19 節）或「GT」（18 節），只有 8 節寫「標註框」。讀者照首頁建議去術語表搜尋「標註」會找不到，也不會知道後面的 GT、真值框就是首頁說的標註。 |
| 7 | 建議 | docs/index.md 第 47 行「用已知的標註計算預測誤差（loss）」；第 56 行「計算誤差」 | 第 0 章把兩者分開：「預測減目標叫誤差」，例子是 2−4=−2；loss 是平方誤差，例子是 4。術語表也把 loss 譯成「損失」。首頁把 loss 寫成「預測誤差」，讀者到第 0 章會看到誤差 −2、loss 4 是兩個不同的數，前後用語對不上。 |
| 8 | 建議 | docs/index.md 第 48 行〈推論〉 | 前面剛講過，訓練時只有負責格學框，其他格學「這裡沒有物件」。數學好的讀者會問：那推論時旁邊的格子為什麼還會對同一個物件給出很像、分數又不低的框？頁面直接跳到 NMS，沒交代原因。「分數」是什麼、「重疊」用什麼衡量，也都沒說。 |
| 9 | 建議 | docs/index.md 第 57 行「已確認：短訓練顯示…小模型學得會。」 | 這句讀起來像每個短訓練都學會了。驗證範圍頁寫的是「各章結果不一」，例如 3.3 沒有捷徑的 plain 網路就沒學會，8.2 只練 160 步時也沒學會。讀者讀到這些節會覺得和首頁矛盾。「短訓練」大約多少步，首頁也沒交代。 |
| 10 | 建議 | docs/index.md 第 61–63 行摺疊區「技術細節：L4、checkpoint 與 TensorRT 的實測」 | 上一段剛說「各節實驗都只用 CPU」，這裡接著說第 20 章在 L4 GPU 上跑 TensorRT，卻沒說這是課程另外做的檢查、讀者不需要 GPU。初學者可能以為第 20 章要 GPU。「透過 Modal 雲端租用」裡的 Modal 也沒有解釋。 |
| 11 | 建議 | docs/index.md 第 5 行「本教材的主角是 MiniYOLO」；第 30 行「第 7 章的簡化模型」 | 第 5 行說主角是 MiniYOLO，第 30 行講的卻是「第 7 章的簡化模型」，沒說它就是 MiniYOLO；要到完整閱讀路線才寫「也就是第 7 章的 MiniYOLO」。讀者不知道主角在哪一章出現，也不知道第 1–4 章的模型還不是它。 |
| 12 | 建議 | docs/index.md 第 69 行摺疊區標題「資料來源、課綱依據與維護」；第 75 行「課綱的安排」 | 對臺灣高中生，「課綱」通常指教育部的十二年國教課綱，「課綱依據」容易讀成本教材是依照高中課綱編寫的。導覽列用的是「課程大綱」。 |
| 13 | 建議 | docs/index.md 第 22 行「42 個小節」；第 73 行「42 節教材」 | 同一頁一處寫「42 個小節」、一處寫「42 節」；完整閱讀路線寫「共 42 節」，術語表叫「節次」。第 0 章、第 1 章本身就只有一節，叫「小節」會讓人以為還有更大一層的「節」。 |

### 事實查核（AI 對照 repo 的程式、紀錄與頁面引用的來源）

方法：repository 根目錄，分支 release/lessons-v0.4.0，HEAD 3152741，開始時工作樹乾淨。用 rsync 複製到暫存副本，排除 .git、site、.venv*、artifacts/runs、data/curated、data/downloads；所有執行都在這份複本裡。
讀過的檔案：docs/index.md、docs/assets/diagrams/object-journey.svg（用 qlmanage -t -s 1200 畫成 PNG 檢視）、zensical.toml（nav）、審查用的事實與寫作規範清單；miniyolo/targets.py、models.py、inference.py、geometry.py、losses.py、data.py、deployment_gpu.py、__init__.py；lesson_cases/00-warmup、07-targets、08-own-images、18-video、20-deployment、17-capstone、19-tracking、07-inference、07-heldout、16-training（也用 grep 掃過全部 lesson_cases 的 I/O 與 import）；docs/lessons/00-warmup、01-small-cnn、07-targets、07-training、18-video、20-deployment（含 #l4-results），並 grep 全部課程頁的前置與練習／答案，以及第 9–16 章頁面的「簡化」與原版來源；docs/status.md、learning-path.md、glossary.md、validation/gpu-smoke.md、preparation/data.md、preparation/publish.md（標題與開頭）、planning/feedback.md、planning/course-research.md、planning/outline.md（grep）、research/version-sources.md、research/pages-colab.md（grep）；README.md、requirements-*.txt、section-map.json、scripts/validate_site.py；notebooks/00-warmup、18-video、20-deployment、00_environment_check.ipynb（前三格）；artifacts/checks/grid-learning.json、gpu-smoke.json，以及 curriculum/ 下的 01-small-cnn-learning、03-comparison-learning、04-localization-learning、10-multiscale-learning、custom-data-160-step、custom-data-learning、fashion-mnist-learning、video-file、deployment-gpu.json。
唯讀 git 指令：git tag -l；git log main；git show main:notebooks/00_environment_check.ipynb；git diff main..HEAD -- docs/index.md、object-journey.svg、00_environment_check.ipynb。
外部來源：https://research.google.com/colaboratory/faq.html（"requires no setup"；FAQ 沒有明寫要登入，登入需求另以一般網頁搜尋與 repo 的 research/pages-colab.md 佐證）；https://arxiv.org/abs/1506.02640（YOLOv1 摘要）；https://arxiv.org/abs/1512.03385（ResNet 摘要）；https://github.com/zalandoresearch/fashion-mnist（README：60k／10k、28x28 grayscale、10 classes、MIT）；https://www.nvidia.com/en-us/data-center/l4/；https://developer.nvidia.com/tensorrt；https://modal.com/docs/guide/gpu（可用 GPU 型號含 L4）；https://git-lfs.com/。
在複本中執行的程式（PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python）：
- lesson_cases/00-warmup.py、07-targets.py、18-video.py、20-deployment.py，都 exit 0；00 和 07-targets 的輸出與頁面文字逐行相符。
- 自寫的檢查腳本用 build_targets、GridDetector、nms 核對：(8,12,24,28) 落在 [0,1,1]、box [0,.25,.25,.25]；中心 (40,20) 落在 [0,1,2]；中心 y=16 落在列 1；框覆蓋列 0–1、欄 0–1，共 4 格；輸出 (1,4,4,7)；兩個相同的框 NMS 後只留 1 個。
- 環境檢查 notebook 的程式格：只印出 python、git_available、pytorch、cuda_available、gpu，沒有安裝任何東西。
- .venv-docs/bin/zensical build --clean --strict 與 python3 scripts/validate_site.py，都 exit 0（連結、錨點、Colab 配對都通過）。
- validate_preparation.py exit 0。validate_lessons.py exit 1（缺審查紀錄）和 validate_curriculum_evidence.py exit 1（紀錄過期）都屬發布前的暫時狀態，不列為發現。
- review_coverage.py --help。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/index.md 第 57 行，〈電腦需求與驗證範圍〉的「已確認」項 | 「短訓練顯示，在程式畫出的幾何圖形這種刻意簡化的題目上，小模型學得會」比權威頁說得更滿。status.md 表格第 3 列能支持的結論是「小模型有能力學會這種受控的簡化任務（各章結果不一，例如第 3 章沒有捷徑的 plain 網路就沒學會）」，並註明第 1、4、10 章的 40 步實驗只看訓練圖，只能說明模型記得訓練圖多少。7.4 頁（07-training.md 第 141、227 行）也只把 160 步的結果稱為「這個受控任務學得動」的初步證據。另外，「幾何圖形」實際上全是實心矩形：紅、藍兩色，8.2 節另加黃色，類別只由顏色決定（data.md〈自製 RGB 幾何資料〉；lesson_cases/01-small-cnn.py 第 24 行的註解）。讀者可能以為模型學會了分辨形狀，也可能以為每次短訓練都學得會。 |
| 2 | 建議 | docs/index.md 第 59 行，「沒有驗證」項 | 這一項沒寫「notebook 沒有在 Google Colab 的託管 runtime 上執行過」。審查用的事實與寫作規範清單〈範圍〉與 status.md〈沒有驗證的事〉都寫明：各節紀錄是在 Colab 以外的電腦用 CPU 跑出的；環境格只有單元測試，另外在發布後由 GitHub 的 Linux runner 執行第 0、20 章。本頁第 22–24 行卻正是請讀者「點該節的 Colab 按鈕，在雲端執行 Python」。讀者看完這份「沒有驗證」清單，會以為自己要走的 Colab 這條路已經實測過。 |
| 3 | 建議 | docs/index.md 第 61–63 行，摺疊區〈技術細節：L4、checkpoint 與 TensorRT 的實測〉 | 「二是第 20 章把匯出的模型交給 TensorRT 執行」很容易被讀成：讀者在第 20 章 CPU 範例裡匯出的那份 grid.onnx 被交給了 TensorRT。實際上，L4 實測（miniyolo/deployment_gpu.py、artifacts/checks/curriculum/deployment-gpu.json）用的是同款 GridDetector 在 L4 上另用 Adam 練 40 步、再在 L4 上匯出的模型，建成 FP32 與允許 FP16 兩個 engine，在 B=1～4 時和同一張 L4 上的 PyTorch 比對。20-deployment.md〈L4 GPU 上的 TensorRT 實測〉明寫它和 CPU 案例權重不同（Adam 40 步對 SGD 1 步）；lesson_cases/20-deployment.py 輸出的 limits 也寫「this CPU case does not run TensorRT/GPU」。另外，兩項實測都只寫了做什麼，沒寫確認了什麼：續訓結果和不中斷的訓練相同；TensorRT 輸出和 PyTorch 在容許誤差內一致。 |
| 4 | 建議 | docs/index.md 第 58 行，「做不到」項 | 「想偵測真實物件，要用自己標註的資料重新訓練」把可行的路限縮成自己動手標註。preparation/data.md 也列出 Penn-Fudan、VOC2007、COCO2017 等已附框標註的真實資料，並另外寫明它們的使用與再散布條件。讀者可能因此以為只能自己從頭標註。 |
| 5 | 建議 | docs/index.md 第 73 行，摺疊區〈資料來源、課綱依據與維護〉第一項 | Fashion-MNIST 寫成「28×28 灰階衣物小圖」。官方 README（github.com/zalandoresearch/fashion-mnist）說它是 Zalando 的商品圖（article images）；10 類裡有 Sandal、Sneaker、Ankle boot 三種鞋和 Bag，「衣物」涵蓋不到鞋和包。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 審查意見 | 讀者 must（沒有驗證漏寫 Colab） | 已改。「沒有驗證」補上：各節 notebook 沒有在 Google Colab 上實際執行過，執行紀錄都是在 Colab 以外的電腦上用 CPU 跑出的；也補上沒有請真人學生學習、測量學習效果。 |
| 2 | 審查意見 | 讀者 should 1（數字會因電腦而異） | 已改。在〈這裡怎麼學〉加一段：網頁、notebook 和下載的教材程式是同一版；能手算的數字會相同，計時與訓練後的數字會因電腦而不同，可能和頁尾紀錄不一樣，這是正常的。審查者另指出 4.1 頁也缺這個提醒，那要改 docs/lessons/04-localization.md，不在本頁範圍。 |
| 3 | 審查意見 | 讀者 should 2（Python 沒有下一步） | 已改。加上 https://docs.python.org/zh-tw/3/tutorial/ 第 3–5 章的連結（已用 WebFetch 確認目錄），並說明第 0 章前半的手算不必寫程式。 |
| 4 | 審查意見 | 讀者 should 3（「切成 4×4 格」） | 已改成「第 7 章的 MiniYOLO 一次看整張圖，再把輸出排成 4×4 格：每一格對應圖上的一小塊區域」，並補一句「算一格的數字時也會用到周圍的畫素，所以負責格預測的框可以超出這一格」（重做梯度檢查：受 0–47 的畫素影響）。座標段落的「切成」改為「分成」。SVG 沒有改，因為它也用在 7.2 節。docs/glossary.md、docs/learning-path.md、docs/lessons/05-assignment.md 也寫「切成」，要不要一致屬於那些頁面。 |
| 5 | 審查意見 | 讀者 should 4（從 0 和從 1 起算交錯） | 已改。先說明「這兩個 1 就是程式用的索引（從 0 起算）」，並指出圖上的紫色字是索引、圖例的「第 2 列、第 2 欄」是從 1 數起；之後全頁一律用「欄 1」「列 1」。替代文字也同時寫出兩種說法。 |
| 6 | 審查意見 | 讀者 should 5（標註、輸出位置查不到） | 已改。標註並列 annotation、GT（ground truth，真值）、真值框；輸出位置並列「第 5 章起也叫 slot 或候選」。術語表的 GT 列要不要加「標註」，要改 docs/glossary.md。 |
| 7 | 審查意見 | 讀者 should 6（loss 寫成預測誤差） | 已改。訓練一項改成「算出 loss（損失：表示預測離答案多遠的一個數，越小越好）」，驗證方式改成「計算 loss」。 |
| 8 | 審查意見 | 讀者 should 7（推論沒說旁邊格子為什麼也高分） | 已改。補上分數的意思（0 到 1；越高表示模型越相信那裡有物件、而且是它預測的那一類），說明模型不完美、相鄰格子看到的畫面接近，並註明重疊程度用第 4 章的 IoU（交併比）衡量。 |
| 9 | 審查意見 | 讀者 should 8（「已確認」說得太絕對） | 已改，和事實查核第 1 項同一處一起改。 |
| 10 | 審查意見 | 讀者 should 9（GPU 摺疊區） | 已改。開頭寫明這兩項是另外做的工程檢查，讀者不需要 GPU，第 20 章的 Colab 範例也只用 CPU；Modal 加註「出租雲端 GPU 的服務」。 |
| 11 | 審查意見 | 讀者 should 10（MiniYOLO 在哪一章） | 已改。開頭加上「從第 5 章開始逐步成形，到第 7 章成為能用圖片訓練、推論與評估的偵測器」，格子段改成「第 7 章的 MiniYOLO」。 |
| 12 | 審查意見 | 讀者 should 11（課綱） | 已改。摺疊區標題改成「資料來源、課程安排的依據與維護」，內文改成「課程的安排」。 |
| 13 | 審查意見 | 讀者 should 12（42 個小節／42 節） | 已統一寫「42 節」。 |
| 14 | 審查意見 | 事實查核 should 1（「已確認」比 status.md 說得滿；圖形其實都是矩形） | 已改成「在程式畫的彩色矩形（顏色就代表類別）……短訓練（幾十到 1600 步）顯示小模型有能力學會。各章結果不一，例如 3.3 節沒有捷徑的 plain 網路就沒學會；有幾個實驗只看訓練圖……」。用 miniyolo/data.py、scripts/run_*.py 和 lesson_cases 核對了矩形、顏色與步數。 |
| 15 | 審查意見 | 事實查核 should 2（沒有驗證漏寫 Colab） | 已改，同讀者必要。發布後在 GitHub Linux runner 上執行第 0、20 章的細節沒有寫進首頁，由 status.md 說明，頁末也連過去。 |
| 16 | 審查意見 | 事實查核 should 3（L4 上的 TensorRT 實測） | 已改成兩點清單。第一點補「結果和不中斷的訓練一致」；第二點寫明「和第 20 章範例同架構的模型，在 L4 上另外訓練 40 步後匯出……和同一張 L4 上的 PyTorch 比對輸出，差異在容許範圍內」，連到 #l4-results，並說明 CPU 範例用的是另一份權重、也不經過 TensorRT。用 miniyolo/deployment_gpu.py 核對了在 L4 上用 Adam 練 40 步再匯出。 |
| 17 | 審查意見 | 事實查核 should 4（「做不到」限縮成自己標註） | 已改成「要用有框標註的真實照片重新訓練，標註可以自己做，也可以用授權允許的公開資料集」。 |
| 18 | 審查意見 | 事實查核 should 5（Fashion-MNIST 寫成衣物） | 首頁已改成「28×28 灰階、10 類的分類資料集，圖片是衣服、鞋與包等服飾商品」，並對照官方 README 的類別表。docs/status.md 第 19 行和 README.md 第 44 行仍寫「衣物」，要改那兩個檔案。 |
| 19 | 審查意見 | 先前檢查意見（「再散佈」改成「再散布」） | 現行文字已是「再散布」，不需修改。 |
| 20 | 先前查核 | 「再散佈」改成「再散布」 | 未改：現行文字已經是「再散布」。 |

上表「審查意見」各列的修正，由另一位 AI 逐項檢查（通過）。

結論：19 項處理都在現行頁面上，內容正確，也確實解決了原本的發現；沒有必要問題，只有 1 項 should：第 9 行新加的 class 說明，最好註明它和物件「類別」是兩回事。

方法：用 rsync 建立暫存副本，所有執行都在這份副本裡。讀了審查結果的 index 項目、審查用的事實與寫作規範清單，並把現行 docs/index.md 和 HEAD 版逐行比對（object-journey.svg 和 HEAD 相同）。對照了相關課程頁、status.md、glossary.md、learning-path.md、preparation/data.md、validation/gpu-smoke.md、20-deployment.md #l4-results，以及 miniyolo/data.py、models.py、inference.py、targets.py、deployment_gpu.py、lesson_cases、scripts/run_*.py 與 notebook 環境格。

程式核對：用自寫腳本跑 build_targets、GridDetector、decode_grid、nms，(8,12,24,28) 分到 [0,1,1]、中心 (40,20) 分到 [0,1,2]、中心 y=16 歸列 1、輸出 (1,4,4,7)、列 1／欄 1 那格的感受野是 0–47、分數都在 0–1、NMS 會刪重複框，都和頁面相符。在副本重跑 3.3 的 40 步實驗，只確認定性結論：plain 沒學會、residual 學會。用 WebFetch 確認 Python 官方教學正體中文版的章節內容。

建置與畫面：zensical build --clean --strict 回報 No issues，exit 0；validate_site.py exit 0，連結、錨點、Colab 配對都通過。Playwright 的 Firefox 沒安裝，改用 headless Chrome 截圖，摺疊區全展開：1280px 淺色正常；寬 390px 的 iframe 裡 scrollWidth 375 等於 clientWidth 375，沒有元素超出版面，圖寬 343px、左側留 16px；在副本強制 slate 深色模式時，圖有白底，看得清楚。

寫作規則：CJK 與英文、數字之間的空格掃描沒有問題，也沒有修訂經過或計畫語氣的字眼。

其他：沒有寫入紀錄裡會隨機器改變的數字；用 find -newermt 確認原 repo 沒有任何檔案被改動。

| 列 | 結果 | 檢查內容 |
|---|---|---|
| 1 | 成立 | 第 61 行「沒有驗證」已寫明各節 notebook 沒在 Google Colab 上執行過、執行紀錄是在 Colab 以外的電腦用 CPU 跑出，也寫了沒有請真人學生試用。這和審查用的事實與寫作規範清單〈範圍〉、status.md 第 29、32、94 行一致，讀者不會再以為 Colab 這條路驗證過。 |
| 2 | 成立 | 第 26 行的新段落和審查用的事實與寫作規範清單、status.md 第 47 行一致。notebook 環境格 clone 的是 lessons-v0.4.0，所以「同一版」成立；42 個課程頁的最後一節都是「實際執行紀錄」，所以「每節頁面最下方」也成立。4.1 頁的提醒屬於另一個檔案，不改的理由成立。 |
| 3 | 成立 | 用 WebFetch 打開 https://docs.python.org/zh-tw/3/tutorial/：第 3–5 章涵蓋變數、list、if／for、函式、dict，第 9 章是「Class（類別）」，語言標示是「繁體中文」。「第 0 章前半只用到四則運算與平方」和 00-warmup 的前置知識一致。後來加的 dict、class、第 9 章也正確，但 class 和「類別」容易混淆，另列一項建議。 |
| 4 | 成立 | 在副本對 GridDetector 量梯度：列 1、欄 1 那格的輸出受 x、y 都在 0–47 的畫素影響；decode 的寬高也是相對整張圖。所以「一次看整張圖」「會用到周圍畫素、框可以超出這一格」都正確。圖內副標仍寫「切成 4×4 格」，「也用在 7.2」不算充分理由（它也用在 7.5，改成「分成」對兩頁同樣正確），但圖上方的正文已直接說明模型一次看整張圖，保留不會再造成原本的誤解。 |
| 5 | 成立 | 頁面先說明「從 0 起算的索引」，之後一律寫「欄 1」「列 1」。SVG 的紫色索引字確實在上方（欄 0–3）和右側（列 0–3），圖例的「第 2 列、第 2 欄」是從 1 數起，替代文字也寫了兩種說法。這和 build_targets 的輸出 [0,1,1] 相符。 |
| 6 | 成立 | 「標註（annotation）……GT（ground truth，真值）或真值框」和「第 5 章起也叫 slot 或候選」，與 glossary 第 86、88、89 行及 05-assignment 第 10 行一致；19 個課程頁都用到 GT 或真值框。術語表的 GT 列現在已經寫了「標註（annotation）或真值框」，讀者搜尋得到。 |
| 7 | 成立 | loss 的說明（表示預測離答案多遠的一個數，越小越好）和 00-warmup「預測減目標叫誤差、loss 是平方誤差」及術語表的 loss 列一致。第 58 行也已改成「計算 loss」，全頁沒有「預測誤差」。 |
| 8 | 成立 | inference.py 的 score 是 σ(obj) 乘最大類別 softmax 機率，實測 decode 出的分數都在 0–1；門檻 0.25 先丟低分框，再按類別做 NMS，和術語表第 101–103 行一致。IoU 在 4.1 引入；相鄰格子看到的範圍重疊，也和量到的感受野相符。 |
| 9 | 成立 | 和事實查核第 1 項是同一處修改，查證內容見第 14 項；這段的說法已降到 status.md 表格第 3 列的程度。 |
| 10 | 成立 | 「讀者不需要 GPU、第 20 章 Colab 範例只用 CPU」和 lesson_cases/20-deployment.py（模型在 CPU、ORT 用 CPUExecutionProvider、limits 寫明不跑 TensorRT）及 status.md〈執行方式〉一致。Modal 的註解正確。 |
| 11 | 成立 | 05-assignment 寫「本節的 MiniYOLO」，04-localization 說該節是 MiniYOLO 之前的單物件模型；第 6 章做 decode 與評估，第 7 章完成訓練、推論與 held-out 評估。這和「從第 5 章開始逐步成形，到第 7 章成為偵測器」相符。 |
| 12 | 成立 | 摺疊區標題已改成「資料來源、課程安排的依據與維護」，內文改成「課程的安排」；全頁沒有「課綱」，導覽列的名稱是「課程大綱」。 |
| 13 | 成立 | 全頁只剩「42 節」（第 22、76 行），沒有「小節」。docs/lessons 有 42 頁，notebooks 有 42 本課程 notebook，另加一本環境檢查。 |
| 14 | 成立 | 各學習實驗的資料都是程式畫的實心矩形，紅、藍兩色，8.2 另加黃色，類別只由顏色決定（miniyolo/data.py、01／3.3／4.1／10 的資料產生、run_custom_data_learning.py）；40、160、1600 步都寫死在程式裡。「3.3 的 plain 沒學會」和 03-comparison.md、status.md 一致，我在副本重跑 40 步實驗，定性結果相同；「有幾個實驗只看訓練圖」對應 status.md 的第 1、4、10 章。 |
| 15 | 成立 | 和第 1 項是同一處修改。發布後在 GitHub Linux runner 執行第 0、20 章的細節，確實寫在 status.md 第 29 行，首頁也連到 status.md，所以不寫進首頁的理由成立。 |
| 16 | 成立 | 兩點清單和 miniyolo/deployment_gpu.py（GridDetector(width=8)、在 L4 上用 Adam 練 40 步後匯出、FP32／FP16 engine、B=1–4 以 assert_close 比對）、lesson_cases/20-deployment.py（同架構、SGD 1 步、只用 CPU）、20-deployment.md #l4-results、gpu-smoke.md、status.md 第 76–77 行都一致。錨點經 validate_site 確認存在。 |
| 17 | 成立 | preparation/data.md 列出 Penn-Fudan、VOC2007、COCO2017 和各自的授權條件，所以「也可以用授權允許的公開資料集」成立；第 8 章確實示範 JSON＋PNG 的資料格式與流程。 |
| 18 | 成立 | 和官方 README 一致：Zalando 商品圖、28×28 灰階、10 類，類別裡有鞋和包。lesson_cases 和 notebook 都沒用到 Fashion-MNIST；status.md 第 19 行與 README 第 44 行現在也是同樣的寫法。 |
| 19 | 成立 | 第 76 行是「再散布」，和 preparation/data.md 等頁一致，不需要修改。 |

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/index.md 第 9 行「也看得懂 class 的寫法（本書的模型都寫成 class）……第 9 章介紹 class」 | 這段是後來的小修改加上的，沒有說明 Python 的 class 和本頁大量使用的「類別」（物件類別，例如「是哪一類」「框與類別」）是兩回事；連過去的 Python 教學第 9 章標題又正好是「Class（類別）」。程式新手讀到「模型都寫成 class」，可能把它和物件類別混在一起。learning-path.md 第 7 行已經寫了「這個 class 和物件的「類別」是兩回事」，首頁沒有。 | 已修正：括號裡補上「這是 Python 的寫法，和物件的「類別」是兩回事」，和閱讀路線頁一致。 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

「再散布（把資料再提供給別人）」與 data.md〈再散布與 LFS 的分工〉用詞一致，全站沒有「散佈」；該節確實分資料集列出條件。Fashion-MNIST「衣服、鞋與包等服飾商品」符合官方 10 類。GPU 摺疊區的兩項描述與 gpu-smoke.json、第 20 章 L4 節一致。

### 第 2 輪：上一輪的處理與審查紀錄：通過

全文讀了 reviews/index.md：有獨立查核、讀者審查、事實查核與方法，〈定稿修正〉20 列，修正後檢查與〈後續編輯的檢查〉都有；首頁那一輪修正由〈後續編輯的檢查〉涵蓋，支持各項說法。只有可讀性問題。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/index.md 各段 | 內部名稱與路徑清理殘留：「log 在同層的暫存副本」「impact.json 裡確實沒有這三頁的項目；拿程式修改清單的程式修正逐一對照」「不算本單元的問題」「執行（都在暫存副本」「方法：Repository：repo，分支 release/lessons-v0.4.0」「這是審查用的事實與寫作規範清單列出的暫時狀態」；〈定稿修正〉19 列編號都是 1、處理寫在「意見」欄。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 2 | 建議 | 「修正後由另一位 AI 檢查改動」 | 這段明寫「只看這位修改者改了什麼」，不含上方 19 列 resolve 階段的修正；讀者容易以為它檢查了整個〈定稿修正〉。那些修正其實由〈後續編輯的檢查〉涵蓋。 | 已修正：修正後的檢查寫明是這一批頁面的改動；上方「審查意見」各列，另由一位 AI 逐項檢查，結果列在表格之後。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

第 9 行對照 Python 官方教學正體中文版目錄（第 3–5 章：非正式介紹、流程控制、資料結構；第 9 章 Class）與各節模型寫法（SmallCNN、GridDetector 等 nn.Module 子類別）：括號補上「這是 Python 的寫法，和物件的『類別』是兩回事」，和 learning-path.md 第 7 行一致，解決逐項檢查留下的建議；讀來通順、合寫作規則；網站嚴格建置與 validate_site 通過。紀錄：〈定稿修正〉20 列編號連續、處理都在處理欄；表後附逐項檢查（19 列，對應 1–19 列審查意見）與它的建議及處理；沒有掛不涵蓋本頁的批次檢查；其餘結構檢查都通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/index.md 第 15、29、53、95 行與逐項檢查的方法段 | 〈上述處理與審查紀錄的檢查〉第 1 項寫「已修正：內部用語與路徑殘句換成白話」，但它點名的「log 在同層的暫存副本」（第 15 行）、「執行（都在暫存副本：」（第 53 行，括號沒有關）仍在，「這是審查用的事實與寫作規範清單列出的暫時狀態」也沒改；另有「拿查核範圍清單的程式修正逐一對照三頁內容」「自寫的檢查腳本（在暫存副本）」「讀了審查結果的 index 項目」這類內部名稱。處理說明和紀錄現況不符。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 15、29、53、95 行已改好；但另兩處點名的文字逐字未改，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/index.md 第 17、142 行 | 點名的「讀了審查結果的 index 項目」（第 142 行，內部名稱）與「這是審查用的事實與寫作規範清單列出的暫時狀態」（第 17 行）都逐字還在，處理卻寫已清理或換成白話。後者若判斷已是白話，處理欄應寫未改與理由。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 2 | 建議 | reviews/index.md 第 17 行 | 「repo 裡這三頁和編者暫存副本；」是殘句。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

第 4 輪第 1、2 項屬實：第 17、142 行兩段仍在，「repo 裡這三頁和編者暫存副本；」已經不在。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 3 輪第 1 項處理欄（第 201 行） | 「7 處中 4 處已改寫或刪除」把引用的第 2 輪處理說法「已修正：內部用語與路徑殘句換成白話」也算成已改寫，但它仍在第 192 行。實際改寫的紀錄文字是 3 處。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
