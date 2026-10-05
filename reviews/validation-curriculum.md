# 審查紀錄：全套實驗與審查

審查範圍：`docs/validation/curriculum.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 讀者審查與技術查核

### 讀者審查（AI 以這一頁的讀者身分閱讀）

方法：所有執行與建置都在暫存副本進行，沒有寫入 repo 根目錄。
- 主要複本
- 驗證器實驗另開一份複本

【閱讀】
- 頁面與對照文件：審查用的事實與寫作規範清單、docs/validation/curriculum.md 全文、docs/glossary.md、docs/status.md、docs/validation/gpu-smoke.md、README.md、docs/preparation/publish.md（第 7–11 步）、docs/research/pages-colab.md 第 70–71 行、docs/preparation/data.md 的 Fashion-MNIST 節。
- 補充紀錄被引用的課程頁段落：01、03.3、04.1、07.4、07.6、08.2、10、18、19、20。
- 程式：scripts/ 下的 verify_curriculum.py、record_evidence.py、evidence_records.py、validate_curriculum_evidence.py、validate_lessons.py、validate_site.py、review_coverage.py、verify_release.py、check_lesson_runtime.py、run_learning_extensions.py、run_multiscale_learning.py、verify_video_file.py、run_fashion_cnn.py；build_lesson_notebooks.py 與 run_custom_data_learning.py 讀了相關片段。另讀 miniyolo/provenance.py 全文，miniyolo/train.py、deployment_gpu.py 以 grep 查看。
- 工作流程：.github/workflows/pages.yml、verify-release.yml、deployment-gpu.yml、gpu-smoke.yml。
- 紀錄與審查：artifacts/checks 與 curriculum/ 的紀錄欄位、GPU 紀錄的 dependencies_sha256，以及 reviews/ 現有檔案。

【建置與看頁面】
- `zensical build --clean --strict` 結果 exit 0，接著讀建置出的 article 文字與連結。
- `scripts/validate_site.py` 結果 exit 0。
- 用 python http.server 提供暫存副本網站。Playwright 的 Firefox 沒有安裝，改用 Playwright 快取裡的 headless Chromium，以 1280 與 390 寬截圖後分段查看。本頁沒有圖。

【比對與試跑】
- 程式逐列比對 42 節與 17 頁表格的節次、標題、連結和導覽標籤、紀錄與審查的命名規則：沒有不符。
- 檢查 114 個 GitHub 連結的路徑：工作樹只缺 17 頁非課程頁審查與 bootstrap 紀錄，依事實清單屬於發布時才寫入的檔案，所以不報。
- 用 curl 確認 repo 公開、幾個連結回 200。
- 跑 run_learning_extensions.py 的 01、03、04：40 步，8／8／2 張，Adam／SGD／Adam，和頁面相符。
- 跑 00-warmup、12-dfl、16-dfl-free、19-tracking：輸出和舊紀錄不同，原因是程式已改、紀錄正在重產，不報。
- 統計 42 節程式的 optimizer.step 與迴圈次數；比對 42 本 notebook 的環境格（41 本相同，第 20 章不同）；grep 全站有沒有定義「機制章」「前導頁」「Pages 工作流程」等詞。
- 驗證器實驗：用測試用腳本 fake_records.py（只在暫存副本使用），讓 42 份紀錄與補充紀錄都「對得上目前程式」；再把 00-warmup 頁尾區塊的輸出改成錯的值，validate_curriculum_evidence.py 仍然 exit 0。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/validation/curriculum.md 第 9 行，〈執行紀錄怎麼產生、怎麼檢查〉第二段：「頁尾的執行紀錄區塊也要和紀錄一致；有一項不符，網站就不會發布。這項檢查確認頁面、notebook 與紀錄出自同一份程式」 | 這句說的檢查實際上沒有做。scripts/validate_curriculum_evidence.py 對 notebook 會逐字比對最後一格的 stdout，對頁面卻只檢查兩件事：開始與結束標記各出現一次，以及紀錄的日期（前 10 字元）、PyTorch 版本、CPU 型號三個字串出現在整頁文字的某處（不限區塊內）。區塊裡顯示的輸出不比對，由紀錄重畫的圖也不檢查。 實測：我在另建的暫存副本把 42 份紀錄做成「對得上目前程式」的狀態，再把 docs/lessons/00-warmup.md 區塊裡的 `prediction=2.00, loss=4.00, gradient=-8.00` 改成 `prediction=7.00, loss=99.00, gradient=+123.00`。validate_curriculum_evidence.py 仍然 exit 0，印出「42 lesson records and 11 supplementary records match the current code」。 本頁第 90 行又說審查不涵蓋這個區塊，validate_site.py 也不看它，所以頁尾顯示的輸出其實沒有任何檢查守著。讀者會以為那段輸出已經被機器逐字核對過。另外〈驗證範圍〉第 45 行與〈Pages 與 Colab〉第 71 行對這項檢查的描述是正確的，三頁的說法不一致。 |
| 2 | 必要 | 第 117 行，〈發布後的公開驗證〉curriculum-release-bootstrap.json 那一項 | 沒寫出四種 PyTorch 情況只用第 0 章（00-warmup）的 notebook，第 20 章也只在「沒有 PyTorch」這一種情況執行一次；其餘 40 本 notebook 沒有以 notebook 的方式在 runner 上執行。「照 notebook 的順序執行環境格與最後一格」讀起來像每本 notebook 都在四種情況下跑過。scripts/verify_release.py 的 check_bootstrap、〈驗證範圍〉第 29 行、〈發布與帳號設定〉第 251 行都寫明是第 0 章。 另外還漏了三件事： - 四種情況的最後一格輸出都要和 notebook 存的輸出相同（第 20 章不比，因為輸出含計時）。 - README 的指令會略過不會自己結束的 `zensical serve`。 - README 的指令包含 pytest 與 `check_lesson_runtime.py`，也就是在 runner 上把 42 節程式各跑一次並和紀錄比對；紀錄的 lesson_runtime 欄記下哪些節輸出相同。 句子也不好讀：開頭是「每種情況都用…」，情況卻到冒號之後才列出。 |
| 3 | 必要 | 第 64 行，〈補充實驗清單〉第一句「各節程式只跑幾步、幾秒鐘。下面這些實驗用同一套程式多跑一些步數」（第 3 行「用同一套程式多跑一些步數的補充實驗」同義） | 和程式不符。各節程式本身的更新次數如下： - 第 17 章：把 GridDetector 從頭訓練 160 步、做兩次（另有 10 步暖機），再用 held-out 圖評估。 - 第 18 章：fit_detector 訓練 160 步。 - 12.4 更新 400 次、16.1 更新 300 次、13.2 更新 100 次、12.1 更新 80 次、14 更新 30 次、第 2 章 20 次。 反過來，第 1、3、4、10 章的補充實驗只有 40 步，比上面這些節的程式還少；第 18、19 章的影片補充實驗也不是多跑步數。讀者會以為各節程式都只是幾步的煙霧測試、較長的訓練都在補充實驗裡，因而低估第 17 章結業任務本身做過的訓練與評估。這也和〈驗證範圍〉不一致：那頁把「CPU 少數幾步更新」舉例為第 0 章 1 步、第 7 章 3 步。 |
| 4 | 建議 | 第 15 行與〈逐節執行清單〉的「實驗範圍」欄 | 只解釋了「固定數值／幾何／張量機制」，沒有說「短步更新／管線」是什麼。兩類的分界也看不出來： - 24 節「固定數值」裡，有 14 節實際呼叫 optimizer.step()（例如 3.1、5、9.1、11.1、14、15.1）。 - 18 節「短步更新」全部都會更新。 - 第 17、18 章是 160 步的訓練，卻和第 0 章的 1 步同屬「短步更新」。 〈驗證範圍〉的表格用的是另一套分類（人工已知答案、CPU 少數幾步更新、合成圖形短訓練），讀者在兩頁之間對應不起來。 |
| 5 | 建議 | 〈GPU 檢查〉第 81–82 行 | 兩項都寫「一致」，但強度不同： - 第 20 章是容許誤差內相符。deployment_gpu.py 對 FP16 engine 的 raw 輸出用 atol 2e-2、rtol 2e-3，框容許 ±0.1 畫素，分數容許 ±0.002；只有類別與候選順序要求完全相同。 - 存檔續訓是最大差異為 0。 讀過術語表「FP16 只有約 3 位有效數字」的讀者，會疑惑 FP16 engine 怎麼能和 PyTorch「一致」，或誤以為輸出逐位相同；〈驗證範圍〉寫的是「在容許誤差內一致」。 同一節還有三個小問題： - 「所以不說全部的層都用 FP16」描述的是作者選擇怎麼寫，不是事實。 - 「紀錄同樣綁定所執行的程式」其實只涵蓋在 Modal 上執行的程式；Actions 端的 scripts/modal_*.py 不在其中（〈GPU／checkpoint 實測〉有寫明）。 - 第 20 章的連結落在頁首，L4 結果在頁末。 |
| 6 | 建議 | 第 121–123 行〈沒有驗證的事〉 | 兩個修飾詞會讓人以為做過一部分：「沒有在真實照片資料集上長時間訓練」暗示做過短時間的真實照片訓練；「沒有…逐節執行」暗示有部分 notebook 在 Colab 上跑過。實際上，偵測實驗只用合成圖或小型自製資料，notebook 也一本都沒在 Colab 的託管 runtime 上執行過。 這份清單也比〈驗證範圍〉的〈沒有驗證的事〉短，少了正式的 GPU 速度測試、用相同資料與預算比較各機制、更大模型或強制 FP16 的 TensorRT。讀者可能以為這就是全部。 |
| 7 | 建議 | 全頁的術語：第 9、90 行「Pages 工作流程」；第 79 行「GitHub Actions 工作流程」「Modal」；第 82 行「container」；第 114、116 行「runner」「嚴格建置」；第 117 行「clone」；第 88、116 行「commit」；第 7 行起的「SHA-256」；第 71 行「訓練 CLI」；第 11 行「機制章」「預算」 | 這位讀者熟 Python 與 PyTorch，不熟 GitHub Actions 與 Modal，這些詞本頁卻都沒有解釋： - 「Pages 工作流程」在 repo 裡的實際名稱是 Publish Learn to YOLO（.github/workflows/pages.yml），照這個詞找不到。 - 「runner」到該節最後一段才說明。 - 「嚴格建置」沒說是 `zensical build --strict`。 - 7.4 節從未用「CLI」這個詞，它寫的是 `python -m miniyolo.train`。 - 「機制章」全站只出現在本頁，看不出指哪幾章。 - 「預算」沒說是訓練步數這類花費。 |
| 8 | 建議 | 第 7 行「都在同一台電腦上用 CPU 從頭跑完」 | 紀錄只在綁定的檔案改了之後才重跑，沒過期的沿用原本的日期與電腦（見本頁第 9 行、〈驗證範圍〉第 40 行、README 發布步驟第 2 步）；README 也只要求在「同一種環境」重產。「同一台電腦」比實際能保證的更強。讀者看到各節頁尾的日期不同時，也不知道原因。 |
| 9 | 建議 | 第 64 行「哪份紀錄綁定哪些檔案，列在 scripts/evidence_records.py」，以及第 68、73、75 行 | - scripts/evidence_records.py 只列每份紀錄的「入口檔」，例如 grid-learning.json 只寫 miniyolo/train.py。完整的綁定清單是程式追查 import 之後，寫在各紀錄的 dependencies_sha256 裡。讀者照這句打開 evidence_records.py，會以為一份紀錄只綁一個檔。 - 第 4 章那列寫了「只在這 2 張訓練圖上評分」；第 1、10 章同樣只在訓練圖上評分（第 10 章的 AP 也是在訓練圖上算的，〈驗證範圍〉寫明第 1、4、10 章只看訓練圖），表格卻沒寫，讀者可能以為它們像第 3 章一樣另有評估圖。 - Fashion-MNIST 那列的「資料規劃」連到頁首，不是 Fashion-MNIST 那一節。 |
| 10 | 建議 | 第 114–116 行〈發布後的公開驗證〉 | 沒告訴讀者怎麼判斷驗證有沒有通過。依〈發布與帳號設定〉第 11 步，只有兩份紀錄都是 `passed=True` 才會 commit 到 main，所以連結到的就是通過的結果；但本頁沒說，讀者打開 JSON 也不知道該看哪一欄。 「逐頁比對公開網站上每一頁的正文與每一張圖」沒說是和什麼比。實際上是和 runner 上剛建置的結果比：正文指 `<article>`，圖要逐 byte 相同。 |
| 11 | 建議 | 第 88 行「前導頁是初學讀者，維護頁是維護者」，以及第 92–110 行的 17 頁表格 | 「前導頁」「維護頁」全站只出現在本頁，沒有定義。表格也沒標出每一頁用哪種讀者審查，讀者無從得知例如〈資料規劃〉或〈Pages 與 Colab〉是以初學讀者還是維護者的身分審查的。 |
| 12 | 建議 | 第 9 行「頁面講得對不對，由下方的審查負責」，以及第 90 行 | 審查綁定的是頁面文字、頁面上的 SVG，以及課程頁程式 import 的模組；產生補充紀錄的腳本與紀錄本身不在其中，例如 18、19 章的審查不含 scripts/verify_video_file.py。 所以會出現這種情形：這類腳本改了、紀錄重產後數字跟著變（例如 18 章用 video-file.json 算出的 IoU 0.50、0.47），只要頁面文字與圖沒變，validate_lessons.py 與 validate_curriculum_evidence.py 都會通過。README 發布步驟第 5 步說，這部分要人工搜尋核對。讀者看到兩道自動檢查加上審查，會以為正文引用的紀錄數字也有自動保證。 |
| 13 | 建議 | 第 88 行「每份審查紀錄都列出全部的發現、每個發現怎麼處理，以及修改後的複查」 | 定稿事實清單只說每份審查紀錄記下每個發現與它的處理，沒有提到「修改後的複查」；寫作規則也要求不要宣稱審查涵蓋審查之後才改的文字。只要有一份審查紀錄沒有獨立的複查段落，讀者照這句去找就會找不到。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 讀者審查 | 誤稱頁尾執行紀錄區塊有和紀錄比對 | 已修正：已查證：validate_curriculum_evidence.py 對頁面只檢查開始／結束標記各一個，以及日期、PyTorch、CPU 字串出現在整頁某處。已改寫成實際檢查的範圍，並寫明區塊由 verify_curriculum.py 從紀錄寫入，但檢查不逐字比對它，也不檢查圖；「出自同一份程式」也改成「紀錄對得上目前的程式、notebook 存的就是紀錄的輸出」。腳本 docstring 與 report scope 的同樣誇大說法在 scripts/ 裡，不在我能改的範圍；區塊內容目前沒有任何檢查守著。 |
| 2 | 讀者審查 | bootstrap 驗證沒寫只用第 0 章，第 20 章只跑一次 | 已修正：已對照 verify_release.py 的 check_bootstrap：四種情況都只用 00-warmup，20-deployment 只在沒有 PyTorch 的環境跑一次，而且不比對輸出（輸出含 median_ms 計時）。已改寫該項：41 本環境格逐字相同所以用第 0 章測、最後一格輸出要和存的輸出相同、略過 zensical serve、README 指令包含 pytest 與 check_lesson_runtime.py。 |
| 3 | 讀者審查 | 「各節程式只跑幾步、幾秒鐘」與程式不符 | 已修正：已查證：17、18 章各訓練 160 步，12-dfl 跑 400 步、16-dfl-free 跑 300 步等。刪掉這句，改成「把第 1、3、4、7、8、10 章的實驗延長，或接上真實的檔案與資料」；頁首摘要也一併改。 |
| 4 | 讀者審查 | 「短步更新／管線」沒有定義 | 已修正：在欄位說明補一句定義：重點是讓參數真的更新，或把整條管線跑通，步數從第 0 章的 1 步到第 17、18 章的 160 步不等。逐列改成更新次數或改用另一套分類屬於較大的改動，沒有做。 |
| 5 | 讀者審查 | GPU 兩項都寫「一致」，強度不同 | 已修正：已對照 deployment_gpu.py：FP16 的容許誤差 atol 2e-2，labels 用 torch.equal。第 20 章改成「在容許誤差內和 PyTorch 相符，類別與候選順序完全相同」，續訓改成「最大差異為 0」；「所以不說…」改成「所以不能確定每一層都用 FP16」；連結改指 #l4-results。綁定那句改成照實描述：只綁 miniyolo 的實驗程式與它 import 的模組，scripts/modal_*.py 不在其中。 |
| 6 | 讀者審查 | 〈沒有驗證的事〉的修飾詞容易誤讀，清單也不完整 | 已修正：「逐節執行」改成「沒有在 Colab 託管 runtime 上執行過」，補上 Colab 的 GPU 沒有測，句首加「偵測實驗都只用合成圖或小型自製資料」，並連到〈驗證範圍〉的完整清單。「長時間」保留：它和審查用的事實與寫作規範清單的說法一致，而且 Fashion-MNIST 的 40 步就是一次真實影像的短訓練。 |
| 7 | 讀者審查 | GitHub Actions、Modal 等術語沒有解釋 | 已修正：在第一次出現的地方補上說明：SHA-256、Pages 工作流程（Publish Learn to YOLO）、GitHub Actions、Modal、container、嚴格建置（zensical build --strict）。「訓練 CLI」改成 `python -m miniyolo.train`，「機制章」改成第 9–16 章，「預算」改成「訓練預算（例如步數）」。runner 第一次出現時已有說明；commit、clone 是 repo 維護的基本用語，沒有加註。 |
| 8 | 讀者審查 | 「同一台電腦」的說法太強 | 已修正：刪掉「在同一台電腦上」，並補一句：沒有過期的紀錄沿用原本的日期與電腦，所以各節頁尾的日期不一定相同。 |
| 9 | 讀者審查 | evidence_records.py 列的只是入口檔；第 1、10 章評分範圍沒寫；Fashion-MNIST 連結錨點 | 已修正：改成：完整的檔案清單記在 dependencies_sha256，evidence_records.py 只列入口檔。第 1、10 章兩列補上「只在訓練圖上評分」，已對照 run_learning_extensions.py 與 run_multiscale_learning.py 確認。連結改成 data.md#fashion-mnist，validate_site 確認錨點存在。 |
| 10 | 讀者審查 | 公開驗證沒說怎麼判斷通過，也沒說和什麼比 | 已修正：比對那句改成「在 runner 上嚴格建置後和公開網站比對：每頁正文與 assets/diagrams 的每張圖都要完全相同」。另補一句：兩份紀錄最外層的 passed 都是 true 才 commit 到 main，這和 publish.md 第 11 步一致。 |
| 11 | 讀者審查 | 「前導頁／維護頁」沒有定義 | 已修正：改用〈驗證範圍〉的說法：寫給學習者的頁面模擬初學讀者，寫給維護者的頁面模擬維護者。逐頁加一欄審查身分需要另外取得每份審查的設定，不屬於小改動，沒有做。 |
| 12 | 讀者審查 | 沒寫正文引用的紀錄數字沒有自動比對 | 已修正：在 coverage 那段後補一句：正文引用的紀錄數字不在這項核對裡，紀錄重產後由維護者搜尋檔名與舊數字逐一核對，再重審那一頁，並連到〈發布與帳號設定〉。 |
| 13 | 讀者審查 | 審查紀錄不一定有「修改後的複查」 | 已修正：查了 reviews/：定稿紀錄只有讀者審查、技術或事實查核、先前檢查意見三段，沒有一份有複查段落。已刪掉「以及修改後的複查」。 |

修正後由另一位 AI 檢查這一批頁面（`docs/validation/curriculum.md`、`docs/validation/gpu-smoke.md`、`docs/preparation/data.md`、`docs/preparation/publish.md`、`docs/preparation/architecture.md`、`docs/planning/outline.md`、`docs/planning/course-research.md`、`docs/planning/feedback.md`、`docs/research/foundation-data.md`、`docs/research/detection-data.md`、`docs/research/lfs.md`、`docs/research/pages-colab.md`、`docs/research/version-sources.md`、`README.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

讀〈審查〉一節，對照 review_coverage.py：digest 排除執行紀錄區塊，Colab tag 會正規化，課程頁另含 lesson case 與它 import 的模組；也對照 validate_lessons.py 第 89–91 行與導覽頁數。17 列「其他頁」表格與 nav 一致，審查檔命名合乎 review_file()（例如 reviews/index.md、reviews/validation-gpu-smoke.md）。正文引用紀錄數字要另外核對的說明與 publish.md 第 7 步一致。〈發布後的公開驗證〉與 verify_release.py 的 site／bootstrap 內容相符。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 〈審查〉第一段：「頁面引用原始論文、官方程式或函式庫文件的說法，另由 AI 對照頁尾〈參考來源〉連到的固定版本逐句核對。」 | 和 docs/status.md 同一個錯：42 節中只有 13 節有「參考來源：」頁尾行；00、02、6.1、7.4 的 PyTorch／torchvision 文件連到 stable，不是固定版本；9.1、10 的 Darknet 說法沒有連結；20 章的來源行不在頁尾。這一頁是審查總覽，讀者會照這句以為每節頁尾都列著審查用的固定版本。 | 已修正：同樣改成打開原始來源逐句核對，查閱的來源記在該頁的審查紀錄。 |

### 第 1 輪：〈全套實驗與審查〉與相同說法的事實查核

逐句核對全頁。

- 執行紀錄：對照 verify_curriculum.py、evidence_records.py、validate_curriculum_evidence.py、record_evidence.py、miniyolo/provenance.py 與 pages.yml，核對紀錄內容（stdout、日期、machine 欄位、case_sha256 與 dependencies_sha256）、綁定方式、過期判定，以及 Pages 建置前會擋下哪些不符。
- 補充實驗：對照 run_learning_extensions.py、run_multiscale_learning.py、run_custom_data_learning.py、miniyolo/train.py、verify_video_file.py、run_fashion_cnn.py、classification.py 與各節 lesson_cases，核對每一列的設定（張數、optimizer、步數、評分範圍、checkpoint 重讀、FFV1、tracker）。
- GPU：用 evidence_records.is_current 確認兩份 GPU 紀錄綁定目前程式；再對照 deployment_gpu.py、gpu_smoke.py、modal_*.py、兩個 GPU workflow 與兩份 JSON。deployment-gpu.json 是 batch 1–4、FP32 與 FP16 engine、類別與順序一致；gpu-smoke.json 的 model、optimizer 與 history 最大差異都是 0.0。
- 審查：對照 review_coverage.py 與 validate_lessons.py 核對審查涵蓋與強制方式；也讀了 reviews/00-warmup.md 與 reviews/index.md。
- 發布後驗證：對照 verify_release.py、verify-release.yml、README 的 bash 區塊、build_lesson_notebooks.py（41 本環境格逐字相同），並用 zensical build --help 確認 --strict 的意思是遇到警告就中止。
- 表格：用程式比對 42 列與 zensical.toml 的導覽標題、section-map 的順序、紀錄與審查的檔名，全部相符。統計各節 optimizer.step 次數，核對「實驗範圍」欄的說明成立（第 0 章 1 步、第 17、18 章 160 步、12.4 有 400 步、16.1 有 300 步）。
- 連結：在副本建站並執行 validate_site.py，通過；#fashion-mnist 與 #l4-results 錨點存在。
- 寫作規則：掃描修訂敘事用語與中英文空格，沒有問題。
- 〈沒有驗證的事〉與 status.md 一致。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 88 行〈審查〉「查到的問題修正後，都由另一位 AI 檢查修正。」 | repo 裡只有兩份 v0.4.0 格式的審查紀錄，兩份都寫成修正由處理者自己核對： - reviews/00-warmup.md 第 3 行：「處理後的頁面沒有交回兩位審查者重讀，而是重新執行本節程式與頁面新增的每個小實驗……」 - reviews/index.md 第 3 行：「由處理者對照原始碼、程式實際輸出與建置結果逐項核對，沒有再請兩位審查者重讀」 兩份都沒有「另一位 AI 檢查修正」的記載；「先前檢查意見」一節也沒寫是誰提的、查的是不是修正。讀者看了本頁，會以為處理欄新加的內容（例如 00-warmup 新增的〈第一次用 Colab〉）經過獨立檢查，打開紀錄卻看到相反的說法。同一句也在 docs/status.md 第 88 行、README.md 第 7 行與審查用的事實與寫作規範清單。 | 已處理：這一項引用的是重產前的舊紀錄。00-warmup 與首頁的紀錄改由各次查核的原始結果產生；兩頁定稿時的修正，另請一位 AI 逐項檢查，結果記在各自的〈定稿修正〉。這一輪的修正也交給另一位 AI 檢查。 |
| 2 | 必要 | 第 88 行〈審查〉「頁面引用原始論文、官方程式或函式庫文件的說法，另由 AI 對照頁尾〈參考來源〉連到的固定版本逐句核對」 | 42 節中，只有 12.1–16.3 與第 20 章共 13 頁在頁末有「參考來源：」段落。 另外 22 頁也引用了論文或官方文件，但沒有〈參考來源〉：00、01、02、3.1–3.3、05、6.1–6.2、7.1–7.6、9.1–9.2、10、11.1–11.4。這些頁的出處是正文內的連結，而且多半不是固定版本： - arXiv 連到沒有版本號的 abs 頁，例如 09-anchors 的 arxiv.org/abs/1612.08242。 - PyTorch 文件連到 /docs/stable/，例如 00-warmup、02-diagnostics、07-training。 00-warmup 的審查紀錄寫明，技術查核用的是 PyTorch v2.9.1 commit d38164a5 與 2.9 版文件，不是頁面連到的版本。讀者照這句去頁尾找審查依據的版本：22 頁找不到；00-warmup 找到的是會變動的 stable 連結。 | 已修正：改成「另由 AI 打開原始來源逐句核對（論文看原文，官方程式看固定的 commit 或 tag，函式庫看官方文件），查閱的來源記在該頁的審查紀錄」；〈驗證範圍〉、README.md 與〈發布與帳號設定〉同步。 |
| 3 | 建議 | 第 88 行〈審查〉「其他 17 頁由 AI 對照……查核，並實際執行頁面上的部分指令」 | 17 頁中有 6 頁沒有任何指令，也就是沒有程式區塊、也沒有行內指令：index.md、learning-path.md、glossary.md、planning/outline.md、planning/course-research.md、planning/feedback.md。這 6 頁沒有「頁面上的指令」可以執行。 審查用的事實與寫作規範清單沒有這一項。除了 reviews/index.md，其餘 16 份非課程頁的審查紀錄現在都不在工作樹，無法查證；reviews/index.md 執行的是建置與相關程式，不是首頁上的指令。 | 已修正：改成「頁面上有指令的，也實際執行其中一部分」；〈驗證範圍〉同步。 |
| 4 | 建議 | 第 7 行「電腦（作業系統、CPU 型號、使用的執行緒數、Python 與 PyTorch 版本）」 | miniyolo/provenance.py 的 machine_info() 還記了架構（architecture）、邏輯 CPU 數（logical_cpus）與 PyTorch 的 CPU 指令集（torch_cpu_capability）；審查用的事實與寫作規範清單也列了邏輯 CPU 數與 CPU 指令集。現在的括號讀起來像完整清單。此外，紀錄另外存有 stderr、exit_code 與 process_wall_seconds，也沒列出。 | 已修正：括號末加「等」。 |
| 5 | 建議 | 第 9 行「這項檢查不逐字比對它，也不檢查由紀錄畫出的圖」 | validate_curriculum_evidence.py 完全不檢查圖，但網站上的實驗圖不只「由紀錄畫出」一種： - 第 17、18、19 章的圖：執行紀錄那次執行產生，再由 verify_curriculum.py 的 SITE_FIGURES 複製到網站。 - *-learning.svg、08-custom-*.svg：補充腳本加 --record 時，和紀錄一起寫出。 - 只有 grid-learning 的兩張圖，是 render_learning_evidence.py 依紀錄重畫的。 照現在的寫法，讀者可能以為其他圖有被檢查。publish.md 第 6 步寫的就是「validate_curriculum_evidence.py 不檢查圖」。 | 已修正：改成「也不檢查網站上的任何實驗圖」。 |
| 6 | 建議 | 第 9 行「其中任何一個改了，紀錄才算過期，要重跑」 | 本頁沒說過期的紀錄由什麼工具列出、怎麼重產。審查用的事實與寫作規範清單寫明由 scripts/record_evidence.py 負責：不加參數只列出，--run 重產 CPU 紀錄，--gpu <分支> 透過 GitHub Actions 重跑 GPU 紀錄；逐節紀錄由它呼叫 verify_curriculum.py。想知道確切流程的技術讀者，在本頁找不到這個工具。 | 已修正：補上過期的紀錄由 `scripts/record_evidence.py` 列出並重產（GPU 紀錄經 GitHub Actions 重跑），並連到〈發布與帳號設定〉。 |
| 7 | 建議 | 第 64 行〈補充實驗清單〉「下面這些實驗把第 1、3、4、7、8、10 章的實驗延長，或接上真實的檔案與資料」 | 表裡還有第 18、19 章的影片檔實驗，而 Fashion-MNIST 不屬於任何一章（解說在〈資料規劃〉）。開頭列的章次和表格對不上。 | 已修正：改成延長第 1、3、4、7、8、10 章的實驗、讓第 18、19 章讀寫真正的影片檔，並用 Fashion-MNIST 核對分類的訓練流程。 |
| 8 | 建議 | 第 64 行「每份紀錄同樣記下執行的電腦」 | grid-learning.json 的電腦資訊由 miniyolo/train.py 的 _hardware() 寫進 hardware 欄位，只有 CPU 型號、Python、PyTorch 與執行緒數（另有 device、cuda_device），沒有作業系統、邏輯 CPU 數與 CPU 指令集。其他補充紀錄用的是完整的 machine_info()，所以「同樣」不精確。validate_curriculum_evidence.py 的 check_machine 也只要求有 CPU，再加上作業系統或 Python 其中一項。 | 已修正：「同樣記下」改成「也都記下」，不再暗示各紀錄的電腦欄位相同。 |
| 9 | 建議 | 第 114 行「結果存成兩份紀錄」與第 119 行「兩份紀錄最外層的 `passed` 都是 `true`，才 commit 到 main」 | verify-release.yml 只有 contents: read 權限，結果只上傳成 workflow artifact（site.json、bootstrap.json），不會寫進 repo： - 兩份紀錄是維護者事後執行 `scripts/verify_release.py save --run <ID>` 才下載寫入 artifacts/checks/ 的。 - 要不要 commit，由維護者依 publish.md 第 11 步判斷，程式不會強制。 - 所以 tag 裡的 curriculum-publication.json 是上一版的驗證（verify_release.py 拿它核對舊 tag）。 本頁沒提 save，讀者會以為工作流程直接寫入 repo，或以為 passed 的條件由程式把關。 | 已修正：補上工作流程把結果上傳成 Actions 的 artifact，由維護者用 `scripts/verify_release.py save --run <run 編號>` 存成紀錄；commit 的條件寫明是發布步驟，並說明 tag 裡的 `curriculum-publication.json` 是上一版的驗證。 |
| 10 | 建議 | 第 117 行「以及把 42 節的實驗格各跑一次、和紀錄比對輸出的 `scripts/check_lesson_runtime.py`」 | check_lesson_runtime.py 只有在某一節報錯或超時時，才以非 0 結束；輸出和紀錄不同不算失敗（README 第 22 行）。bootstrap 的 passed 也只看各指令的 exit code。比對結果只記在紀錄的 lesson_runtime 欄（same_output_as_record／different_output）。讀者容易以為「和紀錄比對輸出」是通過條件。 | 已修正：補上輸出和紀錄不同不算失敗，哪些節逐字相同記在紀錄的 `lesson_runtime`。 |

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

核對第一節、〈補充實驗清單〉開頭、〈審查〉與〈發布後的公開驗證〉：對照 validate_curriculum_evidence.py（42 節逐字比對 notebook stdout；頁面只找日期前 10 字元、PyTorch、CPU 字串與標記各一次；不比對區塊內容、不看圖；--scope gpu 檢查 status、gpu_calls_finished、modal_stop_confirmation）、record_evidence.py、pages.yml（名稱 Publish Learn to YOLO，驗證在建置前，deploy 依賴 build）、evidence_records.py 與各補充腳本寫入的 dependencies_sha256、verify_release.py（site／bootstrap／save）、verify-release.yml（只上傳 artifact、contents: read）、check_lesson_runtime.py（只有報錯或逾時才非 0）與 41 本 notebook 環境格雜湊（只有 20-deployment 不同）。這些句子都屬實，上一輪 10 項處理都對得上頁面，句子通順、合規；暫存副本建站與 validate_site 通過。工作樹的 grid-learning.json 尚無 dependencies_sha256，屬發布前重產紀錄的暫時狀態。〈審查〉段對紀錄的說法見下與各頁紀錄的問題。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 〈審查〉「查到的問題修正後，都由另一位 AI 檢查修正。」（上一輪 must #1 的處理） | 同 docs/status.md：00-warmup〈定稿修正〉的 23 項修正沒有任何紀錄顯示經另一位 AI 檢查；紀錄掛的檢查原文是「這一輪只改了 docs/lessons/01-small-cnn.md 與 docs/lessons/02-diagnostics.md」「並抽查了先前輪次的修改」，〈後續編輯的檢查〉也不含 00-warmup。處理所說「列出每輪修正後另一位 AI 的檢查」對 00-warmup 不成立。 | 已處理：另請一位 AI 逐項檢查 00-warmup 與首頁定稿時的修正，結果記在這兩頁紀錄的〈定稿修正〉；這句說法不變。 |
| 2 | 建議 | reviews/validation-curriculum.md | 留有內部名稱與截斷：「驗證器實驗另開一份複本：同目錄下的暫存副本」「指派的 14 頁裡，只有 docs/validation/curriculum.md 與 docs/validation/gpu-smoke.md 有改動」「8 條 leftover 抽查…」「Playwright 的 Fir…」。〈後續編輯的檢查〉第 1 項引用「reviews/00-warmup.md 第 3 行：「處理後的頁面沒有交回兩位審查者重讀…」」，重產後的 00-warmup 紀錄已沒有這句，讀者對不起來。 | 已處理：〈後續編輯的檢查〉裡引用重產前紀錄的那一項，處理欄已註明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

〈審查〉段的說法和〈驗證範圍〉相同，核對結果也相同：除了 reviews/preparation-publish.md（列在該頁），都有紀錄支持。本頁紀錄沒有〈獨立查核〉，但讀者審查的方法段寫明對照程式、紀錄並實跑腳本，〈後續編輯的檢查〉另有逐句事實查核，可以支持「其他 17 頁…查核、執行部分指令」；13 項讀者意見都在〈定稿修正〉處理；批次檢查只掛在 curriculum 與 gpu-smoke 兩頁（這一批只有它們有改動）。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/validation-curriculum.md 第 10–11 行（讀者審查方法） | 〈上述處理〉第 2 項點名「驗證器實驗另開一份複本：同目錄下的暫存副本」並寫已修正，但這句仍在第 11 行；第 10 行「主要複本：暫存副本」也是同類殘句。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 10–11 行改成「- 主要複本」「- 驗證器實驗另開一份複本」，點名的路徑佔位字已移除。處理說明屬實。

## lessons-v0.4.1 版本引用檢查

2026-10-05，另一位 AI 的獨立審查。**本頁版本引用檢查 closed，沒有必要問題。**

獨立副本為 `/tmp/lessons-v0.4.1-review-version-pages/`，以指定 rsync 從 root 複製，排除 `.git`、`site`、`.venv*`。已讀完整頁面、AGENTS 與發布審查規範，並以 read-only Git 對照 v0.4.0 HEAD `8292a666b72934f86228e5d408ffb922c1855599`。未修改 root、coverage、commit、push、tag 或 GPU。

完整 byte 對照確認：HEAD 本頁僅有一個 `lessons-v0.4.0`，替換成 v0.4.1 後與副本全文完全相同。改的是〈發布後的公開驗證〉的本版 tag，42 節清單、補充實驗、GPU、審查範圍、數據、連結及限制都沒有變。本頁沒有嵌入圖，diagrams 相對 HEAD 無變更。

讀取 `.github/workflows/verify-release.yml` 與 `scripts/verify_release.py`，核對 site／bootstrap／save、五個 fresh cases、已 import torch 時的重新啟動、20 節不要求計時 stdout 相同、README 排除 serve、頁面所列「兩份頂層 passed 均為 true 才 commit」的發布規範；save 本身只下載並保存結果。`notebook_stdout` 只收 stdout stream，接受 ipynb text 的字串與字串列表；新增測試實際覆蓋兩種表示、忽略 stderr／display、空輸出及42本原始JSON輸出與紀錄一致，沒有透過 nbformat 正規化掩蓋原 bug。它修正了 session 前讀取已存輸出的 TypeError，不代表 fresh sessions 已實跑通過。

在副本使用既有 Python 3.12.14／CPU 模型環境實際執行：

- `PYTHONPATH=. python -m pytest -q tests/test_release_verification.py tests/test_notebook_bootstrap.py tests/test_core.py tests/test_checkpoint.py`：35 passed，含修正後 helper 的 4 個 regression cases。
- `python scripts/validate_curriculum_evidence.py --report artifacts/runs/version-pages/evidence.json`：42 節與 11 份補充紀錄相符；既有 GPU 紀錄只查綁定，沒有執行 GPU。
- `PYTHONPATH=. python scripts/check_lesson_runtime.py --section 00-warmup --report artifacts/runs/version-pages/runtime-00.json`：PASS，輸出與紀錄逐字相同。
- `python scripts/validate_preparation.py`、嚴格 Zensical 0.0.67 建置及 `python scripts/validate_site.py`：均 exit 0；60 頁／42 課的連結、錨點、Colab 配對與 Markdown 轉換通過。
- `python scripts/verify_release.py --help`：exit 0；只檢查 CLI，沒有啟動公開 site／bootstrap／save。

另外逐一核對 42 節頁面入口、section-map source_ref、環境格均指 v0.4.1，末格與案例相同，41 個一般環境格逐字相同。42 本 notebook 相對 HEAD 只改 tag 與導言的舊頁名同步，實驗與存的數值輸出未變。此範圍仍清楚區分 CPU 紀錄一致性、編輯審查、發布後公開驗證與 Colab 未實測，沒有新增已完成的公開驗證宣稱。

本頁 SHA-256：`fb7393a8588304af68ef41048f3050db709eb2ecdd9a4c80ea1771ad35dc5285`；HEAD 舊頁：`df33ae0350bac4774c1284665954f27270fdac78d806bb81ac2b834dce85a26f`。結束前副本頁面再次與 root 逐 byte 相同。證據在副本 `artifacts/runs/version-pages/scope.json`、`notebook-tags.json`、`evidence.json` 與 `runtime-00.json`。

這是發布前的版本引用審查。新 tag 尚未 push，不能把本次本地驗證當成 v0.4.1 的公開驗證；主審正在進行的舊公開版本五個 fresh sessions／README 探針，以及之後的新 tag Actions，都未由本次查核證明通過。副本的 publication JSON 仍引用 v0.3.0、沒有頂層 `passed`，bootstrap JSON 尚不存在，沒有拿它們作新版本通過證據。沒有進入 Colab 託管 runtime，也沒有測 GPU 或重跑 42 節的全部實驗。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[reference當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/reference.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/reference.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/reference.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-05：v0.6.0 有界更新審閱

新增10節及官方預訓練紀錄、獨立審閱連結；舊59頁／16f6910紀錄明標歷史範圍，新版發布流程與各source_ref結果分開。

本次僅重查以上變更，既有正文的歷史審閱保留；沒有把全頁或全書重新標成首次盲讀。[導讀／實際網站審閱](clear-tutorial/vision-v0.6.0/guide-visual-review.md)、[新增支線第三輪及實際前置](clear-tutorial/vision-v0.6.0/transitions-review.md)、[首讀修後複查](clear-tutorial/vision-v0.6.0/vit-recheck.md)記錄各自範圍。全版52本notebook的本機cell執行另見[實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)；不是Google Colab登入執行。來源與圖／程式指紋更新於[coverage.json](coverage.json)。
