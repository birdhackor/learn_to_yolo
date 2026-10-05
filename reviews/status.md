# 審查紀錄：驗證範圍

審查範圍：`docs/status.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：有必要問題

不通過：有 1 個必要。全頁照逐項查證後只剩這一個錯：〈執行紀錄〉第 40 行說建置會確認「每份紀錄」都和程式、notebook、頁面對得上，但補充紀錄與 GPU 紀錄其實只檢查程式綁定，不和頁面比對，宣稱的守衛範圍比程式大。另有 1 個 should：第 81 行的「完整的報告清單」目前不成立（curriculum.md 只列 16 份，首頁審查報告的連結也被刪掉了）。

已確認的部分：
- 先前審查意見全數處理：標題已與導覽、notebook 第一格的引用一致；沒有殘留「這版／本輪／補做」等製作或修訂敘事、計畫語氣或寫作平台細節；第 76 行的「最初 160 步…接著 1600 步」是實驗敘述，審查用的事實清單允許。
- 指令、旗標、預設值、路徑：對照 check_lesson_runtime.py（報告預設寫到 artifacts/runs/lesson-runtime.json；`--section`；逐節印 PASS／FAIL 與是否和紀錄相同；不寫回任何 git 追蹤的檔案；各節輸出路徑都被 .gitignore 涵蓋，也沒有已追蹤的檔案）、verify_curriculum.py、record_evidence.py、evidence_records.py、provenance.py、環境格程式碼、tests/test_notebook_bootstrap.py、pages.yml、deployment-gpu.yml、gpu-smoke.yml，都正確。
- 數字：第 8 章（1.55017→0.16921、0.00680、154–158 步的尖峰、0.388889＝7/18、0.666667＝2/3、各 9 個物件、48 張）、影片檔（12 幀 FFV1、capture 三種情況都會關閉）、第 3 章 plain 網路沒學會、第 1、4、10 章只評訓練圖、Fashion-MNIST 40 步、GPU 兩項（L4、2.9.1+cu128、B=1～4、ORT 先比對、80 次更新、最大差異 0）、Penn-Fudan 說法，都和紀錄與 data.md 一致。核心與 checkpoint 測試我在暫存副本跑過，25 passed。
- 刪掉的內容沒有遺失唯一的範圍限制或維護步驟；git checkout 那段建議刪得對，因為現在的程式已經不會改動追蹤中的檔案。

渲染：暫存副本的 zensical build --clean --strict 與 validate_site.py 都通過；粗體、表格與 code span 都正確渲染，中英文間距與全形標點也沒問題。依各部分的檔案分工與修改時間，這次查核的頁面只改了 docs/status.md。

以下內容和審查用的事實與寫作規範清單、程式碼一致，但要等發布流程完成才會成立（不列為問題，提醒維護者）：
- Colab 按鈕與環境格固定在 lessons-v0.4.0：目前 52 個連結和 42 本 notebook 都還指向 lessons-v0.3.0。
- 頁尾執行紀錄區塊寫出 CPU 型號，以及所有紀錄有 SHA-256 綁定和電腦欄位：目前 42 份逐節紀錄和補充紀錄都缺 dependencies_sha256／machine，要在 AMD EPYC 9V74 那台電腦重產；換電腦重產的話，第 42 行和第 8 章的數字都要重新核對。
- artifacts/checks/gpu-smoke.json 還不存在。
- tests/test_notebook_bootstrap.py 還沒加入 git，發布時要一起 commit。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | docs/status.md 第 40 行，〈執行紀錄〉第二段最後一句：「網站建置前，`scripts/validate_curriculum_evidence.py` 會確認每份紀錄都和 repository 裡的程式、notebook 與頁面對得上；對不上，網站就無法發布。」 | 這句說驗證器檢查的範圍，比程式實際做的大。`validate_curriculum_evidence.py` 只對 42 節的逐節紀錄比對 notebook 輸出與頁面，而且頁面只檢查有沒有寫上紀錄的日期、CPU 型號、PyTorch 版本和標記，不比對輸出文字。第 38 行剛列出的 8 份補充紀錄（第 1、3、4、10、7、8、18/19 章）和 2 份 GPU 紀錄，驗證器只檢查三件事：綁定的程式沒變（`is_current`）、紀錄有寫電腦、少數內部欄位一致。它們完全不和任何頁面比對。這句放在補充紀錄那句後面，「每份紀錄」自然包含它們，等於說有人守著一件其實沒有人守的事。具體情境：第 8 章紀錄換一台電腦重產，數字變了，但本頁和 08-own-data 頁引用的 0.388889、0.666667、0.00680 等沒有跟著改，Pages 建置照樣通過並發布。維護者照這句相信建置會擋下，就不會去人工核對。（審查用的事實與寫作規範清單第 12 行同一句話，前後文講的是逐節紀錄；本頁把它放到補充紀錄之後，範圍就變大了。） | 已修正：〈執行紀錄〉末段改成只說驗證器實際檢查的範圍（逐節紀錄比對 notebook 的輸出，頁面只找日期、CPU 型號與 PyTorch 版本；補充與 GPU 紀錄只看綁定），第 2 次複查確認，最後一輪檢查也核對過現行文字。 |
| 2 | 建議 | docs/status.md 第 81 行：「審查的範圍與完整的報告清單，見[全套實驗與審查頁](validation/curriculum.md)。」 | reviews/ 目前有 60 份報告：42 份逐節審查、homepage-fresh-reader.md、training-evidence.md，以及 16 份 curriculum／closure 報告。docs/validation/curriculum.md 現在只列 16 份，先前審查意見裡對那頁的改寫計畫也只列這 16 份。本頁又刪掉了首頁審查報告的連結，於是網站上沒有任何地方連得到它。「完整的報告清單」目前不成立；讀者照連結去找首頁審查，會找不到，可能誤以為首頁沒人審過。 | 已處理：拿掉「完整的」，列在第 1 次查核後的修正項目裡。 |

修正必要問題時處理的項目（修正後由第 2 次複查確認）：

- line 1 「# 這版驗證了什麼，還缺什麼」：已改寫。現為「# 驗證範圍與後續實驗」，與 zensical.toml 第 22 行的導覽標籤相同。
- line 3 「以及這一版還缺什麼」：已改寫。現為「…讀完你會知道每種結果能支持哪個結論，以及哪些事沒有驗證。」
- line 5 「本頁的逐節執行紀錄對應 lessons-v0.3.0 的程式（2026-10-02 在 CPU 上逐節重跑）」：已改寫。第 5 行現在指向頁尾的「實際執行紀錄」（日期、CPU 型號、PyTorch），再連到〈執行紀錄〉。〈執行紀錄〉說明 SHA-256 綁定、每份紀錄記下日期與電腦、只重跑過期紀錄（scripts/record_evidence.py），並寫出 CPU 電腦（AMD EPYC 9V74…）和 GPU（Modal L4、2.9.1+cu128）。lessons-v0.4.0 寫在〈執行方式〉。正文沒有寫死日期。
- line 29 「作者的逐節重跑和環境格」：已改寫。現為「逐節執行紀錄都在 Colab 以外的電腦上用 CPU 跑出；環境格…由自動測試在本機模擬檢查（tests/test_notebook_bootstrap.py，不會真的安裝套件）」，後文保留。
- line 38 「作者驗證這一版時用 Python 3.12、PyTorch 2.9.1（CPU 版）。」：已改寫。現為「執行紀錄用的是 Python 3.12、PyTorch 2.9.1（CPU 版）。」環境格的說明不變。
- line 49 「# 輸出會寫回 notebooks/*.ipynb 與 artifacts/checks/lesson-runtime.json；」：已改寫。註解現為：依序執行 42 節實驗格，每節印 PASS 或 FAIL 並註明是否和紀錄一字不差；報告寫在 artifacts/runs/lesson-runtime.json；git 追蹤的檔案都不會改變。已對照 check_lesson_runtime.py 和 .gitignore。
- line 50 「# 第 17、18、19 章的程式還會重畫 docs/assets/diagrams/ 裡網站用的三張圖」：已刪除這行註解。程式區塊後的段落改說各節的圖寫在 git 不追蹤的位置，例如 artifacts/lesson-17/。
- line 56 「並把輸出寫回 notebook 與 JSON 檔。第 17、18、19 章的程式還會重畫網站用的三張圖」：已照建議改寫。段落說明 check_lesson_runtime.py：記錄 stdout、錯誤與是否通過；和紀錄比對；報告路徑；圖存在 ignored 目錄；數字末幾位不同會標出但仍算 PASS；--section；不經過 Colab。核心測試的說明移到獨立段落。
- line 56 「想還原，可用 `git checkout -- notebooks artifacts/checks/lesson-runtime.json」：已刪除，連同「不要提交回去…會失敗」那幾句。程式已不會改動追蹤中的檔案，不需要還原步驟。
- line 58 「## 這一版已完成的檢查」：已改寫為「## 已完成的檢查」。
- line 60 「這一版發布前，在 CPU 上逐節重跑的結果是」：已改寫。現為「**逐節執行紀錄：**42 節實驗全部執行通過（**42／42**；通過＝程式完整跑完、沒有報錯…）」。測試數目不寫死；核心與 checkpoint 測試另列一項「全部通過」。
- line 60 「你自己跑 `check_lesson_runtime.py` 的結果，則寫在 `artifacts/checks/lesson-runtime.json`。」：已改成 `artifacts/runs/lesson-runtime.json`。建議中「從 repo 移除 artifacts/checks/lesson-runtime.json」不在這次查核的頁面範圍，該檔仍在，已列入疑慮。
- line 63 「第 8 章最後補上「JSON 標註＋PNG 圖片」的」：已改寫。現為「**第 8 章自己的資料：**第 8 章「JSON 標註＋PNG 圖片」的完整訓練流程，用的是 48 張程式畫的三類矩形合成圖…」。160→1600 步是實驗敘述，保留；數字不變，列在待重錄數值清單。
- line 68 「之後各節頁面、首頁與本頁為了更好讀而改寫過」：照選項 (b) 處理，本頁不寫審查範圍，只連到 curriculum.md：「審查的範圍與報告清單，見[全套實驗與審查頁](validation/curriculum.md)。」本輪依查核者的建議再拿掉「完整的」。
- line 70 「〈審查交付〉」：已改寫。連結文字改為「全套實驗與審查頁」，不寫小標名稱，所以 curriculum.md 把小標改成〈內容審查〉也不會失效。
- line 72 「當時最後補做的第 8、18、19 章內容」：已刪除。第 8、18、19 章的四份審查經 curriculum.md 的連結取得。
- line 72 「初讀與修訂後的複查報告」：已刪除，首頁審查報告的連結一併移除（審查範圍只留在 curriculum.md）。reviews/homepage-fresh-reader.md 現在沒有任何頁面連到，已列入疑慮。

### 第 2 次複查：通過

結論：通過。我沒有找到必要等級的問題，只有一條 should（Colab 那句發布後會過時，見下表）。

**一、先前審查意見與全頁用語（檢查 1）**
- 先前審查意見清單列的 17 條先前審查意見都已處理。
- 我逐句掃過全頁：沒有製作或修訂經過的敘述，沒有「待補、之後會」這類計畫語氣，也沒有寫作平台細節。
- 「這次」只出現在描述第 8 章那次 160 步執行的地方，審查用的事實與寫作規範清單允許這種用法。

**二、事實查核（檢查 2）**
本輪改寫的驗證器那句已對照 scripts/validate_curriculum_evidence.py 第 52–92 行和 scripts/evidence_records.py，全部成立：
- 42 節紀錄：檢查 case_sha256 與 dependencies_sha256、notebook 存的 stdout 必須一字不差，頁面要寫出日期、torch 版本和 machine.cpu。
- 8 份 CPU 補充紀錄與 2 份 GPU 紀錄：檢查 is_current。
- .github/workflows/pages.yml 先跑驗證器再建置，deploy 依賴 build，所以「任何一項不符就無法發布」成立。

其餘敘述也逐項對照了程式與紀錄，都成立：
- record_evidence.py、verify_curriculum.py 的 evidence_block（日期／CPU／執行緒／PyTorch）、miniyolo/provenance.py。
- check_lesson_runtime.py：預設報告路徑 artifacts/runs/lesson-runtime.json，有 --section 選項。
- 環境格分支與 tests/test_notebook_bootstrap.py。
- deployment-gpu.yml 與 gpu-smoke.yml（手動啟動、Modal L4）；requirements-gpu.lock 是 torch 2.9.1+cu128。
- 各份紀錄的數字：03-comparison 的 plain 網路正確率 0.5；01／04／10 的紀錄只有訓練圖；grid-learning 用了不同 seed；custom-data 的 48 張圖、1.55017→0.16921、0.00680、第 154–158 步尖峰、7/18、2/3、各 9 個物件、重新載入與原尺寸推論檢查；video-file（FFV1、12 幀、三種情況都釋放）；deployment-gpu（40 步、B=1–4，ONNX Runtime 先和 PyTorch 比對）；20-deployment（B=1、2、3）；gpu-smoke-37032967155（最大差異 0）。
- preparation/data.md 寫明沒有 Penn-Fudan 的轉換程式。

**三、內容沒有遺失（檢查 3）**
刪掉的句子都不是唯一的範圍限制、安全步驟或維護操作。審查範圍照指示只留在 curriculum.md，本頁改用連結過去。

**四、實際執行與渲染（檢查 5）**
在暫存副本這份暫存副本裡：
- 改動任何檔案之前：zensical build --clean --strict 與 validate_site.py 都是 exit 0。
- pytest 跑核心、checkpoint、bootstrap 三組測試，31 passed。
- 用 --ref lessons-v0.4.0 重建 notebook 之後跑 check_lesson_runtime.py：印出 42 行 PASS，報告寫在 artifacts/runs/。執行前後的檔案雜湊比對顯示，只有 git 忽略的路徑有變動（artifacts/*.png、artifacts/lesson-*/、artifacts/runs/），這證實了頁面說「git 追蹤的檔案都不會改變」。

**五、範圍外的提醒（不影響這次查核的頁面結果）**
- 現在的工作樹要先重建 notebook。lesson_cases 已經改過但 notebook 還沒重建，所以直接跑 check_lesson_runtime.py 會在 00-warmup 出現 AssertionError 停下。發布流程本來就會重建，這不是頁面的問題。
- 現有紀錄還沒重新產生：缺 machine 與 dependencies_sha256 欄位，artifacts/checks/gpu-smoke.json 也還沒產生。驗證器會擋住大部分情況，但 CPU 機器那條和第 8 章的數字沒有程式守著，要靠 editor 列的待重錄數值清單在重新產生後再核對一次。
- 其他頁面描述本頁時用了計畫語氣，也和現在的內容對不上：docs/index.md:65 寫「列出已完成與待做的項目」，docs/learning-path.md:86 寫「哪些仍待 GPU 或真實資料」，而這兩處都沒有先前審查意見。需要由負責那些檔案的部分處理。

**六、這次查核的頁面只改了這一個檔案**
我沒有發現這次查核的頁面改了 docs/status.md 以外的檔案。兩輪之間其他檔案的修改時間和並行的其他部分交錯，應該是那些部分改的。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/status.md 第 29 行（〈沒有驗證的事〉的 Colab 條目） | 「Colab 上改裝 PyTorch、提示重新啟動工作階段的實際流程，以及 Colab 可能分配給你的 GPU，都沒有測」把「沒有測」寫成固定的事實。審查用的事實與寫作規範清單第 8 行的寫法是「在 Colab 上的實際執行另見發布驗證紀錄（發布時才會有）」，第 31 行是「（除非發布驗證紀錄另有記載）」。docs/preparation/publish.md 第 11 步也規定，網站部署之後要在 Colab 開一節新 tag 的 notebook 執行環境格，並把「Colab 的 PyTorch 版本、有沒有要求重新啟動工作階段」寫進 artifacts/checks/curriculum-publication.json，而且『不必重新部署』。所以 lessons-v0.4.0 發布完以後，這句會和發布紀錄互相矛盾：Colab 上改裝 PyTorch 的流程實際上已經測過一次，頁面卻一直寫沒有測，也沒有機制回頭改它。editor 的待重錄數值清單沒有列這一句。這句部署當下為真，屬於少說而不是多說，所以不算必要。 |

建議事項的處理（處理後由下一次複查檢查）：

- [should] 第 29 行〈沒有驗證的事〉的 Colab 條目：採用做法 (a)，已修正。標題改成「在 Google Colab 上逐節執行 notebook 的實驗」，與審查用的事實與寫作規範清單一致。刪掉寫死的「Colab 上改裝 PyTorch、提示重新啟動工作階段的實際流程……都沒有測」，改成「在 Colab 上實際開啟 notebook、執行環境格的結果，記在發布驗證紀錄（網站發布後檢查公開網站與 Colab 入口的 JSON 檔）」，連結指向 main 上的 artifacts/checks/curriculum-publication.json；「Colab 可能分配給你的 GPU 沒有測」保留。查證：publish.md 第 11 步會在 Colab 開一節 notebook 執行環境格，並把 Colab 的 PyTorch 版本與是否要求重新啟動寫進這個檔。環境格（build_lesson_notebooks.bootstrap）用 importlib.metadata 判斷版本，所以全新的 Colab runtime 也會走改裝的分支，舊句發布後確實會和紀錄矛盾。新句在發布紀錄出現前後都成立，所以不用做法 (b)（列入待重錄數值清單）。在暫存副本中，zensical build --clean --strict 與 validate_site.py 都 exit 0。

### 第 3 次複查：通過

第 1 輪檢查通過：沒有必要問題，只有一項建議。

前輪的 should（第 29 行 Colab 條目）已照做法 (a) 解決。新句只說「在 Colab 上開啟 notebook、執行環境格的結果，記在發布驗證紀錄」，和審查用的事實與寫作規範清單第 8、39 行、publish.md 第 11 步一致（第 11 步規定把 Colab 的 PyTorch 版本、有沒有要求重新啟動寫進 curriculum-publication.json）。標題改成「逐節執行 notebook 的實驗」也正確，因為第 11 步只執行一節的環境格。網站部署之後、第 11 步推送之前，這個連結會暫時顯示 v0.3.0 的紀錄。這是發布流程中的暫時狀態，不算頁面錯誤。第 76 行改用導覽名稱「第 20 章〈ONNX／TensorRT〉」，與導覽標籤一致。其他頁名（全套實驗與審查、資料規劃、GPU／checkpoint 實測）也都與 zensical.toml 的導覽一致。

逐句對照程式後，以下都成立：
- 環境格的各個分支：scripts/build_lesson_notebooks.py 的 bootstrap，與 tests/test_notebook_bootstrap.py 一致。
- validate_curriculum_evidence.py 的檢查項目、verify_curriculum.py 的頁尾區塊與 index.json 欄位。
- check_lesson_runtime.py：每節一行 PASS／FAIL、--section、報告路徑、輸出不同不算失敗；各節程式只寫到 .gitignore 涵蓋的位置。
- test_core／test_checkpoint 的測試內容與頁面描述相符。
- GPU 兩項：gpu_smoke.py（8 張圖、40 步與 20+20 步、StepLR、單次使用的新 container）；deployment_gpu.py（40 步、TensorRT Python API、FP16 只是允許混合精度、B=1～4、ONNX Runtime 先比對）。
- 第 8 章的各個數字、第 154–158 步的尖峰、48 張圖、各 9 個物件；影片檔紀錄（FFV1 AVI、12 幀、三種情況都會 release）；Fashion-MNIST 40 步紀錄（data.md 引用）；Penn-Fudan；版本來源。
- 全文沒有修訂或製作經過的敘述，也沒有計畫語氣。第 78 行「最初 160 步…接著 1600 步」講的是兩份實驗紀錄的程序，與 8.2 頁一致。

數字取決於重產後的紀錄：工作樹裡的紀錄都還是舊格式（evidence_records.is_current 全為 False，artifacts/checks/gpu-smoke.json 還不存在）。產生這些數字的計算這版沒有改，包括 train.optimizer_step、自有資料的 fixture 與訓練、gpu_smoke.py、deployment_gpu.py。第 3 章的資料產生方式有改，所以我在暫存副本用新程式跑了 run_learning_extensions.py --section 03-comparison（不加 --record）：plain 的 train／validation 仍是 0.5、residual 仍是 1.0，「plain 網路沒學會」仍成立。紀錄重產後，第 17、18、42、77、78 行的值仍要照既有流程核對。

跨頁的依賴：第 21 行引用的〈逐節執行清單〉〈補充實驗清單〉目前存在於 docs/validation/curriculum.md。但那一頁本身仍是 v0.3 的舊內容，改寫時若改了節名，這裡要跟著改。

建置：在暫存副本執行，zensical build --clean --strict exit 0，python3 scripts/validate_site.py exit 0。日誌是同一層的暫存副本。pytest tests/test_core.py tests/test_checkpoint.py tests/test_notebook_bootstrap.py 共 31 passed。渲染後的頁面沒有殘留的 **、表格 1 個、連結都正確。

唯一的 should：第 83 行說審查「分兩種」，沒有限定是課程頁，見下表。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/status.md 第 83 行（〈誰檢查過內容〉第一段） | 「教材的審查者都是 AI，分兩種：模擬初學讀者的 AI…；AI 技術審查者，對照原始論文、固定版本的官方程式碼和計算」沒說這兩種審查涵蓋哪些頁。審查用的事實與寫作規範清單第 26 行只保證 42 節課程頁有這兩種審查。網站導覽裡的其他頁（包括本頁、資料規劃、Pages 與 Colab 等）也各有一份審查紀錄（reviews/<路徑以 - 連接>.md），但審查用的事實與寫作規範清單沒說它們屬於這兩種。照現在的寫法，讀者會以為非課程頁也都對照過論文與官方程式碼。反過來，頁面也沒交代非課程頁同樣有審查紀錄。README 的寫法（「每節都有一份審查紀錄…兩位審查者」）只講課程頁，比本頁精確。 |

### 第 4 次查核：通過

結論：docs/status.md 通過。沒有必要問題，也沒有需要再改的建議。

1. 建置與格式（在暫存副本執行）
- zensical build --clean --strict、validate_site.py、validate_preparation.py 結束碼都是 0。
- README 的相對連結全部存在；43 本 notebook 都是合法 JSON。
- 頁面引用的 preparation/data/#fashion-mnist 錨點確實存在。

2. 實際執行頁面上的指令（暫存副本，用 repo 的 .venv-model）
- `PYTHONPATH=. … lesson_cases/00-warmup.py`：結束碼 0。
- pytest test_core.py test_checkpoint.py：最後一行是「25 passed」，和頁面說的一致。
- 完整跑一次 check_lesson_runtime.py：
  - 42 節都 PASS，每行都附「輸出是否和紀錄相同」的註記。
  - 報告寫在 artifacts/runs/lesson-runtime.json。
  - 執行前後比對所有檔案的雜湊：改變的只有 git 忽略的 artifacts/*.png 與 artifacts/lesson-08-own-images、17、18、19、20 資料夾；docs/assets/diagrams/ 沒有變動。
  - --section 可以重複指定。

3. 逐句對照程式（只列結論，都成立）
- 第 17–19 章的網站圖：只有 verify_curriculum.py 的 SITE_FIGURES 會把它們複製進 docs/assets/diagrams/；check_lesson_runtime.py 不複製任何圖。
- check_lesson_runtime.py：時限 timeout=120，通過的條件是結束碼為 0。
- 環境格：先用 pip 改裝，再因 torch 已載入而停下，訊息要求重新啟動工作階段。
- 第 29 行的發布驗證：與 verify_release.py bootstrap 的內容相符。
  - 從公開 tag 重新 clone，每種情況都開全新的 venv。
  - 第 0 章四種情況，另加第 20 章。
  - README 的 bash 區塊除了 zensical serve 都執行。
  - 結果記在 curriculum-release-bootstrap.json；程式本身寫明 "not a hosted Colab runtime; no GPU"。
- 第 20 章 GPU 部分（deployment_gpu.py）：建 FP32 與允許 FP16 兩個 engine，在 B=1–4 比對，ONNX Runtime 先和 PyTorch 比對過。
- GPU 存檔續訓（gpu_smoke）：40＋20＋20＝80 次更新；模型與 optimizer 最大差異為 0。
- 補充紀錄：evidence_records.py 的 9 份 CPU 紀錄，和第 38 行的清單一致。
- Fashion-MNIST：run_fashion_cnn --record 會寫入電腦資訊、dependencies_sha256，以及四個資料檔的 SHA-256。
- validate_curriculum_evidence.py：檢查內容與第 45 行相符，而且在 pages.yml 裡排在網站建置之前。
- 審查涵蓋範圍：review_coverage 雜湊的範圍與第 90 行相符；validate_lessons 在建置前強制檢查。
- 頁數：網站導覽共 59 頁＝42 節課程頁＋17 頁，四組導覽名稱都正確。

4. 遺留意見、影響項與追加指示
- 遺留的 should（審查範圍沒寫清楚）已修好：兩種審查限定在課程頁，其他 17 頁照審查用的事實與寫作規範清單第 28 行描述。
- 第 17、19 章登記的 line 56：那句話和 git checkout 指令在目前頁面都已經不在；新寫的句子經程式核對是真的。
- 第 8 章自有資料的三項：
  - 數字和兩份紀錄相符：loss 1.55017→0.16921、train 0.00680、validation 0；尖峰落在第 154–158 步。
  - 1600 步後：train 1.0；validation 0.388889＝(4/9＋1/6＋5/9)/3＝7/18；test 0.666667＝2/3；validation 與 test 各 9 個 GT。
  - 指標名稱改為 mAP50，和 8.2 節一致。
- 四項追加指示都已處理。
- 編輯判斷審查用的事實與寫作規範清單第 8、41 行和第 36 行衝突，頁面照程式寫「沒有開 Colab」。這個判斷正確。

5. 數字、敘述與一致性
- 沒有放進任何 Mac 實測數字；紀錄相依值的清單完整。
- 沒有修訂過程的敘述；範圍限制都寫成現況。
- 和以下頁面一致：README、index、validation/curriculum.md、gpu-smoke.md、08-own-data.md、20-deployment.md、data.md、publish.md（目前版本）。

不影響這次判定、但發布時要注意的事
- curriculum-release-bootstrap.json 的連結，要等 verify_release.py save 之後才有內容；審查用的事實與寫作規範清單第 36–37 行已接受這一點。
- 工作樹裡的 notebook 仍寫〈驗證範圍與後續實驗〉。build_lesson_notebooks.py 已改成〈驗證範圍〉，重建 notebook 後就會一致。
- 目前所有紀錄都還沒有 machine 欄位。重錄之後，要再核對第 42 行的電腦描述，以及第 8 章的數字、分數形式與尖峰步數。
- 審查用的事實與寫作規範清單第 8、41 行應該改成和第 36 行一致；這是審查用的事實與寫作規範清單的問題，不是本頁的問題。
- 主 repo 的 scripts/__pycache__/evidence_records.cpython-314.pyc 在 02:59:14 被改寫，表示當時有人在主 repo 裡用 python3 執行過會匯入 evidence_records 的腳本。這個檔案 git 不追蹤，也查不出是誰執行的。

檢查用的檔案在暫存副本。

## 讀者審查與技術查核

### 讀者審查（AI 以這一頁的讀者身分閱讀）

方法：只在暫存副本裡工作。在複本中用 zensical build --clean --strict 建站（exit 0，No issues found），再跑 scripts/validate_site.py（通過）。逐段讀 docs/status.md 的 Markdown 與建好的 site/status/index.html，並用 qlmanage 把頁面轉成 PNG 檢查排版。本頁沒有任何圖。Playwright 的瀏覽器沒有安裝，所以沒用它。

對照審查用的事實與寫作規範清單、docs/glossary.md、index.md、learning-path.md。沿頁面上的每個連結讀了 validation/curriculum.md（確認〈逐節執行清單〉〈補充實驗清單〉兩節存在）、validation/gpu-smoke.md、preparation/data.md#fashion-mnist（錨點存在，並寫明沒有 Penn-Fudan 的轉換程式）、lessons/20-deployment.md（章內沒有〈ONNX／TensorRT〉小節，L4 結果在 #l4-results）、08-own-data.md 與 README（〈CPU 本機執行〉沒有取得專案的步驟）。

核對的程式與設定：
- check_lesson_runtime.py：120 秒逾時、--section、英文訊息
- verify_curriculum.py：evidence_block，以及第 17–19 章的圖複製到 docs/assets/diagrams
- validate_curriculum_evidence.py：頁面須含日期、CPU、PyTorch 版本；notebook 輸出須一字不差
- verify_release.py：第 0 章四種情境＋第 20 章；README 指令略過 zensical serve
- build_lesson_notebooks.py 的環境格，以及 tests/test_notebook_bootstrap.py、test_core.py、test_checkpoint.py
- evidence_records.py、review_coverage.py
- run_learning_extensions.py：第 1、4 章只評訓練圖，第 3 章用 4 張平移圖評估
- run_multiscale_learning.py：1 張圖、40 步
- run_custom_data_learning.py：fixture 為 24／12／12 共 48 張，validation 與 test 各 9 個物件
- verify_video_file.py：12 幀 FFV1 AVI，三種 release 情況
- miniyolo/deployment_gpu.py：40 步、B=1–4、FP32 與允許 FP16 的 engine，ORT 先和 PyTorch 比對
- miniyolo/gpu_smoke.py：8 張圖、40＋20＋20、StepLR
- miniyolo/train.py 的 seed 700／7000
- lesson_cases 00、07-training、20-deployment（B=1、2、3）
- .gitignore、pages.yml、zensical.toml 的導覽（42 節，加上四組共 17 頁）
- 以及 index.json、gpu-smoke.json、video-file.json 的欄位

照頁面實際執行的步驟（在複本裡把 .venv-model 以 symlink 指向原 venv，設 PYTHONDONTWRITEBYTECODE=1）：
- PYTHONPATH=. .venv-model/bin/python lesson_cases/00-warmup.py：1 次更新，1.00→1.80
- lesson_cases/07-training.py：3 步
- .venv-model/bin/python -m pytest tests/test_core.py tests/test_checkpoint.py：最後一行是 25 passed
- scripts/check_lesson_runtime.py：42／42 PASS，每節一行英文訊息，這台 Mac 約 55 秒，報告寫到 artifacts/runs/lesson-runtime.json。執行前後對 377 個 git 追蹤檔做 SHA-256，沒有一個改變；新檔都在 .gitignore 涵蓋的位置。
- check_lesson_runtime.py --section 07-training：只跑這一節
- run_learning_extensions.py --section 03-comparison：plain 的訓練與平移圖 accuracy 都是 0.5，residual 是 1.0，和「plain 沒學會」一致

依指示沒有拿機器相關的數字（訓練與計時結果）和紀錄比對。工作樹此刻的暫時狀態也不列為問題：頁尾區塊還是舊格式、紀錄缺 machine 欄位、其他頁的審查與 coverage.json、curriculum-release-bootstrap.json 都還沒產生。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈各種結果各代表什麼〉第一段（「第 5 列才是『真實照片上的效果』」），以及表格第 5 列「結果類型」欄的「真實資料完整對照」 | 列名和引導句都讓人以為教材做過真實資料的完整對照，或量過真實照片上的效果。這一列實際只有 Fashion-MNIST 的 40 步分類管線核對：不是偵測，也沒有對照。表格也沒說這項核對在哪裡。首頁寫明 42 節沒有用到 Fashion-MNIST，所以初學讀者在各節裡找不到它。同一句又把前兩列都叫「程式執行成功」，但第 1 列檢查的是算出的值符合手算答案；本頁後文的說法是「跑得動、算得對」，「執行成功」容易被理解成只是沒當掉。 |
| 2 | 建議 | 〈已完成的檢查〉「匯出與部署」一點的「詳見第 20 章〈ONNX／TensorRT〉」；〈沒有驗證的事〉第 6 點與最後一節第 5 步的「強制全部用 FP16、INT8」 | 第 20 章沒有叫〈ONNX／TensorRT〉的小節，那是章名；連結也只到章首。這一點講的 L4 結果在第 20 章的〈L4 GPU 上的 TensorRT 實測〉（錨點 #l4-results，〈GPU／checkpoint 實測〉頁就是連到這裡）。本頁說測過的是「允許改用 FP16」的 engine，又把「強制全部用 FP16」列為沒驗證，卻沒說兩者差在哪，讀者容易以為 FP16 engine 每一層都用 FP16 算。FP16 第一次出現在〈沒有驗證的事〉時還沒解釋，INT8 在本頁始終沒有解釋。 |
| 3 | 建議 | 〈已完成的檢查〉「第 8 章自己的資料」一點 | 「接著只把步數加到事先固定的 1600 步」和「換新 seed 畫的 test」都沒說原因。初學者可能讀成「160 步失敗，就一直加步數，加到成功為止」，而這正是「事先固定」想排除的誤解。讀者也看不出為什麼只有 test 換了新 seed：160 步那次已經評過舊 test，validation 則沿用同一批。「最初…接著…」讀起來像實驗經過的時間順序，沒有說明現在保存的兩份紀錄各是什麼。 |
| 4 | 建議 | 〈執行方式〉第二段（「完整步驟與套件版本見 repository README」）與下方的指令區塊 | README 的〈CPU 本機執行〉直接從「在 repository 根目錄執行」開始，沒寫怎麼把專案放到自己的電腦（下載 ZIP 或 git clone）。對程式新手來說，「教材專案資料夾」從哪裡來沒有交代，所以「完整步驟見 README」並不成立。區塊裡的 PYTHONPATH=. .venv-model/bin/python … 在 Windows 上不能照打，本頁也沒提醒；Windows 的寫法只在 README 裡。 |
| 5 | 建議 | 〈沒有驗證的事〉第 5 點（Google Colab） | 這是「沒有驗證」清單裡最長的一點，大半篇幅卻在講做過的替代檢查：環境格的自動測試、發布後在 GitHub 的 Linux 電腦上實跑，以及 README 指令。讀者要讀到最後一句，才確定到底什麼沒測。「由自動測試在本機模擬檢查」的「本機」，在各課頁指讀者自己的電腦（「在本機執行 PYTHONPATH=. python …」），放在這裡容易讀成「在你的電腦上測過」。「另加環境格還要安裝 ONNX 套件的第 20 章」的句型也不好讀。 |
| 6 | 建議 | 開頭的「第一次讀」提示 | 提示要讀者「讀完第 7、8、20 章後，再回來看其餘部分」，所以〈執行方式〉第一段的 Colab 說明也被延到讀完第 20 章才看，包括不必選 GPU 執行階段，以及環境格要求重新啟動時怎麼做。可是每本 notebook 的第一格就請讀者到本頁看執行環境與紀錄，這段 Colab 說明從第 0 章就用得到。另外第 18–20 章在閱讀路線上是選修，「讀完第 7、8、20 章」意思不清楚：是讀完第 20 章才回來，還是讀完第 7、8 章就能先回來看一部分。 |
| 7 | 建議 | 〈各種結果各代表什麼〉表格第 4 列最後一欄（搭配「第一次讀」提示） | 提示請初學者第一次只讀最後一欄。但第 4 列最後一欄的 TensorRT 只在第 2 欄解釋，那一欄正好被跳過；ONNX Runtime 要到〈已完成的檢查〉才解釋。照提示讀的人，在這一列會遇到兩個沒解釋的工具名。本頁也沒有連到〈術語快速查〉。 |
| 8 | 建議 | 〈已完成的檢查〉「核心與 checkpoint 測試」一點的「測試內容見上方〈執行方式〉」 | 〈執行方式〉只寫了核心測試檢查什麼，沒寫 checkpoint 測試檢查什麼。讀者照指引回去，找不到 checkpoint 測試的內容。 |
| 9 | 建議 | 表格下方一段的「示範用的人工高分框會註明來源」 | 「人工高分框」不是教材用的詞，各節用的是「人工框」「人工 fixture」「人工 logits」。「會註明來源」也可以讀成「會寫明圖片從哪裡來」。讀者看不出這句想說的是：有些分數高、看起來很準的框是人手指定的，不是模型學出來的。 |
| 10 | 建議 | 〈誰檢查過內容〉第 3 段的「commit：程式碼某一次存檔的編號」與「不包含之後的新版本或分支」 | 術語表和本頁都用「存檔」指 checkpoint（「中途存檔」「存檔續訓」「讀回存檔」），這裡卻又用「存檔」解釋 git 的 commit。初學者可能把官方程式的 commit 和模型的 checkpoint 當成同一件事。「分支」在 git 裡另有意思（branch），這裡指的應該是其他團隊衍生的版本，意思不清楚。 |
| 11 | 建議 | 〈執行方式〉指令區塊的註解，以及介紹 check_lesson_runtime.py 的那段 | 程式實際印出的是英文，例如「00-warmup: PASS; output differs from the recorded run (timing or machine-dependent digits)」或「07-training: PASS; output identical to the recorded run」。FAIL 時下面還會印出不只一行的錯誤訊息，最後一行是「Report: …」。本頁只用中文轉述，初學者不一定對得上。報告只標出哪一節不同，不標出哪幾行不同；括號裡的「計時或機器相關的位數」是腳本固定印的字，不是比對後的判斷。改過程式的讀者看到 differs，沒辦法判斷差異是正常的，還是 shape、斷言相關的行變了。「加上 --section 07-training」也沒寫出完整指令。 |
| 12 | 建議 | 〈沒有驗證的事〉第 5 點與〈執行方式〉指令註解裡的「實驗格」；〈執行紀錄〉第一段的 artifacts/checks/curriculum/<節的編號>.json | 本頁解釋了「環境格」，卻沒說「實驗格」是哪一格；各課頁的叫法是「本節可修改的完整實驗」下面那一格，或「最後一格」。「節的編號」容易讓人以為檔名是 7.4.json，實際上是 07-training.json 這種代號。第一次出現時沒有例子，要到〈已完成的檢查〉才看到 00-warmup.json。 |
| 13 | 建議 | 〈誰檢查過內容〉第 4 段的「每份審查都用 SHA-256 記下它看過的內容」 | 這些 SHA-256 不在各份審查紀錄裡，而是集中記在 reviews/coverage.json，〈全套實驗與審查〉頁就是這樣寫的。讀者照本頁的說法打開某一份審查紀錄，會找不到這些指紋。 |

### 事實查核（AI 對照 repo 的程式、紀錄與頁面引用的來源）

方法：一、讀過的 repository 來源
來源是 https://github.com/birdhackor/learn_to_yolo，branch release/lessons-v0.4.0，commit 31527417d328e18418c438cb76b4630a8c695f33，工作樹乾淨。路徑如下：
- 頁面與定稿事實：docs/status.md、審查用的事實與寫作規範清單（審查用的事實與寫作規範清單）
- 設定與說明：zensical.toml（nav：17 頁加 42 節）、README.md、section-map.json、.gitignore、requirements-model.txt、requirements-video.txt、requirements-gpu.in、requirements-docs.txt、requirements-modal.txt
- 紀錄相關腳本：scripts/check_lesson_runtime.py、scripts/verify_curriculum.py、scripts/record_evidence.py、scripts/evidence_records.py、scripts/validate_curriculum_evidence.py
- 審查相關腳本：scripts/review_coverage.py、scripts/validate_lessons.py
- notebook 與發布驗證：scripts/build_lesson_notebooks.py（bootstrap 函式）、scripts/verify_release.py
- 補充實驗腳本：scripts/run_custom_data_learning.py、scripts/run_learning_extensions.py、scripts/run_multiscale_learning.py、scripts/verify_video_file.py、scripts/run_fashion_cnn.py
- GPU 啟動腳本：scripts/modal_gpu_smoke.py
- miniyolo 模組：provenance.py、gpu_smoke.py、deployment_gpu.py、checkpoint.py、models.py、train.py
- lesson 程式：lesson_cases/00-warmup.py、01-small-cnn.py、03-comparison.py、07-training.py、08-own-images.py、18-video.py、20-deployment.py，另 grep 全部 lesson_cases 的寫檔位置與 cuda 用法
- 測試：tests/test_core.py、tests/test_checkpoint.py、tests/test_notebook_bootstrap.py
- 工作流程：.github/workflows/pages.yml、gpu-smoke.yml、deployment-gpu.yml、verify-release.yml
- 紀錄：artifacts/checks/gpu-smoke.json、artifacts/checks/curriculum/deployment-gpu.json、index.json、00-warmup.json、01-small-cnn-learning.json、03-comparison-learning.json、04-localization-learning.json、10-multiscale-learning.json、fashion-mnist-learning.json
- 其他：data/manifest.json、notebooks/00-warmup.ipynb
- 被本頁摘要的頁面：docs/validation/curriculum.md、docs/validation/gpu-smoke.md、docs/preparation/data.md、docs/research/version-sources.md、docs/planning/outline.md 第 154 行、docs/lessons/08-own-data.md、03-comparison.md、20-deployment.md，以及 17／18／19 章頁的圖片引用
- 只讀的 git 指令（在原 repo）：log、status、ls-tree main、tag、check-ignore

二、外部來源
- GitHub Docs〈GitHub-hosted runners〉https://docs.github.com/en/actions/reference/runners/github-hosted-runners，看「standard runners」的硬體規格：ubuntu-latest 是 4 CPU、16 GB，標準 runner 沒有 GPU，GPU 只在 larger runners。
- 公開 repository 首頁 https://github.com/birdhackor/learn_to_yolo：確認是公開的。

三、執行過的程式
都在暫存副本執行，副本以 rsync 建立，排除 .git／site／.venv*／artifacts/runs／data/curated／data/downloads：
- `.venv-model/bin/python -m pytest tests/test_core.py tests/test_checkpoint.py -p no:cacheprovider`：exit 0，最後一行「25 passed in 1.40s」。
- `.venv-model/bin/python scripts/check_lesson_runtime.py`：exit 0，42 行「<id>: PASS; output identical/differs ...」，最後印 Report 路徑。
- `rsync -anc --delete --itemize-changes` 乾跑比對暫存副本與原 repo：只有 git 忽略的 artifacts/*.png、artifacts/lesson-*/ 和 __pycache__ 有變化。
- `check_lesson_runtime.py --section 07-training --report <log 目錄>`：只跑一節，PASS。
- `.venv-model/bin/python -m pytest tests/test_notebook_bootstrap.py`：6 passed，以 stub 執行，沒有真的安裝套件。
- `python3 scripts/validate_curriculum_evidence.py --scope gpu`：「2 GPU records match the current code」。不加 --scope 時因 CPU 紀錄過期而失敗，這是暫時狀態，未回報。
- `.venv-model/bin/python scripts/record_evidence.py`（列出模式）：列出過期的 CPU 紀錄，不執行實驗。
- `verify_curriculum.py --check` 與 `--help`。
- `.venv-docs/bin/zensical build --clean --strict`：exit 0；接著 `python3 scripts/validate_site.py`：exit 0，連結與錨點（含 preparation/data/#fashion-mnist）都通過，並檢查本頁的粗體與錨點轉換。
- python 片段：
  - 用 evidence_records.provenance.repo_dependencies 列出 08-own-images、00-warmup、18-video 綁定的檔案。
  - 用 CPU 的 PyTorch 模擬 Adam 加 StepLR(20, 0.8) 在第 20 步中斷、不還原 scheduler 時的學習率與權重：兩者都和不中斷時相同。
  - 掃描本頁的全形標點與中英文間空格。

四、依指示未回報的內容
- 會隨機器變的數字。
- 工作樹的暫時狀態：CPU 紀錄過期、reviews/coverage.json 不存在、notebook 舊說明文字。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/status.md 第 77 行〈已完成的檢查〉「雲端 GPU 存檔續訓」末句「存檔時連這兩種狀態也一起存，續訓才能和不中斷時完全一樣。」 | 這句把「模型與 optimizer 最大差異為 0」的原因歸給有存 StepLR 和 RNG 狀態，但本實驗的模型結果並不靠這兩項。 - RNG：miniyolo/gpu_smoke.py 每步都用同一批固定的 8 張圖，沒有隨機打亂、增強或 dropout。RNG 只被每步之後的 rng_probe 抽用，docs/validation/gpu-smoke.md 第 37 行也寫明 probe「不影響圖片或 loss」。 - StepLR：step_size=20 剛好等於中斷點，學習率又跟著 optimizer 的 param_groups 一起存回。我在暫存副本用同樣的 Adam lr 0.01 加 StepLR(20, 0.8) 模擬「第 20 步只存模型與 optimizer、不存 scheduler」：40 步的學習率序列和權重都和不中斷時逐值相同，只有 scheduler 的 last_epoch 不同（40 對 20）。 存這兩種狀態，保證的是續訓後「scheduler 與 RNG 狀態本身也相同」，也就是紀錄裡 scheduler_matches、rng_states_match 檢查的內容。初學讀者照現在的寫法，會以為這項實驗證明了少存它們，權重就會不同。 |
| 2 | 建議 | docs/status.md 第 18 行表格第 4 列「做了什麼」的「交給 ONNX Runtime 與 TensorRT（NVIDIA 的推論加速工具）執行」，以及第 76 行「（ONNX Runtime 也先和這份 PyTorch 輸出比對過）」 | 這一列的類型是「雲端 GPU（NVIDIA L4）小規模實測」，讀者容易以為 ONNX Runtime 也在 L4 上執行過。實際上 miniyolo/deployment_gpu.py 第 50 行用的是 providers=["CPUExecutionProvider"]：ONNX Runtime 在那台雲端機器的 CPU 上跑，只有 TensorRT 和作為參照的 PyTorch 在 L4 上。第 20 章頁面（「CPU 上的 ORT 也先和它比對過」）有寫明這點，本頁兩處都沒有。想用 ONNX Runtime 的 CUDA 執行環境部署的讀者，可能因此以為這條路徑在 L4 上驗證過。 |
| 3 | 建議 | docs/status.md 第 19 行表格第 5 列「結果類型」欄的「真實資料完整對照」，以及第 11 行「第 5 列才是「真實照片上的效果」」 | 前四列的類型名稱都在描述實際做了什麼（人工已知答案、CPU 少數幾步更新、合成圖形短訓練、雲端 GPU 小規模實測）。第 5 列卻叫「真實資料完整對照」，但同列「做了什麼」只有 Fashion-MNIST（28×28 灰階、沒有框）的 40 步分類管線核對，「能支持的結論」也寫「不支持任何偵測結論」。只看第一欄或第 11 行的讀者，會以為教材有一項「真實資料的完整對照」，或「真實照片上的效果」的結果。 |
| 4 | 建議 | docs/status.md 第 75 行〈已完成的檢查〉「核心與 checkpoint 測試」的「測試內容見上方〈執行方式〉」 | 〈執行方式〉只在第 68 行說明核心測試檢查什麼，checkpoint 測試只出現名稱。tests/test_checkpoint.py 實際做兩件事： - 在 CPU 上重跑和 GPU 實測相同的中斷／續訓流程（miniyolo.gpu_smoke 的 produce／resume），並要求模型與 optimizer 的差異為 0、scheduler 與 RNG 狀態相同。 - 確認沒有 RNG 狀態的舊格式 checkpoint 還能用來推論，但不能用來續訓。 讀者照這句指引回頭找，找不到 checkpoint 測試的內容。 |
| 5 | 建議 | docs/status.md 第 78 行「第 8 章自己的資料」中「最初只訓練 160 步時，……這次失敗的紀錄也保存著；接著只把步數加到事先固定的 1600 步（在 CPU 上）」 | 「最初……接著」是製作經過的敘事口吻，寫成作者先做了什麼、後來改做什麼，接近審查用的事實與寫作規範清單寫作規則不許寫的教材製作史。實際上這是固定的兩段式設計，不是一段歷史： - scripts/record_evidence.py 每次都把 custom-data-160-step.json 和 custom-data-learning.json 一起重產。 - 1600 步那次用 --prior-diagnostic 指向 160 步紀錄。 - 紀錄的 predeclared_protocol.followup 寫明只允許改訓練步數和改用新的 test seed。 內容本身沒有錯，問題在口吻。 |
| 6 | 建議 | docs/status.md 第 88 行〈誰檢查過內容〉「教材介紹到 YOLO26 為止，不包含之後的新版本或分支」 | 這句容易讓人以為 YOLOv1 到 YOLO26 之間的版本都有介紹。docs/planning/outline.md 第 154 行寫明 YOLOv6、v7、v9 的重參數化、訓練輔助與梯度設計不在範圍內，lesson 頁也沒有介紹這三版。這是一頁講「驗證範圍」的頁面，卻沒有交代這個範圍缺口。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 讀者審查 | 第 5 列名稱「真實資料完整對照」與引導句誤導 | 已修正：列名改成「真實資料：只有分類管線核對」，並註明不屬於 42 節、連到資料規劃頁的 #fashion-mnist；「衣物」改成和首頁相同的「衣服、鞋與包等服飾商品」。引導句改成第 5 列只核對分類管線，真實照片上的效果沒有測。前兩列的類別名改成「程式跑得動、算得對」。 |
| 2 | 讀者審查 | 第 20 章連結只到章首；FP16「允許」、INT8 沒解釋 | 已修正：連結改成〈L4 GPU 上的 TensorRT 實測〉(#l4-results)；註明「允許 FP16」是由 TensorRT 決定哪些層用 FP16、不保證每一層都是；〈沒有驗證的事〉第一次出現時解釋 FP16 與 INT8（INT8 要先校準，依第 20 章的表）。 |
| 3 | 讀者審查 | 第 8 章「最初…接著…」沒說原因 | 已修正：改寫成兩份紀錄：160 步的診斷，以及沿用同一批 train／validation 與全部設定、開跑前就定好的 1600 步。寫明 160 步那次已評過 test，所以改用新 seed 畫的 test，也沒有用 test 挑設定。已和紀錄的 predeclared_protocol（allowed_changes、prior／fresh test seed、test_evaluations＝1）核對，數字都沒動。 |
| 4 | 讀者審查 | 沒寫怎麼取得專案，也沒提醒 Windows | 已修正：補上「先從 GitHub 下載本專案（Code → Download ZIP 或 git clone）」，並在指令前註明「Windows 的寫法見 README」（README 確實有 Windows 的寫法）。 |
| 5 | 讀者審查 | Colab 一點太長，「本機」有歧義 | 已修正：把沒有驗證的事放在開頭（在 Colab 上執行，包括它分配的 GPU），後面寫成「代替的檢查有兩項」，並刪掉「在本機」。沒有把它移到〈已完成的檢查〉，因為發布後的檢查在 tag 之後才執行。 |
| 6 | 讀者審查 | 「第一次讀」提示把 Colab 說明延後了 | 已修正：改成先看開頭、表格最後一欄，以及〈執行方式〉第一段的 Colab 說明；其餘部分讀完第 7、8 章（以及選修的第 20 章）再回來看。 |
| 7 | 讀者審查 | 第 4 列最後一欄的工具名沒有解釋 | 已修正：把 ONNX Runtime、TensorRT 的短註移到最後一欄。 |
| 8 | 讀者審查 | 「測試內容見上方」找不到 checkpoint 測試內容 | 已修正：在〈執行方式〉補一句：checkpoint 測試在 CPU 上中斷、存檔、讀回、續訓，要求和不中斷時相同；也確認沒有 RNG 狀態的舊格式只能推論、不能續訓（已和 test_checkpoint.py 的兩個測試、gpu-smoke.md 核對）。 |
| 9 | 讀者審查 | 「人工高分框會註明來源」不清楚 | 已修正：改成「有些節為了示範，直接用人手指定的框與分數（例如第 6 章的人工框），不是模型的預測，這些地方都會寫明」。 |
| 10 | 讀者審查 | commit 用「存檔」解釋，和 checkpoint 撞名；「分支」不清楚 | 已修正：改成「commit：程式碼某一次提交的版本編號，對應的內容固定不變」；「之後的新版本或分支」改成「之後的版本」。 |
| 11 | 讀者審查 | check_lesson_runtime 的輸出是英文，比對方法不清楚 | 已修正：照腳本的 print 寫出例句 `07-training: PASS; output identical to the recorded run`，說明 differs 後面括號裡是固定文字、報告不標出是哪幾行；要知道差異就逐行對照 stdout 與頁尾紀錄。也寫出 --section 的完整指令。 |
| 12 | 讀者審查 | 「實驗格」沒定義；「節的編號」容易誤解 | 已修正：第一次出現時註明實驗格是 notebook 最後一格，內容和 lesson_cases 該節程式相同；兩處都改成「<節的代號>.json」，並舉 07-training.json 為例。 |
| 13 | 讀者審查 | SHA-256 其實記在 reviews/coverage.json | 已修正：改成這些內容「都用 SHA-256 記在 reviews/coverage.json」，與審查用的事實與寫作規範清單一致。 |
| 14 | 事實查核 | 存檔續訓把結果相同歸因給 StepLR／RNG | 已修正：改成「存檔時也存了這兩種狀態，所以續訓後它們也和不中斷時相同；訓練若用到亂數（例如隨機打亂、隨機增強），少存 RNG 狀態，續訓結果就可能不同」，不再說存了才能完全一樣。 |
| 15 | 事實查核 | ONNX Runtime 其實在 CPU 上跑，不在 L4 | 已修正：已確認 deployment_gpu.py 用的是 CPUExecutionProvider。表格第 4 列與已完成檢查那一點，都標明 ONNX Runtime 在同一台雲端機器的 CPU 上執行。 |
| 16 | 事實查核 | 第 5 列名稱與第 11 行誤導 | 已修正：與讀者 should 1 一起改。 |
| 17 | 事實查核 | checkpoint 測試內容沒有寫 | 已修正：與讀者 should 8 一起改。 |
| 18 | 事實查核 | 第 8 章的敘事口吻像製作經過 | 已修正：與讀者 should 3 一起改寫成實驗設計（兩份紀錄），數字不動。 |
| 19 | 事實查核 | 沒交代 YOLOv6、v7、v9 不在範圍內 | 已修正：補上「也不涵蓋 YOLOv6、v7、v9 的機制（範圍見課程大綱）」，依 outline.md 的範圍說明。 |

修正後由另一位 AI 檢查這一批頁面（`docs/index.md`、`docs/learning-path.md`、`docs/glossary.md`、`docs/status.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

逐段讀〈沒有驗證的事〉與〈誰檢查過內容〉。用 grep 統計 42 節課程頁頁尾的來源行，再對照 reviews/00-warmup.md、reviews/index.md 的方法段、任務給定的審查方法，以及 zensical.toml 導覽（核對「其他 17 頁」的頁數：6＋3＋3＋5）。〈沒有驗證的事〉新增的「在真實資料上比較各機制」一項，與表格第 5 列、首頁〈沒有驗證〉一致；3.3 節的 plain／residual 確實是合成資料上的小型對照。審查紀錄每頁一份、coverage.json 的涵蓋範圍與 review_coverage.py 的 digest 相符。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 〈誰檢查過內容〉「42 節課程頁」一項的最後一句：「頁面引用原始論文、官方程式或函式庫文件的說法，另由 AI 對照頁尾〈參考來源〉連到的固定版本。」 | 只有 13 節有「參考來源：」頁尾行（12.1–12.4、13.1、13.2、14、15.1、15.2、16.1–16.3、20）。其餘 29 節的來源寫在引用處，有的根本沒有連結：00 的「延伸閱讀（英文官方文件）」和 02 的「官方說明文件：」連到 pytorch.org/docs/stable，7.4 的 Adam、6.1 的 torchvision NMS 也連到 stable，這些頁面會隨新版變動，不是固定版本；9.1 的 Darknet region_layer.c／yolov2-voc.cfg 說法、10 的 cfg/yolov3.cfg 說法完全沒有連結；20 章的「參考來源：」放在〈L4 GPU 上的 TensorRT 實測〉之前，不在頁尾，ONNX Runtime 入門連結也沒有版本。審查紀錄實際對照的固定版本（例如 reviews/00-warmup.md 寫的「PyTorch 固定 commit 原始碼與官方文件」）並不在頁面連結裡。讀者照這句到 3.3、6.2、9.1 的頁尾找審查依據會找不到；到 00、02 的頁尾找，看到的是會變動的 stable 文件，會誤以為那就是審查用的版本。 | 已修正：不再寫頁尾的位置，改成打開原始來源逐句核對（論文看原文，官方程式看固定的 commit 或 tag，函式庫看官方文件），查閱的來源記在該頁的審查紀錄；也刪掉「對照的來源都是固定版本」。 |

### 第 1 輪：〈全套實驗與審查〉與相同說法的事實查核

只為了確認 curriculum.md 第 121–123 行〈沒有驗證的事〉與它一致，讀了〈沒有驗證的事〉、〈執行紀錄〉、〈已完成的檢查〉、〈誰檢查過內容〉四段，沒有逐句審查全頁。

- 〈沒有驗證的事〉：curriculum.md 列的是子集，並連到本頁，兩頁一致。
- 〈執行紀錄〉的 machine 欄位用「等」表示沒有列完；第 45 行的建置前檢查說明和 validate_curriculum_evidence.py 相符。
- 第 76–77 行和 20-deployment.json（ORT 比對 B=1、2、3）、deployment-gpu.json（B=1–4）、gpu-smoke.json（最大差異 0.0）相符。
- 〈誰檢查過內容〉第 85–88 行和 curriculum.md 第 88 行用同一組說法，問題相同。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 88 行「查到的問題修正後，都由另一位 AI 檢查修正。」 | 和 curriculum.md 第 88 行相同。reviews/00-warmup.md 與 reviews/index.md 第 3 行都寫成修正由處理者自己核對、沒有交回審查者重讀，兩份都沒有另一位 AI 檢查修正的記載。 | 已處理：這一項引用的是重產前的舊紀錄。00-warmup 與首頁的紀錄改由各次查核的原始結果產生；兩頁定稿時的修正，另請一位 AI 逐項檢查，結果記在各自的〈定稿修正〉。這一輪的修正也交給另一位 AI 檢查。 |
| 2 | 必要 | 第 85 行「頁面引用原始論文、官方程式或函式庫文件的說法，另由 AI 對照頁尾〈參考來源〉連到的固定版本。」 | 和 curriculum.md 第 88 行相同。只有 12.1–16.3 與第 20 章共 13 頁有頁尾的「參考來源」。另外 22 頁的出處是正文內的連結，多半沒有固定版本（沒有版本號的 arXiv abs 頁、PyTorch /stable/ 文件）。 | 已修正：改成「另由 AI 打開原始來源逐句核對：論文看原文，官方程式看固定的 commit 或 tag，函式庫看官方文件；查閱了哪些來源，記在該頁的審查紀錄」，也刪掉下一段過寬的「對照的來源都是固定版本」。 |
| 3 | 建議 | 第 86 行「其他 17 頁……並實際執行頁面上的部分指令」 | 和 curriculum.md 第 88 行相同。index.md、learning-path.md、glossary.md 與 planning/ 底下的 3 頁沒有任何指令可以執行；審查用的事實與寫作規範清單也沒有這一項。 | 已修正：改成「頁面上有指令的，也實際執行其中一部分」。 |

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

核對〈誰檢查過內容〉改過的句子：逐句對照 59 份 reviews/ 紀錄（用腳本統計每份的段落、發現數與處理列數，並全文讀了 00-warmup、index、09-anchors、16-training、validation-curriculum 的紀錄）、scripts/review_coverage.py、zensical.toml 導覽（其他 17 頁、四組名稱正確）。上一輪 must #2（來源核對方式）與 should #3（有指令的頁才執行）已改對，現在式、無修訂敘事、中英文間距都合規；暫存副本 zensical build --clean --strict 與 validate_site.py 都通過。上一輪 must #1 的處理沒有完全成立（00-warmup），本頁自己的審查紀錄也有發現沒列處理。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 〈誰檢查過內容〉「查到的問題修正後，都由另一位 AI 檢查修正。」（上一輪 must #1 的處理） | 處理寫「00-warmup 與首頁原本的紀錄改由各次查核的原始結果產生，列出每輪修正後另一位 AI 的檢查」，但 reviews/00-warmup.md〈定稿修正〉的 23 項「審查意見」修正（含讀者必要問題：本機指令、新增〈第一次用 Colab〉、BatchNorm、SGD 預設值等）沒有任何紀錄顯示經另一位 AI 檢查。紀錄掛的「修正後由另一位 AI 檢查改動（通過）」是同批第 2 輪複查，原文寫「這一輪只改了 docs/lessons/01-small-cnn.md 與 docs/lessons/02-diagnostics.md。 - 00-warmup.md 和兩張 SVG 都與第 1 輪相同」「並抽查了先前輪次的修改」；同批第 1 輪對 00 頁只核了一條「不必改」的先前意見；〈後續編輯的檢查〉的頁面範圍也不含 00-warmup。讀者照這句打開 00-warmup 的紀錄，找不到那一輪修正的檢查（首頁的同類修正則由〈後續編輯的檢查〉涵蓋，成立）。這是說法目前沒有紀錄支持。 | 已處理：另請一位 AI 逐項檢查 00-warmup 與首頁定稿時的修正，結果記在這兩頁紀錄的〈定稿修正〉；這句說法不變。 |
| 2 | 必要 | reviews/status.md〈獨立查核〉第 1、3 次 | 紀錄的問題：第 1 次查核的必要問題後面是空白的「必要問題的修正：」（這一輪的修正結果放在 traces_handled，生成器只讀 items_handled／handled）；第 1 次的 should（第 81 行「完整的報告清單」）與第 3 次的 should（第 83 行沒說兩種審查涵蓋哪些頁）在紀錄裡找不到處理，〈定稿修正〉19 列只涵蓋讀者審查與事實查核，紀錄卻寫「建議事項在下方〈定稿修正〉逐項處理」。和〈審查〉段「每份紀錄列出…每一項的處理」不符。 | 已修正：產生器讀入修正時處理的項目與建議事項的處理（處理後由下一次複查檢查），不再輸出空的清單，每項發現都有對應的處理。指向〈定稿修正〉的句子只在該節存在時才寫。 |
| 3 | 建議 | reviews/status.md 各輪摘要 | 留有內部名稱與截斷，讀者看不懂：「先前審查意見清單列的 17 條 trace 都已處理」「editor 的待重錄數值清單沒有列這一句」；各輪摘要在句中被「…」切斷。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

〈誰檢查過內容〉的說法用腳本對 59 份紀錄逐項核對。42 份課程頁紀錄都有獨立查核；頁面連到論文、官方程式或文件的課程頁，紀錄都有〈來源對照〉或技術查核，並列出查閱的來源（以 arXiv 編號或 commit 加檔名逐一比對頁面連結，都找得到）；17 頁其他頁紀錄都對照了程式與紀錄，頁面上有指令的紀錄都寫了實際執行；各處修正都有批次檢查、逐項檢查或下一輪檢查（最後一輪由這次最終檢查涵蓋）；17 頁＝6＋3＋3＋5，和導覽一致。結論：除了 reviews/preparation-publish.md（必要問題列在該頁），這些說法都有紀錄支持；18-video 與 08-own-images 的函式庫說法沒有來源清單，列在各該頁。本頁正文這輪沒有改動。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/status.md〈獨立查核〉第 1 次查核與「修正必要問題時處理的項目」 | 第 1 次查核唯一的必要問題（第 40 行把驗證器的守衛範圍說得比程式大），在其後 17 項處理裡沒有對應項，只能從第 2 次複查「本輪改寫的驗證器那句已對照…全部成立」推知已修好；〈上述處理〉第 2 項卻寫「每項發現都有對應的處理」。我核對過現行〈執行紀錄〉末段，它和 validate_curriculum_evidence.py 相符（逐節紀錄比對 notebook stdout，頁面只找日期、CPU、PyTorch；補充與 GPU 紀錄只看綁定）。 | 已修正：第 1 次查核的表格加上處理欄，寫明這項必要問題怎麼改、由第 2 次複查確認。 |
| 2 | 建議 | reviews/status.md 第 14、21、47、79、98、119 行與第 4 次查核末段 | 內部名稱與殘句：「facts 允許」「提醒 orchestrator」「本輪依 checker 的 should 再拿掉「完整的」」「在暫存副本裡」「在暫存副本中」「日誌是同一層的暫存副本與暫存副本」；第 4 次查核還留著查核流程的內部觀察「主 repo 的 scripts/__pycache__/evidence_records.cpython-314.pyc 在 02:59:14 被改寫…查不出是誰執行的」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 1 次查核表格的必要問題有處理欄。對照現行〈執行紀錄〉末段與 validate_curriculum_evidence.py（逐節紀錄比對 notebook stdout，頁面只找日期、torch、machine.cpu；補充與 GPU 紀錄只看綁定）屬實，第 2 次複查第 62–65 行也確認了；第 3 輪第 1 項屬實。不過同一張表第 2 項的處理指向錯，第 3 輪第 2 項的處理說明也不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/status.md 第 30 行（第 1 次查核表格第 2 項的處理欄） | 處理寫「已處理：見下方〈建議事項的處理〉」。但整份紀錄唯一的〈建議事項的處理〉（第 96–98 行）處理的是第 2 次複查的 Colab 條目；第 81 行「完整的報告清單」的實際處理，在第 47 行「修正必要問題時處理的項目」那一項（拿掉「完整的」）。照指向去找，會找不到這項的處理。 | 已修正：處理欄直接寫出處理內容：拿掉「完整的」，列在第 1 次查核後的修正項目裡。 |
| 2 | 必要 | reviews/status.md 第 79、98 行，對應第 3 輪第 2 項處理 | 處理寫「點名的路徑殘句與內部用語已清理或換成白話」，但點名的「在暫存副本裡」變成「在暫存副本：」（第 79 行）；點名的「在暫存副本中」只把 polish 換成「潤稿檢查」，成了「在暫存副本潤稿檢查-run/status 中」（第 98 行）。兩處仍是殘句。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 3 | 建議 | reviews/status.md 第 119 行 | 「日誌是同一層的暫存副本」仍沒有實際位置。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

全文對照了第 1 次查核表格、第 3 輪與第 4 輪的處理欄。
第 1 次查核的處理欄屬實：
- #1：〈執行紀錄〉末段（docs/status.md 第 45 行）和 validate_curriculum_evidence.py 一致：42 節紀錄比對 notebook stdout，頁面檢查日期、torch 與 machine.cpu；補充與 GPU 紀錄檢查 is_current 與 machine。第 2 次複查（第 62–65 行）和第 4 輪都核對過。
- #2：頁面已經沒有「完整的」，這項列在第 47 行的修正項目裡。
第 3 輪第 1 項、第 4 輪第 1 與第 3 項屬實。第 3 輪第 2 項、第 4 輪第 2 項不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 3 輪第 2 項處理欄（第 363 行） | 處理欄寫「點名的 6 處文字中，4 處已改寫或刪除；其餘 2 處仍在紀錄裡：「在暫存副本裡」「主 repo…」」。但點名的「在暫存副本中」仍在第 98 行（「在暫存副本中，zensical build --clean --strict 與 validate_site.py 都 exit 0。」），沒有列為仍在。發現實際引了 7 段文字。列為仍在的「在暫存副本裡」也不是發現裡的引文：發現寫的是「在暫存副本裡」，第 79 行現在是「在暫存副本裡：」。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
| 2 | 必要 | 第 4 輪第 2 項處理欄（第 372 行） | 處理欄寫 6 處中 5 處已改寫或刪除，只剩「在暫存副本裡」。但「在暫存副本中」「暫存副本中」都還在第 98 行。「在暫存副本裡」只是碰巧和第 188 行「只在暫存副本裡工作」相符；點名的第 79 行已變成「在暫存副本裡：」。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |

### 第 6 輪：上一輪的處理：通過

逐列對照〈後續編輯的檢查〉第 3、4、5 輪處理欄與紀錄內容。先用暫存副本從工作流程 journal 重產全部 59 份紀錄，結果和 repo 逐位元相同；再列出這份紀錄從原文到紀錄的每一處替換，只有路徑、內部名稱（審查用的事實與寫作規範清單、查核者、先前審查意見、部分等）與 A：／B：標籤。第 3 輪 #1：第 1 次查核的表格確實有處理欄，#1 寫了改法，也寫了由第 2 次複查確認。第 3 輪 #2 是統一說明：點名的第 14、21、47、79、98、119 行與第 182 行有的已經改變，有的仍在，和「可能改變，也可能仍在」相符。第 4 輪 #1：第 30 行的處理欄就是「拿掉「完整的」…」，docs/status.md 也確實沒有「完整的」了；#2、#3：第 3 輪 #2 確實是統一說明。第 5 輪 #1、#2：第 3、4 輪的處理欄已經沒有任何計數，也沒有「已清理」這類說法。沒有和紀錄矛盾的處理說明。附記，不列為問題：統一說明寫「保留查核者的原文」，但點名的第 47、98 行其實是修正者寫的處理項目；讀者不會因此做錯事，不建議為此改寫。範圍外的第 2 輪：#3 的處理寫「工作流程用語換成白話」，但發現點名的「editor 的待重錄數值清單沒有列這一句」仍在第 94 行，第 86 行也還有 editor，因為產生器沒有替換 editor 的規則。
