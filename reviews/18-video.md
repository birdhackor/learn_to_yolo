# 審查紀錄：影片串流

審查範圍：`docs/lessons/18-video.md`、頁面上的圖（`docs/assets/diagrams/18-video.svg`），以及 `lesson_cases/18-video.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要等級的問題，只有兩條 should（讀者友善度）。

**1. 對程式的敘述都符合現行程式（sha256 0aa6c171…）**
- 我在暫存副本跑完整程式，exit 0、stderr 為空。
- 以下敘述都和 `main()` 的斷言、輸出相符：暖機只跑 1 幀而且有框；正式結果 12 筆、每張 96×64；index 依序是 0～11；最後時間戳 0.55 秒；`warmup_frames`=1；`median_ms` 涵蓋全部 12 幀，12 是偶數，所以取第 6、7 小兩個值的平均。
- 其他敘述也都對得上程式：
  - `decode_grid` 的預設門檻是 .25，比較用 `>=`，所以「低於 0.1 就丟掉」正確；只有在有候選框時才會呼叫 NMS，所以「第 5、11 幀沒有框，就不跑 NMS」成立。
  - letterbox 實際算出 43 列、上補 10 列、下補 11 列；raw 的 shape 是 [1,4,4,7]；紅色是類別 0。
  - `verify_video_file.py` 預設寫到 `artifacts/runs/video-file/result.json`，加 `--record` 才寫正式紀錄。`artifacts/lesson-*/` 和 `artifacts/runs/` 都被 git 忽略，所以刪掉三處「會覆寫／git checkout 還原」的提醒是對的。
- 練習 1、2 照頁面指示改最後一行後實跑：
  - `main(count=24)`：靜態圖是第 0、11、23 幀；最後時間戳 1.15 秒、總播放 1.2 秒；第 23 幀 `object_visible_per_frame` 是 false。
  - `main(fps=10)`：1.1 秒、1.2 秒、GIF 每幀 100 毫秒，框數和 fps=20 相同。
  - 兩題的參考答案都正確。另外確認 Pillow 會把 33 毫秒存成 30 毫秒。

**2. 先前審查意見與受程式改動影響的段落全部處理到**
- 每一條都處理了。先前審查意見 L144 建議讓程式算 IoU，但最終程式沒有這麼做，所以頁面保留四個 IoU 值並列為紀錄相依。我用 video-file.json 重算得到 0.504／0.473／0.304／0.461，和頁面一致。
- 舊的補丁說明（中位數不計第 0 幀、3.94／0.73 毫秒、英文圖標翻譯、git checkout 還原）都已刪除。全頁沒有修訂或審查經過的敘述；頁中的「本次／這次」都是指紀錄的那次執行。
- 頁尾執行紀錄區塊和 HEAD 逐位元組相同；Colab 連結已經是 lessons-v0.4.0。

**3. 數字**
- 新加的數字都是程式決定的固定值（1 幀、12 幀、第 6、7 小、0.25、160 步），沒有用到這台 Mac 的數字。
- 會隨重錄改變的值，編者都列進待重錄數值清單：表格五個值、0.796 的加總與「少」的方向、約每秒 1127 幀、框數與「8 幀」、四個 IoU、跨格線的幀、面板的中文標籤、`--record` 那一句。
- 本機第 7 幀沒有框（已知的機器差異）。重錄後要照新紀錄複查第 158、160、168 行。

**4. 程式摘錄**
- 摘錄比對工具印出 []。
- Frame、`main()` 暖機那幾行、opencv_frames 三段都已標成摘錄，順序也和程式相同。
- 突變測試：我在程式裡分別改壞暖機行、中位數行、`timestamp_s` 的型別、`cvtColor` 那一行，四處都被摘錄比對工具擋下。
- `run_stream` 是簡化片段，正文已寫明，列出的差異也完整（說明文字、斷言、計時、畫框、`tensor_rgb` 改名、dict 只留三欄）。
- 正文沒有引用程式行號。

**5. 網站建置與圖**
- `zensical build --clean --strict` 與 `validate_site.py` 都 exit 0。8 個程式區塊都正常上色，三個摘錄區塊帶 data-excerpt。
- `validate_lessons` 會跑到最後的審查涵蓋檢查才失敗：每一頁都是 no review，這是發布前要補的事。在它之前，含全部摘錄的檢查與 notebook 一致性都通過。
- 新產生的 panel.svg 用 qlmanage 渲染正常：中文標籤、viewBox、title、desc 都在，內容和第 154 行的圖說一致。
- 網站圖 `docs/assets/diagrams/18-video.svg` 目前還是舊的英文版。`verify_curriculum.py` 重錄時，會在寫紀錄的同一次執行裡用 SITE_FIGURES 把 panel.svg 複製過來。依審查用的事實與寫作規範清單這屬於工作樹的暫時狀態，不算這一頁的問題。
- 和 HEAD 相比，lesson 18 只有這一頁有改動；程式與 notebook 已在 HEAD。

**附帶**：編者疑慮說 18-metrics.json 還留著，其實它已經在 index 中標記刪除了，對這一頁沒有影響。

**檢查用的檔案**
- 紀錄（run.out、ex1.out、ex2.out、build.log、validate-site.log、validate-lessons.log）
- 練習複本
- 突變測試
- 渲染圖：暫存副本（panel-default.svg.png、site-18-video.svg.png）

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/18-video.md 第 98、102 行（「以及 report 的 `warmup_frames`」「也就是 report 裡的 `median_ms`」） | 「report」第一次出現在這裡，但沒說它是什麼、在哪裡看得到。`report.json` 要到第 170 行才出現，摘錄也沒有包含 `summary`。初學讀者讀到這兩句時，不知道 `warmup_frames`、`median_ms` 指的是哪一份輸出，也不知道頁尾〈實際執行紀錄〉印的就是它。 |
| 2 | 建議 | docs/lessons/18-video.md 第 162 行「`run_stream` 裡的註解寫明了理由」 | 讀者在本頁看得到的 `run_stream` 只有第 57–74 行的簡化片段，那段的中文註解只寫「score 門檻 0.1、NMS 的 IoU 門檻 0.5」，沒有這個理由。讀者往上找會找不到，以為頁面說錯。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 「report」第一次出現時沒說明它是什麼 | 已修正：確認 `main()` 印出的 summary 和 report.json 逐字相同（實跑後已用 cmp 比對）。在第一次提到 `warmup_frames` 的句子後補一句：report 是 `main()` 最後印出、也存成 `artifacts/lesson-18/report.json` 的摘要，頁尾〈實際執行紀錄〉印出的 JSON 就是它。 |
| 2 | 先前查核 | 「`run_stream` 裡的註解寫明了理由」，但頁面上的簡化片段沒有這段註解 | 已修正：確認門檻 0.1 的理由只寫在完整程式的英文註解裡。改成「完整程式 `run_stream` 裡的英文註解寫明了理由」。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/17-capstone.md`、`docs/lessons/18-video.md`、`docs/lessons/19-tracking.md`、`docs/lessons/20-deployment.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/18-video.md 第 98 行「report 是 `main()` 最後印出、也存成 `artifacts/lesson-18/report.json` 的摘要」 | `main()` 先在第 150 行印出 JSON，第 151 行才印最後一行 `GIF: …` 路徑；頁尾紀錄也一樣，JSON 後面還有一行 GIF。所以照字面讀，「最後印出」不精確。讀者靠後半句「印出的 JSON 就是它」仍找得到 report，不會因此做錯事，所以列為建議。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

lesson_cases/18-video.py 第 136–151 行把 summary 寫進 artifacts/lesson-18/report.json，並以 `print(json.dumps(summary, indent=2))` 印出；頁尾〈實際執行紀錄〉顯示的就是這段 JSON，其中 warmup_frames 是 len(warmup)。句子正確。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核 2 項在〈定稿修正〉處理；批次檢查留下的 1 項，由〈後續編輯的檢查〉核對；結構檢查通過。頁面沒有外部連結，紀錄沒有〈來源對照〉也沒有技術查核。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/18-video.md 第 84、174–179、249 行；reviews/18-video.md | 頁面寫了函式庫與格式行為：「OpenCV 讀出的顏色順序是 B、G、R」「cv2.VideoCapture…用完要呼叫 capture.release()」「opencv-python-headless…不含開視窗功能」「在 CPU 上，PyTorch 的運算回傳時就已經算完（同步執行）；GPU 則會先把工作排隊」「FFV1 是一種無損影片編碼」。紀錄沒有記下核對過哪份官方文件（只靠 video-file.json 的往返實測支持其中一部分）。判定理由同 08-own-images：這頁和 19-tracking 一樣是點名來源卻沒有連結，上一輪把那種情況判為建議。我讀過的內容和官方說法相符。另有路徑殘句：「- 暫存副本- 紀錄：同一層的暫存副本（run.out…」「- 練習複本：暫存副本、暫存副本」「- 突變測試：暫存副本」。 | 已處理：另做一次函式庫說法的來源對照，見〈補做的來源對照〉。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 補做的來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開原始來源逐句核對。檢查範圍：docs/lessons/18-video.md 正文和摺疊區裡所有關於函式庫與格式行為的敘述。略過 curriculum-evidence 產生區塊，以及描述本課自己程式的敘述。 對照依據是課程固定的版本： - OpenCV 4.13.0（opencv-python-headless 4.13.0.92）：docs.opencv.org 擋自動抓取，改讀 tag 4.13.0 的標頭檔文件註解和 videoio 原始碼。 - PyTorch：2.9 文件與 v2.9.1 原始碼。 - 其他：Python 3.12 文件、NumPy 2.3、Pillow 12.0.0、RFC 9043（FFV1）、GIF89a 規格、FFmpeg 7.1。 - 另在本機 .venv-model 做了幾個小實驗。 結果：沒有發現必要等級的錯誤。 確認正確的敘述： - OpenCV：預設輸出 BGR，需要用 cvtColor(COLOR_BGR2RGB) 轉換；read() 在檔尾或讀不到時回傳 False；get() 遇到不支援的屬性回傳 0（所以只在 FPS 大於 0 時採用是合理的）；isOpened() 的語意；相機編號 0；headless 版不含 GUI。 - Python：generator 被 close() 或被回收時會執行 finally；Colab 留著 traceback 時物件不會被回收；contextlib.closing 的行為；runpy.run_path 回傳 globals，而且 __name__ 是 '<run_path>'；dataclass 會產生 __init__。 - NumPy 與 PyTorch：np.median 在個數為偶數時取中間兩值的平均；from_numpy 和陣列共用記憶體（所以 .copy() 有道理）；PyTorch 在 CPU 上同步執行、在 GPU 上非同步。 - 格式：PTS 的定義；GIF 以 1/100 秒記錄時長，Pillow 用 int(ms/10) 截斷，所以 33 毫秒會存成 30 毫秒。 - perf_counter 是單調時鐘：Python 3.12 文件沒有明寫，3.13 文件以 CPython 實作細節載明；3.12 的 get_clock_info 也回報 monotonic=True，實務上成立。 - opencv-python-headless 4.13.0.92 要求 numpy>=2，和 numpy==2.3.5 相容。 7 項 should（措辭不精確或缺少必要的前提）： 1. 同一環境只能裝一種 OpenCV 套件，而 Colab 已經預裝三種，「需要額外安裝」在 Colab 不成立，照著安裝還會混裝。 2. @torch.no_grad() 套在 generator 上，和在函式本體寫 with 並不等價。 3. torch.set_num_threads 只限制 PyTorch 的 intra-op 執行緒，管不到 OpenCV 和 FFmpeg 的解碼執行緒。 4. 在 GPU 上計時，每次讀計時器之前都要 torch.cuda.synchronize()。 5. FFV1 能逐值相等，前提是寫入端沒有先轉成 YUV 4:2:0；本例成立，是因為 OpenCV 選了 BGR0。 6. VideoCapture 被解構時也會自動 release()，「一直被占住」要加上「被回收或程式結束前」的條件。 7. 在互動環境單寫 count_up()，會顯示 generator 物件本身，不是「什麼都不會印」。

查閱的來源：

- OpenCV 4.13.0 原始碼：https://github.com/opencv/opencv ，tag 4.13.0（commit fe38fc608f6acb8b68953438a62305d8318f4fcd），modules/videoio/include/opencv2/videoio.hpp（docs.opencv.org 4.13.0 的 VideoCapture／VideoWriter 文件就是從這裡產生；網站擋自動抓取，所以改讀原始碼）：第 147 行 CAP_PROP_FPS、第 159 行 CAP_PROP_CONVERT_RGB、第 213–214 行 CAP_PROP_N_THREADS／CAP_PROP_PTS、第 823 行相機編號 0、第 915–927 行 isOpened／release、第 980–988 行 read、第 1009–1013 行 get、第 1199 行 VideoWriter::write 預期 BGR
- 同一 commit，modules/videoio/src/cap_ffmpeg_impl.hpp：第 1021–1027 行解碼執行緒預設 min(CPU 數,16)、第 1882 行讀取預設輸出 BGR24、第 2186–2198 行 get_fps 取 avg_frame_rate、第 3333–3355 行 FFV1 對 BGR24 輸入選 BGR0
- 同一 commit，modules/videoio/src/cap_avfoundation_mac.mm（macOS 後端預設輸出 BGR）；modules/imgproc/include/opencv2/imgproc.hpp 第 3765 行（cvtColor：OpenCV 預設色彩順序其實是 BGR）
- opencv-python 打包 README：https://github.com/opencv/opencv-python ，tag 92（4.13.0.92，commit 4ddfc013fd1f13d9b9e379dbebf2cdbeb052e7f8），README.md 第 35 行（四種套件只能擇一、共用 cv2）、第 42–47 行（headless 不含 GUI 功能）、第 203 行（wheel 內附 LGPL FFmpeg）
- PyPI opencv-python-headless 4.13.0.92 的 requires_dist（Python 3.9 以上需要 numpy>=2，和 numpy==2.3.5 相容）
- Colab 預裝套件清單：https://github.com/googlecolab/backend-info ，commit 92a2364a31cd9b686f88075e52d0ff5e5c098a40，pip-freeze.txt（opencv-python 5.0.0.93、opencv-python-headless 5.0.0.93、opencv-contrib-python 4.14.0.94；README 註明可能比線上 runtime 晚一兩天）
- PyTorch 2.9 文件（docs.pytorch.org/docs/2.9）：notes/cuda.html〈Asynchronous execution〉、generated/torch.no_grad.html、generated/torch.set_num_threads.html、generated/torch.from_numpy.html、generated/torch.cuda.synchronize.html
- PyTorch 原始碼：https://github.com/pytorch/pytorch ，tag v2.9.1（commit d38164a545b4a4e4e0cf73ce67173f70574890b6），torch/utils/_contextlib.py 第 19–70 行 _wrap_generator、第 72–124 行 context_decorator
- Python 3.12 文件（docs.python.org/3.12）：library/time.html（perf_counter、monotonic、sleep）、reference/expressions.html（generator.close() 與被回收時呼叫 close）、library/runpy.html（run_path）、library/contextlib.html（closing）、library/dataclasses.html、library/sys.html（displayhook）；另參照 Python 3.13 文件 library/time.html（perf_counter 的 CPython 實作細節：和 monotonic 用同一個時鐘）
- NumPy 2.3 文件：reference/generated/numpy.median.html（個數為偶數時取中間兩值的平均）
- Pillow 原始碼：https://github.com/python-pillow/Pillow ，tag 12.0.0（commit 693df7b42c666f88c719f9973be0ad71607328e0），src/PIL/GifImagePlugin.py 第 833 行 duration = int(duration / 10)；本機 site-packages 裡的檔案和這個 tag 相同
- GIF89a 規格：https://www.w3.org/Graphics/GIF/spec-gif89a.txt ，Graphic Control Extension 的 Delay Time 以 1/100 秒為單位
- RFC 9043（FFV1 Video Coding Format Versions 0, 1, and 3）：摘要（無損、支援多種像素格式）與 §3.7.2 RGB（JPEG 2000 RCT 可逆轉換）
- FFmpeg 原始碼：https://github.com/FFmpeg/FFmpeg ，tag n7.1（commit b08d7969c550a804a59511c7b83f2dd8cc0499b8），libavutil/frame.h 第 499 行 AVFrame.pts 定義（libavutil 59.39，和 opencv-python-headless 4.13.0.92 內建的版本相同）
- 本機實測：.venv-model（Python 3.12.15、torch 2.9.1、opencv-python-headless 4.13.0、Pillow 12.0.0、macOS 8 核）：no_grad 裝飾器和 with 在 generator 上的差異、set_num_threads 之後 cv2／FFmpeg 的執行緒數、沒 release 直接 del 後檔案控制代碼關閉、FFV1 預設寫法和 yuv420p 重新編碼的畫素差、Pillow 把 GIF 的 33 毫秒存成 30 毫秒、互動模式單寫 count_up() 的輸出

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 174 行〈接真影片的同一個入口〉首段：「它用 OpenCV 讀影片，需要額外安裝 `opencv-python-headless`：這是 OpenCV（Python 裡叫 `cv2`）不含開視窗功能的版本，適合 Colab。」；連帶第 221、256 行的 `python -m pip install -r requirements-video.txt` | 「不含開視窗（GUI）功能、適合雲端環境」和 opencv-python README（tag 92，也就是 4.13.0.92）一致。但同一份 README 用粗體要求：opencv-python、opencv-contrib-python、opencv-python-headless、opencv-contrib-python-headless 這四個套件共用同一個 `cv2` 命名空間，同一環境只能裝一個；已經裝了好幾個，就要全部 `pip uninstall` 再只裝回一個。本頁沒寫這個前提。Colab 目前已經預裝 OpenCV（googlecolab/backend-info 的 pip-freeze.txt 列出 opencv-python 5.0.0.93、opencv-python-headless 5.0.0.93、opencv-contrib-python 4.14.0.94）。所以在 Colab 上「需要額外安裝」不成立：`import cv2` 本來就能用，只是版本是 Colab 的，不是本課固定的 4.13.0.92。若在 Colab 照第 221／256 行執行 `pip install -r requirements-video.txt`，headless 會換成 4.13.0.92，另外兩個版本不同的 OpenCV 套件卻還留在同一個 `cv2` 命名空間，正是 README 要人避免的混裝。本機若已裝過 opencv-python，情況也一樣。 | 已修正：補上 OpenCV 各套件共用 `cv2`、同一環境只能裝一種，已裝別的 OpenCV 套件時先移除；安裝指令旁也加了提醒。 |
| 2 | 建議 | 第 55 行（`run_stream` 簡化片段前的說明）：「`@torch.no_grad()` 寫在 `def` 的上一行，效果和第 0 章的 `with torch.no_grad():` 相同，只是套用在整個函式上：函式裡的計算都不記錄計算圖。」 | `run_stream` 是 generator 函式。PyTorch v2.9.1 的 torch/utils/_contextlib.py 裡，`context_decorator` 遇到 generator 函式會改走 `_wrap_generator`：每次恢復執行（send／throw／close）時才進入 no_grad，`yield` 把結果交回呼叫端時就恢復原本的梯度設定。原始碼註解特別說明這是「Wraps generators in the intuitive way」，和直接把函式本體包進 with 不一樣。若真的在函式本體用 `with torch.no_grad():` 包住迴圈，generator 停在 `yield` 時 no_grad 仍然有效，外面 for 迴圈裡的程式也會跟著不記錄梯度。本機 torch 2.9.1 實測：用裝飾器時，暫停期間呼叫端的 `torch.is_grad_enabled()` 是 True；改用 with 則是 False。所以對這個 generator 而言，「效果相同、只是套用在整個函式上」不精確；後半句「函式裡的計算都不記錄計算圖」是對的。 | 已修正：改成說明裝飾器在 generator 上的行為（只在往下執行時關閉梯度，`yield` 時恢復），並和在函式裡用 with 包住迴圈的差別對照。 |
| 3 | 建議 | 第 233 行（〈換成自己的影片〉程式碼）註解：「torch.set_num_threads(2) # 和完整程式一樣，只用 2 個 CPU 執行緒」 | PyTorch 2.9 文件寫的是：torch.set_num_threads「Sets the number of threads used for intraop parallelism on CPU」，只限制 PyTorch 自己的 intra-op 執行緒。這段程式同時用 OpenCV 解碼 `clip.mp4`：OpenCV 4.13.0 的 FFmpeg 後端在沒有另外指定時，會把解碼執行緒設成 min(CPU 核心數, 16)（cap_ffmpeg_impl.hpp 的 `fill_codec_context`），`cvtColor` 等也走 OpenCV 自己的平行框架。本機實測（8 核、opencv-python-headless 4.13.0）：呼叫之後 `torch.get_num_threads()` 是 2，但 `cv2.getNumThreads()` 是 8，`capture.get(cv2.CAP_PROP_N_THREADS)` 也是 8。所以這段程式並不是「只用 2 個 CPU 執行緒」。 | 已修正：註解改成 PyTorch 的 CPU 運算只用 2 個執行緒，OpenCV 解碼另有自己的執行緒。 |
| 4 | 建議 | 第 84 行〈把管線時間拆開量〉：「GPU 則會先把工作排隊、稍後才算（非同步），計時前要先等它算完。」 | 「GPU 預設非同步」符合 PyTorch 2.9〈CUDA semantics：Asynchronous execution〉。但該文件也寫明：沒有同步就量時間不準，要先呼叫 `torch.cuda.synchronize()` 再量，或改用 `torch.cuda.Event`。本頁沒點出這個 API，「計時前」也容易被讀成「開始計時前等一次就好」。本節是在四段前後各讀一次 `perf_counter()`。在 GPU 上若只在開頭同步，模型 forward 只是把工作排進佇列就回傳，「模型」這段會量得偏小；GPU 真正的計算時間，會落到後面第一個需要等結果的步驟（PyTorch 會在 GPU→CPU 複製時自動同步，例如後處理）。 | 已修正：改成每次讀 `time.perf_counter()` 前都要 `torch.cuda.synchronize()`，或改用 `torch.cuda.Event`。 |
| 5 | 建議 | 第 249 行（摺疊區〈進階：用無損影片檔驗證 adapter〉第一段）：「FFV1 是一種無損影片編碼：存進去再讀出來，每個畫素都不變，所以能逐值比對。……MP4 常用的有損編碼（例如 H.264）為了縮小檔案會丟掉一些細節，讀回的畫素會略有不同。」 | RFC 9043 把 FFV1 定義為無損格式，可以存多種像素格式（RGB 用可逆的 JPEG 2000 RCT 轉換存）。這裡的「無損」是針對交給編碼器的那份像素而言；原始 RGB／BGR 畫素能不能原樣讀回，還要看寫入端在編碼前把畫面轉成哪種像素格式。本頁實驗能逐值相等，是因為 OpenCV 4.13.0 的 FFmpeg 寫入器遇到 8 位元 3 通道的 BGR 輸入時，會替 FFV1 選 BGR0 這種 RGB 格式（cap_ffmpeg_impl.hpp 的 `CODEC_ID_FFV1` 分支），讀取時再轉回 BGR24。若畫面先被轉成 YUV 4:2:0（色度取樣減半）再交給 FFV1，顏色在編碼前就已經改變。本機實測：同一批隨機畫素，OpenCV 預設寫出的 FFV1 讀回完全相同；改用 ffmpeg 以 FFV1 加 `-pix_fmt yuv420p` 重新編碼後，最大誤差達 230。後半句把 MP4 讀回的差異只歸因於有損壓縮，也漏了 RGB→YUV 4:2:0 轉換本身造成的損失。 | 已修正：改成 FFV1 原樣存回交給它的像素，本實驗能逐值相等是因為 OpenCV 讓它直接存 RGB 類格式；並說明 YUV 4:2:0 轉換本身也會改變顏色。 |
| 6 | 建議 | 第 178 行（adapter 三個重點的第一點）：「開啟後，程式會占用這個檔案或相機，用完要呼叫 `capture.release()` 交還系統；不釋放的話，相機可能一直被占住。」 | OpenCV 4.13.0 的 `VideoCapture::release()` 文件寫明：「The method is automatically called by subsequent VideoCapture::open and by VideoCapture destructor.」也就是說，即使沒有明確呼叫 release()，只要 VideoCapture 物件被回收（解構）、再次呼叫 open()，或程式結束，裝置一樣會被關閉。「一直被占住」漏了這個條件，讀者可能以為只有 release() 能釋放，或以為程式結束後相機仍被鎖住。而第 180 行「要等 Python 回收它」那段說明，正是建立在這個解構行為上。本機實測：沒呼叫 release() 就直接 `del capture`，這個影片檔的開啟控制代碼從 1 個變成 0 個。 | 已修正：補上 VideoCapture 物件被回收時也會自動釋放，但回收時間不由程式控制。 |
| 7 | 建議 | 第 53 行：「呼叫它時，函式本體還不會執行，所以單寫 `count_up()` 什麼都不會印。」 | 「函式本體還不會執行」是對的（Python 3.12 語言參考：呼叫 generator 函式只會回傳 generator，要等第一次 next() 才開始執行）。但如果把 `count_up()` 單獨寫在 Colab／IPython 格子的最後一行，或在互動式 `>>>` 直譯器裡輸入，互動環境會把運算結果顯示出來（Python 3.12 的 `sys.displayhook`：互動式工作階段輸入的運算式，結果不是 None 就印出 repr），畫面會出現 `<generator object count_up at 0x…>`。本機 Python 3.12.15 互動模式實測確實印出這一行。本課以 Colab 為主，讀者照著試，會看到和「什麼都不會印」矛盾的輸出。 | 已修正：改成單寫 `count_up()` 時裡面的 `print` 不會執行，在 Colab 只會顯示 generator 物件本身。 |

### 第 4 輪：上一輪的處理：通過

逐句對照固定版本來源並實測。count_up() 句：呼叫 generator 函式不會執行本體，互動環境會顯示 repr，實測印出 <generator object count_up at …>。@torch.no_grad() 套在 generator 上：v2.9.1 torch/utils/_contextlib.py 的 _wrap_generator 只在 send／throw／close 時進入 no_grad；實測用裝飾器時，暫停期間呼叫端 is_grad_enabled 是 True，改用 with 包住迴圈則是 False。GPU 計時句：PyTorch 2.9 CUDA semantics〈Asynchronous execution〉要求先 synchronize 或改用 torch.cuda.Event。OpenCV 套件句與安裝提醒：opencv-python tag 92 README 第 35、42–47 行寫明四個套件共用 cv2、只能擇一、混裝時全部 pip uninstall 再裝一個，headless 不含 GUI；requirements-video.txt 固定 opencv-python-headless==4.13.0.92。capture.release() 句：OpenCV 4.13.0 videoio.hpp 第 922–927 行，解構時會自動呼叫。set_num_threads 註解：它只管 PyTorch 的 intra-op 執行緒，cap_ffmpeg_impl.hpp 另設 thread_count。FFV1／YUV 4:2:0 段：cap_ffmpeg_impl.hpp 第 3333–3359 行，FFV1 對 BGR24 輸入選 BGR0，有損編碼預設 YUV420P；verify_video_file.py 以 FFV1 寫入 8 位元 BGR。都屬實，回應〈補做的來源對照〉7 項，合寫作規則；摘錄比對工具印出 []。B：紀錄有〈補做的來源對照〉、來源清單與 7 項處理；第 3 輪第 1 項屬實（點名的三個殘句都已拿掉路徑）。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/18-video.md 第 174、221 行，以及摺疊區第 256 行的安裝指令 | 「先全部 pip uninstall」沒說怎麼找出已裝的 OpenCV 套件；頁面只點名 opencv-python、opencv-python-headless「等」，而 Colab 另外預裝 opencv-contrib-python，初學者容易漏移除。Colab 的 code cell 要加 ! 才能執行 python -m pip …，這裡沒有提醒。摺疊區第 256 行的安裝指令旁也沒有同樣的移除提醒，而補做的來源對照第 1 項點名過這一行。 | 已修正：列出四個 OpenCV 套件名稱與一行 uninstall 指令，註明 Colab 要加 `!`；摺疊區的安裝指令前也加上指回說明的提醒。 |
| 2 | 建議 | reviews/18-video.md 第 14、51–55 行 | 〈檢查用的檔案〉清單剩下「- 暫存副本紀錄（run.out…）」「- 練習複本」「- 突變測試」「- 渲染圖：暫存副本（…）」這類黏字或空項；第 14 行「我在暫存副本，exit 0」少了動詞。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

查證第 174 行的 OpenCV 段落與第 253 行摺疊區的提醒：
- 套件名稱與規則：opencv-python tag 92 README 第 35 行寫四個套件共用 cv2、只能擇一、混裝時全部 pip uninstall 再只裝一個；第 42–47 行寫 headless 不含 GUI、適合 Docker 或雲端。requirements-video.txt 固定 opencv-python-headless==4.13.0.92。
- 「Colab 通常已經預裝」成立：googlecolab/backend-info 的 pip-freeze.txt（commit 92a2364 與目前的 main）列出 opencv-contrib-python 4.14.0.94、opencv-python 5.0.0.93、opencv-python-headless 5.0.0.93。
- uninstall 指令：用 pip 26.2.1 在暫存 venv 實測同一行指令，沒裝的套件只印 WARNING: Skipping…，結束碼 0，所以四個名稱一次列出可以直接照抄。
- Colab 的 code cell 執行 shell 指令要加 !，屬實。
- 第 253 行的提醒指回〈接真影片的同一個入口〉，第 221 行也有同樣提醒，回應了第 4 輪第 1 項。
讀來通順，符合寫作規則。副本建置、validate_site 與摘錄比對（[]）都通過。
範圍外附記（不列為問題）：本機 macOS 的 opencv-python-headless 4.13.0.92，cv2.getBuildInformation() 顯示 GUI: COCOA，imshow 可以用；「不含開視窗功能」在 Linux／Colab 成立，這句不在這次的編輯範圍。

### 第 5 輪：上一輪的處理：通過

第 4 輪第 2 項的狀態屬實：第 52–55 行四項仍在，第 14 行已改成「我在暫存副本跑完整程式，exit 0、stderr 為空。」。第 3 輪第 1 項是人工寫的「殘句已清理」，就點名的三段而言屬實（路徑已拿掉）。第 4 輪第 1 項是頁面修正，見 docs/lessons/18-video.md。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 4 輪第 2 項處理欄開頭（第 134 行） | 開頭寫「第 3 輪的處理說明改由逐句比對紀錄產生」，但本頁第 3 輪只有一列（第 93 行），仍是人工寫的「…；殘句已清理。」，沒有重寫。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |

### 第 6 輪：上一輪的處理：有必要問題

第 4 輪 #2 與第二個第 5 輪 #1 的處理欄都說第 3 輪已經改成統一說明，但第 3 輪唯一一列仍是人工寫的「殘句已清理」；而且點名的第一段殘句其實還黏在一起。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪 #2 的處理欄（第 134 行）；第二個第 5 輪 #1 的處理欄（第 153 行）；第 3 輪 #1 的處理欄（第 93 行） | 第 4 輪 #2 寫「第 3 輪的處理說明改成統一的說明」，但第 3 輪唯一一列（第 93 行）仍是人工寫的「…見〈補做的來源對照〉；殘句已清理。」。第 5 輪 #1 指出了這點，它的處理「用詞類的處理說明改成統一的說明…」同樣不成立。第 93 行的「殘句已清理」本身也和紀錄不完全相符：journal 原文是兩個清單項目「- 暫存副本：<路徑>」和「- 紀錄：同一層的暫存副本（run.out…）」，紀錄第 52 行卻把它們黏成「- 暫存副本紀錄（run.out…）」，第 4 輪 #2 也把它列為黏字。 | 已修正：第 3 輪那一列改成統一的說明；路徑規則不再吃掉換行，兩個清單項目不再黏在一起。 |
