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

## 紀錄重產後的檢查

2026-10-05，另一位 AI 依 `AGENTS.md` 與 `docs/preparation/publish.md` 第 7 步，在自己的獨立副本逐句閱讀本頁、既有審查、`lesson_cases/18-video.py`、其全部 repo 相依模組，以及補充的 `scripts/verify_video_file.py` 與它使用的第 19 課 tracker。既有審查只作為待核對的證據。本次檢查的是 CPU 紀錄重產、中文面板帶回及計時正文更新後的頁面，沒有啟動 GPU、實體相機或下載資料集。

查核使用 Python 3.12.14、PyTorch 2.9.1+cpu、NumPy 2.3.5、Pillow 12.0.0、opencv-python-headless 4.13.0.92；cwd 和 `PYTHONPATH=.` 都指向自己的副本。模型 Python 使用原 checkout 既有的 `.venv-model/bin/python`，文件工具使用既有的 `.venv-docs`，不變更環境依賴。實際執行的主要命令如下，輸出、練習副本與畫面證據保存在自己的副本 `artifacts/runs/review18-*`：

```bash
LD_LIBRARY_PATH=/tmp/lessons-v0.4.0-tools/extracted/usr/lib/x86_64-linux-gnu \
  /tmp/lessons-v0.4.0-tools/extracted/usr/bin/rsync -a \
  --exclude=.git --exclude=site --exclude='.venv*' \
  /tmp/lessons-v0.4.0-reviews/base/ /tmp/lessons-v0.4.0-reviews/review-18/
cd /tmp/lessons-v0.4.0-reviews/review-18
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python lesson_cases/18-video.py
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python scripts/verify_video_file.py --output artifacts/runs/review18-file
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python artifacts/runs/review18-audit.py
/workspace/learn_to_yolo/.venv-docs/bin/zensical build --clean --strict
/workspace/learn_to_yolo/.venv-model/bin/python scripts/validate_site.py
```

三項執行都以 exit 0 完成，完整課程與檔案腳本的 stderr 為空。查核輔助程式最後印出 `AUDIT_PASS`；它讀取最高 score 的一行產生一則將帶梯度 tensor 轉為純量的 PyTorch 警告，屬輔助程式，沒有造成教材程式或斷言失敗。這次實跑的計時因系統負載而與保存紀錄不同，沒有拿查核時的計時替換教材數字。

**數字與結論。** 直接解析 `18-video.json` 的 stdout，原始中位數依序為 0.2764804958133027、0.30956500268075615、0.43933201231993735、0.06989399844314903、1.1000390077242628 ms；三位小數正好是正文的 0.276／0.310／0.439／0.070／1.100。四項未取整中位數合計 1.0952715092571452 ms，與 total 的方向一致；表上取整值合計 1.095 ms。實跑逐幀四段計時相加與該幀 total 相同，差異來自中位數不能分配到加法。以未取整 total 計算 `1000/total` 得 909.0586724454241，正文「約 909」正確；將它稱為 FPS「上限」的問題見下表。

正式來源仍為 12 幀、20 FPS、96×64 RGB；來源間隔 50 ms，最後時間戳 0.55 s、播放時間 0.6 s、GIF 每幀 50 ms。暖機是一份另外產生的第 0 幀，得到框後丟棄，不進 12 幀統計。實跑的 index 0～11 與圖大小都通過斷言。letterbox 實際內容 43×64、上下補 10／11，水平比例 2/3、垂直比例 43/64；raw shape `[1,4,4,7]`。來源是立即產生下一幀的合成 generator，沒有睡眠或相機佇列；`run_stream` 的計時開始於取得 `Frame` 之後、結束於疊圖完成，並不含影片開啟／解碼、訓練、GIF 輸出、顯示、網路與排隊。

完整程式、簡化片段、FFV1 檔案與保存紀錄的框數均為 `[1,1,0,0,0,0,1,1,0,0,0,0]`。12 幀都有物件且全部處理，8 幀沒有框是偵測漏檢，沒有把它誤算為串流掉幀。以 `video-file.json` 存的來源座標框，獨立用半開 xyxy 的長寬計算交集及聯集，沒有直接拿 repo 的 `box_iou` 當唯一依據：

| 幀 | GT 框 | 交集面積 | 聯集面積 | IoU | 正文取整 |
| --- | --- | --- | --- | --- | --- |
| 0 | `[4,20,18,34]` | 134.9128594930662 | 267.5997388069927 | 0.5041591598501992 | 0.50 |
| 1 | `[8,20,22,34]` | 130.747676403269 | 276.52069891266865 | 0.47283142606464335 | 0.47 |
| 6 | `[28,20,42,34]` | 85.4353223174403 | 280.95689208601834 | 0.3040869426021528 | 0.30 |
| 7 | `[32,20,46,34]` | 116.88559948009788 | 253.5415713355469 | 0.46101157638329404 | 0.46 |

最高 score 實跑為約 `[.930,.894,.090,.012,.006,.052,.263,.114,.001,.0003,.004,.025]`；9／12 幀低於 .25，支持降低顯示門檻的理由。以控制 raw 值另外測試 score 恰為 .1 時留下、門檻改成 .100001 時捨棄，確認比較是 `>=`。程式仍由真正的 forward、objectness×最大 class probability、score 篩選、按類別 NMS 及座標還原產生框，沒有注入 GT；低門檻框數也不是準確率。訓練資料仍是每個框完整落於 16×16 格內的 64×64 圖，框寬高 8～15。來源物件縮放成約 9.33×9.41，y 約 23.44～32.84；x 跨格幀重新算得 `[2,3,4,8,9,10]`，和正文一致。第 5、11 幀不跨 x 格線仍漏檢，因此原文把跨格列為未證實的部分原因，保留了合理限制。

**程式摘錄、練習與資源。** 執行頁面的 `count_up`、簡化 `run_stream` 及〈換成自己的影片〉片段。簡化片段與完整函式逐幀 index、timestamp、boxes、scores、labels 全相同；三個標記摘錄交給 `validate_lessons.py` 的 `excerpt_problems()` 得到 `[]`，notebook 最後一格仍逐字等於 lesson case。`@torch.no_grad()` generator 暫停時呼叫端梯度恢復，函式內用 `with` 則在暫停時仍關閉，兩者實测符合正文。

練習 1、2 都複製完整 case，只改最後一行 `main(count=24)`／`main(fps=10)` 再用子程序實跑，不修改斷言。24 幀的最後時間戳 1.15 s、播放 1.2 s，圖取 0／11／23 幀；第 19～23 幀可見紅色寬度為 14／12／8／4／0，第 20 幀開始裁切，第 23 幀可見欄位 false 且沒有框。10 FPS 的最後時間戳 1.1 s、播放 1.2 s、GIF 100 ms；實際 boxes／scores／labels 與 20 FPS 相同，位移由每秒 80 改為 40 畫素。逐幀讀回 GIF，50／100 ms 正確；另用 Pillow 存 33 ms 再讀回，確為 30 ms。排隊練習用 `Fraction` 重算，第 90 幀完成 4550 ms、延遲 1550 ms；只取最新幀依序處理 0／1／3／4／6／7…，延遲交替為 50 與 200/3 ms，沒有浮點時間剛好到達的判斷誤差。

FFV1 fixture 是 9232 bytes，SHA-256 `1c124d4aae1e30c4eed7f6c2e5cb98613ad1e059bce7f657c78be00c83fd64dc`，與保存紀錄相同。檔案腳本比較全部 12 幀，RGB、boxes、scores、labels、疊圖逐值／逐 tensor 完全相同，時間戳為 0 至 .55 s，EOF／提前 close／開啟失敗都已 release。另包住真實 `VideoCapture` 觀察控制代碼：`closing` 中提前 break、forward 拋錯且保留 traceback、adapter 的 cvtColor 拋錯均關閉；單純 break 且仍保留來源 generator 時控制代碼仍開啟，顯式 close 後才關閉。這支持正文要求用 `closing`，沒有把 generator 暫停誤認成已釋放。自己的影片片段以同一份 AVI fixture 暫時命名為 `clip.mp4` 原樣執行，印出 0～11 與正確框數；OpenCV 依內容識別容器，此項只驗證片段，不能聲稱已測有損 H.264／MP4。影片腳本的檔案管線計時確實包含 capture/open/read/decode 與收集結果，訓練、編碼、磁碟寫出在其外；全驗證計時則包含那些前置工作。實體相機、可變幀率 PTS、現場／網路延遲及正式 throughput benchmark 都未測。

**SVG 與網頁。** 網站圖與原始重產留存的 `artifacts/lesson-18/panel.svg` SHA-256 完全相同。解碼 SVG 內三張 PNG，都是 96×64；第 0 幀圖像與重跑檔案的第一張疊圖逐值相同，含黃框與 score .93；第 5、11 幀與其合成原圖逐值相同、沒有黃框。標籤為第 0／5／11 幀、0.00／0.25／0.55 s、單次總處理 2.28／.91／.91 ms、框 1／0／0，正文將單次時間與中位數區分，沒有舊英文標籤。

嚴格建置與 `validate_site.py` 通過。用 `.venv-docs` 的 Playwright 與 `/usr/bin/chromium`，對獨立副本建出的網站作 1440×1000 與 390×844 檢查並實際查看截圖：SVG 正常載入，標籤未裁切；8 個 code block、3 個 excerpt、7 個折疊區及表格都正確轉換；頁面沒有水平溢位。手機圖中文字縮得很小，列為可選改善，見下表。

**重新打開的原始來源。** 固定版本原始碼均由 HTTPS 原址取得並實際讀取相關段落；PyTorch 的 docs.pytorch.org 2.9 HTML 與 NumPy 網頁回 403，改讀同版官方原始碼／已安裝固定版官方函式原碼，不把抓取失敗當成完成核對。

- PyTorch v2.9.1，commit `d38164a545b4a4e4e0cf73ce67173f70574890b6`：[torch/utils/_contextlib.py](https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/utils/_contextlib.py) 的 `_wrap_generator`／`context_decorator`；[docs/source/notes/cuda.rst](https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/docs/source/notes/cuda.rst) 的 Asynchronous execution；[torch/_torch_docs.py](https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/_torch_docs.py) 的 from_numpy 與 set_num_threads；`torch/autograd/grad_mode.py` 的 no_grad。對照 generator 梯度範圍、GPU 同步計時、共用 NumPy 記憶體、只控制 PyTorch intra-op 執行緒。
- OpenCV 4.13.0，commit `fe38fc608f6acb8b68953438a62305d8318f4fcd`：[videoio.hpp](https://github.com/opencv/opencv/blob/fe38fc608f6acb8b68953438a62305d8318f4fcd/modules/videoio/include/opencv2/videoio.hpp) 的相機編號、isOpened、release／解構、read／get；[cap_ffmpeg_impl.hpp](https://github.com/opencv/opencv/blob/fe38fc608f6acb8b68953438a62305d8318f4fcd/modules/videoio/src/cap_ffmpeg_impl.hpp) 的解碼 thread_count、BGR24 讀回與 FFV1 的 BGR0／BGRA 分支；[imgproc.hpp](https://github.com/opencv/opencv/blob/fe38fc608f6acb8b68953438a62305d8318f4fcd/modules/imgproc/include/opencv2/imgproc.hpp) 的 cvtColor BGR 說明。
- opencv-python tag 92（4.13.0.92），commit `4ddfc013fd1f13d9b9e379dbebf2cdbeb052e7f8`：[README](https://github.com/opencv/opencv-python/blob/4ddfc013fd1f13d9b9e379dbebf2cdbeb052e7f8/README.md) 的四種套件擇一、共用 cv2、混裝全部移除、headless 無 GUI。
- Colab 官方 backend-info，commit `92a2364a31cd9b686f88075e52d0ff5e5c098a40`：[pip-freeze.txt](https://github.com/googlecolab/backend-info/blob/92a2364a31cd9b686f88075e52d0ff5e5c098a40/pip-freeze.txt) 列出 opencv-contrib-python 4.14.0.94、opencv-python 5.0.0.93、opencv-python-headless 5.0.0.93，支持「Colab 通常已經預裝」與避免共用 cv2 混裝的提醒；本次沒有登入或執行託管 Colab 工作階段。
- Python 3.12 官方文件：[dataclasses](https://docs.python.org/3.12/library/dataclasses.html)、[runpy.run_path](https://docs.python.org/3.12/library/runpy.html#runpy.run_path)、[contextlib.closing](https://docs.python.org/3.12/library/contextlib.html#contextlib.closing)、[generator／close](https://docs.python.org/3.12/reference/expressions.html#generator.close)、[time.perf_counter](https://docs.python.org/3.12/library/time.html#time.perf_counter)。本機另查 `get_clock_info('perf_counter')`：CLOCK_MONOTONIC、monotonic=True、adjustable=False。
- NumPy 2.3.5 安裝版官方 `numpy.median` 原碼／docstring，明寫偶數個值取排序後中間兩值平均；Pillow 12.0.0，commit `693df7b42c666f88c719f9973be0ad71607328e0`：[GifImagePlugin.py](https://github.com/python-pillow/Pillow/blob/693df7b42c666f88c719f9973be0ad71607328e0/src/PIL/GifImagePlugin.py) 的 `duration = int(duration / 10)`；[GIF89a](https://www.w3.org/Graphics/GIF/spec-gif89a.txt) 的 Delay Time 以 1/100 秒記錄。
- [RFC 9043](https://www.rfc-editor.org/rfc/rfc9043.txt) 摘要及 §3.7.2，FFV1 無損與可逆 RGB 色彩轉換；FFmpeg n7.1，commit `b08d7969c550a804a59511c7b83f2dd8cc0499b8`：[libavutil/frame.h](https://github.com/FFmpeg/FFmpeg/blob/b08d7969c550a804a59511c7b83f2dd8cc0499b8/libavutil/frame.h) 的 PTS 定義。確認無損的前提是交給編碼器前沒有先損失色度，PTS 是呈現時間，不能拿 i/fps 當實際相機擷取時間。

| # | 嚴重度 | 位置 | 發現 | 處理 |
| --- | --- | --- | --- | --- |
| 1 | 必要 | 〈處理速率與延遲是兩回事〉，原查核版本第 124 行 | 把中位數 total 的倒數約 909 FPS 稱為四段計算的「上限」不成立。中位數不決定整段平均速度；1／100／100 ms 反例得到 median 倒數 10 FPS、實際 14.93 FPS。 | 建議改為典型單幀時間的粗略換算，持續 FPS 用完成幀數／總經過時間。主代理回報已修正；本次只審查原快照，修改後版本交另一位 AI 複查。 |
| 2 | 建議 | 三欄 `18-video.svg`，手機 390 px 檢查 | 圖中文字縮至約 7.50／6.79 px，未裁切但較難讀。 | 可加上連到同一 SVG 的圖片連結，供讀者點開放大；不影響紀錄數字正確性。是否採用由編者記錄。 |

**本次已核對的 SHA-256。** 下列是本次舊快照的涵蓋內容，頁面文字已在查核後另行修正，不能用這一組頁面 hash 宣稱涵蓋修改後版本。全部 `dependencies_sha256` 欄位都逐檔重算相符，完整清單另存於副本 `artifacts/runs/review18-audit/checked-hashes.json`。

```text
docs/lessons/18-video.md 0ceaef138f82b9c43c26c12b03b42c039279bcb17cfa1c7994825e6944ad4a20
docs/assets/diagrams/18-video.svg 9c2bf9984e04e64f5a1f788ea774a74ad74799cb2732b0ce353369ae1d8f3040
artifacts/checks/curriculum/18-video.json f7d0eb05bad4a736b561c75463049d794d829a09397b19b526782e0e838bb609
artifacts/checks/curriculum/video-file.json e03980d47e7e4f87c0c4243f63754ad56a905cfacc4cf2ebc30bfd07b1394d58
lesson_cases/18-video.py 0aa6c1713098e244dd3651c6f319844ad0e17b10ee28a187e8a224ee6e9024a0
scripts/verify_video_file.py ec77505526972a57b74cf757e303ccfea7459dfba2ffd9857b7634129070dcec
miniyolo/__init__.py 785b058b2b011124243b06ee59df8e59dfa75b77f050b897479e5be636763fc9
miniyolo/data.py cccad00e2c4f96eb6567eafc9e12248379c6b715fc1790d75518a253baa6181d
miniyolo/geometry.py 6a6b57d3493888e99dae4a012dab78127b8b543a0e01d8107963a3d6e63d8483
miniyolo/inference.py 995ac8f942d0c1e43d94f2efb3b7adcb91a9feb6b9bab923615209f0c190c313
miniyolo/losses.py 81fa9331c9a2aebe2e9c6c453a5313e64566bb20c3fe77ca45ec9df53424d4e4
miniyolo/metrics.py 53ce982e37cbd96c784f75c7d30faf99d52f79ab83ca7b8114eb21b4327330e0
miniyolo/models.py 49d029ea4ba2650ce8933cf97e3d25dc7aff2ca4e972eec19cdda17b0f4900e6
miniyolo/targets.py 2c8e32f2845b3bf970c77304c5cca0f999083d36a4d5af2f75f14aa89b823f16
miniyolo/provenance.py 21769813c36b45f513c9f0d6125194f3baa5ae265278ffd44a796d298d124ed3
lesson_cases/19-tracking.py e1fc77ce9e6e8e09a34f4dedf8543155154ab58709eaf1ab61756793f7a14efa
```

### 修正後的獨立複查

2026-10-05，由不同於原第 18 課查核者的另一位 AI，在自己的 `/tmp/lessons-v0.4.0-reviews/fix-18/` 副本複查兩項修正。副本依指定 rsync 從修正後的 `base/` 建立，排除 `.git`、`site`、`.venv*`。先讀 `AGENTS.md`、發布流程第 7 步與原查核發現，再閱讀本頁全部正文，逐段比對原 `review-18/` 快照。差異只有第 125 行的 FPS 解釋、第 152 行的 SVG 自連結與第 154 行的點圖說明；lesson case、全部 repository 相依模組、notebook、保存紀錄與 SVG 本體沒有改動。沒有改 root 的追蹤檔案或 coverage，也沒有使用 Git、remote、GPU、實體相機或下載資料集。

**數值與計時範圍。** 直接解析當前 `artifacts/checks/curriculum/18-video.json` 的原始 stdout：全流程中位數是 `1.1000390077242628 ms`，`1000 / 1.1000390077242628 = 909.0586724454241`，正文約 909 的算術正確。另獨立手算原反例：三幀分別花 1、100、100 ms，中位數為 100 ms，所以中位數倒數是 10 FPS；但持續完成速率為 `3 / 0.201 = 1000/67 = 14.925373134328359 FPS`，大於 10 FPS。新文字將 909 稱為「典型單幀計算時間粗略換算的數字」，並明說持續速率以處理完的幀數除以總耗時、未必等於中位數倒數，已解除把它當上限的錯誤。

對照 `run_stream` 第 76–98 行與 `main` 第 122–147 行：計時起點在來源交出 `Frame` 之後，終點在 PIL 疊圖完成、`yield` 交出結果之前；各段取同一幀的相鄰時間差，全流程取該幀的頭尾差，再各自對正式 12 幀取中位數；1 幀暖機不納入統計。本次另外用受控 CPU 輸出與假時鐘做範圍探針：來源每次先推進 2 秒、消費結果後再推進 3 秒，四個量測區間各推進 1 ms；兩幀的 total 仍各為 4 ms，逐段相加亦為 4 ms。這直接確認來源取得與下游消費不在該計時內。正文相鄰段落仍保留解碼、網路、磁碟、顯示及相機佇列未量測，合成 generator 沒有按 20 FPS 實際等待的限制，沒有把這份單幀統計擴大成端到端 FPS 或現場效能證據。

重新打開 PyTorch v2.9.1 官方固定 commit `d38164a545b4a4e4e0cf73ce67173f70574890b6` 的 [CUDA notes 原始碼](https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/docs/source/notes/cuda.rst)，閱讀 `Asynchronous execution`：GPU 工作預設排隊執行，未同步的時間量測不準，應同步或用 CUDA Event。這與正文保留的 CPU／GPU 計時區別相符；本次實跑是 CPU，沒有把 CPU 的計時方法宣稱已在 GPU 驗證。沒有新增論文或函式庫用法，因此未重做與兩項修正無關的外部來源查核。

**執行與手機連結。** 使用既有 Python 3.12.14、PyTorch 2.9.1+cpu，cwd 與 `PYTHONPATH=.` 指向自己的副本，重跑 `lesson_cases/18-video.py` 的預設示範一次；exit 0，stderr 為空，暖機 1 幀、正式 12 幀、20 FPS、末時間戳 0.55 s、框數 `[1,1,0,0,0,0,1,1,0,0,0,0]` 與保存紀錄一致。本次測到 total 中位數約 2.074 ms，未替換作者的 1.100 ms 紀錄。範圍探針修正裝飾器包裝層的注入位置後以 `FIX18_AUDIT_PASS` 完成，notebook 最後一格與 case 逐字相同；沒有重跑未受影響的練習 1、2、FFV1 檔案實驗或額外訓練套件。

自己的 `zensical build --clean --strict` 與 `validate_site.py` 都 exit 0。用 Playwright、`/usr/bin/chromium`，依指定 `--no-sandbox --disable-gpu --disable-dev-shm-usage` 啟動，在自有 HTTP server 的 390×844 手機／觸控 viewport 檢查建置頁面，實際看過正文圖與開啟 SVG 的截圖。圖片載入完整，顯示寬約 357.97 px；文件寬與 viewport 同為 390 px，沒有水平溢位。圖片的 `src` 與外層 `<a>` 的目的地均是同一個 `18-video.svg`，點圖說明已顯示；實際 `tap()` 後的文件導覽回 200、Content-Type 為 `image/svg+xml`，導向同一張 SVG。瀏覽器收到的圖、建置 site 的圖、正文來源圖和舊快照圖的 bytes／SHA-256 全相同。三格仍為第 0、5、11 幀，來源時間 0.00／0.25／0.55 s，處理時間 2.28／0.91／0.91 ms，框數 1／0／0，沒有更換圖碼或數據。

手機頁內標籤仍小；這次採納的是提供原 SVG 的入口。額外嘗試的 headless CDP pinch 沒有改變 `visualViewport.scale`，另以瀏覽器比例模擬可改變 scale，但這兩項不作為實體手機手勢縮放成功的證據。手機原生瀏覽器的縮放操作仍未實測，不影響「可點到同一個 SVG」已完成的複查結論。最終瀏覽器查核以 `FIX18_BROWSER_PASS` 完成；自有 port 8818 server 已停止，root 的 8794 server 沒有操作。

| 原發現 | 修正與獨立複查結果 | 處理狀態 |
| --- | --- | --- |
| 必要：中位數倒數約 909 FPS 被稱為上限 | 已改成典型單幀時間粗估，持續 FPS 明定為完成幀數／總耗時；手算反例、實際紀錄與計時範圍探針相符，限制未超出證據。 | 已修正並通過獨立複查。 |
| 可選：390 px 手機三欄 SVG 標籤較小 | 已採納圖片自連結及可點的正文說明；手機 tap 實際導向原 SVG，回傳圖檔完全相同。 | 建議已採納；實體手機手勢縮放未測。 |

剩餘必要問題：**0**。本次只複查上述兩項修正與其直接相鄰語境，沒有把原查核未重跑的項目重新宣稱為本次實測。

**修正後涵蓋指紋。** 以現行 `scripts/review_coverage.py` 的 `digest()` 只讀計算，已排除頁尾執行紀錄、將 Colab tag 正規化；沒有寫入 coverage。完整逐檔 hash 與結果保存在自己副本的 `artifacts/runs/fix18/audit.json`，頁面差異是 `page.diff`，手機結果是 `browser.json` 與 `mobile-section.png`／`mobile-open-svg.png`；預設程式輸出與建置 log 分別是 `artifacts/runs/fix18-default.stdout`／`.stderr`、`artifacts/runs/fix18-build.log`。

```text
docs/lessons/18-video.md（新正文 coverage hash）
b162e4b4a24c92d58d342ac496f2aca9abeb6a3f14a25efd59e4f93de5c79a7d
docs/assets/diagrams/18-video.svg
9c2bf9984e04e64f5a1f788ea774a74ad74799cb2732b0ce353369ae1d8f3040
lesson_cases/18-video.py
0aa6c1713098e244dd3651c6f319844ad0e17b10ee28a187e8a224ee6e9024a0
artifacts/checks/curriculum/18-video.json
f7d0eb05bad4a736b561c75463049d794d829a09397b19b526782e0e838bb609
```


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[applications當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/applications.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `applications` 實讀本頁 7 個凍結單元；首次使用／前文方法範圍四題位置為 18-video/00:first_use, 18-video/01:first_use, 18-video/02:first_use, 18-video/04:first_use，頁末為 18-video/06。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/applications.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/applications-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/applications.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R016、R019；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-applications.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 de 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。
