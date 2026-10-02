# 第 18 課陌生讀者審查

範圍僅包含 `docs/lessons/18-video.md`、`lesson_cases/18-video.py` 與 `docs/assets/diagrams/18-video.svg`。以具備基礎 Python／PyTorch、已學過圖片推論與 letterbox 的讀者角度審查；未讀其他課程，未修改正文、程式或原圖。

結論：流程與邊界大致清楚，需修以下兩個問題。沒有建議刪除的主要內容。

## 1. 核心程式片段缺少 frame → tensor 與 result 的連接

- 位置：[正文第 15 行](/workspace/learn_to_yolo/docs/lessons/18-video.md:15)，尤其第 17、21 行；實作對照 [程式第 73 行](/workspace/learn_to_yolo/lesson_cases/18-video.py:73) 與第 90 行。
- 問題：迴圈取到的是 `frame`，卻直接使用未定義的 `tensor_rgb`，最後又 `yield` 未定義的 `result`。本課的重點正是把圖片偵測接上逐幀來源；這段省略會讓陌生讀者無法沿著變數看清兩者如何接起來，也容易把它當成可直接搬進函式的片段。
- 具體修法：標成「`run_stream` 函式內的核心片段，省略畫框與計時」，在 `letterbox` 前補上 `tensor_rgb = torch.from_numpy(frame.rgb.copy()).permute(2, 0, 1).float() / 255`。將 `yield result` 換成明確的字典，例如含 `frame.index`、`frame.timestamp_s` 與 `prediction`；如果保留 `result` 名稱，就先建立它。補一句「generator 每次 `yield` 交出一幀結果，下游要求下一筆才繼續處理」，即可連接第 24 行的逐幀／`list()` 說明。

## 2. 改 count／fps 的練習沒有同步輸出播放速度，還會觸發斷言

- 位置：[正文第 56 行](/workspace/learn_to_yolo/docs/lessons/18-video.md:56)；[程式第 118 行](/workspace/learn_to_yolo/lesson_cases/18-video.py:118)、第 119–127 行；物件位置公式在 [程式第 29 行](/workspace/learn_to_yolo/lesson_cases/18-video.py:29)。
- 問題：只將來源 `fps` 改成 10，最後來源時間會變成 1.1 秒，立即不符第 121 行的 `.55` 斷言；即使改掉斷言，第 124 行的 `duration=50` 仍使 GIF 以 20FPS 播放，觀看輸出時不會呈現正文所說的每秒 40 畫素。第 127 行的來源 FPS／時長也仍是固定數字。另一個參數邊界是 `count=24`：最後四幀紅色像素數為 `[168,112,56,0]`，第 23 幀的矩形已完全離開畫面；讀者可能把這個沒有物件的末幀與前文模型漏檢混在一起。
- 具體修法：在 `main()` 集中設定 `count, fps`，把它們傳入 `synthetic_frames(count=count, fps=fps)`；長度與 index 斷言依 `count`，末時間依 `(count-1)/fps`，summary 的 FPS／時長依 `fps` 與 `count/fps`，GIF 的 `duration` 依 `round(1000/fps)`。正文說明改 `fps` 時也會更新 GIF 播放速度。對 `count=24` 的題目加一句「保持向右 4 畫素時，物件會逐漸離開畫面，最後一幀沒有物件」，或另給往返移動的選項；不要把末幀無物件解釋成模型漏檢。若靜態圖仍選第 0、5、11 幀，也應說明它只展示固定三幀，不代表 24 幀中的末幀。

## 已確認清楚且一致的部分

- `Frame` 的 RGB、dtype、shape 與來源時間契約，以及 `[64,96,3] → [1,3,64,64] → 原始座標` 的文字說明，可跟實作接上。
- FPS／latency 的差異、來源間隔、無限排隊造成畫面變舊、處理計時未含解碼／佇列，以及來源沒有 sleep，都有交代。median 不可直接相加的提醒正確。
- 十二幀都完成處理與八幀漏檢的區別清楚；圖中第 0、5、11 幀的來源時間、框數符合程式結果。保留失敗預測有助於讀者理解分佈差異。
- 相機預設不啟用，OpenCV 是額外依賴，CPU 同步計時與 camera adapter 未執行的範圍都有標示。`PTS`、`adapter`、`iterable` 可在首現補中文括註；這屬小幅文字改善，不是第三個關鍵問題。

驗證：在 `/tmp/review-18-video` 執行現有 `.venv-model` 與本課程式，退出碼 0；產生 12 幀、來源末時間 `.55` 秒、96×64 輸出，框數為 `[1,1,0,0,0,0,1,1,0,0,0,0]`。此次逐幀 total median 約 `.954ms`，頁面測得 `1.171ms`；頁面已聲明測量不是速度保證，這項執行間差異無需當成錯誤。另直接檢查 generator 的 count／fps 邊界與 GIF 每幀 50ms 設定。原始 SVG 的三張內嵌圖片已視覺檢查。

## 作者修訂（2026-10-02）

- 問題1：正文標明是run_stream內省略畫框／計時的核心片段，補齊frame.rgb→tensor_rgb，直接yield包含index、timestamp與prediction的字典；解釋generator按下游需求逐幀前進。iterable、adapter與PTS於首現補中文說明。
- 問題2：`main(count=12,fps=20)`統一來源參數，長度、index、末時間斷言、summary、GIF每幀duration與SVG都取自本次參數。圖改選第0、中間和最後一幀。生成器明確裁切右邊界並於完全離場時輸出空畫面，report附逐幀物件可見性；正文說明count24的第23幀無物件，不能算漏檢。
- 驗證：修改後預設案例exit 0，12幀框數保持原結果，末時間.55秒、GIF每幀50ms。隔離實跑count24/fps20，末時間1.15秒、時長1.2秒，最後四幀紅色畫素為168、112、56、0，SVG包含第23幀；實跑count12/fps10，末時間1.1秒、時長1.2秒，讀回GIF全部12幀的duration均為100ms。預設SVG已重新生成並通過XML解析。
