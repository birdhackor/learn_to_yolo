# 第18／19課影片應用補項：獨立技術準確性審查

審查日期：2026-10-02。結論：新增的實際檔案adapter、detector→tracker接線與CPU計時敘述通過本次獨立查核；本範圍沒有required修正。下列結論基於實際函數執行及固定版本官方source，沒有把作者自述當成通過證據。

範圍為`docs/lessons/18-video.md`、`19-tracking.md`、兩份`lesson_cases`、`scripts/verify_video_file.py`、`requirements-video.txt`、`video-file.json`、`18-video.json`、`19-tracking.json`及兩份notebook。保留既有`reviews/18-video.md`、`reviews/19-tracking.md`的初審與作者修訂文字，本檔只記錄這次獨立驗證。沒有修改教材、案例、script或既有JSON；沒有Git操作、遠端mutation或GPU工作。

## 實際執行與證據保護

使用repository現有`.venv-model`，PyTorch `2.9.1+cpu`、NumPy `2.3.5`、OpenCV `4.13.0`，CPU兩個threads。直接消費案例函數，不呼叫案例`main()`，也不執行會寫入固定JSON的`verify()`。試驗程式與結果放在`/tmp/closure-accuracy-application/check_application.py`及`independent-result.json`；2026-10-02 18:08:54 UTC的該次實跑exit 0。

查核的案例SHA-256：

- `18-video.py`：`2d7c5acafb068d819c740fd41c6b30e1d2ca798e94ab0ced29806fb732139f25`。
- `19-tracking.py`：`adf6924a818b9d10e912a08d7aa1437abc27a2ec285a2d094b5834a5f1050895`。
- 當時`verify_video_file.py`：`f7820880f0c89c715c58b2f7ad4682ee9283db4d1a4d76a75787831f155f5b4b`。最新固定JSON的script SHA與此相符；其`code_commit_scope`已明確區分基底commit和此次執行的working-tree bytes。

## 逐項查核結果

| 敘述／入口 | 獨立檢查 | 結果 |
| --- | --- | --- |
| [19頁接線片段](/workspace/learn_to_yolo/docs/lessons/19-tracking.md:63) | 從Markdown擷取完整`import runpy`片段逐字執行；沒有呼叫兩節`main()` | 真正訓練一次model，再完成12幀推論與association；只印12行逐幀IDs |
| [12幀FFV1檔案實驗](/workspace/learn_to_yolo/docs/lessons/18-video.md:65) | 在`/tmp`用真實OpenCV writer把RGB轉BGR寫成FFV1 AVI，再直接消費`opencv_frames` | 12幀、96×64、`index=0…11`、來源時間`0…0.55s`；每幀RGB與記憶體來源逐像素完全相同 |
| 同一model接檔案與記憶體來源 | 比較每幀boxes、scores、labels與PIL疊圖，使用`torch.equal`／`np.array_equal` | 全部完全相同；沒有用GT框代替推論 |
| [捕捉資源釋放](/workspace/learn_to_yolo/lesson_cases/18-video.py:40) | 以觀察代理包住真正的`VideoCapture`，保留實體handle並記錄`read`與`release`呼叫 | EOF讀13次，其中最後一次返回false；explicit `release()`一次，handle關閉 |
| 提前關閉與開檔失敗 | 取得第0幀後`adapter.close()`；另讀取不存在的檔案 | 提前關閉讀一次、release一次；開檔失敗拋`RuntimeError`、讀零次、release一次；兩者handle都關閉 |
| [文件中的`closing`用法](/workspace/learn_to_yolo/docs/lessons/18-video.md:79) | 真正讀檔後`break`；另注入model推論錯誤 | 兩種情況都explicit release一次。`closing`確實覆蓋下游中止，而非靠garbage collection碰巧釋放 |
| `run_stream`惰性／eval／no_grad | 建立generator時檢查未forward；逐次`next()`，以model hook記錄真實forward | 12次均`training=False`、`grad_enabled=False`、`raw.requires_grad=False`；每次yield後caller的grad mode恢復為True |

獨立產生的AVI為9,232 bytes，SHA-256為`1c124d4aae1e30c4eed7f6c2e5cb98613ad1e059bce7f657c78be00c83fd64dc`，與固定`video-file.json`一致。

第18頁硬寫的框數`[1,1,0,0,0,0,1,1,0,0,0,0]`與獨立執行、`18-video.json`及`video-file.json`皆一致。12幀都實際處理，8個空預測沒有被誤稱為掉幀。第19頁接線的IDs也一致：`[[1],[1],[],[],[],[],[2],[2],[],[],[],[]]`。空幀仍呼叫`update`，舊track依frame gap過期；再出現時建立ID2。此接線未讀GT身份，不支援新增ID switch／IDF1／HOTA成績的宣稱；正文與JSON均有保留這個邊界。

也獨立呼叫`tracking.run`核對既有人工案例：預設last-box為3次switch、motion為0；`max_age=1`時motion最後IDs為`[1,3]`、1次switch。反貪心IoU矩陣實得`[[0.8181818,0.6666667],[0.5384616,0.25]]`，精確配對為`[(0,1),(1,0)]`；空detections回傳空配對。這些正文硬寫數字與原JSON一致。

## 官方版本來源與時間契約

下列來源於本輪以公開只讀HTTP取得，版本固定，不憑通用印象推定API行為。

- [OpenCV 4.13.0：VideoCapture properties與PTS](https://github.com/opencv/opencv/blob/4.13.0/modules/videoio/include/opencv2/videoio.hpp#L142)：`CAP_PROP_FPS`是frame rate；`CAP_PROP_POS_FRAMES`指下一個要解碼的frame，案例自己從0計數沒有混用該property；`CAP_PROP_PTS`則是FFmpeg backend專用、最近讀入frame的呈現時間，以FPS time base表示。不能把單純遞增index當媒體PTS。
- [OpenCV 4.13.0：FFmpeg get_fps](https://github.com/opencv/opencv/blob/4.13.0/modules/videoio/src/cap_ffmpeg_impl.hpp#L2186)：先取`avg_frame_rate`，必要時猜測／fallback。`index/fps`不會因此取得VFR各幀的真實間隔。正文第57行要求VFR／網路來源改用可靠PTS，而沒有宣稱adapter已做到，判斷正確。
- [OpenCV 4.13.0：VideoCapture read](https://github.com/opencv/opencv/blob/4.13.0/modules/videoio/include/opencv2/videoio.hpp#L980)：讀取會grab、decode下一幀，無幀時返回false；[release](https://github.com/opencv/opencv/blob/4.13.0/modules/videoio/include/opencv2/videoio.hpp#L922)關閉檔案或裝置。這與case的`finally`及實測EOF行為一致。
- [OpenCV 4.13.0：BGR預設與cvtColor](https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/include/opencv2/imgproc.hpp#L3765)，以及[VideoWriter.write](https://github.com/opencv/opencv/blob/4.13.0/modules/videoio/include/opencv2/videoio.hpp#L1199)：一般color frame以BGR供writer；案例寫入前RGB→BGR，讀取後BGR→RGB，沒有把兩種channel契約弄反。此FFV1實驗的完全相同結論有逐像素證據，沒有外推到有損codec或所有色彩轉換。
- [PyTorch v2.9.1：generator context wrapper](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/_contextlib.py#L19)：`context_decorator`辨識generator後使用`_wrap_generator`；每次`send`、`throw`與`close`都重新進入context。因此`@torch.no_grad()`包住`run_stream`的每次恢復執行，不會只包住建立generator那一下。下載的官方`_contextlib.py`與本機已安裝source逐byte相同，SHA為`64fe29ee3f970cd805a97661596ef05bc680b9694d5c66a79fbf0cd28e1e5d6b`。
- [PyTorch v2.9.1：no_grad](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/autograd/grad_mode.py#L21)明示停用gradient calculation、thread-local，以及factory function的例外。本課觀察的是已建立model的實際forward結果，沒有用該例外誤判。

案例檔案時間分支實際是「來源為str且FPS>0」才用`index/fps`；否則用`perf_counter()-started`，即開檔後相對的讀取完成時間，並非sensor曝光時間。固定FFV1來源已確認FPS=20，所以本輪`0…0.55s`的宣稱成立。可選的小幅文字補強是第18頁第57行把「檔案的frame時間用index/fps」補成「有可用FPS的固定幀率檔案以index/fps估算」並交代FPS未知的fallback；這不是本次受控實驗的required缺陷。

## 計時範圍是否誠實

第18頁表格的`0.199 / 0.215 / 0.337 / 0.045 / 0.887 ms`是固定`18-video.json`數值的正確四捨五入。`run_stream`從取得frame後才開始`perf_counter()`，到畫框完成結束，正確涵蓋前處理、CPU forward、decode/NMS／還原及畫框；不含來源產生／解碼、下游收集與顯示。各項median相加不必等於total median，正文有說明。

本次獨立執行略過首幀的total median約`0.784ms`；這個執行差異不構成數字錯誤，頁面已說明是當次小模型測量、不是速度保證。來源只有20FPS，而且沒有以sleep重演到達節奏；不能由total倒數推論實體相機FPS或畫面年齡，正文第43至45行也沒有這樣宣稱。

最新固定檔案JSON的file pipeline為`0.015970064s`，本次獨立實跑為`0.016384957s`。script把起點放在消費file generator之前、終點放在`list()`完成之後，所以capture constructor／開檔、read／decode、推論、畫框、EOF釋放及Python收集均在內；先前訓練、fixture編碼、保存overlay、tracker以及顯示／網路佇列不在該計時內。[正文第74行](/workspace/learn_to_yolo/docs/lessons/18-video.md:74)及JSON的timing scope與之相符。

`verification_wall_seconds`另計fixture編碼、釋放檢查、訓練、兩條推論、tracking、第一張疊圖保存與hash，排除imports／process startup／最後JSON write；script計時位置符合該敘述。`18-video.json`的`process_wall_seconds=2.4652`與`19-tracking.json`的`1.1865`則明說是整個subprocess時間，沒有被混充模型延遲。這些均只是受控CPU功能檢查，沒有正式throughput benchmark的證據或宣稱。

## Notebook與明示限制

兩份notebook索引3的完整實驗code cell與對應case逐字一致；環境格固定`REF='lessons-v0.3.0'`，選用檔案驗證文字包含固定`requirements-video.txt`與真實script命令。`requirements-video.txt`保留NumPy `2.3.5`、固定`opencv-python-headless==4.13.0.92`，與本機runtime版本一致。

本次未執行notebook的clone／pip環境格，也未測尚待發布的新tag；不把當前remote tag尚未發布解釋成缺少憑證。未測實體相機、有損MP4、VFR PTS、live/network latency、任意真實照片辨識或長訓練。這些已在正文／JSON明示範圍外，不列成required。綜合本次實跑，新增的受控影片檔案驗證與第18→19節接線有充分、範圍一致的證據。

## 修訂後獨立複查：來源時間與自有影片入口

2026-10-02再次讀取修訂正文與原case；保留上面的初始觀察，原行號指修訂前版本。這次只追加查核，不改寫初審或既有artifact，不重跑訓練。

[修訂後的時間段落](/workspace/learn_to_yolo/docs/lessons/18-video.md:59)已明示「有可用FPS的固定幀率檔案，以index/fps估算」、FPS未知或相機改用「從開啟後開始計」的monotonic經過時間，以及不是sensor曝光時間。這與[未變動的adapter](/workspace/learn_to_yolo/lesson_cases/18-video.py:44)相符：先取得FPS、設定`started`，每次成功`read()`後，str且FPS>0才用`index/fps`，否則用`perf_counter()-started`；再BGR→RGB產生Frame。VFR／網路來源仍要求可靠PTS，沒有把固定FPS推算升格成實測擷取時間。初審的可選文字補強已經獨立確認完成，沒有新增required。

為確認fallback沒有誤讀，使用既有`/tmp` FFV1檔案保留真實capture／read，只將`CAP_PROP_FPS`查詢結果注入0，並固定`perf_counter()`值為`100,100.25,100.75`。前兩幀時間實得`.25,.75`，證明用的是從開始點扣除後的讀取時刻，而非index/FPS或絕對clock值。generator明確close；此小檢查exit 0，未做model fit／推論；本機`perf_counter`的clock info也確認`monotonic=True`。案例SHA仍是上列`2d7c5ac…`。

[新版自有影片片段](/workspace/learn_to_yolo/docs/lessons/18-video.md:80)已獨立目視核對：它明確import `runpy`／`torch`／`closing`，先設定兩個CPU threads，從case取出`fit_detector`、`opencv_frames`及`run_stream`，不依賴上一個Python程序的函數定義。`runpy.run_path`未指定`run_name='__main__'`，因此原case的main guard不會執行。模型只fit一次並回傳eval；str來源`'clip.mp4'`接原adapter的FPS／RGB契約，再由`for`實際消費`run_stream`，結果使用真正存在的`index`與`prediction['boxes']`。`closing`包住來源，可沿用本輪已實測的break／推論錯誤釋放保證。新增文字也正確區分Colab同一工作階段與本機不同Python程序。

這次目視複查沒有聲稱另外完整執行新版自有影片片段；它的函數載入、資料介面、惰性消費與釋放方式均與上次已獨立實跑的來源／管線相同，新增`threads=2`也與本次受控CPU檢查相符。複查範圍通過，既有數據和範圍限制繼續適用。
