# Curriculum reader F：第17–20章獨立初讀

審查日期：2026-10-02。讀者立場：第一次接觸本專案，只知道基本Python／PyTorch／NN／CNN，數學不熟。先讀`docs/index.md`、`docs/learning-path.md`，再依序讀17、18、19、20；兼讀`docs/status.md`、`docs/validation/curriculum.md`、`docs/preparation/data.md`。沒有以舊reviews的結論作判斷。

本次只新增本報告。原教材的CPU紀錄用於核對，另在`/tmp/curriculum-reader-f`用`.venv-model/bin/python`、`PYTHONPATH=/workspace/learn_to_yolo`試做指定練習，沒有改教材、範例程式、notebook或正式實測圖。沒有使用GPU、啟動相機、發送外部訊息或發布tag。`lessons-v0.2.0`依任務約定留到最後發布，現在沒有把尚未發布的Colab連結列為教材缺陷。

## 共用導讀與驗證頁

實際讀檔：`docs/index.md`、`docs/learning-path.md`、`docs/assets/diagrams/object-journey.svg`、`docs/status.md`、`docs/validation/curriculum.md`、`docs/preparation/data.md`。輔助核對`miniyolo/data.py`、`miniyolo/geometry.py`、`miniyolo/inference.py`、`requirements-model.txt`。

首頁的64×64、4×4格、每格16畫素與紅框`(8,12)→(24,28)`相符；中心`(16,20)`按右側分界規則落在程式索引`(row1,col1)`。SVG座標映射`(76+4x,116+4y)`也吻合。圖與文字都明說這是已知標註。閱讀路線把17列為整合、18–20列為可選應用，合理；各頁列有前置知識，但應加回連結，初讀者才不用自行猜是哪一節。

發現與建議：

- **P1，資料準備頁的現況與規劃混在一起。** 原句包括「資料產生器、dataset adapter、模型與正式教材留到大綱定稿後實作」、「幾何資料先訂規格，不產生訓練程式」及「尚未實作generator」。17、18已實際呼叫`ShapeDataset`訓練，`miniyolo/data.py`已有可控矩形generator。初讀者會不知道眼前課程是否可執行。建議更新成已實作的64×64紅／藍矩形範圍，將尚未提供的多形狀、重疊／極端比例等需求明確放在後續規劃。尤其「類別不單靠顏色決定」是未來規格，當前`ShapeDataset`明確紅=0、藍=1。
- **P2，GPU現況的入口摘要容易被讀成全部未做。** 首頁說「GPU效能，會在安排硬體後補上」，status表將「真實資料／GPU完整對照」合寫為「未完成」，但同一status與20已有L4部署核對。完整真實資料／正式架構對照確實未完成；建議在入口直接分開「已做的小模型L4／checkpoint／TensorRT管線證據」及「未做的真實資料長訓練與正式效能比較」，避免讀者到了20才發現例外。
- **P2，共用本機執行前置只在status連README。** 各課直接給`PYTHONPATH=. python ...`，第一次跳到應用章的讀者未必知道先裝版本固定依賴、在repo根目錄執行。建議各課命令前加一句共用安裝／工作目錄連結；Colab環境格已有依賴版本，不需重複整份安裝說明。

## 17-capstone

實際讀檔：`docs/lessons/17-capstone.md`、`docs/assets/diagrams/17-capstone.svg`、`lesson_cases/17-capstone.py`、`notebooks/17-capstone.ipynb`、`artifacts/checks/curriculum/17-capstone.json`、`artifacts/checks/curriculum/17-metrics.json`。SVG另用Inkscape轉成暫存PNG目視檢查；沒有改正式SVG。

數字核對：32／16／16張、資料seed1100／2200／3300、模型seed7、batch8、每次160步及15,511參數一致。baseline AP為`.333333/.555556`、mAP`.444444`；weight10 AP為`.625/.777778`、mAP`.701389`。baseline TP9、FP5給`9/14=.642857`與`9/17=.529412`；changed TP12、FP3給`12/15=.8`與`12/17=.705882`。coverage`9+4+4=17`及`12+2+3=17`，與表相符。背景FP是validation#10、prediction1、class1、score`.0680081`、IoU0。chosen test mAP`.445238`、precision`.636364`、recall`.538462`相符。

圖文核對：前兩列實際是相同validation#14、#5、#4、#7；baseline難例排名由程式算出。綠框是GT、橘框是實際訓練預測，下方背景FP紅框也來自預測，沒有人工高分框冒充效果。`.25`顯示與`.05`評估的區別清楚，coverage不等於一對一recall也講清楚。單次短訓練／相同幾何分佈／test波動的限制合理。

計時核對：兩次訓練`.506033/.453295`秒與頁面`.51/.45`秒相符；單張計時為`1.052933ms`，3次warmup、12次取樣，包含RGB轉tensor、forward、decode/NMS與PIL畫框，不含檔案讀取。`draw()`還包含放大到192×192及畫GT；這是本案例的標註視覺化範圍，不能拿來代表純產品推論。整個case的`process_wall_seconds=3.3507`包含imports、輸出與檔案，與上述計時口徑不同，紀錄有說清楚。

程式／notebook／紀錄核對：notebook實驗格與case全文相同，case SHA與紀錄相符，notebook stdout與紀錄完全相同。練習已用`main(changed_weight=2)`實跑：`box_weights=[5,2]`，baseline validation mAP`.444444`、changed`.471528`，因此`keep_change=True`；changed FP為background2、localization3。斷言、report與圖題跟著參數改變。網頁沒有預設「降權重必然變差」，讀者可以依當次結果交付，練習可操作。

發現與建議：

- **P2，類別數字在本頁沒有接上顏色。** 原句「辨識…紅、藍矩形」，但表只寫「class0 AP50／class1 AP50」、圖標籤為`0:1.00`等。建議開頭直接補「紅色為class0、藍色為class1」，讓只知道基本CNN的初讀者能從失敗圖解讀表格。
- **P2，必要前置可改為可點的複習入口。** 原句「前置是資料切分、grid detector、分項loss、AP50與完整圖片推論」不給課名或連結。建議至少連07-loss、07-heldout、06-evaluation及07-inference。概念本身正確，不需新增數學推導。

本節未發現結果數字、人工框來源或練習斷言的阻斷問題。

## 18-video

實際讀檔：`docs/lessons/18-video.md`、`docs/assets/diagrams/18-video.svg`、`lesson_cases/18-video.py`、`notebooks/18-video.ipynb`、`artifacts/checks/curriculum/18-video.json`、`artifacts/checks/curriculum/18-metrics.json`；輔助讀`miniyolo/geometry.py`。SVG另轉暫存PNG目視檢查。

數字／座標核對：12幀、20FPS、來源HWC`[64,96,3]`，末幀`11/20=.55s`、總播放`12/20=.6s`、GIF50ms相符。letterbox取整後為43×64，上padding10、下11；`scale_xy`實際是`(64/96,43/64)`，undo函式用此實際比例還原，沒有錯用單一理論2/3。輸入`[1,3,64,64]`與raw`[1,4,4,7]`相符。

預測與圖核對：每幀候選數`[1,1,0,0,0,0,1,1,0,0,0,0]`，共4幀有框、8幀無框；預設12幀都有紅色物件，因此8幀確有漏檢。模型從零訓練，黃色框是forward/decode結果。SVG選0／5／11幀，來源時間0／.25／.55s、框數1／0／0相符；圖上的單幀pipeline3.92／.89／1.09ms沒有被當成median或來源FPS。頁面沒有將低分候選數宣稱準確率。

計時核對：report的前處理`.186751`、model`.198889`、後處理`.367443`、畫框`.048543`、total`.840675ms`，與頁面三位小數相符。只排除首幀，沒有含來源迭代／影片解碼；median相加與total median不同、20FPS來源不代表1190FPS相機等限制寫得清楚。

notebook實驗格／SHA／stdout與case及紀錄一致。練習已跑`main(count=24,fps=20)`：末幀1.15s、總1.2s、GIF50ms，最後物件可見值false；靜態圖選0／11／23。`x=4+4i`在20／21／22幀為84／88／92，14畫素寬逐步裁切，23幀x96完全離場，網頁答案正確。來源fps10的數學也正確：每秒40畫素、末幀1.1s、总1.2s、100ms播放。

發現與建議：

- **P2，真影片入口缺可直接執行的消費端。** 原句「可用`run_stream(opencv_frames('clip.mp4'),model)`替換來源」。`model`只在`main()`內建立，外部貼這行會沒有model；而裸generator呼叫不會開始處理。前段已解釋yield，但初讀者仍缺完整接線。建議補最小片段`model=fit_detector()`，接`for result in run_stream(opencv_frames('clip.mp4'),model): ...`，用print或逐幀輸出消費；說明檔案位置及OpenCV安裝命令。避免直接把main的`list()`移植到長影片。
- **P2，前置letterbox加座標章連結。** 原句「前置是完整圖片推論與letterbox」缺連結；本節shape和還原寫清楚，連到04-coordinates與07-inference即可。

本節預設CPU範例與自主練習可操作。真影片／相機adapter尚未實測，頁面明示，不將其當已驗證能力。

## 19-tracking

實際讀檔：`docs/lessons/19-tracking.md`、`docs/assets/diagrams/19-tracking.svg`、`lesson_cases/19-tracking.py`、`notebooks/19-tracking.ipynb`、`artifacts/checks/curriculum/19-tracking.json`。SVG另轉暫存PNG目視檢查。

數字核對：六幀A的x1為16／24／32／40／48／56；B為40／32／24／16／8／0，框寬高12，第4幀B被刻意漏掉。交叉時同物件IoU為`48/(144+144-48)=.2`，相反身份框IoU1；last-box換ID符合幾何。反貪心IoU`.818182/.666667/.538462/.25`與表一致，對角總1.068182、交叉1.205128，exact搜尋以有效配對數再IoU總和排序，確有unmatched分支，沒有假稱Hungarian。

結果核對：last-box IDs`[[1,2],[1,2],[2,1],[2,1],[2],[2,3]]`，motion IDs`[[1,2],[1,2],[1,2],[1,2],[1],[1,2]]`；由GT物理身份的上次被觀测ID計switch，分別3與0。第5幀last-box舊ID1未過期但IoU0，網頁已正確說明不是max_age刪掉造成。人工detections為11／12、FP0，不是神經網路學得成績。

圖文核對：色彩表示track ID，上／下兩路一致使用相同detections；A/B顯示分列、matching y相同已註明。圖中交叉後上列換色、下列保持色，最後B上列紫色新ID3、下列藍色ID2，符合輸出。notebook實驗格、SHA及stdout與紀錄一致。

練習已跑`main(max_age=1)`：last-box仍3 switches，motion1 switch，motion最後`[1,3]`；圖title／desc／列標題也跟著變成1。無需backward的理由、等速與漏檢邊界、不能外推正式MOT品質，都講清楚。

發現與建議：

- **P2，輸入對象寫反。** 原句「detector輸入是人工已知的框序列，以隔離association」。此case沒有執行detector，人工detections交給的是tracker；detector通常輸入影像。建議改「tracker的輸入是人工給定的detections框序列，本節不執行detector」，避免初讀者把影片→detector→tracker的方向讀錯。
- **P3，錯字。** 「區域性最大演演算法」應為「區域性最大演算法」。這一段數學對，文字修正即可。
- **P2，前置matching應連評估／影片頁。** 不需假定讀者已知association，可先補一句「association就是把當前框配給既有track」，再連06-evaluation及18-video。

本節沒有發現ID計數、max_age練習或人工框來源的數值問題。

## 20-deployment

實際讀檔：`docs/lessons/20-deployment.md`、`lesson_cases/20-deployment.py`、`notebooks/20-deployment.ipynb`、`artifacts/checks/curriculum/20-deployment.json`、`artifacts/checks/curriculum/20-metrics.json`、`artifacts/checks/curriculum/deployment-gpu.json`；為核對GPU計時範圍兼讀`miniyolo/deployment_gpu.py`。本頁沒有SVG，不捏造圖審查。GPU程式只閱讀，沒有執行。

CPU數字核對：一步SGD後export，opset17、legacy`dynamo=False`、只有batch動態。三份非正方形RGB由亂數產生，B1／2／3確實為不同有效batch，raw最大差全為`4.76837158e-7`。decode與原圖還原的boxes／scores／labels assertions都在程式內；80×80拒絕是實際ORT InvalidArgument，沒有只宣告不支援。notebook實驗格／SHA／stdout與case紀錄一致，初始化格也確實裝ONNX1.19.1與ORT1.23.2。

CPU計時核對：torch raw`.1502505ms`、ORT raw`.0362195ms`，比率約4.148倍；torch全流程`2.2687715ms`、ORT全流程`2.128991ms`，時間減少約6.16%。ORT B2`.042063ms`，`2000/.042063=47547.726 images/s`，與47,548相符。兩backend全流程共用相同preprocess/restore，包含raw轉numpy／tensor的介面成本，不含畫框、解碼影片、螢幕或湊batch等待，不能與17／18的畫框計時直接排行。warmup3、取20次median相符。

GPU數字核對：既有JSON為L4、PyTorch2.9.1+cu128、CUDA build12.8、TensorRT10.13.3.9、15,511參數、40更新，loss`.984767→.0710477`與頁面`.9848→.0710`相符。FP32／允許FP16都`tf32_enabled=false`；B1–4最大raw差都是`4.76837158e-6`，最大框差分別`1.14440918e-5`及`7.62939453e-6`。各batch非空候選數2／4／6／7、類別及順序吻合；B1 median`.1602175/.1616085ms`與`.160/.162`相符。表的差值是B1–4最大值，時間只取B1，建議表下注明這個範圍。

GPU計時程式確實含shape/address設定、輸出配置、enqueue與stream/device同步，沒有CPU拷貝／decode。JSON client wall`223.405937s`与223.41s相符；Volume另container讀取的hash与engine/ONNX hash相符，既有停止紀錄tasks0。僅就保存紀錄確認，未查外部服務現況。允許FP16不保證逐層FP16、不同權重不能CPU/GPU比速度、INT8未測等限制清楚。

練習負例已在暫存執行：只移除第三source，B3輸入shape assertion如預期失敗。沒有把兩張的`batch[:3]`當三張；練習可操作。

發現與建議：

- **P2，部署工具首次出現缺角色解釋。** 原句直接說「真的匯出ONNX、執行checker與ONNX Runtime CPU」，後面即有backend、opset、engine與optimization profile。基本PyTorch讀者可能不知這些是檔案格式、執行器還是模型。建議開頭加三句：ONNX保存模型運算圖；ONNX Runtime（ORT）讀它並執行；TensorRT在NVIDIA GPU建置／執行engine。初次出現opset再說是ONNX運算子版本，profile是engine允許的輸入shape範圍。
- **P2，FP16 CLI與下方API實測的TF32設定不同。** `trtexec ... --saveEngine=grid-fp16.engine --fp16`缺`--noTF32`，而實測兩種engine都停用TF32。CLI確已明說未實測，沒有偽造證據；仍建議FP16命令也補`--noTF32`，或明說它與下表不同，才是讀者可比較的單項精度變更。
- **P2，CPU report的limits容易被誤讀為全章沒做GPU。** 原文`no TensorRT/GPU verification`位於本輪輸出，實際是本CPU case沒做GPU；下節另有GPU紀錄。建議改為「this CPU case does not run TensorRT/GPU; separate L4 evidence is documented」，讓離開網頁只讀JSON的人也能理解範圍。
- **P2，數學記法補一句即可。** 頁面直接使用`4.77e−7`、`atol`、`rtol`。建議`e−7`初次出現時給十進位`.000000477`，說atol是固定允許誤差、rtol按參考值大小容許誤差；這會幫數學不熟的讀者理解checker通過與數值接近是不同層。

本節未發現數據冒充、dynamic batch假測試或GPU計時口徑與程式不吻合的問題。

## 複查待辦

作者修訂後，重新核對以上問題句、前置連結、20精度命令與工具解釋。若變動case或notebook，須確認實驗格同步、受影響紀錄SHA與stdout同步；只有文字變動不需重跑所有實驗。尚未把作者修訂視為本reader已通過複查。

## 第一次修訂後複查：共用頁

重新實讀`docs/index.md`、`docs/status.md`、`docs/preparation/data.md`，並讀`docs/research/foundation-data.md`新增的早期設計說明。

- **index、status的GPU摘要：通過。** 首頁已改為「已有L4短訓練、checkpoint恢復與TensorRT數值核對」，完整真實資料／架構品質效能仍待做；status將L4證據獨立列為一列。初讀者現在能在入口分清已實測範圍與未完成實驗。
- **data主要現況／規劃問題：已修。** 開头明列ShapeDataset、Fashion loader、簡化模型及42課已實作；表首列不再說沒有訓練程式；generator末段明列已做紅藍矩形與尚未全部實作的困難集，非顏色類別是延伸規格。foundation研究頁也明說早期三類构想與當前兩類不同，不再容易誤當實作契約。
- **data仍有兩句需清理。** 再散布區仍寫「自製資料：未來由本專案生成」，應改現在式並保留seed／split規格；Penn-Fudan段「分類與偵測兩份…轉換 adapter 與訓練留待後續」可被讀為Fashion也未訓練，應明說後續僅指Penn-Fudan的真實偵測adapter與訓練。這兩句不影響17–20已執行證據，但會讓單讀資料準備頁的人再次混淆。

17–20的本頁修訂尚待作者完成，再作獨立複查。

## 第二次修訂後獨立複查：全報告

重新實讀`docs/index.md`、`docs/status.md`、`docs/preparation/data.md`、`docs/research/foundation-data.md`、`README.md`及17–20四篇教材。另重新核對四份case、四份notebook、四份`artifacts/checks/curriculum/<id>.json`、17／18／20-metrics、三張SVG的來源／圖題／標籤，以及`deployment-gpu.json`。保留上方初讀與第一次複查，不用作者所稱「已補」代替實際核對。

### 已解決的原問題

- **共用GPU狀態：通過。** 首頁及status仍清楚分開已完成的小L4管線證據與未完成的真實資料／正式品質效能對照，没有退回「所有GPU未做」的說法。
- **資料準備現況：通過。** 自製資料現已改為「目前幾何資料已由本專案生成」；Penn-Fudan段明确只把真實偵測adapter／訓練列為待後續，同句說Fashion分類已實測。首次複查剩餘兩句都已解決，當前兩類矩形與研究頁早期三類構想也有區別。
- **17類別與前置：通過。** 新增紅=class0、藍=class1與`0:1.00`含義。四個前置連結07-heldout、07-loss、06-evaluation、07-inference都指向存在的教材。新增計時包含192×192放大及GT標註，與`draw()`一致。色號說明目前在練習後才出現；建議移到結果表／圖之前，屬閱讀順序的P3建議，內容已正確。
- **18真影片consumer：通過。** 新片段先建立`model=fit_detector()`再用`for result in run_stream(opencv_frames('clip.mp4'),model)`逐幀消費，print有實際存取結果；不再只有裸generator呼叫。檔案位置、OpenCV安裝命令、先在notebook執行完整case定義函數，以及adapter未實測都已明說。合成模型不能辨識任意真實物件的限制也有補充。本次未開真影片或相機，不把接線文字判為OpenCV實測。
- **19tracker輸入／錯字／matching前置：通過。** 開頭明说人工detections交給tracker、本節不執行detector；「演演算法」已修；association定義及06-evaluation／18-video兩個前置連結已補且檔案存在。ID結果與max_age定義沒有受文字修訂影響。
- **20工具／誤差名詞：通過。** ONNX格式、ORT執行器、TensorRT engine、backend、opset與profile在契約前已有角色解釋。`.000000477`與`4.77e−7`一致；`abs(actual−reference)≤atol+rtol×abs(reference)`正確說明固定及相對容差，沒有把接近說成位元相同。
- **20FP16 TF32／L4表範圍：通過。** FP16 CLI已補`--noTF32`，與既有Python API兩條engine設定一致；CLI仍明說未實測。B1–4最大差與僅B1計時已明确區分，没有把CPU一步權重和GPU40步權重混比。
- **20CPU limits：通過。** case、notebook、CPU紀錄、20-metrics與網頁輸出都改為「this CPU case does not run TensorRT/GPU; separate L4 evidence is documented」，不再讀成整章未做GPU。
- **四課本機安裝入口：內容已補。** 四頁都連README固定依賴及repo根目錄，Colab環境格前置也明說；但Python命令仍有下面的新問題。

### 修訂後數據與同步核對

四課均逐一確認：notebook完整實驗格與case全文相同、case SHA與對應紀錄相同、notebook stdout與紀錄stdout相同、網頁evidence block與紀錄stdout相同，紀錄`passed=true`、`exit_code=0`。17、18、19的核心結果與初讀相同；三張SVG仍是實際17 weights5／10難例、18 frames0／5／11及19 switches3／0，没有因文字補充變成人工預測。

20最新紀錄執行時間為`2026-10-02T17:08:12.380897+00:00`，case SHA為`af61839e8af678ab18d115b0a1b56745d6095a6122b3d40ebfd356f3e7854a64`。`20-metrics.json`與stdout內的JSON完整相同，raw最大誤差仍為三次`4.76837158e-7`、B1／2／3與80×80拒絕仍通過。

當次torch／ORT raw為`.1525845/.0364245ms`，比值`4.18906`，正文約4.2倍正確；端到端為`2.3286365/2.1442895ms`，減少`7.91652%`，正文約7.9%正確。B2為`.042584ms`，`2000/.042584=46965.9968 images/s`，正文46,966正確。新計時沒有沿用初讀的6.16%或47,548數字。

### 剩餘／新發現

- **P2，18前置複習連結仍缺。** 開頭仍是「前置是完整圖片推論與letterbox」，全頁尚無04-coordinates／07-inference連結，原報告這一項未解決。建議與17相同在開頭直接連到對應前置。
- **P2，四課命令與README的虛擬環境路徑不一致。** README的安裝及執行全用`.venv-model/bin/python`，沒有activate步驟；四課卻仍給`PYTHONPATH=. python lesson_cases/<id>.py`。初讀者逐字安裝後執行教材命令，可能落到未裝依賴的系統Python。建議四課命令直接改成`.venv-model/bin/python`，或明确要求先啟用虛擬環境，再使用python。這是實際命令前置問題，不是模型數值問題。

本次沒有必要重跑只有文字／limits變更的整套case；先前四項練習實跑證據與最新同步檢查已足以核對其邏輯。剩餘两項修訂後再確認，尚不記為全報告全部通過。

## 最終定點複查

重新實讀`README.md`的CPU本機執行段、17與18教材開頭，並確認四課保留README安裝入口。此次只追前輪剩餘問題與色號位置，不以新章節或未授權GPU實驗擴大範圍。

- **18前置連結：通過。** 開頭現為「[完整圖片推論](../docs/lessons/07-inference.md)與[letterbox座標轉換](../docs/lessons/04-coordinates.md)」對應的教材內連結，實際目標`07-inference.md`與`04-coordinates.md`存在，讀者可直接回查。
- **四課本機Python環境：通過。** README明說若要照網頁`python ...`命令執行，先`source .venv-model/bin/activate`；也可一直使用`.venv-model/bin/python`。Windows完整直譯器路徑及PowerShell PYTHONPATH另列。實際啟用環境後，`sys.executable`為`/workspace/learn_to_yolo/.venv-model/bin/python`，PyTorch2.9.1+cpu、ONNX1.19.1、ORT1.23.2可匯入。現在README與四課命令前置一致，未重跑／覆寫正式圖。
- **17色號位置：通過。** 紅=class0、藍=class1及`0:1.00`說明已移到baseline之前，早於結果表與圖，前輪P3順序建議已解決。

四份case SHA仍與各自實測紀錄吻合，本次定點文字修訂没有造成新的程式或紀錄不同步。至此，本報告初讀、第一次及第二次複查列出的問題均已核對解決，17–20與共用頁在本reader的範圍內通過。這是教材可讀性／一致性與指定CPU練習的審查結論，不替代真人學生學習驗證，也不新增對相機、真實影片、INT8或真實偵測品質的實測聲稱。
