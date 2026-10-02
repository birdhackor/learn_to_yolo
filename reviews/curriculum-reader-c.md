# Curriculum reader C：第一輪獨立新讀者審查

審查日期：2026-10-02。讀者假設：懂基本 Python、PyTorch、NN、CNN，數學不熟，事前不知道這個專案。依指定順序逐節閱讀；沒有讀取或引用既有 `reviews/*.md` 的結論。本報告只記本輪初讀，之後的修訂應另加複查紀錄，保留初讀問題。

`lessons-v0.2.0` 視為待發布的教材版本，沒有把目前尚未發布 tag／Colab URL 當作內容缺陷。沒有啟動 GPU、下載資料或對外發送訊息，也沒有修改教材、程式或實測資料。

檢查方法：七節的 case SHA256 全部等於各 JSON 的 `case_sha256`；七份 notebook 第 3 格的程式與 case 全文一致，保存 stdout 與 JSON 逐字一致。另用 `/workspace/learn_to_yolo/.venv-model/bin/python`、`CUDA_VISIBLE_DEVICES=''`、`PYTHONPATH=/workspace/learn_to_yolo`，在 `/tmp` 重跑七個 case，全部 exit 0，stdout 與現有 JSON 逐字一致。四份相關 SVG 用 Inkscape 渲染到 `/tmp` 後實際看圖。40 步擴充只讀腳本、JSON 和圖並重新計算 IoU，沒有執行會覆寫原紀錄的腳本。

嚴重程度：P2 表示會阻礙此讀者跟做或誤解跨節比較，建議修正；P3 是可改善的教學清楚度，不表示公式或實測結果錯誤。

## 1. 08-own-images

實際閱讀：`docs/lessons/08-own-images.md` 全頁、`lesson_cases/08-own-images.py` 全文、`notebooks/08-own-images.ipynb` 全部四格與保存輸出、`docs/assets/diagrams/08-own-images.svg` 原始結構及渲染圖、`artifacts/checks/curriculum/08-own-images.json`。另讀 `miniyolo/geometry.py` 的 letterbox／undo、`scripts/detect_image.py` 的 loader、CLI 與保存流程。

數字核對：120×80 → 64×43，`sx=0.533333…`、`sy=0.5375`，上／下 padding 10／11；框 `[20,10,60,30]` → `[10.666667,15.375,32,26.125]` → 原框，誤差低於 0.000004 pixel。整張原圖框對應 `[0,10,64,53]`。SVG 原圖使用 2 倍畫圖比例，輸入使用 3 倍比例，紅框的位置及大小與上述變換相符。案例完成三步實際更新、重載 logits 完全相同、尺寸維持 120×80、類別／配置不一致被拒絕；16 個候選明確只是管線證據。

閱讀判斷：RGB／HWC／CHW、原圖與輸入 pixel、metadata 順序、空 GT 的意思都有實例可跟。先前章節已有 logits、GT、NCHW 等定義。沒有把照片推論當作學會新類別；checkpoint 的封裝與裸 state_dict 差別也已解釋。整圖框自主練習可以手算。

問題與建議：

- **P2：Colab 的疊框圖片與 JSON 無法在執行後查閱。** `lesson_cases/08-own-images.py:18` 把整個實驗放在 `TemporaryDirectory()`，第 58–64 行保存到該目錄卻只印「saved」，離開 `main()` 即刪除。notebook 第 3 格相同。頁面第 31 行要讀者核對「輸出原圖座標 JSON 及疊框 PNG」，但沒有檔案路徑、顯示圖片、JSON 內容或下載入口。建議 checkpoint／測試 fixture 繼續暫存，將預測 PNG／JSON 複製到持久的 `artifacts/predictions/08-own-images/`，印路徑，notebook 加一格顯示 PNG 和 JSON；也可在暫存清除前直接顯示並說明不會保留檔案。
- **P3：重載用字。** 頁面第 7、31 行「過載 checkpoint／過載前後」应改「重載」，避免讀者以為是另一種權重操作。

## 2. 08-own-data

實際閱讀：`docs/lessons/08-own-data.md` 全頁、`lesson_cases/08-own-data.py` 全文、`notebooks/08-own-data.ipynb` 全部四格與輸出、`artifacts/checks/curriculum/08-own-data.json`，以及 `miniyolo/custom_data.py`。本頁沒有引用 SVG；已查 `docs/assets/diagrams/`，沒有本節專屬 SVG，不將此視為缺圖。

數字核對：三個 source 各兩張，共六張；train／validation／test 都是 2 筆。train 第 0 張是 R=G、B=0 的黃矩形，第 1 張全黑、框 `[0,4]`。同步變換框仍為 `[10.666667,15.375,32,26.125]`，label 2 不變。兩張 batch `[2,3,64,64]`，三類 head `[2,4,4,8]`，一步 loss 1.4407；空圖維度、錯列長、bool label、NaN、實圖尺寸不符與來源洩漏都被案例拒絕。四類最後軸 `5+4=9` 正確。

閱讀判斷：JSON 契約、原圖尺寸、空圖、各軸、來源切分與 class id 索引定義足够明確；Dataset 實際驗證所有 split 後才篩 train，與頁面一致。已區分人工來源 fixture、真實檔案讀取與真實資料效能。需要重建模型／optimizer 的理由可跟。來源洩漏自主練習可直接修改 `bad[1]['split']` 重現；新增類別的 shape 練習可算。

本節未發現必須修正的數字或程式矛盾。可選改善：頁面第 43 行的 `annotations.json` 是相對目前工作目錄，不是自動相對 `root='my-data'`；若預期讀者把 JSON 與圖片一起放進 `my-data`，加一個最小目錄樹，並用 `JsonDetectionDataset('my-data/annotations.json', root='my-data', ...)` 示範，可減少換成自有資料時的路徑猜測（P3）。

## 3. 09-anchors

實際閱讀：`docs/lessons/09-anchors.md` 全頁、`lesson_cases/09-anchors.py` 全文、`notebooks/09-anchors.ipynb` 全部四格與輸出、`artifacts/checks/curriculum/09-anchors.json`。本頁未引用 SVG，圖目錄也沒有本節 SVG。為判斷既有術語是否已定義，另查 `07-targets.md`、`07-loss.md` 的 logits／sigmoid／class／mask 說明。

數字核對：框中心 `(16,20)`、wh `(16,16)`，cell `(1,1)`、比例 `(0,.25)`；兩尺寸 IoU 1／.25；32 槽是 1 positive + 1 ignore + 30 negative。encode 的 logits 是 `[-9.210240,-1.098612]`，decode x 的兩端都比原框多 .0016003 pixel，吻合 clamp 的 `16×1e-4`。忽略槽的全部梯度與非正槽框梯度為 0，SGD 真正改變 raw 參數。練習的 `[ln2,0]`、`[ln4,ln2]` 正確。

閱讀判斷：anchor 是 wh 起點、與 class 軸獨立、pixel 單位、比例与 logit 兩種 target、ignore 不進 loss 的區別講得清楚。clamp 的邊界近似已明講，沒有假稱完整 YOLOv2 或 AP。

問題與建議：

- **P2：核心新數學 `exp`、`log`／`ln` 未先給白話定義。** 頁面第 11 行先用 `exp(tw)`，第 13 行用 `log`，第 38 行答案換成 `ln`。對不熟數學的讀者，.693147 是已公布答案，仍不清楚為何寬能回復 2 倍或 `log` 和 `ln` 是否同一種。建議首次公式旁說「這裡 `log` 就是自然對數 `ln`；`exp(t)` 把 log 尺寸修正還原成倍率，兩者互相抵消」，再走 `tw=ln2≈.6931 → exp(tw)=2 → 16×2=32`；不需要微積分。
- **P3：更新的是可學槽位表，尚未接圖片網路。** 第 32 行稱「槽位學習實驗」，但 `case:29` 的 `raw` 是直接的 `nn.Parameter`，沒有 GridDetector、圖片或 backbone。建議補一句「這一步直接學 raw 槽位張量，不是已接上 anchor head 的圖片偵測器」，讓「本次僅替換框表示和槽位責任」的尺度更具體。

## 4. 09-anchor-clustering

實際閱讀：`docs/lessons/09-anchor-clustering.md` 全頁、`lesson_cases/09-anchor-clustering.py` 全文、`notebooks/09-anchor-clustering.ipynb` 全部四格與輸出、`artifacts/checks/curriculum/09-anchor-clustering.json`。本頁未引用 SVG，圖目錄也沒有本節 SVG。

數字核對：8×8 vs 9×8 的 IoU `64/72=.888889`，共同比例乘 2 不變。分群 `[0,0,0,1,1,1]`，均值中心 `[25/3,25/3]`、`[94/3,50/3]`。以两个16×16為基線，mean best IoU `0.381712963 → 0.911871608`；median `[8,8]`、`[32,16]` 是 `0.934027778`。新來源兩個極端 wh 的覆蓋 .1975 與 JSON 相符。手算 64→128 時 anchors 乘 2 的練習正確。

閱讀判斷：尺寸群編號与 detector 正負責任已分開；只用 train、實際前處理後 wh、mean 不保證最小化 IoU 距離、空群保留舊值，都有說明。沒有把尺寸覆蓋當 AP。六筆 mean 中心的計算可跟。

問題與建議：

- **P2：基線与前一節的 anchor 組不同，容易誤讀成連續改版的改善幅度。** 第 5 行說「只改 anchor 尺寸的選法，保留…模型分支」，但 `case:33`／頁面第 36 行基線是 `[[16,16],[16,16]]`；上一節的 anchor 是 `[[16,16],[8,8]]`。後者在同六筆資料的 mean best IoU 是 **0.709259259**，不是 .3817。現有數字本身沒有算錯，問題在跨節比較。建議主比較使用上一節配置，顯示 `.7093 → .9119`，或者明講「本節刻意另設兩個相同16×16的弱基線，.3817不是上一節配置的結果」，並另外列上一節結果。
- **P3：覆蓋指標与 median 可再先定義一次。** 第 26、36 行直接用逐維 median、mean best size IoU。建議明講「median 是把每一維各自排序取中間值」、「每個 wh 先挑 IoU 最大的 anchor，再平均六筆最大值」，用前三筆寬 `[8,9,8]` 排序成 `[8,8,9]` 示範；這會讓 .9340 与 .9119 成為可追算的量。

## 5. 10-multiscale

實際閱讀：`docs/lessons/10-multiscale.md` 全頁、`lesson_cases/10-multiscale.py` 全文、`notebooks/10-multiscale.ipynb` 全部四格與輸出、`artifacts/checks/curriculum/10-multiscale.json`、`artifacts/checks/curriculum/10-multiscale-learning.json`、`scripts/run_multiscale_learning.py` 全文、`docs/assets/diagrams/10-multiscale-learning.svg` 的 XML 曲線／框位置和渲染圖。另讀 `miniyolo/inference.py`、`miniyolo/metrics.py`、`miniyolo/models.py`，及 `scripts/run_learning_extensions.py` 的 `load`。

數字核對：小框中心9，在stride16是cell0、offset .5625，在stride8是cell1、offset .125；wh 在兩者都是8/64=.125。大框中心44，coarse cell `(2,2)`，fine未負責位置 `(5,5)`。兩 head `[1,8,8,7]`／`[1,4,4,7]`，80 候選、560 個 logits；相對單個4×4 head 是16 候選、112 logits。人工同框跨尺度合併2→1。40 筆 loss 的首末 3.659366／.07033685、delta 6.503563 与頁面相符；SVG 曲線有40點、綠框與兩 GT 相同，三個橙框與 JSON 坐標相同。真實小框 IoU **.441043**，大框 **.862990**，額外 class1 框對大 GT **.299122**；因此 AP0=0、AP1=1、mAP50=.5、precision 1/3、recall1/2 符合。

閱讀判斷：stride 已定義且與感受野區分；細／粗特徵、head channel、單框與兩類、尺寸分配不按類別、未分配物件所在格仍學背景、跨尺度 NMS 都有具體步驟。已清楚區分人工 decode 驗證與40步真正模型輸出，並直說小框未達 .5、沒有 held-out 或單尺度品質對照。自主練習可按表格算。

可選改善：

- **P3：本節是保留 grid 表示的新 toy 模型，並非對第7章 GridDetector 原封不動加一個 head。** 第 5、7 行的「只增加…head」「起始是第7章grid分支」可能讓讀者直接找舊 model 插入點。實際 `GridDetector` 有 `AdaptiveAvgPool2d`、32→32卷積和 obj／wh bias 初始化；TwoScale 改成16通道的第三層及stride2 deep，沒有同樣 bias 初始化。建議明講沿用的是 target／loss／decode 契約，backbone 另用本頁小模型；若要正式只測多 head，应以相同 TwoScale backbone 去掉 fine head 作基線。
- **P3：40步的 Colab 命令能保存圖，沒有顯示圖的 cell。** 第 74 行 `!python scripts/run_multiscale_learning.py` 只印 JSON，圖寫入 repo。加 `display(SVG(filename='docs/assets/diagrams/10-multiscale-learning.svg'))`，使讀者在改步數後能立即核對新曲線與框。現有網站圖本身沒有問題。

## 6. 11-csp

實際閱讀：`docs/lessons/11-csp.md` 全頁、`lesson_cases/11-csp.py` 全文、`notebooks/11-csp.ipynb` 全部四格與輸出、`docs/assets/diagrams/11-csp.svg` 結構與渲染圖、`artifacts/checks/curriculum/11-csp.json`。為核對前置定義，另查 `01-small-cnn.md` 的參數公式和 `03-identity.md` 的梯度／identity說明。

數字核對：8通道切成4+4，concat／fuse 前後 shape 不變；a／b concat 值 `[1,2,10,20]`、add `[11,22]`。Full `1168+72=1240`，CSP `296+72=368`；兩支輸入梯度和分別 Full `.033093/.029087`、CSP `.255476/.019409` 与 JSON 相符，兩次 SGD 真的更新 fuse。C=10 練習：`2×(5×5×9+5)+(10×10+10)=570`。SVG 完整顯示兩條路與8→8、4→4、concat4+4的關係，沒有把 bypass 畫成整個模組 identity。

閱讀判斷：stage、NCHW、梯度、旁路、concat与add、學得的1×1混合都有先定義或數值例子。已講清楚容量不同、非零梯度只支持连通、隨機 feature 平方loss不是 detection/AP；原版 CSPDarknet／C3 与本例的界線明確。手算自主練習可操作。

本節未發現必須修正的內容矛盾。若希望 C=10 練習也要求執行，建议第45行追加修改點：branch 的兩個 Conv2d 都改5→5、fuse 改10→10、input改10channel、前後梯度切片改5，以及更新固定 counts/assertions（P3）；目前當作手算題已经成立。

## 7. 11-fusion

實際閱讀：`docs/lessons/11-fusion.md` 全頁、`lesson_cases/11-fusion.py` 全文、`notebooks/11-fusion.ipynb` 全部四格與輸出、`docs/assets/diagrams/11-fusion.svg` 結構與渲染圖、`artifacts/checks/curriculum/11-fusion.json`，並回看第10節的 TwoScale 通道與連線。

數字核對：reduce16→8有136參數、mix16→8 3×3有1160，合1296。concat `[1,16,8,8]` 的1024個float32元素是4096bytes。nearest 2×2→4×4每值複製四次，sum loss 的來源梯度全4。兩來源梯度非零，SGD真更新。未 reduction 的concat合法為24ch；add匹配後保留8ch。練習 `[2,12,10,10]` concat到 `[2,24,10,10]` 正確。SVG的top-down、lateral、細head接點與「case停在融合特徵」相符，PAN虚線明确只是未实现路徑的圖例。

閱讀判斷：backbone／neck／head、top-down／bottom-up／lateral、nearest 複製與細節來源、空間 shape 与原圖對齊的區別均有說明。FPN原版add与本例concat分開，沒有把PANet、v4、v5當同一模組；本例不接 detector、不能報AP的限制也明講。自主練習可以按shape規則完成。

問題與建議：

- **P2：第10節介面與本節通道對不上，且1296被寫成上一節新增成本。** 頁面第7行說「起始為第10章兩尺度介面」，第39行說「與沒有neck的兩head基線比較，這些是新增成本」。但第10節 fine/deep 是 **16／32ch**，本節與case第9、10、21、22行是 **8／16ch**；直接把第10節的features送入本節Fusion會在reduce收到32而只接受16時失败，fused8也不能送入原fine_head(16→7)。本節1296的計算正确，只适用本節的縮小隨機feature模型。建议明讲「仅借用stride8/16与兩尺度概念，本case把channel縮為8/16，未直接連接第10節」，把1296說成這個toy neck自身成本；或者提供對第10節的實際接線：reduce32→16、concat32ch、mix32→16、原fine_head不變，此neck參數為 `32×16+16 + 32×16×9+16 = 5152`，才能主張只在原兩head模型新增neck。

## 本輪結論

七節案例、notebook与既有CPU紀錄相符，圖中座標／通道／框也對得上，沒有發現公式答案或保存結果造假。主要需要修正的是：圖片輸出可見性（08）、log／exp入門解釋（09）、聚類基線銜接（09）與融合通道／成本的跨節銜接（11）。各節都已將人工機制驗證、短步真更新与實際偵測品質分開，這一界線應保留。

## 修訂後獨立複查（2026-10-02）

本次重新讀取七節中修訂的完整頁面與08圖片case、所有七節的notebook／case／JSON對應關係，並逐项試做；沒有把修訂說明當作通過證據。初讀以上內容未改。本次只追加本報告，測試輸出與SVG渲染都寫到 `/tmp`。

共同核對：七節case的SHA256仍全部吻合JSON，七份notebook中的case程式与保存stdout均仍吻合对应case／JSON。08圖片新case在獨立目錄 `/tmp/c-review-recheck-ivncxagm` 用現有CPU環境重新執行，exit 0且stdout与更新後JSON逐字相同。其他正式case沒有程式修訂，本次沒有再次重跑全部案例；首次独立重跑结果仍适用。

- **08-own-images 原P2與重載用字P3已解決。** case現在建立持久 `artifacts/lesson-08-own-images/`，沒有TemporaryDirectory清除。在子程序完全结束后，我实際重新讀取 `my_image.png`、`prediction.png`、`prediction.json`、`checkpoint.pt`；PNG為120×80，JSON有16框，checkpoint的image_size64／grid_size4／width8／兩類配置可重載。再用同個checkpoint及输入圖重做推論，boxes逐值相同。目視新prediction.png：确实是三步模型的16个候选和重叠文字，符合网页「管线检查、不是照片偵測效果」的限制，未把幾何示意框混作模型預測。网页第63–72行给持久路徑与Image／JSON查看代码，`.gitignore:12`确實排除此目录。网页第7、31行現在均用「重載」。
- **08-own-data 路徑P3已解決。** 网页第43行現在显式用`my-data/annotations.json`与`root='my-data'`，第73行说明图片相对root、JSON本身仍需完整路径。我照新說明在/tmp建立`my-data/images/example.png`与JSON，以本頁JSON契約与所示DataLoader步骤读入，得到`[1,3,64,64]`，保留三類索引，框仍为`[10.666667,15.375,32,26.125]`。B=1来自该最小JSON只有一筆，不是batch_size失效。
- **09-anchors 数学定义P2与槽位表P3已解決。** 网页第9行在decode公式之前先說log=ln、exp为反運算，并走`ln2→2→16×2=32`，另明确「raw槽位表可學、沒有圖片CNN」。独立计算`ln2=.6931471806`、`exp(ln2)×16=32`、`ln4=1.3862943611`，與公式和自主練習相符。
- **09-anchor-clustering 基線P2已解決。** 网页第48行明确标出兩個相同16×16为弱基線、.3817不是上一节结果，并给上一节16×16+8×8的.7093。调用本节size_iou重算得weak=.381713003、previous=.709259212，与说明吻合。該段亦给mean best IoU的「逐笔先max再mean」定义和median排序例子，原P3的含义问题已补足；定义仍位於較後段，移到首次使用前仍是可選排版改善，不构成新的數據錯誤。
- **10-multiscale 跨節模型P3與顯圖P3已处理。** 网页第76行明确只继承第7章target／loss／decode契约、backbone另用TwoScale，并给固定backbone／初始化／预算的coarse-only對照原则；不再要求读者把旧GridDetector视为只加head后的同一模型。补充的SVG显示入口指向真实存在的学习图。重新渲染并目視当前learning.svg，loss曲线、两GT、三预测框与本节JSON一致，小框未达IoU.5的限制仍在，没有升级为质量比較结论。
- **11-csp 本次沒有內容修訂或新問題。** 重读前次明确的手算练习与当前case／notebook一致，当前SVG重新渲染后分流、拼接及参数数字仍清楚。原C=10執行修改点建议保留为可选P3，手算答案570仍成立。
- **11-fusion 跨節通道与成本P2已解決。** 网页第7行现在先說明toy用8／16ch、沒有直接接第10节16／32ch；第39行把1296限定为toy neck，并给实际接第10节的32→16／concat32／mix32→16與5152。为了核对可接性，我在/tmp的独立CPU探针实例化原TwoScale，取得fine`[1,16,8,8]`、deep`[1,32,4,4]`，按新说明接reduce／upsample／concat／mix，得到joined`[1,32,8,8]`、fused`[1,16,8,8]`，交给原fine_head后logits`[1,8,8,7]`；新增卷积参数实数 **5152**。当前fusion.svg仍明确标8／16toy通道与「case停在融合特徵」，目視与修订边界一致。

複查中發現并已修正的新問題也保留紀錄：08-own-data新增目录說明最初写「JSON内`file_name`写images/example.png」，与本节必填`path`的契约冲突。我用同一合法fixture将`path`改成`file_name`，实际重现`ValueError: missing path`并报告根agent；根agent随即更正为`path`，我重新读到第73行更正并验证正确路径读入成功。08-own-images第7行最初仍残留「暂存目錄」，也已重新核对更正为「持久输出目录」。这些新问题目前没有遗留。

验证范围：本地CPU环境没有IPython，因此没有启动Colab runtime或实际执行IPython显示API；Image.open、JSON读档、checkpoint重载及预测图片的目视均已实际完成，SVG也已本地渲染目视。待发布tag没有被视为内容错误。没有GPU、外部消息或正式图／实测文件覆写。

複查结论：初读四项P2均已解决；修订中发现的JSON字段错误也已纠正并试做通过。当前没有未解决的P2。剩余只为聚类定义摆放及CSP手算改成可执行练习的可选P3，不影响本轮内容通过。
