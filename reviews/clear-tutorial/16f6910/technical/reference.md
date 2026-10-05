# reference：非課程導覽頁技術與證據審查

審查者：/root/clear_first_reference，未參與本組教材撰寫。審查時間：2026-10-05 10:22:55 UTC。方法依 .agents/skills/clear-tutorial/SKILL.md 與 references/review-protocol.md 的第二輪；本檔是技術複查，不能代替第三輪銜接閱讀或瀏覽器檢查。

範圍為網站全部 17 個非 lesson 導覽頁及 README.md。先閱讀 git diff 16f69103d1f216d1024a40610623083557324701 的本組變更，再讀目前各頁完整正文、相關實作、保存紀錄與原始來源。以下 SHA-256 記錄審查完成當下的完整檔案，不是 coverage.json 正規化指紋。

本組沒有尚未處理的 blocking 技術問題。TREF-01 至 TREF-05 均由作者修正後獨立核回；兩張本輪 SVG 只有原始結構與內容語意核對，網站桌面／手機呈現未驗證。發布仍須待別位讀者完成第三輪、全站 browser 檢查、最終 tag 配對和正式 coverage 更新；不得由本報告推論新版已可發布。

## 原始首次閱讀紀錄

reference 首輪完成 143／143 單位：首頁及閱讀路線 10 個前提單位，加上 15 個 target 頁的 133 個單位。當場理解、疑問、猜測與圖解需求保存在 first-read/reference.jsonl；SHA-256 為 738984b16c33fa3263715a7d4277022607cd0f65cc2d6f361f05d586088ddc3c。首輪 summary 另存 first-read/reference-summary.md。沒有修改原始疑問或把它們改寫成 pass。

本輪另外核對 source-manifest.json、六組 parts 與 canonical raw：凍結來源 59 頁，六組各為 110、131、148、109、100、143 個單位，共 741 個。這是包含各組補讀前文的揭露單位總數，不能稱 741 個互不重複段落。manifest 的 protocol_amendments 保留 grid 的 07-data 前提漏列、applications 的 AP 前提漏列、08 後段提早收到作者提示，以及 3200 步提示錯誤後更正 160／1600 步的限制。status／curriculum／README 沒有把先前整頁審查重標為逐段盲讀，也沒有宣稱真人學生看懂。

## 發現及修改後核回

| ID | 嚴重程度 | 位置、發現及建議 | 實際核回結果 |
| --- | --- | --- | --- |
| TREF-01 | burden／增加理解負擔 | outline 的固定紅框指向已被改成分類／偵測概覽的首頁。建議指向真的保存數值例子的 7.1。 | 已改為「7.1 節固定場景的紅框」，核對 07-data 的 [8,12,24,28]、中心 (16,20) 與 outline 相符；關閉。 |
| TREF-02 | burden／增加理解負擔 | 初版 release-evidence-flow.svg 的直線流程先 CPU 再 GPU，與正文步驟 4 GPU、5 CPU 及只重跑過期紀錄矛盾。建議分有／無過期，重產支線先 GPU 再 CPU 並匯合。 | 作者修改後核對 SVG：無過期沿用日期；有過期依需要 GPU→CPU→帶回紀錄／圖，兩路匯合到正文數字／審查；圖前解說兩類都過期時的順序。結構語意關閉；browser 未驗。 |
| TREF-03 | optional／選讀改善 | crowd 段落引用配對迴圈，但 ignore 的最直接設定在 _prepare。建議加固定 L106–L109。 | 已加 8c9bcc3 固定連結；獨立取得官方 raw，L108 把 iscrowd 設為 ignore，和段落一致；關閉。 |
| TREF-04 | burden／增加理解負擔 | detection-data small area range 固定連結一度選 L521–L530，實際是 keypoints 設定，沒有 small。建議改 bbox／segm 的 L502–L511。 | 現在確為 #L502-L511；官方 raw L509 的第二個 range 是 [0²,32²]，L510 對應 small；關閉。 |
| TREF-05 | optional／選讀改善 | status／curriculum「整本书」與 publish「這张圖」違反 repo 繁體中文寫法。 | 現在為「整本書」／「這張圖」，rg 不再找到原字串；關閉。 |

## 實際執行的檢查

| 方法／指令 | 結果及支持範圍 |
| --- | --- |
| .venv-model/bin/python -m pytest -q tests/test_notebook_bootstrap.py tests/test_checkpoint.py tests/test_release_verification.py tests/test_core.py | 35 passed，3.66 秒。支持 bootstrap 各分支、CPU 存檔續訓、發布驗證邊界與核心幾何／配對，不支持實際 Colab runtime。 |
| python3 scripts/validate_preparation.py | exit 0；manifest、節次／notebook 配對、摘錄與網站資產邊界通過。 |
| python3 scripts/validate_curriculum_evidence.py | exit 0；42 節紀錄及 11 份補充紀錄與目前程式相符。檢查保存證據，沒有重訓全部實驗。 |
| .venv-model/bin/python scripts/record_evidence.py | 不加參數、exit 0，列出目前紀錄均未過期，沒有執行訓練。程式會在臨時目錄比較部分重畫圖；不能解讀成 GPU 新跑過。 |
| .venv-model/bin/python scripts/check_lesson_runtime.py --section 07-training --report /tmp/clear-reference-lesson-runtime.json | 該節 CPU 完整執行 PASS，stdout 和既有紀錄相同；這次只重新跑這一節，不聲稱重新跑 42 節。 |
| python3 scripts/download_data.py list；fetch coco2017 | list 列出 6 個資料狀態；候選 COCO 下載被預期的 guard 拒絕、exit 2，沒有下載完整 COCO。 |
| python3 scripts/review_coverage.py | exit 1，列出本輪文字／圖變動造成的過期審查。這是工作中預期結果；沒有寫 coverage 或宣稱正式出版檢查已全過。 |
| git rev-list --objects --all 加 cat-file batch 尺寸盤點 | 本機不是 shallow repository；目前已知 refs 的最大普通 Git blob 1,173,007 bytes，是 MathJax。支持不需因既有大 blob 做 LFS 歷史遷移；不外推未知遠端 refs。 |
| miniyolo.metrics._interpolated_ap([False,True,False,True],3) | 1／3，和 glossary 的 3 個 GT、FP／TP／FP／TP、單調 precision envelope／all-point 插值一致。 |
| Fashion-MNIST 4 個既有 gzip 的 bytes、SHA-256、MD5／IDX 內容核對 | 四檔 bytes 26,421,880／29,515／4,422,102／5,148，SHA 和 manifest 相符；資料讀取檢查不等於重新完成 40 步訓練。 |
| PennFudanPed.zip 既有完整快取：SHA-256、ZipFile.testzip、成員計數、readme.txt／FudanPed00001.txt | SHA 9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2；所有 ZIP member CRC 通過，170 張圖／mask／TXT；README 原權利限制和 PASCAL 1.00 TXT 格式相符。這次沒有重新下載整包 Penn。 |
| 官方 Pet annotations 完整下載後僅讀指定 tar 成員、split／XML／trimap 盤點、PNG chunk CRC 和 mask tight bbox | 19,173,078 bytes，SHA 52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91；沒有 extractall，沒有下載圖片。XML 3,686、trimap 7,390、trainval 3,680／test 3,669、trainval 缺 9 XML、test 全缺 XML，皆與兩頁相符。 |
| 官方 VOC／COCO S3／Pet 圖片的 HEAD 抽查 | 回 200，Content-Length 與 manifest／表格相符；只支持端點及宣告大小，不支持未下載大型包的完整性，也沒有重做 2026-10-04 的 HTTP／range 對照。 |

沒有觸發 GPU／Actions、push、發布、修改追蹤教材、覆寫 artifacts/checks 的證據、移動 tag、重新上傳 LFS 或取用私人 HF checkpoint。所有此次新增執行輸出在 /tmp/clear-reference-*，唯一 repo 寫入是本報告。

## 逐頁實際結果

### 1. docs/index.md

結果：沒有未處理技術疑點。完整讀取新首頁與來源 diff；分類回答整張圖類別，偵測增加各物件的類別、位置與數量，並連到閱讀路線與驗證範圍。新圖只展示任務區別，未再假裝是後面 grid 的數值例子。技術範圍和 status 的教學簡化說明一致。

頁 SHA-256：dc8947e5d0d7baf3ef68a74e6631d788344b5891c50f9cea9701e099a501f6a6

圖：docs/assets/diagrams/classification-vs-detection.svg，SHA-256：a050400b9fe7e35a06073ca97ab15ee893ce1fef62bc57236707539a00d1c90e。核對 SVG 內容與圖說；Zensical 桌面／手機實際呈現未驗。

### 2. docs/learning-path.md

結果：沒有未處理技術疑點。完整讀取 NN／CNN 基礎路線與復習入口，對照 section-map.json／outline 的 0–20 章順序。分類→ResNet→定位→人工框→grid→各分支機制的前提安排一致；明列各版本小實驗，未宣稱完整模型比較。沒有另外假設讀者先懂 detector 術語。無直接 SVG；路線表以文字清楚列先後。

頁 SHA-256：7008aa1575e26490fd7d731450975fc6b67fd8ad5bbc9988a5b1b5f00714f825

### 3. docs/glossary.md

結果：沒有 blocking／burden 技術疑點。全頁讀取並對照 miniyolo/data.py、metrics.py 和 fixed 官方 sources 的術語；核對 0-based／半開框、xyxy／xywh、IoU、GT／prediction、背景、配對、score、precision／recall／AP／mAP 的區別。AP 例子額外手算且呼叫實作得到 1／3。現代術語引用依官方固定版本及原論文，不由名稱推論完整效能。無直接 SVG；首輪 AP envelope 小圖需求仍是選讀改善，沒有用本輪技術正確性消除原始需求。

頁 SHA-256：f1f6fbb72f85924b45fd2aaaf663880a308957a7fb6782e84ba13394a8e67b2e

### 4. docs/status.md

結果：沒有未處理技術疑點。全頁對照 42 節 index、11 份補充紀錄、check_lesson_runtime.py、bootstrap 和 validate_curriculum_evidence.py。42／42 是既有逐節執行通過；個別人工框／未訓練模型和泛化沒有混稱。Fashion 40 步及自製 160／1600 步都限定其資料及用途，不能推論真實照片有效。CPU／GPU 環境、只有兩項 L4 檢查、非 Colab 的公開 bootstrap、RNG 與 code hash 邊界都能回查。

本輪首次閱讀方法聲明核對 manifest／raw／amendments，前提與作者提示的限制保留；審查流程的第 2／3 輪與 browser 是本輪發布前仍須完成的工作，不由文字聲明認定全部完成。舊 lessons-v0.4.1 的公開驗證是歷史紀錄；預計新 tag lessons-v0.5.0 尚未發布，不能提前聲稱它已驗證。

頁 SHA-256：7ebd692718a4545decff806fb9578310e2fe4b3df125e9af5d41b976186ef943

### 5. docs/planning/course-research.md

結果：沒有未處理技術疑點。完整讀取研究事實／設計建議／查核限制，重新打開官方 D2L 4 章、Coursera 公開模組／作業名、CS231n 2025 課表及 A2／A3、李宏毅 2021 CNN PDF／HW3 notebook、2022 HW3 PDF、Harvard CS50／CS109B 2018、MIT Lab 2、Visionbook、fast.ai、NYU 2021、Aladdin model／loss／train 與 Ayoosh README，核對實際支持的段落或 cells。

具體核對：MIT PyTorch Part 2 n_outputs=1 是 face／non-face classifier，非 bbox detector；CS231n A2／A3 並非 YOLO 作業；CS50 Traffic 43 類與 3 類練習、README 實驗紀錄要求；Visionbook §24.5 水平／垂直線的域外失敗；fast.ai Part 1 9 lessons；Ayoosh only detection module／既有權重；Aladdin v1 BatchNorm／FC 496。沒有執行參考專案、讀付費 Coursera 作業或補稱 Michigan 503 時的內容。移動網頁以 2026-10-05 取得的內容複查，非固定歷史快照。

頁 SHA-256：6922ad22cf822cf611bfea9092bc20616732010f92ff8471693ba778ba5973d0

### 6. docs/planning/feedback.md

結果：沒有未處理技術疑點。完整閱讀每個心得條目及其推論限制，再讀原始 Discourse JSON 的指定 post／GitHub API 的指定 issue 或 comment。確認作者、日期、卡點和教學建議的區別，沒有把單一抱怨當成模型或課程品質統計。

核對位置包括：FluffyFirefly 2025-02-05 post 5 的 axis=3→axis=-1；Aaron_L 2021-07-18 post 10 的 background 0／-1；YOLO issue 123 cutoff 0.005 vs 0.2；issue 137 reduction／尺度疑問；bubbliiiing comment 1328073285 的預測框與 GT IoU>0.5 ignore 說明。固定 notebook／repo 心得只支持作者實際敘述，不代表獨立重現其訓練。無直接 SVG；個別公開心得無須憑空新增架構圖。

頁 SHA-256：4b805cc1729450bff04d27ae6ef2f6026adde52b80f50dc87d57ae7541524819

### 7. docs/planning/outline.md

結果：TREF-01 已關閉；沒有未處理技術疑點。全頁對照 section-map 的 42 節、learning-path 及目前各章機制範圍；「分布式回歸」已釐清是距離的機率分布，不是多機運算。紅框定位改 7.1 後實際數值一致。候選 RGB 幾何設計與已實作紅／藍矩形／小型模型分開，沒有把尚未實測的資料規模寫成既有效果。無直接 SVG。

頁 SHA-256：3046aed911dcc50ad22a76f6b19ec9ebc9b04fbb3de5a927457af292a5797518

### 8. docs/preparation/architecture.md

結果：沒有未處理技術疑點。完整正文及 data-excerpt 字面 Markdown fence 範例對照 validate_preparation.py、build_lesson_notebooks.py、pages.yml、verify-release.yml、requirements 與 gitignore。tag 在首次使用處解釋為已發布固定版本標籤。網站建置不需 PyTorch／GPU／資料；notebook 最後一格與 lesson_cases 配對；實驗輸出與 tracked evidence 的路徑邊界符合實作。

validate_preparation.py 實跑通過，支持新摘錄示例仍符合檢查。無直接 SVG；目錄表及頁面↔程式↔notebook 的文字關係足夠，不以未做 browser 檢查聲稱版面通過。

頁 SHA-256：ce1c28d2ce406227575fc611a617f19a42cc1fcc0c665788a5049bf422398305

### 9. docs/preparation/data.md

結果：沒有未處理技術疑點。全頁核對 miniyolo/data.py、classification.py、download_data.py、manifest 和 Fashion 40 步 JSON；抽查實際 gzip、Penn ZIP，獨立重取 Pet annotation 包及 COCO／VOC 官方說明。候選資料與已用資料分組、合成圖 colour／class 限制、COCO xywh 左上原點／單位／category mapping／iscrowd 已說明，crowd 直接連到本頁所需官方評估規則。

Fashion 驗證 18／128=0.140625、test 12／128=0.09375、10 類均勻 loss ln(10)≈2.302585；保存 loss 約 2.109–2.386。這只證明 40 步管線和權重更新，正文沒有稱學會 10 類。Penn／Fashion 合計 84.60 MB 與 manifest 一致；COCO／VOC HEAD 不冒稱整包驗證。

頁 SHA-256：293bd94b298639d544845474106009fadfc357efa08bbdcf78efd25d207a313e

### 10. docs/preparation/publish.md

結果：TREF-02、TREF-05 已關閉；沒有未處理技術疑點。完整 11 步流程、命令、bring-back 表格對照 bootstrap、record_evidence.py、verify_curriculum.py、review_coverage.py、verify_release.py 與五個 workflow。只有 workflow_dispatch，push 不自行發布或租 GPU。新增 git diff --cached 能查看真正 staged 內容；沒有 force-push／覆寫已發布 tag 的指令。

實作核對重點：bootstrap clone 到包含 tag 的 learn_to_yolo_<tag>，並 os.chdir 到該 checkout，因此新增「已執行環境格」分支省略 clone／cd 的說法正確。record_evidence 無參數只列過期／比較圖，--run 固定環境重產 CPU；GPU remote head 必須等於本機 commit。GPU 失敗保存 failed 紀錄後不可直接視為成功，正文要求處理／還原。--render-only 只回填頁尾／notebook／彙總，不帶回實驗圖，bring-back 表格正確。coverage 的正文 fingerprint 包含數字但不核算數值一致，須人工搜尋紀錄檔名和舊數字。

頁 SHA-256：fd842d0b9dd3c85ee8c266439d79153908b6c2de7f6a489ed539f2250c4b792d

圖：docs/assets/diagrams/release-evidence-flow.svg，SHA-256：daf67652ce9f56282fb58571cd670bac0e0d572c8509dd116acf0944ea5fbe90。僅核 SVG 箭頭與分支語意、text 與正文步驟；桌面／手機呈現未驗。

### 11. docs/research/detection-data.md

結果：TREF-03、TREF-04 已關閉；沒有未處理技術疑點。全頁對照官方 Penn／VOC／COCO／Pet 原文、manifest、實際 Penn／Pet 內容及 HEAD。VOC2007 統計／difficult、COCO annotation／照片權利分開，Pet head ROI／trimap 衍生全身框分開；沒有已實作 COCO loader 或已做官方 COCO 評分的誤稱。

新增 crowd 段已從官方 fixed 8c9bcc3 重新抓 raw 獨立核對：mask.py L58–L67 為交集／dt 面積；cocoeval.py L106–L109 crowd GT ignore；L279–L295 crowd 可接多個預測、一般 GT 只接一個；L375–L378 TP／FP 排除 dtIgnore。L502–L511 是 bbox／segm small area range。這是官方評估規則，正文明說訓練另訂 ignore 策略，並且本 repo AP50 沒有 crowd 支援，不能報 COCO 成績。

頁 SHA-256：5f60e12e4819855713470f3998105d28b492eb4a8f547d95f624cc6e4ebd44cc

### 12. docs/research/foundation-data.md

結果：沒有未處理技術疑點。全頁讀取並核對 Fashion 官方 README／MIT 授權、CIFAR 官方格式及授權證據範圍、Pet current CC BY-SA 4.0 網頁與舊包內 research-only README。自製 128×128／1500-300-300 候選規格明示未實作、未量測，不當成現有 ShapeDataset 契約。

獨立 Pet 盤點和固定 Abyssinian_1 樣本：600×400，XML head (333,72,425,158)；trimap 值 1 的半開 bbox [107,81,444,328)，值 1或3 的框 [92,66,460,343)，PNG chunk CRC 通過；與本文一致。未下載原圖／做 XML 疊框，正文也沒有宣稱完成。19,257,361 bytes 的舊網路總量只能核算其加總式，原始 HTTP 紀錄未進 repo，此次沒有重建當年的請求總量。

頁 SHA-256：582aaade84655bcc75e2b338445114bfd4a581575a402360f06859b797abe99c

### 13. docs/research/lfs.md

結果：沒有未處理技術疑點。全頁命令對照 git-lfs fixed 0043a645047926f4bd7f7091299095528253d575 的 config／faq／fetch／fsck／install／pull manual、GitHub 當前 billing／file limits、gitattributes 和本地／遠端歷史 JSON。空字串 --exclude 覆蓋設定、skip-smudge、指定 include、fsck 的 HEAD／index 範圍均有依據。新增已跑 bootstrap 的分支符合實際 cwd 和含 tag 的目錄名。

核算 500 MiB×200／1024=97.65625 GiB，10 GiB 大約容納 20 次完整下載；10 GiB Free／Pro 額度與 2 GB 單檔上限符合現行官方頁。過去 501 失敗、本地 isolate 成功、Actions 空快取遠端成功分開；本輪沒有把讀保存 JSON 說成重新驗遠端 LFS。普通 Git 歷史最大 blob 1.17 MB 的安全只讀盤點支持不用 migration。無直接 SVG。

頁 SHA-256：b1b0547e3cb20653c08bb31b9edce93fb2960363d01c9e95438445cca16cca04

### 14. docs/research/pages-colab.md

結果：沒有未處理技術疑點。全頁對照官方 Zensical docs、Google Colab FAQ、GitHub Pages workflows／limits、requirements-docs 和實際 repo workflow／bootstrap。Zensical 0.0.67、Python 網站依賴與 CPU 模型依賴分開；Pages 不發布 notebook／權重／資料，notebook 從 GitHub tag 開啟，LFS 不供 Pages。Colab 免費 GPU 不保證及 runtime 保存範圍和 FAQ 一致；未把 hosted Linux runner 當成 Colab。無直接 SVG。

頁 SHA-256：c23b21afc8b5c7899d889e2462e6e472f2f61b275c63289b14227b52d59144f0

### 15. docs/research/version-sources.md

結果：沒有未處理技術疑點。完整來源表逐列打開 fixed 官方程式與論文原文；Ultralytics 441632c、THU-MIG 453c6e3、YOLOv12 2abab71 都取得內容，具體位置如下方來源記錄。GFL 原文式(6) 是相鄰兩個 bin 加權 log loss；Attention 原文 §3.2.1 式(1) 為 softmax(QKᵀ／√dk)V。

重點差異直接確認：YOLO26 paper v1 §3.2.1 one-to-one default；固定 predictor.py L469 end2end=self.args.nms is False，default.yaml L61 nms 未指定，故 model.predict 預設 many+NMS。paper §3.3.3 式(5)及 §4.1 是 side<8→16；fixed tal.py L59、L328–L344 則 side<16→16。本文有保留差異，不將通用全域列舉配對冒稱 YOLOv10 的 top-one task-aligned assignment，也不把小實驗當整版 AP／速度比較。無直接 SVG。

頁 SHA-256：7248ab8f8c0e00af80e76314201edd7cf5921b3384ad3fec9bbbe9511935ac44

### 16. docs/validation/curriculum.md

結果：沒有未處理技術疑點。全頁清單對照 section-map、42 份 per-section JSON／index、11 份 supplementary／GPU 紀錄、verify_curriculum.py、validate_curriculum_evidence.py 和 review_coverage.py。逐節數量與連結配對實跑 validator 通過。清楚寫 hash 不變沿用日期，公式／shape／梯度通過不是完整官方模型訓練；頁尾／網站圖的檢查能力也沒有誇大成正文自動審查。

manifest／raw 支持六位讀者、59 凍結頁、有紀錄後才開下一單位，以及協調者方法限制；第三輪及 browser 仍由其他人完成，這段方法聲明不能作為完成證據。已有正式 coverage 在本輪變更後仍過期，沒有由本審查修寫。無直接 SVG。

頁 SHA-256：2575b694468be422f2d42fcc027766ee048f2e133f47805c799b697b269aa7f9

### 17. docs/validation/gpu-smoke.md

結果：沒有未處理技術疑點。全頁逐項核對 gpu-smoke.json 以及 miniyolo/gpu_smoke.py、scripts/modal_gpu_smoke.py 的設備、timers、checkpoint／RNG、Volume／HF 回讀和 cleanup。保存 run 是 40 步連續 baseline vs 20+20 步跨 container 恢復，合計執行 80 步；不是80步單次模型訓練。輸入／targets／model CUDA、15,511 參數、finite nonzero gradients、initial 1.040038→final 0.061703、model／optimizer／history 差異0、scheduler／RNG一致均能回查。

timer 兩端 synchronize；baseline 1.494028 秒／interrupted 0.140341 秒／resume 1.750403 秒包含記錄／驗證，不當 throughput。producer／resume remote wall 和 72.374 秒 client wall 另列。warn_only=True 的 AdaptiveAvgPool CUDA backward 警告保留，不外推跨版本位元相同；L4／torch2.9.1+cu128／CUDA12.8 限定單次保存環境。HF 私有路徑和 Volume SHA、Modal stopped=running0 是既有記錄，未登入私庫或重新租 GPU 核實現在的遠端狀態。正文揭露 Modal wrapper 不在依賴 hash 的限制。無直接 SVG。

頁 SHA-256：18150faefd3b660bcddd8423fddd76a20b7857378ca5671bb1362bf740b5cc92

### 18. README.md（導覽以外的指定範圍）

結果：沒有未處理技術疑點。全頁命令及發布步驟逐項對照 requirements-model／docs／video、train defaults、check_lesson_runtime.py、record_evidence.py、review_coverage.py、workflow 和 verify_release.py。35 個相稱 CPU 測試、資料／證據 validators 與 07-training 的實際執行支持維護說明；預設輸出 artifacts/runs 與明確 --report tracked evidence 的差別正確。以 epochs20／1024圖／batch8 計算 2560 steps，和 argparse defaults 相符。

新增首次閱讀方法／限制的說明與本輪 manifest／canonical raw 一致，未把舊 reviews 稱盲讀。README 不是 59 導覽頁之一，其正文數字仍要人工追查，coverage 不包含 README。當前 lessons-v0.4.1 為舊已發布tag及歷史驗證；未發布的新 lessons-v0.5.0 不作失效憑證判斷，最終發布時仍須同步正文 tag 與新公開驗證。

頁 SHA-256：91445a97a8dd1f082175c8f63e100c91e5855479f9e40b253ebd708b4dc3b680

## 原始來源版本與實際位置

以下都在 2026-10-05 由審查者打開內容；HTTP 200 本身不作語意通過證據。固定程式逐一查其內容，網頁／課表屬移動來源，記本次時間及位置。完整此次來源索引暫存在 /tmp/clear-reference-authority/index.json；報告中的官方 URL／commit／章節可另行重取。

| 來源版本／位置 | 實際支持內容 |
| --- | --- |
| [COCO mask.py，8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9](https://raw.githubusercontent.com/cocodataset/cocoapi/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/mask.py)，L52–L67 | xywh、0-index、crowd 交集／dt 面積。獨立抓 4,591 bytes，SHA 4d4788f995971c97c5548f0bd19941f6ec1faaf5d0d7fb6ec489256ff872c617，和 root 本機 authority/mask.py 全檔相同。 |
| [COCO cocoeval.py，同 commit](https://raw.githubusercontent.com/cocodataset/cocoapi/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py)，L106–L109、L261–L299、L375–L378、L502–L511 | crowd ignore、多配對、TP／FP 排除 dtIgnore、small area range。獨立抓 24,143 bytes，SHA e514af401848d5a4cc5d3512bd93d2c317f92272c6d30f218e0f00b7b4110534，和 root authority/cocoeval.py 全檔相同。 |
| [Ultralytics 441632cdfd19e22e60a4b1b1999d46326ca51ec4](https://github.com/ultralytics/ultralytics/tree/441632cdfd19e22e60a4b1b1999d46326ca51ec4)：head.py Detect、block.py C3k2、tasks.py、cfg/models/11/yolo11.yaml、cfg/models/26/yolo26.yaml | DFL／Identity、one／many detach與輸出、fuse、C3k2 路徑、reg_max1／end2endTrue。head.py 本次 SHA 6b118882874ee2550b2c5c819a712774e1c729faf9fdc5ae6f068996c9893d02。 |
| 同 Ultralytics：utils/loss.py L92–L152、L1322–L1355；utils/tal.py L59、L328–L344／topk2 | DFL 相鄰 CE、CIoU+正規化 L1、0.8／0.2→0.1／0.9、stride-based 小框資格擴張。loss SHA 8eb4e089b7c36cef13bb872f9aa37e346da5c022bcfd979e11e7a57a0666b106；tal SHA c69ec67990777a4d13df219ef9b254085b2df283fbe5fd283245b76d6b461b3c。 |
| 同 Ultralytics：engine/predictor.py L469、nn/autobackend.py、models/yolo/detect/predict.py、cfg/default.yaml L61；optim/muon.py；docs/en/guides/yolo26-training-recipe.md | nms=False 才設 end2end，default many+NMS；MuSGD 與 Objects365→COCO 預訓練／微調及 branch-only 內部設定。predictor SHA d3de9cc0aa0c1f2c0b17c28ea35d1ae971ae7b56f1b7a5c2e3513e5b45e5efaa。 |
| [YOLOv10 THU-MIG 453c6e38a51e9d1d5a2aa5fb7f1014a711913397](https://github.com/THU-MIG/yolov10/tree/453c6e38a51e9d1d5a2aa5fb7f1014a711913397)：head.py v10Detect L497–L525、tal.py；[paper v1](https://arxiv.org/html/2405.14458v1) §3.1／appendix A.2 | detach one2one、top-one選擇、consistent matching metric、NMS-free分支。paper SHA c984835f0b966eb5a695aa63cb10118ca68114959b18e4f52e7aed4b8b0e5024；head SHA b088a15ecc73d05ddab5878b7b7257e84deec3b75e21cbe7fb6e8d0052b35623。 |
| [YOLOv12 2abab7153a065fb2925e8088e9ca2b19016ab7d6](https://github.com/sunsmarterjie/yolov12/tree/2abab7153a065fb2925e8088e9ca2b19016ab7d6)：block.py AAttn L1176–L1274；[paper v1](https://arxiv.org/html/2502.12524v1) §3.2 | QK 與 V 分開投影、連續token reshape 成area、CPU stabilized softmax、5×5 depthwise position Conv。block SHA e8ca08816f17d768bfe1613b7f6e009459ef6e9ca4d8d85bd1807dcb7afe72c8。 |
| [YOLO26 paper 2606.03748v1](https://arxiv.org/html/2606.03748v1) §3.2.1、§3.3.3 式(5)、§4.1 | one-to-one default、side<8→16，與 fixed code 的差異有原文支持。HTML SHA a972a2aa001d4603a66e31d3366edac8062f56a65fd77641fc88a4d3ba1197b0。 |
| [GFL 2006.04388v1 PDF](https://arxiv.org/pdf/2006.04388v1)，§3 的 Distribution Focal Loss 段、式(5)／(6) | expectation 解碼及相鄰 bin 的距離權重 cross entropy。PDF SHA 04a56cf0bd1c144ff940ad12a4134b75090843994bbfc22b84a04c1e5ed68a54。 |
| [Attention Is All You Need 1706.03762v5 PDF](https://arxiv.org/pdf/1706.03762v5)，§3.2.1 式(1) | QKᵀ／√dk→softmax→V。PDF SHA bfaaec89262875f927cf1b38b2da2d775f3309b7bea3537f29b606ca67e79065。 |
| [Git LFS 0043a645047926f4bd7f7091299095528253d575](https://github.com/git-lfs/git-lfs/tree/0043a645047926f4bd7f7091299095528253d575/docs/man)：config／faq／fetch／fsck／install／pull manuals | include／exclude 覆蓋、local skip-smudge、沒有大小自動追蹤。GitHub API 確認 commit 2026-09-22；config raw SHA 6ec196d258b6fd71a348604ca367f044bf23e2f689f58f7b918b7dc542dda827。 |
| [GitHub LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)、[about LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)、[large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[pricing](https://github.com/pricing) | Free／Pro 10 GiB、owner 計費、每個版本完整 storage、2 GB單檔、普通 Git50／100 MiB、Pages不能用LFS；費率讀當前來源，不當永久契約。另用 API 核對 docs 7c06b1d／6c39b7b 的日期與 touched files，不把最後修改日稱規則生效日。 |
| [Fashion-MNIST README／LICENSE](https://github.com/zalandoresearch/fashion-mnist)、[CIFAR 官方頁](https://www.cs.toronto.edu/~kriz/cifar.html)、[Pet 官方頁](https://www.robots.ox.ac.uk/~vgg/data/pets/)、[Penn作者頁](https://www.cis.upenn.edu/~jshi/ped_html/)及包內 README | 分類格式、MIT notice、Pet網頁CC BY-SA4 vs舊research-only文字、Penn原權利限制。Fashion LICENSE raw SHA 13ef4788476d292858fa60eb9a5f74aeca5c65770bc885ccaa05823a17ef7be1；Pet頁SHA a87f14385c70a4344e84c6c22fb7663bb1a6172efdec353ae325a6166d5fc8ec。 |
| [COCO download／format／terms](https://cocodataset.org/)、[VOC2007 dbstats／htmldoc](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/dbstats.html) | split、xywh／category／iscrowd、annotation CC BY4與照片另有權利、VOC difficult／Flickr。大型封裝本輪只HEAD。 |
| planning/course-research 及 feedback 的頁內原始連結 | 上列逐頁結果列出實際查核課表、notebook cells、PDF位置與指定posts／issues；沒有以二手摘要代替。有兩次猜測MIT舊檔名的404，隨後用官方目錄定位PT_Part1_MNIST／PT_Part2_Debiasing，沒有把自己的猜測404當教材錯誤。 |

## 保存紀錄及工具版本

| 本輪逐項查過的紀錄／實作 | SHA-256 |
| --- | --- |
| artifacts/checks/curriculum/fashion-mnist-learning.json | 38bdf713e154792299c885cd919175b866f39175daa1848a7253e84d444e7e38 |
| artifacts/checks/curriculum/custom-data-160-step.json | d1b5582921555b93338f26088ceb2d4e3a84b8a4b04266965df86a4ea37b0f96 |
| artifacts/checks/curriculum/custom-data-learning.json | fb22ce8356fbd747f967c3b82df53bd55de840d826d44fc73abfdba315f8e584 |
| artifacts/checks/gpu-smoke.json | 8a12cb801f0b6e8e8180fa03464fa8caf47980038be82f6c9e63a610d0d439d0 |
| scripts/build_lesson_notebooks.py | 2daddeff86572fc3edd3ba6302f0ff4696d33f5b1cd8037560ceb955f610893a |
| scripts/record_evidence.py | 8fd91178fb6927861ffa1dd1ee20ab80f0438ad151ef1f09da02a2799399c035 |
| scripts/review_coverage.py | b8b617dcaaa495753f329b71e709bd1aa8e7dce73fae40d2c8f3ab32395bb0ea |
| scripts/verify_curriculum.py | 78caa627e6794f6f20cee0a16a48b5efbf434bc35096a1bf867a2c0e5c26e5a9 |
| scripts/validate_curriculum_evidence.py | ed6bc8abfc50f22df97fc7645e6430d3d6e1488e07dafc5b1ca4abe923c94a6c |
| scripts/verify_release.py | fff8a374c293b329cc1caf48219426d9f07f4674cbe548e3a155c952c26e2f79 |
| data/manifest.json | cc643f72e809ccf5e7ea3b268f33bb12265a2f14d1f0653834eebdf8cc3c7f05 |

自有資料 160 步 train mAP50=0.006802721／val=0；1600 步紀錄 train=1／val=8／27≈0.296296／test=7／9≈0.777778，每個held-out split只有9個GT。1600 步是另一次從頭預先設定預算的實驗，沒有把兩份紀錄虛構成 checkpoint 接續，也沒有把其微小樣本的非零 AP 說成穩定泛化。GPU 保存 code_commit 463d3f59fb8042f8edc07fcc18581fab2c6bb418；只核現存紀錄與程式，不新增GPU證據。

仍未驗證：本輪的桌面／手機 Zensical 呈現、真人理解、Google登入與實際Colab runtime、新 lessons-v0.5.0 發布／公開驗證、未知遠端gitrefs、未下載VOC／COCO／CIFAR大型包、歷史網路總byte的逐請求原始證據、本輪新的LFS上傳／空快取回取、私人HF／Modal目前遠端狀態。這些限制按實際審查範圍保留，沒有補寫為已完成。

## 2026-10-05 10:49:18 UTC：最終 pin 與方法文字追加核回

將上述18個頁面SHA逐一與目前檔案比較：16頁完全相同；status與README不同，已重新完整閱讀兩頁及核新版操作／範圍聲明。這是追加複查，不抹除10:22原快照。16頁沿用原SHA與結果；兩頁目前版本如下：

| 重新實讀頁面 | 當前 SHA-256 |
| --- | --- |
| docs/status.md | 43306c2ce119b06b9954403f9cc9b68b8309c1492ffad6b08db5ca21f50c84c8 |
| README.md | 6d73e6ceadb3649f07405719d8428475dae8a327b238218b893bc5586e00c5b2 |

README將第二輪技術細節放在clear-tutorial三輪順序後，沒有以技術摘要取代原段落記錄；首次閱讀前提／作者提示限制、AI與真人區別仍保留。status的分類／偵測、CPU實測／GPU工程檢查、三步／160／1600步、AP候選門檻與顯示門檻、RNG進度及coverage不自動比對數字的界線仍與前次查過的程式／保存紀錄一致。新tag文字是lessons-v0.5.0；再次執行 `python3 scripts/validate_preparation.py` 通過，核目前section配對、notebook程式與網站發布邊界，不重新跑已通過的35項測試或訓練。

目前 `curriculum-publication.json` 和 `curriculum-release-bootstrap.json` 的source_ref仍是lessons-v0.4.1，SHA分別68f1bb41d4c771706efbcbd5166a5a66e8a66c67ef1540aa674ef109da6b64d5／492aa2ccec30effd46009365362c5192ddf889b52339276f4c77f8bc9d264a6b。它們只支持歷史版本，不能支持工作樹已做新版發布後驗證。status對發布後固定tag驗證的說明是新版發布流程應完成的結果；此刻還不能把頁中文字當本輪已跑過的證據。README第6／7步保留刪上一版bootstrap、部署後執行Verify published lessons並保存兩份新結果的順序，真正發布仍須完成它們。不因尚未打tag、futuretag此刻不可訪問判定缺憑證，也不提前聲稱新tag已公開通過。

本次沒有更改教材、coverage或原raw；reference原143行SHA仍為738984b16c33fa3263715a7d4277022607cd0f65cc2d6f361f05d586088ddc3c。後續第三輪evolution另存transitions/evolution.md，其中09圖片已追加站台局部核回；它不改本報告17導覽頁的browser未驗範圍。
