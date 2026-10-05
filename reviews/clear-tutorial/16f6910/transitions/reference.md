# 第三輪：參考頁與 README 的前後銜接

審查者：`clear_first_foundations`，AI；2026-10-05。這是第三輪銜接與修改處複查，不是本組第一輪盲讀，也不是真人學習驗收。我沒有撰寫或做過這 15 個參考頁的第二輪技術審查；曾撰寫其他四個第 11 章頁面、審查第 7–8 章八頁，不能把那些已有知識說成這次才由參考頁首次學到。本報告只寫本次实际讀到的任務、銜接、操作條件及局部變化預測；沒有用先前的技術報告替代讀者回答。

## 方法與範圍

先實際讀 current 首頁、learning-path 全文，以及 outline 的 42 課概要導覽，再沿學習／維護分流閱讀參考頁：查詞與範圍 → 課綱和設計來源 → 資料規劃及兩個資料研究頁 → 架構、LFS、發布與 Pages／Colab → 版本來源 → CPU／GPU 驗證。最後讀 README，核對是否能把短版操作接回發布詳解。以 repo 實際路徑為準，target 是下面 15 頁加 README；首頁與閱讀路線只是本輪前提，另組負責其驗收。

每頁全文讀完後先寫下自己的理解與小變化預測，保存在 `/tmp/clear-reference-reader-notes.md`，然後才開 `first-read/reference.jsonl` 核原始 ID。沒有修改該 raw log 或以後文覆寫當時卡點。舊 reference reader 的 17 個 issue 單位（14 burden、3 optional）逐項實際讀回修正處，見後表。

本輪另外檢查這 15 頁與 README 的 247 個本地 Markdown 連結目的路徑，都存在；這只證明本地路徑存在，沒有聲稱 HTTP／錨點／瀏覽器查詢全通過。唯一 target 內嵌圖是發布流程 SVG，實際讀 XML 的標籤、節點、箭頭和 `title`／`desc`；它把「有過期才重產、兩者過期先 GPU 再 CPU、別機連圖帶回」畫成可追的路線。**我本輪没有使用 browser，未驗桌面／手機字體、表格横向捲動、SVG inline 尺寸、搜尋查詢或快速換頁 MathJax。SVG 原始碼可讀不等於網站呈現已驗。** 協調者另做全站 browser；那不算我這份報告的實際方法。

未執行發布、Git 寫入／LFS 上傳、Colab、GPU、camera 或遠端私有 HF 操作。本輪是沿正文判斷「該做哪步／為何／需要哪些條件」，不是新一輪操作或實測。没有拿新 tag 暫時 404 當認證缺少：`lessons-v0.5.0` 在此時是待發布的計畫，發布頁先推 tag 再部署與公開驗證的次序，能解釋何時 URL 才應有效。

## 逐頁理解與變化預測

### docs/glossary.md

這是讀課文卡住時查詞、再點回課文的七組表，首頁／路線沒有要求先背完。符號 C／N／P 要依當節定義；※和末段把同名詞不同用途分開。四種門檻、score／precision、訓練 assignment／推論 NMS／評估 matching 都能從表中說明比較的是哪一類對象，不必把同一個 IoU 門檻套在所有任務。

小變化：讀到第 15 章「多 head」，要查 attention 的分組，不拿 detector 的末端 head 解釋；DFL 改 K 個 bin 時距離支撐是 0…K−1，不是改物件類別數。AP 列仍要在心裡重建階梯，但給了 2/3×0.5=1/3 的例子和 6.2 詳解，可完成查詞用途；微型包絡圖屬可選。沒有其他必要空間圖缺失。表格手機可讀性未驗。

### docs/planning/outline.md

從首頁／路線到此，我能將 0–7 視為完整偵測核心，8 是自訂輸入，9–16 是相互獨立的機制小實驗，17 可在基礎主線之後做，18–20 有另外的影片／座標前提。第 7 章六個里程碑分清原始真值、target、loss、decode、評估，不把版本名稱當作已逐代拼成同一個完整 detector。完成條件要求說明成功、失敗與成本，不保證新版一定勝出。「距離分佈回歸（DFL）」已和本列的一組機率連上。

小變化：17 章只改顯示門檻，模型及固定 AP 候選不變，不構成新訓練；改成模型寬度對照，就要固定資料與預算，先 validation 選設定再 test。細節可連到 course-research／feedback 看規劃來源，學習者可回閱讀路線進課文；表格足夠比較，沒有必要新增圖。

### docs/planning/course-research.md

這頁從 outline 來說明課程設計借鏡，公開課程事實與本教材建議分開。有些來源只讀公開作業名稱，未執行 repo 或看付費影片；不能把名稱當本教材已完成的照片偵測證據。現行實作範圍回 outline 決定，研究建議不覆蓋它。

小變化是來源範圍辨識：MIT Lab2 的輸出若仍是 face／non-face，即便稱 detection，也不能推成 bbox 定位；將原先建議的連續模型演進採作新規格，必須改課綱／實驗，不能稱它目前已實現。此頁沒有需要自己跑的主要例子，數值更新預測不適用。文字和比較表能支持任務，沒有缺必要圖。

### docs/planning/feedback.md

AI 模擬的大綱讀者與公開帳號個案是兩種證據；它們都不是本教材真人學習效果。每則卡點對應採用的章節，可從此回到 batch、座標、target、score、optimizer 接線的教學目標。公開 issue 作者的說法是個案，不代表其猜測的 bug 已由本教材重現。

小變化：一人說作業 100% 卻接錯 batch，應補 batch=1／>1 的完整管線檢查，不能推出全班比例；單個 accuracy 拼圖不能代替泛化實驗。此頁没有参数或框的數值練習，採证据類型預測即可。列表能追「卡點→教學採用」，沒有必要圖缺失。

### docs/preparation/architecture.md

從學習路線切到維護路線，網站只上傳靜態 site，notebook 另開 Colab，docs／模型環境分開。「ready」只表示頁、case、notebook 存在，不是 GPU 或效果驗收。tag 首次已短解為已發布的固定版本標籤。來源摘錄是 Markdown 圍欄的 `data-excerpt` 標記，最小完整圍欄例子讓我知道改的是說明檔開頭，不是 Python 程式。

小變化：只改標題而保留小節 ID，URL 可維持；改實驗或結果需新 release 配對，不能只部署新網站。下載改 `--output /content/data`，分類命令就改 `--data-root /content/data/fashion-mnist`。LFS pointer 不會變成 Pages 圖；42 節本身不需下載 LFS。路径表足夠回答檔案分工，詳細发布路線已有下頁圖，無需再補一張相同圖。

### docs/preparation/data.md

課內當場生成資料；Fashion 是額外灰階分類、沒有框；只有 Fashion／Penn-Fudan 可 `fetch`，`candidate` 從官方連結取，`generated` 不需下載。兩種不下載的狀態已直指名字，沒有「後兩種」歧義。校驗只完成下載完整性，不代替解壓、標註轉換或訓練。COCO 的左上 xywh、連續類別重編、crowd 定義及官方評估連結都在需要的位置。

小變化：`--output /content/data` 改完，之後分類要用對應 data-root；加入 Penn-Fudan 多人圖，不能只留一人將其他人變背景。crowd 區域不能交給本專案普通一對一 AP 評估；應保留旗標並用官方規則。LFS pack 含來源授權不代表別的照片可任意重包。表格／文字能回答資料路徑與操作，沒有必要圖缺失。

### docs/research/foundation-data.md

從資料規劃來看來源研究：開頭區分候選的 128² 三形狀規格與現行 64² 兩色矩形，不把候選 1500／300／300 當已跑配置。Fashion 沒 bbox；Pet 頭框、trimap 衍生前景框與官方 test 的 XML 覆蓋是不同任務。HEAD bytes 只是傳輸資訊，小 label／完整 annotations 有校驗仍不代表原圖片疊圖已驗。

小變化：把 trimap 的 unknown 值 3 算前景，框可能擴大，要固定轉換並重新核標註，不能說仍是同一官方頭框任務。要實際取現行可用資料，回 preparation/data，研究候選不是另一套已啟用下載指令。來源表足夠，這個查核頁無需補造沒看過的原圖片。

### docs/research/detection-data.md

這是四個真實照片候選來源的核對，教材偵測實驗沒有用它們訓練。来源可讀／range 成功不證完整 archive 或再散布權利；完整下載狀態另見 preparation/data。VOC trainval 包含 val，不能用它訓練後報獨立 val；照片和 annotation 也不是同一份授權。COCO crowd 小節說明群體不是大人物框，官方忽略配對可接多個預測，重疊用交集÷預測面積；不直接沿用普通 TP／FP 配對。

小變化：將 GT 標成 crowd 後，不能再用 repo 一般 IoU、一 GT 一 prediction 的规则報 COCO 成績；原 HTTP 只改 https 若憑證主機名不合，應換同 bucket 的 S3 URL，而不是關 TLS 驗證。VOC 原始端點的數值轉換例仍可補，這頁沒有提供正在執行的 VOC 轉換器，也明說先訂契約，故保留為 optional。無必要圖缺失。

### docs/research/lfs.md

讀完此頁再接發布頁，我能分清一般 Git 小圖／manifest、選定 LFS 封裝與外部完整資料。skip-smudge 保留 pointer，include pull 才取實體；本地 fsck、遠端寫權限、從空快取重取是不同階段。額度／成本歸 repo owner，不能以讀取 repo 成功當 LFS 已上傳。mini-voc 是不存在的示意，文案明确提醒換真實路徑和 manifest。

小變化：500 MiB 檔改一 byte 又占完整新版本；200 次下載约 97.66 GiB，`--depth 1` 不能限 LFS 大小。已跑教材環境格時，範本沿用包含 tag 的目前 repo，不再 clone 到固定 `/content/learn_to_yolo`。尚未安裝 git-lfs 时先裝，之後只 pull include 指名資產。MiB／GiB 的 1024 換算可再短寫；操作意圖仍明確，是原 optional。此頁无需新圖。

### docs/preparation/publish.md

主線可說清：新 tag 配對 → 列過期 → 遠端操作前推分支 → GPU 過期才手動做 → CPU 重產 → 紀錄與圖一起帶回 → 人工核數字與三輪審查 → 本地檢查 → 不可變 tag／Pages → 公開驗證。兩類紀錄都過期先 GPU 再 CPU。純文案改只重審，没有自動 GPU 工作。Git 推送 contents／workflows、啟動 Actions、Pages job token、Modal／HF 與照片再散布權利各有來源；純閱讀者不需要這些寫入條件。

小變化：bundle 直接解開舊 lessons 會丟掉編輯機新正文，故只帶回紀錄／圖，再 render-only；三列表與當場警告明說 render-only 不補圖、不替我改結論。GPU failed 但 hash 相符的結果不能留著讓下次跳過，要依指名檔的已追蹤／未追蹤狀態恢復或移除，修原因再跑。已跑環境格者省 clone／固定 cd，先確認 repo 根目錄。新 tag 不存在时，按步 9 先推 tag 再驗公开入口，404 不要求補帳號認證。发布圖回答了原始流程依賴需求；源码核回已做，inline/browser 未驗。

### docs/research/pages-colab.md

從網站讀者到獨立 Colab，或維護者以 Python 3.12＋docs venv 預覽，都有路線；建站不需模型環境。靜態搜尋／MathJax 檢查只查索引、語言與標記，仍需實際 browser 查詢／快速換頁。公開 tag 的 runner bootstrap 檢查與真人 Google／Colab 操作分開，沒有稱後者已測。

小變化：只改網站主題不需新教材 tag，改 bootstrap／套件／實驗则需新 notebook 配對；先 import 錯版 torch 再改裝，需要 restart；存 Drive 只存 notebook，checkpoint 等 runtime 檔另保存。此頁文字足以說清兩個執行位置，沒有缺必要圖，browser 未驗。

### docs/research/version-sources.md

此頁由 outline 的版本範圍接入，把現代機制定位到固定官方 commit；右欄是教材採用／省略範圍，不能把教學列舉配對、STAL mask 或省 BN 當完整官方模型。論文、固定程式与本專案示範的分界可讀出來。

小變化：若 STAL 從固定程式 <16 改採論文 <8，邊長 10 不再符合條件；YOLO26 名稱不能讓我跳過辨認目前 predict 走 many＋NMS 還是 one。這是依頁上明示差異預測，不聲稱本輪重新讀外部 repo 全部實作。對照表足夠，沒有必需新增圖。

### docs/status.md

首頁先看這頁就能分清人工答案、少數 CPU 更新、合成長些訓練、L4 工程檢查和 Fashion 分類管線；都不等於真實照片 AP、正式速度比較或真人學生已懂。紀錄程式 hash 變表示證據過期，review 內容 hash 變表示重審，不是多跑 GPU。RNG 第一次已展開亂數產生器；同頁雲端續訓段說明少存狀態會影響隨機打亂／增強，並連 GPU 詳解，能理解舊格式可以推論但不能用指定入口續訓。短解當下進度仍可再放在第一次位置，屬可選減少回讀。

小變化：改 decoder 后，綁定該模組的紀錄要重做，不能只換日期；自己的 Colab 數字稍不同，不等於管線壞。正式實驗協議現在分開兩種 score 門檻，顯示 0.25 不能直接截 AP 候選。未發布新 tag 的操作要接 publish 的先推 tag 次序，而非認證推測。範圍表與文字足夠，未做本頁 browser。

### docs/validation/curriculum.md

由 status 進來，可查 42 個 case JSON、補充長實驗、GPU 記錄及每頁 review；通過意思是定義／shape／梯度／流程符合本節要求，不是完整版本品質排名。新紀錄與舊紀錄绑定程式都沒變就沿用日期。內容 hash 包含正文數字，但 validator 不會自動計算它是否和重產 JSON 一致，维护者仍需搜尋舊值／檔名人工核回。

小變化：補充 JSON 重產了但別頁仍舊数字，coverage 不必然提醒；正文没變並非數字仍正确。新 tag 先部署再從公開 runner 檢查，tag 内含上次 publication 結果並不假裝已驗當前 tag。真人 Colab／相機未測的界線可接回 status。索引表正是所需查找方式，沒有必要圖缺失。

### docs/validation/gpu-smoke.md

從發布頁的 GPU 分支進來，能說清 40 步連續對照、另 20 步保存、CPU container 核 Volume／HF、全新 L4 續 20，合計 80 次更新。Volume commit 不是 git commit。無 HF_TOKEN 時可只用 Volume 通過；有 token 但 public repo／上傳失敗是 partial，不能把 checkpoint 發到公開 HF。課程 CPU／照片路線不因此需要開 GPU。

小變化：只改未綁定的 Modal wrapper，dry-run 不會自動過期，要照文中手動重跑 workflow；有錯誤設定先暫停修正，不自动啟 GPU。缺 RNG 狀態可推論但不能接上亂數序列做 exact resume。`warn_only=True` 容許警告與有限誤差，實測差 0 不是跨硬體承諾；不同暖機狀態的計時不能當速度排名。文字把狀態、權限、紀錄路徑與測量范围交代完整，無必要新圖。本輪没有取得私有 checkpoint 或啟動 GPU。

### README.md

從 repo 入口可沿 Python 3.12、CPU venv、套件、暖身／tests／42 節 runtime 做本機學習；執行命令需 activate 或絕對 venv 路徑，Windows 用 Scripts／PYTHONPATH。自己的 runtime 輸出在 artifacts/runs，不改原 checks；建站另用 docs venv。新版發布短版的順序與 publish 詳解一致，其他電腦／GPU 設定／失敗處理接其末端連結。

小變化：`steps=160,samples=32` 改為 `epochs=20,samples=1024`，batch=8 每輪 128 更新，共 2560；兩個步數模式不能同時指定。換輸出目錄會留下舊成果，不表示新的檢查改過原紀錄。初讀後發現可選改善：發布 step 5 仍用舊「一次獨立 AI 查核」長段，可以在起首明指逐段閱讀→非作者技術→另一讀者銜接，與頁首及 publish 的三階段保持同一操作提醒。它目前没有明说只需一次或排除其他階段，且有详解連結，所以沒有擋住理解，列 optional 而非 blocking。沒有必要新圖。

## 原始問題逐項核回

原始 log 仍原樣保留。以下寫的是本輪 current 文的複查結果，不回填第一次答案，也不把 optional 稱成已加圖／已修改。

| Raw unit ID | 原 severity／頁 | 本輪實際讀回與處理 |
|---|---|---|
| reference-0015 | optional／glossary | 109 行仍是文字包絡＋2/3×0.5 例＋6.2 詳解。小包絡圖可選保留；這是查詞表，不要求在此完成整個 AP 推導，沒有主要任務卡點。 |
| reference-0051 | burden／outline | 108 行已改「距離分佈回歸（DFL）」，與同列距離機率連上，關閉。 |
| reference-0060 | burden／architecture | 7 行 tag 第一次短解為「已發布的固定版本標籤」，能辨公開固定教材，關閉。 |
| reference-0061 | burden／architecture | 54 行明說 Markdown 區塊標記不是 Python，接完整三反引號最小例，知道改哪行，關閉。 |
| reference-0066 | burden／data | 16 行直寫 generated 或 candidate，連到官方下載表，不再猜「後兩種」，關閉。 |
| reference-0070 | burden／data | 91 行 COCO 列定左上 xywh／畫素、category ID 重編、iscrowd 群體標記，關閉。 |
| reference-0072 | burden／data | 126 行定 crowd，保存旗標、官方忽略處理、直連 coco-crowd，並說本專案評估不支援，關閉。 |
| reference-0079 | burden／publish | 49 行實際有 `git diff --cached`，54 行說它顯示將 commit 的修改，不只 whitespace check，關閉。 |
| reference-0085 | burden／publish | 125 行先說全新 runtime；137 行已跑環境格者省 clone／固定 cd，沿用 tag 目錄並可確認根目錄，關閉。 |
| reference-0086 | burden／publish | 148 行發布依賴圖、153 行三類檔表；第 6 步再次說 render-only 不會重畫／複製圖。能列 GPU／CPU 條件及帶回清單，關閉文字／圖來源層問題；browser 不在本報告已驗範圍。 |
| reference-0092 | optional／detection-data | 35 行仍要求 VOC 1-based→半開契約一致，沒有數值端點例。保留 optional：本頁是來源研究，沒有讓讀者直接執行不完整轉換器。採用 VOC 時仍須先明訂／驗轉換。 |
| reference-0093 | burden／detection-data | 51 行定群體區域並舉擠在一起的人；下一段說官方忽略、可接多預測及特殊重疊公式，關閉。 |
| reference-0106 | optional／lfs | 39 行仍沒有 1024 換算式。保留 optional：容量／成本意圖與 include 操作不靠先重算小數才能完成。可加 `200×500÷1024`，未假裝已加。 |
| reference-0110 | burden／lfs | 78 行先分既有 tag repo／新 runtime；bash 的 git 根目錄分支沿用目前 checkout，關閉。 |
| reference-0124 | burden／status | 68 行已展開 RNG；77 行解釋隨機操作續訓差異，GPU 詳解說狀態是當下進度，當場任務可理解，原縮寫卡點關閉。第一次位置再加「當下進度」可選，不能聲稱舊原文已經這樣寫。 |
| reference-0127 | burden／status | 104 行明列兩種 score 門檻／用途、不能以顯示 0.25 直接截 AP 候選、每次記實際值，關閉。 |
| reference-0133 | burden／curriculum | 98 行改成核內容指紋但不自動核算正文數字與重產紀錄一致，並接人工搜尋舊值，關閉。 |

本輪沒有未解 blocking 或 burden。原 3 optional 保留理由如上；另有 README step 5 的三階段短提醒、status 第一次 RNG 狀態短解兩個可選改善，已回報協調者。不是以後文復述消除原始卡點；raw log 永久保留。

## 本次來源指紋

下面是實際全文閱讀後核回的 current 工作樹 SHA-256；不是引用舊 coverage 的指紋。若協調者再改正文，需對應 delta 實際再讀。

| 檔案 | SHA-256 |
|---|---|
| docs/glossary.md | f1f6fbb72f85924b45fd2aaaf663880a308957a7fb6782e84ba13394a8e67b2e |
| docs/planning/outline.md | 3046aed911dcc50ad22a76f6b19ec9ebc9b04fbb3de5a927457af292a5797518 |
| docs/planning/course-research.md | 6922ad22cf822cf611bfea9092bc20616732010f92ff8471693ba778ba5973d0 |
| docs/planning/feedback.md | 4b805cc1729450bff04d27ae6ef2f6026adde52b80f50dc87d57ae7541524819 |
| docs/preparation/architecture.md | ce1c28d2ce406227575fc611a617f19a42cc1fcc0c665788a5049bf422398305 |
| docs/preparation/data.md | 293bd94b298639d544845474106009fadfc357efa08bbdcf78efd25d207a313e |
| docs/preparation/publish.md | fd842d0b9dd3c85ee8c266439d79153908b6c2de7f6a489ed539f2250c4b792d |
| docs/research/foundation-data.md | 582aaade84655bcc75e2b338445114bfd4a581575a402360f06859b797abe99c |
| docs/research/detection-data.md | 5f60e12e4819855713470f3998105d28b492eb4a8f547d95f624cc6e4ebd44cc |
| docs/research/lfs.md | b1b0547e3cb20653c08bb31b9edce93fb2960363d01c9e95438445cca16cca04 |
| docs/research/pages-colab.md | c23b21afc8b5c7899d889e2462e6e472f2f61b275c63289b14227b52d59144f0 |
| docs/research/version-sources.md | 7248ab8f8c0e00af80e76314201edd7cf5921b3384ad3fec9bbbe9511935ac44 |
| docs/status.md（追加核回後） | 43306c2ce119b06b9954403f9cc9b68b8309c1492ffad6b08db5ca21f50c84c8 |
| docs/validation/curriculum.md | 2575b694468be422f2d42fcc027766ee048f2e133f47805c799b697b269aa7f9 |
| docs/validation/gpu-smoke.md | 18150faefd3b660bcddd8423fddd76a20b7857378ca5671bb1362bf740b5cc92 |
| README.md（追加核回後） | 6d73e6ceadb3649f07405719d8428475dae8a327b238218b893bc5586e00c5b2 |
| docs/index.md（前提） | dc8947e5d0d7baf3ef68a74e6631d788344b5891c50f9cea9701e099a501f6a6 |
| docs/learning-path.md（前提） | 7008aa1575e26490fd7d731450975fc6b67fd8ad5bbc9988a5b1b5f00714f825 |
| docs/assets/diagrams/release-evidence-flow.svg（target 內唯一嵌圖） | daf67652ce9f56282fb58571cd670bac0e0d572c8509dd116acf0944ea5fbe90 |
| docs/assets/diagrams/classification-vs-detection.svg（實際 current 首頁圖；修正來源標籤時再讀 XML，非 browser） | a050400b9fe7e35a06073ca97ab15ee893ce1fef62bc57236707539a00d1c90e |

本輪讀前筆記、實際路徑與圖指紋明細另存 `/tmp/clear-reference-reader-notes.md`、`/tmp/clear-reference-inventory.json`。只寫本份新 report，未修改 docs、coverage 或舊 reviews；沒有 commit、push、GPU。

## 協調者修正後的追加實讀

協調者收到上面的兩個可選改善後已修改，實際重新讀 README 第 74–82 行與 status 第 66–70 行：README 發布 step 5 現在直接連 clear-tutorial 並列「逐段首次閱讀、非作者技術核對、另一位讀者銜接複查」，後面的長段明標「以下是第二輪的技術細節」。我現在能從這一步列出全部三輪和誰應獨立，不會把一輪技術檢查當完整審閱。這個新 optional 已關閉。

status 第一次 RNG 現在寫「RNG 狀態（亂數產生器當下的進度）」，不用等第 77 行就能與 seed 的起點分開；舊 checkpoint 不足以接上原亂數進度，推論只需權重。這個可選位置改善已關閉，reference-0124 的原問題也在原先需要的位置直接解開。上述初讀段落／原始 issue 表保留我當時讀到的版本，不改成假裝首次就看到了修正版。

最終表已更新這兩頁的指紋；初读版本分别是 README `91445a97a8dd1f082175c8f63e100c91e5855479f9e40b253ebd708b4dc3b680`、status `7ebd692718a4545decff806fb9578310e2fe4b3df125e9af5d41b976186ef943`。其餘 14 個 target／README 頁没有因這個 delta 改動。最終仍為 **0 未解 blocking、0 未解 burden、3 原始 optional 留存**；browser、真人 Colab、camera、發布後新 tag 入口依然未由本審查者驗證。

協調者指出初稿把 `object-journey.svg` 誤列成 current 首頁前提圖。我實際再讀 current `docs/index.md` 全文及 `classification-vs-detection.svg` XML：首頁唯一圖是同一張黑底紅方塊的分類／偵測答案比較，箭頭分別連到「紅色」與含藍色位置框的答案；藍框是預先畫的答案，不是模型預測。已更正上表；object-journey 是我先前第 7 章技術 context，不是現首頁圖，不能借它宣稱已看 current 首頁素材。沒有把這次更正說成原始首次閱讀時的圖檢查。

當場 `/tmp` 筆記已原樣保存為 `transitions/reference-current-read.md`，加明第三輪非盲讀的來源說明；本次 input／initial 與 final 頁指紋、247 路徑檢查、圖來源標籤更正與未驗範圍保存在 `transitions/reference-inputs.json`。兩份都是新報告，原 `first-read/reference.jsonl` 沒有改動。
