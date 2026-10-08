# 審查紀錄：基礎資料

審查範圍：`docs/research/foundation-data.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：有必要問題

結論：不通過。有 1 個 must：第 3 行與第 93 行把 Fashion-MNIST 寫成教材實際使用的資料，和程式現況不符，也和〈資料規劃〉「42 節教材沒有用到」直接矛盾。另有 1 個 should：第 83 行說「結果沒有收進 repo」，用詞不精確；而且「HEAD 紀錄與官方頁面快照不在 repo」這項範圍說明刪掉後已經沒有地方交代。

其餘檢查都通過：
1. 查核指示單列的 9 條先前審查意見全部處理：/workspace 路徑、outline-v2 檔名、研究代理／子任務／後續整合、五個不在 repo 的 JSON 都已移除；兩個標題已改名；第 9 行已併入引言段。全頁 grep「本次、本研究、待測、尚未、日後、後續、現行、早期、現在、目前、新增、修正」等字眼都沒有命中。第 4 節的建議以查核當時的建議句呈現，不算計畫式敘述。
2. 事實逐項核對。合成圖部分：第 1–4 章與第 7 章確實只有紅藍兩類矩形、類別由顏色決定（01-small-cnn、03-comparison、04-localization、04-coordinates、07-*），ShapeDataset 預設 size=64，張數為 2–32（含補充紀錄 8／8／2 與 grid-learning 的 32／16／16）。manifest 中 oxford-iiit-pet 與 fashion-mnist 的 bytes、SHA-256 相符。各節頁尾的「## 實際執行紀錄」42／42 頁都有。另外我自己在暫存副本重新下載 Pet annotations 與 Fashion t10k labels，以下全部重現：19,173,078 bytes、SHA-256 52425fb6…、MD5 95a8c909…；XML 3,686、trimap 7,390；trainval 3,680、test 3,669、兩者無重疊；trainval 缺 9 個 XML，test 全缺；有 41 個 trimap 不在任一 split 清單；Abyssinian_1 的 600×400、depth 3、head (333,72,425,158)、8-bit 灰階、值只有 1／2／3、CRC 全部相符、兩個 bbox；Fashion labels 解壓後 10,008 bytes、magic 2049、每類 1,000。HEAD 結果也相符：CIFAR 170,498,071、Pet images 791,918,971、Fashion 四檔大小與轉址目標。官方頁原文（Pet 的 CC BY-SA 句、CIFAR 的 163 MB／md5 且無授權文字、Fashion README 的 MIT 與 MD5、torchvision 的 MD5）也都對得上。
3. 渲染：在暫存副本執行 zensical build --clean --strict，結果 No issues found；validate_site.py exit 0；另外跑的 validate_preparation.py 也是 exit 0。頁面上的 4 個表格、引言段與連結都正常渲染。這次查核的頁面的 diff 只動到 docs/research/foundation-data.md；工作樹中其他修改來自同時進行的其他部分。

未列為問題的觀察：
- 連結文字〈教學大綱〉與導覽標籤「課程大綱」不一致，但 detection-data.md、feedback.md 也用同一個稱呼，不會誤導讀者。
- annotation 包裡另有 15 個 XML 不在任一 split 清單；本頁沒寫，但第 2 點已建議從有 XML 的 trainval 切分，讀者照做不會出錯。

查核下載的檔案放在暫存副本，建置與驗證日誌放在暫存副本。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/research/foundation-data.md 第 3 行（開頭引言「真實資料另有 Fashion-MNIST 分類支線」）與第 93 行（「教材實際用資料跑出的結果，見〈資料規劃〉（Fashion-MNIST）」） | 兩句都把 Fashion-MNIST 寫成教材實際使用的資料，但 42 節沒有任何一節用到它。lesson_cases/ 和 notebooks/ 都找不到 Fashion，section-map.json 每一節的 datasets 都是空陣列。它只出現在獨立腳本 scripts/run_fashion_cnn.py，以及 miniyolo/classification.py 的 loader。那次 40 步執行也只證明管線接得起來，正確率和亂猜差不多。本頁連到的〈資料規劃〉開頭明寫「Fashion-MNIST 是選用的真實分類資料，42 節教材沒有用到」，docs/index.md 和 docs/preparation/architecture.md 也這樣寫。讀者照本頁會以為某一節拿 Fashion-MNIST 訓練，跑去找那一節；點進〈資料規劃〉又看到相反的說法。 |
| 2 | 建議 | docs/research/foundation-data.md 第 83 行（第 3 節表格下方） | 「split 筆數、XML／trimap 覆蓋與解碼檢查的結果沒有收進 repo」照字面讀是錯的：這些結果就寫在本頁，〈資料規劃〉也重述了 split 與 XML 的數字，兩頁都在 repo 裡。真正不在 repo 的是本查核的原始紀錄。原本第 5 行和第 85 行交代過「資料與來源快照、官方來源原文、HEAD 記錄都不在 repo」，刪掉以後，HEAD 回應和官方頁面快照不在 repo 這件事已經沒有任何地方說明。現在這句只點名 split 等結果，維護者反而可能以為 HEAD 紀錄或頁面快照收在 repo 某處。另外，「就能重查」沒有說 repo 裡沒有盤點程式，得自己寫。 |

修正必要問題時處理的項目（修正後由第 2 次複查確認）：

- line 3 （製作經過的敘述，改寫）「研究代理／子任務／後續整合」橫幅：改寫。現在第 3 行是有日期的查核紀錄引言：實際採用 64×64 紅／藍兩類矩形的 ShapeDataset。Fashion-MNIST 那一句沒有照先前審查意見寫成「分類支線」，因為那是錯的；改成『選用的真實分類資料，42 節教材都沒有用到』，並指向〈資料規劃〉的 40 步分類管線核對（檢查員的必要）。
- line 5 （製作經過的敘述，改寫） `/workspace/yolo-curriculum-outline-v2.md` 與「尚未製作教材」：改寫。第 5 行現在是『查核日期：2026-10-02。依〈教學大綱〉第 1–4 章與第 7 章的資料約定…網路讀取只用 Python 標準函式庫』，已沒有 /workspace 路徑和 v2 檔名。
- line 9 （修訂經過的敘述，刪除）「現在已有 ShapeDataset 與 Fashion-MNIST 實測…早期三類構想」：刪除，內容併入第 3 行的引言。
- line 11 （修訂經過的敘述，改寫）標題「自製幾何資料的早期規格」：改名為『## 1. 自製幾何資料的候選規格』（現在在第 9 行）。
- line 76 （修訂經過的敘述，改寫）標題「本次已取得的小型檢查資產」：改名為『## 3. 查核時取得的小型檢查資產』（現在在第 74 行）。
- line 78 （製作經過的敘述，刪除）「路徑均在 `/workspace/prep-research/data-probe/`」：刪除。保留沒有呼叫 extractall 那一句（現在在第 76 行），但主詞從「壓縮檔」縮成『Pet 的 `annotations.tar.gz`』，因為第 81 行的 Fashion gz 檔確實有解壓。
- line 85 （製作經過的敘述，改寫）五個不在 repo 的 JSON 與 foundation-sources 路徑：改寫（現在在第 83 行）。改成指向 `data/manifest.json` 的 `oxford-iiit-pet`、`fashion-mnist` 兩項，並照檢查員的建議寫明：原始紀錄（下載與 HEAD 請求的紀錄、官方頁面快照、盤點輸出）沒有收進 repo；可以從官方連結重新下載 annotations.tar.gz，核對 bytes 與 SHA-256 後自己盤點；repo 裡沒有做這項盤點的程式。
- line 87 （製作經過的敘述，刪除）「另行資料準備若下載更多檔案，需獨立登錄」分句：刪除。句子現在停在『本查核沒有下載 Fashion-MNIST 的 train images、CIFAR-10 或 Pet 的圖片。』（第 85 行）
- line 95 （其他，改寫）「尚未實測任何資料…」：改寫（現在在第 93 行）。內容是：本查核沒有量測完整影像讀取速度、訓練時間、準確率、AP 或記憶體；實測值包括 byte 數、split、第 3 節的 checksum 與單一 trimap 結果；各節實驗結果看各節頁尾的「實際執行紀錄」；Fashion-MNIST 的 40 步分類管線核對看〈資料規劃〉。原本「教材實際用資料（Fashion-MNIST）」的說法已拿掉（檢查員的必要）。

### 第 2 次複查：通過

通過，沒有必要問題，只有 1 個建議。

1. 先前審查意見與製作經過：9 條先前審查意見都處理了。第 3 行的橫幅改寫了；第 5 行的 /workspace 路徑、outline-v2 檔名和「尚未製作教材」都拿掉了；第 9 行刪除並併入第 3 行；兩個標題改成「候選規格」和「查核時取得」；data-probe 路徑刪了；五個不在 repo 的 JSON 改成指向 manifest；「另行登錄」那個分句刪了；結尾的「尚未實測／待測」改成「本查核沒有量測」。全頁搜過「本次、本輪、原本、現在、目前、後續、日後、待、尚未、早期、研究代理、子任務、workspace、v2、AI」等字眼，唯一命中是第 62 行的「作者對舊文字的更新說明」，講的是資料集作者，不算製作經過。

2. 事實核對：
- 第 3 行對教材資料的描述都對：
  - ShapeDataset 預設 size=64，docstring 寫紅=0、藍=1，畫的是矩形；第 7 章各節都用 size=64。
  - 第 1–4 章需要圖片的實驗，尺寸分別是 32×32、16×16、40×80、32×32，都是紅／藍矩形；要分類的那幾節，類別都只由顏色決定。
  - 張數：01、03 各 8 張，04 是 2 張，7.4 是 32／16／16，都遠少於候選規格的 24 張檢查集和 1,500／300／300。
- 「42 節教材都沒有用到 Fashion-MNIST」成立：section-map 的 43 筆 datasets 都是 []；lesson_cases、notebooks、docs/lessons 都沒有出現 fashion；miniyolo/classification.py 沒有任何一節 import。
- data.md 有「讀取與 40 步訓練核對」一節；fashion-mnist-learning.json 的 steps 是 40。
- 42 個 lesson 頁的最後一個 H2 都是「實際執行紀錄」。
- data/manifest.json 裡兩個檔的 bytes 和 SHA-256，工作樹與 HEAD 相同，也和本頁的表格相同。
- 對照 r1 的快照：Fashion README 的四個 MD5、CIFAR 頁面的 MD5 和「163 MB」、Pet 頁面的 CC BY-SA 原文、torchvision loader 的 MD5 95a8c909… 都相符。
- 對照 probe.log：Pet 的各項數字都相符，包括 3,686 個 XML、7,390 個 trimap、split 3,680／3,669、9 筆缺 XML、41 個多出來的 trimap、class id 1–37，以及 Abyssinian_1 的尺寸、框與兩個 bbox。
- 兩組加總都算過：30,878,645 和 19,257,361。
- repo 裡沒有原始查核紀錄（git ls-files 查過），也沒有 Pet 的盤點程式。

3. 沒有遺失重要內容：打包發布前要先查核作者更新說明的安全步驟還在；19.26 MB 的下載量明確限定在「本查核」；原始紀錄不在 repo 的說明改寫在第 83 行。

4. 渲染：我在暫存副本執行：
- zensical build --clean --strict：exit 0，No issues found。
- validate_site.py：exit 0。
- validate_preparation.py：exit 0。
- 輸出的頁面有 4 個表格、1 個引言區塊，3 個站內連結都正常，沒有殘留的 Markdown 符號。
- 日誌在暫存副本。

5. 這個部分只改了 docs/research/foundation-data.md。工作樹其他的差異是 course-research、feedback、status 和各程式部分的修改，依查核分配都屬於別的部分。

另外有一點觀察，我沒有列成問題：第 70 行寫「XML 的 object name 只有 cat/dog」，這比第 76 行描述的讀取範圍（只讀出一份 XML）強。不過我用 SHA-256 相同的檔案對全部 XML 核對過，只有 cat 1,189 筆、dog 2,498 筆，這句話是真的，沒有讀者會因此做錯決定，所以不建議為這點再改一輪。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/research/foundation-data.md 第 5 行「依〈[教學大綱](../planning/outline.md)〉第 1–4 章與第 7 章的資料約定」 | 本頁用〈〉括起的是頁面名稱：〈資料規劃〉、〈全套實驗與審查〉都和導覽列的名稱一致。只有〈教學大綱〉對不上：那一頁現在的導覽名稱是「課程大綱」（zensical.toml 第 82 行），H1 是「從小 CNN 到 MiniYOLO：課程大綱」，feedback.md 和 architecture.md 也都叫它「課程大綱」。「教學大綱」是從 HEAD 的舊標題「教學大綱第二版」去掉版本字樣得來的，outline 部分已經把那一頁改名。連結本身沒壞，讀者不會卡住，但在導覽列或搜尋裡找不到這個名稱。 |

建議事項的處理（處理後由下一次複查檢查）：

- [should] 第 5 行〈教學大綱〉：已修正。改成〈[課程大綱](../planning/outline.md)〉，連結目標不變。依據有兩項：zensical.toml 第 82 行的導覽名稱，以及 outline.md 的 H1「從小 CNN 到 MiniYOLO：課程大綱」。同時確認〈課程大綱〉第 1–4 章與第 7 章確實有本句所說的資料約定：第 1 章的圖片契約、第 3.3 節固定資料、第 4 章的 bbox 格式與座標及非正方形 resize／padding、第 7.1 節的幾何資料與空圖。同一句的查核日期 2026-10-02 和「只用標準函式庫」都和 HEAD 的原紀錄一致。也抽查了頁內其他事實：ShapeDataset 紅=0／藍=1、64×64；Fashion-MNIST 只由 scripts/run_fashion_cnn.py 使用，不在任何 lesson_cases；data/manifest.json 兩項的 bytes 與 SHA-256；資料規劃頁的 40 步核對；42 個課程頁都有「實際執行紀錄」標題。這些都成立。在暫存副本跑 zensical build --clean --strict 與 scripts/validate_site.py，exit code 都是 0；產出頁面的連結文字是「課程大綱」，指向 ../../planning/outline/。

### 第 3 次複查：通過

結論：通過。沒有必要問題，只有一條 should：第 93 行的「split」範圍寫得太寬。

先說明一件事：repo 的工作樹是乾淨的，所以 `git diff -- docs/research/foundation-data.md` 沒有輸出。這是因為本版的改動已在 00:29:52 提交進 463d3f5。本版對這頁的變更要看 `git diff 0c94e35 463d3f5`。我檢查的是整頁目前的內容（工作樹和 HEAD 相同）。

各項結果：

(5) 建置與網站驗證：通過。我用 rsync 把 repo 複製到暫存副本，排除 .git、兩個 venv、site 和 .cache。在這份複本裡：
- `zensical build --clean --strict` 結束碼 0，輸出「No issues found」。
- `scripts/validate_site.py` 結束碼 0：60 頁，連結與錨點、Colab 配對、Markdown 轉換、artifact 邊界都通過。
- 建出的頁面有 4 張表、1 個引言區塊。三個站內連結分別指向 ../../preparation/data/、../../validation/curriculum/、../../planning/outline/。
- 紀錄檔。

(1) 敘述是否屬實：除了上述建議，全部成立。逐項核對如下：
- 引言區塊說第 1–4 章與第 7 章只用紅、藍矩形，成立。我讀了這兩部分的全部 13 支 lesson_cases：
  - 01、03-comparison、04-localization、04-coordinates 畫的是紅／藍矩形，圖的大小在 16×16 到 40×80 之間；要分類時，類別由顏色決定。
  - 02、03-identity、03-projection 不用圖片。
  - 第 7 章用 `ShapeDataset`（紅=0、藍=1，預設 64×64），7.1 和 7.5 另外手畫紅／藍矩形。
  - 40 步學習實驗直接沿用 lesson 的資料；`miniyolo.train` 預設 32／16／16 張。
- Fashion-MNIST 只出現在 `scripts/run_fashion_cnn.py`（經由 miniyolo/classification.py）。lesson_cases、notebooks、docs/lessons 都沒有它，section-map 裡每一節的 datasets 都是空的。
- 頁面名稱和 zensical.toml 的導覽一致：〈資料規劃〉、〈全套實驗與審查〉、〈課程大綱〉。docs/ 和 README.md 裡已經沒有「教學大綱」這個字。
- 〈課程大綱〉第 1、3.3、4、7.1 節確實寫有本頁引用的資料約定。舊版大綱的章號和現在相同。
- 〈資料規劃〉確實有下載步驟和「讀取與 40 步訓練核對」一節。
- `data/manifest.json` 的 `oxford-iiit-pet` 和 `fashion-mnist` 兩項，bytes 與 SHA-256 都和本頁相同。
- repo 裡沒有本查核的原始紀錄，也沒有盤點 Pet 標註的程式。
- 42 個課程頁最後一個 H2 都是「實際執行紀錄」，這個標題由 `scripts/verify_curriculum.py` 產生，發布重建時也會保留。
- 我另外寫了一支只用標準函式庫的檢查程式，跑在暫存副本裡先前下載的 annotations.tar.gz 上（它的 bytes 與 SHA-256 和頁面相同）。以下數字全部相符：
  - Pet：3,686 個 XML、7,390 個 trimap；trainval 3,680 筆、test 3,669 筆，兩個 split 沒有重疊；trainval 有 9 筆缺 XML，test 全部沒有 XML；41 個 trimap 不在兩個 split 清單裡。
  - class id 是 1–37，species 是 1／2；全部 XML 的 object name 只有 cat 和 dog。
  - Abyssinian_1：XML 尺寸 600×400×3，頭框 (333,72,425,158)；trimap 的 CRC 全部相符，畫素值只有 1／2／3；值 1 的框是 [107,81,444,328)，值 1 或 3 的框是 [92,66,460,343)。
  - Fashion test labels：解壓後 10,008 bytes，magic number 2049，共 10,000 個 label，每類 1,000 個。
  - 程式與輸出：暫存副本和 .log。
- 外部連結和官方來源：
  - 頁上 12 個外部連結都回 200，轉址目的地和頁面記錄的一致；7 個 HEAD 請求回報的大小也和頁面完全相同。
  - Pet 官方頁的兩個檔案連結和 CC BY-SA 4.0 那句原文都相符。
  - CIFAR 官方頁寫 163 MB，MD5 相符，頁上沒有授權文字。
  - Fashion README 有 MIT 那行，四個 MD5 相符，下載表確實是 S3 的 HTTP 網址。
  - torchvision 對 annotations 預期的 MD5 是 95a8c909…，和頁面相同。
  - 頁面的 MB／MiB 換算和各項加總都正確。
  - 頁面沒有提到 Colab、GPU 或審查範圍，所以沒有誇大這些的問題。

(2) 先前的發現：唯一一條是建議，把〈教學大綱〉改成〈課程大綱〉，已經處理好。沒有被駁回的建議。

(3) 修訂經過與計畫語氣：找不到「本輪、這次、修訂、已修正、尚待、之後、待測、日後」這類字眼；唯一命中的是「HTTP 回應本體」，不是在講修訂。第 4 節是以研究當日為準的選型建議，用現在式寫，不是待辦。

(4) 清楚程度：trimap、ROI、HEAD 請求、checksum、glob、tight bbox、CRC 都在原處附了解釋。「畫素」的用法和術語表一致。中英文之間的空格和全形標點，我用程式掃過，沒有問題。這頁的 H1 和導覽名稱不同，但其他非課程頁也都是這樣，屬於全站的慣例，不是這頁的缺陷。

維護者提到的「工作樹沒有 reviews/coverage.json」屬於審查用的事實與寫作規範清單列出、發布時才會補齊的暫時狀態，所以沒有列為這頁的問題。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/research/foundation-data.md 第 93 行「本頁的 byte 數、split、第 3 節的 checksum 與單一 trimap 結果是實測值」 | 這句的用途是界定本頁哪些數字是實測的。checksum 已經限定成「第 3 節的」，split 卻沒有限定。照字面讀，「本頁的 split」也包含第 30 行 Fashion-MNIST 的 60,000 train，以及第 31 行 CIFAR-10 的 50,000 train／10,000 test。這些數字取自官方 README 和官方頁；CIFAR-10 整包沒有下載，Fashion-MNIST 的 train 檔也沒有下載，所以都不是實測。真正實測的只有 Pet 的 split 盤點（第 66–68 行）。不列為必要的理由有兩個：一是那兩列各自寫明「官方 README：」「官方頁：」，CIFAR 列也寫了「完整資料未下載」；二是這些數字和官方來源一致（本輪已逐一對照）。所以讀者不會因此做錯決定。 |

### 第 4 次查核：通過

（這一輪同時查核：`docs/research/pages-colab.md`、`docs/research/lfs.md`、`docs/research/detection-data.md`、`docs/research/foundation-data.md`、`docs/planning/course-research.md`；下表只列和本頁有關的發現。）

結論：通過。沒有必要等級的問題，只有兩條建議，都在 pages-colab.md。

**這次查核的頁面改了哪些檔**
- 只改了 pages-colab、lfs、detection-data、foundation-data 四頁。
- course-research.md 與 HEAD 相同，HEAD 已經是 4 格縮排。
- 受程式改動影響的段落沒有這五個檔的條目。

**遺留意見的判斷：9 條都正確**
- pages-colab #1–#3、lfs #1–#2、foundation-data：已改，改法和程式一致。
- pages-colab #4：判為「事實已改變」是對的。依據有三：
  - verify_release.py 的 scope 明寫 "Colab itself was not opened"。
  - 審查用的事實與寫作規範清單寫的是 Verify published lessons workflow。
  - publish.md 第 6 節第 11 步現在就是這個 workflow，所以編輯者擔心的 publish.md 落差已經不存在。

**detection-data 的 must（COCO 的 HTTPS 網址）**
- 官方 download.htm 列的是 http:// 網址，頁面連結也是這組。
- 我在 2026-10-04 19:21 UTC 用 HEAD／range 重查：三個檔的官方 HTTP 網址與 S3 HTTPS 路徑都回 200／206，Content-Length、ETag、Last-Modified 兩邊相同；https://images.cocodataset.org 則是 curl exit 60。
- 其他說法也都對得上：
  - DNS 的 CNAME 指向 images.cocodataset.org.s3.amazonaws.com。
  - 憑證是 CN=s3.amazonaws.com，SAN 不含 images.cocodataset.org；urllib 回報 Hostname mismatch。
  - 單張圖片用 S3 HTTPS 也回 200。
- 暫存副本的 coco-probe-final.log（16:43:27Z）支持頁面寫的 16:43 UTC 結果。
- data/manifest.json 的 coco2017 是三個 S3 HTTPS 網址，status 是 candidate，與頁面一致。

**逐條核對程式**
- 對照的程式：validate_preparation、validate_lessons（code_lines、SOURCES）、review_coverage、validate_curriculum_evidence（finite_json、check_machine，以及 custom 1600 步、video、fashion 的內容檢查）、evidence_records、validate_site（含 markdown_list_problems）、build_lesson_notebooks 的 bootstrap()、pages.yml、verify-release.yml、verify_release.py、tests/test_notebook_bootstrap.py、zensical.toml、overrides/、docs/assets/（mathjax.js 訂閱 document$；MathJax 3.2.2）。頁面的敘述都成立。
- 外部來源：
  - PyPI 0.0.67 的 Requires-Python 是 >=3.10，wheel 為 cp310-abi3。
  - Zensical 數學公式文件確有 document$ 與 instant navigation 的整合說明。
  - git-lfs 在 0043a645 的 FAQ 有 `git lfs migrate import --everything`；GitHub 的 LFS 單檔上限單位是 GB。
- git 歷史裡最大的 blob 是 1.17 MB。
- 五頁的 82 個外部連結都回 200 或 206。

**實際照頁面做的步驟**
- 用 Python 3.12 建 .venv-docs、裝 requirements-docs.txt。
- 依序跑 5 個指令：
  - validate_preparation 0。
  - validate_lessons 只因所有導覽頁都「no review」而失敗。
  - validate_curriculum_evidence 只因紀錄還沒重產而失敗（00-warmup 的 case_sha256 不符）。
  - zensical build --clean --strict 0。
  - validate_site 0（含 numbered lists 檢查）。
- validate_lessons 與 validate_curriculum_evidence 的失敗都屬於審查用的事實與寫作規範清單列的「發布時才完成」狀態。
- zensical serve 預設在 localhost:8000，根路徑會轉到 /learn_to_yolo/。
- 用 Python 3.9 實跑：validate_lessons 與 validate_site 確實停在 `ModuleNotFoundError: No module named 'tomllib'`；用 3.10 語法解析這些腳本都通過。
- tests/test_notebook_bootstrap.py 6 項通過。

**數字**
- 沒有新加入 Mac 上量的數字，也沒有依紀錄而變的數字。
- 頁面的 byte 數、換算與加總都正確。
- 用 SHA-256 為 52425fb6… 的 annotations.tar.gz 重驗 Pet 的盤點：3,686／7,390／3,680／3,669／9／41，以及 Abyssinian_1 的 [107,81,444,328)、[92,66,460,343)、CRC，全部相符。

**敘述與一致性**
- 沒有敘述修訂經過，範圍限制都寫成現況。
- 與 status.md、validation/curriculum.md、architecture.md、publish.md、data.md、README 一致。
- 「像素」改「畫素」與術語表一致；連結文字〈驗證範圍〉與 status.md 標題相同。

**渲染**
- README 的 9 個相對連結都存在；43 本 notebook 都是有效 JSON。
- course-research 建置後只有一個 8 項的 &lt;ol&gt;，8 行「來源：」都在項目內；headless Chromium 截圖顯示編號 1–8。

**檔案位置**
- 紀錄與截圖：同路徑的暫存副本（prep.log、lessons.log、evidence.log、build.log、site.log、links.log、course-research-top.png）

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

讀了 reviews/research-foundation-data.md 四輪查核。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/research-foundation-data.md | 第 1 次的「必要問題的修正：」是空的（結果在 traces_handled）；第 1 次 should（第 83 行「結果沒有收進 repo」與範圍說明）與第 2 次 should（〈教學大綱〉名稱）的處理沒有列出。另有「派工單列的 9 條 trace 全部處理」。 | 已修正：產生器讀入修正時處理的項目與建議事項的處理（處理後由下一次複查檢查），不再輸出空的清單，每項發現都有對應的處理。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次（1 項必要、1 項建議）的處理清單已補上（line 3、line 95 處理必要問題，line 85 處理建議）；第 2 次的建議有處理清單；第 3 次的建議，由第 4 次確認已改。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/research-foundation-data.md 第 22、31–39、70、94 行 | 殘句與內部標籤：「查核下載的檔案放在暫存副本的暫存副本，建置與驗證日誌放在暫存副本。」「紀錄檔：暫存副本和暫存副本。」；處理清單的「[process-history/rewrite]」「[revision-history/delete]」「[other/rewrite]」；「依查核範圍清單都屬於別的頁面組」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：點名的路徑殘句都已不在；分類標籤改成中文（第 31–37 行）；「查核範圍清單／頁面組」已換掉。處理說明屬實。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[reference當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/reference.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/reference.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/reference.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-05：v0.6.0 有界更新審閱

局部核本版実驗數量42→52。

本次僅重查以上變更，既有正文的歷史審閱保留；沒有把全頁或全書重新標成首次盲讀。[導讀／實際網站審閱](clear-tutorial/vision-v0.6.0/guide-visual-review.md)、[新增支線第三輪及實際前置](clear-tutorial/vision-v0.6.0/transitions-review.md)、[首讀修後複查](clear-tutorial/vision-v0.6.0/vit-recheck.md)記錄各自範圍。全版52本notebook的本機cell執行另見[實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)；不是Google Colab登入執行。來源與圖／程式指紋更新於[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本附頁由獨立銜接與技術審閱者核對當前來源；頁內既有審閱歷史的接觸另記，未冒稱完全無歷史提示。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。
