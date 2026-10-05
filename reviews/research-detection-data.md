# 審查紀錄：偵測資料

審查範圍：`docs/research/detection-data.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題，另有 3 項建議建議。

1. 痕跡清理：清單上 9 條痕跡全部處理完（第 3、5、9、51、55、59、60、61 行與第 63 段）。全頁搜尋本輪、此次、本次、目前、暫、待、之後、後續、使用者、代理、子任務、workspace、現階段、blocker 等字眼，都沒有殘留。原本的簡體字「没有」「備选」已修正，沒有其他簡體字。

2. 事實核對（都成立）：
- lesson_cases/、miniyolo/、scripts/、notebooks/、tests/ 都沒有用到 Penn-Fudan、VOC、COCO、Pet，只有 metrics 的「非 COCO AP」說明會被搜到。橫幅與審查用的事實與寫作規範清單、status.md:26、outline.md:152、data.md 開頭一致。
- 我拿暫存副本自己重數過，它的 SHA-256 52425fb6… 與 manifest 相同：XML 3,686 個、trimap 7,390 個；trainval 3,680 筆、test 3,669 筆，兩者沒有重疊；trainval 有 9 筆缺 XML，test 全部沒有 XML。README 的引文逐字相符。官方頁的「All images have … head ROI」與 CC BY-SA 4.0 字樣也都對得上。
- data/manifest.json 確實有 PennFudanPed.zip 與 Pet annotations 的 sha256；頁面上八個 byte 數全部相符。
- foundation-data.md:66–67 有完整的盤點內容，連結過去找得到。
- 第 4 章、第 4–7 章、第 10–17 章的編號，在 HEAD 與目前的 outline.md 都一致。
- HEAD／200、range GET／206、Content-Length、multipart ETag 的白話說明都正確。
- detection-sources/ 和頁尾提到的各個樣本檔，用 git ls-files、find 都查不到，artifacts/checks 也沒有對應紀錄，所以那段刪掉是對的。

3. 沒有漏掉重要內容：被刪的「優先沿用課綱已提到的 VOC 路線」和「課綱一致」，在目前的大綱已經不成立（大綱只在「課程刻意不做的事」提到 VOC）。其餘被刪的是流程回報。範圍限制都還在：只讀檔案前段、沒有下載完整封裝、不把 ETag 當雜湊、不宣稱已驗證完整封裝。

4. 建置與渲染：在我自己的暫存副本中，zensical build --clean --strict 與 python3 scripts/validate_site.py 都是 exit 0。建出的頁面連到 ../../preparation/data/、../../planning/outline/、../foundation-data/ 都正確，表格與 blockquote 正常，沒有殘留的 Markdown 符號；git diff --check 也沒有問題。依修改時間判斷，這個部分只改了 docs/research/detection-data.md。

3 項建議建議：
- 連結文字「教學大綱」要改成目前頁名與導覽名「課程大綱」。foundation-data.md:5 也是同樣寫法，但不屬這次查核的頁面，請通知該檔 owner。
- 第 57 行要寫明 manifest 裡的雜湊來自另外的完整下載，避免讀者以為與第 27、51 行矛盾。
- 第 61 行的「實測之前」建議改成現況式的範圍限制；這項可由 owner 決定。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/research/detection-data.md:5「文中的章節編號依〈[教學大綱](../planning/outline.md)〉」 | 大綱頁目前的 H1 是「從小 CNN 到 MiniYOLO：課程大綱」，zensical.toml 導覽（第 82 行）也改成「課程大綱」，網站上已經沒有「教學大綱」這個名稱。站內〈〉引用的是頁名或導覽名，例如本頁的〈資料規劃〉。讀者照「教學大綱」去導覽列找，會找不到對應項目；連結本身可以點，所以不算誤導。 |
| 2 | 建議 | docs/research/detection-data.md:57「Penn-Fudan ZIP 與 Pet annotations 封裝的 SHA-256 記錄在 `data/manifest.json`。」 | 這句緊接在「本查核只確認 HTTP 可用性……也不宣稱已驗證完整封裝」之後。第 27 行說本查核「未下載圖片包」，第 51 行說「沒有下載完整圖片或標註包」。讀者看不出 manifest 裡的雜湊是誰、在哪次下載算的，可能以為兩句互相矛盾，或誤以為本查核下載過這兩個封裝。實際來源是另外的完整下載：manifest 的 penn-fudan availability_probe 寫 "Complete ZIP downloaded; size, SHA256, CRC and entry counts verified"，Pet annotations 的完整下載見 foundation-data.md:80。 |
| 3 | 建議 | docs/research/detection-data.md:61「……要以原型實測為準；實測之前不承諾訓練耗時或小模型品質。」（第 10、11 行的「原型實測」同類） | 「實測之前」預設之後還有一次實測，「原型」也沒有解釋。outline.md:152 現在寫課程刻意不在這些資料集上訓練，大綱的盤點也把原型視為已完成。讀者可能以為 VOC／COCO 的原型實測是待辦工作。這句是有日期的建議，內容沒有錯，所以列為建議而不是必要。 |

建議事項的處理（處理後由下一次複查檢查）：

- [should] 第 5 行〈教學大綱〉：已套用。連結文字改成〈[課程大綱](../planning/outline.md)〉，與 zensical.toml 第 82 行的導覽名稱、大綱頁 H1 一致。另依『用導覽名稱稱呼頁面』的規則，把第 51 行的〈基礎資料研究〉改成導覽名稱〈基礎資料〉；第 57 行新加的引用也用這個名稱。
- [should] 第 57 行 SHA-256 來源不清：已套用。現在寫明：Penn-Fudan ZIP 與 Pet annotations 這兩個封裝另有完整下載，`data/manifest.json` 裡的 SHA-256 是由完整檔算出的；校驗結果分別見〈資料規劃〉與〈基礎資料〉；本查核沒有下載這兩個完整封裝。已逐項對照：manifest 的 penn-fudan availability_probe；data.md 的 Penn-Fudan SHA-256 與每個檔的 CRC（各 170 份）；foundation-data.md 第 80 行 Pet annotations 的 SHA-256 與 torchvision MD5；data/local-lfs-verification.json。改後與本頁第 27、51 行『沒有下載完整包』一致，不再像互相矛盾。
- [should] 第 10、11、61 行「原型實測」：已套用，改成現在式的範圍限制，再加上採用時的建議，不再暗示有待辦的實測。第 10 行：『可先選數百張，用小規模實驗量過訓練耗時與結果，再決定要不要擴充』。第 11 行：『本查核不訂張數與下載預算，要採用時先用小規模實驗量測再決定』。第 61 行：『本查核沒有用這些資料集訓練；VOC／COCO 子集需要的張數、下載量、訓練耗時與 GPU 資源都沒有實測，因此不對訓練耗時或小模型品質下結論。要採用時，先用小規模實驗量測。』原句說完整封裝的下載量要靠實測，但表格已列出 Content-Length，所以改成指子集的下載量。主詞用『這些資料集』，避免讀成 Penn-Fudan 或 Pet 有訓練過。
- 驗證：rsync 到暫存副本，比對過副本與工作樹的頁面相同後建置；`zensical build --clean --strict` 與 `python3 scripts/validate_site.py` 都是 exit 0。產出的 HTML 中，課程大綱、資料規劃、基礎資料三個連結都指到正確頁面，「原型」「教學大綱」已不存在。改過的各行都通過 CJK 與英數之間的空格檢查。

### 第 2 次複查：有必要問題

有一個必要，所以不通過：第 42 行 COCO 的三個 https 下載連結因為憑證主機名不符而打不開，還被寫成「官方 canonical URL」；官方下載頁實際列的是 http 網址。第 18 行表格「均 HEAD 200／range 206」也因此對不上。其餘都通過。

逐項結果：
(1) 其他敘述都屬實。
- repo 端：導覽名稱與三個站內連結（資料規劃、課程大綱、基礎資料）一致；章節編號依現行大綱，從研究當時的 commit 起就沒變。lesson_cases、miniyolo、scripts、tests 都沒有提到這四個資料集，「沒有用這四個資料集訓練」成立。manifest 裡 Penn-Fudan ZIP（53,723,336 bytes，9095a9…）與 Pet annotations（19,173,078 bytes，52425f…）的 SHA-256 都在。data.md 第 96 行（Penn-Fudan 的 SHA-256 與 CRC）、foundation-data.md 第 66–67、80 行（Pet 的盤點與 MD5）都對得上本頁第 51、57 行。
- 本機副本：拿雜湊相符的 Penn-Fudan ZIP 與 Pet annotations 核對。README 引文、「這版新標了很小或嚴重遮擋的行人」、0 是背景的 mask 規則、PASCAL 1.00 格式、左上像素 (1,1)、FudanPed00001 的尺寸與框都相符；170/170/170、Penn 96／Fudan 74、每張至少一人。Pet：XML 3,686 份，trainval 缺 9 份，test 全缺；README 落在前 262,144 bytes 內；gzip magic 正確。46,575 bytes 的傳輸量算起來合理。
- 官方來源：VOC 9,963／2,501／2,510／5,011 與 Database Rights、COCO 的 0-based bbox、CC BY 4.0、small 範圍 0–32²、torchvision VOC 讀取程式用 thor、官方頁連結 301 轉到 thor、Pet 轉址到 /pets/、Pet 的 CC BY-SA 4.0、PyTorch 教學的 randperm 切分，都核對過。
- 頁面沒有過度宣稱 Colab、GPU 或審查範圍。

(2) 前一輪的三個建議都已正確套用，沒有被駁回的項目。

(3) 沒有敘述修訂經過，也沒有計畫語氣；「要採用時先量測」是附條件的建議。

(4) 頁名與導覽一致，CJK 與英數間的空格、全形標點掃描都乾淨。

(5) 在暫存副本（頁面與工作樹逐位元相同）中，`zensical build --clean --strict` 與 `python3 scripts/validate_site.py` 都是 exit 0。紀錄在同層的暫存副本。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/research/detection-data.md 第 42 行「可用下載：[train2017.zip](https://images.cocodataset.org/zips/train2017.zip)、[val2017.zip](…)、[annotations_trainval2017.zip](…)……優先保留官方 canonical URL」，以及第 18 行表格 COCO2017 列的「均 HEAD 200／range 206」 | 這三個 `https://images.cocodataset.org/…` 連結，在任何會驗證憑證的瀏覽器或程式裡都打不開，原因是網址設定本身，不是暫時故障。`images.cocodataset.org` 在 DNS 上是 S3 bucket 的別名（指向 images.cocodataset.org.s3.amazonaws.com）。S3 出示的憑證是 CN=s3.amazonaws.com，SAN 只涵蓋 `s3.amazonaws.com`、`*.s3.amazonaws.com` 這類名稱，沒有 `images.cocodataset.org`。2026-10-05 實測結果：curl 回報 `SSL: no alternative certificate subject name matches target host name`，Python 預設 TLS 回報 `CERTIFICATE_VERIFY_FAILED … Hostname mismatch`。COCO 官方下載頁（download.htm）列的是 `http://images.cocodataset.org/zips/train2017.zip` 這類 HTTP 網址，所以本頁把 https 版本當成「官方 canonical URL」也不對。表格的「均 HEAD 200／range 206」如果指的是這三個 https 網址，在目前的 DNS 與憑證設定下，會驗證憑證的用戶端不可能得到這個結果。實際影響：照著點的讀者會看到瀏覽器的憑證警告；用 wget 或 requests 下載的人會失敗，而常見的「解法」是把憑證驗證關掉。另外兩條路實測都可用：HTTP 官方網址與 `https://s3.amazonaws.com/images.cocodataset.org/…` 都回 206，Content-Length 與表格相符（252,907,541／815,585,330／19,336,861,798）。本頁其他 19 個外部連結都正常。這個錯誤在研究頁最初的 commit（5184c2f）就已存在。 |

修正必要問題時處理的項目（修正後由第 3 次複查確認）：

- [must] COCO 三個 https 下載連結、「官方 canonical URL」與表格「均 HEAD 200／range 206」：已修正，分五點。(1) 第 42 行三個連結改成官方 download.htm 列出的 http://images.cocodataset.org/zips/train2017.zip、/zips/val2017.zip、/annotations/annotations_trainval2017.zip，並註明這就是官方下載頁的 HTTP 網址。(2) 新增「**HTTPS 要換網址**」一條：說明 images.cocodataset.org 在 DNS 上以 CNAME 指向 S3 bucket，S3 出示的憑證是 CN=s3.amazonaws.com，SAN 都是 amazonaws.com 名稱，所以改用 https:// 會因憑證主機名不符而失敗（已實測：curl 回 exit 60 no alternative certificate subject name，Python urllib 回 CERTIFICATE_VERIFY_FAILED Hostname mismatch）。需要 HTTPS 時，整包下載與按張取圖都改用 https://s3.amazonaws.com/images.cocodataset.org/…（已用兩張樣本圖 val2017/000000000139.jpg、train2017/000000391895.jpg 驗證可用）；並註明不要為了連上而關掉憑證驗證。(3) 「優先保留官方 canonical URL」改寫成兩種網址各自的用途：記錄來源時寫官方 HTTP 網址，需要 HTTPS 時走 S3 路徑。(4) 表格 COCO 列改成「官方 HTTP 網址與 S3 HTTPS 路徑均 HEAD 200／range 206；images.cocodataset.org 走 HTTPS 會因憑證不符而失敗」。頁首與第 3 節寫明這是 2026-10-04 16:43 UTC（臺北時間 10-05）一次合併量測的結果：兩種網址三個檔都是 HEAD 200、range 206（Content-Range bytes 0-1023/<size>），Content-Length 與表格相符，ETag 與 Last-Modified 兩邊相同；查不到當初 https 主機名測試成功的紀錄，所以沒有寫成那個網址的結果。(5) 通知 manifest owner 的部分我沒有權限改那個檔，已寫進疑慮。
- [連帶，屬建議等級] 第 57 行「本查核只確認 HTTP 可用性」改成「只確認下載網址可用」。頁面現在區分 HTTP 與 HTTPS 網址，原句容易被讀成「只測了 http:// 網址」；改寫後原意不變。
- [連帶，屬建議等級] 第 26、43 行的「像素」改成「畫素」，與術語表和全站用語一致（docs 內「畫素」出現 390 次，術語表也用這個詞）。
- 建置驗證：在暫存副本執行 zensical build --clean --strict 與 python3 scripts/validate_site.py，exit code 都是 0。紀錄在同層的 detection-data-build.log 與 detection-data-site.log。渲染後的 HTML 中，三個 href 都是 http 網址，表格列數正確，新加的 URL 都放在 code span 裡，不會觸發 validate_site 的「URL shown as plain text」檢查。

### 第 3 次複查：通過

判定：通過。docs/research/detection-data.md 沒有必要，也沒有建議。

(2) 先前的發現都已解決。
- 第 1 輪的 must（COCO 的 https 連結失效、「canonical URL」、表格寫「均 HEAD 200／range 206」）已修正。我在 2026-10-04 16:48–16:49 UTC 獨立重測，頁面每一句都成立：
  - DNS：images.cocodataset.org 的 CNAME 指向 images.cocodataset.org.s3.amazonaws.com。
  - 憑證：subject CN=s3.amazonaws.com，SAN 全是 amazonaws.com 的名稱。
  - 改用 https:// 時，curl 回 exit 60，Python urllib 回 CERTIFICATE_VERIFY_FAILED Hostname mismatch。
  - 官方 http 網址與 S3 路徑式 https 網址，三個檔都是 HEAD 200、range 206；Content-Length 與表格相同，ETag、Last-Modified 兩邊一致。
  - 按張取圖：val2017/000000000139.jpg 與 train2017/000000391895.jpg 走 S3 路徑都可取得。
  - 官方 download.htm 對這些檔只列 http:// 網址。
  - 頁面已沒有「canonical」一詞。
- 先前潤稿檢查的紀錄的三條建議在現行文字都已解決：連結名稱改成〈課程大綱〉；寫明 manifest 的 SHA-256 來自另外的完整下載；「原型實測」「實測之前」改成現況的範圍限制加建議。沒有被駁回的建議。

(1) 其餘陳述我也逐項查過，都成立。
- Penn-Fudan：用 range 讀 ZIP，確認大小 53,723,336、HEAD 200／range 206、README 的版權引文與「新增標註小或遮擋行人」、FudanPed00001 的尺寸與兩個框，以及 ZIP 內沒有 split 清單。官方頁的 170 張／345 人／96＋74 張也相符。
- VOC2007：thor 主機的兩個 TAR 大小相符、HEAD 200／range 206；www 連結以 301 轉到 thor；torchvision 的 voc.py 用 thor；Database Rights 原文相符；總數 9,963、train 2,501／val 2,510／trainval 5,011 相符；000005.xml 有 5 個 chair，含 truncated／difficult，也有 owner／source。
- COCO：format-data.htm 寫明 bbox 是 0-indexed；termsofuse.htm 的 CC BY 4.0 與「不擁有圖片版權」相符；download.htm 有 118K/5K 用同一批圖片、以及 FiftyOne 的介紹；cocoeval 的 small 面積範圍是 0–32²。
- Oxford-IIIT Pet：兩個檔的大小相符；先回 308 轉到 /pets/…；伺服器忽略 Range，對 range 請求回 200 並傳整個檔；官網的 CC BY-SA 4.0 原文相符；README 的 research only 引文落在前 262,144 bytes 內。
- PyTorch 官方教學確實用 randperm 隨機留出 50 張。
- 與 repo 的一致性：data.md、foundation-data.md（trainval 缺 9 份 XML、test 全部沒有 XML）、manifest 的 SHA-256、大綱的章節編號，都與頁面一致。「沒有用這四個資料集訓練」與審查用的事實與寫作規範清單、outline.md:152 一致。
- 頁面沒有過度宣稱 Colab、GPU 或審查範圍。
- 舊主機 host.robots.ox.ac.uk 現在直接拒絕連線；頁面寫的是「查核時回 503」，屬於有日期的陳述，不構成矛盾。

(3) 沒有敘述修訂經過，也沒有計畫語氣。頁面中兩處「改成」都是操作說明（把網址改成 https://、不可把 crowd 改成單物件）。

(4) 站內引用的頁名〈資料規劃〉〈課程大綱〉〈基礎資料〉都與導覽列一致；「像素」已全部換成「畫素」；bucket、multipart ETag、HEAD、range 都有白話解釋。

(5) 建置：用 rsync 把 repo 複製到暫存副本（排除 .git、.venv-*、site、.cache；docs 與 repo 內容相同）。
- zensical build --clean --strict：exit 0，輸出「No issues found」。
- python3 scripts/validate_site.py：exit 0，各項都 passed。
- 紀錄檔。
- 渲染結果：三個 COCO 的 href 都是 http 網址；S3 網址都在 code span 裡；表格 5 列，COCO 列 4 格。
- 網路查核的原始輸出。

以下兩點不是本頁的問題，記下供參考：
- data/manifest.json 第 107、112、117 行的 coco2017 網址仍是 https://images.cocodataset.org/…，我實測確認會因憑證不符而失敗，所以維護者提出的疑慮成立，應轉給 manifest 的 owner。目前沒有程式受影響：coco2017 是 candidate，list 只印 id／status／purpose，fetch 會拒絕 candidate，validate_preparation 也略過非 download-ready 的資料集。docs 與 README 裡都沒有這個失效網址。日後若升為 download-ready，能同時通過 validate_preparation 的 https 檢查與 urllib 憑證驗證的，只有 S3 路徑式的 https 網址。
- scripts/review_coverage.py 列出導覽列全部 59 頁都還沒有審查紀錄。這是工作樹此刻所有頁面共同的狀態（定稿審查在發布時才寫入），不是本頁的缺陷。

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

讀了 reviews/research-detection-data.md 三輪查核。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/research-detection-data.md | 第 1 次 should #2（manifest SHA-256 那句）與 #3（「實測之前」「原型」）的處理沒有任何地方提到；第 2 次複查的 must（COCO https 連結）沒有列出修正，只有第 3 次寫了結果。指向的〈定稿修正〉不存在；resolve:detection-data 與 final:detection-data 沒被讀入。 | 已修正：產生器讀入修正時處理的項目與建議事項的處理（處理後由下一次複查檢查），不再輸出空的清單，每項發現都有對應的處理。第 2 次複查的必要問題（COCO 連結）也列出修正，並由下一次複查檢查。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次 3 項建議的處理清單已補上（上一輪指出 #2、#3 沒有處理）；第 2 次的必要問題（COCO 連結）有修正清單，並由第 3 次複查檢查；第 4 次通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/research-detection-data.md 第 26、89、108、110、113 行 | 內部名稱與殘句：「polish/detection-data.json 的三條 should 在現行…」「maintainer 提出的 concern 成立」「紀錄檔：同層的暫存副本與暫存副本。」「網路查核的原始輸出：同層的暫存副本、暫存副本、暫存副本、暫存副本、暫存副本。」「在我自己的副本暫存副本中」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 26 行已清理，第 89 行改成「先前潤稿檢查的紀錄」，第 113 行改成「維護者提出的疑慮」，路徑清單已清理。處理說明屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/research-detection-data.md 第 108、110 行 | 清單只剩「- 紀錄檔。」「- 網路查核的原始輸出。」兩個空項。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
