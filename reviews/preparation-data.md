# 審查紀錄：資料規劃

審查範圍：`docs/preparation/data.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

判定：通過，沒有必要等級的問題；有 3 項建議等級的建議。

【1. 改版痕跡】9 條先前審查意見都已處理。全頁搜尋「本輪、本次、本雲端、本工作區、目前、尚未、仍待、待補、之後會、先採、已備妥、新增、修訂」等字眼，只剩第 47 行的「這次執行」；它指的是那份紀錄的那次執行，審查用的事實與寫作規範清單允許這種用法。40 步那段已經移到 Fashion-MNIST 一節底下，〈完整查核來源〉回到頁尾。幾何資料一節先寫 ShapeDataset 的實際行為，再寫延伸規格。

【2. 事實查核】逐項對照程式與紀錄：
- 下載與執行指令：download_data.py 預設存到 data/downloads/<id>/；--output /content/data 會存到 /content/data/fashion-mnist/；不能下載的項目會報 "Candidate only; see docs/preparation/data.md…"（exit 2）。run_fashion_cnn.py 的 --data-root、--train-steps、--subset、--eval-samples 與輸出位置 artifacts/runs/fashion-cnn/ 都和頁面一致。
- loader 與 ShapeDataset：loader 先核對大小與 SHA-256，再檢查 IDX，並在 seed 7 下切 6k／54k。ShapeDataset 的描述（紅＝0、藍＝1；預設 64×64；每張 0–2 個物件；4×4 格中每格最多一個物件；物件邊長 8–15 px）都和程式相符；miniyolo.train 的 seed 預設是 7／700／7000。
- 各章用的資料：第 1–4 章、5／7.2 的碰撞例、11.3 的裁切例、8.2 的紅藍黃三類 PNG＋JSON，都和程式相符。42 節教材沒有任何一節用到 Fashion-MNIST，也沒有任何一節要下載資料。
- 40 步紀錄：頁上的數字（0.140625＝18/128、0.09375＝12/128、loss 2.1086–2.3858、batch 8、lr 0.001）都和紀錄一致；紀錄裡確實沒有日期、電腦與程式版本。「梯度有限」這一點紀錄本身沒有存，但我查 git 歷史確認：產生這份紀錄的程式（0c94e35）已經有遇到非有限梯度就中止的檢查。
- LFS 封裝：workflow 與 verify_remote_lfs.py 從 run 的 head 5184c2f 到現在都沒改過。它們確實會核對封裝與四個檔的大小、SHA-256，以及 LICENSE 的內容。
- 其他：.gitattributes、.gitignore、「不需登入或 API key」（detection-data.md 有記載）、VOC 舊主機回 503 也都屬實。

【實際重跑】在暫存副本：
- 下載 Fashion-MNIST，四個 MD5 都和 manifest 相符。
- 用 package_fashion_mnist.py 重建封裝，SHA-256 和 manifest 相同；照頁面的 mkdir＋tar -xf 解開後，loader 不加 --data-root 就讀得到。
- 重跑頁面的 40 步指令：正確率與紀錄完全相同，loss 最大差 2.4e-7，符合頁面「最後幾位小數可能不同」的說法。
- 下載 Penn-Fudan ZIP：SHA-256 通過，testzip 無錯，圖片、mask、標註各 170 份；README 裡確實寫著 "may not be reposted without… permission"。

【3. 刪掉的內容】沒有遺失重要資訊。COCO 端點查核的那句雖然刪了，表格第 12 行和 detection-data.md 仍有記載。

【4. manifest】只改了 synthetic-rgb 的 status 與 format，新內容屬實。沒有任何驗證器或執行紀錄綁定 manifest 的 hash；download_data.py list 正常列出 "synthetic-rgb: generated"。

【5. 建置】在暫存副本執行：zensical build --clean --strict、validate_site.py、validate_preparation.py 都是 exit 0。頁面渲染出 2 個表格、9 個標題、6 個程式碼區塊，沒有殘留的 **。和資料相關的程式與紀錄（download_data.py、run_fashion_cnn.py、classification.py、data.py、兩支 LFS 腳本、validate_preparation.py、.gitattributes、.gitignore、各 LFS／Fashion 紀錄）相對 HEAD 都沒有改動。工作樹裡還有其他修改，看起來屬於同時進行的其他部分，我無法逐一歸屬。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/preparation/data.md 第 9 行（〈各階段的資料〉表格第一列第三欄） | 「Fashion-MNIST 分類，見下一節。」這門課的「節」都指教材的節次：同一張表第二列寫「8.2 節」，第 134 行寫「7.2 節」「11.3 節」。讀者可能把「下一節」當成下一節教材，而不是本頁下方的〈Fashion-MNIST：選用的真實分類資料〉。 |
| 2 | 建議 | docs/preparation/data.md 第 41 行（〈讀取與 40 步訓練核對〉第一個項目） | 本頁替 IDX、SHA-256、HEAD、Git LFS、trimap 都加了一句白話，「magic number」卻沒有說明，程式新手看不出 loader 在檢查什麼。另外，這句把「類別編號 0–9」和檔頭欄位寫在一起，看起來像是檔頭的一部分；實際上 loader 是讀完內容後才檢查每個標籤。 |
| 3 | 建議 | docs/preparation/data.md 第 91 行（COCO 列「像素 xywh」）、第 127 行（切分規則 5「像素或正規化座標」） | 課程統一用「畫素」：docs/lessons 有 138 行用「畫素」，「像素」一處也沒有；術語表和本頁第 9 行也寫「畫素」。同一頁兩種寫法混用，違反審查用的事實與寫作規範清單「術語與術語表一致」。第 127 行是這次改寫過的句子，仍然保留了「像素」。 |

建議事項的處理（處理後由下一次複查檢查）：

- [should｜第 9 行「見下一節」] 已採用：改成「見本頁下方〈[Fashion-MNIST：選用的真實分類資料](#fashion-mnist)〉」，加上「本頁」兩字，讀者就不會當成教材的下一節。建置後的 HTML 確實有 id="fashion-mnist"，validate_site 的錨點檢查也通過。
- [should｜第 41 行 magic number] 已採用並照程式修正順序（對照 miniyolo/classification.py）：檔頭檢查寫成「magic number（檔頭最前面、用來辨認檔案種類的固定數字，圖片檔是 2051、標籤檔是 2049）、張數與 28×28」；接著是讀完內容才做的兩項檢查：「核對內容長度和檔頭相符」（程式確實會檢查，原句漏了），最後才是「每個類別編號都在 0–9」。用「檔頭最前面」不用「檔案最前面」，是因為 .gz 檔自己開頭也有 gzip 的標記，寫「檔案最前面」會讓人分不清指哪一個。
- [should｜第 91、127 行「像素」] 已採用：兩處都改成「畫素」，與術語表和 docs/lessons 的用法一致（docs/lessons 有 138 處「畫素」，「像素」0 處）。本頁已沒有「像素」。

### 第 2 次複查：通過

結論：通過，沒有必要問題。只有兩個影響很小的 should（第 16 行段落引用沒寫位置、第 74 行先決條件放在指令後面），兩者都不會讓讀者做出錯誤的動作。

先說明檢查途中 repo 狀態有變：工作樹的改動已經提交成 463d3f5（分支 release/lessons-v0.4.0），所以現在 `git diff -- docs/preparation/data.md` 是空的。本次發布對這頁的改動要改用 `git diff e25db8d HEAD -- docs/preparation/data.md` 看。我逐檔比對過：已提交的 data.md，以及 manifest、download_data.py、run_fashion_cnn.py、classification.py、data.py、train.py、fashion-mnist-learning.json、prepare-lfs.yml、hosted-lfs-run.json、.gitattributes、zensical.toml、evidence_records.py、validate_curriculum_evidence.py，都和我審查時的版本逐位元組相同。

(1) 內容是否屬實：逐句對照程式與紀錄，全部成立。
- 下載器：`list` 會列出 manifest 的每一項。`fetch` 遇到 `generated`／`candidate` 狀態時，會以 parser.error 回「Candidate only; see docs/preparation/data.md…」。`--asset` 只取指定檔名；`--output` 的存放位置是「指定目錄/資料集 id/檔名」。既有檔案大小與 SHA-256 相符就沿用，不符就停下來要求先移走。
- loader 的檢查順序和頁面一致：先核對 gzip 的大小與 SHA-256，再查 magic number 2051／2049、張數與 28×28，接著核對內容長度和檔頭相符，最後確認類別編號在 0–9。
- seed 7 打亂，前 6000 張當 validation、其餘 54000 張當 train；`--data-root` 的預設路徑也和頁面相同。
- 紀錄裡的 loss 介於 2.1086 與 2.3858 之間（頁面寫 2.11–2.39，是取兩位小數）；正確率 18/128 與 12/128。這份紀錄確實沒有日期、電腦與程式版本。它也不在 record_evidence／validate_curriculum_evidence 的清單裡，所以發布時不會重產，頁面這句到時候仍然成立。
- 所有大小、SHA-256 與 84.60 MB 的加總都對得上。
- LFS：指標檔內容正確。封裝成員是 fashion-mnist/ 底下的檔案，解到 data/downloads 後 loader 不必指定 `--data-root`。工作流程各步驟與頁面描述一致，而且從那次執行的 head_sha 5184c2f 到現在，相關檔案都沒有改過。
- Penn-Fudan：用實際下載的 ZIP 驗證，SHA-256 相符，testzip 的 CRC 全部通過，PNG、mask、TXT 各 170 份，README 的再散布條款原文也和頁面說法一致。
- ShapeDataset 的預設值與 miniyolo.train 的 seed 7／700／7000 都對。第 1–4 章沒有用 ShapeDataset；第 5 章與 7.2 節的同格碰撞、11.3 節的裁切、8.2 節三色 PNG＋JSON 都對照過程式。
- 頁面沒有誇大 Colab、GPU 或審查範圍。

(2) 前一輪只有三個建議，都已正確落實，沒有被駁回的項目。

(3) 沒有敘述修訂經過，也沒有計畫語氣。

(4) 除了上面兩個建議，術語都有解釋，頁名也和導覽一致。維護者擔心的〈完整查核來源〉連結文字（帶「研究」後綴）不必改：publish.md 的來源清單也是這種寫法（「Pages／Colab 完整研究」「LFS 完整研究」），而本頁用〈〉寫出的頁名〈全套實驗與審查〉和導覽一致。

(5) 在暫存副本執行的結果：
- `zensical build --clean --strict`：exit 0。
- `python3 scripts/validate_site.py`：exit 0，錨點 #fashion-mnist 存在。
- 另跑了 validate_preparation：也是 exit 0。
- 頁面上的重跑指令實際執行：exit 0，結果寫到 artifacts/runs/fashion-cnn/result.json，正確率與紀錄相同；40 步裡有 7 步的 loss 在小數第 7 位附近不同，和頁面說的「最後幾位小數可能不同」相符。
- 紀錄檔都在暫存副本。

我沒有改動 repo。

另外兩件事不屬於這頁的內容，提出來給負責的人判斷：
- 發布前要補上涵蓋這頁定稿文字的 reviews/preparation-data.md，工作樹目前沒有這個檔。
- docs/lessons/20-deployment.md 第 247 行的標題用了「本輪」，審查用的事實與寫作規範清單把這個詞列為要避免的字眼；要不要改由那一節的負責人決定。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/preparation/data.md 第 16 行末句「候選資料請從〈偵測資料：大小、格式與下載〉表格的官方連結取得」 | 本頁另外兩處指向自己段落的〈〉引用都寫了位置：第 9 行「本頁下方〈Fashion-MNIST：…〉」、第 111 行「上方〈Git LFS 的可再散布封裝〉」。只有這一處沒寫。站內的〈〉多半用來寫頁面名稱，而導覽列「資料與平台查核」底下剛好有一頁叫「偵測資料」（research/detection-data.md）。讀者多半是看到下載器的「Candidate only; see docs/preparation/data.md」才來到這頁，可能會改點那一頁去找下載連結。那一頁也有官方連結，所以不會拿錯資料，只是多繞一趟，屬於清楚度問題。 |
| 2 | 建議 | docs/preparation/data.md 第 74 行第一句「沒有安裝 `git-lfs` 的電腦要先安裝它。」（接在第 67–72 行的指令區塊之後） | 這句要讀者「先」安裝，卻放在需要 git-lfs 的指令後面。照著頁面由上往下執行的讀者，要到區塊第三行 `git lfs install` 才會看到「git: 'lfs' is not a git command」（在沒裝 git-lfs 的電腦上執行 git lfs 指令，實際就是這個訊息）。這時第一行的 clone 已經做完；如果整段重跑，`git clone` 會因為 learn_to_yolo 目錄已經存在而失敗。最後結果不會錯，只是要多排錯一輪。 |

### 第 3 次查核：通過

（這一輪同時查核：`docs/preparation/architecture.md`、`docs/preparation/data.md`；下表只列和本頁有關的發現。）

結論：兩頁都通過，沒有必要或建議問題。檢查用的是暫存副本，兩頁內容與 repo 現況逐位元相同。

1. 程式、指令與路徑的敘述都正確，以下逐項對照過：
   - run_fashion_cnn.py 的 --record 會寫入 executed_at_utc、machine、torch、dependencies_sha256、data_sha256 與 train_fashion_cnn 的結果。綁定的程式實算為 12 個檔：腳本本身加 11 個 repo 模組，所以頁面寫「直接或間接 import」是對的。
   - evidence_records.py 的 CPU_RECORDS 有列 fashion-mnist-learning.json；record_evidence.py 先執行 download_data.py fetch 再重產；validate_curriculum_evidence.py 有檢查這份紀錄。
   - artifacts/checks/curriculum/ 在 git ls-files 裡共 52 檔：42 節、index.json、8 份 CPU 補充紀錄、deployment-gpu.json。grid-learning.json 不在這裡，所以「多數補充紀錄」成立。
   - 5 MiB 與 validate_preparation 的 5*1024*1024 一致；docs/ 裡沒有任何 PNG。
   - verify-release.yml 與 verify_release.py 的 site／bootstrap 兩項檢查，和頁面描述相符。
   - tests/、scripts/、.github/workflows/ 三行的描述都屬實；docs/research/ 那行與 zensical.toml 的導覽名稱一致。
   - 11-fusion 那項影響引用的 data-excerpt 約定（現在在第 53 行），與 validate_lessons.py 的 EXCERPT／PYTHON_BLOCK 規則相符。

2. 實際照頁面步驟跑過以下幾項：
   - download_data.py list 列出的狀態與頁面一致。
   - 對 synthetic-rgb、voc2007、coco2017、oxford-iiit-pet 執行 fetch，都回「Candidate only; see docs/preparation/data.md」，結束碼 2。
   - fetch fashion-mnist 有效：--asset 只下載指定檔、--output 存到 <dir>/fashion-mnist/、第二次執行顯示 Verified cache（沿用已校驗的快取）。
   - 照頁面的 40 步指令跑一次：結果寫到 artifacts/runs/fashion-cnn/result.json，紀錄檔的 SHA-256 前後不變。Mac 上的 loss 與紀錄最多差 2.4e-7，正確率相同。
   - --record 搭配其他設定時，parser 會拒絕執行。
   - package_fashion_mnist.py 重建的 tar，SHA-256 與 manifest 相符。tar 裡的檔案在 fashion-mnist/ 底下，用 -C data/downloads 解開就到 loader 的預設路徑。
   - record_evidence.py 的清單有把 fashion-mnist-learning.json 列為過期。

3. 數字：
   - 我自己下載 Pet annotations.tar.gz（SHA-256 相符）盤點，得到 3,686 份 XML，其中 trainval 3,671、不在 split 清單的 15、test 0、trainval 缺 9，與頁面相符。
   - Penn-Fudan ZIP 的 SHA-256 相符，testzip 沒有壞檔，圖片、mask 與標註 TXT 各 170 份。
   - COCO 實測：官方 download.htm 列的是 http 網址；https://images.cocodataset.org 因憑證不符而失敗；S3 路徑回 200，ETag 與 HTTP 網址相同。伺服器時間是 2026-10-04 UTC。
   - 30,878,645、84.60 MB、約 55 MB 等數值都核對過。
   - 編輯沒有加入任何在 Mac 上量的數字。依紀錄而定的 loss 範圍、正確率與 weights_changed，編輯都已列出。

4. 遺留項目的判斷都正確：
   - architecture 1、2(b)、3 與 data 1、2 都已修好。
   - architecture 2(a) 判為過時是對的；若照原本的要求的修正寫「沒有綁定程式」，現在反而會是錯的。

5. 用語與一致性：
   - 兩頁都沒有敘述修訂經過，與審查用的事實與寫作規範清單、status.md、README.md、publish.md、validation/curriculum.md 都沒有矛盾。
   - data.md 新加的 #detection-downloads 錨點，讓後兩個標題的自動 id 往前移一號（#_4→#_3、#_5→#_4）。全站沒有連結指向這兩個 id。

6. 建置與檢查：zensical build --clean --strict 的結果是 No issues found，validate_site.py 與 validate_preparation.py 結束碼都是 0。README 的相對連結都存在，43 本 notebook 都是合法 JSON。

以下不是頁面問題，交給發布流程注意：
- .github/workflows/verify-release.yml 與 scripts/verify_release.py 目前還沒有被 git 追蹤，發布時一定要 commit。
- data/pages-verification.json 仍被 git 追蹤，但沒有任何頁面引用。依審查用的事實與寫作規範清單，發布前應該刪掉。
- 編輯提的第一項疑慮（detection-data.md 與 manifest 的 COCO 網址互相矛盾）已經不成立：detection-data.md 第 43 行現在寫明 manifest 存的是 S3 HTTPS 網址。
- 兩頁的定稿審查要以現在的文字為準。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

讀了 reviews/preparation-data.md 三輪查核。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/preparation-data.md | 第 1 次 3 項、第 2 次 2 項 should 的處理都沒列，指向的〈定稿修正〉不存在（resolve:data 等結果沒被讀入）。另有「8 條 leftover 抽查…」「指派的 14 頁裡…」。 | 已修正：產生器讀入修正時處理的項目與建議事項的處理（處理後由下一次複查檢查），不再輸出空的清單，每項發現都有對應的處理。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次 3 項建議的處理已補上清單（上一輪要求）；第 2 次 2 項建議，由第 3 次逐項判定「data 1、2 都已修好」；頁面有指令，紀錄記了實際執行（download_data.py list／fetch、40 步指令等）；本頁沒有定稿修正，也沒有掛批次檢查。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/preparation-data.md 第 75、132 行 | 殘句與內部名稱：「紀錄檔都在暫存副本、暫存副本、暫存副本、暫存副本。」「編輯提的第一項 concern（detection-data.md 與 manifest 的 COCO 網址互相矛盾）」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 75 行點名的殘句已刪，第 131 行已改成「疑慮」。處理說明屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/preparation-data.md 第 3 輪第 1 項處理欄 | 「「疑慮」改成「疑慮」」是同語反覆。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
