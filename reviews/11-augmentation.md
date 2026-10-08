# 審查紀錄：圖與框同步增強

審查範圍：`docs/lessons/11-augmentation.md`、頁面上的圖（`docs/assets/diagrams/11-augmentation.svg`），以及 `lesson_cases/11-augmentation.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題，只有兩個小的 should（見下表）：說明摘錄裡 `...` 的意思，以及「切圖」改成和全頁一致的「裁切」或「裁圖」。所有檢查都在暫存副本裡跑，日誌放在同目錄的暫存副本。真實 repo 裡沒有執行任何東西。

1. 程式相關敘述全部屬實（對照 lesson_cases/11-augmentation.py 逐句核對）。
   - 程式結束碼 0。四行輸出和頁面〈預期輸出〉、紀錄 JSON 的 stdout 逐字相同。
   - 練習：照頁面指示走一次第 3 題。在同一個 namespace 先跑 notebook 最後一格，再執行新增那格的程式。印出 [[0.0, 4.0, 8.0, 20.0]] [True, False]，和頁面註解相同。第 1、3 題的手算中間值也用 crop／horizontal_flip 核對過。
   - 第 76 行的反例自己重算過：從 x=12 開始切時，紅色在新圖占 x=0…11；框內 128 個畫素全紅；框外 x=8…11、y=4…19 有 64 個紅畫素；整張圖比對結果為 False。把實際程式改成從 left−4 開始切，結果是 AssertionError、stdout 0 行，符合第 131 行說的「這四行一行也不會印出」。
   - torch.equal（torch 2.9.1）：shape 不同時回 False，連可以 broadcast 的 shape 也一樣；dtype 不同時回 True。頁面只說比對 shape 和值，沒說比對 dtype，這是對的。
   - 第 61 行和〈常見錯誤〉提到的錯法都重算過：用 W−1 得 [39,12,55,28]；沒交換 x1、x2 得 [56,12,40,28]；翻錯軸；以上三種翻兩次都會還原。沒 clone 時得 [40,12,24,28]；flip(0) 會把紅色換到藍色通道。
   - 對照 miniyolo/targets.py 確認了：build_targets 遇到零面積框、或 labels 和 boxes 長度不同時，會拋出 ValueError；沒有 ignore 格；沒有框時 objectness 全是 0。miniyolo 裡也沒有任何增強。

2. 先前審查意見和受程式改動影響的段落都處理了。
   - 第 3 行和內文的 Colab 連結都已是 lessons-v0.4.0（HEAD 裡就已是）。
   - 原第 119 行承認「只核對框內」的那句已刪掉，改成描述程式現在的檢查方式。
   - 第 71 行的選配補句已採用，而且寫成限定範圍的版本，沒有「抓不到裁錯位置」這種說太廣的句子。
   - 頁尾執行紀錄沒有動。
   - 全頁 grep 不到修訂過程的敘述用語。「中文註解是本頁加的」是全站統一的寫法，不算修訂敘述。

3. 數字：新加的數字都是固定值，都重算過，沒有這台 Mac 的計時或機器數字。〈預期輸出〉和頁尾已列在編輯回報的待重錄數值清單裡。目前紀錄的 case_sha256 還是 62189e…，程式已經是 5665b9…。這是發布前重產紀錄之前的暫時狀態，不算問題。

4. 摘錄：摘錄比對工具對真實 repo 和副本都印出 []。我確認它真的比對到兩個區塊（第 50、84 行），因為什麼都沒比對到時也會印 []。在副本裡把兩個被摘錄的程式行各改壞一行，兩個區塊都被抓出；還原後和真實 repo 逐位元相同。參考答案裡的程式片段不是摘錄，驗證器也沒把它判成逐字照抄。正文提到的「第 1～4 行」都是指輸出行，沒有程式行號。

5. 可讀性：新段落的順序講得通，torch.equal 有解釋；AssertionError 在前面的章節已經介紹過。

6. 建置：副本裡 zensical build --clean --strict 和 scripts/validate_site.py 都成功（結束碼 0）。產生的 HTML 裡，兩個摘錄區塊都帶 data-excerpt 屬性，標記沒有漏成內文文字。SVG 這次沒改；它有 viewBox、title、desc，用 qlmanage 轉成圖檔看過，畫面乾淨，數值和程式一致。這一節只改了頁面：notebook 最後一格和程式逐字相同，source_ref 是 lessons-v0.4.0；SVG、紀錄、reviews 都沒動。副本裡 validate_lessons.py 失敗在 09-anchor-clustering 和 20-deployment 的摘錄，validate_curriculum_evidence.py 停在 00-warmup 的過期紀錄。這兩項都屬於其他部分，和本頁無關。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/11-augmentation.md，Crop 小節第二個摘錄區塊（第 82–103 行），包括引言句和單獨一行的 `...` | 區塊中間那行 `...` 代表「這裡省略了幾行」，但頁面沒有說。多數同類頁面（06-decode-nms、07-training、10-multiscale、16-dfl-free 等）都在引言交代 `...` 的意思。程式新手可能以為 `...` 是要執行的程式碼，因為在 Python 裡 `...` 本身就是合法的運算式。另外，main() 裡摘出的兩行在程式中並不相鄰：`cropped, clipped, keep = crop(...)` 與 `cropped_labels = labels[keep]` 中間還有 `assert torch.equal(clipped,torch.tensor([[0.,4.,8.,20.]]))`，摘錄沒標出這處省略，讀起來像連續的兩行。這不會讓讀者誤解程式行為，所以列為建議。 |
| 2 | 建議 | docs/lessons/11-augmentation.md 第 76 行「假設切圖時從 x=12 開始切」 | 全頁只有這裡用「切圖」，其他地方都叫「裁切」或「裁圖」（第 82 行、程式註解）。舊的編輯決定 I11 要求全頁對裁切固定用「裁切（crop）」。讀者不會因此誤解，只是同一個操作多了一個名字。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：讀了：審查用的事實與寫作規範清單、docs/lessons/11-augmentation.md 全文、lesson_cases/11-augmentation.py、notebooks/11-augmentation.ipynb 各格、docs/assets/diagrams/11-augmentation.svg 的原始碼，以及 docs/glossary.md。為了對照前面教過的內容，也讀了 07-data、07-targets（4×4 格、gx/gy、負責格）、07-inference 第 55 行（GT 不能 clamp）、04-coordinates（「畫素之間的格線」）、05-assignment、02-diagnostics（`&`），以及 11-csp、11-fusion、11-iou-loss 的開頭、miniyolo/targets.py（build_targets 會在哪些情況報 ValueError），並 grep miniyolo/ 確認訓練程式沒有接上增強。另外看了 zensical.toml 的導覽和舊的 reviews/11-augmentation.md。

建置：在暫存副本執行 zensical build --clean --strict，exit 0、No issues found；從 site/lessons/11-augmentation/index.html 抽出 article 逐段讀渲染結果（摺疊區、程式區塊、圖片）。Playwright 的瀏覽器沒有安裝，所以沒做截圖。SVG 用 qlmanage -t -s 1200 轉成 PNG，逐項核對座標比例、gx/gy 標示、兩個負責格、裁切區與灰色被裁掉的部分。

執行：PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/11-augmentation.py，exit 0，四行輸出和頁面逐字相同。

練習：模擬 notebook，先在同一個 namespace 執行完整程式，再照頁面原樣新增一格跑第 3 題的程式，得到 [[0.0, 4.0, 8.0, 20.0]] [True, False]，和註解一致；另外核對藍框單獨裁切是 [24,28,32,32]、篩完的 labels 是 [0]。第 1 題用 horizontal_flip 得 [0,0,64,64]。第 2 題用門檻 0.4、0.5、0.5000001、0.6 試，依序是保留、保留、刪除、刪除。

常見錯誤：寫了幾個寫錯的版本逐一執行，包括 W−1、忘了交換 x1 和 x2、flip(0)、flip(1)、沒有 clone、切圖從 x=12 開始、用 > 取代 >=。每一個都停在頁面暗示的那個斷言；前三種錯誤（W−1、沒交換、翻錯軸）翻兩次確實會還原；沒有 clone 時得到 [40,12,24,28]。flip(0)、flip(1) 實際的畫素位置和頁面描述不完全一樣（見發現 4）。用 build_targets 驗證：零面積框、labels 長度不符都會報 ValueError；門檻 0.6 的空標註讓 objectness 全為 0；負責格 flip 前是 (1,1)、flip 後是 (3,1)，裁切再 ×2 後是 (0,1)。

其他：在暫存副本跑 scripts/validate_lessons.py，只有審查涵蓋（no review）的失敗，屬於工作樹的暫時狀態。比對兩段程式摘錄，和程式逐行一致。另外要說明一件事：第一次轉 SVG 時，我不小心把 PNG 寫進了旁邊的暫存副本目錄暫存副本（內容是同一張 SVG 的轉檔）；之後都只寫在自己副本的 _out/。沒有在 repo 根目錄裡執行或寫入任何東西。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | 〈Crop 會改座標，也會改可見面積〉第 2 步「把座標夾回（clamp）新圖的邊界 0 和 32 之間」（Markdown 第 72 行），以及 crop 摘錄裡 clamp 的註解（第 89 行） | 照順序讀的讀者學過兩條規則。7.2 節說不能用 clamp 掩飾錯誤；7.5 節說「這兩步只用於模型的預測。如果是標註（GT，真值框）超出圖片，代表資料錯了，要回頭修資料，不能用截斷（clamp）把錯誤蓋過去」。本頁卻直接對標註框 clamp，而且用的是同一個詞，卻沒有一句話說明兩者為什麼不衝突。讀者會以為前面的規則被推翻；或者得出錯的結論，以為標註超出範圍時夾回去就好，之後在自己的資料上照做。 |
| 2 | 建議 | 〈Flip〉的「框座標代表畫素之間的格線」與「畫素／框」兩條條列（第 43–46 行）；圖 ② 對照卡「中心 x 壓在格線上，取 floor 歸右邊那格」；第 65 行「4×4 格（每格 16 畫素）」 | 同一頁、同一張圖旁邊，「格線」和「格」各有兩種意思。條列裡的格線是相鄰畫素之間的分界（每 1 畫素一條），連單一畫素也叫「一格」（「畫素 i 占據格線 i 到 i+1 之間的一格」）。圖卡和第 65 行的格線則是 4×4 格的格線（每 16 畫素一條），「格」指負責格。讀者剛看過圖卡，再讀到「x1、x2 本身就是格線」，很容易以為框邊要落在 16 畫素的格線上，但紅框的 x1=8 並不在那種格線上。 |
| 3 | 建議 | 第 78 行「這個 0.5 是可見比例的門檻（threshold）」；對照 docs/glossary.md〈門檻（threshold）〉 | 術語表的「門檻」只列 score 門檻兩種、IoU 門檻兩種，還寫「四種門檻用途不同」，看起來已經列完。讀者照著查，找不到本頁的可見比例門檻。而且這裡的值剛好也是 0.5，很容易和 AP50、NMS 的 IoU 門檻 0.5 混在一起；但可見比例是同一個框裁切後剩下的面積除以原面積，不是兩個框的重疊。 |
| 4 | 建議 | 〈常見錯誤〉第一條「翻錯軸」（第 143 行） | 我把 horizontal_flip 裡的 image.flip(-1) 改成 flip(0)、flip(1) 實際執行。flip(0)：紅色移到 channel 2 變成藍色，但圖沒有左右翻，色塊仍在 x=8…23，框卻照公式移到 [40,12,56,28]。flip(1)：紅色移到 y=36…51，x 仍是 8…23，框也是 [40,12,56,28]。頁面只寫「紅色變藍」和「框的 y 卻沒改」，讀者會以為 flip(1) 只是上下對不上，自己試了才發現 x 也完全錯開。 |
| 5 | 建議 | 〈參考答案〉第 3 題後的核對程式（第 163–169 行） | 第 3 題最後問「keep 和 labels 會變成什麼」，核對程式卻只印 kept 和 keep，沒有印 labels。讀者照做只能核對一半，而「labels 用同一個 keep 篩」正是本節反覆強調的步驟。 |
| 6 | 建議 | 〈執行完整程式〉第 4 行輸出的說明（第 127 行）與第 131 行「它們印得出來，表示前面的斷言都已通過」；lesson_cases/11-augmentation.py 第 55 行 | 頁面說，空圖翻轉、裁切後「labels 仍是長度 0 的 long（Long[0]）」，又說第 4 行的固定文字靠前面的斷言撐著。但程式第 55 行，shape 檢查的是篩選後的 `empty_labels[empty_keep]`，dtype 檢查的卻是篩選前的 `empty_labels.dtype`。照頁面去讀程式的讀者會發現，「篩完仍是 long」其實沒有被斷言檢查。結論本身沒錯，因為布林索引不改 dtype，但頁面描述的檢查和程式不符。 |
| 7 | 建議 | 第 13 行「本節把它們固定」、第 139 行談固定 seed；lesson_cases/11-augmentation.py 第 24 行 `torch.manual_seed(7)` | 本頁專門講隨機增強和固定 seed，完整程式開頭也有 torch.manual_seed(7)，但整支程式沒有抽任何亂數。讀者容易以為本節翻轉、裁切之所以固定，是靠這個 seed；或以為程式裡藏著隨機增強。 |
| 8 | 建議 | 第 78 行與第 96 行程式註解的「True/False」、第 137 行的「validation/test」 | 術語表和其他頁大多寫「True／False」「train／validation／test」（全形斜線），本頁用半形，和寫作規則要求的全形標點不一致。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：在自己的暫存副本工作：暫存副本（照指示用 rsync 建立）。來源檔與檢查腳本放在旁邊的 11-augmentation-technical-src/，沒有寫入原 repo。

【一手來源】
1. YOLOv4 論文，arXiv 2004.10934v1。讀的是 https://arxiv.org/html/2004.10934 抽出的文字；PDF https://arxiv.org/pdf/2004.10934v1 也下載了，但本機沒有 poppler，無法顯示。
   - §2.2 Bag of freebies：「We call these methods that only change the training strategy or only increase the training cost as "bag of freebies." What is often adopted by object detection methods and meets the definition of bag of freebies is data augmentation.」前一句：「…without increasing the inference cost.」
   - §3.3：「Mosaic represents a new data augmentation method that mixes 4 training images.」
   - §3.4：「Bag of Freebies (BoF) for detector: … Mosaic data augmentation …」
2. ultralytics/yolov5 tag v6.0，用 git ls-remote 解析為 commit 956be8e642b5c10af4a1533e09084ca32ff4f21f；檔案從 raw.githubusercontent.com/ultralytics/yolov5/956be8e642b5c10af4a1533e09084ca32ff4f21f/ 下載：
   - utils/augmentations.py：函式清單是 Albumentations、augment_hsv、hist_equalize、replicate、letterbox（L92）、random_perspective（L125，含 Perspective／Rotation and Scale／Shear／Translation）、copy_paste、cutout、mixup（L265；L269「labels = np.concatenate((labels, labels2), 0)」）、box_candidates（L273；L207–209 呼叫與篩選；L278 判斷式）。
   - utils/datasets.py：__getitem__（L542–613；L549 load_mosaic、L554 mixup、L579 xyxy2xywhn 正規化、L596–599「if random.random() < hyp['fliplr']: img = np.fliplr(img) … labels[:, 1] = 1 - labels[:, 1]」）；load_mosaic（L670）。
   - data/hyps/hyp.scratch.yaml L31：「fliplr: 0.5」。
   - train.py L439：--hyp 預設為 data/hyps/hyp.scratch.yaml。

【課程檔案】
docs/lessons/11-augmentation.md、lesson_cases/11-augmentation.py、docs/assets/diagrams/11-augmentation.svg；miniyolo/targets.py、losses.py、data.py、geometry.py、models.py、train.py、custom_data.py；lesson_cases/07-data.py；docs/lessons/07-data.md、04-coordinates.md、04-localization.md、08-own-data.md；docs/glossary.md；zensical.toml 的 nav；notebooks/11-augmentation.ipynb（最後一格與 lesson 程式逐字相同）；artifacts/checks/curriculum/11-augmentation.json（stdout 與實跑一致）；scripts/build_lesson_notebooks.py；審查用的事實與寫作規範清單。

【執行的指令】
1. rsync 建立暫存副本。
2. 執行本節程式：PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg …/.venv-model/bin/python lesson_cases/11-augmentation.py。exit 0，印出的四行與頁面逐字相同。
3. python3 …/摘錄比對，結果為 []。
4. qlmanage -t -s 1200 渲染 SVG，看了 PNG，並逐一核對座標換算：比例 3、各圖原點、格線、負責格、中心點都正確。
5. zensical build --clean --strict：exit 0，No issues found；也檢查了輸出 HTML 的摺疊區塊、程式區塊、圖與連結。
6. 自寫 check_claims.py 和兩段 python -c 驗證頁面所有手算值：
   - 翻轉：沒有 clone 得 [40,12,24,28]；用 W−1 得 [39,12,55,28]；忘了交換得 [56,12,40,28]；三種錯法翻兩次都會還原。flip(0)／flip(1)／flip(-1) 後紅色或藍色的範圍；畫素 23→40、8→55。
   - 裁切：從 x=12 切時框內 128 個紅畫素、框外 64 個；門檻 0 又沒有 visible>0 時會留下零面積框，build_targets 拋 ValueError；labels 數量不符時拋 ValueError。
   - 練習 3 程式的輸出；空標註時 objectness 全為 0；放大 2 倍後在 (gy=1, gx=0)；flip 前後負責格是 (1,1)→(1,3)；YOLOv5 的 1−cx/W 與本節公式一致（都是 0.75）。
   - 空圖篩選後的 dtype；裁切區超出原圖時圖的 shape 是 (3,32,16)；>= 0.5 保留、0.6 刪除；torch.equal 在 shape 不同時回傳 False。
   - build_targets 用 image_size=64 與 32 的差別；GridDetector 輸入 32×32 得到 [1,4,4,7]；每個索引固定 seed 時，亂數每輪重複。
7. grep：檢查 miniyolo 與 lesson_cases 都沒有增強；掃描頁面，沒有敘述修訂經過的字眼。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/11-augmentation.md 第 143 行（常見錯誤「翻錯軸」） | 這條只說了一半的後果。把 `horizontal_flip` 裡的 `image.flip(-1)` 換成 `flip(0)` 或 `flip(1)` 時，框的公式照樣把 x 鏡射成 `[40,12,56,28]`，圖卻沒有左右移動。實跑結果：`flip(0)` 後物件變成藍色，位置仍是 x=8…23、y=12…27，框卻在 x 40…56，框住的是黑色背景；`flip(1)` 後紅色在 x=8…23、y=36…51，框是 `[40,12,56,28]`，兩者完全不重疊。頁面對 flip(0) 只說「紅色變藍、類別意義錯」，對 flip(1) 只說「框的 y 卻沒改」，讀者會以為 x 方向仍然對得上，只要補改類別或 y 就好。 |
| 2 | 建議 | docs/lessons/11-augmentation.md 第 109–111 行（裁切後 resize，再建立 targets） | 「若模型仍要求 64×64 輸入」暗示不 resize 也可以，卻沒提醒：本書 `build_targets` 的 `image_size` 預設是 64，而且不會檢查它是否等於實際圖寬。本書的 GridDetector 用 AdaptiveAvgPool2d，32×32 輸入也跑得動（實跑輸出 shape `[1,4,4,7]`），所以讀者可能把 32×32 的裁切圖和框 `[0,4,8,20]` 直接交給預設的 build_targets。這時不會報錯，正格卻變成 (gy=0, gx=0)、wh=(0.125, 0.25)；正確的是 (gy=1, gx=0)、wh=(0.25, 0.5)（傳 `image_size=32` 時）。 |
| 3 | 建議 | lesson_cases/11-augmentation.py 第 12–20 行 `crop`；docs/lessons/11-augmentation.md 第 13、72、82 行 | `crop` 假設裁切區完全落在原圖內，但頁面和程式都沒寫出這個前提。框是夾到參數 width、height（本例 32），圖卻是用切片取的；裁切區超出原圖時，切出來的圖會變小，也不會報錯。實跑 `crop(image,boxes,48,8,32,32)` 得到 shape `[3,32,16]` 的圖，框卻仍以 32 為邊界。left 或 top 是負數時，Python 的負索引還會從另一端切。第 13 行又說實際訓練時「裁切的位置通常也是隨機選的」，讀者若把這個函式改成隨機選位置，圖和框可能不一致，而且沒有任何錯誤訊息。 |
| 4 | 建議 | docs/lessons/11-augmentation.md 第 139 行（固定 seed 與每個 epoch 的增強）；第 13 行（增強接進訓練的位置） | 這段的原則正確，但本書自己的資料正好容易踩到反例：`miniyolo/data.py` 的 `ShapeDataset.__getitem__` 每次呼叫都用 `seed + index*1009` 重新建立 Generator。讀者若照第 13 行把隨機翻轉接進 `__getitem__`，又沿用這個 `rng` 抽亂數，同一張圖每一輪抽到的結果都一樣（實測同一個 index 連抽三次都是 0.3560）。這正是頁面說「固定 seed 不等於每個 epoch 都要用完全相同的增強」要避免的情況。 |
| 5 | 建議 | lesson_cases/11-augmentation.py 第 55 行；docs/lessons/11-augmentation.md 第 127、131 行 | 第 131 行說，第 4 行的固定文字印得出來，代表前面的斷言都已通過；第 127 行說翻轉、裁切後，labels 仍是長度 0 的 long。但程式第 55 行檢查 dtype 的對象是篩選前的 `empty_labels`：它兩行前才用 `dtype=torch.long` 建出來，所以這個檢查一定成立。篩選後的 `empty_labels[empty_keep]` 只檢查了 shape。因此印出的「labels Long[0]」裡，Long 這一項並沒有被這行斷言守住；實際上靠的是 PyTorch 用布林遮罩取值時會保留 dtype，以及第 48 行對 `strict_labels` 的同類斷言。 |
| 6 | 建議 | docs/lessons/11-augmentation.md 第 26 行（〈歷史來源與進階增強〉的隨機透視） | 「框也跟著一起變換」只說了一半。v6.0 的 `random_perspective` 把框變換、夾回之後，還會用 `box_candidates` 刪框（utils/augmentations.py L207–209、L278）：變換後的寬、高都要大於 2 畫素，面積比要大於 0.1（嚴格大於），長寬比要小於 20。而且類別和框存在同一列 `[cls, xyxy]`（L128 註解），`targets = targets[i]` 一次就把兩者一起篩掉。這正是本節「保留規則」和「labels 用同一個 keep 篩」在官方程式裡的對應。不提的話，讀者看不出本節的 0.5 只是本例的規定，官方程式的面積比門檻是 0.1。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 技術查核 | 翻錯軸的後果只說了一半 | 已修正：改寫成完整症狀：flip(0) 變藍而且沒有左右移，框住的是背景；flip(1) 紅色移到 y=36…51，和框完全不重疊；兩種錯都會被預期圖的斷言抓到 |
| 2 | 技術查核 | 不 resize 時 build_targets 預設 image_size=64 | 已修正：補一句：直接用 32×32 時要傳 image_size=32，否則會算錯卻不報錯。已確認 targets.py 的簽名 |
| 3 | 技術查核 | crop 假設裁切區在原圖內 | 已修正：第二段摘錄前寫明前提（0≤left、left+width≤W…），並說明超出時圖和框會對不上。程式不在可改範圍，所以沒有加斷言 |
| 4 | 技術查核 | ShapeDataset 每個索引用固定的 seed | 已修正：第 139 行補一句：不能沿用那個亂數產生器加增強，否則每一輪都一樣。已確認 data.py 的 seed+index*1009 |
| 5 | 技術查核 | 空圖 labels 的 dtype 斷言檢查的是篩選前的 tensor | 已修正：程式不能改，改成照實描述：斷言檢查篩選後的長度；dtype 檢查的是篩選前的 labels，布林遮罩篩選不改 dtype，0.6 那組斷言也核對了篩選後仍是 long |
| 6 | 技術查核 | random_perspective 還會用 box_candidates 刪框 | 已修正：補上三個條件（寬高>2、面積比>0.1、長寬比<20），以及類別和框一起刪。已對照 956be8e 的 augmentations.py 第 207–208、273–278 行 |
| 7 | 讀者審查 | 對標註框 clamp，和 7.5 節的規則看似衝突（must） | 已修正：第 2 步後補一段：裁切讓合法標註有一部分落到圖外，夾回後的框正好框住留下的畫素；原圖上就超界的標註仍是資料錯誤（附 7.5 節連結） |
| 8 | 讀者審查 | 「格線／格」有兩種意思 | 已修正：第 43 行註明這裡的格線每 1 畫素一條，不是圖 ② 裡每 16 畫素一條的 4×4 格線；「之間的一格／這一格」改成「那一段／這個畫素」。圖不用改 |
| 9 | 讀者審查 | 可見比例門檻和其他門檻容易混淆 | 已修正：補一句：和第 6 章的 score 門檻、IoU 門檻不同，比的是同一個框裁切後還看得到多少。術語表不在可改範圍 |
| 10 | 讀者審查 | 翻錯軸的症狀不完整 | 已修正：和技術查核第 0 條一起改寫 |
| 11 | 讀者審查 | 核對程式沒印 labels | 已修正：加一行 print(torch.tensor([0, 1])[keep].tolist())，暫存副本實測印出 [0]；另補上在自己的電腦用 python -i 執行的方法 |
| 12 | 讀者審查 | labels 的 dtype 檢查和頁面描述不符 | 已修正：和技術查核第 4 條一起照實改寫 |
| 13 | 讀者審查 | manual_seed 容易讓人以為程式有隨機增強 | 已修正：〈執行完整程式〉補一句：那兩行是各節共用的設定，本節的參數直接寫在程式裡，沒有抽亂數 |
| 14 | 讀者審查 | True/False、validation/test 用了半形斜線 | 已修正：改成全形「／」（正文兩處、摘錄註解一處；摘錄檢查仍通過） |
| 15 | 先前查核 | 摘錄的 `...` 與不相鄰的行沒說明 | 已修正：引言補上：`...` 表示省略了程式，main() 摘出的兩行之間也省略了一行斷言 |
| 16 | 先前查核 | 「切圖」用字不統一 | 已修正：改成「裁切圖片時從 x=12 開始」 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/11-csp.md`、`docs/lessons/11-fusion.md`、`docs/lessons/11-augmentation.md`、`docs/lessons/11-iou-loss.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

用腳本比對發現與處理列數並讀了修正後檢查。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/11-augmentation.md | 16 項發現中，must 的處理寫在修正後檢查裡，其餘 15 項 should 都沒有處理，指向的〈定稿修正〉不存在（fix:b4b 以「11-augmentation」為 page）。 | 已修正：產生器改以頁名、節名、萬用字元、頁面上的圖與該節 notebook 把處理對應到頁面，〈定稿修正〉列出這一頁每一項的處理。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核 2 項、讀者 8 項（含 1 項必要）與技術 6 項，都在〈定稿修正〉處理（上一輪指出缺漏，已補）；技術查核列出 YOLOv4 論文與 yolov5 固定 commit；批次檢查掛在本頁。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/11-augmentation.md 第 11、91 行 | 殘句：「所有檢查都在副本暫存副本裡跑，日誌放在同目錄的暫存副本」「3. python3 …/摘錄比對工具 <暫存副本> docs/lessons/11-augmentation.md」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 11 行的「暫存副本」已改；但點名的指令殘句仍不完整，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/11-augmentation.md 第 90 行 | 點名的「3. python3 …/摘錄比對工具 <暫存副本> docs/lessons/11-augmentation.md」只變成「3. python3 …/摘錄比對，結果為 []。」，仍是帶省略路徑、不能執行的殘缺指令。處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[evolution當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/evolution.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/evolution.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/evolution.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_a` 實讀本頁 5 個凍結單元；首次使用／前文方法範圍四題位置為 11-augmentation/00:first_use, 11-augmentation/01:first_use, 11-augmentation/02:first_use，頁末為 11-augmentation/04。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_a.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_a-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_a.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁未有需要改寫的已裁定問題，保留原教學內容；仍完整重讀與核對。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 c1 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁未有需要新增修正的來源缺口；保留原文的通過依據在本輪原答與覆核。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `c_foundation` 實讀本頁 7 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/c_foundation/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：無額外局部修正。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/11-augmentation.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
