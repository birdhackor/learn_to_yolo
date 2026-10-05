# 六張既有 SVG 的窄版修訂紀錄

作者：grid；2026-10-05。這是圖作者的來源、修改與實際渲染紀錄，不是獨立讀者的通過判定。此追加任務優先於尚未交付的第三輪非作者頁面閱讀。

## 範圍與版本

已確認聚類圖實名為 `09-anchor-clustering.svg`。本人只修改下列六張既有圖，保留原檔名及教材的圖連結；未修改正文、程式、notebook、coverage、其他 review 紀錄，未執行訓練、GPU、commit 或 push。新紀錄檔是此次明確指定的 `author-changes/legacy-diagrams.md`。

六圖均改為 `viewBox` 寬 480，另明列相同的 intrinsic width、height；正文標籤基本為 24，刻度與補充文字 22，標題 26。字型先用 `Noto Sans CJK TC`，再退回 sans-serif。圖板改為上下排列，避免在手機上把兩至四個小圖同時塞進一列。增加的高度是此次可讀性取捨。

| 圖 | 最終 viewBox | SHA256 前 12 碼 |
| --- | --- | --- |
| [09-anchor-clustering.svg](../../../../docs/assets/diagrams/09-anchor-clustering.svg) | 0 0 480 2152 | 58a4bd6ea6a5 |
| [09-anchors.svg](../../../../docs/assets/diagrams/09-anchors.svg) | 0 0 480 1180 | c429bcd5b05c |
| [11-csp.svg](../../../../docs/assets/diagrams/11-csp.svg) | 0 0 480 1750 | 64377bcc3357 |
| [11-augmentation.svg](../../../../docs/assets/diagrams/11-augmentation.svg) | 0 0 480 1970 | 46a835957fbd |
| [12-anchor-free.svg](../../../../docs/assets/diagrams/12-anchor-free.svg) | 0 0 480 1280 | 0f43b6c74f09 |
| [12-assignment.svg](../../../../docs/assets/diagrams/12-assignment.svg) | 0 0 480 1250 | 4a3817c27401 |

完整原／改 SVG 指紋、正文前後指紋見 [final-manifest.json](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/final-manifest.json)。六個原 SVG 的完整快照在同資料夾的 `original/`。讀取來源是這六頁現行 main 與各原 SVG；沒有為此任務從其他作者摘要取答案。

## 每圖的來源與保留內容

以下 LD 編號只是本次作者紀錄的項目標記，不是新增獨立閱讀 issue 或 closure。

### LD-09-clustering：尺寸 IoU 與兩群的放大圖

來源：[09-anchor-clustering.md](../../../../docs/lessons/09-anchor-clustering.md) 與原 SVG。① 拆成兩個中心對齊的例子，②是六尺寸總覽，③、④分別放大群 0、群 1。綠框／點、紫虛線、黃交集、共同中心十字、初始圈、群平均菱形及更新箭頭都有直接圖例。

保留 8×8、32×16 對弱基線 16×16 的尺寸 IoU：`64/(64+256−64)=0.25`、`256/(512+256−256)=0.5`。兩框交集仍是 min(寬)×min(高)，兩例仍按同一比例繪製。總覽保留 `[8,8]、[9,8]、[8,9]、[32,16]、[30,16]、[32,18]`。

放大圖仍分別畫寬高範圍 7.3～9.7，以及寬 29.8～32.2／高 15.8～18.2，兩張使用同一比例。初始 `(8,8)`、`(32,18)` 移到群平均 `(25/3,25/3)≈(8.33,8.33)`、`(94/3,50/3)≈(31.33,16.67)`；保留最後 anchor 的意義。只是 SVG 顯示比例改了，資料尺寸與座標沒有改。

### LD-09-anchors：同一格兩槽與 1／1／30

來源：[09-anchors.md](../../../../docs/lessons/09-anchors.md) 與原 SVG。先給完整格圖，再上下列三種責任；沒有把兩槽畫成兩個類別。

保留圖 64×64、4×4 格、每格 16、每格兩槽、32 槽，GT `[8,12,24,28]`、寬高 16×16、中心 `(16,20)`、格 `(gx=1,gy=1)`。x=16 在格線上，floor 後歸右格。畫素 `(x,y)` 映射成 `(80+5.5x,152+5.5y)`；GT 與槽 0 的實際矩形邊界完全相同，紅線較粗、紫虛線較細，顏色仍能分辨重合的兩框。槽 1 是中心相同的 8×8，即 `[12,16,20,24]`。

槽 0 尺寸 IoU=1，是 1 positive；槽 1 尺寸 IoU=0.25 > 自訂門檻 0.2，是 1 ignore；其他 15 格×2槽是 30 negative。圖中明寫 ignore 不計任何 loss，negative 只算 objectness 目標 0，positive 的 objectness 目標為 1，並學 GT 框與類別。沒有把本節自訂 ignore 說成原版 YOLOv2 規則。

### LD-11-csp：分支、concat 與參數

來源：[11-csp.md](../../../../docs/lessons/11-csp.md) 與原 SVG。Full 改成直向完整流程；CSP 另用一個上下圖板，左支前 4 旁路、右支後 4 卷積，箭頭分別進 concat，再接 fuse。箭頭沒有穿過標籤。

保留輸入／輸出 `[B,8,8,8]`、每步 H×W 不變。Full：兩個 3×3 8→8，各接 ReLU，再 1×1 8→8。CSP：兩個 3×3 4→4，各接 ReLU；旁路 C=4；沿 channel concat 的 4+4=8；再 1×1 8→8。圖中繼續區分 concat 與 add，保留無 BatchNorm、使用 ReLU、卷積帶 bias、不比較準確度的限制。

算式逐段列出：Full 的 `2×(8×8×9+8)=1168` 加 `8×8+8=72` 等於 1240；CSP 的 `2×(4×4×9+4)=296` 加 72 等於 368。這些是現行正文已列的參數數量，不是新增速度、AP 或訓練測量。

### LD-11-augmentation：①分別到②與③

來源：[11-augmentation.md](../../../../docs/lessons/11-augmentation.md) 與原 SVG。三圖上下排列，但標題明寫 `①→② flip；①→③ crop`，③也明寫「不是從②裁切」，沒有②→③流程箭頭。三張圖仍同一比例，1 畫素=4 SVG單位，32×32 crop 的圖面邊長正好是64×64原圖的一半。

保留原 GT `[8,12,24,28]`、紅畫素 x=8…23／y=12…27，裁切白虛線 x `[16,48)`、y `[8,40)`。Flip 的 `new_x1=64−24=40`、`new_x2=64−8=56`，框 `[40,12,56,28]`，紅 x=40…55；中心 `(16,20)→(48,20)`、責任格 `(1,1)→(3,1)`，淡／亮藍虛線與空心／實心點繼續區分前後。保留每格 16、48/16=3、20/16=1.25、格線上的中心取 floor。

Crop 仍從①切出32×32，留下框 `[0,4,8,20]`、紅畫素 x=0…7／y=4…19；灰色被裁掉的左半在新圖之外。保留可見 `8×16=128`、原 `16×16=256`、比例 `128/256=0.5`。沒有將裁切區或灰半移進新圖。

### LD-12-anchor-free：真實幾何與四邊距離

來源：[12-anchor-free.md](../../../../docs/lessons/12-anchor-free.md) 與原 SVG。完整64×64格圖放在第一圖板，箭頭只標 l/t/r/b，四邊的畫素／格數另用大字列在下一圖板，框外負距離另一圖板。

保留 stride=8、8×8格、畫素刻度0～64、row/column格編號0～7。座標映射 `(64+5.5x,184+5.5y)`；實際 `gt-box` 的矩形反算為 `[12,16,40,36]`，黑點反算為 `(28,28)`，紅點反算為 `(44,28)`。黑點為第 `(3,3)` 格中心，框中心 `(26,26)` 是綠十字，黑點仍在其右下方；四箭頭的左比右長、上比下長。

畫素 LTRB 仍 `[16,12,12,8]`，各除8得到 `[2,1.5,1.5,1]`。紅點為第 `(5,3)` 格中心，仍在同一列、框右側，r=40−44=−4 畫素=−0.5格。作者幾何反算與參數算式見 [author-geometry-check.json](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/author-geometry-check.json)；這份作者檢查不代替獨立技術核圖。

### LD-12-assignment：參考點與預測框

來源：[12-assignment.md](../../../../docs/lessons/12-assignment.md) 與原 SVG。GT／點、資格清單、預測框依序往下；GT與所有預測長條都使用 `x→48+8x`，保留主要刻度及間隔2的小刻度。橘色只在兩條GT長條上標共同x區間，避免直條遮到B的文字。

保留 A `[0,0,20,16]` 類別0、B `[12,0,32,16]` 類別1、重疊x12～20；點 p0 `(8,8)` 在A內、p1 `(16,8)` 在A/B內、p2 `(24,8)` 在B內、p3 `(40,8)` 在兩框之外。四個藍預測框仍是 p0 `[0,0,10,16]`、p1 `[4,0,24,16]`、p2 `[14,0,32,16]`、p3 `[40,0,50,10]`，圖內明寫藍框拿來和A/B算IoU。沒有新增品質表或 owner 結果。

保留所有框y從0開始、所有參考點y=8、GT/p0～p2框高16而p3框高10；長條高度仍不代表框高。正文「由點往上看GT、往下看預測框」方向仍成立。

## 必要正文方向詞

只有 [09-anchor-clustering.md](../../../../docs/lessons/09-anchor-clustering.md) 的既有圖說「左圖／右圖」因重新排版需要更改。本人已向 root 列出，root 已在工作中同步改為「① 圖板／②～④ 圖板」，與最終編號相符；因此這頁前後正文hash不同，其餘五頁一致。完整建議句如下，可與 root 已採用的最小替換核對：

> ①的兩例：兩框的中心疊在一起後，黃色交集的寬取兩框寬的較小者、高取兩框高的較小者，兩例的 IoU 分別是 0.25 與 0.5。②～④對應下一小節〈手做一次聚類〉：②是六個尺寸的寬高總覽，③、④分別放大群0、群1；橫軸是寬、縱軸是高，紫色箭頭是群中心從初始值移到群平均。

其他五頁的圖說沒有因此次排版失效的方向詞。12的左長右短／上長下短仍是幾何事實；11的①→②與①→③仍成立。

## 實際 browser 渲染

使用主機 `/usr/bin/chromium`、Playwright，viewport為手機390×844／桌機1280×800，device scale factor=1。將原／改SVG本身作為 `<img>` data URL 載入一個只有24px頁邊距的HTML harness；等待 image decode 和 fonts ready，再截整張 image。截圖不是由Python重新繪圖產生。旁邊的同來源 offscreen SVG 只用於 `getBBox()` 與字號量測；圖上的PNG來自實際 `<img>` 渲染。

手機可用圖寬實測342px；新版桌機圖寬480px，舊版720px。新版最小SVG字號22（手機約15.7px），正文24（手機約17.1px），標題26。六圖兩viewport共12張新版截圖，皆完成載入、無harness橫向溢出、無text超出viewBox；作者實際查看六張手機與六張桌機截圖，修正過箭頭穿字、concat框寬、row標題靠近64刻度與橘區遮到GT文字等問題。這是原SVG的作者視覺檢查，沒有執行或宣稱Zensical頁面browser驗證。

| 圖 | 原手機 | 原桌機 | 改手機 | 改桌機 |
| --- | --- | --- | --- | --- |
| 09-anchor-clustering | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/09-anchor-clustering-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/09-anchor-clustering-desktop.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/09-anchor-clustering-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/09-anchor-clustering-desktop.png) |
| 09-anchors | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/09-anchors-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/09-anchors-desktop.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/09-anchors-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/09-anchors-desktop.png) |
| 11-csp | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/11-csp-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/11-csp-desktop.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/11-csp-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/11-csp-desktop.png) |
| 11-augmentation | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/11-augmentation-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/11-augmentation-desktop.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/11-augmentation-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/11-augmentation-desktop.png) |
| 12-anchor-free | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/12-anchor-free-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/12-anchor-free-desktop.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/12-anchor-free-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/12-anchor-free-desktop.png) |
| 12-assignment | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/12-assignment-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/before/12-assignment-desktop.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/12-assignment-mobile.png) | [PNG](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/12-assignment-desktop.png) |

完整文字量測與此次測試的來源SHA256：[新版 browser-results.json](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/after/browser-results.json)。原版記錄在相同資料夾的 `before/browser-results.json`，渲染程式為 [render-raw-svg.py](../../../../artifacts/runs/clear-tutorial-16f6910/legacy-diagrams/render-raw-svg.py)。六圖XML解析及最終SVG／截圖來源SHA256一致性核對已完成。

交付後由 modern 作獨立技術核圖，reference 作第三輪最終SVG／正文銜接閱讀；全站實際browser由root另做。作者不替自己填獨立pass。

## 收尾補記：root 的 09 anchors 標題間距修正

作者交付後，modern 的獨立核圖指出 `09-anchors.svg` 圖板標題與欄索引 0／1 太接近。root 已將「64×64 圖；每格 16 畫素」這一個 `<text>` 的 baseline **由 y=128 改為 y=110，向上移18 SVG單位**。x=28、字號22、文字內容、格線與所有框／點的幾何均未變。作者用目前檔案只將此一行還原為 y=128，反算 SHA256 正好等於先前作者交付指紋，確認此收尾差異只有該 baseline。

- 先前作者交付版：`c429bcd5b05c8c9ed22986319693d060ac81a691f0e781acc386fc691cf9a5d8`。
- root 收尾後目前檔案：`93c73f27386a64d9282413f080db87a2891c249115cb165622cb78c13ba3f4e5`。

**上表、`final-manifest.json`、`after/browser-results.json` 及本報告連結的 09 anchors after PNG，保留的是先前作者交付版 y=128，沒有冒稱已渲染 root 的 y=110 版。** 其餘五張現行 SHA256 仍與該 manifest 相同。此次不重渲任何圖；收尾版在實際 Zensical 頁面上的呈現由 root 與 modern 接續核對。獨立核圖發現、root 間距修正與作者先前的原SVG檢查分別保留，作者不將自己的檢查改標獨立通過。

root 已通知第三輪59頁由另外五組完成，本人不再追加這59頁全文閱讀。
