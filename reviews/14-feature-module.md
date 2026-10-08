# 審查紀錄：YOLO11 特徵模組

審查範圍：`docs/lessons/14-feature-module.md`、頁面上的圖（`docs/assets/diagrams/14-split-paths.svg`），以及 `lesson_cases/14-feature-module.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題。只有一個 should：第 122 行有兩個意思不同的 64，建議標清楚，改不改都可以。我沒有修改 repo 根目錄裡的任何檔案。所有實驗都在暫存副本裡跑。

依檢查項目說明：

1. 頁面對程式的描述都和現在的程式一致。
- 原程式跑完 exit 0，印出 6 行，和頁面描述相同。第 5、6 行的標籤已經是 concat-segment。
- 練習照頁面原文做：把 `module = SplitAggregate()` 改成 `SplitAggregate(blocks=3)`。程式 exit 0，印出 `concatenated channels: 4 + 4 + 4 + 4 + 4 = 20; fuse 20 -> 8` 和 `parameters plain / split: 1168 1128`，和參考答案逐字相同。MSE 那行和 blocks=2 時不同，斷言照樣通過。
- 0.97 的驗算照頁面指示做：在 `target = x.roll(1, dims=-1)` 那行之後加 `print(target.square().mean())`。印出 `tensor(0.9730)`，其餘 6 行逐字不變。
- 進階摺疊區新寫的兩條檢查，我直接改壞真正的程式來驗：
  - 只串 a、b、b2（fuse 改成 12→8）：程式停在數值核對 `torch.equal`。
  - b1 留在 concat 裡，但乘 0 之後才進 fuse：只有直接梯度那條斷言失敗。拿掉這條斷言後程式 exit 0，第三段的 L1 是 0.0，b1 的總梯度仍不是 0。
  - 另外確認了 a、b2 的直接梯度等於總梯度，b、b1 則不相等。
  - 這些都和頁面寫的一致。突變程式放在暫存副本。

2. 盤點清單（先前審查意見）和程式修正的影響清單（受程式改動影響的段落）全部處理了。
- 第 3 行的 Colab 連結本來就是 lessons-v0.4.0。
- 兩處「另外算的，程式沒有印出」：一處改成教讀者自己印出來驗算，一處改成寫出算式。
- 「本機請先複製一份」改成全課通用的說法。我確認過 notebook 裡有「本節可修改的完整實驗」這一格，最後一格的程式也和 lesson_cases 檔案相同。
- 頁面上的「槽」全部改成「段」，和第 5 章「槽」的區分說明已刪除。第 76 行原本另有一個「段」，也改成別的說法，現在「段」在這頁只有一個意思。
- 頁尾執行紀錄區塊和 HEAD 逐字相同，等重跑紀錄時重新產生。
- 全頁沒有修訂、審查或寫作過程的敘述。

3. 數字都對。乘加次數 768、1,152、49,152、73,728 我逐一手算過。這次沒有加入任何在 Mac 上跑出來的數字。正文裡唯一依賴紀錄的數字是 1.0049，編輯已經列為待重新產生的紀錄值。

4. 程式摘錄沒問題。摘錄比對工具印出 []。頁面唯一的 Python 區塊沒有標記成逐字摘錄，正文也說明了最後一行 return 是合併改寫的。正文沒有引用程式行號。

5. 讀者可讀性：新用語都有解釋，包括「段」、乘加、兩種梯度。教學順序沒有被打亂。唯一的建議就是上面那個建議。

6. 網站建置和圖都沒問題。
- 暫存副本裡 `zensical build --clean --strict` 和 `validate_site.py` 都 exit 0。產生的 HTML 裡，摺疊區內的新清單顯示正常。
- 14-split-paths.svg 用 qlmanage 渲染正常，viewBox、title、desc 都在，和程式一致。desc 寫的「四段」正好和新用語相同。
- 和本節有關的檔案中，只有頁面本身有改動。SVG、notebook、lesson_cases、紀錄 JSON、reviews 都和 HEAD 相同。
- 另外，validate_lessons 只在審查涵蓋（review coverage）那一步失敗，原因是全站每一頁都還沒有對應目前內容的審查。這是發布前的暫時狀態，和這次編輯無關。validate_preparation 是 exit 0。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/14-feature-module.md 第 122 行（〈本次到底測什麼〉講計算量的那段）：「每一筆輸入有 8×8=64 個位置：split 每個位置做 64（投影）＋4×144（四個 3×3）＋128（融合）＝768 次」 | 同一句裡有兩個 64，意思不一樣。第一個是位置數（高×寬＝8×8）；第二個是投影層每個位置的乘加數（輸出 channel×輸入 channel＝8×8）。第 45 行剛提醒過讀者：C、H、W 都是 8，要分清楚哪個 8 是 channel。這句卻把兩種 8×8 放在一起，又沒標出前者是 H×W。讀者自己手算核對時，可能把「64（投影）」當成位置數。算式本身是對的，所以不會讓讀者做錯事，只是讀起來容易混淆。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py（Bottleneck、C3、C2f、C3k、C3k2）
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/conv.py（Conv：bias=False、BatchNorm2d、default_act=SiLU）
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/11/yolo11.yaml（backbone/head 的 C3k2、SPPF、C2PSA）
- https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/tasks.py（parse_model：m/l/x 尺度把 C3k2 的 c3k 強制設為 True）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/block.py（AAttn、ABlock、A2C2f）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/conv.py（Conv）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/cfg/models/v12/yolov12.yaml（第 6 層 A2C2f area=4、第 8 層 P5/32 area=1，head 的 A2C2f a2=False）
- https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/tasks.py（parse_model：A2C2f 的 n 插在 args[2]）
- https://arxiv.org/abs/2502.12524（Submission history：只有 v1）
- https://arxiv.org/html/2502.12524 §3.2 Area Attention（分成 l 段，(H/l,W) 或 (H,W/l)；只需一次 reshape、速度較快；預設 l=4，感受野變 1/4 但仍然夠大）、§3.4 Architectural Improvements（移除 positional encoding，改用 7×7 large separable convolution『position perceiver』）
- https://arxiv.org/abs/1706.03762 → https://arxiv.org/html/1706.03762 §3.2.1 Scaled Dot-Product Attention 正文（We suspect…）與腳註（變異數 d_k）
- https://ar5iv.labs.arxiv.org/html/1512.03385 §4.1 Deeper Bottleneck Architectures（1×1、3×3、1×1 三層）

這一頁沒有發現與來源不符的說法。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 同一句裡兩個 64（位置數、投影的乘加數）容易混淆 | 已修正：改成「H×W＝8×8＝64 個位置」與「64（投影 8→8）」。已從程式確認 fuse 層是 16→8，所以順帶把融合寫成「128（融合 16→8）」，兩項標法一致。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/13-dual-assignment.md`、`docs/lessons/13-nms-free.md`、`docs/lessons/14-feature-module.md`、`docs/lessons/15-attention-bridge.md`、`docs/lessons/15-area-attention.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

頁尾來源行為「參考來源：」，後面直接接執行紀錄區塊；yolo11.yaml、block.py 的 441632c 連結都回 200。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核的建議在〈定稿修正〉處理；〈來源對照〉列出 Ultralytics 固定 commit，沒有不符；批次檢查掛在本頁；結構檢查通過。沒有發現問題。

### 第 4 輪：上一輪的處理：通過

第 3 輪沒有發現，沒有處理說明需要核對。

## 紀錄重產後的檢查

2026-10-05，由另一位 AI 在 `/tmp/lessons-v0.4.0-reviews/review-14/` 的獨立副本重新審查 `docs/lessons/14-feature-module.md`。依 `docs/preparation/publish.md` 第 7 步，先完整閱讀正文、既有審查、case、實際執行 JSON、SVG 與 notebook，再自行執行、手算、查來源及看瀏覽器畫面；舊審查的結論沒有代替這次查核。

結論：通過。必要問題（required）0 件，建議事項（optional）0 件；沒有需要修改正文、圖或程式的發現。本次未改動根目錄、作者的執行紀錄或 coverage，未執行 git、GPU 或資料下載。

### 實際執行與紀錄核對

- 使用既有 Python 3.12.14／PyTorch 2.9.1+cpu，cwd 與 `PYTHONPATH=.` 都指向獨立副本。預設 `lesson_cases/14-feature-module.py` exit 0：輸入輸出 `(2,8,8,8)`、四份 4 channel 串接為 16、fuse 16→8、參數 plain/split＝1168/800、MSE 1.0722→1.0049、直接梯度 L1 `[0.2967, 0.3349, 0.3481, 0.2779]`，所有 assert 通過。整份 stdout 與目前 JSON、notebook 保存的輸出逐字相同；stderr 為空。
- 依自主練習只把 `module = SplitAggregate()` 改為 `SplitAggregate(blocks=3)`，exit 0：五份 4 channel、concat＝20、fuse 20→8、參數 1168/1128、MSE 1.1248→1.0141，所有 assert 通過。這與第 149–162 行的預測、參考答案及「MSE 會變」說明一致。
- 依第 115 行加入 `print(target.square().mean())`，印出 `tensor(0.9730)`；預設 30 步後的 1.0049 確實還高於這個零輸出基準，正文沒有把下降的 loss 解釋成學會通用位移或提高 AP。
- 完整讀過 notebook 四格，最後一格與 case 逐字相同；另執行抽出的最後一格，exit 0，輸出與 JSON 相同。環境格與 metadata 都固定在 `lessons-v0.4.0`，練習指向的「本節可修改的完整實驗」存在。未執行會 clone／安裝的環境格，也未測 Google Colab 託管 runtime。
- case 只 import PyTorch，沒有其他 repo 程式依賴。以 `miniyolo.provenance.repo_dependencies` 重查，依賴清單與 JSON 的 `dependencies_sha256` 一致，僅列本節 case。已閱讀補充紀錄清單，本節沒有指定需重跑的補充實驗。

### 機制、手算與讀者理解

- 沿實際 forward 追出投影 8→8、chunk 的連續前／後四個 channel、a/b/b1/b2 的 `[2,4,8,8]`、concat `[2,16,8,8]`、融合 `[2,8,8,8]`。正文摘錄明說最後一行是合併改寫；核對其計算與預設 forward 相同，並執行既有摘錄檢查函式，本頁 `excerpt_problems=[]`。
- 獨立手算含 bias 參數：plain＝2×(8×8×9+8)＝1168；split＝72+4×148+136＝800。每筆卷積乘加：split＝64×(64+4×144+128)＝49152，plain＝64×(2×576)＝73728，不計 bias／ReLU／殘差加法／concat，符合正文限定為卷積乘加的口徑。blocks＝3 為 1128；一般式 144+328×blocks，blocks＝4 為 1456。亦從各 Conv2d 的 weight／bias 元素數核對上述結果。
- 理論感受野 1×1、5×5、9×9 正確；四層 stride-1 的 3×3 增加邊長 8。把 9×9 視窗與 8×8 真實輸入範圍取交集，中央 2×2 可看全圖、角落只涵蓋 5×5。roll 的 `[1,2,3,4]→[4,1,2,3]` 實測一致；左端循環值超過本模組可達範圍、且跨位置資訊經 4 channel 的 b 分支，正文有充分解釋為何不能據此宣稱通用右移規則。
- 串接暫存＝2×16×8×8＝2048 個 float32＝8192 bytes＝8 KiB。參數較少不代表 latency 較短、沒有圖片／GT／AP、不等於完整 YOLO11 的限制均有明示。
- 補做梯度探查：a、b2 的直接／總梯度逐元素相同，b、b1 不同；四段數值都與對應 path 完全相同。保留 b1 的下游路徑，但在 fuse 前把第 8–11 channel 乘 0：該段直接梯度 L1＝0，b1 總梯度 L1 約 0.2897，正好驗證進階摺疊區的反例。只串 a、b、b2 時，第 8–11 channel 與 b1 的數值核對不成立。這些是獨立副本中的檢查，沒有修改作者程式或保存的證據。
- 完整讀過 `miniyolo/models.py`，確認 GridDetector 的 adaptive pool 後確實是 32→32 的 3×3／ReLU 模組。於暫時模型以 `SplitAggregate(channels=32, hidden=16)` 替換最後一個 backbone 模組，forward 仍輸出 `[2,4,4,7]`，支持第 139 行建議的 shape 相容性；沒有據此宣稱偵測效果相同。
- 以數學好、程式初學的高中讀者角度逐段閱讀：plain、project、fuse、hidden、bottleneck、concat segment、直接／總梯度、activation 的用法均有說明，前置章節與進階可跳過的範圍清楚。舊建議「兩個 64 容易混淆」已在第 122 行標為 H×W 位置數與投影 8→8 的乘加數，查核後維持原處理，不新增建議。

### 重新開啟官方來源

頁面兩個 GitHub 引用連結均重新請求，回 HTTP 200；並下載同一固定 commit 的官方 raw source 逐段閱讀，不用最新分支或記憶替代。

- [Ultralytics YOLO11 配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/11/yolo11.yaml)，commit `441632cdfd19e22e60a4b1b1999d46326ca51ec4`：確認 backbone／head 的 C3k2、SPPF、C2PSA、上取樣與串接，支持整版還有其他設定的說明。
- [同版 block.py](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py)：閱讀 C2f（291–322 行）、C3（325–348）、Bottleneck（460–482）、C3k2（1069–1106）、C3k（1109–1127）。確認 C3k2 繼承 C2f 的切分／保留／串接 forward；C2f 的內部 Bottleneck 明傳 e=1.0，C3k2 在 `c3k=False`、attention 預設關閉時換回預設 e=0.5 的 Bottleneck；C3k 可自訂 kernel。頁面對其他選項的省略有明示。
- [同版 conv.py](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/conv.py) 的 Conv（48–98 行）：確認 bias=False、BatchNorm2d、預設 SiLU 及卷積→BN→activation 順序。
- [同版 tasks.py](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/tasks.py) 的 parse_model（2172 行附近）：確認 m/l/x 可強制 c3k=True，沒有把配置的個別 False 誤解為所有 YOLO11 尺度都用 Bottleneck。
- [同版 YOLOv8 配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/v8/yolov8.yaml) 與 [YOLOv5 v7.0 配置](https://github.com/ultralytics/yolov5/blob/v7.0/models/yolov5s.yaml)：重新取得 HTTP 200，確認 C2f／C3 的版本歸屬。
- [He 等人 CVPR 2016 原文 PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)，§4.1「Deeper Bottleneck Architectures」及 Figure 5：原文為 1×1→3×3→1×1 三層、先縮再還原維度，與本頁提醒各種 bottleneck 的層數／channel 不同一致。
- PyTorch 2.9 的 retain_grad 官方 HTML 請求回 HTTP 403，沒有宣稱看過該 HTML。改讀 [PyTorch v2.9.1 官方 `_tensor_docs.py`](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_tensor_docs.py)，raw 請求 HTTP 200；核對 `.grad`、`is_leaf`、`retain_grad` 文件，與本頁保留中間 tensor 梯度的說明一致。

以上來源沒有查出與正文不符之處。raw source／請求結果留在獨立副本的 `artifacts/runs/review-14/sources/`。

### 圖與實際網頁

使用既有 Zensical 0.0.67 在獨立副本執行 `zensical build --clean --strict`，exit 0；`scripts/validate_site.py` 亦 exit 0，連結／錨點、Markdown 轉換與 Colab 配對等現有檢查通過。另以本節專用 HTTP 8814 與 Chromium／Playwright 看實際頁面，沒有使用根目錄 HTTP 8794。

完整讀 SVG 的 viewBox、title、desc 與全部線段／標籤，再看桌面 1440×1100 和手機 390×844 的實際像素：a/b/b1/b2 的路徑、兩個殘差公式、四段 channel 0–3／4–7／8–11／12–15、16ch→8ch 均與正文／程式相符，無裁切或箭頭錯接。此圖是結構示意，沒有資料曲線可對照。頁面 HTTP 200、SVG 正常載入、5 個摺疊區存在；已展開檢查梯度說明、參考答案與執行紀錄，表格、編號清單及 code block 顯示正常，pageerror 為空，手機 body 沒有超出 viewport。畫面留在 `artifacts/runs/review-14/` 的 `diagram-desktop.png`、`gradient-detail.png`、`exercise-answer.png`、`mobile-viewport.png`。專用伺服器於查核後停止。

### 快照與發現處理

本次覆蓋的頁面正文依 `review_coverage.digest` 排除 Colab tag 與自動 footer 後，SHA-256 是 `c70b8c7dc8e2c67eb46e77f6c92f100b8a11e2f133a4482e95587cefefb15a1d`。補回尾端兩個換行後會精確得到舊 covered 摘要 `900440826d70c71ce0146c3e13d837809eabc745b269d1e96e5a48daa2a20a21`；圖與 case 摘要原本就相同。這確認 stale 的直接原因是 generator 刪除正文尾兩個換行，沒有拿這點取代實際逐頁審查。

| 檔案 | 本次快照 SHA-256 |
| --- | --- |
| docs/lessons/14-feature-module.md（完整檔案） | d0bf6a60c3ff7f877e5806bf116cf9c89487871ecb12ad857df4d9bc7f5c6ea7 |
| docs/assets/diagrams/14-split-paths.svg | 3dcd7a8b95cd422632559f86b3b91d573a3cf6fd19a944e291c12130c44ca299 |
| lesson_cases/14-feature-module.py | 5fceb84d094061f30744f2ac138702d55f24e746fe5780e3ca653e56843fac6c |
| notebooks/14-feature-module.ipynb | 03a3c0a86271d6202c97dabcb213ca0dffdf38ad1c69c631da542cebaac4c0cd |
| artifacts/checks/curriculum/14-feature-module.json | 05bc8b0e823bac904f8d18ce9f7ab30ccd2079fb306138a6ffe3c3a3b99843c4 |
| reviews/14-feature-module.md（既有紀錄） | fd06d92977f4d717dc9b530a407f92b17f8ea54d1243f7d8ebb584b8d6e68de2 |

| 項目 | 結果與處理 |
| --- | --- |
| 本次必要問題 | 0 件；無修正需求。 |
| 本次建議事項 | 0 件；不為產生發現而添加修改。 |
| 舊審查的兩個 64 建議 | 第 122 行已分別標明 H×W 與投影 8→8，核對數字及畫面後維持已修正的結論。 |
| stale／尾端換行 | 本次已對當前文字、圖、程式、紀錄與來源真正重新查核；本報告可附加到既有審查，由發布主流程處理 coverage。 |

限制：這是 AI 獨立審查，沒有真人學生試讀；未測發布後網站、新 tag 的公開取得、Colab 託管環境、GPU、真實偵測 AP 或 latency。本次 CPU 檢查不覆寫原紀錄的日期、機器、計時或數值，也不構成上述未執行項目的驗證。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[modern當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/modern.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `evolution_b` 實讀本頁 4 個凍結單元；首次使用／前文方法範圍四題位置為 14-feature-module/00:first_use, 14-feature-module/01:first_use，頁末為 14-feature-module/03。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/evolution_b.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/evolution_b-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/evolution_b.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R032、R040；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-detector-evolution.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。本頁圖內部分小字在手機仍偏小；相鄰正文提供必要對應，保留為可選的可讀性改善，對應 TVIS04。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-08：最新版 skill 的 B–E 審閱與既有待修

本頁由 c2 依實際前文逐段保存首讀，正文封存後才補讀選讀與執行紀錄。範圍起點為93dc8d8；首讀、技術與銜接角色分開，原答未回寫。

本頁相關處置：DEC-024；包含採用、保留或後文撤回的來源與理解收益。必要與可選建議均由主 Agent 逐項裁定，詳見[決策表](clear-tutorial/remainder-2026-10-08-93dc8d8/coordinator/decisions.json)及[本輪範圍](clear-tutorial/remainder-2026-10-08-93dc8d8/README.md)。修後的技術、圖文、銜接與實頁範圍見[技術複查](clear-tutorial/remainder-2026-10-08-93dc8d8/technical/post-repair.json)、[銜接複查](clear-tutorial/remainder-2026-10-08-93dc8d8/audit/post-repair.json)和 [post-repair](clear-tutorial/remainder-2026-10-08-93dc8d8/post-repair/)；不把局部複查稱作全書新首讀，也不等同真人學生測試。


## 2026-10-08：B–E 敘事重寫與舊新對照

本頁按最新版 clear-tutorial 的學習問題、材料、做法、可觀察結果與理由重寫。開頭與 A 保留。本輪以 `7a8b9d7` 保存舊稿；新稿亦另凍結，初讀判斷不回寫。

- 獨立順讀由 `c_modern` 實讀本頁 7 個正文單位，先完成整組正文並封存，再補讀選讀／執行紀錄；[原答、摘要與實際限制](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/readers/c_modern/)保留首次需要及頁末四題、猜測與後文釐清。de 與 e_tail 的補讀按頁 batch 記錄，沒有冒稱逐單位 gate 全部提交。
- [舊新保存性對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/comparison.json)逐頁覈對原目標、例子、程式摘錄、練習、失敗與結論邊界；[既有26項對照](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/known-fix-regression.json)另記恢復與保留。
- 必要及可選項由主 Agent 依來源與理解收益裁定；本頁採用局部修正：R006, R020。原分級與具體處置見[決策表](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/decisions.json)。[獨立銜接檢查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/transition/new-initial.json)與修後addendum分開，不當成另一份未提示首讀。
- 本頁 CPU lesson case 已於本輪實際重跑並PASS，現行紀錄在 `artifacts/checks/curriculum/14-feature-module.json`；原程式與Notebook code不變。必要摘錄來源、實際輸出與保存性見[最後核對](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/coordinator/final-preservation.json)。
- 46頁桌面／手機皆有實際瀏覽器capture與DOM掃描；實看範圍以[technical/visual.json](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/technical/visual.json)、[主Agent抽查](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/visual/root-sampling.json)及後續有界delta為準。capture不代表所有圖都已人工視判，不把來源PNG當真實頁面。

方法、校準、先備路線調整、圖視判時序及AI限制見[本輪總覽](clear-tutorial/rewrite-b-e-2026-10-08-7a8b9d7/README.md)。最新頁面、圖片與實驗依賴另綁定 coverage；沒有真人學生效果驗收。
