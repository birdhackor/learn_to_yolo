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
