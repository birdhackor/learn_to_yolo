# Modern 作者交付：第 3–5 章四頁

作者：clear_first_modern。此階段承接作者分工，不是獨立首次閱讀或獨立終審。Modern 組 109 單位首次閱讀與原始當場紀錄已另存 `/tmp/clear-tutorial-modern-summary.md`、`/tmp/clear-tutorial-modern-original.jsonl`。

本次只修改：

- `docs/lessons/03-identity.md`
- `docs/lessons/03-comparison.md`
- `docs/lessons/04-coordinates.md`
- `docs/lessons/05-assignment.md`
- 專用非生成圖 `docs/assets/diagrams/05-assignment.svg`
- 新增專用非生成圖 `docs/assets/diagrams/04-coordinates-rounding.svg`

未修改 lesson source、miniyolo、notebooks、scripts、reviews、已生成 learning SVG、公開實測 JSON。未 commit、push 或使用 GPU。

## 原始 ID 與作者處理

閱讀來源是 `reviews/clear-tutorial/16f6910/first-read/foundations.jsonl` 與 `evolution.jsonl` 原始 `disclosed_source`／`reader_notes`，包含 prerequisite 記錄；沒有改寫或刪除原記錄。

|原始 ID|頁與角色|作者處理|
|---|---|---|
|foundations-0039|03-identity，target|公式前明說 x／y 暫作單一數，一維例子後再回到張量；說明卷積多路影響、雅可比矩陣與反傳相加，不能把整圖 Δy÷Δx。保留直接路徑，但明說總梯度還有主分支且可抵消；零分支全 1 才是特定機制測試。|
|foundations-0054|03-comparison，target|手算 986 參數、248840 MAC、3072 逐值加法及程式核對摘錄留主文；forward hook、modules／handle／旗標細節整段收進選讀。|
|foundations-0058|03-comparison，target|主文新增三步 stem 梯度數字表，全部取自已公布輸出：plain 0.000984／0.001001／0.001011，residual 0.146701／0.147590／0.146128。相差約 150 倍，搭配原有 loss 和 validation=0.50，讓現象具體可見。均勻平方平均、交叉項抵消、ReLU 衰減及 He 尺度估算改選讀，明示估算假設及未逐層實測。|
|evolution-0004|04-coordinates，prerequisite|開場先分「往返還原」與「對齊真實 resize 邊界」；理想／實際倍率表列出往返都能成功，但下緣分別約 45.53／46。新增局部放大圖，round() 與奇數補邊的長細節移選讀；實際倍率 metadata 契約留主文。|
|evolution-0005|04-coordinates，prerequisite|用兩列摘要區分 allclose 往返與主例紅區精確比對。tolist／item／where、float32 詳細例及插值 0.5 門檻收進選讀；保留 [N,4]、空框 [0,4]、浮點框、正規化先乘 64、減補邊再除倍率等必要契約。|
|foundations-0081|05-assignment，target|責任圖先只列正 2／負 14，移除提前出現的 ignore 0。手機實看原並排圖文字太小，改成 420 寬上下排列，保留原兩框與兩組責任格、中心、target、兩種分母。|
|evolution-0010|05-assignment，prerequisite|同上：圖不搶先介紹 ignore；主文到正負／mask 段才介紹第三種狀態。|
|evolution-0008|05-assignment，prerequisite|YOLOv1 confidence／多框／softmax 比較仍為選讀，移到本節例子與練習之後；開場指明例子讀完再看。|
|evolution-0011|05-assignment，prerequisite|主文先講負格 target=0 會被推向背景，ignore 在指定 loss 不計入；加正／負／ignore 三列對照。未來 anchor、多 slot、尺寸 IoU>0.2 的規則收進選讀。|
|evolution-0014|05-assignment，prerequisite|主要練習只呼叫實際函式 `build`（本節 target builder），保留原可執行答案。改完整 main 的同步修改從答案中拆成獨立操作選讀表，列明主例 grid=8、features、prediction shape、負格 126、兩正格與 target；保留預設 grid=4 的 odd／collision 核對與故障原因。|

另依 root 的技術審查訊息，移除 05 的兩處舊首頁責任格依賴：row／col 改為自足 `[b,1,2]` 索引例，floor 邊界直接說右／下格。

以上是「作者已處理」的對照，不是獨立讀者重新讀後的驗收宣告。

## Source 與實測依據

- `lesson_cases/03-identity.py`：`IdentityBlock.forward` 是 `x + branch(x)`，無相加後 ReLU；零分支 `y.sum().backward()` 與 `x.grad == ones`，第二部分另建隨機 block。來源 SHA256 與現有 `artifacts/checks/curriculum/03-identity.json` 的 case_sha256 一致。
- `lesson_cases/03-comparison.py`：兩邊同 stem／3 block／head，複製 state_dict，記 stem 權重 grad.norm；hook 的三類成本規則、副本暖機、每步先 backward 再 step。六個梯度值、三步 loss／accuracy、986／248840／3072 全取現有 `03-comparison.json` stdout；來源 SHA256 一致。
- `artifacts/checks/curriculum/03-comparison-learning.json` 與既有 `03-comparison-learning.svg`：保留公布 40 步 loss、8 張 train／4 張 validation、seed 7、SGD lr=0.1、CPU、本設定無 BatchNorm 的限定。未產生新的逐 block 幅度數字或圖；不能把初始化的粗略平方平均推算說成實測。
- `lesson_cases/04-coordinates.py`：`new_w/new_h=round(...)`；`left/top` 整除；bilinear、align_corners=False；框和 metadata 用 `new_w/width`、`new_h/height`；undo 先扣 padding 再除 scales；整數框先轉 float32；紅區只對主例精確比對。29、17／18、45.53／46 是原正文既有幾何例的直接換算。來源 SHA256 與 `04-coordinates.json` 一致。
- `lesson_cases/05-assignment.py`：函式名實際是 `build`；中心／grid、floor 得 col／row、整圖正規化 wh、正格 mask、同格 raise ValueError、人工 head 和負格輸出梯度斷言。來源 SHA256 與 `05-assignment.json` 一致。新操作表僅同步既有 fixture 的六處值，未改 source 或公布 S=8 訓練小數。
- 引用的論文與原版限定保留：ResNet 原始論文、Identity Mappings、YOLOv1；本次未聲稱重新重現官方結果。

## 作者層檢查

- 四頁 `data-excerpt` 與頁尾 `curriculum-evidence` 區塊對 `/tmp/clear-tutorial-modern-before/` 原始副本逐一比對：程式內容及 evidence 保持不變，只有 hook 摘錄隨選讀增加外層縮排。
- 呼叫現有 `scripts/validate_lessons.py` 的原有 `code_lines`／`excerpt_problems` 定義，對四頁限定檢查：通過；四份 lesson source 的 SHA256 也都與公開 CPU JSON 相符。
- Zensical：把 docs／overrides 複製到 `/tmp/clear-tutorial-modern-preview-project/`，用既有 `.venv-docs/bin/zensical` 嚴格建站，最終文字變更後再次通過。設定與建站輸出均在 /tmp，未寫共同 site／設定。
- 實際 HTTP 網頁與系統 Chromium：desktop 1280×900、mobile 390×844，四頁 HTTP 200、全部圖片載入、無整頁橫向溢出；新增／搬移的選讀預設收合，互不巢狀。圖與頁面預覽及 DOM 報告在 `/tmp/clear-tutorial-modern-author-previews/`。
- 實看新 04 局部圖 desktop／mobile、新 05 圖 desktop／mobile。兩圖 SVG 均有 viewBox／title／desc，DOM `getBBox` 沒有文字超出 viewBox。05 因上下排列圖較高，另以 1280×1500 重截 desktop 圖避免浮動導航遮住 col 標籤。
- S=8 操作選讀的六處修改實際套用到 `/tmp/clear-tutorial-modern-05-s8.py`，CPU 執行完整案例：正格位置、target、負格 62／126、不對稱例保留 S=4、碰撞 ValueError、人工 head 的兩步梯度／更新斷言全部通過。輸出在 `/tmp/clear-tutorial-modern-05-s8-output.txt`，沒有回寫公開 evidence。
- 指定已修改檔案 `git diff --check` 通過。沒有為正文改動重跑 40 步學習、完整課程、GPU 或改任何 source。

環境沿用既有工具；亦讀取 cloud-environment-onboarding:setup 與 onboarding reference，確認無需安裝或新增持久設定。Playwright 的預設下載目錄不存在，實際改用已安裝 `/usr/bin/chromium`，沒有下載瀏覽器。

## 還待獨立確認

- 四頁作者自查與網頁檢查不代替獨立讀者確認，尤其新 Jacobian 範圍說明是否足夠輕、主文表格是否比原長推導更好讀，要由另一位讀者判斷。
- 沒有逐 block 特徵或梯度幅度的實測；本次以既有 stem 三步表處理原視覺需求，明確區別觀察與推算。
- reviews 的 hash／新文驗收仍由 root 的獨立審查流程處理。未執行會要求所有頁最新審查 hash 的完整 `validate_lessons.py`；作者層限定 excerpt 檢查與建站結果如上，不聲稱 42 頁終審已通過。
