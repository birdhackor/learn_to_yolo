# 第 11 課：特徵融合審閱

審閱範圍：只讀 `docs/lessons/11-fusion.md`、`lesson_cases/11-fusion.py`。本課沒有引用融合圖，`docs/assets/diagrams/` 也沒有對應的 fusion 圖；未讀其他教材，未改教材與程式。讀者假設只會基本 PyTorch／CNN，Colab 佔位不列問題。

整體判斷：沿 channel concat 的 shape 流程、nearest 複製與反向梯度、新增參數／activation 成本都能跟著算；程式與文字數值一致。v4/v5 僅作歷史定位、範例不是完整 PANet 的界線交代清楚。不過不同 stride 的空間意義、concat 與 add 的差別，以及 FPN／PAN 的路徑術語仍需要三處小補充。

1. **修正「忘記 channel reduction 是錯誤」，並具體比較 concat／add。** `docs/lessons/11-fusion.md:41` 將「忘記 deep 的 channel reduction」列為常見錯誤，但 concat 只要求非拼接維度一致，不要求兩支 channel 相同。此例若不 reduction，8ch shallow 與 16ch deep 上採樣後仍可合法 concat 成 `[B,24,8,8]`，只需將 mix 改為 `nn.Conv2d(24,8,3,padding=1)`；本例 reduction 的作用是控制通道數與成本，或滿足既定 mix 的 16ch 輸入。建議將該項改為「改動 reduction 後，未同步修改 mix 的輸入 channel」。同段再補一句：「concat 保留兩組通道，結果 16ch；add 將對應位置、對應通道相加，結果仍為 8ch，因此兩支 shape 必須一致，mix 也須改為 8ch 輸入。」目前只有「信息混合方式也會改變」，初學者無法據此理解差別。

2. **在 shape 例子中補總 stride 與原圖的關係。** `docs/lessons/11-fusion.md:9`、`:18` 直接從 8×8／4×4 shape 講到 stride／padding 管理，沒有說明 stride 是相對原輸入的採樣間距。「必須一致管理」也可能被讀成兩層 stride 或 padding 必須相同。建議加入具體例子：「同一張 64×64 輸入，總 stride 8 的 shallow 為 8×8，總 stride 16 的 deep 為 4×4；1×1 reduction 只改通道，上採樣把 deep 搬到 shallow 的網格大小。兩層 stride 本來就不同，但必須追蹤各自相對原圖的格點位置；不同裁切造成的偏移不會因 shape 相同而消失。」可在同一小圖標出原圖、兩種 stride 與上採樣箭頭，讓 alignment 不只是一句提醒。

3. **補一個 FPN／PAN 與本例的路徑對照，解開未定義術語。** `docs/lessons/11-fusion.md:3`、`:5` 的 head、neck、lateral connection、top-down／bottom-up 集中出現，只有基本 CNN 知識的讀者容易知道名稱卻不知道模組接在哪裡。建議用三句或小圖交代：「neck 在 backbone 特徵與預測 head 之間做融合；FPN 的 top-down 把深層特徵送往淺層，lateral connection 接入 backbone 同尺度特徵（原始 FPN 使用相加）；PAN 再增加淺層往深層的 bottom-up 融合。本例只有兩尺度的一次 deep→shallow concat，再送給細 head。」保留目前 v4/v5 的簡化界線即可，無需擴寫完整架構。這也避免讀者把本例 concat 誤當成原始 FPN 的固定做法。

驗證：在 repo 根目錄執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/11-fusion.py`，退出碼 0。`lesson_cases/11-fusion.py:27` 的 `[1,8,8,8]` 輸出、`:31` 的兩支非零梯度、`:33` 的 SGD 權重更新及 `:39`、`:41` 的 nearest 矩陣／來源梯度 assertions 全通過。參數為 `136 + 1160 = 1296`；concat 的 `1×16×8×8×4 = 4096` bytes 正確；自主練習 `[2,24,10,10]` 正確。文字已說明實驗未訓練 detector、沒有 AP／速度結論，沒有需要刪除的離題內容。


## 作者修訂與驗證（2026-10-02）

修正channel reduction不是concat必要条件，未reduce可合法8+16→24ch，需同步mix輸入；加concat16ch與add8ch對照檢查。補同張64輸入stride8／16格點、neck／head／FPN lateral與PAN方向；新增靜態11-fusion.svg。case有限梯度／更新、nearest數值與來源梯度4均通過。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/11-fusion.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。
