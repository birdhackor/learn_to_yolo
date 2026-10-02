# 第 08 課完整資料閉環：獨立陌生讀者審讀

日期：2026-10-02。讀者假設：有基本 Python／PyTorch 與大學數學，未熟悉 YOLO。範圍：`docs/lessons/08-own-data.md` 新增閉環、`notebooks/08-own-data.ipynb` 主例與可選入口、`scripts/run_custom_data_learning.py`、160／1600 步 JSON 與 SVG。先前 42 節審查不視為這段新入口的驗證。

初讀判定：**沒有 required 阻塞；兩項 optional 閱讀改善。** 一步更新與完整訓練的用途、從 JSON 到 Dataset／head／target／checkpoint 的鏈條可理解。保留失敗結果並解釋 loss 降低不等於定位學好，讓讀者知道先查哪一階段。三類受控合成實验足以檢查本課接口及小批 overfit；不要求原版 YOLO、GPU 或真實資料長訓練。

## 初讀項目（保留原始觀察）

### CD-O1 — 頁首「一步」的範圍可再明確

分類：optional。初讀位置：`docs/lessons/08-own-data.md:7`。

原文「案例……僅對train做一步backward／step」在尚未看到後半段時，會使讀者暫時以為整頁只有一步。後半 `:75` 已明確補上完整閉環，因此不是錯誤主張或執行障礙。建議把主語寫成「下方主例」，並提示另有後半完整閉環即可；不用刪除一步介面檢查。

初讀狀態：optional，已告知作者；修後狀態只以本 reviewer 的重讀為準。

### CD-O2 — 圖板紅圖的兩個文字標籤稍重疊

分類：optional。初讀位置：`docs/assets/diagrams/08-custom-learning.svg` 的 validation #1。

用本次重跑 SVG 實際渲染後，`GT red` 與 `red 1.00` 略有重疊。SVG 兩個文字的 y 為 606.27／612.70，字級均 12，垂直只差 6.43px。框與顏色仍可辨識，其他三張圖和數字無誤，因此不影響完成條件。建議將 GT 與預測文字分開擺放；不應改框或預測分數來整理圖片。

初讀狀態：optional，已告知作者；修後狀態只以本 reviewer 的重新渲染為準。

## 獨立實際執行與證據

使用既有 `.venv-model/bin/python`，PyTorch 2.9.1+cpu、CPU 2 threads。README 已交代啟用 venv 或使用其完整直譯器路徑，因此將教材 `python` 換成上述路徑不是補造教材前置。沒有執行 GPU、seed search、Git 命令、network 或 Actions；沒有修改教材、程式、notebook、作者報告／資料／權重。輸出只在 `/tmp`，repo 只新增本 review。

實際命令（從 repo 根目錄）：

```bash
.venv-model/bin/python scripts/run_custom_data_learning.py --fixture --steps 1600 --fixture-test-seed 7001 --output /tmp/closure-reader-custom-data --report /tmp/closure-reader-custom-data/report.json --diagram /tmp/closure-reader-custom-data/learning.svg
PYTHONPATH=. .venv-model/bin/python scripts/detect_image.py --image /tmp/closure-reader-custom-data/fixture/validation/scene-001.png --checkpoint /tmp/closure-reader-custom-data/checkpoint.pt --output /tmp/closure-reader-custom-data/cli-validation-001.png
```

兩命令 exit 0。重跑沒有 `--prior-diagnostic`，實際新建資料／報告／圖／checkpoint。這直接驗證 notebook 的可選入口不依賴 ignored 舊 fixture。程式以自身路徑加入 repo root；不是靠 notebook kernel 的 `sys.path` 才能 import。

- 48 張非正方形 PNG；train／validation／test 為 24／12／12，每 split 3 張空圖，GT 為 21／9／9。類別數、有序名稱、來源及跨 split PNG／RGB 完全重複檢查與文字一致。
- 同一完整 train loss：1.5501668453216553 → 0.000354691524989903。train AP50／precision／recall 都 1；validation AP50 .3888888888888889，test AP50 .6666666666666666。
- 1600 項 loss history、初末分項 loss、gradient range、weight delta、參數數、全部 train／validation／test metrics、reload checks 逐項與 `custom-data-learning.json` 相等。所有梯度有限且非零；raw／decoded／Adam／RNG 重讀一致。
- 本次 manifest 與 PNG inventory SHA-256 均與作者 1600 步報告相同；本次生成 SVG 與教材 SVG 位元組完全相同。圖板順序確為紅／藍／黃／空背景，圖像、GT、預測標籤和分數能對上 JSON。曲線清楚標示 minibatch update 前 loss，表格清楚標示完整 train loss，兩者不混用。
- 獨立讀取舊 160 步 inventory，比對本次 inventory 的 train／validation，36 張 PNG 的 path／split／SHA 完全一致；舊 test 和本次 test 的 PNG SHA 沒有交集。另直接讀本次 24 張 train PNG，逐張取得前景 bounding box，全部與原始 JSON 框吻合。舊診斷中最低高度乘 64 為 .0548070073 pixels，支持教材「約 .05 畫素」描述。
- 本次 protocol 在訓練前寫出固定 1600 步、test seed 7001、test evaluations 0；程式只在訓練完後的一次 split loop 評估 test，報告為 1。作者歷史決策另有先於訓練的 protocol、舊診斷 hash 與 fresh seed 紀錄；獨立重跑不把作者口述當作測量結果。
- 額外 CLI 只用 validation 圖，沒有再次評估 test。CLI 與腳本 callable 的 classes／boxes／scores／labels 完全相同；PNG 是原圖 80×120，letterbox 反變換检查全通過。這與教材對兩套座標的區分一致。
- 從 notebook 擷取主例 code cell 在既有環境執行，所有 rejection、6 張 PNG、框座標、`(2,4,4,8)` 和一步 loss 1.4407 逐項吻合既存输出。

獨立觀察訓練耗時約 8.067 秒；作者約 6.807 秒是其單次觀察值，教材已限定環境和計時範圍，不要求重跑秒數相同。

Notebook 可選說明明確提供 1600／7001、沒有 `--prior-diagnostic`，並讀 `artifacts/runs/custom-data-learning/learning.svg`。腳本預設 `diagram = output / 'learning.svg'`，因此會顯示當次產物。沒有驗證遠端發布 tag 或線上 Colab 啟動（本輪禁止 network／Git）；以上驗證的是當前 checkout 的 CPU 入口。

## 作者修後獨立複查

作者回覆後，reviewer 重新讀取目前頁首、renderer、重新生成的 JSON／SVG，並獨立執行相關修正邏輯。初讀意見保留在上方，不以作者自称修好作為關閉依據。

- **CD-O1：獨立複查通過。** `docs/lessons/08-own-data.md:7` 現在寫「主例……僅對train做一步……後半的完整學習閉環另用48張PNG訓練1600步」。重讀時一開始就能辨別兩種用途。
- **CD-O2：獨立複查通過。** 目前 renderer 將 GT 與預測標籤放在各圖板左上方的不同列。reviewer 使用自己前次 `/tmp` 執行留下的 validation PNG／targets／predictions，直接呼叫新版 `render_svg`，沒有再訓練或評估 test。獨立產生的 `/tmp/closure-reader-custom-data/rechecked-learning.svg` 與作者新版教材 SVG 位元組完全相同。再以 Chromium 僅 `set_content` 渲染本地 SVG，實際看到紅／藍／黃三組標籤分別位於 y=460／475，文字清楚且不重疊；框、圖片、分數未變。

新版腳本 SHA-256 為 `e970cb91ba6e7e350bf476452727ed73a6f371083996d6f5681e508546adf98f`；作者新版報告記錄的 script SHA 與目前檔案相同，完成時間 `2026-10-02T18:15:14.565152+00:00`。reviewer 逐項比對新版報告與自己先前的獨立 1600 步結果：全部 loss history、初末分項 loss、gradient range、weight delta、train／validation／test metrics、reload checks 維持相等。這次複查只執行展示邏輯，沒有擴張為另一輪超參數或 seed 選擇。

最終判定：**本課新的完整 CPU 入口可理解、可執行，圖文數字一致；沒有 required 或 optional 未結事項。** 本 review 沒有驗證遠端發布或真實照片泛化，兩者不列為這個受控課程閉環的完成障礙。
