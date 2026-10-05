# applications 作者修改記錄

本次身份是作者，僅依 canonical 首次閱讀紀錄修課文；沒有把作者自查當成獨立技術審查或首次閱讀複驗。原始 first-read JSONL 沒有改寫。

## 修改範圍

- `docs/lessons/07-training.md`
- `docs/lessons/09-anchors.md`
- `docs/lessons/10-multiscale.md`
- `docs/assets/diagrams/10-multiscale-coarse.svg`
- `docs/assets/diagrams/10-multiscale-fine.svg`
- `docs/assets/diagrams/10-multiscale-branches.svg`
- `docs/assets/diagrams/10-multiscale-observed-predictions.svg`

沒有修改 lesson_cases、miniyolo、scripts、notebooks、reviews coverage 或保存的實測 JSON；沒有 commit、push、GPU／CPU 訓練。`10-multiscale-learning.svg` 原生成圖及三頁的 curriculum-evidence 區塊逐位元保留。

## 依原始 ID 處理

原始來源：`reviews/clear-tutorial/16f6910/first-read/grid.jsonl` 與 `evolution.jsonl`，只取本次三頁 target issues。

| 原始 ID | 原卡點 | 作者處理 |
| --- | --- | --- |
| grid-0088 | 160 步設計、曲線、評估、計時、重跑與檔案角色混在一起 | 補充拆成設計、看結果、結論範圍、重跑／查檔案。32／16／16 張切分、seed、步數、優化器、固定評估門檻留主線；操作、檔案表、COCO 差異與數字來源折疊。合併重複限制，保留實測數字。 |
| evolution-0065 | 前段像把中心 raw tx／ty 當真正 target，末段才說中心學比例 | 算步驟前明示 loss 真正使用 `[ox,oy,ln(w/aw),ln(h/ah)]`；列出本例 `[0,.25,0,0]` 與 source 名 `offsets`／`wh`。raw 中心反解、clamp、encoded→decoded 誤差移獨立折疊，明示不替換訓練 target。 |
| evolution-0066 | 範例和完整程式的四組別名增加追蹤負擔 | 換成 source 真正的 `ious`、`best`、`wh`、`offsets`、`regression`、`objectness`、`classification` 摘錄，帶中心 loss 一起看。保留 `raw` 人工張量、224 個可更新數、固定 anchor 與實際更新項目；輸出／執行細節折疊。 |
| evolution-0079 | 原版七項差異在主例前打斷多尺度動機 | 開頭只留簡化範圍與 TwoScale 不是 GridDetector 對照；完整原版差異移到 40 步主例之後的摺疊。 |
| evolution-0080 | 同圖疊 coarse／fine，淡色正格和斜線負格互相壓住 | 改兩張同座標獨立圖，綠格標各自正格，斜線標另一物件中心仍是負格，白格也負。每張單獨可讀；不修改原疊圖。 |
| evolution-0081 | fine／coarse 同時叫特徵與 head prediction，長文追線 | 新分支圖同時寫 feature NCHW 與 pred NHWC shapes；主例用 `fine_features`／`coarse_features`、`fine_pred`／`coarse_pred` 的角色命名。完整程式的原名原文放摺疊對照；執行程式未變。 |
| evolution-0083 | 實測圖只有 class／score，難連回配對；底邊不容易看全 | 依同一保存報告新增靜態預測圖，用 prediction 列號 #0–#2 標 TP／FP，小 GT 標 FN；score／IoU 與緊鄰表可直接比。完整顯示 #2 的 y=64 底邊；原生成圖留折疊。 |

applications-0029／0037 的 AP 前置漏列依協調者指示不當成課文缺陷；沒有另加整段 AP 入門教學。

## 圖的 source 依據

- coarse／fine：`lesson_cases/10-multiscale.py` 的 `small` `[5,5,13,13]`、`large` `[32,32,56,56]`、`fine_small` 與 `coarse_large`。中心 (9,9) 與 (44,44)，正格 fine(1,1)／coarse(2,2)；每圖都只是一個固定責任分配示意，不是預測。
- branches：同檔的 `TwoScale`，early 三層 stride 2，deep 一層 stride 2，兩個 1×1 head。圖中 features／pred 後綴是教學角色命名；所有層、channel、shape 和 loss 相加方式以 source 為準。圖末明示 backward 後還需要 optimizer.step()。
- observed-predictions：`artifacts/checks/curriculum/10-multiscale-learning.json` 的三列 boxes／scores／labels 原值；GT 和黑底紅／藍填色取 `scripts/run_multiscale_learning.py` 同次固定資料建構。圖依 1 pixel=6 SVG units 繪製，顯示值才四捨五入。這張是靜態實測實例，不能由 `--record` 自動重產覆蓋。

報告 SHA-256：`b825d38aa28ad98d0c84fe38f8be79431fae3f13d18f2690e5028da6f122ec09`。

同類 GT IoU 的重算值：#0=0.4410430692566806、#1=0.862989802612396、#2=0.29912168867113625；#1／#2 預測框 IoU=0.2788788844380188。這些是已存框的幾何計算，不是新的模型實測。

## 作者自查與剩疑點

- 既有文件環境的 strict build 通過。後續改用 `/tmp/clear-tutorial-applications-project/out` 的專用建置產物，最後 strict build 1.55 秒、沒有 issues；沒有再寫共同 `site`。
- 以 scoped 方式使用 `scripts/validate_lessons.py` 的 excerpt 邏輯，三頁摘錄／圖片路徑通過；確認三頁 evidence 區塊未動、原生成 SVG 未動。四個新 SVG 的 XML、viewBox、title、desc 通過。
- Playwright 在 1440 與 360 px 上實際載入三頁，所有圖載入、沒有 pageerror 或整頁橫向溢出。四張 SVG 的文字 bbox 都在 viewBox 裡；已實際看桌面／手機圖，責任格、分支 shape、預測 ID 與下沿可辨識。
- 瀏覽器紀錄：`/tmp/clear-tutorial-applications-browser.json`；截圖：`/tmp/clear-tutorial-applications-visual/`。SVG 浮動導覽鈕偶爾覆在頁面角落，是既有全站 UI；圖本身沒有文字裁切。
- 獨立技術審查與 fresh first-read 複驗尚未由本作者做，留給另一位審查者。原始生成 loss 圖在折疊中仍保留原面板，主線已改用可直接配對的新圖。
- 若未來保存報告換了一次訓練，新增靜態圖需跟新報告重畫；SVG 內 source 註解已記報告 hash。這次不宣稱小物件 AP 提升或單尺度品質優勢。

總字元及主線字元（主線只扣除 `???` 摺疊內容，仍含 Markdown／程式，僅作篇幅自查）：

| 頁 | 總字元前→後 | 主線字元前→後 |
| --- | --- | --- |
| 07-training | 14962→13455 | 12311→9245 |
| 09-anchors | 12066→12265 | 10069→8397 |
| 10-multiscale | 14816→12709 | 12840→6694 |
