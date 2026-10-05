# grid 首次閱讀總結

凍結版本：`16f69103d1f216d1024a40610623083557324701`。Gate 已回報 `READING COMPLETE`，共 131/131 個單位：原先 66 個 prerequisite、後補 7 個 07-data prerequisite，以及第 7–8 章 7 個 target 頁面的 58 個單位。每個單位均實際閱讀後各自保存當場理解、元素含義、後續動作預期、問題、圖解需求與小變化預測，未批量產生 pass。

我有基本 Python/PyTorch、NN/CNN 背景，依 gate 補讀必要前文；沒有從首頁順讀全書，也沒有讀第 3 章或其他未提供頁面。閱讀期間未開教材以外實作、完整紀錄連結、其他組 reviews 或外部解釋，未修改正文、未訓練或執行教材程式。

原始紀錄：`/workspace/learn_to_yolo/reviews/clear-tutorial/16f6910/first-read/grid.jsonl`，由 gate 逐單位保存；`/tmp/clear-tutorial-grid-note.json` 只是最後一筆暫存筆記，不是完整紀錄。以下以原始 unit ID 定位，不改寫較早疑惑。

## 每個 target 頁面

### 07-targets.md（grid-0066–0071）

- `grid-0066` burden：開場要求先讀 07-data，但協調者當時沒有把實際 07-data 放進 gate。這是協調者提供材料的缺漏，不是正文缺漏。協調者確認後在已記錄的 grid-0076 後插入 7 個凍結單位，我實讀補上；原始疑惑保留。
- `grid-0069` optional：當時讀到「物件中心不會落在格線上」時，尚未看見生成資料的根據。後補資料說明矩形完全在各自 16-pixel 格中且寬高 8–15，現在可推知中心嚴格在格內。宜在相關正文直接連結這個生成規則與邊界例子的用途；較早紀錄沒有覆寫。
- 應保留：一個物件一路從 pixel 框、中心、負責格到 target 的具體數字；xy 用格內尺、wh 用整圖尺；正格 mask、負格填值與同格拒絕各有明確用途。圖上的格線中心例子讓「不屬於生成資料常態，但用來說明規則」可區分。

### 07-loss.md（grid-0072–0082）

- 本頁未記錄新的必要卡點；沒有以此宣稱技術測試通過。
- 應保留：5×box MSE、全格 objectness BCE、正格 class CE 的不同分母，零 logits 的具體 loss 與梯度，負格 box/class 不學，以及空圖分支的 graph-preserving zero。有限差分與 shared-weight 梯度區別，能把公式連回更新用途。

### 07-training.md（grid-0083–0090）

- `grid-0088` burden：160 步補充從設定、loss 曲線、五個評估時點、初始巧合配對一路接到報告欄位、重跑與計時，讀者需暫存太多檔案角色和時間點。建議先完成「看結果」主線，將「重跑／報告欄位」分成選讀或清楚的小節。
- 應保留：tensor shape 到模型輸出的對照；三步固定小資料只驗更新不宣稱學會；監看有限梯度、權重變化、正格 IoU 與全背景捷徑；160 步實測與三步 smoke 的目的區分。

### 07-inference.md（grid-0091–0095）

- 本頁未記錄新的必要卡點。
- 應保留：未訓練模型與人工 fixture 的區別；decode 的 sigmoid、浮點端點、裁切與零面積丟棄；每張圖都回 boxes/scores/labels 字典，即使 0 個或 batch 1；score 截斷發生在同類 NMS 前，刪掉的候選不能靠 NMS 補回。

### 07-heldout.md（grid-0096–0102）

- 本頁未記錄新的必要卡點。
- 應保留：人工 AP 例、三步模型 AP=0、160 步實測的目的分開；display score 與計算 AP 候選截斷的用途區分；GT/pred/TP/FP/FN 實際數量與分母，固定 split、seed、IoU 及 all-points AP 設定。四張圖不代表整個 split 的提醒與每框判定可對照。

### 08-own-images.md（grid-0103–0113）

- 本頁未記錄新的必要卡點。
- 應保留：120×80 實際 resize 成 64×43，y 比例 43/80 而非理想統一比例；top/bottom padding 10/11；每圖保留 meta 並還原/裁切；class_names 有順序，checkpoint 權重不等於架構；存載前後比 logits 而非只比框數；推論空 annotations 代表未知 GT，不能當成真實沒有物件。

### 08-own-data.md（grid-0114–0123）

- `grid-0120` burden：正在建立「160 步總 loss 降了，評估卻落在不穩之後」的主線，曲線前插入舊資料重建、SHA-256、三種 seed、prior-diagnostic、命令及輸出路徑細節。宜正文保留比較前提，再接曲線；把核對細節集中到已有「自己跑」折疊內容。
- `grid-0121` burden：TP/FP/FN 圖解中途轉入直接呼叫與另開程序的推論核對、两份 JSON 比較；後接計時和 checkpoint，主要結果與保存機制混在長節。宜讓「四張不是全部，仍看整體 mAP」緊接圖解，CLI 相容性與保存/計時另作選讀。
- 應保留：JSON 原圖框合約、label 不接受 bool、空圖 shape、source 整組切分；來源/path 檢查與完全相同 RGB 檢查的範圍區別；從 Dataset/letterbox 到可變長 annotations list；三類 head 從 7 變 8，重新建 model/optimizer；一步更新與 1600 步 overfit/泛化證據分開；手工畫素編號右下端點 +1 的具體例子；train 同格拒絕但 validation/test 保留全部 GT。

## 主要閱讀斷點與前文問題

目標內容的主要斷點是實測補充的旁支過長，不是缺少公式定義。07-training 與 08-own-data 需要把結果主線、重現細節、輸出欄位和保存驗證分層，保留設定与限制但讓第一次讀者先完成因果鏈。

Prerequisite 中另保存：`grid-0003` 主例尚未收束就引入多項 autograd 陷阱的 burden；`grid-0009` VGG 歷史說明的 optional；`grid-0011` receptive field 邊界與中央計數需較具體圖解的 burden；`grid-0044` 圖先出 ignore=0 名稱的 optional（下一單位 grid-0045 釐清）。後補 07-data 第 4 單位另保存把「完全在格內」規則直接接到中心不落格線的 optional。這些是前文閱讀紀錄，不冒稱 target 頁面缺陷。

協調者在末尾閱讀期間提供了後續作者修改範圍，其中包含「非零 gradient 不等於每步參數實際更新」的要求。這項外部方向不列為我新增的獨立 first-read 發現。協調者已確認會在 manifest 標註 08 頁部分後段紀錄受提前作者 brief 影響，不能把整頁宣稱嚴格盲讀；原紀錄保留，修改後由未讀過 08 的另一讀者分段複查。作者修改與首次閱讀紀錄分開。

## 實際視覺範圍與限制

實際看過 15 張不同的 gate PNG preview（部分在後續頁再看），均為凍結 SVG 的靜態預覽：

- 前文：小 CNN 錯誤圖板與 40-step loss；diagnostics a/b 散點；localization IoU 和 loss；座標 letterbox；assignment 網格；NMS；evaluation PR/envelope。
- 後補前文：07-data 半開框／生成資料示意。
- 目標：07 object-journey；07 160-step loss 與四張 validation 圖；08 own-images letterbox；08 custom-learning 1600-step 曲線與四張 validation 圖。

最後一張 `48aaa9b0d3cfe4d2.png` 中實際讀到紅線160與尖峰、validation #1/#2/#3/#0、GT與預測框、分數、IoU0.585/0.406/0.421和各自TP/FP/FN。四張中紅伴隨額外黄色FP，藍/黃位置不足造成FP+FN，空圖無GT與框。提供解析度下必要字與框可辨讀。

未做 Zensical 或手機頁面排版、Colab、CLI、training、GPU、assert、checkpoint 或數值復現測試；沒有開實測完整 JSON 或原始程式。上面疑點皆未經作者修改後由獨立讀者複查，不標記 closed。
