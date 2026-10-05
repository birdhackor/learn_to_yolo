# evolution 首次閱讀紀錄

本組透過 gate 逐單位閱讀並手寫當場記錄，gate 已回報 `READING COMPLETE`，完成 148/148 單位：64 個 prerequisite（evolution-0000–0063）及 84 個 target（0064–0147，共 11 頁）。凍結 commit 為 `16f69103d1f216d1024a40610623083557324701`。沒有預讀未揭露正文、程式實作、其他人的 reviews 或外部解釋，沒有修改教材。

實際補讀的 prerequisite 是 04-coordinates、05-assignment、06-decode-nms、06-evaluation、07-data、07-targets、07-loss、07-training。我不是從首頁一路讀完本書，也沒有把未提供的其他前文視為讀過。背景是基本 Python/PyTorch、NN/CNN，加上不熟練的大學數學與非 detector 專家。

每單位原始 JSON 已由 gate 保存至 `/workspace/learn_to_yolo/reviews/clear-tutorial/16f6910/first-read/evolution.jsonl`，共 148 筆；這個 canonical 路徑由協調者提供，讀者未查看 reviews 目錄。`/tmp/clear-tutorial-evolution-note.json` 只保留最後 evolution-0147 的單位筆記，不是全部原始紀錄。下列問題皆是首次閱讀回饋，尚未經作者修正及重新閱讀，全部保持未解決。此摘要不取代當場紀錄。

## 各 target 頁面

### 09-anchors（0064–0068）

- **burden，0065**：六步 target 推導混入 raw tx/ty 編碼 round-trip，但實際訓練用 sigmoid 後的 xy 比例與 raw log-wh。原文「接著用紅框 `[8,12,24,28]`（xyxy，單位 pixel）一步一步算出它的 target，順序和完整程式相同：」使我先把六步當成同一個訓練 target 流程，較後面才分開。應在推導前列出訓練比較值與 round-trip 編碼值的兩列對照。
- **burden，0066**：摘錄與完整程式有 gt_wh/wh、wh_loss/regression、obj_loss/objectness 等別名，還有 inline target_log_wh。原文承認「有幾個變數名稱和完整程式不同，對應關係寫在註解裡。」但首次走流程仍要記四組對應，建議統一名字。
- **保留**：固定 anchor 只有尺寸的定義、positive/ignore/negative 圖、教學 ignore 與原版不同的明示、ignored 梯度為零的手算，以及這是直接學 raw 數字而非 CNN 的限制。偏移 target 在邊界，round-trip clamp 的小誤差有交代。
- **視覺**：已實看 `ec01384b3b26d809.png`，槽、anchor 與責任角色清楚。需要的是訓練/驗證 target 對照表，無須另加大型示意圖。

### 09-anchor-clustering（0069–0078）

- 沒有當場 blocking/burden/optional 問題；wh 聚類、同中心 size IoU、分群/均值、收斂、覆蓋與新分佈測試可以逐步讀通。
- **保留**：明示 weak baseline、均值更新不是 1-IoU 的通用最佳解並附反例、median 比較、只能用 train 的預處理後尺寸、size coverage 不等於 AP。
- **視覺**：已實看 `c2df34c161c99f51.png`，尺寸交集與兩群 wh 的圖、均值移動箭頭有效。

### 10-multiscale（0079–0084）

- **burden，0079**：主例前先展開七項原版差異；在尚未建立兩尺度模型之前，原版歷史、融合及訓練差異先佔掉工作記憶。應把主要實驗先走完，再提供可選版本對照。
- **burden，0080**：粗細格疊在同一圖，淡色正樣本與另一尺度負樣本斜線互相覆蓋。原文「兩種格子疊在同一張圖上，所以斜線會疊到淡色上」雖解釋原因，仍需回讀確認同一空間的兩種責任。建議並列 fine/coarse 兩張格圖。
- **burden，0081**：`fine`/`coarse` 在 forward 是特徵，在 main 又是預測。原文「同樣的名字用了兩次：`forward` 裡的 `fine`、`coarse` 是兩張特徵圖」揭露重用，但讀者仍需切換語意。改為 *_features、*_pred，並畫真實 early/deep/head 分支及梯度流。
- **burden，0083**：40 步真實預測圖只寫類別/分數，缺 TP/FP/FN、GT 與逐框 IoU 對照；「另有一個分數約 0.13 的類別 1 框」要靠文字辨认，該低分框又碰圖的下緣。應編號框並註明 IoU/配對、漏掉的小物件與完整框邊界。
- **保留**：不同尺度需合併到相同 pixel 後共同 NMS、每尺度先 NMS 不足；80 個候選的 shape；兩尺度各自的 target；小框訓練 loss 降卻 IoU=.44、AP=0 的誠實例子，以及同一訓練圖不能當泛化或品質比較。
- **視覺**：已實看 `0c72577993208aa9.png` 與 `f88fc925bdd5aebf.png`。前者的粗細疊格需要重排；後者的訓練曲線有用，預測框標記與邊界需補強。

### 11-csp（0085–0093）

- **burden，0085**：主實驗之前的 DenseNet、梯度/transition、v4/v5/C3/BN 背景太多，尤其「**CSPNet 的梯度說法**」出現在尚未走過本例兩路結構之前。先建立 Full 與 CSP 路徑，再把論文/版本細節留在後方選讀。
- **保留**：concat 與 add 的數字例子、CSP 不等於 identity 的限制、bias-inclusive 1240/368 參數、每個 branch parameter 的梯度檢查而非只看輸入梯度，以及沒有測準確度/延遲/重複梯度的界定。
- **視覺**：已實看 `1fc27b05ccf876ae.png`，Full 與 bypass/branch/concat/fuse 的並列有效。

### 11-fusion（0094–0104）

- **burden，0095**：圖把淺層放上、深層放下，於是 top-down 箭頭向上，與術語隱含的金字塔方向相反；backbone/head/PAN 等上下文又同圖出現。原文「圖中淺層畫在上方、深層畫在下方，所以 top-down 路徑」需要讀者先解方向再解功能。應明示語意方向，灰化本例未做的 PAN/backbone/head 區域。
- **optional，0098**：四數 nearest 的主線已清楚，後面「PyTorch 預設 `align_corners=False`」接上雙線性座標、True/False 值及平方梯度的大段內容，對本例必要性低。可移為進階選讀，必要時補像素中心圖。
- **保留**：reduce before up 的 MAC 比較、concat 的 B/H/W 要相同但 channel 可不同、size=shallow.shape 而非盲用 scale_factor、shape 相同仍不保證 pixel 對齊的限制、參數成本与完整原模型成本分開，以及中間 activation 不等於激勵函數。
- **視覺**：已實看 `d4d358f525b0dfb7.png`，主融合流程可讀，但語意方向與實作範圍邊界需清楚。

### 11-augmentation（0105–0110）

- **burden，0107**：「裁切後，圖和框都在 32×32 的座標裡。」之後同一單位接續 clipping、visible ratio、keep/labels、保留像素但刪GT、32→64 resize、重算責任格，概念切換很密。應分成 crop 幾何、是否保留標註、恢復輸入尺寸/重建 target 三段，補 32 crop→64 resize→新責任格圖。
- **保留**：pixel index 的 W-1-x 與框邊的 W-x 不同、clone 防止 inplace 污染、double flip 仍不足以檢查完整正確性、drop GT 不等於移除前景像素、labels 用同一 keep、seed 重現序列不等於每 epoch 同一增強，以及沒有整合訓練/AP 的明示。
- **視覺**：已實看 `aea78368752c82a8.png`，原圖/flip/crop 三站清楚，缺的是 crop 之後 resize 與責任重算的圖。

### 11-iou-loss（0111–0121）

- **burden，0115**：「### DIoU：把兩個中心拉近」容易先形成每方向都會拉近的理解，後面例子卻顯示垂直偏移增大中心距離，因 enclosing diagonal 分母增得更快，DIoU loss 仍下降。應從標題就稱中心距離比例，補垂直位移前後的 enclosing box/ρ²/c² 圖。
- **保留**：只有中心可學、寬高固定的實驗範圍，IoU 不相交時零梯度，GIoU 負值合法、fold 处梯度0與微小偏移不一樣，DIoU 比例的反直覺例子，CIoU aspect term/alpha detach，以及正式訓練不能沿推論 no_grad/NMS decode 路徑反傳。
- **視覺**：已實看 `bfbde9baa2c597b7.png`，G/P/C、空白區、ρ/c 主圖清楚；缺 DIoU 垂直移動的成對例圖。

### 12-anchor-free（0122–0127）

- 沒有當場 blocking/burden/optional issue。候選點不是 GT 中心、pixel/grid 單位、外部點負距離、softplus/Smooth L1 梯度與學習範圍都可讀通。
- **保留**：把 points 與距離單位分開、四距離 target 非對稱以抓左右錯誤、沒有分類/圖片/CNN 的明示、anchor_points 的名稱陷阱、anchor-free 不代表 NMS-free，以及大距離收斂慢的延伸題。
- **視覺**：已實看 `a2f1561370e731e4.png`，黑候選點、綠GT中心、四邊箭頭及紅外部點非常有用。可選补 Smooth L1 值/斜率比較與 stride8/16 小框含點對照，但不屬必要修正。

### 12-decoupled-head（0128–0134）

- **optional，0130**：「兩條路在共用特徵 f 會合，梯度在那裡相加。」文字與長程式可以讀通，但三次 backward 必須在記憶中對照 branch None 與 shared 梯度。建議並列 box-only、class-only、total 的回流箭頭，shared 標 g_box+g_cls，未走分支標 None。
- **保留**：共享 backbone 與後段分開的邊界、對稱 shape 的逐軸解釋、retain_graph/clone/zero_grad 因果、None 與零 tensor 區分、detach 後 loss 仍能下降但 shared 收不到分類梯度的例子、cosine 接近0不能叫明顯衝突，以及 coupled 容量比較不公平的提醒。
- **視覺**：本頁沒有 PNG，實際看的是文字 forward 流程、shape 表、公式、程式及紀錄；沒有聲稱驗過圖。

### 12-assignment（0135–0140）

- **burden，0138**：「本例另外建立 4 個 logits，每個候選一個，從 0 開始，表示『是不是前景』」使「接回 loss」從剛讀完的無 objectness 兩類 head 換到另一組單一前景 logits。文字有明示簡化，但我仍須自行搭 owner→GT類別→兩類target，再壓成前景，前後接口有負擔。應列四行候選→owner→GT類別→兩類硬target（[1,0]/[1,0]/[0,1]/[0,0]）→本例前景target 的橋接表；正式軟品質值另註在正類欄。
- **保留**：固定點資格與預測框 IoU 不能交換、score表是取GT類別而非類別機率總和、top-k兩階段索引走查、k1顯示品質排序的作用、衝突後不保證每GT有k正樣本、只改正score可能owner不變、每GT正樣本計數、assignment detach不等於模型輸出全detach、空GT與評估一對一配對的區分。
- **視覺**：已實看 `9d3e243c946e2cc9.png`，分層同x軸呈現GT/點/預測框非常有效，圖中長條高度不代表實際框高的說明必要。

### 12-dfl（0141–0147）

- **optional，0147**：「learned bin probabilities: [[0.0025, 0.7475, 0.2474, 0.0025]]」以數列呈現，本頁沒有圖；能讀通，但相鄰兩bin內插、uniform/target/學得形狀仍得腦中拼圖。可加bin0–3三組柱圖、1.25數線與.75/.25權重，並列相同期望a/b、不同DFL的例子。
- **保留**：DFL非物件類別CE、左右bin不是框左右、初始p-t梯度與有限差分、最低loss為target熵而非0、同期期望不同分佈、分佈寬度未校準不等於可信信心、0權重仍會讀超範圍索引、reg_max兩種定義、reduction先逐邊加權再mean，以及正式head reshape/正樣本與IoU loss 的接口。
- **視覺**：七個單位均無 PNG。實讀公式、程式與數列；柱圖僅是建議，未聲稱已看過。

## 主要閱讀斷點

1. **先堆版本/論文差異再建立主實驗**：10-multiscale、11-csp 的開場最明顯。術語有定義，不等於先讀大量差異容易建立正在做什麼。
2. **表示、訓練target、驗證或別名的角色切換**：09-anchors 的 raw編碼與實際訓練比較值、10-multiscale 的 features/pred 同名、12-assignment 的兩類head與單一前景logits橋接，需要先把角色對齊。
3. **圖的空間方向或重疊使責任語意難讀**：粗細網格疊圖、fusion 的 top-down 物理向上，以及DIoU標題簡化與比例反例，需圖與文字同步改。
4. **真實輸出缺少任務判讀標記**：多尺度40步框圖應把score、IoU、TP/FP/FN和GT對上，不能只靠遠處段落辨認。

沒有 blocking 級問題。當場 target 紀錄共有 11 個 burden 與 3 個 optional issue；此數目指有正式 quote 的 issues，不把所有可選圖想法都算成問題。所有 issue 均未修正、未複查，不能列為 closed。

## 實際視覺及驗證邊界

target 共實看10張 gate 提供的 preview PNG：09-anchors 1、09-anchor-clustering 1、10-multiscale 2、11-csp 1、11-fusion 1、11-augmentation 1、11-iou-loss 1、12-anchor-free 1、12-assignment 1。12-decoupled-head 與12-dfl無PNG。這些是提供的SVG靜態preview，不是Zensical整頁或手機排版測試。

沒有做技術執行、training/GPU、程式測試、AP實測、瀏覽器/Zensical/手機頁面驗證，也沒有讀連結的外部原文。文中執行紀錄是被揭露教材的一部分，不能當成讀者親自跑過。首次閱讀理解与教材所宣稱的技術結果保持分開。
