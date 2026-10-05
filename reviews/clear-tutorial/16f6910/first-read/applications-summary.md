# applications 首次閱讀摘要

凍結 commit：16f69103d1f216d1024a40610623083557324701。僅依 gate 逐單位揭露閱讀並 record，已完成 applications-0000 至 applications-0099，gate 回覆 `READING COMPLETE`。沒有讀其他教材來源、程式實作、既有 review 或外部解釋，也沒有修改教材或執行訓練。每一單位的當場筆記由 gate 保存為 applications 原始 JSONL；`/tmp/clear-tutorial-applications-note.json` 只保留最後一個提交單位，不能當全部紀錄。

實際補讀的背景是 7.1 data（7 單位）、7.2 targets（6）、7.3 loss（11）、7.4 training（8）、7.5 inference（5）、7.6 heldout（7）、8.1 own-images（11）、8.2 own-data（10），共 65 單位。我沒有從首頁順讀全書，也未實際補讀第 4 章或第 6 章全文；座標和 AP 主線主要由已揭露背景中的具體複習取得。

目標閱讀為 17-capstone（applications-0065–0071，7 單位）、18-video（0072–0081，10）、19-tracking（0082–0089，8）、20-deployment（0090–0099，10），共 35 單位。

## 17 靜態偵測結業

沒有 blocking。已保存兩個疑點，尚未經作者修正複查：

- applications-0067，burden，原文「例如 class 0 裡 #4 那兩個高分的定位 FP（score 都約 1.00）。」同頁已存在圖片編號與預測編號兩套，此處未標示 #4 屬哪一套。讀至 0068 圖及 0071 JSON 才確認為 validation 圖片 4。建議當句明寫「validation 圖片 #4 裡的兩個 class 0…」。後文釐清保留在後單位理解，沒有改掉早期筆記。
- applications-0068，optional，原文「每個橘框左上角的深色小標籤寫『class:score』。」兩列對照圖可看偏移，但無法自己判斷 #4 兩框或 #14 改版框是否跨過 IoU 0.5；需要靠正文與後方 JSON。若加 TP/FP 和 IoU，尤其 #14 的 0.49，能降低從外觀猜「變準」的負擔。

主要主線可跟上：同初始化與同資料只改 box 權重，先固定 validation 保留規則，再測 chosen 的 test。加權梯度不等於 Adam 更新倍增，總 loss 不可直接比，coverage 與正式一對一 recall 不同，低分 FP 會影響 precision 卻未必影響 AP。應保留這些區分，以及保留失敗、test 不回頭選模型的設計。

實際看圖：19b22ec8bd1521ec.png（17-capstone.svg 的靜態預覽）。上中兩列與底部低分背景 FP 清楚；未看 Zensical 或手機排版，未跑權重 2 練習或任何計時。

## 18 影片串流

原始 issues 無 blocking/burden，沒有必須補內容才能理解的斷點。可選圖解：以兩條時間軸並排畫「全排隊」與「只取最新幀」，以及假影片在 64×64 輸入跨格的示意，能讓計算更快形成直覺；現有毫秒列表、公式和來源/輸入座標數字已足以手算。

應保留 Frame 源時間與處理耗時的區分、generator 先小例再 run_stream、暖機需真的走 NMS/畫框、各段中位數不可直接相加、throughput 與排隊 latency 的數字反例。漏檢與掉幀分清楚且沒有用 GT 冒充輸出；固定 FPS 時戳、VFR/PTS 未實作和 closing 資源釋放範圍都說明清楚。

實際看圖：a70599f511482096.png（18-video.svg）。第 0/5/11 幀與框數、源時刻清楚；第 0 幀 score 比文字稍糊但可讀 0.93。沒有看 GIF，沒有跑 OpenCV、無損 AVI、MP4 或相機，沒有量擷取或排隊延遲。

## 19 簡易追蹤

原始 issues 無 blocking/burden，沒有必要閱讀斷點。六幀位置、IoU 矩陣、先配對數再 IoU 和的規則能手算。速度乘幀間隔與 max_age 壽命分工清楚；max_age=2 只容一幀漏檢的邊界也有解釋。

應保留最後框法交叉時 IoU 選錯的具體矩陣、同一組偵測 recall 11/12 卻 ID switch 3→0 的分離、A/B 真身份只評估不餵 tracker、真模型接線先篩 class 0 且 ids 只對篩後 boxes 的說明。沒有把匀速人工案例或真模型接線冒稱完整 MOT 品質。

實際看圖：66f41cacf2de2a8d.png（19-tracking.svg）。ID 顏色及 A/B 標籤清楚；上下分開只為排字、實際同 y 的說明有助避免誤解。未執行 tracker、max_age 練習或 IDF1/HOTA。

## 20 部署

原始 issues 無 blocking/burden，沒有必要閱讀斷點。ONNX 只包模型、metadata 在外旁路的圖很好地支撐檔案介面。checker、raw 容差、原圖框/分數/類別三層比較與空對空防護可跟上。動態 batch 與固定空間、切片可能不足 B、80×80 的池化/解碼問題都有具體反例。

應保留成對交替計時、中位數差與差值中位數不同的反例、raw 加速不代表整段加速、凑 batch 吞吐提高却可能增加服務等待的例子。TensorRT 命令未執行與作者 Python L4 實測分開，允許 FP16 不代表逐層 FP16、與 CPU 不可跨硬體/权重/範圍比速度的限制清楚。

實際看圖：e0fa515030012d04.png（20-deployment.svg）。全部箭頭與三層紫色標記可見；未執行匯出、ORT、trtexec、TensorRT 或 GPU 工作流程，數值只是閱讀作者紀錄，未由我驗證。

## 背景閱讀問題（不冒稱目標章問題）

- applications-0029（7-training）與 applications-0037（7-heldout）各保存一次 AP 前提 burden：導覽列 AP50 / all-points / 包絡，但本組沒有先揭露 06-evaluation。0038 的人工排序表、PR 面積定義與 AP=0.5 算例釐清了基本算法。這是後文釐清，不是作者修正後複查；可改善閱讀導覽或把基本 AP 複習提前到第一次使用之前。
- applications-0061（8-own-data），burden：「自己跑 160 步與接續的 1600 步」容易誤讀為讀回 160 步 checkpoint 後續訓，而前 160 步逐值相同的敘述實指同初始化重跑總 1600 步。需回讀「從零」確認。建議標明重跑、非 checkpoint 續訓。
- applications-0062（8-own-data），burden：「每步都有限且不是 0，表示每一步都有更新」把非零梯度直接當每步更新證據；7-training 剛教了 backward 有梯度不代表 optimizer.step 已更新。建議此欄僅說有可用梯度，另指 step 與參數變化檢查。

背景實際視覺範圍還包含 b2bcab88a92808d4.png（紅藍 pixel 與半開框）、0420f5474adeda77.png（中心責任格）、9d80e3250df1853d.png（160 步 loss）、122dfec30b5d05e2.png（validation GT/預測與 TP/FP/FN）、8a83baa6fd58595f.png（letterbox）、48aaa9b0d3cfe4d2.png（1600 步曲線與三類/空圖失敗）。各單位提供圖時實際查看預覽；相同 SVG 同單位重複引用只查看一份。全都是靜態瀏覽器呈現，沒有驗證教材網站、手機、GIF、技術執行或作者修正後結果。所有疑點均維持未經修正複查狀態。
