# 第20課審查

範圍只含 `docs/lessons/20-deployment.md`、`lesson_cases/20-deployment.py` 與本課引用的圖；本課沒有引用圖。以陌生專案、具基本 Python／PyTorch／CNN 知識的讀者為準，未讀其他課文，未修改教材或案例。

實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/20-deployment.py` 成功。ONNX checker、ORT CPU 的 B=1／2、raw 數值與 decoded／還原框比對皆通過，兩次 raw 最大誤差均為 `4.76837158203125e-7`，80×80 輸入確實被拒絕。此次 raw median 為 PyTorch 0.149 ms、ORT 0.0353 ms；兩者端到端約 2.231／2.070 ms；ORT B=2 raw 約 0.0402 ms、49,789 images/s。課文已說明計時會變動，故與第42行歷次數字不同不構成矛盾。

整體而言，匯出僅含模型 raw、兩 backend 共用前後處理、動態 batch／固定空間、CPU FP32 一致性與偵測品質的區別，以及 TensorRT／GPU 未測範圍都交代清楚。以下三處值得修正。

1. **把 throughput 寫成了完成時間。** `docs/lessons/20-deployment.md:36` 的「B=2的throughput是兩張合計完成所需時間」混淆單位；`lesson_cases/20-deployment.py:113` 實際計算的是 `2000 / batch2_ms`，程式正確。將該句改成：「B=2 的 batch latency 是一次完成兩張所需的時間；raw throughput 是 `2 ÷ batch latency（秒）`，單位為 images/s。它不包含湊 batch 的等待，也不能當成單張請求的服務延遲。」這樣才與第42行的數字及程式一致。

2. **B=3 練習缺少真正的第三張輸入，直接改迴圈會假測 B=3。** `docs/lessons/20-deployment.md:60` 要求「B改3」，但 `lesson_cases/20-deployment.py:68–76` 只有兩張 source；把迴圈改為 `(1, 2, 3)` 時，`batch[:3]` 仍是 `[2,3,64,64]`，不會報錯，容易把 B=2 再跑一次當成 B=3。已實證此切片形狀；另外用真正 `[3,3,64,64]` 輸入 ORT 可得到 `[3,4,4,7]`，所以契約本身成立。請在練習補明確步驟：先新增第三張 RGB source 並重新建構 processed、batch、metadata，再將迴圈與 report 的 `batches_checked` 改成 `[1,2,3]`，並在每次比較前加入 `assert batch[:b].shape[0] == b`。保留 raw 與還原框的兩層比對，才算完成練習。

3. **TensorRT 的「FP32 engine」沒有區分預設 TF32。** `docs/lessons/20-deployment.md:51–58` 用未指定精度的 `trtexec` 產生 `grid-fp32.engine`，隨後要求先比較 FP32 engine。對支援 TF32 的 GPU／TensorRT 版本，預設可能允許 TF32 計算；FP32 輸入或 engine 檔名不代表所有乘法都採完整 FP32 精度。若此處要建立完整 FP32 基線，在第51行命令加入該 TensorRT 版本支援的 `--noTF32`，並於第58行說明它的目的；若保留預設效能設定，改稱「預設精度 engine」，明說可能使用 TF32，記錄實際設定並重新檢查 raw／decoded 誤差。此項是範例精度契約的靜態檢查，本次沒有 GPU，沒有執行 `trtexec`。

## 作者修訂（2026-10-02）

- 問題1：區分batch latency與raw throughput，明寫`2÷batch latency（秒）`及images/s單位，說明不含湊batch等待、不能當單張服務延遲；保留原實跑數字及量測範圍限制。
- 問題2：實際新增80×48的第三張RGB source，processed、batch與metadata均有三份。集中`batches_checked=[1,2,3]`，每次比較前assert輸入`[B,3,64,64]`，ORT與PyTorch raw都assert`[B,4,4,7]`；保留raw／還原框兩層比較。B=2吞吐計時明確使用batch[:2]，避免新增source後誤測B=3。正文與練習同步為真正B3，並要求移除第三張時應觸發shape斷言。
- 問題3：FP32 trtexec命令加`--noTF32`，解釋目的及檔名不能保證計算精度，明確記錄該命令與GPU／TensorRT仍未實測。
- 驗證：修改後案例exit 0，ONNX checker、ORT B=1／2／3、raw與三張還原框比對全通過，三個batch的raw最大誤差均為4.76837158203125e-7，80×80仍被拒絕。隔離實跑刪除第三張source的負面檢查，在B3輸入shape斷言失敗，未把兩張假稱三張。無GPU，沒有新增TensorRT驗證宣稱。
