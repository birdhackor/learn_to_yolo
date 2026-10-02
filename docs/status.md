# 這版驗證了什麼，還缺什麼

把「程式執行成功」、「模型學到合成圖形」、「真實照片上的效果」分開，讀者才能知道結果支援哪個結論。

## 各種結果各代表什麼

| 結果型別 | 本版用途 | 可以支援的結論 |
| --- | --- | --- |
| 人工已知答案 | 座標、target、NMS、AP、assignment、DFL 等 | 此實作在所列例子符合定義；不是學得的效果 |
| CPU 短步更新 | 檢查 loss 有限、gradient 連得上、參數確實更新 | 訓練程式可跑；少數步驟不代表已收斂 |
| 固定合成資料短訓練 | 小 MiniYOLO 與獨立 seed 的幾何圖形 | 此模型能學這個受控任務；不能外推到真實影像 |
| L4訓練／checkpoint／TensorRT數值 | 已完成小規模實測 | 原專案模型能在CUDA更新、持久保存恢復及跨runtime核對；不是正式速度比較 |
| 真實資料完整對照 | 未完成；Fashion僅40步管線檢查 | 真實偵測AP、泛化、各機制效果差異與正式GPU效率仍待測 |

[全套實驗與審查清單](validation/curriculum.md)列出42節逐項紀錄及補充訓練。各節 notebook 儲存其 CPU 實驗的文字輸出。這些輸出來自對應程式；演示用人工高分框會明說來源。沒有訓練的模型畫出框，只代表管線能接起來。

## 執行方式

本版作者驗證環境為 Python 3.12、PyTorch 2.9.1 CPU。Colab 有版本固定的初始化格；如果你已在同一工作階段匯入舊 PyTorch，更新後需按 notebook 指示重啟工作階段。每個實驗都可單獨執行。

本機完整步驟與依賴版本見 [repository README](https://github.com/birdhackor/learn_to_yolo#readme)。已設定環境後，在 repository 根目錄：

```bash
PYTHONPATH=. .venv-model/bin/python lesson_cases/00-warmup.py
.venv-model/bin/python -m pytest tests/test_core.py tests/test_checkpoint.py
.venv-model/bin/python scripts/check_lesson_runtime.py
```

`check_lesson_runtime.py` 執行 notebook 中的實驗格，逐節記錄 stdout、錯誤與通過狀態。它不登入 Google，也不聲稱測過 Google 分配的 GPU。核心測試則檢查空圖、錯誤標註、同格碰撞、座標取整還原、重複框配對等容易壞的地方。

本輪逐節重跑的檢查結果：**42／42 節實驗執行通過，24／24 項核心與checkpoint測試通過**。ONNX 章實際匯出模型並用 ONNX Runtime 比對 batch 1、2、3 的結果；TensorRT另完成L4的Python API建置與B1–4比對，詳見[部署章](lessons/20-deployment.md)。完整實驗輸出見 [CPU 執行紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/index.json)。

另已完成 [L4／checkpoint 實測](validation/gpu-smoke.md)：真實 GridDetector 在 CUDA 上跑兩條 40 步路徑（合計 80 次更新），從私有 HF 下載的中途 checkpoint 恢復後，模型及 optimizer 與不中斷對照的最大差異為 0；排程與 RNG 也相符。Volume 經另一 container 校驗，Modal 已停止且 tasks=0。這項驗證使用目前 `main` 的新入口，不修改已發布教材的 release tag；尚未驗證真實資料長訓練或正式 GPU 效能。

本輪以六位全新讀者subagent逐節檢查42節的理解、圖文與數字，再以另外六位技術reviewer核對原始論文、固定官方程式碼及計算。必要修改已由reviewer獨立複查；[12份審查與覆蓋紀錄](validation/curriculum.md)保留原始發現與關閉結果。

收尾另補[自有JSON／PNG訓練閉環](lessons/08-own-data.md)：三類模型固定1600個CPU步驟後，train AP50=1.0；獨立validation／新test為.388889／.666667，checkpoint重讀與原圖座標推論通過。最初160步定位不足的證據保留，沒有只靠loss下降宣布overfit。另完成12幀實際AVI讀檔、RGB／預測一致、capture釋放及[真正預測接tracking](lessons/19-tracking.md)；實體攝影機仍未測。新增內容另經四位獨立reviewer檢查，見[收尾紀錄](validation/curriculum.md)。

首頁另有[初讀與修訂版複查報告](https://github.com/birdhackor/learn_to_yolo/blob/main/reviews/homepage-fresh-reader.md)，包含圖文一致性及手機呈現的檢查。這些是編輯審查紀錄，沒有做過真人學生的學習實驗。

## 之後有 GPU 時的實驗順序

1. 先重跑少量資料，確認圖片與框對齊、參數更新、loss 有限，然後少量 overfit。這一步失敗就先修管線。
2. 固定 train／validation／test 切分、類別、圖片大小、score 截斷、NMS 與 AP 定義。只用 validation 選設定，test 留到設定確定後。
3. 跑真實資料 baseline，儲存設定、seed、checkpoint、成功與失敗圖，以及端到端時間。
4. 每次只改一項機制，在相同資料與訓練預算下比較；記錄候選數、參數量、記憶體與額外設定。不因某個 seed 的小差異就宣佈勝負。
5. 對有希望的設定增加 seed，再試影片與部署。TensorRT已有本教材小模型的L4核對；更大模型、真實場景與其他精度仍須分別驗證。

YOLO26 之後的分支，只有在找到可核實原文或公開實作後加入。已介紹版本的來源固定到論文或官方程式 commit；這份教材沒有為尚未確認的版本編造架構。
