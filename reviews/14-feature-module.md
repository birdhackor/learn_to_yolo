# 第 14 課獨立閱讀審查

審查範圍只有 `docs/lessons/14-feature-module.md` 與 `lesson_cases/14-feature-module.py`，以懂基本 PyTorch／CNN、完全不熟悉本專案的讀者為準。本頁沒有引用圖，因此圖的部分無可核對。未修改教材。

整體可讀：第 7 行清楚區分 YOLO11 官方 C3k2 與教學用 `SplitAggregate`；第 11–28 行的 channel 帳本和程式足以重建 a、b、b1、b2 的路徑；第 34–44 行也清楚限制小實驗的結論，並交代參數、activation 記憶體及延遲的取捨。沒有需要刪除的整段內容。以下有兩個關鍵補正。

1. **練習實際執行後會印出錯誤的 concat／fuse 尺寸。**
   - 位置：教材第 48 行；程式第 37、51 行。
   - 教材要求將 blocks 改為 3，答案 20→8 與 1,128 參數正確，而且程式第 22 行已會自動調整 fuse。但第 51 行固定印出 `4 + 4 + 4 + 4 = 16; fuse 16 -> 8`，讀者照做後會同時看到錯的 16→8 與正確的 1,128，無法判斷自己的修改是否成功。
   - 具體修法：教材明确寫出將第 37 行改為 `module = SplitAggregate(blocks=3)`，補充 fuse 的構造式已自動使用 `(2 + blocks) * hidden`，不用再手改。程式第 51 行改為由 `module.project.out_channels // 2`、`len(module.blocks)` 和 `module.fuse.in_channels/out_channels` 產生輸出，讓練習也會印出五份 4 channels、concat 20、fuse 20→8。

2. **梯度檢查不足以支持「各路徑都有梯度」或「保留中間成果已被驗證」。**
   - 位置：教材第 30、38 行；程式第 48–49、54 行。
   - 目前只檢查 project 的權重與第一個 bottleneck 的第一層權重。這能確認兩個參數張量參與訓練，不能證明第二個 bottleneck、融合層或四個 concat 槽各自得到梯度。尤其 b1 即使沒有直接交給 concat，只要仍作為 b2 的輸入，第一個 bottleneck 也會有梯度；因此讀者可能把「串接了中間成果」與「前段權重有梯度」誤認為同一個檢查。
   - 具體修法：最小修正是把第 38 行的「各路徑梯度存在」改為「投影與第一個 bottleneck 的權重有非零梯度」，第 30 行補一句「這項檢查不代表逐一驗證 concat 的四份輸入；b1 也會經 b2 收到梯度」。若要保留原來較強的教學承諾，則增加一次專門的 forward，對 a、b、b1、b2 呼叫 `retain_grad()`，在 backward 後逐一檢查，並另核對 concat 輸入的份數與順序；不要只擴充權重梯度檢查便宣稱已驗證保留路徑。

驗證：`PYTHONPATH=. .venv-model/bin/python lesson_cases/14-feature-module.py` 成功，輸入／輸出 `(2, 8, 8, 8)`，參數 plain／split 為 1,168／800，MSE 為 1.0722→1.0049。另在記憶體內把實例改成 `SplitAggregate(blocks=3)` 執行，參數為 1,128，MSE 為 1.1248→1.0141，但仍印出 16→8，重現第一項問題；檔案未改。


## 作者修訂紀錄（2026-10-02）

已將concat/fuse印出尺寸由實際hidden、block數與fuse通道動態生成。新增專門檢查forward，retain_grad核對concat槽的值與順序、每槽直接梯度、各中間特徵總梯度，並說明b1另經b2收到間接梯度。主例800參數與blocks=3練習1,128參數均CPU通過；練習正確印20→8，不再硬編16→8。
