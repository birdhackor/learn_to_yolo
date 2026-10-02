# 第 08 課 own-data 審閱

範圍只含 `docs/lessons/08-own-data.md`、`lesson_cases/08-own-data.py`；本課沒有引用圖片。以只有基礎 PyTorch、未讀其他教材的讀者為準。類別索引、空圖表示、來源切分與 leakage 的說明基本可理解；Colab 佔位不列為問題。

驗證：`PYTHONPATH=. .venv-model/bin/python lesson_cases/08-own-data.py` 成功，輸出 `rejected source leakage`、每個 split 2 筆、head `(2, 4, 4, 8)`、loss `1.6998`，與教材數值一致。

## 1. 真實標註接到訓練的步驟仍缺少

- 位置：教材第 20–22、32–40 行；案例第 55–60 行。
- 問題：教材正確揭露這是人工 fixture，但「用自己的資料」的讀者仍只能看到硬編碼的 `torch.zeros(2,3,64,64)`。不知道 JSON 的 `path` 如何成為 RGB tensor、實際尺寸在哪裡驗證、非 64×64 圖片的框如何同步轉換，也沒有交代 `build_targets(anns,4,64,len(classes))` 各參數的意思。直接換 JSON 後，程式仍然訓練原本兩張人工圖片。
- 具體修法：在第 34–40 行附近增加一個最小的真實樣本銜接示例或明確的步驟：以資料根目錄解析 `path`，讀取圖片並驗證原始寬高，轉 RGB，將圖片與 pixel xyxy 框一起 resize／letterbox 到 64×64，再轉成 `[3,64,64]`、`float32`、數值範圍 `[0,1]`；同一 batch 的圖片與 annotations 保持相同順序。明寫此處 `4` 是 grid size、`64` 是轉換後的正方形圖片尺寸，並示範把該 batch 傳入目前第 60–68 行。可保留無下載的人工案例，但須讓讀者知道替換資料時的具體入口。

## 2. validator 會接受不符合契約的資料

- 位置：教材第 20–22 行；案例第 19–28 行。
- 問題：先 `reshape(-1,4)` 再檢查，會掩蓋每個框必須有四個座標的契約。例如 `boxes=[[8,12],[24,28]], labels=[2]` 會被拼成一個框並通過。`isinstance(True,int)` 也讓 JSON 的 `true` 通過 class id 檢查；空圖則完全不檢查寬高，`width=0,height=-1` 仍通過。三種情況均已呼叫本案例的 `validate` 驗證，回傳 `True`。
- 具體修法：轉 tensor 前先要求 `boxes` 為陣列，非空時每一列恰好四個數值，確認原始列數等於 labels 數；只有合法空陣列才轉成 `[0,4]`。對 class id 使用 `type(x) is int` 或等效的排除 bool 檢查；在是否空圖的分支之外檢查 width／height 是正整數。並在教材第 20 行明寫座標次序 `[x_min,y_min,x_max,y_max]`，第 22 行明寫邊界 `0≤x_min<x_max≤width`、`0≤y_min<y_max≤height`，讓契約本身可直接查核。


## 作者修訂與驗證（2026-10-02）

新增miniyolo/custom_data.py的JsonDetectionDataset，全部records先驗證原始row形狀／int非bool標籤／空圖正整數尺寸及來源split，查驗真檔尺寸，再轉RGB／CHW／letterbox且同步框。case實際保存6張PNG／JSON並讀train，拒絕錯row、NaN、bool、錯尺寸、來源洩漏；三類head實際一步loss1.4407。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/08-own-data.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。
