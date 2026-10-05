# 審查紀錄：版本來源查證

審查範圍：`docs/research/version-sources.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過，沒有必要問題，也沒有需要回報的建議問題。檢查對象是 docs/research/version-sources.md。

1. 痕跡與敘事：唯一的痕跡（原第 17 行第 20 章 ONNX／ORT 與 L4 TensorRT 那一句）已整句刪除，連到 20-deployment 的連結也一併拿掉。同一行的 GFL／Attention 句保留，並補上 12.4、15.1 節的連結。全頁沒有製作或修訂經過的敘述，沒有「待補、之後會」這類計畫語氣，也沒有提到撰寫用的平台或工具。「查核日期：2026-10-02」與第 12–16 章各頁頁尾的「來源查核」日期一致，data.md 開頭也用同樣格式。

2. 事實查核（全部成立）：
- 對照 12 個 lesson 頁（12.1–16.3）、審查用的事實與寫作規範清單、reviews/curriculum-accuracy-e.md 和 -f.md（前一版的審查紀錄，本版已移除，留在 git 歷史中）逐項核對。
- 另外直接下載固定 commit 的上游檔案確認：
  - recipe L21：Objects365v1 預訓練 150 epochs，再做 COCO 微調。
  - recipe L26：推論預設走 NMS，要設 `nms=False` 才走 NMS-free head。
  - recipe L98、L172、L325：內部參數只在實驗分支上有，發布的套件會拒收。
  - predictor.py L469 是 `end2end=self.args.nms is False`；default.yaml L61 的 `nms` 預設是 None。
  - tal.py：`stride_val` 取 `stride[1]`，邊長小於 16 的框擴到 16；另有 `topk2`、`make_anchors`、`dist2bbox`。
  - loss.py：有 `DFLoss`；`BboxLoss` 在 `reg_max<=1` 時用 CIoU 加正規化 L1；`E2ELoss` 的權重 0.8→0.1，one2one 用 `tal_topk2=1`。
  - yolo26.yaml 寫 `end2end: True`、`reg_max: 1`；yolo11.yaml 用到 C3k2。
  - YOLOv12 的 AAttn：有 `self.qk`、`self.v`，位置卷積是 depthwise 5×5（`g=dim`），條件 `x.is_cuda and USE_FLASH_ATTN` 不成立時走一般矩陣乘法。
  - YOLOv10 論文原文：一對一配對用 top one selection，"same performance as Hungarian matching"，所以頁面說 YOLOv10 不用匈牙利演算法是對的。
- 頁上 11 個外部連結都回 HTTP 200。各節編號與 zensical 導覽一致。
- 「本書各節的實驗都只用 CPU、不使用預訓練權重，偵測實驗只用合成圖或小型自製資料」與審查用的事實與寫作規範清單第 8、30 行一致；status.md、pages-colab.md、architecture.md 也用同樣說法。程式碼裡搜不到下載預訓練權重的地方，Penn-Fudan 資料集也沒有任何 lesson 程式使用。L4 GPU 紀錄是另外的驗證，第 20 章頁面自己有說明。
- 「第 12–16 章各節都寫明驗證了什麼、沒有驗證什麼」：12 頁逐頁確認都有。

3. 沒有遺失重要內容：第 20 章的實測範圍與限制都還在 20-deployment.md（L27、L243–270）。原頁其他的範圍限制（不代表重現該版本、不能和 checkpoint 公平比較、各頁寫明驗證範圍）都保留了下來。

4. 可讀性：many／one、STAL 在第一次出現處有說明；術語與 glossary 一致（一致的雙重分配、資格遮罩、超參數）；中英文之間的空格和全形標點已逐一掃描，沒有問題；沒有指向已刪內容的殘留參照。

5. 渲染：在我自己的暫存副本執行 `zensical build --clean --strict` 與 `scripts/validate_site.py`，兩者都 exit 0（log 是同目錄的暫存副本與 .validate.log）。頁面渲染出 1 張表（表頭加 5 列）和 2 項條列。從修改時間看（本檔 22:43:00，前後其他檔案屬於別的部分），這個部分只改了這一個檔案。

給維護者的備註（都不是這個部分的缺陷）：
- 審查涵蓋範圍：reviews/curriculum-accuracy-e.md L146 記錄的 SHA 是 HEAD 版本，所以 2026-10-02 的審查不涵蓋現在的文字。不過本頁沒有宣稱自己審查過，docs/validation/curriculum.md L74 也已經寫明審查只涵蓋當時的文字。
- validate_lessons 目前失敗，原因是其他部分改了 lesson_cases 但 notebook 還沒重新產生，與本頁無關。
- 兩處用字不精確，但沒有讀者會因此做錯決定，所以不列為問題：「task-aligned 選擇（TAL）」和 12.3 節對 TAL 的定義略有出入；「不用 FlashAttention 時的 CPU 算法」其實在沒裝 flash-attn 的 GPU 上也會走這條路，但與 15.2 節對「CPU attention」的定義一致。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：通過

讀了紀錄：獨立查核下載固定 commit 的上游檔案與論文逐項核對，沒有發現。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/research-version-sources.md | 本頁沒有任何修正，紀錄仍寫「修正後由另一位 AI 檢查改動（通過）」並放別批頁面的內容；摘要引用「reviews/curriculum-accuracy-e.md 和 -f.md」，這兩份已不在工作樹，讀者找不到。 | 已修正：沒有修正的頁不再掛修正後的檢查；那兩份舊紀錄註明是前一版的審查紀錄，留在 git 歷史中。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：本頁沒有修正，所以不掛修正後的檢查（上一輪的建議已處理）；舊審查紀錄 curriculum-accuracy-e／-f 已註明是前一版、留在 git 歷史中；頁面沒有指令，紀錄記了逐項下載固定 commit 核對。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/research-version-sources.md 第 35、37 行 | 殘句與內部名稱：「在我自己的複本暫存副本執行…（log 是同目錄的暫存副本.build.log 與 .validate.log）」「這個頁面組只改了這一個檔案」「給派工端的備註（都不是這個頁面組的缺陷）」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 35 行開頭已清理，第 37 行改成「給維護者的備註（都不是這個部分的缺陷）」；但第 35 行括號裡的殘缺路徑還在，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/research-version-sources.md 第 35 行 | 點名的「（log 是同目錄的暫存副本.build.log 與 .validate.log）」只刪掉 .build.log，變成「（log 是同目錄的暫存副本與 .validate.log）」，仍是殘缺路徑。處理卻寫路徑已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
