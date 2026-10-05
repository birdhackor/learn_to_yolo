# lessons-v0.4.1 發布驗證工具審查

原發布 commit 為 `8292a666b72934f86228e5d408ffb922c1855599`（`lessons-v0.4.0`）。
[首次發布後驗證](https://github.com/birdhackor/learn_to_yolo/actions/runs/37281420623)的公開網站 job 通過（59 頁、40 張圖、42 個 notebook 配對），但 bootstrap job 在讀取 notebook 保存輸出時拋出 TypeError，尚未建立第一個測試環境。該次沒有 bootstrap 結果 JSON，兩份發布紀錄均未保存或提交。

依外部發布清單附錄 C 保留既有 tag；新版使用 `lessons-v0.4.1`。修正 verifier 的 stdout 讀取並新增 4 個回歸測試，重建 42 本 notebook 與 Colab 連結。導言的舊頁名也由產生器同步為「驗證範圍」。README 與三頁明示的版本引用更新；三頁另有獨立審查。實驗程式、51 份 CPU 紀錄、2 份既有 GPU 紀錄與所有圖均未改動；實際執行 `record_evidence.py --run` 回報沒有過期紀錄，沒有重新訓練。

修正版對舊公開 tag 的本機完整探針在 README 第二次 clone 遇到 GitHub 503 而中止，沒有最終 JSON；不將它記成 五個全新環境 或完整 README 通過。錯誤 log SHA-256：`b88ebd3d01b13d98122667397100c3b50a2639530aa6ce0bb2334e694fcdf616`。新版完整實跑結果必須另由發布後的 Actions 保存。

## 發布驗證腳本的獨立初審

2026-10-05，檢查 `scripts/verify_release.py` 修正前的完整腳本、兩本實際 notebook、五個環境分支、README 所有 bash 區塊、工作流程及 Jupyter 官方 notebook 格式。使用在 stdout helper 加入前保存的獨立 rsync 副本 `/tmp/lessons-v0.4.1-verifier-initial/`；原腳本 SHA-256 為 `bf1d351d86aeaa669a130d3e8b1cc00f72eed590a861ccd1ad6eb6f5175ed045`。本次不把編輯者後來加入的 helper 當成原版，也不是修正後的獨立複查。

判定：確認 1 項必要程式問題。其餘探針沒有發現第二項已證實會阻擋後續實跑的必要程式問題；完整的新環境安裝與全部 README 命令仍須由實際發布驗證完成，不能以本次受控探針替代。編輯者另外回報：修正後的本機完整探針在 README 階段的第二次 public clone 遇到 GitHub HTTP 503，中止且沒有結果 JSON；本報告不據此宣告五個真實環境分支或全套 README 通過。

| # | 嚴重度／位置 | 原發現與實證 | 修正建議／處理 |
| --- | --- | --- | --- |
| 1 | 必要；原 `scripts/verify_release.py` 第 256–257 行 | `saved = ''.join(output.get('text', '') for output in ...)` 假設每個 stdout stream 的 `text` 都是字串。實際 `00-warmup.ipynb` 與 `20-deployment.ipynb` 存的是字串列表，兩本都直接重現 `TypeError: sequence item 0: expected str instance, list found`。原 `check_bootstrap()` 在第一個 `new_environment()` 呼叫前就失敗，完全沒有建立／執行五個驗證環境；因此不能把該次 workflow 當成 notebook 執行失敗的證據。原 `main()` 同時因未收到結果而沒有寫出 `bootstrap.json`，這也實際重現。 | 對每個 stdout stream 先做 `''.join(output.get('text', ''))`，再合併全部 stdout stream；字串逐字 join 不改內容、列表逐項 join 可還原換行。保持 `output_type == 'stream'` 與 `name == 'stdout'` 的篩選，不把 stderr 或 rich display 當成實驗輸出。用 str／list、混合多筆 stream、stderr／display 排除、空輸出與 42 本真實 JSON 對保存紀錄核對。編輯者後續的實作與新 tag，應另由不同 AI 複查。 |

### 官方格式與根因

官方 HTML `https://nbformat.readthedocs.io/en/stable/format_description.html` 本次回 HTTP 403；沒有將它記為已讀。改讀 Jupyter nbformat **v5.10.4 固定 tag** 的兩份官方原檔，皆 HTTP 200：

- [docs/format_description.rst](https://github.com/jupyter/nbformat/blob/v5.10.4/docs/format_description.rst)，第 53–59 行明說直接讀 notebook JSON 時，多行欄位可為 string 或 list of strings，列表應以 `''` 連接。本次已讀這段原文及 stream output 定義。
- [nbformat.v4.5.schema.json](https://github.com/jupyter/nbformat/blob/v5.10.4/nbformat/v4/nbformat.v4.5.schema.json)，stream.text 參照 `misc/multiline_string`；該定義的 `oneOf` 是 string 或 items 為 string 的 array。兩本實際 notebook 都是 nbformat 4.5，符合這個範圍。

`json.loads()` 保留磁碟上 list 的型別，並不經過 nbformat API 的多行文字正規化。外層 `str.join()` 的元素因此是 list，才會拋錯；這不是 notebook 格式壞掉，也不需要改動 42 本 notebook 的保存方式。內層 join 是對直接讀 JSON 的兩種合法表示進行正規化。

### 實際探針與結果

1. **完整原分支重現。** 在副本載入未改過的腳本，只把 public clone 換成副本中保存的 notebook／README 複製操作，把 `new_environment()` 換成呼叫紀錄器；原 `check_bootstrap()` 在已確認的第 256 行拋出同一 TypeError，環境建立呼叫數為 0。另實際呼叫原 `main()`，捕捉相同例外後確認指定輸出 JSON 不存在。這是原問題的影響，沒有另列成第二個根因。
2. **獨立驗算文字正規化。** 讀取 section-map 的 42 本實際 notebook JSON，對 stdout stream 做內層 join 後，每一本的最後程式格輸出都與 `artifacts/checks/curriculum/<節>.json` 的 stdout 逐字相同。再把各 stream.text 轉成單一字串，得到同樣結果。混合 stdout list／stdout string／stderr／display_data 的小例子得到精確的 `a\nb\n`，沒有多插換行或納入其他輸出。
3. **五個分支的排程與結果判定。** 僅在探針的記憶體 AST 中補上內層 join，沒有改副本或教材的原腳本。用明列的 clone／pip／torch_state／session fixtures 走完整 `check_bootstrap()`，實際順序為 no-torch、other-torch、same-torch、imported-torch 首次、imported-torch 重啟、deployment。五項都走到結果判定，成功 fixtures 得到 passed；other 版本的前置安裝、same 版本的 distribution RECORD 狀態保留、已 import 後必須看見重啟訊息，以及最後輸出從 cell 分隔線切出後比對的條件均核對。此項驗證排程和邏輯，沒有聲稱 fixtures 是實際安裝。
4. **實際 SESSION 子程序。** 使用現有 Python 3.12.14／PyTorch 2.9.1+cpu 執行原 `SESSION`，環境格與最後一格在同一個 namespace，原 `<<next cell>>` 分隔線與 PYTHONPATH 移除方式照原腳本。只替換 clone／pip 呼叫及安裝版本 metadata，避免安裝或變動套件；第 0 章在 no／other／same 三個受控分支皆 exit 0，切出的實驗輸出逐字等於 notebook。已 import 分支 exit 1，stderr 明確含「重新啟動工作階段」，且未執行最後一格；另開程序並用 pinned metadata 重跑兩格，exit 0、輸出相同。第 20 章實際做了一步更新、ONNX export／checker、ORT CPU 及 B=1、2、3 的 raw／框／score／label 對照，exit 0，spatial80 拒絕檢查通過。第 20 章包含計時，其 stdout 與保存紀錄不同；原腳本明確允許 deployment 不逐字匹配，但仍要求程式成功及 pinned 版本，這不是新失敗。
5. **既有環境格測試。** 在副本執行 `tests/test_notebook_bootstrap.py`，**6 passed**。它檢查無 torch、同版本含 CUDA suffix、不同版本、已 import 後要求重啟、錯誤 tag 與 ONNX 套件分支。這些是明列的 stub 測試，不是套件安裝實測。
6. **README 全部 bash 區塊。** 原腳本的 regex 讀出 4 個區塊，去掉空行／註解及唯一不返回的 `zensical serve` 後，共 16 條命令；順序與 README 相同。現有命令全是完整單行，沒有 `cd`／`source`、跨行續行、shell 變數或依賴前一個 shell 的狀態，因此逐條 `run(command, readme_folder, environment)` 沒有已證實的狀態遺失問題。完整受控流程實際呼叫了這 16 條命令，順序相符、PYTHONPATH 已移除。

   | 順序 | 目前 README 命令範圍 | 核對結果 |
   | --- | --- | --- |
   | 1–3 | 建 model venv、CPU torch、requirements-model | torch／numerical／pytest／ONNX 固定版本和第 0、20 章需要相符。 |
   | 4–7 | 暖身、tests、42 節 runtime、160 步合成偵測 | 原命令使用完整 venv 路徑；暖身明設 PYTHONPATH；runtime 腳本自行將 repo 根目錄加入 import path。沒有把全部 42 節或 160 步訓練聲稱成本次重跑。 |
   | 8–9 | Fashion-MNIST fetch 與 2 步分類 | loader 和 download script 的預設資料目錄相同，fetch 在分類之前；本次沒有下載資料或重跑這兩條。 |
   | 10–16 | docs venv、requirements-docs、三項 validator、strict build、site validator | 三項 validator 用現有 Python 在副本實際執行，全部 exit 0，42 節及 11 項補充紀錄符合現程式。沒有把此項稱為全套 README 新環境驗證。 |

### 完整發布探針的外部失敗

本次收尾前，編輯者回報修正後本機完整探針已結束：進入 README 的第二次 public clone 時遇到 GitHub HTTP 503，未產生 `/tmp/lessons-v0.4.1-verifier-bootstrap-probe.json`；其 read-only Git 與 GitHub API 探針亦為 503。這是編輯者實測回報，本 AI 沒有再做網路請求或重新安裝；也沒有讀到可核對每項 passed 的完整結果 JSON，因此不把任何真實 fresh-venv 分支寫為已通過。該暫時的對外連線失敗不是本次新證實的教材／配對程式缺陷；後續應在連線恢復後完成整套驗證。

上述 TypeError 探針與編輯者的 clone 失敗都顯示：正常 `run()` 非零結束會被放進結果，但 `check_bootstrap()` 外逸的例外不會由原 `main()` 寫成失敗 JSON。本次將此列為實際限制及診斷保存建議，不把它宣稱為另一個會使安裝正常時必然失敗的根因。

### 快照與限制

| 檔案／證據 | SHA-256 |
| --- | --- |
| 原 `scripts/verify_release.py` | `bf1d351d86aeaa669a130d3e8b1cc00f72eed590a861ccd1ad6eb6f5175ed045` |
| 原 README | `b9c96076cb65d27ee3d3afd83938e4d9021c42cd3e43d70fdc550df85c1bb19f` |
| 原 `00-warmup.ipynb` | `be9ba9daa6700f0db56775fc46b2e52700553867c876f0ee84c2d853e5f52c3a` |
| 原 `20-deployment.ipynb` | `2898170995e7f67b17d34e29a90bd97cef611554a9cdbb61bd4bf5d448f3fec1` |
| nbformat v5.10.4 格式原文 | `2f6f094c7d33de17da0aafb4693d75f063265613eb3bfb602ebe8623197c3d70` |
| nbformat v5.10.4 的 v4.5 schema | `523e3578ddbfcad52933d2423dc5951114ef73f48df180b6a601498ba5ca071a` |

實際探針、五個子程序的 stdout／stderr、重啟後結果、排程檢查及官方來源快照保存在副本 `artifacts/runs/verifier-initial/`。本次沒有安裝或更動套件，沒有做新的 public clone、建立／覆寫 tag、推送或 GPU 執行；對發布工作流程的結果只採用工作指定的失敗背景，沒有假裝本次重新下載並驗證其 run log。這份報告支持原錯誤的修正，不支持「五個 fresh venv 與全部 README 已在此初審真實跑完」的宣告。

# 發布驗證 stdout 修正：不同 AI 的獨立複查

2026-10-05，在 `/tmp/lessons-v0.4.1-verifier-fix/` 從 root 建立獨立 rsync 副本，讀 `AGENTS.md`、完整 `scripts/verify_release.py`、新增回歸測試、既有 bootstrap 測試、workflow、README，以及 section-map 選出的 42 本課程 notebook 原始 JSON。沒有修改 root、coverage、git、遠端、GPU 或共享模型套件。

結論：**stdout 解析缺陷已修好，必要問題 0 項、建議問題 0 項。** 原版失敗與修正版通過都實際重現，新測試確實涵蓋原問題；session 與五種流程的原有判定沒有被改動。完整 fresh 安裝／README 的本機 probe 受 GitHub 503 中止、沒有最終 JSON，不能報成全流程通過；`lessons-v0.4.1` 尚未公開，仍須新 tag 公開後由 GitHub-hosted Actions 做完整 fresh 驗證。

## 原問題、合法格式與修正

以 v0.4.0 審查快照中的完整原版 `check_bootstrap()` 和真實 `00-warmup.ipynb` 執行，僅把 clone 替換為本地同版檔案複製，且在 `new_environment`／`session` 設置「一被呼叫就失敗」的觀察器。原程式確實拋出：

```text
TypeError: sequence item 0: expected str instance, list found
```

只有本地 fixture clone 被呼叫，沒有建立 venv、安裝套件或啟動 session。這獨立確認解析處先失敗，不能把先前 Actions bootstrap 中止解讀成 notebook 實驗格失敗。

修正把每個 stdout stream 的 `text` 先內層 `''.join(...)`，再按輸出順序外層 join。`text` 是 str 時重組字符仍得到原字串；是 list[str] 時串回原多行字串。既有篩選保持只收 `output_type=='stream'` 且 `name=='stdout'`：stderr、display_data、execute_result、error 不混入比較；空 outputs 得空字串，沒有自行插入或刪除換行。

原始官方格式已獨立重讀：

- [nbformat v5.10.4 的 v4.5 schema](https://raw.githubusercontent.com/jupyter/nbformat/v5.10.4/nbformat/v4/nbformat.v4.5.schema.json)，`definitions.misc.multiline_string` 明列 string 或 items 為 string 的 array，stream 的 text 指向該定義。
- [同 tag 的官方 format_description.rst](https://raw.githubusercontent.com/jupyter/nbformat/v5.10.4/docs/format_description.rst)，第 53–59 行要求直接讀磁碟 notebook 時允許 string／list of strings，list 用 `''` join；stream 區分 stdout／stderr。

此次重新抓固定 raw URL 與 GitHub contents API 均遇 HTTP 503（最初漏 `v` 前綴的 URL 則為 404，已改正）；因此實際閱讀另一位初審者先前以 HTTP 200 保存的上述固定版官方原檔，沒有沿用其結論。schema 16,104 bytes，SHA-256 `523e3578ddbfcad52933d2423dc5951114ef73f48df180b6a601498ba5ca071a`；官方文字 15,492 bytes，SHA-256 `2f6f094c7d33de17da0aafb4693d75f063265613eb3bfb602ebe8623197c3d70`。另用實際 schema 對 42 本原始 JSON 做 Draft4 validation，全通過；沒有先經 nbformat 正規化而掩蓋 list 格式。

## 實跑結果與回歸涵蓋

模型／測試命令使用指定 Python 3.12.14、PyTorch 2.9.1+cpu 環境，以副本為 cwd 和 `PYTHONPATH=.`。

| 檢查 | 結果 |
| --- | --- |
| 完整測試 `pytest tests/ -q` | **72 passed in 4.16s**，包含新增四個測試結果。 |
| 新增測試數量 | str／list 參數化測試兩個結果，空輸出一個，全部課程原始 JSON 一個，共四個；不把一個 parametrized function 誤算為只有一次檢查。 |
| 回歸效力反證 | 把新測試使用的 helper 暫時替成原版 outer-join 邏輯，str 與空 outputs 通過，但 list 參數與 42 本實際 JSON 核對各自拋 TypeError；兩項確實能抓到原缺陷。只在 probe 的函式 globals 暫時替換，未修改 repo 程式。 |
| 真實 notebook 範圍 | 以 `section-map.json` 的 `kind=='lesson'` 選出 42 本；磁碟實際有 43 本，多的一本是 `00_environment_check.ipynb`。沒有將它算進 42 本課程。 |
| 42 本 source／輸出／版本 | 每本最後一格 source 逐字等於同名 lesson case；helper 提取的 stdout 逐字等於各自 curriculum record。42 本的 metadata、bootstrap REF、section-map source_ref、頁面 Colab 按鈕均為 `lessons-v0.4.1`。 |
| 額外 stdout 邊界 | 真實提取 `甲\n乙` 的 list stream 加 `丙\n` 的 str stream，結果 `甲\n乙丙\n`；stderr warning、display、execute_result、error 都排除，輸出順序和換行保留。 |
| 新版本變更範圍 | 42 本相對 v0.4.0 的整份檔案，在只正規化「v0.4.0→v0.4.1」和導言舊頁名「驗證範圍與後續實驗→驗證範圍」後逐 byte 相同；不是聲稱只改 tag。末格 source／outputs 沒變。 |
| 既有實驗未改 | `lesson_cases/`、`miniyolo/`、`artifacts/checks/curriculum/`、`docs/assets/diagrams/` 相對原快照共 150 個非 pycache 檔案逐 byte 相同。這個範圍不含另外更新的發布驗證 metadata。 |

## Session 與五種流程沒有被修錯

對原版與修正版做 AST 比對：新增函式只有 `notebook_stdout`；既有函式只有 `check_bootstrap` 不同，而差異只在 saved stdout 的提取。`SESSION` 和 `TORCH_STATE` 字串逐字相同；`session`、`run`、`clone`、`check_site`、`save`、CLI 等其他函式的 AST 均相同。

實際以原 `session()` 啟動兩個新 subprocess，分別用 `plain`、`import-torch-first`，讓第一格建立 shared=41、第二格讀它並印 42。兩次都得到 `ENV __main__\n<<next cell>>\n42\n`，stderr 各自保存 `warning\n`；cell 可共用 namespace、`__name__=='__main__'`、`PYTHONPATH` 從子程序移除、分格 marker 正常。這是既有指定環境的新 process，不是新 venv／fresh 套件安裝實測。

另直接執行完整修正版 `check_bootstrap()`，僅將 clone、venv／pip、torch metadata 和昂貴 notebook session 依賴換成受控 fixture，確認如下原有分支。mock 使用真實 notebook 的兩格原文和保存的 stdout，沒有放寬程式內的 passed 判定。

| flow | 檢查的判定與執行順序 |
| --- | --- |
| no-torch | before 為空、after 是 pinned、一次 plain session；錯把 before 設為已裝 torch 時應失敗。 |
| other-torch | before 為 2.8.0、after 為 pinned；after 仍 2.8.0 時應失敗。 |
| same-torch | before／after version 和 RECORD mtime 必須完全相同；模擬重新安裝導致 mtime 改變時應失敗。 |
| imported-torch | 第一輪 import-torch-first 必須非零退出且 stderr 包含重新啟動訊息，再有第二輪 plain session；只把訊息放 stdout、或第一輪 exit 0，都應失敗。 |
| deployment | pinned、最後一輪 exit 0；保留原有允許最後實驗小數輸出不同的例外，不要求和 saved stdout 完全相同。 |

五 flow 的 baseline 全通過，session 呼叫數依序為 1、1、1、2、1；部署 saved stdout 故意不同仍通過，與原有範圍相符。另七個負向探針（same 重新安裝、after 版本錯、no-torch 起始狀態錯、restart exit 0、restart 訊息只在 stdout、暖身 stdout 不相等、README 命令失敗）都讓總 passed 為 false。README 的命令解析與先後順序也與真實 bash 區塊相同，只排除註解、空行和不結束的 `zensical serve`。**這些是受控 flow 探針，不是五套 fresh 安裝或 README 命令已實跑通過的證據。**

## 完整 fresh probe 的限制

已讀 root 實跑 `/tmp/lessons-v0.4.1-verifier-bootstrap-probe.log`：修正後的 script 使用已公開且不變的 `lessons-v0.4.0`，在進入 README 的第二次 public clone 時，GitHub／代理傳輸回 503，`clone()` 拋 CalledProcessError（exit 128），所以沒有產生 `/tmp/lessons-v0.4.1-verifier-bootstrap-probe.json`。原版與修正版 `clone()` 的 AST 相同，這是外部網路阻斷，沒有證據顯示是 stdout 修正造成。

該 log 為 1,576 bytes，SHA-256 `b88ebd3d01b13d98122667397100c3b50a2639530aa6ce0bb2334e694fcdf616`，已保存在副本供核對。因為沒有最終 case 結果，不從「程式進到 README」推論五 sessions 已通過，也不宣稱 README 已通過。本次不重試完整 fresh probe、不 override clone、不放寬 passed 判定。

`lessons-v0.4.1` 的 42 本 notebook／本機配對一致性已查；**尚未 cold clone／安裝驗證這個未公開 tag**。既有 v0.4.0 immutable release 不覆寫的處理方向與 repo 指示一致；實際遠端 tag 保留與新公開網站比對不在此次本地修正複查中驗證。發布後仍需 GitHub-hosted `Verify published lessons` 對新 tag 完成 site、五種 fresh bootstrap 和 README 全流程；本次不能替代該公開驗證。

## 發現與 closure

| 發現 | 處理／closure |
| --- | --- |
| 原版必要缺陷：合法 stdout `text=list[str]` 在 outer join 拋 TypeError，bootstrap 尚未進入第一個 session | 已修正並獨立重現舊版失敗；新版解析 42 本原始 JSON、schema validation、新測試反證及 72 tests 均確認，closure 通過。 |
| fresh probe 無最終 JSON，不能宣稱完整 bootstrap／README 通過 | 無須改此解析修正；報告明示 GitHub 503 和實際未完成範圍，留待公開新 tag 的 Actions 驗證。沒有將外部阻斷冒稱為修正成功的全流程證據。 |
| 此次新增必要／建議問題 | 均為 0，沒有待修程式項目。 |

## 快照與可追溯檔案

| 檔案 | SHA-256 |
| --- | --- |
| `scripts/verify_release.py` | `fff8a374c293b329cc1caf48219426d9f07f4674cbe548e3a155c952c26e2f79` |
| `tests/test_release_verification.py` | `59c9079006b455b28de51b3929a8b5ffd9bed7eb9ccd0f96220f0c6031e56c01` |
| `tests/test_notebook_bootstrap.py` | `08ae79cbdc1e887083c83e81428e6986ca03eadf41281747dbf3cbbca0978d60` |
| `section-map.json` | `490393fd15d4c294cd670ded579cd2fcdfa527ab51fb3a343ac6cb340fb8afa5` |
| `README.md` | `6732d75dab1314105722092ef0d02e7202c19a61087d7692bc8311f7d3442df3` |
| `scripts/build_lesson_notebooks.py` | `2daddeff86572fc3edd3ba6302f0ff4696d33f5b1cd8037560ceb955f610893a` |
| `.github/workflows/verify-release.yml` | `e490d35241d29a69cb3abb7ef19a6bfd4cbf738fce5a08eb692ac45d2b392a0d` |

42 本保存的 stdout stream.text 全部是 list[str]。完整 42 本 notebook 的 SHA-256 清單（清單自身 SHA-256 `4a3774fb98a3a76706b0a1616824b372a84185d834675753dd7cd35ba4a68fcf`）和上述檔案 hash 在副本 `artifacts/runs/verifier-review/snapshot-hashes.json`。同目錄另有 `probe.py`、`probe.json`、`probe.stdout`、`pytest.log`、`verifier.diff`、`release-delta.json`、官方原文／schema、來源抓取失敗記錄、`root-bootstrap-probe.log` 與 `root-probe-limit.json`。交付前再比對 root 的 verifier、tests、section-map、README 與本副本逐 byte 相同。
